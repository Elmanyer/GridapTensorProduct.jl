# CLAUDE.md — GridapTensorProduct.jl

This file is a comprehensive technical reference for AI agents working on this codebase.
It documents the current state of the library, module-by-module design, implementation
status, known limitations, and next-step priorities (as of 2026-05-19).

---

## 1. Purpose and Motivation

`GridapTensorProduct.jl` extends [Gridap.jl](https://github.com/gridap/Gridap.jl) to
efficiently assemble and solve PDEs on **tensor-product domains**:

```
Ω = Ω₁ × Ω₂ × ... × Ω_N  ⊆  ℝᴺ
```

When the domain has Cartesian-product structure and the PDE operator is *separable*,
global FEM matrices decompose as sums/products of **Kronecker products** of small 1D
subdomain matrices. The library exploits this to replace expensive N-D global assembly
with N independent 1D assemblies followed by cheap Kronecker operations.

**Key application context:** Far-field / exterior domain problems at TU Delft (WP2,
author: pmanyerfuertes@tudelft.nl). The primary target is 2D and 3D structured Cartesian
tensor grids arising in atmospheric / ocean models.

---

## 2. Dependencies

```toml
Gridap      # FEM framework: reference elements, quadrature, spaces, assembly
Kronecker   # Lazy Kronecker products (KroneckerProduct struct, matrix-free matvec)
BlockArrays # Block matrix utilities
FillArrays  # Fill arrays for DOF indexing
LinearAlgebra # stdlib
```

**Why Kronecker.jl for lazy evaluation:**
`kron(A, B)` from `LinearAlgebra` materialises a dense `(n²×n²)` matrix.
`Kronecker.kronecker(A, B)` returns a lightweight struct (~1 KB) and computes
matrix–vector products without forming the global matrix. This is critical for
3D+ problems where the global matrix would be infeasible to materialise.

---

## 3. Five-Stage Pipeline Architecture

The library is organised into five stages; subdomains are the primary computational
unit throughout stages 1–4 and tensor coupling is deferred to stage 5.

```
Stage 1: TensorProductMeasure        — bundle N subdomain quadrature measures
Stage 2: TensorProductFESpace        — bundle N subdomain FE spaces + DOF mapping
Stage 3: TensorProductGeometry       — lazy triangulation and cell-field wrappers
Stage 4: TensorProductFEMOperators   — extract 1D matrices; form Kronecker operators
Stage 5: TensorProductAssembly       — high-level API; assemble and return (A, b)
```

---

## 4. DOF Ordering Convention

**Subdomain 1 = fastest-varying (innermost).** The global linear DOF index satisfies:

```
global_dof(i₁, i₂, ..., iₙ) = i₁ + (i₂−1)·n₁ + (i₃−1)·n₁·n₂ + ...
```

Kronecker products are always built **innermost first** (subdomain 1 at the right):

```julia
# For N=2:  kron(M₂, M₁)    (M₂ outer, M₁ inner)
# For N=3:  kron(M₃, kron(M₂, M₁))
```

This convention is implemented and verified in `TensorProductAssemblers.jl`
(`tensor_tuple_to_linear_index`, `linear_index_to_tensor_tuple`) and tested in
`test/DOFOrderingTests.jl` and `test/KroneckerAssemblerTests.jl`.

---

## 5. Kronecker Decomposition Formulas

| Operator | Global Formula |
|---|---|
| Mass | `M = M⁽¹⁾ ⊗ M⁽²⁾ ⊗ ... ⊗ M⁽ᴺ⁾` |
| Stiffness (Laplacian) | `A = Σₖ M⁽¹⁾ ⊗ ... ⊗ K⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾` |
| Gradient | `G = vcat( M⊗...⊗G⁽ᵏ⁾⊗...⊗M )` (block column, N blocks) |
| Divergence | `B = Gᵀ` |
| Curl-Curl | `C = Σₖ Σ_{l≠k} (⊗_{j≠k,l} M⁽ʲ⁾) ⊗ (Cᵀ Dₖᵀ Dₗ − Dₗᵀ Dₖ)` |
| Advection | `T = Σₖ bₖ · M⁽¹⁾ ⊗ ... ⊗ A⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾` |

Per-subdomain fundamental matrices:
- `M⁽ᵏ⁾`: `∫ φᵢ φⱼ dxₖ` — mass on Ωₖ
- `K⁽ᵏ⁾`: `∫ (∂φᵢ/∂xₖ)(∂φⱼ/∂xₖ) dxₖ` — stiffness on Ωₖ
- `G⁽ᵏ⁾`: `∫ (∂φᵢ/∂xₖ) φⱼ dxₖ` — gradient on Ωₖ (square, n×n)
- `D⁽ᵏ⁾`: same as G⁽ᵏ⁾, kept separate for curl-curl
- `A⁽ᵏ⁾`: `∫ φᵢ (bₖ · ∂φⱼ/∂xₖ) dxₖ` — advection on Ωₖ (asymmetric)

---

## 6. Module-by-Module Reference

### 6.1 `TensorProductMeasures.jl` — STATUS: COMPLETE AND SATISFACTORY

**Type:** `TensorProductMeasure{N, T<:NTuple{N,Measure}} <: Measure`

A simple, clean wrapper over N Gridap `Measure` objects. Subtype of `Measure` so it
can appear anywhere a Gridap Measure is expected.

**Construction:**
```julia
dΩtp = dΩ1 ⊗ dΩ2          # ⊗ operator, auto-flattens nested TensorProductMeasures
dΩtp = dΩ1 ⊗ dΩ2 ⊗ dΩ3    # N=3
dΩtp = TensorProductMeasure((dΩ1, dΩ2))  # tuple form
```

**Key interface functions (all exported):**
- `get_measures(m)` → `NTuple{N, Measure}`
- `get_measure(m, k)` → `Measure` for subdomain k
- `get_triangulations(m)` → NTuple of triangulations
- `num_subdomains(m)` → N
- `get_quadrature_degree(m)`, `get_quadrature_degree(m, k)`
- `get_quadrature_points(m, k)`, `get_quadrature_weights(m, k)`
- `get_tensor_product_points(m)`, `get_tensor_product_weights(m)`

**Assessment:** Well-implemented. The `⊗` operator correctly flattens nested tensor
products (`dΩ1 ⊗ dΩ2 ⊗ dΩ3` produces a single `TensorProductMeasure{3}`, not nested).
All accessors are defensive (bounds-checked). The `Base.show` implementation is useful.

---

### 6.2 `TensorProductFESpaces.jl` — STATUS: COMPLETE AND SATISFACTORY

**Type:** `TensorProductFESpace{T<:Tuple} <: FESpace`

Bundles N independent 1D FE spaces into a single tensor-product space. DOF count and
indexing are correctly handled via Cartesian product arithmetic.

**Construction:**
```julia
V_tp = TensorProductFESpace(Vx, Vy)        # variadic
V_tp = TensorProductFESpace(Vx, Vy, Vz)    # 3D
V_tp = TensorProductFESpace((Vx, Vy))      # tuple form
```

**Key interface functions (all implemented):**
- `num_free_dofs(V_tp)` → `prod(num_free_dofs(Vk))`  ✓
- `get_free_dof_ids(V_tp)` → `Base.OneTo(num_free_dofs(V_tp))`  ✓
- `get_triangulation(V_tp)` → `TensorProductTriangulation`  ✓
- `get_triangulations(V_tp)` → NTuple of per-subdomain triangulations  ✓
- `get_tensor_cell_ids(V_tp)` → Table mapping TP cells to subdomain cell tuples  ✓
- `get_tensor_cell_dof_ids(V_tp)` → Table mapping TP cells to subdomain DOF tuples  ✓

**`TensorProductFEFunction`:** Placeholder struct storing `free_values` and
`dirichlet_values`. `cell_field::CF` is stored as `Nothing` — full cell-field
scatter (mapping solution vector entries back to cell-local arrays) is not yet
implemented.

**Known limitations (documented as `[PHASE 3+] TODO`):**
- `get_fe_basis`, `get_fe_dof_basis` return `Vector` of bases, not a single wrapped object.
- `get_cell_constraints`, `get_cell_isconstrained` return lists, not combined tensor versions.
- `CellField(fs::TensorProductFESpace, cell_vals)` throws `ErrorException` (not implemented).

**Assessment:** The core DOF arithmetic and space construction are correct and well-
implemented. The partial Gridap FESpace interface compliance (deviations are documented)
is acceptable for the current use case where assembly goes through the custom Kronecker
path rather than Gridap's standard assembler.

---

### 6.3 `TensorProductGeometry.jl` — STATUS: PARTIAL (structural wrapper only)

**Types:**
- `TensorProductTriangulation{T<:Tuple}` — wraps N component triangulations; pre-
  computes `CartesianIndices` over all tensor cells.
- `TensorProductCellField{CF<:Tuple, DATA}` — wraps N component CellFields.

**What works:**
- `TensorProductTriangulation(t1, t2, ...)` construction
- `num_cells(tptp)`, `length(tptp)`
- `get_triangulation(tpfc)` for a `TensorProductCellField`

**What is NOT implemented:**
- `TensorProductCellField` indexing (`getindex` throws `error("not yet fully implemented")`)
- Full Gridap Triangulation interface (`get_cell_coordinates`, `get_node_coordinates`, etc.)
- Does NOT inherit from `Gridap.Geometry.Triangulation` (planned for PHASE 4)

**Assessment:** Lightweight structural container. Sufficient for the current workflow
where geometry data stays per-subdomain and only the Kronecker algebra lives globally.

---

### 6.4 `TensorProductAssemblers.jl` — STATUS: LOW-LEVEL UTILITIES COMPLETE

Contains two assembler types and index-conversion utilities used throughout:

**Types:**
- `KroneckerAssembler <: TensorProductAssembler` — separable Kronecker path
- `FallbackAssembler <: TensorProductAssembler` — cell-wise fallback (stub; hard errors)

**Index conversion (exported, tested in `DOFOrderingTests.jl`):**
```julia
tensor_tuple_to_linear_index((i1,i2,...), (n1,n2,...)) → Int
linear_index_to_tensor_tuple(linear, (n1,n2,...))      → NTuple
```

**Matrix/vector utilities:**
```julia
tensor_kron((A1, A2, ...))         # kron(A_N, ..., kron(A2, A1)) — innermost-first
tensor_kron_vector((v1, v2, ...))  # kron(v_N, ..., kron(v2, v1))
tensor_matrix_sizes(mats)          # (n1, n2, ...) row sizes
```

**Assembly entry points:**
```julia
assemble_tensor_operator(KroneckerAssembler(), (A1, A2))                     # simple product
assemble_tensor_operator(KroneckerAssembler(), ((α1,(A1,B1)), (α2,(A2,B2))), Val(:sum))  # linear combo
assemble_tensor_rhs(KroneckerAssembler(), (b1, b2))                          # kron vector
apply_tensor_operator(KroneckerAssembler(), (A1, A2), x)                     # matrix-free Ax
assemble_tensor_operator(FallbackAssembler(), row_sizes, col_sizes, entry_fun)  # from callback
```

**Note:** These are low-level utilities. The high-level assembly API lives in
`TensorProductFEMOperators.jl` and `TensorProductAssembly.jl`.

---

### 6.5 `TensorProductIntegration.jl` — STATUS: HOOK + FORM CONTRIBUTION + PLACEHOLDER

This module provides the Gridap hook that intercepts `∫(expr) * dΩ_tp` expressions,
captures multi-term 2-arg forms via `TensorProductFormContribution`, and retains
`TensorProductDomainContribution` as a placeholder for future direct-integration work.

**`TensorProductSeparableTerm` and `TensorProductWeakForm` have been moved to
`TensorProductOperator.jl`** (the Stage 4 translation layer, where they conceptually
belong).

**Types retained:**

**`TensorProductIntegrand`** — produced by `∫(expr) * dΩ_tp`:
```julia
struct TensorProductIntegrand
    object::Any
    measure::TensorProductMeasure
end
Base.:*(integrand::Integrand, m::TensorProductMeasure) = TensorProductIntegrand(...)
```
`Gridap.CellData.integrate(::TensorProductIntegrand)` **always throws an error**
directing the user to `TensorProductAffineOperator` or `TensorProductWeakForm`.

**`TensorProductFormContribution`** — list of terms captured from a 2-arg form:
```julia
struct TensorProductFormContribution
    terms::Vector{TensorProductIntegrand}
end
```
Produced when `+` is called between `TensorProductIntegrand` objects. This keeps terms
separate (rather than merging as Gridap's `DomainContribution` would). Four `+` overloads:
`TPI+TPI`, `TPFC+TPI`, `TPI+TPFC`, `TPFC+TPFC`.

**`TensorProductDomainContribution`** — per-subdomain contribution container (placeholder):
```julia
mutable struct TensorProductDomainContribution
    contributions::Vector{DomainContribution}
    measure::TensorProductMeasure
    operator_type::Symbol
end
```
Not actively used; retained for future direct-integration work.

---

### 6.6 `TensorProductFEMOperators.jl` — STATUS: CORE ALGORITHM COMPLETE; LAZY ASSEMBLY IMPLEMENTED

This is the **mathematical heart** of the library. It implements:
1. Lazy storage of per-subdomain matrices
2. On-demand assembly of each matrix type
3. All six Kronecker assemblers

#### 6.6.1 `TensorProductSubdomainOperators{N}` — Lazy Container

```julia
mutable struct TensorProductSubdomainOperators{N}
    spaces::NTuple{N, FESpace}
    measures::NTuple{N, Measure}
    b_coeffs::Union{NTuple, Nothing}       # advection velocity coefficients
    _M_ops::Vector{Union{Matrix, Nothing}} # cache; nothing = not yet assembled
    _K_ops::Vector{Union{Matrix, Nothing}}
    _G_ops::Vector{Union{Matrix, Nothing}}
    _D_ops::Vector{Union{Matrix, Nothing}}
    _A_ops::Vector{Union{Matrix, Nothing}}
end
```

**Lazy access pattern:** `Base.getproperty` is overridden so that `ops.M_ops` (public
name, no underscore) triggers lazy assembly of all N mass matrices and returns them as
`NTuple{N, Matrix}`. Direct field access to `ops._M_ops` (underscore, goes to
`getfield`) skips the lazy assembly and is used internally.

```julia
function Base.getproperty(ops::TensorProductSubdomainOperators, s::Symbol)
    s === :M_ops && return _get_M_ops!(ops)   # triggers lazy assembly
    s === :K_ops && return _get_K_ops!(ops)
    s === :G_ops && return _get_G_ops!(ops)
    s === :D_ops && return _get_D_ops!(ops)
    s === :A_ops && return _get_A_ops!(ops)
    return getfield(ops, s)                   # raw field access for everything else
end
```

**Per-subdomain assembly helpers** (`_assemble_M_k!`, `_assemble_K_k!`, etc.):
each uses standard Gridap `AffineFEOperator` on the 1D subdomain. Note that all
gradient/derivative extraction uses `∇(u) ⋅ VectorValue(1.0)` (projects to scalar
in 1D) rather than `∇(u)[1]` (which is not valid in Gridap's symbolic framework).

**Factory function (now O(1)):**
```julia
function assemble_subdomain_operators(spaces, measures; b_coeffs=nothing) where N
    noth() = Vector{Union{Matrix,Nothing}}(fill(nothing, N))
    return TensorProductSubdomainOperators{N}(
        spaces, measures, b_coeffs, noth(), noth(), noth(), noth(), noth())
end
```
No assembly happens at construction. All five matrix types are initialised to `nothing`.

**Memory benefit:** A call to `assemble_stiffness_tensor(ops)` assembles only M and K
(2×N matrices total). G, D, A remain `nothing`. If `assemble_mass_tensor` is then
called on the same `ops`, M is already cached — no re-assembly.

#### 6.6.2 Kronecker Assemblers

All six assemblers follow the same pattern: extract the needed properties once at the
top (e.g. `M = ops.M_ops; K = ops.K_ops`), then work with local NTuples in inner loops.

```julia
# assemble_mass_tensor — product rule
M = ops.M_ops
result = M[1]; for k in 2:N: result = kronecker(M[k], result); end
return collect(result)

# assemble_stiffness_tensor — sum rule: Σ_k kron(M_N,...,K_k,...,M_1)
M = ops.M_ops; K = ops.K_ops
for k in 1:N:
    term = (k==1) ? K[1] : M[1]
    for j in 2:N: term = kronecker((j==k) ? K[j] : M[j], term); end
    A_global += collect(term)
end

# assemble_gradient_tensor — block column
# assemble_divergence_tensor — transpose of gradient
# assemble_curl_curl_tensor — N(N-1) anti-symmetric pairs
# assemble_advection_tensor — scaled sum rule with A instead of K
```

All assemble to dense `Matrix{Float64}` via `collect()` on the lazy Kronecker object.

#### 6.6.3 `assemble_weak_form` — New Auto-Detection Assembly

```julia
function assemble_weak_form(wf::TensorProductWeakForm,
                             spaces::NTuple{N,<:FESpace},
                             measures::NTuple{N,Measure}) where N
```

This function implements the new "3-arg form" assembly path:
1. Compute reference mass matrices `M_k` once for all terms.
2. For each `TensorProductSeparableTerm` in `wf.terms`:
   a. Evaluate `term.form(u_k, v_k, dΩ_k)` on each subdomain k → per-subdomain `A_k`
   b. **Product/sum heuristic:** if `‖A_k − M_k‖ ≤ tol` for ALL k → product rule;
      otherwise → sum rule.
   c. Accumulate into global matrix.

**The heuristic works for:** mass (product), stiffness (sum), advection (sum).
**The heuristic cannot handle:** variable-coefficient operators, mixed terms, or operators
with coupling between subdomains other than via the standard product/sum pattern.

#### 6.6.4 Helper: `extract_gradient_matrix`, `extract_derivative_matrix`, `extract_advection_matrix`

These extract the rectangular/asymmetric subdomain matrices by assembling the appropriate
weak form on a single 1D space using standard Gridap `AffineFEOperator`.

---

### 6.7 `TensorProductOperator.jl` — STATUS: COMPLETE (Stage 4 translation layer + high-level operator)

This is the **central Stage 4 module** that translates user-written weak forms into
Kronecker-factored operators. It now contains:

#### `TensorProductSeparableTerm` — now dual-representation (moved here from Integration.jl)

```julia
struct TensorProductSeparableTerm
    form::Union{Function, Nothing}                      # Track A: 3-arg function
    subdomain_matrices::Union{Vector{Matrix{Float64}}, Nothing}  # Track B: pre-assembled
    label::Symbol       # :mass, :stiffness, :advection, :unknown
    coefficient::Float64
    kronecker_rule::Symbol  # :product, :sum, :unknown (explicit for Track B)
end
```

Track A constructors (3-arg form):
```julia
TensorProductSeparableTerm(form)                  # label=:unknown, coeff=1.0, rule=:unknown
TensorProductSeparableTerm(form, :stiffness)      # coeff=1.0
TensorProductSeparableTerm(form, :mass, -1.0)     # explicit coefficient
```

Track B constructors (pre-assembled matrices):
```julia
TensorProductSeparableTerm(mats::Vector{Matrix{Float64}}, label, rule)
TensorProductSeparableTerm(mats::Vector{Matrix{Float64}}, label, rule, coeff)
```

#### `TensorProductWeakForm` — intermediary between weak form and operator (moved here)

```julia
mutable struct TensorProductWeakForm
    terms::Vector{TensorProductSeparableTerm}
    _kronecker_rules::Union{Nothing, Vector{Symbol}}  # :product|:sum, cached after assembly
end
# Constructors:
TensorProductWeakForm(form::Function)
TensorProductWeakForm(forms::Vector{<:Function})
TensorProductWeakForm(term::TensorProductSeparableTerm)
TensorProductWeakForm(terms::Vector{TensorProductSeparableTerm})
# Composition:
stiff + mass  # TensorProductWeakForm([stiff, mass])
wf + term     # append a term
wf1 + wf2    # merge two weak forms
```

#### New helper functions

**`normalize_to_list(result)`** → `Vector{TensorProductIntegrand}` — converts a
`TensorProductIntegrand` or `TensorProductFormContribution` to a flat list of integrands,
providing a uniform input for the translation engine.

**`assemble_subdomain_matrix(dc::DomainContribution, U_k, V_k)`** → `Matrix{Float64}` —
assembles a per-subdomain matrix from a pre-computed `DomainContribution` using
`collect_cell_matrix + assemble_matrix`.

**`classify_term(A_ops, M_ops, K_ops)`** → `(label, rule, coeff)` — detects scaled
mass/stiffness operators via Frobenius-norm proportionality to reference mass/stiffness
matrices; returns the inferred `label`, Kronecker `rule`, and scalar `coeff`.

**`translate_bilinear_form(a, V_tp, quad_order)`** → `TensorProductWeakForm` — full 2-arg
form translation engine. Evaluates `a(u_k, v_k)` per subdomain (using subdomain
measures created at `quad_order`), calls `normalize_to_list` to handle both single and
multi-term forms, then calls `classify_term` per term to build a `TensorProductWeakForm`
with fully populated Track B `TensorProductSeparableTerm`s.

#### `assemble_weak_form` — translate weak form to global matrix (moved here from FEMOperators.jl)

Evaluates each `TensorProductSeparableTerm` on every 1D subdomain and accumulates via
the product/sum Kronecker heuristic. Handles both Track A (3-arg function, uses
heuristic) and Track B (pre-assembled matrices, uses `classify_term` result). Scales by
`term.coefficient`. Caches the per-term rule in `wf._kronecker_rules`.

#### `TensorProductAffineOperator` — MOVED HERE from TensorProductAssembly.jl

The struct gains an additional `_tp_operator::Union{TensorProductOperator, Nothing}` field.
`_assemble_tp!` now has three paths:
1. Pre-built `weak_form` (3-arg and vector API) — existing path
2. Legacy `op_type` dispatch — existing path
3. 2-arg form → `translate_bilinear_form` → `TensorProductOperator` — new path: when
   `a` is detected as a 2-arg form (no `Measure` argument), `translate_bilinear_form` is
   called and the result is stored in `_tp_operator` for re-use.

#### `TensorProductOperator{OP_TYPE}` — typed, caching wrapper

```julia
# Explicit type API (backward compatible):
op = TensorProductOperator(:stiffness, (Vx, Vy), (dΩx, dΩy))
A  = get_global_matrix(op)

# Weak form API (new):
stiff = TensorProductSeparableTerm((u, v, dΩ) -> ∫(∇(u)⋅∇(v)) * dΩ, :stiffness)
mass  = TensorProductSeparableTerm((u, v, dΩ) -> ∫(u*v) * dΩ, :mass)
wf    = stiff + mass
op    = TensorProductOperator(wf, (Vx, Vy), (dΩx, dΩy))
A     = get_global_matrix(op)   # OP_TYPE = :weak_form
```

The `weak_form` field stores the `TensorProductWeakForm` for `:weak_form`-typed operators.
All assembly is lazy — triggered only on the first `get_global_matrix` call.

**Exported:** `TensorProductSeparableTerm`, `TensorProductWeakForm`, `num_terms`,
`get_terms`, `normalize_to_list`, `assemble_subdomain_matrix`, `classify_term`,
`translate_bilinear_form`, `assemble_weak_form`, `TensorProductOperator`,
`get_global_matrix`, `get_subdomain_operators`, `get_operator_type`,
`TensorProductAffineOperator`

---

### 6.8 `TensorProductAssembly.jl` — STATUS: COMPLETE (high-level functional API only)

`TensorProductAffineOperator`, `_assemble_tp!`, `get_matrix`, and `get_vector` have been
**REMOVED from this file** and moved to `TensorProductOperator.jl` (Section 6.7). Only
the two functional entry points remain here:

#### `assemble_tensor_affine_operator` (function)

Legacy functional API. Accepts explicit `op_type::Symbol`:

```julia
A, b = assemble_tensor_affine_operator(a, l, U_tp, V_tp; op_type=:stiffness, quad_order=2)
```

Internally calls `assemble_subdomain_operators` + the appropriate Kronecker assembler.
RHS is assembled as `b = kron(b_N, ..., kron(b_2, b_1))` from per-subdomain unit loads.

#### `assemble_tensor_system` (function)

Convenience wrapper that calls `assemble_tensor_affine_operator` and returns `(A, b)`
for direct use.

---

### 6.9 `GridapTensorProduct.jl` — Module Entry Point

Loads all submodules in dependency order. Key exports:

```julia
# Stage 1
export TensorProductMeasure, ⊗

# Stage 2
export TensorProductFESpace

# Stage 3
export TensorProductTriangulation, TensorProductCellField

# Stage 4
export TensorProductSubdomainOperators, TensorProductGlobalOperator
export assemble_subdomain_operators
export assemble_mass_tensor, assemble_stiffness_tensor, assemble_gradient_tensor
export assemble_divergence_tensor, assemble_curl_curl_tensor, assemble_advection_tensor
export assemble_weak_form
export extract_gradient_matrix, extract_derivative_matrix, extract_advection_matrix

# Stage 4 — FEMOperators (Kronecker assemblers, subdomain ops)
export TensorProductSubdomainOperators, assemble_subdomain_operators
export assemble_mass_tensor, assemble_stiffness_tensor, ...

# Stage 4 — Operator (weak form types + translation + typed wrapper)
export TensorProductSeparableTerm, TensorProductWeakForm, num_terms, get_terms
export normalize_to_list, assemble_subdomain_matrix, classify_term
export translate_bilinear_form
export assemble_weak_form
export TensorProductOperator, get_global_matrix, get_subdomain_operators, get_operator_type
export TensorProductAffineOperator  # moved here from Assembly section

# Stage 5
export assemble_tensor_affine_operator, assemble_tensor_system

# Assemblers
export TensorProductAssembler, KroneckerAssembler, FallbackAssembler

# Integration types (hook + placeholder)
export TensorProductIntegrand
export TensorProductFormContribution
export TensorProductDomainContribution
```

**Load order (fixed):** `Measures → Integration → Geometry → FESpaces → Assemblers →
FEMOperators → Operator → Assembly` (previously Assembly loaded before FEMOperators).

---

## 7. Test Suite

Located in `test/`. Run via `Pkg.test("GridapTensorProduct")` or `include("runtests.jl")`.

| Test file | What it tests |
|---|---|
| `TensorProductQuadraturesTests.jl` | Quadrature accessors via `TensorProductMeasure` |
| `TensorProductFESpaceInterfaceTests.jl` | DOF counts, cell IDs, cell DOF IDs, triangulation |
| `DOFOrderingTests.jl` | Bijection `tensor_tuple ↔ linear_index` for N=1,2,3 |
| `FEFunctionTests.jl` | `TensorProductFEFunction` construction with free/Dirichlet values |
| `KroneckerAssemblerTests.jl` | `tensor_kron`, `apply_tensor_operator`, `assemble_tensor_rhs`, sum-of-kron |
| `PoissonEquivalenceTests.jl` | Tensor DOF count; `KroneckerAssembler` vs. explicit `kron` |

**Validation (examples, not in test suite):**
Three Poisson problems validated against standard Gridap reference solutions with L² error = 0:
- 2D homogeneous: [0,1]², 3×3 DOFs, A = K₁⊗M₂ + M₁⊗K₂
- 3D homogeneous: [0,1]³, 3×3×3 DOFs, A = K₁⊗M₂⊗M₃ + ...
- 2D heterogeneous: 3×7 DOFs (different mesh refinements per subdomain)

---

## 8. Weak Form Translation — Current State and Limitations

### What works today

| Use case | How to do it |
|---|---|
| Poisson (stiffness) | `TensorProductSeparableTerm((u,v,dΩ)->∫(∇u⋅∇v)*dΩ, :stiffness)` |
| Mass matrix | `TensorProductSeparableTerm((u,v,dΩ)->∫(u*v)*dΩ, :mass)` |
| Helmholtz (K + M) | `stiff + mass` → `TensorProductWeakForm` → `TensorProductOperator` |
| Scaled term (e.g. -k²M) | `TensorProductSeparableTerm(form, :mass, -k²)` |
| Any composite form | `TensorProductOperator(wf, spaces, measures)` |
| Advection–diffusion | `op_type=:advection` + `b_coeffs` (legacy), or custom 3-arg |
| Gradient, divergence, curl-curl | `op_type=:gradient` etc. (explicit path) |
| Custom per-subdomain RHS | `rhs_forms=(v->∫(f1*v)*dΩ1, v->∫(f2*v)*dΩ2)` |
| 2-arg form `a(u,v) = ∫(expr)*dΩ_tp` | `TensorProductAffineOperator(a, l, U_tp, V_tp)` → automatic classification via `translate_bilinear_form` |
| Scaled mass `∫(k²*u*v)*dΩ_tp` | Detected automatically: label=:mass, coeff=k², product rule with normalized M matrices |
| Scaled stiffness `∫(c*∇u⋅∇v)*dΩ_tp` | Detected automatically: label=:stiffness, coeff=c, sum rule with normalized K matrices |
| Multi-term 2-arg `a(u,v)=∫(∇u⋅∇v)*dΩ_tp + ∫(k²*u*v)*dΩ_tp` | Two separate `TensorProductIntegrand`s captured via `TensorProductFormContribution`, each classified independently |

### What does NOT work

1. **Non-separable operators:** Variable-coefficient Laplacians `∫(a(x,y)*∇u⋅∇v)*dΩ`
   cannot be decomposed into Kronecker products and require full N-D assembly.

2. **Symbolic analysis of weak form trees:** There is no mechanism to inspect a Gridap
   `DomainContribution` and automatically determine its Kronecker structure.

3. **Mixed operators with cross-subdomain coupling:** For example, `∫(∂u/∂x * ∂v/∂y)*dΩ`
   couples derivatives from different subdomains and does not fit the standard sum/product
   rules.

### The product/sum heuristic (3-arg path)

When a 3-arg form is used, `assemble_weak_form` applies this rule per term:
- Evaluate `form(u_k, v_k, dΩ_k)` on each subdomain k to get `A_k`
- If `‖A_k − M_k‖ ≤ tol * (‖M_k‖ + 1)` for ALL k: **product rule** → `kron(A_N,...,A_1)`
- Otherwise: **sum rule** → `Σ_k kron(M_N,...,A_k,...,M_1)`

This correctly identifies mass operators (product) and stiffness/advection (sum) since
on a 1D subdomain `∫(∇u⋅∇v)*dΩ ≠ ∫(u*v)*dΩ`. However, the heuristic has no
understanding of the mathematical structure beyond this simple comparison — it cannot
handle non-standard terms or terms where per-subdomain matrices happen to be close to M
by coincidence.

### Recommended next steps

1. **Extend `translate_bilinear_form` to more operator types:** Currently classifies only
   mass and stiffness (via Frobenius proportionality). Advection and gradient operators
   require additional heuristics or explicit `label` hints.

2. **Implement tensor-product quadrature evaluation:** Walk the `TensorProductMeasure`'s
   component quadratures, build the Cartesian-product quadrature rule, and evaluate the
   integrand at all tensor quadrature points. This is the "full" path that bypasses the
   Kronecker assumption entirely and handles arbitrary integrands.

3. **Symbolic/numerical rank-1 detection:** Given per-subdomain matrices `A_k`, check
   whether `A = ⊗_k A_k` (rank-1 in Kronecker sense) or `A = Σ_k (⊗_j≠k M_j) ⊗ A_k`
   (sum rule). More robust than the current mass-comparison heuristic.

---

## 9. Workflow Recipes for a Future Agent

### Minimal 2D Poisson solve

```julia
using Gridap, GridapTensorProduct

model_x = CartesianDiscreteModel((0,1), 20)
model_y = CartesianDiscreteModel((0,1), 20)
reffe = ReferenceFE(lagrangian, Float64, 2)

Vx = TestFESpace(Interior(model_x), reffe; conformity=:H1, dirichlet_tags="boundary")
Vy = TestFESpace(Interior(model_y), reffe; conformity=:H1, dirichlet_tags="boundary")
Ux = TrialFESpace(Vx, 0.0)
Uy = TrialFESpace(Vy, 0.0)
V_tp = TensorProductFESpace(Vx, Vy)
U_tp = TensorProductFESpace(Ux, Uy)

# New 3-arg API (auto-detection, no op_type needed):
op = TensorProductAffineOperator(
    (u, v, dΩ) -> ∫(∇(u)⋅∇(v)) * dΩ,
    (v, dΩ)    -> ∫(1.0 * v) * dΩ,
    U_tp, V_tp; quad_order=4)

A = get_matrix(op)
b = get_vector(op)
u = A \ b   # or: u = lu(sparse(A)) \ b  for large systems
```

### Gridap-style 2-arg form (automatic translation, new in Phase 2)

```julia
dΩ_tp = dΩx ⊗ dΩy
a(u, v) = ∫(∇(u)⋅∇(v)) * dΩ_tp   # closed over dΩ_tp
l(v)    = ∫(1.0 * v) * dΩ_tp
op = TensorProductAffineOperator(a, l, U_tp, V_tp)  # translate_bilinear_form called lazily
A  = get_matrix(op)  # labels: [:stiffness], rules: [:sum], diff vs K: 0.0
```

### Multi-term 2-arg form (Helmholtz, new in Phase 2)

```julia
k = 2.5
dΩ_tp = dΩx ⊗ dΩy
a(u, v) = ∫(∇(u)⋅∇(v)) * dΩ_tp + ∫(k^2 * u * v) * dΩ_tp
l(v)    = ∫(1.0 * v) * dΩ_tp
op = TensorProductAffineOperator(a, l, U_tp, V_tp)
# → TensorProductFormContribution with 2 terms
# → labels: [:stiffness, :mass], coeffs: [1.0, k²], rules: [:sum, :product]
A  = get_matrix(op)
```

### Using TensorProductOperator for explicit control

```julia
dΩx = Measure(Interior(model_x), 4)
dΩy = Measure(Interior(model_y), 4)

# Lazy operator — no assembly at construction
op_stiff = TensorProductOperator(:stiffness, (Vx, Vy), (dΩx, dΩy))
A = get_global_matrix(op_stiff)   # assembled once, cached

# Low-level: get subdomain matrices directly
ops = assemble_subdomain_operators((Vx, Vy), (dΩx, dΩy))
M = assemble_mass_tensor(ops)     # assembles M only
K = assemble_stiffness_tensor(ops) # assembles M (from cache) + K
```

### Using TensorProductWeakForm for explicit control

```julia
dΩx = Measure(Interior(model_x), 4)
dΩy = Measure(Interior(model_y), 4)

# Compose a Helmholtz form: K − k²·M
k = 2.5
stiff = TensorProductSeparableTerm((u, v, dΩ) -> ∫(∇(u)⋅∇(v)) * dΩ, :stiffness)
mass  = TensorProductSeparableTerm((u, v, dΩ) -> ∫(u*v) * dΩ, :mass, -k^2)
wf    = stiff + mass

op = TensorProductOperator(wf, (Vx, Vy), (dΩx, dΩy))
A  = get_global_matrix(op)
# After assembly: wf._kronecker_rules == [:sum, :product]
```

### Helmholtz via TensorProductAffineOperator (alternative)

```julia
k_wavenumber = 2.5
op = TensorProductAffineOperator(
    [(u,v,dΩ) -> ∫(∇(u)⋅∇(v))*dΩ,
     (u,v,dΩ) -> ∫((-k_wavenumber^2)*u*v)*dΩ],
    (v,dΩ) -> ∫(1.0*v)*dΩ,
    U_tp, V_tp)
```

### Advection–diffusion

```julia
bx, by = 1.0, 2.0
op = TensorProductAffineOperator(
    (u,v,dΩ) -> ∫(∇(u)⋅∇(v))*dΩ,   # diffusion term only
    (v,dΩ)   -> ∫(1.0*v)*dΩ,
    U_tp, V_tp)

# Add advection separately:
ops = assemble_subdomain_operators((Vx,Vy), (dΩx,dΩy); b_coeffs=(bx,by))
A = get_matrix(op) + assemble_advection_tensor(ops, (bx, by))
```

---

## 10. File Index

```
src/
├── GridapTensorProduct.jl        — module entry point, all exports
├── TensorProductMeasures.jl      — TensorProductMeasure, ⊗ operator
│                                   STATUS: COMPLETE AND SATISFACTORY
├── TensorProductIntegration.jl   — TensorProductIntegrand (∫*dΩ_tp hook),
│                                   TensorProductFormContribution (multi-term 2-arg capture),
│                                   TensorProductDomainContribution (placeholder)
│                                   STATUS: HOOK + FORM CONTRIBUTION + PLACEHOLDER
├── TensorProductGeometry.jl      — TensorProductTriangulation, TensorProductCellField
│                                   STATUS: STRUCTURAL WRAPPERS ONLY; GRIDAP INTERFACE INCOMPLETE
├── TensorProductFESpaces.jl      — TensorProductFESpace, TensorProductFEFunction
│                                   STATUS: COMPLETE AND SATISFACTORY (core DOF arithmetic correct)
├── TensorProductAssemblers.jl    — KroneckerAssembler, FallbackAssembler,
│                                   tensor_kron, tensor_tuple_to_linear_index, etc.
│                                   STATUS: COMPLETE (low-level utilities)
├── TensorProductAssembly.jl      — assemble_tensor_affine_operator, assemble_tensor_system
│                                   STATUS: COMPLETE (functional API only; TensorProductAffineOperator moved)
├── TensorProductFEMOperators.jl  — TensorProductSubdomainOperators (lazy), all Kronecker assemblers,
│                                   extract_* helpers
│                                   STATUS: COMPLETE (lazy assembly, all 6 operators)
└── TensorProductOperator.jl      — TensorProductSeparableTerm (dual-rep), TensorProductWeakForm,
                                    normalize_to_list, assemble_subdomain_matrix, classify_term,
                                    translate_bilinear_form, assemble_weak_form, TensorProductOperator,
                                    TensorProductAffineOperator (moved from Assembly)
                                    STATUS: COMPLETE (Stage 4 translation layer + high-level operator)

examples/
├── tutorial.ipynb                — Jupyter tutorial (37 cells, covers all API variants)
├── 3DPoisson.ipynb               — 3D tensor Poisson
└── gridap_workflow.ipynb         — Standard Gridap workflow for reference

test/
├── runtests.jl                   — Runs all test files
├── TensorProductQuadraturesTests.jl
├── TensorProductFESpaceInterfaceTests.jl
├── DOFOrderingTests.jl
├── FEFunctionTests.jl
├── KroneckerAssemblerTests.jl
└── PoissonEquivalenceTests.jl
```

---

## 11. Known Issues and Cautions

1. **`_get_all!` NTuple conversion:** `NTuple{N, Matrix}(cache)` where `cache::Vector{Union{Matrix,Nothing}}` —
   this succeeds at runtime only if all elements have been filled. If any `nothing` remains (i.e.
   `_assemble_*_k!` threw an error), the conversion will throw. Error messages in this case may be
   confusing.

2. **`NTuple{N, Matrix}` is abstract:** `Matrix` is `Array{T,2} where T`, so the return type of
   `_get_*_ops!` is type-unstable. For performance-critical code, consider storing concrete
   element types.

3. **`assemble_stiffness_tensor` etc. materialise via `collect()`:** The Kronecker lazy product is
   collected into a dense `Matrix{Float64}` at the end of each assembler. For large systems,
   consider returning the `KroneckerProduct` directly and using an iterative solver.

4. **`FallbackAssembler` is a stub:** `assemble_tensor_operator(::FallbackAssembler, op, V)` throws
   a hard error. It is not a working fallback for non-separable operators.

5. **2-arg weak form with `dΩ_tp` — restricted classification:** Writing
   `a(u,v) = ∫(∇u⋅∇v)*(dΩx⊗dΩy)` and passing it to `TensorProductAffineOperator` now
   works for mass and stiffness operators via `translate_bilinear_form`. However, operators
   that are not proportional to mass or stiffness (e.g. advection, variable-coefficient
   forms) will not be classified correctly and will fall through to `:unknown` label.
   The 3-arg form API remains the more robust path for non-standard operators.

6. **No boundary condition tensor coupling:** Dirichlet constraints are applied per subdomain
   before Kronecker assembly. A tensor DOF is free iff all its component DOFs are free. Non-trivial
   boundary conditions (e.g. Neumann on one face of the tensor domain) are not yet handled.

7. **`get_vector_type` on `TensorProductFESpace`:** Returns `Vector{Tuple{...}}` which may not be
   compatible with all Gridap solvers expecting `Vector{Float64}`.

---

## 12. What to Work on Next

In rough priority order:

1. **General weak form evaluation for `TensorProductMeasure`** (hardest, most impactful):
   Implement `integrate(::TensorProductIntegrand)` by building a Cartesian product
   quadrature rule from the component measures and evaluating the integrand at all
   tensor quadrature points. This unlocks arbitrary 2-arg weak forms.

2. **Improve the 3-arg heuristic:** The current product/sum rule (compare `A_k` to `M_k`)
   is fragile. A more robust approach would classify terms by inspecting the weak form
   function signature or by running a small symbolic analysis.

3. **Sparse output from Kronecker assemblers:** Currently all assemblers return dense
   `Matrix{Float64}`. Add a `sparse=true` kwarg to return `SparseMatrixCSC` and avoid
   materialising large dense arrays.

4. **Full `TensorProductFEFunction` with cell-field scatter:** Implement the `cell_field`
   field in `TensorProductFEFunction` so that solution vectors can be evaluated pointwise
   and visualised via Gridap's VTK output.

5. **`TensorProductTriangulation` as a proper Gridap `Triangulation`:** Implement the full
   Gridap triangulation interface so that `TensorProductTriangulation` can be used in
   standard Gridap weak forms without special-casing.
