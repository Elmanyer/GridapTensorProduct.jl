# CLAUDE.md — GridapTensorProduct.jl

This file is a comprehensive technical reference for AI agents working on this codebase.
It documents the current state of the library, module-by-module design, implementation
status, known limitations, and next-step priorities (as of 2026-05-20).

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

```
Stage 1: TensorProductMeasure        — bundle N subdomain quadrature measures
Stage 2: TensorProductFESpace        — bundle N subdomain FE spaces + DOF mapping
Stage 3: TensorProductGeometry       — lazy triangulation and cell-field wrappers
Stage 4: TensorProductFEMOperators   — extract 1D matrices; form Kronecker operators
Stage 5: TensorProductOperator       — explicit-label weak form → global system
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

**Symbol conflict:** Both Gridap and GridapTensorProduct export `⊗`. User scripts
must resolve with `import GridapTensorProduct: ⊗` after `using` both packages.

---

### 6.2 `TensorProductFESpaces.jl` — STATUS: COMPLETE AND SATISFACTORY

**Type:** `TensorProductFESpace{T<:Tuple} <: FESpace`

Bundles N independent 1D FE spaces into a single tensor-product space.

**Construction:**
```julia
V_tp = TensorProductFESpace(Vx, Vy)        # variadic
V_tp = TensorProductFESpace(Vx, Vy, Vz)    # 3D
V_tp = TensorProductFESpace((Vx, Vy))      # tuple form
```

**Key interface functions:**
- `num_free_dofs(V_tp)` → `prod(num_free_dofs(Vk))`
- `get_free_dof_ids(V_tp)` → `Base.OneTo(num_free_dofs(V_tp))`
- `get_triangulation(V_tp)` → `TensorProductTriangulation`
- `get_triangulations(V_tp)` → NTuple of per-subdomain triangulations
- `get_tensor_cell_ids(V_tp)`, `get_tensor_cell_dof_ids(V_tp)`

**`TensorProductFEFunction`:** Placeholder struct storing `free_values` and
`dirichlet_values`. Full cell-field scatter not yet implemented.

---

### 6.3 `TensorProductGeometry.jl` — STATUS: PARTIAL (structural wrapper only)

**Types:**
- `TensorProductTriangulation{T<:Tuple}` — wraps N component triangulations.
- `TensorProductCellField{CF<:Tuple, DATA}` — wraps N component CellFields.

Does NOT inherit from `Gridap.Geometry.Triangulation` (planned for future phase).

---

### 6.4 `TensorProductAssemblers.jl` — STATUS: LOW-LEVEL UTILITIES COMPLETE

**Types:**
- `KroneckerAssembler <: TensorProductAssembler` — separable Kronecker path
- `FallbackAssembler <: TensorProductAssembler` — stub; throws hard error

**Index conversion (exported, tested):**
```julia
tensor_tuple_to_linear_index((i1,i2,...), (n1,n2,...)) → Int
linear_index_to_tensor_tuple(linear, (n1,n2,...))      → NTuple
```

**Matrix/vector utilities:**
```julia
tensor_kron((A1, A2, ...))         # kron(A_N, ..., kron(A2, A1)) — innermost-first
tensor_kron_vector((v1, v2, ...))  # kron(v_N, ..., kron(v2, v1))
```

---

### 6.5 `TensorProductIntegration.jl` — STATUS: COMPLETE

Provides the Gridap hook and labeled contribution types for the explicit-label API.

#### `TensorProductIntegrand`
```julia
struct TensorProductIntegrand
    object::Any
    measure::TensorProductMeasure
end
Base.:*(integrand::Integrand, m::TensorProductMeasure) = TensorProductIntegrand(...)
```
`Gridap.CellData.integrate(::TensorProductIntegrand)` throws an error directing the
user to `TensorProductAffineOperator`.

#### `TensorProductDomainContribution` — labeled bilinear form recipe
```julia
struct TensorProductDomainContribution
    label::Symbol              # :mass, :stiffness, :gradient, :divergence, :curl_curl, :advection
    measure::TensorProductMeasure
    coefficient::Float64       # scalar multiplier (default 1.0)
    b_coeffs::Union{Nothing, NTuple}   # velocity components for :advection only
end
# Constructors:
TensorProductDomainContribution(label, measure; coefficient=1.0, b_coeffs=nothing)
TensorProductDomainContribution(intg::TensorProductIntegrand, label; ...)
```

| label        | subdomain ops assembled |
|:-------------|:------------------------|
| `:mass`      | M_k                     |
| `:stiffness` | M_k, K_k                |
| `:gradient`  | M_k, G_k                |
| `:divergence`| M_k, G_k (transposed)   |
| `:curl_curl` | M_k, D_k                |
| `:advection` | M_k, A_k (needs b_coeffs) |

#### `TensorProductLinearContribution` — labeled linear (RHS) form recipe
```julia
struct TensorProductLinearContribution
    measure::TensorProductMeasure
    rhs_forms::Union{Nothing, Tuple}   # nothing = unit load per subdomain
    coefficient::Float64
end
TensorProductLinearContribution(measure; rhs_forms=nothing, coefficient=1.0)
```

---

### 6.6 `TensorProductFEMOperators.jl` — STATUS: COMPLETE

**`TensorProductSubdomainOperators{N}`** — lazy container:
```julia
assemble_subdomain_operators(spaces, measures; b_coeffs=nothing)
```
Construction is O(1). `ops.M_ops`, `ops.K_ops`, etc. trigger lazy assembly on first
access and cache results. Each matrix type assembled at most once.

**Six Kronecker assemblers (all return `Matrix{Float64}`):**
```julia
assemble_mass_tensor(ops)
assemble_stiffness_tensor(ops)
assemble_gradient_tensor(ops)
assemble_divergence_tensor(ops)
assemble_curl_curl_tensor(ops)
assemble_advection_tensor(ops, b_coeffs)
```

**Helper extractors:**
```julia
extract_gradient_matrix(ops, k)
extract_derivative_matrix(ops, k)
extract_advection_matrix(ops, k)
```

---

### 6.7 `TensorProductOperator.jl` — STATUS: COMPLETE (explicit-label hierarchy)

This is the Stage 5 module. Contains the full explicit-label type hierarchy.

#### Type hierarchy (top = containing, bottom = contained)

```
TensorProductAffineOperator
  └── TensorProductOperator
        └── TensorProductWeakForm
              ├── lhs_terms :: Vector{TensorProductDomainContribution}
              └── rhs_terms :: Vector{TensorProductLinearContribution}
```

#### `TensorProductWeakForm`
```julia
struct TensorProductWeakForm
    lhs_terms::Vector{TensorProductDomainContribution}
    rhs_terms::Vector{TensorProductLinearContribution}
    # inner constructor validates N consistency across all contributions
end
num_lhs_terms(wf), num_rhs_terms(wf)
```

#### `TensorProductOperator` — lazy caching wrapper
```julia
mutable struct TensorProductOperator
    weak_form::TensorProductWeakForm
    test_space::TensorProductFESpace
    trial_space::TensorProductFESpace
    _matrix::Union{AbstractMatrix, Nothing}
    _vector::Union{AbstractVector, Nothing}
end
TensorProductOperator(wf, test_space, trial_space)
get_matrix(op)  # lazy — triggers _assemble_op! on first call
get_vector(op)
```

#### `TensorProductAffineOperator` — thin high-level wrapper
```julia
struct TensorProductAffineOperator
    weak_form::TensorProductWeakForm
    test_space::TensorProductFESpace
    trial_space::TensorProductFESpace
    _operator::TensorProductOperator
end
TensorProductAffineOperator(wf, test_space, trial_space)
get_matrix(op)  # delegates to _operator
get_vector(op)
```

**No quadrature degree or measure information** is stored in `TensorProductAffineOperator`;
those are embedded in the `TensorProductDomainContribution` objects.

#### Assembly internals (`_assemble_op!`)
1. Extract per-subdomain spaces and measures from `wf.lhs_terms[1].measure`
2. Build one shared `TensorProductSubdomainOperators` (lazy; each M_k, K_k, etc. assembled at most once)
3. Accumulate LHS: `A += dc.coefficient * _assemble_contribution(dc, subdomain_ops)` for each term
4. Assemble RHS: `_assemble_rhs(wf.rhs_terms, spaces)` → `kron(b_N, ..., b_1)`

`_assemble_contribution` dispatches on `dc.label` to the correct Kronecker assembler.

`_assemble_rhs` builds per-subdomain vectors via per-subdomain `rhs_forms[k]` (if given)
or unit load `∫(1*v)*dΩ_k` (fallback), then combines with `kron`.

---

### 6.8 `TensorProductAssembly.jl` — STATUS: COMPLETE (thin functional wrappers)

```julia
assemble_tensor_affine_operator(label, measure, U_tp, V_tp;
    coefficient=1.0, b_coeffs=nothing, rhs_forms=nothing)
# → (A::Matrix{Float64}, b::Vector{Float64})

assemble_tensor_system(label, measure, U_tp, V_tp; kwargs...)
# → (A, b, n_dofs)
```

These are convenience entry points for the single-term case. For multi-term weak
forms (Helmholtz, advection–diffusion), use the `TensorProductWeakForm` /
`TensorProductAffineOperator` API directly.

---

### 6.9 `GridapTensorProduct.jl` — Module Entry Point

Load order: `Measures → Integration → Geometry → FESpaces → Assemblers →
FEMOperators → Operator → Assembly`

Key exports:
```julia
# Stage 1 — Measures
TensorProductMeasure, ⊗

# Stage 2 — Integration types
TensorProductIntegrand
TensorProductDomainContribution
TensorProductLinearContribution

# Stage 3 — Geometry
TensorProductTriangulation, TensorProductCellField

# Stage 2 (load order) — FE Spaces
TensorProductFESpace

# Assembler utilities
TensorProductAssembler, KroneckerAssembler, FallbackAssembler
tensor_tuple_to_linear_index, linear_index_to_tensor_tuple
tensor_kron, tensor_kron_vector
apply_tensor_operator
assemble_tensor_rhs, assemble_tensor_operator

# Stage 4 — Subdomain operators + six Kronecker assemblers
TensorProductSubdomainOperators, assemble_subdomain_operators
assemble_mass_tensor, assemble_stiffness_tensor, assemble_gradient_tensor
assemble_divergence_tensor, assemble_curl_curl_tensor, assemble_advection_tensor
extract_gradient_matrix, extract_derivative_matrix, extract_advection_matrix

# Stage 5 — Explicit-label weak form + operator hierarchy
TensorProductWeakForm, num_lhs_terms, num_rhs_terms
TensorProductOperator
TensorProductAffineOperator
get_matrix, get_vector

# Functional convenience API
assemble_tensor_affine_operator, assemble_tensor_system
```

---

## 7. Workflow Recipes

### Poisson (2D)
```julia
using Gridap, GridapTensorProduct
import GridapTensorProduct: ⊗

model_x = CartesianDiscreteModel((0,1), 20)
model_y = CartesianDiscreteModel((0,1), 20)

Vx = TestFESpace(Interior(model_x), ReferenceFE(lagrangian, Float64, 1);
                  conformity=:H1, dirichlet_tags="boundary")
Vy = TestFESpace(Interior(model_y), ReferenceFE(lagrangian, Float64, 1);
                  conformity=:H1, dirichlet_tags="boundary")
V_tp = TensorProductFESpace(Vx, Vy)
U_tp = TensorProductFESpace(TrialFESpace(Vx, 0.0), TrialFESpace(Vy, 0.0))

dΩx = Measure(Interior(model_x), 2)
dΩy = Measure(Interior(model_y), 2)
dΩ_tp = dΩx ⊗ dΩy

lhs = [TensorProductDomainContribution(:stiffness, dΩ_tp)]
rhs = [TensorProductLinearContribution(dΩ_tp)]
op  = TensorProductAffineOperator(TensorProductWeakForm(lhs, rhs), V_tp, U_tp)
A   = get_matrix(op)
b   = get_vector(op)
u   = A \ b
```

### Helmholtz (K + k²M)
```julia
k = 2.5
lhs = [
    TensorProductDomainContribution(:stiffness, dΩ_tp),
    TensorProductDomainContribution(:mass, dΩ_tp; coefficient=k^2)
]
rhs = [TensorProductLinearContribution(dΩ_tp)]
op  = TensorProductAffineOperator(TensorProductWeakForm(lhs, rhs), V_tp, U_tp)
```

### Advection–diffusion
```julia
lhs = [
    TensorProductDomainContribution(:stiffness, dΩ_tp),
    TensorProductDomainContribution(:advection, dΩ_tp; b_coeffs=(2.0, 1.5))
]
```

### Custom separable RHS
```julia
rhs = [TensorProductLinearContribution(dΩ_tp;
        rhs_forms=(v -> ∫(f_x * v) * dΩx, v -> ∫(f_y * v) * dΩy))]
```

### Functional shortcut (single-term)
```julia
A, b = assemble_tensor_affine_operator(:stiffness, dΩ_tp, U_tp, V_tp)
A, b = assemble_tensor_affine_operator(:mass, dΩ_tp, U_tp, V_tp; coefficient=k^2)
```

### Low-level: direct Kronecker assembly
```julia
ops = assemble_subdomain_operators((Vx, Vy), (dΩx, dΩy))
M   = assemble_mass_tensor(ops)       # only assembles M_k
A   = assemble_stiffness_tensor(ops)  # assembles M_k (cached) + K_k
```

---

## 8. Test Suite

Located in `test/`. Run individual files (not `runtests.jl`) due to the pre-existing
`TensorQuadrature` error in `TensorProductQuadraturesTests.jl`.

```bash
julia --project test/DOFOrderingTests.jl
julia --project test/KroneckerAssemblerTests.jl
julia --project test/PoissonEquivalenceTests.jl
julia --project test/TensorProductFESpaceInterfaceTests.jl
```

| Test file | Status | What it tests |
|---|---|---|
| `TensorProductQuadraturesTests.jl` | KNOWN FAILURE (`TensorQuadrature` undefined) | Quadrature accessors |
| `TensorProductFESpaceInterfaceTests.jl` | PASS | DOF counts, cell IDs, triangulation |
| `DOFOrderingTests.jl` | PASS | Bijection `tensor_tuple ↔ linear_index` for N=1,2,3 |
| `FEFunctionTests.jl` | PASS | `TensorProductFEFunction` construction |
| `KroneckerAssemblerTests.jl` | PASS | `tensor_kron`, `apply_tensor_operator` |
| `PoissonEquivalenceTests.jl` | PASS | Tensor DOF count; Kronecker vs. explicit `kron` |

**Additional smoke test (2026-05-20, verified passing):**
- Stiffness equivalence: `norm(A_tp - (kron(My,Kx)+kron(Ky,Mx))) == 0.0`
- Helmholtz equivalence: `norm(A_helm - (A_stiff + k²*kron(My,Mx))) == 0.0`
- Poisson solve: non-trivial solution vector
- Functional API consistency: `assemble_tensor_affine_operator` matches `TensorProductAffineOperator`

---

## 9. Known Issues and Cautions

1. **`⊗` symbol conflict:** Both `Gridap` and `GridapTensorProduct` export `⊗`. In user
   scripts that `using` both packages, resolve with `import GridapTensorProduct: ⊗`.

2. **`get_matrix`/`get_vector` extension:** `TensorProductOperator.jl` imports and extends
   `Gridap.FESpaces.get_matrix` and `Gridap.FESpaces.get_vector`. This is required so
   that internal assembly calls like `get_vector(AffineFEOperator(...))` dispatch correctly.

3. **`assemble_weak_form` spaces argument typed as `Tuple`:** Subdomain spaces may have
   heterogeneous concrete types (P1 on x, P2 on y). Julia's `NTuple{N, T} where T`
   constraint fails for such tuples; `Tuple` with `N = length(spaces)` is correct.

4. **Materialised dense matrices:** All Kronecker assemblers currently collect into
   dense `Matrix{Float64}`. For large systems (N>2 or many DOFs), return the
   `KroneckerProduct` object directly and use iterative solvers.

5. **`FallbackAssembler` is a stub:** Throws a hard error.

6. **`TensorProductTriangulation` not a Gridap `Triangulation`:** Does not implement
   the full Gridap triangulation interface.

7. **No boundary condition tensor coupling:** Dirichlet constraints applied per subdomain.

8. **Pre-existing test failure:** `TensorProductQuadraturesTests.jl` references
   `TensorQuadrature` which is not defined anywhere in the codebase; skip this test.

---

## 10. What to Work on Next

In rough priority order:

1. **Update examples directory:** `examples/PoissonTensorProduct2D_Fixed.jl`,
   `examples/PoissonTensorProduct3D.jl`, `examples/PoissonTensorProductHeterogeneous.jl`
   use the OLD `assemble_tensor_affine_operator(a::Function, l::Function, U, V; op_type)`
   API. Must be updated to the new explicit-label API.

2. **Sparse output from Kronecker assemblers:** Add `sparse=true` kwarg to all six
   Kronecker assemblers to return `SparseMatrixCSC` for large problems.

3. **Full `TensorProductFEFunction` with cell-field scatter:** Implement the `cell_field`
   field so solutions can be visualised via Gridap VTK.

4. **`TensorProductTriangulation` as proper Gridap `Triangulation`:** Full interface
   implementation for use in standard Gridap weak forms.

5. **2-arg RHS translation:** Decompose `l(v) = ∫(f*v)*dΩ_tp` into per-subdomain RHS
   vectors for separable `f`.

---

## 11. File Index

```
src/
├── GridapTensorProduct.jl        — module entry point, all exports
├── TensorProductMeasures.jl      — TensorProductMeasure, ⊗ operator
│                                   STATUS: COMPLETE AND SATISFACTORY
├── TensorProductIntegration.jl   — TensorProductIntegrand (∫*dΩ_tp hook),
│                                   TensorProductDomainContribution (labeled bilinear),
│                                   TensorProductLinearContribution (labeled RHS)
│                                   STATUS: COMPLETE
├── TensorProductGeometry.jl      — TensorProductTriangulation, TensorProductCellField
│                                   STATUS: STRUCTURAL WRAPPERS ONLY
├── TensorProductFESpaces.jl      — TensorProductFESpace, TensorProductFEFunction
│                                   STATUS: COMPLETE AND SATISFACTORY
├── TensorProductAssemblers.jl    — KroneckerAssembler, FallbackAssembler,
│                                   tensor_kron, index conversion utilities
│                                   STATUS: COMPLETE (low-level utilities)
├── TensorProductFEMOperators.jl  — TensorProductSubdomainOperators (lazy),
│                                   six Kronecker assemblers (mass/stiff/grad/div/curl/adv),
│                                   extract_* helpers
│                                   STATUS: COMPLETE (lazy assembly, all 6 operators)
├── TensorProductOperator.jl      — TensorProductWeakForm,
│                                   TensorProductOperator (lazy, mutable),
│                                   TensorProductAffineOperator (thin wrapper),
│                                   _assemble_op!, _assemble_contribution, _assemble_rhs
│                                   STATUS: COMPLETE (explicit-label hierarchy)
└── TensorProductAssembly.jl      — assemble_tensor_affine_operator (functional wrapper),
                                    assemble_tensor_system
                                    STATUS: COMPLETE

examples/
├── tutorial.ipynb                — Jupyter tutorial (uses OLD API — needs update)
├── 3DPoisson.ipynb               — 3D tensor Poisson (uses OLD API — needs update)
├── gridap_workflow.ipynb         — Standard Gridap workflow for reference
├── PoissonTensorProduct2D_Fixed.jl  — NEEDS UPDATE to new API
├── PoissonTensorProduct3D.jl        — NEEDS UPDATE to new API
└── PoissonTensorProductHeterogeneous.jl — NEEDS UPDATE to new API

test/
├── runtests.jl                   — Runs all test files (fails on Quadratures; run others individually)
├── TensorProductQuadraturesTests.jl  — KNOWN FAILURE (TensorQuadrature undefined)
├── TensorProductFESpaceInterfaceTests.jl — PASS
├── DOFOrderingTests.jl           — PASS
├── FEFunctionTests.jl            — PASS
├── KroneckerAssemblerTests.jl    — PASS
└── PoissonEquivalenceTests.jl    — PASS
```
