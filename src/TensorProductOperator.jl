"""
    TensorProductOperator

Stage 4 of the tensor product FEM pipeline: translation of user-written weak forms
into Kronecker-factored operators ready for assembly.

# Workflow

```julia
# 1. Define separable terms (3-arg form convention)
stiff = TensorProductSeparableTerm((u, v, dΩ) -> ∫(∇(u)⋅∇(v)) * dΩ, :stiffness)
mass  = TensorProductSeparableTerm((u, v, dΩ) -> ∫(u*v) * dΩ,        :mass)

# 2. Compose into a weak form
wf = stiff + mass

# 3. Translate to a TensorProductOperator
op = TensorProductOperator(wf, (Vx, Vy), (dΩx, dΩy))

# 4. Assemble (lazy: triggered on first access)
A = get_global_matrix(op)
```

Alternatively, the explicit operator-type API is still supported:
```julia
op = TensorProductOperator(:stiffness, (Vx, Vy), (dΩx, dΩy))
A  = get_global_matrix(op)
```
"""

import LinearAlgebra: kron, norm

export TensorProductSeparableTerm
export TensorProductWeakForm
export num_terms, get_terms
export assemble_weak_form
export normalize_to_list, assemble_subdomain_matrix, classify_term
export translate_bilinear_form

export TensorProductOperator
export get_global_matrix
export get_subdomain_operators
export get_operator_type

export TensorProductAffineOperator

import Gridap.FESpaces: TestFESpace, collect_cell_matrix, SparseMatrixAssembler,
                        assemble_matrix, get_matrix, get_vector
import Gridap.CellData: Measure
import Gridap


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductSeparableTerm
# ═══════════════════════════════════════════════════════════════════════════

"""
    struct TensorProductSeparableTerm

A single separable term in a tensor product weak form.

`form(u, v, dΩ)` is a 3-argument function: given 1D FEBasis objects and a 1D
Gridap `Measure`, it returns a standard Gridap `DomainContribution`. The library
calls this independently on each subdomain Ω_k, obtains per-subdomain matrices,
then combines them via Kronecker products.

# Fields
- `form::Function` — 3-arg `(u, v, dΩ::Measure) -> DomainContribution`
- `label::Symbol` — operator type hint (`:mass`, `:stiffness`, `:advection`, `:unknown`)
- `coefficient::Float64` — scalar multiplier applied at assembly time (default 1.0)

# Example
```julia
stiff = TensorProductSeparableTerm((u, v, dΩ) -> ∫(∇(u)⋅∇(v)) * dΩ, :stiffness)
mass  = TensorProductSeparableTerm((u, v, dΩ) -> ∫(u*v) * dΩ,        :mass)
wf    = stiff + mass
```
"""
# Dual-representation struct:
# - Track A: form ≠ nothing — 3-arg (u,v,dΩ) function, evaluated per subdomain.
# - Track B: subdomain_matrices ≠ nothing — pre-assembled matrices from 2-arg translation.
struct TensorProductSeparableTerm
    form::Union{Function, Nothing}
    subdomain_matrices::Union{Vector{Matrix{Float64}}, Nothing}
    label::Symbol       # :mass, :stiffness, :advection, :unknown
    coefficient::Float64
    kronecker_rule::Symbol  # :product, :sum, :unknown (explicit rule for Track B)
end

# ── Track A constructors (3-arg form) ─────────────────────────────────────
TensorProductSeparableTerm(form::Function) =
    TensorProductSeparableTerm(form, nothing, :unknown, 1.0, :unknown)
TensorProductSeparableTerm(form::Function, label::Symbol) =
    TensorProductSeparableTerm(form, nothing, label, 1.0, :unknown)
TensorProductSeparableTerm(form::Function, label::Symbol, coeff::Float64) =
    TensorProductSeparableTerm(form, nothing, label, coeff, :unknown)

# ── Track B constructor (pre-assembled per-subdomain matrices) ─────────────
TensorProductSeparableTerm(
    mats::Vector{Matrix{Float64}}, label::Symbol, rule::Symbol
) = TensorProductSeparableTerm(nothing, mats, label, 1.0, rule)

TensorProductSeparableTerm(
    mats::Vector{Matrix{Float64}}, label::Symbol, rule::Symbol, coeff::Float64
) = TensorProductSeparableTerm(nothing, mats, label, coeff, rule)


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductWeakForm
# ═══════════════════════════════════════════════════════════════════════════

"""
    mutable struct TensorProductWeakForm

Intermediary container that translates user-written tensor-product weak forms into
`TensorProductOperator` objects via Kronecker decomposition.

Users compose weak forms as sums of `TensorProductSeparableTerm` objects using
the high-level `∫() * dΩ` API:

```julia
stiff = TensorProductSeparableTerm((u, v, dΩ) -> ∫(∇(u)⋅∇(v)) * dΩ, :stiffness)
mass  = TensorProductSeparableTerm((u, v, dΩ) -> ∫(u*v) * dΩ,        :mass)
wf    = stiff + mass           # Helmholtz-type form

op = TensorProductOperator(wf, (Vx, Vy), (dΩx, dΩy))
A  = get_global_matrix(op)
```

# Fields
- `terms::Vector{TensorProductSeparableTerm}` — list of separable terms
- `_kronecker_rules::Union{Nothing, Vector{Symbol}}` — cached per-term Kronecker
  classification (`:product` or `:sum`), populated by `assemble_weak_form`
"""
mutable struct TensorProductWeakForm
    terms::Vector{TensorProductSeparableTerm}
    _kronecker_rules::Union{Nothing, Vector{Symbol}}
end

# ── Constructors ──────────────────────────────────────────────────────────

TensorProductWeakForm(terms::Vector{TensorProductSeparableTerm}) =
    TensorProductWeakForm(terms, nothing)

TensorProductWeakForm(term::TensorProductSeparableTerm) =
    TensorProductWeakForm([term], nothing)

TensorProductWeakForm(form::Function) =
    TensorProductWeakForm(TensorProductSeparableTerm(form))

TensorProductWeakForm(forms::AbstractVector{<:Function}) =
    TensorProductWeakForm([TensorProductSeparableTerm(f) for f in forms])

# ── Composition operators ─────────────────────────────────────────────────

Base.:+(t1::TensorProductSeparableTerm, t2::TensorProductSeparableTerm) =
    TensorProductWeakForm([t1, t2])

Base.:+(wf::TensorProductWeakForm, t::TensorProductSeparableTerm) =
    TensorProductWeakForm([wf.terms..., t])

Base.:+(t::TensorProductSeparableTerm, wf::TensorProductWeakForm) =
    TensorProductWeakForm([t, wf.terms...])

Base.:+(wf1::TensorProductWeakForm, wf2::TensorProductWeakForm) =
    TensorProductWeakForm([wf1.terms..., wf2.terms...])

# ── Accessors ─────────────────────────────────────────────────────────────

"""
    num_terms(wf::TensorProductWeakForm) -> Int

Return the number of separable terms in this weak form.
"""
num_terms(wf::TensorProductWeakForm) = length(wf.terms)

"""
    get_terms(wf::TensorProductWeakForm) -> Vector{TensorProductSeparableTerm}

Return the list of separable terms.
"""
get_terms(wf::TensorProductWeakForm) = wf.terms

function Base.show(io::IO, wf::TensorProductWeakForm)
    n = num_terms(wf)
    rules = wf._kronecker_rules
    rule_str = rules === nothing ? "not yet analyzed" :
               join(string.(rules), " + ")
    print(io, "TensorProductWeakForm($n term$(n==1 ? "" : "s"), Kronecker rules: $rule_str)")
end


# ═══════════════════════════════════════════════════════════════════════════
# assemble_weak_form — translate weak form to global matrix
# ═══════════════════════════════════════════════════════════════════════════

"""
    assemble_weak_form(wf::TensorProductWeakForm,
                       spaces::NTuple{N,<:FESpace},
                       measures::NTuple{N,Measure}) -> Matrix

Assemble the global matrix for all separable terms in `wf`.

For each `TensorProductSeparableTerm` in `wf.terms`:
1. Evaluate `term.form(u_k, v_k, dΩ_k)` on each subdomain k → per-subdomain matrix `A_k`
2. Apply product/sum heuristic by comparing `A_k` to the reference mass `M_k`:
   - If `‖A_k − M_k‖ ≤ tol` for ALL k → **product rule**: `global += coeff * kron(A_N,…,A_1)`
   - Otherwise → **sum rule**: `global += coeff * Σ_k kron(M_N,…,A_k,…,M_1)`
3. Accumulate contributions scaled by `term.coefficient`.

The per-term Kronecker rules are cached in `wf._kronecker_rules` after assembly.
Reference mass matrices are computed once and shared across all sum-rule terms.

# Arguments
- `wf` — `TensorProductWeakForm` (one or more separable terms)
- `spaces` — N test spaces (one per subdomain)
- `measures` — N integration measures (one per subdomain)

# Returns
Dense `Matrix{Float64}` of size `(∏ nk) × (∏ nk)` where `nk = num_free_dofs(spaces[k])`.
"""
function assemble_weak_form(
    wf::TensorProductWeakForm,
    spaces::NTuple{N,<:FESpace},
    measures::NTuple{N,Measure}
) where N
    # Compute reference mass matrices once — shared passive factors in sum rule.
    M_ops = ntuple(N) do k
        Uk = TrialFESpace(spaces[k])
        Matrix(get_matrix(AffineFEOperator(
            (u, v) -> ∫(u * v) * measures[k],
            (v)    -> ∫(0 * v) * measures[k],
            Uk, spaces[k])))
    end

    n_total  = prod(size(M_ops[k], 1) for k in 1:N)
    A_global = zeros(n_total, n_total)
    rules    = Symbol[]

    # K_ops computed lazily — only when the heuristic needs it.
    K_ops_cache = nothing
    function get_K_ops()
        if K_ops_cache === nothing
            K_ops_cache = ntuple(N) do k
                Uk = TrialFESpace(spaces[k])
                Matrix(get_matrix(AffineFEOperator(
                    (u, v) -> ∫(∇(u) ⋅ ∇(v)) * measures[k],
                    (v)    -> ∫(0 * v) * measures[k],
                    Uk, spaces[k])))
            end
        end
        return K_ops_cache
    end

    for term in wf.terms
        coeff = term.coefficient

        # Obtain per-subdomain matrices: Track A (evaluate form) or Track B (use stored).
        A_ops = if term.form !== nothing
            ntuple(N) do k
                Uk = TrialFESpace(spaces[k])
                Matrix(get_matrix(AffineFEOperator(
                    (u, v) -> term.form(u, v, measures[k]),
                    (v)    -> ∫(0 * v) * measures[k],
                    Uk, spaces[k])))
            end
        else
            @assert length(term.subdomain_matrices) == N "Expected $N subdomain matrices, got $(length(term.subdomain_matrices))"
            ntuple(k -> term.subdomain_matrices[k], N)
        end

        # Determine rule and effective matrices.
        # Track B with explicit rule: stored matrices are already normalized; use directly.
        # Track A or Track B(:unknown): classify via proportionality heuristic.
        eff_A_ops, eff_coeff, is_product = if term.kronecker_rule === :product
            A_ops, coeff, true
        elseif term.kronecker_rule === :sum
            A_ops, coeff, false
        else
            # Run classify_term to detect scaled mass / scaled stiffness.
            lbl, rule, c = classify_term(A_ops, M_ops, get_K_ops())
            norm_ops = if lbl === :mass
                M_ops
            elseif lbl === :stiffness
                get_K_ops()
            else
                A_ops
            end
            norm_ops, coeff * c, (rule === :product)
        end
        push!(rules, is_product ? :product : :sum)

        if is_product
            result = eff_A_ops[1]
            for k in 2:N
                result = kronecker(eff_A_ops[k], result)
            end
            A_global .+= eff_coeff .* collect(result)
        else
            for k in 1:N
                t = (k == 1) ? eff_A_ops[1] : M_ops[1]
                for j in 2:N
                    t = kronecker((j == k) ? eff_A_ops[j] : M_ops[j], t)
                end
                A_global .+= eff_coeff .* collect(t)
            end
        end
    end

    wf._kronecker_rules = rules
    return A_global
end


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductOperator Type Definition
# ═══════════════════════════════════════════════════════════════════════════

"""
    mutable struct TensorProductOperator{OP_TYPE}

Wrapper for a translated tensor-product operator.

# Type Parameters
- `OP_TYPE`: Symbol indicating operator type (`:mass`, `:stiffness`, `:weak_form`, etc.)

# Fields
- `subdomain_ops::TensorProductSubdomainOperators` — all fundamental matrices (M, K, G, D, A)
- `operator_type::Symbol` — one of: `:mass`, `:stiffness`, `:gradient`, `:divergence`,
  `:curl_curl`, `:advection`, `:weak_form`
- `global_matrix::Union{Matrix, Nothing}` — cached global matrix (computed once on-demand)
- `n_subdomains::Int` — number of subdomains N
- `weak_form::Union{TensorProductWeakForm, Nothing}` — stored weak form (only for `:weak_form` type)

# Supported Operator Types
- `:mass`      → `M = M⁽¹⁾ ⊗ M⁽²⁾ ⊗ ... ⊗ M⁽ᴺ⁾`
- `:stiffness` → `A = Σ_k [M⁽¹⁾ ⊗ ... ⊗ K⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾]`
- `:gradient`  → `G = ⊕_k [M⁽¹⁾ ⊗ ... ⊗ G⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾]`
- `:divergence` → `B = Gᵀ`
- `:curl_curl` → anti-symmetric sum
- `:advection` → `T = Σ_k bₖ [M⁽¹⁾ ⊗ ... ⊗ A⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾]`
- `:weak_form` → assembled via `assemble_weak_form(op.weak_form, spaces, measures)`
"""
mutable struct TensorProductOperator{OP_TYPE}
    subdomain_ops::TensorProductSubdomainOperators
    operator_type::Symbol
    global_matrix::Union{Matrix, Nothing}
    n_subdomains::Int
    weak_form::Union{TensorProductWeakForm, Nothing}
end


# ═══════════════════════════════════════════════════════════════════════════
# Constructors
# ═══════════════════════════════════════════════════════════════════════════

"""
    TensorProductOperator(
        operator_type::Symbol,
        spaces::NTuple{N, FESpace},
        measures::NTuple{N, Measure};
        b_coeffs = nothing
    ) -> TensorProductOperator

Construct a tensor-product operator with an explicit operator type.

# Arguments
- `operator_type` — one of `:mass`, `:stiffness`, `:gradient`, `:divergence`,
  `:curl_curl`, `:advection`
- `spaces` — N test spaces (one per subdomain)
- `measures` — N integration measures
- `b_coeffs` — velocity coefficients (required when `operator_type = :advection`)

# Example
```julia
op = TensorProductOperator(:stiffness, (Vx, Vy), (dΩx, dΩy))
A  = get_global_matrix(op)
```
"""
function TensorProductOperator(
    operator_type::Symbol,
    spaces::NTuple{N, FESpace},
    measures::NTuple{N, Measure};
    b_coeffs::Union{NTuple, Nothing}=nothing
) where {N}
    valid_types = (:mass, :stiffness, :gradient, :divergence, :curl_curl, :advection)
    @assert operator_type ∈ valid_types "Unsupported operator type: $operator_type. Must be one of $valid_types"
    @assert !isempty(spaces) "Need at least one space"
    @assert all(s -> s isa Gridap.FESpaces.FESpace, spaces) "All spaces must be FESpace instances"
    @assert length(spaces) == length(measures) "Number of spaces and measures must match"

    subdomain_ops = assemble_subdomain_operators(spaces, measures, b_coeffs=b_coeffs)
    return TensorProductOperator{operator_type}(subdomain_ops, operator_type, nothing, N, nothing)
end

"""
    TensorProductOperator(
        wf::TensorProductWeakForm,
        spaces::NTuple{N, FESpace},
        measures::NTuple{N, Measure}
    ) -> TensorProductOperator{:weak_form}

Construct a tensor-product operator from a `TensorProductWeakForm`.

The weak form is stored and evaluated lazily when `get_global_matrix` is first called.
Each separable term is evaluated on each 1D subdomain and assembled via the
product/sum Kronecker heuristic.

# Example
```julia
stiff = TensorProductSeparableTerm((u, v, dΩ) -> ∫(∇(u)⋅∇(v)) * dΩ, :stiffness)
mass  = TensorProductSeparableTerm((u, v, dΩ) -> ∫(u*v) * dΩ,        :mass)
wf    = stiff + mass

op = TensorProductOperator(wf, (Vx, Vy), (dΩx, dΩy))
A  = get_global_matrix(op)
```
"""
function TensorProductOperator(
    wf::TensorProductWeakForm,
    spaces::NTuple{N, FESpace},
    measures::NTuple{N, Measure}
) where N
    @assert !isempty(spaces) "Need at least one space"
    @assert length(spaces) == length(measures) "Number of spaces and measures must match"

    subdomain_ops = assemble_subdomain_operators(spaces, measures)
    return TensorProductOperator{:weak_form}(subdomain_ops, :weak_form, nothing, N, wf)
end


# ═══════════════════════════════════════════════════════════════════════════
# Accessors
# ═══════════════════════════════════════════════════════════════════════════

"""
    get_global_matrix(op::TensorProductOperator) -> Matrix

Assemble and return the global matrix for this operator. Cached after first call.
"""
function get_global_matrix(op::TensorProductOperator)
    if op.global_matrix === nothing
        op.global_matrix = _assemble_operator_matrix(op)
    end
    return op.global_matrix
end

"""
    get_subdomain_operators(op::TensorProductOperator) -> TensorProductSubdomainOperators

Return the fundamental subdomain operators (M, K, G, D, A).
"""
function get_subdomain_operators(op::TensorProductOperator)
    return op.subdomain_ops
end

"""
    get_operator_type(op::TensorProductOperator) -> Symbol

Return the operator type classification (`:mass`, `:stiffness`, `:weak_form`, etc.)
"""
function get_operator_type(op::TensorProductOperator)
    return op.operator_type
end


# ═══════════════════════════════════════════════════════════════════════════
# Internal: Kronecker Assembly Dispatch
# ═══════════════════════════════════════════════════════════════════════════

function _assemble_operator_matrix(op::TensorProductOperator)
    ops = op.subdomain_ops

    if op.operator_type == :weak_form
        return assemble_weak_form(op.weak_form, ops.spaces, ops.measures)
    elseif op.operator_type == :mass
        return assemble_mass_tensor(ops)
    elseif op.operator_type == :stiffness
        return assemble_stiffness_tensor(ops)
    elseif op.operator_type == :gradient
        return assemble_gradient_tensor(ops)
    elseif op.operator_type == :divergence
        return assemble_divergence_tensor(ops)
    elseif op.operator_type == :curl_curl
        return assemble_curl_curl_tensor(ops)
    elseif op.operator_type == :advection
        return assemble_advection_tensor(ops)
    else
        error("Unknown operator type: $(op.operator_type)")
    end
end


# ═══════════════════════════════════════════════════════════════════════════
# Helpers for 2-arg weak form translation
# ═══════════════════════════════════════════════════════════════════════════

"""
    normalize_to_list(result) -> Vector{TensorProductIntegrand}

Convert whatever `a(u_k, v_k)` returns into a flat list of `TensorProductIntegrand`
objects, one per additive term.
"""
function normalize_to_list(result)::Vector{TensorProductIntegrand}
    if result isa TensorProductIntegrand
        return [result]
    elseif result isa TensorProductFormContribution
        return result.terms
    else
        error("translate_bilinear_form: unexpected result type $(typeof(result)) from bilinear form evaluation. " *
              "Make sure the form is written over a TensorProductMeasure (dΩ_tp = dΩ1 ⊗ dΩ2 ...).")
    end
end

"""
    assemble_subdomain_matrix(dc, U_k, V_k) -> Matrix{Float64}

Assemble the FE matrix from a pre-computed `DomainContribution` on a single subdomain.
Uses `collect_cell_matrix` to convert the `DomainContribution` to the matdata format
expected by `assemble_matrix`, bypassing the need to re-evaluate the bilinear form.
"""
function assemble_subdomain_matrix(dc::DomainContribution, U_k::FESpace, V_k::FESpace)
    matdata  = collect_cell_matrix(U_k, V_k, dc)
    assembler = SparseMatrixAssembler(U_k, V_k)
    return Matrix(assemble_matrix(assembler, matdata))
end

"""
    classify_term(A_ops, M_ops, K_ops; tol=1e-12) -> (label::Symbol, rule::Symbol, coeff::Float64)

Classify a tensor-product term by comparing its per-subdomain matrices `A_ops`
against reference mass `M_ops` and stiffness `K_ops`.

Detects scalar-weighted operators: `A_k ≈ c·M_k` or `A_k ≈ c·K_k` for any `c`.
The returned `coeff` is the proportionality constant; the caller should store
the **normalized** reference matrices (`M_k` / `K_k`) together with `coeff` so
that `assemble_weak_form` computes `coeff × kron(M_N,…,M_1)` (or the sum rule
equivalent) without squaring the coefficient.

| Condition | label | rule | coeff |
|---|---|---|---|
| `A_k ≈ c·M_k` for ALL k (same c) | `:mass`      | `:product` | `c` |
| `A_k ≈ c·K_k` for ALL k (same c) | `:stiffness` | `:sum`     | `c` |
| otherwise                         | `:unknown`   | `:sum`     | `1.0` |
"""
function classify_term(
    A_ops::NTuple{N, Matrix{Float64}},
    M_ops::NTuple{N, Matrix{Float64}},
    K_ops::NTuple{N, Matrix{Float64}};
    tol::Float64 = 1e-10
) where N
    # Frobenius-inner-product proportionality: c = <A,B>_F / <B,B>_F
    _c(A, B) = dot(vec(A), vec(B)) / max(dot(vec(B), vec(B)), eps())

    # Check A_k ≈ c·M_k for all k (same c)
    c0_m = _c(A_ops[1], M_ops[1])
    is_mass = all(1:N) do k
        ck = _c(A_ops[k], M_ops[k])
        abs(ck - c0_m) ≤ tol * (abs(c0_m) + 1) &&
        norm(A_ops[k] - ck * M_ops[k]) ≤ tol * (norm(M_ops[k]) + 1)
    end
    is_mass && return :mass, :product, c0_m

    # Check A_k ≈ c·K_k for all k (same c)
    c0_k = _c(A_ops[1], K_ops[1])
    is_stiff = all(1:N) do k
        ck = _c(A_ops[k], K_ops[k])
        abs(ck - c0_k) ≤ tol * (abs(c0_k) + 1) &&
        norm(A_ops[k] - ck * K_ops[k]) ≤ tol * (norm(K_ops[k]) + 1)
    end
    is_stiff && return :stiffness, :sum, c0_k

    return :unknown, :sum, 1.0
end

"""
    translate_bilinear_form(a, V_tp, quad_order) -> TensorProductWeakForm

Translate a user-supplied bilinear form `a(u,v)` (closed over a
`TensorProductMeasure`) into a `TensorProductWeakForm` containing labelled,
Kronecker-classified `TensorProductSeparableTerm`s.

# Algorithm

**Fast path (3-arg form):** If `a` accepts `(u, v, dΩ::Measure)`, wrap it as a
single `TensorProductSeparableTerm` with Track A representation and return immediately.
No per-subdomain analysis is performed.

**Translation path (2-arg form closed over `dΩ_tp`):**
1. For each subdomain k, evaluate `a(u_k, v_k)` using the 1D basis functions of
   subdomain k. The `∫(expr)*dΩ_tp` hooks fire and return `TensorProductIntegrand`
   objects; `+` between them produces a `TensorProductFormContribution`.
2. Re-integrate each term's `object` against the standard 1D measure `dΩ_k` to get
   a per-subdomain `DomainContribution`.
3. Assemble the per-subdomain matrix from the `DomainContribution` via
   `collect_cell_matrix` + `assemble_matrix`.
4. Classify each term as `:mass`/`:product`, `:stiffness`/`:sum`, or `:unknown`/`:sum`
   by comparing against reference mass and stiffness matrices.
5. Build `TensorProductSeparableTerm`s with Track B (pre-assembled matrices) and return
   a `TensorProductWeakForm`.

# Arguments
- `a`           — bilinear form: either `(u,v,dΩ)->...` (3-arg) or `(u,v)->...` (2-arg)
- `V_tp`        — test `TensorProductFESpace` (trial spaces derived via `TrialFESpace`)
- `quad_order`  — quadrature degree used to build per-subdomain measures
"""
function translate_bilinear_form(
    a::Function,
    V_tp::TensorProductFESpace,
    quad_order::Int
)
    # Fast path: 3-arg form is already in the format assemble_weak_form expects.
    if hasmethod(a, Tuple{Any, Any, Gridap.CellData.Measure})
        return TensorProductWeakForm([TensorProductSeparableTerm(a)])
    end

    N = length(V_tp.spaces)
    spaces_test = V_tp.spaces
    measures = ntuple(
        k -> Measure(Gridap.Geometry.get_triangulation(spaces_test[k]), quad_order), N)

    # Step 1: evaluate on each subdomain; capture terms as TensorProductIntegrand list.
    term_lists = Vector{Vector{TensorProductIntegrand}}(undef, N)
    for k in 1:N
        u_k = get_trial_fe_basis(TrialFESpace(spaces_test[k]))
        v_k = get_fe_basis(spaces_test[k])
        term_lists[k] = normalize_to_list(a(u_k, v_k))
    end

    n_terms = length(term_lists[1])
    @assert all(k -> length(term_lists[k]) == n_terms, 1:N) "Bilinear form produced a different number of terms per subdomain (expected $n_terms for all)"

    # Step 2: reference mass and stiffness for classification (assembled once).
    M_ref = ntuple(N) do k
        Uk = TrialFESpace(spaces_test[k])
        Matrix(get_matrix(AffineFEOperator(
            (u, v) -> ∫(u * v) * measures[k],
            (v)    -> ∫(0 * v) * measures[k],
            Uk, spaces_test[k])))
    end
    K_ref = ntuple(N) do k
        Uk = TrialFESpace(spaces_test[k])
        Matrix(get_matrix(AffineFEOperator(
            (u, v) -> ∫(∇(u) ⋅ ∇(v)) * measures[k],
            (v)    -> ∫(0 * v) * measures[k],
            Uk, spaces_test[k])))
    end

    # Steps 3–5: for each term, assemble per-subdomain matrices, classify, and store.
    tp_terms = TensorProductSeparableTerm[]
    for j in 1:n_terms
        A_mats = Matrix{Float64}[]
        for k in 1:N
            intg = term_lists[k][j]
            dc   = ∫(intg.object) * measures[k]   # standard Gridap DomainContribution
            A_kj = assemble_subdomain_matrix(dc, TrialFESpace(spaces_test[k]), spaces_test[k])
            push!(A_mats, A_kj)
        end

        A_ntuple        = ntuple(k -> A_mats[k], N)
        label, rule, c  = classify_term(A_ntuple, M_ref, K_ref)

        # Store NORMALIZED reference matrices to avoid squaring the coefficient
        # when the Kronecker product is formed (see classify_term docstring).
        stored_mats = if label == :mass
            [M_ref[k] for k in 1:N]
        elseif label == :stiffness
            [K_ref[k] for k in 1:N]
        else
            A_mats
        end

        push!(tp_terms, TensorProductSeparableTerm(stored_mats, label, rule, Float64(c)))
    end

    return TensorProductWeakForm(tp_terms)
end


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductAffineOperator
# ═══════════════════════════════════════════════════════════════════════════

"""
    mutable struct TensorProductAffineOperator

High-level tensor-product analogue of Gridap's `AffineFEOperator`. Lazily assembles
the global `(A, b)` system via Kronecker factorisation on first access to
`get_matrix(op)` or `get_vector(op)`.

Internally wraps a `TensorProductOperator` (stored in `_tp_operator`) which is
created during assembly.

## Construction APIs

**2-arg Gridap-style (new — automatic translation):**
```julia
dΩ_tp = dΩx ⊗ dΩy
a(u, v) = ∫(∇(u)⋅∇(v)) * dΩ_tp
l(v)    = ∫(1.0 * v)    * dΩ_tp   # dΩ_tp used as unit measure here
op = TensorProductAffineOperator(a, l, U_tp, V_tp)
A  = get_matrix(op)
b  = get_vector(op)
```

**3-arg form (existing — automatic detection, no op_type needed):**
```julia
a(u, v, dΩ) = ∫(∇(u)⋅∇(v)) * dΩ
l(v, dΩ)    = ∫(1.0 * v)    * dΩ
op = TensorProductAffineOperator(a, l, U_tp, V_tp)
```

**Vector of 3-arg forms (Helmholtz, advection–diffusion, etc.):**
```julia
op = TensorProductAffineOperator(
    [(u,v,dΩ) -> ∫(∇u⋅∇v)*dΩ,
     (u,v,dΩ) -> ∫(κ²*u*v)*dΩ],
    l_unit, U_tp, V_tp)
```

**Legacy API (explicit op_type, backward compatible):**
```julia
op = TensorProductAffineOperator(a, l, U_tp, V_tp; op_type=:stiffness, quad_order=4)
```
"""
mutable struct TensorProductAffineOperator
    _tp_operator::Union{TensorProductOperator, Nothing}
    bilinear_form::Union{Function, Nothing}
    linear_form::Function
    trial_space::TensorProductFESpace
    test_space::TensorProductFESpace
    quad_order::Int
    op_type::Union{Symbol, Nothing}     # nothing = translate_bilinear_form path
    weak_form::Union{TensorProductWeakForm, Nothing}  # pre-built form (vector API)
    rhs_forms::Union{Nothing, NTuple}
    b_coeffs::Union{Nothing, NTuple}
    _matrix::Union{AbstractMatrix, Nothing}
    _vector::Union{AbstractVector, Nothing}
end

# ── Primary constructor: single Function ──────────────────────────────────

function TensorProductAffineOperator(
    a::Function, l::Function,
    U_tp::TensorProductFESpace, V_tp::TensorProductFESpace;
    op_type::Union{Symbol, Nothing} = nothing,
    quad_order::Int = 2,
    rhs_forms::Union{Nothing, NTuple} = nothing,
    b_coeffs::Union{Nothing, NTuple} = nothing
)
    @assert length(U_tp.spaces) == length(V_tp.spaces) "Trial and test spaces must have the same number of subdomains"

    if op_type !== nothing
        valid = (:mass, :stiffness, :gradient, :divergence, :curl_curl, :advection)
        @assert op_type ∈ valid "Unknown op_type: $op_type. Must be one of $valid"
        return TensorProductAffineOperator(
            nothing, a, l, U_tp, V_tp, quad_order, op_type, nothing,
            rhs_forms, b_coeffs, nothing, nothing)
    end

    if hasmethod(a, Tuple{Any, Any, Gridap.CellData.Measure})
        # 3-arg form: build TensorProductWeakForm eagerly (fast path)
        wf = TensorProductWeakForm([TensorProductSeparableTerm(a)])
        return TensorProductAffineOperator(
            nothing, a, l, U_tp, V_tp, quad_order, nothing, wf,
            rhs_forms, nothing, nothing, nothing)
    end

    # 2-arg form (closed over dΩ_tp): translate lazily in _assemble_tp!
    return TensorProductAffineOperator(
        nothing, a, l, U_tp, V_tp, quad_order, nothing, nothing,
        rhs_forms, b_coeffs, nothing, nothing)
end

# ── Vector / TensorProductWeakForm constructor ────────────────────────────

function TensorProductAffineOperator(
    a_terms::Union{TensorProductWeakForm, AbstractVector{<:Function}},
    l::Function,
    U_tp::TensorProductFESpace, V_tp::TensorProductFESpace;
    quad_order::Int = 2,
    rhs_forms::Union{Nothing, NTuple} = nothing
)
    @assert length(U_tp.spaces) == length(V_tp.spaces) "Trial and test spaces must have the same number of subdomains"
    wf = a_terms isa TensorProductWeakForm ? a_terms :
         TensorProductWeakForm([TensorProductSeparableTerm(f) for f in a_terms])
    return TensorProductAffineOperator(
        nothing, nothing, l, U_tp, V_tp, quad_order, nothing, wf,
        rhs_forms, nothing, nothing, nothing)
end

# ── Lazy accessors ────────────────────────────────────────────────────────

function get_matrix(op::TensorProductAffineOperator)
    op._matrix === nothing && _assemble_tp!(op)
    return op._matrix
end

function get_vector(op::TensorProductAffineOperator)
    op._vector === nothing && _assemble_tp!(op)
    return op._vector
end

# ── Internal assembly ─────────────────────────────────────────────────────

function _assemble_tp!(op::TensorProductAffineOperator)
    N           = length(op.test_space.spaces)
    spaces_test = op.test_space.spaces
    measures    = ntuple(
        k -> Measure(Gridap.Geometry.get_triangulation(spaces_test[k]), op.quad_order), N)

    # ── Matrix ─────────────────────────────────────────────────────────────
    A = if op.weak_form !== nothing
        # Path 1: pre-built TensorProductWeakForm (3-arg and vector API)
        assemble_weak_form(op.weak_form, spaces_test, measures)

    elseif op.op_type !== nothing
        # Path 2: legacy explicit op_type
        subdomain_ops = assemble_subdomain_operators(spaces_test, measures; b_coeffs=op.b_coeffs)
        if op.op_type == :mass
            assemble_mass_tensor(subdomain_ops)
        elseif op.op_type == :stiffness
            assemble_stiffness_tensor(subdomain_ops)
        elseif op.op_type == :gradient
            assemble_gradient_tensor(subdomain_ops)
        elseif op.op_type == :divergence
            assemble_divergence_tensor(subdomain_ops)
        elseif op.op_type == :curl_curl
            assemble_curl_curl_tensor(subdomain_ops)
        elseif op.op_type == :advection
            @assert op.b_coeffs !== nothing "op_type=:advection requires b_coeffs keyword"
            assemble_advection_tensor(subdomain_ops, op.b_coeffs)
        end

    else
        # Path 3: 2-arg form closed over dΩ_tp → translate now
        wf = translate_bilinear_form(op.bilinear_form, op.test_space, op.quad_order)
        op._tp_operator = TensorProductOperator(wf, spaces_test, measures)
        get_global_matrix(op._tp_operator)
    end

    # ── RHS ────────────────────────────────────────────────────────────────
    rhs_vecs = if op.rhs_forms !== nothing
        ntuple(N) do k
            Uk = TrialFESpace(spaces_test[k])
            Vector{Float64}(get_vector(AffineFEOperator(
                (u, v) -> ∫(0*u*v)*measures[k],
                op.rhs_forms[k],
                Uk, spaces_test[k])))
        end
    elseif hasmethod(op.linear_form, Tuple{Any, Gridap.CellData.Measure})
        ntuple(N) do k
            Uk = TrialFESpace(spaces_test[k])
            Vector{Float64}(get_vector(AffineFEOperator(
                (u, v) -> ∫(0*u*v)*measures[k],
                v -> op.linear_form(v, measures[k]),
                Uk, spaces_test[k])))
        end
    else
        ntuple(N) do k
            Uk = TrialFESpace(spaces_test[k])
            Vector{Float64}(get_vector(AffineFEOperator(
                (u, v) -> ∫(0*u*v)*measures[k],
                (v)    -> ∫(1*v)*measures[k],
                Uk, spaces_test[k])))
        end
    end

    b = rhs_vecs[1]
    for k in 2:N
        b = kron(rhs_vecs[k], b)
    end

    op._matrix = A
    op._vector = b
    nothing
end
