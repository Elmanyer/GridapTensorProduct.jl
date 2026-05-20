"""
    TensorProductOperator

High-level explicit-label weak form hierarchy for tensor-product FEM.

## Type hierarchy (top = containing, bottom = contained)

```
TensorProductAffineOperator
  └── TensorProductOperator
        └── TensorProductWeakForm
              ├── lhs_terms :: Vector{TensorProductDomainContribution}
              └── rhs_terms :: Vector{TensorProductLinearContribution}
```

## Typical usage

```julia
dΩ_tp = dΩx ⊗ dΩy

# Poisson:
lhs = [TensorProductDomainContribution(:stiffness, dΩ_tp)]
rhs = [TensorProductLinearContribution(dΩ_tp)]
op  = TensorProductAffineOperator(TensorProductWeakForm(lhs, rhs), V_tp, U_tp)
A   = get_matrix(op)
b   = get_vector(op)
u   = A \\ b

# Helmholtz (K + k²M):
k   = 2.5
lhs = [
    TensorProductDomainContribution(:stiffness, dΩ_tp),
    TensorProductDomainContribution(:mass,      dΩ_tp; coefficient = k^2)
]

# Advection–diffusion:
lhs = [
    TensorProductDomainContribution(:stiffness, dΩ_tp),
    TensorProductDomainContribution(:advection, dΩ_tp; b_coeffs = (2.0, 1.5))
]

# Custom separable RHS:
rhs = [TensorProductLinearContribution(dΩ_tp;
        rhs_forms = (v -> ∫(f_x * v) * dΩx, v -> ∫(f_y * v) * dΩy))]
```
"""

import LinearAlgebra: kron
import Gridap.FESpaces: get_matrix, get_vector


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductWeakForm
# ═══════════════════════════════════════════════════════════════════════════

"""
    struct TensorProductWeakForm

Container for the labeled terms of a tensor-product weak form.

Holds separate arrays of bilinear (LHS) and linear (RHS) contribution recipes.
Validates at construction that all contributions use compatible measures
(same number of subdomains).

# Fields
- `lhs_terms::Vector{TensorProductDomainContribution}`
- `rhs_terms::Vector{TensorProductLinearContribution}`

# Construction

```julia
wf = TensorProductWeakForm(
    [TensorProductDomainContribution(:stiffness, dΩ_tp)],   # LHS
    [TensorProductLinearContribution(dΩ_tp)]                 # RHS
)
```
"""
struct TensorProductWeakForm
    lhs_terms::Vector{TensorProductDomainContribution}
    rhs_terms::Vector{TensorProductLinearContribution}

    function TensorProductWeakForm(
        lhs::Vector{TensorProductDomainContribution},
        rhs::Vector{TensorProductLinearContribution}
    )
        isempty(lhs) && error("TensorProductWeakForm: lhs_terms must be non-empty.")
        N = num_subdomains(lhs[1].measure)
        for dc in lhs
            num_subdomains(dc.measure) == N ||
                error("TensorProductWeakForm: incompatible measure in lhs_terms — " *
                      "expected $N subdomains, got $(num_subdomains(dc.measure)).")
        end
        for lc in rhs
            num_subdomains(lc.measure) == N ||
                error("TensorProductWeakForm: incompatible measure in rhs_terms — " *
                      "expected $N subdomains, got $(num_subdomains(lc.measure)).")
        end
        new(lhs, rhs)
    end
end

"""
    num_lhs_terms(wf) -> Int

Number of bilinear form terms.
"""
num_lhs_terms(wf::TensorProductWeakForm) = length(wf.lhs_terms)

"""
    num_rhs_terms(wf) -> Int

Number of linear form terms.
"""
num_rhs_terms(wf::TensorProductWeakForm) = length(wf.rhs_terms)

function Base.show(io::IO, wf::TensorProductWeakForm)
    lhs_labels = join([string(dc.label) for dc in wf.lhs_terms], " + ")
    print(io, "TensorProductWeakForm(lhs: [$lhs_labels], rhs: $(num_rhs_terms(wf)) term(s))")
end

export TensorProductWeakForm, num_lhs_terms, num_rhs_terms


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductOperator
# ═══════════════════════════════════════════════════════════════════════════

"""
    mutable struct TensorProductOperator

Wraps a `TensorProductWeakForm` and the trial/test `TensorProductFESpace`s.
Lazily assembles the global matrix and RHS vector on first access to
`get_matrix` or `get_vector`.

# Fields
- `weak_form::TensorProductWeakForm`
- `test_space::TensorProductFESpace`
- `trial_space::TensorProductFESpace`
- `_matrix` — cached after first assembly
- `_vector` — cached after first assembly

# Construction

```julia
op = TensorProductOperator(wf, test_space, trial_space)
A  = get_matrix(op)   # triggers assembly on first call
b  = get_vector(op)
```
"""
mutable struct TensorProductOperator
    weak_form::TensorProductWeakForm
    test_space::TensorProductFESpace
    trial_space::TensorProductFESpace
    _matrix::Union{AbstractMatrix, Nothing}
    _vector::Union{AbstractVector, Nothing}
end

function TensorProductOperator(
    wf::TensorProductWeakForm,
    test_space::TensorProductFESpace,
    trial_space::TensorProductFESpace
)
    length(test_space.spaces) == length(trial_space.spaces) ||
        error("TensorProductOperator: test and trial spaces must have the same number of subdomains.")
    TensorProductOperator(wf, test_space, trial_space, nothing, nothing)
end

export TensorProductOperator

function get_matrix(op::TensorProductOperator)
    op._matrix === nothing && _assemble_op!(op)
    return op._matrix
end

function get_vector(op::TensorProductOperator)
    op._vector === nothing && _assemble_op!(op)
    return op._vector
end

export get_matrix, get_vector


# ═══════════════════════════════════════════════════════════════════════════
# Internal assembly
# ═══════════════════════════════════════════════════════════════════════════

"""
    _assemble_op!(op::TensorProductOperator)

Assemble the global matrix and RHS vector and cache them in `op`.

Algorithm:
1. Extract per-subdomain test spaces and measures from the first LHS contribution.
2. Create a single `TensorProductSubdomainOperators` (lazy; each M_k, K_k, … is
   assembled at most once thanks to the caching in that struct).
3. Accumulate each LHS contribution via `_assemble_contribution`.
4. Assemble the RHS via `_assemble_rhs`.
"""
function _assemble_op!(op::TensorProductOperator)
    wf     = op.weak_form
    spaces = op.test_space.spaces   # NTuple (or Tuple) of 1-D test FESpaces
    N      = length(spaces)

    # Per-subdomain measures from the first LHS contribution.
    # (All contributions are validated to use compatible measures at WeakForm construction.)
    measures = get_measures(wf.lhs_terms[1].measure)  # NTuple{N, Measure}

    # Advection velocity coefficients — use the first :advection term if present.
    b_coeffs = _extract_b_coeffs(wf.lhs_terms)

    # Build one shared lazy subdomain operator container.
    subdomain_ops = assemble_subdomain_operators(spaces, measures; b_coeffs = b_coeffs)

    # Assemble LHS.
    n_total  = prod(num_free_dofs(spaces[k]) for k in 1:N)
    A_global = zeros(Float64, n_total, n_total)
    for dc in wf.lhs_terms
        A_term = _assemble_contribution(dc, subdomain_ops)
        A_global .+= dc.coefficient .* A_term
    end

    # Assemble RHS.
    b_global = _assemble_rhs(wf.rhs_terms, spaces)

    op._matrix = A_global
    op._vector = b_global
    nothing
end

"""
    _assemble_contribution(dc, subdomain_ops) -> Matrix{Float64}

Dispatch to the correct Kronecker assembler based on `dc.label`.
"""
function _assemble_contribution(
    dc::TensorProductDomainContribution,
    ops::TensorProductSubdomainOperators
) :: Matrix{Float64}
    label = dc.label
    if label === :mass
        return assemble_mass_tensor(ops)
    elseif label === :stiffness
        return assemble_stiffness_tensor(ops)
    elseif label === :gradient
        return assemble_gradient_tensor(ops)
    elseif label === :divergence
        return assemble_divergence_tensor(ops)
    elseif label === :curl_curl
        return assemble_curl_curl_tensor(ops)
    elseif label === :advection
        return assemble_advection_tensor(ops, dc.b_coeffs)
    else
        error("_assemble_contribution: unknown label `$label`.")
    end
end

"""
    _assemble_rhs(rhs_terms, spaces) -> Vector{Float64}

Assemble the global RHS vector as `kron(b_N, …, kron(b_2, b_1))`.

For each `TensorProductLinearContribution`:
- If `rhs_forms` is provided, call each per-subdomain form `rhs_forms[k]`.
- Otherwise use a unit load `∫(1*v)*dΩ_k` on every subdomain.
The per-subdomain vectors are combined via nested `kron`.
"""
function _assemble_rhs(
    rhs_terms::Vector{TensorProductLinearContribution},
    spaces::Tuple
) :: Vector{Float64}
    N        = length(spaces)
    n_total  = prod(num_free_dofs(spaces[k]) for k in 1:N)
    b_global = zeros(Float64, n_total)

    for lc in rhs_terms
        lc_measures = get_measures(lc.measure)
        b_vecs = if lc.rhs_forms !== nothing
            ntuple(N) do k
                Uk = TrialFESpace(spaces[k])
                Vector{Float64}(get_vector(AffineFEOperator(
                    (u, v) -> ∫(0 * u * v) * lc_measures[k],
                    lc.rhs_forms[k],
                    Uk, spaces[k])))
            end
        else
            ntuple(N) do k
                Uk = TrialFESpace(spaces[k])
                Vector{Float64}(get_vector(AffineFEOperator(
                    (u, v) -> ∫(0 * u * v) * lc_measures[k],
                    v      -> ∫(1.0 * v)   * lc_measures[k],
                    Uk, spaces[k])))
            end
        end
        b_term = b_vecs[1]
        for k in 2:N
            b_term = kron(b_vecs[k], b_term)
        end
        b_global .+= lc.coefficient .* b_term
    end

    return b_global
end

"""
    _extract_b_coeffs(lhs_terms) -> Union{Nothing, NTuple}

Return `b_coeffs` from the first `:advection` term, or `nothing`.
"""
function _extract_b_coeffs(lhs_terms::Vector{TensorProductDomainContribution})
    for dc in lhs_terms
        dc.label === :advection && dc.b_coeffs !== nothing && return dc.b_coeffs
    end
    return nothing
end


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductAffineOperator
# ═══════════════════════════════════════════════════════════════════════════

"""
    struct TensorProductAffineOperator

High-level analogue of Gridap's `AffineFEOperator` for tensor-product spaces.
Holds the weak form and FE spaces; contains NO quadrature degree or measure
information (those are embedded in the `TensorProductDomainContribution` objects).

Lazily assembles the global `(A, b)` system via the wrapped `TensorProductOperator`.

# Fields
- `weak_form::TensorProductWeakForm`
- `test_space::TensorProductFESpace`
- `trial_space::TensorProductFESpace`
- `_operator::TensorProductOperator`

# Construction

```julia
lhs = [TensorProductDomainContribution(:stiffness, dΩ_tp)]
rhs = [TensorProductLinearContribution(dΩ_tp)]
wf  = TensorProductWeakForm(lhs, rhs)

op  = TensorProductAffineOperator(wf, V_tp, U_tp)
A   = get_matrix(op)
b   = get_vector(op)
u   = A \\ b
```
"""
struct TensorProductAffineOperator
    weak_form::TensorProductWeakForm
    test_space::TensorProductFESpace
    trial_space::TensorProductFESpace
    _operator::TensorProductOperator
end

function TensorProductAffineOperator(
    wf::TensorProductWeakForm,
    test_space::TensorProductFESpace,
    trial_space::TensorProductFESpace
)
    op = TensorProductOperator(wf, test_space, trial_space)
    TensorProductAffineOperator(wf, test_space, trial_space, op)
end

get_matrix(op::TensorProductAffineOperator) = get_matrix(op._operator)
get_vector(op::TensorProductAffineOperator) = get_vector(op._operator)

export TensorProductAffineOperator

function Base.show(io::IO, op::TensorProductAffineOperator)
    n = num_free_dofs(op.test_space)
    print(io, "TensorProductAffineOperator(n_dofs=$n, $(op.weak_form))")
end
