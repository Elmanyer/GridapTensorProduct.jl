"""
    TensorProductIntegration

Integration hooks and contribution types for tensor-product weak forms.

# Overview

This module provides:

1. `TensorProductIntegrand` — the marker type produced by `∫(expr) * dΩ_tp`.
   Carries the integrand object and the `TensorProductMeasure`; not evaluated directly.

2. `TensorProductDomainContribution` — an explicit-label recipe for a single
   **bilinear** form term.  The user annotates each term with its operator type
   (`:stiffness`, `:mass`, etc.) and the tensor-product measure.  The assembler
   uses the label to dispatch to the correct Kronecker formula without any
   heuristic classification.

3. `TensorProductLinearContribution` — recipe for a single **linear** (RHS)
   form term.  Stores the tensor-product measure and optional per-subdomain
   linear forms.

# Typical usage

```julia
dΩ_tp = dΩx ⊗ dΩy

# Poisson LHS:
dc = TensorProductDomainContribution(:stiffness, dΩ_tp)

# Helmholtz LHS (K + k²M):
lhs = [
    TensorProductDomainContribution(:stiffness, dΩ_tp),
    TensorProductDomainContribution(:mass, dΩ_tp; coefficient = k^2)
]

# Unit RHS:
lc = TensorProductLinearContribution(dΩ_tp)

# Custom separable RHS:
lc = TensorProductLinearContribution(dΩ_tp;
        rhs_forms = (v -> ∫(f_x * v) * dΩx, v -> ∫(f_y * v) * dΩy))
```
"""


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductIntegrand — marker produced by ∫(expr) * dΩ_tp
# ═══════════════════════════════════════════════════════════════════════════

"""
    struct TensorProductIntegrand

Marker produced when `∫(expr) * dΩ_tp` is evaluated on a `TensorProductMeasure`.
Carries the integrand object and the tensor-product measure.  Actual assembly is
deferred to `TensorProductAffineOperator`.
"""
struct TensorProductIntegrand
    object::Any
    measure::TensorProductMeasure
end

export TensorProductIntegrand

"""
    *(integrand::Integrand, m::TensorProductMeasure) -> TensorProductIntegrand

Gridap hook: intercepts `∫(expr) * dΩ_tp` and wraps the result.
"""
function Base.:*(integrand::Gridap.CellData.Integrand, m::TensorProductMeasure)
    return TensorProductIntegrand(integrand.object, m)
end


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductDomainContribution — labeled bilinear form recipe
# ═══════════════════════════════════════════════════════════════════════════

"""
    struct TensorProductDomainContribution

Explicit-label recipe for a single **bilinear** form term in a tensor-product
weak form.

The user provides the operator type `label` and the `TensorProductMeasure` that
carries all per-subdomain `Measure` objects.  The assembler uses the label to
dispatch to the correct Kronecker formula without any heuristic classification.

# Fields
- `label::Symbol`  — operator type: `:mass`, `:stiffness`, `:gradient`,
  `:divergence`, `:curl_curl`, or `:advection`
- `measure::TensorProductMeasure`  — bundle of per-subdomain measures
- `coefficient::Float64`  — scalar multiplier applied at assembly (default 1.0)
- `b_coeffs::Union{Nothing, NTuple}`  — velocity components `(b₁, …, bₙ)` required
  when `label == :advection`; `nothing` for all other labels

# Which subdomain operators are needed per label

| label        | subdomain ops assembled |
|:-------------|:------------------------|
| `:mass`      | M_k                     |
| `:stiffness` | M_k, K_k                |
| `:gradient`  | M_k, G_k                |
| `:divergence`| M_k, G_k                |
| `:curl_curl` | M_k, D_k                |
| `:advection` | M_k, A_k                |

The lazy caching in `TensorProductSubdomainOperators` ensures each subdomain
matrix is assembled at most once, even when multiple contributions share the
same label.

# Constructors

```julia
# From a label and measure directly:
TensorProductDomainContribution(:stiffness, dΩ_tp)
TensorProductDomainContribution(:mass, dΩ_tp; coefficient = k^2)
TensorProductDomainContribution(:advection, dΩ_tp; b_coeffs = (2.0, 1.5))

# From a TensorProductIntegrand (extracts measure automatically):
intg = ∫(∇(u)⋅∇(v)) * dΩ_tp   # TensorProductIntegrand
TensorProductDomainContribution(intg, :stiffness)
TensorProductDomainContribution(intg, :mass; coefficient = k^2)
```
"""
struct TensorProductDomainContribution
    label::Symbol
    measure::TensorProductMeasure
    coefficient::Float64
    b_coeffs::Union{Nothing, NTuple}
end

const _VALID_BILINEAR_LABELS = (:mass, :stiffness, :gradient, :divergence, :curl_curl, :advection)

function TensorProductDomainContribution(
    label::Symbol,
    measure::TensorProductMeasure;
    coefficient::Real = 1.0,
    b_coeffs::Union{Nothing, NTuple} = nothing
)
    label ∈ _VALID_BILINEAR_LABELS ||
        error("TensorProductDomainContribution: unknown label `$label`. " *
              "Must be one of $_VALID_BILINEAR_LABELS.")
    label === :advection && b_coeffs === nothing &&
        error("TensorProductDomainContribution: label=:advection requires `b_coeffs` keyword.")
    TensorProductDomainContribution(label, measure, Float64(coefficient), b_coeffs)
end

function TensorProductDomainContribution(
    intg::TensorProductIntegrand,
    label::Symbol;
    coefficient::Real = 1.0,
    b_coeffs::Union{Nothing, NTuple} = nothing
)
    TensorProductDomainContribution(label, intg.measure;
                                     coefficient = coefficient,
                                     b_coeffs    = b_coeffs)
end

export TensorProductDomainContribution


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductLinearContribution — labeled linear form (RHS) recipe
# ═══════════════════════════════════════════════════════════════════════════

"""
    struct TensorProductLinearContribution

Recipe for a single **linear** (right-hand side) form term in a tensor-product
weak form.

# Fields
- `measure::TensorProductMeasure`  — bundle of per-subdomain measures
- `rhs_forms::Union{Nothing, Tuple}`  — per-subdomain linear forms
  `(v -> ∫(f₁*v)*dΩ₁, v -> ∫(f₂*v)*dΩ₂, …)`.  When `nothing`, each subdomain
  uses a unit load `∫(1*v)*dΩₖ`.
- `coefficient::Float64`  — scalar multiplier (default 1.0)

# Constructors

```julia
# Unit load on every subdomain:
TensorProductLinearContribution(dΩ_tp)

# Custom separable load:
TensorProductLinearContribution(dΩ_tp;
    rhs_forms = (v -> ∫(f_x * v) * dΩx, v -> ∫(f_y * v) * dΩy))

# Scaled unit load:
TensorProductLinearContribution(dΩ_tp; coefficient = 2.0)
```
"""
struct TensorProductLinearContribution
    measure::TensorProductMeasure
    rhs_forms::Union{Nothing, Tuple}
    coefficient::Float64
end

function TensorProductLinearContribution(
    measure::TensorProductMeasure;
    rhs_forms::Union{Nothing, Tuple} = nothing,
    coefficient::Real = 1.0
)
    TensorProductLinearContribution(measure, rhs_forms, Float64(coefficient))
end

export TensorProductLinearContribution


# ═══════════════════════════════════════════════════════════════════════════
# Direct integration — error stub
# ═══════════════════════════════════════════════════════════════════════════

"""
    Gridap.CellData.integrate(::TensorProductIntegrand)

Direct `integrate()` on a `TensorProductIntegrand` is not supported.
Use the high-level assembly API:

    lhs = [TensorProductDomainContribution(:stiffness, dΩ_tp)]
    rhs = [TensorProductLinearContribution(dΩ_tp)]
    op  = TensorProductAffineOperator(TensorProductWeakForm(lhs, rhs), V_tp, U_tp)
    A   = get_matrix(op)
    b   = get_vector(op)
"""
function Gridap.CellData.integrate(_contrib::TensorProductIntegrand)
    error("""
    Direct integrate() on TensorProductIntegrand is not supported.
    Use TensorProductAffineOperator with TensorProductDomainContribution instead:

        lhs = [TensorProductDomainContribution(:stiffness, dΩ_tp)]
        rhs = [TensorProductLinearContribution(dΩ_tp)]
        op  = TensorProductAffineOperator(TensorProductWeakForm(lhs, rhs), V_tp, U_tp)
        A, b = get_matrix(op), get_vector(op)
    """)
end
