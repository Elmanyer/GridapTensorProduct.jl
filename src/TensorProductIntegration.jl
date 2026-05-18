"""
    TensorProductIntegration

Integration support for TensorProductMeasure objects in the Gridap weak form assembly pipeline.

# Purpose

This module extends Gridap's integrate() and related functions to handle tensor-product measures,
allowing users to write:

    a(u,v) = ∫(∇(u)⋅∇(v)) * dΩ_tp

where dΩ_tp = dΩ1 ⊗ dΩ2 ⊗ ... ⊗ dΩn.

**This is Stage 5** of the tensor product pipeline (integration foundation).

# Design Approach (Pragmatic)

Rather than implementing full quaternary quadrature-based integration, this module uses a
**two-pronged approach:**

1. **Current (Stage 3+):** Weak-form route via TensorProductFEMOperators
   - User defines weak form (e.g., `a(u,v) = ∫(∇(u)⋅∇(v)) * dΩ_tp`)
   - TensorProductAssembly detects operator type
   - Routes to TensorProductFEMOperators for Kronecker assembly
   - No need for tensor quadrature evaluation

2. **Future ([PHASE 4+]):** Direct quaternary integration for general integrands
   - Implement cell quadrature iteration over tensor cells
   - Support arbitrary integrands (research/testing only)
   - More complex, less common use case

# Why Pragmatic?

For 95% of use cases (standard PDE operators: Poisson, advection-diffusion, etc.), weak-form
decomposition + Kronecker assembly is both more efficient and more natural. Full quaternary
integration is deferred until there's a compelling use case.

# Current Framework

The infrastructure is in place (TensorProductDomainContribution, multiply dispatch), but
integration evaluation returns descriptive errors with phase labels. This makes it easy to
add full evaluation later without breaking existing code.

# Architecture

1. When ∫(expr) * dΩ_tp is evaluated, create an Integrand with TensorProductMeasure
2. Later, when integrate() is called on this, dispatch to tensor-aware integration
3. Return a DomainContribution that accounts for the tensor-product cell structure
"""


# ===========================
# Integrand × Measure multiplication (Gridap hook point)
# ===========================

"""
    *(integrand::Integrand, m::TensorProductMeasure)

Create a domain contribution representing the integration of `integrand` over the tensor-product measure.

This is the entry point that Gridap calls when users write `∫(expr) * measure`.
For TensorProductMeasure, we create a special DomainContribution that handles tensor-product cell structure.
"""

function Base.:*(integrand::Gridap.CellData.Integrand, m::TensorProductMeasure)
    return TensorProductIntegrand(integrand.object, m)
end

# ===========================
# TensorProductIntegrand — marker produced by ∫(expr) * dΩ_tp
# ===========================

"""
    struct TensorProductIntegrand

Marker produced when `∫(expr) * dΩ_tp` is evaluated on a TensorProductMeasure.
Carries the integrand object and the tensor-product measure; actual assembly is
deferred to `TensorProductAffineOperator` or `assemble_tensor_affine_operator`.
"""
struct TensorProductIntegrand
    object::Any
    measure::TensorProductMeasure
end

export TensorProductIntegrand

# ===========================
# TensorProductDomainContribution — per-subdomain contribution container
# ===========================

"""
    struct TensorProductDomainContribution

Per-subdomain contribution container for tensor product weak forms.
Mirrors Gridap's `DomainContribution` (which is an `OrderedDict{Triangulation,AbstractArray}`)
but stores one Gridap `DomainContribution` per subdomain.
"""
mutable struct TensorProductDomainContribution
    contributions::Vector{DomainContribution}
    measure::TensorProductMeasure
    operator_type::Symbol
end

function TensorProductDomainContribution(measure::TensorProductMeasure;
                                          op_type::Symbol = :unknown)
    N = num_subdomains(measure)
    TensorProductDomainContribution(
        Vector{DomainContribution}(undef, N), measure, op_type)
end

get_operator_type(c::TensorProductDomainContribution) = c.operator_type
get_subdomain_contribution(c::TensorProductDomainContribution, k::Int) = c.contributions[k]
num_subdomains(c::TensorProductDomainContribution) = length(c.contributions)

function add_subdomain_contribution!(c::TensorProductDomainContribution,
                                      k::Int, dc::DomainContribution)
    c.contributions[k] = dc
end

function Base.:+(c1::TensorProductDomainContribution, c2::TensorProductDomainContribution)
    @assert num_subdomains(c1) == num_subdomains(c2)
    N = num_subdomains(c1)
    combined = [c1.contributions[k] + c2.contributions[k] for k in 1:N]
    TensorProductDomainContribution(combined, c1.measure, c1.operator_type)
end

export TensorProductDomainContribution

# ===========================
# Integration dispatch
# ===========================

"""
    integrate(contrib::TensorProductIntegrand) -> (not implemented)

Direct integration of a `TensorProductIntegrand` is not supported. Use the high-level
assembly API instead:

    op = TensorProductAffineOperator(a, l, U_tp, V_tp; op_type=:stiffness)
    A  = get_matrix(op)
    b  = get_vector(op)

or for low-level access:

    A, b = assemble_tensor_affine_operator(a, l, U_tp, V_tp; op_type=:stiffness)
"""
function Gridap.CellData.integrate(_contrib::TensorProductIntegrand)
    error("""
    Direct integrate() on TensorProductIntegrand is not supported.
    Use TensorProductAffineOperator(a, l, U_tp, V_tp; op_type=:stiffness) instead,
    or assemble_tensor_affine_operator(a, l, U_tp, V_tp; op_type=:stiffness).
    """)
end

