"""
    TensorProductAssembly

Thin functional wrappers over the explicit-label tensor-product assembly API.

These functions are convenience entry points for the common single-term case.
For multi-term weak forms (Helmholtz, advection–diffusion, etc.), use the
`TensorProductWeakForm` / `TensorProductAffineOperator` API directly.

# Example

```julia
# Single stiffness term (Poisson):
A, b = assemble_tensor_affine_operator(:stiffness, dΩ_tp, U_tp, V_tp)

# Stiffness with scaled mass (Helmholtz):
lhs = [TensorProductDomainContribution(:stiffness, dΩ_tp),
       TensorProductDomainContribution(:mass, dΩ_tp; coefficient = k^2)]
rhs = [TensorProductLinearContribution(dΩ_tp)]
op  = TensorProductAffineOperator(TensorProductWeakForm(lhs, rhs), V_tp, U_tp)
A, b = get_matrix(op), get_vector(op)
```
"""

export assemble_tensor_affine_operator
export assemble_tensor_system

"""
    assemble_tensor_affine_operator(
        label    :: Symbol,
        measure  :: TensorProductMeasure,
        U_tp     :: TensorProductFESpace,
        V_tp     :: TensorProductFESpace;
        coefficient :: Real = 1.0,
        b_coeffs    :: Union{Nothing, NTuple} = nothing,
        rhs_forms   :: Union{Nothing, Tuple}  = nothing
    ) -> (A::Matrix{Float64}, b::Vector{Float64})

Assemble a single-label tensor-product affine system and return `(A, b)`.

# Arguments
- `label`       — operator type: `:mass`, `:stiffness`, `:gradient`, `:divergence`,
  `:curl_curl`, or `:advection`
- `measure`     — `TensorProductMeasure` carrying per-subdomain `Measure` objects
- `U_tp`        — trial `TensorProductFESpace`
- `V_tp`        — test `TensorProductFESpace`
- `coefficient` — scalar multiplier for the bilinear term (default 1.0)
- `b_coeffs`    — velocity tuple `(b₁, …, bₙ)` required for `label=:advection`
- `rhs_forms`   — per-subdomain linear forms `(v->∫(f₁*v)*dΩ₁, …)`;
  `nothing` uses a unit load on every subdomain

# Example
```julia
A, b = assemble_tensor_affine_operator(:stiffness, dΩ_tp, U_tp, V_tp)
u    = A \\ b
```
"""
function assemble_tensor_affine_operator(
    label   :: Symbol,
    measure :: TensorProductMeasure,
    U_tp    :: TensorProductFESpace,
    V_tp    :: TensorProductFESpace;
    coefficient :: Real                   = 1.0,
    b_coeffs    :: Union{Nothing, NTuple} = nothing,
    rhs_forms   :: Union{Nothing, Tuple}  = nothing
)
    dc = TensorProductDomainContribution(label, measure;
                                          coefficient = Float64(coefficient),
                                          b_coeffs    = b_coeffs)
    lc = TensorProductLinearContribution(measure; rhs_forms = rhs_forms)
    wf = TensorProductWeakForm([dc], [lc])
    op = TensorProductAffineOperator(wf, V_tp, U_tp)
    return get_matrix(op), get_vector(op)
end

"""
    assemble_tensor_system(label, measure, U_tp, V_tp; kwargs...)
        -> (A, b, n_dofs)

Convenience wrapper: returns `(A, b, n_dofs)` instead of just `(A, b)`.
Accepts the same keyword arguments as `assemble_tensor_affine_operator`.
"""
function assemble_tensor_system(
    label   :: Symbol,
    measure :: TensorProductMeasure,
    U_tp    :: TensorProductFESpace,
    V_tp    :: TensorProductFESpace;
    kwargs...
)
    A, b = assemble_tensor_affine_operator(label, measure, U_tp, V_tp; kwargs...)
    return A, b, size(A, 1)
end
