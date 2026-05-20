"""
    GridapTensorProduct

Extension to Gridap.jl enabling efficient solution of PDEs on tensor-product domains
via Kronecker product decomposition.

## Pipeline Architecture

```
Stage 1  TensorProductMeasure        — bundle N subdomain measures; ⊗ operator
Stage 2  TensorProductFESpace        — bundle N subdomain FE spaces + DOF mapping
Stage 3  TensorProductGeometry       — lazy triangulation and cell-field wrappers
Stage 4  TensorProductFEMOperators   — lazy subdomain operators; six Kronecker assemblers
Stage 5  TensorProductOperator       — explicit-label weak form → global system
```

## Quick-start (2D Poisson)

```julia
using Gridap, GridapTensorProduct

model_x = CartesianDiscreteModel((0,1), 10)
model_y = CartesianDiscreteModel((0,1), 10)
Vx = TestFESpace(Interior(model_x), ReferenceFE(lagrangian, Float64, 1);
                  conformity=:H1, dirichlet_tags="boundary")
Vy = TestFESpace(Interior(model_y), ReferenceFE(lagrangian, Float64, 1);
                  conformity=:H1, dirichlet_tags="boundary")
Ux = TrialFESpace(Vx, 0.0); Uy = TrialFESpace(Vy, 0.0)

dΩx = Measure(Interior(model_x), 2)
dΩy = Measure(Interior(model_y), 2)
dΩ_tp = dΩx ⊗ dΩy

V_tp = TensorProductFESpace(Vx, Vy)
U_tp = TensorProductFESpace(Ux, Uy)

lhs = [TensorProductDomainContribution(:stiffness, dΩ_tp)]
rhs = [TensorProductLinearContribution(dΩ_tp)]
op  = TensorProductAffineOperator(TensorProductWeakForm(lhs, rhs), V_tp, U_tp)

A = get_matrix(op)
b = get_vector(op)
u = A \\ b
```

## Kronecker Decompositions

| Operator     | Global formula                                      |
|:-------------|:----------------------------------------------------|
| Mass         | `M = M⁽¹⁾ ⊗ M⁽²⁾ ⊗ … ⊗ M⁽ᴺ⁾`                    |
| Stiffness    | `A = Σₖ M⁽¹⁾ ⊗ … ⊗ K⁽ᵏ⁾ ⊗ … ⊗ M⁽ᴺ⁾`             |
| Gradient     | `G = ⊕ₖ M⁽¹⁾ ⊗ … ⊗ G⁽ᵏ⁾ ⊗ … ⊗ M⁽ᴺ⁾`             |
| Divergence   | `B = Gᵀ`                                           |
| Curl-curl    | `C = Σₖ Σₗ≠ₖ (⊗ⱼ≠ₖ,ₗ M⁽ʲ⁾) ⊗ (Cᵀ Dₖᵀ Dₗ − …)`  |
| Advection    | `T = Σₖ bₖ · M⁽¹⁾ ⊗ … ⊗ A⁽ᵏ⁾ ⊗ … ⊗ M⁽ᴺ⁾`       |
"""
module GridapTensorProduct

using LinearAlgebra
using BlockArrays
using Kronecker

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.Helpers
using Gridap.ODEs
using Gridap.ReferenceFEs
using Gridap.TensorValues

import FillArrays: Fill

# ── Stage 1: Measures ───────────────────────────────────────────────────────
export TensorProductMeasure, ⊗
include("TensorProductMeasures.jl")

# ── Stage 2 integration types: integrand hook + explicit-label contributions ──
export TensorProductIntegrand
export TensorProductDomainContribution
export TensorProductLinearContribution
include("TensorProductIntegration.jl")

# ── Stage 3: Geometry ───────────────────────────────────────────────────────
export TensorProductTriangulation, TensorProductCellField
include("TensorProductGeometry.jl")

# ── Stage 2: FE Spaces ─────────────────────────────────────────────────────
export TensorProductFESpace
include("TensorProductFESpaces.jl")

# ── Assembler utilities (low-level) ─────────────────────────────────────────
export TensorProductAssembler, KroneckerAssembler, FallbackAssembler
export tensor_tuple_to_linear_index, linear_index_to_tensor_tuple
export tensor_kron, tensor_kron_vector
export apply_tensor_operator
export assemble_tensor_rhs, assemble_tensor_operator
include("TensorProductAssemblers.jl")

# ── Stage 4: Subdomain operators + six Kronecker assemblers ─────────────────
export TensorProductSubdomainOperators
export assemble_subdomain_operators
export assemble_mass_tensor
export assemble_stiffness_tensor
export assemble_gradient_tensor
export assemble_divergence_tensor
export assemble_curl_curl_tensor
export assemble_advection_tensor
export extract_gradient_matrix
export extract_derivative_matrix
export extract_advection_matrix
include("TensorProductFEMOperators.jl")

# ── Stage 5: Explicit-label weak form + operator hierarchy ──────────────────
export TensorProductWeakForm, num_lhs_terms, num_rhs_terms
export TensorProductOperator
export TensorProductAffineOperator
export get_matrix, get_vector
include("TensorProductOperator.jl")

# ── Functional convenience API ───────────────────────────────────────────────
export assemble_tensor_affine_operator
export assemble_tensor_system
include("TensorProductAssembly.jl")

end # module GridapTensorProduct
