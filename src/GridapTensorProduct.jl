"""
    GridapTensorProduct

Extension to Gridap.jl enabling efficient solution of PDEs on tensor-product domains.

## Pipeline Architecture

The library organizes tensor product FEM into **5 stages**, enabling subdomains to be treated
independently until assembly:

### Stage 1: Measures (TensorProductMeasure)
Bundle quadrature rules and integration measures for each subdomain.
```julia
dΩ1 = Measure(trian1, 2)
dΩ2 = Measure(trian2, 2)
dΩ_tp = dΩ1 ⊗ dΩ2  # Tensor product measure
```

### Stage 2: FE Spaces (TensorProductFESpace)
Bundle FE basis functions and DOF structure for each subdomain.
```julia
V1 = TestFESpace(Ω1, reffe)
V2 = TestFESpace(Ω2, reffe)
V_tp = TensorProductFESpace(V1, V2)
```

### Stage 3: Geometry (TensorProductGeometry)
Lazy wrappers for triangulations and cell fields (used only when assembling weak forms).
```julia
trian_tp = TensorProductTriangulation(get_triangulation(V1), get_triangulation(V2))
```

### Stage 4: Operator Translation (TensorProductOperator, TensorProductFEMOperators)
Translate global weak-form operators into Kronecker product decompositions.
- Six operator types supported: Mass, Stiffness, Gradient, Divergence, Curl-Curl, Advection
```julia
op_stiff = TensorProductOperator(:stiffness, (V1, V2), (dΩ1, dΩ2))
A = get_global_matrix(op_stiff)  # Triggers Kronecker assembly
```

### Stage 5: Assembly & Solving (TensorProductAssembly, TensorProductAssemblers)
Assemble via Kronecker relations and solve using Gridap solvers.
```julia
a(u, v) = ∫(∇(u) ⋅ ∇(v)) * dΩ_tp
l(v) = ∫(v) * dΩ_tp
A, b = assemble_tensor_affine_operator(a, l, V_tp, V_tp)
u = A \\ b
```

## Key Design Principles
- **Subdomain Independence:** Each subdomain is a primary computational unit
- **Lazy Instantiation:** Tensor operations only instantiated when needed for assembly
- **Kronecker Factorization:** Global operators decomposed as sums/products of subdomain matrices
  - Mass: M = M₁ ⊗ M₂ ⊗ ... ⊗ Mₙ
  - Stiffness: A = ΣK₁⊗M₂⊗... + M₁⊗K₂⊗... + ...
  - Gradient: G = ⊕[M₁⊗G₂⊗...], [G₁⊗M₂⊗...], ...
  - etc.

## Usage Example (2D Poisson)
```julia
using Gridap, GridapTensorProduct

# Stage 1-3: Set up subdomains (work independently)
model_x = CartesianDiscreteModel((0,1), 10)
model_y = CartesianDiscreteModel((0,1), 10)
Vx = TestFESpace(Interior(model_x), ReferenceFE(lagrangian, Float64, 2))
Vy = TestFESpace(Interior(model_y), ReferenceFE(lagrangian, Float64, 2))
dΩx = Measure(Interior(model_x), 2)
dΩy = Measure(Interior(model_y), 2)

# Stage 2: Create tensor space
V_tp = TensorProductFESpace(Vx, Vy)

# Stage 4: Define weak form and assemble
a(u, v) = ∫(∇(u) ⋅ ∇(v)) * (dΩx ⊗ dΩy)
l(v) = ∫(1.0 * v) * (dΩx ⊗ dΩy)

# Stage 5: Assemble and solve
A, b = assemble_tensor_affine_operator(a, l, V_tp, V_tp)
u = A \\ b
```

## Kronecker Product Lazy Evaluation

This library uses **Kronecker.jl** for efficient Kronecker product assembly with lazy evaluation.

### Why Lazy Evaluation?

Traditional approaches compute Kronecker products eagerly, materialize the full global matrix:

```
Example: 2D Poisson with 100×100 1D mesh
┌─────────────────────────────────────────────────────────────────┐
│ LinearAlgebra.kron (eager):                                     │
│  K1 (10K×10K) ⊗ M2 (10K×10K) → 100M×100M DENSE matrix          │
│  Memory: ~40 GB per term × N terms → Infeasible              │
│                                                                  │
│ Kronecker.jl (lazy):                                            │
│  KroneckerProduct(K1, M2) → Lightweight struct (~1 KB)         │
│  Memory: ~1 KB per term + efficiency on matrix-vector ops      │
│  Scales to 3D/4D problems naturally                            │
└─────────────────────────────────────────────────────────────────┘
```

### Solver Integration

For **iterative solvers** (CG, GMRES, etc.), lazy Kronecker products work seamlessly:
- Matrix-vector products `y = A*x` computed implicitly without materializing `A`
- Memory footprint stays small throughout solving
- Solver framework automatically handles the KroneckerProduct type

For **direct solvers** (LU, QR), convert to sparse when needed:
```julia
using LinearAlgebra: sparse
A_sparse = sparse(A)  # Explicit conversion only when necessary
u = lu(A_sparse) \\ b
```

### Performance Benefits

| Problem Size    | LinearAlgebra.kron | Kronecker.jl  | Scaling |
|-----------------|-------------------|--------------|---------|
| 2D small        | ~1 GB              | ~1 KB        | 1M× |
| 2D medium       | OOM (>100 GB)      | ~1 KB        | Enables |
| 3D small        | OOM                | ~1 KB        | Enables |

## Validation
All implementations validated against standard Gridap reference solutions.
Tested on 2D (Poisson, advection, curl), 3D tensor products, and heterogeneous problems.
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

export TensorProductMeasure, ⊗
include("TensorProductMeasures.jl")

include("TensorProductIntegration.jl")

export TensorProductTriangulation, TensorProductCellField
include("TensorProductGeometry.jl")

export TensorProductFESpace
include("TensorProductFESpaces.jl")

export TensorProductAssembler
export KroneckerAssembler
export FallbackAssembler
include("TensorProductAssemblers.jl")

export TensorProductSubdomainOperators
export TensorProductGlobalOperator
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
include("TensorProductOperator.jl")

include("TensorProductAssembly.jl")


end # module GridapTensorProduct
