# GridapTensorProduct.jl

A Julia library extending [Gridap.jl](https://github.com/gridap/Gridap.jl) for the efficient solution of PDEs on tensor-product domains using Kronecker-factored finite element operators.

---

## Goal

Many PDE problems of practical interest arise on domains that are Cartesian products of lower-dimensional components:

```
Ω = Ω₁ × Ω₂ × ... × Ω_N  ⊆  ℝᴺ
```

Examples include:
- **2D/3D structured grids** in computational fluid dynamics (channel flows, atmospheric models)
- **Space–time formulations** where the spatial domain is time-independent
- **Far-field/exterior domains** with a radial × angular decomposition
- **Separable Cartesian problems** in scientific computing

When the domain has this tensor-product structure and the PDE operator is *separable*, global FEM matrices decompose as sums of **Kronecker products** of small 1D subdomain matrices. GridapTensorProduct exploits this structure to:

1. **Replace N-D assembly** (expensive, memory-heavy) with N independent 1D assemblies followed by cheap Kronecker operations
2. **Avoid materialising large global matrices** using lazy Kronecker products (via [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl)), enabling matrix-free or low-memory linear solves
3. **Reuse Gridap's full FE infrastructure** — reference elements, quadrature, boundary conditions, solvers — on each 1D subdomain, then compose globally

---

## Key Features

### Kronecker Factorisation of FEM Operators

Six operator types are supported, each with an exact Kronecker decomposition:

| Operator | Formula |
|---|---|
| **Mass** | `M = M⁽¹⁾ ⊗ M⁽²⁾ ⊗ ... ⊗ M⁽ᴺ⁾` |
| **Stiffness (Laplacian)** | `A = Σₖ M⁽¹⁾ ⊗ ... ⊗ K⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾` |
| **Gradient** | `G = [ M⊗...⊗G⁽ᵏ⁾⊗...⊗M ; ... ]` (block column) |
| **Divergence** | `B = Gᵀ` |
| **Curl-Curl** | `C = Σₖ Σ_{l≠k} M⊗...⊗(Dₖᵀ Dₗ − Dₗᵀ Dₖ)⊗...⊗M` |
| **Advection** | `T = Σₖ bₖ · M⁽¹⁾ ⊗ ... ⊗ A⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾` |

where `M⁽ᵏ⁾`, `K⁽ᵏ⁾`, `G⁽ᵏ⁾`, `D⁽ᵏ⁾`, `A⁽ᵏ⁾` are standard 1D Gridap matrices on `Ω_k`.

The DOF ordering convention is **subdomain 1 = fastest-varying** (innermost index), so the global linear index satisfies:

```
global_dof(i₁, i₂, ..., iₙ) = i₁ + (i₂−1)·n₁ + (i₃−1)·n₁·n₂ + ...
```

and Kronecker products are built as `kron(M_N, kron(M_{N-1}, ..., kron(M_2, M_1)...))`.

### Lazy Evaluation via Kronecker.jl

Global operators are represented as lightweight `KroneckerProduct` objects (a few hundred bytes each) rather than dense or sparse global matrices:

```
Traditional: K⁽¹⁾ ⊗ M⁽²⁾  →  n²×n² dense matrix  (O(n⁴) memory)
Lazy:         kronecker(M2, K1)  →  struct of two n×n matrices  (O(n²) memory)
```

For iterative solvers (CG, GMRES) the lazy representation is used directly; matrix-vector products `y = A*x` are computed without ever forming the full global matrix. For direct solvers a sparse conversion is available when required.

### Gridap Integration

GridapTensorProduct is designed as a non-intrusive extension of Gridap. Each subdomain is treated as a standard Gridap problem:
- `TestFESpace`, `TrialFESpace`, `AffineFEOperator` on each `Ω_k`
- Standard Gridap reference elements, quadrature rules, and boundary conditions
- Tensor product coupling happens only at assembly, not at the subdomain level

---

## Pipeline Architecture

The library is organised into five stages:

```
Stage 1: TensorProductMeasure        — bundle N subdomain quadrature measures
Stage 2: TensorProductFESpace        — bundle N subdomain FE spaces + DOF mapping
Stage 3: TensorProductGeometry       — triangulation and cell-field wrappers
Stage 4: TensorProductFEMOperators   — extract 1D matrices; form Kronecker global operators
Stage 5: TensorProductAssembly       — high-level API; assemble and return (A, b)
```

Subdomains are the primary computational unit throughout stages 1–4; the tensor product coupling is deferred to stage 5 assembly.

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/your-org/GridapTensorProduct.jl")
```

**Dependencies:** `Gridap.jl`, `Kronecker.jl`, `BlockArrays.jl`, `FillArrays.jl`, `LinearAlgebra` (stdlib).

---

## Main Workflow

### High-Level API (`TensorProductAffineOperator`)

The recommended entry point for most users:

```julia
using Gridap, GridapTensorProduct

# Stage 1-2: set up 1D subdomains independently
model_x = CartesianDiscreteModel((0, 1), 50)
model_y = CartesianDiscreteModel((0, 1), 50)

reffe = ReferenceFE(lagrangian, Float64, 2)

Vx = TestFESpace(Interior(model_x), reffe; conformity=:H1, dirichlet_tags="boundary")
Vy = TestFESpace(Interior(model_y), reffe; conformity=:H1, dirichlet_tags="boundary")

# Stage 2: bundle into tensor-product space
V_tp = TensorProductFESpace(Vx, Vy)
U_tp = TensorProductFESpace(TrialFESpace(Vx, 0.0), TrialFESpace(Vy, 0.0))

# Weak form (written for documentation; the operator type is given explicitly below)
a(u, v) = ∫(∇(u) ⋅ ∇(v)) * (dΩx ⊗ dΩy)
l(v)    = ∫(1.0 * v)       * (dΩx ⊗ dΩy)

# Stage 5: lazy assembly
op = TensorProductAffineOperator(a, l, U_tp, V_tp; op_type=:stiffness, quad_order=4)

A = get_matrix(op)   # assembled on first call, cached afterwards
b = get_vector(op)
u = A \ b
```

The `op_type` keyword selects which Kronecker formula to use. Supported values: `:mass`, `:stiffness`, `:gradient`, `:divergence`, `:curl_curl`, `:advection`.

### Low-Level API

For direct control over operator construction and reuse:

```julia
using Gridap, GridapTensorProduct

# Set up 1D subdomains
dΩx = Measure(Interior(model_x), 4)
dΩy = Measure(Interior(model_y), 4)

# Stage 4a: extract all fundamental 1D matrices in one pass
ops = assemble_subdomain_operators((Vx, Vy), (dΩx, dΩy))

# Stage 4b: assemble any global operator from the same subdomain data
A = assemble_stiffness_tensor(ops)   # Laplacian: K₁⊗M₂ + M₁⊗K₂
M = assemble_mass_tensor(ops)        # L² projection: M₁⊗M₂
G = assemble_gradient_tensor(ops)    # Gradient: block [G₁⊗M₂; M₁⊗G₂]

# Or use TensorProductOperator for lazy caching
op_stiff = TensorProductOperator(:stiffness, (Vx, Vy), (dΩx, dΩy))
A = get_global_matrix(op_stiff)  # computed once and cached
```

### 3D Tensor Product (N=3)

The API is dimension-agnostic; adding a third subdomain requires no code changes:

```julia
V_tp = TensorProductFESpace(Vx, Vy, Vz)
op = TensorProductAffineOperator(a, l, U_tp, V_tp; op_type=:stiffness)
A = get_matrix(op)
# A ≡ K₁⊗M₂⊗M₃ + M₁⊗K₂⊗M₃ + M₁⊗M₂⊗K₃   (lazy, O(n²) memory per term)
```

### Advection–Diffusion

```julia
b_x, b_y = 1.0, 2.0  # advection velocity components

ops = assemble_subdomain_operators((Vx, Vy), (dΩx, dΩy);
                                    b_coeffs=(b_x, b_y))

A_diff = assemble_stiffness_tensor(ops)
A_adv  = assemble_advection_tensor(ops, (b_x, b_y))
A      = A_diff + A_adv

# Or via the high-level operator:
op = TensorProductAffineOperator(a, l, U_tp, V_tp;
                                  op_type=:advection,
                                  b_coeffs=(b_x, b_y))
```

---

## Module Structure

```
src/
├── GridapTensorProduct.jl        — module entry point, exports
├── TensorProductMeasures.jl      — TensorProductMeasure, ⊗ operator
├── TensorProductIntegration.jl   — TensorProductIntegrand, TensorProductDomainContribution
├── TensorProductGeometry.jl      — TensorProductTriangulation, TensorProductCellField
├── TensorProductFESpaces.jl      — TensorProductFESpace, DOF mapping
├── TensorProductAssemblers.jl    — KroneckerAssembler, FallbackAssembler
├── TensorProductAssembly.jl      — assemble_tensor_affine_operator,
│                                   TensorProductAffineOperator
├── TensorProductFEMOperators.jl  — assemble_subdomain_operators,
│                                   assemble_{mass,stiffness,gradient,...}_tensor
└── TensorProductOperator.jl      — TensorProductOperator (typed, caching wrapper)
```

---

## Limitations and Assumptions

- **Separability required:** the PDE operator must be expressible as a sum/product of 1D integrals. Non-separable operators (e.g. variable-coefficient Laplacians with `a(x,y) ∇u · ∇v`) cannot use the Kronecker path and require a full global assembly.
- **Separable forcing:** the default RHS assembler assumes a constant unit load on each subdomain. For non-constant separable forcing, supply explicit per-subdomain `rhs_forms` to `TensorProductAffineOperator`.
- **Structured tensor grids:** the library targets structured Cartesian product meshes. Unstructured global meshes are not supported.
- **Boundary conditions per subdomain:** Dirichlet constraints are applied independently on each `Ω_k` before Kronecker assembly; the tensor DOF is free if and only if all component DOFs are free.

---

## Testing

```julia
using Pkg
Pkg.test("GridapTensorProduct")
```

The test suite includes:
- **`PoissonEquivalenceTests`** — Kronecker stiffness/RHS assembly matches expected `kron(M₂,K₁) + kron(K₂,M₁)` decomposition
- **`TensorProductFESpaceInterfaceTests`** — DOF count, cell IDs, cell DOF IDs, triangulation access
- **`DOFOrderingTests`** — bijection between Cartesian DOF tuples and linear global indices for N=1,2,3
- **`FEFunctionTests`** — `TensorProductFEFunction` construction with free and Dirichlet values
- **`KroneckerAssemblerTests`** — smoke tests for `KroneckerAssembler` and `FallbackAssembler`
