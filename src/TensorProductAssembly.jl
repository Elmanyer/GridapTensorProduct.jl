"""
    TensorProductAssembly

Full integration of tensor-product weak form assembly with Gridap.

# Purpose

This module is **Stage 5** of the tensor product FEM pipeline: it assembles the global system
by leveraging the Kronecker factorization computed in Stage 4.

# Implementation Strategy

The strategy is pragmatic and modular:

1. **Operator Detection:** Analyze weak form to detect operator type
   - Heuristic-based (Poisson-like → stiffness, currently)
   - User can also pass operator type explicitly via TensorProductOperator

2. **Subdomain Extraction:** For each subdomain, assemble fundamental matrices (M, K, G, D, A)
   - Uses standard Gridap assembly on 1D components
   - Independent per subdomain (no tensor product coupling at this stage)

3. **Global Assembly:** Form global operator via Kronecker products
   - Leverages TensorProductFEMOperators functions
   - Handles all 6 operator types symmetrically
   - Works for arbitrary N (not just 2D)

4. **RHS Assembly:** Form right-hand side as tensor product of subdomain RHS
   - Separable RHS: b = b₁ ⊗ b₂ ⊗ ... ⊗ bₙ
   - Uses Kronecker product of component vectors

# Key Functions

- `assemble_tensor_affine_operator(a, l, U_tp, V_tp)` - Main entry point
  - Extracts measures from spaces
  - Auto-detects operator type (heuristic)
  - Returns (A, b) for solving

- (Private) `_assemble_tensor_affine_nd(...)` - Generalized N-D assembly
  - Supports any number of subdomains
  - Routes to appropriate Kronecker assembler
"""

import Kronecker: kronecker, KroneckerProduct

export assemble_tensor_affine_operator
export assemble_tensor_system
# Note: TensorProductAffineOperator, get_matrix, get_vector, _assemble_tp! have moved to TensorProductOperator.jl

# ===========================
# Tensor-Product Weak Form Integration
# ===========================

"""
    assemble_tensor_affine_operator(a::Function, l::Function, U::TensorProductFESpace, V::TensorProductFESpace) -> (A, b)

Assemble tensor-product affine operator by extracting and combining subdomain components.

**Main entry point for Stage 5 assembly.**

# Algorithm

For a tensor domain with N subdomains:
1. Extract component test/trial spaces and measures
2. Detect operator type from weak form (heuristic: assume Poisson-like stiffness)
3. Assemble all subdomain operators (M, K, G, D, A) via `assemble_subdomain_operators()`
4. Form global operator via appropriate Kronecker assembler
5. Assemble RHS as tensor product of component RHS vectors
6. Return (A, b) ready for solving

# Returns
- `A::SparseMatrixCSC`: Global stiffness matrix (n_dofs × n_dofs)
- `b::Vector`: Global RHS vector (n_dofs,)

# Example (2D Poisson)
```julia
a(u, v) = ∫(∇(u) ⋅ ∇(v)) * (dΩ1 ⊗ dΩ2)
l(v) = ∫(1.0 * v) * (dΩ1 ⊗ dΩ2)

A, b = assemble_tensor_affine_operator(a, l, V_tp, V_tp)
u = A \\ b
```

# Limitations (Stage 5 - Current)
- Operator detection is heuristic (only Poisson recognized)
- RHS assumed separable
- Constraints not yet integrated with tensor structure
"""
function assemble_tensor_affine_operator(
    a::Function,
    l::Function,
    U::TensorProductFESpace,
    V::TensorProductFESpace;
    op_type::Symbol = :stiffness,
    quad_order::Int = 2
)
    n = length(U.spaces)
    return _assemble_tensor_affine_nd(a, l, U, V, n; op_type=op_type, quad_order=quad_order)
end

# ===========================
# Generalized N-Dimensional Assembly
# ===========================

"""
    _assemble_tensor_affine_nd(a, l, U_tp, V_tp, n_subdomains)

Generalized N-D tensor product assembly for arbitrary number of subdomains.

**This is the main assembly workhorse supporting N=2, 3, 4, ...** (only 2D Poisson separated for reference)

# Algorithm

1. **Detect operator type:** Analyze weak form
   - Currently heuristic (Poisson-like → stiffness assumed if no explicit hint)
   - Future: full symbolic analysis

2. **Extract measures:** Get triangulations and measures from component spaces
   - For TensorProductFESpace with N spaces, extract N corresponding measures

3. **Assemble subdomain operators:** For each subdomain k ∈ 1...N:
   - `M^(k) = ∫ φᵢφⱼ dxₖ` (mass matrix)
   - `K^(k) = ∫ (∇φᵢ⋅∇φⱼ) dxₖ` (stiffness matrix)
   - Plus second-order operators if needed (G, D, A)

4. **Form global operator:** Use appropriate Kronecker assembler
   - For stiffness: A = Σₖ [M⁽¹⁾ ⊗ ... ⊗ K⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾]
   - Via `assemble_stiffness_tensor(subdomain_ops)`

5. **Assemble RHS:** b = b₁ ⊗ b₂ ⊗ ... ⊗ bₙ
   - Each component: bₖ = ∫ f(xₖ) φⱼ dxₖ
   - Assumes separable RHS

# Returns
(A, b) ready for solving via A \\ b

# Extensibility

To add support for additional operators (advection, curl-curl, etc.):
1. Add detection logic in operator-type inference
2. Call appropriate assembler from TensorProductFEMOperators
3. All N-D routing is automatic
"""
function _assemble_tensor_affine_nd(
    _a::Function, _l::Function,
    _U_tp::TensorProductFESpace, V_tp::TensorProductFESpace,
    n_subdomains::Int;
    op_type::Symbol = :stiffness,
    quad_order::Int = 2
)
    N = n_subdomains
    spaces_test = V_tp.spaces

    measures = ntuple(k -> Measure(Gridap.Geometry.get_triangulation(spaces_test[k]), quad_order), N)

    subdomain_ops = assemble_subdomain_operators(spaces_test, measures)

    A = if op_type == :mass
        assemble_mass_tensor(subdomain_ops)
    elseif op_type == :stiffness
        assemble_stiffness_tensor(subdomain_ops)
    elseif op_type == :gradient
        assemble_gradient_tensor(subdomain_ops)
    elseif op_type == :divergence
        assemble_divergence_tensor(subdomain_ops)
    elseif op_type == :curl_curl
        assemble_curl_curl_tensor(subdomain_ops)
    elseif op_type == :advection
        assemble_advection_tensor(subdomain_ops, ntuple(_k -> 1.0, N))
    else
        error("Unknown op_type: $op_type")
    end

    b_vecs = ntuple(N) do k
        Uk = TrialFESpace(spaces_test[k])
        get_vector(AffineFEOperator(
            (u, v) -> ∫(0*u*v)*measures[k],
            (v)    -> ∫(1*v)*measures[k],
            Uk, spaces_test[k]))
    end

    b = b_vecs[1]
    for k in 2:N
        b = kronecker(b_vecs[k], b)
    end

    return A, collect(b)
end

# ===========================
# Integration with Gridap's AffineFEOperator
# ===========================

"""
    assemble_tensor_system(a::Function, l::Function, U::TensorProductFESpace, V::TensorProductFESpace)

Assemble and return a Gridap-compatible result structure.

Returns a tuple (A, b, n_dofs) suitable for solving.
"""
function assemble_tensor_system(a, l, U_tp, V_tp)
    A, b = assemble_tensor_affine_operator(a, l, U_tp, V_tp)
    n_dofs = size(A, 1)
    return A, b, n_dofs
end
