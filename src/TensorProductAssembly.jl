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

# ===========================
# TensorProductAffineOperator
# ===========================

export TensorProductAffineOperator

"""
    mutable struct TensorProductAffineOperator

High-level tensor product analogue of Gridap's `AffineFEOperator`.
Lazily assembles the global `(A, b)` system via Kronecker factorisation on first access.

# Usage
```julia
op = TensorProductAffineOperator(a, l, U_tp, V_tp; op_type=:stiffness, quad_order=2)
A  = get_matrix(op)   # assembled on first call, cached afterwards
b  = get_vector(op)
u  = A \\ b
```

# Optional kwargs
- `op_type`   — one of `:mass`, `:stiffness`, `:gradient`, `:divergence`, `:curl_curl`, `:advection`
- `quad_order` — quadrature degree for subdomain measures (default 2)
- `rhs_forms` — NTuple of per-subdomain linear forms `(v) -> ∫(f_k*v)*dΩ_k`; if `nothing` uses unit load
- `b_coeffs`  — NTuple of advection velocity components (required for `op_type=:advection`)
"""
mutable struct TensorProductAffineOperator
    bilinear_form::Function
    linear_form::Function
    trial_space::TensorProductFESpace
    test_space::TensorProductFESpace
    op_type::Symbol
    quad_order::Int
    rhs_forms::Union{Nothing, NTuple}
    b_coeffs::Union{Nothing, NTuple}
    _matrix::Union{AbstractMatrix, Nothing}
    _vector::Union{AbstractVector, Nothing}
end

function TensorProductAffineOperator(
    a::Function, l::Function,
    U_tp::TensorProductFESpace, V_tp::TensorProductFESpace;
    op_type::Symbol   = :stiffness,
    quad_order::Int   = 2,
    rhs_forms::Union{Nothing, NTuple} = nothing,
    b_coeffs::Union{Nothing, NTuple}  = nothing
)
    valid = (:mass, :stiffness, :gradient, :divergence, :curl_curl, :advection)
    @assert op_type ∈ valid "Unknown op_type: $op_type. Must be one of $valid"
    @assert length(U_tp.spaces) == length(V_tp.spaces) "Trial and test spaces must have the same number of subdomains"
    TensorProductAffineOperator(a, l, U_tp, V_tp, op_type, quad_order,
                                 rhs_forms, b_coeffs, nothing, nothing)
end

import Gridap.FESpaces: get_matrix, get_vector

function get_matrix(op::TensorProductAffineOperator)
    op._matrix === nothing && _assemble_tp!(op)
    return op._matrix
end

function get_vector(op::TensorProductAffineOperator)
    op._vector === nothing && _assemble_tp!(op)
    return op._vector
end

function _assemble_tp!(op::TensorProductAffineOperator)
    N = length(op.test_space.spaces)
    spaces_test = op.test_space.spaces
    measures = ntuple(k -> Measure(Gridap.Geometry.get_triangulation(spaces_test[k]),
                                   op.quad_order), N)

    subdomain_ops = assemble_subdomain_operators(spaces_test, measures; b_coeffs=op.b_coeffs)

    A = if op.op_type == :mass
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

    rhs_vecs = if op.rhs_forms !== nothing
        ntuple(N) do k
            Uk = TrialFESpace(spaces_test[k])
            Vector{Float64}(get_vector(AffineFEOperator(
                (u, v) -> ∫(0*u*v)*measures[k],
                op.rhs_forms[k],
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
