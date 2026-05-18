"""
    TensorProductOperator

Wrapper encapsulating the translation of weak-form operators on tensor domains
into factored Kronecker decompositions.

This module is **Stage 4** of the tensor product FEM pipeline: it bridges the gap between
user-defined weak forms and efficient Kronecker-factored assembly.

# Purpose

Users write weak forms like:
- `a(u, v) = ∫(∇(u) ⋅ ∇(v)) * dΩ_tp`  (stiffness)
- `a(u, v) = ∫(u * v) * dΩ_tp`        (mass)
- etc.

This module detects the operator type and translates it to the appropriate Kronecker product
decomposition for assembly via Stage 5.

# Design

The `TensorProductOperator{OP_TYPE}` struct wraps:
- Fundamental subdomain operators (`TensorProductSubdomainOperators`)
- Operator type classification (`:mass`, `:stiffness`, `:gradient`, etc.)
- Optional cached global matrix (lazy evaluation)

# Usage Pattern

```julia
# Stage 1-3: Set up subdomains, spaces, measures (independent work)
V1, V2 = ...  # Two 1D test spaces
dΩ1, dΩ2 = ...  # Two measures

# Stage 4: Detect operator type and translate to Kronecker form
# Option A: Explicit constructor (user annotates)
op_mass = TensorProductOperator(:mass, (V1, V2), (dΩ1, dΩ2))
op_stiff = TensorProductOperator(:stiffness, (V1, V2), (dΩ1, dΩ2))

# Option B: Extract from weak form (future: automatic routing)
# op_tp = TensorProductOperator(a, V_tp)  # Detect operator type automatically

# Stage 5: Assemble (get_global_matrix triggers Kronecker assembly)
A = get_global_matrix(op_stiff)
b = assemble_rhs_tensor(...)
u = A \\ b
```
"""

import LinearAlgebra: kron

export TensorProductOperator
export get_global_matrix
export get_subdomain_operators
export get_operator_type


# Import Gridap types
import Gridap.FESpaces: TestFESpace
import Gridap.CellData: Measure
import Gridap


# ═══════════════════════════════════════════════════════════════════════════
# TensorProductOperator Type Definition
# ═══════════════════════════════════════════════════════════════════════════

"""
    struct TensorProductOperator{OP_TYPE}

Wrapper for a translated tensor-product operator.

# Type Parameters
- `OP_TYPE`: Symbol indicating operator type (`:mass`, `:stiffness`, etc.)
  These are used for dispatch in the Kronecker assembly layer.

# Fields
- `subdomain_ops::TensorProductSubdomainOperators` - All fundamental matrices (M, K, G, D, A)
- `operator_type::Symbol` - One of: `:mass`, `:stiffness`, `:gradient`, `:divergence`, `:curl_curl`, `:advection`
- `global_matrix::Union{Matrix, Nothing}` - Cached global matrix (computed once on-demand)
- `n_subdomains::Int` - Number of subdomains N

# Supported Operator Types
- `:mass` → M = M⁽¹⁾ ⊗ M⁽²⁾ ⊗ ... ⊗ M⁽ᴺ⁾
- `:stiffness` → A = Σ_k [M⁽¹⁾ ⊗ ... ⊗ K⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾]
- `:gradient` → G = ⊕_k [M⁽¹⁾ ⊗ ... ⊗ G⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾]
- `:divergence` → B = Gᵀ
- `:curl_curl` → C = Σ_k Σ_{l≠k} [...]
- `:advection` → T = Σ_k bₖ [M⁽¹⁾ ⊗ ... ⊗ A⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾]
"""
mutable struct TensorProductOperator{OP_TYPE}
    subdomain_ops::TensorProductSubdomainOperators
    operator_type::Symbol
    global_matrix::Union{Matrix, Nothing}
    n_subdomains::Int
end

# ═══════════════════════════════════════════════════════════════════════════
# Constructor & Factory
# ═══════════════════════════════════════════════════════════════════════════

"""
    TensorProductOperator(
        operator_type::Symbol,
        spaces::NTuple{N, TestFESpace},
        measures::NTuple{N, Measure};
        b_coeffs::Union{NTuple, Nothing}=nothing
    ) -> TensorProductOperator

Construct a tensor-product operator by extracting and wrapping subdomain operators.

# Arguments
- `operator_type::Symbol` - Type to assemble (`:mass`, `:stiffness`, etc.)
- `spaces::NTuple{N, TestFESpace}` - N test spaces (one per subdomain)
- `measures::NTuple{N, Measure}` - N integration measures
- `b_coeffs` - Velocity coefficients if operator type is `:advection` (optional)

# Returns
`TensorProductOperator{OP_TYPE}` wrapping all subdomain operators.

# Example
```julia
op = TensorProductOperator(:stiffness, (V1, V2), (dΩ1, dΩ2))
A = get_global_matrix(op)  # Triggers Kronecker assembly
```

# Validation
- Checks that `operator_type` is one of the 6 supported types
- Validates all spaces are TestFESpace instances
- Validates all measures are Gridap Measure instances
"""
function TensorProductOperator(
    operator_type::Symbol,
    spaces::NTuple{N},
    measures::NTuple{N};
    b_coeffs::Union{NTuple, Nothing}=nothing
) where {N}

    # Validate operator type
    valid_types = (:mass, :stiffness, :gradient, :divergence, :curl_curl, :advection)
    @assert operator_type ∈ valid_types "Unsupported operator type: $operator_type. Must be one of $valid_types"

    # Validate inputs
    @assert !isempty(spaces) "Need at least one space"
    @assert all(s -> s isa Gridap.FESpaces.FESpace, spaces) "All spaces must be FESpace instances"
    @assert length(spaces) == length(measures) "Number of spaces and measures must match"

    # Stage 4.1: Extract all subdomain operators (M, K, G, D, A)
    subdomain_ops = assemble_subdomain_operators(spaces, measures, b_coeffs=b_coeffs)

    # Create wrapped operator (global matrix deferred until needed)
    return TensorProductOperator{operator_type}(subdomain_ops, operator_type, nothing, N)
end

# ═══════════════════════════════════════════════════════════════════════════
# Accessors
# ═══════════════════════════════════════════════════════════════════════════

"""
    get_global_matrix(op::TensorProductOperator) -> Matrix

Assemble and return the global matrix for this operator.

If already cached, returns cached copy. Otherwise, triggers Kronecker assembly
via the appropriate assembler function and caches the result.

# Stage 5 Integration
This function is called during Stage 5 assembly to get the factored, assembled
global operator ready for solving.

# Example
```julia
op = TensorProductOperator(:stiffness, spaces, measures)
A = get_global_matrix(op)  # Triggers kron assembly of K terms
size(A)  # (n_dofs_global, n_dofs_global)
```
"""
function get_global_matrix(op::TensorProductOperator)
    if op.global_matrix === nothing
        op.global_matrix = _assemble_operator_matrix(op)
    end
    return op.global_matrix
end

"""
    get_subdomain_operators(op::TensorProductOperator) -> TensorProductSubdomainOperators

Return the fundamental subdomain operators (M, K, G, D, A) stored in this wrapper.
"""
function get_subdomain_operators(op::TensorProductOperator)
    return op.subdomain_ops
end

"""
    get_operator_type(op::TensorProductOperator) -> Symbol

Return the operator type classification (`:mass`, `:stiffness`, etc.)
"""
function get_operator_type(op::TensorProductOperator)
    return op.operator_type
end

# ═══════════════════════════════════════════════════════════════════════════
# Internal: Kronecker Assembly Dispatch
# ═══════════════════════════════════════════════════════════════════════════

"""
    _assemble_operator_matrix(op::TensorProductOperator) -> Matrix

Internal dispatcher that routes to appropriate Kronecker assembler based on operator type.

# Dispatch Table
- `:mass` → `assemble_mass_tensor()`
- `:stiffness` → `assemble_stiffness_tensor()`
- `:gradient` → `assemble_gradient_tensor()`
- `:divergence` → `assemble_divergence_tensor()`
- `:curl_curl` → `assemble_curl_curl_tensor()`
- `:advection` → `assemble_advection_tensor()`

Each assembler function takes subdomain operators and returns factored global matrix.
"""
function _assemble_operator_matrix(op::TensorProductOperator)
    ops = op.subdomain_ops

    if op.operator_type == :mass
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
