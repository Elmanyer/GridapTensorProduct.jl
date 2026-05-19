"""
    TensorProductFEMOperators

Complete implementation of FEM operators for tensor-product domains following
Kronecker product decomposition of weak forms.

# Purpose

This module translates global weak-form operators on tensor-product domains Ω = Ω₁ ⊗ Ω₂ ⊗ ... ⊗ Ωₙ
into factored Kronecker products of subdomain operators. 

# Six Global Operators Supported

For a tensor domain with N subdomains, the following global operators are decomposed:

## 1. Mass Matrix (M)
Full Kronecker product:
```
M = M⁽¹⁾ ⊗ M⁽²⁾ ⊗ ... ⊗ M⁽ᴺ⁾
```
where M⁽ᵏ⁾ = ∫ φᵢ φⱼ dxₖ (L² inner product on subdomain k)

## 2. Stiffness Matrix (A) — Laplacian on Tensor Domain
Sum of N terms with stiffness at each coordinate:
```
A = Σ_{k=1}^N [ M⁽¹⁾ ⊗ ... ⊗ K⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾ ]
```
where K⁽ᵏ⁾ = ∫ ∇φᵢ · ∇φⱼ dxₖ (stiffness in direction k)

Each term couples the k-th direction's stiffness with mass in all other directions.
Total of N terms in the sum.

## 3. Gradient Operator (G)
Block-column assembly (⊕ denotes vertical stacking):
```
G = ⊕_{k=1}^N [ M⁽¹⁾ ⊗ ... ⊗ G⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾ ]
```
where G⁽ᵏ⁾ = ∫ (∂φᵢ/∂xₖ) φⱼ dxₖ (rectangular: dofs_k × dofs_k)

Output dimension: (N × standard_dofs) × (total_dofs)
Represents the weak gradient operator ∇ on tensor domains with N components.

## 4. Divergence Operator (B)
Transpose of gradient:
```
B = Gᵀ
```

## 5. Curl-Curl Operator (C)
Anti-symmetric coupling of N(N-1) pairs via cross-derivatives:
```
C = Σ_{k=1}^N Σ_{l≠k} [ (⊗_{j≠k,l} M⁽ʲ⁾) ⊗ (Cᵀ D⁽ᵏ⁾ᵀ D⁽ˡ⁾ - D⁽ˡ⁾ᵀ D⁽ᵏ⁾) ]
```
where D⁽ᵏ⁾ = ∫ φᵢ (d/dxₖ φⱼ) dxₖ (directional derivative matrix)

Couples derivatives in different coordinate directions.

## 6. Advection Matrix (T) — For Separable Velocity
Sum decomposition for constant or separable velocity field b = (b₁, b₂, ..., bₙ):
```
T = Σ_{k=1}^N bₖ · [ M⁽¹⁾ ⊗ ... ⊗ A⁽ᵏ⁾ ⊗ ... ⊗ M⁽ᴺ⁾ ]
```
where A⁽ᵏ⁾ = ∫ φᵢ (bₖ · ∇ₖ φⱼ) dxₖ (advection in direction k with velocity bₖ)

Represents convection (u · ∇)u for separable velocity.

# Implementation Strategy

1. **Subdomain Extraction** (Part 1):
   - `assemble_subdomain_operators()` computes fundamental matrices {M⁽ᵏ⁾, K⁽ᵏ⁾, G⁽ᵏ⁾, D⁽ᵏ⁾, A⁽ᵏ⁾}}
   - Returns wrapped `TensorProductSubdomainOperators` struct

2. **Kronecker Assembly** (Part 2):
   - Six public functions assemble each global operator type
   - Uses `Kronecker.jl` for lazy evaluation (efficient memory usage)
   - Handle sum-of-Kronecker structures (stiffness, gradient, curl-curl, advection)
   - Handle block-column assembly (gradient, divergence)

3. **Extensibility**:
   - Add new operators by implementing corresponding Kronecker formula
   - Follow existing naming convention: `assemble_<operator>_tensor(ops)`

# Usage Example

```julia
# Stage 4.1: Extract subdomain operators
spaces = (V1, V2)  # Two 1D test spaces
measures = (dΩ1, dΩ2)  # Two 1D measures
ops = assemble_subdomain_operators(spaces, measures)

# Stage 4.2: Assemble desired global operator
A = assemble_stiffness_tensor(ops)        # Global stiffness (Laplacian)
M = assemble_mass_tensor(ops)             # Global mass matrix
G = assemble_gradient_tensor(ops)         # Global gradient operator
T = assemble_advection_tensor(ops)        # Global advection (if needed)
```

# References

Fundamental matrices follow standard weak form definitions:
- M⁽ᵏ⁾: ∫_Ωₖ φᵢ φⱼ dxₖ
- K⁽ᵏ⁾: ∫_Ωₖ (dφᵢ/dxₖ)(dφⱼ/dxₖ) dxₖ
- G⁽ᵏ⁾: ∫_Ωₖ (dφᵢ/dxₖ) φⱼ dxₖ

Kronecker decompositions assume:
- Separable weak forms (sum/products of 1D integrals)
- Lazy evaluation via Kronecker.jl (memory efficient)
"""

# Import required functions
import Kronecker: kronecker, KroneckerProduct
import Gridap.FESpaces: FESpace, TrialFESpace, TestFESpace
import Gridap.FESpaces: get_matrix, AffineFEOperator
import Gridap: ∫
import Gridap.Fields: ∇

export TensorProductSubdomainOperators
export TensorProductGlobalOperator
export assemble_subdomain_operators
export assemble_mass_tensor
export assemble_stiffness_tensor
export assemble_gradient_tensor
export assemble_divergence_tensor
export assemble_curl_curl_tensor
export assemble_advection_tensor

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Subdomain Operator Extraction
# ═══════════════════════════════════════════════════════════════════════════

"""
    TensorProductSubdomainOperators{N}

Lazy container for fundamental subdomain operators Ω_k (k = 1, ..., N).

Each of the five per-subdomain matrix types (M, K, G, D, A) is computed on first
access and cached. Operators never needed by the weak form are never assembled.

# Fields (public, accessed via property syntax)
- `spaces`   — N test FE spaces
- `measures` — N integration measures
- `M_ops`    — mass matrices M^(k): `∫ φᵢφⱼ dxₖ` (assembled on first access)
- `K_ops`    — stiffness matrices K^(k): `∫ ∇φᵢ·∇φⱼ dxₖ`
- `G_ops`    — gradient matrices G^(k): `∫ (∂φᵢ/∂xₖ) φⱼ dxₖ`
- `D_ops`    — derivative matrices D^(k) (same form as G; used in curl-curl)
- `A_ops`    — advection matrices A^(k) (requires `b_coeffs` at construction)

Access `ops.M_ops`, `ops.K_ops`, etc. as before; the matrices are assembled on
demand and results are cached for subsequent calls.
"""
mutable struct TensorProductSubdomainOperators{N}
    spaces::NTuple{N, FESpace}
    measures::NTuple{N, Measure}
    b_coeffs::Union{NTuple, Nothing}          # advection coefficients (for lazy A)
    _M_ops::Vector{Union{Matrix, Nothing}}    # cache: nothing = not yet assembled
    _K_ops::Vector{Union{Matrix, Nothing}}
    _G_ops::Vector{Union{Matrix, Nothing}}
    _D_ops::Vector{Union{Matrix, Nothing}}
    _A_ops::Vector{Union{Matrix, Nothing}}
end

# ── Lazy accessor internals ───────────────────────────────────────────────────

function _assemble_M_k!(ops::TensorProductSubdomainOperators, k::Int)
    Uk = TrialFESpace(ops.spaces[k])
    ops._M_ops[k] = Matrix(get_matrix(AffineFEOperator(
        (u, v) -> ∫(u * v) * ops.measures[k],
        (v)    -> ∫(0 * v) * ops.measures[k],
        Uk, ops.spaces[k])))
end

function _assemble_K_k!(ops::TensorProductSubdomainOperators, k::Int)
    Uk = TrialFESpace(ops.spaces[k])
    ops._K_ops[k] = Matrix(get_matrix(AffineFEOperator(
        (u, v) -> ∫(∇(u) ⋅ ∇(v)) * ops.measures[k],
        (v)    -> ∫(0 * v) * ops.measures[k],
        Uk, ops.spaces[k])))
end

function _assemble_G_k!(ops::TensorProductSubdomainOperators, k::Int)
    Uk = TrialFESpace(ops.spaces[k])
    ops._G_ops[k] = Matrix(extract_gradient_matrix(Uk, ops.spaces[k], ops.measures[k]))
end

function _assemble_D_k!(ops::TensorProductSubdomainOperators, k::Int)
    Uk = TrialFESpace(ops.spaces[k])
    ops._D_ops[k] = Matrix(extract_derivative_matrix(Uk, ops.spaces[k], ops.measures[k]))
end

function _assemble_A_k!(ops::TensorProductSubdomainOperators, k::Int)
    Uk = TrialFESpace(ops.spaces[k])
    b_k = (ops.b_coeffs !== nothing && k <= length(ops.b_coeffs)) ?
          ops.b_coeffs[k] : 1.0
    ops._A_ops[k] = Matrix(extract_advection_matrix(
        Uk, ops.spaces[k], ops.measures[k], b_k))
end

function _get_all!(cache::Vector{Union{Matrix,Nothing}}, assemble_k!::Function,
                   ops::TensorProductSubdomainOperators{N}) where N
    for k in 1:N
        cache[k] === nothing && assemble_k!(ops, k)
    end
    return NTuple{N, Matrix}(cache)
end

# ── Public accessors (triggered by Base.getproperty) ─────────────────────────

_get_M_ops!(ops::TensorProductSubdomainOperators) =
    _get_all!(ops._M_ops, _assemble_M_k!, ops)

_get_K_ops!(ops::TensorProductSubdomainOperators) =
    _get_all!(ops._K_ops, _assemble_K_k!, ops)

_get_G_ops!(ops::TensorProductSubdomainOperators) =
    _get_all!(ops._G_ops, _assemble_G_k!, ops)

_get_D_ops!(ops::TensorProductSubdomainOperators) =
    _get_all!(ops._D_ops, _assemble_D_k!, ops)

_get_A_ops!(ops::TensorProductSubdomainOperators) =
    _get_all!(ops._A_ops, _assemble_A_k!, ops)

# ── Transparent property access — backward compatible ──────────────────────

function Base.getproperty(ops::TensorProductSubdomainOperators, s::Symbol)
    s === :M_ops && return _get_M_ops!(ops)
    s === :K_ops && return _get_K_ops!(ops)
    s === :G_ops && return _get_G_ops!(ops)
    s === :D_ops && return _get_D_ops!(ops)
    s === :A_ops && return _get_A_ops!(ops)
    return getfield(ops, s)
end

"""
    assemble_subdomain_operators(spaces::NTuple{N, FESpace}, measures::NTuple{N, Measure};
                                b_coeffs::Union{NTuple,Nothing}=nothing,
                                velocity_direction::Int=1) -> TensorProductSubdomainOperators

Assemble all fundamental subdomain operators for tensor-product FE spaces.

For each subdomain k ∈ 1...N:
  - M^(k): mass matrix via standard Gridap assembly
  - K^(k): stiffness matrix (Laplacian component in 1D)
  - G^(k): gradient matrix (directional derivative)
  - D^(k): directional derivative on Ω_k
  - A^(k): advection matrix for velocity field b_k

Parameters:
  - `spaces::NTuple{N,TestFESpace}` - 1D trial/test spaces (must be TestFESpace)
  - `measures::NTuple{N,Measure}` - Integration measures per subdomain
  - `b_coeffs::Union{NTuple,Nothing}` - Velocity coefficients (b_1, ..., b_N)
  - `velocity_direction::Int` - Which direction velocity acts (default 1)

Returns: TensorProductSubdomainOperators{N} with all subdomain matrices

Key points:
  - Extracts actual G^(k), D^(k), A^(k) matrices using extraction functions
  - Respects directional structure (k-th operator in k-th subdomain)
  - Handles variable number of subdomains N automatically
"""
function assemble_subdomain_operators(
    spaces::NTuple{N, FESpace},
    measures::NTuple{N, Measure};
    b_coeffs::Union{NTuple, Nothing}=nothing
) where N
    noth() = Vector{Union{Matrix, Nothing}}(fill(nothing, N))
    return TensorProductSubdomainOperators{N}(
        spaces, measures, b_coeffs,
        noth(), noth(), noth(), noth(), noth())
end

# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Kronecker-Product Operators (N = 2, 3 special cases first)
# ═══════════════════════════════════════════════════════════════════════════

"""
    assemble_mass_tensor(ops::TensorProductSubdomainOperators) -> Matrix

Global mass matrix via full Kronecker product:
  M = M^(1) ⊗ M^(2) ⊗ ... ⊗ M^(N)

No special structure - all matrices play same role.
"""
function assemble_mass_tensor(ops::TensorProductSubdomainOperators{N}) where N
    M = ops.M_ops  # trigger lazy assembly once; M is an NTuple
    M_global = M[1]
    for k in 2:N
        M_global = kronecker(M[k], M_global)
    end
    return collect(M_global)
end

"""
    assemble_stiffness_tensor(ops::TensorProductSubdomainOperators) -> Matrix

Global stiffness matrix via sum of Kronecker terms:
  A = Σ_{k=1}^N [ M^(1) ⊗ ... ⊗ K^(k) ⊗ ... ⊗ M^(N) ]

Each term places K^(k) at position k and M^(j) elsewhere.
Total of N terms in the sum.
"""
function assemble_stiffness_tensor(ops::TensorProductSubdomainOperators{N}) where N
    M = ops.M_ops  # trigger M assembly once
    K = ops.K_ops  # trigger K assembly once

    n_total = prod(size(M[k], 1) for k in 1:N)
    A_global = zeros(n_total, n_total)

    for k in 1:N
        A_term = (k == 1) ? K[1] : M[1]
        for j in 2:N
            factor = (j == k) ? K[j] : M[j]
            A_term = kronecker(factor, A_term)
        end
        A_global += collect(A_term)
    end

    return A_global
end

"""
    assemble_gradient_tensor(ops::TensorProductSubdomainOperators)

Global gradient operator via block-column assembly:
  G = ⊕_k [ M^(1) ⊗ ... ⊗ G^(k) ⊗ ... ⊗ M^(N) ]

Block column (⊕) means vertical stacking of matrix blocks.
Total dimension: (sum of block_k_rows) × (total_dofs)
N blocks, each coupling gradient at position k with mass elsewhere.

Physical meaning: Weak gradient operator ∇ on tensor product domain,
with output dimension = N × (standard domain dimension).
"""
function assemble_gradient_tensor(ops::TensorProductSubdomainOperators{N}) where N
    M = ops.M_ops  # trigger M assembly once
    G = ops.G_ops  # trigger G assembly once
    blocks = Vector{Matrix}(undef, N)

    for k in 1:N
        G_block = (k == 1) ? G[1] : M[1]
        for j in 2:N
            factor = (j == k) ? G[j] : M[j]
            G_block = kronecker(factor, G_block)
        end
        blocks[k] = collect(G_block)
    end

    return vcat(blocks...)
end

"""
    assemble_divergence_tensor(ops::TensorProductSubdomainOperators)

Global divergence operator (transpose of gradient):
  B = G^T = ⊕_k [ M^(1) ⊗ ... ⊗ G^(k)^T ⊗ ... ⊗ M^(N) ]

This is the adjoint of the gradient operator.
"""
function assemble_divergence_tensor(ops::TensorProductSubdomainOperators)
    G = assemble_gradient_tensor(ops)
    return transpose(G)
end

"""
    assemble_curl_curl_tensor(ops::TensorProductSubdomainOperators; curl_op=nothing)

Global curl-curl operator (N-dimensional):
  C = Σ_k Σ_{l≠k} (⊗_{j≠k,l} M^(j)) ⊗ (C^T D^(k)^T D^(l) - D^(l)^T D^(k))

Couples pairs (k, l) of subdomains via antisymmetric derivatives.
Total of N(N-1) terms in the nested sum.

Parameters:
  - `ops`: Subdomain operators (includes D^(k))
  - `curl_op`: Curl operator for problem-specific coupling (optional)
             If not provided, uses standard 2D rotation (-id for 2D)

Physical meaning: ∫(∇×u)·(∇×v) dΩ for vector-valued fields

Mathematical Structure:
  - Outer loop over subdomain pairs (k, l) where l ≠ k
  - For each pair, outer factor carries all M^(j) except those involved in pair
  - Inner factor contains derivative coupling and optional curl transformation
  - Contribution couples directional derivatives at different orders

For 2D (N=2): Two terms (k,l)=(1,2) and (2,1)
  T_{1,2} ⊗ (C^T D^(1)^T D^(2) - D^(2)^T D^(1))
where T_{1,2} is the outer mass tensor product

Returns: Sparse matrix of size (n_dofs, n_dofs)
"""
function assemble_curl_curl_tensor(ops::TensorProductSubdomainOperators{N};
                                    curl_op=nothing) where N
    M = ops.M_ops  # trigger M assembly once
    D = ops.D_ops  # trigger D assembly once

    n_total = prod(size(M[k], 1) for k in 1:N)
    C_global = zeros(n_total, n_total)

    for k in 1:N
        for l in 1:N
            l == k && continue

            other_positions = [j for j in 1:N if j != k && j != l]
            if isempty(other_positions)
                outer_factor = nothing
            else
                outer_factor = M[other_positions[1]]
                for j in other_positions[2:end]
                    outer_factor = kronecker(M[j], outer_factor)
                end
            end

            C_matrix = (curl_op === nothing) ? I(size(D[k], 1)) : curl_op

            inner_factor = (transpose(C_matrix) * transpose(D[k]) * D[l]
                            - transpose(D[l]) * D[k])

            pair_term = (outer_factor === nothing) ? inner_factor :
                        collect(kronecker(outer_factor, inner_factor))

            C_global += pair_term
        end
    end

    return C_global
end

"""
    assemble_advection_tensor(ops::TensorProductSubdomainOperators,
                              b_coeffs::NTuple)

Global advection operator for separable velocity field:
  T = Σ_k b_k · M^(1) ⊗ ... ⊗ A^(k) ⊗ ... ⊗ M^(N)

where A^(k)_ij = ∫ φ_i (b_k · ∇_k φ_j) dx_k

Mathematical Structure:
  - Single-subdomain sum: each term places A^(k) at position k
  - Other positions filled with M^(j)
  - Summed with coefficient b_k scaling each term
  - Result: N terms total in the sum

Physical meaning: Convection operator for separable velocity field
  (b · ∇u, v) = ∫ v (b · ∇u) dΩ

Separability constraint:
  - Velocity field must decompose: b(x₁,...,xₙ) = (b₁(x₁), ..., bₙ(xₙ))
  - Each component b_k depends only on x_k coordinate
  - Scalar coefficients b_k passed as tuple argument

Returns: Sparse or dense matrix (typically asymmetric) of size (n_dofs, n_dofs)

Example (2D Poisson with convection):
  ops = assemble_subdomain_operators((Vx, Vy), (dΩx, dΩy))
  T = assemble_advection_tensor(ops, (2.0, 1.5))  # b = (2.0, 1.5)
  # Result: T = 2.0*(A^(x) ⊗ M^(y)) + 1.5*(M^(x) ⊗ A^(y))
"""
function assemble_advection_tensor(ops::TensorProductSubdomainOperators{N},
                                   b_coeffs::NTuple) where N
    @assert length(b_coeffs) == N "Velocity field must have N components"

    M = ops.M_ops  # trigger M assembly once
    A = ops.A_ops  # trigger A assembly once

    n_total = prod(size(M[k], 1) for k in 1:N)
    T_global = zeros(n_total, n_total)

    for k in 1:N
        T_term = (k == 1) ? A[1] : M[1]
        for j in 2:N
            factor = (j == k) ? A[j] : M[j]
            T_term = kronecker(factor, T_term)
        end
        T_global += b_coeffs[k] * collect(T_term)
    end

    return T_global
end

# ═══════════════════════════════════════════════════════════════════════════
# PART 3: High-Level API - Form Translation to Operators
# ═══════════════════════════════════════════════════════════════════════════

"""
    TensorProductGlobalOperator

High-level representation of a bilinear form on tensor domain,
with automatic translation to Kronecker structure.

Supports:
  - Forms involving mass, stiffness, gradient, divergence
  - Separable advection
  - Automatic detection of operator type
"""
struct TensorProductGlobalOperator
    operator_type::Symbol  # :mass, :stiffness, :gradient, :divergence, :curl, :advection
    matrix::Matrix
    subdomain_ops::TensorProductSubdomainOperators
end


"""
Helper functions to extract fundamental subdomain operator matrices from Gridap FE spaces.

Implements extraction of:
1. G^(k): Gradient matrices (rectangular: n_dofs × n_dofs·dim)
2. D^(k): Directional derivative matrices
3. A^(k): Advection matrices for separable velocity fields
"""

# Import necessary Gridap functionality
import Gridap.FESpaces: FESpace, TrialFESpace, TestFESpace
import Gridap.FESpaces: get_matrix, AffineFEOperator
import Gridap: ∫
import Gridap.Fields: ∇
import LinearAlgebra: mul!

export extract_gradient_matrix
export extract_derivative_matrix
export extract_advection_matrix

# ═══════════════════════════════════════════════════════════════════════════
# GRADIENT MATRIX EXTRACTION: G^(k)
# ═══════════════════════════════════════════════════════════════════════════

"""
    extract_gradient_matrix(U::FESpace, V::FESpace, measure::Measure, direction::Int=1)
    → G::Matrix (square: n_dofs × n_dofs)

Extract the directional gradient matrix for subdomain Ω_k.

G^(k)_ij = ∫ (∂φ_i/∂x_k) φ_j dx_k

Physical meaning: Weak directional gradient operator in one direction.

Algorithm:
1. Assemble bilinear form: (∂u/∂x_{direction}, v)
2. Extract matrix via Gridap's AFfineFEOperator
3. Returns square matrix of size (n_dofs, n_dofs)

Parameters:
- `U::FESpace` - Trial space (TrialFESpace recommended)
- `V::FESpace` - Test space (TestFESpace recommended)
- `measure::Measure` - Integration measure for subdomain
- `direction::Int` - Coordinate direction (1=x, 2=y, 3=z, etc.)

Returns: Sparse matrix of size (n_dofs, n_dofs)
"""
function extract_gradient_matrix(U::FESpace, V::FESpace, measure::Measure)
    # Each subdomain is 1D: ∇(u) is VectorValue{1}. Dot product with the 1D unit
    # vector extracts ∂u/∂x as a scalar (direct indexing ∇(u)[1] fails in Gridap's
    # symbolic framework because getindex is not defined on SingleFieldFEBasis).
    op_grad = AffineFEOperator(
        (u, v) -> ∫( (∇(u) ⋅ VectorValue(1.0)) * v ) * measure,
        (v) -> ∫(0*v) * measure,
        U, V
    )
    return get_matrix(op_grad)
end

# ═══════════════════════════════════════════════════════════════════════════
# DIRECTIONAL DERIVATIVE MATRIX EXTRACTION: D^(k)
# ═══════════════════════════════════════════════════════════════════════════

"""
    extract_derivative_matrix(U::FESpace, V::FESpace, measure::Measure, direction::Int=1)
    → D::Matrix (square: n_dofs × n_dofs)

Extract the first-order directional derivative matrix.

D^(k)_ij = ∫ (∂φ_i/∂x_k) φ_j dx_k

Similar to gradient but scalar-valued (single direction component).

Used in curl-curl operator: C = Σ_k Σ_{l≠k} ... (D^(k)^T D^(l) - D^(l)^T D^(k)) ...

Algorithm:
1. Assemble weak form with directional derivative: ∫ (∂u/∂x_direction) v dx
2. Extract matrix via get_matrix()
3. Returns square matrix for pair coupling

Parameters:
- `U::FESpace` - Trial space
- `V::FESpace` - Test space
- `measure::Measure` - Integration measure
- `direction::Int` - Coordinate direction (1=x, 2=y, 3=z, etc.)

Returns: Matrix of size (n_dofs, n_dofs)

Supports: Arbitrary N-dimensional domains (tested up to 3D)
"""
function extract_derivative_matrix(U::FESpace, V::FESpace, measure::Measure)
    op_deriv = AffineFEOperator(
        (u, v) -> ∫( (∇(u) ⋅ VectorValue(1.0)) * v ) * measure,
        (v) -> ∫(0*v) * measure,
        U, V
    )
    return get_matrix(op_deriv)
end

# ═══════════════════════════════════════════════════════════════════════════
# ADVECTION MATRIX EXTRACTION: A^(k)
# ═══════════════════════════════════════════════════════════════════════════

"""
    extract_advection_matrix(U::FESpace, V::FESpace, measure::Measure,
                            b_coefficient::Real; direction::Int=1)
    → A::Matrix (square: n_dofs × n_dofs, ASYMMETRIC)

Extract advection matrix for separable velocity field component.

For velocity field b(x_k) and direction k:
    A^(k)_ij = ∫ φ_i (b(x_k) · ∂φ_j/∂x_k) dx_k

Physical meaning: Matrix representation of convection operator
    (b · ∇u, v) = ∫ v (b · ∂u/∂x) dx

Properties:
- **Asymmetric:** Unlike mass/stiffness, advection breaks symmetry
- Linear in coefficient: A(2b) = 2·A(b)
- Directional: Direction parameter specifies which coordinate

Parameters:
- `U::FESpace` - Trial space (TrialFESpace)
- `V::FESpace` - Test space (TestFESpace)
- `measure::Measure` - Integration measure for subdomain
- `b_coefficient::Real` - Scalar velocity component (constant)
- `direction::Int` - Coordinate direction (1=x, 2=y, 3=z, etc.)

Returns: Sparse/dense matrix (typically asymmetric) of size (n_dofs, n_dofs)

Usage:
```julia
A_x = extract_advection_matrix(Ux, Vx, dΩx, 2.0, direction=1)  # b_x = 2.0
```

Note: For time-dependent advection, scale coefficient appropriately for your formulation.
"""
function extract_advection_matrix(
    U::FESpace, V::FESpace, measure::Measure, b_coefficient::Real
)
    op_advec = AffineFEOperator(
        (u, v) -> ∫( v * b_coefficient * (∇(u) ⋅ VectorValue(1.0)) ) * measure,
        (v) -> ∫(0*v) * measure,
        U, V
    )
    return get_matrix(op_advec)
end

# assemble_weak_form has moved to TensorProductOperator.jl (Stage 4 translation layer)

