import Gridap.FESpaces: get_cell_dof_ids
import Gridap.FESpaces: get_free_dof_ids
import Gridap.FESpaces: num_free_dofs
import LinearAlgebra: kron

export tensor_tuple_to_linear_index
export linear_index_to_tensor_tuple
export tensor_matrix_sizes
export tensor_vector_sizes
export tensor_kron
export tensor_kron_vector
export apply_tensor_operator
export assemble_tensor_rhs
export assemble_tensor_operator

abstract type TensorProductAssembler end

"""
Kronecker-based assembly strategy for separable tensor-product operators.

The first implementation keeps the type light-weight and routes all tensor
index access through the `TensorProductFESpace` wrapper. The actual algebraic
assembly kernels will be filled in next.
"""
struct KroneckerAssembler <: TensorProductAssembler
end

"""
Fallback cell-wise assembly strategy for non-separable operators.
"""
struct FallbackAssembler <: TensorProductAssembler
end

TensorProductAssembler(::Val{:kron}) = KroneckerAssembler()
TensorProductAssembler(::Val{:fallback}) = FallbackAssembler()

tensor_cell_dof_ids(f::TensorProductFESpace) = get_cell_dof_ids(f)
tensor_free_dof_ids(f::TensorProductFESpace) = get_free_dof_ids(f)

"""
Convert a tensor tuple index to a 1-based linear index using the current
wrapper convention.

The first subdomain index varies fastest, so the last subdomain is the slowest
varying dimension. This matches the `kron(A₂, A₁)` convention used below.
"""
function tensor_tuple_to_linear_index(idxs::NTuple{N,Int}, sizes::NTuple{N,Int}) where {N}
    linear = 1
    stride = 1
    for i in 1:N
        linear += (idxs[i] - 1) * stride
        stride *= sizes[i]
    end
    return linear
end

"""
Inverse of `tensor_tuple_to_linear_index`.

This helper is intentionally explicit so the assembly layer can recover tensor
coordinates for diagnostics or for future sparse assembly kernels.
"""
function linear_index_to_tensor_tuple(linear::Integer, sizes::NTuple{N,Int}) where {N}
    remaining = linear - 1
    values = Vector{Int}(undef, N)
    for (i, size) in enumerate(sizes)
        values[i] = rem(remaining, size) + 1
        remaining = div(remaining, size)
    end
    return Tuple(values)
end

"""
Return the tensor-product size tuple of a vector of matrices.

This is a tiny helper used by the Kronecker assembler to document the size of
the resulting operator and to support future index-based assembly kernels.
"""
function tensor_matrix_sizes(mats::Tuple)
    return tuple((size(A, 1) for A in mats)...)
end

"""
Return the per-subdomain vector lengths for a tuple of vectors.
"""
function tensor_vector_sizes(vs::Tuple)
    return tuple((length(v) for v in vs)...)
end

"""
Validate that matrix factors can form a tensor product operator.

Returns `(row_sizes, col_sizes)` where each tuple stores the per-subdomain
matrix dimensions.
"""
function _validate_tensor_matrix_factors(mats::Tuple)
    @assert !isempty(mats) "At least one matrix factor is required"
    @assert all(A -> A isa AbstractMatrix, mats) "All factors must be matrices"
    row_sizes = tuple((size(A, 1) for A in mats)...)
    col_sizes = tuple((size(A, 2) for A in mats)...)
    return row_sizes, col_sizes
end

"""
Build a Kronecker product using the tensor ordering already used by the wrapper.

For a pair `(A1, A2)` this returns `kron(A2, A1)`, which matches the current
`tp_pair_to_dof` convention where the first subdomain index varies fastest.
"""
function tensor_kron(mats::Tuple)
    @assert !isempty(mats) "tensor_kron needs at least one matrix"
    acc = mats[1]
    for i in 2:length(mats)
        acc = kron(mats[i], acc)
    end
    return acc
end

"""
Tensor-product of vectors with the same ordering convention as `tensor_kron`.

For `(v1, v2)` this returns `kron(v2, v1)`.
"""
function tensor_kron_vector(vs::Tuple)
    @assert !isempty(vs) "tensor_kron_vector needs at least one vector"
    @assert all(v -> v isa AbstractVector, vs) "All factors must be vectors"
    acc = vs[1]
    for i in 2:length(vs)
        acc = kron(vs[i], acc)
    end
    return acc
end

"""
Apply a tensor-product operator to a global vector without forming the global
Kronecker matrix explicitly.

This implementation uses the tuple-index representation to keep the mapping
transparent. It is not the most optimized kernel yet, but it provides a
correct matrix-free path that the library can optimize later.
"""
function apply_tensor_operator(::KroneckerAssembler, factors::Tuple, x::AbstractVector)
    row_sizes, col_sizes = _validate_tensor_matrix_factors(factors)
    ncols = prod(col_sizes)
    nrows = prod(row_sizes)
    @assert length(x) == ncols "Input vector length does not match tensor operator columns"

    y = zeros(eltype(x), nrows)

    for i_lin in 1:nrows
        i_idx = linear_index_to_tensor_tuple(i_lin, row_sizes)
        acc = zero(eltype(x))

        for j_lin in 1:ncols
            j_idx = linear_index_to_tensor_tuple(j_lin, col_sizes)
            coeff = one(eltype(x))
            for d in 1:length(factors)
                coeff *= factors[d][i_idx[d], j_idx[d]]
            end
            acc += coeff * x[j_lin]
        end

        y[i_lin] = acc
    end

    return y
end

"""
Assemble tensor-product RHS vectors from separable factors.

Given `(b1, b2, ..., bN)` this returns the global tensor RHS vector with the
same index ordering convention used across this package.
"""
function assemble_tensor_rhs(::KroneckerAssembler, factors::Tuple)
    return tensor_kron_vector(factors)
end

"""
Assemble a purely factorized tensor operator from a tuple of subdomain matrices.

This is the first concrete Kronecker assembly path in the library. It does not
yet inspect a weak form; instead it directly maps tensor factors to a global
operator using the wrapper's tensor ordering convention.
"""
function assemble_tensor_operator(::KroneckerAssembler, factors::Tuple)
    @assert all(f -> f isa AbstractMatrix, factors) "assemble_tensor_operator expects a tuple of matrices"
    return tensor_kron(factors)
end

"""
Assemble a linear combination of tensor-product operators.

Each entry in `terms` can be either:
- `factors::Tuple` representing one Kronecker term with unit weight, or
- `(α, factors)` where `α` is a scalar coefficient and `factors::Tuple`.

This is the first building block to represent weak forms that become sums of
Kronecker products at algebraic level.
"""
function assemble_tensor_operator(::KroneckerAssembler, terms::Tuple, ::Val{:sum})
    @assert !isempty(terms) "At least one Kronecker term is required"

    A = nothing
    for term in terms
        α, factors = _normalize_kron_term(term)
        Ak = assemble_tensor_operator(KroneckerAssembler(), factors)
        if A === nothing
            A = α * Ak
        else
            A .+= α .* Ak
        end
    end

    return A
end

"""
Assemble a tensor operator from a tuple of factors and a tensor-product space.

The space is accepted so the assembler API can evolve toward a full FE operator
builder while still supporting the simple dense Kronecker path.
"""
function assemble_tensor_operator(::KroneckerAssembler, op, V::TensorProductFESpace)
    if op isa Tuple && all(f -> f isa AbstractMatrix, op)
        return assemble_tensor_operator(KroneckerAssembler(), op)
    end
    error("Kronecker assembly currently expects a tuple of subdomain matrices or a vararg matrix input.")
end

"""
Fallback assembler entry point.

This is intentionally a hard error for now so we do not silently return a
matrix built with the wrong semantics. The next implementation step will fill
this in with a cell-wise FE assembly path.
"""
function assemble_tensor_operator(::FallbackAssembler, op, V::TensorProductFESpace)
    error("Fallback tensor assembly is not implemented yet; tensor-space plumbing is ready and will be filled in next.")
end

"""
Dense fallback tensor assembly from an entry callback.

`entry_fun` receives `(i_tuple, j_tuple)` and returns the scalar matrix entry.
This is the first non-separable-capable assembly path and is intended as the
reference implementation for correctness checks.
"""
function assemble_tensor_operator(::FallbackAssembler, row_sizes::NTuple{N,Int}, col_sizes::NTuple{N,Int}, entry_fun::Function) where {N}
    nrows = prod(row_sizes)
    ncols = prod(col_sizes)
    A = zeros(Float64, nrows, ncols)

    for i_lin in 1:nrows
        i_idx = linear_index_to_tensor_tuple(i_lin, row_sizes)
        for j_lin in 1:ncols
            j_idx = linear_index_to_tensor_tuple(j_lin, col_sizes)
            A[i_lin, j_lin] = entry_fun(i_idx, j_idx)
        end
    end

    return A
end

"Normalize one Kronecker sum term to `(coefficient, factors_tuple)` form." 
function _normalize_kron_term(term)
    if term isa Tuple && length(term) == 2 && term[2] isa Tuple
        α = term[1]
        factors = term[2]
        @assert all(A -> A isa AbstractMatrix, factors) "Each Kronecker term needs a tuple of matrices"
        return α, factors
    elseif term isa Tuple && all(A -> A isa AbstractMatrix, term)
        return 1.0, term
    else
        error("Invalid Kronecker term. Use `factors::Tuple` or `(α, factors)`.")
    end
end
