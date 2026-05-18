using Gridap
using GridapTensorProduct
using LinearAlgebra

println("Kronecker assembler: placeholder and smoke checks")

sizes = (2, 3, 4)
idxs = (2, 1, 4)
linear = GridapTensorProduct.tensor_tuple_to_linear_index(idxs, sizes)
recovered = GridapTensorProduct.linear_index_to_tensor_tuple(linear, sizes)

if linear != 20
    error("tensor_tuple_to_linear_index returned $linear, expected 20")
end
if recovered != idxs
    error("linear_index_to_tensor_tuple returned $recovered, expected $idxs")
end

A1 = [1.0 2.0; 3.0 4.0]
A2 = [5.0 6.0; 7.0 8.0]
expected = kron(A2, A1)
actual = GridapTensorProduct.tensor_kron((A1, A2))
assembled = GridapTensorProduct.assemble_tensor_operator(GridapTensorProduct.KroneckerAssembler(), (A1, A2))

if actual != expected
    error("tensor_kron does not match kron(A2, A1)")
end
if assembled != expected
    error("assemble_tensor_operator does not match kron(A2, A1)")
end

# Matrix-free apply must match dense multiplication by the Kronecker matrix.
x = [1.0, -1.0, 2.0, 0.5]
y_dense = expected * x
y_apply = GridapTensorProduct.apply_tensor_operator(GridapTensorProduct.KroneckerAssembler(), (A1, A2), x)
if y_apply != y_dense
    error("apply_tensor_operator does not match dense kron-matrix multiplication")
end

# Tensor RHS assembly should follow the same factor ordering convention.
b1 = [1.0, 2.0]
b2 = [3.0, 4.0]
b_expected = kron(b2, b1)
b_actual = GridapTensorProduct.assemble_tensor_rhs(GridapTensorProduct.KroneckerAssembler(), (b1, b2))
if b_actual != b_expected
    error("assemble_tensor_rhs does not match kron(b2, b1)")
end

# Dense fallback assembly from callback should reproduce the separable matrix.
row_sizes = (2, 2)
col_sizes = (2, 2)
entry_fun = (i, j) -> A1[i[1], j[1]] * A2[i[2], j[2]]
A_fallback = GridapTensorProduct.assemble_tensor_operator(
    GridapTensorProduct.FallbackAssembler(),
    row_sizes,
    col_sizes,
    entry_fun,
)
if A_fallback != expected
    error("Fallback entry-callback assembly does not reproduce separable Kronecker matrix")
end

# Sum-of-Kronecker assembly should match an explicit linear combination.
B1 = [2.0 0.0; 0.0 1.0]
B2 = [1.0 1.0; 0.0 1.0]
expected_sum = 1.5 .* kron(A2, A1) .- 0.25 .* kron(B2, B1)

assembled_sum = GridapTensorProduct.assemble_tensor_operator(
    GridapTensorProduct.KroneckerAssembler(),
    ((1.5, (A1, A2)), (-0.25, (B1, B2))),
    Val(:sum),
)

if assembled_sum != expected_sum
    error("Sum-of-Kronecker assembly does not match explicit linear combination")
end
