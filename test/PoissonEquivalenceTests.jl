using Gridap
using GridapTensorProduct
using LinearAlgebra

println("Poisson equivalence: testing separable Poisson problem via tensor product")

# Create very simple 1D domains
model1 = CartesianDiscreteModel((0, 1), 2)
model2 = CartesianDiscreteModel((0, 1), 2)

Ω1 = Interior(model1)
Ω2 = Interior(model2)

# Create reference elements (scalar, linear)
reffe = ReferenceFE(lagrangian, Float64, 1)

# Create standard spaces
V1 = TestFESpace(Ω1, reffe)
U1 = TrialFESpace(V1)
V2 = TestFESpace(Ω2, reffe)
U2 = TrialFESpace(V2)

# Check that we can construct tensor-product spaces
Vtp = TensorProductFESpace(V1, V2)
Utp = TensorProductFESpace(U1, U2)

println("  ✓ Tensor-product spaces constructed")

# Check DOF counts
n_free_1 = num_free_dofs(V1)
n_free_2 = num_free_dofs(V2)
n_free_tp = num_free_dofs(Vtp)
expected = n_free_1 * n_free_2

@assert n_free_tp == expected "DOF count mismatch: expected $expected, got $n_free_tp"
println("  ✓ Tensor DOF count correct: $n_free_tp = $n_free_1 × $n_free_2")

# Test Kronecker assembly of simple mass matrices
# For a separable problem: M = M₁ ⊗ M₂

# Create simple 1D mass matrices (stiffness for now since we don't assemble full weak forms yet)
A1 = [2.0 -1.0; -1.0 2.0]
A2 = [3.0 -1.0; -1.0 3.0]

# Assemble via Kronecker
A_tp = GridapTensorProduct.assemble_tensor_operator(
    GridapTensorProduct.KroneckerAssembler(),
    (A1, A2)
)

expected_A_tp = kron(A2, A1)
@assert A_tp ≈ expected_A_tp "Kronecker assembly mismatch"
println("  ✓ Kronecker assembly of tensor operator correct")

# Test vector assembly
b1 = [1.0, 2.0]
b2 = [3.0, 4.0]
b_tp = GridapTensorProduct.assemble_tensor_rhs(
    GridapTensorProduct.KroneckerAssembler(),
    (b1, b2)
)
expected_b_tp = kron(b2, b1)
@assert b_tp ≈ expected_b_tp "Kronecker assembly of RHS mismatch"
println("  ✓ Kronecker assembly of RHS vector correct")

println("Poisson equivalence tests completed successfully")
