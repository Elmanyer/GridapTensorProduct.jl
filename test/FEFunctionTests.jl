using Gridap
using GridapTensorProduct

println("FEFunction: testing tensor-product FEFunction construction")

# Create simple test setup
model1 = CartesianDiscreteModel((0, 1), 2)
model2 = CartesianDiscreteModel((0, 1), 3)

Ω1 = Interior(model1)
Ω2 = Interior(model2)

reffe = ReferenceFE(lagrangian, Float64, 1)
V1 = TestFESpace(Ω1, reffe)
V2 = TestFESpace(Ω2, reffe)

Vtp = TensorProductFESpace(V1, V2)

# Get DOF counts
n_free_dofs = num_free_dofs(Vtp)
n_dirichlet_dofs = 0

# Create test vectors
free_values = ones(n_free_dofs)
dirichlet_values = Float64[]

println("  ✓ Created test vectors: $n_free_dofs free DOFs")

# Create FEFunction without Dirichlet values
f1 = FEFunction(Vtp, free_values)
@assert f1 isa GridapTensorProduct.TensorProductFEFunction "FEFunction construction failed"
println("  ✓ FEFunction created (minimal constructor)")

# Create FEFunction with Dirichlet values
f2 = FEFunction(Vtp, free_values, dirichlet_values)
@assert f2 isa GridapTensorProduct.TensorProductFEFunction "FEFunction construction with Dirichlet values failed"
println("  ✓ FEFunction created (with Dirichlet values)")

# Check that values are stored correctly (use internal methods)
@assert length(f2.free_values) == n_free_dofs "Free values not stored correctly"
@assert f2.fe_space === Vtp "FE space not stored correctly"
println("  ✓ FEFunction values and space stored correctly")

# Test with different value types
test_values_int = ones(Int32, n_free_dofs)
f3 = FEFunction(Vtp, test_values_int)
@assert f3 isa GridapTensorProduct.TensorProductFEFunction "FEFunction construction with Int32 values failed"
println("  ✓ FEFunction supports different value types")

println("FEFunction tests completed successfully")
