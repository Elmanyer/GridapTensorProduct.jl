using Gridap
using GridapTensorProduct

println("TensorProductFESpace interface: testing tensor-product FE space API")

# Create small test domains and spaces
domain1 = (0, 1)
domain2 = (0, 1)
n_cells_1 = 2
n_cells_2 = 3

# Create simple Cartesian models
model1 = CartesianDiscreteModel(domain1, n_cells_1)
model2 = CartesianDiscreteModel(domain2, n_cells_2)

# Get triangulations
Ω1 = Interior(model1)
Ω2 = Interior(model2)

# Create reference element types
reffe1 = ReferenceFE(lagrangian, Float64, 1)
reffe2 = ReferenceFE(lagrangian, Float64, 1)

# Create FESpaces
V1 = TestFESpace(Ω1, reffe1)
V2 = TestFESpace(Ω2, reffe2)

# Create tensor-product FESpace
Vtp = TensorProductFESpace(V1, V2)

println("  ✓ TensorProductFESpace created successfully")

# Test number of free DOFs
n_dofs_1 = num_free_dofs(V1)
n_dofs_2 = num_free_dofs(V2)
n_dofs_tp_expected = n_dofs_1 * n_dofs_2
n_dofs_tp_actual = num_free_dofs(Vtp)
@assert n_dofs_tp_actual == n_dofs_tp_expected "Expected $n_dofs_tp_expected DOFs, got $n_dofs_tp_actual"
println("  ✓ Correct total number of DOFs: $n_dofs_tp_actual = $n_dofs_1 × $n_dofs_2")

# Test cell IDs
cell_ids = get_cell_ids(Vtp)
expected_n_cells = n_cells_1 * n_cells_2
@assert num_cells(get_triangulation(Vtp)[1]) == n_cells_1 "First triangulation cell count mismatch"
@assert num_cells(get_triangulation(Vtp)[2]) == n_cells_2 "Second triangulation cell count mismatch"
println("  ✓ Tensor cell IDs correctly structured")

# Test cell DOF IDs
cell_dof_ids = get_cell_dof_ids(Vtp)
println("  ✓ Tensor cell DOF IDs accessible")

# Test free DOF IDs
free_dof_ids = get_free_dof_ids(Vtp)
println("  ✓ Tensor free DOF IDs accessible")

# Test DOF value type (should be a tuple type)
dof_value_type = get_dof_value_type(Vtp)
println("  ✓ DOF value type: $dof_value_type")

# Test vector type
vec_type = get_vector_type(Vtp)
println("  ✓ Vector type: $vec_type")

println("TensorProductFESpace interface tests completed successfully")
