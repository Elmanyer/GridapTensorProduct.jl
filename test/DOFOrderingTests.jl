using Gridap
using GridapTensorProduct

println("DOF ordering: testing tensor index ↔ global DOF bijection")

# Test tensor_tuple_to_linear_index and linear_index_to_tensor_tuple bijection
# for various problem sizes

test_cases = [
    (sizes=(2,), indices=[(1,), (2,)]),
    (sizes=(2, 3), indices=[(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]),
    (sizes=(2, 3, 4), indices=[(1, 1, 1), (1, 1, 2), (2, 2, 3), (2, 3, 4)]),
]

for (sizes, indices) in test_cases
    total_dofs = prod(sizes)
    println("  Testing sizes=$sizes (total DOFs: $total_dofs)")

    for idx in indices
        # Forward mapping
        linear = GridapTensorProduct.tensor_tuple_to_linear_index(idx, sizes)
        @assert 1 <= linear <= total_dofs "Linear index $linear out of range [1, $total_dofs]"

        # Backward mapping
        recovered_idx = GridapTensorProduct.linear_index_to_tensor_tuple(linear, sizes)
        @assert recovered_idx == idx "Index mismatch: sent $idx, got $recovered_idx"
    end
    println("    ✓ Bijection verified for all indices in this configuration")
end

# Test with homogeneous spaces (all subdomains identical)
println("  Testing homogeneous spaces (3×5×2 tensor product)...")
sizes_homo = (3, 5, 2)
n_dofs_homo = prod(sizes_homo)
for linear in 1:min(n_dofs_homo, 20)  # Test first 20 for efficiency
    idx = GridapTensorProduct.linear_index_to_tensor_tuple(linear, sizes_homo)
    linear_recovered = GridapTensorProduct.tensor_tuple_to_linear_index(idx, sizes_homo)
    @assert linear_recovered == linear "Bijection failed for linear=$linear"
end
println("    ✓ Bijection verified for homogeneous system")

# Test with heterogeneous spaces (different number of DOFs per subdomain)
println("  Testing heterogeneous spaces (with different DOF counts per subdomain)...")
model1 = CartesianDiscreteModel((0, 1), 2)
model2 = CartesianDiscreteModel((0, 1), 3)
model3 = CartesianDiscreteModel((0, 1), 4)

Ω1 = Interior(model1)
Ω2 = Interior(model2)
Ω3 = Interior(model3)

reffe = ReferenceFE(lagrangian, Float64, 1)
V1 = TestFESpace(Ω1, reffe)
V2 = TestFESpace(Ω2, reffe)
V3 = TestFESpace(Ω3, reffe)

Vtp = TensorProductFESpace(V1, V2, V3)
n_dofs_total = num_free_dofs(Vtp)
n_dofs_1 = num_free_dofs(V1)
n_dofs_2 = num_free_dofs(V2)
n_dofs_3 = num_free_dofs(V3)

expected_total = n_dofs_1 * n_dofs_2 * n_dofs_3
@assert n_dofs_total == expected_total "Heterogeneous DOF count mismatch"
println("    ✓ Heterogeneous DOF count correct: $n_dofs_total = $n_dofs_1 × $n_dofs_2 × $n_dofs_3")

println("DOF ordering tests completed successfully")
