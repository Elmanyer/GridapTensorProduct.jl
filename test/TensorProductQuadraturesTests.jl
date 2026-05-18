using Gridap
using GridapTensorProduct

println("TensorProductQuadratures: testing tensor-product quadrature construction")

# Create simple 1D quadratures
q1_points = [0.5]
q1_weights = [2.0]  # Full interval weight
q1 = (points=q1_points, weights=q1_weights)

q2_points = [0.25, 0.75]
q2_weights = [1.0, 1.0]
q2 = (points=q2_points, weights=q2_weights)

# Create tensor-product quadrature
q_tp = TensorQuadrature(q1, q2)

# Check number of quadrature points
expected_n_points = length(q1_points) * length(q2_points)
actual_n_points = length(q_tp.weights)
@assert actual_n_points == expected_n_points "Expected $expected_n_points quadrature points, got $actual_n_points"
println("  ✓ Correct number of quadrature points: $actual_n_points")

# Check weight sum (should equal product of weight sums)
expected_weight_sum = sum(q1_weights) * sum(q2_weights)
actual_weight_sum = sum(q_tp.weights)
@assert isapprox(actual_weight_sum, expected_weight_sum) "Weight sum mismatch: expected $expected_weight_sum, got $actual_weight_sum"
println("  ✓ Correct total weight: $actual_weight_sum ≈ $expected_weight_sum")

# Check that points are correctly formed
for (i, pt) in enumerate(q_tp.points)
    @assert length(pt) == 2 "Each point should be 2D, got $(length(pt))D"
    @assert eltype(pt) == Float64 "Points should be Float64, got $(eltype(pt))"
end
println("  ✓ All points have correct dimensionality and type")

# Check individual weights product property
q_tp_manual = TensorQuadrature(q1, q2)
expected_weights = [q1_weights[i] * q2_weights[j] for i in 1:length(q1_weights) for j in 1:length(q2_weights)]
@assert isapprox(sort(q_tp_manual.weights), sort(expected_weights)) "Weights don't match expected product"
println("  ✓ Weights correctly computed as products of subdomain weights")

println("TensorProductQuadratures tests completed successfully")
