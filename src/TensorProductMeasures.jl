"""
    TensorProductMeasure{N, T<:NTuple{N,Measure}}

A measure representing the tensor product of multiple 1D or lower-dimensional measures.

Simple wrapper that combines N subdomain Measure objects into a unified structure for
tensor-product integration operations.

## Type Parameters
- `N::Int` - Number of subdomains (tensor product dimensionality)
- `T<:NTuple{N,Measure}` - Tuple type of measures

## Fields
- `measures::T` - Tuple of N Gridap Measure objects (dΩ₁, dΩ₂, ..., dΩ_N)

All other data (triangulations, cell quadratures, degrees) are accessed on-demand
from the individual measures via interface functions.

## Constructor
```julia
TensorProductMeasure(measures::Tuple)      # Tuple form
TensorProductMeasure(m₁, m₂, ..., m_N)     # Variadic form
dΩ1 ⊗ dΩ2 ⊗ dΩ3                           # Operator form
```

## Examples
```julia
dΩ1 = Measure(Ω1, 2)  # 1D measure
dΩ2 = Measure(Ω2, 2)  # 1D measure
dΩtp = TensorProductMeasure((dΩ1, dΩ2))   # Tensor product
# Or more conveniently:
dΩtp = dΩ1 ⊗ dΩ2  # Using the ⊗ operator
```
"""

import Base: ⊗
import Gridap.CellData: Measure
import Gridap.Geometry: get_triangulation
import Gridap.CellData: get_cell_quadrature

export TensorProductMeasure

struct TensorProductMeasure{N, T<:NTuple{N,Measure}} <: Measure
    measures::T  # (Measure₁, Measure₂, ..., Measureₙ)

    function TensorProductMeasure(measures::NTuple{N,Measure}) where N
        @assert N > 0 "Must have at least 1 subdomain measure"
        @assert all(m isa Measure for m in measures) "All elements must be Gridap Measure"
        new{N, typeof(measures)}(measures)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCTORS
# ═══════════════════════════════════════════════════════════════════════════

"""
    TensorProductMeasure(m₁::Measure, m₂::Measure, ..., m_N::Measure)

Construct a tensor-product measure from N individual Gridap Measure objects (variadic form).
"""
TensorProductMeasure(ms::Measure...) = TensorProductMeasure(tuple(ms...))

# ═══════════════════════════════════════════════════════════════════════════
# TENSOR PRODUCT OPERATOR (⊗)
# ═══════════════════════════════════════════════════════════════════════════

"""
    ⊗(m1::Measure, m2::Measure) -> TensorProductMeasure

Tensor product operator for measures. Enables convenient syntax:
```julia
dΩ1 ⊗ dΩ2 ⊗ dΩ3
```

Automatically flattens nested tensor products.
"""
function ⊗(m1::Measure, m2::Measure)::TensorProductMeasure
    case1 = m1 isa TensorProductMeasure
    case2 = m2 isa TensorProductMeasure

    if case1 && case2
        return TensorProductMeasure((m1.measures..., m2.measures...))
    elseif case1
        return TensorProductMeasure((m1.measures..., m2))
    elseif case2
        return TensorProductMeasure((m1, m2.measures...))
    else
        return TensorProductMeasure((m1, m2))
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

export ⊗, TensorProductMeasure
export get_measures, get_measure
export get_triangulations, get_triangulation
export get_cell_quadratures
export num_subdomains, get_num_subdomains
export get_quadrature_degree
export get_quadrature_points, get_quadrature_weights
export num_quadrature_points
export get_tensor_product_points, get_tensor_product_weights

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE FUNCTIONS: ACCESSORS
# ═══════════════════════════════════════════════════════════════════════════

"""
    get_measures(m::TensorProductMeasure) -> NTuple{N,Measure}

Return the tuple of all N component measures.
"""
get_measures(m::TensorProductMeasure) = m.measures

"""
    get_measure(m::TensorProductMeasure, dim::Int) -> Measure

Return the Measure object for subdomain dim.
"""
function get_measure(m::TensorProductMeasure, dim::Int)
    @assert 1 ≤ dim ≤ num_subdomains(m) "Dimension $dim out of range [1, $(num_subdomains(m))]"
    return m.measures[dim]
end

"""
    get_triangulations(m::TensorProductMeasure) -> NTuple{N,Triangulation}

Return the tuple of all N component triangulations (extracted from measures).
"""
function get_triangulations(m::TensorProductMeasure)
    N = num_subdomains(m)
    return ntuple(i -> get_triangulation(m.measures[i]), N)
end

"""
    get_triangulation(m::TensorProductMeasure, dim::Int) -> Triangulation

Return the Triangulation object for subdomain dim.
"""
function get_triangulation(m::TensorProductMeasure, dim::Int)
    @assert 1 ≤ dim ≤ num_subdomains(m) "Dimension $dim out of range [1, $(num_subdomains(m))]"
    return get_triangulation(m.measures[dim])
end

"""
    get_cell_quadratures(m::TensorProductMeasure) -> NTuple{N,Any}

Return the tuple of all N component cell quadratures (extracted from measures).
"""
function get_cell_quadratures(m::TensorProductMeasure)
    N = num_subdomains(m)
    return ntuple(i -> get_cell_quadrature(m.measures[i]), N)
end

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE FUNCTIONS: DIMENSIONALITY
# ═══════════════════════════════════════════════════════════════════════════

"""
    num_subdomains(m::TensorProductMeasure{N}) -> Int

Return the number of subdomains (tensor product dimensionality).
"""
num_subdomains(m::TensorProductMeasure{N}) where N = N

"""
    get_num_subdomains(m::TensorProductMeasure{N}) -> Int

Alias for num_subdomains().
"""
get_num_subdomains(m::TensorProductMeasure{N}) where N = N

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE FUNCTIONS: DEGREES
# ═══════════════════════════════════════════════════════════════════════════

"""
    get_quadrature_degree(m::TensorProductMeasure) -> Int

Return the combined quadrature degree (product of component degrees).
"""
function get_quadrature_degree(m::TensorProductMeasure)
    N = num_subdomains(m)
    degree = 1
    for i in 1:N
        degree *= get_quadrature_degree(m, i)
    end
    return degree
end

"""
    get_quadrature_degree(m::TensorProductMeasure, dim::Int) -> Int

Return the quadrature degree for subdomain dim.
"""
function get_quadrature_degree(m::TensorProductMeasure, dim::Int)
    @assert 1 ≤ dim ≤ num_subdomains(m) "Dimension $dim out of range [1, $(num_subdomains(m))]"

    cq = get_cell_quadrature(m.measures[dim])
    if hasfield(typeof(cq), :degree)
        return cq.degree
    else
        return 1
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE FUNCTIONS: QUADRATURE POINTS AND WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════

"""
    get_quadrature_points(m::TensorProductMeasure, dim::Int) -> Vector

Return the quadrature points for subdomain dim.
"""
function get_quadrature_points(m::TensorProductMeasure, dim::Int)
    @assert 1 ≤ dim ≤ num_subdomains(m) "Dimension $dim out of range [1, $(num_subdomains(m))]"

    cq = get_cell_quadrature(m.measures[dim])

    if hasfield(typeof(cq), :coords)
        return cq.coords
    elseif hasfield(typeof(cq), :points)
        return cq.points
    else
        error("Cannot extract points from cell quadrature of type $(typeof(cq))")
    end
end

"""
    get_quadrature_weights(m::TensorProductMeasure, dim::Int) -> Vector

Return the quadrature weights for subdomain dim.
"""
function get_quadrature_weights(m::TensorProductMeasure, dim::Int)
    @assert 1 ≤ dim ≤ num_subdomains(m) "Dimension $dim out of range [1, $(num_subdomains(m))]"

    cq = get_cell_quadrature(m.measures[dim])

    if hasfield(typeof(cq), :weights)
        return cq.weights
    else
        error("Cannot extract weights from cell quadrature of type $(typeof(cq))")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE FUNCTIONS: POINT AND WEIGHT COUNTS
# ═══════════════════════════════════════════════════════════════════════════

"""
    num_quadrature_points(m::TensorProductMeasure) -> Int

Return the total number of tensor product quadrature points.
"""
function num_quadrature_points(m::TensorProductMeasure)
    n_total = 1
    for dim in 1:num_subdomains(m)
        n_total *= num_quadrature_points(m, dim)
    end
    return n_total
end

"""
    num_quadrature_points(m::TensorProductMeasure, dim::Int) -> Int

Return the number of quadrature points for subdomain dim.
"""
function num_quadrature_points(m::TensorProductMeasure, dim::Int)
    @assert 1 ≤ dim ≤ num_subdomains(m) "Dimension $dim out of range [1, $(num_subdomains(m))]"

    pts = get_quadrature_points(m, dim)
    return length(pts)
end

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE FUNCTIONS: TENSOR PRODUCT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

"""
    get_tensor_product_points(m::TensorProductMeasure) -> Vector{Vector}

Return all tensor product quadrature points as Cartesian product.
"""
function get_tensor_product_points(m::TensorProductMeasure)
    N = num_subdomains(m)
    all_points = [get_quadrature_points(m, i) for i in 1:N]

    index_ranges = [1:length(all_points[i]) for i in 1:N]
    index_combos = Iterators.product(index_ranges...)

    tensor_pts = Vector{Float64}[]
    for indices in index_combos
        pt = Float64[all_points[i][indices[i]] for i in 1:N]
        push!(tensor_pts, pt)
    end

    return tensor_pts
end

"""
    get_tensor_product_weights(m::TensorProductMeasure) -> Vector

Return all tensor product quadrature weights.
"""
function get_tensor_product_weights(m::TensorProductMeasure)
    N = num_subdomains(m)
    all_weights = [get_quadrature_weights(m, i) for i in 1:N]

    index_ranges = [1:length(all_weights[i]) for i in 1:N]
    index_combos = Iterators.product(index_ranges...)

    tensor_ws = Float64[]
    for indices in index_combos
        w = prod(all_weights[i][indices[i]] for i in 1:N)
        push!(tensor_ws, w)
    end

    return tensor_ws
end

# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

"""
    Base.show(io::IO, m::TensorProductMeasure)

Display summary of tensor product measure.
"""
function Base.show(io::IO, m::TensorProductMeasure{N}) where N
    print(io, "TensorProductMeasure{$N}:\n")
    print(io, "  Subdomains (N): $N\n")
    print(io, "  Combined degree: $(get_quadrature_degree(m))\n")
    print(io, "  Total points: $(num_quadrature_points(m))\n")

    for i in 1:N
        deg_i = get_quadrature_degree(m, i)
        pts_i = num_quadrature_points(m, i)
        print(io, "    [$i]: degree=$deg_i, points=$pts_i\n")
    end
end
