"""
    TensorProductGeometry

Wrapper types for tensor-product geometric objects (triangulations, cell fields).

This module provides containers that allow Gridap to work with tensor-product domains
while maintaining the abstraction that a tensor domain is still a valid domain for FEM.

# Design Principles
- **Simplicity first**: Keep the library simple and minimal for now
- **Subdomain independence**: Treat subdomains independently as much as possible
- **Lazy approach**: Only define TensorProductTriangulation when needed for weak form
  integrands in the form of TensorProductCellField
- **Future extension**: Full integration with Gridap's triangulation interface planned
"""

import Gridap.CellData: CellField
import Gridap.Geometry: num_cells

export TensorProductTriangulation, TensorProductCellField

# ===========================
# TensorProductTriangulation
# ===========================

"""
    struct TensorProductTriangulation{T<:Tuple}

Represents a triangulation of a tensor-product domain.

The tensor product domain Ω = Ω₁ ⊗ Ω₂ ⊗ ... ⊗ Ωₙ has cells indexed by tuples
(c₁, c₂, ..., cₙ) where cᵢ ∈ [1, n_cells(Ωᵢ)].

# Fields
- `triangulations::T` : Tuple of component triangulations (Ω₁, Ω₂, ..., Ωₙ)
- `cell_indices::Any` : Cached Cartesian product of cell indices (computed once)

# Design Notes
- Acts as a **simple wrapper** over component triangulations (treat subdomains independently)
- Delegates most methods to components; tensor-aware methods are specialized
- Does NOT inherit from Gridap.Triangulation (will be addressed in [PHASE 4])
- **Currently only used when defining weak form integrands** as TensorProductCellField
- Instantiation should be minimal; subdomains are the primary computational units

# Lazy Instantiation Pattern

This type is intentionally lightweight and created only when weak form assembly requires
coordinating indexing across subdomains. Most work stays at the subdomain level:

```julia
# Stage 1-2: Work with subdomains independently (don't create tensor triangulation yet)
Ω1 = Interior(model1)  # 1D subdomain
Ω2 = Interior(model2)  # 1D subdomain
V1 = TestFESpace(Ω1, reffe)  # Test space on Ω₁
V2 = TestFESpace(Ω2, reffe)  # Test space on Ω₂

# Stage 3: Only instantiate tensor triangulation when defining weak form integrands
V_tp = TensorProductFESpace(V1, V2)
trian_tp = TensorProductTriangulation(get_triangulation(V1), get_triangulation(V2))

# Stage 4-5: Use for coordinating tensor product assembly
dΩ_tp = Measure(trian_tp, 2)
a(u, v) = ∫(∇(u) ⋅ ∇(v)) * dΩ_tp  # Weak form on tensor domain
```

# Future Work
- Integrate with Gridap's triangulation interface (get_cell_coordinates, get_node_coordinates, etc.)
- May require custom implementations for tensor-specific geometric queries
"""
struct TensorProductTriangulation{T<:Tuple}
    triangulations::T  # Component triangulations
    cell_indices_cache::Any  # Cache for CartesianIndices over cells

    function TensorProductTriangulation(triangulations::Tuple)
        @assert !isempty(triangulations) "Need at least one triangulation"
        @assert all(t -> t isa Gridap.Geometry.Triangulation, triangulations) "All components must be Triangulation objects"

        # Pre-compute cell indices for efficient iteration
        n_cells_per_dim = tuple((Gridap.Geometry.num_cells(t) for t in triangulations)...)
        cell_indices = CartesianIndices(n_cells_per_dim)

        new{typeof(triangulations)}(triangulations, cell_indices)
    end
end

"""
    TensorProductTriangulation(t1, t2, ..., tN)

Construct from multiple triangulations (variadic form).
"""
TensorProductTriangulation(ts::Gridap.Geometry.Triangulation...) = TensorProductTriangulation(tuple(ts...))

# Basic accessors
get_triangulations(tptp::TensorProductTriangulation) = tptp.triangulations
get_cell_indices(tptp::TensorProductTriangulation) = tptp.cell_indices_cache

function Base.length(tptp::TensorProductTriangulation)
    prod(Gridap.Geometry.num_cells(t) for t in tptp.triangulations)
end

Gridap.Geometry.num_cells(t::TensorProductTriangulation) = length(t)
Base.getindex(t::TensorProductTriangulation, i::Int) = t.triangulations[i]

# ===========================
# TensorProductCellField
# ===========================

"""
    struct TensorProductCellField{CF<:Tuple, DATA}

Represents a cell field evaluated on a tensor-product domain.

For a tensor domain Ω = Ω₁ ⊗ Ω₂, a cell field u ∈ H¹(Ω) is represented as:
- Component scalar fields: u₁ ∈ H¹(Ω₁), u₂ ∈ H¹(Ω₂)
- Tensor value at cell (c₁, c₂): u(c₁, c₂) = u₁(c₁) * u₂(c₂)

For vector fields or component-wise products, indexing is specialized.

# Fields
- `cell_fields::CF` : Tuple of component cell fields (cf₁, cf₂, ..., cfₙ)
- `data::DATA` : Cached evaluation data (Gridap-compatible format)

# Design Notes
- Wraps component cell fields without materializing the full tensor product field
- Lazy evaluation: actual tensor values computed on-demand via indexing
- Interops with Gridap's CellData evaluation machinery
"""
struct TensorProductCellField{CF<:Tuple, DATA}
    cell_fields::CF  # Component cell fields
    data::DATA  # Cached data for Gridap integration

    function TensorProductCellField(cell_fields::Tuple, data=nothing)
        @assert !isempty(cell_fields) "Need at least one cell field"
        @assert all(cf -> cf isa CellField, cell_fields) "All components must be CellField objects"
        new{typeof(cell_fields), typeof(data)}(cell_fields, data)
    end
end

"""
    TensorProductCellField(cf1, cf2, ..., cfN)

Construct from multiple cell fields (variadic form).
"""
TensorProductCellField(cfs::CellField...) = TensorProductCellField(tuple(cfs...))

# Basic accessors
get_cell_fields(tpfc::TensorProductCellField) = tpfc.cell_fields

"""
    evaluate(cf::TensorProductCellField, indices::Tuple)

Evaluate a tensor-product cell field at a specific tensor cell index.

For cell indices (i₁, i₂, ..., iₙ), retrieves component evaluations:
- u₁_value = evaluate(cf.cell_fields[1], i₁)
- u₂_value = evaluate(cf.cell_fields[2], i₂)
- ...
- Result = product or component-wise combination
"""
function Base.getindex(tpfc::TensorProductCellField, cell_idx::Tuple)
    # Retrieve component values and combine via tensor product rule
    # This is a placeholder; full implementation involves Gridap's CellData machinery
    error("TensorProductCellField indexing not yet fully implemented")
end

# ===========================
# Integration helpers
# ===========================

"""
    get_triangulation(tpfc::TensorProductCellField)

Return the triangulation on which the cell field is defined.
"""
function Gridap.Geometry.get_triangulation(tpfc::TensorProductCellField)
    trians = ntuple(i -> Gridap.Geometry.get_triangulation(tpfc.cell_fields[i]),
                    length(tpfc.cell_fields))
    return TensorProductTriangulation(trians)
end

# ===========================
# Future extensions
# ===========================

# These will be needed for full integration:
# - define Base.iterate for looping over tensor cells
# - implement Gridap integration methods (evaluate, apply_map, etc.)
# - support different value types (scalar, vector, tensor)
