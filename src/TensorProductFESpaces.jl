"""
    TensorProductFESpaces

Wrapper types for tensor-product finite element spaces (node/edge/face/cell bases).

This module provides a container that bundles N independent 1D FE spaces into a single
tensor-product space for systems on Ω = Ω₁ ⊗ Ω₂ ⊗ ... ⊗ Ωₙ.

# Design Philosophy
- **Subdomain Independence**: Each subdomain is a primary computational unit; DOFs and
  basis functions are computed per-subdomain and combined via tensor products.
- **Transparent Access**: Methods return tuples of component spaces, allowing user code
  to work with subdomains individually when needed.
- **Tensor DOF Ordering**: Global DOFs are indexed by Cartesian tuples (i₁, i₂, ..., iₙ)
  internally; mappings to linear indices handle compatibility with Gridap's row/column indexing.

# Known Limitations (Documented for [PHASE 3+] integration)
- ⚠️ `get_triangulation()` returns tuple of triangulations, not single wrapped object
- ⚠️ `get_fe_basis()` returns list of bases ([PHASE 3]: wrap in TensorProductFEBasis)
- ⚠️ `get_cell_coordinates()`, `get_node_coordinates()` not yet wrapped
- These limitations are intentional; full Gridap interface integration planned for [PHASE 3+]

# Usage Example
```julia
V1 = TestFESpace(Ω1, reffe1)  # 1D space on [0,1]
V2 = TestFESpace(Ω2, reffe2)  # 1D space on [0,1]
V_tp = TensorProductFESpace(V1, V2)  # Tensor product space on [0,1]²

# Access component spaces independently:
spaces = get_triangulations(V_tp)  # (Triangulation1, Triangulation2)
n_dofs_global = num_free_dofs(V_tp)  # prod(num_free_dofs(Vi))
```
"""

# ===========================
# ALL IMPORTS (Consolidated at top)
# ===========================

# Gridap.FESpaces imports
import Gridap.FESpaces: FESpace
import Gridap.FESpaces: get_cell_dof_ids
import Gridap.FESpaces: get_free_dof_ids
import Gridap.FESpaces: num_free_dofs
import Gridap.FESpaces: get_fe_basis
import Gridap.FESpaces: get_fe_dof_basis
import Gridap.FESpaces: get_dof_value_type
import Gridap.FESpaces: get_vector_type
import Gridap.FESpaces: get_cell_constraints
import Gridap.FESpaces: get_cell_isconstrained

# Gridap.Geometry imports
import Gridap.Geometry: get_triangulation
import Gridap.Geometry: get_cell_type
import Gridap.Geometry: get_node_coordinates
import Gridap.Geometry: get_reffes
import Gridap.Geometry: get_cell_node_ids
import Gridap.Geometry: num_cells

# ===========================
# ALL EXPORTS (Consolidated at top, after imports)
# ===========================

export TensorProductFESpace
export TensorProductFEFunction
export get_cell_ids
export get_tensor_cell_ids
export get_tensor_cell_dof_ids
export get_tensor_free_dof_ids
export get_triangulations
# Re-export helpers
export get_cell_isconstrained
export get_dof_value_type
export get_vector_type

# Wrapper for tensor product finite element spaces constituted of subspaces
struct TensorProductFESpace{T<:Tuple} <: FESpace
    spaces::T

    function TensorProductFESpace(spaces::Tuple)
        @assert all(s -> s isa FESpace, spaces) "TensorProductFESpace expects a tuple of Gridap FESpaces"
        new{typeof(spaces)}(spaces)
    end
end

TensorProductFESpace(spaces::FESpace...) = TensorProductFESpace(tuple(spaces...))
TensorProductFESpace(spaces::AbstractVector{<:FESpace}) = TensorProductFESpace(tuple(spaces...))

Gridap.FESpaces.ConstraintStyle(::Type{<:TensorProductFESpace}) = UnConstrained()

_tensor_tuple_type(::Type{Tuple{}}) = Tuple{}

function _tensor_tuple_type(f::TensorProductFESpace, fun)
    types = map(fun, f.spaces)
    return Tuple{types...}
end

function _tensor_cartesian_rows(rows)
    iter = Iterators.product(rows...)
    return collect(Tuple, iter)
end

function _tensor_ids_table(rows)
    tuples = _tensor_cartesian_rows(rows)
    return Table(map(Vector{Int32}, tuples))
end

function _tensor_rows_from_subspaces(f::TensorProductFESpace, getter)
    map(getter, f.spaces)
end

# TensorProductFESpace interface methods
function get_tensor_cell_ids(f::TensorProductFESpace)
    # 1. Get number of cells in each direction
    ncells_axes = [num_cells(get_triangulation(s)) for s in f.spaces]
    # 2. Total number of tensor-product cells
    n_tp_cells = prod(ncells_axes)
    dim = length(f.spaces)
    # 3. Pointers: Each TP cell is a "row" containing 'dim' indices
    # Length must be n_tp_cells + 1
    ptrs = collect(Int32, 1:dim:(n_tp_cells * dim + 1))
    # 4. Data: Flattened list of indices
    data = Vector{Int32}(undef, n_tp_cells * dim)
    # 5. Fill data using CartesianIndices to handle the "odometer" counting
    iter = CartesianIndices(Tuple(ncells_axes))
    
    for (i, cell_idx) in enumerate(iter)
        # cell_idx is something like CartesianIndex(1, 2)
        # We store it as [1, 2] in the flat data array
        start_idx = (i - 1) * dim + 1
        for d in 1:dim
            data[start_idx + d - 1] = Int32(cell_idx[d])
        end
    end

    return Table(data, ptrs)
end

get_cell_ids(f::TensorProductFESpace) = get_tensor_cell_ids(f)
get_cell_ids(f::TensorProductFESpace, ttrian) = get_tensor_cell_ids(f)

function get_tensor_cell_dof_ids(f::TensorProductFESpace)
    N = length(f.spaces)
    nf = ntuple(k -> num_free_dofs(f.spaces[k]), N)
    sub_cell_dofs = ntuple(k -> Gridap.FESpaces.get_cell_dof_ids(f.spaces[k]), N)
    ncells_per_dim = ntuple(k -> length(sub_cell_dofs[k]), N)

    rows = Vector{Vector{Int32}}()
    sizehint!(rows, prod(ncells_per_dim))

    for cell_idx in CartesianIndices(ncells_per_dim)
        local_dofs = ntuple(k -> sub_cell_dofs[k][cell_idx[k]], N)
        local_sizes = ntuple(k -> length(local_dofs[k]), N)
        cell_row = Int32[]

        for dof_idx in CartesianIndices(local_sizes)
            dofs = ntuple(k -> Int32(local_dofs[k][dof_idx[k]]), N)
            is_free = all(d -> d > 0, dofs)
            if is_free
                lin = Int32(dofs[1])
                stride = nf[1]
                for k in 2:N
                    lin   += Int32((dofs[k] - 1) * stride)
                    stride *= nf[k]
                end
                push!(cell_row, lin)
            else
                abs_dofs = ntuple(k -> Int32(abs(dofs[k])), N)
                n_range  = ntuple(k -> (dofs[k] < 0 ?
                    Gridap.FESpaces.num_dirichlet_dofs(f.spaces[k]) : nf[k]), N)
                lin = Int32(abs_dofs[1])
                stride = n_range[1]
                for k in 2:N
                    lin   += Int32((abs_dofs[k] - 1) * stride)
                    stride *= n_range[k]
                end
                push!(cell_row, -lin)
            end
        end
        push!(rows, cell_row)
    end
    return Table(rows)
end

get_cell_dof_ids(f::TensorProductFESpace) = get_tensor_cell_dof_ids(f)
get_cell_dof_ids(f::TensorProductFESpace, ttrian) = get_tensor_cell_dof_ids(f)

# [PHASE 3+] TODO: These methods currently return lists, but Gridap API expects single objects.
# They are included for introspection but should NOT be used directly in Gridap routines.
# Future implementation should wrap results appropriately or implement proper tensor-product semantics.
function get_tensor_free_dof_ids(f::TensorProductFESpace)
    free_ids = [Gridap.FESpaces.get_free_dof_ids(s) for s in f.spaces]
    tuples = _tensor_cartesian_rows(free_ids)
    # Convert each tuple to an Int32 vector
    return Table([Vector{Int32}(collect(t)) for t in tuples])
end

num_free_dofs(f::TensorProductFESpace) = prod(num_free_dofs(s) for s in f.spaces)
get_free_dof_ids(f::TensorProductFESpace) = Base.OneTo(num_free_dofs(f))

# ⚠️  WARNING: The following methods return lists, not single objects.
# This deviates from standard Gridap API and is a known limitation.
# These methods are suitable for introspection and debugging only.
# Full Gridap integration requires wrapping or alternative approaches.

function get_triangulation(f::TensorProductFESpace)
    trians = ntuple(i -> Gridap.Geometry.get_triangulation(f.spaces[i]), length(f.spaces))
    return TensorProductTriangulation(trians)
end
get_fe_basis(f::TensorProductFESpace) = [Gridap.FESpaces.get_fe_basis(s) for s in f.spaces]
get_fe_dof_basis(f::TensorProductFESpace) = [Gridap.FESpaces.get_fe_dof_basis(s) for s in f.spaces]
get_dof_value_type(f::TensorProductFESpace) = _tensor_tuple_type(f, get_dof_value_type)
get_vector_type(f::TensorProductFESpace) = Vector{get_dof_value_type(f)}
get_reffes(f::TensorProductFESpace) = [Gridap.Geometry.get_reffes(s) for s in f.spaces]
get_cell_type(f::TensorProductFESpace) = [Gridap.Geometry.get_cell_type(s) for s in f.spaces]



"""
    struct TensorProductFEFunction{T<:CellField} <: FEFunction
"""
struct TensorProductFEFunction{CF,T1<:AbstractVector,T2<:AbstractVector} <: FEFunction
    cell_field::CF
    cell_dof_values::Any
    free_values::T1
    dirichlet_values::T2
    fe_space::TensorProductFESpace
end


function FEFunction(fs::TensorProductFESpace, free_values::AbstractVector, dirichlet_values::AbstractVector)
    # Minimal constructor: store provided vectors and the parent FE space.
    # Detailed cell-field construction (scattering values to per-cell local arrays)
    # will be implemented once the DOF global-indexing scheme is finalized.
    return TensorProductFEFunction{Nothing,typeof(free_values),typeof(dirichlet_values)}(nothing, nothing, free_values, dirichlet_values, fs)
end

function FEFunction(fs::TensorProductFESpace, free_values::AbstractVector)
    # Mirror the SingleFieldFESpace API: if no Dirichlet vector is supplied yet,
    # create an empty placeholder and keep the free values untouched.
    dirichlet_values = similar(free_values, 0)
    return FEFunction(fs, free_values, dirichlet_values)
end


function CellField(fs::TensorProductFESpace, cell_vals)
    throw(ErrorException("CellField construction for TensorProductFESpace is not yet implemented. Use FEFunction(...) to create a placeholder FEFunction and evaluate via provided utilities."))
end

get_data(f::TensorProductFEFunction) = f.cell_dof_values
get_triangulation(f::TensorProductFEFunction) = get_triangulation(f.fe_space)
DomainStyle(::Type{TensorProductFEFunction{CF,T1,T2}}) where {CF,T1,T2} = DomainStyle(CellField)

function get_triangulations(f::TensorProductFESpace)
    return ntuple(i -> Gridap.Geometry.get_triangulation(f.spaces[i]), length(f.spaces))
end

get_free_dof_values(f::TensorProductFEFunction) = f.free_values
get_cell_dof_values(f::TensorProductFEFunction) = f.cell_dof_values
get_fe_space(f::TensorProductFEFunction) = f.fe_space


function get_cell_constraints(f::TensorProductFESpace)
    # [PHASE 3+] TODO: Implement tensor-product constraint handling
    # Currently returns list of constraints from each subdomain
    # Future: may need to combine constraints in tensor-product semantics
    return [Gridap.FESpaces.get_cell_constraints(s) for s in f.spaces]
end

function get_cell_isconstrained(f::TensorProductFESpace)
    # [PHASE 3+] TODO: Implement tensor-product constraint detection
    # Currently returns list of constraint flags from each subdomain
    # Future: may need to combine flags in tensor-product semantics
    return [Gridap.FESpaces.get_cell_isconstrained(s) for s in f.spaces]
end
