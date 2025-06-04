import dolfinx as df
import numpy as np

def split_mesh_comm(mesh):
    """
    Given a mesh, return a new mesh with identical geometry, topology and distribution 
    but defined on a split MPI communicator, defined based on whether the mesh has cells 
    on a given MPI rank or not.

    Arguments:
      * mesh      - original mesh

    Returns:
      * new_mesh  - mesh with new comm
    """
    tdim = mesh.topology.dim

    # create a split comm
    new_comm = mesh.comm.Split(mesh.topology.index_map(tdim).size_local>0)

    # map from the rank on the original comm to the new split comm
    # used to map index map owners to their new ranks
    # this will include undefined negative numbers on ranks with no corresponding entry on the split comm that should cause failure if the mesh has ghosts on unexpected ranks
    rank_map = np.asarray(mesh.comm.group.Translate_ranks(None, new_comm.group), dtype=np.int32)

    # create a new topology on the new comm
    new_topo = df.cpp.mesh.Topology(new_comm, mesh.topology.cell_type)

    new_topo_im_0 = df.common.IndexMap(new_comm, 
                                    mesh.topology.index_map(0).size_local, 
                                    mesh.topology.index_map(0).ghosts, 
                                    rank_map[mesh.topology.index_map(0).owners])
    new_topo.set_index_map(0, new_topo_im_0)

    new_topo_im_tdim = df.common.IndexMap(new_comm, 
                                    mesh.topology.index_map(tdim).size_local, 
                                    mesh.topology.index_map(tdim).ghosts, 
                                    rank_map[mesh.topology.index_map(tdim).owners])
    new_topo.set_index_map(tdim, new_topo_im_tdim)

    new_topo.set_connectivity(mesh.topology.connectivity(0,0), 0, 0)
    new_topo.set_connectivity(mesh.topology.connectivity(tdim, 0), tdim, 0)

    gdim = mesh.geometry.dim
    new_geom_im = df.common.IndexMap(new_comm,
                                    mesh.geometry.index_map().size_local,
                                    mesh.geometry.index_map().ghosts,
                                    rank_map[mesh.geometry.index_map().owners])
    new_geom = type(mesh.geometry._cpp_object)(new_geom_im, mesh.geometry.dofmap, 
                                    mesh.geometry.cmap._cpp_object, 
                                    mesh.geometry.x[:,:gdim], 
                                    mesh.geometry.input_global_indices)
    
    # set up a new mesh
    new_mesh = df.mesh.Mesh(type(mesh._cpp_object)(
                     new_comm, new_topo, new_geom), mesh.ufl_domain())
    return new_mesh

def create_submesh(mesh, cell_indices, 
                   cell_tags=None, facet_tags=None, split_comm=True):
    """
    Function to return a submesh based on the cell indices provided.

    Arguments:
      * mesh         - original (parent) mesh
      * cell_indices - cell indices of the parent mesh to include in the submesh

    Keyword Arguments:
      * cell_tags    - cell tags on parent mesh that will be mapped and returned relative to submesh (default=None)
      * facet_tags   - facet tags on parent mesh that will be mapped and returned relative to submesh (default=None)
      * split_comm   - whether to create the submesh on a split comm, excluding ranks with no local cells (default=True)

    Returns:
      * submesh            - submesh of mesh given cell_indices
      * submesh_cell_tags  - cell tags relative to submesh if cell_tags provided (otherwise None)
      * submesh_facet_tags - facet tags relative to submesh if facet_tags provided (otherwise None)
      * submesh_cell_map   - map from submesh cells to parent cells
    """
    tdim = mesh.topology.dim
    fdim = tdim-1
    submesh, submesh_cell_map, submesh_vertex_map, submesh_geom_map = \
                  df.mesh.create_submesh(mesh, tdim, cell_indices)

    
    if split_comm: submesh = split_mesh_comm(submesh)

    submesh.topology.create_connectivity(fdim, tdim)

    # if cell_tags are provided then map to the submesh
    submesh_cell_tags = None
    if cell_tags is not None:
        submesh_cell_tags_indices = []
        submesh_cell_tags_values  = []
        # loop over the submesh cells, checking if they're included in
        # the parent cell_tags
        for i,parentind in enumerate(submesh_cell_map):
            parent_cell_tags_indices = np.argwhere(cell_tags.indices==parentind)
            if parent_cell_tags_indices.shape[0]>0:
                submesh_cell_tags_indices.append(i)
                submesh_cell_tags_values.append(cell_tags.values[parent_cell_tags_indices[0][0]])
        submesh_cell_tags_indices = np.asarray(submesh_cell_tags_indices)
        submesh_cell_tags_values  = np.asarray(submesh_cell_tags_values)

        # create a new meshtags object
        # indices should already be sorted by construction
        submesh_cell_tags = df.mesh.meshtags(submesh, tdim, 
                                             submesh_cell_tags_indices, 
                                             submesh_cell_tags_values)
            

    # if facet_tags are provided then map to the submesh
    submesh_facet_tags = None
    if facet_tags is not None:
        # parent facet to vertices adjacency list
        f2vs = mesh.topology.connectivity(fdim, 0)

        # submesh facet to vertices adjaceny list
        submesh.topology.create_connectivity(fdim, 0)
        submesh_f2vs = submesh.topology.connectivity(fdim, 0)
        # create a map from the parent vertices to the submesh facets
        # (only for the facets that exist in the submesh)
        submesh_parentvs2subf = dict()
        for i in range(submesh_f2vs.num_nodes):
            submesh_parentvs2subf[tuple(sorted([submesh_vertex_map[j] for j in submesh_f2vs.links(i)]))] = i

        # loop over the facet_tags and map from the parent facet to the submesh facet
        # via the vertices, copying over the facet_tag values
        submesh_facet_tags_indices = []
        submesh_facet_tags_values  = []
        for i,parentind in enumerate(facet_tags.indices):
            subind = submesh_parentvs2subf.get(tuple(sorted(f2vs.links(parentind))), None)
            if subind is not None:
                submesh_facet_tags_indices.append(subind)
                submesh_facet_tags_values.append(facet_tags.values[i])
        submesh_facet_tags_indices = np.asarray(submesh_facet_tags_indices)
        submesh_facet_tags_values  = np.asarray(submesh_facet_tags_values)

        perm = np.argsort(submesh_facet_tags_indices)
        submesh_facet_tags = df.mesh.meshtags(submesh, fdim, 
                                              submesh_facet_tags_indices[perm], 
                                              submesh_facet_tags_values[perm])
    
    return submesh, submesh_cell_tags, submesh_facet_tags, submesh_cell_map

def get_cell_collisions(x, mesh):
    """
    Given a list of points and a mesh, return the first cells that each point lies in in the mesh.

    Arguments:
      * x    - coordinates of points
      * mesh - mesh

    Returns:
      * first_cells - a list of cells corresponding to each point in x
    """
    tree = df.geometry.bb_tree(mesh, mesh.geometry.dim)
    xl = x
    if len(x.shape)==1: xl = [x]
    cinds = []
    cells = []
    for i, x0 in enumerate(xl):
        cell_candidates = df.geometry.compute_collisions_points(tree, x0)
        cell = df.geometry.compute_colliding_cells(mesh, cell_candidates, x0).array
        if len(cell) > 0:
            cinds.append(i)
            cells.append(cell[0])
    return cinds, cells
