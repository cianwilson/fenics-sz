from mpi4py import MPI
import dolfinx as df
import pyvista as pv
import numpy as np
import functools
import vtk

try:
    pv.start_xvfb()
except OSError:
    pass

Myr_to_s = lambda a: a*1.e6*365.25*24*60*60
s_to_Myr = lambda a: a/1.e6/365.25/24/60/60
nondim_to_K = lambda T: T + 273.15
mmpyr_to_mps = lambda v: v*1.0e-3/365.25/24/60/60
mps_to_mmpyr = lambda v: v*1.0e3*365.25*24*60*60

def create_submesh(mesh, cell_indices, cell_tags=None, facet_tags=None):
    """
    Function to return a submesh based on the cell indices provided using a split comm
    that excludes ranks with no cells.

    Arguments:
      * mesh         - original (parent) mesh
      * cell_indices - cell indices of the parent mesh to include in the submesh

    Keyword Arguments:
      * cell_tags    - cell tags on parent mesh that will be mapped and returned relative to submesh (default=None)
      * facet_tags   - facet tags on parent mesh that will be mapped and returned relative to submesh (default=None)

    Returns:
      * submesh            - submesh of mesh given cell_indices
      * submesh_cell_tags  - cell tags relative to submesh if cell_tags provided (otherwise None)
      * submesh_facet_tags - facet tags relative to submesh if facet_tags provided (otherwise None)
      * submesh_cell_map   - map from submesh cells to parent cells
    """
    tdim = mesh.topology.dim
    fdim = tdim-1
    submesh0, submesh0_cell_map, submesh0_vertex_map, submesh0_geom_map = \
                  df.mesh.create_submesh(mesh, tdim, cell_indices)
    
    # create a split comm
    new_comm = mesh.comm.Split(submesh0.topology.index_map(tdim).size_local>0)

    # map from the rank on the original comm to the new split comm
    # used to map index map owners to their new ranks
    # this will include undefined negative numbers on ranks with no corresponding entry on the split comm that should cause failure if the mesh has ghosts on unexpected ranks
    rank_map = np.asarray(submesh0.comm.group.Translate_ranks(None, new_comm.group), dtype=np.int32)

    # create a new topology on the new comm
    new_topo = df.cpp.mesh.Topology(new_comm, submesh0.topology.cell_type)

    new_topo_im_0 = df.common.IndexMap(new_comm, 
                                    submesh0.topology.index_map(0).size_local, 
                                    submesh0.topology.index_map(0).ghosts, 
                                    rank_map[submesh0.topology.index_map(0).owners])
    new_topo.set_index_map(0, new_topo_im_0)

    new_topo_im_tdim = df.common.IndexMap(new_comm, 
                                    submesh0.topology.index_map(tdim).size_local, 
                                    submesh0.topology.index_map(tdim).ghosts, 
                                    rank_map[submesh0.topology.index_map(tdim).owners])
    new_topo.set_index_map(tdim, new_topo_im_tdim)

    new_topo.set_connectivity(submesh0.topology.connectivity(0,0), 0, 0)
    new_topo.set_connectivity(submesh0.topology.connectivity(tdim, 0), tdim, 0)

    gdim = submesh0.geometry.dim
    new_geom_im = df.common.IndexMap(new_comm,
                                    submesh0.geometry.index_map().size_local,
                                    submesh0.geometry.index_map().ghosts,
                                    rank_map[submesh0.geometry.index_map().owners])
    new_geom = type(submesh0.geometry._cpp_object)(new_geom_im, submesh0.geometry.dofmap, 
                                    submesh0.geometry.cmap._cpp_object, 
                                    submesh0.geometry.x[:,:gdim], 
                                    submesh0.geometry.input_global_indices)
    
    # set up a new submesh
    submesh = df.mesh.Mesh(type(submesh0._cpp_object)(new_comm, new_topo, new_geom), submesh0.ufl_domain())
    submesh.topology.create_connectivity(fdim, tdim)

    # if cell_tags are provided then map to the submesh
    submesh_cell_tags = None
    if cell_tags is not None:
        submesh_cell_tags_indices = []
        submesh_cell_tags_values  = []
        # loop over the submesh cells, checking if they're included in
        # the parent cell_tags
        for i,parentind in enumerate(submesh0_cell_map):
            parent_cell_tags_indices = np.argwhere(cell_tags.indices==parentind)
            if parent_cell_tags_indices.shape[0]>0:
                submesh_cell_tags_indices.append(i)
                submesh_cell_tags_values.append(cell_tags.values[parent_cell_tags_indices[0][0]])
        submesh_cell_tags_indices = np.asarray(submesh_cell_tags_indices)
        submesh_cell_tags_values  = np.asarray(submesh_cell_tags_values)

        # create a new meshtags object
        # indices should already be sorted by construction
        submesh_cell_tags = df.mesh.meshtags(mesh, tdim, 
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
            submesh_parentvs2subf[tuple(sorted([submesh0_vertex_map[j] for j in submesh_f2vs.links(i)]))] = i

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
        submesh_facet_tags = df.mesh.meshtags(mesh, fdim, 
                                              submesh_facet_tags_indices[perm], 
                                              submesh_facet_tags_values[perm])
    
    return submesh, submesh_cell_tags, submesh_facet_tags, submesh0_cell_map

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

@functools.singledispatch
def vtk_mesh(mesh: df.mesh.Mesh):
    tdim = mesh.topology.dim
    if mesh.topology.index_map(tdim).size_local > 0:
        return df.plot.vtk_mesh(mesh)
    else:
        cell_type = df.cpp.mesh.cell_entity_type(mesh.topology.cell_type, tdim, 0)
        vtk_type = df.cpp.io.get_vtk_cell_type(cell_type, tdim)
        cell_types = np.full(0, vtk_type)
        x = mesh.geometry.x
        num_nodes_per_cell = mesh.geometry.dofmap.shape[-1]
        topology = np.empty((0, num_nodes_per_cell + 1), dtype=np.int32)
        return topology.reshape(-1), cell_types, x

@vtk_mesh.register
def _(V: df.fem.FunctionSpace):
    if V.ufl_element().degree == 0:
        return vtk_mesh(V.mesh)
    else:
        return df.plot.vtk_mesh(V)

@vtk_mesh.register
def _(u: df.fem.Function):
    return vtk_mesh(u.function_space)


@functools.singledispatch
def pyvista_grids(cells: np.ndarray, types: np.ndarray, x: np.ndarray, 
                  comm: MPI.Intracomm=None, gather: bool=False):
    grids = []
    if gather:
        cells_g = comm.gather(cells, root=0)
        types_g = comm.gather(types, root=0)
        x_g = comm.gather(x, root=0)
        if comm.rank == 0:
            for r in range(comm.size):
                grids.append(pv.UnstructuredGrid(cells_g[r], types_g[r], x_g[r]))
    else:
        grids.append(pv.UnstructuredGrid(cells, types, x))
    return grids

@pyvista_grids.register
def _(mesh: df.mesh.Mesh, gather=False):
    return pyvista_grids(*vtk_mesh(mesh), comm=mesh.comm, gather=gather)

@pyvista_grids.register
def _(V: df.fem.FunctionSpace, gather=False):
    return pyvista_grids(*vtk_mesh(V), comm=V.mesh.comm, gather=gather)

@pyvista_grids.register
def _(u: df.fem.Function, gather=False):
    return pyvista_grids(*vtk_mesh(u), comm=u.function_space.mesh.comm, gather=gather)

def plot_mesh(mesh, tags=None, plotter=None, gather=False, **pv_kwargs):
    """
    Plot a dolfinx mesh using pyvista.

    Arguments:
      * mesh        - the mesh to plot

    Keyword Arguments:
      * tags        - mesh tags to color plot by (either cell or facet, default=None)
      * plotter     - a pyvista plotter, one will be created if none supplied (default=None)
      * gather      - gather plot to rank 0 (default=False)
      * **pv_kwargs - kwargs for adding the mesh to the plotter
    """

    comm = mesh.comm

    grids = pyvista_grids(mesh, gather=gather)

    tdim = mesh.topology.dim
    fdim = tdim - 1
    if tags is not None:
        cell_imap = mesh.topology.index_map(tdim)
        num_cells = cell_imap.size_local + cell_imap.num_ghosts
        marker = np.zeros(num_cells)
        if tags.dim == tdim:
            for i, ind in enumerate(tags.indices):
                marker[ind] = tags.values[i]
        elif tags.dim == fdim:
            mesh.topology.create_connectivity(fdim, tdim)
            fcc = mesh.topology.connectivity(fdim, tdim)
            for f,v in enumerate(tags.values):
                for c in fcc.links(tags.indices[f]):
                    marker[c] = v
        else:
            raise Exception("Unknown tag dimension!")

        if gather:
            marker_g = comm.gather(marker, root=0)
        else:
            marker_g = [marker]

        for r, grid in enumerate(grids):
            grid.cell_data["Marker"] = marker_g[r]
            grid.set_active_scalars("Marker")
    
    if len(grids) > 0 and plotter is None: plotter = pv.Plotter()

    if plotter is not None:
        for grid in grids: 
            if grid.GetNumberOfPoints() > 0:
                plotter.add_mesh(grid, **pv_kwargs)
        if mesh.geometry.dim == 2:
            plotter.enable_parallel_projection()
            plotter.view_xy()

    return plotter

def grids_scalar(scalar, scale=1.0, gather=False):
    """
    Return a list of pyvista grids for a scalar Function.

    Arguments:
      * scalar      - the dolfinx scalar Function to grid

    Keyword Arguments:
      * scale       - a scalar scale factor that the values are multipled by (default=1.0)
      * gather      - gather plot to rank 0 (default=False)
    """
    
    comm = scalar.function_space.mesh.comm
    
    grids = pyvista_grids(scalar, gather=gather)

    if scalar.function_space.ufl_element().degree == 0:
        tdim = scalar.function_space.mesh.topology.dim
        cell_imap = scalar.function_space.mesh.topology.index_map(tdim)
        num_cells = cell_imap.size_local + cell_imap.num_ghosts
        perm = [scalar.function_space.dofmap.cell_dofs(c)[0] for c in range(num_cells)]
        values = scalar.x.array.real[perm]*scale
    else:
        values = scalar.x.array.real*scale
        
    if gather:
        values_g = comm.gather(values, root=0)
    else:
        values_g = [values]

    for r, grid in enumerate(grids):
        if scalar.function_space.element.space_dimension==1:
            grid.cell_data[scalar.name] = values_g[r]
        else:
            grid.point_data[scalar.name] = values_g[r]
        grid.set_active_scalars(scalar.name)

    return grids

def plot_scalar(scalar, scale=1.0, plotter=None, gather=False, **pv_kwargs):
    """
    Plot a dolfinx scalar Function using pyvista.

    Arguments:
      * scalar      - the dolfinx scalar Function to plot

    Keyword Arguments:
      * scale       - a scalar scale factor that the values are multipled by (default=1.0)
      * plotter     - a pyvista plotter, one will be created if none supplied (default=None)
      * gather      - gather plot to rank 0 (default=False)
      * **pv_kwargs - kwargs for adding the mesh to the plotter
    """

    grids = grids_scalar(scalar, scale=scale, gather=gather)

    if len(grids) > 0 and plotter is None: plotter = pv.Plotter()

    if plotter is not None:
        for grid in grids: plotter.add_mesh(grid, **pv_kwargs)
        if scalar.function_space.mesh.geometry.dim == 2:
            plotter.enable_parallel_projection()
            plotter.view_xy()

    return plotter

def plot_scalar_values(scalar, scale=1.0, fmt=".2f", plotter=None, gather=False, **pv_kwargs):
    """
    Print values of a dolfinx scalar Function using pyvista.

    Arguments:
      * scalar  - the dolfinx scalar Function to plot

    Keyword Arguments:
      * scale       - a scalar scale factor that the values are multipled by (default=1.0)
      * fmt         - string formatting (default='.2f')
      * plotter     - a pyvista plotter, one will be created if none supplied (default=None)
      * gather      - gather plot to rank 0 (default=False)
      * **pv_kwargs - kwargs for the point labels
    """

    comm = scalar.function_space.mesh.comm
    
    # based on plot_function_dofs in febug
    V = scalar.function_space

    x = V.tabulate_dof_coordinates()

    size_local = V.dofmap.index_map.size_local
    num_ghosts = V.dofmap.index_map.num_ghosts
    bs = V.dofmap.bs
    values = scalar.x.array.reshape((-1, bs))*scale

    if gather:
        # only gather the local entries
        x_g = comm.gather(x[:size_local], root=0)
        values_g = comm.gather(values[:size_local], root=0)
        size_local = None
        num_ghosts = 0
    else:
        x_g = [x]
        values_g = [values]
    
    formatter = lambda x: "\n".join((f"{u_:{fmt}}" for u_ in x))

    if values_g is not None and plotter is None: plotter = pv.Plotter()
    
    if plotter is not None:
        if size_local is None or size_local > 0:
            for r in range(len(values_g)):
                x_local_polydata = pv.PolyData(x_g[r][:size_local])
                x_local_polydata["labels"] = list(
                    map(formatter, values_g[r][:size_local]))
                plotter.add_point_labels(
                    x_local_polydata, "labels", **pv_kwargs,
                    point_color="black")
    
        # we only get here if gather is False so can use x and values
        if num_ghosts > 0:
            x_ghost_polydata = pv.PolyData(x[size_local:size_local+num_ghosts])
            x_ghost_polydata["labels"] = list(
                map(formatter, values[size_local:size_local+num_ghosts]))
            plotter.add_point_labels(
                x_ghost_polydata, "labels", **pv_kwargs,
                point_color="pink")
    
        if V.mesh.geometry.dim == 2:
            plotter.enable_parallel_projection()
            plotter.view_xy()

    return plotter

def plot_vector(vector, scale=1.0, plotter=None, gather=False, **pv_kwargs):
    """
    Plot a dolfinx vector Function using pyvista.

    Arguments:
      * vector      - the dolfinx vector Function to plot

    Keyword Arguments:
      * scale       - a scalar scale factor that the values are multipled by (default=1.0)
      * plotter     - a pyvista plotter, one will be created if none supplied (default=None)
      * gather      - gather plot to rank 0 (default=False)
      * **pv_kwargs - kwargs for adding the mesh to the plotter
    """

    comm = vector.function_space.mesh.comm

    grids = pyvista_grids(vector, gather=gather)

    imap = vector.function_space.dofmap.index_map
    nx = imap.size_local + imap.num_ghosts
    values = np.zeros((nx, 3))
    values[:, :len(vector)] = vector.x.array.real.reshape((nx, len(vector)))*scale

    if gather:
        values_g = comm.gather(values, root=0)
    else:
        values_g = [values]

    for r, grid in enumerate(grids):
        grid[vector.name] = values_g[r]
    
    if len(grids) > 0 and plotter is None: plotter = pv.Plotter()

    if plotter is not None:
        for grid in grids: plotter.add_mesh(grid, **pv_kwargs)

        if vector.function_space.mesh.geometry.dim == 2:
            plotter.enable_parallel_projection()
            plotter.view_xy()

    return plotter

def plot_vector_glyphs(vector, factor=1.0, scale=1.0, plotter=None, gather=False, **pv_kwargs):
    """
    Plot dolfinx vector Function as glyphs using pyvista.

    Arguments:
      * vector      - the dolfinx vector Function to plot

    Keyword Arguments:
      * factor      - scale for glyph size (default=1.0)
      * scale       - a scalar scale factor that the values are multipled by (default=1.0)
      * plotter     - a pyvista plotter, one will be created if none supplied (default=None)
      * gather      - gather plot to rank 0 (default=False)
      * **pv_kwargs - kwargs for adding the mesh to the plotter
    """

    comm = vector.function_space.mesh.comm

    grids = pyvista_grids(vector, gather=gather)

    imap = vector.function_space.dofmap.index_map
    nx = imap.size_local + imap.num_ghosts
    values = np.zeros((nx, 3))
    values[:, :len(vector)] = vector.x.array.real.reshape((nx, len(vector)))*scale

    if gather:
        values_g = comm.gather(values, root=0)
    else:
        values_g = [values]

    glyphs_g = []
    for r, grid in enumerate(grids):
        grid[vector.name] = values_g[r]
        geom = pv.Arrow()
        glyphs_g.append(grid.glyph(orient=vector.name, factor=factor, geom=geom))
    
    if len(grids) > 0 and plotter is None: plotter = pv.Plotter()

    if plotter is not None:
        for glyphs in glyphs_g: plotter.add_mesh(glyphs, **pv_kwargs)
    
        if vector.function_space.mesh.geometry.dim == 2:
            plotter.enable_parallel_projection()
            plotter.view_xy()

    return plotter

def plot_slab(slab, plotter=None, **pv_kwargs):
    """
    Plot a slab spline using pyvista.

    Arguments:
      * slab        - the slab spline object to plot

    Keyword Arguments:
      * plotter     - a pyvista plotter, one will be created if none supplied (default=None)
      * **pv_kwargs - kwargs for adding the mesh to the plotter
    """
    slabpoints = np.empty((2*len(slab.interpcurves), 3))
    for i, curve in enumerate(slab.interpcurves):
        slabpoints[2*i] = [curve.points[0].x, curve.points[0].y, 0.0]
        slabpoints[2*i+1] = [curve.points[-1].x, curve.points[-1].y, 0.0]
    if plotter is None: plotter = pv.Plotter()
    if plotter is not None:
        plotter.add_lines(slabpoints, **pv_kwargs)
    return plotter

def plot_geometry(geom, plotter=None, **pv_kwargs):
    """
    Plot the subduction zone geometry using pyvista.

    Arguments:
      * geom        - the geometry object to plot

    Keyword Arguments:
      * plotter     - a pyvista plotter, one will be created if none supplied (default=None)
      * **pv_kwargs - kwargs for adding the mesh to the plotter
    """
    lines = [line for lineset in geom.crustal_lines for line in lineset] + \
             geom.slab_base_lines + \
             geom.wedge_base_lines + \
             geom.slab_side_lines + \
             geom.wedge_side_lines + \
             geom.wedge_top_lines + \
             geom.slab_spline.interpcurves
    points = np.empty((len(lines)*2, 3))
    for i, line in enumerate(lines):
        points[2*i] = [line.points[0].x, line.points[0].y, 0.0]
        points[2*i+1] = [line.points[-1].x, line.points[-1].y, 0.0]
    if plotter is None: plotter = pv.Plotter()
    if plotter is not None:
        plotter.add_lines(points,**pv_kwargs)
    return plotter

def plot_couplingdepth(slab, plotter=None, **pv_kwargs):
    """
    Plot a point representing the coupling depth using pyvista.

    Arguments:
      * slab        - the slab spline object containing the 'Slab::FullCouplingDepth' point

    Keyword Arguments:
      * plotter     - a pyvista plotter, one will be created if none supplied (default=None)
      * **pv_kwargs - kwargs for adding the mesh to the plotter
    """
    point = slab.findpoint('Slab::FullCouplingDepth')
    if plotter is None: plotter = pv.Plotter()
    if plotter is not None:
        plotter.add_points(np.asarray([point.x, point.y, 0.0]), **pv_kwargs)
    return plotter

def plot_show(plotter, **pv_kwargs):
    """
    Display a pyvista plotter.

    Arguments:
      * plotter  - the pyvista plotter
    """    
    if plotter is not None and not pv.OFF_SCREEN:
        plotter.show(**pv_kwargs)

def plot_save(plotter, filename, **pv_kwargs):
    """
    Display a pyvista plotter.

    Arguments:
      * plotter  - the pyvista plotter
      * filename - filename to save image to
    """
    if plotter is not None:
        figure = plotter.screenshot(filename, **pv_kwargs)

class PVGridProbe:
    """
    A class that probes a pyvista grid and given coordinates.
    """
    
    def __init__(self, grid, xyz):
        """
        A class that probes a pyvista grid and given coordinates.

        Arguments:
          * grid - a pyvista grid
          * xyz  - coordinates
        """
        # save the grid
        self.grid = grid
        
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(grid)
        locator.SetTolerance(10.0)
        locator.Update()
        
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        ilen, jlen = xyz.shape
        for i in range(ilen):
            points.InsertNextPoint(xyz[i][0], xyz[i][1], xyz[i][2])
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        # set up the probe
        self.probe = vtk.vtkProbeFilter()
        self.probe.SetInputData(polydata)
        self.probe.SetSourceData(self.grid)
        self.probe.Update()
        
        valid_ids = self.probe.GetValidPoints()
        num_locs = valid_ids.GetNumberOfTuples()
        valid_loc = 0
        # save a list of invalid nodes not found by the probe
        self.invalidNodes = []
        for i in range(ilen):
            if valid_loc < num_locs and valid_ids.GetTuple1(valid_loc) == i:
                valid_loc += 1
            else:
                nearest = locator.FindClosestPoint([xyz[i][0], xyz[i][1], xyz[i][2]])
                self.invalidNodes.append((i, nearest))
        
    def get_field(self, name):
        """
        Return a numpy array containing the named field at the saved coordinates.

        Arguments:
          * name - name of field to probe
        """
        probepointData = self.probe.GetOutput().GetPointData()
        probevtkData = probepointData.GetArray(name)
        nc = probevtkData.GetNumberOfComponents()
        nt = probevtkData.GetNumberOfTuples()
        array = np.asarray([probevtkData.GetValue(i) for i in range(nt * nc)])
        
        if len(self.invalidNodes) > 0:
            field = self.grid.GetPointData().GetArray(name)
            if field is None: field = self.grid.GetCellData().GetArray(name)
            if field is None: 
                raise Exception("ERROR: no point of cell data with name {}.".format(name,))
            components = field.GetNumberOfComponents()
            for invalidNode, nearest in self.invalidNodes:
                for comp in range(nc):
                    array[invalidNode * nc + comp] = field.GetValue(nearest * nc + comp)
        
        if nc==9:
            array = array.reshape(nt, 3, 3)
        elif nc==4:
            array = array.reshape(nt, 2, 2)
        elif nc==1:
            array = array.reshape(nt,)
        else:
            array = array.reshape(nt, nc)
        
        return array

def pvgrid_test_points(grid1, grid2, tol=1.e-6):
    """
    Test if two grids have the same point coordinates to the given tolerance.

    Arguments:
      * grid1 - first grid
      * grid2 - second grid

    Keyword Arguments:
      * tol - tolerance (defaults to 1.e-6)
    """
    locs1 = grid1.points
    locs2 = grid2.points
    if not len(locs1) == len(locs2):
        return False
    for i in range(len(locs1)):
        if not len(locs1[i]) == len(locs2[i]):
            return False
        for j in range(len(locs1[i])):
            if np.abs(locs1[i][j] - locs2[i][j]) > tol:
                return False
    return True

def pv_diff(grid1, grid2, field_name_map={}, pass_point_data=False, pass_cell_data=False):
    """
    Take the difference between the fields on two pyvista grids, grid1 - grid2.

    This functionality overlaps with the pyvista sample filter but tries to handle coordinates
    that aren't found better.

    Arguments:
      * grid1 - first grid
      * grid2 - second grid
      * field_name_map - map between names of the fields on the first grid to the names on the second grid
      * pass_point_data - keep the original point data using the names _name_1 and _name_2
      * pass_cell_data  - keep the original cell data using the names _name_1 and _name_2
    """
    outgrid = pv.UnstructuredGrid(grid1.cells, grid1.celltypes, grid1.points)

    useprobe = not pvgrid_test_points(grid1, grid2)
    if useprobe: probe = PVGridProbe(grid2, grid1.points)

    pointnames1 = grid1.point_data.keys()
    pointnames2 = grid2.point_data.keys()
    for name1 in pointnames1:
        name2 = field_name_map.get(name1, name1)
        field1 = grid1.point_data[name1]
        if name2 in pointnames2:
            if useprobe:
                field2 = probe.get_field(name2)
            else:
                field2 = grid2.point_data[name2]
            outgrid.point_data[name1] = field1-field2
            if pass_point_data: outgrid.point_data["_"+name1+"_2"] = field2
        if pass_point_data: outgrid.point_data["_"+name1+"_1"] = field1

    cellnames1 = grid1.cell_data.keys()
    cellnames2 = grid2.cell_data.keys()
    if useprobe:
        for name1 in cellnames1:
            name2 = field_name_map.get(name1, name1)
            if pass_cell_data: 
                outgrid.cell_data["_"+name1+"_1"] = grid1.point_data[name1]
                if name2 in cellnames2: outgrid.cell_data["_"+name1+"_2"] = grid2.cell_data[name2]
    else:
        for name1 in cellnames1:
            name2 = field_name_map.get(name1, name1)
            field1 = grid1.cell_data(name1)
            if name2 in cellnames2:
                field2 = grid2.cell_data[name2]
                outgrid.cell_data[name1] = field1 - field2
                if pass_cell_data: outgrid.cell_data["_"+name1+"_2"] = field2
            if pass_cell_data: outgrid.point_data["_"+name1+"_1"] = field1

    return outgrid
