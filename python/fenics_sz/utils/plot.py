from mpi4py import MPI
import dolfinx as df
import pyvista as pv
import numpy as np
import functools
import vtk
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation

try:
    pv.start_xvfb()
except OSError:
    pass

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

def grids_mesh(mesh, tags=None, gather=False):
    """
    Return a list of pyvista grids for a dolfinx mesh, optionally coloring by mesh tags.

    Arguments:
      * mesh        - the mesh to grid

    Keyword Arguments:
      * tags        - mesh tags to color plot by (either cell or facet, default=None)
      * gather      - gather plot to rank 0 (default=False)
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

    return grids

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

    grids = grids_mesh(mesh, tags=tags, gather=gather)

    if len(grids) > 0 and plotter is None: plotter = pv.Plotter()

    if plotter is not None:
        for grid in grids: 
            if grid.GetNumberOfPoints() > 0:
                plotter.add_mesh(grid, **pv_kwargs)
        if mesh.geometry.dim == 2:
            plotter.enable_parallel_projection()
            plotter.view_xy()

    return plotter

def mpl_plot_pv_mesh(grid, ax, **mpl_kwargs):
    """
    Plot a pyvista (unstructured) grid as a mesh using matplotlib.

    Arguments:
      * grid - a pyvista grid
      * ax   - a matplotlib axis to plot on

    Keyword Arguments:
      * **mpl_kwargs - kwargs for the matplotlib PolyCollection
    """

    if grid.n_cells == 0: return ax

    polygons = [grid.points[grid.get_cell(i).point_ids, :2] for i in range(grid.n_cells)]

    active_scalars_name = grid.active_scalars_name
    if active_scalars_name is not None and active_scalars_name in grid.cell_data:
        collection = PolyCollection(polygons, array=grid.cell_data[active_scalars_name], **mpl_kwargs)
    else:
        collection = PolyCollection(polygons, **mpl_kwargs)

    ax.add_collection(collection)
    ax.autoscale_view()

    return ax

def mpl_plot_mesh(mesh, ax, tags=None, gather=False, **mpl_kwargs):
    """
    Plot a dolfinx mesh using matplotlib.

    Arguments:
      * mesh        - the mesh to plot
      * ax          - a matplotlib axis to plot on

    Keyword Arguments:
      * tags         - mesh tags to color plot by (either cell or facet, default=None)
      * gather       - gather plot to rank 0 (default=False)
      * **mpl_kwargs - kwargs for the matplotlib PolyCollection
    """

    grids = grids_mesh(mesh, tags=tags, gather=gather)

    for grid in grids:
        if grid.GetNumberOfPoints() > 0:
            mpl_plot_pv_mesh(grid, ax, **mpl_kwargs)

    if mesh.geometry.dim == 2:
        ax.set_aspect("equal")

    return ax

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

def mpl_pv_triangles(grid):
    """
    Return a matplotlib-compatible (n, 3) triangle connectivity array for a pyvista grid.

    Straight-sided (3-node) triangle cells map to a single matplotlib triangle.
    Quadratic (6-node) triangle cells are subdivided into 4 linear sub-triangles
    using their existing corner and mid-edge points. This exactly reproduces the
    quadratic field along the sub-triangle edges but only approximates it in the
    interior of each sub-triangle, since matplotlib only supports linear (Gouraud)
    interpolation across a triangle.

    Arguments:
      * grid - a pyvista grid (triangle or quadratic-triangle cells only)
    """
    triangles = []
    for i in range(grid.n_cells):
        cell = grid.get_cell(i)
        pids = cell.point_ids
        if cell.type == vtk.VTK_TRIANGLE:
            triangles.append(pids)
        elif cell.type in (vtk.VTK_QUADRATIC_TRIANGLE, vtk.VTK_LAGRANGE_TRIANGLE) and len(pids) == 6:
            p0, p1, p2, p3, p4, p5 = pids
            triangles.extend([[p0, p3, p5], [p3, p1, p4], [p5, p4, p2], [p3, p4, p5]])
        else:
            raise Exception("mpl_pv_triangles only supports linear (3-node) and quadratic (6-node) triangle cells, not cell type {} with {} points!".format(cell.type, len(pids)))
    return np.asarray(triangles, dtype=np.int64)

def mpl_plot_pv_scalar(grid, ax, **mpl_kwargs):
    """
    Plot a scalar field on a pyvista grid using matplotlib.

    Cell data (e.g. from a DG0 Function) is drawn as flat-shaded polygons, one
    color per cell, via mpl_plot_pv_mesh. Point data (e.g. from a continuous
    Lagrange Function) is drawn with Gouraud-shaded triangles via mpl_pv_triangles,
    which is exact for linear (P1) triangles and a piecewise-linear approximation
    for quadratic (P2) triangles. Only triangle and quadratic-triangle cells are
    supported for point data.

    Arguments:
      * grid - a pyvista grid with an active scalar field
      * ax   - a matplotlib axis to plot on

    Keyword Arguments:
      * **mpl_kwargs - kwargs for the matplotlib PolyCollection (cell data) or
                        tripcolor (point data)
    """

    if grid.n_cells == 0: return ax

    active_scalars_name = grid.active_scalars_name
    if active_scalars_name is None:
        raise Exception("grid has no active scalars to plot!")

    if active_scalars_name in grid.cell_data:
        return mpl_plot_pv_mesh(grid, ax, **mpl_kwargs)

    values = grid.point_data[active_scalars_name]
    points = grid.points
    triangles = mpl_pv_triangles(grid)
    triangulation = Triangulation(points[:, 0], points[:, 1], triangles)

    mpl_kwargs.setdefault("shading", "gouraud")
    ax.tripcolor(triangulation, values, **mpl_kwargs)
    ax.autoscale_view()

    return ax

def mpl_plot_scalar(scalar, ax, scale=1.0, gather=False, **mpl_kwargs):
    """
    Plot a dolfinx scalar Function using matplotlib.

    Arguments:
      * scalar - the dolfinx scalar Function to plot
      * ax     - a matplotlib axis to plot on

    Keyword Arguments:
      * scale        - a scalar scale factor that the values are multipled by (default=1.0)
      * gather       - gather plot to rank 0 (default=False)
      * **mpl_kwargs - kwargs for the matplotlib PolyCollection (cell data) or
                        tripcolor (point data)
    """

    grids = grids_scalar(scalar, scale=scale, gather=gather)

    for grid in grids:
        if grid.GetNumberOfPoints() > 0:
            mpl_plot_pv_scalar(grid, ax, **mpl_kwargs)

    if scalar.function_space.mesh.geometry.dim == 2:
        ax.set_aspect("equal")

    return ax

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
            pv_kwargs.pop('shape_color', None)
            pv_kwargs.pop('point_color', None)
            plotter.add_point_labels(
                x_ghost_polydata, "labels", **pv_kwargs,
                point_color="pink", shape_color="pink")
    
        if V.mesh.geometry.dim == 2:
            plotter.enable_parallel_projection()
            plotter.view_xy()

    return plotter

def plot_dofmap(V, plotter=None, gather=False, **pv_kwargs):
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

    comm = V.mesh.comm

    x = V.tabulate_dof_coordinates()

    size_local = V.dofmap.index_map.size_local
    num_ghosts = V.dofmap.index_map.num_ghosts
    bs = V.dofmap.bs
    dtype = V.dofmap.index_map.ghosts.dtype
    values = np.concatenate((np.arange(*V.dofmap.index_map.local_range, dtype=dtype), V.dofmap.index_map.ghosts), axis=0, dtype=dtype)

    if gather:
        # only gather the local entries
        x_g = comm.gather(x[:size_local], root=0)
        values_g = comm.gather(values[:size_local], root=0)
        size_local = None
        num_ghosts = 0
    else:
        x_g = [x]
        values_g = [values]
    
    fmt='d'
    formatter = lambda x: "".join((f"{x:{fmt}}"))

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
                point_color="pink", shape_color="pink")
    
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

def plot_vector_glyphs(vector, factor=1.0, scale=1.0, plotter=None, gather=False, tolerance=None, **pv_kwargs):
    """
    Plot dolfinx vector Function as glyphs using pyvista.

    Arguments:
      * vector      - the dolfinx vector Function to plot

    Keyword Arguments:
      * factor      - scale for glyph size (default=1.0)
      * scale       - a scalar scale factor that the values are multipled by (default=1.0)
      * plotter     - a pyvista plotter, one will be created if none supplied (default=None)
      * gather      - gather plot to rank 0 (default=False)
      * tolerance   - tolerance for pyvista glyph selection (default=None)
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
        glyphs_g.append(grid.glyph(orient=vector.name, factor=factor, geom=geom, tolerance=tolerance))
    
    if len(grids) > 0 and plotter is None: plotter = pv.Plotter()

    if plotter is not None:
        for glyphs in glyphs_g: plotter.add_mesh(glyphs, **pv_kwargs)
    
        if vector.function_space.mesh.geometry.dim == 2:
            plotter.enable_parallel_projection()
            plotter.view_xy()

    return plotter

def plot_points(points, plotter=None, **pv_kwargs):
    """
    Plot points using pyvista.

    Arguments:
      * points      - array of [[x, y, z]] points

    Keyword Arguments:
      * plotter     - a pyvista plotter, one will be created if none supplied (default=None)
      * **pv_kwargs - kwargs for adding the mesh to the plotter
    """
    if plotter is None: plotter = pv.Plotter()
    if plotter is not None:
        plotter.add_points(np.asarray(points), **pv_kwargs)
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

def plot_save_graphic(plotter, filename, **pv_kwargs):
    """
    Save graphic of a pyvista plotter.

    Arguments:
      * plotter  - the pyvista plotter
      * filename - filename to save image to
    """
    if plotter is not None:
        figure = plotter.save_graphic(filename, **pv_kwargs)

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
