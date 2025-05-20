#!/usr/bin/env python
# coding: utf-8

from mpi4py import MPI
import dolfinx as df
from petsc4py import PETSc
import dolfinx.fem.petsc
import numpy as np
import ufl
import matplotlib.pyplot as pl
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))
import utils
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
if __name__ == "__main__":
    output_folder = pathlib.Path(os.path.join(basedir, "output"))
    output_folder.mkdir(exist_ok=True, parents=True)


def solve_poisson_2d(ne, p=1, petsc_options=None):
    """
    A python function to solve a two-dimensional Poisson problem
    on a unit square domain.
    Parameters:
    * ne - number of elements in each dimension
    * p  - polynomial order of the solution function space
    * petsc_options - 
    """

    # Set the default PETSc solver options if none have been supplied
    if petsc_options is None:
        petsc_options = {"ksp_type": "preonly", \
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"}
    opts = PETSc.Options()
    for k, v in petsc_options.items(): opts[k] = v
    
    # Describe the domain (a unit square)
    # and also the tessellation of that domain into ne 
    # equally spaced squares in each dimension which are
    # subdivided into two triangular elements each
    with df.common.Timer("Poisson Mesh"):
        mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, ne, ne, ghost_mode=df.mesh.GhostMode.none)

    # Define the solution function space using Lagrange polynomials
    # of order p
    with df.common.Timer("Poisson Functions"):
        V = df.fem.functionspace(mesh, ("Lagrange", p))
        T_i = df.fem.Function(V)

    with df.common.Timer("Poisson Dirichlet BCs"):
        # Define the location of the boundary condition, x=0 and y=0
        def boundary(x):
            return np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0))
        boundary_dofs = df.fem.locate_dofs_geometrical(V, boundary)
        # Specify the value and define a Dirichlet boundary condition (bc)
        gD = df.fem.Function(V)
        gD.interpolate(lambda x: np.exp(x[0] + x[1]/2.))
        bc = df.fem.dirichletbc(gD, boundary_dofs)

    with df.common.Timer("Poisson Neumann BCs"):
        # Get the coordinates
        x = ufl.SpatialCoordinate(mesh)
        # Define the Neumann boundary condition function
        gN = ufl.as_vector((ufl.exp(x[0] + x[1]/2.), 0.5*ufl.exp(x[0] + x[1]/2.)))
        # Define the right hand side function, h
        h = -5./4.*ufl.exp(x[0] + x[1]/2.)

    with df.common.Timer("Poisson Forms"):
        T_a = ufl.TrialFunction(V)
        T_t = ufl.TestFunction(V)
        # Get the unit vector normal to the facets
        n = ufl.FacetNormal(mesh)
        # Define the integral to be assembled into the stiffness matrix
        S = df.fem.form(ufl.inner(ufl.grad(T_t), ufl.grad(T_a))*ufl.dx)
        # Define the integral to be assembled into the forcing vector,
        # incorporating the Neumann boundary condition weakly
        f = df.fem.form(T_t*h*ufl.dx + T_t*ufl.inner(gN, n)*ufl.ds)

    # The next two sections "Poisson Assemble" and "Poisson Solve"
    # are the equivalent of the much simpler:
    # ```
    # problem = df.fem.petsc.LinearProblem(S, f, bcs=[bc], \
    #                                      petsc_options=petsc_options)
    # T_i = problem.solve()
    # ```
    # We split them up here so we can time and profile each step separately.
    with df.common.Timer("Poisson Assemble"):
        # Assemble the matrix from the S form
        A = df.fem.petsc.assemble_matrix(S, bcs=[bc])
        A.assemble()
        # Assemble the R.H.S. vector from the f form
        b = df.fem.petsc.assemble_vector(f)

        # Set the boundary conditions
        df.fem.petsc.apply_lifting(b, [S], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        df.fem.petsc.set_bc(b, [bc])

    with df.common.Timer("Poisson Solve"):
        # Setup the solver
        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setFromOptions()

        pc = ksp.getPC()
        try:
            pc.setFactorSetUpSolverType()
        except PETSc.Error as e:
            if e.ierr == 92:
                print("The required PETSc solver/preconditioner is not available. Exiting.")
                print(e)
                exit(0)
            else:
                raise e
        
        # Call the solver
        ksp.solve(b, T_i.x.petsc_vec)
        # Communicate the solution across processes
        T_i.x.scatter_forward()

    return T_i


if __name__ == "__main__":
    ne = 4
    p = 1
    petsc_options = {'ksp_type':'preonly', 'pc_type':'lu', 'pc_factor_mat_solver_type':'superlu_dist', 'ksp_view': None}
    petsc_options = {'ksp_type':'cg', 'pc_type':'gamg', 'ksp_view': None}
    T_P1 = solve_poisson_2d(ne, p=p, petsc_options=petsc_options)
    T_P1.name = "T (P1)"


if __name__ == "__main__":
    # plot the solution as a colormap
    plotter_P1 = utils.plot_scalar(T_P1, gather=True, cmap='coolwarm')
    # plot the mesh
    utils.plot_mesh(T_P1.function_space.mesh, plotter=plotter_P1, gather=True, show_edges=True, style="wireframe", color='k', line_width=2)
    # plot the values of the solution at the nodal points 
    utils.plot_scalar_values(T_P1, plotter=plotter_P1, gather=True, point_size=15, font_size=22, shape_color='w', text_color='k', bold=False)
    # show the plot
    utils.plot_show(plotter_P1)
    # save the plot
    utils.plot_save(plotter_P1, output_folder / "2d_poisson_P1_solution.png")
    comm = T_P1.function_space.mesh.comm
    if comm.size > 1:
        # if we're running in parallel (e.g. from a script) then save an image per process as well
        plotter_P1_p = utils.plot_scalar(T_P1)
        utils.plot_mesh(T_P1.function_space.mesh, plotter=plotter_P1_p, show_edges=True, style="wireframe", color='k', line_width=2)
        utils.plot_scalar_values(T_P1, plotter=plotter_P1_p, point_size=15, font_size=22, shape_color='w', text_color='k', bold=False)
        utils.plot_save(plotter_P1_p, output_folder / "2d_poisson_P1_solution_p{:d}.png".format(comm.rank,))


if __name__ == "__main__":
    ne = 4
    p = 2
    T_P2 = solve_poisson_2d(ne, p=p)
    T_P2.name = "T (P2)"


if __name__ == "__main__":
    # plot the solution as a colormap
    plotter_P2 = utils.plot_scalar(T_P2, gather=True, cmap='coolwarm')
    # plot the mesh
    utils.plot_mesh(T_P2.function_space.mesh, plotter=plotter_P2, gather=True, show_edges=True, style="wireframe", color='k', line_width=2)
    # plot the values of the solution at the nodal points 
    utils.plot_scalar_values(T_P2, plotter=plotter_P2, gather=True, point_size=15, font_size=12, shape_color='w', text_color='k', bold=False)
    # show the plot
    utils.plot_show(plotter_P2)
    # save the plot
    utils.plot_save(plotter_P2, output_folder / "2d_poisson_P2_solution.png")
    comm = T_P2.function_space.mesh.comm
    if comm.size > 1:
        # if we're running in parallel (e.g. from a script) then save an image per process as well
        plotter_P2_p = utils.plot_scalar(T_P2, cmap='coolwarm')
        utils.plot_mesh(T_P2.function_space.mesh, plotter=plotter_P2_p, show_edges=True, style="wireframe", color='k', line_width=2)
        #utils.plot_scalar_values(T_P2, plotter=plotter_P2_p, point_size=15, font_size=12, shape_color='w', text_color='k', bold=False)
        utils.plot_save(plotter_P2_p, output_folder / "2d_poisson_P2_solution_p{:d}.png".format(comm.rank,))


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/background --NbConvertApp.output_base=poisson_2d 2.3b_poisson_2d.ipynb')




