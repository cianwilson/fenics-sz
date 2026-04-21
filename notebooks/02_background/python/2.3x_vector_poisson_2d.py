# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dolfinx-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Vector Poisson Example 2D

# %% [markdown]
# ### Preamble

# %% [markdown]
# We start by loading all the modules we will require and initializing our plotting preferences through [pyvista](https://pyvista.org/).

# %%
from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
from petsc4py import PETSc
import numpy as np
import ufl
import matplotlib.pyplot as pl
import basix
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python', 'fenics_sz'))
import fenics_sz.utils.plot
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


# %% [markdown]
# ### Solution

# %%
def solve_nested_poisson_2d(ne, p=1, petsc_options=None):
    """
    A python function to solve a two-dimensional corner flow 
    problem on a unit square domain.
    Parameters:
    * ne - number of elements in each dimension
    * p  - polynomial order of the solution (defaults to 1)
    * petsc_options - a dictionary of petsc options to pass to the solver 
                      (defaults to an LU direct solver using the MUMPS library)
    """

    if petsc_options is None:
        petsc_options = {"ksp_type": "preonly", \
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"}
    pc_type = petsc_options.get('pc_type', None)
    
    opts = PETSc.Options(); opts.clear()
    for k, v in petsc_options.items(): opts[k] = v

    # Describe the domain (a unit square)
    # and also the tessellation of that domain into ne
    # equally spaced squared in each dimension, which are
    # subduvided into two triangular elements each
    with df.common.Timer("Poisson Mesh"):
        mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, ne, ne, ghost_mode=df.mesh.GhostMode.none)

    # Define the solution function space using Lagrange polynomials
    # of order p
    with df.common.Timer("Poisson Functions"):
        e = basix.ufl.element("Lagrange", mesh.basix_cell(), p, shape=(mesh.geometry.dim,))
        V = df.fem.functionspace(mesh, e)

    with df.common.Timer("Poisson Dirichlet BCs"):
        bcs = []
        # Define the location of the boundary condition, x=0 and y=0
        def boundary(x):
            return np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0))
        boundary_dofs = df.fem.locate_dofs_geometrical(V, boundary)
        # Specify the value and define a Dirichlet boundary condition (bc)
        gD = df.fem.Function(V)
        gD.interpolate(lambda x: [np.exp(x[0] + x[1]/2.), np.exp(x[0] + x[1]/2.)])
        bcs.append(df.fem.dirichletbc(gD, boundary_dofs))

    with df.common.Timer("Poisson Neumann BCs"):
        # Get the coordinates
        x = ufl.SpatialCoordinate(mesh)
        # Define the Neumann boundary condition function
        gN = ufl.as_vector((ufl.exp(x[0] + x[1]/2.), 0.5*ufl.exp(x[0] + x[1]/2.)))
        # Define the right hand side function, h
        h = -5./4.*ufl.exp(x[0] + x[1]/2.)
    
    with df.common.Timer("Poisson Forms"):
        T_t, T_a = ufl.TestFunction(V), ufl.TrialFunction(V)
        # Get the unit vector normal to the facets
        n = ufl.FacetNormal(mesh)
        # Define the integral to be assembled into the stiffness matrix
        Si = lambda Ti_t, Ti_a: ufl.inner(ufl.grad(Ti_t), ufl.grad(Ti_a))*ufl.dx
        S = df.fem.form(Si(T_t[0], T_a[0]) + Si(T_t[1], T_a[1]))
        # Define the integral to be assembled into the forcing vector,
        # incorporating the Neumann boundary condition weakly
        fi = lambda Ti_t: Ti_t*h*ufl.dx + Ti_t*ufl.inner(gN, n)*ufl.ds
        f = df.fem.form(fi(T_t[0]) + fi(T_t[1]))

    with df.common.Timer("Poisson Assemble"):
        A = df.fem.petsc.assemble_matrix(S, bcs=bcs)
        A.assemble()

        b = df.fem.petsc.assemble_vector(f)

        # Set the boundary conditions
        df.fem.petsc.apply_lifting(b, [S], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        df.fem.petsc.set_bc(b, bcs)
    
    with df.common.Timer("Batchelor Solve"):
        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setFromOptions()

        # Compute the solution
        T_i = df.fem.Function(V)
        ksp.solve(b, T_i.x.petsc_vec)
        T_i.x.scatter_forward()

    opts.clear()

    return T_i.sub(0).collapse(), T_i.sub(1).collapse()


# %% [markdown]
# We can now numerically solve the equations using, e.g., 10 elements in each dimension and piecewise linear polynomials for pressure.

# %%
if __name__ == "__main__":
    ne = 48
    p = 1
    U = 1
    # petsc_options = {
    #                  'ksp_type':'preonly', 
    #                  'pc_type':'fieldsplit', 
    #                  'pc_fieldsplit_type': 'additive',
    #                  'ksp_converged_reason':None,
    #                  'ksp_view':None,
    #                  #'ksp_view':None,
    #                  'fieldsplit_T1_ksp_type':'cg',
    #                  'fieldsplit_T1_ksp_rtole':5.e-9,
    #                  'fieldsplit_T1_pc_type':'gamg',
    #                  #'fieldsplit_T1_pc_factor_mat_solver_type':'mumps',
    #                  'fieldsplit_T1_ksp_converged_reason':None,
    #                  'fieldsplit_T2_ksp_type':'cg',
    #                  'fieldsplit_T2_ksp_rtol':5.e-9,
    #                  'fieldsplit_T2_pc_type':'gamg',
    #                  #'fieldsplit_T2_pc_factor_mat_solver_type':'mumps',
    #                  'fieldsplit_T2_ksp_converged_reason':None,
    #                  }
    petsc_options = {
                     'ksp_type':'preonly',
                     'pc_type':'lu',
                     'pc_factor_mat_solver_type':'mumps',
                     'ksp_converged_reason':None,
    }
    petsc_options = {
                     'ksp_type':'cg',
                     'pc_type':'gamg',
                     'pc_factor_mat_solver_type':'mumps',
                     'ksp_converged_reason':None,
                     'ksp_view':None,
    }
    #petsc_options=None
    T1, T2 = solve_nested_poisson_2d(ne, p=p, petsc_options=petsc_options)
    T1.name = "T1"
    T2.name = "T2"

# %%
# plot the solution as a colormap
plotter_P1 = fenics_sz.utils.plot.plot_scalar(T1, gather=True, cmap='coolwarm')
# plot the mesh
fenics_sz.utils.plot.plot_mesh(T1.function_space.mesh, plotter=plotter_P1, gather=True, show_edges=True, style="wireframe", color='k', line_width=2)
# plot the values of the solution at the nodal points 
fenics_sz.utils.plot.plot_scalar_values(T1, plotter=plotter_P1, gather=True, point_size=15, font_size=22, shape_color='w', text_color='k', bold=False)
# show the plot
fenics_sz.utils.plot.plot_show(plotter_P1)
# save the plot
#fenics_sz.utils.plot.plot_save(plotter_P1, output_folder / "2d_nested_poisson_T1_P1_solution.png")

# %%
# plot the solution as a colormap
plotter_P1 = fenics_sz.utils.plot.plot_scalar(T2, gather=True, cmap='coolwarm')
# plot the mesh
fenics_sz.utils.plot.plot_mesh(T2.function_space.mesh, plotter=plotter_P1, gather=True, show_edges=True, style="wireframe", color='k', line_width=2)
# plot the values of the solution at the nodal points 
fenics_sz.utils.plot.plot_scalar_values(T2, plotter=plotter_P1, gather=True, point_size=15, font_size=22, shape_color='w', text_color='k', bold=False)
# show the plot
fenics_sz.utils.plot.plot_show(plotter_P1)


# %% [markdown]
# ## Testing

# %% [markdown]
# ### Error analysis

# %% [markdown]
# We can quantify the error in cases where the analytical solution is known by taking the L2 norm of the difference between the numerical and exact solutions.

# %%
# %%px
def evaluate_error(T_i):
    """
    A python function to evaluate the l2 norm of the error in 
    the two dimensional Poisson problem given a known analytical
    solution.
    """
    # Define the exact solution
    x  = ufl.SpatialCoordinate(T_i.function_space.mesh)
    Te = ufl.exp(x[0] + x[1]/2.)
    
    # Define the error between the exact solution and the given
    # approximate solution
    l2err = df.fem.assemble_scalar(df.fem.form((T_i - Te)*(T_i - Te)*ufl.dx))
    l2err = T_i.function_space.mesh.comm.allreduce(l2err, op=MPI.SUM)**0.5
    
    # Return the l2 norm of the error
    return l2err



# %% [markdown]
# ### Convergence test

# %%
# %%px
if __name__ == "__main__":
    if MPI.COMM_WORLD.rank == 0:
        # Open a figure for plotting
        fig = pl.figure()
        ax = fig.gca()
    
    petsc_options = {'ksp_type':'preonly', 
                     'pc_type':'fieldsplit', 
                     'pc_fieldsplit_type': 'additive',
                     'fieldsplit_T1_ksp_type':'preonly',
                     'fieldsplit_T1_pc_type':'lu',
                     'fieldsplit_T1_pc_factor_mat_solver_type':'mumps',
                     'fieldsplit_T2_ksp_type':'preonly',
                     'fieldsplit_T2_pc_type':'lu',
                     'fieldsplit_T1_pc_factor_mat_solver_type':'mumps',}
    petsc_options = {'ksp_type':'preonly', 
                     'pc_type':'fieldsplit', 
                     'pc_fieldsplit_type': 'additive',
                     'fieldsplit_T1_ksp_type':'cg',
                     'fieldsplit_T1_pc_type':'gamg',
                     'fieldsplit_T1_ksp_rtol':1.e-12,
                     'fieldsplit_T2_ksp_type':'cg',
                     'fieldsplit_T2_pc_type':'gamg',
                     'fieldsplit_T2_ksp_rtol':1.e-12}
    # petsc_options = {'ksp_type':'cg', 
    #                  'pc_type':'gamg',
    #                  'ksp_rtol':1.e-12}
    
    # List of polynomial orders to try
    ps = [1, 2]
    # List of resolutions to try
    nelements = [10, 20, 40, 80, 160]
    # Keep track of whether we get the expected order of convergence
    test_passes = True
    # Loop over the polynomial orders
    for p in ps:
        # Accumulate the errors
        errors_l2_1_a = []
        errors_l2_2_a = []
        # Loop over the resolutions
        for ne in nelements:
            # Solve the 2D Poisson problem
            T1_i, T2_i = solve_nested_poisson_2d(ne, p, petsc_options=petsc_options)
            # Evaluate the error in the approximate solution
            l2error1 = evaluate_error(T1_i)
            l2error2 = evaluate_error(T2_i)
            # Print to screen and save if on rank 0
            if T1_i.function_space.mesh.comm.rank == 0:
                print('ne = ', ne, ', l2error1 = ', l2error1, ', l2error2 = ', l2error2)
            errors_l2_1_a.append(l2error1)
            errors_l2_2_a.append(l2error2)
        
        # Work out the order of convergence at this p
        hs = 1./np.array(nelements)/p

        # Fit a line to the convergence data
        fit1 = np.polyfit(np.log(hs), np.log(errors_l2_1_a),1)
        fit2 = np.polyfit(np.log(hs), np.log(errors_l2_2_a),1)

        # Test if the order of convergence is as expected
        test_passes = test_passes and fit1[0] > p+0.9 and fit2[0] > p+0.9
        
        if T1_i.function_space.mesh.comm.rank == 0:
            print("***********  order of accuracy p={}, order1={:.2f}, order2={:.2f}".format(p,fit1[0],fit2[0]))
        
            # log-log plot of the error  
            ax.loglog(hs,errors_l2_1_a,'o-',label='p={}, order1={:.2f}'.format(p,fit1[0]))
            ax.loglog(hs,errors_l2_2_a,'ok--',label='p={}, order2={:.2f}'.format(p,fit2[0]))
        
    
    # Write convergence to disk
    if MPI.COMM_WORLD.rank == 0:
        # Tidy up the plot
        ax.set_xlabel('h')
        ax.set_ylabel('||e||_2')
        ax.grid()
        ax.set_title('Convergence')
        ax.legend()

        fig.savefig(output_folder / '2d_poisson_convergence.pdf')
        
        print("***********  convergence figure in output/2d_poisson_convergence.pdf")
    
    # Check if we passed the test
    assert(test_passes)

# %%
rc.shutdown()

# %%
