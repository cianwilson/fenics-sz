#!/usr/bin/env python
# coding: utf-8

from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
import numpy as np
import ufl
import matplotlib.pyplot as pl
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))
import utils.mesh
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


def solve_poisson_1d(ne, p=1):
    """
    A python function to solve a one-dimensional Poisson problem
    on a unit interval domain.
    Parameters:
      * ne - number of elements
      * p  - polynomial order of the solution function space
    """
    
    # Describe the domain (a one-dimensional unit interval)
    # and also the tessellation of that domain into ne 
    # equally spaced elements
    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, ne)

    # Define the solution function space using Lagrange polynomials
    # of order p
    V = df.fem.functionspace(mesh, ("Lagrange", p))

    # Define the location of the boundary, x=0
    def boundary(x):
        return np.isclose(x[0], 0)
    # Specify the value and define a boundary condition (bc)
    boundary_dofs = df.fem.locate_dofs_geometrical(V, boundary)
    gD = df.fem.Constant(mesh, df.default_scalar_type(0.0))
    bc = df.fem.dirichletbc(gD, boundary_dofs, V)

    # Define the right hand side function, h
    x = ufl.SpatialCoordinate(mesh)
    h = (ufl.pi**2)*ufl.sin(ufl.pi*x[0]/2)/4

    # Define the trial and test functions on the same function space (V)
    T_a = ufl.TrialFunction(V)
    T_t = ufl.TestFunction(V)
    
    # Define the integral to be assembled into the stiffness matrix
    S = ufl.inner(ufl.grad(T_t), ufl.grad(T_a))*ufl.dx
    # Define the integral to be assembled into the forcing vector
    f = T_t*h*ufl.dx

    # Compute the solution (given the boundary condition, bc)
    problem = df.fem.petsc.LinearProblem(S, f, bcs=[bc], \
                                         petsc_options={"ksp_type": "preonly", \
                                                        "pc_type": "lu",
                                                        "pc_factor_mat_solver_type": "mumps"})
    T_i = problem.solve()

    # Return the solution
    return T_i


def plot_1d(T, x, filename=None):
    nx = len(x)
    # convert 1d points to 3d points (necessary for eval call)
    xyz = np.stack((x, np.zeros_like(x), np.zeros_like(x)), axis=1)
    # work out which cells those points are in using a utility function we provide
    mesh = T.function_space.mesh
    cinds, cells = utils.mesh.get_cell_collisions(xyz, mesh)
    # evaluate the numerical solution
    T_x = T.eval(xyz[cinds], cells)[:,0]
    # if running in parallel gather the solution to the rank-0 process
    cinds_g = mesh.comm.gather(cinds, root=0)
    T_x_g = mesh.comm.gather(T_x, root=0)
    # only plot on the rank-0 process
    if mesh.comm.rank == 0:
        T_x = np.empty_like(x)
        for r, cinds_p in enumerate(cinds_g):
            for i, cind in enumerate(cinds_p):
                T_x[cind] = T_x_g[r][i]
        # plot
        fig = pl.figure()
        ax = fig.gca()
        ax.plot(x, T_x, label=T.name)                              # numerical solution
        ax.plot(x[::int(nx/ne/p)], T_x[::int(nx/ne/p)], 'o')       # nodal points (uses globally defined ne and p)
        ax.plot(x, np.sin(np.pi*x/2), '--g', label='T (exact)')    # analytical solution
        ax.legend()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$T$')
        ax.set_title('Numerical and exact solutions')
        # save the figure
        if filename is not None: fig.savefig(filename)


def evaluate_error(T_i):
    """
    A python function to evaluate the l2 norm of the error in 
    the one dimensional Poisson problem given a known analytical
    solution.
    """
    # Define the exact solution
    x  = ufl.SpatialCoordinate(T_i.function_space.mesh)
    Te = ufl.sin(ufl.pi*x[0]/2)
    
    # Define the error between the exact solution and the given
    # approximate solution
    l2err = df.fem.assemble_scalar(df.fem.form((T_i - Te)*(T_i - Te)*ufl.dx))
    l2err = T_i.function_space.mesh.comm.allreduce(l2err, op=MPI.SUM)**0.5
    
    # Return the l2 norm of the error
    return l2err




