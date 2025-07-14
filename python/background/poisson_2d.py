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
import utils.plot
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


def solve_poisson_2d(ne, p=1, petsc_options=None):
    """
    A python function to solve a two-dimensional Poisson problem
    on a unit square domain.
    Parameters:
    * ne - number of elements in each dimension
    * p  - polynomial order of the solution function space
    * petsc_options - a dictionary of petsc options to pass to the solver 
                      (defaults to an LU direct solver using the MUMPS library)
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
    with df.common.Timer("Mesh"):
        mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, ne, ne, ghost_mode=df.mesh.GhostMode.none)

    # Define the solution function space using Lagrange polynomials
    # of order p
    with df.common.Timer("Function spaces"):
        V = df.fem.functionspace(mesh, ("Lagrange", p))

    with df.common.Timer("Dirichlet BCs"):
        # Define the location of the boundary condition, x=0 and y=0
        def boundary(x):
            return np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0))
        boundary_dofs = df.fem.locate_dofs_geometrical(V, boundary)
        # Specify the value and define a Dirichlet boundary condition (bc)
        gD = df.fem.Function(V)
        gD.interpolate(lambda x: np.exp(x[0] + x[1]/2.))
        bc = df.fem.dirichletbc(gD, boundary_dofs)

    with df.common.Timer("Neumann BCs"):
        # Get the coordinates
        x = ufl.SpatialCoordinate(mesh)
        # Define the Neumann boundary condition function
        gN = ufl.as_vector((ufl.exp(x[0] + x[1]/2.), 0.5*ufl.exp(x[0] + x[1]/2.)))
        # Define the right hand side function, h
        h = -5./4.*ufl.exp(x[0] + x[1]/2.)

    with df.common.Timer("Forms"):
        T_a = ufl.TrialFunction(V)
        T_t = ufl.TestFunction(V)
        # Get the unit vector normal to the facets
        n = ufl.FacetNormal(mesh)
        # Define the integral to be assembled into the stiffness matrix
        S = df.fem.form(ufl.inner(ufl.grad(T_t), ufl.grad(T_a))*ufl.dx)
        # Define the integral to be assembled into the forcing vector,
        # incorporating the Neumann boundary condition weakly
        f = df.fem.form(T_t*h*ufl.dx + T_t*ufl.inner(gN, n)*ufl.ds)

    # The next two sections "Assemble" and "Solve"
    # are the equivalent of the much simpler:
    # ```
    # problem = df.fem.petsc.LinearProblem(S, f, bcs=[bc], \
    #                                      petsc_options=petsc_options)
    # T_i = problem.solve()
    # ```
    # We split them up here so we can time and profile each step separately.
    with df.common.Timer("Assemble"):
        # Assemble the matrix from the S form
        Sm = df.fem.petsc.assemble_matrix(S, bcs=[bc])
        Sm.assemble()
        # Assemble the R.H.S. vector from the f form
        fm = df.fem.petsc.assemble_vector(f)

        # Set the boundary conditions
        df.fem.petsc.apply_lifting(fm, [S], bcs=[[bc]])
        fm.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        df.fem.petsc.set_bc(fm, [bc])

    with df.common.Timer("Solve"):
        # Setup the solver (a PETSc KSP object)
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(Sm)
        solver.setFromOptions()
        
        # Set up the solution function
        T_i = df.fem.Function(V)
        # Call the solver
        solver.solve(fm, T_i.x.petsc_vec)
        # Communicate the solution across processes
        T_i.x.scatter_forward()

    with df.common.Timer("Cleanup"):
        solver.destroy()
        Sm.destroy()
        fm.destroy()

    return T_i




