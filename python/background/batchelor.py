#!/usr/bin/env python
# coding: utf-8

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
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))
import utils.plot
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


def v_exact_batchelor(mesh, U=1):
    """
    A python function that returns the exact Batchelor velocity solution
    using UFL on the given mesh.
    Parameters:
    * mesh - the mesh on which we wish to define the coordinates for the solution
    * U    - convergence speed of lower boundary (defaults to 1)
    """
    # Define the coordinate systems
    x = ufl.SpatialCoordinate(mesh)
    theta = ufl.atan2(x[1],x[0])

    # Define the derivative to the streamfunction psi
    d_psi_d_r = -U*(-0.25*ufl.pi**2*ufl.sin(theta) \
                    +0.5*ufl.pi*theta*ufl.sin(theta) \
                    +theta*ufl.cos(theta)) \
                    /(0.25*ufl.pi**2-1)
    d_psi_d_theta_over_r = -U*(-0.25*ufl.pi**2*ufl.cos(theta) \
                               +0.5*ufl.pi*ufl.sin(theta) \
                               +0.5*ufl.pi*theta*ufl.cos(theta) \
                               +ufl.cos(theta) \
                               -theta*ufl.sin(theta)) \
                               /(0.25*ufl.pi**2-1)

    # Rotate the solution into Cartesian and return
    return ufl.as_vector([ufl.cos(theta)*d_psi_d_theta_over_r + ufl.sin(theta)*d_psi_d_r, \
                          ufl.sin(theta)*d_psi_d_theta_over_r - ufl.cos(theta)*d_psi_d_r])


def unit_square_mesh(ne):
    """
    A python function to set up a mesh of a unit square domain.
    Parameters:
    * ne   - number of elements in each dimension
    Returns:
    * mesh - the mesh
    """
    # Describe the domain (a unit square)
    # and also the tessellation of that domain into ne
    # equally spaced squared in each dimension, which are
    # subduvided into two triangular elements each
    with df.common.Timer("Mesh"):
        mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, ne, ne)
    return mesh


def functionspaces(mesh, p):
    """
    A python function to set up velocity and pressure function spaces.
    Parameters:
    * mesh - the mesh to set up the functions on
    * p    - polynomial order of the pressure solution (defaults to 1)
    Returns:
    * V_v  - velocity function space of polynomial order p+1
    * V_p  - pressure function space of polynomial order p
    """
    with df.common.Timer("Function spaces"):
        # Define velocity and pressure elements
        v_e = basix.ufl.element("Lagrange", mesh.basix_cell(), p+1, shape=(mesh.geometry.dim,))
        p_e = basix.ufl.element("Lagrange", mesh.basix_cell(), p)

        # Define the velocity and pressure function spaces
        V_v = df.fem.functionspace(mesh, v_e)
        V_p = df.fem.functionspace(mesh, p_e)
    
    return V_v, V_p


def velocity_bcs(V_v, U=1):
    """
    A python function to set up the velocity boundary conditions.
    Parameters:
    * V_v - velocity function space
    * U   - convergence speed of lower boundary (defaults to 1)
    Returns:
    * bcs - a list of boundary conditions
    """

    with df.common.Timer("Dirichlet BCs"):
        # Declare a list of boundary conditions
        bcs = []
        
        # Grab the mesh
        mesh = V_v.mesh

        # Define the location of the left boundary and find the velocity DOFs
        def boundary_left(x):
            return np.isclose(x[0], 0)
        dofs_v_left = df.fem.locate_dofs_geometrical(V_v, boundary_left)
        # Specify the velocity value and define a Dirichlet boundary condition
        zero_v = df.fem.Constant(mesh, df.default_scalar_type((0.0, 0.0)))
        bcs.append(df.fem.dirichletbc(zero_v, dofs_v_left, V_v))

        # Define the location of the bottom boundary and find the velocity DOFs
        # for x velocity (0) and y velocity (1) separately
        def boundary_base(x):
            return np.isclose(x[1], 0)
        dofs_v_base = df.fem.locate_dofs_geometrical(V_v, boundary_base)
        # Specify the value of the x component of velocity and define a Dirichlet boundary condition
        U_v = df.fem.Constant(mesh, df.default_scalar_type((U, 0.0)))
        bcs.append(df.fem.dirichletbc(U_v, dofs_v_base, V_v))

        # Define the location of the right and top boundaries and find the velocity DOFs
        def boundary_rightandtop(x):
            return np.logical_or(np.isclose(x[0], 1), np.isclose(x[1], 1))
        dofs_v_rightandtop = df.fem.locate_dofs_geometrical(V_v, boundary_rightandtop)
        # Specify the exact velocity value and define a Dirichlet boundary condition
        exact_v = df.fem.Function(V_v)
        # Interpolate from a UFL expression, evaluated at the velocity interpolation points
        exact_v.interpolate(df.fem.Expression(v_exact_batchelor(mesh, U=U), V_v.element.interpolation_points()))
        bcs.append(df.fem.dirichletbc(exact_v, dofs_v_rightandtop))
    
    return bcs

def pressure_bcs(V_p):
    """
    A python function to set up the pressure boundary conditions.
    Parameters:
    * V_p - pressure function space
    Returns:
    * bcs - a list of boundary conditions
    """

    with df.common.Timer("Dirichlet BCs"):
        # Declare a list of boundary conditions
        bcs = []

        # Grab the mesh
        mesh = V_p.mesh

        # Define the location of the lower left corner of the domain and find the pressure DOF there
        def corner_lowerleft(x):
            return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))
        dofs_p_lowerleft = df.fem.locate_dofs_geometrical(V_p, corner_lowerleft)
        # Specify the arbitrary pressure value and define a Dirichlet boundary condition
        zero_p = df.fem.Constant(mesh, df.default_scalar_type(0.0))
        bcs.append(df.fem.dirichletbc(zero_p, dofs_p_lowerleft, V_p))
    
    return bcs


def stokes_weakforms(V_v, V_p):  
    """
    A python function to return the weak forms of the Stokes problem.
    Parameters:
    * V_v - velocity function space
    * V_p - pressure function space
    Returns:
    * S   - a bilinear form
    * f   - a linear form
    """  
    with df.common.Timer("Forms"):
        # Grab the mesh
        mesh = V_p.mesh

        # Define the trial functions for velocity and pressure
        v_a, p_a = ufl.TrialFunction(V_v), ufl.TrialFunction(V_p)
        # Define the test functions for velocity and pressure
        v_t, p_t = ufl.TestFunction(V_v),  ufl.TestFunction(V_p)

        # Define the integrals to be assembled into the stiffness matrix
        K = ufl.inner(ufl.sym(ufl.grad(v_t)), ufl.sym(ufl.grad(v_a))) * ufl.dx
        G = -ufl.div(v_t)*p_a*ufl.dx
        D = -p_t*ufl.div(v_a)*ufl.dx
        S = df.fem.form([[K, G], [D, None]])
        # S = df.fem.form([[K, G], [D, p_t*p_a*zero_p*ufl.dx]])

        # Define the integral to the assembled into the forcing vector
        # which in this case is just zero
        zero_p = df.fem.Constant(mesh, df.default_scalar_type(0.0))
        zero_v = df.fem.Constant(mesh, df.default_scalar_type((0.0, 0.0)))
        f = df.fem.form([ufl.inner(v_t, zero_v)*ufl.dx, zero_p*p_t*ufl.dx])
    
    return S, f

def pressure_weakforms(V_p):
    """
    A python function to return a dummy (zero) weak form for the pressure block of 
    the Stokes problem.
    Parameters:
    * V_p - pressure function space
    Returns:
    * M   - a bilinear form
    """  
    with df.common.Timer("Forms"):
        # Grab the mesh
        mesh = V_p.mesh

        # Define the trial function for the pressure
        p_a = ufl.TrialFunction(V_p)
        # Define the test function for the pressure
        p_t = ufl.TestFunction(V_p)

        # Define the dummy integrals to be assembled into a zero pressure mass matrix
        zero_p = df.fem.Constant(mesh, df.default_scalar_type(0.0))
        M = df.fem.form(p_t*p_a*zero_p*ufl.dx)
    
    return M


def assemble(S, f, bcs):
    """
    A python function to assemble the forms into a matrix and a vector.
    Parameters:
    * S   - Stokes form
    * f   - RHS form
    * bcs - list of boundary conditions
    Returns:
    * A   - a matrix
    * b   - a vector
    """  
    with df.common.Timer("Assemble"):
        A = df.fem.petsc.assemble_matrix_block(S, bcs=bcs)
        A.assemble()
        b = df.fem.petsc.assemble_vector_block(f, S, bcs=bcs)
    return A, b


def solve(A, b, V_v, V_p, petsc_options=None):
    """
    A python function to solve a matrix vector system.
    Parameters:
    * A   - matrix
    * b   - vector
    * V_v - velocity function space
    * V_p - pressure function space
    * petsc_options - a dictionary of petsc options to pass to the solver 
                      (defaults to an LU direct solver using the MUMPS library)
    Returns:
    * v_i - velocity solution function
    * p_i - pressure solution function
    """  

    if petsc_options is None:
        petsc_options = {"ksp_type": "preonly", \
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"}
    
    opts = PETSc.Options()
    for k, v in petsc_options.items(): opts[k] = v

    with df.common.Timer("Solve"):
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A)
        solver.setFromOptions()

        x = A.createVecRight()
        solver.solve(b, x)

        # Set up the solution functions
        v_i = df.fem.Function(V_v)
        p_i = df.fem.Function(V_p)
        # Extract the velocity and pressure solutions for the coupled problem
        offset = V_v.dofmap.index_map.size_local*V_v.dofmap.index_map_bs
        v_i.x.array[:offset] = x.array_r[:offset]
        p_i.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
        v_i.x.scatter_forward()
        p_i.x.scatter_forward()
    
    with df.common.Timer("Cleanup"):
        solver.destroy()
        x.destroy()

    return v_i, p_i


def solve_batchelor(ne, p=1, U=1, petsc_options=None):
    """
    A python function to solve a two-dimensional corner flow 
    problem on a unit square domain.
    Parameters:
    * ne - number of elements in each dimension
    * p  - polynomial order of the pressure solution (defaults to 1)
    * U  - convergence speed of lower boundary (defaults to 1)
    * petsc_options - a dictionary of petsc options to pass to the solver 
                      (defaults to an LU direct solver using the MUMPS library)
    Returns:
    * v_i - velocity solution function
    * p_i - pressure solution function
    """

    # 1. Set up a mesh
    mesh = unit_square_mesh(ne)
    # 2. Declare the appropriate function spaces
    V_v, V_p = functionspaces(mesh, p)
    # 3. Collect all the boundary conditions into a list
    bcs  = velocity_bcs(V_v, U=U)
    bcs += pressure_bcs(V_p)
    # 4. Declare the weak forms
    S, f = stokes_weakforms(V_v, V_p)
    # 5. Include a dummy zero pressure mass matrix to allow us to set a pressure constraint
    S[1][1] = pressure_weakforms(V_p)
    # 6. Assemble the matrix equation
    A, b = assemble(S, f, bcs)
    # 7. Solve the matrix equation using the supplied petsc_options
    v_i, p_i = solve(A, b, V_v, V_p, petsc_options=petsc_options)
    
    with df.common.Timer("Cleanup"):
        A.destroy()
        b.destroy()

    return v_i, p_i


def evaluate_error(v_i, U=1):
    """
    A python function to evaluate the l2 norm of the error in 
    the two dimensional Batchelor corner flow problem.
    Parameters:
    * v_i - numerical solution for velocity
    * U   - convergence speed of lower boundary (defaults to 1)
    Returns:
    * l2err - l2 error of solution
    """
    # Define the exact solution (in UFL)
    ve = v_exact_batchelor(v_i.function_space.mesh, U=U)

    # Define the error as the squared difference between the exact solution and the given approximate solution
    l2err = df.fem.assemble_scalar(df.fem.form(ufl.inner(v_i - ve, v_i - ve)*ufl.dx))
    l2err = v_i.function_space.mesh.comm.allreduce(l2err, op=MPI.SUM)**0.5

    # Return the l2 norm of the error
    return l2err


def convergence_errors(ps, nelements, U=1, petsc_options=None):
    """
    A python function to run a convergence test of a two-dimensional corner flow 
    problem on a unit square domain.
    Parameters:
    * ps        - a list of pressure polynomial orders to test
    * nelements - a list of the number of elements to test
    * U         - convergence speed of lower boundary (defaults to 1)
    * petsc_options - a dictionary of petsc options to pass to the solver 
                      (defaults to an LU direct solver using the MUMPS library)
    Returns:
    * errors_l2 - a list of l2 errors
    """
    errors_l2 = []
    # Loop over the polynomial orders
    for p in ps:
        # Accumulate the errors
        errors_l2_p = []
        # Loop over the resolutions
        for ne in nelements:
            # Solve the 2D Batchelor corner flow problem
            v_i, p_i = solve_batchelor(ne, p=p, U=U, 
                                       petsc_options=petsc_options)
            # Evaluate the error in the approximate solution
            l2error = evaluate_error(v_i, U=U)
            # Print to screen and save if on rank 0
            if MPI.COMM_WORLD.rank == 0:
                print('p={}, ne={}, l2error={}'.format(p, ne, l2error))
            errors_l2_p.append(l2error)
        if MPI.COMM_WORLD.rank == 0:
            print('*************************************************')
        errors_l2.append(errors_l2_p)
    
    return errors_l2


def test_plot_convergence(ps, nelements, errors_l2, output_basename=None):
    """
    A python function to test and plot convergence of the given errors.
    Parameters:
    * ps              - a list of pressure polynomial orders to test
    * nelements       - a list of the number of elements to test
    * errors_l2       - errors_l2 from convergence_errors
    * output_basename - basename for output (defaults to no output)
    Returns:
    * test_passes     - a boolean indicating if the convergence test has passed
    """

    # Open a figure for plotting
    if MPI.COMM_WORLD.rank == 0:
        fig = pl.figure()
        ax = fig.gca()

    # Keep track of whether we get the expected order of convergence
    test_passes = True

    # Loop over the polynomial orders
    for i, p in enumerate(ps):
        # Work out the order of convergence at this p
        hs = 1./np.array(nelements)/p
        # Fit a line to the convergence data
        fit = np.polyfit(np.log(hs), np.log(errors_l2[i]),1)
        # Test if the order of convergence is as expected (first order)
        test_passes = test_passes and abs(fit[0]-1) < 0.1

        # Write the errors to disk
        if MPI.COMM_WORLD.rank == 0:
            if output_basename is not None:
                with open(str(output_basename) + '_p{}.csv'.format(p), 'w') as f:
                    np.savetxt(f, np.c_[nelements, hs, errors_l2[i]], delimiter=',', 
                             header='nelements, hs, l2errs')
            
            print("order of accuracy p={}, order={}".format(p,fit[0]))
        
            # log-log plot of the L2 error 
            ax.loglog(hs,errors_l2[i],'o-',label='p={}, order={:.2f}'.format(p,fit[0]))
        
    if MPI.COMM_WORLD.rank == 0:
        # Tidy up the plot
        ax.set_xlabel(r'$h$')
        ax.set_ylabel(r'$||e||_2$')
        ax.grid()
        ax.set_title('Convergence')
        ax.legend()

        # Write convergence to disk
        if output_basename is not None:
            fig.savefig(str(output_basename) + '.pdf')

            print("***********  convergence figure in "+str(output_basename)+".pdf")
    
    return test_passes




