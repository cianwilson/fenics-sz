#!/usr/bin/env python
# coding: utf-8

from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
from petsc4py import PETSc
import ufl
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
path = os.path.join(basedir, os.path.pardir, os.path.pardir, 'python')
sys.path.append(path)
import utils.ipp
import utils.plot
from background.batchelor import (unit_square_mesh, 
                                  functionspaces, 
                                  velocity_bcs, 
                                  pressure_bcs,
                                  stokes_weakforms,
                                  dummy_pressure_weakform,
                                  evaluate_error,
                                  test_plot_convergence)
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


def pressure_preconditioner_weakform(V_p):
    """
    A python function to return a weak form for the pressure preconditioner of 
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

        # Define the integrals to be assembled into a pressure mass matrix
        M = df.fem.form(p_t*p_a*ufl.dx)
    
    return M


def assemble_nest(S, f, bcs, M=None, attach_nullspace=False, attach_nearnullspace=True):
    """
    A python function to assemble the forms into a nested matrix and a vector.
    Parameters:
    * S   - Stokes form
    * f   - RHS form
    * bcs - list of boundary conditions
    * M   - pressure mass matrix form (defaults to None)
    * attach_nullspace - attach the pressure nullspace to the matrix (defaults to False)
    * attach_nearnullspace - attach the possible (near) velocity nullspaces to the preconditioning matrix (defaults to True)
    Returns:
    * A   - a matrix
    * B   - preconditioner matrix (None if P is None)
    * b   - a vector
    """  
    with df.common.Timer("Assemble"):
        # assemble the matrix
        A = df.fem.petsc.assemble_matrix_nest(S, bcs=bcs)
        A.assemble()
        # set a flag to indicate that the velocity block is
        # symmetric positive definite (SPD)
        A00 = A.getNestSubMatrix(0, 0)
        A00.setOption(PETSc.Mat.Option.SPD, True)

        # assemble the RHS vector
        b = df.fem.petsc.assemble_vector_nest(f)
        # apply the boundary conditions
        df.fem.petsc.apply_lifting_nest(b, S, bcs=bcs)
        # update the ghost values
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bcs_by_block = df.fem.bcs_by_block(df.fem.extract_function_spaces(f), bcs)
        df.fem.petsc.set_bc_nest(b, bcs_by_block)

        # assemble the pre-conditioner (if M was supplied)
        B = None
        if M is not None:
            BM = df.fem.petsc.assemble_matrix(M, bcs=bcs_by_block[1])
            BM.assemble()
            B = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, BM]])

            # set the SPD flag on the diagonal blocks of the preconditioner
            B00, B11 = B.getNestSubMatrix(0, 0), B.getNestSubMatrix(1, 1)
            B00.setOption(PETSc.Mat.Option.SPD, True)
            B11.setOption(PETSc.Mat.Option.SPD, True)
            
            if attach_nearnullspace:
                V_v_cpp = df.fem.extract_function_spaces(f)[0]
                
                bs = V_v_cpp.dofmap.index_map_bs
                length0 = V_v_cpp.dofmap.index_map.size_local
                ns_basis = [df.la.vector(V_v_cpp.dofmap.index_map, bs=bs, dtype=PETSc.ScalarType) for i in range(3)]
                ns_arrays = [ns_b.array for ns_b in ns_basis]
                
                dofs = [V_v_cpp.sub([i]).dofmap.map().flatten() for i in range(bs)]
                
                # Set the three translational rigid body modes
                for i in range(2):
                    ns_arrays[i][dofs[i]] = 1.0
                
                x = V_v_cpp.tabulate_dof_coordinates()
                dofs_block = V_v_cpp.dofmap.map().flatten()
                x0, x1 = x[dofs_block, 0], x[dofs_block, 1]
                ns_arrays[2][dofs[0]] = -x1
                ns_arrays[2][dofs[1]] = x0
                
                df.la.orthonormalize(ns_basis)
                
                ns_basis_petsc = [PETSc.Vec().createWithArray(ns_b[: bs * length0], bsize=bs, comm=V_v_cpp.mesh.comm) for ns_b in ns_arrays]
                nns = PETSc.NullSpace().create(vectors=ns_basis_petsc)
                B00.setNearNullSpace(nns)

        if attach_nullspace:
            # set up a null space vector indicating the null space 
            # in the pressure DOFs
            null_vec = df.fem.petsc.create_vector_nest(f)
            null_vecs = null_vec.getNestSubVecs()
            null_vecs[0].set(0.0)
            null_vecs[1].set(1.0)
            null_vec.normalize()
            nsp = PETSc.NullSpace().create(vectors=[null_vec])
            # test the null space is actually a null space
            assert(nsp.test(A))
            A.setNullSpace(nsp)

    with df.common.Timer("Cleanup"):
        if attach_nullspace: null_vec.destroy()
        if M is not None and attach_nearnullspace:
            for ns_b_p in ns_basis_petsc: ns_b_p.destroy()
            nns.destroy()
        
    return A, B, b


def solve_nest(A, b, V_v, V_p, B=None):
    """
    A python function to solve a nested matrix vector system.
    Parameters:
    * A   - matrix
    * b   - vector
    * V_v - velocity function space
    * V_p - pressure function space
    * B   - preconditioner matrix (defaults to None)
    Returns:
    * v_i - velocity solution function
    * p_i - pressure solution function
    """  

    # retrieve the petsc options
    opts = PETSc.Options()
    pc_type = opts.getString('pc_type')

    with df.common.Timer("Solve"):
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A, B)
        solver.setFromOptions()

        # a fieldsplit preconditioner allows us to precondition
        # each block of the matrix independently but we first
        # have to set the index sets (ISs) of the DOFs on which 
        # each block is defined
        if pc_type == "fieldsplit":
            iss = B.getNestISs()
            solver.getPC().setFieldSplitIS(("v", iss[0][0]), ("p", iss[0][1]))

        # Set up the solution functions
        v_i = df.fem.Function(V_v)
        p_i = df.fem.Function(V_p)

        # Create a solution vector and solve the system
        x = PETSc.Vec().createNest([v_i.x.petsc_vec, p_i.x.petsc_vec])
        solver.solve(b, x)

        # Update the ghost values
        v_i.x.scatter_forward()
        p_i.x.scatter_forward()
    
    with df.common.Timer("Cleanup"):
        solver.destroy()
        x.destroy()
        if pc_type == "fieldsplit":
            for isl in iss[0]: isl.destroy()

    return v_i, p_i


def solve_batchelor_nest(ne, p=1, U=1, petsc_options=None, attach_nullspace=False, attach_nearnullspace=True):
    """
    A python function to solve a two-dimensional corner flow 
    problem on a unit square domain.
    Parameters:
    * ne - number of elements in each dimension
    * p  - polynomial order of the pressure solution (defaults to 1)
    * U  - convergence speed of lower boundary (defaults to 1)
    * petsc_options - a dictionary of petsc options to pass to the solver 
                      (defaults to an LU direct solver using the MUMPS library)
    * attach_nullspace - flag indicating if the null space should be removed 
                         iteratively rather than using a pressure reference point
                         (defaults to False)
    * attach_nearnullspace - flag indicating if the preconditioner should be made
                             aware of the possible (near) nullspaces in the velocity
                             (defaults to True)
    Returns:
    * v_i - velocity solution function
    * p_i - pressure solution function
    """

    # 0. Set some default PETSc options
    if petsc_options is None:
        petsc_options = {"ksp_type": "preonly", \
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"}
    # and load them into the PETSc options system
    opts = PETSc.Options()
    for k, v in petsc_options.items(): opts[k] = v
    pc_type = opts.getString('pc_type')
    
    # 1. Set up a mesh
    mesh = unit_square_mesh(ne)
    # 2. Declare the appropriate function spaces
    V_v, V_p = functionspaces(mesh, p)
    # 3. Collect all the boundary conditions into a list
    bcs  = velocity_bcs(V_v, U=U)
    #    We only require the pressure bc if we're not attaching the nullspace
    if not attach_nullspace: bcs += pressure_bcs(V_p)
    # 4. Declare the weak forms
    S, f = stokes_weakforms(V_v, V_p)
    #    If not attaching the nullspace, include a dummy zero pressure mass 
    #    matrix to allow us to set a pressure constraint
    if not attach_nullspace: S[1][1] = dummy_pressure_weakform(V_p)
    #    If we're not using a direct LU method we need to set up
    #    a weak form for the pressure preconditioner block (also a 
    #    pressure mass matrix
    M = None
    if pc_type != "lu": M = pressure_preconditioner_weakform(V_p)
    # 5. Assemble the matrix equation (now using _nest)
    A, B, b = assemble_nest(S, f, bcs, M=M, attach_nullspace=attach_nullspace, attach_nearnullspace=attach_nearnullspace)
    # 6. Solve the matrix equation (now using _nest)
    v_i, p_i = solve_nest(A, b, V_v, V_p, B=B)
    
    with df.common.Timer("Cleanup"):
        A.destroy()
        if B is not None: B.destroy()
        b.destroy()

    return v_i, p_i


def convergence_errors_nest(ps, nelements, U=1, petsc_options=None, attach_nullspace=False, attach_nearnullspace=True):
    """
    A python function to run a convergence test of a two-dimensional corner flow 
    problem on a unit square domain.
    Parameters:
    * ps        - a list of pressure polynomial orders to test
    * nelements - a list of the number of elements to test
    * U         - convergence speed of lower boundary (defaults to 1)
    * petsc_options - a dictionary of petsc options to pass to the solver 
                      (defaults to an LU direct solver using the MUMPS library)
    * attach_nullspace - whether to remove the null space iteratively (defaults to False)
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
            v_i, p_i = solve_batchelor_nest(ne, p=p, U=U, 
                                            petsc_options=petsc_options,
                                            attach_nullspace=attach_nullspace,
                                            attach_nearnullspace=attach_nearnullspace)
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




