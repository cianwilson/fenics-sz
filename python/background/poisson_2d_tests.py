#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))
from background.poisson_2d import solve_poisson_2d
from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
import numpy as np
import ufl
import utils.plot
import matplotlib.pyplot as pl
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


def evaluate_error(T_i):
    """
    A python function to evaluate the l2 norm of the error in 
    the two dimensional Poisson problem given a known analytical
    solution.
    Parameters:
      * T_i - numerical solution
    Returns:
      * l2err - l2 norm of the error
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


def convergence_errors(ps, nelements, petsc_options=None):
    """
    A python function to evaluate the convergence errors in a two-dimensional 
    Poisson problem on a unit square domain.
    Parameters:
    * ps        - a list of polynomial orders to test
    * nelements - a list of the number of elements to test
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
            # Solve the 2D Poisson problem
            T_i = solve_poisson_2d(ne, p, petsc_options=petsc_options)
            # Evaluate the error in the approximate solution
            l2error = evaluate_error(T_i)
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
    * ps              - a list of polynomial orders to test
    * nelements       - a list of the number of elements to test
    * errors_l2       - errors_l2 from convergence_errors
    * output_basename - basename for output (defaults to no output)
    Returns:
    * test_passes     - a boolean indicating if the convergence test has passed
    """

    if MPI.COMM_WORLD.rank == 0:
        # Open a figure for plotting
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
        # Test if the order of convergence is as expected (polynomial degree plus 1)
        test_passes = test_passes and fit[0] > p+0.9
        
        # Write the errors to disk
        if MPI.COMM_WORLD.rank == 0:
            if output_basename is not None:
                with open(str(output_basename) + '_p{}.csv'.format(p), 'w') as f:
                    np.savetxt(f, np.c_[nelements, hs, errors_l2[i]], delimiter=',', 
                            header='nelements, hs, l2errs')
            
            print("order of accuracy p={}, order={:.2f}".format(p,fit[0]))

            # log-log plot of the error  
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
            
            print("***********  convergence figure in "+str(output_basename)+ ".pdf")
    
    # Return if we passed the test
    return test_passes




