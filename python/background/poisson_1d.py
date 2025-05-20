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
import utils
import pathlib
if __name__ == "__main__":
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

    # Define the trial and test functions on the same function space (V)
    T_a = ufl.TrialFunction(V)
    T_t = ufl.TestFunction(V)

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


if __name__ == "__main__":
    ne = 4
    p = 1
    T_P1 = solve_poisson_1d(ne, p=p)
    T_P1.name = "T (P1)"


def plot_1d(T, x, filename=None):
    nx = len(x)
    # convert 1d points to 3d points (necessary for eval call)
    xyz = np.stack((x, np.zeros_like(x), np.zeros_like(x)), axis=1)
    # work out which cells those points are in
    mesh = T.function_space.mesh
    cinds, cells = utils.get_cell_collisions(xyz, mesh)
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
        ax.plot(x, T_x, label=T.name)           # numerical solution
        ax.plot(x[::int(nx/ne/p)], T_x[::int(nx/ne/p)], 'o') # nodal points (uses globally defined ne and p)
        ax.plot(x, np.sin(np.pi*x/2), '--g', label='T (exact)')    # analytical solution
        ax.legend()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$T$')
        ax.set_title('Numerical and exact solutions')
        # save the figure
        if filename is not None: fig.savefig(filename)


if __name__ == "__main__":
    x = np.linspace(0, 1, 201)
    plot_1d(T_P1, x, filename=output_folder / '1d_poisson_P1_solution.pdf')


if __name__ == "__main__":
    ne = 4
    p = 2
    T_P2 = solve_poisson_1d(ne, p=p)
    T_P2.name = "T (P2)"


if __name__ == "__main__":
    x = np.linspace(0, 1, 201)
    plot_1d(T_P2, x, filename=output_folder / '1d_poisson_P2_solution.pdf')


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


if __name__ == "__main__":
    # Open a figure for plotting
    fig = pl.figure()
    ax = fig.gca()
    
    # List of polynomial orders to try
    ps = [1, 2]
    # List of resolutions to try
    nelements = [10, 20, 40, 80, 160, 320]
    # Keep track of whether we get the expected order of convergence
    test_passes = True
    # Loop over the polynomial orders
    for p in ps:
        # Accumulate the errors
        errors_l2_a = []
        # Loop over the resolutions
        for ne in nelements:
            # Solve the 1D Poisson problem
            T_i = solve_poisson_1d(ne, p)
            # Evaluate the error in the approximate solution
            l2error = evaluate_error(T_i)
            # Print to screen and save if on rank 0
            if T_i.function_space.mesh.comm.rank == 0:
                print('ne = ', ne, ', l2error = ', l2error)
            errors_l2_a.append(l2error)
    
        # Work out the order of convergence at this p
        hs = 1./np.array(nelements)/p
        
        # Write the errors to disk
        if T_i.function_space.mesh.comm.rank == 0:
            with open(output_folder / '1d_poisson_convergence_p{}.csv'.format(p), 'w') as f:
                np.savetxt(f, np.c_[nelements, hs, errors_l2_a], delimiter=',', 
                           header='nelements, hs, l2errs')
            
        # Fit a line to the convergence data
        fit = np.polyfit(np.log(hs), np.log(errors_l2_a),1)
        
        if T_i.function_space.mesh.comm.rank == 0:
            print("***********  order of accuracy p={}, order={:.2f}".format(p,fit[0]))
        
        # log-log plot of the error  
        ax.loglog(hs,errors_l2_a,'o-',label='p={}, order={:.2f}'.format(p,fit[0]))
        
        # Test if the order of convergence is as expected
        test_passes = test_passes and fit[0] > p+0.9
    
    # Tidy up the plot
    ax.set_xlabel('h')
    ax.set_ylabel('||e||_2')
    ax.grid()
    ax.set_title('Convergence')
    ax.legend()
    
    # Write convergence to disk
    if T_i.function_space.mesh.comm.rank == 0:
        fig.savefig(output_folder / '1d_poisson_convergence.pdf')
        
        print("***********  convergence figure in output/1d_poisson_convergence.pdf")
    
    # Check if we passed the test
    assert(test_passes)


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/background --NbConvertApp.output_base=poisson_1d 2.2b_poisson_1d.ipynb')




