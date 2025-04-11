#!/usr/bin/env python
# coding: utf-8

# # Poisson Example 2D
# 
# Authors: Kidus Teshome, Cameron Seebeck, Cian Wilson

# ## Description

# As a reminder, in this case we are seeking the approximate solution to
# \begin{equation}
# - \nabla^2 T = -\tfrac{5}{4} \exp \left( x+\tfrac{y}{2} \right)
# \end{equation}
# in a unit square, $\Omega=[0,1]\times[0,1]$, imposing the boundary conditions
# \begin{align}
#   T &= \exp\left(x+\tfrac{y}{2}\right) && \text{on } \partial\Omega \text{ where } x=0 \text{ or } y=0 \\
#   \nabla T\cdot \hat{\vec{n}} &= \exp\left(x + \tfrac{y}{2}\right) && \text{on } \partial\Omega \text{ where } x=1  \\
#   \nabla T\cdot \hat{\vec{n}} &= \tfrac{1}{2}\exp\left(x + \tfrac{y}{2}\right) && \text{on } \partial\Omega \text{ where } y=1
#  \end{align}
# 
# The analytical solution to this problem is $T(x,y) = \exp\left(x+\tfrac{y}{2}\right)$.

# ## Themes and variations

# * Given that we know the exact solution to this problem is $T(x,y)$=$\exp\left(x+\tfrac{y}{2}\right)$ write a python function to evaluate the error in our numerical solution.
# * Loop over a variety of numbers of elements, `ne`, and polynomial degrees, `p`, and check that the numerical solution converges with an increasing number of degrees of freedom.
# * Write an equation for the gradient of $\tilde{T}$, describe it using UFL, solve it, and plot the solution.

# ### Preamble

# Start by loading `solve_poisson_2d` from `notebooks/poisson_2d.ipynb` and setting up some paths.

# In[ ]:


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
import utils
import matplotlib.pyplot as pl
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
if __name__ == "__main__":
    output_folder = pathlib.Path(os.path.join(basedir, "output"))
    output_folder.mkdir(exist_ok=True, parents=True)


# ### Error analysis

# We can quantify the error in cases where the analytical solution is known by taking the L2 norm of the difference between the numerical and (known) exact solutions.

# In[ ]:


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


# ### Convergence test

# Repeating the numerical experiments with increasing `ne` allows us to test the **convergence** of our approximate finite element solution to the known analytical solution.  A key feature of any discretization technique is that with an increasing number of degrees of freedom (DOFs) these solutions should converge, i.e. the error in our approximation should decrease.

# In[ ]:


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
            # Solve the 2D Poisson problem
            T_i = solve_poisson_2d(ne, p)
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
            with open(output_folder / '2d_poisson_convergence_p{}.csv'.format(p), 'w') as f:
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
        fig.savefig(output_folder / '2d_poisson_convergence.pdf')
        
        print("***********  convergence figure in output/2d_poisson_convergence.pdf")
    
    # Check if we passed the test
    assert(test_passes)


# The convergence tests show that we achieve the expected orders of convergence for all polynomial degrees tested.

# ### Gradient

# To find the gradient of the approximate solution $\tilde{T}$ we seek the approximate solution to
# \begin{equation}
#   \vec{g} = \nabla \tilde{T}
# \end{equation}
# where $\vec{g}$ is the gradient solution we seek in the domain $\Omega=[0,1]\times[0,1]$.  This is a projection operation and no boundary conditions are required.
# 
# We proceed as before
# 1. we solve for $\tilde{T}$ using elements with polynomial degree `p` on a mesh of $2 \times$ `ne` $\times$ `ne` triangular elements or cells
# 2. we reuse the mesh to declare a _vector_ **function space** for $\vec{g} \approx \tilde{\vec{g}}$, `Vg`, to use Lagrange polynomials of degree `pg`
# 3. using this function space we declare trial, `g_a`, and test, `g_t`, functions
# 4. we define the right hand side using the gradient of $\tilde{T}$
# 5. we describe the **discrete weak forms**, `Sg` and `fg`, that will be used to assemble the matrix $\mathbf{S}_g$ and vector $\mathbf{f}_g$
# 6. we solve the matrix problem using a linear algebra back-end and return the solution

# In[ ]:


if __name__ == "__main__":
    # solve for T
    ne = 10
    p = 1
    T = solve_poisson_2d(ne, p)
    T.name = 'T'

    # reuse the mesh from T
    mesh = T.function_space.mesh

    # define the function space for g to be of polynomial degree pg and a vector of length mesh.geometry.dim
    pg = 2
    Vg = df.fem.functionspace(mesh, ("Lagrange", pg, (mesh.geometry.dim,)))

    # define trial and test functions using Vg
    g_a = ufl.TrialFunction(Vg)
    g_t = ufl.TestFunction(Vg)

    # define the bilinear and linear forms, Sg and fg
    Sg = ufl.inner(g_t, g_a) * ufl.dx
    fg = ufl.inner(g_t, ufl.grad(T)) * ufl.dx

    # assemble the problem and solve
    problem = df.fem.petsc.LinearProblem(Sg, fg, bcs=[], 
                                         petsc_options={"ksp_type": "preonly", 
                                                        "pc_type": "lu", 
                                                        "pc_factor_mat_solver_type": "mumps"})
    gh = problem.solve()
    gh.name = "grad(T)"


# We can then plot the solutions.

# In[ ]:


if __name__ == "__main__":
    # plot T as a colormap
    plotter_g = utils.plot_scalar(T, gather=True)
    # plot g as glyphs
    utils.plot_vector_glyphs(gh, plotter=plotter_g, gather=True, factor=0.03, cmap='coolwarm')
    utils.plot_show(plotter_g)
    utils.plot_save(plotter_g, output_folder / "2d_poisson_gradient.png")
    comm = mesh.comm
    if comm.size > 1:
        # if we're running in parallel (e.g. from a script) then save an image per process as well
        plotter_g_p = utils.plot_scalar(T)
        # plot g as glyphs
        utils.plot_vector_glyphs(gh, plotter=plotter_g_p, factor=0.03, cmap='coolwarm')
        utils.plot_save(plotter_g_p, output_folder / "2d_poisson_gradient_p{:d}.png".format(comm.rank,))


# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/background --NbConvertApp.output_base=poisson_2d_tests 2.3c_poisson_2d_tests.ipynb')


# In[ ]:




