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


from poisson_2d import solve_poisson_2d
from mpi4py import MPI
import numpy as np
import ufl
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, 'python'))
import utils
import matplotlib.pyplot as pl
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
if __name__ == "__main__":
    output_folder = pathlib.Path(os.path.join(basedir, "output"))
    output_folder.mkdir(exist_ok=True, parents=True)
import time
import dolfinx as df


# In[ ]:


print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)


# In[ ]:


if __name__ == "__main__":
    ne = 640
    p = 2
    # Solve the 2D Poisson problem
    start_time = time.time()
    T_i = solve_poisson_2d(ne, p)
    end_time = time.time()
    comm = T_i.function_space.mesh.comm
    print(f"{comm.rank} ({T_i.function_space.dofmap.index_map.size_local}) - time taken: {end_time - start_time} seconds")


# In[ ]:


if __name__ == "__main__":
    # plot the solution as a colormap
    plotter = utils.plot_scalar(T_i, gather=True)
    # save the plot
    utils.plot_save(plotter, output_folder / "2d_poisson_test_single_solution_ne{:d}.png".format(ne,))
    comm = T_i.function_space.mesh.comm
    if comm.size > 1:
        # if we're running in parallel (e.g. from a script) then save an image per process as well
        plotter_p = utils.plot_scalar(T_i)
        utils.plot_save(plotter_p, output_folder / "2d_poisson_test_single_solution_ne{:d}_p{:d}.png".format(ne,comm.rank,))


# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True poisson_2d_test_single.ipynb')


# In[ ]:




