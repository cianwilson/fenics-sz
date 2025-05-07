#!/usr/bin/env python
# coding: utf-8

# # Subduction Zone Benchmark
# 
# Authors: Kidus Teshome, Cian Wilson

# ## Convergence testing

# ### Preamble

# Let's start by adding the path to the modules in the `python` folder to the system path (so we can find the our custom modules).

# In[ ]:


import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


# Then load everything we need from `sz_problem` and also set our default plotting preferences.

# In[ ]:


import utils
from sz_problems.sz_base import default_params, allsz_params
from sz_problems.sz_slab import create_slab
from sz_problems.sz_geometry import create_sz_geometry
from sz_problems.sz_problem import SubductionProblem, plot_slab_temperatures
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
if __name__ == "__main__":
    output_folder = pathlib.Path(os.path.join(basedir, "output"))
    output_folder.mkdir(exist_ok=True, parents=True)


# ### Benchmark case 1

# In[ ]:


def solve_benchmark_case1(resscale):
    xs = [0.0, 140.0, 240.0, 400.0]
    ys = [0.0, -70.0, -120.0, -200.0]
    lc_depth = 40
    uc_depth = 15
    coast_distance = 0
    extra_width = 0
    sztype = 'continental'
    io_depth_1 = 139
    A      = 100.0      # age of subducting slab (Myr)
    qs     = 0.065      # surface heat flux (W/m^2)
    Vs     = 100.0      # slab speed (mm/yr)

    # create the slab
    slab = create_slab(xs, ys, resscale, lc_depth)
    # construct the geometry
    geom = create_sz_geometry(slab, resscale, sztype, io_depth_1, extra_width, 
                               coast_distance, lc_depth, uc_depth)
    # set up a subduction zone problem
    sz = SubductionProblem(geom, A=A, Vs=Vs, sztype=sztype, qs=qs)

    # solve it using a steady state assumption and an isoviscous rheology
    sz.solve_steadystate_isoviscous()

    return sz


# In[ ]:


if __name__ == "__main__":
    resscales = [4.0, 2.0, 1.0]
    diagnostics_case1 = []
    for resscale in resscales:
        sz = solve_benchmark_case1(resscale)
        diagnostics_case1.append((resscale, sz.get_diagnostics()))


# In[ ]:


if __name__ == "__main__":
    print('')
    print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
    for resscale, diag in diagnostics_case1:
        print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale, *diag))    


# For comparison here are the values reported for case 1 using [TerraFERMA](https://terraferma.github.io) in [Wilson & van Keken, 2023](http://dx.doi.org/10.1186/s40645-023-00588-6):
# 
# | `resscale` | $T_{\text{ndof}} $ | $T_{(200,-100)}^*$ | $\overline{T}_s^*$ | $ \overline{T}_w^* $ |  $V_{\text{rms},w}^*$ |
# | - | - | - | - | - | - |
# | 2.0 | 21403  | 517.17 | 451.83 | 926.62 | 34.64 |
# | 1.0 | 83935  | 516.95 | 451.71 | 926.33 | 34.64 |
# | 0.5 | 332307 | 516.86 | 451.63 | 926.15 | 34.64 |

# ### Benchmark case 2

# In[ ]:


def solve_benchmark_case2(resscale):
    xs = [0.0, 140.0, 240.0, 400.0]
    ys = [0.0, -70.0, -120.0, -200.0]
    lc_depth = 40
    uc_depth = 15
    coast_distance = 0
    extra_width = 0
    sztype = 'continental'
    io_depth_2 = 154
    A      = 100.0      # age of subducting slab (Myr)
    qs     = 0.065      # surface heat flux (W/m^2)
    Vs     = 100.0      # slab speed (mm/yr)

    # create the slab
    slab = create_slab(xs, ys, resscale, lc_depth)
    # construct the geometry
    geom = create_sz_geometry(slab, resscale, sztype, io_depth_2, extra_width, 
                               coast_distance, lc_depth, uc_depth)
    # set up a subduction zone problem
    sz = SubductionProblem(geom, A=A, Vs=Vs, sztype=sztype, qs=qs)

    # solve it using a steady state assumption and a dislocation creep rheology
    sz.solve_steadystate_dislocationcreep()

    return sz


# In[ ]:


if __name__ == "__main__":
    resscales = [4.0, 2.0, 1.0]
    diagnostics_case2 = []
    for resscale in resscales:
        sz = solve_benchmark_case2(resscale)
        diagnostics_case2.append((resscale, sz.get_diagnostics()))


# In[ ]:


if __name__ == "__main__":
    print('')
    print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
    for resscale, diag in diagnostics_case2:
        print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale, *diag))    


# For comparison here are the values reported for case 2 using [TerraFERMA](https://terraferma.github.io) in [Wilson & van Keken, 2023](http://dx.doi.org/10.1186/s40645-023-00588-6):
# 
# | `resscale` | $T_{\text{ndof}} $ | $T_{(200,-100)}^*$ | $\overline{T}_s^*$ | $ \overline{T}_w^* $ |  $V_{\text{rms},w}^*$ |
# | - | - | - | - | - | - |
# | 2.0 | 21403  | 683.05 | 571.58 | 936.65 | 40.89 |
# | 1.0 | 83935 | 682.87 | 572.23 | 936.11 | 40.78 |
# | 0.5 | 332307 | 682.80 | 572.05 | 937.37 | 40.77 |

# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/sz_problems --NbConvertApp.output_base=sz_benchmark 3.2b_sz_benchmark.ipynb')

