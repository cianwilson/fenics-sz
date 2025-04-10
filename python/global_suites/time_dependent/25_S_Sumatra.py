#!/usr/bin/env python
# coding: utf-8

# # 25 S Sumatra
# 
# Authors: Cameron Seebeck, Cian Wilson

# ## Time-dependent implementation

# ### Preamble

# Set some path information.

# In[ ]:


import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir))
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, os.path.pardir, 'python'))


# Loading everything we need from `sz_problem` and also set our default plotting and output preferences.

# In[ ]:


import utils
from sz_problems.sz_base import allsz_params
from sz_problems.sz_slab import create_slab, plot_slab
from sz_problems.sz_geometry import create_sz_geometry
from sz_problems.sz_problem import SubductionProblem
import numpy as np
import dolfinx as df
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)
import hashlib
import zipfile
import requests


# ### Parameters

# We first select the name and resolution scale, `resscale` and target Courant number `cfl` of the model.
# 
# ```{admonition} Resolution
# By default the resolution (both spatial and temporal) is low to allow for a quick runtime and smaller website size.  If sufficient computational resources are available set a lower `resscale` and a lower `cfl` to get higher spatial and temporal resolutions respectively. This is necessary to get results with sufficient accuracy for scientific interpretation.
# ```
# 

# In[ ]:


name = "25_S_Sumatra"
resscale = 3.0
cfl      = 3.0


# Then load the remaining parameters from the global suite.

# In[ ]:


szdict = allsz_params[name]
print("{}:".format(name))
print("{:<20} {:<10}".format('Key','Value'))
print("-"*85)
for k, v in allsz_params[name].items():
    if v is not None and k not in ['z0', 'z15']: print("{:<20} {}".format(k, v))


# Any of these can be modified in the dictionary.
# 
# Several additional parameters can be modified, for details see the documentation for the `SubductionProblem` class.

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    get_ipython().run_line_magic('pinfo', 'SubductionProblem')


# The `if __name__ == "__main__" and "__file__" not in globals():` logic above is only necessary to make sure that this only runs in the Jupyter notebook version of this code and not the python version.  It is not generally necessary when getting the docstring of a function or class in Jupyter.

# ### Setup

# Setup a slab.

# In[ ]:


slab = create_slab(szdict['xs'], szdict['ys'], resscale, szdict['lc_depth'])
_ = plot_slab(slab)


# Create the subduction zome geometry around the slab.

# In[ ]:


geom = create_sz_geometry(slab, resscale, szdict['sztype'], szdict['io_depth'], szdict['extra_width'], 
                             szdict['coast_distance'], szdict['lc_depth'], szdict['uc_depth'])
_ = geom.plot()


# Finally, declare the `SubductionZone` problem class using the dictionary of parameters.

# In[ ]:


sz = SubductionProblem(geom, **szdict)


# ### Solve

# Let's first set up a gif of the temperature.

# In[ ]:


fps = 5
plotter = pv.Plotter(notebook=False, off_screen=True)
utils.plot_scalar(sz.T_i, plotter=plotter, scale=sz.T0, gather=True, cmap='coolwarm', clim=[0.0, sz.Tm*sz.T0], scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
utils.plot_geometry(geom, plotter=plotter, color='green', width=2)
utils.plot_couplingdepth(slab, plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
plotter.open_gif( str(output_folder / "{}_td_solution_resscale_{:.2f}_cfl_{:.2f}.gif".format(name, resscale, cfl,)), fps=fps)


# Solve using a dislocation creep rheology.

# In[ ]:


# Select the timestep based on the approximate target Courant number
dt = cfl*resscale/szdict['Vs']
# Reduce the timestep to get an integer number of timesteps
dt = szdict['As']/np.ceil(szdict['As']/dt)
sz.solve_timedependent_dislocationcreep(szdict['As'], dt, theta=0.5, rtol=1.e-1, verbosity=1, plotter=plotter)
plotter.close()


# ### Plot

# Plot the solution.

# In[ ]:


plotter = pv.Plotter()
utils.plot_scalar(sz.T_i, plotter=plotter, scale=sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
utils.plot_vector_glyphs(sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz.v0))
utils.plot_vector_glyphs(sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz.v0))
utils.plot_geometry(geom, plotter=plotter, color='green', width=2)
utils.plot_couplingdepth(slab, plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
utils.plot_show(plotter)
utils.plot_save(plotter, output_folder / "{}_td_solution_resscale_{:.2f}_cfl_{:.2f}.png".format(name, resscale, cfl,))


# Save it to disk so that it can be examined with other visualization software (e.g. [Paraview](https://www.paraview.org/)).

# In[ ]:


filename = output_folder / "{}_td_solution_resscale_{:.2f}_cfl_{:.2f}.bp".format(name, resscale, cfl,)
with df.io.VTXWriter(sz.mesh.comm, filename, [sz.T_i, sz.vs_i, sz.vw_i]) as vtx:
    vtx.write(0.0)
# zip the .bp folder so that it can be downloaded from Jupyter lab
if __name__ == "__main__" and "__file__" not in globals():
    zipfilename = filename.with_suffix(".zip")
    get_ipython().system('zip -r $zipfilename $filename')


# ## Comparison

# Compare to the published result from [Wilson & van Keken, PEPS, 2023 (II)](http://dx.doi.org/10.1186/s40645-023-00588-6) and [van Keken & Wilson, PEPS, 2023 (III)](https://doi.org/10.1186/s40645-023-00589-5).  The original models used in these papers are also available as open-source repositories on [github](https://github.com/cianwilson/vankeken_wilson_peps_2023) and [zenodo](https://doi.org/10.5281/zenodo.7843967).
# 
# First download the minimal necessary data from zenodo and check it is the right version.

# In[ ]:


zipfilename = pathlib.Path(os.path.join(basedir, os.path.pardir, os.path.pardir, os.path.pardir, "data", "vankeken_wilson_peps_2023_TF_lowres_minimal.zip"))
if not zipfilename.is_file():
    zipfileurl = 'https://zenodo.org/records/13234021/files/vankeken_wilson_peps_2023_TF_lowres_minimal.zip'
    r = requests.get(zipfileurl, allow_redirects=True)
    open(zipfilename, 'wb').write(r.content)
assert hashlib.md5(open(zipfilename, 'rb').read()).hexdigest() == 'a8eca6220f9bee091e41a680d502fe0d'


# In[ ]:


tffilename = os.path.join('vankeken_wilson_peps_2023_TF_lowres_minimal', 'sz_suite_td', szdict['dirname']+'_minres_2.00_cfl_2.00.vtu')
tffilepath = os.path.join(basedir, os.path.pardir, os.path.pardir, os.path.pardir, 'data')
with zipfile.ZipFile(zipfilename, 'r') as z:
    z.extract(tffilename, path=tffilepath)


# In[ ]:


fxgrid = utils.grids_scalar(sz.T_i)[0]

tfgrid = pv.get_reader(os.path.join(tffilepath, tffilename)).read()

diffgrid = utils.pv_diff(fxgrid, tfgrid, field_name_map={'T':'Temperature::PotentialTemperature'}, pass_point_data=True)


# In[ ]:


diffgrid.set_active_scalars('T')
plotter_diff = pv.Plotter()
clim = None
plotter_diff.add_mesh(diffgrid, cmap='coolwarm', clim=clim, scalar_bar_args={'title': 'Temperature Difference (deg C)', 'bold':True})
utils.plot_geometry(geom, plotter=plotter_diff, color='green', width=2)
utils.plot_couplingdepth(slab, plotter=plotter_diff, render_points_as_spheres=True, point_size=5.0, color='green')
plotter_diff.enable_parallel_projection()
plotter_diff.view_xy()
plotter_diff.show()


# In[ ]:


integrated_data = diffgrid.integrate_data()
error = integrated_data['T'][0]/integrated_data['Area'][0]
print("Average error = {}".format(error,))
assert np.abs(error) < 5


# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../../python/global_suites/time_dependent --NbConvertApp.output_base=25_S_Sumatra 25_S_Sumatra.ipynb')


# In[ ]:




