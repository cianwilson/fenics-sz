# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dolfinx-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 43 Izu with a time-dependent coupling depth

# %% [markdown]
# ## Time-dependent implementation

# %% [markdown]
# ### Preamble

# %% [markdown]
# Set some path information.

# %%
import sys, os, shutil
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))

# %% [markdown]
# Loading everything we need from `sz_problem` and also set our default plotting and output preferences.

# %%
import fenics_sz.utils
from fenics_sz.sz_problems.sz_params import allsz_params, default_params
from fenics_sz.sz_problems.sz_slab import create_slab, plot_slab
from fenics_sz.sz_problems.sz_geometry import create_sz_geometry
from fenics_sz.applications.codillo_setup import TDCDDislSubductionProblem
import numpy as np
import dolfinx as df
import pyvista as pv
import pathlib
import copy
import matplotlib.pyplot as pl
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### Parameters

# %% [markdown]
# We first select the name and resolution scale, `resscale` and target Courant number `cfl` of the model.
#
# ```{admonition} Resolution
# By default the resolution (both spatial and temporal) is low to allow for a quick runtime and smaller website size.  If sufficient computational resources are available set a lower `resscale` and a lower `cfl` to get higher spatial and temporal resolutions respectively. This is necessary to get results with sufficient accuracy for scientific interpretation.
# ```
#

# %%
name = "43_Izu"
resscale = 3.0
cfl      = 3.0

# %% [markdown]
# Then load the remaining parameters from the global suite.

# %%
szdict = allsz_params[name]
print("{}:".format(name))
print("{:<20} {:<10}".format('Key','Value'))
print("-"*85)
for k, v in allsz_params[name].items():
    if v is not None and k not in ['z0', 'z15']: print("{:<20} {}".format(k, v))

# %% [markdown]
# ### Setup

# %% [markdown]
# Setup a slab.

# %%
slab = create_slab(szdict['xs'], szdict['ys'], resscale, szdict['lc_depth'])
_ = plot_slab(slab)

# %% [markdown]
# Create the subduction zome geometry around the slab.

# %%
geom = create_sz_geometry(slab, resscale, szdict['sztype'], szdict['io_depth'], szdict['extra_width'], 
                             szdict['coast_distance'], szdict['lc_depth'], szdict['uc_depth'])
_ = geom.plot()

# %% [markdown]
# Finally, declare the `TDDislSubductionProblem` problem class using the dictionary of parameters.

# %%
sz = TDCDDislSubductionProblem(geom, **szdict, 
                               cd0=30, cdf=80, dcd=default_params['coupling_depth_range'],
                               tc0=0.0, tcf=szdict['As'])

# %% [markdown]
# ### Solve

# %% [markdown]
# Solve using a dislocation creep rheology.

# %%
# Select the timestep based on the approximate target Courant number
dt = cfl*resscale/szdict['Vs']
# Reduce the timestep to get an integer number of timesteps
dt = szdict['As']/np.ceil(szdict['As']/dt)

# set up a gif (FIXME: we need time-dependent output from these simulations)
fps = 5
plotter = pv.Plotter(notebook=False, off_screen=True)
fenics_sz.utils.plot.plot_scalar(sz.T_i, plotter=plotter, scale=sz.T0, gather=True, cmap='coolwarm', clim=[0.0, sz.Tm*sz.T0], scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
geom.pyvistaplot(plotter=plotter, color='green', width=2)
cd = min(max(sz.cd0, sz.cd0 + (sz.cdf - sz.cd0)/(sz.tcf - sz.tc0)*(sz.t_Myr - sz.tc0)), sz.cdf)
cdpt = slab.findpointy(-cd)
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
plotter.open_gif( str(output_folder / "{}_td_solution_resscale_{:.2f}_cfl_{:.2f}.gif".format(name, resscale, cfl,)), fps=fps)

solutions = sz.solve(szdict['As'], dt, theta=0.5, rtol=1.e-1, verbosity=1, plotter=plotter, save_period=1.0)

plotter.close()

# %% [markdown]
# ### Plot

# %% [markdown]
# Plot the solution at the finish time.

# %%
plotter = pv.Plotter()
fenics_sz.utils.plot.plot_scalar(sz.T_i, plotter=plotter, scale=sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
fenics_sz.utils.plot.plot_vector_glyphs(sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz.v0))
fenics_sz.utils.plot.plot_vector_glyphs(sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz.v0))
geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = slab.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter)
fenics_sz.utils.plot.plot_save(plotter, output_folder / "{}_td_solution_resscale_{:.2f}_cfl_{:.2f}.png".format(name, resscale, cfl,))

# %% [markdown]
# Save it to disk so that it can be examined with other visualization software (e.g. [Paraview](https://www.paraview.org/)).

# %%
filename = output_folder / "{}_td_solution_resscale_{:.2f}_cfl_{:.2f}.bp".format(name, resscale, cfl,)
with df.io.VTXWriter(sz.mesh.comm, filename, [sz.T_i, sz.vs_i, sz.vw_i]) as vtx:
    vtx.write(0.0)
# zip the .bp folder so that it can be downloaded from jupyter lab
shutil.make_archive(str(filename), 'zip', root_dir=str(filename.parent), base_dir=str(filename.name))

# %% [markdown]
# ### Plot slab temperatures over time

# %%

# get some points along the slab
slabpoints = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in sz.geom.slab_spline.interpcurves])
cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(slabpoints, sz.mesh)

# do the same along a spline deeper in the slab
slabmoho = copy.deepcopy(sz.geom.slab_spline)
slabmoho.translatenormalandcrop(-7.0)
slabmohopoints = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in slabmoho.interpcurves])
mcinds, mcells = fenics_sz.utils.mesh.get_cell_collisions(slabmohopoints, sz.mesh)

# set up a figure
fig, (ax, axm) = pl.subplots(1,2)

for sol in solutions[::5]:
    t = sol['t']
    T = sol['T']
    # plot the slab temperatures
    ax.plot(T.eval(slabpoints, cells)[:,0], -slabpoints[:,1], label='t = {:.2f} Myr'.format(t))
    # plot the moho temperatures
    axm.plot(T.eval(slabmohopoints, mcells)[:,0], -slabmohopoints[:,1], label='t = {:.2f} Myr'.format(t))
# labels, title etc.
ax.set_xlabel('T ($^\circ$C)')
ax.set_ylabel('z (km)')
ax.set_title('Slab surface temperatures')
ax.legend()
ax.invert_yaxis()

axm.set_xlabel('T ($^\circ$C)')
axm.set_ylabel('z (km)')
axm.set_title('Moho temperatures')
axm.legend()
axm.invert_yaxis()

# %%
