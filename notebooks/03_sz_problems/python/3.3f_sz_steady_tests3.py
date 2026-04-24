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
# # Steady-State Subduction Zone Setup

# %% [markdown]
# ## Themes and variations - global suite geometries
#
# In this notebook we will try using more realistic geometries from the global suite.

# %% [markdown]
# ### Preamble
#
# Let's start by adding the path to the modules in the `python` folder to the system path (so we can find the our custom modules).

# %%
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))

# %% [markdown]
# Then load everything we need from `sz_problems` and other modules.

# %%
import fenics_sz.utils
from fenics_sz.sz_problems.sz_params import default_params, allsz_params
from fenics_sz.sz_problems.sz_slab import create_slab
from fenics_sz.sz_problems.sz_geometry import create_sz_geometry
from fenics_sz.sz_problems.sz_steady_isoviscous import SteadyIsoSubductionProblem
from fenics_sz.sz_problems.sz_steady_dislcreep import SteadyDislSubductionProblem
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### Alaska Peninsula (dislocation creep, low res)
#
# ```{admonition} Resolution
# By default the resolution is low to allow for a quick runtime and smaller website size.  If sufficient computational resources are available set a lower `resscale` to get higher resolutions and results with sufficient accuracy.
# ```

# %%
resscale_ak = 5.0
szdict_ak = allsz_params['01_Alaska_Peninsula']
slab_ak = create_slab(szdict_ak['xs'], szdict_ak['ys'], resscale_ak, szdict_ak['lc_depth'])
geom_ak = create_sz_geometry(slab_ak, resscale_ak, szdict_ak['sztype'], szdict_ak['io_depth'], szdict_ak['extra_width'], 
                             szdict_ak['coast_distance'], szdict_ak['lc_depth'], szdict_ak['uc_depth'])
sz_ak = SteadyDislSubductionProblem(geom_ak, **szdict_ak)

# %%
print("\nSolving steady state flow with isoviscous rheology...")
sz_ak.solve()

# %%
plotter_ak = fenics_sz.utils.plot.plot_scalar(sz_ak.T_i, scale=sz_ak.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
fenics_sz.utils.plot.plot_vector_glyphs(sz_ak.vw_i, plotter=plotter_ak, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz_ak.v0))
fenics_sz.utils.plot.plot_vector_glyphs(sz_ak.vs_i, plotter=plotter_ak, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz_ak.v0))
sz_ak.geom.pyvistaplot(plotter=plotter_ak, color='green', width=2)
cdpt = sz_ak.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter_ak, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter_ak)
fenics_sz.utils.plot.plot_save(plotter_ak, output_folder / "sz_steady_tests_ak_solution.png")

# %%
eta_ak = sz_ak.project_dislocationcreep_viscosity()
plotter_eta_ak = fenics_sz.utils.plot.plot_scalar(eta_ak, scale=sz_ak.eta0, gather=True, log_scale=True, show_edges=True, scalar_bar_args={'title': 'Viscosity (Pa s)', 'bold':True})
sz_ak.geom.pyvistaplot(plotter=plotter_eta_ak, color='green', width=2)
cdpt = sz_ak.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter_eta_ak, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter_eta_ak)
fenics_sz.utils.plot.plot_save(plotter_eta_ak, output_folder / "sz_steady_tests_ak_eta.png")

# %%
# clean up
del plotter_eta_ak
del sz_ak
del geom_ak
del slab_ak

# %% [markdown]
# ### N Antilles (dislocation creep, low res)
#
# ```{admonition} Resolution
# By default the resolution is low to allow for a quick runtime and smaller website size.  If sufficient computational resources are available set a lower `resscale` to get higher resolutions and results with sufficient accuracy.
# ```

# %%
resscale_ant = 5.0
szdict_ant = allsz_params['19_N_Antilles']
slab_ant = create_slab(szdict_ant['xs'], szdict_ant['ys'], resscale_ant, szdict_ant['lc_depth'])
geom_ant = create_sz_geometry(slab_ant, resscale_ant, szdict_ant['sztype'], szdict_ant['io_depth'], szdict_ant['extra_width'], 
                              szdict_ant['coast_distance'], szdict_ant['lc_depth'], szdict_ant['uc_depth'])
sz_ant = SteadyDislSubductionProblem(geom_ant, **szdict_ant)

# %%
print("\nSolving steady state flow with isoviscous rheology...")
sz_ant.solve()

# %%
plotter_ant = fenics_sz.utils.plot.plot_scalar(sz_ant.T_i, scale=sz_ant.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
fenics_sz.utils.plot.plot_vector_glyphs(sz_ant.vw_i, plotter=plotter_ant, gather=True, factor=0.25, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz_ant.v0))
fenics_sz.utils.plot.plot_vector_glyphs(sz_ant.vs_i, plotter=plotter_ant, gather=True, factor=0.25, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz_ant.v0))
sz_ant.geom.pyvistaplot(plotter=plotter_ant, color='green', width=2)
cdpt = sz_ant.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter_ant, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter_ant)
fenics_sz.utils.plot.plot_save(plotter_ant, output_folder / "sz_steady_tests_ant_solution.png")

# %%
eta_ant = sz_ant.project_dislocationcreep_viscosity()
plotter_eta_ant = fenics_sz.utils.plot.plot_scalar(eta_ant, scale=sz_ant.eta0, gather=True, log_scale=True, show_edges=True, scalar_bar_args={'title': 'Viscosity (Pa s)', 'bold':True})
sz_ant.geom.pyvistaplot(plotter=plotter_eta_ant, color='green', width=2)
cdpt = sz_ant.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter_eta_ant, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter_eta_ant)
fenics_sz.utils.plot.plot_save(plotter_eta_ant, output_folder / "sz_steady_tests_ant_eta.png")

# %%
# clean up
del plotter_eta_ant
del sz_ant
del geom_ant
del slab_ant

# %% [markdown]
# Having played with some test examples using a steady state solver we will [next](./3.4a_sz_benchmark_intro.ipynb) formally examine the published benchmark solutions and test how our implementation performs.

# %%
