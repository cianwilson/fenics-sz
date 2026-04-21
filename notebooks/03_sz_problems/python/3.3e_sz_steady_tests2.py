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
# ## Themes and variations - varying the coupling depth
#
# In this notebook we will try seeing the effect of varying the coupling depth on the benchmark solution.

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
# We will now re-use all of the parameters for case 2

# %%
xs = [0.0, 140.0, 240.0, 400.0]
ys = [0.0, -70.0, -120.0, -200.0]
lc_depth = 40
uc_depth = 15
coast_distance = 0
extra_width = 0
sztype = 'continental'
io_depth_2 = 154.0
A      = 100.0      # age of subducting slab (Myr)
qs     = 0.065      # surface heat flux (W/m^2)
Vs     = 100.0      # slab speed (mm/yr)

# %% [markdown]
# but vary the coupling depth by passing in an additional keyword argument `coupling_depth` to `create_slab`.  The rest of the solution procedure is the same as [before](./3.3d_sz_steady_tests1.ipynb).
#
# Let's loop over a series of coupling depths to see how varying it changes the solution.

# %%
# set a list of coupling depths to try
coupling_depths = [60.0, 80.0, 100.0]
resscale3 = 3.0

# set up a list to save the diagnostics from each
diagnostics = []
# loop over the couplings depths
for coupling_depth in coupling_depths:
    # create the slab object, all of the input arguments are the same as in case 2
    # but this time we also pass in the coupling_depth keyword argument to override
    # the default value (80 km)
    slab_dc = create_slab(xs, ys, resscale3, lc_depth, coupling_depth=coupling_depth)
    # set up the geometry
    geom_dc = create_sz_geometry(slab_dc, resscale3, sztype, io_depth_2, extra_width, 
                                            coast_distance, lc_depth, uc_depth)
    # set up the subduction zone problem
    sz_dc = SteadyDislSubductionProblem(geom_dc, A=A, Vs=Vs, sztype=sztype, qs=qs)

    # solve the steady state problem
    if sz_dc.comm.rank == 0: print(f"\nSolving steady state flow with coupling depth = {coupling_depth}km...")
    sz_dc.solve()

    # retrieve the diagnostics
    diagnostics.append(sz_dc.get_diagnostics())

    # plot the solution
    plotter_dc = fenics_sz.utils.plot.plot_scalar(sz_dc.T_i, scale=sz_dc.T0, gather=True, cmap='coolwarm', 
                                   scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    fenics_sz.utils.plot.plot_vector_glyphs(sz_dc.vw_i, plotter=plotter_dc, gather=True, factor=0.05, color='k', 
                             scale=fenics_sz.utils.mps_to_mmpyr(sz_dc.v0))
    fenics_sz.utils.plot.plot_vector_glyphs(sz_dc.vs_i, plotter=plotter_dc, gather=True, factor=0.05, color='k', 
                             scale=fenics_sz.utils.mps_to_mmpyr(sz_dc.v0))
    sz_dc.geom.pyvistaplot(plotter=plotter_dc, color='green', width=2)
    cdpt = sz_dc.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
    fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter_dc, render_points_as_spheres=True, point_size=10.0, color='green')
    fenics_sz.utils.plot.plot_show(plotter_dc)
    fenics_sz.utils.plot.plot_save(plotter_dc, output_folder / f"sz_steady_tests_dc{coupling_depth}_solution.png")

    # clean up
    del plotter_dc
    del sz_dc
    del geom_dc
    del slab_dc


# %% [markdown]
# As well as visualizing the solutions we can see what effect varying the coupling depth has on the global diagnostics from the benchmark.

# %%
# print the varying coupling depth output
print('')
print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('d_c', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
for dc, diag in zip(coupling_depths, diagnostics):
    print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(dc, *diag.values()))

# %% [markdown]
# Note the dramatic drop in temperature at (200, -100), `T_{200,-100}`, once the coupling depth reaches 100km.
#
# In the [next notebook](./3.3f_sz_steady_tests3.ipynb) we will try more realistic geometries.

# %%
