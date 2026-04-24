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
# ## Themes and variations - higher resolution
#
# In this notebook we will try increasing the resolution of the benchmark cases to see if we match the benchmarks better.

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
# ### Benchmark case 1

# %%
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

# %%
resscale2 = 2.5
slab_resscale2 = create_slab(xs, ys, resscale2, lc_depth)
geom_resscale2 = create_sz_geometry(slab_resscale2, resscale2, sztype, io_depth_1, extra_width, 
                           coast_distance, lc_depth, uc_depth)
sz_case1_resscale2 = SteadyIsoSubductionProblem(geom_resscale2, A=A, Vs=Vs, sztype=sztype, qs=qs)

# %%
print("\nSolving steady state flow with isoviscous rheology...")
sz_case1_resscale2.solve()

# %%
diag_resscale2 = sz_case1_resscale2.get_diagnostics()

print('')
print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale2, *diag_resscale2.values()))

# %% [markdown]
# For comparison here are the values reported for case 1 using [TerraFERMA](https://terraferma.github.io) in [Wilson & van Keken, 2023](http://dx.doi.org/10.1186/s40645-023-00588-6):
#
# | `resscale` | $T_{\text{ndof}} $ | $T_{(200,-100)}^*$ | $\overline{T}_s^*$ | $ \overline{T}_w^* $ |  $V_{\text{rms},w}^*$ |
# | - | - | - | - | - | - |
# | 2.0 | 21403  | 517.17 | 451.83 | 926.62 | 34.64 |
# | 1.0 | 83935  | 516.95 | 451.71 | 926.33 | 34.64 |
# | 0.5 | 332307 | 516.86 | 451.63 | 926.15 | 34.64 |

# %%
plotter_case1_resscale2 = fenics_sz.utils.plot.plot_scalar(sz_case1_resscale2.T_i, scale=sz_case1_resscale2.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
fenics_sz.utils.plot.plot_vector_glyphs(sz_case1_resscale2.vw_i, plotter=plotter_case1_resscale2, gather=True, factor=0.05, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz_case1_resscale2.v0))
fenics_sz.utils.plot.plot_vector_glyphs(sz_case1_resscale2.vs_i, plotter=plotter_case1_resscale2, gather=True, factor=0.05, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz_case1_resscale2.v0))
sz_case1_resscale2.geom.pyvistaplot(plotter=plotter_case1_resscale2, color='green', width=2)
cdpt = sz_case1_resscale2.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter_case1_resscale2, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter_case1_resscale2)
fenics_sz.utils.plot.plot_save(plotter_case1_resscale2, output_folder / "sz_steady_tests_case1_resscale2_solution.png")

# %%
# clean up
del plotter_case1_resscale2
del sz_case1_resscale2
del geom_resscale2

# %% [markdown]
# ### Benchmark case 2

# %%
io_depth_2 = 154.0
geom_case2_resscale2 = create_sz_geometry(slab_resscale2, resscale2, sztype, io_depth_2, extra_width, 
                                          coast_distance, lc_depth, uc_depth)
sz_case2_resscale2 = SteadyDislSubductionProblem(geom_case2_resscale2, A=A, Vs=Vs, sztype=sztype, qs=qs)

# %%
print("\nSolving steady state flow with dislocation creep rheology...")
sz_case2_resscale2.solve()

# %%
diag_case2_resscale2 = sz_case2_resscale2.get_diagnostics()

print('')
print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale2, *diag_case2_resscale2.values()))

# %% [markdown]
# For comparison here are the values reported for case 2 using [TerraFERMA](https://terraferma.github.io) in [Wilson & van Keken, 2023](http://dx.doi.org/10.1186/s40645-023-00588-6):
#
# | `resscale` | $T_{\text{ndof}} $ | $T_{(200,-100)}^*$ | $\overline{T}_s^*$ | $ \overline{T}_w^* $ |  $V_{\text{rms},w}^*$ |
# | - | - | - | - | - | - |
# | 2.0 | 21403  | 683.05 | 571.58 | 936.65 | 40.89 |
# | 1.0 | 83935 | 682.87 | 572.23 | 936.11 | 40.78 |
# | 0.5 | 332307 | 682.80 | 572.05 | 937.37 | 40.77 |

# %%
plotter_case2_resscale2 = fenics_sz.utils.plot.plot_scalar(sz_case2_resscale2.T_i, scale=sz_case2_resscale2.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
fenics_sz.utils.plot.plot_vector_glyphs(sz_case2_resscale2.vw_i, plotter=plotter_case2_resscale2, gather=True, factor=0.05, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz_case2_resscale2.v0))
fenics_sz.utils.plot.plot_vector_glyphs(sz_case2_resscale2.vs_i, plotter=plotter_case2_resscale2, gather=True, factor=0.05, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz_case2_resscale2.v0))
sz_case2_resscale2.geom.pyvistaplot(plotter=plotter_case2_resscale2, color='green', width=2)
cdpt = sz_case2_resscale2.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter_case2_resscale2, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter_case2_resscale2)
fenics_sz.utils.plot.plot_save(plotter_case2_resscale2, output_folder / "sz_steady_tests_case2_resscale2_solution.png")

# %%
# clean up
del plotter_case2_resscale2
del sz_case2_resscale2
del geom_case2_resscale2
del slab_resscale2

# %% [markdown]
# In the [next notebook](./3.3e_sz_steady_tests2.ipynb) we will try seeing the effect of varying the coupling depth.

# %% [markdown]
#
