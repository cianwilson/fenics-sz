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
# # Subduction Zone Benchmark

# %% [markdown]
# ## Convergence testing

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
def solve_benchmark_case1(resscale, petsc_options_s=None, petsc_options_T=None, partition_by_region=True):
    """Solve benchmark case 1 with resolution scale resscale and petsc options petsc_options_s for the Stokes solver and petsc_options_T for the temperature solver."""
    xs = [0.0, 140.0, 240.0, 400.0]
    ys = [0.0, -70.0, -120.0, -200.0]
    lc_depth = 40
    uc_depth = 15
    coast_distance = 0
    extra_width = 0
    sztype = 'continental'
    io_depth = 139
    A      = 100.0      # age of subducting slab (Myr)
    qs     = 0.065      # surface heat flux (W/m^2)
    Vs     = 100.0      # slab speed (mm/yr)

    # create the slab
    slab = create_slab(xs, ys, resscale, lc_depth)
    # construct the geometry
    geom = create_sz_geometry(slab, resscale, sztype, io_depth, extra_width, 
                               coast_distance, lc_depth, uc_depth)
    # set up a subduction zone problem
    sz = SteadyIsoSubductionProblem(geom, A=A, Vs=Vs, sztype=sztype, qs=qs, partition_by_region=partition_by_region)

    # solve it using a steady state assumption and an isoviscous rheology
    sz.solve(petsc_options_s=petsc_options_s, petsc_options_T=petsc_options_T)

    return sz

def benchmark_case1_diagnostics(*args, **kwargs):
    """Return the diagnostics from benchmark case 1 with resolution scale resscale and petsc options petsc_options_s for the Stokes solver and petsc_options_T for the temperature solver."""
    return solve_benchmark_case1(*args, **kwargs).get_diagnostics()


# %% tags=["active-ipynb"]
# resscales = [4.0, 2.0, 1.0]
# diagnostics_case1 = []
# for resscale in resscales:
#     diagnostics_case1.append((resscale, benchmark_case1_diagnostics(resscale)))

# %% tags=["active-ipynb"]
# print('')
# print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
# for resscale, diag in diagnostics_case1:
#     print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale, *diag.values()))    

# %% [markdown]
# For comparison here are the values reported for case 1 using [TerraFERMA](https://terraferma.github.io) in [Wilson & van Keken, 2023](http://dx.doi.org/10.1186/s40645-023-00588-6):
#
# | `resscale` | $T_{\text{ndof}} $ | $T_{(200,-100)}^*$ | $\overline{T}_s^*$ | $ \overline{T}_w^* $ |  $V_{\text{rms},w}^*$ |
# | - | - | - | - | - | - |
# | 2.0 | 21403  | 517.17 | 451.83 | 926.62 | 34.64 |
# | 1.0 | 83935  | 516.95 | 451.71 | 926.33 | 34.64 |
# | 0.5 | 332307 | 516.86 | 451.63 | 926.15 | 34.64 |
#
# Which we can test against our values.

# %%
values_wvk_case1 = [
    {'resscale': 2, 'T_ndof': 21403, 'T_{200,-100}': 517.17, 'Tbar_s': 451.83, 'Tbar_w': 926.62, 'Vrmsw': 34.64},
    {'resscale': 1, 'T_ndof': 83935, 'T_{200,-100}': 516.95, 'Tbar_s': 451.71, 'Tbar_w': 926.33, 'Vrmsw': 34.64},
    {'resscale': 0.5, 'T_ndof': 332307, 'T_{200,-100}': 516.86, 'Tbar_s': 451.63, 'Tbar_w': 926.15, 'Vrmsw': 34.64},
]


# %% tags=["active-ipynb"]
# print('')
# print('{:<12} {:<12}'.format('', 'error'))
# for key, val in diagnostics_case1[-1][-1].items():
#     if key != "T_ndof":
#         err = abs(values_wvk_case1[1][key]-val)/val
#         print('{:<12} {:<12.4g}'.format(key, err))
#         assert(err < 1.e-3) # check error is less than 0.1%

# %% [markdown]
# ### Benchmark case 2

# %%
def solve_benchmark_case2(resscale, petsc_options_s=None, petsc_options_T=None, partition_by_region=True):
    xs = [0.0, 140.0, 240.0, 400.0]
    ys = [0.0, -70.0, -120.0, -200.0]
    lc_depth = 40
    uc_depth = 15
    coast_distance = 0
    extra_width = 0
    sztype = 'continental'
    io_depth = 154
    A      = 100.0      # age of subducting slab (Myr)
    qs     = 0.065      # surface heat flux (W/m^2)
    Vs     = 100.0      # slab speed (mm/yr)

    # create the slab
    slab = create_slab(xs, ys, resscale, lc_depth)
    # construct the geometry
    geom = create_sz_geometry(slab, resscale, sztype, io_depth, extra_width, 
                               coast_distance, lc_depth, uc_depth)
    # set up a subduction zone problem
    sz = SteadyDislSubductionProblem(geom, A=A, Vs=Vs, sztype=sztype, qs=qs, partition_by_region=partition_by_region)

    # solve it using a steady state assumption and a dislocation creep rheology
    sz.solve(petsc_options_s=petsc_options_s, petsc_options_T=petsc_options_T)

    return sz

def benchmark_case2_diagnostics(*args, **kwargs):
    """Return the diagnostics from benchmark case 2 with resolution scale resscale and petsc options petsc_options_s for the Stokes solver and petsc_options_T for the temperature solver."""
    return solve_benchmark_case2(*args, **kwargs).get_diagnostics()


# %% tags=["active-ipynb"]
# resscales = [4.0, 2.0, 1.0]
# diagnostics_case2 = []
# for resscale in resscales:
#     diagnostics_case2.append((resscale, benchmark_case2_diagnostics(resscale)))

# %% tags=["active-ipynb"]
# print('')
# print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
# for resscale, diag in diagnostics_case2:
#     print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale, *diag.values()))    

# %% [markdown]
# For comparison here are the values reported for case 2 using [TerraFERMA](https://terraferma.github.io) in [Wilson & van Keken, 2023](http://dx.doi.org/10.1186/s40645-023-00588-6):
#
# | `resscale` | $T_{\text{ndof}} $ | $T_{(200,-100)}^*$ | $\overline{T}_s^*$ | $ \overline{T}_w^* $ |  $V_{\text{rms},w}^*$ |
# | - | - | - | - | - | - |
# | 2.0 | 21403  | 683.05 | 571.58 | 936.65 | 40.89 |
# | 1.0 | 83935 | 682.87 | 572.23 | 936.11 | 40.78 |
# | 0.5 | 332307 | 682.80 | 572.05 | 937.37 | 40.77 |
#
# Which we can test against our values.

# %%
values_wvk_case2 = [
    {'resscale': 2, 'T_ndof': 21403, 'T_{200,-100}': 683.05, 'Tbar_s': 571.58, 'Tbar_w': 936.65, 'Vrmsw': 40.89},
    {'resscale': 1, 'T_ndof': 83935, 'T_{200,-100}': 682.87, 'Tbar_s': 572.23, 'Tbar_w': 936.11, 'Vrmsw': 40.78},
    {'resscale': 0.5, 'T_ndof': 332307, 'T_{200,-100}': 682.80, 'Tbar_s': 572.05, 'Tbar_w': 937.37, 'Vrmsw': 40.77},
]

# %% tags=["active-ipynb"]
# print('')
# print('{:<12} {:<12}'.format('', 'error'))
# for key, val in diagnostics_case2[-1][-1].items():
#     if key != "T_ndof":
#         err = abs(values_wvk_case2[1][key]-val)/val
#         print('{:<12} {:<12.4g}'.format(key, err))
#         assert(err < 1.e-3) # check error is less than 0.1%

# %% [markdown]
# In the [next notebook](./3.4c_sz_benchmark_parallel.ipynb) we will check that these small errors are reproducible using our implementation in parallel.

# %%
