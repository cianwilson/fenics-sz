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
# ## Parallel Scaling
#
# In [the previous notebook](./3.4b_sz_benchmark.ipynb) we tested that the error in our implementation of a steady-state thermal convection problem in two-dimensions converged towards the published benchmark value as the resolution scale (`resscale`) decreased (the number of elements increased).  We also wish to test for parallel scaling of this problem, assessing if the simulation wall time decreases as the number of processors used to solve it is increases.
#
# Here we perform strong scaling tests on our functions `solve_benchmark_case1` and `solve_benchmark_case2` from [the previous notebook](./3.4b_sz_benchmark.ipynb).

# %% [markdown]
# ### Preamble
#
# We start by loading all the modules we will require.

# %%
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
path = os.path.join(basedir, os.path.pardir, os.path.pardir, 'python')
sys.path.insert(0, path)
import fenics_sz.utils.ipp
import matplotlib.pyplot as pl
import matplotlib.ticker as ticker
import numpy as np
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### Implementation
#
# We perform the strong parallel scaling test using a utility function (from `python/utils/ipp.py`) that loops over a list of the number of processors calling our function for a given resolution scale, `rescale`.  It runs our function `solve_benchmark_case1` a specified `number` of times and evaluates and returns the time taken for each of a number of requested `steps`.

# %%
# the list of the number of processors we will use
nprocs_scale = [1, 2, 4]

# resolution scale factor (equivalent to minimum element size)
resscale_scale = 2

# perform the calculation a set number of times
number = 1

# We are interested in the time to create the mesh,
# declare the functions, assemble the problem and solve it.
# From our implementation in `solve_poisson_2d` it is also
# possible to request the time to declare the Dirichlet and
# Neumann boundary conditions and the forms.
steps = [
          'Assemble Temperature', 'Assemble Stokes',
          'Solve Temperature', 'Solve Stokes'
         ]

# %% [markdown]
# #### Case 1 - Direct
#
# We start by running benchmark case 1 - isoviscous - and will compare direct and iterative solver strategies.

# %%
# declare a dictionary to store the times each step takes
maxtimes_1 = {}

# %%
petsc_options_s = {'ksp_type':'preonly', 
                   'pc_type':'lu', 
                   'pc_factor_mat_solver_type' : 'mumps',
                  }

maxtimes_1['Direct Stokes'], _, _ = fenics_sz.utils.ipp.profile_parallel(nprocs_scale, steps, path, 
                                                         'fenics_sz.sz_problems.sz_benchmark', 'solve_benchmark_case1', 
                                                        resscale_scale, number=number, petsc_options_s=petsc_options_s,
                                                        output_filename=output_folder / 'sz_benchmark_scaling_direct_1.png')

# %% [markdown]
# The behavior of the scaling test will strongly depend on the computational resources available on the machine where this notebook is run.  In particular when the website is generated it has to run as quickly as possible on github, hence we limit our requested numbers of processors, size of the problem (`resscale`) and number of calculations to average over (`number`) in the default setup of this notebook.
#
# For comparison we provide the output of this notebook using `resscale = 0.5` and `number = 10` in Figure 3.4.1 generated on a dedicated machine with a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software.
#
# > ![Direct Scaling](images/sz_benchmark_scaling_direct_1.png)
#
# *Figure 3.4.1 Scaling results for a direct solver with `resscale = 0.5` averaged over `number = 10` calculations using a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software on a dedicated machine.*
#
# We can see in Figure 3.4.1 that assembly scales well up to four processes at this resolution.  As in previous scaling tests on steady state problems we see that the direct solves do not scale but that the Stokes solves (in the slab and wedge) are more expensive than the (global) temperature solve.
#
# We also need to test that we are still converging to the benchmark solutions in parallel.

# %%
# the list of the number of processors to test the convergence on
nprocs_conv = [1, 4,]

# List of resolutions to try
resscales_conv = [2,]

# %%
diagnostics = []
for resscalel in resscales_conv:
    diagnostics.append(fenics_sz.utils.ipp.run_parallel(nprocs_conv, path, 
                                           'fenics_sz.sz_problems.sz_benchmark', 
                                           'solve_benchmark_case1', 
                                           resscalel))

print('')
print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('nprocs', 'resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
for resscalel, diag_all in zip(resscales_conv, diagnostics):
    for i, nproc in enumerate(nprocs_conv):
        print('{:<12d} {:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(nproc, resscalel, *diag_all[i].values()))
         # check error (compared to the serial case) is less than 0.1%
        if i > 0: 
            for k, v in diag_all[i].items():
                assert(abs(v - diag_all[0][k])/abs(diag_all[0][k]) < 1.e-3)

# %% [markdown]
# #### Case 1 - Iterative
#
# We can try to improve the scaling of the Stokes solution steps by using an iterative solver.

# %%
petsc_options_s = {'ksp_type':'minres', 
                   'ksp_rtol' : 5.e-9,
                   'pc_type':'fieldsplit', 
                   'pc_fieldsplit_type': 'additive',
                   'fieldsplit_v_ksp_type':'preonly',
                   'fieldsplit_v_pc_type':'gamg',
                   'pc_gamg_threshold_scale' : 1.0, 
                   'pc_gamg_threshold' : 0.01, 
                   'pc_gamg_coarse_eq_limit' : 800,
                   'fieldsplit_p_ksp_type':'preonly',
                   'fieldsplit_p_pc_type':'jacobi'}

maxtimes_1['Iterative Stokes'], _, _ = fenics_sz.utils.ipp.profile_parallel(nprocs_scale, steps, path, 
                                                            'fenics_sz.sz_problems.sz_benchmark', 'solve_benchmark_case1', 
                                                            resscale_scale, petsc_options_s=petsc_options_s, number=number,
                                                            output_filename=output_folder / 'sz_benchmark_scaling_iterative_1.png')

# %% [markdown]
# If sufficient computational resources are available when running this notebook (unlikely during website generation) this should show that the iterative method scales better than the direct method but has a much higher absolute cost (wall time).  
#
# For reference we provide the output of this notebook using `resscale = 0.5` and `number = 10` in Figure 3.4.2 generated on a dedicated machine with a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software.
#
# > ![Iterative Scaling](images/sz_benchmark_scaling_iterative_1.png)
#
# *Figure 3.4.2 Scaling results for a direct solver with `resscale = 0.5` averaged over `number = 10` calculations using a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software on a dedicated machine.*
#
# Here we can see the improved scaling (for the Stokes solve, the temperature is still using a direct method) but increased cost.
#
# We also test if the iterative method is converging to the benchmark solution below.
#

# %%
diagnostics = []
for resscalel in resscales_conv:
    diagnostics.append(fenics_sz.utils.ipp.run_parallel(nprocs_conv, path, 
                                           'fenics_sz.sz_problems.sz_benchmark', 
                                           'solve_benchmark_case1', 
                                           resscalel, petsc_options_s=petsc_options_s))

print('')
print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('nprocs', 'resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
for resscalel, diag_all in zip(resscales_conv, diagnostics):
    for i, nproc in enumerate(nprocs_conv):
        print('{:<12d} {:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(nproc, resscalel, *diag_all[i].values()))
         # check error (compared to the serial case) is less than 0.1%
        if i > 0: 
            for k, v in diag_all[i].items():
                assert(abs(v - diag_all[0][k])/abs(diag_all[0][k]) < 1.e-3)

# %% [markdown]
# #### Comparison
#
# We can more easily compare the different solution method directly by plotting their walltimes for assembly and solution.

# %%
# choose which steps to compare
compare_steps = ['Assemble Stokes', 'Solve Stokes']

# set up a figure for plotting
fig, axs = pl.subplots(nrows=len(compare_steps), figsize=[6.4,4.8*len(compare_steps)], sharex=True)
if len(compare_steps) == 1: axs = [axs]
for i, step in enumerate(compare_steps):
    s = steps.index(step)
    for name, lmaxtimes in maxtimes_1.items():
        axs[i].plot(nprocs_scale, [t[s] for t in lmaxtimes], 'o-', label=name)
    axs[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[i].set_title(step)
    axs[i].legend()
    axs[i].set_ylabel('wall time (s)')
    axs[i].grid()
axs[-1].set_xlabel('number processors')
# save the figure
fig.savefig(output_folder / 'sz_benchmark_scaling_comparison_1.png')

# %% [markdown]
# With sufficient computational resources we will see that both methods have approximately the same assembly costs but the iterative methods solution wall time costs only become competitive above approximately four processes (for `resscale = 0.5`) at which point the assembly scaling has stalled anyway.
#
# For reference we provide the output of this notebook using `resscale = 0.5` and `number = 10` in Figure 3.4.3 generated on a dedicated machine with a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software.
#
# > ![Scaling Comparison](images/sz_benchmark_scaling_comparison_1.png)
#
# *Figure 3.4.3 Scaling results for a direct solver with `resscale = 0.5` averaged over `number = 10` calculations using a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software on a dedicated machine.*

# %% [markdown]
# #### Case 2 - Direct
#
# Time constraints mean we do not run case 2 - temperature and strain-rate dependent rheology - but we do present some previously run results below.  
#
# Uncommenting the cells below will allow the direct solution strategy to be tested interactively.

# %%
# declare a dictionary to store the times each step takes
maxtimes_2 = {}

# note that the iterative solver requires sufficient resolution to converge
# in the non-linear case so we decrease the resscale for these cases
resscale_scale_2 = 0.5
resscales_conv_2 = [0.5,]

# %%
# petsc_options_s = {'ksp_type':'preonly', 
#                    'pc_type':'lu', 
#                    'pc_factor_mat_solver_type' : 'mumps',
#                   }

# maxtimes_2['Direct Stokes'], _, _ = fenics_sz.utils.ipp.profile_parallel(nprocs_scale, steps, path, 
#                                                          'fenics_sz.sz_problems.sz_benchmark', 'solve_benchmark_case2', 
#                                                         resscale_scale_2, number=number, petsc_options_s=petsc_options_s,
#                                                         output_filename=output_folder / 'sz_benchmark_scaling_direct_2.png')

# %% [markdown]
# As before, the behavior of the scaling test will strongly depend on the computational resources available on the machine where this notebook is run.
#
# For comparison we provide the output of this notebook using `resscale = 0.5` and `number = 10` in Figure 3.4.4 generated on a dedicated machine with a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software.
#
# > ![Direct Scaling](images/sz_benchmark_scaling_direct_2.png)
#
# *Figure 3.4.4 Scaling results for a direct solver with `resscale = 0.5` averaged over `number = 10` calculations using a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software on a dedicated machine.*
#
# We can see in Figure 3.4.4 that assembly scales well.  The improved scaling behavior of the solves seen in [non-linear Blankenbach convection case](../02_background/2.5c_blankenbach_parallel.ipynb) is seen again.  This is likely because case 2 takes multiple non-linear iterations, caching the direct solver's initial analysis step between iterations and improving the scaling relative to the linear case 1.
#
# We can also test that we are still converging to the benchmark solutions in parallel.

# %%
# diagnostics = []
# for resscalel in resscales_conv_2:
#     diagnostics.append(utils.ipp.run_parallel(nprocs_conv, path, 
#                                            'fenics_sz.sz_problems.sz_benchmark', 
#                                            'solve_benchmark_case2', 
#                                            resscalel))

# print('')
# print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('nprocs', 'resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
# for resscalel, diag_all in zip(resscales_conv_2, diagnostics):
#     for i, nproc in enumerate(nprocs_conv):
#         print('{:<12d} {:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(nproc, resscalel, *diag_all[i].values()))
#          # check error (compared to the serial case) is less than 0.1%
#         if i > 0: 
#             for k, v in diag_all[i].items():
#                 assert(abs(v - diag_all[0][k])/abs(diag_all[0][k]) < 1.e-3)

# %% [markdown]
# #### Case 2 - Iterative
#
# Time constraints mean we do not run case 2 - temperature and strain-rate dependent rheology - but do present some previously run results below.
#
# Uncommenting the cells below will allow the iterative solution strategy to be tested interactively.

# %%
# petsc_options_s = {'ksp_type':'minres', 
#                    'ksp_rtol' : 5.e-9,
#                    'pc_type':'fieldsplit', 
#                    'pc_fieldsplit_type': 'additive',
#                    'fieldsplit_v_ksp_type':'preonly',
#                    'fieldsplit_v_pc_type':'gamg',
#                    'pc_gamg_threshold_scale' : 1.0, 
#                    'pc_gamg_threshold' : 0.01, 
#                    'pc_gamg_coarse_eq_limit' : 800,
#                    'fieldsplit_p_ksp_type':'preonly',
#                    'fieldsplit_p_pc_type':'jacobi'}

# maxtimes_2['Iterative Stokes'], _, _ = fenics_sz.utils.ipp.profile_parallel(nprocs_scale, steps, path, 
#                                                             'fenics_sz.sz_problems.sz_benchmark', 'solve_benchmark_case2', 
#                                                             resscale_scale_2, petsc_options_s=petsc_options_s, number=number,
#                                                             output_filename=output_folder / 'sz_benchmark_scaling_iterative_2.png')

# %% [markdown]
# If sufficient computational resources are available when running this cell interactively this should show that the iterative method scales better than the direct method but has a much higher absolute cost (wall time).  
#
# For reference we provide the output of this notebook using `resscale = 0.5` and `number = 10` in Figure 3.4.5 generated on a dedicated machine with a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software.
#
# > ![Iterative Scaling](images/sz_benchmark_scaling_iterative_2.png)
#
# *Figure 3.4.5 Scaling results for a direct solver with `resscale = 0.5` averaged over `number = 10` calculations using a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software on a dedicated machine.*
#
# Here we can see the improved scaling (for the Stokes solve, the temperature is still using a direct method) but hugely increased cost.  This increase is even more significant here as the non-linearity in case 2 compared to 1 has increased the number of nonlinear iterations taken and the iterative method does not take advantage of the analysis caching employed by the direct solver strategy.
#
# We can also test if the iterative method is converging to the benchmark solution by uncommenting the cells below.

# %%
# diagnostics = []
# for resscalel in resscales_conv_2:
#     diagnostics.append(utils.ipp.run_parallel(nprocs_conv, path, 
#                                            'fenics_sz.sz_problems.sz_benchmark', 
#                                            'solve_benchmark_case2', 
#                                            resscalel, petsc_options_s=petsc_options_s))

# print('')
# print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('nprocs', 'resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
# for resscalel, diag_all in zip(resscales_conv_2, diagnostics):
#     for i, nproc in enumerate(nprocs_conv):
#         print('{:<12d} {:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(nproc, resscalel, *diag_all[i].values()))
#          # check error (compared to the serial case) is less than 0.1%
#         if i > 0: 
#             for k, v in diag_all[i].items():
#                 assert(abs(v - diag_all[0][k])/abs(diag_all[0][k]) < 1.e-3)

# %% [markdown]
# #### Case 2 - Comparison
#
# Time constraints mean we do not run case 2 - temperature and strain-rate dependent rheology - but do present some previously run results below.  
#
# Uncommenting the cells below will allow the solution strategies to be compared interactively.

# %%
# # choose which steps to compare
# compare_steps = ['Assemble Stokes', 'Solve Stokes']

# # set up a figure for plotting
# fig, axs = pl.subplots(nrows=len(compare_steps), figsize=[6.4,4.8*len(compare_steps)], sharex=True)
# if len(compare_steps) == 1: axs = [axs]
# for i, step in enumerate(compare_steps):
#     s = steps.index(step)
#     for name, lmaxtimes in maxtimes_2.items():
#         axs[i].plot(nprocs_scale, [t[s] for t in lmaxtimes], 'o-', label=name)
#     axs[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#     axs[i].set_title(step)
#     axs[i].legend()
#     axs[i].set_ylabel('wall time (s)')
#     axs[i].grid()
# axs[-1].set_xlabel('number processors')
# # save the figure
# fig.savefig(output_folder / 'sz_benchmark_scaling_comparison_2.png')

# %% [markdown]
# If running interactively with sufficient computational resources we will see that both methods have approximately the same assembly costs but the iterative methods wall time costs are substantially higher and never become competitive despite scaling better at higher number of processors.
#
# For reference we provide the output of this notebook using `resscale = 0.5` and `number = 10` in Figure 3.4.6 generated on a dedicated machine using a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software.
#
# > ![Scaling Comparison](images/sz_benchmark_scaling_comparison_2.png)
#
# *Figure 3.4.6 Scaling results for a direct solver with `resscale = 0.5` averaged over `number = 10` calculations using a local [conda](../01_introduction/1.2_usage.ipynb) installation of the software on a dedicated machine.*
#
# In the [next section](../04_global_suites/4.1_global_suites_intro.ipynb) we will apply what we have learned here to the global suite of subduction zone thermal solvers.  First however we implement a [time-dependent subduction problem](./3.5a_sz_tdep_problem.ipynb).

# %%
