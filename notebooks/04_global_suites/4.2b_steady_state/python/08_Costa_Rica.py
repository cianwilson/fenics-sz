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
# # 08 Costa Rica

# %% [markdown]
# ## Steady state implementation

# %% [markdown]
# ### Preamble

# %% [markdown]
# We are going to run this notebook in parallel using `ipyparallel`.  To do this we launch a parallel engine with `nprocs` processes.  Following this, all cells we wish to run in parallel need to start with the jupyter magic `%%px`.
#
# ```{admonition} %%px
# Cells without the `%%px` magic will run locally and will not have access to anything loaded on the parallel engine.
# ```

# %%
import ipyparallel as ipp
nprocs = 2
rc = ipp.Cluster(engine_launcher_class="mpi", n=nprocs).start_and_connect_sync()

# %% [markdown]
# Set some path information.

# %%
# %%px
import sys, os, shutil
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(basedir, os.path.pardir, os.path.pardir, os.path.pardir, 'python'))

# %% [markdown]
# Loading everything we need from `sz_problem` and also set our default plotting and output preferences.

# %%
# %%px
import fenics_sz.utils
from fenics_sz.sz_problems.sz_params import allsz_params
from fenics_sz.sz_problems.sz_slab import create_slab, plot_slab
from fenics_sz.sz_problems.sz_geometry import create_sz_geometry
from fenics_sz.sz_problems.sz_steady_dislcreep import SteadyDislSubductionProblem
import numpy as np
import dolfinx as df
import pyvista as pv
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)
import hashlib
import zipfile
import requests
from mpi4py import MPI
comm = MPI.COMM_WORLD
my_rank = comm.rank

# %% [markdown]
# ### Parameters

# %% [markdown]
# We first select the name and resolution scale, `resscale` of the model.
#
# ```{admonition} Resolution
# By default the resolution is low to allow for a quick runtime and smaller website size.  If sufficient computational resources are available set a lower `resscale` to get higher resolutions and results with sufficient accuracy.
# ```
#

# %%
# %%px
name = "08_Costa_Rica"
resscale = 3.0

# %% [markdown]
# Then load the remaining parameters from the global suite.

# %%
# %%px
szdict = allsz_params[name]
if my_rank == 0:
    print("{}:".format(name))
    print("{:<20} {:<10}".format('Key','Value'))
    print("-"*85)
    for k, v in allsz_params[name].items():
        if v is not None and k not in ['z0', 'z15']: print("{:<20} {}".format(k, v))

# %% [markdown]
# Any of these can be modified in the dictionary.
#
# Several additional parameters can be modified, for details see the documentation for the `SteadyDislSubductionProblem` class.

# %%
# %%px
if my_rank == 0: help(SteadyDislSubductionProblem.__init__)

# %% [markdown]
# ### Setup

# %% [markdown]
# Setup a slab.

# %%
# %%px
slab = create_slab(szdict['xs'], szdict['ys'], resscale, szdict['lc_depth'])
if my_rank == 0: _ = plot_slab(slab)

# %% [markdown]
# Create the subduction zome geometry around the slab.

# %%
# %%px
geom = create_sz_geometry(slab, resscale, szdict['sztype'], szdict['io_depth'], szdict['extra_width'], 
                             szdict['coast_distance'], szdict['lc_depth'], szdict['uc_depth'])
if my_rank == 0: _ = geom.plot()

# %% [markdown]
# Finally, declare the `SubductionZone` problem class using the dictionary of parameters.

# %%
# %%px
sz = SteadyDislSubductionProblem(geom, **szdict)

# %% [markdown]
# ### Solve

# %% [markdown]
# Solve using a dislocation creep rheology and assuming a steady state.

# %%
# %%px
sz.solve()

# %% [markdown]
# ### Plot

# %% [markdown]
# Plot the solution.

# %%
# %%px
plotter = pv.Plotter()
fenics_sz.utils.plot.plot_scalar(sz.T_i, plotter=plotter, scale=sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
fenics_sz.utils.plot.plot_vector_glyphs(sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz.v0))
fenics_sz.utils.plot.plot_vector_glyphs(sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz.v0))
geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = slab.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
if my_rank == 0: 
    fenics_sz.utils.plot.plot_show(plotter)
    fenics_sz.utils.plot.plot_save(plotter, output_folder / "{}_ss_solution_resscale_{:.2f}.png".format(name, resscale))

# %% [markdown]
# Save it to disk so that it can be examined with other visualization software (e.g. [Paraview](https://www.paraview.org/)).

# %%
# %%px
filename = output_folder / "{}_ss_solution_resscale_{:.2f}.bp".format(name, resscale)
with df.io.VTXWriter(sz.mesh.comm, filename, [sz.T_i, sz.vs_i, sz.vw_i]) as vtx:
    vtx.write(0.0)
# zip the .bp folder so that it can be downloaded from jupyter lab
if my_rank == 0: shutil.make_archive(str(filename), 'zip', root_dir=str(filename.parent), base_dir=str(filename.name))

# %% [markdown]
# ## Comparison

# %% [markdown]
# Compare to the published result from [Wilson & van Keken, PEPS, 2023 (II)](http://dx.doi.org/10.1186/s40645-023-00588-6) and [van Keken & Wilson, PEPS, 2023 (III)](https://doi.org/10.1186/s40645-023-00589-5).  The original models used in these papers are also available as open-source repositories on [github](https://github.com/cianwilson/vankeken_wilson_peps_2023) and [zenodo](https://doi.org/10.5281/zenodo.7843967).
#
# First download the minimal necessary data from zenodo and check it is the right version.

# %%
# %%px
zipfilename = pathlib.Path(os.path.join(basedir, os.path.pardir, os.path.pardir, os.path.pardir, "data", "vankeken_wilson_peps_2023_TF_lowres_minimal.zip"))
# only one process should download the data
if my_rank == 0:
    if not zipfilename.is_file():
        zipfileurl = 'https://zenodo.org/records/13234021/files/vankeken_wilson_peps_2023_TF_lowres_minimal.zip'
        r = requests.get(zipfileurl, allow_redirects=True)
        open(zipfilename, 'wb').write(r.content)
# wait until rank 0 has downloaded the data
comm.barrier()
assert hashlib.md5(open(zipfilename, 'rb').read()).hexdigest() == 'a8eca6220f9bee091e41a680d502fe0d'

# %%
# %%px
tffilename = os.path.join('vankeken_wilson_peps_2023_TF_lowres_minimal', 'sz_suite_ss', szdict['dirname']+'_minres_2.00.vtu')
tffilepath = os.path.join(basedir, os.path.pardir, os.path.pardir, os.path.pardir, 'data')
# one process should extract the file
if my_rank == 0:
    with zipfile.ZipFile(zipfilename, 'r') as z:
        z.extract(tffilename, path=tffilepath)
# other processes should wait
comm.barrier()

# %%
# %%px
fxgrid = fenics_sz.utils.plot.grids_scalar(sz.T_i)[0]

tfgrid = pv.get_reader(os.path.join(tffilepath, tffilename)).read()

diffgrid = fenics_sz.utils.plot.pv_diff(fxgrid, tfgrid, field_name_map={'T':'Temperature::PotentialTemperature'}, pass_point_data=True)

# %%
# %%px
# first gather the data onto rank 0
diffgrid_g = fenics_sz.utils.plot.pyvista_grids(diffgrid.cells, diffgrid.celltypes, diffgrid.points, comm, gather=True)
T_g = comm.gather(diffgrid.point_data['T'], root=0)
for r, grid in enumerate(diffgrid_g):
    grid.point_data['T'] = T_g[r]
    grid.set_active_scalars('T')
    grid.clean(tolerance=1.e-2)
diffgrid_g

# then plot it
plotter_diff = pv.Plotter()
clim = None
for grid in diffgrid_g: 
    plotter_diff.add_mesh(grid, cmap='coolwarm', clim=clim, scalar_bar_args={'title': 'Temperature Difference (deg C)', 'bold':True})
geom.pyvistaplot(plotter=plotter_diff, color='green', width=2)
cdpt = slab.findpoint('Slab::FullCouplingDepth')
#fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter_diff, render_points_as_spheres=True, point_size=10.0, color='green')
plotter_diff.enable_parallel_projection()
plotter_diff.view_xy()
if my_rank == 0: plotter_diff.show()

# %%
# %%px
integrated_data = diffgrid.integrate_data()
totalarea = comm.allreduce(integrated_data['Area'][0], op=MPI.SUM)
error = comm.allreduce(integrated_data['T'][0], op=MPI.SUM)/totalarea
if my_rank == 0: print("Average error = {}".format(error,))
assert np.abs(error) < 5

# %%
