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
data_folder = pathlib.Path(os.path.join(basedir, "data"))
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)
import json

# %%
with open(data_folder/"maxtimes_1.json", "r") as f:
    maxtimes_1 = json.load(f)
with open(data_folder/"maxtimes_2.json", "r") as f:
    maxtimes_2 = json.load(f)
with open(data_folder/"maxtimes_mumps_1.json", "r") as f:
    maxtimes_mumps_1 = json.load(f)
with open(data_folder/"maxtimes_mumps_2.json", "r") as f:
    maxtimes_mumps_2 = json.load(f)
with open(data_folder/"extradiag_1.json", "r") as f:
    extradiag_1 = json.load(f)
with open(data_folder/"extradiag_2.json", "r") as f:
    extradiag_2 = json.load(f)

# %%
maxtimes_1.keys(), maxtimes_mumps_1.keys(), maxtimes_mumps_1['Direct Stokes'][0].keys()

# %%
extradiag_1['Direct Stokes'][0].keys()

# %%
import matplotlib.font_manager

# %%
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 10
handlelength = 4
mpl.rcParams['legend.handlelength'] = handlelength
fontsize = 9
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['figure.labelsize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize+2
mpl.rcParams['axes.titlesize'] = fontsize+2
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
# mpl.rcParams['font.family'] = ['sans-serif']
# mpl.rcParams['font.sans-serif'] = ['Arial']

# the list of the number of processors we will use
nprocs_scale = [1, 2, 3, 4, 6, 8]

# We are interested in the time to create the mesh,
# declare the functions, assemble the problem and solve it.
# From our implementation in `solve_poisson_2d` it is also
# possible to request the time to declare the Dirichlet and
# Neumann boundary conditions and the forms.
allsteps = [('Assemble Temperature', 'Temperature assembly'), \
         ('Assemble Stokes', 'Stokes assembly'), \
         ('Solve Temperature', 'Temperature solve'), \
         ('Solve Stokes', 'Stokes solve')]

steps = [('Solve Stokes', 'Stokes solve')]
Tsteps = [('Solve Temperature', 'Temperature solve')]

mumpssteps = [('MUMPS analysis', 'MUMPS analysis'), ('MUMPS factorization', 'MUMPS factorization')]#, ('MUMPS solve', 'MUMPS solve')]

versions = [('Direct Stokes nosplit', 'direct, no split'), ('Direct Stokes', 'direct, split'), ('Iterative Stokes', 'iterative, split')]

# set up a figure for plotting
scale = 0.75
nplots = len(steps)+3
fig, axs = pl.subplots(nrows=nplots, ncols=2, figsize=[scale*2*6.4,scale*4.8*nplots], sharex=True, layout='constrained')

lss = ['--', '-', '-.', ':']
markers = ['^', 'P', 's', 'o', 'X']
cs = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 
      'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

key, name = versions[1]
alss = ['--', '-.', ':', '-']
row0 = 1
dofrow = 0
for j, (step, sname) in enumerate(allsteps):
    s = allsteps.index((step, sname))
    axs[row0][0].plot(nprocs_scale, [t[s] for t in maxtimes_1[key]], markers[j]+lss[j], c=cs[1], label='{} ({})'.format(sname, name))
    # axs[row0][0].legend()
    axs[row0][1].plot(nprocs_scale, [t[s] for t in maxtimes_2[key]], markers[j]+lss[j], c=cs[1], label='{} ({})'.format(sname, name))

axs[row0][0].set_ylabel('(i) wall time (s)')
axs[row0][1].legend(handlelength=handlelength)

for i, (key, name) in enumerate(versions):
    for j, (step, sname) in enumerate(steps):
        s = allsteps.index((step, sname))
        for m, maxtimes in enumerate([maxtimes_1, maxtimes_2]):
            axs[j+row0+1][m].plot(nprocs_scale, [t[s] for t in maxtimes[key]], markers[3]+lss[i], c=cs[i], label='{} ({})'.format(sname, name))
            axs[row0+j+2][m].plot(nprocs_scale, maxtimes[key][0][s]/np.asarray([t[s] for t in maxtimes[key]]), markers[3]+lss[i], c=cs[i], label='{} ({})'.format(sname, name))
        for m, maxtimes_mumps in enumerate([maxtimes_mumps_1, maxtimes_mumps_2]):
            for mi, (mstep, msname) in enumerate(mumpssteps):
                if key in maxtimes_mumps:
                    axs[j+row0+1][m].plot(nprocs_scale, [lmumpsts[mstep] for lmumpsts in maxtimes_mumps[key]], markers[mi]+lss[i], c=cs[i], label='{} ({})'.format(msname, name))
                    axs[row0+j+2][m].plot(nprocs_scale, maxtimes_mumps[key][0][mstep]/np.asarray([lmumpsts[mstep] for lmumpsts in maxtimes_mumps[key]]), markers[mi]+lss[i], c=cs[i], label='{} ({})'.format(msname, name))

# for i, (key, name) in enumerate(versions[:-1]):
#     for j, (step, sname) in enumerate(Tsteps):
#         s = allsteps.index((step, sname))
#         for m, maxtimes in enumerate([maxtimes_1, maxtimes_2]):
#             axs[j+row0+1][m].plot(nprocs_scale, [t[s] for t in maxtimes[key]], markers[2]+lss[i], c=cs[i], label='{} ({})'.format(sname, name))
#             axs[row0+j+2][m].plot(nprocs_scale, maxtimes[key][0][s]/np.asarray([t[s] for t in maxtimes[key]]), markers[2]+lss[i], c=cs[i], label='{} ({})'.format(sname, name))


ymin = max(axs[row0+2][0].get_ylim()[0], axs[row0+2][1].get_ylim()[0])
ymax = max(axs[row0+2][0].get_ylim()[1], axs[row0+2][1].get_ylim()[1])
axs[row0+2][0].plot(nprocs_scale, [nproc/nprocs_scale[0] for nproc in nprocs_scale], 'k:', label='Ideal')
ideal = axs[row0+2][1].plot(nprocs_scale, [nproc/nprocs_scale[0] for nproc in nprocs_scale], 'k:', label='Ideal')[0]
axs[row0+2][0].set_ylim(ymin, ymax)
axs[row0+2][1].set_ylim(ymin, ymax)

axs[row0+1][0].set_ylim(-0.8, axs[row0+1][0].get_ylim()[1])
# axs[row0+1][1].set_ylim(axs[row0+1][1].get_ylim()[0], 2.5*axs[row0+1][1].get_ylim()[1])

axs[row0+1][0].set_ylabel('(ii) wall time (s)')
axs[row0+2][0].set_ylabel('(iii) speed up')
axs[row0+1][1].legend(handlelength=handlelength)
axs[row0+2][1].legend([ideal], ['Ideal'], handlelength=handlelength)

### DOFs and ghosts

if nplots == len(steps)+3:

    T_ndofs_avg = np.asarray([extradiag['T_ndofs_avg'] for extradiag in extradiag_2['Direct Stokes']])
    T_ndofs_min = np.asarray([extradiag['T_ndofs_min'] for extradiag in extradiag_2['Direct Stokes']])
    T_ndofs_max = np.asarray([extradiag['T_ndofs_max'] for extradiag in extradiag_2['Direct Stokes']])
    T_ndofs_errs = [np.abs(T_ndofs_min-T_ndofs_avg), np.abs(T_ndofs_max-T_ndofs_avg)]

    T_nghosts_sum = np.asarray([extradiag['T_nghosts_sum'] for extradiag in extradiag_2['Direct Stokes']])
    T_nghosts_nosplit_sum = np.asarray([extradiag['T_nghosts_sum'] for extradiag in extradiag_2['Direct Stokes nosplit']])

    v_wedge_ndofs_min = np.asarray([extradiag['v_wedge_ndofs_min'] for extradiag in extradiag_2['Direct Stokes']])
    v_wedge_ndofs_max = np.asarray([extradiag['v_wedge_ndofs_max'] for extradiag in extradiag_2['Direct Stokes']])
    v_wedge_ndofs_avg = 0.5*(v_wedge_ndofs_min + v_wedge_ndofs_max)
    v_wedge_ndofs_errs = [np.abs(v_wedge_ndofs_min-v_wedge_ndofs_avg), np.abs(v_wedge_ndofs_max-v_wedge_ndofs_avg)]

    v_slab_ndofs_min = np.asarray([extradiag['v_slab_ndofs_min'] for extradiag in extradiag_2['Direct Stokes']])
    v_slab_ndofs_max = np.asarray([extradiag['v_slab_ndofs_max'] for extradiag in extradiag_2['Direct Stokes']])
    v_slab_ndofs_avg = 0.5*(v_slab_ndofs_min + v_slab_ndofs_max)
    v_slab_ndofs_errs = [np.abs(v_slab_ndofs_min-v_slab_ndofs_avg), np.abs(v_slab_ndofs_max-v_slab_ndofs_avg)]

    T_ndofs_nosplit_avg = np.asarray([extradiag['T_ndofs_avg'] for extradiag in extradiag_2['Direct Stokes nosplit']])
    T_ndofs_nosplit_min = np.asarray([extradiag['T_ndofs_min'] for extradiag in extradiag_2['Direct Stokes nosplit']])
    T_ndofs_nosplit_max = np.asarray([extradiag['T_ndofs_max'] for extradiag in extradiag_2['Direct Stokes nosplit']])
    T_ndofs_nosplit_errs = [np.abs(T_ndofs_nosplit_min-T_ndofs_nosplit_avg), np.abs(T_ndofs_nosplit_max-T_ndofs_nosplit_avg)]

    lines = []
    lines.append(axs[dofrow][0].errorbar(nprocs_scale, T_ndofs_avg, yerr=T_ndofs_errs, c=cs[-1], linestyle='-', marker='s', label='Temperature, DOFs/process (split)')[0])
    lines[-1].set_label('Temperature, DOFs/process (split)')
    # axs[dofrow][0].fill_between(nprocs_scale, T_ndofs_min, T_ndofs_max, color='tab:blue', alpha=0.5)

    # lines.append(axs[dofrow][0].errorbar(nprocs_scale, T_ndofs_nosplit_avg, yerr=T_ndofs_nosplit_errs, c='k', linestyle='--', marker='o', label='DOFs')[0])
    # # lines[-1].set_label('DOFs')
    # # axs[dofrow][0].fill_between(nprocs_scale, T_ndofs_nosplit_min, T_ndofs_nosplit_max, color='tab:orange', alpha=0.5)

    lines.append(axs[dofrow][0].errorbar(nprocs_scale, v_wedge_ndofs_avg, yerr=v_wedge_ndofs_errs, c=cs[-2], linestyle='-', marker='D', label='Velocity wedge, DOFs/process (split)')[0])
    lines[-1].set_label('Velocity wedge, DOFs/process (split)')
    lines.append(axs[dofrow][0].errorbar(nprocs_scale, v_slab_ndofs_avg, yerr=v_slab_ndofs_errs, c=cs[-3], linestyle='-', marker='D', label='Velocity slab, DOFs/process (split)')[0])
    lines[-1].set_label('Velocity slab, DOFs/process (split)')

    axs[dofrow][0].set_ylim(-20000, axs[dofrow][0].get_ylim()[1])
    axs[dofrow][0].legend(lines, [line.get_label() for line in lines], handlelength=handlelength)

    T_nghosts_sum = np.asarray([extradiag['T_nghosts_sum'] for extradiag in extradiag_2['Direct Stokes']])
    v_wedge_nghosts_sum = np.asarray([extradiag['v_wedge_nghosts_sum'] for extradiag in extradiag_2['Direct Stokes']])
    v_slab_nghosts_sum = np.asarray([extradiag['v_slab_nghosts_sum'] for extradiag in extradiag_2['Direct Stokes']])
    lines = []
    lines.append(axs[dofrow][1].plot(nprocs_scale, T_nghosts_sum, c=cs[-1], ls='-', marker='s', label='Temperature, ghosts (split)')[0])
    lines.append(axs[dofrow][1].plot(nprocs_scale, T_nghosts_nosplit_sum, c=cs[-1], ls=':', marker='s', label='Temperature, ghosts (no split)')[0])
    lines.append(axs[dofrow][1].plot(nprocs_scale, v_wedge_nghosts_sum, c=cs[-2], ls='-', marker='D', label='Velocity wedge, ghosts (split)')[0])
    lines.append(axs[dofrow][1].plot(nprocs_scale, v_slab_nghosts_sum, c=cs[-3], ls='-', marker='D', label='Velocity slab, ghosts (split)')[0])

    axs[dofrow][1].legend(lines, [line.get_label() for line in lines], handlelength=handlelength)

    axs[dofrow][0].set_ylabel('no. DOFs')

for j in range(len(axs)):
    axs[j][0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[j][1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[j][0].grid(axis='y')
    axs[j][1].grid(axis='y')
axs[len(axs)-1][0].set_xlabel('number processors')
axs[len(axs)-1][1].set_xlabel('number processors')

axs[row0][0].set_title('(c) isoviscous rheology', loc='left')
axs[row0][1].set_title('(d) non-linear rheology', loc='left')
axs[dofrow][0].set_title('(a) degrees of freedom (DOFs)/process', loc='left')
axs[dofrow][1].set_title('(b) total no. ghosts', loc='left')

fig.savefig(output_folder/"strong_scaling.pdf", dpi=400)

# %%

# %%

# %%
import ipyparallel as ipp
nprocs = 8
rc = ipp.Cluster(engine_launcher_class="mpi", n=nprocs).start_and_connect_sync()

# %%
# %%px
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
path = os.path.join(basedir, os.path.pardir, os.path.pardir, 'python')
sys.path.insert(0, path)

from fenics_sz.sz_problems.sz_benchmark import solve_benchmark_case2

def setup_benchmark_case2(resscale, partition_by_region=True):
    from fenics_sz.sz_problems.sz_slab import create_slab
    from fenics_sz.sz_problems.sz_geometry import create_sz_geometry
    from fenics_sz.sz_problems.sz_steady_dislcreep import SteadyDislSubductionProblem
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

    return sz

szhr = solve_benchmark_case2(resscale=0.5, partition_by_region=True)
szlr = setup_benchmark_case2(resscale=2, partition_by_region=True)

# %%
# %%px
import dolfinx as df
import numpy as np
from mpi4py import MPI
tdim = szhr.mesh.topology.dim
num_cells = szhr.mesh.topology.index_map(tdim).size_local
rank_tags = df.mesh.meshtags(szhr.mesh, tdim, np.arange(num_cells), szhr.mesh.comm.rank*np.ones(num_cells, dtype=np.int32))
rank_tags.name = 'rank'

# %%
# %%px

values = np.zeros_like(szhr.cell_tags.values)
for i in range(num_cells):
    if szhr.cell_tags.values[i] in szhr.wedge_rids:
        values[i] = 1
    elif szhr.cell_tags.values[i] in szhr.crust_rids:
        values[i] = 2

region_tags = df.mesh.meshtags(szhr.mesh, tdim, np.arange(num_cells), values)
region_tags.name = 'region'

# %%
520/768

# %%
# %%px
import fenics_sz.utils.plot
cs = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
      'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
import pyvista as pv
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
pv.global_theme.font.family = 'arial'
fontsize = 14
sbar_fontsize = 18


zoom = 1.65
nrows = 2
ncols = 2
windowsize = [1000, 520]
xscale = windowsize[0]/1024
yscale = windowsize[1]/768
plotter = pv.Plotter(shape=(nrows,ncols), border=False, window_size=[windowsize[0]*nrows, windowsize[1]*ncols])


# %%
# %%px

# plotter = pv.Plotter(window_size=windowsize)

plotter.subplot(0,0)

fenics_sz.utils.plot.plot_mesh(szhr.mesh, plotter=plotter, tags=region_tags, show_edges=False, line_width=1, 
                                gather=True, cmap='gray', clim=[-2,2], show_scalar_bar=False, lighting=False, opacity=0.6)#,
                                # scalar_bar_args={'n_labels': 0, 'title': 'regions', 'width':0.6, 'position_x':0.205, 'position_y':0.1})
szhr.geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = szhr.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, 
                                 render_points_as_spheres=True, 
                                 point_size=10.0, color='green')
plotter.add_text('slab', (xscale*300, yscale*300), color='green', font_size=fontsize)
plotter.add_text('wedge', (xscale*750, yscale*400), color='green', font_size=fontsize)
plotter.add_text('upper crust', (xscale*500, yscale*625), color='green', font_size=fontsize)
plotter.add_text('lower crust', (xscale*502, yscale*565), color='green', font_size=fontsize)
plotter.add_text('coupling depth', (xscale*450, yscale*430), color='green', font_size=fontsize)
plotter.add_text('(a) domain', (xscale*120, yscale*700), font_size=fontsize)
# plotter.enable_parallel_projection()
# plotter.view_xy()
# plotter.show_bounds(show_xlabels=True, bold=False, font_size=sbar_fontsize, use_2d=True, use_3d_text=False, ytitle="km", xtitle="km", n_xlabels=5, n_ylabels=2)
plotter.show_bounds(show_zlabels=True, bold=False, font_size=8, use_3d_text=False, ytitle="km", xtitle="km", n_xlabels=2, n_ylabels=2)
plotter.camera.zoom(zoom)


# if szhr.mesh.comm.rank==0: 
#     fenics_sz.utils.plot.plot_show(plotter)
#     fenics_sz.utils.plot.plot_save(plotter, output_folder / "benchmark_geometry.png")
#     fenics_sz.utils.plot.plot_save_graphic(plotter, output_folder / "benchmark_geometry.pdf")



# %%
# %%px
import copy

# plotter = pv.Plotter(window_size=windowsize)

plotter.subplot(1,0)


fenics_sz.utils.plot.plot_scalar(szhr.T_i, plotter=plotter, scale=szhr.T0, gather=True, lighting=False, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'width':0.6, 'position_x':0.205, 'position_y':0, 'title_font_size':sbar_fontsize, 'label_font_size': sbar_fontsize})
bounds = copy.deepcopy(plotter.bounds)
if szhr.mesh.comm.rank==0: print(plotter.bounds)
fenics_sz.utils.plot.plot_vector_glyphs(szhr.vw_i, plotter=plotter, gather=True, factor=0.2, tolerance=0.08, color='k', scale=fenics_sz.utils.mps_to_mmpyr(szhr.v0))
fenics_sz.utils.plot.plot_vector_glyphs(szhr.vs_i, plotter=plotter, gather=True, factor=0.2, tolerance=0.08, color='k', scale=fenics_sz.utils.mps_to_mmpyr(szhr.v0))
if szhr.mesh.comm.rank==0: print(plotter.bounds)
plotter.reset_camera(render=False, bounds=bounds)
if szhr.mesh.comm.rank==0: print(plotter.bounds)
szhr.geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = szhr.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, 
                                 point_size=10.0, color='green')
plotter.add_text('(c) solution', (xscale*120, yscale*700), font_size=fontsize)
plotter.camera.zoom(zoom)

# if szhr.mesh.comm.rank==0: 
#     fenics_sz.utils.plot.plot_show(plotter)
#     fenics_sz.utils.plot.plot_save(plotter, output_folder / "benchmark_solution.png")
#     fenics_sz.utils.plot.plot_save_graphic(plotter, output_folder / "benchmark_solution.pdf")

# %%
# %%px

# plotter = pv.Plotter(window_size=windowsize)

plotter.subplot(0,1)


fenics_sz.utils.plot.plot_mesh(szlr.mesh, plotter=plotter, show_edges=True, line_width=1, 
                                              gather=True, lighting=False, color='white')
szlr.geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = szlr.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, 
                                 point_size=10.0, color='green')
plotter.add_text('(b) mesh', (xscale*120, yscale*700), font_size=fontsize)
plotter.camera.zoom(zoom)

# if szhr.mesh.comm.rank==0: 
#     fenics_sz.utils.plot.plot_show(plotter)
#     fenics_sz.utils.plot.plot_save(plotter, output_folder / "benchmark_mesh.png")
#     fenics_sz.utils.plot.plot_save_graphic(plotter, output_folder / "benchmark_mesh.pdf")


# %%
# %%px

# plotter = pv.Plotter(window_size=windowsize)

plotter.subplot(1,1)


fenics_sz.utils.plot.plot_mesh(szhr.mesh, plotter=plotter, tags=rank_tags, show_edges=False, line_width=1, lighting=False, 
                                gather=True, cmap=cs[:szhr.mesh.comm.size], show_scalar_bar=True,
                                scalar_bar_args={'n_labels': 0, 'title': 'rank', 'width':0.6, 'position_x':0.205, 'position_y':0, 'title_font_size':sbar_fontsize, 'label_font_size': sbar_fontsize})
szhr.geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = szhr.geom.slab_spline.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, 
                                 point_size=10.0, color='green')
plotter.add_text('(d) partitioning', (xscale*120, yscale*700), font_size=fontsize)
# plotter.show_bounds(show_zlabels=True, bold=False, font_size=fontsize, ytitle="km", xtitle="km", n_xlabels=5, n_ylabels=2)
plotter.camera.zoom(zoom)

# if szhr.mesh.comm.rank==0: 
#     fenics_sz.utils.plot.plot_show(plotter)
#     fenics_sz.utils.plot.plot_save(plotter, output_folder / "benchmark_partitioning.png")
#     fenics_sz.utils.plot.plot_save_graphic(plotter, output_folder / "benchmark_partitioning.pdf")

# %%
# %%px
if szhr.mesh.comm.rank==0: 
    fenics_sz.utils.plot.plot_show(plotter)
    fenics_sz.utils.plot.plot_save(plotter, output_folder / "benchmark_tiled.png")
    fenics_sz.utils.plot.plot_save_graphic(plotter, output_folder / "benchmark_tiled.pdf")

# %%

# %%

# %%

# %%
rc.shutdown()

# %%
