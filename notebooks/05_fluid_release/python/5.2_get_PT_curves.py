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
# ## Solve and get PT curves

# %% [markdown]
# In this notebook, we'll solve a couple subduction problem from the global suite, and develop tools to plot and probe the temperature solution along paths of interest along a slab. The paths of interest will be paths parallel to the slab surface at varying distances away such that each layer of the slab is appropriately probed. We'll also build in functionality now to change how far into the mantle we want to probe, which will be useful later for exploring the effect of different thicknesses of the serpentinized mantle on the global water budget.

# %% [markdown]
# Start with the usual imports and file path settings:

# %%
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))

# %%
import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as pl
import matplotlib.image as img
import subprocess
import pathlib
import pyvista as pv
import copy

import fenics_sz.utils
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)

# %%
from fenics_sz.sz_problems.sz_slab import create_slab, plot_slab
from fenics_sz.sz_problems.sz_geometry import create_sz_geometry
from fenics_sz.sz_problems.sz_steady_dislcreep import SteadyDislSubductionProblem
from fenics_sz.sz_problems.sz_tdep_dislcreep import TDDislSubductionProblem
from fenics_sz.sz_problems.sz_params import default_params, allsz_params

# %% [markdown]
# Set resolution for the FEM solver

# %%
resscale = 5.0

# %% [markdown]
# Printing the list of subduction zones in the global suite, to recall their names

# %%
print("\n".join(allsz_params))

# %% [markdown]
# ### Tonga

# %% [markdown]
# We'll solve one of these subduction problems now, so that we can use it to test the temperature probing functions that we develop.

# %% [markdown]
# #### Solve SZ problem (case 2; TD)

# %%
tonga_sz_dict = allsz_params["35_Tonga"]

# %% [markdown]
# As a reminder, we can look at the given parameters for the Tonga SZ problem:

# %%
for key, value in tonga_sz_dict.items():
    print(f"{key}, : {value}")


# %% [markdown]
# Create and plot the slab:

# %%
tonga_slab = create_slab(tonga_sz_dict['xs'], tonga_sz_dict['ys'], resscale, tonga_sz_dict['lc_depth'])

# %%
plot_slab(tonga_slab)

# %% [markdown]
# Now create and solve the subduction zone problem:

# %%
tonga_geom = create_sz_geometry(tonga_slab, resscale, tonga_sz_dict['sztype'], tonga_sz_dict['io_depth'], tonga_sz_dict['extra_width'], 
                             tonga_sz_dict['coast_distance'], tonga_sz_dict['lc_depth'], tonga_sz_dict['uc_depth'])
tonga_sz = TDDislSubductionProblem(tonga_geom, **tonga_sz_dict)

# %%
tonga_sz.solve(tonga_sz_dict['As'], dt=0.05, theta=0.5, rtol=1.e-1, verbosity=1)

# %% [markdown]
# Plot the solution.
# Comment and uncomment certain lines according to what you want to plot / whether or not you want to save the plot.

# %%
plotter = pv.Plotter()
fenics_sz.utils.plot.plot_scalar(tonga_sz.T_i, plotter=plotter, scale=tonga_sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
# fenics_sz.utils.plot.plot_vector_glyphs(tonga_sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(tonga_sz.v0))
# fenics_sz.utils.plot.plot_vector_glyphs(tonga_sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(tonga_sz.v0))
tonga_geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = tonga_slab.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter)
# fenics_sz.utils.plot.plot_save(plotter, output_folder / "{}_td_solution_resscale_{:.2f}.png".format("Tonga", resscale))

# %% [markdown]
# #### Probing paths along slab

# %% [markdown]
# For the hydration modeling, we'll be examining the temperatures of many paths parallel to the curve of our slabs. Therefore, we will now define a function that, given a subduction problem, will generate paths of interest and return their temperatures.

# %%
#add a resolution parameter? thickness with which to discretize the probed layers?
#add thickness params for other layers besides hydrated mantle?
#add ability to probe an array of custom depths?

def probe(sz, depth, h_serp=None):

    """
    Arguements:
    * sz    - the solved sz problem
    * depth - the layer of the slab to probe temperature of.
            Must be a name in the list specified below

    kwargs: 
    * h_serp - thickness of the serpentinized mantle. deafaults to None.
            Must be provided if probing the mantle!
    """

    assert (depth in ["sediments", "upper_volc", "lower_volc", "dikes", "gabbro", "mantle"] or type(depth) == float)
    
    if depth == "sediments":
        slabpoints = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in sz.geom.slab_spline.interpcurves])
        cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(slabpoints, sz.mesh)
        return (sz.T_i.eval(slabpoints, cells)[:,0], -slabpoints[:,1])
    
    elif depth == "upper_volc":
        upper_volcs = copy.deepcopy(sz.geom.slab_spline)
         # considering an upper volcanics layer that's 300m thick, so we want to get a path
         # 150m normal to the slab surface, cropped such that it fits w/n the problem's geometry
        upper_volcs.translatenormalandcrop(-0.15)
        probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in upper_volcs.interpcurves])
        cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(probe_points, sz.mesh)
        return (sz.T_i.eval(probe_points, cells)[:,0], -probe_points[:,1])
    
    elif depth == "lower_volc":
        lower_volcs = copy.deepcopy(sz.geom.slab_spline)
        lower_volcs.translatenormalandcrop(-0.45)
        probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in lower_volcs.interpcurves])
        cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(probe_points, sz.mesh)
        return (sz.T_i.eval(probe_points, cells)[:,0], -probe_points[:,1])

    elif depth == "dikes":
        results = []
        i=1
        while i < 4:
            dikes = copy.deepcopy(sz.geom.slab_spline)
            dikes.translatenormalandcrop(-0.6 - 0.35*i)
            probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in dikes.interpcurves])
            cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(probe_points, sz.mesh)
            results.append([sz.T_i.eval(probe_points, cells)[:,0], -probe_points[:,1]])
            i+=1
        return results

    elif depth == "gabbro":
        results = []
        i=1
        while i < 11:
            gabbro = copy.deepcopy(sz.geom.slab_spline)
            gabbro.translatenormalandcrop(-2 - 0.5*i)
            probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in gabbro.interpcurves])
            cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(probe_points, sz.mesh)
            results.append([sz.T_i.eval(probe_points, cells)[:,0], -probe_points[:,1]])
            i+=1
        return results
    
    elif depth == "mantle":
        assert h_serp is not None
        results = []
        i=1
        while i < h_serp//0.5 + 1:
            mantle = copy.deepcopy(sz.geom.slab_spline)
            mantle.translatenormalandcrop(-7 - 0.5*i)
            probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in mantle.interpcurves])
            cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(probe_points, sz.mesh)
            results.append([sz.T_i.eval(probe_points, cells)[:,0], -probe_points[:,1]])
            i+=1
        return results


# %%
slabpoints = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in tonga_sz.geom.slab_spline.interpcurves])
cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(slabpoints, tonga_sz.mesh)

print("cell indices")
print(cinds)
print("cells")
print(cells)
print("Temperatures, depths")
print(tonga_sz.T_i.eval(slabpoints, cells)[:,0], -slabpoints[:,1])

# %%
print((tonga_sz.x))


# %% [markdown]
# Lets also add a way to plot all of this temperature data:
# This function is almost identical to the previous one, except now instead of returning the data, the temperatures of each point are being plotted as a function of the point's depth.

# %%
# development notes:
# legend generator is maybe a bit strange

def plot_geotherms(sz, sz_name, h_serp, legend=False, save=False):

    """
    Arguements:
    * sz        - the solved sz problem
    * sz_name   - name of the subduction zone, used for plot title and file naming
    * h_serp    - thickness (depth) of serpentinized mantle under the slab

    kwargs: 
    * legend - color-coded legend is generated if set to true
    * save   - plot is saved to output folder if set to true
    """

    fig = pl.figure()
    ax = fig.gca()

    slabpoints = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in sz.geom.slab_spline.interpcurves])
    cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(slabpoints, sz.mesh)
    ax.plot(sz.T_i.eval(slabpoints, cells)[:,0], -slabpoints[:,1], color = 'orange', label = "Sediments")
    
    upper_volcs = copy.deepcopy(sz.geom.slab_spline)
    upper_volcs.translatenormalandcrop(-0.15) # get a path 7km normal to the slab surface, cropped such that it fits w/n the problem's geometry
    probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in upper_volcs.interpcurves])
    cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(probe_points, sz.mesh)
    ax.plot(sz.T_i.eval(probe_points, cells)[:,0], -probe_points[:,1], color = 'gray', label = "Upper volcanics")


    lower_volcs = copy.deepcopy(sz.geom.slab_spline)
    lower_volcs.translatenormalandcrop(-0.45)
    probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in lower_volcs.interpcurves])
    cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(probe_points, sz.mesh)
    ax.plot(sz.T_i.eval(probe_points, cells)[:,0], -probe_points[:,1], color = 'black', label = "Lower volcanics")


    i=1
    while i < 4:
        dikes = copy.deepcopy(sz.geom.slab_spline)
        dikes.translatenormalandcrop(-0.6 - 0.35*i)
        probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in dikes.interpcurves])
        cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(probe_points, sz.mesh)
        ax.plot(sz.T_i.eval(probe_points, cells)[:,0], -probe_points[:,1], color = 'b', label = "Dikes")
        i+=1

    i=1
    while i < 11:
        gabbro = copy.deepcopy(sz.geom.slab_spline)
        gabbro.translatenormalandcrop(-2 - 0.5*i)
        probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in gabbro.interpcurves])
        cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(probe_points, sz.mesh)
        ax.plot(sz.T_i.eval(probe_points, cells)[:,0], -probe_points[:,1], color = 'r', label = "Gabbro")
        i+=1

    i=1
    while i < h_serp//0.5 + 1:
        mantle = copy.deepcopy(sz.geom.slab_spline)
        mantle.translatenormalandcrop(-7 - 0.5*i)
        probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in mantle.interpcurves])
        cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(probe_points, sz.mesh)
        ax.plot(sz.T_i.eval(probe_points, cells)[:,0], -probe_points[:,1], color = 'g', label = "Serpentinized mantle")
        i+=1

    if legend == True:
        handles, labels = ax.get_legend_handles_labels()
        indexes = [labels.index(x) for x in set(labels)]
        new_handles = []
        new_labels = []
        for i in indexes:
            new_handles.append(handles[i])
            new_labels.append(labels[i])
        ax.legend(new_handles, new_labels)

    ax.set_xlabel('T ($^\circ$C)')
    ax.set_ylabel('z (km)')
    ax.set_title("{} Geotherms: {:.1f}km-thick Serpentinized Mantle".format(sz_name, h_serp))

    if save == True:
        fig.savefig(output_folder / "{}_geotherms_{:.1f}_km_serp.png".format(sz_name, h_serp))




# %% [markdown]
# Some tests to make sure the probe function works as expected:

# %%
results = probe(tonga_sz, "upper_volc")

# %%
results = probe(tonga_sz, "dikes", 2.0)

# %%
results = probe(tonga_sz, "mantle")
#make sure function doesn't probe mantle w/o a specified mantle depth

# %%
results = probe(tonga_sz, "mantle", h_serp = 2)

# %% [markdown]
# Now we'll use the plotting function to plot the depth-temperature paths of the Tonga slab, considering a 12km thick hydrated mantle.

# %%
plot_geotherms(tonga_sz, "Tonga", 12, legend=True, save=True)

# %% [markdown]
# #### Kamchatka: comparison against Gies et al.

# %% [markdown]
# One paper in the literature that uses similar depth-temperature plots is Gies et al. 2024, which correlates depth to pressue and plots the curves as P-T curves. They consider a 12km thick serpentinized mantle layer, and use the Kamchatka subduction zone as one of their examples, so we'll use our model to generate the same conditions and compare our results against theirs.

# %%
h_serp = 12

# %%
kamchatka_dict = allsz_params["52_Kamchatka"]

# %% [markdown]
# We can see the differences between the default parameters for Tonga and the default parameters for Kamchatka.

# %%
print("Tonga:")
for key, value in tonga_sz_dict.items():
    print(f"{key}, : {value}")

print("Kamchatka")
for key, value in kamchatka_dict.items():
    print(f"{key}, : {value}")

# %% [markdown]
# Create and plot the slab as before:

# %%
kamchatka_slab = create_slab(kamchatka_dict['xs'], kamchatka_dict['ys'], resscale, kamchatka_dict['lc_depth'])
plot_slab(kamchatka_slab)

# %% [markdown]
# Create and solve the subduction problem as before:

# %%
kamchatka_geom = create_sz_geometry(kamchatka_slab, resscale, kamchatka_dict['sztype'], kamchatka_dict['io_depth'], kamchatka_dict['extra_width'], 
                             kamchatka_dict['coast_distance'], kamchatka_dict['lc_depth'], kamchatka_dict['uc_depth'])
kamchatka_sz = TDDislSubductionProblem(kamchatka_geom, **kamchatka_dict)

kamchatka_sz.solve(kamchatka_dict['As'], dt=0.05, theta=0.5, rtol=1.e-1, verbosity=1)

# %% [markdown]
# Plot the solution as a sanity check to ensure everything as run correctly.

# %%
plotter = pv.Plotter()
fenics_sz.utils.plot.plot_scalar(kamchatka_sz.T_i, plotter=plotter, scale=kamchatka_sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
# fenics_sz.utils.plot.plot_vector_glyphs(kamchatka_sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(kamchatka_sz.v0))
# fenics_sz.utils.plot.plot_vector_glyphs(kamchatka_sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(kamchatka_sz.v0))
kamchatka_geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = kamchatka_slab.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter)
# fenics_sz.utils.plot.plot_save(plotter, output_folder / "{}_td_solution_resscale_{:.2f}.png".format("kamchatka", resscale))

# %% [markdown]
# Now we can plot the depth-temperature paths of the kamchatka slab, considering 12km of serpentinized mantle, and compare our results with the P-T curves from Gies et al.

# %%
plot_geotherms(kamchatka_sz, "kamchatka", h_serp, legend=True, save=True)


# %% [markdown]
# Let's also add a function that returns the points that are being probed, so that they can be plotted on top of the subduction zone geometry in order to visualize where we are probing.
#
# This function is very similar to the probing and plotting functions previously defined in this notebook, but this returns arrays that we will use to plot connecting line segments.

# %%
def get_probe_paths(sz, h_serp):

    """

    Arguements:
    * sz        - the solved sz problem
    * h_serp    - thickness (depth) of serpentinized mantle under the slab
    """
    
    probe_paths = []
    upper_volcs = copy.deepcopy(sz.geom.slab_spline)
    upper_volcs.translatenormalandcrop(-0.15) # get a path 7km normal to the slab surface, cropped such that it fits w/n the problem's geometry
    probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in upper_volcs.interpcurves])
    probe_paths.append(probe_points)

    lower_volcs = copy.deepcopy(sz.geom.slab_spline)
    lower_volcs.translatenormalandcrop(-0.45)
    probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in lower_volcs.interpcurves])
    probe_paths.append(probe_points)


    i=1
    while i < 4:
        dikes = copy.deepcopy(sz.geom.slab_spline)
        dikes.translatenormalandcrop(-0.6 - 0.35*i)
        probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in dikes.interpcurves])
        probe_paths.append(probe_points)
        i+=1

    i=1
    while i < 11:
        gabbro = copy.deepcopy(sz.geom.slab_spline)
        gabbro.translatenormalandcrop(-2 - 0.5*i)
        probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in gabbro.interpcurves])
        probe_paths.append(probe_points)
        i+=1

    i=1
    while i < h_serp//0.5 + 1:
        mantle = copy.deepcopy(sz.geom.slab_spline)
        mantle.translatenormalandcrop(-7 - 0.5*i)
        probe_points = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in mantle.interpcurves])
        probe_paths.append(probe_points)
        i+=1

    return(probe_paths)

# %%
kamchatka_paths = get_probe_paths(kamchatka_sz, h_serp)


# %% [markdown]
# We can define a function to use this data to plot a few representative paths, as plotting all of the paths would make them appear as just one thick curve beneath the slab:

# %%
def plot_temp_probes(sz, sz_geom, sz_slab, sz_paths, res=5, filename=None):

    """

    A function that plots some of the paths along which the temperature is being probed
    on top of the subduction zone's geometry.

    Arguements:
    * sz        - the solved sz problem
    * sz_geom   - geometry object representing the subduction zone
    * sz_slab   - the slab around which the subduction zone geometry was built
    * sz_paths  - the paths parallel to the solve where temp. is being probed

    kwargs:
    * res       - int: number of paths to skip before one is plotted
                defaults to 5
    * filename  - plot is saved to output folder if filename is given
    """

    plotter = pv.Plotter()
    fenics_sz.utils.plot.plot_scalar(sz.T_i, plotter=plotter, scale=sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    # fenics_sz.utils.plot.plot_vector_glyphs(sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz.v0))
    # fenics_sz.utils.plot.plot_vector_glyphs(sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(sz.v0))
    sz_geom.pyvistaplot(plotter=plotter, color='green', width=2)
    cdpt = sz_slab.findpoint('Slab::FullCouplingDepth')
    fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')

    for i in range(len(sz_paths)//res):
        for j in range(len(kamchatka_paths[res*i])-1):            
            plotter.add_lines(np.array([sz_paths[res*i][j], sz_paths[res*i][j+1]]), color = 'w', width = 0.1)

    fenics_sz.utils.plot.plot_show(plotter)
    if filename is not None:
        fenics_sz.utils.plot.plot_save(plotter, output_folder / "{}_td_solution_resscale_{:.2f}.png".format("filename", resscale))



# %%
plot_temp_probes(kamchatka_sz, kamchatka_geom, kamchatka_slab, kamchatka_paths, res=5, filename="Kamchatka_lines_res_5")


# %% [markdown]
# Note that lines do not extend all the way to the bottom of the subduction zone geometry because the points defining the line segments that make up these lines (ie. the points we probe) all exist within the sz geometry.

# %% [markdown]
# Next, we'll be using the density of the surrounding rock in each subduction zone to relate depth to pressure, letting us generate proper P-T curves. All of this is leading up to comparing pressure-temperature data of the slab with pressure-temperature phase stability diagrams of the minerals in each slab.

# %% [markdown]
# First, we need a function to convert depth to pressure:

# %%
def get_pressure(depth):
    #simplified pressure calculation:
    # pressure = depth (in meters) * 3300kg/m^3 (density) * g
    # then convert from Pa to GPa
    return depth*1000 * 3300 * 9.81 *1e-9


# %%
def get_depth(pressure):
    return pressure/1000 * 3300 * 9.81


# %%
results = probe(tonga_sz, "mantle", 12)

# %%
len(results)

# %%
print(results[2][1])

# %%
p=[]
t=[]
for i in range(len(results)):
    path_p = []
    path_t = []
    for j in range(len(results[i][0])):
        # simplified pressure 
        point_p = get_pressure(results[i][1][j])
        point_t = results[i][0][j]
        path_p.append(point_p)
        path_t.append(point_t)
    p.append(path_p)
    t.append(path_t)

# %% [markdown]
# We now have P-T information from our depth-temp. data. Let's plot it to make sure it looks all good.

# %%
fig = pl.figure()
ax = fig.gca()
for i in range(len(t)):
    ax.plot(t[i], p[i])
ax.set_title("Tonga serp mantle P-T curves")
ax.set_xlabel('T ($^\circ$C)')
ax.set_ylabel('Pressure (GPa)')
fig.savefig(output_folder / "{}_P-T_geotherms_{:.1f}_km_serp.png".format("Tonga", 12))

# %% [markdown]
# We now have a way to get P-T curves from paths along the slab. Now we can use this with our outputs from Perple_X to get some information about what the slab hydration should look like.

# %% [markdown]
# As a reminder, we've used Perple_X, a Gibbs free energy minimizer, to predict the hydration states at different pressures and temperatures of various initial lithologies. We can import one of those P-T grids now to use in conjuction with our geotherms. Because we've just plotted P-T curves for the serpentinized mantle under Tonga, let's import the hydration data for the damp DMM case.

# %% [markdown]
# First we need to input functions from the previous notebook:

# %%
from fenics_sz.fluid_release.perple_x_integration import get_PT_data_from_tabs, plot_PT_data
# import fenics_sz.thermo_calcs.perple_x_integration

# %%
p, t, H2O = get_PT_data_from_tabs("DMMdamp_25")
DMM_data = [p,t,H2O]

# %%
plot_PT_data(["DMMdamp_25"],[DMM_data])

# %%
results = probe(tonga_sz, "mantle", 12)

p=[]
t=[]
for i in range(len(results)):
    path_p = []
    path_t = []
    for j in range(len(results[i][0])):
        # simplified pressure 
        point_p = get_pressure(results[i][1][j])
        point_t = results[i][0][j]
        path_p.append(point_p)
        path_t.append(point_t)
    p.append(path_p)
    t.append(path_t)

# %%
fig = pl.figure()
ax = fig.gca()
for i in range(len(t)):
    ax.plot(t[i], p[i])
ax.set_title("Tonga serp mantle P-T curves")
ax.set_xlabel('T ($^\circ$C)')
ax.set_ylabel('Pressure (GPa)')
fig.savefig(output_folder / "{}_P-T_geotherms_{:.1f}_km_serp.png".format("Tonga", 12))

# %%
fig, ax = pl.subplots(figsize=(7, 4.5))

vmin = 0.0
vmax = 5.5
dv = 0.25

levels = np.arange(vmin, vmax+dv, dv)
c = ax.contourf(DMM_data[1], DMM_data[0], DMM_data[2], levels=levels, cmap="jet_r")
cbar = fig.colorbar(c, label=r"H$_2$O (wt%)")
cbar.set_ticks(np.arange(vmin, vmax, 1, dtype=np.int32))

for i in range(len(t)):
    ax.plot(t[i], p[i])


ax.set_ylabel(r"P (GPa)")
ax.set_xlabel(r"T ($^\circ$C)")
ax.set_title("Tonga serp mantle P-T curves")
ax.set_box_aspect(1)

# %%
results_kamchatka = probe(kamchatka_sz, "mantle", 12)

kamchatka_p=[]
kamchatka_t=[]
for i in range(len(results_kamchatka)):
    path_p = []
    path_t = []
    for j in range(len(results_kamchatka[i][0])):
        # simplified pressure 
        point_p = get_pressure(results_kamchatka[i][1][j])
        point_t = results_kamchatka[i][0][j]
        path_p.append(point_p)
        path_t.append(point_t)
    kamchatka_p.append(path_p)
    kamchatka_t.append(path_t)

# %%
fig, ax = pl.subplots(figsize=(7, 4.5))

vmin = 0.0
vmax = 5.5
dv = 0.25

levels = np.arange(vmin, vmax+dv, dv)
c = ax.contourf(DMM_data[1], DMM_data[0], DMM_data[2], levels=levels, cmap="jet_r")
cbar = fig.colorbar(c, label=r"H$_2$O (wt%)")
cbar.set_ticks(np.arange(vmin, vmax, 1, dtype=np.int32))

for i in range(len(kamchatka_t)):
    ax.plot(kamchatka_t[i], kamchatka_p[i])


ax.set_ylabel(r"P (GPa)")
ax.set_xlabel(r"T ($^\circ$C)")
ax.set_title("Kamchatka mantle P-T curves over DMM stability")
ax.set_box_aspect(1)
fig.savefig(output_folder / "{}_geotherms_over_phase_{:.1f}_km_serp.png".format("Kamchatka_SZ_mantle", h_serp))

# %% [markdown]
# This looks great! Let's try it with a mid-temp and colder subduction zone, to make sure the colder slabs dehydrate

# %% [markdown]
# Workflow: building the SZ problem to plotting the mantle geotherms on top of the mantle phase stabilty diagram.
#
# * All that's missing is extracting data from that comparision (and doing it for the rest of the slab besides the mantle)

# %%
print("\n".join(allsz_params))

# %%
BC_sz_dict = allsz_params["03_British_Columbia"]

# %%
BC_sz_slab = create_slab(BC_sz_dict['xs'], BC_sz_dict['ys'], resscale, BC_sz_dict['lc_depth'])
plot_slab(BC_sz_slab)

# %% [markdown]
# Create and solve the subduction problem as before:

# %%
BC_sz_geom = create_sz_geometry(BC_sz_slab, resscale, BC_sz_dict['sztype'], BC_sz_dict['io_depth'], BC_sz_dict['extra_width'], 
                             BC_sz_dict['coast_distance'], BC_sz_dict['lc_depth'], BC_sz_dict['uc_depth'])
BC_sz_sz = TDDislSubductionProblem(BC_sz_geom, **BC_sz_dict)

BC_sz_sz.solve(BC_sz_dict['As'], dt=0.05, theta=0.5, rtol=1.e-1, verbosity=1)

# %% [markdown]
# Plot the solution as a sanity check to ensure everything as run correctly.

# %%
plotter = pv.Plotter()
fenics_sz.utils.plot.plot_scalar(BC_sz_sz.T_i, plotter=plotter, scale=BC_sz_sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
# fenics_sz.utils.plot.plot_vector_glyphs(BC_sz_sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(BC_sz_sz.v0))
# fenics_sz.utils.plot.plot_vector_glyphs(BC_sz_sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(BC_sz_sz.v0))
BC_sz_geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = BC_sz_slab.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter)
# fenics_sz.utils.plot.plot_save(plotter, output_folder / "{}_td_solution_resscale_{:.2f}.png".format("kamchatka", resscale))

# %% [markdown]
# Now we can plot the depth-temperature paths of the kamchatka slab, considering 12km of serpentinized mantle, and compare our results with the P-T curves from Gies et al.

# %%
plot_geotherms(BC_sz_sz, "BC", h_serp, legend=True, save=True)

# %%
results_BC = probe(BC_sz_sz, "mantle", 12)

BC_p=[]
BC_t=[]
for i in range(len(results_BC)):
    path_p = []
    path_t = []
    for j in range(len(results_BC[i][0])):
        # simplified pressure 
        point_p = get_pressure(results_BC[i][1][j])
        point_t = results_BC[i][0][j]
        path_p.append(point_p)
        path_t.append(point_t)
    BC_p.append(path_p)
    BC_t.append(path_t)

# %%
fig, ax = pl.subplots(figsize=(7, 4.5))

vmin = 0.0
vmax = 5.5
dv = 0.25

levels = np.arange(vmin, vmax+dv, dv)
c = ax.contourf(DMM_data[1], DMM_data[0], DMM_data[2], levels=levels, cmap="jet_r")
cbar = fig.colorbar(c, label=r"H$_2$O (wt%)", location='left')
cbar.set_ticks(np.arange(vmin, vmax, 1, dtype=np.int32))

for i in range(len(BC_t)):
    ax.plot(BC_t[i], BC_p[i])


ax.set_ylabel(r"P (GPa)")
ax.set_xlabel(r"T ($^\circ$C)")
ax.set_title("BC mantle P-T curves over DMM stability")
ax.set_box_aspect(1)

ax.secondary_yaxis('right', functions=(get_depth,get_pressure), label=(r"z (km)"))

fig.savefig(output_folder / "{}_geotherms_over_phase_{:.1f}_km_serp.png".format("BC_SZ_mantle", h_serp))

# %%
points = (DMM_data[0], DMM_data[1])
values = (DMM_data[2][0],DMM_data[2][1])

# %%
print(DMM_data[2][0],DMM_data[2][1])

# %%
predict = sci.interpolate.RegularGridInterpolator(points, DMM_data[2], method = 'linear')

# %%
print(len(BC_t[0]))

# %%
test_points_shorter = []
for j in range(len(BC_t[0])):
    test_points_shorter.append([BC_p[0][j],BC_t[0][j]])

# %%
interpolate_test = predict(test_points_shorter)

# %%
test_points = []
for i in range(len(BC_t)):
    test_path=[]
    for j in range(len(BC_t[i])):
        test_path.append([BC_p[i][j],BC_t[i][j]])
    test_points.append(test_path)

# %%
longer_intrp_test =[]
for i in range(len(test_points)):
    longer_intrp_test.append(predict(test_points[i]))

# %%
print(len(interpolate_test))
print(len(longer_intrp_test[23]))


# %%
def find_water_line(H20):
    for i in range(len(H20)):
        if H20[i] == 0:
            return i


# %%
fig, ax = pl.subplots(figsize=(7, 4.5))

vmin = 0.0
vmax = 5.5
dv = 0.25

levels = np.arange(vmin, vmax+dv, dv)
c = ax.contourf(DMM_data[1], DMM_data[0], DMM_data[2], levels=levels, cmap="jet_r")
cbar = fig.colorbar(c, label=r"H$_2$O (wt%)", location='left')
cbar.set_ticks(np.arange(vmin, vmax, 1, dtype=np.int32))


# ax.plot(BC_t[i][0:80], BC_p[i][0:80])


for i in range(len(BC_t)):
    intrp = predict(test_points[i])
    water_line_ind = find_water_line(intrp)
    ax.plot(BC_t[i][0:water_line_ind], BC_p[i][0:water_line_ind])


ax.set_ylabel(r"P (GPa)")
ax.set_xlabel(r"T ($^\circ$C)")
ax.set_title("BC mantle P-T curves over DMM stability")
ax.set_box_aspect(1)

ax.secondary_yaxis('right', functions=(get_depth,get_pressure), label=(r"z (km)"))

fig.savefig(output_folder / "{}_geotherms_over_phase_{:.1f}_km_serp.png".format("BC_SZ_mantle", h_serp))

# %% [markdown]
# progress tracker:
#
# have a way to interpolate hydration values on the pt grid
# have a way to read what those values are (right now I'm searching for the first instance of a zero, but I can use that same for_loop structure to find the hydration value at each point that it has been predicted at.)
#
# need to:
# * do the interpolating for different mineralogies' h20 data
# * make sure hydration values are actually percents
# * calculate water *lost* at each depth

# %% [markdown]
# Meeting qs:
#
# * where do i check for things summing up to 100 again? I needed make sure that some set of numbers were percents and not something else
# * for some reason, when I save the new notebooks I'm making, it's not automatically saving equivalent py files to the python folder that exists outside of the notebooks directory

# %%
