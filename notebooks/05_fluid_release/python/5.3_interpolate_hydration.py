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
# # Interpolating hydrations for P-T curves

# %% [markdown]
# In the last notebook, we built up the tools to extract P-T curves from our solved subduction problems, and we started to explore using those P-T curves in conjunction with a phase-stability diagram. In this notebook, we'll expand upon that comparison, and develop the tools to model the hydration state of an entire slab.

# %% [markdown]
# Start with the usual imports and file path settings

# %%
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))

# %%
print(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))

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

# %%
from fenics_sz.fluid_release.perple_x_integration import get_PT_data_from_tabs, plot_PT_data
import fenics_sz.fluid_release.get_PT_curves 

# %%
resscale = 5.0
cascadia_dict = allsz_params["04_Cascadia"]
cascadia_slab = create_slab(cascadia_dict['xs'], cascadia_dict['ys'], resscale, cascadia_dict['lc_depth'])
plot_slab(cascadia_slab)

# %%
cascadia_geom = create_sz_geometry(cascadia_slab, resscale, cascadia_dict['sztype'], cascadia_dict['io_depth'], cascadia_dict['extra_width'], 
                             cascadia_dict['coast_distance'], cascadia_dict['lc_depth'], cascadia_dict['uc_depth'])
cascadia_sz = TDDislSubductionProblem(cascadia_geom, **cascadia_dict)

cascadia_sz.solve(cascadia_dict['As'], dt=0.05, theta=0.5, rtol=1.e-1, verbosity=1)

plotter = pv.Plotter()
fenics_sz.utils.plot.plot_scalar(cascadia_sz.T_i, plotter=plotter, scale=cascadia_sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
# fenics_sz.utils.plot.plot_vector_glyphs(cascadia_sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(cascadia_sz.v0))
# fenics_sz.utils.plot.plot_vector_glyphs(cascadia_sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=fenics_sz.utils.mps_to_mmpyr(cascadia_sz.v0))
cascadia_geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = cascadia_slab.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
fenics_sz.utils.plot.plot_show(plotter)
# fenics_sz.utils.plot.plot_save(plotter, output_folder / "{}_td_solution_resscale_{:.2f}.png".format("kamchatka", resscale))

# %%
h_serp = 12
fenics_sz.fluid_release.get_PT_curves.plot_geotherms(cascadia_sz, "Cascadia", h_serp, legend=True, save=False)

# %%
basenames = ["DMMdamp_25" , "dike_25" , "gabbro_25" , "lovolc_25" , "upvolc_25"]

# %%
data = []
for basename in basenames:
    data.append(get_PT_data_from_tabs(basename))

# %%
paths = ["mantle" , "dikes" , "gabbro" , "lower_volc" , "upper_volc"]

# %%
all_probe = []
for path in paths:
    all_probe.append(fenics_sz.fluid_release.get_PT_curves.probe(cascadia_sz, path, 12))

# %%
pt_dict = {}
for i in range(len(paths)):
    pt_dict[paths[i]] = all_probe[i]


# %%
def interpolate_h20_content(grid_points, data, points_to_find, method='linear'):
    """
    A function to interpolate hydration data given a set of P-T_H2O data, and a set of P-T points to interpolate for

    Arguements:
      * grid_points     - the pressure and temperature points that define coordinate space
      * data            - the hydration data on the P-T grid
      * points_to_find  - points on the slab in P-T space to interpolate H2O for
    
    kwargs:
      * method          - method of interpolation (defaults to linear)

    Returns:
      * 
    """
    predict = sci.interpolate.RegularGridInterpolator(grid_points, data, method = method)
    return predict(points_to_find)


# %%
points = fenics_sz.fluid_release.get_PT_curves.get_probe_paths(cascadia_sz, 12)

# %%
print(len(points[0][0]))

# %%
plotter = pv.Plotter()
fenics_sz.utils.plot.plot_scalar(cascadia_sz.T_i, plotter=plotter, scale=cascadia_sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
cascadia_geom.pyvistaplot(plotter=plotter, color='green', width=2)
cdpt = cascadia_slab.findpoint('Slab::FullCouplingDepth')
fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')

for i in range(len(points)):
    for j in range(len(points[i])):            
        fenics_sz.utils.plot.plot_points([[points[i][j][0], points[i][j][1], 0.0]], plotter=plotter, point_size=0.5, color='white')

fenics_sz.utils.plot.plot_show(plotter)

# %% [markdown]
# I plotted this to see if making a slab-based coordinate system from the points I'd already probed was feasible. I determined it was, but elected for the orthogonal grid approach to avoid parsing for depth and calculating varying areas later on.

# %%
print(cascadia_dict['ys'])

# %%
slab_points = (cascadia_sz.geom.slab_spline.points)

# %%
for i in range(len(slab_points)):
    print(slab_points[i].x , " , ", slab_points[i].y)

# %% [markdown]
# Try this setup for the lower_volcs path

# %%
upper_volcs = copy.deepcopy(cascadia_sz.geom.slab_spline)
upper_volcs.translatenormalandcrop(-0.15)

# %%
upper_volcs_points = (upper_volcs.points)
for i in range(len(upper_volcs_points)):
    print(upper_volcs_points[i].x , " , ", upper_volcs_points[i].y)


# %% [markdown]
# I now know how to return points that define the boundaries between the layers.

# %%
def interpret_curve(slab_spline, input_x):
    points = slab_spline.points
    xs = []
    ys = []
    for i in range(len(points)):
        xs.append(points[i].x)
        ys.append(points[i].y)
    predicted_y = np.interp(input_x, xs, ys)
    return predicted_y


# %% [markdown]
# Give this function an x, it will output what the corresponding y should be based on its interpolation. This can be used to find out if a point is above or below a curve. Take a point's x value, plug it into this curve interpretter for the corresponding curve. If the predicted y is higher than the actual y, (comparision made using some epsilon )the point is below or on the curve, otherwise it's above the curve

# %%
def is_below_curve(x, y, curve):
    curve_points = curve.points
    xs = []
    ys = []
    for i in range(len(curve_points)):
        xs.append(curve_points[i].x)
        ys.append(curve_points[i].y)
    predict = sci.interpolate.CubicSpline(xs, ys)
    predicted_y = predict(x)
    if predicted_y >= y:
        return True
    else:
        return False
    
# this code neglects issue of floating point precision, effects seem like they should be negligable


# %%
resx = 500
resy = 250
xs = np.linspace(0, 468.0, resx)
ys = np.linspace(0, -240.0, resy)
points = []
for x in xs:
    for y in ys:
        points.append([x,y])

# %%
for point in points:
    if is_below_curve(point[0], point[1], cascadia_sz.geom.slab_spline) == False:
        points.remove(point)

# %%
print(len(points))


# %% [markdown]
# Define bounding curves:
# (starting depth of each zone used)

# %%
def bounding_curves(sz, h_serp):

    lower_volcs = copy.deepcopy(sz.geom.slab_spline)
    lower_volcs.translatenormalandcrop(-0.3)

    dikes = copy.deepcopy(sz.geom.slab_spline)
    dikes.translatenormalandcrop(-0.6)

    gabbro = copy.deepcopy(sz.geom.slab_spline)
    gabbro.translatenormalandcrop(-2)

    mantle = copy.deepcopy(sz.geom.slab_spline)
    mantle.translatenormalandcrop(-7)

    end = copy.deepcopy(sz.geom.slab_spline)
    end.translatenormalandcrop(-7 - h_serp)

    return lower_volcs, dikes, gabbro, mantle, end



# %%
lower_volcs, dikes, gabbro, mantle, end = bounding_curves(cascadia_sz, 8)

# %%
xs = np.linspace(0, 468.0, resx)
ys = np.linspace(0, -240.0, resy)
points = []
for x in xs:
    for y in ys:
        points.append([x,y])

# %%
upper_volcs_points = []
for point in points:
    if is_below_curve(point[0], point[1], cascadia_sz.geom.slab_spline) == True and is_below_curve(point[0], point[1], lower_volcs) == False:
        upper_volcs_points.append([point[0], point[1]])

# %%
lower_volcs_points = []
for point in points:
    if is_below_curve(point[0], point[1], lower_volcs) == True and is_below_curve(point[0], point[1], dikes) == False:
        lower_volcs_points.append([point[0], point[1]])

# %%
dikes_points = []
for point in points:
    if is_below_curve(point[0], point[1], dikes) == True and is_below_curve(point[0], point[1], gabbro) == False:
        dikes_points.append([point[0], point[1]])

# %%
gabbros_points = []
for point in points:
    if is_below_curve(point[0], point[1], gabbro) == True and is_below_curve(point[0], point[1], mantle) == False:
        gabbros_points.append([point[0], point[1]])

# %%
mantle_points = []
for point in points:
    if is_below_curve(point[0], point[1], mantle) == True and is_below_curve(point[0], point[1], end) == False:
        mantle_points.append([point[0], point[1]])

# %% [markdown]
# Sanity check plotter

# %%
plotter = pv.Plotter()
fenics_sz.utils.plot.plot_scalar(cascadia_sz.T_i, plotter=plotter, scale=cascadia_sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
cascadia_geom.pyvistaplot(plotter=plotter, color='green', width=2)

for i in range(len(dikes_points)):
        fenics_sz.utils.plot.plot_points([[dikes_points[i][0], dikes_points[i][1], 0.0]], plotter=plotter, point_size=0.5, color='white')

fenics_sz.utils.plot.plot_show(plotter)

# %% [markdown]
# Now have sets of points corresponding to each zone of the slab

# %% [markdown]
# Probe P-T for each of those sets, and compare to set's minerology stability:

# %%
DMM_data = get_PT_data_from_tabs('DMMdamp_25')
DMM_points = (DMM_data[0], DMM_data[1])

predict = sci.interpolate.RegularGridInterpolator(DMM_points, DMM_data[2], method = 'linear')

mantle_points_with_depth = np.array([[mantle_points[i][0],mantle_points[i][1],0.0] for i in range(len(mantle_points))])
cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(mantle_points_with_depth, cascadia_sz.mesh)
t, z, p = (cascadia_sz.T_i.eval(mantle_points_with_depth, cells)[:,0], -mantle_points_with_depth[:,1], (-mantle_points_with_depth[:,1])*1000 * 3300 * 9.81 *1e-9)


# %%
print(len(t), len(z), len(p))
print(len(mantle_points))
# print(p)
# print(t)
# print((DMM_data[1]))

# %%
print(len(mantle_points[0]))
print(len(mantle_points_with_depth[0]))

# %%
predicter_test=[]
for i in range(len(mantle_points)):
    predicter_test.append(predict((p[i], t[i])))

# %%
print(len(predicter_test))

# %%
cascadia_probe = fenics_sz.fluid_release.get_PT_curves.probe(cascadia_sz, "mantle", 8)

mantle_p=[]
mantle_t=[]
for i in range(len(cascadia_probe)):
    path_p = []
    path_t = []
    for j in range(len(cascadia_probe[i][0])):
        # simplified pressure 
        point_p = fenics_sz.fluid_release.get_PT_curves.get_pressure(cascadia_probe[i][1][j])
        point_t = cascadia_probe[i][0][j]
        path_p.append(point_p)
        path_t.append(point_t)
    mantle_p.append(path_p)
    mantle_t.append(path_t)

fig, ax = pl.subplots(figsize=(7, 4.5))

vmin = 0.0
vmax = 5.5
dv = 0.25

levels = np.arange(vmin, vmax+dv, dv)
c = ax.contourf(DMM_data[1], DMM_data[0], DMM_data[2], levels=levels, cmap="jet_r")
cbar = fig.colorbar(c, label=r"H$_2$O (wt%)", location='left')
cbar.set_ticks(np.arange(vmin, vmax, 1, dtype=np.int32))

for i in range(len(mantle_t)):
    ax.plot(mantle_t[i], mantle_p[i])


ax.set_ylabel(r"P (GPa)")
ax.set_xlabel(r"T ($^\circ$C)")
ax.set_title("Cascadia mantle P-T curves over DMM stability")
ax.set_box_aspect(1)

ax.secondary_yaxis('right', functions=(fenics_sz.fluid_release.get_PT_curves.get_depth, fenics_sz.fluid_release.get_PT_curves.get_pressure), label=(r"z (km)"))

# %%
fig = pl.figure()
ax = fig.gca()
ax.plot(z, predicter_test, 'bo', markersize=2)
ax.set_xlabel("Depth (km)")
ax.set_ylabel("%wt. H2O")
ax.set_title("Cascadia Mantle depth vs H2O %wt at points")

# %%
print(len(mantle_points[0]))

# %% [markdown]
# Get volume of one cell per meter of trench length:

# %%
grid_volume = (cascadia_dict['xs'][-1] / resx) * (-cascadia_dict['ys'][-1] / resy)

# remember to redimensionalize before plotting!!
# currently in km^3 : volume of one cell on this "new grid" per km trench length 

grid_mass_mantle = grid_volume * 3300 *1e6 #redimensionalizing : units are kg/m

# %%
print(grid_volume)
print(grid_mass_mantle)

# %%
mantle_start = -7.0
def find_depth_ind(depth, ys):
    for i in range(len(ys)):
        if ys[i] <= depth:
            return i

mantle_depth_ind = find_depth_ind(mantle_start, ys)
print(mantle_depth_ind)
mantle_ys = ys[mantle_depth_ind:]

# was planning to use this to only index depths where the mantle acutally starts, but I couldn't get it to work without bugs

# %%
depth_water_sums = []
for y in ys:
    depth_water_sum = 0
    for i in range(len(mantle_points)):
        if y == mantle_points[i][1]:
            depth_water_sum += predicter_test[i]
    depth_water_sum = depth_water_sum * 1e-2 * grid_mass_mantle * 1e-9
    depth_water_sums.append(depth_water_sum)

# %%
print(len(depth_water_sums))

# %%
fig = pl.figure()
ax = fig.gca()

ax.plot(-ys, depth_water_sums, label="Mantle")
ax.set_xlim([20,250])
ax.set_xlabel("Depth (km)")
ax.set_ylabel("H2O (Tg/m)")
ax.legend()
ax.set_title("Cascadia slab H2O weight per meter of trench distance vs depth")

# %% [markdown]
# ### Now we repeat this process for the gabbros:

# %%
gabbro_data = get_PT_data_from_tabs('gabbro_25')
gabbro_PT_points = (gabbro_data[0], gabbro_data[1])

gabbro_predict = sci.interpolate.RegularGridInterpolator(gabbro_PT_points, gabbro_data[2], method = 'linear')

gabbro_points_3d = np.array([[gabbros_points[i][0],gabbros_points[i][1],0.0] for i in range(len(gabbros_points))])
cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(gabbro_points_3d, cascadia_sz.mesh)
gabbro_t, gabbro_z, gabbro_p = (cascadia_sz.T_i.eval(gabbro_points_3d, cells)[:,0], -gabbro_points_3d[:,1], (-gabbro_points_3d[:,1])*1000 * 3300 * 9.81 *1e-9)

# %%
print((gabbro_p))
print((gabbro_t))

# %%
gabbro_h2o=[]
for i in range(len(gabbros_points)):
    if gabbro_t[i] < 200:
        gabbro_h2o.append(gabbro_data[2][0][1])
    else:
        gabbro_h2o.append(gabbro_predict((gabbro_p[i], gabbro_t[i])))

# %%
print(len(gabbro_h2o))
print(len(gabbro_z))

# %%
cascadia_probe_gabbro = fenics_sz.fluid_release.get_PT_curves.probe(cascadia_sz, "gabbro")

gabbro_geotherm_p=[]
gabbro_geotherm_t=[]
for i in range(len(cascadia_probe_gabbro)):
    path_p = []
    path_t = []
    for j in range(len(cascadia_probe_gabbro[i][0])):
        # simplified pressure 
        point_p = fenics_sz.fluid_release.get_PT_curves.get_pressure(cascadia_probe_gabbro[i][1][j])
        point_t = cascadia_probe_gabbro[i][0][j]
        path_p.append(point_p)
        path_t.append(point_t)
    gabbro_geotherm_p.append(path_p)
    gabbro_geotherm_t.append(path_t)

fig, ax = pl.subplots(figsize=(7, 4.5))

vmin = 0.0
vmax = 5.5
dv = 0.25

levels = np.arange(vmin, vmax+dv, dv)
c = ax.contourf(gabbro_data[1], gabbro_data[0], gabbro_data[2], levels=levels, cmap="jet_r")
cbar = fig.colorbar(c, label=r"H$_2$O (wt%)", location='left')
cbar.set_ticks(np.arange(vmin, vmax, 1, dtype=np.int32))

for i in range(len(gabbro_geotherm_p)):
    ax.plot(gabbro_geotherm_t[i], gabbro_geotherm_p[i])


ax.set_ylabel(r"P (GPa)")
ax.set_xlabel(r"T ($^\circ$C)")
ax.set_title("Cascadia gabbro P-T curves over gabbro mineral stability")
ax.set_box_aspect(1)

ax.secondary_yaxis('right', functions=(fenics_sz.fluid_release.get_PT_curves.get_depth, fenics_sz.fluid_release.get_PT_curves.get_pressure), label=(r"z (km)"))

# %%
fig = pl.figure()
ax = fig.gca()
ax.plot(gabbro_z, gabbro_h2o, 'red', markersize=2)
ax.set_xlabel("Depth (km)")
ax.set_ylabel("%wt. H2O")
ax.set_title("Cascadia Gabbro depth vs H2O %wt at points")

# %%
gabbro_depth_water_sums = []
for y in ys:
    gabbro_depth_water_sum = 0
    for i in range(len(gabbros_points)):
        if y == gabbros_points[i][1]:
            gabbro_depth_water_sum += gabbro_h2o[i]
    gabbro_depth_water_sum = gabbro_depth_water_sum* 1e-2 * grid_mass_mantle * 1e-9
    gabbro_depth_water_sums.append(gabbro_depth_water_sum)

# %%
print(len(gabbro_depth_water_sums))

# %%
fig = pl.figure()
ax = fig.gca()

ax.plot(-ys, gabbro_depth_water_sums, label="Gabbro")
ax.set_xlim([10, 250])
ax.set_xlabel("Depth (km)")
ax.set_ylabel("H2O (Tg/m)")
ax.legend()
ax.set_title("Cascadia Gabbro H2O weight per meter of trench distance vs depth")

# %%
fig = pl.figure()
ax = fig.gca()

ax.plot(-ys, gabbro_depth_water_sums, label="Gabbro")
ax.plot(-ys, depth_water_sums, label="Mantle")


ax.set_xlim([20, 250])
ax.set_xlabel("Depth (km)")
ax.set_ylabel("H2O (Tg/m)")
ax.legend()
ax.set_title("Cascadia slab H2O per meter of trench distance")
