# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dolfinx-env (3.12.3.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Concise, barebones workflow for getting to a TSM Line

# %% [markdown]
# #### Imports

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

# %%
from fenics_sz.fluid_release.perple_x_integration import get_PT_data_from_tabs, plot_PT_data
import fenics_sz.fluid_release.get_PT_curves 


# %% [markdown]
# ### Test to make sure a slab orthogonal grid can be made

# %%
def in_domain(sz, point):
    return ((0 < point[0] < int(sz.geom.slab_spline(1)[0])) and (int(sz.geom.slab_spline(1)[1]) < point[1] < 0))


# %%
# zslab = np.asarray([-slab.intersectx(x)[1] for x in nscoords[:,0]])

# use this to get the propper depth later on

# %%
# variable sediment offset

def get_st_grid(sz, h_serp, u_res, z0, z15):
    
    new_xys = []
    spline_dist = []
    spline_depth = []
    us = np.linspace(0,1,u_res)


# -----Checking which points will fall in the domain:-----
    test_spline = copy.deepcopy(sz.geom.slab_spline)
    test_spline.translatenormal(-7 -h_serp -z15)
    test_xys = np.asarray([test_spline(u) + [0.0] for u in us])

    slab_xys = np.asarray([sz.geom.slab_spline(u) + [0.0] for u in us])
    # xy points on the surface of the slab

    scoords = np.asarray(slab_xys).transpose()  # transpose now to avoid some issues later

    y = scoords[1,:]
    y0 = sz.geom.slab_spline.y[0]

    # -------sediments stuff----------------
    ssthicks = np.where(y>y0, z0, np.where(y<-15, z15, (z0-z15)*(y-y0)/(y0+15) + z0)).reshape(y.shape[0],1)
    print(ssthicks)
    # --------------------------------------

    normals = np.stack([-sz.geom.slab_spline.cs(test_xys[:,0], nu=1), np.ones(u_res), np.zeros(u_res)], axis=1)
    normags = np.sqrt(np.sum(normals**2, axis=1))
    normals = (normals.T/normags).T


    new_test_xys_new = slab_xys + (-ssthicks -7 -h_serp)*normals
    #'corrected' test points
    
    good_us = [] # list of u values to use when creating the set of xy points in slab space 
    good_ssthicks = []
    for u in range(u_res):
        if in_domain(sz, new_test_xys_new[u]) ==True:
            good_us.append(us[u])
            good_ssthicks.append(ssthicks[u])

    good_slab_xys = np.asarray([sz.geom.slab_spline(u) + [0.0] for u in good_us])
    # valid xy points on the surface of the slab

    good_ssthicks = np.asarray(good_ssthicks)

    print(good_us)
# -------------------------------------------------------

    layer_offsets = [-0.0, -0.3, -0.6, -1.3] + np.arange(-2.0, -10.0, -1).tolist()

    new_xys.append(np.asarray(good_slab_xys))
    spline_dist.append(np.asarray([u*sz.geom.slab_spline.length for u in good_us]))
    spline_depth.append((np.zeros(len(good_us))).reshape(len(good_us), 1))


    for depth in layer_offsets:
        new_spline = copy.deepcopy(sz.geom.slab_spline)
        new_spline.translatenormal(depth)
        xys = (np.asarray([new_spline(u) + [0.0] for u in good_us]))
        spline_dist.append(np.asarray([u*new_spline.length for u in good_us]))
        spline_depth.append(np.asarray(-good_ssthicks + depth))

        normals = np.stack([-sz.geom.slab_spline.cs(xys[:,0], nu=1), np.ones(len(good_us)), np.zeros(len(good_us))], axis=1)
        normags = np.sqrt(np.sum(normals**2, axis=1))
        normals = (normals.T/normags).T
        new_xys.append(np.asarray(good_slab_xys + (-good_ssthicks + depth)*normals))

    xys_as_array = np.asarray(new_xys)
    return xys_as_array, spline_dist, layer_offsets, spline_depth


# %%
def get_interpolator(perple_x_data):
    PT_points = (perple_x_data[0], perple_x_data[1])
    predict = sci.interpolate.RegularGridInterpolator(PT_points, perple_x_data[2], method = 'linear')
    return predict

    #create an interpolator object given the p, t, h2o perplex output

def predict_h2o(interpolator, t, p):
    # print("Temp: ", t)
    # print("Pressure: ", p)
    if t < 200:
        return interpolator([p, 200])
    # case for if the temperatures fall outside the bounds of the data used to generate the interpolator

    else:
        return (interpolator([p, t]))

    #predict the %wt hydration at a point given the pressure and temperature

# %% [markdown]
# ## The Cell class

# %%
# pass in a PT H2O interpolator when initializing the class, instead of the perpleX data
# one interpolator per lithology

# i can also try structuring it such that the temp of each vertex is passed in as a parameter.
# this way, mesh-point collisions and temperature evals are only done once per point, as opposed to four times


class Cell:
    def __init__(self, sz, slab_depth_1, slab_depth_2, slab_depth_3, slab_depth_4, 
                 slab_dist_1, slab_dist_2, slab_dist_3, slab_dist_4,
                 z1, z2, z3, z4, p1, p2, p3, p4, t1, t2, t3, t4, interpolator):
        self.sz = sz
        self.slab_depth_1 = slab_depth_1
        self.slab_depth_2 = slab_depth_2
        self.slab_depth_3 = slab_depth_3
        self.slab_depth_4 = slab_depth_4

        self.slab_dist_1 = slab_dist_1
        self.slab_dist_2 = slab_dist_2
        self.slab_dist_3 = slab_dist_3
        self.slab_dist_4 = slab_dist_4

        self.pressures = np.array([p1, p2, p3, p4])
        self.temps = np.array([t1, t2, t3, t4])
        self.vertex_depths = np.array([z1, z2, z3, z4])
        self.interpolator = interpolator
        self._water_pct = None
        self._hydrations = []
    
    def get_area(self):
        return((((self.slab_depth_1 - self.slab_depth_3) + (self.slab_depth_2 - self.slab_depth_4)) / 2) * (((self.slab_dist_2 - self.slab_dist_1) + (self.slab_dist_4 - self.slab_dist_3)) / 2))
    # very height times avg. length approx

    def get_depth(self):
        return(sum(self.vertex_depths)/4)

    def get_high_depth(self):
        pass

    def get_low_depth(self):
        pass

    def get_temp(self):
        return(sum(self.temps)/4)

    def get_water_pct(self):
        if self._water_pct is None:

            for i in range(len(self.temps)):
                if self.temps[i] < 200:
                    self._hydrations.append(self.interpolator([self.pressures[i], 200]))
                # case for if the temperatures fall outside the bounds of the data used to generate the interpolator

                else:
                    self._hydrations.append(self.interpolator([self.pressures[i], self.temps[i]]))

            self._water_pct = sum(self._hydrations)/4
        return self._water_pct

    def get_water_wt(self):
        return (self.get_water_pct() * 1e-2 * self.get_area() * 3300 * 1e-3)
        # units documentation:
        # first putting percents in decimal form
        # multiplying by area to get km^2

        # multiply density by 1e9 to get the density in units of kg/km^3
        # multiply current value by 1e-9 to convert from kg to Tg
        # two conversions above cancel out

        # left with units of Tg/km
        # multiply by 1e-3 to get Tg/m



    def __repr__(self):
        pass


# %% [markdown]
# #### Disallowing rehydration

# %%
def remove_rehydration(cell_hydrations):
    for i in range(len(cell_hydrations)):
        for j in range(len(cell_hydrations[i])-1):
            if cell_hydrations[i][j+1] > cell_hydrations[i][j]:
                cell_hydrations[i][j+1] = cell_hydrations[i][j]
    # FIXME more of a warning: this function transforms the input, it doesn't create a new variable
    # that is, when used, an initial hydrations array that allows for rehydration is not retained
    return cell_hydrations


# %% [markdown]
# #### Water Loss

# %%
def get_water_loss(cells, cell_hydrations):
    slab_losses_and_depths = []
    for i in range(len(cell_hydrations)):
        for j in range(len(cell_hydrations[i])-1):
            if (cell_hydrations[i][j+1] < cell_hydrations[i][j]):
                # should I check against an epsilon instead?

                # losses_and_depths = [cell_hydrations[i][j] - cell_hydrations[i][j+1] , ((cells[i][j].get_depth() + cells[i][j+1].get_depth()) / 2)]


                #This line stores total water lost, not water pct. water pct needs to be used for comparison, not water loss (I think...)
                losses_and_depths = [cells[i][j].get_water_wt() - cells[i][j+1].get_water_wt() , ((cells[i][j].get_depth() + cells[i][j+1].get_depth()) / 2)]
                
                #FIXME depths are currently global depths, not depths to surface of the slab

                # water_losses.append(cell_hydrations[i][j] - cell_hydrations[i][j+1])
                # water_loss_depths.append(((cells[i][j].get_depth() + cells[i][j+1].get_depth()) / 2))
                slab_losses_and_depths.append(losses_and_depths)

    return slab_losses_and_depths


# %% [markdown]
# #### Sidebar: splitting up water loss tracking by lithology

# %%
#FIXME this function is very specifc
# to how I've manually defined the layers of cells in each lithology

def sorted_water_loss_by_layer(cells, cell_hydrations):
    sediment_losses_and_depths = []
    uvolc_losses_and_depths = []
    lvolc_losses_and_depths = []
    dike_losses_and_depths = []
    gabbros_losses_and_depths = []
    mantle_losses_and_depths = []

    for j in range(len(cell_hydrations[0])-1):
        if (cell_hydrations[0][j+1] < cell_hydrations[0][j]):
            losses_and_depths = [cells[0][j].get_water_wt() - cells[0][j+1].get_water_wt(),
                                  ((cells[0][j].get_depth() + cells[0][j+1].get_depth()) / 2)]
            sediment_losses_and_depths.append(losses_and_depths)
            uvolc_losses_and_depths.append(losses_and_depths)
            lvolc_losses_and_depths.append(losses_and_depths)
            dike_losses_and_depths.append(losses_and_depths)
            gabbros_losses_and_depths.append(losses_and_depths)
            mantle_losses_and_depths.append(losses_and_depths)


    for j in range(len(cell_hydrations[1])-1):
        if (cell_hydrations[1][j+1] < cell_hydrations[1][j]):

            #This line stores total water lost, not water pct. water pct needs to be used for comparison, not water loss (I think...)
            losses_and_depths = [cells[1][j].get_water_wt() - cells[1][j+1].get_water_wt(),
                                  ((cells[1][j].get_depth() + cells[1][j+1].get_depth()) / 2)]
            
            #FIXME depths are currently global depths, not depths to surface of the slab

            uvolc_losses_and_depths.append(losses_and_depths)
            lvolc_losses_and_depths.append(losses_and_depths)
            dike_losses_and_depths.append(losses_and_depths)
            gabbros_losses_and_depths.append(losses_and_depths)
            mantle_losses_and_depths.append(losses_and_depths)


    for j in range(len(cell_hydrations[2])-1):
        if (cell_hydrations[2][j+1] < cell_hydrations[2][j]):
            losses_and_depths = [cells[2][j].get_water_wt() - cells[2][j+1].get_water_wt(),
                                  ((cells[2][j].get_depth() + cells[2][j+1].get_depth()) / 2)]
            lvolc_losses_and_depths.append(losses_and_depths)
            dike_losses_and_depths.append(losses_and_depths)
            gabbros_losses_and_depths.append(losses_and_depths)
            mantle_losses_and_depths.append(losses_and_depths)

    for j in range(len(cell_hydrations[3])-1):
        if (cell_hydrations[3][j+1] < cell_hydrations[3][j]):
            losses_and_depths = [cells[3][j].get_water_wt() - cells[3][j+1].get_water_wt(),
                                  ((cells[3][j].get_depth() + cells[3][j+1].get_depth()) / 2)]
            dike_losses_and_depths.append(losses_and_depths)
            gabbros_losses_and_depths.append(losses_and_depths)
            mantle_losses_and_depths.append(losses_and_depths)

    for j in range(len(cell_hydrations[4])-1):
        if (cell_hydrations[4][j+1] < cell_hydrations[4][j]):
            losses_and_depths = [cells[4][j].get_water_wt() - cells[4][j+1].get_water_wt(),
                                  ((cells[4][j].get_depth() + cells[4][j+1].get_depth()) / 2)]
            dike_losses_and_depths.append(losses_and_depths)
            gabbros_losses_and_depths.append(losses_and_depths)
            mantle_losses_and_depths.append(losses_and_depths)

    for i in range(5):
        for j in range(len(cell_hydrations[i+5])-1):
            if (cell_hydrations[i+5][j+1] < cell_hydrations[i+5][j]):
                losses_and_depths = [cells[i+5][j].get_water_wt() - cells[i+5][j+1].get_water_wt(),
                                      ((cells[i+5][j].get_depth() + cells[i+5][j+1].get_depth()) / 2)]

                gabbros_losses_and_depths.append(losses_and_depths)
                mantle_losses_and_depths.append(losses_and_depths)

    for i in range(2):
        for j in range(len(cell_hydrations[i+10])-1):
            if (cell_hydrations[i+10][j+1] < cell_hydrations[i+10][j]):
                losses_and_depths = [cells[i+10][j].get_water_wt() - cells[i+10][j+1].get_water_wt(),
                                      ((cells[i+10][j].get_depth() + cells[i+10][j+1].get_depth()) / 2)]

                mantle_losses_and_depths.append(losses_and_depths)



    
    return sorted(sediment_losses_and_depths, key=lambda l:l[1]), sorted(uvolc_losses_and_depths, key=lambda l:l[1]), sorted(lvolc_losses_and_depths, key=lambda l:l[1]), sorted(dike_losses_and_depths, key=lambda l:l[1]), sorted(gabbros_losses_and_depths, key=lambda l:l[1]), sorted(mantle_losses_and_depths, key=lambda l:l[1]),


# %% [markdown]
# ### Master Function

# %%
def get_TSMstye_line(sz_dict, h_serp, u_res, resscale, interps):

    sediment_data = get_PT_data_from_tabs(sz_dict['sediment'])
    sediment_interp = get_interpolator(sediment_data)

    uvolcs_interp = interps[0]
    lvolcs_interp = interps[1]
    dikes_interp = interps[2]
    gabbros_interp = interps[3]
    damp_DMM_interp = interps[4]

    # ------------------create SZ problem, solve, plot solution------------------
    slab = create_slab(sz_dict['xs'], sz_dict['ys'], resscale, sz_dict['lc_depth'])
    plot_slab(slab)
    geom = create_sz_geometry(slab, resscale, sz_dict['sztype'], sz_dict['io_depth'], sz_dict['extra_width'], 
                             sz_dict['coast_distance'], sz_dict['lc_depth'], sz_dict['uc_depth'])
    sz = TDDislSubductionProblem(geom, **sz_dict)

    sz.solve(sz_dict['As'], dt=0.05, theta=0.5, rtol=1.e-1, verbosity=1)

    plotter = pv.Plotter()
    fenics_sz.utils.plot.plot_scalar(sz.T_i, plotter=plotter, scale=sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    geom.pyvistaplot(plotter=plotter, color='green', width=2)
    cdpt = slab.findpoint('Slab::FullCouplingDepth')
    fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
    fenics_sz.utils.plot.plot_show(plotter)
    # ---------------------------------------------------------------------------


    # ------------------create and plot slab-based coord system------------------

    z0 = sz_dict["z0"]
    z15 = sz_dict["z15"]

    regular_points, spline_dist, layer_depths, true_normal_depths = get_st_grid(sz, h_serp, u_res, z0, z15)
    # layer_depths represent distances from the base of the sediments

    #plot s-t points
    plotter = pv.Plotter()
    fenics_sz.utils.plot.plot_scalar(sz.T_i, plotter=plotter, scale=sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    geom.pyvistaplot(plotter=plotter, color='green', width=2)
    cdpt = slab.findpoint('Slab::FullCouplingDepth')
    fenics_sz.utils.plot.plot_points([[cdpt.x, cdpt.y, 0.0]], plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')

    for i in range(len(regular_points)):
        for j in range(len(regular_points[i])):            
            fenics_sz.utils.plot.plot_points([[regular_points[i][j][0], regular_points[i][j][1], 0.0]], plotter=plotter, point_size=0.5, color='black')
    fenics_sz.utils.plot.plot_show(plotter)

    # ---------------------------------------------------------------------------


    # -------------precalculate temps and pressures at cell corners--------------

    ts = []
    zs = []
    ps = []

    for i in range(len(regular_points)):
        cinds, cells = fenics_sz.utils.mesh.get_cell_collisions(regular_points[i], sz.mesh)
        t, z, p = (sz.T_i.eval(regular_points[i], cells)[:,0] + (0.3 * -regular_points[i][:,1]), -regular_points[i][:,1], (-regular_points[i][:,1])*1000 * 3300 * 9.81 *1e-9)
        ts.append(t)
        zs.append(z)
        ps.append(p)

    # ---------------------------------------------------------------------------


    # -------------------------------create cells--------------------------------

    cells = []
    interp = None
    for i in range(12):
        layer_cells = []

        if i==0:
            interp = sediment_interp
        elif i==1:
            interp = uvolcs_interp
        elif i==2:
            interp = lvolcs_interp
        elif i in range(5):
            interp = dikes_interp
        elif i in range(10):
            interp = gabbros_interp
        elif i in range(12):
            interp = damp_DMM_interp
        else:
            raise Exception("Improper cell depth")


        for j in range(len(regular_points[i])-1): # problem with area approx is that spline dist varies between layers (taking average of spline dist btwn cell's 2 defining layers might make it a bit better)
            cell = Cell(sz, true_normal_depths[i][j][0], true_normal_depths[i][j+1][0], true_normal_depths[i+1][j][0], true_normal_depths[i+1][j+1][0], 
                        spline_dist[i][j], spline_dist[i][j+1], spline_dist[i+1][j], spline_dist[i+1][j+1],
                        zs[i][j], zs[i][j+1], zs[i+1][j], zs[i+1][j+1], ps[i][j], ps[i][j+1], ps[i+1][j], ps[i+1][j+1], 
                        ts[i][j], ts[i][j+1], ts[i+1][j], ts[i+1][j+1], interp)
            # might want to consider going back to passing in an interpolator object, and doing the hydration prediction within the class
            layer_cells.append(cell)
        cells.append(layer_cells)

    # ---------------------------------------------------------------------------

 
    arr_TND = np.asarray((true_normal_depths))
    slab_surf_dist = [] # distance along slab *surface* , not along the layer
    for i in range(len(spline_dist[0])):
        slab_surf_dist.append(spline_dist[0][i])


    # -----------------------------cell temperatures-----------------------------

    cell_temps = []
    for i in range(len(cells)):
        layer_cell_temps = []
        for j in range(len(cells[i])):
            layer_cell_temps.append(cells[i][j].get_temp())
        cell_temps.append(layer_cell_temps)

    fig, ax = pl.subplots()
    c = ax.pcolor(slab_surf_dist, arr_TND[:,-1,0], cell_temps, cmap='RdBu_r')
    ax.set_title('Cell Temps; 5x Slab-Normal Exageration')
    ax.set_xlabel('Distance along slab surface (km)')
    ax.set_ylabel('Distance normal to slab surface (km)')

    slab_normal_exag = 5
    ax.set_box_aspect((-arr_TND[:,-1,0][-1] / slab_surf_dist[-1]) * slab_normal_exag)

    fig.colorbar(c, ax=ax)
    pl.show()


    # -----------------------------cell hydrations-----------------------------

    cell_hydrations = []
    for i in range(len(cells)):
        layer_cell_hydrations = []
        for j in range(len(cells[i])):
            layer_cell_hydrations.append(cells[i][j].get_water_pct()[0])
        cell_hydrations.append(layer_cell_hydrations)

    fig, ax = pl.subplots()
    c = ax.pcolor(slab_surf_dist, arr_TND[:,-1,0], cell_hydrations, cmap='Blues')
    ax.set_title('Cell HYDRATIONS; 5x Slab-Normal Exageration')
    ax.set_xlabel('Distance along slab surface (km)')
    ax.set_ylabel('Distance normal to slab surface (km)')

    slab_normal_exag = 5
    ax.set_box_aspect((-arr_TND[:,-1,0][-1] / slab_surf_dist[-1]) * slab_normal_exag)

    fig.colorbar(c, ax=ax)
    pl.show()


    # -------------------------disallowing rehydration-------------------------
    cell_hydrations_no_rehydration = remove_rehydration(cell_hydrations)
    fig, ax = pl.subplots()
    c = ax.pcolor(slab_surf_dist, arr_TND[:,-1,0], cell_hydrations_no_rehydration, cmap='Blues') 
    ax.set_title('Cell Hydration NO REHYDRATION; 5x Slab-Normal Exageration')
    ax.set_xlabel('Distance along slab surface (km)')
    ax.set_ylabel('Distance normal to slab surface (km)')

    slab_normal_exag = 5
    ax.set_box_aspect((-arr_TND[:,-1,0][-1] / slab_surf_dist[-1]) * slab_normal_exag)

    fig.colorbar(c, ax=ax)
    pl.show()


    # -------------------------------water loss--------------------------------
    water_losses_and_depths = get_water_loss(cells, cell_hydrations_no_rehydration)
    sorted_water_losses_and_depths = sorted(water_losses_and_depths, key=lambda l:l[1])

    cum_sum_array = []
    cum_sum = 0
    for i in range(len(sorted_water_losses_and_depths)):
        cum_sum += sorted_water_losses_and_depths[i][0]
        # print(cum_sum)
        cum_sum_array.append(cum_sum[0])
        # print(cum_sum_array)


    distance_increment = sz.geom.slab_spline.length / u_res
    # (in km) slab-tangent distance between two st points on the surface of the slab

    time_standarized_losses = []
    for i in range(len(sorted_water_losses_and_depths)):
        time_standarized_losses.append(sorted_water_losses_and_depths[i][0][0] * sz_dict['Vs'] / distance_increment)

    cum_sum_time_standardized_array = []
    cum_sum_time_standardized = 0
    for i in range(len(time_standarized_losses)):
        cum_sum_time_standardized += time_standarized_losses[i]
        # print(cum_sum_time_standardized)
        cum_sum_time_standardized_array.append(cum_sum_time_standardized)
        # print(cum_sum_time_standardized_array)


    fig, ax = pl.subplots()
    ax.plot(cum_sum_time_standardized_array, [row[1] for row in sorted_water_losses_and_depths])
    ax.yaxis.set_inverted(True)  # inverted axis with autoscaling

    ax.set_title(str(sz_dict['dirname']) + " TSM Line")
    ax.set_xlabel('Tg/MYr/m lost')
    ax.set_ylabel('Depth of water loss (km)')
    ax.set_box_aspect(2.5)
    pl.show()

    print("Total water loss: " , cum_sum_time_standardized)
    fig.savefig(output_folder / (str(sz_dict['dirname']) + "TSM_line"))
    # ---------------------------------------------------------------------------



    # --------------------------water loss by layer------------------------------
    sediment_losses_and_depths, uvolc_losses_and_depths, lvolc_losses_and_depths, dike_losses_and_depths, gabbros_losses_and_depths, mantle_losses_and_depths = sorted_water_loss_by_layer(cells, cell_hydrations_no_rehydration)
    time_standarized_sediment_losses = []
    time_standarized_uvolc_losses = []
    time_standarized_lvolc_losses = []
    time_standarized_dike_losses = []
    time_standarized_gabbros_losses = []
    time_standarized_mantle_losses = []

    for i in range(len(sediment_losses_and_depths)):
        time_standarized_sediment_losses.append(sediment_losses_and_depths[i][0][0] * sz_dict['Vs'] / distance_increment)

    for i in range(len(uvolc_losses_and_depths)):
        time_standarized_uvolc_losses.append(uvolc_losses_and_depths[i][0][0] * sz_dict['Vs'] / distance_increment)

    for i in range(len(lvolc_losses_and_depths)):
        time_standarized_lvolc_losses.append(lvolc_losses_and_depths[i][0][0] * sz_dict['Vs'] / distance_increment)

    for i in range(len(dike_losses_and_depths)):
        time_standarized_dike_losses.append(dike_losses_and_depths[i][0][0] * sz_dict['Vs'] / distance_increment)

    for i in range(len(gabbros_losses_and_depths)):
        time_standarized_gabbros_losses.append(gabbros_losses_and_depths[i][0][0] * sz_dict['Vs'] / distance_increment)

    for i in range(len(mantle_losses_and_depths)):
        time_standarized_mantle_losses.append(mantle_losses_and_depths[i][0][0] * sz_dict['Vs'] / distance_increment)

    cum_sum_array_sediment = []
    cum_sum_array_uvolc = []
    cum_sum_array_lvolc = []
    cum_sum_array_dike = []
    cum_sum_array_gabbros = []
    cum_sum_array_mantle = []

    cum_sum = 0
    for i in range(len(time_standarized_sediment_losses)):
        cum_sum += time_standarized_sediment_losses[i]
        cum_sum_array_sediment.append(cum_sum)

    cum_sum = 0
    for i in range(len(time_standarized_uvolc_losses)):
        cum_sum += time_standarized_uvolc_losses[i]
        cum_sum_array_uvolc.append(cum_sum)

    cum_sum = 0
    for i in range(len(time_standarized_lvolc_losses)):
        cum_sum += time_standarized_lvolc_losses[i]
        cum_sum_array_lvolc.append(cum_sum)

    cum_sum = 0
    for i in range(len(time_standarized_dike_losses)):
        cum_sum += time_standarized_dike_losses[i]
        cum_sum_array_dike.append(cum_sum)

    cum_sum = 0
    for i in range(len(time_standarized_gabbros_losses)):
        cum_sum += time_standarized_gabbros_losses[i]
        cum_sum_array_gabbros.append(cum_sum)

    cum_sum = 0
    for i in range(len(time_standarized_mantle_losses)):
        cum_sum += time_standarized_mantle_losses[i]
        cum_sum_array_mantle.append(cum_sum)

    fig, ax = pl.subplots()
    ax.plot(cum_sum_time_standardized_array, [row[1] for row in sorted_water_losses_and_depths])

    ax.plot(cum_sum_array_sediment, [row[1] for row in sediment_losses_and_depths], label = "sediments")
    ax.plot(cum_sum_array_uvolc, [row[1] for row in uvolc_losses_and_depths], label = "upper_volcs")
    ax.plot(cum_sum_array_lvolc, [row[1] for row in lvolc_losses_and_depths], label = "lower_volcs")
    ax.plot(cum_sum_array_dike, [row[1] for row in dike_losses_and_depths], label = "dikes")
    ax.plot(cum_sum_array_gabbros, [row[1] for row in gabbros_losses_and_depths], label = "gabbros")
    ax.plot(cum_sum_array_mantle, [row[1] for row in mantle_losses_and_depths], label = "mantle")

    ax.yaxis.set_inverted(True)
    ax.set_title(str(sz_dict['dirname']) + " Layer-Seperated TSM Line")
    ax.set_xlabel('Tg/MYr/m lost')
    ax.set_ylabel('Depth where water loss occurs (km)')

    ax.set_box_aspect(2.5)
    ax.legend()
    pl.show()
    fig.savefig(output_folder / (str(sz_dict['dirname']) + "layer_seperated_TSM_line"))


    # ----------------------------function outputs-------------------------------
    layer_losses= None      # make this a dict; layer name, cum water loss of layer and layers above
    return cum_sum_time_standardized, layer_losses
