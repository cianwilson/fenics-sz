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

# %%
import sys, os, shutil
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))

# %%
import pyvista as pv
import copy
import pathlib
import hashlib
import zipfile
import requests
import numpy as np
import matplotlib.pyplot as pl
import itertools
import math
from dataclasses import dataclass
import json
from scipy import integrate as integ

# %%
import fenics_sz.utils
from fenics_sz.sz_problems.sz_params import allsz_params
from fenics_sz.sz_problems.sz_slab import create_slab, plot_slab
from fenics_sz.fluid_release.perple_x_class import PerpleXGrid
from fenics_sz.fluid_release.perple_x_class_meemum import PerpleXMeemum
from fenics_sz.fluid_release.slab_dehydration_class import SlabDehydration, SlabMesh


# %%
class SlabDehydrationVertical(SlabDehydration):

    @property
    def mesh(self):
        if not hasattr(self, '_mesh') or self._mesh is None:
            # work out initial slab coordinates
            xmin = self.slab.intersecty(-self.zmin)[0]
            umin = self.slab.x2delu(xmin)
            umax = 1.0
            if self.zmax is not None and self.zmax < -self.slab.y[-1]:
                xmax = self.slab.intersecty(-self.zmax)[0]
                umax = self.slab.x2dely(xmax)
            slab_vertex_us = np.arange(umin, umax, self.sres/self.slab.length)
            slab_vertex_xys = np.asarray([self.slab(u) for u in slab_vertex_us])

            # accumulate the offsets from layers above, at and below the slab (and reverse the list - base first)
            offsets = (list(itertools.accumulate([t for t in self.layer_thicknesses if t > 0])) + \
                       [0.0] + \
                       list(itertools.accumulate([t for t in self.layer_thicknesses if t < 0])))[::-1]
            base_vertex_xys = np.asarray([self.slab.intersectx(x, offset=offsets[0]) for x in slab_vertex_xys[:,0]])
            valid_vertex_ind_0, valid_vertex_ind_f = self.inds_in_domain(base_vertex_xys)
            if offsets[-1] > 0.0:
                top_ind_0, top_ind_f = self.inds_in_domain(np.asarray([self.slab.intersectx(x, offset=offsets[-1]) for x in slab_vertex_xys[:,0]]))
                valid_vertex_ind_0 = max(valid_vertex_ind_0, top_ind_0)
                valid_vertex_ind_f = min(valid_vertex_ind_f, top_ind_f)

            # restrict the vertex coordinates to the valid values
            slab_vertex_xys = slab_vertex_xys[valid_vertex_ind_0:valid_vertex_ind_f,:]
            base_vertex_xys = base_vertex_xys[valid_vertex_ind_0:valid_vertex_ind_f,:]

            # work out the dimensions of our grid
            num_valid_us = len(slab_vertex_xys)
            num_outer_layers = len(self.layer_thicknesses)
            # we fill in nums_sub_layers from top to bottom to match layer_h2os and other user input
            # and will reverse access it in the assembly loop
            nums_sub_layers = [math.ceil(abs(thickness)/tres) 
                               for thickness, tres in zip(self.layer_thicknesses, self.layer_tres)]
            total_layers = sum(nums_sub_layers)

            # pre-allocate memory for the full coordinate system and the cells
            vertex_xys = np.empty(((total_layers+1)*num_valid_us, 2))
            vertex_xys[:num_valid_us,:] = base_vertex_xys
            cells = np.empty((total_layers*(num_valid_us-1), 4), dtype=np.int32)
            # we set up in layer_cell_inds from top to bottom to match layer_h2os and other user input
            layer_cell_inds = [np.empty((num_sub_layers, num_valid_us-1), dtype=np.int32) for num_sub_layers in nums_sub_layers]
            dof_xys = np.empty((total_layers*(num_valid_us-1), 2))
            # memory for cell areas
            cell_areas = np.empty(total_layers*(num_valid_us-1))

            # set up near-orthogonal slab coordinate system
            vertex_ss = slab_vertex_us[valid_vertex_ind_0:valid_vertex_ind_f]*self.slab.length
            vertex_ss = vertex_ss - vertex_ss[0]
            vertex_ts = np.empty(total_layers + 1)
            # note that this is bottom to top
            vertex_ts[0] = offsets[0]

            layer_ind = 0
            for l, offset in enumerate(offsets[1:]):
                prev_offset = offsets[l]
                # the number of sublayers based on the layer thickness and resolution
                num_sub_layers = nums_sub_layers[num_outer_layers-1-l]
                sub_offsets = np.linspace(prev_offset, offset, num=num_sub_layers+1)
                # loop over the sub layers
                for sl, sub_offset in enumerate(sub_offsets[1:]):
                    prev_sub_offset = sub_offsets[sl]
                    sub_thickness = sub_offset - prev_sub_offset
                    # work out coordinates...
                    vertex_ind = (layer_ind + 1)*num_valid_us
                    current_vertex_inds = np.arange(vertex_ind, vertex_ind+num_valid_us, dtype=np.int32)
                    # in xy space
                    vertex_xys[current_vertex_inds,:] = np.asarray([self.slab.intersectx(x, offset=sub_offset) for x in slab_vertex_xys[:,0]])
                    # and t space
                    vertex_ts[layer_ind + 1] = sub_offset
                    # the current cells
                    current_cell_inds = np.arange(layer_ind*(num_valid_us-1), (layer_ind+1)*(num_valid_us-1), dtype=np.int32)
                    # use a halfway offset to work out the dof xys at the mid x points of each cell
                    half_sub_offset = prev_sub_offset + 0.5*sub_thickness
                    dof_xys[current_cell_inds,:] = np.asarray([self.slab.intersectx(x, offset=half_sub_offset) for x in 0.5*(slab_vertex_xys[:-1,0] + slab_vertex_xys[1:,0])])
                    # work out cells
                    for i in range(num_valid_us-1):
                        # lower left, lower right, upper right, upper left
                        cell = [
                                (layer_ind)*num_valid_us + i,   (layer_ind)*num_valid_us + i + 1,
                                (layer_ind + 1)*num_valid_us + i + 1, (layer_ind + 1)*num_valid_us + i
                               ]
                        cells[layer_ind*(num_valid_us-1) + i, :] = cell
                    # work out the cell areas by numerical integration
                    cell_areas[current_cell_inds] = np.asarray([integ.quad(lambda x: self.slab.intersectx(x, offset=sub_offset)[1] - self.slab.intersectx(x, offset=prev_sub_offset)[1], x0, x1)[0] for (x0,x1) in itertools.pairwise(slab_vertex_xys[:,0])])
                    # we fill in layer_cell_inds from top to bottom to match layer_h2os and other user input
                    layer_cell_inds[num_outer_layers-1-l][num_sub_layers-1-sl,:] = current_cell_inds
                    # increment layer index
                    layer_ind += 1

            self._mesh = SlabMesh(vertex_xys=vertex_xys, vertex_ss=vertex_ss, vertex_ts=vertex_ts, 
                                  cells=cells, layer_cell_inds=layer_cell_inds, 
                                  dof_xys=dof_xys, 
                                  cell_areas=cell_areas)
            # NOTE: during construction we reversed layer_cell_inds to be in the same 
            # order as layer_h2os above (i.e. it goes from top to bottom, while 
            # everything else is natively ordered from bottom to top but this should 
            # be OK as the other things should be accesses through layer_cell_inds)
        return self._mesh


# %%
class SlabDehydrationVerticalFlux(SlabDehydrationVertical):
    @property
    def maxH2Os(self):
        if not hasattr(self, '_maxH2Os') or self._maxH2Os is None:
            self._maxH2Os = np.empty(self.mesh.cells.shape[0])
            for cell_inds, h2ogrid in zip(self.mesh.layer_cell_inds, self.layer_h2os):
                for sl_cell_inds in cell_inds:
                    comps = h2ogrid.component_masses
                    for sl_cell_ind in sl_cell_inds:
                        comps = h2ogrid.eval(self.Ps[sl_cell_ind], self.Ts[sl_cell_ind], **comps)
                        self._maxH2Os[sl_cell_ind] = comps['H2O']/100.0
        return self._maxH2Os

# %% tags=["active-ipynb"]
# name = "03_British_Columbia"
# resscale = 5.0

# %% tags=["active-ipynb"]
# szdict = allsz_params[name]
# print("{}:".format(name))
# print("{:<20} {:<10}".format('Key','Value'))
# print("-"*85)
# for k, v in allsz_params[name].items():
#     if v is not None: print("{:<20} {}".format(k, v))

# %% tags=["active-ipynb"]
#

# %% tags=["active-ipynb"]
# slab1 = create_slab(szdict['xs'], szdict['ys'], resscale, szdict['lc_depth'])
# _ = plot_slab(slab1)

# %% tags=["active-ipynb"]
# zipfilename = pathlib.Path(os.path.join(basedir, os.path.pardir, os.path.pardir, "data", "vankeken_wilson_peps_2023_TF_lowres_minimal.zip"))
# if not zipfilename.is_file():
#     zipfileurl = 'https://zenodo.org/records/13234021/files/vankeken_wilson_peps_2023_TF_lowres_minimal.zip'
#     r = requests.get(zipfileurl, allow_redirects=True)
#     open(zipfilename, 'wb').write(r.content)
# assert hashlib.md5(open(zipfilename, 'rb').read()).hexdigest() == 'a8eca6220f9bee091e41a680d502fe0d'

# %% tags=["active-ipynb"]
# tffilename = os.path.join('vankeken_wilson_peps_2023_TF_lowres_minimal', 'sz_suite_td', szdict['dirname']+'_minres_2.00_cfl_2.00.vtu')
# tffilepath = os.path.join(basedir, os.path.pardir, os.path.pardir, 'data')
# with zipfile.ZipFile(zipfilename, 'r') as z:
#     z.extract(tffilename, path=tffilepath)
# tfgrid = pv.get_reader(os.path.join(tffilepath, tffilename)).read()

# %% tags=["active-ipynb"]
# dmm_thickness = 2.0
#
# tres = 1
# sres = 20
#
# # negative number implies below slab, positive implies above it
# layer_thicknesses = [
#                 #  2.0,            # above slab mantle
#                  -szdict['z15'], # sediments
#                  -0.3,           # upper volcanics
#                  -0.3,           # lower volcanics
#                  -1.4,           # dikes
#                  -5.0,           # gabbro
#                  -dmm_thickness  # subslab mantle
#                 ]
#
# csv_path = os.path.join(os.pardir, os.pardir, 'data', 'perple_x_v7.1.9', 'abers_25')
# layer_h2os = [
#     # PerpleXGrid(csv_file=os.path.join(csv_path, 'DMMdry_25_h2o.csv')),
#     PerpleXGrid(csv_file=os.path.join(csv_path, szdict['sed_type']+'_h2o.csv')),
#     PerpleXGrid(csv_file=os.path.join(csv_path, 'upvolc_25_h2o.csv')),
#     PerpleXGrid(csv_file=os.path.join(csv_path, 'lovolc_25_h2o.csv')),
#     PerpleXGrid(csv_file=os.path.join(csv_path, 'dike_25_h2o.csv')),
#     PerpleXGrid(csv_file=os.path.join(csv_path, 'gabbro_25_h2o.csv')),
#     PerpleXGrid(csv_file=os.path.join(csv_path, 'DMMdamp_25_h2o.csv'))
# ]
#
# layer_tres = [
#     None,
#     None,
#     None,
#     None,
#     1.4,
# ]

# %% tags=["active-ipynb"]
# testslab = SlabDehydrationVertical(sres, tres, layer_thicknesses, layer_h2os, layer_tres=None,
#                            slab=slab1, Tgrid=tfgrid, 
#                            Tname='Temperature::PotentialTemperature', 
#                            coast_distance=szdict['coast_distance'], 
#                            sztype=szdict['sztype'], lc_depth=szdict['lc_depth'], trench_length=szdict['trench_length'], Vs=szdict['Vs'])

# %% tags=["active-ipynb"]
# testslab.conservation_errors

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(10,8))
# ax = fenics_sz.utils.plot.mpl_plot_pv_scalar(tfgrid, ax, cmap='coolwarm')
# testslab.plot_xy(ax, C=testslab.Ts, cmap='coolwarm', edgecolor = 'none', lw=0.1, vmin=0, vmax=tfgrid[tfgrid.active_scalars_name].max())
# # some discrepancy because adiabat added in
# ax.set_aspect(3)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(10,8))
# ax = fenics_sz.utils.plot.mpl_plot_pv_scalar(tfgrid, ax, cmap='coolwarm')
# ax = fenics_sz.utils.plot.mpl_plot_pv_mesh(tfgrid, ax, facecolor='none', edgecolor='black', linewidth=0.2)
# testslab.plot_xy(ax, C=testslab.H2Os, cmap='Blues', edgecolor = 'none', lw=0.1)
# ax.set_aspect(3)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslab.plot_st(ax, C=testslab.cumulative_H2O_losses, cmap='coolwarm', edgecolor = 'black', lw=0.5)
# ax.set_aspect(5)
# fig.show()

# %% tags=["active-ipynb"]
# fig, (ax, axc) = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1,0.05])
# # work out areas relative to first cell of each row of mesh (easier to visualize)
# relcellareas = np.empty_like(testslab.mesh.cell_areas)
# for cell_inds in testslab.mesh.layer_cell_inds:
#     for sub_cell_inds in cell_inds:
#         relcellareas[sub_cell_inds] = testslab.mesh.cell_areas[sub_cell_inds]/testslab.mesh.cell_areas[sub_cell_inds[0]]
# pcm = testslab.plot_st(ax, C=relcellareas, cmap = 'viridis', edgecolor = 'black', linewidth=0.5, shading='flat')
# ax.set_aspect(10.0)
# fig.colorbar(pcm, cax=axc, orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fig, (ax, axc) = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1,0.05])
# pcm = testslab.plot_st(ax, C=testslab.mesh.cell_areas, cmap = 'viridis', edgecolor = 'black', linewidth=0.5, shading='flat')
# ax.set_aspect(10.0)
# fig.colorbar(pcm, cax=axc, orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# with open(os.path.join(basedir, os.pardir, "data", "perple_x_v7.1.9", "abers_25", "abers_25.json"), "r") as file:
#     abers_25 = json.load(file)

# %% tags=["active-ipynb"]
# layer_h2os_meemum = [
#     PerpleXMeemum(basename, abers_25[basename]['component_masses'], abers_25[basename]['excluded_phases'], abers_25[basename]['solution_models']) for basename in [szdict['sed_type'], 'upvolc_25', 'lovolc_25', 'dike_25', 'gabbro_25', 'DMMdamp_25']
# ]

# %% tags=["active-ipynb"]
# testslabmeemum = SlabDehydrationVertical(sres, tres, layer_thicknesses, layer_h2os_meemum, layer_tres=None,
#                            slab=slab1, Tgrid=tfgrid, 
#                            Tname='Temperature::PotentialTemperature', 
#                            coast_distance=szdict['coast_distance'], 
#                            sztype=szdict['sztype'], lc_depth=szdict['lc_depth'], trench_length=szdict['trench_length'], Vs=szdict['Vs'])

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslabmeemum.plot_st(ax, C=testslabmeemum.cumulative_H2O_losses, cmap='coolwarm', edgecolor = 'black', lw=0.5)
# ax.set_aspect(5)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslab.plot_st(ax, C=testslab.cumulative_H2O_losses, cmap='coolwarm', edgecolor = 'black', lw=0.5)
# ax.set_aspect(5)
# fig.show()

# %% tags=["active-ipynb"]
# testslabmeemumflux = SlabDehydrationVerticalFlux(sres, tres, layer_thicknesses, layer_h2os_meemum, layer_tres=None,
#                            slab=slab1, Tgrid=tfgrid, 
#                            Tname='Temperature::PotentialTemperature', 
#                            coast_distance=szdict['coast_distance'], 
#                            sztype=szdict['sztype'], lc_depth=szdict['lc_depth'], trench_length=szdict['trench_length'], Vs=szdict['Vs'])

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslabmeemumflux.plot_st(ax, C=testslabmeemumflux.cumulative_H2O_losses, cmap='coolwarm', edgecolor = 'black', lw=0.5)
# ax.set_aspect(5)
# fig.show()

# %% tags=["active-ipynb"]
# testslabmeemumflux.H2O_losses.min(), testslabmeemumflux.cumulative_H2O_losses.min(), (testslabmeemumflux.maxH2Os-testslabmeemumflux.H2Os).max()

# %% tags=["active-ipynb"]
# testslabmeemum.H2O_losses.min(), testslabmeemum.cumulative_H2O_losses.min(), (testslabmeemum.maxH2Os-testslabmeemum.H2Os).max()

# %% tags=["active-ipynb"]
# testslab.H2O_losses.min(), testslab.cumulative_H2O_losses.min()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslabmeemum.plot_st(ax, C=testslabmeemum.maxH2Os, cmap='coolwarm', edgecolor = 'black', lw=0.5)
# ax.set_aspect(5)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslab.plot_st(ax, C=testslab.maxH2Os, cmap='coolwarm', edgecolor = 'black', lw=0.5)
# ax.set_aspect(5)
# fig.show()

# %% tags=["active-ipynb"]
# testslabog = SlabDehydration(sres, tres, layer_thicknesses, layer_h2os, layer_tres=None,
#                            slab=slab1, Tgrid=tfgrid, 
#                            Tname='Temperature::PotentialTemperature', 
#                            coast_distance=szdict['coast_distance'], 
#                            sztype=szdict['sztype'], lc_depth=szdict['lc_depth'], trench_length=szdict['trench_length'], Vs=szdict['Vs'])

# %% tags=["active-ipynb"]
# testslabog.H2O_losses.min(), testslabog.cumulative_H2O_losses.min()

# %% tags=["active-ipynb"]
# fix, ax = pl.subplots(figsize=(5,10))
# indices = np.argsort(-testslab.mesh.dof_xys[:,1])
# ax.plot(testslab.total_cumulative_H2O_losses/1000.0, testslab.mesh.dof_xys[indices,1], label='vertical')
# indicesog = np.argsort(-testslabog.mesh.dof_xys[:,1])
# ax.plot(testslabog.total_cumulative_H2O_losses/1000.0, testslabog.mesh.dof_xys[indicesog,1], label='normal', ls='--')
# indicesmm = np.argsort(-testslabmeemum.mesh.dof_xys[:,1])
# ax.plot(testslabmeemum.total_cumulative_H2O_losses/1000.0, testslabmeemum.mesh.dof_xys[indicesmm,1], label='meemum', ls='-.')
# indicesmmf = np.argsort(-testslabmeemumflux.mesh.dof_xys[:,1])
# ax.plot(testslabmeemumflux.total_cumulative_H2O_losses/1000.0, testslabmeemumflux.mesh.dof_xys[indicesmmf,1], label='flux', ls=':')
# ax.set_xlabel('Cumulative water loss')
# ax.set_ylabel('y (km)')
# _ = ax.legend()

# %%

# %%
