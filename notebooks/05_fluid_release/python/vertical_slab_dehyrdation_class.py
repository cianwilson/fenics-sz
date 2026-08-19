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

# %%
import fenics_sz.utils
from fenics_sz.sz_problems.sz_params import allsz_params
from fenics_sz.sz_problems.sz_slab import create_slab, plot_slab
from fenics_sz.fluid_release.perple_x_class import PerpleXGrid
from fenics_sz.fluid_release.perple_x_class_meemum import PerpleXMeemum
from fenics_sz.fluid_release.slab_dehydration_class import SlabDehydration, SlabMesh


# %%
@dataclass
class SlabMeshFlux(SlabMesh):
    cell_out_flux : np.typing.NDArray[np.float64]


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
            base_slab = copy.deepcopy(self.slab)
            base_slab.translatenormal(offsets[0])
            base_vertex_xys = np.asarray([base_slab.intersectx(x) for x in slab_vertex_xys[:,0]])
            valid_vertex_ind_0, valid_vertex_ind_f = self.inds_in_domain(base_vertex_xys)
            if offsets[-1] > 0.0:
                top_slab = copy.deepcopy(self.slab)
                top_slab.translatenormal(offsets[-1])
                top_ind_0, top_ind_f = self.inds_in_domain(np.asarray([top_slab.intersectx(x) for x in slab_vertex_xys[:,0]]))
                valid_vertex_ind_0 = max(valid_vertex_ind_0, top_ind_0)
                valid_vertex_ind_f = min(valid_vertex_ind_f, top_ind_f)

            # restrict the vertex coordinates to the valid values
            slab_vertex_xys = slab_vertex_xys[valid_vertex_ind_0:valid_vertex_ind_f,:]
            base_vertex_xys = base_vertex_xys[valid_vertex_ind_0:valid_vertex_ind_f,:]

            # work out the dimensions of our grid
            num_valid_us = len(slab_vertex_xys)
            nums_sub_layers = [math.ceil(abs(thickness)/tres) 
                               for thickness, tres in zip(self.layer_thicknesses, self.layer_tres)][::-1] # NOTE reversed
            total_layers = sum(nums_sub_layers)

            # pre-allocate memory for the full coordinate system and the cells
            vertex_xys = np.empty(((total_layers+1)*num_valid_us, 2))
            vertex_xys[:num_valid_us,:] = base_vertex_xys
            cells = np.empty((total_layers*(num_valid_us-1), 4), dtype=np.int32)
            cell_out_flux = np.empty((total_layers*(num_valid_us-1),2))
            layer_cell_inds = [np.empty((num_sub_layers, num_valid_us-1), dtype=np.int32) for num_sub_layers in nums_sub_layers]
            dof_xys = np.empty((total_layers*(num_valid_us-1), 2))

            # set up near-orthogonal slab coordinate system
            vertex_ss = slab_vertex_us[valid_vertex_ind_0:valid_vertex_ind_f]*self.slab.length
            vertex_ss = vertex_ss - vertex_ss[0]
            vertex_ts = np.empty(total_layers + 1)
            vertex_ts[0] = offsets[0]

            layer_ind = 0
            for l, offset in enumerate(offsets[1:]):
                prev_offset = offsets[l]
                # the number of sublayers based on the layer thickness and resolution
                num_sub_layers = nums_sub_layers[l]
                sub_offsets = np.linspace(prev_offset, offset, num=num_sub_layers+1)
                # loop over the sub layers
                for sl, sub_offset in enumerate(sub_offsets[1:]):
                    prev_sub_offset = sub_offsets[sl]
                    sub_thickness = sub_offset - prev_sub_offset
                    # set up a spline for this layer
                    layer_spline = copy.deepcopy(self.slab)
                    layer_spline.translatenormal(sub_offset)
                    # work out coordinates...
                    vertex_ind = (layer_ind + 1)*num_valid_us
                    current_vertex_inds = list(range(vertex_ind, vertex_ind+num_valid_us))
                    # in xy space
                    vertex_xys[current_vertex_inds,:] = np.asarray([layer_spline.intersectx(x) for x in slab_vertex_xys[:,0]])
                    # and t space
                    vertex_ts[layer_ind + 1] = sub_offset
                    # set up a spline for halfway through the layer
                    halflayer_spline = copy.deepcopy(self.slab)
                    halflayer_spline.translatenormal(prev_sub_offset + 0.5*sub_thickness)
                    # use the halfway spline to figure out flux vectors out (of the rhs) of the cell
                    halflayer_tangents = np.stack([np.ones(num_valid_us-1), halflayer_spline.cs(slab_vertex_xys[1:,0], nu=1)], axis=1)
                    halflayer_tangmags = np.sqrt(np.sum(halflayer_tangents**2, axis=1))
                    halflayer_tangents = (halflayer_tangents.T/halflayer_tangmags).T
                    current_cell_inds = list(range(layer_ind*(num_valid_us-1), (layer_ind+1)*(num_valid_us-1)))
                    cell_out_flux[current_cell_inds,:] = halflayer_tangents
                    # also use the halfway spline to work out the dof xys at the mid x points of each cell
                    dof_xys[current_cell_inds,:] = np.asarray([halflayer_spline.intersectx(x) for x in 0.5*(slab_vertex_xys[:-1,0] + slab_vertex_xys[1:,0])])
                    # work out cells
                    for i in range(num_valid_us-1):
                        cell = [(layer_ind + 1)*num_valid_us + i + 1, (layer_ind + 1)*num_valid_us + i,
                                (layer_ind)*num_valid_us + i,   (layer_ind)*num_valid_us + i + 1]
                        cells[layer_ind*(num_valid_us-1) + i, :] = cell
                    layer_cell_inds[l][sl,:] = np.arange(layer_ind*(num_valid_us-1), (layer_ind+1)*(num_valid_us-1), dtype=np.int32)
                    # increment layer index
                    layer_ind += 1

            self._mesh = SlabMeshFlux(vertex_xys=vertex_xys, vertex_ss=vertex_ss, vertex_ts=vertex_ts, 
                                  cells=cells, layer_cell_inds=layer_cell_inds[::-1], 
                                  dof_xys=dof_xys, cell_out_flux=cell_out_flux)
            # NOTE: we reverse layer_cell_inds to be in the same order as layer_h2os above
        return self._mesh

    @property
    def H2O_fluxes(self):
        if not hasattr(self, '_H2O_fluxes') or self._H2O_fluxes is None:
            self._H2O_fluxes = np.empty(self.mesh.cells.shape[0])
            for cell_inds in self.mesh.layer_cell_inds:
                for sub_cell_inds in cell_inds:
                    thicknesses = np.linalg.norm(self.mesh.vertex_xys[self.mesh.cells[sub_cell_inds,1]] - 
                                                 self.mesh.vertex_xys[self.mesh.cells[sub_cell_inds,2]], axis=1)
                    self._H2O_fluxes[sub_cell_inds] = self.H2Os[sub_cell_inds]*thicknesses*self.rhom*self.Vs*self.mesh.cell_out_flux[sub_cell_inds,0]
        return self._H2O_fluxes

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
# slab = create_slab(szdict['xs'], szdict['ys'], resscale, szdict['lc_depth'])
# _ = plot_slab(slab)

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
# tres = 1.0
# sres = 1.0
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
# csv_path = os.path.join(os.pardir, os.pardir, 'data', 'perple_x_v7.1.9')
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
#                            slab=slab, Tgrid=tfgrid, 
#                            Tname='Temperature::PotentialTemperature', 
#                            coast_distance=szdict['coast_distance'], 
#                            sztype=szdict['sztype'], lc_depth=szdict['lc_depth'], trench_length=szdict['trench_length'], Vs=szdict['Vs'])

# %% tags=["active-ipynb"]
# testslab.conservation_errors

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(10,8))
# testslab.plot_xy(ax, C=testslab.H2Os, cmap='coolwarm', edgecolor = 'none', lw=0.5)
# ax.set_aspect(3)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslab.plot_st(ax, C=testslab.cumulative_H2O_losses, cmap='coolwarm', edgecolor = 'black', lw=0.5)
# ax.set_aspect(5)
# fig.show()

# %% tags=["active-ipynb"]
# import json
# with open(os.path.join(basedir, os.pardir, "data", "perple_x_v7.1.9", "abers_25.json"), "r") as file:
#     abers_25 = json.load(file)

# %% tags=["active-ipynb"]
# layer_h2os_meemum = [
#     PerpleXMeemum(basename, abers_25[basename]['component_masses'], abers_25[basename]['excluded_phases'], abers_25[basename]['solution_models']) for basename in [szdict['sed_type'], 'upvolc_25', 'lovolc_25', 'dike_25', 'gabbro_25', 'DMMdamp_25']
# ]

# %% tags=["active-ipynb"]
# testslabmeemum = SlabDehydrationVertical(sres, tres, layer_thicknesses, layer_h2os_meemum, layer_tres=None,
#                            slab=slab, Tgrid=tfgrid, 
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
#                            slab=slab, Tgrid=tfgrid, 
#                            Tname='Temperature::PotentialTemperature', 
#                            coast_distance=szdict['coast_distance'], 
#                            sztype=szdict['sztype'], lc_depth=szdict['lc_depth'], trench_length=szdict['trench_length'], Vs=szdict['Vs'])

# %% tags=["active-ipynb"]
# fix, ax = pl.subplots(figsize=(5,10))
# indices = np.argsort(-testslab.mesh.dof_xys[:,1])
# ax.plot(testslab.total_cumulative_H2O_losses/1000.0, testslab.mesh.dof_xys[indices,1], label='vertical')
# indicesog = np.argsort(-testslabog.mesh.dof_xys[:,1])
# ax.plot(testslabog.total_cumulative_H2O_losses/1000.0, testslabog.mesh.dof_xys[indicesog,1], label='normal', ls='--')
# indicesmm = np.argsort(-testslabmeemum.mesh.dof_xys[:,1])
# ax.plot(testslabmeemum.total_cumulative_H2O_losses/1000.0, testslabmeemum.mesh.dof_xys[indicesmm,1], label='meemum', ls='-.')
# ax.set_xlabel('Cumulative water loss')
# ax.set_ylabel('y (km)')
# _ = ax.legend()

# %%
