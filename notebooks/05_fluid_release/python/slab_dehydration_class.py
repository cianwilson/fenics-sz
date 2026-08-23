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
import numpy as np
import math
import matplotlib.pyplot as pl
import shapely.geometry as geo
import pyvista as pv
import pathlib
import hashlib
import zipfile
import requests
from dataclasses import dataclass
import itertools

# %%
import fenics_sz.utils
from fenics_sz.sz_problems.sz_params import allsz_params
from fenics_sz.sz_problems.sz_slab import create_slab, plot_slab
from fenics_sz.fluid_release.perple_x_class import PerpleXGrid


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

# %%

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

# %%
@dataclass
class SlabMesh:
    vertex_xys : np.typing.NDArray[np.float64]
    vertex_ss : np.typing.NDArray[np.float64]
    vertex_ts : np.typing.NDArray[np.float64]
    cells : np.typing.NDArray[np.int_]
    layer_cell_inds : list
    dof_xys : np.typing.NDArray[np.float64]


# %%
class SlabDehydration:
    """
    
    """
    def __init__(self, sres : float, tres : float, 
                 layer_thicknesses : (list[np.float64] | np.typing.NDArray[np.float64]), 
                 layer_h2os : list[np.int_], 
                 layer_tres : (list[np.float64] | np.typing.NDArray[np.float64]) = None, 
                 sz = None, Tgrid = None, slab = None, 
                 zmin : float = 15.0, zmax : float = None,
                 coast_distance : float = None, sztype : str = None, lc_depth : float = None,
                 trench_length : float = None,
                 rhoc : float = None, rhom : float = None,  Vs : float = None,
                 Tname : str ='T', add_adiabat : bool =True):

        # resolutions
        self.sres = sres
        self.tres = tres

        # layer specifications:
        # - thicknesses
        self.layer_thicknesses = layer_thicknesses
        # - hydrations
        if len(layer_h2os) != len(layer_thicknesses):
            raise RuntimeError("Length of layer_h2os must match layer_thicknesses.")
        self.layer_h2os = layer_h2os
        # - resolutions (per layer)
        if layer_tres is not None:
            self.layer_tres = [layer_tres[t] if t < len(layer_tres) 
                                                    and layer_tres[t] is not None 
                                                else self.tres 
                                                for t in range(len(layer_thicknesses))]
        else:
            self.layer_tres = [self.tres]*len(layer_thicknesses)

        # subduction zone class
        self._sz = sz

        # pyvista temperature grid
        self._Tgrid = Tgrid
        # temperature field name in pyvista grid
        if not Tname in self.Tgrid.array_names:
            raise RuntimeError("Temperature field {:s} not found.  Please check Tname parameter.".format(Tname))
        self.Tname = Tname

        # the slab spline
        self._slab = slab

        # the minimum and maximum depths allowed to be considered
        self.zmin = zmin
        self.zmax = zmax

        # other parameters required if sz not supplied
        self._coast_distance = coast_distance
        self._sztype = sztype
        self._lc_depth = lc_depth
        self._trench_length = trench_length
        self._rhoc = rhoc
        self._rhom = rhom
        self._Vs = Vs

        # does the temperature field require an adiabat to be added?
        self.add_adiabat = add_adiabat

    @property
    def sz(self):
        if self._sz is None: raise RuntimeError("No sz supplied.")
        return self._sz

    @property
    def Tgrid(self):
        if self._Tgrid is None:
            try:
                self._Tgrid = fenics_sz.utils.plot.grids_scalar(self.sz.T_i)[0]
            except RuntimeError:
                raise RuntimeError("Must supply Tgrid if sz not supplied.")
        return self._Tgrid

    @property
    def slab(self):
        if self._slab is None:
            try:
                self._slab = self.sz.geometry.slab_spline
            except RuntimeError:
                raise RuntimeError("Must supply slab if sz not supplied.")
        return self._slab

    @property
    def coast_distance(self):
        if self._coast_distance is None: 
            try:
                self._coast_distance = self.sz.deltaxcoast
            except RuntimeError:
                raise RuntimeError("Must supply coast_distance if sz not supplied.")
        return self._coast_distance

    @property
    def sztype(self):
        if self._sztype is None: 
            try:
                self._sztype = self.sz.sztype
            except RuntimeError:
                raise RuntimeError("Must supply sztype if sz not supplied.")
        return self._sztype

    @property
    def lc_depth(self):
        if self._lc_depth is None: 
            try:
                self._lc_depth = self.sz.deltazc if self.sztype == "continental" else 7.0
            except RuntimeError:
                raise RuntimeError("Must supply lc_depth if sz not supplied and sztype is continental.")
        return self._lc_depth

    @property
    def trench_length(self):
        if self._trench_length is None:
            raise RuntimeError("Must supply trench_length.")
        return self._trench_length

    @property
    def rhoc(self):
        if self._rhoc is None: 
            try:
                self._rhoc = self.sz.rhoc*self.sz.rho0 if self.sztype == "continental" else 3.0e3
            except RuntimeError:
                self._rhoc = 2750.0 if self.sztype == "continental" else 3.0e3
        return self._rhoc

    @property
    def rhom(self):
        if self._rhom is None:
            try:
                self._rhom = self.sz.rhom*self.sz.rho0
            except RuntimeError:
                self._rhom = 3300.0
        return self._rhom

    @property
    def Vs(self):
        if self._Vs is None:
            try:
                self._Vs = self.sz.Vs
            except RuntimeError:
                raise RuntimeError("Must supply Vs if sz not supplied.")
        return self._Vs

    def inds_in_domain(self, coords):
        # FIXME: this logic could be improved to take into account the bathymetry etc
        below_trench = (coords[:,0] > self.slab.x[0]) & (coords[:,1] < self.slab.y[0])
        valid_ind_0 = np.argmax(below_trench) if np.any(below_trench) else None
        above_base = (coords[:,0] < self.slab.x[-1]) & (coords[:,1] > self.slab.y[-1])
        valid_ind_f = len(coords) - np.argmax(above_base[::-1]) if np.any(above_base) else None
        return valid_ind_0, valid_ind_f

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

            # work out slab vertex normal
            slab_vertex_normals = np.stack([-self.slab.cs(slab_vertex_xys[:,0], nu=1), 
                                            np.ones_like(slab_vertex_us)], axis=1)
            slab_vertex_normmags = np.sqrt(np.sum(slab_vertex_normals**2, axis=1))
            slab_vertex_normals = (slab_vertex_normals.T/slab_vertex_normmags).T

            # accumulate the offsets from layers above, at and below the slab (and reverse the list - base first)
            offsets = (list(itertools.accumulate([t for t in self.layer_thicknesses if t > 0])) + \
                       [0.0] + \
                       list(itertools.accumulate([t for t in self.layer_thicknesses if t < 0])))[::-1]
            # find the valid range of coordinates based on the basal (and top if offset) layer
            base_vertex_xys = slab_vertex_xys+offsets[0]*slab_vertex_normals
            valid_vertex_ind_0, valid_vertex_ind_f = self.inds_in_domain(base_vertex_xys)
            if offsets[-1] > 0.0:
                top_ind_0, top_ind_f = self.inds_in_domain(slab_vertex_xys+offsets[-1]*slab_vertex_normals)
                valid_vertex_ind_0 = max(valid_vertex_ind_0, top_ind_0)
                valid_vertex_ind_f = min(valid_vertex_ind_f, top_ind_f)

            # restrict the vertex coordinates to the valid values
            slab_vertex_xys = slab_vertex_xys[valid_vertex_ind_0:valid_vertex_ind_f,:]
            slab_vertex_normals = slab_vertex_normals[valid_vertex_ind_0:valid_vertex_ind_f,:]
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

            # set up orthogonal slab coordinate system
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
                    # work out coordinates...
                    vertex_ind = (layer_ind + 1)*num_valid_us
                    current_vertex_inds = list(range(vertex_ind, vertex_ind+num_valid_us))
                    # in xy space
                    vertex_xys[current_vertex_inds,:] = slab_vertex_xys + sub_offset*slab_vertex_normals
                    # and t space
                    vertex_ts[layer_ind + 1] = sub_offset
                    # work out cells
                    for i in range(num_valid_us-1):
                        # lower left, lower right, upper right, upper left
                        cell = [
                                (layer_ind)*num_valid_us + i,   (layer_ind)*num_valid_us + i + 1,
                                (layer_ind + 1)*num_valid_us + i + 1, (layer_ind + 1)*num_valid_us + i
                               ]
                        cells[layer_ind*(num_valid_us-1) + i, :] = cell
                        # work out dof xys (here the centroids of the cells)
                        centroid = geo.Polygon(vertex_xys[cell,:]).centroid
                        dof_xys[layer_ind*(num_valid_us-1) + i, :] = [centroid.x, centroid.y]
                    # we fill in layer_cell_inds from top to bottom to match layer_h2os and other user input
                    layer_cell_inds[num_outer_layers-1-l][num_sub_layers-1-sl,:] = np.arange(layer_ind*(num_valid_us-1), (layer_ind+1)*(num_valid_us-1), dtype=np.int32)
                    layer_ind += 1

            self._mesh = SlabMesh(vertex_xys=vertex_xys, vertex_ss=vertex_ss, vertex_ts=vertex_ts, 
                                  cells=cells, layer_cell_inds=layer_cell_inds, dof_xys=dof_xys)
            # NOTE: during construction we reversed layer_cell_inds to be in the same 
            # order as layer_h2os above (i.e. it goes from top to bottom, while 
            # everything else is natively ordered from bottom to top but this should 
            # be OK as the other things should be accesses through layer_cell_inds)
        return self._mesh

    def zsurface(self, x):
        return np.minimum(np.maximum(-self.slab.y[0]*(1.0 - x/max(self.coast_distance, np.finfo(float).eps)), 
                                     0.0), 
                          -self.slab.y[0])

    @property
    def Ts(self):
        if not hasattr(self, '_Ts') or self._Ts is None:
            points = pv.PolyData(np.concatenate([self.mesh.dof_xys, np.zeros((self.mesh.dof_xys.shape[0],1))], axis=1))
            self._Ts = points.sample(self.Tgrid)[self.Tname]
            if self.add_adiabat: self._Ts -= 0.3*(self.mesh.dof_xys[:,1]+self.zsurface(self.mesh.dof_xys[:,0]))
        return self._Ts

    @property
    def Ps(self):
        if not hasattr(self, '_Ps') or self._Ps is None:
            rhow = 1000.0 # water density, kg/m^3
            g = 9.81 # gravity magnitude, m/s^2
            # depth of the slab above the dofs
            zslab = np.asarray([-self.slab.intersectx(x)[1] for x in self.mesh.dof_xys[:,0]])
            # depth of surface above the dofs
            zsurface = self.zsurface(self.mesh.dof_xys[:,0])
            self._Ps = (rhow*zsurface + 
                        # ^- contribution of water
                        self.rhoc*(np.minimum(self.lc_depth, zslab)-zsurface) + 
                        # ^- contribution of crust
                        self.rhom*(np.maximum(0.0, zslab-self.lc_depth)) + 
                        # ^- contribution of overriding mantle above slab
                        self.rhom*(-self.mesh.dof_xys[:,1]-zslab))*g/1.e6 
                        # ^- contribution of subducting slab 
                        # (subtracts contribution if above slab 
                        #  so only valid if rhom is used in slab as well as mantle)
        return self._Ps

    @property
    def maxH2Os(self):
        if not hasattr(self, '_maxH2Os') or self._maxH2Os is None:
            self._maxH2Os = np.empty(self.mesh.cells.shape[0])
            for cell_inds, h2ogrid in zip(self.mesh.layer_cell_inds, self.layer_h2os):
                cell_inds_f = cell_inds.flatten()
                self._maxH2Os[cell_inds_f] = h2ogrid.eval(self.Ps[cell_inds_f], self.Ts[cell_inds_f])['H2O']/100.0
        return self._maxH2Os
    
    @property
    def H2Os(self):
        if not hasattr(self, '_H2Os') or self._H2Os is None:
            self._H2Os = np.empty(self.mesh.cells.shape[0])
            for cell_inds in self.mesh.layer_cell_inds:
                self._H2Os[cell_inds] = np.minimum.accumulate(self.maxH2Os[cell_inds], axis=1)
        return self._H2Os
    
    @property
    def H2O_fluxes(self):
        if not hasattr(self, '_H2O_fluxes') or self._H2O_fluxes is None:
            self._H2O_fluxes = np.empty(self.mesh.cells.shape[0])
            # basing the flux off the thickness alone is valid for both
            # a slab normal coordinate system and a vertical cell boundary
            # system
            thicknesses = (self.mesh.vertex_ts[1:]-self.mesh.vertex_ts[:-1])[::-1]
            # thicknesses has to be reversed to match the top to bottom ordered of layer_cell_inds
            i = 0
            for cell_inds in self.mesh.layer_cell_inds:
                for sub_cell_inds in cell_inds:
                    self._H2O_fluxes[sub_cell_inds] = self.H2Os[sub_cell_inds]*thicknesses[i]*self.rhom*self.Vs
                    i += 1
        return self._H2O_fluxes

    @property
    def H2O_losses(self):
        if not hasattr(self, '_H2O_losses') or self._H2O_losses is None:
            # we set this to zero as we assume that the first entry is always 0
            # i.e. that the flux out of our first cell exactly matches the flux in
            self._H2O_losses = np.zeros(self.mesh.cells.shape[0])
            for cell_inds in self.mesh.layer_cell_inds:
                for sub_cell_inds in cell_inds:
                    self._H2O_losses[sub_cell_inds[1:]] = self.H2O_fluxes[sub_cell_inds[:-1]] - self.H2O_fluxes[sub_cell_inds[1:]]
        return self._H2O_losses

    @property
    def cumulative_H2O_losses(self):
        if not hasattr(self, '_cumulative_H2O_losses') or self._cumulative_H2O_losses is None:
            self._cumulative_H2O_losses = np.empty(self.mesh.cells.shape[0])
            for cell_inds in self.mesh.layer_cell_inds:
                self._cumulative_H2O_losses[cell_inds] = np.cumsum(self.H2O_losses[cell_inds], axis=1)
        return self._cumulative_H2O_losses

    @property
    def total_cumulative_H2O_losses(self):
        if not hasattr(self, '_total_cumulative_H2O_losses') or self._total_cumulative_H2O_losses is None:
            indices = np.argsort(-self.mesh.dof_xys[:,1])
            self._total_cumulative_H2O_losses = np.cumsum(self.H2O_losses[indices])
        return self._total_cumulative_H2O_losses

    @property
    def conservation_errors(self):
        if not hasattr(self, '_conservation_errors') or self._conservation_errors is None:
            self._conservation_errors = [np.empty((len(cell_inds))) for cell_inds in self.mesh.layer_cell_inds]
            for l, cell_inds in enumerate(self.mesh.layer_cell_inds):
                for s, sub_cell_inds in enumerate(cell_inds):
                    self._conservation_errors[l][s] = (self.H2O_fluxes[sub_cell_inds[0]] - \
                                                       self.H2O_losses[sub_cell_inds].sum() - \
                                                       self.H2O_fluxes[sub_cell_inds[-1]])/\
                                                          max(self.H2O_fluxes[sub_cell_inds[0]], np.finfo(float).eps)
        return self._conservation_errors


    @property
    def water_retention(self):
        if not hasattr(self, '_water_retention') or self._water_retention is None:
            self._water_retention = sum([self.H2O_fluxes[sub_cell_inds[-1]] for cell_inds in self.mesh.layer_cell_inds for sub_cell_inds in cell_inds])*self.trench_length
        return self._water_retention

    def plot_xy(self, ax, C=None, **mpl_kwargs):
        nvt = len(self.mesh.vertex_ts)
        nvs = len(self.mesh.vertex_ss)
        X = self.mesh.vertex_xys[:,0].reshape((nvt, nvs))
        Y = self.mesh.vertex_xys[:,1].reshape((nvt, nvs))
        return self.plot(ax, X, Y, C=C, **mpl_kwargs)
    
    def plot_st(self, ax, C=None, **mpl_kwargs):
        coords = np.empty_like(self.mesh.vertex_xys)
        S, T = np.meshgrid(self.mesh.vertex_ss, self.mesh.vertex_ts)
        return self.plot(ax, S, T, C=C, **mpl_kwargs)
        
    def plot(self, ax, X, Y, C=None, **mpl_kwargs):
        if C is None: 
            C = np.zeros((Y.shape[0] - 1, X.shape[1] - 1))
        if isinstance(C, np.ndarray) and len(C.shape) == 1:
            C = C.reshape((Y.shape[0]-1, X.shape[1]-1))
        pcm = ax.pcolormesh(X, Y, C, **mpl_kwargs)
        return pcm

# %% tags=["active-ipynb"]
# dmm_thickness = 2.0
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
#     # None,
#     None,
#     None,
#     None,
#     1.4,
# ]

# %% tags=["active-ipynb"]
# testslab = SlabDehydration(10.0, 1.0, layer_thicknesses, layer_h2os, layer_tres=None,
#                            slab=slab, Tgrid=tfgrid, 
#                            Tname='Temperature::PotentialTemperature', 
#                            coast_distance=szdict['coast_distance'], 
#                            sztype=szdict['sztype'], lc_depth=szdict['lc_depth'], trench_length=szdict['trench_length'], Vs=szdict['Vs'])

# %% tags=["active-ipynb"]
# sum([errors.sum() for errors in testslab.conservation_errors])

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(10,8))
# testslab.plot_xy(ax, C=testslab.Ts, cmap='coolwarm', edgecolor = 'none')
# ax.set_aspect(3)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(10,8))
# testslab.plot_xy(ax, C=testslab.H2O_fluxes, cmap='coolwarm', edgecolor = 'none', lw=0.5)
# ax.set_aspect(3)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslab.plot_st(ax, C=testslab.maxH2Os, cmap = 'coolwarm', edgecolor = 'black', linewidth=0.5, shading='flat')
# ax.set_aspect(10.0)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslab.plot_st(ax, C=testslab.H2Os, cmap = 'coolwarm', edgecolor = 'black', linewidth=0.5, shading='flat')
# ax.set_aspect(10.0)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslab.plot_st(ax, C=testslab.H2O_fluxes, cmap = 'coolwarm', edgecolor = 'black', linewidth=0.5, shading='flat')
# ax.set_aspect(10.0)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslab.plot_st(ax, C=testslab.H2O_losses, cmap = 'coolwarm', edgecolor = 'black', linewidth=0.5, shading='flat')
# ax.set_aspect(10.0)
# fig.show()

# %% tags=["active-ipynb"]
# fig, (ax, axc) = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1,0.05])
# pcm = testslab.plot_st(ax, C=testslab.cumulative_H2O_losses/1000.0, cmap = 'coolwarm', edgecolor = 'black', linewidth=0.5, shading='flat')
# ax.set_aspect(10.0)
# fig.colorbar(pcm, cax=axc, orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fix, ax = pl.subplots(figsize=(5,10))
# indices = np.argsort(-testslab.mesh.dof_xys[:,1])
# ax.plot(testslab.total_cumulative_H2O_losses/1000.0, testslab.mesh.dof_xys[indices,1])

# %%

# %%
