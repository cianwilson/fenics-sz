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
import warnings
from dataclasses import dataclass
import json
from scipy import integrate as integ
from scipy import optimize as opt

# %%
import fenics_sz.utils
from fenics_sz.sz_problems.sz_params import allsz_params
from fenics_sz.sz_problems.sz_slab import create_slab, plot_slab
from fenics_sz.fluid_release.perple_x_class import PerpleXGrid
from fenics_sz.fluid_release.perple_x_class_meemum import PerpleXMeemum
from fenics_sz.fluid_release.vertical_slab_dehyrdation_class import SlabDehydrationVerticalFlux


# %%
class SlabRehydration(SlabDehydrationVerticalFlux):
    def __init__(self, *args, **kwargs):
        # call parent constructor
        super().__init__(*args, **kwargs)

        # this version of the class requires that all layers contain H2O and CO2 components
        for h2ogrid in self.layer_h2os:
            missing = set(['H2O', 'CO2']) - h2ogrid.component_masses.keys()
            if len(missing) > 0:
                if not h2ogrid.initialized:
                    for m in missing: h2ogrid.component_masses[m] = 0.0
                else:
                    raise RuntimeError(f"{g.basename} (layer_h2os) has already been initialized and is missing components: {sorted(missing)}")

    @property
    def maxH2Os(self):
        if not hasattr(self, '_maxH2Os') or self._maxH2Os is None:
            self._maxH2Os = np.empty(self.mesh.cells.shape[0])

            thicknesses = (self.mesh.vertex_ts[1:]-self.mesh.vertex_ts[:-1])[::-1]
            deltaxs = self.mesh.vertex_xys[1:,0]-self.mesh.vertex_xys[:-1,0]
            fluid_components = ['H2O', 'CO2']
            param_names = fluid_components + ['q']
            rhof = 1000.0
            rhos = 3300.0

            def residuals(x, P : float, T :float, 
                          thickness : float, deltax : float, 
                          comps : dict, cfsb : dict, cssl : dict, 
                          qb : float, phil : float, Vs : float):
                assert(len(x) == len(fluid_components)+1)
                for i, k in enumerate(fluid_components):
                    comps[k] = x[i]
                q = x[-1]

                pxout = h2ogrid.eval(P, T, **comps)

                F = pxout['F_f'][0]/100.0
                css = {k : pxout[k][0]/100.0       for k in fluid_components}
                cfs = {k : pxout[k+'_f'][0]/100.0  for k in fluid_components}

                phi = rhos*F/(rhof*(1-F) + rhos*F)

                # residuals for cfs
                rs = [rhof*deltax*(q*cfs[k] - qb*cfsb[k]) + \
                      thickness*Vs*rhos*(phi*css[k] - phil*cssl[k]) 
                      for k in fluid_components]
                # residual for q
                rs.append(rhof*deltax*(q - qb) + thickness*Vs*rhos*(phi - phil))

                return rs

            # bounds: bulk wt% for each fluid component in [0, 100], flux q in [0, inf)
            lb = [0.0]*len(fluid_components) + [0.0]
            ub = [100.0]*len(fluid_components) + [np.inf]

            # tolerances for the post-solve diagnostics below - tune these to the
            # actual residual/bound scale of the problem if they prove too tight/loose
            restol = 1.e-6
            boundtol = 1.e-6

            qs_below = np.zeros(self.mesh.layer_cell_inds[0].shape[1])     
            Fs_below = np.zeros(self.mesh.layer_cell_inds[0].shape[1])     
            cfs_below = {k:np.zeros(self.mesh.layer_cell_inds[0].shape[1]) for k in fluid_components}
            l = 0
            # reverse everything so that we're going from bottom to top
            for cell_inds, h2ogrid in zip(self.mesh.layer_cell_inds[::-1], self.layer_h2os[::-1]):
                # also reversed to go from bottom to top
                for sl_cell_inds in cell_inds[::-1]:
                    # initial bulk component masses (set by user input)
                    comps = h2ogrid.component_masses.copy()
                    # incoming porosity from the left
                    phil = 0.0
                    # incoming solid composition of components
                    # (that also occur in the fluid)
                    cssl = {k : comps[k]/100.0 for k in fluid_components}
                    for c, sl_cell_ind in enumerate(sl_cell_inds):
                        # incoming fluid flux and mass fraction from below
                        qb = qs_below[c]
                        Fb = Fs_below[c]
                        # incoming fluid compositions from below
                        cfsb = {k : v[c] for k,v in cfs_below.items()}

                        x0 = []
                        for k in fluid_components:
                            x0.append((Fb*cfsb[k]+ (1-Fb)*cssl[k])*100)
                        x0.append(qb)
                        sol = opt.least_squares(lambda x: residuals(x, 
                                                           self.Ps[sl_cell_ind], self.Ts[sl_cell_ind], thicknesses[l], deltaxs[c], 
                                                           comps, cfsb, cssl, qb, phil, self.Vs), 
                                        x0, bounds=(lb, ub))
                        if not sol.success:
                            raise RuntimeError(
                                f"Nonlinear solve failed for cell {sl_cell_ind} "
                                f"(layer {l}, column {c}, P={self.Ps[sl_cell_ind]}, T={self.Ts[sl_cell_ind]}): "
                                f"{sol.message} (max|residual|={np.max(np.abs(sol.fun))})"
                            )

                        # a successful solve just means the optimizer stalled (ftol/xtol/gtol) -
                        # it doesn't guarantee we actually hit a zero residual, so check explicitly
                        resnorm = np.max(np.abs(sol.fun))
                        if resnorm > restol:
                            warnings.warn(
                                f"Cell {sl_cell_ind} (layer {l}, column {c}): solve reported success but "
                                f"max|residual|={resnorm:.3e} exceeds restol={restol:.1e}."
                            )

                        # and check whether any parameter ended up sitting on a bound - if so, the
                        # true (unconstrained) root likely lies outside the physically valid region
                        at_lower = np.isclose(sol.x, lb, atol=boundtol, rtol=0.0)
                        at_upper = np.isclose(sol.x, ub, atol=boundtol, rtol=0.0)
                        if np.any(at_lower) or np.any(at_upper):
                            hit = [f"{name} at {'lower' if lo else 'upper'} bound ({val:.6g})"
                                   for name, lo, up, val in zip(param_names, at_lower, at_upper, sol.x) if lo or up]
                            warnings.warn(
                                f"Cell {sl_cell_ind} (layer {l}, column {c}): solve hit bound(s): {', '.join(hit)} "
                                f"(max|residual|={resnorm:.3e})."
                            )

                        # need to find a better way of getting these out of the actual solution
                        # rather than recalculating after the nonlinear solve
                        for i, k in enumerate(fluid_components): comps[k] = sol.x[i]
                        q = sol.x[-1]
                        pxout = h2ogrid.eval(self.Ps[sl_cell_ind], self.Ts[sl_cell_ind], **comps)
                        F = pxout['F_f'][0]/100.0

                        qs_below[c] = q
                        Fs_below[c] = F
                        for k in fluid_components: 
                            cfs_below[k][c] = pxout[k+'_f'][0]/100.0
                            cssl[k] = pxout[k][0]/100.0
                        phil = rhos*F/(rhof*(1-F) + rhos*F)

                        # save wt % H2O of solid from perple_x
                        self._maxH2Os[sl_cell_ind] = pxout['H2O'][0]/100.0
                    l += 1
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
# tres = 2
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
# with open(os.path.join(basedir, os.pardir, "data", "perple_x_v7.1.9", "abers_25", "abers_25.json"), "r") as file:
#     abers_25 = json.load(file)

# %% tags=["active-ipynb"]
# layer_h2os_meemum = [
#     PerpleXMeemum(basename, abers_25[basename]['component_masses'], abers_25[basename]['excluded_phases'], abers_25[basename]['solution_models']) for basename in [szdict['sed_type'], 'upvolc_25', 'lovolc_25', 'dike_25', 'gabbro_25', 'DMMdamp_25']
# ]

# %%
reslab = SlabRehydration(sres, tres, layer_thicknesses, layer_h2os_meemum, layer_tres=None,
                           slab=slab1, Tgrid=tfgrid, 
                           Tname='Temperature::PotentialTemperature', 
                           coast_distance=szdict['coast_distance'], 
                           sztype=szdict['sztype'], lc_depth=szdict['lc_depth'], trench_length=szdict['trench_length'], Vs=szdict['Vs'])

# %%
fig, ax = pl.subplots(figsize=(20,20))
reslab.plot_st(ax, C=reslab.maxH2Os, cmap='coolwarm', edgecolor = 'black', lw=0.5)
ax.set_aspect(5)
fig.show()

# %%
fig, ax = pl.subplots(figsize=(20,20))
reslab.plot_st(ax, C=reslab.cumulative_H2O_losses, cmap='coolwarm', edgecolor = 'black', lw=0.5)
ax.set_aspect(5)
fig.show()

# %% tags=["active-ipynb"]
# testslabmeemum = SlabDehydrationVerticalFlux(sres, tres, layer_thicknesses, layer_h2os_meemum, layer_tres=None,
#                            slab=slab1, Tgrid=tfgrid, 
#                            Tname='Temperature::PotentialTemperature', 
#                            coast_distance=szdict['coast_distance'], 
#                            sztype=szdict['sztype'], lc_depth=szdict['lc_depth'], trench_length=szdict['trench_length'], Vs=szdict['Vs'])

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslabmeemum.plot_st(ax, C=testslabmeemum.maxH2Os, cmap='coolwarm', edgecolor = 'black', lw=0.5)
# ax.set_aspect(5)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# testslabmeemum.plot_st(ax, C=testslabmeemum.cumulative_H2O_losses, cmap='coolwarm', edgecolor = 'black', lw=0.5)
# ax.set_aspect(5)
# fig.show()

# %%
