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
from tqdm.auto import tqdm

# %%
import fenics_sz.utils
from fenics_sz.sz_problems.sz_params import allsz_params
from fenics_sz.sz_problems.sz_slab import create_slab, plot_slab
from fenics_sz.fluid_release.perple_x_class import PerpleXGrid
from fenics_sz.fluid_release.perple_x_class_meemum import PerpleXMeemum
from fenics_sz.fluid_release.vertical_slab_dehyrdation_class import SlabDehydrationVerticalFlux


# %%
@dataclass
class SlabSolution:
    cfs : dict[str,np.typing.NDArray[np.float64]]
    css : dict[str,np.typing.NDArray[np.float64]]
    cbs : dict[str,np.typing.NDArray[np.float64]]
    Ffs : np.typing.NDArray[np.float64]
    qs  : np.typing.NDArray[np.float64]
    rho : np.typing.NDArray[np.float64]


# %%
class SlabRehydration(SlabDehydrationVerticalFlux):
    def __init__(self, *args, **kwargs):
        # call parent constructor
        super().__init__(*args, **kwargs)

        self.fluid_components = ['H2O', 'CO2']

        # this version of the class requires that all layers contain H2O and CO2 components
        for h2ogrid in self.layer_h2os:
            missing = set(['H2O', 'CO2']) - h2ogrid.component_masses.keys()
            if len(missing) > 0:
                if not h2ogrid.initialized:
                    for m in missing: h2ogrid.component_masses[m] = 0.0
                else:
                    raise RuntimeError(f"{g.basename} (layer_h2os) has already been initialized and is missing components: {sorted(missing)}")

    @property
    def solution(self):
        if getattr(self, '_solution', None) is None:
            all_components = set.union(*(set(h2ogrid.component_masses.keys()) for h2ogrid in self.layer_h2os))

            cfs = {k:np.zeros(self.mesh.cells.shape[0]) for k in all_components}
            css = {k:np.zeros(self.mesh.cells.shape[0]) for k in all_components}
            cbs = {k:np.zeros(self.mesh.cells.shape[0]) for k in all_components}
            Ffs = np.zeros(self.mesh.cells.shape[0])
            qs  = np.zeros(self.mesh.cells.shape[0])
            rho = np.zeros(self.mesh.cells.shape[0])

            thicknesses = (self.mesh.vertex_ts[1:]-self.mesh.vertex_ts[:-1])[::-1]
            deltaxs = self.mesh.vertex_xys[1:,0]-self.mesh.vertex_xys[:-1,0]
            
            param_names = ['c_b'+fc for fc in self.fluid_components] + ['q']

            def residuals(x, P : float, T :float, 
                          thickness : float, deltax : float, 
                          fluid_components : list[str],
                          comps : dict, cfsb : dict, cssl : dict, 
                          rhob : float, rhol : float, 
                          qb : float, Fl : float, Vs : float):
                assert(len(x) == len(fluid_components)+1)
                for i, k in enumerate(fluid_components):
                    comps[k] = x[i]
                q = x[-1]

                pxout = h2ogrid.eval(P, T, **comps)

                Fwp = pxout['F_f'][0]
                rho = pxout['rho'][0]
                css = {k : pxout[k][0]/100.0       for k in fluid_components}
                cfs = {k : pxout[k+'_f'][0]/100.0  for k in fluid_components}

                rs = []
                if np.isnan(Fwp):
                    cfs = {k : cfsb[k] for k in fluid_components}
                    rs += [-deltax*qb*rhob*cfsb[k] \
                        + thickness*Vs*(rho*css[k] - (1.-Fl)*rhol*cssl[k]) 
                        for k in fluid_components]
                    rs.append(deltax*q*rho) # q = 0
                else:
                    F = Fwp/100.0
                    rs += [deltax*(q*rho*cfs[k] - qb*rhob*cfsb[k]) \
                        + thickness*Vs*((1.-F)*rho*css[k] - (1.-Fl)*rhol*cssl[k]) 
                        for k in fluid_components]
                    rs.append(deltax*(q*rho - qb*rhob) + thickness*Vs*((1.-F)*rho - (1.-Fl)*rhol))
                
                return rs

            # bounds: bulk wt% for each fluid component in [0, 100], flux q in [0, inf)
            lb = [0.0]*len(self.fluid_components) + [0.0]
            ub = [100.0]*len(self.fluid_components) + [np.inf]

            # tolerances for the post-solve diagnostics below - tune these to the
            # actual residual/bound scale of the problem if they prove too tight/loose
            restol = 1.e-6
            boundtol = 1.e-6

            qs_below = np.zeros(self.mesh.layer_cell_inds[0].shape[1])     
            Fs_below = np.zeros(self.mesh.layer_cell_inds[0].shape[1])     
            cfs_below = {k:np.zeros(self.mesh.layer_cell_inds[0].shape[1]) for k in self.fluid_components}
            rho_below = np.zeros(self.mesh.layer_cell_inds[0].shape[1])
            l = 0
            # progress bars: an outer one over sublayers (one tick per row, bottom to
            # top) and an inner one over the cells within the current sublayer, which
            # resets each time we move to a new sublayer
            n_sublayers = sum(len(cell_inds) for cell_inds in self.mesh.layer_cell_inds)
            outer_pbar = tqdm(total=n_sublayers, desc="Layers     ", position=0)
            inner_pbar = tqdm(total=0, desc="Layer cells", position=1, leave=False)
            # reverse everything so that we're going from bottom to top
            for cell_inds, h2ogrid in zip(self.mesh.layer_cell_inds[::-1], self.layer_h2os[::-1]):
                # also reversed to go from bottom to top
                for sl_cell_inds in cell_inds[::-1]:
                    # reset the inner progress bar for this sublayer
                    inner_pbar.reset(total=len(sl_cell_inds))
                    # initial bulk component masses (set by user input)
                    comps = h2ogrid.component_masses.copy()
                    pxout = h2ogrid.eval(self.Ps[sl_cell_inds[0]], self.Ts[sl_cell_inds[0]], **comps)
                    # incoming fluid from the left
                    Fwp = pxout['F_f'][0]
                    if np.isnan(Fwp):
                        Fl = 0.0
                    else:
                        Fl = Fwp/100.0
                    rhol = pxout['rho'][0]
                    # incoming solid composition of components
                    # (that also occur in the fluid)
                    cssl = {k : pxout[k][0]/100.0 for k in self.fluid_components}
                    for c, sl_cell_ind in enumerate(sl_cell_inds):
                        # incoming fluid flux and mass fraction from below
                        qb = qs_below[c]
                        Fb = Fs_below[c]
                        # incoming fluid compositions from below
                        cfsb = {k : v[c] for k,v in cfs_below.items()}
                        # incoming density from below
                        rhob = rho_below[c]

                        # set the initial guess (in wt %)
                        x0 = []
                        for k in self.fluid_components:
                            x0.append((Fb*cfsb[k]+ (1-Fb)*cssl[k])*100)
                        x0.append(qb)
                        sol = opt.least_squares(lambda x: residuals(x, 
                                                           self.Ps[sl_cell_ind], self.Ts[sl_cell_ind], 
                                                           thicknesses[l], deltaxs[c], 
                                                           self.fluid_components,
                                                           comps, cfsb, cssl, 
                                                           rhob, rhol, 
                                                           qb, Fl, self.Vs), 
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
                        for i, k in enumerate(self.fluid_components): comps[k] = sol.x[i]
                        q = sol.x[-1]
                        pxout = h2ogrid.eval(self.Ps[sl_cell_ind], self.Ts[sl_cell_ind], **comps)
                        rhol = pxout['rho'][0]

                        Fwp = pxout['F_f'][0]
                        if np.isnan(Fwp):
                            Fl = 0.0
                        else:
                            Fl = Fwp/100.0
                        
                        # record in array for next row
                        qs_below[c] = q
                        Fs_below[c] = Fl
                        rho_below[c] = rhol
                        for k in self.fluid_components: 
                            if not np.isnan(Fwp):
                                cfs_below[k][c] = pxout[k+'_f'][0]/100.0
                                # otherwise leave the fluid composition as it was
                                # in the cell below this one as it is not defined 
                                # if no fluid is present
                            cssl[k] = pxout[k][0]/100.0

                        # save data to grids
                        for k in all_components:
                            css[k][sl_cell_ind] = pxout.get(k, [0.0])[0]/100.0
                            cfs[k][sl_cell_ind] = pxout.get(k+'_f', [0.0])[0]/100.0
                            cbs[k][sl_cell_ind] = comps.get(k, 0.0)
                        Ffs[sl_cell_ind] = Fl
                        qs[sl_cell_ind] = q
                        rho[sl_cell_ind] = rhol

                        # update the inner progress bar
                        inner_pbar.update(1)
                    l += 1
                    # update the outer progress bar
                    outer_pbar.update(1)
            # close the progress bars
            inner_pbar.close()
            outer_pbar.close()
            # save the solution
            self._solution = SlabSolution(cfs=cfs, css=css, cbs=cbs, Ffs=Ffs, qs=qs, rho=rho)
        return self._solution

    @property
    def maxH2Os(self):
        return self.solution.css['H2O']

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
# tres = 1
# sres = 20
#
# # negative number implies below slab, positive implies above it
# layer_thicknesses = [
#                  2.0,            # above slab mantle
#                  -szdict['z15'], # sediments
#                  -0.3,           # upper volcanics
#                  -0.3,           # lower volcanics
#                  -1.4,           # dikes
#                  -5.0,           # gabbro
#                  -dmm_thickness  # subslab mantle
#                 ]

# %% tags=["active-ipynb"]
# with open(os.path.join(basedir, os.pardir, "data", "perple_x_v7.1.9", "abers_25", "abers_25.json"), "r") as file:
#     abers_25 = json.load(file)

# %% tags=["active-ipynb"]
# layer_h2os_meemum = [
#     PerpleXMeemum(basename, abers_25[basename]['component_masses'], abers_25[basename]['excluded_phases'], abers_25[basename]['solution_models']) for basename in ['DMMdry_25', szdict['sed_type'], 'upvolc_25', 'lovolc_25', 'dike_25', 'gabbro_25', 'DMMdamp_25']
# ]

# %% tags=["active-ipynb"]
# reslab = SlabRehydration(sres, tres, layer_thicknesses, layer_h2os_meemum, layer_tres=None,
#                            slab=slab1, Tgrid=tfgrid, 
#                            Tname='Temperature::PotentialTemperature', 
#                            coast_distance=szdict['coast_distance'], 
#                            sztype=szdict['sztype'], lc_depth=szdict['lc_depth'], trench_length=szdict['trench_length'], Vs=szdict['Vs'])

# %% tags=["active-ipynb"]
# _ = reslab.solution

# %% tags=["active-ipynb"]
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=reslab.solution.rho, cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=reslab.solution.css['H2O'], cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=reslab.solution.cbs['H2O'], cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=reslab.solution.Ffs, cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=reslab.solution.qs, cmap='coolwarm', edgecolor = 'none', lw=0.5, norm='log', vmin=1.e-6)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=reslab.solution.cbs['SiO2'], cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=reslab.solution.css['SiO2'], cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=reslab.solution.cfs['SiO2'], cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=np.sum(cbk for cbk in reslab.solution.cbs.values()), cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# rhos = 3300.0
# rhof = 1000.0
# F = reslab.solution.Ffs
# rho = reslab.solution.rho
#
# dcss = {k:np.zeros_like(csk) for k, csk in reslab.solution.css.items()}
# for cell_inds in reslab.mesh.layer_cell_inds:
#     for sub_cell_inds in cell_inds:
#         for k, csk in reslab.solution.css.items():
#             dcss[k][sub_cell_inds[1:]] = ((1-F[sub_cell_inds[1:]])*rho[sub_cell_inds[1:]]*csk[sub_cell_inds[1:]] \
#                 - (1-F[sub_cell_inds[:-1]])*rho[sub_cell_inds[:-1]]*csk[sub_cell_inds[:-1]])
#
#
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=dcss['SiO2'], cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# rhos = 3300.0
# rhof = 1000.0
# F = reslab.solution.Ffs
# phi = rhos*F/(rhof*(1-F) + rhos*F)
#
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=phi, cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=reslab.solution.Ffs, cmap='coolwarm', edgecolor = 'none', lw=0.5)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal')
# fig.show()

# %% tags=["active-ipynb"]
# # Lambda := rho/M_total (M_total = sum of all bulk component masses, cbs) must be
# # invariant along a row for holding non-fluid bulk masses fixed to exactly satisfy
# # their own mass-balance equations - see discussion. lam_step is the resulting
# # per-step fractional mass-balance error (identical for every non-fluid component,
# # so a single number captures all of them); lam_cum accumulates that drift along
# # each row. Computed here as postprocessing from the saved rho/cbs grids
# rho = reslab.solution.rho
# Mtot = sum(reslab.solution.cbs.values())
# Lambda = rho/Mtot
#
# lam_step = np.zeros_like(rho)
# lam_cum = np.zeros_like(rho)
# for cell_inds in reslab.mesh.layer_cell_inds:
#     for sub_cell_inds in cell_inds:
#         lam_step[sub_cell_inds[1:]] = (Lambda[sub_cell_inds[1:]] - Lambda[sub_cell_inds[:-1]]) / Lambda[sub_cell_inds[:-1]]
#         lam_cum[sub_cell_inds[1:]] = np.cumsum(np.log(Lambda[sub_cell_inds[1:]] / Lambda[sub_cell_inds[:-1]]))

# %% tags=["active-ipynb"]
# vmax = np.max(np.abs(lam_step))
#
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=lam_step, cmap='RdBu_r', edgecolor = 'none', lw=0.5, vmin=-vmax, vmax=vmax)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal', label='lam_step (relative)')
# fig.show()

# %% tags=["active-ipynb"]
# vmax = np.max(np.abs(lam_cum))
#
# fig, axs = pl.subplots(figsize=(20,5), nrows=2, height_ratios=[1, 0.1])
# pcm = reslab.plot_st(axs[0], C=lam_cum, cmap='RdBu_r', edgecolor = 'none', lw=0.5, vmin=-vmax, vmax=vmax)
# axs[0].set_aspect(10)
# fig.colorbar(pcm, cax=axs[1], orientation='horizontal', label='lam_cum (row-cumulative, log)')
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# reslab.plot_st(ax, C=reslab.solution.qs, cmap='coolwarm', edgecolor = 'none', lw=0.5)
# ax.set_aspect(10)
# fig.show()

# %% tags=["active-ipynb"]
# fig, ax = pl.subplots(figsize=(20,20))
# reslab.plot_st(ax, C=reslab.cumulative_H2O_losses, cmap='coolwarm', edgecolor = 'none', lw=0.5)
# ax.set_aspect(10)
# fig.show()

# %% tags=["active-ipynb"]
# # need to rethink this because it is no longer necessarily >= 0
# reslab.cumulative_H2O_losses.min()

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

# %% tags=["active-ipynb"]
# fix, ax = pl.subplots(figsize=(5,10))
# indices = np.argsort(-testslabmeemum.mesh.dof_xys[:,1])
# ax.plot(testslabmeemum.total_cumulative_H2O_losses/1000.0, testslabmeemum.mesh.dof_xys[indices,1], label='no rehydration')
# indicesre = np.argsort(-reslab.mesh.dof_xys[:,1])
# ax.plot(reslab.total_cumulative_H2O_losses/1000.0, reslab.mesh.dof_xys[indicesre,1], label='rehydration', ls='--')
# ax.set_xlabel('Cumulative water loss')
# ax.set_ylabel('y (km)')
# _ = ax.legend()

# %%
