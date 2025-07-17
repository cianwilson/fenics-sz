#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


import utils
from sz_problems.sz_params import default_params, allsz_params
from sz_problems.sz_slab import create_slab
from sz_problems.sz_geometry import create_sz_geometry
from sz_problems.sz_steady_isoviscous import SteadyIsoSubductionProblem
from sz_problems.sz_steady_dislcreep import SteadyDislSubductionProblem
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


def solve_benchmark_case1(resscale, petsc_options_s=None, petsc_options_T=None):
    xs = [0.0, 140.0, 240.0, 400.0]
    ys = [0.0, -70.0, -120.0, -200.0]
    lc_depth = 40
    uc_depth = 15
    coast_distance = 0
    extra_width = 0
    sztype = 'continental'
    io_depth = 139
    A      = 100.0      # age of subducting slab (Myr)
    qs     = 0.065      # surface heat flux (W/m^2)
    Vs     = 100.0      # slab speed (mm/yr)

    # create the slab
    slab = create_slab(xs, ys, resscale, lc_depth)
    # construct the geometry
    geom = create_sz_geometry(slab, resscale, sztype, io_depth, extra_width, 
                               coast_distance, lc_depth, uc_depth)
    # set up a subduction zone problem
    sz = SteadyIsoSubductionProblem(geom, A=A, Vs=Vs, sztype=sztype, qs=qs)

    # solve it using a steady state assumption and an isoviscous rheology
    sz.solve(petsc_options_s=petsc_options_s, petsc_options_T=petsc_options_T)

    # evaluate the diagnostics
    diag = sz.get_diagnostics()

    return diag


values_wvk_case1 = [
    {'resscale': 2, 'T_ndof': 21403, 'T_{200,-100}': 517.17, 'Tbar_s': 451.83, 'Tbar_w': 926.62, 'Vrmsw': 34.64},
    {'resscale': 1, 'T_ndof': 83935, 'T_{200,-100}': 516.95, 'Tbar_s': 451.71, 'Tbar_w': 926.33, 'Vrmsw': 34.64},
    {'resscale': 0.5, 'T_ndof': 332307, 'T_{200,-100}': 516.86, 'Tbar_s': 451.63, 'Tbar_w': 926.15, 'Vrmsw': 34.64},
]


def solve_benchmark_case2(resscale, petsc_options_s=None, petsc_options_T=None):
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
    sz = SteadyDislSubductionProblem(geom, A=A, Vs=Vs, sztype=sztype, qs=qs)

    # solve it using a steady state assumption and a dislocation creep rheology
    sz.solve(petsc_options_s=petsc_options_s, petsc_options_T=petsc_options_T)

    # evaluate the diagnostics
    diag = sz.get_diagnostics()

    return diag


values_wvk_case2 = [
    {'resscale': 2, 'T_ndof': 21403, 'T_{200,-100}': 683.05, 'Tbar_s': 571.58, 'Tbar_w': 936.65, 'Vrmsw': 40.89},
    {'resscale': 1, 'T_ndof': 83935, 'T_{200,-100}': 682.87, 'Tbar_s': 572.23, 'Tbar_w': 936.11, 'Vrmsw': 40.78},
    {'resscale': 0.5, 'T_ndof': 332307, 'T_{200,-100}': 682.80, 'Tbar_s': 572.05, 'Tbar_w': 937.37, 'Vrmsw': 40.77},
]




