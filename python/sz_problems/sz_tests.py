#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


import utils
from sz_problems.sz_base import default_params, allsz_params
from sz_problems.sz_slab import create_slab
from sz_problems.sz_geometry import create_sz_geometry
from sz_problems.sz_problem import SubductionProblem, plot_slab_temperatures
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


xs = [0.0, 140.0, 240.0, 400.0]
ys = [0.0, -70.0, -120.0, -200.0]
lc_depth = 40
uc_depth = 15
coast_distance = 0
extra_width = 0
sztype = 'continental'
io_depth_1 = 139
A      = 100.0      # age of subducting slab (Myr)
qs     = 0.065      # surface heat flux (W/m^2)
Vs     = 100.0      # slab speed (mm/yr)


resscale2 = 2.0
slab_resscale2 = create_slab(xs, ys, resscale2, lc_depth)
geom_resscale2 = create_sz_geometry(slab_resscale2, resscale2, sztype, io_depth_1, extra_width, 
                           coast_distance, lc_depth, uc_depth)
sz_case1_resscale2 = SubductionProblem(geom_resscale2, A=A, Vs=Vs, sztype=sztype, qs=qs)


if sz_case1_resscale2.comm.rank == 0: print("\nSolving steady state flow with isoviscous rheology...")
sz_case1_resscale2.solve_steadystate_isoviscous()


diag_resscale2 = sz_case1_resscale2.get_diagnostics()

print('')
print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale2, *diag_resscale2))


plotter_case1_resscale2 = utils.plot.plot_scalar(sz_case1_resscale2.T_i, scale=sz_case1_resscale2.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
utils.plot.plot_vector_glyphs(sz_case1_resscale2.vw_i, plotter=plotter_case1_resscale2, gather=True, factor=0.05, color='k', scale=utils.mps_to_mmpyr(sz_case1_resscale2.v0))
utils.plot.plot_vector_glyphs(sz_case1_resscale2.vs_i, plotter=plotter_case1_resscale2, gather=True, factor=0.05, color='k', scale=utils.mps_to_mmpyr(sz_case1_resscale2.v0))
utils.plot.plot_geometry(sz_case1_resscale2.geom, plotter=plotter_case1_resscale2, color='green', width=2)
utils.plot.plot_couplingdepth(sz_case1_resscale2.geom.slab_spline, plotter=plotter_case1_resscale2, render_points_as_spheres=True, point_size=10.0, color='green')
utils.plot.plot_show(plotter_case1_resscale2)
utils.plot.plot_save(plotter_case1_resscale2, output_folder / "sz_tests_case1_resscale2_solution.png")


io_depth_2 = 154.0
geom_case2_resscale2 = create_sz_geometry(slab_resscale2, resscale2, sztype, io_depth_2, extra_width, 
                                          coast_distance, lc_depth, uc_depth)
sz_case2_resscale2 = SubductionProblem(geom_case2_resscale2, A=A, Vs=Vs, sztype=sztype, qs=qs)


if sz_case2_resscale2.comm.rank == 0: print("\nSolving steady state flow with dislocation creep rheology...")
sz_case2_resscale2.solve_steadystate_dislocationcreep()


diag_case2_resscale2 = sz_case2_resscale2.get_diagnostics()

print('')
print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale2, *diag_case2_resscale2))


plotter_case2_resscale2 = utils.plot.plot_scalar(sz_case2_resscale2.T_i, scale=sz_case2_resscale2.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
utils.plot.plot_vector_glyphs(sz_case2_resscale2.vw_i, plotter=plotter_case2_resscale2, gather=True, factor=0.05, color='k', scale=utils.mps_to_mmpyr(sz_case2_resscale2.v0))
utils.plot.plot_vector_glyphs(sz_case2_resscale2.vs_i, plotter=plotter_case2_resscale2, gather=True, factor=0.05, color='k', scale=utils.mps_to_mmpyr(sz_case2_resscale2.v0))
utils.plot.plot_geometry(sz_case2_resscale2.geom, plotter=plotter_case2_resscale2, color='green', width=2)
utils.plot.plot_couplingdepth(sz_case2_resscale2.geom.slab_spline, plotter=plotter_case2_resscale2, render_points_as_spheres=True, point_size=10.0, color='green')
utils.plot.plot_show(plotter_case2_resscale2)
utils.plot.plot_save(plotter_case2_resscale2, output_folder / "sz_tests_case2_resscale2_solution.png")


# set a list of coupling depths to try
coupling_depths = [60.0, 80.0, 100.0]

# set up a list to save the diagnostics from each
diagnostics = []
# loop over the couplings depths
for coupling_depth in coupling_depths:
    # create the slab object, all of the input arguments are the same as in case 2
    # but this time we also pass in the coupling_depth keyword argument to override
    # the default value (80 km)
    slab_dc = create_slab(xs, ys, resscale2, lc_depth, coupling_depth=coupling_depth)
    # set up the geometry
    geom_dc = create_sz_geometry(slab_dc, resscale2, sztype, io_depth_2, extra_width, 
                                            coast_distance, lc_depth, uc_depth)
    # set up the subduction zone problem
    sz_dc = SubductionProblem(geom_dc, A=A, Vs=Vs, sztype=sztype, qs=qs)

    # solve the steady state problem
    if sz_dc.comm.rank == 0: print(f"\nSolving steady state flow with coupling depth = {coupling_depth}km...")
    sz_dc.solve_steadystate_dislocationcreep()

    # retrieve the diagnostics
    diagnostics.append(sz_dc.get_diagnostics())

    # plot the solution
    plotter_dc = utils.plot.plot_scalar(sz_dc.T_i, scale=sz_dc.T0, gather=True, cmap='coolwarm', 
                                   scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    utils.plot.plot_vector_glyphs(sz_dc.vw_i, plotter=plotter_dc, gather=True, factor=0.05, color='k', 
                             scale=utils.mps_to_mmpyr(sz_dc.v0))
    utils.plot.plot_vector_glyphs(sz_dc.vs_i, plotter=plotter_dc, gather=True, factor=0.05, color='k', 
                             scale=utils.mps_to_mmpyr(sz_dc.v0))
    utils.plot.plot_geometry(sz_dc.geom, plotter=plotter_dc, color='green', width=2)
    utils.plot.plot_couplingdepth(sz_dc.geom.slab_spline, plotter=plotter_dc, render_points_as_spheres=True, 
                             point_size=10.0, color='green')
    utils.plot.plot_show(plotter_dc)
    utils.plot.plot_save(plotter_dc, output_folder / f"sz_tests_dc{coupling_depth}_solution.png")


# print the varying coupling depth output
print('')
print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('d_c', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
for dc, diag in zip(coupling_depths, diagnostics):
    print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(dc, *diag))


resscale_ak = 5.0
szdict_ak = allsz_params['01_Alaska_Peninsula']
slab_ak = create_slab(szdict_ak['xs'], szdict_ak['ys'], resscale_ak, szdict_ak['lc_depth'])
geom_ak = create_sz_geometry(slab_ak, resscale_ak, szdict_ak['sztype'], szdict_ak['io_depth'], szdict_ak['extra_width'], 
                             szdict_ak['coast_distance'], szdict_ak['lc_depth'], szdict_ak['uc_depth'])
sz_ak = SubductionProblem(geom_ak, **szdict_ak)


if sz_ak.comm.rank==0: print("\nSolving steady state flow with isoviscous rheology...")
sz_ak.solve_steadystate_dislocationcreep()


plotter_ak = utils.plot.plot_scalar(sz_ak.T_i, scale=sz_ak.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
utils.plot.plot_vector_glyphs(sz_ak.vw_i, plotter=plotter_ak, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz_ak.v0))
utils.plot.plot_vector_glyphs(sz_ak.vs_i, plotter=plotter_ak, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz_ak.v0))
utils.plot.plot_geometry(sz_ak.geom, plotter=plotter_ak, color='green', width=2)
utils.plot.plot_couplingdepth(sz_ak.geom.slab_spline, plotter=plotter_ak, render_points_as_spheres=True, point_size=10.0, color='green')
utils.plot.plot_show(plotter_ak)
utils.plot.plot_save(plotter_ak, output_folder / "sz_tests_ak_solution.png")


eta_ak = sz_ak.project_dislocationcreep_viscosity()
plotter_eta_ak = utils.plot.plot_scalar(eta_ak, scale=sz_ak.eta0, gather=True, log_scale=True, show_edges=True, scalar_bar_args={'title': 'Viscosity (Pa s)', 'bold':True})
utils.plot.plot_geometry(sz_ak.geom, plotter=plotter_eta_ak, color='green', width=2)
utils.plot.plot_couplingdepth(sz_ak.geom.slab_spline, plotter=plotter_eta_ak, render_points_as_spheres=True, point_size=10.0, color='green')
utils.plot.plot_show(plotter_eta_ak)
utils.plot.plot_save(plotter_eta_ak, output_folder / "sz_tests_ak_eta.png")


resscale_ant = 5.0
szdict_ant = allsz_params['19_N_Antilles']
slab_ant = create_slab(szdict_ant['xs'], szdict_ant['ys'], resscale_ant, szdict_ant['lc_depth'])
geom_ant = create_sz_geometry(slab_ant, resscale_ant, szdict_ant['sztype'], szdict_ant['io_depth'], szdict_ant['extra_width'], 
                              szdict_ant['coast_distance'], szdict_ant['lc_depth'], szdict_ant['uc_depth'])
sz_ant = SubductionProblem(geom_ant, **szdict_ant)


if sz_ant.comm.rank==0: print("\nSolving steady state flow with isoviscous rheology...")
sz_ant.solve_steadystate_dislocationcreep()


plotter_ant = utils.plot.plot_scalar(sz_ant.T_i, scale=sz_ant.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
utils.plot.plot_vector_glyphs(sz_ant.vw_i, plotter=plotter_ant, gather=True, factor=0.25, color='k', scale=utils.mps_to_mmpyr(sz_ant.v0))
utils.plot.plot_vector_glyphs(sz_ant.vs_i, plotter=plotter_ant, gather=True, factor=0.25, color='k', scale=utils.mps_to_mmpyr(sz_ant.v0))
utils.plot.plot_geometry(sz_ant.geom, plotter=plotter_ant, color='green', width=2)
utils.plot.plot_couplingdepth(sz_ant.geom.slab_spline, plotter=plotter_ant, render_points_as_spheres=True, point_size=10.0, color='green')
utils.plot.plot_show(plotter_ant)
utils.plot.plot_save(plotter_ant, output_folder / "sz_tests_ant_solution.png")


eta_ant = sz_ant.project_dislocationcreep_viscosity()
plotter_eta_ant = utils.plot.plot_scalar(eta_ant, scale=sz_ant.eta0, gather=True, log_scale=True, show_edges=True, scalar_bar_args={'title': 'Viscosity (Pa s)', 'bold':True})
utils.plot.plot_geometry(sz_ant.geom, plotter=plotter_eta_ant, color='green', width=2)
utils.plot.plot_couplingdepth(sz_ant.geom.slab_spline, plotter=plotter_eta_ant, render_points_as_spheres=True, point_size=10.0, color='green')
utils.plot.plot_show(plotter_eta_ant)
utils.plot.plot_save(plotter_eta_ant, output_folder / "sz_tests_ant_eta.png")


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/sz_problems --NbConvertApp.output_base=sz_tests 3.2e_sz_tests.ipynb')

