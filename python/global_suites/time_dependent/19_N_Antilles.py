#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir))
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, os.path.pardir, 'python'))


import utils
from sz_problems.sz_base import allsz_params
from sz_problems.sz_slab import create_slab, plot_slab
from sz_problems.sz_geometry import create_sz_geometry
from sz_problems.sz_problem import SubductionProblem
import numpy as np
import dolfinx as df
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)
import hashlib
import zipfile
import requests


name = "19_N_Antilles"
resscale = 3.0
cfl      = 3.0


szdict = allsz_params[name]
print("{}:".format(name))
print("{:<20} {:<10}".format('Key','Value'))
print("-"*85)
for k, v in allsz_params[name].items():
    if v is not None and k not in ['z0', 'z15']: print("{:<20} {}".format(k, v))


if __name__ == "__main__" and "__file__" not in globals():
    get_ipython().run_line_magic('pinfo', 'SubductionProblem')


slab = create_slab(szdict['xs'], szdict['ys'], resscale, szdict['lc_depth'])
_ = plot_slab(slab)


geom = create_sz_geometry(slab, resscale, szdict['sztype'], szdict['io_depth'], szdict['extra_width'], 
                             szdict['coast_distance'], szdict['lc_depth'], szdict['uc_depth'])
_ = geom.plot()


sz = SubductionProblem(geom, **szdict)


fps = 5
plotter = pv.Plotter(notebook=False, off_screen=True)
utils.plot.plot_scalar(sz.T_i, plotter=plotter, scale=sz.T0, gather=True, cmap='coolwarm', clim=[0.0, sz.Tm*sz.T0], scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
utils.plot.plot_geometry(geom, plotter=plotter, color='green', width=2)
utils.plot.plot_couplingdepth(slab, plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
plotter.open_gif( str(output_folder / "{}_td_solution_resscale_{:.2f}_cfl_{:.2f}.gif".format(name, resscale, cfl,)), fps=fps)


# Select the timestep based on the approximate target Courant number
dt = cfl*resscale/szdict['Vs']
# Reduce the timestep to get an integer number of timesteps
dt = szdict['As']/np.ceil(szdict['As']/dt)
sz.solve_timedependent_dislocationcreep(szdict['As'], dt, theta=0.5, rtol=1.e-1, verbosity=1, plotter=plotter)
plotter.close()


plotter = pv.Plotter()
utils.plot.plot_scalar(sz.T_i, plotter=plotter, scale=sz.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
utils.plot.plot_vector_glyphs(sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz.v0))
utils.plot.plot_vector_glyphs(sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz.v0))
utils.plot.plot_geometry(geom, plotter=plotter, color='green', width=2)
utils.plot.plot_couplingdepth(slab, plotter=plotter, render_points_as_spheres=True, point_size=10.0, color='green')
utils.plot.plot_show(plotter)
utils.plot.plot_save(plotter, output_folder / "{}_td_solution_resscale_{:.2f}_cfl_{:.2f}.png".format(name, resscale, cfl,))


filename = output_folder / "{}_td_solution_resscale_{:.2f}_cfl_{:.2f}.bp".format(name, resscale, cfl,)
with df.io.VTXWriter(sz.mesh.comm, filename, [sz.T_i, sz.vs_i, sz.vw_i]) as vtx:
    vtx.write(0.0)
# zip the .bp folder so that it can be downloaded from Jupyter lab
if __name__ == "__main__" and "__file__" not in globals():
    zipfilename = filename.with_suffix(".zip")
    get_ipython().system('zip -r $zipfilename $filename')


zipfilename = pathlib.Path(os.path.join(basedir, os.path.pardir, os.path.pardir, os.path.pardir, "data", "vankeken_wilson_peps_2023_TF_lowres_minimal.zip"))
if not zipfilename.is_file():
    zipfileurl = 'https://zenodo.org/records/13234021/files/vankeken_wilson_peps_2023_TF_lowres_minimal.zip'
    r = requests.get(zipfileurl, allow_redirects=True)
    open(zipfilename, 'wb').write(r.content)
assert hashlib.md5(open(zipfilename, 'rb').read()).hexdigest() == 'a8eca6220f9bee091e41a680d502fe0d'


tffilename = os.path.join('vankeken_wilson_peps_2023_TF_lowres_minimal', 'sz_suite_td', szdict['dirname']+'_minres_2.00_cfl_2.00.vtu')
tffilepath = os.path.join(basedir, os.path.pardir, os.path.pardir, os.path.pardir, 'data')
with zipfile.ZipFile(zipfilename, 'r') as z:
    z.extract(tffilename, path=tffilepath)


fxgrid = utils.plot.grids_scalar(sz.T_i)[0]

tfgrid = pv.get_reader(os.path.join(tffilepath, tffilename)).read()

diffgrid = utils.plot.pv_diff(fxgrid, tfgrid, field_name_map={'T':'Temperature::PotentialTemperature'}, pass_point_data=True)


diffgrid.set_active_scalars('T')
plotter_diff = pv.Plotter()
clim = None
plotter_diff.add_mesh(diffgrid, cmap='coolwarm', clim=clim, scalar_bar_args={'title': 'Temperature Difference (deg C)', 'bold':True})
utils.plot.plot_geometry(geom, plotter=plotter_diff, color='green', width=2)
utils.plot.plot_couplingdepth(slab, plotter=plotter_diff, render_points_as_spheres=True, point_size=5.0, color='green')
plotter_diff.enable_parallel_projection()
plotter_diff.view_xy()
plotter_diff.show()


integrated_data = diffgrid.integrate_data()
error = integrated_data['T'][0]/integrated_data['Area'][0]
print("Average error = {}".format(error,))
assert np.abs(error) < 5


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../../python/global_suites/time_dependent --NbConvertApp.output_base=19_N_Antilles 19_N_Antilles.ipynb')




