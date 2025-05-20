#!/usr/bin/env python
# coding: utf-8

from mpi4py import MPI
if MPI.COMM_WORLD.size > 1:
    raise Warning("This script shouldn't be run in parallel!")


import ipyparallel as ipp
import numpy as np


labels = [
          'Blankenbach Mesh', 'Blankenbach Functions',
          'Blankenbach Dirichlet BCs', 'Blankenbach Forms',
          'Blankenbach Problem Setup Stokes', 'Blankenbach Problem Setup Temperature',
          'Blankenbach Residual', 'Blankenbach Solve Stokes', 'Blankenbach Solve Temperature',
         ]


def profile_blankenbach(Ra, ne, pp, pT, labels, b=None, petsc_options_s=None, petsc_options_T=None):
     # import necessary modules
     import sys, os
     basedir = ''
     if "__file__" in globals(): basedir = os.path.dirname(__file__)
     sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))
     from background.blankenbach import solve_blankenbach
     import dolfinx as df
     from mpi4py import MPI
    
     # solve the Blankenbach problem
     v_i, p_i, T_i = solve_blankenbach(Ra, ne, pp=pp, pT=pT, b=b, 
                                       petsc_options_s=petsc_options_s, 
                                       petsc_options_T=petsc_options_T)

     # extract and return the computation times from dolfinx
     times = [df.common.timing(l)[1] for l in labels]
     maxtimes = T_i.function_space.mesh.comm.reduce(times, op=MPI.MAX)
     return maxtimes


ne = 128
pp = 1
pT = 1
Ra = 1.e4
b = np.log(1.e3)
petsc_options_s = {'ksp_type' : 'preonly', 'pc_type' : 'lu', 'pc_factor_mat_solver_type' : 'superlu_dist', 'mat_superlu_dist_iterrefine' : True, 'ksp_view':None}

profile_blankenbach(Ra, ne, pp, pT, labels, b=b, petsc_options_s=petsc_options_s)


nproc = 2
ne = 160
pp = 1
pT = 1
Ra = 1.e4
petsc_options_s = {'ksp_type' : 'preonly', 'pc_type' : 'lu', 'pc_factor_mat_solver_type' : 'superlu_dist', 'mat_superlu_dist_iterrefine' : True}

cluster = ipp.Cluster(engine_launcher_class="mpi", n=nproc)
rc = cluster.start_and_connect_sync()
view = rc[:]

maxtimes = [view.remote(block=True)(profile_blankenbach)(Ra, ne, pp, pT, labels, petsc_options_s=petsc_options_s)[0]]

print('=========================', flush=True)
print('\t'.join(['\t']+[repr(nproc) for nproc in [nproc]]))
for l, label in enumerate(labels):
    print('\t'.join([label]+[repr(t[l]) for t in maxtimes]))
print('=========================')

rc.shutdown(hub=True)








def scale_blankenbach_parallel(nprocs, Ra, ne, pp, pT, labels, b=None, petsc_options_s=None, petsc_options_T=None):
    maxtimes = []
    for nproc in nprocs:
        cluster = ipp.Cluster(engine_launcher_class="mpi", n=nproc)
        rc = cluster.start_and_connect_sync()
        view = rc[:]

        maxtimes.append(view.remote(block=True)(profile_blankenbach)(Ra, ne, pp, pT, labels, b=b, petsc_options_s=petsc_options_s, petsc_options_T=petsc_options_T)[0])

        rc.shutdown(hub=True)

    print('=========================', flush=True)
    print('\t'.join(['\t']+[repr(nproc) for nproc in nprocs]))
    for l, label in enumerate(labels):
        print('\t'.join([label]+[repr(t[l]) for t in maxtimes]))
    print('=========================')

    return maxtimes


ne = 256
pp = 1
pT = 1
Ra = 1.e4
nprocs = [1, 2, 4, 8]

scale_blankenbach_parallel(nprocs, Ra, ne, pp, pT, labels)


ne = 256
pp = 1
pT = 1
Ra = 1.e4
b = np.log(1.e3)
nprocs = [1, 2, 4, 8]

scale_blankenbach_parallel(nprocs, Ra, ne, pp, pT, labels, b=b)


ne = 120
pp = 1
pT = 1
Ra = 1.e4
b = np.log(1.e3)
petsc_options_s = {'ksp_type' : 'preonly', 'pc_type' : 'lu', 'pc_factor_mat_solver_type' : 'superlu_dist', 'mat_superlu_dist_iterrefine' : True}
nprocs = [1, 2, 4, 8]

scale_blankenbach_parallel(nprocs, Ra, ne, pp, pT, labels, b=b, petsc_options_s=petsc_options_s)











if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/background --NbConvertApp.output_base=blankenbach_parallel 2.5c_blankenbach_parallel.ipynb')




