#!/usr/bin/env python
# coding: utf-8

from mpi4py import MPI
if MPI.COMM_WORLD.size > 1:
    raise Warning("This script shouldn't be run in parallel!")


import ipyparallel as ipp





labels = [
              'Poisson Mesh', 'Poisson Functions',
              'Poisson Dirichlet BCs', 'Poisson Neumann BCs',
              'Poisson Forms', 'Poisson Assemble',
              'Poisson Solve'
         ]


def profile_poisson(ne, p, labels, petsc_options=None):
     # import necessary modules
     import sys, os
     basedir = ''
     if "__file__" in globals(): basedir = os.path.dirname(__file__)
     sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))
     from background.poisson_2d import solve_poisson_2d
     import dolfinx as df
     from mpi4py import MPI
    
     # solve the Poisson problem
     T_i = solve_poisson_2d(ne, p, petsc_options=petsc_options)

     # extract and return the computation times from dolfinx
     times = [df.common.timing(l)[1] for l in labels]
     maxtimes = T_i.function_space.mesh.comm.reduce(times, op=MPI.MAX)
     return maxtimes


nproc = 2
ne = 64
p = 2

cluster = ipp.Cluster(engine_launcher_class="mpi", n=nproc)
rc = cluster.start_and_connect_sync()
view = rc[:]

maxtimes = [view.remote(block=True)(profile_poisson)(ne, p, labels)[0]]

print('=========================', flush=True)
print('\t'.join(['\t']+[repr(nproc) for nproc in [nproc]]))
for l, label in enumerate(labels):
    print('\t'.join([label]+[repr(t[l]) for t in maxtimes]))
print('=========================')

rc.shutdown(hub=True)








def scale_poisson_parallel(nprocs, ne, p, petsc_options=None):
    maxtimes = []
    for nproc in nprocs:
        cluster = ipp.Cluster(engine_launcher_class="mpi", n=nproc)
        rc = cluster.start_and_connect_sync()
        view = rc[:]

        maxtimes.append(view.remote(block=True)(profile_poisson)(ne, p, 
                                                labels, petsc_options=petsc_options)[0])
        
        rc.shutdown(hub=True)

    print('=========================', flush=True)
    print('\t'.join(['\t']+[repr(nproc) for nproc in nprocs]))
    for l, label in enumerate(labels):
        print('\t'.join([label]+[repr(t[l]) for t in maxtimes]))
    print('=========================')

    return maxtimes



ne = 320
p = 2
nprocs = [1, 2, 4, 8]

scale_poisson_parallel(nprocs, ne, p)


ne = 320
p = 2
nprocs = [1, 2, 4, 8]
petsc_options = {'ksp_type':'cg', 'pc_type':'sor'}

scale_poisson_parallel(nprocs, ne, p, petsc_options=petsc_options)


ne = 320
p = 2
nprocs = [1, 2, 4, 8]
petsc_options = {'ksp_type':'cg', 'pc_type':'gamg'}

scale_poisson_parallel(nprocs, ne, p, petsc_options=petsc_options)


ne = 320
p = 2
nprocs = [1, 2, 4, 8]
petsc_options = {'ksp_type':'preonly', 'pc_type':'lu', 'pc_factor_mat_solver_type':'superlu_dist'}

scale_poisson_parallel(nprocs, ne, p, petsc_options=petsc_options)


ne = 320
p = 2
nprocs = [1, 2, 4, 8]
petsc_options = {'ksp_type':'preonly', 'pc_type':'lu', 'pc_factor_mat_solver_type':'superlu_dist', 'mat_superlu_dist_iterfine':True}

scale_poisson_parallel(nprocs, ne, p, petsc_options=petsc_options)





if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/background --NbConvertApp.output_base=poisson_2d_parallel 2.3d_poisson_2d_parallel.ipynb')

