#!/usr/bin/env python
# coding: utf-8

import ipyparallel as ipp
import sys, os


path = os.path.join(os.path.pardir, os.path.pardir, 'python')
sys.path.append(path)
import utils
from background.batchelor import test_convergence_batchelor


labels = [
          'Mesh', 'Functions',
          #'Dirichlet BCs', 'Forms',
          'Assemble', 'Solve',
         ]
number = 2
nprocs = [1, 2, 3, 4]


ne = 256
p = 1
petsc_options=None

_ = utils.profile_parallel(nprocs, labels, path, 'background.batchelor', 'solve_batchelor', 
                           ne, p, number=number, petsc_options=petsc_options)


# List of polynomial orders to try
ps = [1,2]
# List of resolutions to try
nelements = [10, 20, 40, 80]

errors_l2_all = utils.run_parallel([2,], path, 'background.batchelor', 'run_convergence_batchelor', 
                                   ps, nelements, petsc_options=petsc_options, attach_nullspace=False)

for errors in errors_l2_all:
    test_convergence_batchelor(ps, nelements, errors)


ne = 256
p = 1
petsc_options = {'ksp_type':'minres', 
                 'pc_type':'fieldsplit', 
                 'pc_fieldsplit_type': 'additive',
                 'ksp_view':None,
                 'fieldsplit_v_ksp_type':'preonly',
                 'fieldsplit_v_pc_type':'gamg',
                 'fieldsplit_p_ksp_type':'preonly',
                 'fieldsplit_p_pc_type':'jacobi'}

_ = utils.profile_parallel(nprocs, labels, path, 'background.batchelor', 'solve_batchelor', 
                           ne, p, number=number, petsc_options=petsc_options)


# List of polynomial orders to try
ps = [1,2]
# List of resolutions to try
nelements = [10, 20, 40, 80]

errors_l2_all = utils.run_parallel([2,], path, 'background.batchelor', 'run_convergence_batchelor', 
                                   ps, nelements, petsc_options=petsc_options)

for errors in errors_l2_all:
    test_convergence_batchelor(ps, nelements, errors)


ne = 256
p = 1
petsc_options = {'ksp_type':'minres', 
                 'pc_type':'fieldsplit', 
                 'pc_fieldsplit_type': 'additive',
                 'ksp_view':None,
                 'fieldsplit_v_ksp_type':'preonly',
                 'fieldsplit_v_pc_type':'gamg',
                 'fieldsplit_p_ksp_type':'preonly',
                 'fieldsplit_p_pc_type':'jacobi'}

_ = utils.profile_parallel(nprocs, labels, path, 'background.batchelor', 'solve_batchelor', 
                           ne, p, number=number, petsc_options=petsc_options, attach_nullspace=True)


# List of polynomial orders to try
ps = [1,2]
# List of resolutions to try
nelements = [10, 20, 40, 80]

errors_l2_all = utils.run_parallel([2,], path, 'background.batchelor', 'run_convergence_batchelor', 
                                   ps, nelements, petsc_options=petsc_options, attach_nullspace=True)

for errors in errors_l2_all:
    test_convergence_batchelor(ps, nelements, errors)













