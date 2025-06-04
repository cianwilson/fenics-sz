#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


from sz_problems.sz_params import default_params, allsz_params
from sz_problems.sz_slab import create_slab
from sz_problems.sz_geometry import create_sz_geometry
from sz_problems.sz_problem import StokesSolverNest, TemperatureSolver
from sz_problems.sz_steady_problem import SteadySubductionProblem


import geometry as geo
import utils
from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
from petsc4py import PETSc
import numpy as np
import scipy as sp
import ufl
import basix.ufl as bu
import matplotlib.pyplot as pl
import copy
import pyvista as pv
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


class SteadyIsoSubductionProblem(SteadySubductionProblem):
    def solve(self, petsc_options_s=None, petsc_options_T=None):
        """
        Solve the coupled temperature-velocity-pressure problem assuming an isoviscous rheology

        Keyword Arguments:
          * petsc_options_s - a dictionary of petsc options to pass to the Stokes solver 
                              (defaults to an LU direct solver using the MUMPS library) 
          * petsc_options_T - a dictionary of petsc options to pass to the temperature solver 
                              (defaults to an LU direct solver using the MUMPS library) 
        """

        # first solve both Stokes systems
        self.solve_stokes_isoviscous(petsc_options=petsc_options_s)

        # retrieve the temperature forms
        ST, fT, _ = self.temperature_forms()
        solver_T = TemperatureSolver(ST, fT, self.bcs_T, self.T_i, 
                                     petsc_options=petsc_options_T)
        # and solve the temperature problem
        self.T_i = solver_T.solve()

        # only update the pressure at the end as it is not necessary earlier
        self.update_p_functions()


def plot_slab_temperatures(sz):
    """
    Plot the slab surface and Moho (7 km slab depth)

    Arguments:
      * sz - a solved SubductionProblem instance
    """
    # get some points along the slab
    slabpoints = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in sz.geom.slab_spline.interpcurves])
    # do the same along a spline deeper in the slab
    slabmoho = copy.deepcopy(sz.geom.slab_spline)
    slabmoho.translatenormalandcrop(-7.0)
    slabmohopoints = np.array([[curve.points[0].x, curve.points[0].y, 0.0] for curve in slabmoho.interpcurves])
    # set up a figure
    fig = pl.figure()
    ax = fig.gca()
    # plot the slab temperatures
    cinds, cells = utils.mesh.get_cell_collisions(slabpoints, sz.mesh)
    ax.plot(sz.T_i.eval(slabpoints, cells)[:,0], -slabpoints[:,1], label='slab surface')
    # plot the moho temperatures
    mcinds, mcells = utils.mesh.get_cell_collisions(slabmohopoints, sz.mesh)
    ax.plot(sz.T_i.eval(slabmohopoints, mcells)[:,0], -slabmohopoints[:,1], label='slab moho')
    # labels, title etc.
    ax.set_xlabel('T ($^\circ$C)')
    ax.set_ylabel('z (km)')
    ax.set_title('Slab surface and Moho temperatures')
    ax.legend()
    ax.invert_yaxis()
    return fig




