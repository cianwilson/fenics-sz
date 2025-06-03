#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


from sz_problems.sz_params import default_params, allsz_params
from sz_problems.sz_slab import create_slab
from sz_problems.sz_geometry import create_sz_geometry
from sz_problems.sz_problem import TemperatureSolver
from sz_problems.sz_tdep_problem import TDSubductionProblem


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


class TDIsoSubductionProblem(TDSubductionProblem):
    def solve(self, tf, dt, theta=0.5, verbosity=2, 
              petsc_options_s=None, petsc_options_T=None, plotter=None):
        """
        Solve the coupled temperature-velocity-pressure problem assuming an isoviscous rheology with time dependency

        Arguments:
          * tf - final time  (in Myr)
          * dt - the timestep (in Myr)
          
        Keyword Arguments:
          * theta         - theta parameter for timestepping (0 <= theta <= 1, defaults to theta=0.5)
          * petsc_options - a dictionary of petsc options to pass to the solver (defaults to mumps)
          * verbosity     - level of verbosity (<1=silent, >0=basic, >1=timestep, defaults to 2)
          * petsc_options_s - a dictionary of petsc options to pass to the Stokes solver 
                              (defaults to an LU direct solver using the MUMPS library) 
          * petsc_options_T - a dictionary of petsc options to pass to the temperature solver 
                              (defaults to an LU direct solver using the MUMPS library)
        """
        assert(theta >= 0 and theta <= 1)
        
        # set the timestepping options based on the arguments
        # these need to be set before calling self.temperature_forms_timedependent
        self.dt = df.fem.Constant(self.mesh, df.default_scalar_type(dt/self.t0_Myr))
        self.theta = df.fem.Constant(self.mesh, df.default_scalar_type(theta))

        # reset the initial conditions
        self.setup_boundaryconditions()
        
        # first solve both Stokes systems
        self.solve_stokes_isoviscous(petsc_options=petsc_options_s)

        # retrieve the temperature forms
        ST, fT, _ = self.temperature_forms()
        solver_T = TemperatureSolver(ST, fT, self.bcs_T, self.T_i, 
                                     petsc_options=petsc_options_T)

        # and solve the temperature problem repeatedly with time step dt
        t = 0
        ti = 0
        tf_nd = tf/self.t0_Myr
        if self.comm.rank == 0 and verbosity>0:
            print("Entering timeloop with {:d} steps (dt = {:g} Myr, final time = {:g} Myr)".format(int(np.ceil(tf_nd/self.dt.value)), dt, tf,))
        # enter the time-loop
        while t < tf_nd - 1e-9:
            if self.comm.rank == 0 and verbosity>1:
                print("Step: {:>6d}, Times: {:>9g} -> {:>9g} Myr".format(ti, t*self.t0_Myr, (t+self.dt.value)*self.t0_Myr))
            if plotter is not None:
                for mesh in plotter.meshes:
                    if self.T_i.name in mesh.point_data:
                        mesh.point_data[self.T_i.name][:] = self.T_i.x.array
                plotter.write_frame()
            # set the old solution to the new solution
            self.T_n.x.array[:] = self.T_i.x.array
            # solve for the new solution
            self.T_i = solver_T.solve()
            # increment the timestep number
            ti+=1
            # increment the time
            t+=self.dt.value
        if self.comm.rank == 0 and verbosity>0:
            print("Finished timeloop after {:d} steps (final time = {:g} Myr)".format(ti, t*self.t0_Myr,))

        # only update the pressure at the end as it is not necessary earlier
        self.update_p_functions()




