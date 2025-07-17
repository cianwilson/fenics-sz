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


class TDDislSubductionProblem(TDSubductionProblem):
    def calculate_residual(self, rw, rs, rT):
        """
        Given forms for the vpw, vps and T residuals, 
        return the total residual of the problem.

        Arguments:
          * rw - residual form for the wedge velocity and pressure
          * rs - residual form for the slab velocity and pressure
          * rT - residual form for the temperature
        
        Returns:
          * r  - 2-norm of the combined residual
        """
        # because some of our forms are defined on different MPI comms
        # we need to calculate a squared 2-norm locally and use the global
        # comm to reduce it
        def calc_r_norm_sq(r, bcs, this_rank=True):
            r_norm_sq = 0.0
            if this_rank:
                r_vec = df.fem.petsc.assemble_vector_nest(r)
                # update the ghost values
                for r_vec_sub in r_vec.getNestSubVecs():
                    r_vec_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                # set bcs
                bcs_by_block = df.fem.bcs_by_block(df.fem.extract_function_spaces(r), bcs)
                df.fem.petsc.set_bc_nest(r_vec, bcs_by_block, alpha=0.0)
                r_arr = r_vec.getArray()
                r_norm_sq = np.inner(r_arr, r_arr)
            return r_norm_sq
        with df.common.Timer("Assemble Stokes"):
            r_norm_sq  = calc_r_norm_sq(rw, self.bcs_vw, self.wedge_rank)
            r_norm_sq += calc_r_norm_sq(rs, self.bcs_vs, self.slab_rank)
        self.comm.barrier()
        with df.common.Timer("Assemble Temperature"):
            r_norm_sq += calc_r_norm_sq(rT, self.bcs_T)
        r = self.comm.allreduce(r_norm_sq, op=MPI.SUM)**0.5
        return r

    def solve(self, tf, dt, theta=0.5, rtol=5.e-6, atol=5.e-9, maxits=50, verbosity=2,
              petsc_options_s=None, petsc_options_T=None, 
              plotter=None):
        """
        Solve the coupled temperature-velocity-pressure problem assuming a dislocation creep rheology with time dependency

        Arguments:
          * tf - final time  (in Myr)
          * dt - the timestep (in Myr)
          
        Keyword Arguments:
          * theta         - theta parameter for timestepping (0 <= theta <= 1, defaults to theta=0.5)
          * rtol          - nonlinear iteration relative tolerance
          * atol          - nonlinear iteration absolute tolerance
          * maxits        - maximum number of nonlinear iterations
          * verbosity     - level of verbosity (<1=silent, >0=basic, >1=timestep, >2=nonlinear convergence, defaults to 2)
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
        
        # first solve the isoviscous problem
        self.solve_stokes_isoviscous(petsc_options=petsc_options_s)

        # retrieve the temperature forms (implemented in the parent class)
        ST, fT, rT = self.temperature_forms()
        solver_T = TemperatureSolver(ST, fT, self.bcs_T, self.T_i, 
                                     petsc_options=petsc_options_T)

        # retrive the non-linear Stokes forms for the wedge
        Ssw, fsw, rsw, Msw = self.stokes_forms(self.wedge_vw_t, self.wedge_pw_t, 
                                                self.wedge_vw_a, self.wedge_pw_a, 
                                                self.wedge_vw_i, self.wedge_pw_i, 
                                                eta=self.etadisl(self.wedge_vw_i, self.wedge_T_i))        
        # set up a solver for the wedge velocity and pressure
        solver_s_w = StokesSolverNest(Ssw, fsw, self.bcs_vw, 
                                      self.wedge_vw_i, self.wedge_pw_i, 
                                      M=Msw, isoviscous=False,  
                                      petsc_options=petsc_options_s)

        # retrive the non-linear Stokes forms for the slab
        Sss, fss, rss, Mss = self.stokes_forms(self.slab_vs_t, self.slab_ps_t, 
                                                self.slab_vs_a, self.slab_ps_a, 
                                                self.slab_vs_i, self.slab_ps_i, 
                                                eta=self.etadisl(self.slab_vs_i, self.slab_T_i))
        # set up a solver for the slab velocity and pressure
        solver_s_s = StokesSolverNest(Sss, fss, self.bcs_vs,
                                      self.slab_vs_i, self.slab_ps_i,
                                      M=Mss, isoviscous=False,
                                      petsc_options=petsc_options_s)
        
        t = 0
        ti = 0
        tf_nd = tf/self.t0_Myr
        # time loop
        if self.comm.rank == 0 and verbosity>0:
            print("Entering timeloop with {:d} steps (dt = {:g} Myr, final time = {:g} Myr)".format(int(np.ceil(tf_nd/self.dt.value)), dt, tf,))
        # enter the time-loop
        while t < tf_nd - 1e-9:
            if self.comm.rank == 0 and verbosity>1:
                print("Step: {:>6d}, Times: {:>9g} -> {:>9g} Myr".format(ti, t*self.t0_Myr, (t+self.dt.value)*self.t0_Myr,))
            if plotter is not None:
                for mesh in plotter.meshes:
                    if self.T_i.name in mesh.point_data:
                        mesh.point_data[self.T_i.name][:] = self.T_i.x.array
                plotter.write_frame()
            # set the old solution to the new solution
            self.T_n.x.array[:] = self.T_i.x.array
            # calculate the initial residual
            r = self.calculate_residual(rsw, rss, rT)
            r0 = r
            rrel = r/r0  # 1
            if self.comm.rank == 0 and verbosity>2:
                    print("    {:<11} {:<12} {:<17}".format('Iteration','Residual','Relative Residual'))
                    print("-"*42)

            it = 0
            # enter the Picard Iteration
            if self.comm.rank == 0 and verbosity>2: print("    {:<11} {:<12.6g} {:<12.6g}".format(it, r, rrel,))
            while r > atol and rrel > rtol:
                if it > maxits: break
                # solve for temperature and interpolate it
                self.T_i = solver_T.solve()
                self.update_T_functions()

                # solve for v & p and interpolate the velocity
                if self.wedge_rank: self.wedge_vw_i, self.wedge_pw_i = solver_s_w.solve()
                if self.slab_rank:  self.slab_vs_i,  self.slab_ps_i  = solver_s_s.solve()
                self.update_v_functions()

                # wait for all ranks to catch up 
                # (some may not have done anything above and 
                # letting them carry on messes with profiling)
                self.comm.barrier()
                
                # calculate a new residual
                r = self.calculate_residual(rsw, rss, rT)
                rrel = r/r0
                # increment iterations
                it+=1
                if self.comm.rank == 0 and verbosity>2: print("    {:<11} {:<12.6g} {:<12.6g}".format(it, r, rrel,))
            # check for convergence failures
            if it > maxits:
                raise Exception("Nonlinear iteration failed to converge after {} iterations (maxits = {}), r = {} (atol = {}), rrel = {} (rtol = {}).".format(it, \
                                                                                                                                                          maxits, \
                                                                                                                                                          r, \
                                                                                                                                                          rtol, \
                                                                                                                                                          rrel, \
                                                                                                                                                          rtol,))
            # increment the timestep number
            ti+=1
            # increate time
            t+=self.dt.value
        if self.comm.rank == 0 and verbosity>0:
            print("Finished timeloop after {:d} steps (final time = {:g} Myr)".format(ti, t*self.t0_Myr,))

        # only update the pressure at the end as it is not necessary earlier
        self.update_p_functions()




