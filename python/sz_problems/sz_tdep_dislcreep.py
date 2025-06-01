#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


from sz_problems.sz_params import default_params, allsz_params
from sz_problems.sz_slab import create_slab
from sz_problems.sz_geometry import create_sz_geometry
from sz_problems.sz_problem import SubductionProblem, StokesSolverNest, TemperatureSolver


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


class SubductionProblem(SubductionProblem):
    def solve_timedependent_dislocationcreep(self, tf, dt, theta=0.5, rtol=5.e-6, atol=5.e-9, maxits=50, verbosity=2, 
                                             petsc_options=None, plotter=None):
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
          * petsc_options - a dictionary of petsc options to pass to the solver (defaults to mumps)
        """
        
        if petsc_options is None:
            petsc_options={"ksp_type": "preonly", 
                           "pc_type" : "lu", 
                           "pc_factor_mat_solver_type" : "mumps"}

        assert theta >= 0 and theta <= 1

        # set the timestepping options based on the arguments
        # these need to be set before calling self.temperature_forms_timedependent
        self.dt = df.fem.Constant(self.mesh, df.default_scalar_type(dt/self.t0_Myr))
        self.theta = df.fem.Constant(self.mesh, df.default_scalar_type(theta))
            
        # reset the initial conditions
        self.setup_boundaryconditions()
        
        # first solve the isoviscous problem
        self.solve_stokes_isoviscous(petsc_options=petsc_options)

        # retrieve the temperature forms
        ST, fT = self.temperature_forms_timedependent()
        problem_T = df.fem.petsc.LinearProblem(ST, fT, bcs=self.bcs_T, u=self.T_i,
                                               petsc_options=petsc_options)
        
        # retrive the non-linear Stokes forms for the wedge
        Ssw, fsw = self.stokes_forms(self.wedge_vpw_t, self.wedge_vpw_a, \
                                     self.wedge_submesh, eta=self.etadisl(self.wedge_vw_i, self.wedge_T_i))
        problem_vpw = df.fem.petsc.LinearProblem(Ssw, fsw, bcs=self.bcs_vpw, u=self.wedge_vpw_i, 
                                                 petsc_options=petsc_options)

        # retrive the non-linear Stokes forms for the slab
        Sss, fss = self.stokes_forms(self.slab_vps_t, self.slab_vps_a, \
                                     self.slab_submesh, eta=self.etadisl(self.slab_vs_i, self.slab_T_i))
        problem_vps = df.fem.petsc.LinearProblem(Sss, fss, bcs=self.bcs_vps, u=self.slab_vps_i,
                                                 petsc_options=petsc_options)

        # define the non-linear residual for the wedge velocity-pressure
        rw = ufl.action(Ssw, self.wedge_vpw_i) - fsw
        # define the non-linear residual for the slab velocity-pressure
        rs = ufl.action(Sss, self.slab_vps_i) - fss
        # define the non-linear residual for the temperature
        rT = ufl.action(ST, self.T_i) - fT
        
        t = 0
        ti = 0
        tf_nd = tf/self.t0_Myr
        # time loop
        if self.comm.rank == 0 and verbosity>0:
            print("Entering timeloop with {:d} steps (dt = {:g} Myr, final time = {:g} Myr)".format(int(np.ceil(tf_nd/self.dt.value)), dt, tf,))
        while t < tf_nd - 1e-9:
            if self.comm.rank == 0 and verbosity>1:
                print("Step: {:>6d}, Times: {:>9g} -> {:>9g} Myr".format(ti, t*self.t0_Myr, (t+self.dt.value)*self.t0_Myr,))
            if plotter is not None:
                for mesh in plotter.meshes:
                    if self.T_i.name in mesh.point_data:
                        mesh.point_data[self.T_i.name][:] = self.T_i.x.array
                plotter.write_frame()
            self.T_n.x.array[:] = self.T_i.x.array
            # calculate the initial residual
            r = self.calculate_residual(rw, rs, rT)
            r0 = r
            rrel = r/r0  # 1
            if self.comm.rank == 0 and verbosity>2:
                    print("    {:<11} {:<12} {:<17}".format('Iteration','Residual','Relative Residual'))
                    print("-"*42)

            it = 0
            # Picard Iteration
            if self.comm.rank == 0 and verbosity>2: print("    {:<11} {:<12.6g} {:<12.6g}".format(it, r, rrel,))
            while r > atol and rrel > rtol:
                if it > maxits: break
                self.T_i = problem_T.solve()
                self.update_T_functions()
                
                if self.wedge_rank: self.wedge_vpw_i = problem_vpw.solve()
                if self.slab_rank:  self.slab_vps_i  = problem_vps.solve()
                self.update_v_functions()

                r = self.calculate_residual(rw, rs, rT)
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
            # increment time
            ti+=1
            t+=self.dt.value
        if self.comm.rank == 0 and verbosity>0:
            print("Finished timeloop after {:d} steps (final time = {:g} Myr)".format(ti, t*self.t0_Myr,))

        # only update the pressure at the end as it is not necessary earlier
        self.update_p_functions()


if __name__ == "__main__":
    geom_case2td = create_sz_geometry(slab, resscale, sztype, io_depth_2, extra_width, 
                              coast_distance, lc_depth, uc_depth)
    sz_case2td = SubductionProblem(geom_case2td, A=A, Vs=Vs, sztype=sztype, qs=qs)
    
    fps = 5
    plotter_gif2 = pv.Plotter(notebook=False, off_screen=True)
    utils.plot.plot_scalar(sz_case2td.T_i, plotter=plotter_gif2, scale=sz_case2td.T0, gather=True, cmap='coolwarm', clim=[0.0, sz_case2td.Tm*sz_case2td.T0], scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    utils.plot.plot_geometry(sz_case2td.geom, plotter=plotter_gif2, color='green', width=2)
    utils.plot.plot_couplingdepth(sz_case2td.geom.slab_spline, plotter=plotter_gif2, render_points_as_spheres=True, point_size=10.0, color='green')
    plotter_gif2.open_gif( str(output_folder / "sz_problem_case2td_solution.gif"), fps=fps)
    
    sz_case2td.solve_timedependent_dislocationcreep(10, 0.05, theta=0.5, rtol=1.e-3, plotter=plotter_gif2)

    plotter_gif2.close()


if __name__ == "__main__":
    plotter_distd = utils.plot.plot_scalar(sz_case2td.T_i, scale=sz_case2td.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    utils.plot.plot_vector_glyphs(sz_case2td.vw_i, plotter=plotter_distd, factor=0.1, gather=True, color='k', scale=utils.mps_to_mmpyr(sz_case2td.v0))
    utils.plot.plot_vector_glyphs(sz_case2td.vs_i, plotter=plotter_distd, factor=0.1, gather=True, color='k', scale=utils.mps_to_mmpyr(sz_case2td.v0))
    utils.plot.plot_geometry(sz_case1td.geom, plotter=plotter_distd, color='green', width=2)
    utils.plot.plot_couplingdepth(sz_case1td.geom.slab_spline, plotter=plotter_distd, render_points_as_spheres=True, point_size=10.0, color='green')
    utils.plot.plot_show(plotter_distd)
    utils.plot.plot_save(plotter_distd, output_folder / "sz_problem_case2td_solution.png")


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/sz_problems --NbConvertApp.output_base=sz_problem 3.2d_sz_problem.ipynb')




