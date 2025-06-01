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
    def temperature_forms_timedependent(self):
        """
        Return the forms ST and fT for the matrix problem ST*T = fT for the time dependent
        temperature advection-diffusion problem.

        Returns:
          * ST - lhs bilinear form for the temperature problem
          * fT - rhs linear form for the temperature problem
        """
        # integration measures that know about the cell and facet tags

        # set the crustal conductivity and density
        kc   = self.kc
        rhoc = self.rhoc
        if self.sztype=='oceanic':
            # if we are oceanic then we use the mantle values
            kc   = self.km
            rhoc = self.rhom

        # advection diffusion in the slab
        STs = self.T_t*self.rhom*self.cp*(self.T_a+self.dt*self.theta*ufl.inner(self.vs_i, ufl.grad(self.T_a)))*self.dx(self.slab_rids) + \
               self.dt*self.km*self.theta*ufl.inner(ufl.grad(self.T_t), ufl.grad(self.T_a))*self.dx(self.slab_rids)
        
        # advection diffusion in the wedge
        STw = self.T_t*self.rhom*self.cp*(self.T_a+self.dt*self.theta*ufl.inner(self.vw_i, ufl.grad(self.T_a)))*self.dx(self.wedge_rids) + \
               self.dt*self.km*self.theta*ufl.inner(ufl.grad(self.T_t), ufl.grad(self.T_a))*self.dx(self.wedge_rids)
        
        # just diffusion in the crust
        STc = self.T_t*rhoc*self.cp*(self.T_a)*self.dx(self.crust_rids) + \
               self.dt*kc*self.theta*ufl.inner(ufl.grad(self.T_t), ufl.grad(self.T_a))*self.dx(self.crust_rids)
        
        # the complete bilinear form
        ST  = STs + STw + STc
        
        fTs = self.T_t*self.rhom*self.cp*(self.T_n-self.dt*(1-self.theta)*ufl.inner(self.vs_i, ufl.grad(self.T_n)))*self.dx(self.slab_rids) - \
                (1-self.theta)*self.dt*self.km*ufl.inner(ufl.grad(self.T_t), ufl.grad(self.T_n))*self.dx(self.slab_rids)
        
        fTw = self.T_t*self.rhom*self.cp*(self.T_n-self.dt*(1-self.theta)*ufl.inner(self.vw_i, ufl.grad(self.T_n)))*self.dx(self.wedge_rids) - \
                (1-self.theta)*self.dt*self.km*ufl.inner(ufl.grad(self.T_t), ufl.grad(self.T_n))*self.dx(self.wedge_rids)
        
        fTc = self.T_t*rhoc*self.cp*self.T_n*self.dx(self.crust_rids) - \
                (1-self.theta)*self.dt*kc*ufl.inner(ufl.grad(self.T_t), ufl.grad(self.T_n))*self.dx(self.crust_rids)

        if self.sztype=='continental':
            # if the sztype is 'continental' then put radiogenic heating in the rhs form
            lc_rids = tuple([self.geom.crustal_layers['Crust']['rid']])
            uc_rids = tuple([self.geom.crustal_layers['UpperCrust']['rid']])
            
            fTc += self.T_t*self.dt*self.H1*self.dx(uc_rids) + self.T_t*self.dt*self.H2*self.dx(lc_rids)
        
        fT = fTs + fTw + fTc

        # return the forms
        return ST, fT


class SubductionProblem(SubductionProblem):
    def solve_timedependent_isoviscous(self, tf, dt, theta=0.5, verbosity=2, petsc_options=None, plotter=None):
        """
        Solve the coupled temperature-velocity-pressure problem assuming an isoviscous rheology with time dependency

        Arguments:
          * tf - final time  (in Myr)
          * dt - the timestep (in Myr)
          
        Keyword Arguments:
          * theta         - theta parameter for timestepping (0 <= theta <= 1, defaults to theta=0.5)
          * petsc_options - a dictionary of petsc options to pass to the solver (defaults to mumps)
          * verbosity     - level of verbosity (<1=silent, >0=basic, >1=timestep, defaults to 2)
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
        
        # first solve both Stokes systems
        self.solve_stokes_isoviscous(petsc_options=petsc_options)

        # retrieve the temperature forms
        ST, fT = self.temperature_forms_timedependent()
        problem_T = df.fem.petsc.LinearProblem(ST, fT, bcs=self.bcs_T, u=self.T_i,
                                               petsc_options=petsc_options)

        # and solve the temperature problem repeatedly with time step dt
        t = 0
        ti = 0
        tf_nd = tf/self.t0_Myr
        if self.comm.rank == 0 and verbosity>0:
            print("Entering timeloop with {:d} steps (dt = {:g} Myr, final time = {:g} Myr)".format(int(np.ceil(tf_nd/self.dt.value)), dt, tf,))
        while t < tf_nd - 1e-9:
            if self.comm.rank == 0 and verbosity>1:
                print("Step: {:>6d}, Times: {:>9g} -> {:>9g} Myr".format(ti, t*self.t0_Myr, (t+self.dt.value)*self.t0_Myr))
            if plotter is not None:
                for mesh in plotter.meshes:
                    if self.T_i.name in mesh.point_data:
                        mesh.point_data[self.T_i.name][:] = self.T_i.x.array
                plotter.write_frame()
            self.T_n.x.array[:] = self.T_i.x.array
            self.T_i = problem_T.solve()
            ti+=1
            t+=self.dt.value
        if self.comm.rank == 0 and verbosity>0:
            print("Finished timeloop after {:d} steps (final time = {:g} Myr)".format(ti, t*self.t0_Myr,))

        # only update the pressure at the end as it is not necessary earlier
        self.update_p_functions()


if __name__ == "__main__":
    geom_case1td = create_sz_geometry(slab, resscale, sztype, io_depth_1, extra_width, 
                              coast_distance, lc_depth, uc_depth)
    sz_case1td = SubductionProblem(geom_case1td, A=A, Vs=Vs, sztype=sztype, qs=qs)

    fps = 5
    plotter_gif = pv.Plotter(notebook=False, off_screen=True)
    utils.plot.plot_scalar(sz_case1td.T_i, plotter=plotter_gif, scale=sz_case1td.T0, gather=True, cmap='coolwarm', clim=[0.0, sz_case1td.Tm*sz_case1td.T0], scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    utils.plot.plot_geometry(sz_case1td.geom, plotter=plotter_gif, color='green', width=2)
    utils.plot.plot_couplingdepth(sz_case1td.geom.slab_spline, plotter=plotter_gif, render_points_as_spheres=True, point_size=10.0, color='green')
    plotter_gif.open_gif( str(output_folder / "sz_problem_case1td_solution.gif"), fps=fps)
    
    sz_case1td.solve_timedependent_isoviscous(25, 0.05, theta=0.5, plotter=plotter_gif)
    
    plotter_gif.close()


if __name__ == "__main__":
    plotter_isotd = utils.plot.plot_scalar(sz_case1td.T_i, scale=sz_case1td.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    utils.plot.plot_vector_glyphs(sz_case1td.vw_i, plotter=plotter_isotd, factor=0.1, gather=True, color='k', scale=utils.mps_to_mmpyr(sz_case1td.v0))
    utils.plot.plot_vector_glyphs(sz_case1td.vs_i, plotter=plotter_isotd, factor=0.1, gather=True, color='k', scale=utils.mps_to_mmpyr(sz_case1td.v0))
    utils.plot.plot_geometry(sz_case1td.geom, plotter=plotter_isotd, color='green', width=2)
    utils.plot.plot_couplingdepth(sz_case1td.geom.slab_spline, plotter=plotter_isotd, render_points_as_spheres=True, point_size=10.0, color='green')
    utils.plot.plot_show(plotter_isotd)
    utils.plot.plot_save(plotter_isotd, output_folder / "sz_problem_case1td_solution.png")


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/sz_problems --NbConvertApp.output_base=sz_problem 3.2d_sz_problem.ipynb')




