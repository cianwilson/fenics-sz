#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


from sz_problems.sz_params import default_params, allsz_params
from sz_problems.sz_slab import create_slab
from sz_problems.sz_geometry import create_sz_geometry
from sz_problems.sz_problem import SubductionProblem


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


class SteadySubductionProblem(SubductionProblem):
    def temperature_forms(self):
        """
        Return the forms ST and fT for the matrix problem ST*T = fT for the steady-state 
        temperature advection-diffusion problem.

        Returns:
          * ST - lhs bilinear form for the temperature problem
          * fT - rhs linear form for the temperature problem
          * rT - residual linear form for the temperature problem
        """
        with df.common.Timer("Forms Temperature"):
            # set the crustal conductivity
            kc   = self.kc
            if self.sztype=='oceanic':
                # if we are oceanic then we use the mantle value
                kc   = self.km
            
            # advection diffusion in the slab
            STs = (self.T_t*self.rhom*self.cp*ufl.inner(self.vs_i, ufl.grad(self.T_a)) + \
                ufl.inner(ufl.grad(self.T_a), self.km*ufl.grad(self.T_t)))*self.dx(self.slab_rids)
            # advection diffusion in the wedge
            STw = (self.T_t*self.rhom*self.cp*ufl.inner(self.vw_i, ufl.grad(self.T_a)) + \
                ufl.inner(ufl.grad(self.T_a), self.km*ufl.grad(self.T_t)))*self.dx(self.wedge_rids)
            # just diffusion in the crust
            STc = ufl.inner(ufl.grad(self.T_a), kc*ufl.grad(self.T_t))*self.dx(self.crust_rids)
            # the complete bilinear form
            ST  = STs + STw + STc
            if self.sztype=='continental':
                # if the sztype is 'continental' then put radiogenic heating in the rhs form
                lc_rids = tuple([self.geom.crustal_layers['Crust']['rid']])
                uc_rids = tuple([self.geom.crustal_layers['UpperCrust']['rid']])
                fT  = self.T_t*self.H1*self.dx(uc_rids) + self.T_t*self.H2*self.dx(lc_rids)
            else:
                # if the sztype is 'oceanic' then create a zero rhs form
                zero_c = df.fem.Constant(self.mesh, df.default_scalar_type(0.0))
                fT = self.T_t*zero_c*self.dx
            # residual form
            # (created as a list of forms so we can assemble into a nest vector)
            rT = df.fem.form([ufl.action(ST, self.T_i) - fT])
        # return the forms
        return df.fem.form(ST), df.fem.form(fT), df.fem.form(rT)

