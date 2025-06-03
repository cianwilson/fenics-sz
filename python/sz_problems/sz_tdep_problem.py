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


class TDSubductionProblem(SubductionProblem):
    def members(self):
        super().members()

        # timestepping options
        self.theta = None
        self.dt    = None


class TDSubductionProblem(TDSubductionProblem):
    def temperature_forms(self):
        """
        Return the forms ST and fT for the matrix problem ST*T = fT for the time dependent
        temperature advection-diffusion problem.

        Returns:
          * ST - lhs bilinear form for the temperature problem
          * fT - rhs linear form for the temperature problem
          * rT - residual linear form for the temperature problem
        """
        with df.common.Timer("Forms Temperature"):
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


            # residual form
            # (created as a list of forms so we can assemble into a nest vector)
            rT = df.fem.form([ufl.action(ST, self.T_i) - fT])

        # return the forms
        return df.fem.form(ST), df.fem.form(fT), df.fem.form(rT)




