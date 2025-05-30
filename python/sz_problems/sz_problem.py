#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


from sz_problems.sz_base import default_params, allsz_params
from sz_problems.sz_slab import create_slab
from sz_problems.sz_geometry import create_sz_geometry


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
if __name__ == "__main__":
    output_folder = pathlib.Path(os.path.join(basedir, "output"))
    output_folder.mkdir(exist_ok=True, parents=True)


class SubductionProblem:
    """
    A class describing a kinematic slab subduction zone thermal problem.
    """

    def members(self):
        # geometry object
        self.geom = None
    
        # case specific
        self.A      = None      # age of subducting slab (Myr)
        self.Vs     = None      # slab speed (mm/yr)
        self.sztype = None      # type of sz ('continental' or 'oceanic')
        self.Ac     = None      # age of over-riding plate (Myr) - oceanic only
        self.As     = None      # age of subduction (Myr) - oceanic only
        self.qs     = None      # surface heat flux (W/m^2) - continental only
        
        # non-dim parameters
        self.Ts      = 0.0       # surface temperature (non-dim, also deg C)
        self.Tm      = 1350.0    # mantle temperature (non-dim, also deg C)
        self.kc      = 0.8064516 # crustal thermal conductivity (non-dim) - continental only
        self.km      = 1.0       # mantle thermal conductivity (non-dim)
        self.rhoc    = 0.8333333 # crustal density (non-dim) - continental only
        self.rhom    = 1.0       # mantle density (non-dim)
        self.cp      = 1.0       # heat capacity (non-dim)
        self.H1      = 0.419354  # upper crustal volumetric heat production (non-dim) - continental only
        self.H2      = 0.087097  # lower crustal volumetric heat production (non-dim) - continental only
    
        # dislocation creep parameters
        self.etamax = 1.0e25    # maximum viscosity (Pa s)
        self.nsigma = 3.5       # stress viscosity power law exponent (non-dim)
        self.Aeta   = 28968.6   # pre-exponential viscosity constant (Pa s^(1/nsigma))
        self.E      = 540.0e3   # viscosity activation energy (J/mol)
        
        # finite element degrees
        self.p_p = 1
        self.p_T = 2

        # only allow these options to be set from the update and __init__ functions
        self.allowed_input_parameters = ['A', 'Vs', 'sztype', 'Ac', 'As', 'qs', \
                                         'Ts', 'Tm', 'km', 'rhom', 'cp', \
                                         'etamax', 'nsigma', 'Aeta', 'E', \
                                         'p_p', 'p_T']
        self.allowed_if_continental   = ['kc', 'rhoc', 'H1', 'H2']
    
        self.required_parameters     = ['A', 'Vs', 'sztype']
        self.required_if_continental = ['qs']
        self.required_if_oceanic     = ['Ac', 'As']
    
        # reference values
        self.k0     = 3.1       # reference thermal conductivity (W/m/K)
        self.rho0   = 3300.0    # reference density (kg/m^3)
        self.cp0    = 1250.0    # reference heat capacity (J/kg/K)
        self.h0     = 1000.0    # reference length scale (m)
        self.eta0   = 1.0e21    # reference viscosity (Pa s)
        self.T0     = 1.0       # reference temperature (K)
        self.R      = 8.3145    # gas constant (J/mol/K)
    
        # derived reference values
        self.kappa0 = None  # reference thermal diffusivity (m^2/s)
        self.v0     = None  # reference velocity (m/s)
        self.t0     = None  # reference time-scale (s)
        self.e0     = None  # reference strain rate (/s)
        self.p0     = None  # reference pressure (Pa)
        self.H0     = None  # reference heat source (W/m^3)
        self.q0     = None  # reference heat flux (W/m^2)
        
        # derived parameters
        self.A_si   = None  # age of subducting slab (s)
        self.Vs_nd  = None  # slab speed (non-dim)
        self.Ac_si  = None  # age of over-riding plate (s) - oceanic only
        self.As_si  = None  # age of subduction (s) - oceanic only
        self.qs_nd  = None  # surface heat flux (non-dim) - continental only
    
        # derived from the geometry object
        self.deltaztrench = None
        self.deltaxcoast  = None
        self.deltazuc     = None
        self.deltazc      = None
    
        # mesh related
        self.mesh       = None
        self.cell_tags  = None
        self.facet_tags = None
    
        # MPI communicator
        self.comm       = None
    
        # dimensions and mesh statistics
        self.gdim = None
        self.tdim = None
        self.fdim = None
        self.num_cells = None

        # region ids
        self.wedge_rids       = None
        self.slab_rids        = None
        self.crust_rids       = None
    
        # wedge submesh
        self.wedge_submesh    = None
        self.wedge_cell_tags  = None
        self.wedge_facet_tags = None 
        self.wedge_cell_map   = None
        self.wedge_rank       = True
    
        # slab submesh
        self.slab_submesh    = None
        self.slab_cell_tags  = None
        self.slab_facet_tags = None
        self.slab_cell_map   = None
        self.slab_rank       = True
    
        # integral measures
        self.dx = None
        self.wedge_ds = None
    
        # functionspaces
        self.Vslab_vp  = None
        self.Vslab_v   = None
        self.Vwedge_vp = None
        self.Vwedge_v  = None
        self.V_T       = None
    
        # Functions
        self.slab_vps_i  = None
        self.wedge_vpw_i = None
        self.T_i         = None
        self.T_n         = None
    
        # Functions that need interpolation
        self.vs_i      = None
        self.ps_i      = None
        self.vw_i      = None
        self.pw_i      = None
        self.slab_T_i  = None
        self.wedge_T_i = None
    
        # sub (split) functions
        self.slab_vs_i  = None
        self.slab_ps_i  = None
        self.wedge_vw_i = None
        self.wedge_pw_i = None
    
        # test functions
        self.slab_vps_t = None
        self.wedge_vw_i = None
        self.T_t        = None
    
        # trial functions
        self.slab_vps_a  = None
        self.wedge_vpw_a = None
        self.T_a         = None
        
        # boundary conditions
        self.bcs_T   = None # temperature
        self.bcs_vpw = None # wedge velocity/pressure
        self.bcs_vps = None # slab velocity/pressure

        # timestepping options
        self.theta = None
        self.dt    = None
    


class SubductionProblem(SubductionProblem):
    def setup_meshes(self):
        """
        Generate the mesh from the supplied geometry then extract submeshes representing
        the wedge and slab for the Stokes problems in these regions.
        """
        # check we have a geometry object attached
        assert self.geom is not None

        # generate the mesh using gmsh
        # this command also returns cell and facets tags identifying regions and boundaries in the mesh
        self.mesh, self.cell_tags, self.facet_tags = self.geom.generatemesh()
        self.comm = self.mesh.comm

        # record the dimensions
        self.gdim = self.mesh.geometry.dim
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1

        # get the number of cells
        cell_imap = self.mesh.topology.index_map(self.tdim)
        self.num_cells = cell_imap.size_local + cell_imap.num_ghosts

        # record the region ids for the wedge, slab and crust based on the geometry
        self.wedge_rids = tuple(set([v['rid'] for k,v in self.geom.wedge_dividers.items()]+[self.geom.wedge_rid]))
        self.slab_rids  = tuple([self.geom.slab_rid])
        self.crust_rids = tuple(set([v['rid'] for k,v in self.geom.crustal_layers.items()]))

        # generate the wedge submesh
        # this command also returns cell and facet tags mapped from the parent mesh to the submesh
        # additionally a cell map maps cells in the submesh to the parent mesh
        self.wedge_submesh, self.wedge_cell_tags, self.wedge_facet_tags, self.wedge_cell_map = \
                            utils.mesh.create_submesh(self.mesh, 
                                                 np.concatenate([self.cell_tags.find(rid) for rid in self.wedge_rids]), \
                                                 self.cell_tags, self.facet_tags)
        # record whether this MPI rank has slab DOFs or not
        self.wedge_rank = self.wedge_submesh.topology.index_map(self.tdim).size_local > 0
        
        # generate the slab submesh
        # this command also returns cell and facet tags mapped from the parent mesh to the submesh
        # additionally a cell map maps cells in the submesh to the parent mesh
        self.slab_submesh, self.slab_cell_tags, self.slab_facet_tags, self.slab_cell_map  = \
                            utils.mesh.create_submesh(self.mesh, 
                                                 np.concatenate([self.cell_tags.find(rid) for rid in self.slab_rids]), \
                                                 self.cell_tags, self.facet_tags)
        # record whether this MPI rank has wedge DOFs or not
        self.slab_rank = self.slab_submesh.topology.index_map(self.tdim).size_local > 0

        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags)
        self.wedge_ds = ufl.Measure("ds", domain=self.wedge_submesh, subdomain_data=self.wedge_facet_tags)


class SubductionProblem(SubductionProblem):
    def setup_functionspaces(self):
        """
        Set up the functionspaces for the problem.
        """
        # create finite elements for velocity and pressure
        # use a P2P1 (Taylor-Hood) element pair where the velocity
        # degree is one higher than the pressure (so only the pressure
        # degree can be set)
        v_e = bu.element("Lagrange", self.mesh.basix_cell(), self.p_p+1, shape=(self.gdim,), dtype=df.default_real_type)
        p_e = bu.element("Lagrange", self.mesh.basix_cell(), self.p_p, dtype=df.default_real_type)
        # combine them into a mixed finite element
        vp_e = bu.mixed_element([v_e, p_e])
        
        def create_vp_functions(mesh, name_prefix):
            """
            Create velocity and pressure functions
            """
            # set up the mixed velocity, pressure functionspace
            V_vp = df.fem.functionspace(mesh, vp_e)
            # set up a collapsed velocity functionspace
            V_v, _ = V_vp.sub(0).collapse()

            # set up a mixed velocity, pressure function
            vp_i = df.fem.Function(V_vp)
            vp_i.name = name_prefix+"vp"
            # split the velocity and pressure subfunctions
            (v_i, p_i) = vp_i.split()
            v_i.name = name_prefix+"v"
            p_i.name = name_prefix+"p"
            # set up the mixed velocity, pressure test function
            vp_t = ufl.TestFunction(V_vp)
            # set up the mixed velocity, pressure trial function
            vp_a = ufl.TrialFunction(V_vp)

            # return everything
            return V_vp, V_v, vp_i, v_i, p_i, vp_t, vp_a
        
        # set up slab functionspace, collapsed velocity sub-functionspace, 
        # combined velocity-pressure Function, split velocity and pressure Functions,
        # and trial and test functions for
        # 1. the slab submesh
        self.Vslab_vp,  self.Vslab_v,  \
                        self.slab_vps_i, \
                        self.slab_vs_i, self.slab_ps_i, \
                        self.slab_vps_t, self.slab_vps_a = create_vp_functions(self.slab_submesh, "slab_")
        # 2. the wedge submesh
        self.Vwedge_vp, self.Vwedge_v, \
                        self.wedge_vpw_i, \
                        self.wedge_vw_i, self.wedge_pw_i, \
                        self.wedge_vpw_t, self.wedge_vpw_a = create_vp_functions(self.wedge_submesh, "wedge_")

        # set up the mixed velocity, pressure functionspace (not stored)
        V_vp   = df.fem.functionspace(self.mesh, vp_e)
        V_v, _ = V_vp.sub(0).collapse()
        V_p, _ = V_vp.sub(1).collapse()

        # set up functions defined on the whole mesh
        # to interpolate the wedge and slab velocities
        # and pressures to
        self.vs_i = df.fem.Function(V_v)
        self.vs_i.name = "vs"
        self.ps_i = df.fem.Function(V_p)
        self.ps_i.name = "ps"
        self.vw_i = df.fem.Function(V_v)
        self.vw_i.name = "vw"
        self.pw_i = df.fem.Function(V_p)
        self.pw_i.name = "pw"
        
        # temperature element
        # the degree of the element can be set independently through p_T
        T_e = bu.element("Lagrange", self.mesh.basix_cell(), self.p_T, dtype=df.default_real_type)
        # and functionspace on the overall mesh
        self.V_T  = df.fem.functionspace(self.mesh, T_e)

        # create a dolfinx Function for the temperature
        self.T_i = df.fem.Function(self.V_T)
        self.T_i.name = "T"
        self.T_n = df.fem.Function(self.V_T)
        self.T_n.name = "T_n"
        self.T_t = ufl.TestFunction(self.V_T)
        self.T_a = ufl.TrialFunction(self.V_T)
        
        # on the slab submesh
        Vslab_T = df.fem.functionspace(self.slab_submesh, T_e)
        # and on the wedge submesh
        Vwedge_T = df.fem.functionspace(self.wedge_submesh, T_e)
        # set up Functions so the solution can be interpolated to these subdomains
        self.slab_T_i  = df.fem.Function(Vslab_T)
        self.slab_T_i.name = "slab_T"
        self.wedge_T_i = df.fem.Function(Vwedge_T)
        self.wedge_T_i.name = "wedge_T"


class SubductionProblem(SubductionProblem):
    def update_T_functions(self):
        """
        Update the temperature functions defined on the submeshes, given a solution on the full mesh.
        """
        self.slab_T_i.interpolate(self.T_i, cells0=self.slab_cell_map, cells1=np.arange(len(self.slab_cell_map)))
        self.wedge_T_i.interpolate(self.T_i, cells0=self.wedge_cell_map, cells1=np.arange(len(self.wedge_cell_map)))
    
    def update_v_functions(self):
        """
        Update the velocity functions defined on the full mesh, given solutions on the sub meshes.
        """
        self.vs_i.interpolate(self.slab_vs_i, cells0=np.arange(len(self.slab_cell_map)), cells1=self.slab_cell_map)
        self.vw_i.interpolate(self.wedge_vw_i, cells0=np.arange(len(self.wedge_cell_map)), cells1=self.wedge_cell_map)

    def update_p_functions(self):
        """
        Update the pressure functions defined on the full mesh, given solutions on the sub meshes.
        """
        self.ps_i.interpolate(self.slab_ps_i, cells0=np.arange(len(self.slab_cell_map)), cells1=self.slab_cell_map)
        self.pw_i.interpolate(self.wedge_pw_i, cells0=np.arange(len(self.wedge_cell_map)), cells1=self.wedge_cell_map)


class SubductionProblem(SubductionProblem):
    def T_trench(self, x):
        """
        Return temperature at the trench
        """
        zd = 2*np.sqrt(self.kappa0*self.A_si)/self.h0  # incoming slab scale depth (non-dim)
        deltazsurface = np.minimum(np.maximum(self.deltaztrench*(1.0 - x[0,:]/max(self.deltaxcoast, np.finfo(float).eps)), 0.0), self.deltaztrench)
        return self.Ts + (self.Tm-self.Ts)*sp.special.erf(-(x[1,:]+deltazsurface)/zd)


class SubductionProblem(SubductionProblem):
    def T_backarc_o(self, x):
        """
        Return temperature at the trench
        """
        zc = 2*np.sqrt(self.kappa0*(self.Ac_si-self.As_si))/self.h0 # overriding plate scale depth (non-dim)
        deltazsurface = np.minimum(np.maximum(self.deltaztrench*(1.0 - x[0,:]/max(self.deltaxcoast, np.finfo(float).eps)), 0.0), self.deltaztrench)
        return self.Ts + (self.Tm-self.Ts)*sp.special.erf(-(x[1,:]+deltazsurface)/zc)


class SubductionProblem(SubductionProblem):
    def T_backarc_c(self, x):
        """
        Return continental backarc temperature
        """
        T = np.empty(x.shape[1])
        deltazsurface = np.minimum(np.maximum(self.deltaztrench*(1.0 - x[0,:]/max(self.deltaxcoast, np.finfo(float).eps)), 0.0), self.deltaztrench)
        for i in range(x.shape[1]):
            if -(x[1,i]+deltazsurface[i]) < self.deltazuc:
                # if in the upper crust
                deltaz = -(x[1,i]+deltazsurface[i])
                T[i] = self.Ts - self.H1*(deltaz**2)/(2*self.kc) + (self.qs_nd/self.kc)*deltaz
            elif -(x[1,i]+deltazsurface[i]) < self.deltazc:
                # if in the lower crust
                deltaz1 = self.deltazuc #- deltazsurface[i]
                T1 = self.Ts - self.H1*(deltaz1**2)/(2*self.kc) + (self.qs_nd/self.kc)*deltaz1
                q1 = - self.H1*deltaz1 + self.qs_nd
                deltaz = -(x[1,i] + deltazsurface[i] + self.deltazuc)
                T[i] = T1 - self.H2*(deltaz**2)/(2*self.kc) + (q1/self.kc)*deltaz
            else:
                # otherwise, we're in the mantle
                deltaz1 = self.deltazuc # - deltazsurface[i]
                T1 = self.Ts - self.H1*(deltaz1**2)/(2*self.kc) + (self.qs_nd/self.kc)*deltaz1
                q1 = - self.H1*deltaz1 + self.qs_nd
                deltaz2 = self.deltazc - self.deltazuc #- deltazsurface[i]
                T2 = T1 - self.H2*(deltaz2**2)/(2*self.kc) + (q1/self.kc)*deltaz2
                q2 = - self.H2*deltaz2 + q1
                deltaz = -(x[1,i] + deltazsurface[i] + self.deltazc)
                T[i] = min(self.Tm, T2 + (q2/self.km)*deltaz)
        return T
        


class SubductionProblem(SubductionProblem):
    def vw_slabtop(self, x):
        """
        Return the wedge velocity on the slab surface
        """
        # grab the partial and full coupling depths so we can set up a linear ramp in velocity between them
        pcd = -self.geom.slab_spline.findpoint("Slab::PartialCouplingDepth").y
        fcd = -self.geom.slab_spline.findpoint("Slab::FullCouplingDepth").y
        dcd = fcd-pcd
        v = np.empty((self.gdim, x.shape[1]))
        for i in range(x.shape[1]):
            v[:,i] = min(max(-(x[1,i]+pcd)/dcd, 0.0), 1.0)*self.Vs_nd*self.geom.slab_spline.unittangentx(x[0,i])
        return v
    


class SubductionProblem(SubductionProblem):
    def vs_slabtop(self, x):
        """
        Return the slab velocity on the slab surface
        """
        v = np.empty((self.gdim, x.shape[1]))
        for i in range(x.shape[1]):
            v[:,i] = self.Vs_nd*self.geom.slab_spline.unittangentx(x[0,i])
        return v


class SubductionProblem(SubductionProblem):
    def setup_boundaryconditions(self):
        """
        Set the boundary conditions and apply them to the functions
        """
        # locate the degrees of freedom (dofs) where various boundary conditions will be applied
        # on the top of the wedge for the wedge velocity
        wedgetop_dofs_Vwedge_v = df.fem.locate_dofs_topological((self.Vwedge_vp.sub(0), self.Vwedge_v), self.fdim,
                                                                np.concatenate([self.wedge_facet_tags.find(sid) for sid in set([line.pid for line in self.geom.crustal_lines[0]])]))
        # on the slab surface for the slab velocity
        slab_dofs_Vslab_v = df.fem.locate_dofs_topological((self.Vslab_vp.sub(0), self.Vslab_v), self.fdim, 
                                                           np.concatenate([self.slab_facet_tags.find(sid) for sid in set(self.geom.slab_spline.pids)]))
        # on the slab surface for the wedge velocity
        slab_dofs_Vwedge_v = df.fem.locate_dofs_topological((self.Vwedge_vp.sub(0), self.Vwedge_v), self.fdim, 
                                                            np.concatenate([self.wedge_facet_tags.find(sid) for sid in set(self.geom.slab_spline.pids)]))
        # on the top of the domain for the temperature
        top_dofs_V_T = df.fem.locate_dofs_topological(self.V_T, self.fdim, 
                                                      np.concatenate([self.facet_tags.find(self.geom.coast_sid), self.facet_tags.find(self.geom.top_sid)]))
        # on the side of the slab side of the domain for the temperature
        slabside_dofs_V_T = df.fem.locate_dofs_topological(self.V_T, self.fdim, 
                                                           np.concatenate([self.facet_tags.find(sid) for sid in set([line.pid for line in self.geom.slab_side_lines])]))
        # on the side of the wedge side of the domain for the temperature
        wedgeside_dofs_V_T = df.fem.locate_dofs_topological(self.V_T, self.fdim, 
                                                            np.concatenate([self.facet_tags.find(sid) for sid in set([line.pid for line in self.geom.wedge_side_lines[1:]])]))
        
        # temperature boundary conditions        
        self.bcs_T = []
        # zero on the top of the domain
        zero_c = df.fem.Constant(self.mesh, df.default_scalar_type(0.0))
        self.bcs_T.append(df.fem.dirichletbc(zero_c, top_dofs_V_T, self.V_T))
        # an incoming slab thermal profile on the lhs of the domain
        T_trench_f = df.fem.Function(self.V_T)
        T_trench_f.interpolate(self.T_trench)
        self.bcs_T.append(df.fem.dirichletbc(T_trench_f, slabside_dofs_V_T))
        # on the top (above iodepth) of the incoming wedge side of the domain
        if self.sztype=='continental':
            T_backarc_f = df.fem.Function(self.V_T)
            T_backarc_f.interpolate(self.T_backarc_c)
            self.bcs_T.append(df.fem.dirichletbc(T_backarc_f, wedgeside_dofs_V_T))
        else:
            T_backarc_f = df.fem.Function(self.V_T)
            T_backarc_f.interpolate(self.T_backarc_o)
            self.bcs_T.append(df.fem.dirichletbc(T_backarc_f, wedgeside_dofs_V_T))
            
        # wedge velocity (and pressure) boundary conditions
        self.bcs_vpw = []
        # zero velocity on the top of the wedge
        zero_vw_f = df.fem.Function(self.Vwedge_v)
        zero_vw_f.x.array[:] = 0.0
        self.bcs_vpw.append(df.fem.dirichletbc(zero_vw_f, wedgetop_dofs_Vwedge_v, self.Vwedge_vp.sub(0)))
        # kinematic slab on the slab surface of the wedge
        vw_slabtop_f = df.fem.Function(self.Vwedge_v)
        vw_slabtop_f.interpolate(self.vw_slabtop)
        self.bcs_vpw.append(df.fem.dirichletbc(vw_slabtop_f, slab_dofs_Vwedge_v, self.Vwedge_vp.sub(0)))

        # slab velocity (and pressure) boundary conditions
        self.bcs_vps = []
        # kinematic slab on the slab surface of the slab
        vs_slabtop_f = df.fem.Function(self.Vslab_v)
        vs_slabtop_f.interpolate(self.vs_slabtop)
        self.bcs_vps.append(df.fem.dirichletbc(vs_slabtop_f, slab_dofs_Vslab_v, self.Vslab_vp.sub(0)))

        # interpolate the temperature boundary conditions as initial conditions/guesses
        # to the whole domain (not just the boundaries)
        # on the wedge and crust side of the domain apply the wedge condition
        nonslab_cells = np.concatenate([self.cell_tags.find(rid) for domain in [self.crust_rids, self.wedge_rids] for rid in domain])
        self.T_i.interpolate(T_backarc_f, cells0=nonslab_cells)
        # on the slab side of the domain apply the slab condition
        slab_cells = np.concatenate([self.cell_tags.find(rid) for rid in self.slab_rids])
        self.T_i.interpolate(T_trench_f, cells0=slab_cells)
        # update the interpolated T functions for consistency
        self.update_T_functions()

        # just set the boundary conditions on the boundaries for the velocities
        self.wedge_vpw_i.x.array[:] = 0.0
        self.slab_vps_i.x.array[:] = 0.0
        df.fem.set_bc(self.wedge_vpw_i.x.array, self.bcs_vpw)
        df.fem.set_bc(self.slab_vps_i.x.array, self.bcs_vps)
        # and update the interpolated v functions for consistency
        self.update_v_functions()


class SubductionProblem(SubductionProblem):
    def update(self, geom=None, **kwargs):
        """
        Update the subduction problem with the allowed input parameters
        """

        # loop over the keyword arguments and apply any that are allowed as input parameters
        # this loop happens before conditional loops to make sure sztype gets set
        for k,v in kwargs.items():
            if k in self.allowed_input_parameters and hasattr(self, k):
                setattr(self, k, v)

        # loop over additional input parameters, giving a warning if they will do nothing
        for k,v in kwargs.items():
            if k in self.allowed_if_continental and hasattr(self, k):
                setattr(self, k, v)
                if self.sztype == "oceanic":
                    raise Warning("sztype is '{}' so setting '{}' will have no effect.".format(self.sztype, k))
        
        # check required parameters are set
        for param in self.required_parameters:
            value = getattr(self, param)
            if value is None:
                raise Exception("'{}' must be set but isn't.  Please supply a value.".format(param,))

        # check sztype dependent required parameters are set
        if self.sztype == "continental":
            for param in self.required_if_continental:
                value = getattr(self, param)
                if value is None:
                    raise Exception("'{}' must be set if the sztype is continental.  Please supply a value.".format(param,))
        elif self.sztype == "oceanic":
            for param in self.required_if_oceanic:
                value = getattr(self, param)
                if value is None:
                    raise Exception("'{}' must be set if the sztype is oceanic.  Please supply a value.".format(param,))
        else:
            raise Exception("Unknown sztype ({}).  Please set a valid sztype (continental or oceanic).".format(self.sztype))
            
        # set the geometry and generate the meshes and functionspaces
        if geom is not None:
            self.geom = geom
            self.setup_meshes()
            self.setup_functionspaces()

        # derived reference values
        self.kappa0 = self.k0/self.rho0/self.cp0   # reference thermal diffusivity (m^2/s)
        self.v0     = self.kappa0/self.h0          # reference velocity (m/s)
        self.t0     = self.h0/self.v0              # reference time (s)
        self.t0_Myr = utils.s_to_Myr(self.t0)      # reference time (Myr)
        self.e0     = self.v0/self.h0              # reference strain rate (/s)
        self.p0     = self.e0*self.eta0            # reference pressure (Pa)
        self.H0     = self.k0*self.T0/(self.h0**2) # reference heat source (W/m^3)
        self.q0     = self.H0*self.h0              # reference heat flux (W/m^2)

        # derived parameters
        self.A_si      = utils.Myr_to_s(self.A)   # age of subducting slab (s)
        self.Vs_nd     = utils.mmpyr_to_mps(self.Vs)/self.v0 # slab speed (non-dim)
        if self.sztype == 'oceanic':
            self.Ac_si = utils.Myr_to_s(self.Ac)  # age of over-riding plate (s)
            self.As_si = utils.Myr_to_s(self.As)  # age of subduction (s)
        else:
            self.qs_nd = self.qs/self.q0          # surface heat flux (non-dim)
        
        # parameters derived from from the geometry
        # depth of the trench
        self.deltaztrench = -self.geom.slab_spline.findpoint('Slab::Trench').y
        # coastline distance
        self.deltaxcoast  = self.geom.coast_distance
        # crust depth
        self.deltazc      = -self.geom.crustal_lines[0][0].y.min()
        if self.sztype == "continental":
            # upper crust depth
            self.deltazuc     = -self.geom.crustal_lines[-1][0].y.min()

        self.setup_boundaryconditions()
    
    def __init__(self, geom, **kwargs):
        """
        Initialize a SubductionProblem.

        Arguments:
          * geom  - an instance of a subduction zone geometry

        Keyword Arguments:
         required:
          * A      - age of subducting slab (in Myr) [required]
          * Vs     - incoming slab speed (in mm/yr) [required]
          * sztype - type of subduction zone (either 'continental' or 'oceanic') [required]
          * Ac     - age of the over-riding plate (in Myr) [required if sztype is 'oceanic']
          * As     - age of subduction (in Myr) [required if sztype is 'oceanic']
          * qs     - surface heat flux (in W/m^2) [required if sztype is 'continental']

         optional:
          * Ts   - surface temperature (deg C, corresponds to non-dim)
          * Tm   - mantle temperature (deg C, corresponds to non-dim)
          * kc   - crustal thermal conductivity (non-dim) [only has an effect if sztype is 'continental']
          * km   - mantle thermal conductivity (non-dim)
          * rhoc - crustal density (non-dim) [only has an effect if sztype is 'continental']
          * rhom - mantle density (non-dim)
          * cp   - isobaric heat capacity (non-dim)
          * H1   - upper crustal volumetric heat production (non-dim) [only has an effect if sztype is 'continental']
          * H2   - lower crustal volumetric heat production (non-dim) [only has an effect if sztype is 'continental']

         optional (dislocation creep rheology):
          * etamax - maximum viscosity (Pas) [only relevant for dislocation creep rheologies]
          * nsigma - stress viscosity power law exponents (non-dim) [only relevant for dislocation creep rheologies]
          * Aeta   - pre-exponential viscosity constant (Pa s^(1/n)) [only relevant for dislocation creep rheologies]
          * E      - viscosity activation energy (J/mol) [only relevant for dislocation creep rheologies]
        """
        self.members() # declare all the members
        self.update(geom=geom, **kwargs)


if __name__ == "__main__":
    resscale = 5.0


if __name__ == "__main__":
    xs = [0.0, 140.0, 240.0, 400.0]
    ys = [0.0, -70.0, -120.0, -200.0]
    lc_depth = 40
    uc_depth = 15
    coast_distance = 0
    extra_width = 0
    sztype = 'continental'
    io_depth_1 = 139


if __name__ == "__main__":
    A      = 100.0      # age of subducting slab (Myr)
    qs     = 0.065      # surface heat flux (W/m^2)
    Vs     = 100.0      # slab speed (mm/yr)


if __name__ == "__main__":
    slab = create_slab(xs, ys, resscale, lc_depth)
    geom = create_sz_geometry(slab, resscale, sztype, io_depth_1, extra_width, 
                              coast_distance, lc_depth, uc_depth)
    sz_case1 = SubductionProblem(geom, A=A, Vs=Vs, sztype=sztype, qs=qs)


if __name__ == "__main__":
    plotter_ic = utils.plot.plot_scalar(sz_case1.T_i, scale=sz_case1.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    utils.plot.plot_vector_glyphs(sz_case1.vw_i, plotter=plotter_ic, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz_case1.v0))
    utils.plot.plot_geometry(sz_case1.geom, plotter=plotter_ic, color='green', width=2)
    utils.plot.plot_couplingdepth(sz_case1.geom.slab_spline, plotter=plotter_ic, render_points_as_spheres=True, point_size=10.0, color='green')
    utils.plot.plot_show(plotter_ic)
    utils.plot.plot_save(plotter_ic, output_folder / "sz_problem_case1_ics.png")


class SubductionProblem(SubductionProblem):
    def stokes_forms(self, vp_t, vp_a, mesh, eta=1):
        """
        Return the forms Ss and fs for the matrix problem Ss*us = fs for the Stokes problems
        given the test and trial functions and the mesh.

        Arguments:
          * vp_t - velocity-pressure test function
          * vp_a - velocity-pressure trial function
          * mesh - mesh

        Keyword Arguments:
          * eta  - viscosity (defaults to 1 for isoviscous)

        Returns:
          * Ss - lhs bilinear form for the Stokes problem
          * fs - rhs linear form for the Stokes problem
        """
        (v_t, p_t) = ufl.split(vp_t)
        (v_a, p_a) = ufl.split(vp_a)
        # the stiffness block
        Ks = ufl.inner(ufl.sym(ufl.grad(v_t)), 2*eta*ufl.sym(ufl.grad(v_a)))*ufl.dx
        # gradient of pressure
        Gs = -ufl.div(v_t)*p_a*ufl.dx
        # divergence of velcoity
        Ds = -p_t*ufl.div(v_a)*ufl.dx
        # combined matrix form
        Ss = Ks + Gs + Ds
        # this problem has no rhs so create a dummy form by multiplying by a zero constant
        zero_c = df.fem.Constant(mesh, df.default_scalar_type(0.0))
        fs = zero_c*(sum(v_t) + p_t)*ufl.dx
        # return the forms
        return Ss, fs


class SubductionProblem(SubductionProblem):
    def temperature_forms_steadystate(self):
        """
        Return the forms ST and fT for the matrix problem ST*T = fT for the steady-state 
        temperature advection-diffusion problem.

        Returns:
          * ST - lhs bilinear form for the temperature problem
          * fT - rhs linear form for the temperature problem
        """
        # integration measures that know about the cell and facet tags

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
        # return the forms
        return ST, fT


class SubductionProblem(SubductionProblem):
    def solve_stokes_isoviscous(self, petsc_options=None):
        """
        Solve the Stokes problems assuming an isoviscous rheology.

        Keyword Arguments:
          * petsc_options - a dictionary of petsc options to pass to the solver (defaults to mumps)
        """
        if petsc_options is None:
            petsc_options={"ksp_type": "preonly", 
                           "pc_type" : "lu", 
                           "pc_factor_mat_solver_type" : "mumps"}

        # retrive the Stokes forms for the wedge
        Ssw, fsw = self.stokes_forms(self.wedge_vpw_t, self.wedge_vpw_a, self.wedge_submesh)
        problem_vpw = df.fem.petsc.LinearProblem(Ssw, fsw, bcs=self.bcs_vpw, u=self.wedge_vpw_i, 
                                                 petsc_options=petsc_options)
        
        # retrive the Stokes forms for the slab
        Sss, fss = self.stokes_forms(self.slab_vps_t, self.slab_vps_a, self.slab_submesh)
        problem_vps = df.fem.petsc.LinearProblem(Sss, fss, bcs=self.bcs_vps, u=self.slab_vps_i,
                                                 petsc_options=petsc_options)

        # solve the Stokes problems
        if self.wedge_rank: self.wedge_vpw_i = problem_vpw.solve()
        if self.slab_rank:  self.slab_vps_i = problem_vps.solve()
        
        # interpolate the solutions to the whole mesh
        self.update_v_functions()
        
    def solve_steadystate_isoviscous(self, petsc_options=None):
        """
        Solve the coupled temperature-velocity-pressure problem assuming an isoviscous rheology

        Keyword Arguments:
          * petsc_options - a dictionary of petsc options to pass to the solver (defaults to mumps)
        """
        if petsc_options is None:
            petsc_options={"ksp_type": "preonly", 
                           "pc_type" : "lu", 
                           "pc_factor_mat_solver_type" : "mumps"}

        # first solve both Stokes systems
        self.solve_stokes_isoviscous(petsc_options=petsc_options)

        # retrieve the temperature forms
        ST, fT = self.temperature_forms_steadystate()
        problem_T = df.fem.petsc.LinearProblem(ST, fT, bcs=self.bcs_T, u=self.T_i,
                                               petsc_options=petsc_options)
        # and solve the temperature problem
        self.T_i = problem_T.solve()

        # only update the pressure at the end as it is not necessary earlier
        self.update_p_functions()


class SubductionProblem(SubductionProblem):
    def get_diagnostics(self):
        """
        Retrieve the benchmark diagnostics.

        Returns:
          * Tndof     - number of degrees of freedom for temperature
          * Tpt       - spot temperature on the slab at 100 km depth
          * Tslab     - average temperature along the diagnostic region of the slab surface
          * Twedge    - average temperature in the diagnostic region of the wedge
          * vrmswedge - average rms velocity in the diagnostic region of the wedge
        """
        # work out number of T dofs
        Tndof = self.V_T.dofmap.index_map.size_global * self.V_T.dofmap.index_map_bs
        if self.comm.rank == 0: print("T_ndof = {:d}".format(Tndof,))
        
        # work out location of spot tempeterature on slab and evaluate T
        xpt = np.asarray(self.geom.slab_spline.intersecty(-100.0)+[0.0])
        cinds, cells = utils.mesh.get_cell_collisions(xpt, self.mesh)
        Tpt = np.nan
        if len(cells) > 0: Tpt = self.T0*self.T_i.eval(xpt, cells[0])[0]
        # FIXME: does this really have to be an allgather?
        Tpts = self.comm.allgather(Tpt)
        Tpt = float(next(T for T in Tpts if not np.isnan(T)))
        if self.comm.rank == 0: print("T_(200,-100) = {:.2f} deg C".format(Tpt,))

        # evaluate the length of the slab along which we will take the average T
        slab_diag_sids = tuple([self.geom.wedge_dividers['WedgeFocused']['slab_sid']])
        slab_diag_length = df.fem.assemble_scalar(df.fem.form(df.fem.Constant(self.wedge_submesh, df.default_scalar_type(1.0))*self.wedge_ds(slab_diag_sids)))
        slab_diag_length = self.comm.allreduce(slab_diag_length, op=MPI.SUM)
        if self.comm.rank == 0: print("slab_diag_length = {:.2f}".format(slab_diag_length,))
        
        # evaluate average T along diagnostic section of slab
        # to avoid having to share facets in parallel we evaluate the slab temperature
        # on the wedge submesh so first we update the wedge_T_i function
        self.update_T_functions()
        Tslab = self.T0*df.fem.assemble_scalar(df.fem.form(self.wedge_T_i*self.wedge_ds(slab_diag_sids)))
        Tslab = self.comm.allreduce(Tslab, op=MPI.SUM)/slab_diag_length
        if self.comm.rank == 0: print("T_slab = {:.2f} deg C".format(Tslab,))
        
        # evaluate the area of the wedge in which we will take the average T and vrms
        wedge_diag_rids = tuple([self.geom.wedge_dividers['WedgeFocused']['rid']])
        wedge_diag_area = df.fem.assemble_scalar(df.fem.form(df.fem.Constant(self.mesh, df.default_scalar_type(1.0))*self.dx(wedge_diag_rids)))
        wedge_diag_area = self.comm.allreduce(wedge_diag_area, op=MPI.SUM)
        if self.comm.rank == 0: print("wedge_diag_area = {:.2f}".format(wedge_diag_area,))

        # evaluate average T in wedge diagnostic region
        Twedge = self.T0*df.fem.assemble_scalar(df.fem.form(self.T_i*self.dx(wedge_diag_rids)))
        Twedge = self.comm.allreduce(Twedge, op=MPI.SUM)/wedge_diag_area
        if self.comm.rank == 0: print("T_wedge = {:.2f} deg C".format(Twedge,))

        # evaluate average vrms in wedge diagnostic region
        vrmswedge = df.fem.assemble_scalar(df.fem.form(ufl.inner(self.vw_i, self.vw_i)*self.dx(wedge_diag_rids)))
        vrmswedge = ((self.comm.allreduce(vrmswedge, op=MPI.SUM)/wedge_diag_area)**0.5)*utils.mps_to_mmpyr(self.v0)
        if self.comm.rank == 0: print("V_rms,w = {:.2f} mm/yr".format(vrmswedge,))

        # return results
        return Tndof, Tpt, Tslab, Twedge, vrmswedge


if __name__ == "__main__":
    sz_case1 = SubductionProblem(geom, A=A, Vs=Vs, sztype=sztype, qs=qs)


if __name__ == "__main__":
    sz_case1.solve_steadystate_isoviscous()


if __name__ == "__main__":
    diag = sz_case1.get_diagnostics()

    if sz_case1.comm.rank == 0:
        print('')
        print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
        print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale, *diag))


if __name__ == "__main__":
    plotter_iso = utils.plot.plot_scalar(sz_case1.T_i, scale=sz_case1.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    utils.plot.plot_vector_glyphs(sz_case1.vw_i, plotter=plotter_iso, factor=0.1, gather=True, color='k', scale=utils.mps_to_mmpyr(sz_case1.v0))
    utils.plot.plot_vector_glyphs(sz_case1.vs_i, plotter=plotter_iso, factor=0.1, gather=True, color='k', scale=utils.mps_to_mmpyr(sz_case1.v0))
    utils.plot.plot_geometry(sz_case1.geom, plotter=plotter_iso, color='green', width=2)
    utils.plot.plot_couplingdepth(sz_case1.geom.slab_spline, plotter=plotter_iso, render_points_as_spheres=True, point_size=10.0, color='green')
    utils.plot.plot_show(plotter_iso)
    utils.plot.plot_save(plotter_iso, output_folder / "sz_problem_case1_solution.png")


if __name__ == "__main__":
    filename = output_folder / "sz_problem_case1_solution.bp"
    with df.io.VTXWriter(sz_case1.mesh.comm, filename, [sz_case1.T_i, sz_case1.vs_i, sz_case1.vw_i]) as vtx:
        vtx.write(0.0)
    # zip the .bp folder so that it can be downloaded from Jupyter lab
    if "__file__" not in globals():
        zipfilename = filename.with_suffix(".zip")
        get_ipython().system('zip -r $zipfilename $filename')


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


if __name__ == "__main__":
    fig = plot_slab_temperatures(sz_case1)
    fig.savefig(output_folder / "sz_problem_case1_slabTs.png")


class SubductionProblem(SubductionProblem):
    def etadisl(self, v_i, T_i):
        """
        Return a dislocation creep viscosity given a velocity and temperature

        Arguments:
          * v_i - velocity Function
          * T_i - temperature Function

        Returns:
          * eta - viscosity ufl description
        """
        
        # get the mesh
        mesh = v_i.function_space.mesh
        x = ufl.SpatialCoordinate(mesh)
        zero_c = df.fem.Constant(mesh, df.default_scalar_type(0.0))
        deltaztrench_c = df.fem.Constant(mesh, df.default_scalar_type(self.deltaztrench))
        deltazsurface = ufl.operators.MinValue(ufl.operators.MaxValue(self.deltaztrench*(1. - x[0]/max(self.deltaxcoast, np.finfo(df.default_scalar_type).eps)), zero_c), deltaztrench_c)
        z = -(x[1]+deltazsurface)
        
        # dimensional temperature in Kelvin with an adiabat added
        Tdim = utils.nondim_to_K(T_i) + 0.3*z

        # we declare some of the coefficients as dolfinx Constants to prevent the form compiler from
        # optimizing them out of the code due to their small (dimensional) values
        E_c          = df.fem.Constant(mesh, df.default_scalar_type(self.E))
        invetamax_c  = df.fem.Constant(mesh, df.default_scalar_type(self.eta0/self.etamax))
        neII         = (self.nsigma-1.0)/self.nsigma
        invetafact_c = df.fem.Constant(mesh, df.default_scalar_type(self.eta0*(self.e0**neII)/self.Aeta))
        neII_c       = df.fem.Constant(mesh, df.default_scalar_type(neII))
    
        # strain rate
        edot = ufl.sym(ufl.grad(v_i))
        eII  = ufl.sqrt(0.5*ufl.inner(edot, edot))
        # inverse dimensionless dislocation creep viscosity
        invetadisl = invetafact_c*ufl.exp(-E_c/(self.nsigma*self.R*Tdim))*(eII**neII_c)
        # inverse dimensionless effective viscosity
        inveta = invetadisl + invetamax_c
        # "harmonic mean" viscosity (actually twice the harmonic mean)
        return 1./inveta

    def project_dislocationcreep_viscosity(self, p_eta=0, petsc_options=None):
        """
        Project the dislocation creep viscosity to the mesh.

        Keyword Arguments:
          * peta          - finite element degree of viscosity Function (defaults to 0)
          * petsc_options - a dictionary of petsc options to pass to the solver (defaults to mumps)

        Returns:
          * eta_i - the viscosity Function
        """
        if petsc_options is None:
            petsc_options={"ksp_type": "preonly", 
                           "pc_type" : "lu", 
                           "pc_factor_mat_solver_type" : "mumps"}
        # set up the functionspace
        V_eta = df.fem.functionspace(self.mesh, ("DG", p_eta))
        # declare the domain wide Function
        eta_i = df.fem.Function(V_eta)
        eta_i.name = "eta"
        # set it to etamax everywhere (will get overwritten)
        eta_i.x.array[:] = self.etamax/self.eta0
        
        def solve_viscosity(v_i, T_i):
            """
            Solve for the viscosity in subdomains and interpolate it to the parent Function
            """
            mesh = T_i.function_space.mesh
            V_eta = df.fem.functionspace(mesh, ("DG", p_eta))
            eta_a = ufl.TrialFunction(V_eta)
            eta_t = ufl.TestFunction(V_eta)
            Seta = eta_t*eta_a*ufl.dx
            feta = eta_t*self.etadisl(v_i, T_i)*ufl.dx
            problem = df.fem.petsc.LinearProblem(Seta, feta, petsc_options=petsc_options)
            return problem.solve()

        # solve in the wedge
        if self.wedge_rank:
            leta_i = solve_viscosity(self.wedge_vw_i, self.wedge_T_i)
            eta_i.interpolate(leta_i, cells0=np.arange(len(self.wedge_cell_map)), 
                              cells1=self.wedge_cell_map)
        # solve in the slab
        if self.slab_rank:
            leta_i = solve_viscosity(self.slab_vs_i, self.slab_T_i)
            eta_i.interpolate(leta_i, cells0=np.arange(len(self.slab_cell_map)), 
                              cells1=self.slab_cell_map)

        # return the viscosity
        return eta_i
    


class SubductionProblem(SubductionProblem):

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
                r_vec = df.fem.assemble_vector(df.fem.form(r))
                r_vec.scatter_reverse(df.la.InsertMode.add)
                df.fem.set_bc(r_vec.array, bcs, scale=0.0)
                sl = r_vec.index_map.size_local
                r_norm_sq = np.inner(r_vec.array[:sl], r_vec.array[:sl])
            return r_norm_sq
        r_norm_sq  = calc_r_norm_sq(rw, self.bcs_vpw, self.wedge_rank)
        r_norm_sq += calc_r_norm_sq(rs, self.bcs_vps, self.slab_rank)
        r_norm_sq += calc_r_norm_sq(rT, self.bcs_T)
        r = self.comm.allreduce(r_norm_sq, op=MPI.SUM)**0.5
        return r

    def solve_steadystate_dislocationcreep(self, rtol=5.e-6, atol=5.e-9, maxits=50,
                                           petsc_options=None):
        """
        Solve the Stokes problems assuming a dislocation creep rheology.

        Keyword Arguments:
          * rtol          - nonlinear iteration relative tolerance
          * atol          - nonlinear iteration absolute tolerance
          * maxits        - maximum number of nonlinear iterations
          * petsc_options - a dictionary of petsc options to pass to the solver (defaults to mumps)
        """
        if petsc_options is None:
            petsc_options={"ksp_type": "preonly", 
                           "pc_type" : "lu", 
                           "pc_factor_mat_solver_type" : "mumps"}
            
        # first solve the isoviscous problem
        self.solve_stokes_isoviscous(petsc_options=petsc_options)

        # retrieve the temperature forms
        ST, fT = self.temperature_forms_steadystate()
        problem_T = df.fem.petsc.LinearProblem(ST, fT, bcs=self.bcs_T, u=self.T_i,
                                               petsc_options=petsc_options)
        # and solve the temperature problem, given the isoviscous Stokes solution
        self.T_i = problem_T.solve()
        self.update_T_functions()
        
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

        # calculate the initial residual
        r = self.calculate_residual(rw, rs, rT)
        r0 = r
        rrel = r/r0  # 1
        if self.comm.rank == 0:
          print("{:<11} {:<12} {:<17}".format('Iteration','Residual','Relative Residual'))
          print("-"*42)

        # iterate until the residual converges (hopefully)
        it = 0
        if self.comm.rank == 0: print("{:<11} {:<12.6g} {:<12.6g}".format(it, r, rrel,))
        while rrel > rtol and r > atol:
            if it > maxits: break
            # solve for v & p and interpolate it
            if self.wedge_rank: self.wedge_vpw_i = problem_vpw.solve()
            if self.slab_rank:  self.slab_vps_i  = problem_vps.solve()
            self.update_v_functions()
            # solve for T and interpolate it
            self.T_i = problem_T.solve()
            self.update_T_functions()
            # calculate a new residual
            r = self.calculate_residual(rw, rs, rT)
            rrel = r/r0
            it += 1
            if self.comm.rank == 0: print("{:<11} {:<12.6g} {:<12.6g}".format(it, r, rrel,))

        # check for convergence failures
        if it > maxits:
            raise Exception("Nonlinear iteration failed to converge after {} iterations (maxits = {}), r = {} (atol = {}), rrel = {} (rtol = {}).".format(it, \
                                                                                                                                                          maxits, \
                                                                                                                                                          r, \
                                                                                                                                                          rtol, \
                                                                                                                                                          rrel, \
                                                                                                                                                          rtol,))

        # only update the pressure at the end as it is not necessary earlier
        self.update_p_functions()


if __name__ == "__main__":
    io_depth_2 = 154.0
    geom_case2 = create_sz_geometry(slab, resscale, sztype, io_depth_2, extra_width, 
                                    coast_distance, lc_depth, uc_depth)
    sz_case2 = SubductionProblem(geom_case2, A=A, Vs=Vs, sztype=sztype, qs=qs)


if __name__ == "__main__":
    if sz_case2.comm.rank == 0: print("\nSolving steady state flow with dislocation creep rheology...")
    sz_case2.solve_steadystate_dislocationcreep()


if __name__ == "__main__":
    diag_case2 = sz_case2.get_diagnostics()

    if sz_case2.comm.rank == 0:
        print('')
        print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
        print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale, *diag_case2))


if __name__ == "__main__":
    plotter_dis = utils.plot.plot_scalar(sz_case2.T_i, scale=sz_case2.T0, gather=True, cmap='coolwarm', scalar_bar_args={'title': 'Temperature (deg C)', 'bold':True})
    utils.plot.plot_vector_glyphs(sz_case2.vw_i, plotter=plotter_dis, factor=0.1, gather=True, color='k', scale=utils.mps_to_mmpyr(sz_case2.v0))
    utils.plot.plot_vector_glyphs(sz_case2.vs_i, plotter=plotter_dis, factor=0.1, gather=True, color='k', scale=utils.mps_to_mmpyr(sz_case2.v0))
    utils.plot.plot_geometry(sz_case2.geom, plotter=plotter_dis, color='green', width=2)
    utils.plot.plot_couplingdepth(sz_case2.geom.slab_spline, plotter=plotter_dis, render_points_as_spheres=True, point_size=10.0, color='green')
    utils.plot.plot_show(plotter_dis)
    utils.plot.plot_save(plotter_dis, output_folder / "sz_problem_case2_solution.png")


if __name__ == "__main__":
    filename = output_folder / "sz_problem_case2_solution.bp"
    with df.io.VTXWriter(sz_case2.mesh.comm, filename, [sz_case2.T_i, sz_case2.vs_i, sz_case2.vw_i]) as vtx:
        vtx.write(0.0)
    # zip the .bp folder so that it can be downloaded from Jupyter lab
    if "__file__" not in globals():
        zipfilename = filename.with_suffix(".zip")
        get_ipython().system('zip -r $zipfilename $filename')


if __name__ == "__main__":
    eta_i = sz_case2.project_dislocationcreep_viscosity()
    plotter_eta = utils.plot.plot_scalar(eta_i, scale=sz_case2.eta0, log_scale=True, show_edges=True, scalar_bar_args={'title': 'Viscosity (Pa) [log scale]', 'bold':True})
    utils.plot.plot_geometry(sz_case2.geom, plotter=plotter_eta, color='green', width=2)
    utils.plot.plot_couplingdepth(sz_case2.geom.slab_spline, plotter=plotter_eta, render_points_as_spheres=True, point_size=10.0, color='green')
    utils.plot.plot_show(plotter_eta)
    utils.plot.plot_save(plotter_eta, output_folder / "sz_problem_case2_eta.png")


if __name__ == "__main__":
    plot_slab_temperatures(sz_case2)
    fig.savefig(output_folder / "sz_problem_case2_slabTs.png")


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




