#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


from sz_problems.sz_params import default_params, allsz_params


import geometry as geo
from mpi4py import MPI
import matplotlib.pyplot as pl
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


def create_slab(xs, ys, resscale, lc_depth, 
                **kwargs):
    """
    Function to construct and return a spline object that is used to describe a subducting slab
    in a kinematic-slab model of a subduction zone.  Optional keyword arguments default to parameters 
    in the global default_params dictionary if not specified.
    
    Arguments:
      * xs             - list of x points in slab spline
      * ys             - list of y points in slab spline (must be the same length as xs)
      * resscale       - resolution scale factor that multiplies all _res_fact parameters
      * lc_depth       - depth of lower crustal boundary ("Moho")

    Keyword Arguments:
     distances:
      * slab_diag1_depth - starting depth of slab diagnostic region
      * slab_diag2_depth - end depth of slab diagnostic region
      * coupling_depth   - partial coupling depth on slab
      * coupling_depth_range - depth range over which slab goes from being partially to fully coupled
      * slab_det_depth       - detector depth on slab

     resolutions factors (that get multiplied by the resscale to get the resolutions):
      * partial_coupling_depth_res_fact - partial coupling depth on slab
      * full_coupling_depth_res_fact    - full coupling depth on slab

     surface ids:
      * fault_sid            - fault
      * slab_sid             - default slab surface id
      * slab_diag_sid        - diagnostic region of slab

    Returns:
      * slab - subduction zone slab spline instance
    """
    
    # get input parameters
    # depths
    slab_diag1_depth       = kwargs.get('slab_diag1_depth', default_params['slab_diag1_depth'])
    slab_diag2_depth       = kwargs.get('slab_diag2_depth', default_params['slab_diag2_depth'])
    partial_coupling_depth = kwargs.get('coupling_depth', default_params['coupling_depth'])
    full_coupling_depth    = partial_coupling_depth + kwargs.get('coupling_depth_range', default_params['coupling_depth_range'])
    slab_det_depth         = kwargs.get('slab_det_depth', default_params['slab_det_depth'])
    
    # resolutions
    partial_coupling_depth_res = kwargs.get('partial_coupling_depth_res_fact', default_params['partial_coupling_depth_res_fact'])*resscale
    full_coupling_depth_res    = kwargs.get('full_coupling_depth_res_fact', default_params['full_coupling_depth_res_fact'])*resscale

    # surface ids
    fault_sid      = kwargs.get('fault_sid', default_params['fault_sid'])
    slab_sid       = kwargs.get('slab_sid', default_params['slab_sid'])
    slab_diag_sid  = kwargs.get('slab_diag_sid', default_params['slab_diag_sid'])
       
    # set up resolutions along the slab depending on depth
    # high resolution at shallow depths, lower resolution below the "diagnostic"
    # region required in the benchmark case
    # FIXME: these are currently hard-coded relative to the resolutions specified at the partial and full coupling
    # depths for simplicity but could be separate parameters
    res = [partial_coupling_depth_res if y >= -slab_diag2_depth else 3*full_coupling_depth_res for y in ys]
    
    # set up the surface ids for the slab depending on depth
    # above the "Moho" use fault_sid
    # in the diagnostic region use the slab_diag_sid
    # everywhere else use the default slab_sid
    sids = []
    for y in ys[1:]:
        if y >= -lc_depth: 
            sid = fault_sid
        elif y >= -slab_diag1_depth:
            sid = slab_sid
        elif y >= -slab_diag2_depth:
            sid = slab_diag_sid
        else:
            sid = slab_sid
        sids.append(sid)
    
    # set up the slab spline object
    slab = geo.SlabSpline(xs, ys, res=res, sid=sids, name="Slab")

    assert(full_coupling_depth > partial_coupling_depth)
    # adding the coupling depths may or may not be necessary
    # depending on if they were included in the slab spline data already or not
    # the slab class should ignore them if they aren't necessary
    slab.addpoint(partial_coupling_depth, "Slab::PartialCouplingDepth", 
                  res=partial_coupling_depth_res, 
                  sid=slab_diag_sid)
    slab.addpoint(full_coupling_depth, "Slab::FullCouplingDepth", 
                  res=full_coupling_depth_res, 
                  sid=slab_diag_sid)
    # add the slab detector point
    slab.addpoint(slab_det_depth, "Slab::DetectorPoint", 
                  res=full_coupling_depth_res,
                  sid=slab_diag_sid)

    # and return it
    return slab


def plot_slab(slab):
    """
    A python function to plot the given slab and return a matplotlib figure.
    """
    interpx = [curve.points[0].x for curve in slab.interpcurves]+[slab.interpcurves[-1].points[1].x]
    interpy = [curve.points[0].y for curve in slab.interpcurves]+[slab.interpcurves[-1].points[1].y]
    fig = pl.figure()
    ax = fig.gca()
    ax.plot(interpx, interpy)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_aspect('equal')
    ax.set_title('Slab Geometry')
    return fig




