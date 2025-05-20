#!/usr/bin/env python
# coding: utf-8

import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


from sz_problems.sz_base import default_params, allsz_params
from sz_problems.sz_slab import create_slab


import geometry as geo
import utils
from mpi4py import MPI
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
if __name__ == "__main__":
    output_folder = pathlib.Path(os.path.join(basedir, "output"))
    output_folder.mkdir(exist_ok=True, parents=True)


def create_sz_geometry(slab, resscale, sztype, io_depth, extra_width, 
                       coast_distance, lc_depth, uc_depth, 
                       **kwargs):
    """
    Function to construct and return a subduction zone geometry object that is used to generate
    a mesh of a subduction zone.  Optional keyword arguments default to parameters in the global 
    default_params dictionary if not specified.
    
    Arguments:
      * slab           - an instance of a slab spline object
      * resscale       - resolution scale factor that multiplies all _res_fact parameters
      * sztype         - either 'continental' or 'oceanic', which determines if an upper crust is included
      * io_depth       - prescribed input/output depth on wedge side
      * extra_width    - extra width at the base of the domain
      * coast_distance - distance from trench to coast
      * lc_depth       - depth of lower crustal boundary ("Moho")
      * uc_depth       - depth of upper crustal boundary [only has an effect if sztype is 'continental']

    Keyword Arguments:
     distances:
      * slab_diag1_depth - starting depth along slab of slab diagnostic region
      * slab_diag2_depth - end depth along slab of slab diagnostic region

     resolutions factors (that get multiplied by the resscale to get the resolutions):
      * io_depth_res_fact        - input/output depth
      * coast_res_fact           - coastal point on top
      * lc_slab_res_fact         - lower crust slab intersection
      * lc_side_res_fact         - lower crust side intersection
      * uc_slab_res_fact         - upper crust slab intersection
      * uc_side_res_fact         - upper crust side intersection
      * slab_diag1_res_fact      - start of slab diagnostic region
      * slab_diag2_res_fact      - end of slab diagnostic region
      * wedge_side_top_res_fact  - top of the wedge side
      * wedge_side_base_res_fact - base of the wedge side
      * slab_side_base_res_fact  - base of the slab side

     surface ids:
      * coast_sid            - coastal slope
      * top_sid              - top of domain
      * fault_sid            - fault
      * lc_side_sid          - side of lower crust
      * lc_base_sid          - base of lower crust
      * uc_side_sid          - side of upper crust
      * uc_base_sid          - base of upper crust
      * slab_sid             - default slab surface id
      * slab_diag_sid        - diagnostic region of slab
      * slab_side_sid        - side of slab
      * wedge_side_sid       - side of wedge
      * upper_wedge_side_sid - side of upper wedge
      * slab_base_sid        - base of slab
      * wedge_base_sid       - base of wedge

     region ids:
      * slab_rid       - slab
      * wedge_rid      - wedge
      * lc_rid         - lower crust
      * uc_rid         - upper crust
      * wedge_diag_rid - wedge diagnostic region

    Returns:
      * geom - subduction zone geometry class instance
    """

    # get input parameters
    # depths
    slab_diag1_depth = kwargs.get('slab_diag1_depth', default_params['slab_diag1_depth'])
    slab_diag2_depth = kwargs.get('slab_diag2_depth', default_params['slab_diag2_depth'])
    
    # resolutions
    io_depth_res = kwargs.get('io_depth_res_fact', default_params['io_depth_res_fact'])*resscale
    coast_res   = kwargs.get('coast_res_fact', default_params['coast_res_fact'])*resscale
    lc_slab_res = kwargs.get('lc_slab_res_fact', default_params['lc_slab_res_fact'])*resscale
    lc_side_res = kwargs.get('lc_side_res_fact', default_params['lc_side_res_fact'])*resscale
    uc_slab_res = kwargs.get('uc_slab_res_fact', default_params['uc_slab_res_fact'])*resscale
    uc_side_res = kwargs.get('uc_side_res_fact', default_params['uc_side_res_fact'])*resscale
    slab_diag1_res = kwargs.get('slab_diag1_res_fact', default_params['slab_diag1_res_fact'])*resscale
    slab_diag2_res = kwargs.get('slab_diag2_res_fact', default_params['slab_diag2_res_fact'])*resscale
    wedge_side_top_res  = kwargs.get('wedge_side_top_res_fact', default_params['wedge_side_top_res_fact'])*resscale
    wedge_side_base_res = kwargs.get('wedge_side_base_res_fact', default_params['wedge_side_base_res_fact'])*resscale
    slab_side_base_res  = kwargs.get('slab_side_base_res_fact', default_params['slab_side_base_res_fact'])*resscale

    # surface ids
    coast_sid = kwargs.get('coast_sid', default_params['coast_sid'])
    top_sid   = kwargs.get('top_sid', default_params['top_sid'])
    fault_sid = kwargs.get('fault_sid', default_params['fault_sid'])
    lc_side_sid = kwargs.get('lc_side_sid', default_params['lc_side_sid'])
    lc_base_sid = kwargs.get('lc_base_sid', default_params['lc_base_sid'])
    uc_side_sid = kwargs.get('uc_side_sid', default_params['uc_side_sid'])
    uc_base_sid = kwargs.get('uc_base_sid', default_params['uc_base_sid'])
    slab_sid    = kwargs.get('slab_sid', default_params['slab_sid'])
    slab_diag_sid  = kwargs.get('slab_diag_sid', default_params['slab_diag_sid'])
    slab_side_sid  = kwargs.get('slab_side_sid', default_params['slab_side_sid'])
    wedge_side_sid = kwargs.get('wedge_side_sid', default_params['wedge_side_sid'])
    slab_base_sid  = kwargs.get('slab_base_sid', default_params['slab_base_sid'])
    wedge_base_sid = kwargs.get('wedge_base_sid', default_params['wedge_base_sid'])
    upper_wedge_side_sid = kwargs.get('upper_wedge_side_sid', default_params['upper_wedge_side_sid'])
    
    # region ids
    slab_rid       = kwargs.get('slab_rid', default_params['slab_rid'])
    wedge_rid      = kwargs.get('wedge_rid', default_params['wedge_rid'])
    lc_rid         = kwargs.get('lc_rid', default_params['lc_rid'])
    uc_rid         = kwargs.get('uc_rid', default_params['uc_rid'])
    wedge_diag_rid = kwargs.get('wedge_diag_rid', default_params['wedge_diag_rid'])
    
    assert sztype in ['continental', 'oceanic']
    

    # pass the slab object into the SubductionGeometry class to construct the geometry
    # around it
    geom = geo.SubductionGeometry(slab, 
                                  coast_distance=coast_distance, 
                                  extra_width=extra_width, 
                                  slab_side_sid=slab_side_sid, 
                                  wedge_side_sid=wedge_side_sid, 
                                  slab_base_sid=slab_base_sid, 
                                  wedge_base_sid=wedge_base_sid, 
                                  coast_sid=coast_sid, 
                                  top_sid=top_sid, 
                                  slab_rid=slab_rid, 
                                  wedge_rid=wedge_rid, 
                                  coast_res=coast_res, 
                                  slab_side_base_res=slab_side_base_res, 
                                  wedge_side_top_res=wedge_side_top_res, 
                                  wedge_side_base_res=wedge_side_base_res)
    
    if sztype=='oceanic':
        # add a single crust layer
        # (using the lc parameters & ignoring the uc ones)
        geom.addcrustlayer(lc_depth, "Crust",
                           sid=lc_base_sid, rid=lc_rid,
                           slab_res=lc_slab_res,
                           side_res=lc_side_res,
                           slab_sid=fault_sid,
                           side_sid=lc_side_sid)
    else:
        # add a lower crust
        geom.addcrustlayer(lc_depth, "Crust", 
                           sid=lc_base_sid, rid=lc_rid,
                           slab_res=lc_slab_res,
                           side_res=lc_side_res,
                           slab_sid=fault_sid,
                           side_sid=lc_side_sid)
        
        # add an upper crust
        geom.addcrustlayer(uc_depth, "UpperCrust", 
                           sid=uc_base_sid, rid=uc_rid,
                           slab_res=uc_slab_res,
                           side_res=uc_side_res,
                           slab_sid=fault_sid,
                           side_sid=uc_side_sid)
    
    # add the pre-defined in-out point on the wedge side
    geom.addwedgesidepoint(io_depth, "WedgeSide::InOut", line_name="UpperWedgeSide", 
                           res=io_depth_res, 
                           sid=upper_wedge_side_sid)
    
    # add wedge dividers for the diagnostics
    geom.addwedgedivider(slab_diag1_depth, "ColdCorner", 
                         slab_res=slab_diag2_res, 
                         top_res=slab_diag2_res,
                         rid=wedge_rid, 
                         slab_sid=slab_sid)
    
    # add wedge dividers for the diagnostics
    geom.addwedgedivider(slab_diag2_depth, "WedgeFocused", 
                         slab_res=slab_diag1_res, 
                         top_res=slab_diag1_res,
                         rid=wedge_diag_rid, 
                         slab_sid=slab_diag_sid)

    # return the geometry object
    return geom


if __name__ == "__main__":
    resscale = 5.0
    # points in slab (just linear)
    xs = [0.0, 140.0, 240.0, 400.0]
    ys = [0.0, -70.0, -120.0, -200.0]
    lc_depth = 40
    slab = create_slab(xs, ys, resscale, lc_depth)


if __name__ == "__main__":
    coast_distance = 0
    extra_width = 0
    uc_depth = 15
    lc_depth = 40
    sztype = 'continental'


if __name__ == "__main__":
    io_depth_1 = 139


if __name__ == "__main__":
    geom = create_sz_geometry(slab, resscale, sztype, io_depth_1, extra_width, 
                              coast_distance, lc_depth, uc_depth)


if __name__ == "__main__":
    if MPI.COMM_WORLD.rank == 0:
        fig = geom.plot(label_sids=False, label_rids=False)
        fig.savefig(output_folder / "sz_geometry_benchmark.png")


if __name__ == "__main__":
    mesh, cell_tags, facet_tags = geom.generatemesh()


if __name__ == "__main__":
    plotter_mesh = utils.plot_mesh(mesh, tags=cell_tags, gather=True, show_edges=True, line_width=1)
    utils.plot_geometry(geom, plotter=plotter_mesh, color='green', width=2)
    utils.plot_couplingdepth(slab, plotter=plotter_mesh, render_points_as_spheres=True, point_size=10.0, color='green')
    utils.plot_show(plotter_mesh)
    utils.plot_save(plotter_mesh, output_folder / "sz_geometry_benchmark_mesh.png")


if __name__ == "__main__":
    filename = output_folder / "sz_geometry_benchmark"
    geom.writegeofile(str(filename.with_suffix('.geo_unrolled')))


if __name__ == "__main__":
    if MPI.COMM_WORLD.rank == 0:
        szdict_ak = allsz_params['01_Alaska_Peninsula']
        print("{:<35} {:<10}".format('Key','Value'))
        print("-"*100)
        for k, v in szdict_ak.items():
            if v is not None: print("{:<35} {}".format(k, v))


if __name__ == "__main__":
    resscale = 5.0
    slab_ak = create_slab(szdict_ak['xs'], szdict_ak['ys'], resscale, szdict_ak['lc_depth'])
    geom_ak = create_sz_geometry(slab_ak, resscale, szdict_ak['sztype'], szdict_ak['io_depth'], szdict_ak['extra_width'], 
                                 szdict_ak['coast_distance'], szdict_ak['lc_depth'], szdict_ak['uc_depth'])


if __name__ == "__main__":
    if MPI.COMM_WORLD.rank == 0:
        fig_ak = geom_ak.plot(label_sids=False, label_rids=False)
        fig_ak.savefig(output_folder / "sz_geometry_ak.png")


if __name__ == "__main__":
    mesh_ak, cell_tags_ak, facet_tags_ak = geom_ak.generatemesh()
    


if __name__ == "__main__":
    plotter_mesh_ak = utils.plot_mesh(mesh_ak, tags=cell_tags_ak, gather=True, show_edges=True, line_width=1)
    utils.plot_geometry(geom_ak, plotter=plotter_mesh_ak, color='green', width=2)
    utils.plot_couplingdepth(slab_ak, plotter=plotter_mesh_ak, render_points_as_spheres=True, point_size=10.0, color='green')
    utils.plot_show(plotter_mesh_ak)
    utils.plot_save(plotter_mesh_ak, output_folder / "sz_geometry_ak_mesh.png")


if __name__ == "__main__":
    filename = output_folder / "sz_geometry_ak"
    geom_ak.writegeofile(str(filename.with_suffix('.geo_unrolled')))


if __name__ == "__main__":
    if MPI.COMM_WORLD.rank == 0:
        szdict_ant = allsz_params['19_N_Antilles']
        print("{:<35} {:<10}".format('Key','Value'))
        print("-"*100)
        for k, v in szdict_ant.items():
            if v is not None: print("{:<35} {}".format(k, v))


if __name__ == "__main__":
    resscale = 5.0
    slab_ant = create_slab(szdict_ant['xs'], szdict_ant['ys'], resscale, szdict_ant['lc_depth'])
    geom_ant = create_sz_geometry(slab_ant, resscale, szdict_ant['sztype'], szdict_ant['io_depth'], szdict_ant['extra_width'], 
                                  szdict_ant['coast_distance'], szdict_ant['lc_depth'], szdict_ant['uc_depth'])


if __name__ == "__main__":
    if MPI.COMM_WORLD.rank == 0:
        fig_ant = geom_ant.plot(label_sids=False, label_rids=False)
        fig_ant.savefig(output_folder / "sz_geometry_ant.png")


if __name__ == "__main__":
    mesh_ant, cell_tags_ant, facet_tags_ant = geom_ant.generatemesh()


if __name__ == "__main__":
    plotter_mesh_ant = utils.plot_mesh(mesh_ant, tags=cell_tags_ant, gather=True, show_edges=True, line_width=1)
    utils.plot_geometry(geom_ant, plotter=plotter_mesh_ant, color='green', width=2)
    utils.plot_couplingdepth(slab_ant, plotter=plotter_mesh_ant, render_points_as_spheres=True, point_size=10.0, color='green')
    utils.plot_show(plotter_mesh_ant)
    utils.plot_save(plotter_mesh_ant, output_folder / "sz_geometry_ant_mesh.png")


if __name__ == "__main__":
    filename = output_folder / "sz_geometry_ant"
    geom_ant.writegeofile(str(filename.with_suffix('.geo_unrolled')))


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/sz_problems --NbConvertApp.output_base=sz_geometry 3.1d_sz_geometry.ipynb')




