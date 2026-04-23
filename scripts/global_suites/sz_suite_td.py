def run(name, resscale, cfl, verbosity, petsc_options_s=None, petsc_options_T=None, output=None):
    """
    Run the simulation.
    """
    from fenics_sz.sz_problems.sz_params import allsz_params
    from fenics_sz.sz_problems.sz_slab import create_slab
    from fenics_sz.sz_problems.sz_geometry import create_sz_geometry
    from fenics_sz.sz_problems.sz_tdep_dislcreep import TDDislSubductionProblem
    from mpi4py import MPI
    import numpy as np

    szdict = allsz_params[name]
    if MPI.COMM_WORLD.rank == 0:
        print("{}:".format(name))
        print("{:<20} {:<10}".format('Key','Value'))
        print("-"*85)
        for k, v in allsz_params[name].items():
            if v is not None and k not in ['z0', 'z15']: print("{:<20} {}".format(k, v))
    
    slab = create_slab(szdict['xs'], szdict['ys'], resscale, szdict['lc_depth'])
    geom = create_sz_geometry(slab, resscale, szdict['sztype'], szdict['io_depth'], szdict['extra_width'], 
                             szdict['coast_distance'], szdict['lc_depth'], szdict['uc_depth'])
    
    # Select the timestep based on the approximate target Courant number
    dt = cfl*resscale/szdict['Vs']
    # Reduce the timestep to get an integer number of timesteps
    dt = szdict['As']/np.ceil(szdict['As']/dt)
    sz = TDDislSubductionProblem(geom, **szdict)

    sz.solve(szdict['As'], dt, theta=0.5, rtol=1.e-1, verbosity=verbosity,
             petsc_options_s=petsc_options_s, petsc_options_T=petsc_options_T)

    print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format(' ', 'resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
    print ('-'*(7*12))
    print('{:<12} {:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(name, resscale, *sz.get_diagnostics().values()))  
    print ('-'*(7*12))
    
    if output is not None:
        import dolfinx as df
        with df.io.VTXWriter(sz.V_T.mesh.comm, output, [sz.T_i, sz.vw_i, sz.vs_i]) as vtx:
            vtx.write(0.0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( \
                            description="""
                            Example script to solve a steady state subduction zone thermal model from the global suite.
                            """)
    parser.add_argument('name', metavar='name', type=str, 
                        help='specifiy the sz name to run (can be any of the global suite, e.g. "02_Alaska")')
    parser.add_argument('minres', metavar='minres', type=float, 
                        help='specifiy the minimum mesh spacing, e.g. 1 for 1km minimum mesh spacing')
    parser.add_argument('cfl', metavar='cfl', type=float, 
                        help='specifiy the target Courant number to select the timestep, e.g. 1 for a target Courant number of 1')
    parser.add_argument('-v', '--verbosity', metavar='verbosity', type=int, required=False, default=1,
                        help='specifiy the verbosity of the time-loop, higher values are more verbose, default is 1')
    parser.add_argument('-o', '--output', metavar='output', type=str, required=False, default=None,
                        help='specify an output .bp file for the solution (default is no output, only affects runs without -t specified)')
    args, unknown = parser.parse_known_args()

    run(args.name, args.minres, args.cfl, args.verbosity, output=args.output)