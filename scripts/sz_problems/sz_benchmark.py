def run(case, resscale, petsc_options_s=None, petsc_options_T=None, output=None):
    """
    Run the simulation.
    """
    import numpy as np
    if case == 1:
        from fenics_sz.sz_problems.sz_benchmark import solve_benchmark_case1 as solve_benchmark
        from fenics_sz.sz_problems.sz_benchmark import values_wvk_case1 as values_wvk
    elif case == 2:
        from fenics_sz.sz_problems.sz_benchmark import solve_benchmark_case2 as solve_benchmark
        from fenics_sz.sz_problems.sz_benchmark import values_wvk_case2 as values_wvk
    else:
        raise ValueError(f"Unknown benchmark case {case}")

    sz = solve_benchmark(resscale, petsc_options_s=petsc_options_s, petsc_options_T=petsc_options_T)

    print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format(' ', 'resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
    print ('-'*(7*12))
    print('{:<12} {:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format('simulation: ', resscale, *sz.get_diagnostics().values()))  
    print ('-'*(7*12))
    i = np.asarray([abs(values['resscale']-resscale) for values in values_wvk]).argmin()
    print('{:<12} {:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format('benchmark:', *values_wvk[i].values()))
    
    if output is not None:
        import dolfinx as df
        with df.io.VTXWriter(sz.V_T.mesh.comm, output, [sz.T_i, sz.vw_i, sz.vs_i]) as vtx:
            vtx.write(0.0)


def profile(case, resscale, number, petsc_options_s=None, petsc_options_T=None):
    """
    Profile the simulation.
    """
    from fenics_sz.utils.ipp import profile_local
    from mpi4py import MPI

    function = ''
    if case == 1:
        function = 'solve_benchmark_case1'
    elif case == 2:
        function = 'solve_benchmark_case2'
    else:
        raise ValueError(f"Unknown benchmark case {case}")

    labels = [
          'Assemble Temperature', 'Assemble Stokes',
          'Solve Temperature', 'Solve Stokes'
         ]

    extra_parallel_diagnostics = lambda sz: sz.get_diagnostics()
    
    maxts, maxmumpsts, ediag  = profile_local(labels, '', 
                                        'fenics_sz.sz_problems.sz_benchmark', function,
                                            resscale, number=number, 
                                            extra_diagnostics_func=extra_parallel_diagnostics,
                                            petsc_options_s=petsc_options_s,
                                            petsc_options_T=petsc_options_T)

    if MPI.COMM_WORLD.rank == 0:
        maxlen = max([len(label) for label in labels])
        maxlen = max([maxlen]+[len(k)-4 for k in ediag.keys() if k.endswith('_avg')])
        print('='*(maxlen+2+12))
        print("{0:<{1}}".format("nproc", maxlen+2)+"{:<12d}".format(MPI.COMM_WORLD.size))
        for l, label in enumerate(labels):
            print("{0:<{1}}".format(label, maxlen+2)+"{:<12g}".format(maxts[l]))
        print('='*(maxlen+2+12))
        for k in ediag.keys():
            if k.endswith('_avg'):
                print("{0:<{1}}".format(k[:-4], maxlen+2)+"{:<12g}".format(ediag[k]))
        print('='*(maxlen+2+12))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( \
                            description="""
                            Example script to solve the subduction zone benchmarks.
                            """)
    parser.add_argument('case', metavar='case', type=int, 
                        help='specifiy the benchmark case to run (either 1 or 2)')
    parser.add_argument('minres', metavar='minres', type=float, 
                        help='specifiy the minimum mesh spacing')
    parser.add_argument('-o', '--output', metavar='output', type=str, required=False, default=None,
                        help='specify an output .bp file for the solution (default is no output, only affects runs without -t specified)')
    parser.add_argument('-t', '--time', metavar='nr', type=int, nargs='?', required=False,
                        const=1, default=None,
                        help='time running the simulation nr times (nr defaults to 1 if not provided)')
    args, unknown = parser.parse_known_args()

    if args.time is not None:
        profile(args.case, args.minres, args.time)
    else:
        run(args.case, args.minres, output=args.output)