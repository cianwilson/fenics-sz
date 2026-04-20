def run(ne, p, petsc_options=None, output=None):
    """
    Run the simulation.
    """
    from fenics_sz.background.poisson_2d import solve_poisson_2d

    T = solve_poisson_2d(ne, p=p, petsc_options=petsc_options)

    if output is not None:
        import dolfinx as df
        with df.io.VTXWriter(T.function_space.mesh.comm, output, [T]) as vtx:
            vtx.write(0.0)


def profile(ne, p, number, petsc_options=None):
    """
    Profile the simulation.
    """
    from fenics_sz.utils.ipp import profile_local
    from mpi4py import MPI

    include_mumps_times = petsc_options is not None and \
                          petsc_options.get('pc_factor_mat_solver_type', 'unknown') == 'mumps'

    def extra_parallel_diagnostics(T):
        diag = dict()
        diag['ndofs']   = T.function_space.dofmap.index_map.size_local
        diag['nghosts'] = T.function_space.dofmap.index_map.num_ghosts
        return diag

    labels = ['Mesh', 'Function spaces', 'Assemble', 'Solve']
    maxts, maxmumpsts, ediag  = profile_local(labels, '', 
                                        'fenics_sz.background.poisson_2d', 'solve_poisson_2d',
                                            ne, p, number=number, include_mumps_times=include_mumps_times, 
                                            extra_diagnostics_func=extra_parallel_diagnostics,
                                            petsc_options=petsc_options)

    if MPI.COMM_WORLD.rank == 0:
        maxlen = max([len(label) for label in labels])
        if include_mumps_times: 
            maxlen = max([maxlen]+[len(k) for k in maxmumpsts.keys()])
        maxlen = max([maxlen]+[len(k) for k in ediag.keys()])
        print('=========================')
        print("{0:<{1}}".format("nproc", maxlen+2)+"{:<12d}".format(MPI.COMM_WORLD.size))
        for l, label in enumerate(labels):
            print("{0:<{1}}".format(label, maxlen+2)+"{:<12g}".format(maxts[l]))
        print('=========================')
        if include_mumps_times:
            for k in maxmumpsts.keys():
                print("{0:<{1}}".format(k, maxlen+2)+"{:<12g}".format(maxmumpsts[k]))
            print('=========================')
        for k in ediag.keys():
            print("{0:<{1}}".format(k, maxlen+2)+"{:<12g}".format(ediag[k]))
        print('=========================')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( \
                            description="""
                            Example script using  solve_poisson_2d.  Additional arguments not described below are interpretted as petsc options.

                            For example:
                            
                            `mpiexec -np 8 python3 poisson_2d.py -ne 1024 -p 2 -t 5 -- -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -mat_mumps_icntl_4 2`
                            
                            will time the execution of solve_poisson_2d 5 times on 8 processes with 1024*1024*2 elements and P2 polynomials with the petsc options (MUMPS LU solver) specified after the -- flag.
                            """)
    parser.add_argument('ne', metavar='ne', type=int, 
                        help='specifiy the number of elements in each dimension of the mesh')
    parser.add_argument('-p', metavar='p', type=int, required=False, default=1, 
                        help='specifiy the polynomial degree of the function space')
    parser.add_argument('-o', '--output', metavar='output', type=str, required=False, default=None,
                        help='specify an output .bp file for the solution (default is no output, only affects runs without -t specified)')
    parser.add_argument('-t', '--time', metavar='nr', type=int, nargs='?', required=False,
                        const=1, default=None,
                        help='time running the simulation nr times (nr defaults to 1 if not provided)')
    args, unknown = parser.parse_known_args()

    petsc_options = None
    if len(unknown) > 0:
        petsc_options = dict()
        i = 0
        while i < len(unknown):
            assert(unknown[i].startswith('-'))
            k = unknown[i][1:]
            v = None
            i += 1
            if k == '-': continue
            if i < len(unknown) and not unknown[i].startswith('-'):
                v = unknown[i]
                i += 1
            petsc_options[k] = v

    if args.time is not None:
        profile(args.ne, args.p, args.time,
                petsc_options=petsc_options)
    else:
        run(args.ne, args.p, 
            petsc_options=petsc_options, 
            output=args.output)