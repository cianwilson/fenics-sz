from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.ticker as ticker
import ipyparallel as ipp
import logging

def run_local(path, module_name, func_name, *func_args, **func_kwargs):
    """
    A python function that runs a function using local imports to allow it to run 
    on multiple processes.  Essentially acts like a decorator that imports the function to
    be run locally.
    Parameters:
    * path        - system path where the module can be found
    * module_name - name of module where function can be found
    * func_name   - name of function to profile
    * func_args   - arguments to pass to the function
    * func_kwargs - keyword arguments to pass to the function
    Returns:
    * output      - output of function
    """
    import sys
    sys.path.append(path)
    import importlib
    module = importlib.import_module(module_name)
    # get the function
    func = getattr(module, func_name)

    output = func(*func_args, **func_kwargs)

    return output

# this needs to be decorated as interactive to pass this module's namespace 
# in rather than the system globals (which doesn't include this module on 
# the remote process)
ipp_run_local = ipp.interactive(run_local)

def run_parallel(nprocs, *args, **kwargs):
    """
    A python function that runs a function in parallel on a number of proceses.
    Parameters:
    * nprocs          - list of the number of processes on which to run
    * args            - additional arguments to pass to run_local function
        * path        - system path where the module can be found
        * module_name - name of module where function can be found
        * func_name   - name of function to profile
        * func_args   - arguments to pass to the function
    * kwargs          - additional keyword arguments to pass to profile_local function
        * func_kwargs - keyword arguments to pass to the function
    Returns:
    * outputs         - list of outputs of the functions
    """
    outputs = []
    for nproc in nprocs:
        cluster = ipp.Cluster(engine_launcher_class="mpi", n=nproc, log_level=logging.FATAL)
        rc = cluster.start_and_connect_sync()
        view = rc[:]

        outputs.append(view.remote(block=True)(ipp_run_local)(*args, **kwargs)[0])

        rc.shutdown(hub=True)

    return outputs

def profile_local(labels, path, module_name, func_name, *func_args, number=1, include_mumps_times=False, extra_diagnostics_func=None, **func_kwargs):
    """
    A python function that profiles a function using local imports to allow it to run 
    on multiple processes.  Essentially acts like a decorator that imports the function to
    be profiled internally.
    Parameters:
    * labels      - labels to extract from dolfinx timing 
    * path        - system path where the module can be found
    * module_name - name of module where function can be found
    * func_name   - name of function to profile
    * func_args   - arguments to pass to the function
    * number      - number of runs of function to time
    * include_mumps_times - extract the analysis, factorization 
                    and solve times from the petsc log (requires 
                    mat_mumps_icntl_4 >= 2
                    otherwise these steps will return 0 times)
    * extra_diagnostics_func - function to return extra diagnostics in a dictionary
    * func_kwargs - keyword arguments to pass to the function
    Returns:
    * maxtimes    - computation walltimes corresponding to each label
    """
    # import necessary modules
    import dolfinx as df
    import numpy as np
    from mpi4py import MPI
    from wurlitzer import pipes, STDOUT
    from io import StringIO
    import re
    import sys
    sys.path.append(path)
    import importlib
    module = importlib.import_module(module_name)
    # get the function
    func = getattr(module, func_name)

    # we deal with mumps timings by forcing increased logging and accumulating the timings in a dictionary as we go
    mumpstimes = {'MUMPS analysis':0.0, 
                  'MUMPS factorization':0.0, 
                  'MUMPS solve':0.0}

    extra_diagnostics = dict()

    # run the function the specified number of times
    for n in range(number):
        stdout = StringIO()
        with pipes(stdout=stdout, stderr=STDOUT):
            output = func(*func_args, **func_kwargs)
        if include_mumps_times:
            # collect the numps timings
            for step in ['analysis', 'factorization', 'solve']:
                for match in re.finditer(f'Elapsed time in {step} driver', stdout.getvalue()):
                    i0 = match.start()
                    if i0 > -1:
                        ie = stdout.getvalue()[i0:].find('\n')
                        mumpstimes[f'MUMPS {step}'] += float(stdout.getvalue()[i0:i0+ie].split()[-1])
        if extra_diagnostics_func is not None:
            tmp_extra_diagnostics = extra_diagnostics_func(output)
            for k, v in tmp_extra_diagnostics.items():
                extra_diagnostics[k] = extra_diagnostics.get(k, 0.0) + v
        stdout.close()


    vals_to_max = []
    vals_to_min = []
    vals_to_sum = []
    
    # extract and return the computation times from dolfinx
    for label in labels:
        try:
            vals_to_max.append(df.common.timing(label)[1]/number)
        except RuntimeError:
            vals_to_max.append(0.0)
    # extract mumps times
    if include_mumps_times:
        for label, time in mumpstimes.items():
            vals_to_max.append(time/number)
    # extract extract diagnostics
    if extra_diagnostics_func is not None:
        for k, v in extra_diagnostics.items():
            vals_to_max.append(v/number)
            vals_to_min.append(v/number)
            vals_to_sum.append(v/number)
            
    vals_to_max = np.array(vals_to_max)
    vals_to_min = np.array(vals_to_min)
    vals_to_sum = np.array(vals_to_sum)

    def reduce_vals(vals_to_, mpi_op):
        vals = None
        if MPI.COMM_WORLD.rank==0: vals = np.zeros_like(vals_to_)
        MPI.COMM_WORLD.Reduce(vals_to_, vals, op=mpi_op)
        return vals

    maxvals = reduce_vals(vals_to_max, MPI.MAX)
    minvals = reduce_vals(vals_to_min, MPI.MIN)
    sumvals = reduce_vals(vals_to_sum, MPI.SUM)

    maxtimes = None
    maxmumpstimes = None
    extradiagout = None
    if MPI.COMM_WORLD.rank==0: 
        maxtimes = maxvals.tolist()[:len(labels)]
        o = len(labels)
        if include_mumps_times:
            maxmumpstimes = dict()
            for i, k in enumerate(mumpstimes.keys()):
                maxmumpstimes[k] = maxvals[o+i].tolist()
            o += len(mumpstimes)
        if extra_diagnostics_func is not None:
            extradiagout = dict()
            for i, k in enumerate(extra_diagnostics.keys()):
                extradiagout[k+'_max'] = maxvals[o+i].tolist()
                extradiagout[k+'_min'] = minvals[i].tolist()
                sumval = sumvals[i].tolist()
                extradiagout[k+'_sum'] = sumval
                extradiagout[k+'_avg'] = sumval/MPI.COMM_WORLD.size

    return maxtimes, maxmumpstimes, extradiagout

# this needs to be decorated as interactive to pass this module's namespace 
# in rather than the system globals (which doesn't include this module on 
# the remote process)
ipp_profile_local = ipp.interactive(profile_local)

def profile_parallel(nprocs, labels, *args, output_filename=None, number=1, include_mumps_times=False, extra_diagnostics_func=None, **kwargs):
    """
    A python function that runs a function over a series of number of proceses and prints and returns the timings.
    Parameters:
    * nprocs          - list of the number of processes on which to run
    * labels          - labels to extract from dolfinx timing
    * args            - additional arguments to pass to profile_local function
        * path        - system path where the module can be found
        * module_name - name of module where function can be found
        * func_name   - name of function to profile
        * func_args   - arguments to pass to the function
    * output_filename - filename for plot (defaults to no output)
    * include_mumps_times - extract the analysis, factorization 
                        and solve times from the petsc log (requires 
                        mat_mumps_icntl_4 >= 2
                        otherwise these steps will return 0 times, 
                        defaults to False)
    * extra_diagnostics_func - function to return extra diagnostics in a dictionary (default to None)
    * number          - number of runs of function to time (defaults to 1)
    * kwargs          - additional keyword arguments to pass to profile_local function
        * func_kwargs - keyword arguments to pass to the function
    Returns:
    * maxtimes        - computation walltimes corresponding to each label and number of processes
    * maxmumpstimes   - computation walltimes corresponding to each stage of mumps
    * 
    """
    maxtimes = []
    maxmumpstimes = []
    extradiagout = []
    for nproc in nprocs:
        cluster = ipp.Cluster(engine_launcher_class="mpi", n=nproc, log_level=logging.FATAL)
        rc = cluster.start_and_connect_sync()
        view = rc[:]

        nmaxtimes, nmaxmumpstimes, nextradiagout = view.remote(block=True)(ipp_profile_local)(labels, *args, number=number, include_mumps_times=include_mumps_times, extra_diagnostics_func=extra_diagnostics_func, **kwargs)[0]
        maxtimes.append(nmaxtimes)
        maxmumpstimes.append(nmaxmumpstimes)
        extradiagout.append(nextradiagout)

        rc.shutdown(hub=True)

    maxlen = max([len(label) for label in labels])
    if include_mumps_times: 
        maxlen = max([maxlen]+[len(k) for k in maxmumpstimes[0].keys()])
    if extra_diagnostics_func is not None:
        maxlen = max([maxlen]+[len(k) for k in extradiagout[0].keys()])
    print('=========================')
    print("{0:<{1}}".format(" ", maxlen+2)+"".join(["{:<12d}".format(nproc) for nproc in nprocs]))
    for l, label in enumerate(labels):
        print("{0:<{1}}".format(label, maxlen+2)+"".join(["{:<12g}".format(t[l]) for t in maxtimes]))
    print('=========================')
    if include_mumps_times:
        for k in maxmumpstimes[0].keys():
            print("{0:<{1}}".format(k, maxlen+2)+"".join(["{:<12g}".format(m[k]) for m in maxmumpstimes]))
        print('=========================')
    if extra_diagnostics_func is not None:
        for k in extradiagout[0].keys():
            print("{0:<{1}}".format(k, maxlen+2)+"".join(["{:<12g}".format(e[k]) for e in extradiagout]))
        print('=========================')


    if MPI.COMM_WORLD.rank == 0:
        fig, (ax, ax_r) = pl.subplots(nrows=2, figsize=[6.4,9.6], sharex=True)
        for l, label in enumerate(labels):
            ax.plot(nprocs, [t[l] for t in maxtimes], 'o-', label=label)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylabel('wall time (s)')
        ax.grid()
        ax.legend()

        ax_r.plot(nprocs, [nproc/nprocs[0] for nproc in nprocs], 'k--', label='Ideal')
        for l, label in enumerate(labels):
            ax_r.plot(nprocs, maxtimes[0][l]/np.asarray([t[l] for t in maxtimes]), 'o-', label=label)
        ax_r.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_r.set_xlabel('number processors')
        ax_r.set_ylabel('speed up')
        ax_r.grid()
        ax_r.legend()

        # Write profiling to disk
        if output_filename is not None:
            fig.savefig(output_filename)

            print("***********  profiling figure in "+str(output_filename))

    return maxtimes, maxmumpstimes, extradiagout