from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.ticker as ticker
import ipyparallel as ipp
import logging

# this needs to be decorated as interactive to pass this module's namespace 
# in rather than the system globals (which doesn't include this module on 
# the remote process)
@ipp.interactive
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

        outputs.append(view.remote(block=True)(run_local)(*args, **kwargs)[0])

        rc.shutdown(hub=True)

    return outputs

# this needs to be decorated as interactive to pass this module's namespace 
# in rather than the system globals (which doesn't include this module on 
# the remote process)
@ipp.interactive
def profile_local(labels, path, module_name, func_name, *func_args, number=1, **func_kwargs):
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
    import sys
    sys.path.append(path)
    import importlib
    module = importlib.import_module(module_name)
    # get the function
    func = getattr(module, func_name)

    # we deal with mumps timings by forcing increased logging and accumulating the timings in a dictionary as we go
    test_mumps = any([label.startswith('MUMPS') for label in labels])
    mumps_timings = {'MUMPS analysis':0.0, 
                     'MUMPS factorization':0.0, 
                     'MUMPS solve':0.0}
    if test_mumps:
        # turn on mumps logging to get the timings we need
        if 'petsc_options' in func_kwargs:
            func_kwargs['petsc_options']['mat_mumps_icntl_4'] = \
                max(func_kwargs['petsc_options'].get('mat_mumps_icntl_4', -1), 2)
        else:
            func_kwargs['petsc_options'] = {'ksp_type' : 'preonly', 'pc_type' : 'lu', 'pc_factor_mat_solver_type' : 'mumps', 'mat_mumps_icntl_4' : 2}

    # run the function the specified number of times
    for n in range(number):
        out = StringIO()
        with pipes(stdout=out, stderr=STDOUT):
            _ = func(*func_args, **func_kwargs)
        if test_mumps:
            # collect the numps timings
            for step in ['analysis', 'factorization', 'solve']:
                i0 = out.getvalue().find(f'Elapsed time in {step} driver')
                if i0 > -1:
                    ie = out.getvalue()[i0:].find('\n')
                    mumps_timings[f'MUMPS {step}'] += float(out.getvalue()[i0:i0+ie].split()[-1])
        out.close()


    # extract and return the computation times from dolfinx
    times = []
    for label in labels:
        try:
            times.append(df.common.timing(label)[1]/number)
        except RuntimeError:
            if label.startswith('MUMPS'):
                # get the timing from our dictionary
                times.append(mumps_timings[label]/number)
            else:
                # unknown label
                times.append(0.0)
    times = np.array(times)
    maxtimes = None
    if MPI.COMM_WORLD.rank==0: maxtimes = np.zeros_like(times)
    MPI.COMM_WORLD.Reduce(times, maxtimes, op=MPI.MAX)
    if MPI.COMM_WORLD.rank==0: maxtimes = maxtimes.tolist()
    return maxtimes

def profile_parallel(nprocs, labels, *args, output_filename=None, **kwargs):
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
    * kwargs          - additional keyword arguments to pass to profile_local function
        * number      - number of runs of function to time
        * func_kwargs - keyword arguments to pass to the function
    Returns:
    * maxtimes        - computation walltimes corresponding to each label and number of processes
    """
    maxtimes = []
    for nproc in nprocs:
        cluster = ipp.Cluster(engine_launcher_class="mpi", n=nproc, log_level=logging.FATAL)
        rc = cluster.start_and_connect_sync()
        view = rc[:]

        maxtimes.append(view.remote(block=True)(profile_local)(labels, *args, **kwargs)[0])

        rc.shutdown(hub=True)

    maxlen = max([len(label) for label in labels])
    print('=========================')
    print("{0:<{1}}".format(" ", maxlen+2)+"".join(["{:<12d}".format(nproc) for nproc in nprocs]))
    for l, label in enumerate(labels):
        print("{0:<{1}}".format(label, maxlen+2)+"".join(["{:<12g}".format(t[l]) for t in maxtimes]))
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

    return maxtimes