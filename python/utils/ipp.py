from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as pl
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
    from mpi4py import MPI
    import sys
    sys.path.append(path)
    import importlib
    module = importlib.import_module(module_name)
    # get the function
    func = getattr(module, func_name)

    for n in range(number):
        _ = func(*func_args, **func_kwargs)

    # extract and return the computation times from dolfinx
    times = [df.common.timing(l)[1]/number for l in labels]
    maxtimes = MPI.COMM_WORLD.reduce(times, op=MPI.MAX)
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

    print('=========================')
    print('\t'.join(['\t']+[repr(nproc) for nproc in nprocs]))
    for l, label in enumerate(labels):
        print('\t'.join([label]+[repr(t[l]) for t in maxtimes]))
    print('=========================')

    if MPI.COMM_WORLD.rank == 0:
        fig, (ax, ax_r) = pl.subplots(nrows=2, figsize=[6.4,9.6], sharex=True)
        for l, label in enumerate(labels):
            ax.plot(nprocs, [t[l] for t in maxtimes], 'o-', label=label)
        ax.set_ylabel('wall time (s)')
        ax.grid()
        ax.legend()

        ax_r.plot(nprocs, [nproc/nprocs[0] for nproc in nprocs], 'k--', label='Ideal')
        for l, label in enumerate(labels):
            ax_r.plot(nprocs, maxtimes[0][l]/np.asarray([t[l] for t in maxtimes]), 'o-', label=label)
        ax_r.set_xlabel('number processors')
        ax_r.set_ylabel('speed up')
        ax_r.grid()
        ax_r.legend()

        # Write profiling to disk
        if output_filename is not None:
            fig.savefig(output_filename)

            print("***********  profiling figure in "+str(output_filename))

    return maxtimes