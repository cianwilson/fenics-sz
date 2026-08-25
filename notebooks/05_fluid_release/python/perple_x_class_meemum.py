# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dolfinx-env (3.12.3.final.0)
#     language: python
#     name: python3
# ---

# %%
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))

# %%
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
import subprocess
import pathlib
import tempfile
import shutil
import re
import pty
import select
import time
from dataclasses import dataclass
from typing import TextIO

output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)
work_folder = pathlib.Path(os.path.join(os.getcwd(), "work"))
work_folder.mkdir(exist_ok=True, parents=True)


# %%
@dataclass
class Meemum:
    process : subprocess.Popen = None
    fd      : int              = None
    log     : TextIO           = None
    err     : TextIO           = None


# %%
class PerpleXMeemum:
    def __init__(self, basename : str, 
                 component_masses : dict, excluded_phases : list,
                 solution_models : list,
                 csv_file : str = None, version : str ='7.1.9',
                 clean_tmp_folder : bool = True,
                 work_folder : str = None):
        self.basename = basename
        self.component_masses = component_masses
        self.excluded_phases = excluded_phases
        self.solution_models = solution_models
        self._df = None
        if csv_file is not None:
            self._df = pd.read_csv(csv_file, index_col=0)
            self._df.columns = self._df.columns.astype(float)
        if version not in ['7.1.9',]:
            raise RuntimeError("Unknown perple_x version.")
        self.version  = version
        self.clean_tmp_folder = clean_tmp_folder
        self.work_folder = work_folder
        self.data_folder = pathlib.Path(os.path.join(basedir, os.pardir, "data", "perple_x_v"+self.version))
    
    def __del__(self):
        if self.initialized:
            try:
                self.meemum_write("0 0")
            except OSError:
                pass
            try:
                self._meemum.process.wait(timeout=10)
            except Exception:
                self._meemum.process.kill()
            os.close(self._meemum.fd)
            self._meemum.log.close()
            self._meemum.err.close()
        if self.clean_tmp_folder and hasattr(self, '_tmp_work_folder') and self._tmp_work_folder is not None:
            self._tmp_work_folder.cleanup()

    @property
    def initialized(self) -> bool:
        return getattr(self, '_meemum', None) is not None

    @property
    def tmp_work_folder(self):
        if not hasattr(self, '_tmp_work_folder') or self._tmp_work_folder is None:
            self._tmp_work_folder = tempfile.TemporaryDirectory(dir=self.work_folder)
        return pathlib.Path(self._tmp_work_folder.name)

    # -- a persistent, interactive meemum session, driven through a pty --------
    #
    # meemum is a REPL: give it a T,P (and, since we opt in below, a bulk
    # composition) and it prints a full report and loops back for the next
    # one, until fed "0 0". Keeping ONE meemum process alive across calls
    # (rather than spawning a fresh one per eval call) avoids paying its
    # startup cost - reading the thermodynamic data and regenerating
    # pseudocompounds for the solution models - more than once, and lets
    # later calls change the bulk composition without re-running build.
    #
    # meemum's stdout is only line-buffered when it thinks it's talking to a
    # terminal; piped (as subprocess.PIPE would give it) it's fully
    # block-buffered, so prompts we need to see before writing our next
    # input may never actually reach us. Running it behind a pty (as opened
    # by the pty module) makes it behave as if it has a real terminal
    # attached, so prompts are flushed as they're printed.

    @property
    def meemum(self):
        if not self.initialized:
            shutil.copy( self.data_folder / 'perplex_option.dat', self.tmp_work_folder)
            shutil.copy( self.data_folder / 'solution_model.dat', self.tmp_work_folder)
            shutil.copy( self.data_folder / 'hp622ver.dat', self.tmp_work_folder)

            # build - this is where component_masses, excluded_phases and
            # solution_models are actually consumed, so nothing has been
            # "initialized" until this has run
            stdout = open(os.path.join(self.tmp_work_folder, 'build_'+self.basename + '.log'), 'w')
            stderr = open(os.path.join(self.tmp_work_folder, 'build_'+self.basename + '.err'), 'w')
            # basename
            # thermodynamic data file - hp622ver.dat
            # perplex option file - perplex_option.dat
            # transform default base components - n
            # computational mode - 2, constrained minimization on a 2d grid
            # calculation with a saturated fluid - n
            # calculation with saturated components - n
            # use chemical potentials, activities or fugacities as independent variables - n
            # select thermondynamic components (1 per line)
            # make P and T dependent - n
            # x-axis variable - 2, T
            # min and max T
            # min and max P
            # specify components by mass - y
            # component masses
            # output print file - y?
            # exclude pure and/or endmember phases - y
            # prompt for phases - n
            # excluded phases
            # include solution models - y
            # solution model file name - solution_model.dat
            # solution models (end with blank line)
            # calculation title
            input = self.basename+"""
            hp622ver.dat
            perplex_option.dat
            n
            2
            n
            n
            n
            """+os.linesep.join(k for k in self.component_masses.keys())+"""

            n
            2
            473 1673
            1000 80000
            y
            """+os.linesep.join(str(v) for v in self.component_masses.values())+"""
            y
            y
            n
            """+os.linesep.join(self.excluded_phases)+"""

            y
            solution_model.dat
            """+os.linesep.join(self.solution_models)+"""

            """+\
            self.basename
            input = os.linesep.join(line.lstrip() for line in input.splitlines())
            subprocess.run(["build-v"+self.version], input=input, text=True, stdout=stdout, stderr=stderr, cwd=self.tmp_work_folder)
            stdout.close()
            stderr.close()

            parent_fd, child_fd = pty.openpty()

            # this must be assigned before the first call to meemum_write below
            self._meemum = Meemum()

            self._meemum.err = open(os.path.join(self.tmp_work_folder, 'meemum_'+self.basename+'.err'), 'w')

            self._meemum.process = subprocess.Popen(
                ["meemum-v"+self.version],
                stdin=child_fd, stdout=child_fd, stderr=self._meemum.err,
                cwd=self.tmp_work_folder, close_fds=True,
            )
            # the child now holds its own copy of the child fd; we only need the parent
            os.close(child_fd)
            self._meemum.fd = parent_fd
            self._meemum.log = open(os.path.join(self.tmp_work_folder, 'meemum_'+self.basename+'.log'), 'w')

            # drive the session through to the first T,P prompt: give the
            # project name, then opt into per-point interactive
            # compositions (rather than the fixed one baked into the build)
            self.meemum_write(self.basename)
            self.meemum_read_until("Interactively enter bulk compositions")
            self.meemum_write("y")
            self.meemum_read_until("Enter (zeroes to quit)")
        return self._meemum

    def meemum_write(self, text):
        os.write(self.meemum.fd, (text + os.linesep).encode())

    def meemum_read_until(self, marker, timeout=30.0):
        buffer = ""
        deadline = time.monotonic() + timeout
        while marker not in buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {marker!r} from meemum. "
                                    f"Buffer tail:"+os.linesep+f"{buffer[-2000:]}")
            ready, _, _ = select.select([self.meemum.fd], [], [], remaining)
            if not ready:
                raise TimeoutError(f"Timed out (no output) waiting for {marker!r} from meemum. "
                                    f"Buffer tail:"+os.linesep+f"{buffer[-2000:]}")
            try:
                chunk = os.read(self.meemum.fd, 4096)
            except OSError:
                # the child closed its end of the pty (process exited)
                break
            if not chunk:
                break
            decoded = chunk.decode(errors="replace")
            buffer += decoded
            self.meemum.log.write(decoded)
        return buffer

    def eval(self, P : float, T : float, **comps):
        Parr = np.clip(np.atleast_1d(P)*10000.0, a_min=1000.0, a_max=80000.0)
        Tarr = np.clip(np.atleast_1d(T)+273.15,  a_min=473.0,  a_max=1673.0)

        if len(Parr) != len(Tarr):
            raise RuntimeError("P and T must have same length")

        compsarr = np.empty((len(Parr), len(self.component_masses)))
        for i, (k, v) in enumerate(self.component_masses.items()):
            comparr = np.atleast_1d(comps.get(k, [v]*len(Parr)))
            if len(Parr) != len(comparr):
                raise RuntimeError("P and {:s} values must have same length".format(k,))
            compsarr[:, i] = comparr

        wt_pct = {name: np.empty(len(Parr)) for name in self.component_masses}
        for name in self.component_masses:
            wt_pct[name+'_f'] = np.zeros(len(Parr))
        wt_pct['F_f'] = np.zeros(len(Parr))

        for i, (Pi, Ti, compsi) in enumerate(zip(Parr, Tarr, compsarr)):
            self.meemum_write(str(Ti)+" "+str(Pi))
            self.meemum_write(" ".join(str(v) for v in compsi))
            block_text = self.meemum_read_until("Enter (zeroes to quit)")

            match = re.search(r'Bulk Composition:\s*\n(.*?)\n\s*\nOther Bulk Properties:',
                               block_text, re.S)
            if match is None:
                raise RuntimeError("Could not find 'Bulk Composition:' block in meemum output.\n"
                                    + block_text[-3000:])
            block = match.group(1)

            # 'Complete Assemblage'/'Solid Only' side-by-side columns only appear when a
            # free fluid is stable; without one the single (unlabeled) block already is
            # the solid-only composition, so there is no fluid to report.
            dual_block = 'Complete Assemblage' in block and 'Solid Only' in block

            component_lines = {line.split()[0]: line for line in block.splitlines() if line.split()}
            component_values = {}
            for name in self.component_masses:
                line = component_lines.get(name)
                if line is None:
                    raise RuntimeError(f"Could not find {name} row in meemum Bulk Composition.")
                values = [float(v) for v in line.split()[1:]]
                component_values[name] = values
                wt_pct[name][i] = values[6] if dual_block else values[2]

            if dual_block:
                fluid_g = {name: values[1] - values[5] for name, values in component_values.items()}
                total_fluid_g = sum(fluid_g.values())
                total_complete_g = sum(values[1] for values in component_values.values())
                wt_pct['F_f'][i] = total_fluid_g/total_complete_g*100.0
                if total_fluid_g > 0:
                    for name in self.component_masses:
                        wt_pct[name+'_f'][i] = fluid_g[name]/total_fluid_g*100.0

        return wt_pct

# %% tags=["active-ipynb"]
# import json
# with open(os.path.join(basedir, os.pardir, "data", "perple_x_v7.1.9", "abers_25", "abers_25.json"), "r") as file:
#     abers_25 = json.load(file)

# %% tags=["active-ipynb"]
# basename = 'dike_25'
# grid = PerpleXMeemum(basename, abers_25[basename]['component_masses'], abers_25[basename]['excluded_phases'], abers_25[basename]['solution_models'], work_folder=work_folder)

# %% tags=["active-ipynb"]
# grid.tmp_work_folder

# %% tags=["active-ipynb"]
# grid.eval([0.2, 1.7], [400, 1200.0], SiO2=[40.2, 57.1], H2O=[1.2, 5.6])

# %% tags=["active-ipynb"]
# grid.eval(3.0, 1000.0)

# %% tags=["active-ipynb"]
# grid.tmp_work_folder

# %% tags=["active-ipynb"]
#
# from fenics_sz.fluid_release.perple_x_class import PerpleXGrid

# %% tags=["active-ipynb"]
# oggrid = PerpleXGrid(csv_file = '../../data/perple_x_v7.1.9/abers_25/dike_25_h2o.csv')

# %% tags=["active-ipynb"]
# oggrid.eval(0.2, 400)

# %% tags=["active-ipynb"]
# oggrid.eval(3.0, 1000)

# %% tags=["active-ipynb"]
# oggrid.tmp_work_folder

# %%
