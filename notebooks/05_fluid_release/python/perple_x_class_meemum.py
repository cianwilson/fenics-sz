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
import glob
import re

output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


# %%
class PerpleXMeemum:
    def __init__(self, basename : str, 
                 component_masses : dict, excluded_phases : list,
                 solution_models : list,
                 csv_file : str = None, version : str ='7.1.9',
                 clean_tmp_folder : bool = True):
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
        self.data_folder = pathlib.Path(os.path.join(basedir, os.pardir, "data", "perple_x_v"+self.version))
    
    def __del__(self):
        if self.clean_tmp_folder and hasattr(self, '_tmp_work_folder') and self._tmp_work_folder is not None:
            self._tmp_work_folder.cleanup()

    @property
    def tmp_work_folder(self):
        if not hasattr(self, '_tmp_work_folder') or self._tmp_work_folder is None:
            work_folder = pathlib.Path(os.path.join(os.getcwd(), "work"))
            work_folder.mkdir(exist_ok=True, parents=True)
            self._tmp_work_folder = tempfile.TemporaryDirectory(dir=work_folder)

            shutil.copy( self.data_folder / 'perplex_option.dat', self.tmp_work_folder)
            shutil.copy( self.data_folder / 'solution_model.dat', self.tmp_work_folder)
            shutil.copy( self.data_folder / 'hp622ver.dat', self.tmp_work_folder)

            # build
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
            """+os.linesep.join(v for v in self.component_masses.values())+"""
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
        return pathlib.Path(self._tmp_work_folder.name)

    def eval_h2o(self, P : float, T : float):
        Parr = np.clip(np.atleast_1d(P)*10000.0, a_min=1000.0, a_max=80000.0)
        Tarr = np.clip(np.atleast_1d(T)+273.15,  a_min=473.0,  a_max=1673.0)

        if len(Parr) != len(Tarr):
            raise RuntimeError("P and T must have same length")

        points = os.linesep.join(str(Ti)+" "+str(Pi) for Pi, Ti in zip(Parr, Tarr))
        input = self.basename+"""
        n
        """+points+"""
        0 0"""
        input = os.linesep.join(line.lstrip() for line in input.splitlines())
        result = subprocess.run(["meemum-v"+self.version], input=input, text=True, capture_output=True, cwd=self.tmp_work_folder)

        blocks = re.findall(r'Bulk Composition:\s*\n(.*?)\n\s*\nOther Bulk Properties:',
                             result.stdout, re.S)
        if len(blocks) != len(Parr):
            print(result.stdout)
            raise RuntimeError(f"Expected {len(Parr)} 'Bulk Composition:' blocks in meemum "
                                f"output, found {len(blocks)}.")

        h2oarr = np.empty(len(Parr))
        for i, block in enumerate(blocks):
            h2o_line = next((line for line in block.splitlines() if line.split()[:1] == ['H2O']), None)
            if h2o_line is None:
                raise RuntimeError("Could not find H2O row in meemum Bulk Composition.")
            values = [float(v) for v in h2o_line.split()[1:]]

            # 'Complete Assemblage'/'Solid Only' side-by-side columns only appear when a
            # free fluid is stable; without one the single (unlabeled) block already is
            # the solid-only composition.
            dual_block = 'Complete Assemblage' in block and 'Solid Only' in block
            h2oarr[i] = values[6] if dual_block else values[2]
        return h2oarr


# %%
import json
with open(os.path.join(basedir, os.pardir, "data", "perple_x_v7.1.9", "abers_25.json"), "r") as file:
    abers_25 = json.load(file)

# %%
basename = 'dike_25'
grid = PerpleXMeemum(basename, abers_25[basename]['component_masses'], abers_25[basename]['excluded_phases'], abers_25[basename]['solution_models'])

# %%
grid.eval_h2o(0.2, 400)

# %%
grid.eval_h2o(3.0, 1000.0)

# %%
grid.tmp_work_folder

# %%

from fenics_sz.fluid_release.perple_x_class import PerpleXGrid

# %%
oggrid = PerpleXGrid(csv_file = '../../data/perple_x_v7.1.9/dike_25_h2o.csv')


# %%
oggrid.eval_h2o(0.2, 400)

# %%
oggrid.eval_h2o(3.0, 1000)

# %%
