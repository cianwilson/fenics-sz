# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)

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

output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


# %%
class PerpleXGrid:
    def __init__(self, dat_file : str = None, csv_file : str = None, version : str ='7.1.9'):
        self.version  = version

        self._df = None
        if csv_file is not None:
            self._df = pd.read_csv(csv_file)
            if dat_file is not None:
                raise RuntimeWarning("csv_file and dat_file provided, ignoring dat_file.")
            file = pathlib.Path(csv_file)
        elif dat_file is not None:
            file = pathlib.Path(dat_file)
            if not file.exists(): raise RuntimeError("Provided dat_file does not exist.")
        else:
            raise RuntimeError("Require csv_file or dat_file to be provided.")
        self.basename = file.stem
        self.data_folder = file.parent

    @property
    def df(self):
        if not hasattr(self, '_df') or self._df is None:
            with tempfile.TemporaryDirectory() as tmp_work_folder:
                shutil.copy(((self.data_folder / self.basename)).with_suffix('.dat'), tmp_work_folder)
                shutil.copy( self.data_folder / 'perplex_option.dat', tmp_work_folder)

                # vertex
                stdout = open(os.path.join(tmp_work_folder, 'vertex_'+ self.basename + '.log'), 'w')
                stderr = open(os.path.join(tmp_work_folder, 'vertex_'+ self.basename + '.err'), 'w')
                if self.version == '7.1.9':
                    input = self.basename
                    subprocess.run(["vertex-v"+self.version], input=input, text=True, stdout=stdout, stderr=stderr, cwd=tmp_work_folder)
                else:
                    raise RuntimeError("Unknown Perple_X vertex version")
                stdout.close()
                stderr.close()

                stdout = open(os.path.join(tmp_work_folder, 'werami_'+ self.basename + '.log'), 'w')
                stderr = open(os.path.join(tmp_work_folder, 'werami_'+ self.basename + '.err'), 'w')
                if self.version == '7.1.9':
                    input=self.basename+"""
                    2
                    36
                    1
                    n
                    y
                    473 1673
                    1000 80000
                    241 396
                    0
                    """
                    subprocess.run(["werami-v"+self.version], input=input, text=True, stdout=stdout, stderr=stderr, cwd=tmp_work_folder)
                else:
                    raise RuntimeError("Unknown Perple_X werami version")
                stdout.close()
                stderr.close()

                datafile = os.path.join(tmp_work_folder, self.basename + '_1.tab')

                cols = ["T(K)", "P(bar)", "H2O,wt%"]

                # we need to find the row of the file that contains the header
                header_idx = None
                with open(datafile, 'r') as f:
                    i = 0
                    for line in f:
                        if all([c in line for c in cols]):
                            header_idx = i
                            break
                        i += 1

                # some sanity checks
                if header_idx is None:
                    raise RuntimeError("Could not find header row")

                if header_idx < 1:
                    raise RuntimeError("Unexpected number of header rows")

                self._df = pd.read_csv(datafile, sep=r"\s+", skiprows=header_idx-1, header=1, usecols=cols)

                # reset other stored variables
                self._P = None
                self._T = None
                self._H2O = None
                self._interpolator = None
        return self._df

    def save_h2o(self, filename=None):
        if filename is None: filename = self.data_folder / str(self.basename + '_h2o.csv')
        self.df.to_csv(filename, index=False)

    @property
    def P(self):
        if not hasattr(self, '_P') or self._P is None: 
            self._P = np.unique(self.df['P(bar)'].to_numpy())/10000.0
        return self._P

    @property
    def T(self):
        if not hasattr(self, '_T') or self._T is None: 
            self._T = np.unique(self.df['T(K)'].to_numpy()) - 273.15
        return self._T

    @property
    def H2O(self):
        if not hasattr(self, '_H2O') or self._H2O is None: 
            self._H2O = self.df['H2O,wt%'].to_numpy().reshape(len(self.P),len(self.T))
        return self._H2O
    
    def plot_h2o(self):
        fig, ax = pl.subplots(figsize=(7, 4.5))
        vmin = 0.0
        vmax = 5.5
        dv = 0.01
        levels = np.arange(vmin, vmax+dv, dv)
        c = ax.contourf(self.T, self.P, self.H2O, levels=levels, cmap="jet_r")
        cbar = fig.colorbar(c, label=r"H$_2$O (wt%)")
        cbar.set_ticks(np.arange(vmin, vmax, 1, dtype=np.int32))
        ax.set_ylabel(r"P (GPa)")
        ax.set_xlabel(r"T ($^\circ$C)")
        ax.set_box_aspect(1)
        return fig, ax

    @property
    def interpolator(self):
        if not hasattr(self, '_interpolator') or self._interpolator is None:
            self._interpolator = sp.interpolate.RegularGridInterpolator((self.P, self.T), self.H2O, method='linear')
        return self._interpolator

    def eval_h2o(self, P, T):
        Parr = np.clip(np.atleast_1d(P), a_min=self.P.min(), a_max=self.P.max())
        Tarr = np.clip(np.atleast_1d(T), a_min=self.T.min(), a_max=self.T.max())
        PT = np.stack((Parr, Tarr), axis=1)
        return self.interpolator(PT)

# %% tags=["active-ipynb"] vscode={"languageId": "raw"}
# files = glob.glob('../../data/perple_x_v7.1.9/*_25.dat')
# for dat_file in files:
#     grid = PerpleXGrid(dat_file = dat_file)
#     print(grid.basename)
#     fig, ax = grid.plot_h2o()
#     fig.savefig(output_folder / str(grid.basename+'.png'), dpi=400)
#     grid.save_h2o()

# %% tags=["active-ipynb"]
# files = glob.glob('../../data/perple_x_v7.1.9/*_25_h2o.csv')
# for csv_file in files:
#     grid = PerpleXGrid(csv_file = csv_file)
#     print(grid.basename)
#     fig, ax = grid.plot_h2o()
#     ax.set_title(grid.basename)

# %%
