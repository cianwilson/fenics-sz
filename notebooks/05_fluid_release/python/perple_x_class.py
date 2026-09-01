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
work_folder = pathlib.Path(os.path.join(os.getcwd(), "work"))
work_folder.mkdir(exist_ok=True, parents=True)


# %%
class PerpleXGrid:
    def __init__(self, dat_file : str = None, csv_file : str = None, 
                 version : str ='7.1.9',
                 clean_tmp_folder : bool = True,
                 work_folder : str = None,
                 melt_in_fluid : bool = False):
        self._df = None
        self.dat_file = None
        self.basename = None
        if csv_file is not None:
            self._df = pd.read_csv(csv_file, index_col=0)
            self._df.columns = self._df.columns.astype(float)
            if dat_file is not None:
                raise RuntimeWarning("csv_file and dat_file provided, ignoring dat_file.")
        elif dat_file is not None:
            file = pathlib.Path(dat_file)
            if not file.exists(): raise RuntimeError("Provided dat_file does not exist.")
            self.dat_file = file
            self.basename = file.stem
        else:
            raise RuntimeError("Require csv_file or dat_file to be provided.")

        if version not in ['7.1.9',]:
            raise RuntimeError("Unknown perple_x version.")
        self.version = version
        self.clean_tmp_folder = clean_tmp_folder
        self.work_folder = work_folder
        self.melt_in_fluid = melt_in_fluid
        self.data_folder = pathlib.Path(os.path.join(basedir, os.pardir, "data", "perple_x_v"+self.version))
    
    def __del__(self):
        if self.clean_tmp_folder and hasattr(self, '_tmp_work_folder') and self._tmp_work_folder is not None:
            self._tmp_work_folder.cleanup()

    @property
    def initialized(self) -> bool:
        return getattr(self, '_df', None) is not None

    @property
    def tmp_work_folder(self):
        if not hasattr(self, '_tmp_work_folder') or self._tmp_work_folder is None:
            self._tmp_work_folder = tempfile.TemporaryDirectory(dir=self.work_folder)
        return pathlib.Path(self._tmp_work_folder.name)

    @property
    def df(self):
        if not self.initialized:
            shutil.copy(self.dat_file, self.tmp_work_folder)
            shutil.copy( self.data_folder / 'perplex_option.dat', self.tmp_work_folder)

            # patch the copied option file's melt_is_fluid flag to match melt_in_fluid,
            # preserving its column alignment and line ending
            option_file = self.tmp_work_folder / 'perplex_option.dat'
            value = 'T' if self.melt_in_fluid else 'F'
            lines = option_file.read_text().splitlines(keepends=True)
            for i, line in enumerate(lines):
                m = re.match(r'^(melt_is_fluid\s+)(\S+)(\s*)(\|.*)$', line)
                if m:
                    key, old_value, pad, rest = m.groups()
                    newline = '\n' if line.endswith('\n') else ''
                    lines[i] = key + value.ljust(len(old_value) + len(pad)) + rest + newline
                    break
            option_file.write_text(''.join(lines))

            # vertex
            stdout = open(os.path.join(self.tmp_work_folder, 'vertex_'+ self.basename + '.log'), 'w')
            stderr = open(os.path.join(self.tmp_work_folder, 'vertex_'+ self.basename + '.err'), 'w')
            input = self.basename
            subprocess.run(["vertex-v"+self.version], input=input, text=True, stdout=stdout, stderr=stderr, cwd=self.tmp_work_folder)
            stdout.close()
            stderr.close()

            stdout = open(os.path.join(self.tmp_work_folder, 'werami_'+ self.basename + '.log'), 'w')
            stderr = open(os.path.join(self.tmp_work_folder, 'werami_'+ self.basename + '.err'), 'w')
            # basename
            # 2D grid - 2
            # all phase and/or system properties (could try more compact output here) - 36
            # one system symmary per node (3 gives this plus all phases) - 1
            # include fluid in modal properties - n
            # change grid definition - y
            # min and max T
            # min and max P
            # num T, P nodes (designed for convenient/even 5C/0.02GPa grid)
            # end - 0
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
            subprocess.run(["werami-v"+self.version], input=input, text=True, stdout=stdout, stderr=stderr, cwd=self.tmp_work_folder)
            stdout.close()
            stderr.close()

            datafile = os.path.join(self.tmp_work_folder, self.basename + '_1.tab')

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

            long_df = pd.read_csv(datafile, sep=r"\s+", skiprows=header_idx-1, header=1, usecols=cols)
            self._df = long_df.pivot(index='P(bar)', columns='T(K)', values='H2O,wt%')

            # reset other stored variables
            self._interpolator = None
        return self._df

    def save_h2o(self, filename):
        self.df.to_csv(filename)

    @property
    def P(self):
        return self.df.index.to_numpy()/10000.0

    @property
    def T(self):
        return self.df.columns.to_numpy() - 273.15

    @property
    def H2O(self):
        return self.df.to_numpy()
    
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

    def eval(self, P, T):
        Parr = np.clip(np.atleast_1d(P), a_min=self.P.min(), a_max=self.P.max())
        Tarr = np.clip(np.atleast_1d(T), a_min=self.T.min(), a_max=self.T.max())
        PT = np.stack((Parr, Tarr), axis=1)
        return {'H2O' : self.interpolator(PT)}

# %% tags=["active-ipynb"] vscode={"languageId": "raw"}
# grid = PerpleXGrid(dat_file = '../../data/perple_x_v7.1.9/abers_25/dike_25.dat')
# fig, ax = grid.plot_h2o()

# %% tags=["active-ipynb"] vscode={"languageId": "raw"}
# files = glob.glob('../../data/perple_x_v7.1.9/abers_25/*_25.dat')
# for dat_file in files:
#     grid = PerpleXGrid(dat_file = dat_file)
#     print(grid.basename)
#     fig, ax = grid.plot_h2o()
#     fig.savefig(output_folder / str(grid.basename+'.png'), dpi=400)
#     grid.save_h2o(os.path.splitext(dat_file)[0]+'_h2o.csv')

# %% tags=["active-ipynb"]
# files = glob.glob('../../data/perple_x_v7.1.9/abers_25/*_25_h2o.csv')
# for csv_file in files:
#     basename = os.path.basename(csv_file).split('.')[0][:-4]
#     grid = PerpleXGrid(csv_file = csv_file)
#     print(basename)
#     fig, ax = grid.plot_h2o()
#     _ = ax.set_title(basename)

# %%
