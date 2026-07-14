# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dolfinx-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interpolating hydrations for P-T curves

# %% [markdown]
# In the last notebook, we built up the tools to extract P-T curves from our solved subduction problems, and we started to explore using those P-T curves in conjunction with a phase-stability diagram. In this notebook, we'll expand upon that comparison, and develop the tools to model the hydration state of an entire slab.

# %% [markdown]
# Start with the usual imports and file path settings

# %%
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))

# %%
print(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))

# %%
import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as pl
import matplotlib.image as img
import subprocess
import pathlib
import pyvista as pv
import copy

import fenics_sz.utils
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)

# %%
from fenics_sz.sz_problems.sz_slab import create_slab, plot_slab
from fenics_sz.sz_problems.sz_geometry import create_sz_geometry
from fenics_sz.sz_problems.sz_steady_dislcreep import SteadyDislSubductionProblem
from fenics_sz.sz_problems.sz_tdep_dislcreep import TDDislSubductionProblem
from fenics_sz.sz_problems.sz_params import default_params, allsz_params

# %%
from fenics_sz.fluid_release import perple_x_integration

# %%
