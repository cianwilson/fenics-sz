# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Structure
#
# FEniCS-SZ is intended as a hyper open source resource for developing and understanding thermal models of subduction zones.  It is presented [online](https://cianwilson.github.io/fenics-sz) as a non-interactive jupyter book and as a set of jupyter notebooks in a [github repository](https://github.com/cianwilson/fenics-sz), which are accessible interactively through a suitable jupyter server (see [the usage instructions](1.2_usage.ipynb)).  The vast majority of development takes places in the jupyter notebooks, which are contained in the `notebooks` folder of the repository and are converted to html pages online.  All remaining folders of the repository contain auxiliary or automatically generated files.
#
# The jupyter notebooks are split into three sections both in the repository and online:
#
# 1.  Introduction:
#
#     The section containing this notebook as well as usage information.  Contained in the `notebooks/01_introduction` folder of the repository.
#
# 2.  Background: 
#     
#     This tutorial section contains a set of notebooks introducing the finite element method for simple problems relevant to geodynamics and is contained in the `notebooks/02_background` folder of the repository.
#
# 3.  Subduction Zone Problems: 
#
#     This section contains the bulk of the code development of FEniCS-SZ. A series of notebooks introduces thermal problems in subduction zones and develops a set of python classes for their solution.  These are contained in the `notebooks/03_sz_problems` folder.
#
# 4.  Global Suites:
#
#     This section utilizes the python code developed in the previous section to present full global suites of subduction zone models using both steady state and time-dependent assumptions.  By default, these are low resolution examples to keep the website generation and loading speed low but they can be run at more appropriate resolutions interactively and can be found in the `notebooks/04_global_suites` folder of the repository.

# %% [markdown]
# ## Python Modules
#
# To allow inline development of python classes and subsequent usage in both jupyter notebooks and python scripts we use [jupytext](https://jupytext.readthedocs.io/en/latest/) to automatically convert notebooks into python modules whenever the notebooks are saved.  Cells in the notebooks that should not be run in the modules should be marked with an `active-ipynb` tag (see [the jupytext documentation](https://jupytext.readthedocs.io/en/latest/advanced-options.html#active-and-inactive-cells)) to indicate that they should only be run in the notebook version.
#
# Python modules necessary for `fenics_sz` are linked to the `python` directory of the repository in a directory tree that mirrors the structure of the notebooks described above.  A few module files are not generated from the notebooks and exist outside of this structure (e.g. `python/fenics_sz/utils`).  All modules are imported in place by the notebooks to avoid requiring an installation step when editing the notebooks.
#
# The full suite of notebook-generated python modules can be updated using the script `bin/update_python`. 

# %% [markdown]
# ## Git Branches
#
# The `release` branch of the repository corresponds to the webpage version.  Pushes to the `release` branch of the [github repository](https://github.com/cianwilson/fenics-sz) trigger rebuilds of the website.  All such pushes must be done by an administrator and through a pull request.
#
# The `main` branch of the repository typically corresponds to the `release` branch with all changes being merged into the `main` branch (through pull requests) before `main` is merged into `release`.  Pull requests to either branch trigger github actions, which runs all jupyter notebooks and tests the website build.  Test website builds are available as downloadable artifacts through [github actions](https://github.com/cianwilson/fenics-sz/actions).

# %% [markdown]
#
