# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FEniCS-SZ
#
# Authors: Cian Wilson, Cameron Seebeck, Kidus Teshome, Nathan Sime, Peter van Keken
#
# Welcome to [_FEniCS Subduction Zones_ (FEniCS-SZ)](https://cianwilson.github.io/fenics-sz), a python-based package for the thermal modeling of subduction zones!
#
# This package was developed by undergraduate interns Kidus Teshome and Cameron Seebeck at the [Carnegie Science Earth & Planets Laboratory](https://epl.carnegiescience.edu).  It is based on [Wilson & van Keken, PEPS, 2023 (II)](http://dx.doi.org/10.1186/s40645-023-00588-6), which is part II of a three part introductory review of the thermal structure of subduction zones by Peter van Keken and Cian Wilson.
#
# Our goal is both to develop software tools for kinematic-slab thermal models of subduction zones in as open-source a way as possible and to provide an easily accessible and modifiable source of the global suite of subduction zone models as described in [Wilson & van Keken, PEPS, 2023 (II)](http://dx.doi.org/10.1186/s40645-023-00588-6) and [van Keken & Wilson, PEPS, 2023 (III)](https://doi.org/10.1186/s40645-023-00589-5).  We do this by basing all of our development in jupyter notebooks and making them accessible through the [FEniCS-SZ website](https://cianwilson.github.io/fenics-sz).
#
# This website is published as a [jupyter book](https://jupyterbook.org/). Each page is a jupyter notebook that can be run interactively in the browser.  To start such an interactive session using [binder](https://mybinder.org/) click the ![Binder symbol](images/binder.png)-symbol in the top right corner of the relevant page.  For more details on running an interactive session see the [usage section](./01_introduction/1.2_usage.ipynb).
#
# Comments and corrections to the package should be submitted to the issue tracker by going to the relevant page in the jupyter book, then clicking the ![git](images/git.png)-symbol in the top right corner and "open issue".

# %% [markdown]
# ## Acknowledgments
#
# We make heavy use of the finite element library [FEniCSx](https://fenicsproject.org) and linear algebra package [PETSc](https://petsc.org).  Simulations are run in parallel using [ipyparallel](https://ipyparallel.readthedocs.io) and we rely on python's extensive scientific ecosystem, such as [numpy](https://numpy.org), [scipy](https://scipy.org), [matplotlib](https://matplotlib.org), and [pyvista](https://pyvista.org), combined with [project jupyter's](https://jupyter.org) web-based, interactive, annotated notebooks.
#
# This jupyter book is based on the [FEniCSx Tutorial](https://jsdokken.com/dolfinx-tutorial/) by [Jørgen S. Dokken](https://jsdokken.com/), which is an excellent resource for learning how to use [FEniCS](https://fenicsproject.org/) in a similar interactive jupyter book.
#
