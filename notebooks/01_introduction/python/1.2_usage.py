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
# # Usage
#
# There are three ways to use FEniCS-SZ:
#  1. non-interactively by reading the FEniCS-SZ **website**
#  2. interactively in a web browser from any **jupyter** server with the appropriate software packages installed
#  3. as a python module in python **scripts**

# %% [markdown]
# ## 1. Website
#
# To use the website non-interactively please open [https://cianwilson.github.io/fenics-sz](https://cianwilson.github.io/fenics-sz) in a web browser and use the panel on the left-hand side to navigate the contents.  Each page corresponds to a jupyter notebook, converted to html in a [jupyter book](https://jupyterbook.org/).

# %% [markdown]
# ## 2. Jupyter
#
# FEniCS-SZ is a jupyter book made up of a set of jupyter notebook (.ipynb) files, each one of which can be run interactively in a web browser on a jupyter server.  On the website these are converted into html webpages.  When viewing through a jupyter server these are contained in the `notebooks` directory of the repository that will be visible in the left hand file directory.  Files are numbered corresponding to the order in which they appear on the website to make navigation as easy as possible.
#
# Once opened the notebooks are made up of cells that can contain annotations in a [markdown format](https://www.markdownguide.org/) or executable python code.  Files will typically open with the first cell highlighted.  To run that cell click the play button at the top of the frame and the highlighted cell will progress to the next cell.  A useful keyboard shortcut for this is `Shift + Enter`.  Repeatedly click play or `Shift + Enter` to progress through the notebook.
#
# Not all cells will produce output (e.g., markdown cells won't) but python cells that do will show their results inline below the cell.  Python cells performing a lot of calculations will take some time to run and may temporarily stall progression to the next cell.  Once a cell has finished executing it will display any output and progress the active cell down the page.
#
# Many excellent tutorials for jupyter exist (e.g. for [jupyter lab](https://jupyterlab.readthedocs.io/en/latest/getting_started/overview.html)).  If you're new to interacting with jupyter notebooks we recommend checking some of them out.
#
# Here we discuss how to launch jupyter using
#  1. [fastest but limited resources] **binder**, an online interactive jupyter server
#  2. [fast, requires a local docker installation] **docker**, a containerized software environment
#  3. [fast, sometimes compatibility issues] **conda**, a python package manager
#  4. [complicated, developers only] an installation from **source**

# %% [markdown]
# ### 2.1 Online Binder
#
# Starting on the [website](https://cianwilson.github.io/fenics-sz), navigate to whichever page you want to run interactively, then simply click the ![Binder symbol](../images/binder.png)-symbol in the top right corner of the page.  This will launch an online interactive jupyter session using [binder](https://mybinder.org/) and a containerized docker development environment for FEniCS-SZ.  Note that binder may take some time to start while it downloads and starts the docker image.
#
# Any page can be opened this way but once a binder session is running it will likely be easier to navigate the pages from within the jupyter session using the left hand file directory rather than returning to the FEniCS-SZ website and repeatedly opening new binder sessions.
#
# ```{admonition} Interactive Changes
# Binder allows users to run notebooks interactively, making changes to the published jupyter book.  Note that these changes will be lost once the binder session ends unless the changed files are manually downloaded by selecting them in the jupyter file browser and downloading them to the local machine.
# ```
#
# ```{admonition} Computational Costs
# Binder limits the amount of computational resources available.  High resolution simulations may therefore not be feasible online.
# ```
#
# If this method works and provides sufficient computational resources to explore the notebooks then there's no need to read any further!  The instructions after this refer to running FEniCS-SZ on a local machine rather than online.

# %% [markdown]
# ### 2.2 Local Docker
#
# To run the notebooks locally, outside of [binder](https://mybinder.org/), an installation of [FEniCSx](https://fenicsproject.org/) is required. We strongly recommend new users do this using a local installation of [docker](https://www.docker.com/) and our pre-packaged docker development image.
#
# Docker is software that uses _images_ and _containers_ to supply virtualized installations of software across different kinds of operating systems (Linux, Mac, Windows).  Start by making sure a working installation of docker is available, following the instructions on their [webpage](https://docs.docker.com/get-started/).
#
# ```{admonition} Computational Resources
# On non-linux operating systems docker limits the computational resources available to the virtualized docker container, which may limit the size of simulations it is possible to run locally.  Modify the docker settings to make more resources available.
# ```
#
# Once docker is installed we provide compatible docker images using [github packages](https://github.com/users/cianwilson/packages/container/package/fenics-sz).  To use these images on a local machine open a terminal window, navigate to an appropriate location on the file system, and use git to clone the [FEniCS-SZ repository](https://github.com/cianwilson/fenics-sz) into that directory
# ```bash
#   git clone -b release https://github.com/cianwilson/fenics-sz.git
#   cd fenics-sz
# ```
#
# After this step there are two options for launching the FEniCS-SZ development container:
# 1. [likely for most users] automatically starting the jupyter server and opening it in a **browser**
# 2. [useful for developers] starting the container in a **terminal** and then manually launching the jupyter server
# 3. [also primarily for developers] starting a development container in an integrated development environment (**IDE**)

# %% [markdown]
# #### 2.2.1 Browser
#
# To start the jupyter server automatically in a browser run the following docker command
#
# ```bash
#   docker run --init --rm -p 8888:8888 --workdir /root/shared -v "$(pwd)":/root/shared ghcr.io/cianwilson/fenics-sz:release
# ```
# The first time this is run, it will automatically download the docker image and start jupyter in the docker container on the local machine.  To view and interact with the notebooks, copy and paste the URL printed in the terminal into a web browser.  Look for output similar to, e.g.:
# ```bash
#     To access the server, open this file in a browser:
#         file:///root/.local/share/jupyter/runtime/jpserver-6-open.html
#     Or copy and paste one of these URLs:
#         http://e7ba9ed551e1:8888/lab?token=35522c8f002cc75803ccd1f5764a6e480e5dd8569d8f82bf
#         http://127.0.0.1:8888/lab?token=35522c8f002cc75803ccd1f5764a6e480e5dd8569d8f82bf
# ```
# and copy the full URL starting with `http://127.0.0.1:8888` into a browser window.
#
# ```{admonition} Updates
# `docker run` will only download the docker image the first time it is called.  To get updates to the images run
#
#     docker pull ghcr.io/cianwilson/fenics-sz:release
#
# before calling `docker run`.
# ```
#
# If this method works and provides sufficient access to the repository and its notebooks then there's no need to read any further!  The instructions after this refer to running FEniCS-SZ while maintaining terminal access.

# %% [markdown]
# #### 2.2.2 Terminal
#
# Alternatively, the image can be used through an interactive terminal by running
#
# ```bash
#   docker run -it --rm -p 8888:8888 --workdir /root/shared -v "$(pwd)":/root/shared  --entrypoint="/bin/bash" ghcr.io/cianwilson/fenics-sz:release
# ```
# This can be useful, for example, to install extra packages in the docker container through the terminal.  Again, the first time this is run, it will automatically download the docker image on the local machine.  To get any subsequent updates it is necessary to run `docker pull` (see above).
#
# Jupyter can then be started from within the terminal running the docker container using
# ```bash
#   jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
# ```
# and again copying and pasting the resulting URL into a web browser to access the notebooks.  Look for output similar to, e.g.:
# ```bash
#     To access the server, open this file in a browser:
#         file:///root/.local/share/jupyter/runtime/jpserver-6-open.html
#     Or copy and paste one of these URLs:
#         http://e7ba9ed551e1:8888/lab?token=35522c8f002cc75803ccd1f5764a6e480e5dd8569d8f82bf
#         http://127.0.0.1:8888/lab?token=35522c8f002cc75803ccd1f5764a6e480e5dd8569d8f82bf
# ```
# and copy the full URL starting with `http://127.0.0.1:8888` into a browser window.
#
# Alternatively, to run notebooks non-interactively from the terminal navigate to the desired notebook and run
# ```bash
#   jupyter execute <notebook name>.ipynb
# ```
# replacing `<notbook name>.ipynb` with the filename of the notebook.
#
#
# If this method works and provides sufficient access to the repository and its notebooks then there's no need to read any further!  The instructions immediately after this refer to running FEniCS-SZ in an integrated development environment.

# %% [markdown]
# #### 2.2.3 IDE
#
# Some integrated development environments (IDEs), e.g. [Visual Studio Code (VSCode)](https://code.visualstudio.com), support [development containers](https://containers.dev) that can load a docker image and run jupyter notebooks inside the container from within the IDE (no browser required).  `.devcontainer/devcontainer.json` describes a development container for FEniCS-SZ that has been used and tested in VSCode.  Simply select the option to "Reopen in container" after opening the clone of the [repository](https://github.com/cianwilson/fenics-sz) in the IDE.  When first running a notebook select the `dolfinx-env` python environment if prompted.
#
#
# If this method works and provides sufficient access to the repository and its notebooks then there's no need to read any further!  The instructions immediately after this refer to installing the dependencies of FEniCS-SZ using conda.

# %% [markdown]
# ### 2.3 Conda
#
# Aside from using the recommended method through docker, [conda](https://docs.conda.io) provides the easiest way of installing the necessary software packages.  Below we provide some guidance on setting up a conda environment for running the notebooks.
#
# ```{admonition} Xvfb
# Note that we use [pyvista](https://docs.pyvista.org/) for plotting and, on linux machines, this requires [Xvfb](https://x.org/releases/X11R7.7/doc/man/man1/Xvfb.1.xhtml) to be installed on the system independently of the instructions below.
# ```
#
# Assuming a working [conda installation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html), we provide a `conda/environment.yml` file that can be used to install a conda environment that matches the docker installation as closely as possible
# ```bash
#   conda env create -f conda/environment.yml
# ```
# This can then be activated using
# ```bash
#   conda activate fenics-sz
# ```
# before starting jupyter lab with
# ```bash
#   jupyter lab
# ```
# which will display a URL to copy and paste to the web browser (as above) if it doesn't automatically open.
#
# If this method works and provides sufficient access to the repository and its notebooks then there's no need to read any further!  The instructions immediately after this refer to installing the dependencies of FEniCS-SZ from source.

# %% [markdown]
# ### 2.4 Source Installation
#
# If not using docker or conda, a local installation of FEniCSx and its dependencies is necessary, including all of its components
#  * [UFL](https://github.com/FEniCS/ufl)
#  * [Basix](https://github.com/FEniCS/basix)
#  * [FFCx](https://github.com/FEniCS/ffcx)
#  * [DOLFINx](https://github.com/FEniCS/dolfinx)
#
# along with other dependencies, which can be seen in the files `docker/pyproject.toml` and `docker/Dockerfile`.
#
# ```{admonition} Xvfb
# Note that we use [pyvista](https://docs.pyvista.org/) for plotting and, on linux machines, this requires [Xvfb](https://x.org/releases/X11R7.7/doc/man/man1/Xvfb.1.xhtml) to be installed on the system.
# ```
#
# Installation instructions for FEniCSx are available on the [FEniCS project homepage](https://fenicsproject.org/download/). This jupyter book was built with and is known to be compatible with

# %%
import dolfinx
print(f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")

# %% [markdown]
# ## 3. Scripts
#
# FEniCS-SZ is intended to be used through jupyter notebooks however the python modules developed in those notebooks are also installable as a `fenics_sz` python package.  This can then be used in standalone python scripts.  
#
# Install the `fenics_sz` package using
# ```bash
#   cd python
#   pip install -e .
# ```
# where we recommend installing in editable mode using the `-e` flag above so that changes to the notebooks immediately reflect in the installed package.
#
# Several example python scripts are available in subdirectories of the `scripts` directory and can be run using, e.g.
# ```bash
#   cd scripts/background
#   python3 poisson_2d.py 128 -o poisson_2d_128.bp
# ```
# or in parallel using, e.g.
# ```bash
#   cd scripts/background
#   mpiexec -np 2 python3 poisson_2d.py 128 -o poisson_2d_128.bp
# ```
# Documentation about the parameters each script is available using, e.g.
# ```bash
#   cd scripts/background
#   python3 poisson_2d.py -h
# ```
#
# The `scripts` directory only contains examples of how to use the `fenics_sz` package.  It does not include any functionality not already available through the notebooks.
