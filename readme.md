# Overview

Authors: Cian Wilson, Cameron Seebeck, Kidus Teshome, Nathan Sime, Peter van Keken

Welcome to the [_FEniCS Subduction Zone_ Jupyter Book](https://cianwilson.github.io/fenics-sz), an online resource for modeling subduction zones!

This repository was developed by undergraduate interns Kidus Teshome and Cameron Seebeck at the [Carnegie Science Earth & Planets Laboratory](https://epl.carnegiescience.edu).  It is based on [Wilson & van Keken, PEPS, 2023 (II)](http://dx.doi.org/10.1186/s40645-023-00588-6), which is part II of a three part introductory review of the thermal structure of subduction zones by Peter van Keken and Cian Wilson.

Our goal is both to demonstrate how to build kinematic-slab thermal models of subduction zones using the finite element library [FEniCSx](https://fenicsproject.org) and to provide an easily accessible and modifiable source of the global suite of subduction zone models as described in [Wilson & van Keken, PEPS, 2023 (II)](http://dx.doi.org/10.1186/s40645-023-00588-6) and [van Keken & Wilson, PEPS, 2023 (III)](https://doi.org/10.1186/s40645-023-00589-5).  For comparison, the original models used in these papers are also available as open-source repositories on [github](https://github.com/cianwilson/vankeken_wilson_peps_2023) and [zenodo](https://doi.org/10.5281/zenodo.7843967).

# Usage

The easiest way to read this [Jupyter Book](https://jupyterbook.org/) is on the [FEniCS-SZ website](https://cianwilson.github.io/fenics-sz) however all the pages of the book can also be run interactively in a web browser from any jupyter server with the appropriate software packages installed.  Below we start by providing instructions for running the notebooks through [docker](https://docs.docker.com/get-started/), a containerized software environment, for which we supply [pre-built container images](https://github.com/cianwilson/fenics-sz/pkgs/container/fenics-sz).  These can be most easily run online (but with limited computational resources) through [binder](https://mybinder.org/) or locally on any machine with docker installed.  In addition we provide some guidelines for installing the necessary software packages on any jupyter server using [conda](https://docs.conda.io).

## Online Binder
This website is published as a [Jupyter Book](https://jupyterbook.org/). Each page is a Jupyter notebook that can be run interactively in the browser.  To start such an interactive session using [binder](https://mybinder.org/) click the ![Binder symbol](notebooks/images/binder.png)-symbol in the top right corner of the relevant page.  Note that [binder](https://mybinder.org/) may take some time to start while it downloads and starts the docker image.

```{admonition} Interactive Changes
Binder allows users to run notebooks interactively, making changes to the published Jupyter Book.  Note that these changes will be lost once the binder session ends unless the changed files are manually downloaded by selecting them in the Jupyter lab file browser and downloading them to the local machine.
```

```{admonition} Computational Costs
Binder limits the amount of computational resources available.  Extremely high resolution simulations may therefore not be feasible online.
```

## Local Docker
To run the notebooks locally, outside of [binder](https://mybinder.org/), an installation of the FEniCSx is required. We strongly recommend new users do this using [docker](https://www.docker.com/).

Docker is software that uses _images_ and _containers_ to supply virtualized installations of software across different kinds of operating systems (Linux, Mac, Windows).  The first step is to install docker, following the instructions at their [webpage](https://docs.docker.com/get-started/).

Once docker is installed we provide compatible docker images using [github packages](https://github.com/users/cianwilson/packages/container/package/fenics-sz).

```{admonition} Computational Resources
On non-linux operating systems docker limits the computational resources available to the virtualized docker container, which may limit the size of simulations it is possible to run locally.  Modify the docker settings to change these settings and make more resources available.
```

To use these images with this book on a local machine, first (using a terminal) clone the repository and change into that directory
```bash
  git clone -b release https://github.com/cianwilson/fenics-sz.git
  cd fenics-sz
```

### Browser

If running the book in a browser run the following docker command

```bash
  docker run --init --rm -p 8888:8888 --workdir /root/shared -v "$(pwd)":/root/shared ghcr.io/cianwilson/fenics-sz:release
```
The first time this is run it will automatically download the docker image and start Jupyter lab in the docker container on the local machine.  To view the notebooks and modify them locally, copy and paste the URL printed in the terminal into a web-browser.

```{admonition} Updates
`docker run` will only download the docker image the first time it is called.  To get updates to the images run

   docker pull ghcr.io/cianwilson/fenics-sz:release

before calling `docker run`.
```

### Terminal

Alternatively, the image can be used through an interactive terminal by running

```bash
  docker run -it --rm -p 8888:8888 --workdir /root/shared -v "$(pwd)":/root/shared  --entrypoint="/bin/bash" ghcr.io/cianwilson/fenics-sz:release
```
This can be useful, for example to install extra packages in the docker container through the terminal.

Jupyter lab can also be started from within the docker container
```bash
  jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```
again copying and pasting the resulting URL into a web browser to access the notebooks.

## Local Installation

If not using docker a local installation of FEniCSx is necessary, including all of its components
 * [UFL](https://github.com/FEniCS/ufl)
 * [Basix](https://github.com/FEniCS/basix)
 * [FFCx](https://github.com/FEniCS/ffcx)
 * [DOLFINx](https://github.com/FEniCS/dolfinx)

along with other dependencies, which can be seen in the files `docker/pyproject.toml` and `docker/Dockerfile`.

```{admonition} Xvfb
Note that we use [pyvista](https://docs.pyvista.org/) for plotting and, on linux machines, this requires [Xvfb](https://x.org/releases/X11R7.7/doc/man/man1/Xvfb.1.xhtml) to be installed on the system.
```

Installation instructions for FEniCSx are available on the [FEniCS project homepage](https://fenicsproject.org/download/).

### Conda

Aside from using the recommended method through docker, [conda](https://docs.conda.io) provides the easiest way of installing the necessary software packages.  Below we provide some guidance on setting up a conda environment for running the notebooks.

Assuming a working [conda installation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html), we provide a `docker/requirements.txt` file that can be used to install a conda environment matching the docker installation
```bash
  conda create -n fenics-sz python=3.12.3 -c conda-forge --file docker/requirements.txt
```
This can then be activated using
```bash
  conda activate fenics-sz
```
We recommend disabling threading by setting
```bash
  export OMP_NUM_THREADS=1
```
before starting jupyter lab with
```bash
  jupyter lab
```
which will display a URL to copy and paste to the web browser if it doesn't automatically open.

`docker/requirements.txt` contains the strictest list of version requirements to match the docker installation.  This sometimes causes conda to fail to resolve the dependencies so we provide an alternative `docker/requirements_min.txt` file that contains less stringent version restrictions and can be installed and started using
```bash
  conda create -n fenics-sz-min python=3.12.3 -c conda-forge --file docker/requirements_min.txt
  conda activate fenics-sz-min
  export OMP_NUM_THREADS=1
  jupyter lab
```
Note that as this doesn't guarantee the same software versions as used in our docker environment, some changes may occur between the published website output and output produced using this conda environment.

```{admonition} Xvfb
Remember that linux machines require Xvfb to be installed independently of the above instructions for plotting to work.
```

## Acknowledgments

This Jupyter Book is based on the [FEniCSx Tutorial](https://jsdokken.com/dolfinx-tutorial/) by [Jørgen S. Dokken](https://jsdokken.com/), which is an excellent resource for learning how to use [FEniCS](https://fenicsproject.org/) in a similar interactive Jupyter Book.

