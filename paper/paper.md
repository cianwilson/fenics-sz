---
title: 'FEniCS-SZ: Two-dimensional modeling of the thermal structure of subduction zones using finite elements'
tags:
  - Python
  - Jupyter
  - geology and geophysics
  - finite element methods
  - benchmarking
authors:
  - name: Cian R. Wilson
    orcid: 0000-0002-4083-6529
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1  # (Multiple affiliations must be quoted)
    equal-contrib: true
  - name: Cameron Seebeck
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Kidus Teshome
    affiliation: 1
    equal-contrib: true
  - name: Nathan Sime
    orcid: 0000-0002-2319-048X
    equal-contrib: true
    affiliation: 1  # (Multiple affiliations must be quoted)
  - name: Peter E. van Keken
    orcid: 0000-0003-0377-8830
    equal-contrib: true
    affiliation: 1  # (Multiple affiliations must be quoted)
affiliations:
 - name: Earth and Planets Laboratory, Carnegie Institution for Science, Washington D.C., United States
   index: 1
   ror: https://ror.org/04jr01610
date: 17 February 2026
bibliography: paper.bib
---

# Summary

`FEniCS-SZ` is a hyper open source software tool that aims to make thermal models of subduction zones as accessible as possible.  Development is exposed through jupyter notebooks, which are accessible online as a jupyter book at [cianwilson.github.io/fenics-sz](https://cianwilson.github.io/fenics-sz) and can be run on any jupyter server.  Parallel scaling is demonstrated in the notebooks.  The software can also be installed as a python module and used in python scripts.  It serves a dual purpose as both a research and teaching tool and was developed as part of two undergraduate research projects.

Suggested editor: Jed Brown

Suggested referees: Jørgen Dokken, Adam Holt, Scott King.

# Statement of need

Plate tectonics on Earth is the surface expression of the slow convective release of heat from its interior.
Subduction zones are found worldwide (see \autoref{fig:map}) and form the location of the return flow of mantle convection.  They are sites of mountain building and significant 
natural hazards in the form of earthquakes and explosive volcanism. 
The depth extent of large and sometimes tsunamogenic earthquakes,
intermediate-depth earthquakes, and melt formation are linked to thermal transitions and 
corresponding thermally activated processes such as metamorphic reactions. 
To understand the short- and long-term evolution of the tectonic and geological processes it is critically important to understand
the thermal structure of subduction zones [@vanKeken2023-a].

![Global locations of subdution zones modeled by FEniCS-SZ [after @Syracuse2010].\label{fig:map}](./images/syracuselocations.pdf)


@Wilson2023 provided an introductory review of the numerical strategy for finding the thermal structure of subduction zones using the finite element method and, in @vanKeken2023-b, updated a global suite of subduction zone models originally proposed by @Syracuse2010 (see \autoref{fig:map}).  They provided access to an open source implementation using `TerraFERMA` [@Wilson2017] and an online archive of simulation inputs and results [@vanKeken2023-c].  However, these results were based on a legacy version of the finite element library `FEniCS` [@Langtangen2016].  `FEniCS-SZ` is a hyper-open-source reimplementation of the subduction zone models as described in @Wilson2023 using the latest version of the `FEniCS` library, `FEniCSx` [@Baratta2023].

In addition to providing a community resource for subduction zone modeling, `FEniCS-SZ` is also intended for classroom use.  It has been developed entirely using jupyter notebooks [@Thomas2016], providing online interactive access to the finite element examples described in @Wilson2023.
These problems progress from implementations of the Poisson and Stokes equations, to a reproduction of mantle convection benchmarks, 
before demonstrating how to implement the fully coupled set of time-dependent equations used in the subduction models.  The resulting code is scalable (thanks to the inherent parallelism of `FEniCSx` [@Baratta2023] and the underlying linear algebra library `PETSc` [@petsc-web-page]) and we provide demonstrations of parallel scaling as part of the tutorial.  This workflow is based on and augments the `FEniCSx` tutorial [@fenicsx], which is itself built on the `FEniCS` Tutorial [@Langtangen2016].

# State of the field

Thermal models of subduction zones that are most useful in the prediction of metamorphic dehydration reactions and their role in seismogenesis and seismic structure,
slab dehydration, arc volcanism, and the long term chemical evolution of the Earth require high numerical resolution, 
faithful gridding of material boundaries (such as the slab surface and oceanic Moho), and the ability to handle velocity discontinuities along the
seismogenic zone and its extension to a coupling depth about 80 km depth. 
Semi-analytical techniques can be used successfully along the shallow plate interface to limited depth (see discussion and references in @vanKeken2019), 
but the effects of the cornerflow with realistic mantle rheology requires the numerical solution of the Stokes and heat equations.
A number of dynamical approaches exist that can be used to trace subduction zone thermal evolution
(@HoltCondit2021, @Gerya2011) but these provide slab evolution models that are difficult to use when 
predicting the thermal structure of present-day subduction zones 
since geometry and convergence parameters such as convergence speed cannot be controlled.
Other workers have provided finite element and finite difference approaches to study the thermal structure 
(e.g. @WadaWang2009; @LeeKing2009; @Lin2010; @ReesJones2018; @vanZelst2023). 
These approaches have shown good comparisons with other codes in a code intercomparison [@vanKeken2008], 
by reproduction of benchmark cases therein,
or in direct intercomparisons [@vanKeken2023-b]. 
Many of these subduction implementations, however, are not readily available as open source software even if they are based on general
open source finite element software.

# Software design

`FEniCS-SZ` is designed to be as accessible as possible.  It is provided for use in three formats:

1. non-interactively online
2. interactively as a set of jupyter notebooks 
3. as an installable python package for use in scripts

Non-interactive use is made possible by publishing the jupyter notebooks online as a jupyter book [@JupyterBook] at [cianwilson.github.io/fenics-sz](https://cianwilson.github.io/fenics-sz).  The notebooks can also be used interactively on any jupyter server.  Online usage is possible through binder [@Binder] however resource limitations often mean that it is better run locally.  We provide docker [@Merkel2014] images and conda [@conda] environments to facilitate this.  The python package is installable using pip [@pip2026].

With the exception of a few utility functions for plotting and meshing, all of the code of `FEniCS-SZ` has been developed and is available and editable through the jupyter notebooks.  These are arranged in four sections

1. introductory material
2. background finite element method tutorials
3. subduction zone thermal problem implementation
4. subduction zone global suite (see \autoref{fig:map}) using the implementation from section 3

We use jupytext [@jupytext] to automatically sync the notebooks with importable python modules for use in subsequent notebooks and scripts.  Although we primarily encourage use through the notebooks several example scripts are provided.

![Simplified benchmark subduction zone problem: (a) domain geometry, (b) low resolution (minimum element size ~2km) example mesh, (c) temperature and velocity solution for benchmark case 2 (minimum element size ~0.5km), (d) example parallel partitioning strategy for 8 processes.\label{fig:domain}](./images/benchmark_tiled.pdf)

The subduction zone geometry is divided into several subdomains (see \autoref{fig:domain}(a)) and several corresponding subproblems.  The temperature is defined and solved for globally while a discontinuity in the velocity solution above the coupling depth requires that we solve the Stokes equations twice, once in each of the "slab" and "wedge" subdomains (the velocity is assumed to be zero in both layers of the crust).  Figure \autoref{fig:domain}(b) shows a low-resolution example of the global mesh, with most cells concentrated near the coupling depth.  Figure \autoref{fig:domain}(c) shows an example temperature solution overlain with glyphs showing the velocities from both the wedge and slab subproblems.

![Parallel behavior of the subduction zone problems with a minimum element size of ~0.5km up to 8 processes. (a) Number of degrees of freedom (DOFs, minimum and maximum represented by error bars) for the global temperature and wedge and slab subdomain velocity solutions.  (b) Total number of ghost DOFs for each subproblem (compared to temperature for a case with no split enforced along the subdomain boundary). Strong scaling averaged over 10 simulations on a dedicated machine for (c) an isoviscous linear rheology (single Stokes solve per subdomain) and (d) a non-linear rheology (Stokes solve per iteration per subdomain).  (i) Wall time for different assembly and solution steps using a split strategy.  (ii) Total (sum of both subdomains) Stokes solve wall times including analysis and factorization steps for a direct strategy using MUMPS and an iterative solver.  (iii) Speed up of the different parts of the Stokes solver compared to ideal scaling.\label{fig:scaling}](./images/strong_scaling.pdf){width="80%"}

To facilitate running in parallel within the notebooks we use ipyparallel [@ipyparallel] (standard MPI commands also work on scripts, no noticeable cost penalty was noted comparing parallel usage in the notebooks with the scripts) and demonstrate scaling (on dedicated machines with sufficient resources) in several notebooks.  For the subduction zone problems we adopt a single "split" domain decomposition strategy for all subproblems.  This divides the domain along the subdomain boundary between the slab and wedge (see \autoref{fig:domain}(d)), allowing us to split the global MPI communicator and solve both subdomain Stokes problems in parallel when more than one MPI process is available.  A single domain decomposition for all problems allows interpolation between the subproblems to remain local to a process.  It comes at the expense of increasing the communication costs for the global temperature solution (see \autoref{fig:scaling}(b), cf. Temperature, ghosts (split) vs. (no split)) but given that this has a much lower overall computational cost than the sum of the Stokes solutions (see \autoref{fig:scaling}(c,d)(i)) we focus on improving the scaling behavior of the Stokes solvers.

Our default solver strategy is to use a direct LU decomposition, as implemented by MUMPS [@MUMPS].  This includes an initial serial analysis step that only needs to be performed once per matrix.  For isoviscous problems (which require only a single Stokes solution per subdomain) this step leads to poor scaling of our direct solver with wall times similar to an iterative strategy (see \autoref{fig:scaling}(c)).  For time-dependent and non-linear rheologies (which require a Stokes solution once per iteration per timestep per subdomain) the cost (in wall time) of a direct approach is much lower than an iterative strategy for the two-dimensional problems considered here.  Furthermore, our split domain decomposition strategy makes scaling comparable to an iterative approach (see \autoref{fig:scaling}(d)).

Our implementation is tested on initiation of a pull request using github workflows.  These take the form of

1. convergence tests for background tutorial problems with analytic or benchmark solutions
2. convergence tests for the subduction zone benchmark of @Wilson2023
3. regression tests for the full global suite of subduction zone models compared to the results of @vanKeken2023-b
4. build tests of the docker image and conda environment
5. build tests of the jupyter book

Failure of any of these tests prevents publication of the jupyter book.

Note that as the jupyter book is built through github workflows we limit the default resolution of the global suite of subduction zone problems to a minimum element size of ~3km.  This should be decreased for use in scientific applications.

# Research impact statement

<!-- Add citations to publications that used vanKeken & Wilson -->
`FEniCS-SZ` implements the numerical strategy outlined in @Wilson2023 and applied in @vanKeken2023-b.  These papers (along with @vanKeken2023-a) have been used extensively in recent literature [e.g., @_]. `FEniCS-SZ` provides additional accessibility to these results and allows users to modify subduction zone geometries and thermal parameters at ease.

Beyond its research application, it was primarily developed as part of an intensive summer research experience by two undergraduates and has been used in practical class room demonstrations [@Cornell].

# AI usage disclosure

AI was not used in the development of `FEniCS-SZ`.

# Acknowledgements

`FEniCS-SZ` was primarily developed by undergraduate summer interns as part of the Research Experience for Undergraduates (REU) programs (SURI, 2023 and EPIIC, 2024) at the Carnegie Institution for Science, sponsored by National Science Foundation (NSF) grant EAR-2244322.  We additionally acknowledge support from NSF grants EAR-1850634 and EAR-202102.

# References
