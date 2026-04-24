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
# # Background
#
# The equations we need to solve to make predictions of the thermal structure of subduction zones are derived from the fundamental equations governing the conservation of mass, momentum, and thermal energy. The conservation of mass and momentum lead, under the assumption of an infinite-Prandtl number "Boussinesq" incompressible material, to the nondimensional Stokes equation and the condition of incompressibility
# \begin{align}
#     - \nabla\cdot\left(2\eta\frac{\nabla\vec{v} + \nabla\vec{v}^T}{2}\right) + \nabla P ~&=~ \vec{f}_B\\
#     \nabla \cdot \vec{v} ~&=~ 0
# \end{align}
# Given a viscosity, $\eta$, and a buoyancy force, $\vec{f}_B$, that
# can depend on temperature and composition,
# the Stokes equation balances viscous, pressure, and buoyancy forces. 
# Further imposition of the incompressibility constraint 
# allows us to find the velocity, $\vec{v}$, and pressure, $P$. 
# The conservation of thermal energy leads to the nondimensional heat advection-diffusion
# equation
# \begin{equation}
# \rho c_p\left( \frac{\partial T}{\partial t} + \vec{v} \cdot \nabla T \right) ~=~ \nabla \cdot \left( k \nabla T \right) + H
# \end{equation}
# which, given the density, $\rho$, heat capacity, $c_p$, and thermal conductivity, $k$, balances the transport of heat by diffusion and advection with heat production, $H$.  
#
# The heat equation can be modeled to be stationary $\left(\frac{\partial T}{\partial t}=0\right)$ and the Stokes equation can be nonlinear due to the dependence of the viscosity on stress. The Stokes equation and the incompressibility constraint are generally nonlinearly coupled with the heat advection-diffusion equation.  In this section, rather than immediately solving the full nonlinear set of equations, we will provide examples of how to solve these equations one by one, under various simplifying assumptions, before embarking on a fully coupled problem.
#
# We will start with a simple worked-out example of a [1D Poisson equation](./2.2a_poisson_1d_intro.ipynb) which is arguably the simplest form of the heat equation under the assumption of zero velocity, which also eliminates the mass (incompressibility constraint) and momentum (Stokes) equations entirely.  This example includes a discussion of the generation of finite element "shape functions," the construction of the matrix-vector system, solution on a coarse mesh, comparisons between linear and quadratic elements, and convergence tests. This section is particularly intended for those new to finite element methodology and nomenclature.
#
# This finite element introduction is followed by the extension of the [Poisson heat-diffusion problem to more than one dimension](./2.3a_poisson_2d_intro.ipynb) and the solution of the linear Stokes equation for a traditional [cornerflow problem](./2.4a_batchelor_intro.ipynb), neglecting temperature effects. We then combine the heat and Stokes equation in coupled problems using a standard [mantle convection](./2.5a_blankenbach_intro.ipynb) benchmark before focusing on simplified models of [subduction zones](../03_sz_problems/3.1_sz_intro.ipynb) in the next section.  Unless explicitly mentioned otherwise we will assume in all examples in this section that the equations are in nondimensional form.

# %% [markdown]
#
