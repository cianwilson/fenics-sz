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

# %% [markdown]
# # Blankenbach Thermal Convection Example

# %% [markdown]
# ## Description
#
# We are going to explore solving the equations governing a buoyancy-driven convection in a unit square domain 
# following the steady-state mantle convection benchmarks from [Blankenbach et al. (1989)](https://doi.org/10.1111/j.1365-246X.1989.tb05511.x).  This example allows us to couple a steady-state advection-diffusion equation for temperature to the Stokes and mass conservation (incompressibility constraint) equations we have already discussed. This also provides an example of solving a nonlinearly coupled system and will show how we can test a model for which no analytical solution exists.
#
# <img src="images/blankenbachgeometry.png" width="500" title="Blankenbach geometry and boundary conditions">
#
# *Figure 2.5.1 Blankenbach thermal convection geometry and boundary conditions.*
#
# The flow in the square is driven by heating from below and cooling from above (see Figure 2.5.1).
# We solve 
# \begin{align}
#     - \nabla\cdot\left(2\eta\frac{\nabla\vec{v} + \nabla\vec{v}^T}{2}\right) + \nabla P &= -\textrm{Ra}~T \hat{\vec{g}} && \text{in } \Omega  \\
#     \nabla \cdot \vec{v} &= 0  && \text{in } \Omega 
# \end{align}
# where variable rheology is permitted through the inclusion of the viscosity, $\eta = \eta(T)$, and the buoyancy force vector, $-RaT\hat{\vec{g}}$, uses the temperature $T$, nondimensional Rayleigh number, Ra, and unit vector in the direction of gravity, $\hat{\vec{g}}$. 
# The Rayleigh number arises from the nondimensionalization of the governing equations and is a ratio that balances
# factors that enhance convective vigor (e.g., thermal expansivity, gravity) with those that retard convective vigor 
# (e.g., viscosity). In general, convective vigor increases with increasing Ra when it exceeds a critical value for the Rayleigh number (see, e.g., [Turcotte & Schubert, 2002](https://doi.org/10.1017/CBO9780511843877)),
#
# Under the assumption of steady state ($\tfrac{\partial T}{\partial t}$=0), constant material properties $(k=1)$, and zero internal heating $(H=0)$ the heat equation becomes
# \begin{align}
# \vec{v} \cdot \nabla T &= \nabla^2 T  && \text{in } \Omega  
# \end{align}

# %% [markdown]
# ## Discretization
#
# We discretize the trial function spaces for temperature ($T\approx\tilde{T}$), velocity 
# ($\vec{v}\approx\tilde{\vec{v}}$), and pressure ($P\approx\tilde{P}$) as before using 
# \begin{align}
#  T \approx \tilde{T} &= \sum_j \phi_j T_j \\
#  \vec{v} \approx \tilde{\vec{v}} &= \sum_j \omega^k_j v^k_j  \\
#  P \approx \tilde{P} &= \sum_j  \chi_j  P_j
# \end{align}
# with similarly defined discrete test functions, $\tilde{T}_t$, $\tilde{\vec{v}}_t$ and $\tilde{P}_t$. For the Stokes equation we use the P*n+1*P*n*, *n* > 0, Taylor-Hood Lagrange element pair for the shape functions $(\vec{\omega}_j,\chi_j)$ (as in the [Batchelor example](./2.4a_batchelor_intro.ipynb)) and P*n*, *n* > 0, elements for the heat equation ($\phi_j$). 

# %% [markdown]
# ## Boundary conditions
#
# For the Stokes problem we assume free-slip boundaries. These are formed by the combination
# of a Dirichlet boundary condition of zero normal velocity 
# ($v_n=\tilde{\vec{v}}\cdot{\hat{\vec{n}}}=0$) and a Neumann zero tangential stress condition
# ($\tau_t=( {\boldsymbol\tau} \cdot \hat{\vec{n}}) \cdot \hat{\vec{t}}=0$). Here, $\hat{\vec{n}}$ is the unit normal to the boundary, $\hat{\vec{t}}$ is the unit tangent on the boundary
# (see Figure 2.5.1), and $\boldsymbol\tau$ is the deviatoric stress tensor
# \begin{equation}
# {\boldsymbol\tau} = 2\eta \frac{\nabla\tilde{\vec{v}} + \nabla\tilde{\vec{v}}^T}{2} = 2\eta
# \begin{bmatrix}
# \frac{\partial \tilde{v}_x}{\partial x} & \frac{1}{2} \left( \frac{\partial \tilde{v}_x}{\partial y} + \frac{\partial \tilde{v}_y}{\partial x} \right) \\
# \frac{1}{2} \left( \frac{\partial \tilde{v}_x}{\partial y} + \frac{\partial \tilde{v}_y}{\partial x} \right) & \frac{\partial \tilde{v}_y}{\partial y}
# \end{bmatrix}
# \end{equation}
# As in the [Batchelor example](./2.4a_batchelor_intro.ipynb), this set of velocity boundary conditions results in a pressure null space.  Following our [earlier tests](./2.4e_batchelor_nest_parallel.ipynb) we either choose to impose the extra condition that $\tilde{P}(0,0) = 0$ or we remove the null space from the solution at each iteration to force a unique solution to exist.
#
# For the heat equation the side boundaries are insulating (imposed by the Neumann boundary condition $\partial{\tilde{T}}/\partial{x}=0$) with Dirichlet boundary conditions for the top boundary ($\tilde{T}=0$) and bottom boundary ($\tilde{T}=1$).

# %% [markdown]
# ## Nonlinearity
#
# Unlike the previous examples, which were linear problems of their solution variables, the thermal convection equations are nonlinear.  For an isoviscous or temperature dependent, $\eta$=$\eta(\vec{v})$, rheology the equations are individually linear but the buoyancy contribution and possible temperature dependence of viscosity in the momentum equation and the advective component in the heat equation mean that the coupled system of equations is nonlinear, with $\vec{v}$ depending on $T$ and vice versa.  For non-Newtonian rheologies, where $\eta$=$\eta(\vec{v})$, the momentum equation itself becomes nonlinear too.  Because of this, rather than immediately defining the weak forms of the linear operator $\mathbf{S}$ we begin by considering the weak form of the nonlinear residual, ${\bf r}$.  This is derived in exactly the same manner as before by multiplying the momentum equation by $\vec{v}_t$, the continuity equation by $P_t$ and the heat equation by $T_t$, discretizing the functions, integrating (by parts) over the domain $\Omega$,  dropping the resulting surface integrals (either to enforce the weak boundary conditions or because they are unnecessary due to the essential boundary conditions), 
# and defining the discrete weak forms of the residuals as
# \begin{align}
# {\bf r}_{\vec{v}} &= r_{\vec{v}_{i_1}} :=  \sum_k \int_{e_k} \left[ \left(\frac{\nabla\vec{\omega}_{i_1} + \nabla\vec{\omega}_{i_1}^T}{2}\right):2\eta\left(\frac{\nabla\tilde{\vec{v}} + \nabla\tilde{\vec{v}}^T}{2}\right) - \nabla \cdot \vec{\omega}_{i_1} \tilde{P} + \vec{\omega}_{i_1}\cdot \vec{g}~\textrm{Ra}~\tilde{T} \right] dx = 0 \\
# {\bf r}_P &= r_{P_{i_2}} := -  \sum_k \int_{e_k} \chi_{i_2} \nabla \cdot \tilde{\vec{v}} dx = 0 \\
# {\bf r}_T &= r_{T_{i_3}} := \sum_k \int_{e_k} \left[ \phi_{i_3} \tilde{\vec{v}}\cdot\nabla\tilde{T} + \nabla \phi_{i_3} \cdot \nabla\tilde{T} \right] dx = 0 
# \end{align}
# Here ${\bf r} = \left({\bf r}_{\vec{v}}, {\bf r}_P, {\bf r}_T\right)^T = \left(r_{\vec{v}_{i_1}}, r_{P_{i_2}}, r_{T_{i_3}}\right)^T$ is a residual vector, the root of which must be found in order to find an approximate solution to our equations.  Finding the exact root is not generally possible.  Instead we aim to find
# ${\bf r}={\bf 0}$ within some tolerance. For example we can use an $L^2$ norm and an absolute $||{\bf r}||_2 = \sqrt{{\bf r}\cdot{\bf r}} < \epsilon_\text{atol}$, or relative, $\frac{||{\bf r}||_2}{||{\bf r}^0||_2} = \frac{\sqrt{{\bf r}\cdot{\bf r}}}{\sqrt{{\bf r}^0\cdot{\bf r}^0}} < \epsilon_\text{rtol}$, tolerance, where ${\bf r}^0$ is the residual evaluated using the initial guess at the solution.  Two commonly used approaches to approximately finding the residual root are Newton's method and Picard's method.  Convergence of the Newton iteration method depends on having a good initial guess, which is not always possible, especially when solving steady-state problems so, for simplicity we only implement Picard's method here and briefly outline it below.

# %% [markdown]
# ### Picard's method
#
# Picard's method splits the equations into multiple linearized subsets and solves them sequentially and repeatedly, updating the nonlinear terms at each iteration, until convergence is achieved.  
#
# The thermal convection equations can be split into two systems, the first for the Stokes system
# \begin{align}
# {\bf S}_s &= \left(\begin{array}{cc}{\bf K}_s & {\bf G}_s \\ {\bf D}_s & {\bf 0}\end{array}\right) \\
# {\bf K}_s &= K_{s_{i_1j_1}} = \sum_k\int_{e_k} \left(\frac{\nabla\vec{\omega}_{i_1} + \nabla\vec{\omega}_{i_1}^T}{2}\right):2\eta \left(\frac{\nabla\vec{\omega}_{j_1} + \nabla\vec{\omega}_{j_1}^T}{2}\right) dx \\
# {\bf G}_s &= G_{s_{i_1j_2}} = - \sum_k \int_{e_k} \nabla \cdot \vec{\omega}_{i_1} \chi_{j _2} dx \\
# {\bf D}_s &= D_{s_{i_2j_1}} = -  \sum_k \int_{e_k} \chi_{i_2} \nabla \cdot \vec{\omega}_{j_1} dx \\
# {\bf u}_s &= \left({\bf v}, {\bf P}\right)^T = \left(\vec{v}_{j_1}, P_{j_2}\right)^T \\
# {\bf f}_s &= f_{s_{i_1}} = -\sum_k \int_{e_k} \vec{\omega}_{i_1}\cdot\hat{\vec{g}}~\textrm{Ra}~\tilde{T} dx
# \end{align}
# and the second for the temperature equation
# \begin{align}
# {\bf S}_T &= S_{T_{ij}} = \sum_k\int_{e_k} \left(\phi_i \tilde{\vec{v}}\cdot\nabla \phi_j + \nabla\phi_i\cdot\nabla\phi_j\right) dx \\
# {\bf u}_T &= {\bf T} = T_{j} \\
# {\bf f}_T &= f_{T_{i}} = 0 
# \end{align}
#
# The full system solution vector remains ${\bf u}=\left({\bf u}_s, {\bf u}_T\right)^T=\left({\bf v}, {\bf P}, {\bf T}\right)^T$ and the best guess at the solution is ${\bf u}^i$.  ${\bf S}_s({\bf u}^i){\bf u}_s^{i+1}={\bf f}_s({\bf u}_T^i)$ is solved for ${\bf u}_s^{i+1}$, which is used to update ${\bf u}$ such that ${\bf u}_s^{i+1}\rightarrow{\bf u}_s^{i}$ before solving ${\bf S}_T({\bf u}^i){\bf u}_T^{i+1}={\bf 0}$ for an updated solution for temperature, ${\bf u}_T^{i+1}$.  This iteration is repeated until ${\bf r} = {\bf 0}$ in some norm and to some tolerance.
#
# Convergence of nonlinear methods is not guaranteed.  Various methods are available for solutions that do not converge.  These include finding a better initial guess (e.g., a solution from a case with lower convective vigor), "relaxing" the solution by only applying a partial update at each iteration, or linearizing terms.  It should also be noted that, if applied to the linear problems discussed in previous sections, any nonlinear iteration should converge in a single iteration.
#
# In the cases that follow we use a harmonic perturbation to the conductive state $T(x,y)=1-y+0.1 \cos \pi x \sin \pi y$ as an initial guess for temperature.  We apply a default relaxation factor of 0.8 to the Picard iteration to improve convergence. By default both are solved to a nonlinear relative tolerance, $\epsilon_\text{rtol}$, of $5 \times 10^{-6}$ or an absolute tolerance, $\epsilon_\text{atol}$, of $5 \times 10^{-9}$.  
#

# %% [markdown]
# ## Diagnostics
#
# We do not have an analytical solution available for the thermal convection equations.  Instead we compare our results to published benchmark results. To do this we focus on two measures of convective vigor. The first is the Nusselt number Nu which is the
# integrated nondimensional surface heatflow
# \begin{equation}
# \textrm{Nu} ~=~  - \int_0^1 \frac{\partial T}{\partial y}(x,y=1) dx
# \end{equation}
# The second is the root-mean-square velocity $V_\text{rms}$ defined as 
# \begin{equation}
# V_\text{rms} ~=~ \sqrt{ 
# \frac{\int_\Omega \vec{v}\cdot\vec{v} dx}{\int_\Omega dx}
# }
# \end{equation}
#
# | case | Ra    | $\eta$                  | Nu (BB)       | $V_\text{rms}$ (BB) | Nu (WvK)        | $V_\text{rms}$ (WvK) |
# |------|---------|--------------------------|------------|------------------|-------------|--------------------|
# | 1a   | $10^4$ | 1                        | 4.884409  | 42.864947       | 4.88440907 | 42.8649484        |
# | 1b   | $10^5$ | 1                        | 10.534095 | 193.21454       | 10.53404   | 193.21445         |
# | 1c   | $10^6$ | 1                        | 21.972465 | 833.98977       | 21.97242   | 833.9897          |
# | 2a   | $10^4$  | $e^{-\ln(10^3) T}$ | 10.0660   | 480.4334        | 10.06597   | 480.4308          |
#
# *Table 2.5.1 Best values from [Blankenbach et al. (1989)](https://doi.org/10.1111/j.1365-246X.1989.tb05511.x) (BB) and averaged extrapolated values from [Wilson & van Keken (2023)](http://dx.doi.org/10.1186/s40645-023-00588-6) (WvK) for Nu and $V_\text{rms}$*
#
# Table 9 in [Blankenbach et al. (1989)](https://doi.org/10.1111/j.1365-246X.1989.tb05511.x) and Table 1 in [Wilson & van Keken (2023)](http://dx.doi.org/10.1186/s40645-023-00588-6) specify best estimates for various quantities of the benchmark. We will focus on Nu and $V_\text{rms}$ (see Table 2.5.1) and show results in the [next notebook](./2.5b_blankenbach.ipynb) for the steady-benchmarks 1a-1c (isoviscous, $\eta$=$1$, with Ra increasing from $10^4$ to $10^6$) and benchmark 2a which has Ra $=10^4$ and a temperature-dependent viscosity $\eta({T})=\exp \left( - bT\right)$ with $b=\ln(10^3)$.

# %% [markdown]
#
