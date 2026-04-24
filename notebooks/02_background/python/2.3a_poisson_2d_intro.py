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
# # Poisson Example 2D

# %% [markdown]
# ## General description
#
# We can generalize (and formalize) the description of the Poisson equation 
# using the steady-state heat diffusion equation in multiple dimensions
# \begin{align}
# -\nabla \cdot\left( k \nabla T \right) &= H && \text{in }\Omega
# \end{align}
# $T$ is the temperature solution we are seeking, $k$ is the thermal conductivity, $H$ is a heat source, and $\Omega$ is the domain with boundary $\partial\Omega$.  If $k$ is constant in space we can simplify the equation to
# \begin{align}
# -\nabla^2 T &= h && \text{in }\Omega
# \end{align}
# where $h = \frac{H}{k}$.  

# %% [markdown]
# ### Boundary conditions
#
# We supplement the Poisson equation with some combination of boundary conditions 
# \begin{align}
# T &= g_D && \text{on } \partial\Omega_D \subset \partial\Omega \\
# \nabla T\cdot\hat{\vec{n}} &= g_N && \text{on } \partial\Omega_N \subset \partial\Omega \\
# aT + \nabla T\cdot\hat{\vec{n}} &= g_R && \text{on } \partial\Omega_R \subset \partial\Omega 
# \end{align}
# where $\partial\Omega_D$, $\partial\Omega_N$ and $\partial\Omega_R$ are
# segments of the domain boundary that do not overlap $(\partial\Omega_D \bigcap \partial\Omega_N =\emptyset, \partial\Omega_D \bigcap \partial\Omega_R =\emptyset, \partial\Omega_N \bigcap \partial\Omega_R =\emptyset)$ and that together span the entire boundary $(\partial\Omega_D \bigcup \partial\Omega_N \bigcup \partial\Omega_R = \partial\Omega)$.  The unit outward-pointing normal to the boundary $\partial\Omega$ is denoted by $\hat{\vec{n}}$ and $g_D = g_D(\vec{x}, t)$, $g_N = g_N(\vec{x}, t)$, $g_R = g_R(\vec{x}, t)$ and $a = a(\vec{x}, t)$ are known functions of space and time.  
#
# The first boundary condition is known as a Dirichlet boundary condition and specifies the value of the solution on $\partial\Omega_D$. The second is a Neumann boundary condition and specifies the value of the flux through $\partial\Omega_N$. Finally, the third is a Robin boundary condition, which describes a linear combination of the flux and the solution on $\partial\Omega_R$.

# %% [markdown]
# ### Weak form
#
# The first step in the finite element discretization is to transform the equation into its **weak form**.  This requires multiplying the equation by a test function,  $T_t$,  and integrating over the domain $\Omega$
# \begin{equation}
# -\int_\Omega T_t \nabla^2 T ~dx = \int_\Omega T_t h ~dx
# \end{equation}
# After integrating the left-hand side by parts
# \begin{equation}
# \int_\Omega \nabla T_t \cdot \nabla T ~dx - \int_{\partial\Omega} T_t \nabla T\cdot\hat{\vec{n}}~ds = \int_\Omega T_t h ~dx
# \end{equation}
# we can see that we have reduced the continuity requirements on $T$ by only requiring its first derivative to be bounded across $\Omega$. Integrating by parts also allows Neumann and Robin boundary conditions to be imposed "naturally" through the second integral on the left-hand side since this directly incorporates the flux components across the boundary.  In this formulation, Dirichlet conditions cannot be imposed weakly and are referred to as essential boundary conditions,  that are required of the solution but do not arise naturally in the weak form.  
#
# The weak form therefore becomes: find $T$ such that $T = g_D$ on $\partial\Omega_D$ and
# \begin{equation}
# \int_\Omega \nabla T_t \cdot \nabla T ~dx - \int_{\partial\Omega_N} T_t g_N ~ds - \int_{\partial\Omega_R} T_t \left(g_R - aT\right)~ds = \int_\Omega T_t h ~dx
# \end{equation}
# for all $T_t$ such that $T_t = 0$ on $\partial\Omega_D$.  

# %% [markdown]
# ### Discretization
#
# The weak and strong forms of the problem are equivalent so long as the solution is sufficiently smooth.  We make our first approximation by, instead of seeking $T$ such that $T = g_D$ on $\partial\Omega_D$, seeking the discrete trial function $\tilde{T}$ such that $\tilde{T} = g_D$ on $\partial\Omega_D$ where
# \begin{equation}
# T \approx \tilde{T} = \sum_j \phi_j T_j
# \end{equation}
# for all test functions $\tilde{T}_t$ where
# \begin{equation}
# T_t \approx \tilde{T}_t = \sum_i \phi_i T_{ti}
# \end{equation}
# noting again that $\tilde{T}_t = 0$ on $\partial\Omega_D$.
#
# $\phi_j$ are the finite element shape functions. Assuming these are continuous across elements of the mesh, $\tilde{T}$ and $\tilde{T}_t$ can be substituted into the weak form to yield
#
# \begin{multline}
# \sum_i\sum_j T_{ti}T_j\sum_k  \int_{e_k} \nabla \phi_i \cdot \nabla \phi_j ~dx 
#  + \sum_i\sum_j T_{ti}T_j \sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i a\phi_j ~ds
# \\- \sum_i T_{ti} \sum_k \int_{\partial e_k \cap {\partial\Omega_N}} \phi_i g_N ~ds 
# - \sum_i T_{ti} \sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i g_R 
# = \sum_i T_{ti} \sum_k \int_{e_k} \phi_i h ~dx
# \end{multline}
#
# where we are integrating over the whole domain by summing the integrals over all the elements  $e_k$ ($\int_\Omega dx$=$\sum_k\int_{e_k} dx$).  Note that in practice, because the shape functions are zero over most of the domain, only element integrals with non-zero values need be included in the summation.  The element boundaries, $\partial e_k$, are only of interest (due to the assumed continuity of the shape functions between the elements) if they either intersect with $\partial\Omega_N$, $\partial e_k \cap {\partial\Omega_N}$, or $\partial\Omega_R$, $\partial e_k \cap {\partial\Omega_R}$.  Since the solution of the now discretized weak form should be valid for all $\tilde{T}_t$ we can drop $T_{ti}$
#
# \begin{multline}
# \sum_jT_j\sum_k  \int_{e_k} \nabla \phi_i \cdot \nabla \phi_j ~dx 
#  + \sum_jT_j\sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i a \phi_j ~ds
# \\- \sum_k \int_{\partial e_k \cap {\partial\Omega_N}} \phi_i g_N ~ds 
# - \sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i g_R~ds 
# = \sum_k \int_{e_k} \phi_i h ~dx
# \end{multline}

# %% [markdown]
# ### Matrix equation
#
# The discrete weak form represents a matrix-vector system
# \begin{equation}
# {\bf S} {\bf u} = {\bf f}
# \end{equation}
# with
#
# \begin{align}
# {\bf S} &= S_{ij} = \sum_k\int_{e_k} \nabla \phi_i \cdot \nabla \phi_j ~dx + \sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i a\phi_j ~ds  \\
# {\bf f} &= f_i = \sum_k \int_{e_k} \phi_i h ~dx + \sum_k \int_{\partial e_k \cap {\partial\Omega_N}} \phi_i g_N ~ds 
# + \sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i g_R~ds \\
# {\bf u} &= {\bf T} = T_j 
# \end{align}
#
# The compact support of the shape functions $\phi_{(i,j)}$, which limits their nonzero values to the elements immediately neighboring DOF $i$ or $j$, means that the integrals can be evaluated efficiently by only considering shape functions associated with an element $e_k$.  It also means that the resulting matrix ${\bf S}$ is sparse, with most entries being zero.
#
# In the [next page](./2.3b_poisson_2d.ipynb) we will implement this solution strategy for a specific example of the Poisson problem in 2D.

# %% [markdown]
#
