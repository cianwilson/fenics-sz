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

# %%
from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
import ufl
import numpy as np

# create mesh and define function space
mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = df.fem.functionspace(mesh, ("Lagrange", 1))

# define the location of the boundary condition, x=0 and x=1
def boundary(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))
boundary_dofs = df.fem.locate_dofs_geometrical(V, boundary)

# specify the value and define a Dirichlet boundary condition (bc)
u0 = df.fem.Constant(mesh, df.default_scalar_type(0.0)) 
bc = df.fem.dirichletbc(u0, boundary_dofs, V)

# define variational problem
u_a = ufl.TrialFunction(V) 
u_t = ufl.TestFunction(V)
f = df.fem.Function(V)
f.interpolate(lambda x: 10*np.exp(-((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / 0.02))
g = df.fem.Function(V)
g.interpolate(lambda x: np.sin(5*x[0]))
a = ufl.inner(ufl.grad(u_t), ufl.grad(u_a))*ufl.dx
L = u_t*f*ufl.dx + u_t*g*ufl.dx

# compute solution
problem = df.fem.petsc.LinearProblem(a, L, bcs=[bc], 
                            petsc_options={"ksp_type": "preonly",
                                        "pc_type": "lu",
                                        "pc_factor_mat_solver_type": "mumps"})
u_i = problem.solve()

# save solution to file
with df.io.VTXWriter(mesh.comm, "poisson.bp", [u_i]) as vtx:
    vtx.write(0.0)

# %%
