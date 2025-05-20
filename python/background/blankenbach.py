#!/usr/bin/env python
# coding: utf-8

from mpi4py import MPI
import gmsh
import dolfinx as df
import dolfinx.fem.petsc
import numpy as np
import ufl
import matplotlib.pyplot as pl
import basix
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))
import utils
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
if __name__ == "__main__":
    output_folder = pathlib.Path(os.path.join(basedir, "output"))
    output_folder.mkdir(exist_ok=True, parents=True)


def create_transfinite_square(comm, nx, ny, beta=1):
    """
    A python function to create a mesh of a square domain with refinement
    near the top and bottom boundaries, depending on the value of coeff.
    Parameters:
    * comm  - MPI comm used to distribute the mesh in parallel
    * nx    - number of cells in the horizontal x direction
    * ny    - number of cells in the vertical y direction
    * beta  - beta mesh refinement coefficient, < 1 refines the mesh at 
              the top and bottom boundaries (defaults to  1, no refinement]
    """
    
    if not gmsh.is_initialized(): gmsh.initialize()

    gmsh.model.add("square")
    # make gmsh quieter
    gmsh.option.setNumber('General.Verbosity', 3)

    lc = 1e-2
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 1, 0, lc, 4)
    gmsh.model.geo.addLine(1, 4, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(1, 2, 3)
    gmsh.model.geo.addLine(4, 3, 4)
    gmsh.model.geo.addCurveLoop([3, 2, -4, -1], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.mesh.setTransfiniteCurve(1, ny+1, "Bump", beta)
    gmsh.model.geo.mesh.setTransfiniteCurve(2, ny+1, "Bump", beta)
    gmsh.model.geo.mesh.setTransfiniteCurve(3, nx+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(4, nx+1)

    gmsh.model.geo.mesh.setTransfiniteSurface(1, "Left")

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [1], 1)
    gmsh.model.addPhysicalGroup(1, [2], 2)
    gmsh.model.addPhysicalGroup(1, [3], 3)
    gmsh.model.addPhysicalGroup(1, [4], 4)
    gmsh.model.addPhysicalGroup(2, [1], 1)

    if comm.rank == 0:
        gmsh.model.mesh.generate(2)

    return df.io.gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)[0]


def stokes_function(mesh, pp=1):
    """
    A python function to return the Stokes finite element function.
    Parameters:
    * mesh - the mesh
    * pp   - polynomial order of the pressure solution (defaults to 1)
    """
    # Define velocity, pressure and temperature elements
    v_e = basix.ufl.element("Lagrange", mesh.basix_cell(), pp+1, shape=(mesh.geometry.dim,))
    p_e = basix.ufl.element("Lagrange", mesh.basix_cell(), pp)

    # Define the mixed element of the coupled velocity and pressure
    vp_e = basix.ufl.mixed_element([v_e, p_e])

    # Define the mixed velocity-pressure function space
    V_vp = df.fem.functionspace(mesh, vp_e)

    # Define the finite element functions for the velocity and pressure functions
    vp = df.fem.Function(V_vp)

    return vp

def temperature_function(mesh, pT=1):
    """
    A python function to return the temperature finite element function.
    Parameters:
    * mesh - the mesh
    * pT   - polynomial order (defaults to 1)
    """
    # Define velocity, pressure and temperature elements
    T_e = basix.ufl.element("Lagrange", mesh.basix_cell(), pT)

    # Define the temperature function space
    V_T  = df.fem.functionspace(mesh, T_e)

    # Define the finite element function for the temperature and initialize it
    # with the initial guess
    T = df.fem.Function(V_T)
    T.interpolate(lambda x: 1.-x[1] + 0.2*np.cos(x[0]*np.pi)*np.sin(x[1]*np.pi))

    return T


def stokes_bcs(vp):
    """
    A python function to return a list of boundary conditions on the Stokes problem.
    Parameters:
    * vp - the velocity-pressure finite element function
    """
    # Grab the velocity-pressure function space
    V_vp = vp.function_space
    
    # Define velocity and pressure sub function spaces
    V_v,  _ = V_vp.sub(0).collapse()
    V_vx, _ = V_v.sub(0).collapse()
    V_vy, _ = V_v.sub(1).collapse()
    V_p,  _ = V_vp.sub(1).collapse()

    # Declare a list of boundary conditions for the Stokes problem
    bcs = []

    # Define the location of the left and right boundary and find the x-velocity DOFs
    def boundary_leftandright(x):
        return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))
    dofs_vx_leftright = df.fem.locate_dofs_geometrical((V_vp.sub(0).sub(0), V_vx), boundary_leftandright)
    # Specify the velocity value and define a Dirichlet boundary condition
    zero_vx = df.fem.Function(V_vx)
    zero_vx.x.array[:] = 0.0
    bcs.append(df.fem.dirichletbc(zero_vx, dofs_vx_leftright, V_vp.sub(0).sub(0)))

    # Define the location of the top and bottom boundary and find the y-velocity DOFs
    def boundary_topandbase(x):
        return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))
    dofs_vy_topbase = df.fem.locate_dofs_geometrical((V_vp.sub(0).sub(1), V_vy), boundary_topandbase)
    zero_vy = df.fem.Function(V_vy)
    zero_vy.x.array[:] = 0.0
    bcs.append(df.fem.dirichletbc(zero_vy, dofs_vy_topbase, V_vp.sub(0).sub(1)))

    # Define the location of the lower left corner of the domain and find the pressure DOF there
    def corner_lowerleft(x):
        return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))
    dofs_p_lowerleft = df.fem.locate_dofs_geometrical((V_vp.sub(1), V_p), corner_lowerleft)
    # Specify the arbitrary pressure value and define a Dirichlet boundary condition
    zero_p = df.fem.Function(V_p)
    zero_p.x.array[:] = 0.0
    bcs.append(df.fem.dirichletbc(zero_p, dofs_p_lowerleft, V_vp.sub(1)))

    return bcs

def temperature_bcs(T):
    """
    A python function to return a list of boundary conditions on the temperature problem.
    Parameters:
    * T - the temperature finite element function
    """
    # Grab the temperature function space
    V_T = T.function_space
    
    # Declare a list of boundary conditions for the temperature problem
    bcs = []

    # Define the location of the top boundary and find the temperature DOFs
    def boundary_top(x):
        return np.isclose(x[1], 1)
    dofs_T_top = df.fem.locate_dofs_geometrical(V_T, boundary_top)
    zero_T = df.fem.Function(V_T)
    zero_T.x.array[:] = 0.0
    bcs.append(df.fem.dirichletbc(zero_T, dofs_T_top))

    # Define the location of the base boundary and find the temperature DOFs
    def boundary_base(x):
        return np.isclose(x[1], 0)
    dofs_T_base = df.fem.locate_dofs_geometrical(V_T, boundary_base)
    one_T = df.fem.Function(V_T)
    one_T.x.array[:] = 1.0
    bcs.append(df.fem.dirichletbc(one_T, dofs_T_base))

    return bcs


def stokes_weakforms(vp, T, Ra, b=None):
    """
    A python function to return the weak forms for the Stokes problem.
    By default this assumes an isoviscous rheology but supplying b allows 
    a temperature dependent viscosity to be used.
    Parameters:
    * vp - the velocity-pressure finite element function
    * T  - the temperature finite element function
    * b  - temperature dependence of viscosity (defaults to isoviscous)
    """
    # Grab the velocity-pressure function space and the mesh
    V_vp = vp.function_space
    mesh = V_vp.mesh
    
    # Define extra constants
    Ra_c = df.fem.Constant(mesh, df.default_scalar_type(Ra))
    gravity = df.fem.Constant(mesh, df.default_scalar_type((0.0,-1.0)))
    eta = 1
    if b is not None: 
        b_c  = df.fem.Constant(mesh, df.default_scalar_type(b))
        eta = ufl.exp(-b_c*T)

    # Define the velocity and pressure test functions
    v_t, p_t = ufl.TestFunctions(V_vp)

    # Define the velocity and pressure trial functions
    v_a, p_a = ufl.TrialFunctions(V_vp)

    # Define the integrals to be assembled into the stiffness matrix for the Stokes system
    K = ufl.inner(ufl.sym(ufl.grad(v_t)), 2*eta*ufl.sym(ufl.grad(v_a)))*ufl.dx
    G = -ufl.div(v_t)*p_a*ufl.dx
    D = -p_t*ufl.div(v_a)*ufl.dx
    S = K + G + D

    # Define the integral to the assembled into the forcing vector for the Stokes system
    f = -ufl.inner(v_t, gravity)*Ra_c*T*ufl.dx

    return S, f

def temperature_weakforms(vp, T):
    """
    A python function to return the weak forms for the temperature problem.
    Parameters:
    * vp - the velocity-pressure finite element function
    * T  - the temperature finite element function
    """
    # Grab the temperature function space, mesh and the velocity
    V_T = T.function_space
    mesh = V_T.mesh
    v = vp.sub(0)
    
    # Define the temperature test function
    T_t = ufl.TestFunction(V_T)

    # Define the temperature trial function
    T_a = ufl.TrialFunction(V_T)

    # Define the integrals to be assembled into the stiffness matrix for the temperature system
    S = (T_t*ufl.inner(v, ufl.grad(T_a)) + ufl.inner(ufl.grad(T_t), ufl.grad(T_a)))*ufl.dx

    # Define the integral to the assembled into the forcing vector for the temperature system
    # which in this case is just zero
    f = df.fem.Constant(mesh, df.default_scalar_type(0.0))*T_t*ufl.dx

    return S, f


def solve_blankenbach(Ra, ne, pp=1, pT=1, b=None, beta=1,
                      alpha=0.8, rtol=5.e-6, atol=5.e-9, maxits=50, 
                      petsc_options_s=None, petsc_options_T=None, verbose=True):
    """
    A python function to solve two-dimensional thermal convection 
    in a unit square domain.  By default this assumes an isoviscous rheology 
    but supplying b allows a temperature dependent viscosity to be used.
    Parameters:
    * Ra      - the Rayleigh number
    * ne      - number of elements in each dimension
    * pp      - polynomial order of the pressure solution (defaults to 1)
    * pT      - polynomial order of the temperature solutions (defaults to 1)
    * b       - temperature dependence of viscosity (defaults to isoviscous)
    * beta    - beta distribution parameter for mesh refinement, 
                <1 refines the mesh at the top and bottom (defaults to 1, no refinement)
    * alpha   - nonlinear iteration relaxation parameter (defaults to 0.8)
    * rtol    - nonlinear iteration relative tolerance (defaults to 5.e-6)
    * atol    - nonlinear iteration absolute tolerance (defaults to 5.e-9)
    * maxits  - maximum number of nonlinear iterations (defaults to 50)
    * petsc_options_s - a dictionary of petsc options to pass to the Stokes solver 
                        (defaults to an LU direct solver using the MUMPS library)
    * petsc_options_T - a dictionary of petsc options to pass to the temperature solver 
                        (defaults to an LU direct solver using the MUMPS library)
    * verbose - print convergence information (defaults to True)
    """

    # Set the default PETSc solver options if none have been supplied
    # opts = PETSc.Options()
    
    # prefix_s = "stokes"
    if petsc_options_s is None:
        petsc_options_s = {"ksp_type": "preonly", \
                           "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps"}
    # opts.prefixPush(prefix_s)
    # for k, v in petsc_options_s.items(): opts[k] = v
    # opts.prefixPop()

    # prefix_T = "temperature"
    if petsc_options_T is None:
        petsc_options_T = {"ksp_type": "preonly", \
                           "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps"}
    # opts.prefixPush(prefix_T)
    # for k, v in petsc_options_T.items(): opts[k] = v
    # opts.prefixPop()

    # Describe the domain (a unit square)
    # and also the tessellation of that domain into ne
    # squares in each dimension, which are
    # subdivided into two triangular elements each
    with df.common.Timer("Blankenbach Mesh"):
        mesh = create_transfinite_square(MPI.COMM_WORLD, ne, ne, beta=beta)

    with df.common.Timer("Blankenbach Functions"):
        vp = stokes_function(mesh, pp=pp)
        vp_i = df.fem.Function(vp.function_space)
        T = temperature_function(mesh, pT=pT)
        T_i = df.fem.Function(T.function_space)

    with df.common.Timer("Blankenbach Dirichlet BCs"):
        bcs_s = stokes_bcs(vp)
        bcs_T = temperature_bcs(T)

    with df.common.Timer("Blankenbach Forms"):
        Ss, fs = stokes_weakforms(vp, T, Ra, b=b)
        ST, fT = temperature_weakforms(vp, T)

        # Define the non-linear residual for the Stokes problem
        rs = ufl.action(Ss, vp) - fs
        # Define the non-linear residual for the temperature problem
        rT = ufl.action(ST, T) - fT
    
    with df.common.Timer("Blankenbach Problem Setup Stokes"):
        # Set up the Stokes problem (given the boundary conditions, bcs)
        problem_s = df.fem.petsc.LinearProblem(Ss, fs, bcs=bcs_s, u=vp_i, \
                                            petsc_options=petsc_options_s)
        
    with df.common.Timer("Blankenbach Problem Setup Temperature"):
        # Set up the Stokes problem (given the boundary conditions, bcs)
        problem_T = df.fem.petsc.LinearProblem(ST, fT, bcs=bcs_T, u=T_i, \
                                            petsc_options=petsc_options_T)

    def calculate_residual():
        """
        Return the total residual of the problem
        """
        rs_vec = df.fem.assemble_vector(df.fem.form(rs))
        df.fem.set_bc(rs_vec.array, bcs_s, scale=0.0)
        rs_vec.scatter_reverse(df.la.InsertMode.add)
        rT_vec = df.fem.assemble_vector(df.fem.form(rT))
        df.fem.set_bc(rT_vec.array, bcs_T, scale=0.0)
        rT_vec.scatter_reverse(df.la.InsertMode.add)
        r = np.sqrt(rs_vec.petsc_vec.norm()**2 + \
                    rT_vec.petsc_vec.norm()**2)
        return r

    with df.common.Timer("Blankenbach Residual"):
        # calculate the initial residual
        r = calculate_residual()
        r0 = r
        rrel = r/r0 # 1
    
    if mesh.comm.rank and verbose == 0:
        print("{:<11} {:<12} {:<17}".format('Iteration','Residual','Relative Residual'))
        print("-"*42)

    # Iterate until the residual converges (hopefully)
    it = 0
    if mesh.comm.rank == 0 and verbose: print("{:<11} {:<12.6g} {:<12.6g}".format(it, r, rrel,))
    while rrel > rtol and r > atol:
        if it > maxits: break
        with df.common.Timer("Blankenbach Solve Stokes"):
            vp_i = problem_s.solve()
            vp.x.array[:] = (1-alpha)*vp.x.array + alpha*vp_i.x.array
        with df.common.Timer("Blankenbach Solve Temperature"):
            T_i = problem_T.solve()
            T.x.array[:] = (1-alpha)*T.x.array + alpha*T_i.x.array
        # calculate a new residual
        with df.common.Timer("Blankenbach Residual"):
            r = calculate_residual()
            rrel = r/r0
            it += 1
        if mesh.comm.rank == 0 and verbose: print("{:<11} {:<12.6g} {:<12.6g}".format(it, r, rrel,))

    # Check for convergence failures
    if it > maxits:
        raise Exception("Nonlinear iteration failed to converge after {} iterations (maxits = {}), r = {} (atol = {}), rrel = {} (rtol = {}).".format(it, \
                                                                                                                                                      maxits, \
                                                                                                                                                      r, \
                                                                                                                                                      rtol, \
                                                                                                                                                      rrel, \
                                                                                                                                                      rtol,))

    # Return the subfunctions for velocity and pressure and the function for temperature
    return vp.sub(0).collapse(), vp.sub(1).collapse(), T


def blankenbach_diagnostics(v, T):
    mesh = T.function_space.mesh
    
    fdim = mesh.topology.dim - 1
    top_facets = df.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 1))
    facet_tags = df.mesh.meshtags(mesh, fdim, np.sort(top_facets), np.full_like(top_facets, 1))
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=facet_tags)

    Nu = -df.fem.assemble_scalar(df.fem.form(T.dx(1)*ds(1)))
    Nu = mesh.comm.allreduce(Nu, op=MPI.SUM)

    vrms = df.fem.assemble_scalar(df.fem.form((ufl.inner(v, v)*ufl.dx)))
    vrms = mesh.comm.allreduce(vrms, op=MPI.SUM)**0.5

    return Nu, vrms


if __name__ == "__main__":
    # code for Stokes Equation
    ne = 40
    pp = 1
    pT = 1
    # Case 1a
    Ra = 1.e4
    v_1a, p_1a, T_1a = solve_blankenbach(Ra, ne, pp=pp, pT=pT)
    T_1a.name = 'Temperature'
    print('Nu = {}, vrms = {}'.format(*blankenbach_diagnostics(v_1a, T_1a)))


if __name__ == "__main__":
    # visualize
    plotter_1a = utils.plot_scalar(T_1a, cmap='coolwarm')
    utils.plot_vector_glyphs(v_1a, plotter=plotter_1a, color='k', factor=0.0005)
    utils.plot_show(plotter_1a)


if __name__ == "__main__":
    # code for Stokes Equation
    ne = 40
    pp = 1
    pT = 1
    # Case 1b
    Ra = 1.e5
    v_1b, p_1b, T_1b = solve_blankenbach(Ra, ne, pp=pp, pT=pT)
    T_1b.name = 'Temperature'
    print('Nu = {}, vrms = {}'.format(*blankenbach_diagnostics(v_1b, T_1b)))


if __name__ == "__main__":
    # visualize
    plotter_1b = utils.plot_scalar(T_1b, cmap='coolwarm')
    utils.plot_vector_glyphs(v_1b, plotter=plotter_1b, color='k', factor=0.00005)
    utils.plot_show(plotter_1b)


if __name__ == "__main__":
    # code for Stokes Equation
    ne = 60
    pp = 1
    pT = 1
    # Case 1c
    Ra = 1.e6
    v_1c, p_1c, T_1c = solve_blankenbach(Ra, ne, pp=pp, pT=pT)
    T_1c.name = 'Temperature'
    print('Nu = {}, vrms = {}'.format(*blankenbach_diagnostics(v_1c, T_1c)))


if __name__ == "__main__":
    # visualize
    plotter_1c = utils.plot_scalar(T_1c, cmap='coolwarm')
    utils.plot_vector_glyphs(v_1c, plotter=plotter_1c, color='k', factor=0.00001)
    utils.plot_show(plotter_1c)


if __name__ == "__main__":
    # code for Stokes Equation
    ne = 60
    pp = 1
    pT = 1
    # Case 2a
    Ra = 1.e4
    v_2a, p_2a, T_2a = solve_blankenbach(Ra, ne, pp=pp, pT=pT, b=np.log(1.e3))
    T_2a.name = 'Temperature'
    print('Nu = {}, vrms = {}'.format(*blankenbach_diagnostics(v_2a, T_2a)))


if __name__ == "__main__":
    # visualize
    plotter_2a = utils.plot_scalar(T_2a, cmap='coolwarm')
    utils.plot_vector_glyphs(v_2a, plotter=plotter_2a, color='k', factor=0.00002)
    utils.plot_show(plotter_2a)


values_wvk = {
    '1a': {'Nu': 4.88440907, 'vrms': 42.8649484},
    '1b': {'Nu': 10.53404, 'vrms': 193.21445},
    '1c': {'Nu': 21.97242, 'vrms': 833.9897},
    '2a': {'Nu': 10.06597, 'vrms': 480.4308},
    }
values_bb = {
    '1a': {'Nu': 4.884409, 'vrms': 42.864947},
    '1b': {'Nu': 10.534095, 'vrms': 193.21454},
    '1c': {'Nu': 21.972465, 'vrms': 833.98977},
    '2a': {'Nu': 10.0660, 'vrms': 480.4334},
    }
params = {
    '1a': {'Ra': 1.e4, 'b': None},
    '1b': {'Ra': 1.e5, 'b': None},
    '1c': {'Ra': 1.e6, 'b': None},
    '2a': {'Ra': 1.e4, 'b': np.log(1.e3)}
    }


if __name__ == "__main__":
    # Open a figure for plotting
    fig, (axNu, axvrms) = pl.subplots(1,2, figsize=(15,7.5))

    cases = ['1a', '1b', '1c', '2a']
    # List of polynomial orders to try
    pTs = [1]
    # List of resolutions to try
    nelements = [32, 64, 128]
    # Keep track of whether we get the expected order of convergence
    test_passes = True
    for case in cases:
        params_c = params[case]
        Ra = params_c['Ra']
        b  = params_c['b']
        values_c = values_wvk[case]
        Nu_e   = values_c['Nu']
        vrms_e = values_c['vrms']
        # Loop over the polynomial orders
        for pT in pTs:
            # Accumulate the values and errors
            Nus = []
            vrmss = []
            errors_Nu   = []
            errors_vrms = []
            # Loop over the resolutions
            for ne in nelements:
                # Solve the 2D Batchelor corner flow problem
                v_i, p_i, T_i = solve_blankenbach(Ra, ne, pp=1, pT=pT, b=b, verbose=False)
                Nu, vrms = blankenbach_diagnostics(v_i, T_i)
                Nus.append(Nu)
                vrmss.append(vrms)
                Nuerr = np.abs(Nu - Nu_e)/Nu_e
                vrmserr = np.abs(vrms - vrms_e)/vrms_e
                errors_Nu.append(Nuerr)
                errors_vrms.append(vrmserr)
                # Print to screen and save if on rank 0
                if T_i.function_space.mesh.comm.rank == 0:
                    print('case={}, pT={}, ne={}, Nu={:.3f}, vrms={:.3f}, Nu err={:.3e}, vrms err={:.3e}'.format(case,pT,ne,Nu,vrms,Nuerr,vrmserr,))
    
            # Work out the order of convergence at this pT
            hs = 1./np.array(nelements)/pT
    
            # Write the errors to disk
            if T_i.function_space.mesh.comm.rank == 0:
                with open(output_folder / 'blankenbach_convergence_case{}_pT{}.csv'.format(case, pT), 'w') as f:
                    np.savetxt(f, np.c_[nelements, hs, Nus, vrmss, errors_Nu, errors_vrms], delimiter=',', 
                           header='nelements, hs, Nu, vrms, Nu_err, vrms_err')
    
            # Fit a line to the convergence data
            fitNu = np.polyfit(np.log(hs), np.log(errors_Nu),1)
            fitvrms = np.polyfit(np.log(hs), np.log(errors_vrms),1)
            if T_i.function_space.mesh.comm.rank == 0:
                print("***********  case {} order of accuracy pT={}, Nu order={}, vrms order={}".format(case,pT,fitNu[0],fitvrms[0]))

            # log-log plot of the error 
            label = '{}'.format(case,)
            if len(pTs) > 1: label = label+',pT={}'.format(pT,)
            axNu.loglog(hs, errors_Nu, 'o-', label=label+',order={:.2f}'.format(fitNu[0],))
            axvrms.loglog(hs, errors_vrms, 'o-', label=label+',order={:.2f}'.format(fitvrms[0],))
        
            # Test if the order of convergence is as expected (first order)
            #test_passes = test_passes and abs(fit[0]-1) < 0.1

    # Tidy up the plot
    axNu.set_xlabel('h')
    axNu.set_ylabel('$|\\Delta Nu|/Nu$')
    axNu.grid()
    axNu.legend()
    axvrms.set_xlabel('h')
    axvrms.set_ylabel('$|\\Delta v_\\text{rms}|/v_\\text{rms}$')
    axvrms.grid()
    axvrms.legend()

    fig.tight_layout()

    # Write convergence to disk
    if T_i.function_space.mesh.comm.rank == 0:
        fig.savefig(output_folder / 'blankenbach_convergence.pdf')
    
        print("***********  convergence figure in output/blankenbach_convergence.pdf")
    
    # Check if we passed the test
    assert(test_passes)


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/background --NbConvertApp.output_base=blankenbach 2.5b_blankenbach.ipynb')




