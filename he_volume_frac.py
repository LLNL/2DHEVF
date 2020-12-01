from firedrake import *
from firedrake_adjoint import *
import numpy as np
from ufl import replace, conditional, tanh
from ufl import min_value, max_value

from params import (height, width, inlet_width, dist_center, inlet_depth, line_sep,
                                INMOUTH1, INMOUTH2, OUTMOUTH1, OUTMOUTH2, INLET1, INLET2, OUTLET1, OUTLET2, WALLS)
import sys
from pyMMAopt import MMASolver
from pyadjoint.placeholder import Placeholder
import signal

direct_parameters = {
    "mat_type" : "aij",
    "ksp_type" : "preonly",
    "pc_type" : "lu",
    "pc_factor_mat_solver_type" : "mumps"
}

import argparse
parser = argparse.ArgumentParser(description='Level set method parameters')
parser.add_argument('--enthalpy_scale', action="store", dest="enthalpy_scale", type=float, help='Enthalpy scale coefficient', default=0.04)
parser.add_argument('--alphabar', action="store", dest="alphabar", type=float, help='Brinkmann penalty parameter scale', default=3e-4)
parser.add_argument('--mu', action="store", dest="mu", type=float, help='Viscosity', default=1e-1)
parser.add_argument('--output_dir', action="store", dest="output_dir", type=str, help='Output directory', default="./")
args = parser.parse_args()

enthalpy_scale = args.enthalpy_scale
output_dir = args.output_dir
mu_value = args.mu
alphabar_scale = args.alphabar

simp_coeff = 5
ramp_coeff = 0.01
pressure_drop = 1e-2



def main():

    mesh = Mesh("./2D_mesh.msh")
    x, y = SpatialCoordinate(mesh)

    RHO = FunctionSpace(mesh, 'CG', 1)
    rho = Function(RHO)
    rho.interpolate(Constant(0.5))

    # Total design volume
    with stop_annotating():
        vol_design = assemble(Constant(1.0)*dx(0, domain=mesh))
    vol_constraint_ratio = 2.0
    vol_constraint = vol_design * vol_constraint_ratio

    # Build function space
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2*P1
    W = FunctionSpace(mesh, TH)

    U = TrialFunction(W)
    u, p = split(U)
    V = TestFunction(W)
    v, q = split(V)

    mu = Constant(mu_value)                   # viscosity
    alphabar = Constant(2.5 / (alphabar_scale))      # parameter for \alpha
    psimp = Constant(simp_coeff)
    Placeholder(psimp)
    def alpha(rho):
        return alphabar * (rho)**psimp

    f = Constant((0.0, 0.0))
    a = mu*inner(grad(u), grad(v)) + inner(grad(p), v) + q*div(u)
    L = inner(f, v)*dx

    def a1(rho):
        return a*dx + alpha(rho)*inner(u, v)*dx(0) + alphabar*inner(u, v)*dx(INMOUTH2) + alphabar*inner(u, v)*dx(OUTMOUTH2)
    def a2(rho):
        return a*dx + alpha(Constant(1.0) - rho)*inner(u, v)*dx(0) + alphabar*inner(u, v)*dx(INMOUTH1) + alphabar*inner(u, v)*dx(OUTMOUTH1)


    u_inflow = 2e-3
    # Dirichelt boundary conditions
    inflow1 = as_vector([u_inflow*sin(((y - (line_sep - (dist_center + inlet_width))) * pi )/ inlet_width), 0.0])
    inflow2 = as_vector([u_inflow*sin(((y - (line_sep + dist_center)) * pi )/ inlet_width), 0.0])
    pressure_outlet = Constant(0.0)
    noslip = Constant((0.0, 0.0))

    Re = u_inflow * width / mu.values()[0]
    print("Reynolds number: {:.5f}".format(Re), flush=True)

    # Stokes 1
    bcs1_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs1_2 = DirichletBC(W.sub(0), inflow1, INLET1)
    bcs1_3 = DirichletBC(W.sub(1), pressure_outlet, OUTLET1)
    bcs1_4 = DirichletBC(W.sub(0), noslip, INLET2)
    bcs1_5 = DirichletBC(W.sub(0), noslip, OUTLET2)

    bcs2_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs2_2 = DirichletBC(W.sub(0), inflow2, INLET2)
    bcs2_3 = DirichletBC(W.sub(1), pressure_outlet, OUTLET2)
    bcs2_4 = DirichletBC(W.sub(0), noslip, INLET1)
    bcs2_5 = DirichletBC(W.sub(0), noslip, OUTLET1)


    U1 = Function(W, name="Vel-P 1")
    u1, p1 = split(U1)

    U2 = Function(W, name="Vel-P 2")
    u2, p2 = split(U2)

    # Convection difussion equation
    T = FunctionSpace(mesh, 'DG', 1)

    t = Function(T, name="Temperature")
    w = TestFunction(T)
    theta = TrialFunction(T)

    # Mesh-related functions
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    h_avg = (h('+') + h('-'))/2

    kf = Constant(1e0)
    ks = Constant(1e0)
    cp_value = 5.0e5
    cp = Constant(cp_value)

    Pe = u_inflow * width * cp_value / ks.values()[0]
    print("Peclet number: {:.5f}".format(Pe), flush=True)

    # Temperature problem
    tin1 = Constant(10.0)
    tin2 = Constant(100.0)

    def eT(u1, u2):

        u1n = (dot(u1, n) + abs(dot(u1, n)))/2.0
        u2n = (dot(u2, n) + abs(dot(u2, n)))/2.0
        # Penalty term
        alpha = Constant(50000.0) # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
        # Bilinear form
        a_int = dot(grad(w), ks*grad(t) - cp*(u1 + u2)*t)*dx

        a_fac = - ks*dot(avg(grad(w)), jump(t, n))*dS \
              - ks*dot(jump(w, n), avg(grad(t)))*dS \
              + ks('+')*(alpha('+')/avg(h))*dot(jump(w, n), jump(t, n))*dS

        a_vel = dot(jump(w), Constant(cp_value)*(u1n('+') + u2n('+'))*t('+') - \
                Constant(cp_value)*(u1n('-') + u2n('-'))*t('-'))*dS + \
                dot(w, Constant(cp_value)*(u1n + u2n)*t)*ds

        a_bnd = dot(w, Constant(cp_value)*dot(u1 + u2, n)*t)*(ds(INLET1) + ds(INLET2)) \
                + w*t*(ds(INLET1) + ds(INLET2)) \
                - w*tin1*ds(INLET1) \
                - w*tin2*ds(INLET2) \
                + alpha/h * ks *w*t*(ds(INLET1) + ds(INLET2)) \
                - ks * dot(grad(w), t*n)*(ds(INLET1) + ds(INLET2)) \
                - ks * dot(grad(t), w*n)*(ds(INLET1) + ds(INLET2))

        aT = a_int + a_fac + a_vel + a_bnd

        LT_bnd = alpha/h * ks * tin1 * w * ds(INLET1)  \
                + alpha/h * ks * tin2 * w * ds(INLET2) \
                - tin1 * ks * dot(grad(w), n) * ds(INLET1) \
                - tin2 * ks * dot(grad(w), n) * ds(INLET2)

        return aT, LT_bnd

    bcT1 = DirichletBC(T, tin1, INLET1)
    bcT2 = DirichletBC(T, tin2, INLET2)
    LT = Constant(0.0)*w*dx



    def forward(rho):

        # Stokes 1 problem
        bcs=[bcs1_1,bcs1_2,bcs1_3,bcs1_4, bcs1_5]
        solve(a1(rho)==L, U1, bcs=bcs, solver_parameters=direct_parameters)

        # Stokes 2 problem
        bcs=[bcs2_1,bcs2_2,bcs2_3,bcs2_4, bcs2_5]
        solve(a2(rho)==L, U2, bcs=bcs, solver_parameters=direct_parameters)

        u1, p1 = split(U1)
        u2, p2 = split(U2)

        # Temperature problem
        aT, LT = eT(u1, u2)
        problem = aT - LT
        # Temperature problem
        solve(problem==0, t, solver_parameters=direct_parameters)

        return U1, U2, t

    U1, U2, t = forward(rho)

    # Marking these as control to be able to plot them
    U1control = Control(U1)
    U2control = Control(U2)
    tcontrol = Control(t)

    u1, p1 = split(U1)
    u2, p2 = split(U2)

    def Pdropfunc1(p1):
        p_drop_value = assemble(p1*ds(INLET1))
        return p_drop_value
    def Pdropfunc2(p2):
        p_drop_value = assemble(p2*ds(INLET2))
        return p_drop_value

    def Jfunc(u1, u2, t):
        Jform =  Constant(-enthalpy_scale*cp_value)*inner(t*(u1), n)*(ds(OUTLET1))
        J = assemble(Jform)
        return J


    #print("Enthalpy {0:.5f}, Pressure drop {1:.5f}".format(J, Pinlet))
    c = Control(rho)

    controls_f = File(output_dir + "/control_iterations_f.pvd")
    rho_viz_f = Function(RHO, name="rho")

    derivative_pvd = File(output_dir + "/derivative_iterations.pvd")
    deriv_viz = Function(RHO, name="Derivative")

    primal_u1 = File(output_dir + "/primal_u1.pvd")

    primal_u2 = File(output_dir + "/primal_u2.pvd")

    primal_t = File(output_dir + "/primal_t.pvd")
    t_viz = Function(T, name="t")

    import itertools
    global_counter1 = itertools.count()
    def deriv_cb(j, dj, rho):
        iter = next(global_counter1)
        if iter % 10 == 0:
            with open(output_dir + '/final_output.txt', 'w') as final_output:
                final_output.write(solver.current_state())
            deriv_viz.assign(dj)
            derivative_pvd.write(deriv_viz)
            with stop_annotating():
                t_viz.assign(tcontrol.tape_value())
                rho_viz_f.assign(rho)
            u1, _ = U1control.tape_value().split()
            u2, _ = U2control.tape_value().split()
            u1.rename("Velocity")
            u2.rename("Velocity")
            primal_u1.write(u1)
            primal_u2.write(u2)
            primal_t.write(t_viz)
            controls_f.write(rho_viz_f)

    J = Jfunc(u1, u2, t)
    Pd1 = Pdropfunc1(p1)
    Pd2 = Pdropfunc2(p2)
    print("Heat exchanger initial value: {0:.6}".format(J), flush=True)
    print("Pressure Drop 1: {0:.6}".format(Pd1), flush=True)
    print("Pressure Drop 2: {0:.6}".format(Pd2), flush=True)
    Pcontrol1 = Control(Pd1)
    Pcontrol2 = Control(Pd2)

    Jhat = ReducedFunctional(J, c, derivative_cb_post=deriv_cb)
    Phat1 = ReducedFunctional(Pd1, c)
    Phat2 = ReducedFunctional(Pd2, c)
    Pdrop = pressure_drop # Best design with 1.8e-3

    Jvalue = Jhat(rho)

    class PressureConstraint(InequalityConstraint):
         def __init__(self, Phat, Pdrop, Pcontrol):
             self.Phat  = Phat
             self.Pdrop  = float(Pdrop)
             self.Pcontrol  = Pcontrol
             self.tmpvec = Function(RHO)

         def function(self, m):
             from pyadjoint.reduced_functional_numpy import set_local
             set_local(self.tmpvec, m)

             # Compute the integral of the control over the domain
             integral = self.Pcontrol.tape_value()
             print("Pressure drop: {0:.6f}, Pressure constraint: {1:.4f}".format(integral, self.Pdrop), flush=True)
             with stop_annotating():
                 value = - integral / self.Pdrop + 1.0
             return [value]

         def jacobian(self, m):
             from pyadjoint.reduced_functional_numpy import set_local
             set_local(self.tmpvec, m)

             with stop_annotating():
                 gradients = self.Phat.derivative()
                 with gradients.dat.vec as v:
                     v.scale(-1.0 / self.Pdrop)
             return [gradients]

         def output_workspace(self):
             return [0.0]

         def length(self):
             """Return the number of components in the constraint vector (here, one)."""
             return 1

    # Bound constraints
    lb = 0.0
    ub = 1.0
    # Solve the optimisation problem with q = 0.01

    psimp_arr = [5.0, 3.0]
    change_arr = [1e-4, 1e-8]
    accepted_tol = [1e-2, 1e-3]
    for simp, change, acpt_tol in zip(psimp_arr, change_arr, accepted_tol):
        psimp.assign(simp)
        problem = MinimizationProblem(Jhat, bounds=(lb, ub),
                    constraints=[PressureConstraint(Phat1, Pdrop, Pcontrol1),
                        PressureConstraint(Phat2, Pdrop, Pcontrol2)])

        tape = get_working_tape()
        tape.optimize_for_controls([c])

        parameters_mma = {'move': 0.5, 'maximum_iterations': 10000, 'm': 2, 'IP': 1, 'tol' : change, 'accepted_tol' : acpt_tol}
        solver = MMASolver(problem, parameters=parameters_mma)

        rho_opt = solver.solve()

        c.update(rho_opt)

if __name__ == '__main__':
    main()
