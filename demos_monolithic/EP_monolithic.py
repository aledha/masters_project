from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx.fem.petsc
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)


def v_exact(x, t):
    return ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)


def s_exact(x, t):
    return -ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.cos(t)


def states_exact(x, t):
    return ufl.as_vector((v_exact(x, t), s_exact(x, t)))


def forward_euler(states, t, dt, params):
    v, s = states
    output = np.zeros_like(states)
    output[0] = v - s * dt
    output[1] = s + v * dt
    return output


def error(h, dt, theta, T, quad_degree):
    N = int(np.ceil(1 / h))

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V_pde = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    v = ufl.TrialFunction(V_pde)
    phi = ufl.TestFunction(V_pde)

    v_pde = dolfinx.fem.Function(V_pde)
    v_pde.interpolate(lambda x: 0 * x[0])

    t = dolfinx.fem.Constant(mesh, 0.0)
    x = ufl.SpatialCoordinate(mesh)
    I_stim = 8 * ufl.pi**2 * ufl.sin(t) * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])

    V_ode_element = basix.ufl.quadrature_element(
        cell=mesh.ufl_cell().cellname(), degree=quad_degree, value_shape=(2,), scheme="default"
    )
    V_ode = dolfinx.fem.functionspace(mesh, V_ode_element)

    states_old = dolfinx.fem.Function(V_ode)
    states_old.interpolate(
        dolfinx.fem.Expression(states_exact(x, t), V_ode.element.interpolation_points())
    )

    def f_impl(v_pde: npt.NDArray, states_old: npt.NDArray):
        states_data = states_old.reshape(states_old.shape[0], -1, 2)
        s = states_data[:, :, 1]
        states = np.vstack([v_pde.flatten(), s.flatten()])
        states_new = forward_euler(states, t.value, dt, params=None)
        return states_new.T.flatten()

    def f_external(derivatives: tuple[int, ...]):
        if derivatives == (0, 0):  # no derivation, the function itself
            return f_impl
        elif derivatives == (1, 0):  # the derivative with respect to the operand `uh`
            return NotImplementedError
        elif derivatives == (0, 1):
            return NotImplementedError
        elif derivatives == (0, 0):
            return NotImplementedError
        else:
            return NotImplementedError

    states = FEMExternalOperator(
        v_pde, states_old, function_space=V_ode, external_function=f_external
    )
    v_ode = ufl.split(states)[0]

    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quad_degree})
    a = phi * v * dx + dt * theta * ufl.inner(ufl.grad(phi), ufl.grad(v)) * dx
    L = (
        phi * v_ode * dx + dt * phi * I_stim * dx
        # - dt * (1 - theta) * ufl.dot(ufl.grad(phi), ufl.grad(v_ode)) * dx
    )

    L_updated, operators = replace_external_operators(L)
    L_compiled = dolfinx.fem.form(L_updated)

    solver = dolfinx.fem.petsc.LinearProblem(a, L_compiled, u=v_pde)
    dolfinx.fem.petsc.assemble_matrix(solver.A, solver.a)
    solver.A.assemble()
    v_ex_func = lambda x, t: ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)
    v_ex_expr = dolfinx.fem.Expression(v_ex_func(x, t), V_pde.element.interpolation_points())
    v_ex = dolfinx.fem.Function(V_pde, name="v_exact")

    vtx = dolfinx.io.VTXWriter(MPI.COMM_WORLD, "mono_conv.bp", [v_pde, v_ex], engine="BP4")

    while t.value < T:
        coefficients = evaluate_operands(operators)
        evaluate_external_operators(operators, coefficients)

        with solver.b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(solver.b, solver.L)
        solver.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        solver.solver.solve(solver.b, v_pde.x.petsc_vec)
        v_pde.x.scatter_forward()

        states_old.interpolate(states.ref_coefficient)
        t.value += dt
        v_ex.interpolate(v_ex_expr)

        vtx.write(t.value)

    vtx.close()

    error = dolfinx.fem.form((v_pde - v_ex) ** 2 * dx)
    E = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))
    return E


# print(error(0.1, 0.01, 1.0, 1, 2))


def dual_convergence_plot(hs, dts, theta, T, quad_degree=4, plot_title=None):
    num_spatial = len(hs)
    num_temporal = len(dts)

    errors = np.zeros((num_temporal, num_spatial))

    for i_time in range(num_temporal):
        for i_space in range(num_spatial):
            errors[i_time, i_space] = error(hs[i_space], dts[i_time], theta, T, quad_degree)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("L2 error convergence plot. " + r"$\theta=$" + f"{theta}")

    for i_time in range(len(dts)):
        order = (np.log(errors[i_time, -1]) - np.log(errors[i_time, -2])) / (
            np.log(hs[-1]) - np.log(hs[-2])
        )

        ax1.loglog(
            hs,
            errors[i_time],
            "-o",
            label="dt = {:.3f}  ".format(dts[i_time]) + "p = {:.2f}".format(order),
        )
    ax1.set_xlabel(r"$h$")
    ax1.set_ylabel("Error")
    ax1.set_title(r"Error as a function of $h$ for different d$t$.")
    ax1.legend()
    ax1.set_xticks(hs, hs)
    ax1.minorticks_off()

    for i_space in range(len(hs)):
        order = (np.log(errors[-1, i_space]) - np.log(errors[-2, i_space])) / (
            np.log(dts[-1]) - np.log(dts[-2])
        )
        ax2.loglog(
            dts,
            errors[:, i_space],
            "-o",
            label="h = {:.3f}  ".format(hs[i_space]) + "p = {:.2f}".format(order),
        )
    ax2.set_xlabel(r"d$t$")
    ax2.set_ylabel("Error")
    ax2.set_title(r"Error as a function of d$t$ for different $h$.")
    ax2.legend()
    ax2.set_xticks(dts, dts)
    ax2.minorticks_off()
    if plot_title:
        fig.savefig(plot_title, bbox_inches="tight")
    plt.show()


hs = [1 / (2**i) for i in range(6, 2, -1)]
dts = [1 / (2**i) for i in range(7, 3, -1)]

# dual_convergence_plot(hs, dts, theta=1.0, T=1)#, plot_title="theta1_dual_convergence.png")
