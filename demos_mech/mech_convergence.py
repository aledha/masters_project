from pathlib import Path

from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, io

from nmcemfem.hyperelasticity import HyperelasticProblem
from nmcemfem.utils import interpolate_to_mesh


def solve(h, lagrange_order=2, save_solution=False):
    problem = HyperelasticProblem(h, lagrange_order)
    L = (1, 1, 1)
    f0 = ufl.unit_vector(0, 3)
    s0 = ufl.unit_vector(1, 3)
    problem.set_rectangular_domain(L, f0, s0)

    def left(x):
        return np.isclose(x[0], 0)

    def right(x):
        return np.isclose(x[0], 1)

    problem.boundary_conditions([left, right], vals=[0.0, 0], bc_types=["d1", "n"])
    problem.setup_solver()

    def tension(x):
        return 5 * np.exp(-5 * ((x[1] - 0.5) ** 2 + (x[2] - 0.5) ** 2))

    problem.set_tension(tension)

    if save_solution:
        # Save reference configuration
        func_dir = Path(__file__).parents[1] / "saved_funcs"
        vtx = io.VTXWriter(MPI.COMM_WORLD, func_dir / "convergence.bp", [problem.u], engine="BP4")
        vtx.write(0.0)
    problem.solve()
    if save_solution:
        # Save current configuration
        vtx.write(0.1)
        vtx.close()
    return problem.u


def L2_diff_coarse_fine(h, u_fine, lagrange_order=2):
    # Interpolate the coarser solution into finer mesh and compute error there?
    # Opposite would lose details from the fine solution.
    u_coarse = solve(h, lagrange_order)

    u_coarse_interp = interpolate_to_mesh(u_from=u_coarse, V_to=u_fine.function_space)

    comm = u_fine.function_space.mesh.comm
    error = fem.form((u_fine - u_coarse_interp) ** 2 * ufl.dx)
    E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
    return E


def convergence_plot(h_fine, hs, plot_title, lagrange_order=2):
    u_fine = solve(h_fine)
    errors = np.zeros_like(hs)
    for i in range(len(hs)):
        errors[i] = L2_diff_coarse_fine(hs[i], u_fine, lagrange_order=lagrange_order)

    order = (np.log(errors[-1]) - np.log(errors[-2])) / (np.log(hs[-1]) - np.log(hs[-2]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(hs, errors, "-o", label="order = {:.3f}".format(order))
    ax.set_xlabel(r"$h$")
    ax.set_ylabel("Error")
    ax.set_title("Convergence plot of hyperelastic problem")
    ax.legend()
    figure_dir = Path(__file__).parents[1] / "saved_figures"
    fig.savefig(figure_dir / plot_title, bbox_inches="tight")
    fig.show()


# solve(h=0.1, save_solution=True)
h_fine = 0.075
hs = [0.1, 0.15, 0.2, 0.3, 0.4]
convergence_plot(h_fine, hs, "mech_convergence.png")
