import numpy as np
import matplotlib.pyplot as plt
from dolfinx import io, fem, mesh
import dolfinx
import packaging
from mpi4py import MPI
import ufl
import sys

sys.path.append("../")

dolfinx_version = packaging.version.parse(dolfinx.__version__)

from src.hyperelasticity import HyperelasticProblem

h_fine = 0.5
h_coarse = 0.75
lagrange_order = 2

problem_coarse = HyperelasticProblem(h_coarse, lagrange_order)
problem_fine = HyperelasticProblem(h_fine, lagrange_order)
problem_coarse.set_rectangular_domain(Lx=1, Ly=1, Lz=1, f_dir=0, s_dir=1)
problem_fine.set_rectangular_domain(Lx=1, Ly=1, Lz=1, f_dir=0, s_dir=1)


def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], 1)


problem_coarse.boundary_conditions([left, right], vals=[0.0, 0], bc_types=["d1", "n"])
problem_fine.boundary_conditions([left, right], vals=[0.0, 0], bc_types=["d1", "n"])
problem_coarse.setup_solver()
problem_fine.setup_solver()


def tension(x):
    return 5 * np.exp(-5 * ((x[1] - 0.5) ** 2 + (x[2] - 0.5) ** 2))


problem_coarse.set_tension(tension)
problem_fine.set_tension(tension)

problem_coarse.solve()
problem_fine.solve()

u_coarse = problem_coarse.u
u_fine = problem_fine.u

u_coarse_interp = fem.Function(u_fine.function_space)
if dolfinx_version < packaging.version.parse("0.9.0"):
    u_coarse_interp.interpolate(
        u_coarse,
        nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
            u_fine.function_space.mesh,
            u_fine.function_space.element,
            u_coarse.function_space.mesh,
        ),
    )
else:
    # see https://github.com/FEniCS/dolfinx/issues/3562#issuecomment-2543112976
    V_coarse, V_fine = u_coarse.function_space, u_fine.function_space
    domain_coarse = V_coarse.mesh
    domain_fine = V_fine.mesh

    bbox_coarse = dolfinx.geometry.bb_tree(
        domain_coarse, domain_coarse.topology.dim, padding=1e-6
    )
    bbox_fine = dolfinx.geometry.bb_tree(
        domain_fine, domain_fine.topology.dim, padding=1e-6
    )
    global_b1_tree = bbox_coarse.create_global_tree(domain_coarse.comm)
    collisions = dolfinx.geometry.compute_collisions_trees(bbox_fine, global_b1_tree)
    cells = np.unique(collisions[:, 0])

    interpolation_data1 = fem.create_interpolation_data(V_fine, V_coarse, cells)
    u_coarse_interp.interpolate_nonmatching(u_coarse, cells, interpolation_data1)

comm = u_fine.function_space.mesh.comm
error = fem.form((u_fine - u_coarse_interp) ** 2 * ufl.dx)
E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
print(E)
