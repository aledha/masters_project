import numpy as np
from dolfinx import io
import ufl
from mpi4py import MPI
import sys

sys.path.append("../")

from src.hyperelasticity import HyperelasticProblem

problem = HyperelasticProblem(h=0.2, lagrange_order=2)
L = (1, 1, 1)
f0 = ufl.unit_vector(0, 3)
s0 = ufl.unit_vector(1, 3)
problem.set_rectangular_domain(L, f0, s0)


left = lambda x: np.isclose(x[0], 0)
front = lambda x: np.isclose(x[1], 0)
bottom = lambda x: np.isclose(x[2], 0)
right = lambda x: np.isclose(x[0], 1)
boundaries = [left, front, bottom, right]
vals = [0, 1, 2, -1.0]
bc_types = ["d2", "d2", "d2", "n"]

problem.boundary_conditions(boundaries, vals, bc_types)
problem.setup_solver()

vtx = io.VTXWriter(MPI.COMM_WORLD, "unit_cube_dir2.bp", [problem.u], engine="BP4")

T_as = np.linspace(0, 100, 11)
for T_a in T_as:
    problem.set_tension(T_a)
    problem.solve()
    vtx.write(T_a)
    if problem.domain.comm.rank == 0:
        print(f"Solved for T_a={T_a}")
vtx.close()
