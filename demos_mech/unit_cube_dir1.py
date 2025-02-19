import numpy as np
from dolfinx import io
import ufl
from mpi4py import MPI
from pathlib import Path


from nmcemfem.hyperelasticity import HyperelasticProblem


problem = HyperelasticProblem(h=0.2, lagrange_order=2)
L = (1, 1, 1)
f0 = ufl.unit_vector(0, 3)
s0 = ufl.unit_vector(1, 3)
problem.set_rectangular_domain(L, f0, s0)

left = lambda x: np.isclose(x[0], 0)
right = lambda x: np.isclose(x[0], 1)
problem.boundary_conditions([left, right], vals=[0.0, -1.0], bc_types=["d1", "n"])
problem.setup_solver()

func_dir = Path(__file__).parents[1] / "saved_funcs"
vtx = io.VTXWriter(MPI.COMM_WORLD, func_dir / "unit_cube_dir1.bp", [problem.u], engine="BP4")

T_as = np.linspace(0, 100, 11)
for T_a in T_as:
    problem.set_tension(T_a)
    problem.solve()
    vtx.write(T_a)
    if problem.domain.comm.rank == 0:
        print(f"Solved for T_a={T_a}")
vtx.close()
