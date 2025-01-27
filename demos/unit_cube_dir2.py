import numpy as np
from dolfinx import io
from mpi4py import MPI
import sys
sys.path.append('../')

from src.hyperelasticity import HyperelasticProblem

problem = HyperelasticProblem(h=0.2, lagrange_order=2)
problem.set_rectangular_domain(1, 1, 1, 0, 1)

def left(x):
    return np.isclose(x[0], 0)
def front(x):
    return np.isclose(x[1], 0)
def bottom(x):
    return np.isclose(x[2], 0)
def right(x):
    return np.isclose(x[0], 1)
boundaries = [left, front, bottom, right]
vals = [0, 1, 2, -1.0]
tags = [1, 2, 3, 4]
bc_types = ['d2', 'd2', 'd2', 'n']

problem.boundary_conditions(boundaries, vals, tags, bc_types)
problem.holzapfel_ogden_model()
problem.incompressible()
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