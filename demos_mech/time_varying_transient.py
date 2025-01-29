import numpy as np
from dolfinx import io
from mpi4py import MPI
import sys
sys.path.append('../')
from src.hyperelasticity import HyperelasticProblem

def ca_transient(t, tstart=0.05):
    tau1 = 0.05
    tau2 = 0.110

    ca_diast = 0.0
    ca_ampl = 1.0

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - (tau1 / tau2) ** (-1 / (1 - tau2 / tau1))
    ca = np.zeros_like(t)

    ca[t <= tstart] = ca_diast

    ca[t > tstart] = (ca_ampl - ca_diast) / beta * (
        np.exp(-(t[t > tstart] - tstart) / tau1) - np.exp(-(t[t > tstart] - tstart) / tau2)
    ) + ca_diast
    return ca

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
bc_types = ['d2', 'd2', 'd2', 'n']

problem.boundary_conditions(boundaries, vals, bc_types)
problem.holzapfel_ogden_model()
problem.incompressible()
problem.setup_solver()

vtx = io.VTXWriter(MPI.COMM_WORLD, "time_varing_transient.bp", [problem.u], engine="BP4")

ts = np.arange(0, 1, 0.01)
cas = ca_transient(ts)
for t, ca in zip(ts, cas):
    problem.set_tension(ca)
    problem.solve()
    vtx.write(t) 
    if problem.domain.comm.rank == 0:
        print(f"Solved for T_a={t}")
vtx.close()