from pathlib import Path

from mpi4py import MPI

import adios4dolfinx
import numpy as np
import ufl
from dolfinx import fem, io

from nmcemfem.hyperelasticity import HyperelasticProblem

mech = HyperelasticProblem(1, 2)
Lx, Ly, Lz = 3, 7, 20  # mm
L = (Lx, Ly, Lz)
f0 = ufl.unit_vector(2, 3)
s0 = ufl.unit_vector(1, 3)
mech.set_rectangular_domain(L, f0, s0)

left = lambda x: np.isclose(x[0], 0)
front = lambda x: np.isclose(x[1], 0)
bottom = lambda x: np.isclose(x[2], 0)

boundaries = [left, front, bottom]
vals = [0, 1, 2]
bc_types = ["d2", "d2", "d2"]
mech.boundary_conditions(boundaries, vals, bc_types)
mech.setup_solver()

func_dir = Path(__file__).parents[1] / "saved_funcs"
mesh_filename = "L2_mesh"
function_filename = "saved_Ta_L2"
with io.XDMFFile(MPI.COMM_WORLD, (func_dir / mesh_filename).with_suffix(".xdmf"), "r") as xdmf:
    in_mesh = xdmf.read_mesh()
element = ("Lagrange", 2)
V = fem.functionspace(in_mesh, element)
Ta_in = fem.Function(V)

saveto_file = "saved ta"
vtx = io.VTXWriter(
    MPI.COMM_WORLD,
    (func_dir / saveto_file).with_suffix(".bp"),
    [Ta_in],
    engine="BP4",
)
time = np.arange(1, 70, 1)

for t in time:
    adios4dolfinx.read_function(
        (func_dir / function_filename).with_suffix(".bp"),
        Ta_in,
        time=t,
        name="Ta",
    )
    mech.set_tension(Ta_in)
    mech.solve()
    vtx.write(t)
vtx.close()
