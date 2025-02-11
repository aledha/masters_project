import adios4dolfinx
from dolfinx import fem, io
from mpi4py import MPI
import ufl
import numpy as np
from dataclasses import dataclass
import src.monodomain as monodomain
import src.hyperelasticity as hyperelasticity
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


@dataclass
class WeaklyCoupledModel:
    ep: monodomain.MonodomainSolver
    mech: hyperelasticity.HyperelasticProblem

    def __post_init__(self):
        self.t = self.ep.t
        self.dt = self.ep.dt
        self.domain = self.ep.domain
        self.Ta_index = self.ep.ode.model.monitor_index("Ta")
        self.Ta_lagrange = fem.Function(self.ep.pde.V)
        self.Ta = self.mech.Ta

        with io.XDMFFile(MPI.COMM_WORLD, "Ta.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.ep.domain)

    def _save_Ta(self, title):
        Ta_array = self.ep.ode.model.monitor_values(
            self.t.value, self.ep.ode.states, self.ep.ode.params
        )[self.Ta_index]
        self.Ta_lagrange.x.array[:] = Ta_array
        self.Ta.interpolate(self.Ta_lagrange)

        adios4dolfinx.write_function_on_input_mesh(
            title,
            self.Ta,
            mode=adios4dolfinx.adios2_helpers.adios2.Mode.Write,
            time=self.t.value,
            name="Ta",
        )

    def _transfer_lmbda(self, N=10):
        f = self.mech.F * self.mech.f0
        lmbda_exp = fem.Expression(
            ufl.sqrt(f**2), self.ep.pde.V.element.interpolation_points()
        )
        lmbda_func = fem.Function(self.ep.pde.V)
        lmbda_func.interpolate(lmbda_exp)
        lmbda = lmbda_func.x.petsc_vec[:]
        self.ep.ode.set_param("lmbda", lmbda)
        self._transfer_dlmbda_dt(lmbda, N)

    def _transfer_dlmbda_dt(self, lmbda, N):
        if not hasattr(self, "prev_lmbda"):
            self.prev_lmbda = np.ones_like(lmbda)
        dlmbda_dt = (lmbda - self.prev_lmbda) / (N * self.dt)
        self.ep.ode.set_param("dLambda", dlmbda_dt)
        self.prev_lmbda = lmbda

    def _transfer_Ta(self, Ta = None):
        # Now I am saving T_a to a function defined on the EP ode space,
        # and then interpolating it into the T_a space (DG, 0).
        if Ta:
            Ta_array = Ta.x.petsc_vec
        else:
            Ta_array = self.ep.ode.model.monitor_values(
                self.t.value, self.ep.ode.states, self.ep.ode.params
            )[self.Ta_index]
        self.Ta_lagrange.x.array[:] = Ta_array
        self.mech.set_tension(self.Ta_lagrange)


    def solve(self, T, N=10, save_displacement=False):
        if save_displacement:
            vtx = io.VTXWriter(
                MPI.COMM_WORLD,
                "coupling1way.bp",
                [self.mech.u, self.ep.pde.vn],
                engine="BP4",
            )
        n = 0
        while self.t.value < T + self.dt:
            n += 1
            self.ep.step()
            if n % N == 0:
                self._transfer_Ta()
                self.mech.solve()
                self._transfer_lmbda()
            if self.domain.comm.rank == 0:
                print(f"Solved for t={self.t.value}")
            if save_displacement:
                vtx.write(self.t.value)
        if save_displacement:
            vtx.close()

    def solve_ep_save_Ta(self, T, title):
        while self.t.value < T + self.dt:
            self.ep.step()
            self._save_Ta(title)
            print("Solved for t = ", self.t.value)

    def solve_mech_with_saved_Ta(self, function_filename, time, element):
        with io.XDMFFile(MPI.COMM_WORLD, "Ta.xdmf", "r") as xdmf:
            in_mesh = xdmf.read_mesh()
        V = fem.functionspace(in_mesh, element)
        Ta_in = fem.Function(V)
        adios4dolfinx.read_function(function_filename, Ta_in, time=time, name="Output")
        self._transfer_Ta(Ta_in)
        self.mech.solve()

