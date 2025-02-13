import adios4dolfinx
from dolfinx import fem, io
from mpi4py import MPI
import ufl
import numpy as np
from dataclasses import dataclass
import src.monodomain as monodomain
import src.hyperelasticity as hyperelasticity
from src.utils import pprint
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
        self.Ta_ode = fem.Function(self.ep.pde.V_ode)
        self.Ta = self.mech.Ta

    def _transfer_lmbda(self, N: int = 10):
        f = self.mech.F * self.mech.f0
        lmbda_exp = fem.Expression(
            ufl.sqrt(f**2), self.ep.pde.V_ode.element.interpolation_points()
        )
        lmbda_func = fem.Function(self.ep.pde.V_ode)
        lmbda_func.interpolate(lmbda_exp)
        lmbda = lmbda_func.x.petsc_vec[:]
        self.ep.ode.set_param("lmbda", lmbda)
        self._transfer_dlmbda_dt(lmbda, N)

    def _transfer_dlmbda_dt(self, lmbda: np.ndarray, N: int):
        if not hasattr(self, "prev_lmbda"):
            self.prev_lmbda = np.ones_like(lmbda)
        dlmbda_dt = (lmbda - self.prev_lmbda) / (N * self.dt)
        self.ep.ode.set_param("dLambda", dlmbda_dt)
        self.prev_lmbda = lmbda

    def _transfer_Ta(self, Ta: fem.Function | None = None):
        """Transfer tension from EP model to mechanics model.

        Args:
            Ta (fem.Function | None, optional): Tension function. If None (Default), gets tension from EP.
        """
        if Ta:
            Ta_array = Ta.x.petsc_vec
        else:
            Ta_array = self.ep.ode.model.monitor_values(
                self.t.value, self.ep.ode.states, self.ep.ode.params
            )[self.Ta_index]
        self.Ta_ode_space.x.array[:] = Ta_array
        self.mech.set_tension(self.Ta_ode_space)

    def solve(self, T: float, N: int = 10, save_displacement: bool = False):
        """Solve weakly coupled model until end time T. 
        Solves EP at each step, and solves mechanics at each Nth step. 
        Mech is more expensive to solve than EP.

        Args:
            T (float): end time
            N (int, optional): Number of. Defaults to 10.
            save_displacement (bool, optional): Option to save solution. Defaults to False.
        """
        if save_displacement:
            vtx = io.VTXWriter(
                MPI.COMM_WORLD,
                "coupling1way.bp",
                [self.mech.u, self.ep.pde.v_ode],
                engine="BP4",
            )
        n = 0
        while self.t.value < T:
            n += 1
            self.ep.step()
            pprint(f"Solved EP for t={np.round(self.t.value, 2)}", self.domain)
            if n % N == 0:
                self._transfer_Ta()
                self.mech.solve()
            if save_displacement:
                vtx.write(self.t.value)
        if save_displacement:
            vtx.close()

    def _save_Ta(self, function_filename: Path):
        """Save tension to file using adios4dolfinx.

        Args:
            function_filename (Path): file to save tension to
        """
        Ta_array = self.ep.ode.model.monitor_values(
            self.t.value, self.ep.ode.states, self.ep.ode.params
        )[self.Ta_index]
        self.Ta_ode.x.array[:] = Ta_array
        self.Ta.interpolate(self.Ta_ode)
        if function_filename.exists():
            writemode = adios4dolfinx.adios2_helpers.adios2.Mode.Append
        else:
            writemode = adios4dolfinx.adios2_helpers.adios2.Mode.Write

        adios4dolfinx.write_function_on_input_mesh(
            function_filename.with_suffix(".bp"),
            self.Ta,
            mode=writemode,
            time=np.round(self.t.value, 2),
            name="Ta",
        )

    def solve_ep_save_Ta(self, T: float, function_filename: Path, mesh_filename: Path):
        """Solve EP and save tension to file.

        Args:
            T (float): end time
            function_filename (Path): file to save tension to
            mesh_filename (Path): file to save mesh to
        """
        with io.XDMFFile(
            MPI.COMM_WORLD, mesh_filename.with_suffix(".xdmf"), "w"
        ) as xdmf:
            xdmf.write_mesh(self.ep.domain)
        while self.t.value < T:
            self.ep.step()
            self._save_Ta(function_filename)
            pprint(f"Solved EP for t={np.round(self.t.value, 2)}", self.domain)

    def solve_mech_with_saved_Ta(
        self,
        function_filename: Path,
        mesh_filename: Path,
        time: float,
        element: tuple[str, int],
    ):
        with io.XDMFFile(
            MPI.COMM_WORLD, mesh_filename.with_suffix(".xdmf"), "r"
        ) as xdmf:
            in_mesh = xdmf.read_mesh()
        V = fem.functionspace(in_mesh, element)
        Ta_in = fem.Function(V)
        adios4dolfinx.read_function(function_filename, Ta_in, time=time, name="Ta")
        self._transfer_Ta(Ta_in)
        self.mech.solve()
