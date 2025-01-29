import adios4dolfinx
from dolfinx import fem, mesh, default_scalar_type, io, log
from mpi4py import MPI
from dataclasses import dataclass
import src.monodomain as monodomain
import src.hyperelasticity as hyperelasticity
import gotranx

import sys
from pathlib import Path

import adios4dolfinx

sys.path.append(str(Path(__file__).resolve().parents[1]))

def translateODE(odeFileName, schemes):
    odeFolder = str(Path.cwd().parent) + "/odes/"
    model_path = Path(odeFolder + odeFileName + ".py")
    if not model_path.is_file():
        ode = gotranx.load_ode(odeFolder + odeFileName + ".ode")
        code = gotranx.cli.gotran2py.get_code(ode, schemes)
        model_path.write_text(code)
    else:
        print("ODE already translated")

#translateODE('tentusscher_land_1way', [gotranx.schemes.Scheme.generalized_rush_larsen])

@dataclass
class WeaklyCoupledModel:
    ep: monodomain.MonodomainSolver
    #mech: hyperelasticity.HyperelasticProblem

    def __post_init__(self):
        self.t = self.ep.t
        self.dt = self.ep.dt
        self.domain = self.ep.domain
        self.Ta_index = self.ep.ode.model.monitor_index("Ta")
        self.Ta_space = fem.functionspace(self.domain, ("DG", 1))
        self.Ta = fem.Function(self.Ta_space)
        self.Ta_lagrange = fem.Function(self.ep.pde.V)

        with io.XDMFFile(MPI.COMM_WORLD, "Ta.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.ep.domain)

    def _save_Ta(self):
        Ta_array = self.ep.ode.model.monitor_values(self.t.value, 
                                              self.ep.ode.states, 
                                              self.ep.ode.params)[self.Ta_index]
        self.Ta_lagrange.x.array[:] = Ta_array
        self.Ta.interpolate(self.Ta_lagrange)

        adios4dolfinx.write_function_on_input_mesh("Ta.bp",
                                                   self.Ta,
                                                   mode=adios4dolfinx.adios2_helpers.adios2.Mode.Write,
                                                   time=self.t.value,
                                                   name="Ta")

    def solve_ep_save_Ta(self, T):
        while self.t.value < T + self.dt:
            self.ep.step()
            self._save_Ta()
            print("Solved for t = ", self.t.value)


