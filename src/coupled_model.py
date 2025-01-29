from dolfinx import fem, mesh, default_scalar_type, io, log
import numpy as np
from mpi4py import MPI
import ufl
from scifem import NewtonSolver
from dataclasses import dataclass
import ufl.geometry

import hyperelasticity, monodomain
import gotranx
import importlib
import sys
from pathlib import Path

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

translateODE('land', [gotranx.schemes.Scheme.generalized_rush_larsen])

@dataclass
class WeaklyCoupledModel:
    ep: monodomain.MonodomainSolver
    mech: hyperelasticity.HyperelasticProblem

    def __post_init__(self):
        self.cai_ep_index = self.ep.ode.model.state_index('Ca_i')
        self.cai_mech_index = self.mech.model.state_index('cai')

    def _cai_to_mech(self):
        # TODO: interpolate to allow coarser mech mesh
        self.mech.states[self.cai_mech_index, :] = self.ep.ode.states[self.cai_ep_index, :]


