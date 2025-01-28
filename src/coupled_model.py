from dolfinx import fem, mesh, default_scalar_type, io, log
import numpy as np
from mpi4py import MPI
import ufl
from scifem import NewtonSolver
from dataclasses import dataclass
import ufl.geometry

import hyperelasticity, monodomain

@dataclass
class WeaklyCoupledModel:
    ep: monodomain.MonodomainSolver
    mech: hyperelasticity.HyperelasticProblem
