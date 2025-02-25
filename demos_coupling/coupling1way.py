import logging
from pathlib import Path

import numpy as np
import ufl
from pint import UnitRegistry

from nmcemfem.coupled_model import WeaklyCoupledModel
from nmcemfem.hyperelasticity import HyperelasticProblem
from nmcemfem.monodomain import MonodomainSolver

logging.basicConfig(level=logging.INFO)
ureg = UnitRegistry()


initial_states = {
    "V": -85.23,  # mV
    "Xr1": 0.00621,
    "Xr2": 0.4712,
    "Xs": 0.0095,
    "m": 0.00172,
    "h": 0.7444,
    "j": 0.7045,
    "d": 3.373e-05,
    "f": 0.7888,
    "f2": 0.9755,
    "fCass": 0.9953,
    "s": 0.999998,
    "r": 2.42e-08,
    "Ca_i": 0.000126,  # millimolar
    "R_prime": 0.9073,
    "Ca_SR": 3.64,  # millimolar
    "Ca_ss": 0.00036,  # millimolar
    "Na_i": 8.604,  # millimolar
    "K_i": 136.89,  # millimolar
}
h = 0.5
dt = 0.05
theta = 1


def get_model_problem(ode_element, Ta_element):
    ep_solver = MonodomainSolver(h, dt, theta)

    Lx, Ly, Lz = 3, 7, 20  # mm
    L = (Lx, Ly, Lz)

    ep_solver.set_rectangular_mesh(L, ode_element)

    ep_solver.set_cell_model(
        odefile="tentusscher_land_1way",
        scheme="generalized_rush_larsen",
        initial_states=initial_states,
        v_name="V",
    )

    chi = 1400 * ureg("1/cm")
    C_m = 1 * ureg("uF/cm**2")

    intra_trans, extra_trans = 0.019 * ureg("S/m"), 0.24 * ureg("S/m")  # Sm^-1
    intra_long, extra_long = 0.17 * ureg("S/m"), 0.62 * ureg("S/m")
    trans_conductivity = intra_trans * extra_trans / (intra_trans + extra_trans)
    long_conductivity = intra_long * extra_long / (intra_long + extra_long)

    trans_conductivity_scaled = (trans_conductivity / (chi * C_m)).to("mm**2/ms").magnitude
    long_conductivity_scaled = (long_conductivity / (chi * C_m)).to("mm**2/ms").magnitude
    M = ufl.tensors.as_tensor(
        np.diag([trans_conductivity_scaled, trans_conductivity_scaled, long_conductivity_scaled])
    )

    ep_solver.set_conductivity(M)

    stim_amplitude = 50000 * ureg("uA/cm**3")
    amplitude_magnitude = (stim_amplitude / (C_m * chi)).to("mV/ms").magnitude

    def I_stim(x, t):
        condition = ufl.And(ufl.And(x[0] <= 1.5, x[1] <= 1.5), ufl.And(x[2] <= 1.5, t <= 2))
        return ufl.conditional(condition, amplitude_magnitude, 0)

    ep_solver.set_stimulus(I_stim)
    ep_solver.setup_solver()

    mech_solver = HyperelasticProblem(h=h, lagrange_order=2)
    f0 = ufl.unit_vector(2, 3)
    s0 = ufl.unit_vector(1, 3)
    mech_solver.set_existing_domain(ep_solver.domain, f0, s0, Ta_element)

    left = lambda x: np.isclose(x[0], 0)
    front = lambda x: np.isclose(x[1], 0)
    bottom = lambda x: np.isclose(x[2], 0)

    boundaries = [left, front, bottom]
    vals = [0, 1, 2]
    bc_types = ["d2", "d2", "d2"]
    mech_solver.boundary_conditions(boundaries, vals, bc_types)
    mech_solver.setup_solver()

    coupled_solver = WeaklyCoupledModel(ep_solver, mech_solver)
    return coupled_solver

func_dir = Path(__file__).parents[1] / "saved_funcs"

# Only works for quadrature degree 2.
# Otherwise, ValueError: Mismatch of tabulation points and element points.
ode_element = ("Q", 4)

Ta_element = ("DG", 1)
problem = get_model_problem(ode_element, Ta_element)
problem.solve(60, save_tofile=False)
