import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ufl
from pint import UnitRegistry

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


def solve_model_problem(h, dt, ode_element, T=80):
    ep_solver = MonodomainSolver(h, dt, theta=1)

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
    points = np.array(
        [
            [0, 0, 0],
            [0, Ly, 0],
            [0, 0, Lz],
            [0, Ly, Lz],
            [Lx, 0, 0],
            [Lx, Ly, 0],
            [Lx, 0, Lz],
            [Lx, Ly, Lz],
            [Lx / 2, Ly / 2, Lz / 2],
        ]
    )
    line = np.linspace([0, 0, 0], [Lx, Ly, Lz], 100)
    times_points, times_line = ep_solver.solve_activation_times(points, line, T)

    # Save data
    data_directory = (
        Path(__file__).parent
        / "activation_times_data"
        / f"h={h},dt={dt}"
        / f"{ode_element[0]}{ode_element[1]}"
    )
    if ep_solver.mesh_comm.rank == 0:
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        np.savetxt(data_directory / "points.txt", times_points, fmt="%1.2f")
        np.savetxt(data_directory / "line.txt", times_line, fmt="%1.2f")
    print(f"Saved for h={h}, dt={dt}")


def solve_benchmark(ode_element, skip = [], T=70):
    hs = [0.5, 0.2, 0.1]
    dts = [0.05, 0.01, 0.005]
    for h in hs:
        for dt in dts:
            if [h, dt] not in skip:
                solve_model_problem(h, dt, ode_element, T)

def read_file_plot_line(h, dt, elements):
    data_directory = (
        Path(__file__).parent
        / "activation_times_data"
        / f"h={h},dt={dt}"
    )
    Lx, Ly, Lz = 3, 7, 20 #mm
    fig, ax = plt.subplots(figsize=(8,5))
    for element in elements:
        activation_time_line = np.loadtxt(data_directory / element / "line.txt")
        dist = np.linspace(0, np.sqrt(Lx**2 + Ly**2 + Lz**2), len(activation_time_line))
        ax.plot(dist, activation_time_line, label=element)
    ax.set_ylabel('activation time (ms)')
    ax.set_xlabel('distance (mm)')
    ax.grid(True)
    ax.set_title('Activation time along line')
    ax.legend()
    fig.savefig(f"lineplot_h={h}_dt={dt}.png")

ode_element = ("Q", 2)
solve_model_problem(0.2, 0.05, ode_element)
#read_file_plot_line(h=0.5, dt=0.05, elements=["Lagrange1", "Q2", "Q3", "Q4", "Q5"])
