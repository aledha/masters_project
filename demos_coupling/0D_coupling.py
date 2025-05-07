# From https://computationalphysiology.github.io/zero-mech/examples/electro-mechanics/electro_mechanics.html
from pathlib import Path

from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from scipy.optimize import root

from nmcemfem.monodomain import ODESolver

figure_dir = Path("saved_figures")


def subplus(x):
    if x > 0:
        return x
    return 0


class zeroD_coupling:
    def __init__(self, dt, T):
        self.dt = dt
        self.t = np.arange(0, T, dt)
        ode = ODESolver(
            odefile="tentusscher_land_1way",
            scheme="generalized_rush_larsen",
            num_nodes=1,
            v_name="V",
        )
        self.odesolver = ode.odesolver
        self.monitor_values = ode.model.monitor_values
        ode.set_param("stim_period", T)
        self.states = ode.states.T[0]
        self.params = ode.params.T[0]
        self.V_index = ode.model.state_index("V")
        self.Ca_index = ode.model.state_index("Ca_i")
        self.Ta_index = ode.model.monitor_index("Ta")
        self.lmbda_index = ode.model.parameter_index("lmbda")
        self.dLmbda_index = ode.model.parameter_index("dLambda")

        self.V = np.zeros(len(self.t))
        self.Ca = np.zeros(len(self.t))
        self.Ta = np.zeros(len(self.t))
        self.lmbdas = np.ones(len(self.t))
        self.dLambdas = np.zeros(len(self.t))
        self.ps = np.zeros(len(self.t))

    def _func(self, x, Ta):
        lmbda, p = x
        a, b, af, bf = 2.28, 9.726, 1.685, 15.779
        L1 = (
            Ta
            + a * (lmbda**2 - 1 / lmbda) * np.exp(b * (lmbda**2 + 2 / lmbda - 3))
            + 2 * lmbda**2 * af * subplus(lmbda**2 - 1) * np.exp(bf * subplus(lmbda**2 - 1) ** 2)
            + p
        )
        L2 = (
            2 * a * (lmbda**2 - 1 / lmbda) * np.exp(b * (lmbda**2 + 2 / lmbda - 3))
            + 4 * lmbda**2 * af * subplus(lmbda**2 - 1) * np.exp(bf * subplus(lmbda**2 - 1) ** 2)
            - p
        )
        return np.array([L1, L2], dtype=np.float64)

    def _step(self, i, ti):
        self.states = self.odesolver(self.states, ti, self.dt, self.params)
        self.V[i] = self.states[self.V_index]
        self.Ca[i] = self.states[self.Ca_index]
        self.Ta[i] = self.monitor_values(ti, self.states, self.params)[self.Ta_index]

        solution = root(
            self._func,
            x0=np.array([self.lmbdas[i - 1], self.ps[i - 1]]),
            args=(self.Ta[i],),
            method="hybr",
        )
        self.lmbdas[i], self.ps[i] = solution.x
        self.dLambdas[i] = (self.lmbdas[i] - self.lmbdas[i - 1]) / self.dt

    def solve_weak(self):
        for i, ti in enumerate(self.t):
            self._step(i, ti)

    def solve_strong(self):
        for i, ti in enumerate(self.t):
            self._step(i, ti)
            self.params[self.lmbda_index] = self.lmbdas[i]
            self.params[self.dLmbda_index] = self.dLambdas[i]

    def _func_monolithic(self, x, ti, states_new, states_copy, prev_lmbda):
        lmbda, p = x
        dLmbda = (lmbda - prev_lmbda) / self.dt
        self.params[self.lmbda_index] = lmbda
        self.params[self.dLmbda_index] = dLmbda
        states_new[:] = self.odesolver(states_copy, ti, self.dt, self.params)
        Ta = self.monitor_values(ti, states_new, self.params)[self.Ta_index]

        a, b, af, bf = 2.28, 9.726, 1.685, 15.779
        L1 = (
            Ta
            + a * (lmbda**2 - 1 / lmbda) * np.exp(b * (lmbda**2 + 2 / lmbda - 3))
            + 2 * lmbda**2 * af * subplus(lmbda**2 - 1) * np.exp(bf * subplus(lmbda**2 - 1) ** 2)
            + p
        )
        L2 = (
            2 * a * (lmbda**2 - 1 / lmbda) * np.exp(b * (lmbda**2 + 2 / lmbda - 3))
            + 4 * lmbda**2 * af * subplus(lmbda**2 - 1) * np.exp(bf * subplus(lmbda**2 - 1) ** 2)
            - p
        )
        return np.array([L1, L2], dtype=np.float64)

    def solve_monolithic(self):
        for i, ti in enumerate(self.t):
            solution = root(
                self._func_monolithic,
                x0=np.array([self.lmbdas[i - 1], self.ps[i - 1]]),
                args=(ti, self.states, self.states.copy(), self.lmbdas[i - 1]),
                method="hybr",
            )
            self.lmbdas[i], self.ps[i] = solution.x
            self.dLambdas[i] = (self.lmbdas[i] - self.lmbdas[i - 1]) / self.dt
            self.V[i] = self.states[self.V_index]
            self.Ca[i] = self.states[self.Ca_index]
            self.Ta[i] = self.monitor_values(ti, self.states, self.params)[self.Ta_index]

    def plot(self, filename):
        fig, ax = plt.subplots(2, 3, sharex=True)
        ax[0, 0].plot(self.t, self.V)
        ax[1, 0].plot(self.t, self.Ta)
        ax[0, 1].plot(self.t, self.Ca)
        ax[1, 1].plot(self.t, self.dLambdas)
        ax[0, 2].plot(self.t, self.lmbdas)
        ax[1, 2].plot(self.t, self.ps)
        ax[1, 0].set_xlabel("Time (ms)")
        ax[1, 1].set_xlabel("Time (ms)")
        ax[0, 0].set_ylabel("V (mV)")
        ax[1, 0].set_ylabel("Ta (kPa)")
        ax[0, 1].set_ylabel("Ca (mM)")
        ax[1, 1].set_ylabel(r"$\dot{\lambda}$")
        ax[0, 2].set_ylabel(r"$\lambda$")
        ax[1, 2].set_ylabel("p")
        for axi in ax.flatten():
            axi.grid()
        fig.tight_layout()
        fig.savefig(figure_dir / filename.with_suffix(".png"))

    def plot_simple(self, filename):
        fig, ax = plt.subplots(1, 3, figsize=(10,4))
        ax[0].plot(self.t, self.V)
        ax[1].plot(self.t, self.Ta)
        ax[2].plot(self.t, self.lmbdas)
        ax[0].set_xlabel("Time (ms)")
        ax[1].set_xlabel("Time (ms)")
        ax[2].set_xlabel("Time (ms)")
        ax[0].set_ylabel("v (mV)")
        ax[1].set_ylabel("Ta (kPa)")
        ax[2].set_ylabel(r"$\lambda$")
        for axi in ax.flatten():
            axi.grid()
        fig.tight_layout()
        fig.savefig(figure_dir / filename.with_suffix(".png"))

    def make_animation(self, filename, T = 400, frame_step = 10):
        n = 8
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
        def u_func(x, lmbda):
            return [(1 - lmbda) * (1 - x[0]),
                    - (1 - np.sqrt(lmbda)) * (1 - x[1]),
                    - (1 - np.sqrt(lmbda)) * (1 - x[2])]

        def v_func(x, v):
            return x[0] * 0 + v

        U = fem.functionspace(domain, ("Lagrange", 2, (3,)))
        u = fem.Function(U)
        u.name = "Displacement"

        V = fem.functionspace(domain, ("Lagrange", 2))
        v = fem.Function(V)
        v.name = "Membrane potential"
        vtx = io.VTXWriter(
                MPI.COMM_WORLD, figure_dir / filename.with_suffix(".bp"), [u, v], engine="BP4"
            )

        stop = int(T / self.dt)

        for ti, lmbda, vi in zip(self.t[:stop:frame_step], self.lmbdas[:stop:frame_step], self.V[:stop:frame_step]):
            u.interpolate(lambda x: u_func(x, lmbda))
            v.interpolate(lambda x: v_func(x, vi))
            vtx.write(ti)
        vtx.close()


weakcoupling = zeroD_coupling(dt=0.1, T=400)
weakcoupling.solve_weak()
weakcoupling.plot_simple(Path("zeroD_weak_simple"))
weakcoupling.make_animation(Path("zeroD_weak"))

strongcoupling = zeroD_coupling(dt=0.1, T=400)
strongcoupling.solve_strong()
strongcoupling.plot_simple(Path("zeroD_strong_simple"))
strongcoupling.make_animation(Path("zeroD_strong"), T = 100, frame_step=5)

monolithiccoupling = zeroD_coupling(dt=0.1, T=400)
monolithiccoupling.solve_monolithic()
monolithiccoupling.plot_simple(Path("zeroD_monolithic_simple"))
monolithiccoupling.make_animation(Path("zeroD_monolithic"))
