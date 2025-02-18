# From https://computationalphysiology.github.io/zero-mech/examples/electro-mechanics/electro_mechanics.html
import ufl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import sys

sys.path.append("../")
from src.monodomain import ODESolver


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
        P11 = (
            Ta
            + a * (lmbda**2 - 1 / lmbda) * np.exp(b * (lmbda**2 + 2 / lmbda - 3))
            + 2
            * lmbda**2
            * af
            * subplus(lmbda**2 - 1)
            * np.exp(bf * subplus(lmbda**2 - 1) ** 2)
            + p
        )
        P22 = (
            a * (lmbda**2 - 1 / lmbda) * np.exp(b * (lmbda**2 + 2 / lmbda - 3))
            + 2
            * lmbda**2
            * af
            * subplus(lmbda**2 - 1)
            * np.exp(bf * subplus(lmbda**2 - 1) ** 2)
            - p
        )
        return np.array([P11, P22], dtype=np.float64)

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
        P11 = (
            Ta
            + a * (lmbda**2 - 1 / lmbda) * np.exp(b * (lmbda**2 + 2 / lmbda - 3))
            + 2
            * lmbda**2
            * af
            * subplus(lmbda**2 - 1)
            * np.exp(bf * subplus(lmbda**2 - 1) ** 2)
            + p
        )
        P22 = (
            a * (lmbda**2 - 1 / lmbda) * np.exp(b * (lmbda**2 + 2 / lmbda - 3))
            + 2
            * lmbda**2
            * af
            * subplus(lmbda**2 - 1)
            * np.exp(bf * subplus(lmbda**2 - 1) ** 2)
            - p
        )
        return np.array([P11, P22], dtype=np.float64)

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
            self.Ta[i] = self.monitor_values(ti, self.states, self.params)[
                self.Ta_index
            ]

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
        fig.savefig(filename)


weakcoupling = zeroD_coupling(dt=0.1, T=400)
weakcoupling.solve_weak()
weakcoupling.plot("zeroD_weakcoupling.png")

strongcoupling = zeroD_coupling(dt=0.1, T=400)
strongcoupling.solve_strong()
strongcoupling.plot("zeroD_strongcoupling.png")

monolithiccoupling = zeroD_coupling(dt=0.1, T=400)
monolithiccoupling.solve_monolithic()
monolithiccoupling.plot("zeroD_monolithic.png")
