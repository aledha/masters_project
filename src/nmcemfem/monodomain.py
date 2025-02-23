from dolfinx import fem, mesh, io
import dolfinx.fem.petsc as petsc
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
import ufl
from pathlib import Path
from dataclasses import dataclass
import importlib
import ufl.tensors


class PDESolver:
    def __init__(self, domain: mesh.Mesh, ode_element: tuple[str, int]):
        """Intialize PDESolver instance. PDE space is always ("Lagrange", 1).

        Args:
            domain (mesh.Mesh): existing domain
            ode_element (tuple[str, int]): element to use for the ODE space. Example ("Lagrange", 2)
        """
        self.domain = domain
        self.V_ode = fem.functionspace(domain, ode_element)
        pde_element = ("Lagrange", 1)
        self.V_pde = fem.functionspace(domain, pde_element)

        self.t = fem.Constant(domain, 0.0)
        self.x = ufl.SpatialCoordinate(domain)

        self.v_ode = fem.Function(self.V_ode)
        self.v_pde = fem.Function(self.V_pde)
        self.v_pde.name = "membrane potential"

    def setup_pde_solver(
        self,
        M: ufl.tensors.ListTensor,
        I_stim,
        dt: float,
        theta: float,
        solver_type: str,
    ):
        """Initialize PDE solver

        Args:
            M (ufl.tensors.ListTensor): Conductivity tensor,
            I_stim (function): takes in spatial coordinate x and time t and outputs the stimulating current in that point.
            dt (float): timestep
            theta (float): parameter for operator splitting. theta=1/2 gives Strang splitting. theta between 0 and 1.
            solver_type (str): "PREONLY" for a direct method, "CG" for an iterative method
        """
        v = ufl.TrialFunction(self.V_pde)
        phi = ufl.TestFunction(self.V_pde)
        dx = ufl.dx(domain=self.domain)
        a = phi * v * dx + dt * theta * ufl.dot(ufl.grad(phi), M * ufl.grad(v)) * dx
        L = (
            phi * (self.v_pde + dt * I_stim) * dx
            - dt * (1 - theta) * ufl.dot(ufl.grad(phi), M * ufl.grad(self.v_pde)) * dx
        )
        compiled_a = fem.form(a)
        A = petsc.assemble_matrix(compiled_a)
        A.assemble()
        self.compiled_L = fem.form(L)
        self.b = fem.Function(self.V_pde)

        self.solver = PETSc.KSP().create(self.domain.comm)
        if solver_type == "PREONLY":
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)
            self.solver.setErrorIfNotConverged(True)
            self.solver.getPC().setFactorSolverType("mumps")
        elif solver_type == "CG":
            self.solver.setErrorIfNotConverged(True)
            self.solver.setType(PETSc.KSP.Type.CG)
            self.solver.getPC().setType(PETSc.PC.Type.SOR)
        self.solver.setOperators(A)

    def solve_pde_step(self):
        """Take one step of PDE solver"""
        self.b.x.array[:] = 0
        petsc.assemble_vector(self.b.x.petsc_vec, self.compiled_L)
        self.solver.solve(self.b.x.petsc_vec, self.v_pde.x.petsc_vec)
        self.v_ode.interpolate(self.v_pde)
        self.v_ode.x.scatter_forward()


class ODESolver:
    def __init__(
        self,
        odefile: str,
        scheme: str,
        num_nodes: int,
        v_name: str = "v",
        initial_states: dict | None = None,
    ):
        """Intialize ODESolver instance

        Args:
            odefile (str): name of .ode file in odes/
            scheme (str): scheme to use for solving ODEs. Either "forward_explicit_euler" or "generalized_rush_larsen".
            num_nodes (int): number of nodes (locally or globally)
            initial_states (dict or None): dictionary of initial states. If None (Default), uses default from .ode file.
            v_name (str): name of transmembrane potential in .odefile. Defaults to "v".

        Raises:
            ImportError: if odefile cannot be found.
        """
        try:
            self.model = importlib.import_module(f"odes.{odefile}")
        except ImportError as e:
            raise ImportError(f"Failed to import {odefile}: {e}")

        if initial_states:
            init = self.model.init_state_values(**initial_states)
        else:
            init = self.model.init_state_values()

        self.states = np.tile(init, (num_nodes, 1)).T
        self.v_index = self.model.state_index(v_name)

        init_params = self.model.init_parameter_values()
        self.params = np.tile(init_params, (num_nodes, 1)).T
        self.odesolver = getattr(self.model, scheme)

    def set_param(self, name: str, value: float | np.ndarray):
        """Set parameter

        Args:
            name (str): name of parameter
            value (float | np.ndarray): value or array to set parameter to
        """
        param_index = self.model.parameter_index(name)
        self.params[param_index] = value

    def set_state(self, state_name: str, state: np.ndarray):
        """Set state

        Args:
            state_name (str): name of state
            value (float | np.ndarray): array to set state to
        """
        state_index = self.model.state_index(state_name)
        self.states[state_index, :] = state[:]

    def solve_ode_step(self, t: float, dt: float):
        """Take step of scheme

        Args:
            t (float): time
            dt (float): timestep
        """
        self.states[:] = self.odesolver(self.states, t, dt, self.params)

    def update_v(self, v_ode: fem.Function):
        """Update potential in states

        Args:
            v_ode (fem.Function): potential function in ODE space
        """
        self.states[self.v_index, :] = v_ode.x.array[:]

    def get_v(self):
        """Get potential from states

        Returns:
            np.ndarray
        """
        return self.states[self.v_index, :]


@dataclass
class MonodomainSolver:
    h: float
    dt: float
    theta: float
    """Intialize operator splitting solver for monodomain electrophysiology model.

    Args:
        h (float): spatial step size
        dt (float): temporal step size
        theta (float): parameter for operator splitting. theta=1/2 gives Strang splitting. theta between 0 and 1.
    """

    def set_rectangular_mesh(self, L: tuple[float, float, float], ode_element: tuple[str, int]):
        """Set rectangular mesh

        Args:
            L (tuple[int]): size of rectangle in x-, y-, and z-direction. L=(Lx, Ly, Lz).
            ode_element (tuple[str, int]): element to use for the ODE space. Example ("Lagrange", 2)
        """
        self.mesh_comm = MPI.COMM_WORLD
        Lx, Ly, Lz = L
        self.domain = mesh.create_box(
            self.mesh_comm,
            [[0, 0, 0], [Lx, Ly, Lz]],
            n=[int(Lx / self.h), int(Ly / self.h), int(Lz / self.h)],
        )
        self.pde = PDESolver(self.domain, ode_element)
        self.x = self.pde.x
        self.t = self.pde.t

    def set_cell_model(
        self,
        odefile: str,
        scheme: str,
        initial_states: dict | None = None,
        v_name: str = "v",
    ):
        """Set cell model from .ode file.

        Args:
            odefile (str): name of .ode file found in odes/
            scheme (str): scheme to use for solving ODEs. Either "forward_explicit_euler" or "generalized_rush_larsen".
            initial_states (dict or None): dictionary of initial states. If none, uses default from .ode file.
            v_name (str): name of transmembrane potential in .odefile. Defaults to "v".
        """
        # num_nodes = self.pde.V_ode.dofmap.index_map.size_local
        num_nodes = self.pde.v_ode.x.array.shape[0]
        self.ode = ODESolver(
            odefile, scheme, num_nodes, initial_states=initial_states, v_name=v_name
        )

    def set_stimulus(self, I_stim):
        """Set stimulating current. = 1/(chi*C_m) * I_stim

        Args:
            I_stim (function): takes in spatial coordinate x and time t and outputs the stimulating current in that point.
        """
        self.ode.set_param("stim_amplitude", 0)
        self.I_stim = I_stim(self.x, self.t)

    def set_conductivity(self, M: ufl.tensors.ListTensor):
        """Set conductivity tensor defined as M = 1/(chi*C_m) * lambda/(1+lambda) * M_i

        Args:
            M (ufl.tensors.ListTensor): conductivity tensor
        """
        self.M = M

    def setup_solver(self, solver_type: str = "PREONLY"):
        """Setup solver. set_conductivity, set_stimulus, and set_cell_model should be called prior to this function.

        Args:
            solver_type (str, optional): "PREONLY" for a direct method, "CG" for an iterative method. Defaults to "PREONLY".
        """
        self.pde.setup_pde_solver(self.M, self.I_stim, self.dt, self.theta, solver_type)

    def _transfer_ode_to_pde(self):
        self.pde.v_ode.x.array[:] = self.ode.get_v()

    def _transfer_pde_to_ode(self):
        self.ode.update_v(self.pde.v_ode)

    def step(self):
        # Step 1
        self.ode.solve_ode_step(self.t.value, self.theta * self.dt)
        self.t.value += self.theta * self.dt
        self._transfer_ode_to_pde()

        # Step 2
        self.pde.solve_pde_step()
        self._transfer_pde_to_ode()

        # Step 3
        if self.theta < 1.0:
            self.ode.solve_ode_step(self.t.value, (1 - self.theta) * self.dt)
            self.t.value += (1 - self.theta) * self.dt
            self._transfer_ode_to_pde()

    def solve(self, T: float, vtx_title: str | None = None):
        """Solve electrophysiology without mechanics feedback.

        Args:
            T (float): end time
            vtx_title (str | None, optional): Filename to save solution to. Does not save if None. Defaults to None.

        Returns:
            v_pde (fem.Function): Transmembrane potential function at last timestep. In PDE space ("Lagrange", 1).
            x (ufl.SpatialCoordinate): spatial coordinate of domain.
            t (fem.Constant): time at last timestep.
        """
        if vtx_title:
            vtx = io.VTXWriter(MPI.COMM_WORLD, vtx_title + ".bp", [self.pde.v_pde], engine="BP4")
        while self.t.value < T + self.dt:
            self.step()
            if vtx_title:
                vtx.write(self.t.value)

        return self.pde.v_pde, self.pde.x, self.t
