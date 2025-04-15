import importlib
import logging
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import numba
import numpy as np
import ufl
import ufl.tensors
from dolfinx import fem, geometry, io, mesh
from dolfinx.fem.petsc import LinearProblem
from numba import jit

from nmcemfem.utils import pprint

logger = logging.getLogger(__name__)

class PDESolver:
    def __init__(self, domain: mesh.Mesh, ode_element: tuple[str, int]):
        """Intialize PDESolver instance. PDE space is always ("Lagrange", 1).

        Args:
            domain (mesh.Mesh): existing domain
            ode_element (tuple[str, int]): element to use for the ODE space. Example ("Lagrange", 2)
        """
        self.domain = domain
        if ode_element[0] == "Q":
            element = basix.ufl.quadrature_element(
                cell=self.domain.ufl_cell().cellname(), degree=ode_element[1], scheme="default"
            )
            self.V_ode = fem.functionspace(domain, element)
            self.dx = ufl.dx(domain=self.domain, metadata={"quadrature_degree": ode_element[1]})
        else:
            self.V_ode = fem.functionspace(domain, ode_element)
            self.dx = ufl.dx(domain=self.domain, metadata={"quadrature_degree": 4})

        pde_element = ("Lagrange", 1)
        self.V_pde = fem.functionspace(domain, pde_element)

        self.t = fem.Constant(domain, 0.0)
        self.x = ufl.SpatialCoordinate(domain)

        self.v_ode = fem.Function(self.V_ode)
        self.v_pde = fem.Function(self.V_pde)
        self.v_pde.name = "membrane potential"
        self.v_expr = fem.Expression(self.v_pde, self.V_ode.element.interpolation_points())

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
            I_stim (function): takes in spatial coordinate x and time t and
                outputs the stimulating current in that point.
            dt (float): timestep
            theta (float): parameter for operator splitting.
                theta=1/2 gives Strang splitting. theta between 0 and 1.
            solver_type (str): "PREONLY" for a direct method, "CG" for an iterative method
        """
        v = ufl.TrialFunction(self.V_pde)
        phi = ufl.TestFunction(self.V_pde)
        a = phi * v * self.dx + dt * theta * ufl.dot(ufl.grad(phi), M * ufl.grad(v)) * self.dx
        if np.isclose(theta, 1.0):
            L = phi * (self.v_ode + dt * I_stim) * self.dx
        else:
            L = (
                phi * (self.v_ode + dt * I_stim) * self.dx
                - dt * (1 - theta) * ufl.dot(ufl.grad(phi), M * ufl.grad(self.v_ode)) * self.dx
            )
        self.solver = LinearProblem(a, L, u=self.v_pde)
        fem.petsc.assemble_matrix(self.solver.A, self.solver.a)
        if solver_type == "PREONLY":
            self.solver.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.solver.getPC().setType(PETSc.PC.Type.LU)
            self.solver.solver.setErrorIfNotConverged(True)
            self.solver.solver.getPC().setFactorSolverType("mumps")
        elif solver_type == "CG":
            self.solver.solver.setErrorIfNotConverged(True)
            self.solver.solver.setType(PETSc.KSP.Type.CG)
            self.solver.solver.getPC().setType(PETSc.PC.Type.SOR)
        self.solver.A.assemble()

    def solve_pde_step(self):
        """Take one step of PDE solver"""
        with self.solver.b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector(self.solver.b, self.solver.L)
        self.solver.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        self.solver.solve()
        self.v_pde.x.scatter_forward()
        self.v_ode.interpolate(self.v_expr)

def compile_ode(odefile, scheme) -> bool:
    try:
        import gotranx
    except ModuleNotFoundError:
        raise ImportError("gotranx not found. Please install gotranx.")
    ode_path = (Path(__file__).parent / "odes" / odefile).with_suffix(".ode")
    if not ode_path.exists():
        raise ValueError(f"Could not find {ode_path}")
    if MPI.COMM_WORLD.rank == 0:
        loaded_ode = gotranx.load_ode(ode_path)
        code = gotranx.cli.gotran2py.get_code(loaded_ode, scheme=[gotranx.schemes.Scheme[scheme]])
        with open(ode_path.with_suffix(".py"), "w") as f:
            f.write(code)
    MPI.COMM_WORLD.barrier()
    return ode_path.with_suffix(".py").exists()


class ODESolver:
    def __init__(
        self,
        odefile: str,
        scheme: str,
        num_nodes: int,
        v_name: str = "v",
        initial_states: dict | None = None,
        recompile_ode: bool = False,
    ):
        """Intialize ODESolver instance

        Args:
            odefile (str): name of .ode file in odes/
            scheme (str): scheme to use for solving ODEs.
                Either "forward_explicit_euler" or "generalized_rush_larsen".
            num_nodes (int): number of nodes (locally or globally)
            initial_states (dict or None): dictionary of initial states.
                If None (Default), uses default from .ode file.
            v_name (str): name of transmembrane potential in .odefile. Defaults to "v".
            recompile_ode (bool): recompile ode file. Defaults to False.
        Raises:
            ImportError: if odefile cannot be found.
        """
        if recompile_ode:
            compile_ode(odefile, scheme)
        try:
            # Try importing python module if exists
            self.model = importlib.import_module(f".odes.{odefile}", package=__package__)
        except ImportError as e:
            recompile_ode = compile_ode(odefile, scheme)
            self.model = importlib.import_module(f".odes.{odefile}", package=__package__)

        if initial_states:
            init = self.model.init_state_values(**initial_states)
        else:
            init = self.model.init_state_values()

        self.states = np.tile(init, (num_nodes, 1)).T
        self.v_index = self.model.state_index(v_name)

        init_params = self.model.init_parameter_values()
        self.params = np.tile(init_params, (num_nodes, 1)).T
        self.odesolver = getattr(self.model, scheme)
        self.odesolver = jit(nopython=True)(self.odesolver)

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
        theta (float): parameter for operator splitting.
            theta=1/2 gives Strang splitting. theta between 0 and 1.
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
            scheme (str): scheme to use for solving ODEs.
                Either "forward_explicit_euler" or "generalized_rush_larsen".
            initial_states (dict or None): dictionary of initial states.
                If None, uses default from .ode file.
            v_name (str): name of transmembrane potential in .odefile. Defaults to "v".
        """
        # size_local is different from size of v_ode
        # size_local = size(v_ode) + ghost nodes
        # num_nodes = self.pde.V_ode.dofmap.index_map.size_local
        num_nodes = self.pde.v_ode.x.array.shape[0]
        self.ode = ODESolver(
            odefile, scheme, num_nodes, initial_states=initial_states, v_name=v_name
        )

    def set_stimulus(self, I_stim):
        """Set stimulating current. = 1/(chi*C_m) * I_stim

        Args:
            I_stim (function): takes in spatial coordinate x and time t
                and outputs the stimulating current in that point.
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
        """Setup solver. set_conductivity, set_stimulus, and set_cell_model should be called prior.

        Args:
            solver_type (str, optional): "PREONLY" for a direct method, "CG" for an iterative method
                Defaults to "PREONLY".
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

    def solve(self, T: float, vtx_title: Path | bool = False):
        """Solve electrophysiology without mechanics feedback.

        Args:
            T (float): end time
            vtx_title (Path | Bool, optional): Filename to save solution to. Does not save if False.
                Defaults to False.

        Returns:
            v_pde (fem.Function): Transmembrane potential function at last timestep.
                In PDE space ("Lagrange", 1).
            x (ufl.SpatialCoordinate): spatial coordinate of domain.
            t (fem.Constant): time at last timestep.
        """
        if vtx_title:
            vtx = io.VTXWriter(
                MPI.COMM_WORLD, vtx_title.with_suffix(".bp"), [self.pde.v_pde], engine="BP4"
            )
        while self.t.value < T + self.dt:
            self.step()
            if vtx_title:
                vtx.write(self.t.value)

        return self.pde.v_pde, self.pde.x, self.t

    def solve_activation_times(self, points, line, T):
        bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)
        # Find cells whose bounding-box collide with the the points
        potential_colliding_cells_points = geometry.compute_collisions_points(bb_tree, points)
        # Choose one of the cells that contains the point
        adj_points = geometry.compute_colliding_cells(
            self.domain, potential_colliding_cells_points, points
        )
        indices_points = np.flatnonzero(adj_points.offsets[1:] - adj_points.offsets[:-1])
        cells_points = adj_points.array[adj_points.offsets[indices_points]]
        points_on_proc = points[indices_points]

        # Find cells whose bounding-box collide with the the points
        potential_colliding_cells_line = geometry.compute_collisions_points(bb_tree, line)
        # Choose one of the cells that contains the point
        adj_line = geometry.compute_colliding_cells(
            self.domain, potential_colliding_cells_line, line
        )
        indices_line = np.flatnonzero(adj_line.offsets[1:] - adj_line.offsets[:-1])
        cells_line = adj_line.array[adj_line.offsets[indices_line]]
        line_on_proc = line[indices_line]

        # Global
        times_points = -np.ones(len(points))
        times_line = -np.ones(len(line))
        # Local
        times_points_on_proc = -np.ones(len(points_on_proc))
        times_line_on_proc = -np.ones(len(line_on_proc))

        while self.t.value <= T and np.min(times_points) < 0:
            self.step()
            if self.domain.comm.rank == 0:
                logger.info(f"Solved for t = {np.round(self.t.value, 3)}")

            evaluated_points = self.pde.v_pde.eval(points_on_proc, cells_points)
            for i in range(len(points_on_proc)):
                if times_points_on_proc[i] < 0 and evaluated_points[i] > 0:
                    times_points_on_proc[i] = np.round(self.t.value, 3)
                    times_points[indices_points[i]] = times_points_on_proc[i]
                    logger.info(f"Point {indices_points[i]} activated")

            evaluated_lines = self.pde.v_pde.eval(line_on_proc, cells_line)
            for i in range(len(line_on_proc)):
                if times_line_on_proc[i] < 0 and evaluated_lines[i] > 0:
                    times_line_on_proc[i] = np.round(self.t.value, 3)
                    times_line[indices_line[i]] = times_line_on_proc[i]

        times_points_global = np.copy(times_points)
        self.mesh_comm.Allreduce(times_points, times_points_global, op=MPI.MAX)
        times_line_global = np.copy(times_line)
        self.mesh_comm.Allreduce(times_line, times_line_global, op=MPI.MAX)

        return times_points_global, times_line_global
