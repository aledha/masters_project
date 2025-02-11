from dolfinx import fem, mesh, io
import dolfinx.fem.petsc as petsc
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
import ufl
from pathlib import Path
from dataclasses import dataclass
import importlib
import sys

import ufl.tensors

sys.path.append(str(Path(__file__).resolve().parents[1]))


class PDESolver:
    def __init__(self, domain, element):
        self.domain = domain

        self.V = fem.functionspace(domain, element)
        self.t = fem.Constant(domain, 0.0)
        self.x = ufl.SpatialCoordinate(domain)

        self.vn = fem.Function(self.V)
        self.vn.name = "vn"

    def setup_pde_solver(self, M, I_stim, dt, theta, solver_type="PREONLY"):
        v = ufl.TrialFunction(self.V)
        phi = ufl.TestFunction(self.V)
        dx = ufl.dx(domain=self.domain)
        a = phi * v * dx + dt * theta * ufl.dot(ufl.grad(phi), M * ufl.grad(v)) * dx
        L = (
            phi * (self.vn + dt * I_stim) * dx
            - dt * (1 - theta) * ufl.dot(ufl.grad(phi), M * ufl.grad(self.vn)) * dx
        )
        compiled_a = fem.form(a)
        A = petsc.assemble_matrix(compiled_a)
        A.assemble()

        self.compiled_L = fem.form(L)
        self.b = fem.Function(self.V)
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
        self.b.x.array[:] = 0
        petsc.assemble_vector(self.b.x.petsc_vec, self.compiled_L)
        self.solver.solve(self.b.x.petsc_vec, self.vn.x.petsc_vec)
        self.vn.x.scatter_forward()


class ODESolver:
    def __init__(self, odefile, scheme, num_nodes, initial_states=None, v_name="v"):
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

    def set_param(self, name, value):
        param_index = self.model.parameter_index(name)
        self.params[param_index] = value

    def set_state(self, state_name, state):
        state_index = self.model.state_index(state_name)
        self.states[state_index, :] = state[:]

    def solve_ode_step(self, t, dt):
        self.states[:] = self.odesolver(self.states, t, dt, self.params)

    def update_vn(self, vn):
        self.states[self.v_index, :] = vn.x.array[:]

    def get_vn(self):
        return self.states[self.v_index, :]


@dataclass
class MonodomainSolver:
    h: float
    dt: float
    theta: float

    def set_rectangular_mesh(self, L, element):
        self.mesh_comm = MPI.COMM_WORLD
        Lx, Ly, Lz = L
        self.domain = mesh.create_box(
            self.mesh_comm,
            [[0, 0, 0], [Lx, Ly, Lz]],
            n=[int(Lx / self.h), int(Ly / self.h), int(Lz / self.h)],
        )
        self.pde = PDESolver(self.domain, element)
        self.x = self.pde.x
        self.t = self.pde.t

    def set_cell_model(self, odefile, scheme, initial_states=None, v_name="v"):
        num_nodes = self.pde.V.dofmap.index_map.size_global
        self.ode = ODESolver(
            odefile, scheme, num_nodes, initial_states=initial_states, v_name=v_name
        )

    def set_stimulus(self, I_stim):
        self.ode.set_param("stim_amplitude", 0)
        self.I_stim = I_stim(self.x, self.t)  # = 1/(chi*C_m) * I_stim

    def set_conductivity(self, M):
        self.M = M  # =1/(chi*C_m) * lambda/(1+lambda) * M_i

    def setup_solver(self, solver_type="PREONLY"):
        self.pde.setup_pde_solver(self.M, self.I_stim, self.dt, self.theta, solver_type)

    def _transfer_ode_to_pde(self):
        self.pde.vn.x.array[:] = self.ode.get_vn()

    def _transfer_pde_to_ode(self):
        self.ode.update_vn(self.pde.vn)

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

    def solve(self, T, vtx_title=None):
        if vtx_title:
            vtx = io.VTXWriter(
                MPI.COMM_WORLD, vtx_title + ".bp", [self.pde.vn], engine="BP4"
            )
        while self.t.value < T + self.dt:
            self.step()
            if vtx_title:
                vtx.write(self.t.value)

        return self.pde.vn, self.pde.x, self.t
