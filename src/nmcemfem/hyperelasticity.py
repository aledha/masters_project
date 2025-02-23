from dataclasses import dataclass

from mpi4py import MPI

import numpy as np
import ufl
import ufl.geometry
from dolfinx import default_scalar_type, fem, mesh
from scifem import NewtonSolver

from nmcemfem.utils import pprint


@dataclass
class HyperelasticProblem:
    h: float
    lagrange_order: int

    def __post_init__(self):
        self.bcs = []
        self.states = []
        self.teststates = []
        self.trialstates = []
        self.L = ufl.as_ufl(0.0)

    def _init_functions(self):
        self.V = fem.functionspace(
            self.domain, ("Lagrange", self.lagrange_order, (self.domain.geometry.dim,))
        )
        self.x = ufl.SpatialCoordinate(self.domain)
        self.t = fem.Constant(self.domain, 0.0)
        metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure("dx", domain=self.domain, metadata=metadata)

        self.u = fem.Function(self.V)
        self.v = ufl.TestFunction(self.V)
        self.du = ufl.TrialFunction(self.V)

        self.states.append(self.u)
        self.teststates.append(self.v)
        self.trialstates.append(self.du)

        self.d = len(self.u)
        self.I = ufl.variable(ufl.Identity(self.d))
        self.F = ufl.variable(self.I + ufl.grad(self.u))
        self.C = ufl.variable(self.F.T * self.F)
        self.J = ufl.variable(ufl.det(self.F))

        self.T_space = fem.functionspace(self.domain, ("DG", 0))
        self.Ta = fem.Function(self.T_space)

    def _init_invariants(self, f0: ufl.tensors.ListTensor, s0: ufl.tensors.ListTensor):
        self.f0 = f0
        self.s0 = s0

        self.I1 = ufl.variable(ufl.tr(self.C))
        self.I4f = ufl.variable(ufl.dot(self.f0, ufl.dot(self.C, self.f0)))
        self.I4s = ufl.variable(ufl.dot(self.s0, ufl.dot(self.C, self.s0)))
        self.I8fs = ufl.variable(ufl.dot(self.s0, ufl.dot(self.C, self.f0)))

    def set_rectangular_domain(
        self,
        L: tuple[float, float, float],
        f0: ufl.tensors.ListTensor,
        s0: ufl.tensors.ListTensor,
    ):
        """Set rectangular domain.

        Args:
            L (tuple[float, float, float]): size of domain. L = (Lx, Ly, Lz)
            f0 (ufl.tensors.ListTensor): fiber direction
            s0 (ufl.tensors.ListTensor): transversal direction
        """
        Lx, Ly, Lz = L
        mesh_comm = MPI.COMM_WORLD
        self.domain = mesh.create_box(
            mesh_comm,
            [[0, 0, 0], [Lx, Ly, Lz]],
            n=[int(Lx / self.h), int(Ly / self.h), int(Lz / self.h)],
        )
        self._init_functions()
        self._init_invariants(f0, s0)

    def set_existing_domain(
        self, domain: mesh.Mesh, f0: ufl.tensors.ListTensor, s0: ufl.tensors.ListTensor
    ):
        """Set existing domain

        Args:
            domain (mesh.Mesh): domain to use
            f0 (ufl.tensors.ListTensor): fiber direction
            s0 (ufl.tensors.ListTensor): transversal direction
        """
        self.domain = domain
        self._init_functions()
        self._init_invariants(f0, s0)

    def _dirichlet1(self, val: float, tag: int, facet_tag: mesh.MeshTags):
        u_bc = np.array((val,) * self.domain.geometry.dim, dtype=default_scalar_type)
        dofs = fem.locate_dofs_topological(self.V, facet_tag.dim, facet_tag.find(tag))
        self.bcs.append(fem.dirichletbc(u_bc, dofs, self.V))

    def _dirichlet2(self, val: float, tag: int, facet_tag: mesh.MeshTags):
        dofs = fem.locate_dofs_topological(self.V.sub(val), facet_tag.dim, facet_tag.find(tag))
        self.bcs.append(fem.dirichletbc(default_scalar_type(0), dofs, self.V.sub(val)))

    def _neumann(self, val: float, tag: int, facet_tag: mesh.MeshTags):
        ds = ufl.Measure(
            "ds",
            domain=self.domain,
            subdomain_data=facet_tag,
            metadata={"quadrature_degree": 4},
        )
        N = ufl.geometry.FacetNormal(self.domain)
        t = fem.Constant(self.domain, default_scalar_type(val))
        self.R_neumann += ufl.inner(t * N, self.v) * ds(tag)

    def boundary_conditions(
        self,
        boundaries,
        vals: list[float],
        bc_types: list[str],
        tags: list[int] | None = None,
    ):
        """Apply Dirichlet (type 1 or 2) and/or Neumann boundary conditions

        Args:
            boundaries (list of callables): functions returning true if coordinate is near boundary
            vals (list of floats):
                if Dirichlet type 1: value to hold u at
                if Dirichlet type 2: dimension to restrict u
                if Neumann: traction value
            bc_types (list of strings): choice of boundary conditions:
                d1 (dirichlet type 1), d2 (dirichlet type 2), or n (neumann)
            tags (list of ints): tags to assign boundary.
                Defaults to None, in which case it will be assigned automatically
        """
        if not tags:
            tags = range(1, len(boundaries) + 1)
        fdim = self.domain.topology.dim - 1
        marked_facets = np.array([])
        marked_values = np.array([])
        for boundary, tag in zip(boundaries, tags):
            facets = mesh.locate_entities_boundary(self.domain, fdim, boundary)
            marked_facets = np.hstack([marked_facets, facets])
            marked_values = np.hstack([marked_values, np.full_like(facets, tag)])
        sorted_facets = np.argsort(marked_facets)
        facet_tag = mesh.meshtags(
            self.domain,
            fdim,
            marked_facets[sorted_facets].astype(np.int32),
            marked_values[sorted_facets].astype(np.int32),
        )

        self.R_neumann = ufl.as_ufl(0)
        for val, tag, bc_type in zip(vals, tags, bc_types):
            if bc_type == "d1":  # Dirichlet type 1
                self._dirichlet1(val, tag, facet_tag)
            elif bc_type == "d2":  # Dirichlet type 2
                self._dirichlet2(val, tag, facet_tag)
            elif bc_type == "n":  # Neumann
                self._neumann(val, tag, facet_tag)
            else:
                raise TypeError("Unknown boundary condition type")

    def _incompressible(self):
        self.Q = fem.functionspace(self.domain, ("Lagrange", self.lagrange_order - 1))
        self.p = fem.Function(self.Q)
        self.dp = ufl.TrialFunction(self.Q)
        self.q = ufl.TestFunction(self.Q)

        self.states.append(self.p)
        self.teststates.append(self.q)
        self.trialstates.append(self.dp)
        self.L += self.p * (self.J - 1) * self.dx

    def _holzapfel_ogden_model(self):
        def subplus(x):
            return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)

        a, b, af, bf = 2.28, 9.726, 1.685, 15.779
        psi_p = a / (2 * b) * (ufl.exp(b * (self.I1 - 3)) - 1) + af / (2 * bf) * (
            ufl.exp(bf * subplus(self.I4f - 1) ** 2) - 1
        )
        psi_a = self.Ta * self.J / 2 * (self.I4f - 1)  # eta=0
        self.psi = psi_p + psi_a
        self.L += self.psi * self.dx

    def setup_solver(self):
        self._incompressible()
        self._holzapfel_ogden_model()
        # Residuals
        R = []
        for state, teststate in zip(self.states, self.teststates):
            R.append(ufl.derivative(self.L, state, teststate))
        R[0] += self.R_neumann
        # Jacobian
        K = []
        for r in R:
            Kr = []
            for state, trialstate in zip(self.states, self.trialstates):
                Kr.append(ufl.derivative(r, state, trialstate))
            K.append(Kr)

        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        self.solver = NewtonSolver(
            R,
            K,
            self.states,
            max_iterations=25,
            bcs=self.bcs,
            petsc_options=petsc_options,
        )

        def post_solve(solver: NewtonSolver):
            norm = solver.dx.norm(0)
            pprint(f"Solve completed in with correction norm {norm}", self.domain)

        self.solver.set_post_solve_callback(post_solve)

    def set_tension(self, tension: float | fem.Function):
        """Set active tension from cell model

        Args:
            tension (float | fem.Function): Tension to set to.
        """
        if isinstance(tension, float):
            # ? better way to set function equal to constant?
            self.Ta.interpolate(lambda x: tension + 0 * x[0])
        else:
            # todo: allow different meshes
            self.Ta.interpolate(tension)

    def solve(self):
        self.solver.solve()
        self.u.x.scatter_forward()
        self.p.x.scatter_forward()
