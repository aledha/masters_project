from dolfinx import fem, mesh, default_scalar_type, io, log
import numpy as np
from mpi4py import MPI
import ufl
from scifem import NewtonSolver
from dataclasses import dataclass

import ufl.geometry

@dataclass
class HyperelasticProblem:
    h: float 
    lagrange_order: float

    def __post_init__(self):
        self.bcs = []
        self.states = []
        self.teststates = []
        self.trialstates = []
        self.L = ufl.as_ufl(0.0)

    def _init_functions(self):
        self.V = fem.functionspace(self.domain, ("Lagrange", self.lagrange_order, (self.domain.geometry.dim, )))    
        self.x = ufl.SpatialCoordinate(self.domain)
        metadata = {'quadrature_degree': 4}
        self.dx = ufl.Measure('dx', domain=self.domain, metadata=metadata)

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

        self.B = fem.Constant(self.domain, default_scalar_type((0, 0, 0)))
        self.T = fem.Constant(self.domain, default_scalar_type((0, 0, 0)))

    def _init_invariants(self, f_dir, s_dir):
        f0 = ufl.unit_vector(f_dir, 3)
        s0 = ufl.unit_vector(s_dir, 3)

        self.I1 = ufl.variable(ufl.tr(self.C))
        self.I4f = ufl.variable(ufl.dot(f0, ufl.dot(self.C, f0)))
        self.I4s = ufl.variable(ufl.dot(s0, ufl.dot(self.C, s0)))
        self.I8fs = ufl.variable(ufl.dot(s0, ufl.dot(self.C, f0)))

    def set_rectangular_domain(self, Lx, Ly, Lz, f_dir, s_dir):
        mesh_comm = MPI.COMM_WORLD
        self.domain = mesh.create_box(mesh_comm, [[0,0,0], [Lx,Ly,Lz]], n = [int(Lx/self.h), int(Ly/self.h), int(Lz/self.h)])
        self._init_functions()
        self._init_invariants(f_dir, s_dir)

    def boundary_conditions(self, boundaries, vals = [0.0, -1.0], tags = [1, 2], bc_types=['d', 'n']):
        fdim = self.domain.topology.dim - 1
        marked_facets = np.array([])
        marked_values = np.array([])
        for boundary, tag in zip(boundaries, tags):
            facets = mesh.locate_entities_boundary(self.domain, fdim, boundary)
            marked_facets = np.hstack([marked_facets, facets])
            marked_values = np.hstack([marked_values, np.full_like(facets, tag)])
        sorted_facets = np.argsort(marked_facets)
        facet_tag = mesh.meshtags(self.domain, 
                                  fdim, 
                                  marked_facets[sorted_facets].astype(np.int32), 
                                  marked_values[sorted_facets].astype(np.int32))

        ds = ufl.Measure('ds', domain=self.domain, subdomain_data=facet_tag, metadata={'quadrature_degree': 4})
        N = ufl.geometry.FacetNormal(self.domain)
        self.R_neumann = ufl.as_ufl(0)
        for val, tag, bc_type in zip(vals, tags, bc_types):
            if bc_type == 'd':      # Dirichlet
                u_bc = np.array((val,) * self.domain.geometry.dim, dtype=default_scalar_type)
                dofs = fem.locate_dofs_topological(self.V, facet_tag.dim, facet_tag.find(tag))
                self.bcs.append(fem.dirichletbc(u_bc, dofs, self.V))
            elif bc_type == 'n':    # Neumann
                t = fem.Constant(self.domain, default_scalar_type(val))
                n = t * self.J * ufl.inv(self.F).T * N   
                self.R_neumann += ufl.inner(n, self.v) * ds(tag)
        
    def holzapfel_ogden_model(self):
        def subplus(x):
            return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)

        a, b, af, bf = 2.28, 9.726, 1.685, 15.779
        psi_p = a/(2*b) * (ufl.exp(b * (self.I1-3)) - 1) + af/(2*bf) * (ufl.exp(bf * subplus(self.I4f-1)**2) - 1)
        self.T_a = fem.Constant(self.domain, 0.0)
        psi_a = self.T_a * self.J / 2 * (self.I4f - 1)      #eta=0
        self.psi = psi_p + psi_a 
        self.L += self.psi * self.dx

    def set_tension(self, val):
        self.T_a.value = val

    def incompressible(self):
        self.Q = fem.functionspace(self.domain, ("Lagrange", self.lagrange_order - 1)) 
        self.p = fem.Function(self.Q)
        self.dp = ufl.TrialFunction(self.Q)
        self.q = ufl.TestFunction(self.Q)

        self.states.append(self.p)
        self.teststates.append(self.q)
        self.trialstates.append(self.dp)
        self.L += self.p * (self.J-1) * self.dx

    def setup_solver(self):
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

        petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
        self.solver = NewtonSolver(R, K, self.states, max_iterations=25, bcs=self.bcs, petsc_options=petsc_options)
        def post_solve(solver: NewtonSolver):
            print(f"Solve completed in with correction norm {solver.dx.norm(0)}")
        self.solver.set_post_solve_callback(post_solve)

    def solve(self):
        self.solver.solve()

problem = HyperelasticProblem(0.2, 2)
problem.set_rectangular_domain(1, 1, 1, 0, 1)

def left(x):
    return np.isclose(x[0], 0)
def right(x):
    return np.isclose(x[0], 1)

problem.boundary_conditions([left, right], vals=[0.0, -1.0], tags=[1, 2], bc_types=['d', 'n'])
problem.holzapfel_ogden_model()
problem.incompressible()
problem.setup_solver()

vtx = io.VTXWriter(MPI.COMM_WORLD, "unit_cube.bp", [problem.u], engine="BP4")
problem.set_tension(2.0)
problem.solve()
vtx.write(0.0)
vtx.close()
# T_as = np.linspace(0, 100, 11)
# for T_a in T_as:
#     problem.set_tension(T_a)
#     problem.solve()
#     for i in range(10):
#         vtx.write(T_a + i) 
#     print(f"Solved for T_a={T_a}")
# vtx.close