from dolfinx import fem, mesh, default_scalar_type, io, log
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

h = 0.75

mesh_comm = MPI.COMM_WORLD
Lx, Ly, Lz = 3, 7, 20 #mm
domain = mesh.create_box(mesh_comm, [[0,0,0], [Lx,Ly,Lz]], n = [int(Lx/h), int(Ly/h), int(Lz/h)])

lagrange_order = 2
V = fem.functionspace(domain, ("Lagrange", lagrange_order, (domain.geometry.dim, )))
Q = fem.functionspace(domain, ("Lagrange", lagrange_order - 1))

x = ufl.SpatialCoordinate(domain)
t = fem.Constant(domain, 0.0)

def left(x):
    return np.isclose(x[2], 0)

fdim = domain.topology.dim - 1
facets = mesh.locate_entities_boundary(domain, fdim, left)
facet_tag = mesh.meshtags(domain, fdim, facets, np.full_like(facets, 1))

u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

u = fem.Function(V)
p = fem.Function(Q)

v = ufl.TestFunction(V)
q = ufl.TestFunction(Q)

d = len(u)
I = ufl.variable(ufl.Identity(d))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)
J = ufl.variable(ufl.det(F))

B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

# Invariants
f0 = ufl.unit_vector(2, 3)
s0 = ufl.unit_vector(1, 3)
n0 = ufl.unit_vector(0, 3)

I1 = ufl.variable(ufl.tr(C))
I4f = ufl.variable(ufl.dot(f0, ufl.dot(C, f0)))
I4s = ufl.variable(ufl.dot(s0, ufl.dot(C, s0)))
I8fs = ufl.variable(ufl.dot(s0, ufl.dot(C, f0)))
a, b, af, bf = 2.28, 9.726, 1.685, 15.779

T_a = ufl.variable(ufl.sin(ufl.pi * t))

def subplus(x):
    return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)

psi_p = a/(2*b) * (ufl.exp(b * (I1-3)) - 1) + af/(2*bf) * (ufl.exp(bf * subplus(I4f-1)**2) - 1)
psi_a = T_a * J / 2 * (I4f - 1)
psi = psi_p + psi_a 
#P = ufl.diff(psi, F) + J * p * ufl.compound_expressions.inverse_expr(F.T)

metadata = {'quadrature_degree': 4}
dx = ufl.Measure('dx', domain=domain, metadata=metadata)
pi_i = (p * (J - 1) + psi) * dx
pi_e = - ufl.inner(B, u) * dx
pi = pi_i + pi_e

Ffunc = ufl.derivative(pi, u, v) + ufl.derivative(pi, p, q)
#Ffunc = ufl.inner(ufl.grad(v) + p*J*ufl.compound_expressions.inverse_expr(F.T), P) * dx + q * (J - 1) * dx - ufl.inner(B, v) * dx

states = [u, p]
problem = NonlinearProblem(Ffunc, states, bcs)

solver = NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

vtx = io.VTXWriter(MPI.COMM_WORLD, "active_stress.bp", [u], engine="BP4")

log.set_log_level(log.LogLevel.INFO)

solver.solve(u)
vtx.write(t.value)
T = 1
dt = 0.1
while t.value + dt < T:
    t.value += dt
    num_its, converged = solver.solve(u)
    assert(converged)
    vtx.write(t.value)

vtx.close
