from dolfinx import fem, mesh, default_scalar_type
import numpy as np
from mpi4py import MPI
import ufl

h = 0.5

mesh_comm = MPI.COMM_WORLD
Lx, Ly, Lz = 3, 7, 20 #mm
domain = mesh.create_box(mesh_comm, [[0,0,0], [Lx,Ly,Lz]], n = [int(Lx/h), int(Ly/h), int(Lz/h)])
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))
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

v = ufl.Testfunction(V)
u = fem.Function(V)

d = len(u)
I = ufl.variable(ufl.Identity(d))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)
J = ufl.variable(ufl.det(F))

# Invariants
f0 = np.array([0, 0, 1])
s0 = np.array([0, 1, 0])
n0 = np.array([1, 0, 0])

I1 = ufl.variable(ufl.tr(C))
I4f = ufl.variable(f0 * (C * f0))
#I4s = ufl.variable(s0 * (C * s0))
#I8fs = ufl.variable(s0 * (C * f0))
a, b, af, bf = 1, 1, 1, 1

T_a = ufl.sin(2 * ufl.pi * t)

psi_p = a/(2*b) * (ufl.exp(b * (I1-3)) - 1) + af/(2*bf) * (ufl.exp(bf * (I4f-1)**2) - 1)
psi_a = T_a * J / 2 * (I4f - 1)
psi = psi_p + psi_a
P = ufl.diff(psi, F)

metadata = {'quadrature_degree': 4}
dx = ufl.Measure('dx', domain=domain, metadata=metadata)

F = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx