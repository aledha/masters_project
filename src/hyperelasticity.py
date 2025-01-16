from dolfinx import fem, mesh, default_scalar_type, io, log
import numpy as np
from mpi4py import MPI
import ufl
from scifem import NewtonSolver

h = 0.5

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
du = ufl.TrialFunction(V)
dp = ufl.TrialFunction(Q)
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

T_a = ufl.variable(30*ufl.sin(ufl.pi * t))

def subplus(x):
    return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)

psi_p = a/(2*b) * (ufl.exp(b * (I1-3)) - 1) + af/(2*bf) * (ufl.exp(bf * subplus(I4f-1)**2) - 1)
psi_a = T_a * J / 2 * (I4f - 1)
psi = psi_p + psi_a 

metadata = {'quadrature_degree': 4}
dx = ufl.Measure('dx', domain=domain, metadata=metadata)

L = psi * dx + p * (J - 1) * dx
r_u = ufl.derivative(L, u, v) 
r_p = ufl.derivative(L, p, q)
R = [r_u, r_p]
K = [
    [ufl.derivative(r_u, u, du), ufl.derivative(r_u, p, dp)],
    [ufl.derivative(r_p, u, du), ufl.derivative(r_p, p, dp)],
]

petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
solver = NewtonSolver(R, K, [u, p], max_iterations=25, bcs=bcs, petsc_options=petsc_options)


vtx = io.VTXWriter(MPI.COMM_WORLD, "active_stress.bp", [u], engine="BP4")

def pre_solve(solver: NewtonSolver):
    print(f"Starting solve with {solver.max_iterations} iterations")

def post_solve(solver: NewtonSolver):
    print(f"Solve completed in with correction norm {solver.dx.norm(0)}")
solver.set_pre_solve_callback(pre_solve)
solver.set_post_solve_callback(post_solve)

solver.solve()
vtx.write(t.value)
T = 1
dt = 0.1
while t.value + dt < T:
    t.value += dt
    solver.solve()
    vtx.write(t.value)

vtx.close
