from mpi4py import MPI
import dolfinx.fem.petsc
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)
import numpy.typing as npt
import numpy as np
import basix.ufl
import ufl
import matplotlib.pyplot as plt


def f_impl(Pu: npt.NDArray, g: npt.NDArray, x: npt.NDArray):
    # Input data for Pu and x is packed as (num_cells, num_quadrature_points, gdim)
    # g is packed as (num_cells, num_quadrature_points)
    grad_u_data = Pu.reshape(Pu.shape[0], -1, 2)
    x_data = x.reshape(Pu.shape[0], -1, 2)

    # Exercise:
    # Print x_data and grad_u_data here and verify that grad(u_n) matches x_data at each quadrature point
    # Remember that this example only uses two cells, which makes it easy to draw by hand!
    print("x: ", x_data)
    print("gradu: ", grad_u_data)
    plt.scatter(x_data[0, :, 0], x_data[0, :, 1], c='r')
    plt.scatter(x_data[1, :, 0], x_data[1, :, 1], c='b')
    #plt.show()

    output = np.zeros((Pu.shape[0], grad_u_data.shape[1]))
    # u_n.dx(0) + u_n.dx(1) + g * x[0] * x[1]
    output[:, :] = grad_u_data[:, :, 0] + grad_u_data[:, :, 1] + g * x_data[:, :, 0] * x_data[:, :, 1]
    # Output shape (num_cells, num_points)
    return output.flatten()


def f_external(derivatives: tuple[int, ...]):
    if derivatives == (0, 0, 0):  # no derivation, the function itself
        return f_impl
    elif derivatives == (1, 0, 0):  # the derivative with respect to the operand `uh`
        return NotImplementedError
    elif derivatives == (0, 1, 0):
        return NotImplementedError
    elif derivatives == (0, 0, 1):
        return NotImplementedError
    else:
        return NotImplementedError


mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

u_n = dolfinx.fem.Function(V)
u_n.interpolate(lambda x: 1 + x[0] + 2 * x[1] * x[1])


def P(u):
    return ufl.grad(u)


x = ufl.SpatialCoordinate(mesh)
S_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=4)
g = dolfinx.fem.Constant(mesh, 3.2)
S = dolfinx.fem.functionspace(mesh, S_element)


# I = I(grad(u), g, x)
I = FEMExternalOperator(P(u_n), g, x, function_space=S, external_function=f_external)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(I, v) * ufl.dx

# Create quadrature coefficients/functiosn to store the various dependencies of I
L_updated, operators = replace_external_operators(L)
L_compiled = dolfinx.fem.form(L_updated)

# Temporal Loop
# Update u_n first
# Then
coefficients = evaluate_operands(operators)
ext = evaluate_external_operators(operators, coefficients)
print(ext)
b = dolfinx.fem.petsc.assemble_vector(L_compiled)
print(b.array)


reference_b = ufl.inner(u_n.dx(0) + u_n.dx(1) + g * x[0] * x[1], v) * ufl.dx
reference_b_compiled = dolfinx.fem.form(reference_b)
reference_b = dolfinx.fem.petsc.assemble_vector(reference_b_compiled)
print(reference_b.array)