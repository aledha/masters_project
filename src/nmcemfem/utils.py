from pathlib import Path

import dolfinx
import gotranx
import numpy as np
import packaging

dolfinx_version = packaging.version.parse(dolfinx.__version__)


def translateODE(odeFileName, schemes):
    odeFolder = str(Path.cwd().parent) + "/odes/"
    model_path = Path(odeFolder + odeFileName + ".py")
    if not model_path.is_file():
        ode = gotranx.load_ode(odeFolder + odeFileName + ".ode")
        code = gotranx.cli.gotran2py.get_code(ode, schemes)
        model_path.write_text(code)
    else:
        print("ODE already translated")


def pprint(string: str, domain: dolfinx.mesh.Mesh):
    if domain.comm.rank == 0:
        print(string)


def interpolate_to_mesh(u_from: dolfinx.fem.Function, V_to: dolfinx.fem.FunctionSpace):
    u_from_interp = dolfinx.fem.Function(V_to)
    if dolfinx_version < packaging.version.parse("0.9.0"):
        u_from_interp.interpolate(
            u_from,
            nmm_interpolation_data=dolfinx.fem.create_nonmatching_meshes_interpolation_data(
                V_to.mesh,
                V_to.element,
                u_from.function_space.mesh,
            ),
        )
    else:
        # see https://github.com/FEniCS/dolfinx/issues/3562#issuecomment-2543112976
        V_from = u_from.function_space
        domain_from = V_from.mesh
        domain_to = V_to.mesh

        bbox_from = dolfinx.geometry.bb_tree(
            domain_from, domain_from.topology.dim, padding=1e-6
        )
        bbox_to = dolfinx.geometry.bb_tree(
            domain_to, domain_to.topology.dim, padding=1e-6
        )
        global_b1_tree = bbox_from.create_global_tree(domain_from.comm)
        collisions = dolfinx.geometry.compute_collisions_trees(bbox_to, global_b1_tree)
        cells = np.unique(collisions[:, 0])

        interpolation_data1 = dolfinx.fem.create_interpolation_data(V_to, V_from, cells)
        u_from_interp.interpolate_nonmatching(u_from, cells, interpolation_data1)
    return u_from_interp
