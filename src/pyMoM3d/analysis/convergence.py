"""Mesh convergence study helper."""

import numpy as np
from typing import List, Tuple

from ..mesh.mesh_data import Mesh
from ..mesh.trimesh_mesher import PythonMesher
from ..mesh.rwg_connectivity import compute_rwg_connectivity
from ..mom.impedance import fill_impedance_matrix
from ..mom.excitation import PlaneWaveExcitation
from ..mom.solver import solve_direct
from ..fields.far_field import compute_far_field
from ..fields.rcs import compute_monostatic_rcs
from ..utils.constants import eta0, c0


def mesh_convergence_study(
    geometry,
    frequency: float,
    subdivisions_list: List[int],
    theta_back: float = np.pi,
    phi_back: float = 0.0,
    quad_order: int = 4,
) -> List[Tuple[int, float]]:
    """Run a mesh-refinement convergence study.

    For each subdivision level, mesh the geometry, solve, and compute
    monostatic RCS.

    Parameters
    ----------
    geometry : object
        Geometry primitive with `to_trimesh(subdivisions=...)` method.
    frequency : float
        Operating frequency (Hz).
    subdivisions_list : list of int
        Subdivision levels to test.
    theta_back, phi_back : float
        Backscatter direction (default: -z).
    quad_order : int

    Returns
    -------
    results : list of (N_unknowns, rcs_dBsm)
    """
    k = 2 * np.pi * frequency / c0
    results = []

    exc = PlaneWaveExcitation(
        E0=np.array([1.0, 0.0, 0.0]),
        k_hat=np.array([0.0, 0.0, -1.0]),
    )

    mesher = PythonMesher()

    for subdiv in subdivisions_list:
        trimesh_obj = geometry.to_trimesh(subdivisions=subdiv)
        mesh = mesher.mesh_from_geometry(trimesh_obj)
        basis = compute_rwg_connectivity(mesh)

        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=quad_order)
        V = exc.compute_voltage_vector(basis, mesh, k)
        I = solve_direct(Z, V)

        theta_arr = np.array([theta_back])
        phi_arr = np.array([phi_back])
        E_theta, E_phi = compute_far_field(I, basis, mesh, k, eta0, theta_arr, phi_arr)
        rcs = compute_monostatic_rcs(E_theta[0], E_phi[0])

        results.append((basis.num_basis, rcs))

    return results
