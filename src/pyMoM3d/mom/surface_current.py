"""Surface current reconstruction from MoM solution coefficients."""

import numpy as np

from ..mesh.mesh_data import Mesh
from ..mesh.rwg_basis import RWGBasis


def evaluate_surface_current(
    I_coeffs: np.ndarray,
    rwg_basis: RWGBasis,
    mesh: Mesh,
    points: np.ndarray,
) -> np.ndarray:
    """Reconstruct J(r) = sum(I_n * f_n(r)) at given surface points.

    Parameters
    ----------
    I_coeffs : ndarray, shape (N,), complex128
        Current expansion coefficients.
    rwg_basis : RWGBasis
        RWG basis data.
    mesh : Mesh
        Surface mesh.
    points : ndarray, shape (M, 3)
        Evaluation points (should be on the mesh surface).

    Returns
    -------
    J : ndarray, shape (M, 3), complex128
        Surface current density at each point.
    """
    points = np.asarray(points, dtype=np.float64)
    M = len(points)
    J = np.zeros((M, 3), dtype=np.complex128)

    for p_idx in range(M):
        r = points[p_idx]

        # Find which triangle contains this point (brute force)
        tri_idx = _find_containing_triangle(mesh, r)
        if tri_idx < 0:
            continue

        for n in range(rwg_basis.num_basis):
            # Check if this basis function involves tri_idx
            if rwg_basis.t_plus[n] == tri_idx:
                fv = mesh.vertices[rwg_basis.free_vertex_plus[n]]
                sign = +1.0
                area = rwg_basis.area_plus[n]
            elif rwg_basis.t_minus[n] == tri_idx:
                fv = mesh.vertices[rwg_basis.free_vertex_minus[n]]
                sign = -1.0
                area = rwg_basis.area_minus[n]
            else:
                continue

            l_n = rwg_basis.edge_length[n]
            rho = r - fv if sign > 0 else fv - r
            f_n = sign * l_n / (2.0 * area) * (r - fv)
            J[p_idx] += I_coeffs[n] * f_n

    return J


def _find_containing_triangle(mesh: Mesh, point: np.ndarray) -> int:
    """Find triangle containing point (nearest centroid heuristic)."""
    min_dist = np.inf
    best = -1
    for i in range(len(mesh.triangles)):
        verts = mesh.vertices[mesh.triangles[i]]
        centroid = np.mean(verts, axis=0)
        dist = np.linalg.norm(point - centroid)
        if dist < min_dist:
            min_dist = dist
            best = i
    return best
