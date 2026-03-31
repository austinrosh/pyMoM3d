"""Shared impedance matrix assembly loop.

Provides ``fill_matrix(operator, ...)`` — the single entry point for
assembling any formulation (EFIE, MFIE, CFIE) into an N×N complex
impedance matrix.

The operator strategy pattern separates per-pair kernel logic from the
shared outer loop, geometry precomputation, and backend dispatch.
"""

import numpy as np

from ..mesh.mesh_data import Mesh
from ..mesh.rwg_basis import RWGBasis
from ..greens.quadrature import triangle_quad_rule
from .operators.base import AbstractOperator


def _resolve_backend(requested: str, operator: AbstractOperator) -> str:
    """Map 'auto' to a concrete backend; validate explicit requests.

    Priority for 'auto': cpp → numba → numpy.
    Falls back to numpy if the operator does not support the fast backend.

    Raises ``RuntimeError`` only when the user *explicitly* requests a
    backend that the operator cannot satisfy.
    """
    if requested == 'auto':
        for candidate in ('cpp', 'numba', 'numpy'):
            if operator.supports_backend(candidate):
                return candidate
        return 'numpy'

    if not operator.supports_backend(requested):
        raise RuntimeError(
            f"backend='{requested}' is not supported by "
            f"{type(operator).__name__}.  Available backends: "
            + ', '.join(b for b in ('cpp', 'numba', 'numpy')
                        if operator.supports_backend(b))
        )
    return requested


def fill_matrix(
    operator: AbstractOperator,
    rwg_basis: RWGBasis,
    mesh: Mesh,
    k: float,
    eta: float,
    quad_order: int = 4,
    near_threshold: float = 0.2,
    backend: str = 'auto',
    progress_callback=None,
) -> np.ndarray:
    """Assemble an N×N impedance matrix using *operator*.

    Parameters
    ----------
    operator : AbstractOperator
        Formulation strategy (EFIEOperator, MFIEOperator, CFIEOperator, …).
    rwg_basis : RWGBasis
        RWG basis function data.
    mesh : Mesh
        Surface mesh.
    k : float
        Wavenumber (rad/m).
    eta : float
        Intrinsic impedance of the medium (Ohms).
    quad_order : int
        Quadrature order for triangle integration.
    near_threshold : float
        Near-field threshold for singularity extraction.
    backend : str
        Compute backend: ``'auto'`` (default), ``'numpy'``, ``'numba'``,
        or ``'cpp'``.  ``'auto'`` selects the fastest backend the operator
        supports.
    progress_callback : callable, optional
        Called once per outer row with ``progress_callback(fraction)``
        where *fraction* ∈ [0, 1).  Ignored for fast backends.

    Returns
    -------
    Z : ndarray, shape (N, N), complex128
        Impedance matrix.
    """
    resolved = _resolve_backend(backend, operator)

    N = rwg_basis.num_basis
    Z = np.zeros((N, N), dtype=np.complex128)

    # ------------------------------------------------------------------
    # Precompute per-triangle geometry (vectorized, done once)
    # ------------------------------------------------------------------
    tri_verts = mesh.vertices[mesh.triangles]          # (N_t, 3, 3)
    tri_centroids = tri_verts.mean(axis=1)             # (N_t, 3)

    e0 = tri_verts[:, 1] - tri_verts[:, 0]
    e1 = tri_verts[:, 2] - tri_verts[:, 1]
    e2 = tri_verts[:, 0] - tri_verts[:, 2]
    tri_mean_edge = (
        np.linalg.norm(e0, axis=1) +
        np.linalg.norm(e1, axis=1) +
        np.linalg.norm(e2, axis=1)
    ) / 3.0                                            # (N_t,)
    tri_twice_area = 2.0 * mesh.triangle_areas         # (N_t,)

    # Outward triangle normals — cross(v1-v0, v2-v0)
    # Orientation is mesh-winding-consistent.
    raw_normals = np.cross(e0, -e2)                    # (N_t, 3)
    norm_lens = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    norm_lens = np.where(norm_lens < 1e-30, 1.0, norm_lens)
    tri_normals = raw_normals / norm_lens              # (N_t, 3)

    # Quadrature rule — computed once, shared across all pairs
    weights, bary = triangle_quad_rule(quad_order)

    # ------------------------------------------------------------------
    # Fast backend (cpp / numba) — operator handles dispatch
    # ------------------------------------------------------------------
    if resolved != 'numpy':
        operator.fill_fast(
            resolved, Z, rwg_basis, mesh, k, eta,
            tri_centroids, tri_mean_edge, tri_twice_area, tri_normals,
            weights, bary, quad_order, near_threshold,
        )
        operator.post_assembly(Z, rwg_basis, mesh, k, eta)
        return Z

    # ------------------------------------------------------------------
    # NumPy assembly loop — calls operator.compute_pair_numpy per pair
    # ------------------------------------------------------------------
    near_thresh_scaled = near_threshold * 3.0
    symmetric = operator.is_symmetric

    if symmetric:
        total_pairs = N * (N + 1) // 2
    else:
        total_pairs = N * N

    for m in range(N):
        if progress_callback is not None:
            if symmetric:
                done = m * (2 * N - m - 1) // 2
            else:
                done = m * N
            progress_callback(done / total_pairs if total_pairs > 0 else 0.0)

        tris_m = (
            (rwg_basis.t_plus[m],  rwg_basis.free_vertex_plus[m],
             +1.0, rwg_basis.area_plus[m]),
            (rwg_basis.t_minus[m], rwg_basis.free_vertex_minus[m],
             -1.0, rwg_basis.area_minus[m]),
        )

        n_start = m if symmetric else 0
        for n in range(n_start, N):
            Z_mn = 0.0 + 0.0j

            tris_n = (
                (rwg_basis.t_plus[n],  rwg_basis.free_vertex_plus[n],
                 +1.0, rwg_basis.area_plus[n]),
                (rwg_basis.t_minus[n], rwg_basis.free_vertex_minus[n],
                 -1.0, rwg_basis.area_minus[n]),
            )

            for (tri_m, fv_m, sign_m, A_m) in tris_m:
                centroid_m = tri_centroids[tri_m]
                n_hat_m    = tri_normals[tri_m]

                for (tri_n, fv_n, sign_n, A_n) in tris_n:
                    mean_edge_n = tri_mean_edge[tri_n]
                    dist = np.linalg.norm(centroid_m - tri_centroids[tri_n])
                    is_near = (
                        dist < near_thresh_scaled * mean_edge_n
                        if mean_edge_n > 1e-30 else True
                    )

                    Z_mn += operator.compute_pair_numpy(
                        k, eta, mesh,
                        tri_m, tri_n, fv_m, fv_n,
                        sign_m, sign_n,
                        rwg_basis.edge_length[m], rwg_basis.edge_length[n],
                        A_m, A_n,
                        quad_order, near_threshold,
                        weights, bary,
                        tri_twice_area[tri_m], tri_twice_area[tri_n],
                        is_near,
                        n_hat_m,
                    )

            Z[m, n] = Z_mn
            if symmetric and m != n:
                Z[n, m] = Z_mn

    operator.post_assembly(Z, rwg_basis, mesh, k, eta)
    return Z
