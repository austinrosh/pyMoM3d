"""MFIE operator: Magnetic Field Integral Equation."""

import numpy as np

from ...greens.quadrature import triangle_quad_rule
from .base import AbstractOperator

# ---------------------------------------------------------------------------
# Optional fast backends
# ---------------------------------------------------------------------------
try:
    from .._cpp_kernels import (
        fill_impedance_mfie_cpp as _fill_impedance_mfie_cpp,
    )
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False

try:
    from ..numba_kernels import (
        NUMBA_AVAILABLE as _NUMBA_AVAILABLE,
        fill_Z_mfie_numba as _fill_Z_mfie_numba,
    )
except ImportError:
    _NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Gram matrix helper
# ---------------------------------------------------------------------------

def _tri_rho_dot_integral(v0, v1, v2, a, b):
    """Analytically compute ∫_T (r−a)·(r−b) dS over triangle (v0,v1,v2).

    Uses the standard result:
        ∫_T (r−a)·(r−b) dS = A * [ (|v0|²+|v1|²+|v2|²+v0·v1+v1·v2+v2·v0)/6
                                    − (a+b)·(v0+v1+v2)/3 + a·b ]
    where A = triangle area.
    """
    cross = np.cross(v1 - v0, v2 - v0)
    A = 0.5 * np.linalg.norm(cross)
    if A < 1e-30:
        return 0.0

    r2_sum = (np.dot(v0, v0) + np.dot(v1, v1) + np.dot(v2, v2)
              + np.dot(v0, v1) + np.dot(v1, v2) + np.dot(v2, v0))
    centroid = (v0 + v1 + v2) / 3.0
    return A * (r2_sum / 6.0 - np.dot(a + b, centroid) + np.dot(a, b))


def compute_gram_matrix(rwg_basis, mesh) -> np.ndarray:
    """Compute the RWG Gram (mass) matrix B_mn = ∫_S f_m(r) · f_n(r) dS.

    Only basis pairs that share at least one triangle contribute.  The
    result is returned as a dense N×N real matrix (imaginary part is zero).

    Parameters
    ----------
    rwg_basis : RWGBasis
    mesh : Mesh

    Returns
    -------
    B : ndarray, shape (N, N), float64
    """
    N = rwg_basis.num_basis
    verts = mesh.vertices
    tris  = mesh.triangles

    # Build map: triangle index → list of (basis_index, sign, area)
    tri_to_basis = {}
    for m in range(N):
        for tri, sign, area in (
            (int(rwg_basis.t_plus[m]),  +1.0, float(rwg_basis.area_plus[m])),
            (int(rwg_basis.t_minus[m]), -1.0, float(rwg_basis.area_minus[m])),
        ):
            tri_to_basis.setdefault(tri, []).append((m, sign, area))

    B = np.zeros((N, N), dtype=np.float64)

    for tri, entries in tri_to_basis.items():
        v0, v1, v2 = verts[tris[tri, 0]], verts[tris[tri, 1]], verts[tris[tri, 2]]

        for i, (m, sign_m, A_m) in enumerate(entries):
            r_fv_m = verts[
                int(rwg_basis.free_vertex_plus[m])
                if sign_m > 0 else
                int(rwg_basis.free_vertex_minus[m])
            ]
            l_m = float(rwg_basis.edge_length[m])
            amp_m = sign_m * l_m / (2.0 * A_m)

            for j, (n, sign_n, A_n) in enumerate(entries):
                r_fv_n = verts[
                    int(rwg_basis.free_vertex_plus[n])
                    if sign_n > 0 else
                    int(rwg_basis.free_vertex_minus[n])
                ]
                l_n = float(rwg_basis.edge_length[n])
                amp_n = sign_n * l_n / (2.0 * A_n)

                integral = _tri_rho_dot_integral(v0, v1, v2, r_fv_m, r_fv_n)
                B[m, n] += amp_m * amp_n * integral

    return B


# ---------------------------------------------------------------------------
# MFIEOperator
# ---------------------------------------------------------------------------

class MFIEOperator(AbstractOperator):
    """MFIE impedance matrix operator.

    Assembles the matrix:

        Z_mn^MFIE = (1/2) B_mn + K_mn

    where B_mn is the RWG Gram matrix and K_mn is the K-operator:

        K_mn = ∫∫ (f_m(r) · f_n(r')) · n̂_m · ∇'G(r,r') dS' dS

    with the scalar kernel:

        C(r,r') = n̂_m · (r − r') · (1 + jkR) exp(−jkR) / (4πR³)

    The (1/2)B_mn identity term is added in ``post_assembly``.

    Notes
    -----
    - Valid only for **closed** PEC surfaces (boundary edges must be absent).
    - The matrix is **not symmetric** (K_mn uses n̂_m; K_nm uses n̂_n).
    - Fast cpp/numba backends available when built.
    """

    is_symmetric: bool = False

    def supports_backend(self, backend: str) -> bool:
        if backend == 'cpp':   return _CPP_AVAILABLE
        if backend == 'numba': return _NUMBA_AVAILABLE
        return backend == 'numpy'

    def fill_fast(
        self,
        backend: str,
        Z: np.ndarray,
        rwg_basis,
        mesh,
        k: float,
        eta: float,
        tri_centroids: np.ndarray,
        tri_mean_edge: np.ndarray,
        tri_twice_area: np.ndarray,
        tri_normals: np.ndarray,
        weights: np.ndarray,
        bary: np.ndarray,
        quad_order: int,
        near_threshold: float,
    ) -> None:
        verts    = mesh.vertices.astype(np.float64, copy=False)
        tris_arr = mesh.triangles.astype(np.int32, copy=False)
        normals  = np.ascontiguousarray(tri_normals, dtype=np.float64)
        t_plus   = rwg_basis.t_plus.astype(np.int32, copy=False)
        t_minus  = rwg_basis.t_minus.astype(np.int32, copy=False)
        fv_plus  = rwg_basis.free_vertex_plus.astype(np.int32, copy=False)
        fv_minus = rwg_basis.free_vertex_minus.astype(np.int32, copy=False)
        a_plus   = rwg_basis.area_plus.astype(np.float64, copy=False)
        a_minus  = rwg_basis.area_minus.astype(np.float64, copy=False)
        elen     = rwg_basis.edge_length.astype(np.float64, copy=False)
        cents    = np.ascontiguousarray(tri_centroids, dtype=np.float64)
        medge    = np.ascontiguousarray(tri_mean_edge, dtype=np.float64)
        tarea    = np.ascontiguousarray(tri_twice_area, dtype=np.float64)

        # Pre-compute higher-order quadrature rule for near-field pairs
        near_order = min(quad_order + 3, 13)
        weights_near, bary_near = triangle_quad_rule(near_order)
        weights_near = np.ascontiguousarray(weights_near, dtype=np.float64)
        bary_near    = np.ascontiguousarray(bary_near,    dtype=np.float64)

        if backend == 'cpp':
            _fill_impedance_mfie_cpp(
                Z, verts, tris_arr, normals,
                t_plus, t_minus, fv_plus, fv_minus,
                a_plus, a_minus, elen, cents, medge, tarea,
                weights, bary, weights_near, bary_near,
                float(k), float(near_threshold), int(quad_order),
                0,  # num_threads: 0 = OMP default
            )
        elif backend == 'numba':
            _fill_Z_mfie_numba(
                Z, verts, tris_arr, normals,
                t_plus, t_minus, fv_plus, fv_minus,
                a_plus, a_minus, elen, cents, medge, tarea,
                weights, bary, weights_near, bary_near,
                float(k), float(near_threshold),
            )
        else:
            raise ValueError(f"MFIEOperator.fill_fast: unknown backend '{backend}'")

    def compute_pair_numpy(
        self,
        k, eta, mesh,
        tri_test, tri_src,
        fv_test, fv_src,
        sign_test, sign_src,
        l_test, l_src,
        A_test, A_src,
        quad_order, near_threshold,
        weights, bary,
        twice_area_test, twice_area_src,
        is_near,
        n_hat_test,
    ) -> complex:
        """Compute MFIE K-term contribution from one triangle pair.

        Uses higher-order quadrature for near pairs to mitigate the O(1/R)
        near-field singularity of the MFIE kernel.
        """
        verts_test = mesh.vertices[mesh.triangles[tri_test]]
        verts_src  = mesh.vertices[mesh.triangles[tri_src]]
        r_fv_test  = mesh.vertices[fv_test]
        r_fv_src   = mesh.vertices[fv_src]

        # Use denser quadrature for near pairs
        if is_near:
            q = min(quad_order + 3, 13)
            w, b = triangle_quad_rule(q)
        else:
            w, b = weights, bary

        four_pi = 4.0 * np.pi
        I_K_raw = 0.0 + 0.0j

        for i in range(len(w)):
            r_obs  = (b[i, 0] * verts_test[0]
                      + b[i, 1] * verts_test[1]
                      + b[i, 2] * verts_test[2])
            rho_m  = r_obs - r_fv_test

            I_K_inner = 0.0 + 0.0j
            for j in range(len(w)):
                r_src  = (b[j, 0] * verts_src[0]
                          + b[j, 1] * verts_src[1]
                          + b[j, 2] * verts_src[2])
                rho_n  = r_src - r_fv_src
                R_vec  = r_obs - r_src
                R      = np.linalg.norm(R_vec)
                if R < 1e-30:
                    continue
                # C = n̂·(r−r') * (1+jkR)*exp(−jkR) / (4πR³)
                jkR = 1j * k * R
                C = (np.dot(n_hat_test, R_vec)
                     * (1.0 + jkR)
                     * np.exp(-jkR)
                     / (four_pi * R ** 3))
                I_K_inner += w[j] * np.dot(rho_m, rho_n) * C

            I_K_inner *= twice_area_src
            I_K_raw   += w[i] * I_K_inner

        I_K_raw *= twice_area_test

        scale = ((sign_test * l_test / (2.0 * A_test))
                 * (sign_src  * l_src  / (2.0 * A_src)))
        return scale * I_K_raw

    def post_assembly(self, Z, rwg_basis, mesh, k, eta) -> None:
        """Add the (1/2) Gram matrix identity term."""
        if rwg_basis.num_boundary_edges > 0:
            raise ValueError(
                "MFIEOperator requires a closed surface "
                f"(found {rwg_basis.num_boundary_edges} boundary edge(s))."
            )
        B = compute_gram_matrix(rwg_basis, mesh)
        Z += 0.5 * B
