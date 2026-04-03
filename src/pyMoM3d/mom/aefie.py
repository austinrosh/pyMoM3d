"""Augmented EFIE (A-EFIE) for frequency-stable impedance extraction.

The standard EFIE becomes ill-conditioned at low frequencies (kD << 1)
because the scalar-potential term (proportional to 1/omega) dominates
the vector-potential term (proportional to omega) by a factor of
(kD)^{-2}.  This makes inductance extraction unreliable — extracted L(f)
varies by orders of magnitude instead of being flat.

The A-EFIE introduces surface charge as an additional unknown, solving
an augmented system that remains well-conditioned at all frequencies:

    [Z_A          D^T @ G_s] [I  ]   [V]
    [-D        jk/eta * I_T] [rho'] = [0]

where Z_A is the vector-potential matrix with jkη prefactor (N x N), G_s is the triangle-
to-triangle scalar Green's function matrix (T x T), D is the sparse
divergence matrix (T x N), and rho' = (eta/jk) * rho is the normalized
charge.

Both diagonal blocks scale as omega, and the off-diagonal blocks are
frequency-independent, giving O(1) condition number at all frequencies.

References
----------
* Qian, Z.-G. and Chew, W. C. (2009). "Enhanced A-EFIE with
  perturbation method." IEEE Trans. Antennas Propag., 57(10).
* Zhao, J.-S. and Chew, W. C. (2000). "Integral equation solution of
  Maxwell's equations from zero frequency to microwave frequencies."
  IEEE Trans. Antennas Propag., 48(10).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Optional C++ backend
# ---------------------------------------------------------------------------
try:
    from ._cpp_kernels import fill_scalar_green_cpp as _fill_scalar_green_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


def build_divergence_matrix(rwg_basis, mesh) -> sparse.csc_matrix:
    """Build the T x N sparse divergence matrix D.

    D[t, n] = +l_n / A_t   if t == t_plus[n]
    D[t, n] = -l_n / A_t   if t == t_minus[n]
    D[t, n] = 0             otherwise

    The RWG basis function f_n = l_n/(2A)·ρ has surface divergence
    ∇_s · f_n = l_n/(2A) · (∇_s · ρ) = l_n/(2A) · 2 = l_n / A,
    since ∇_s · ρ = 2 on a planar surface.

    Each column has exactly 2 nonzeros, corresponding to the positive
    and negative divergence of the RWG basis function on its two
    supporting triangles.

    Parameters
    ----------
    rwg_basis : RWGBasis
    mesh : Mesh

    Returns
    -------
    D : sparse.csc_matrix, shape (T, N)
    """
    N = rwg_basis.num_basis
    T = len(mesh.triangles)
    areas = mesh.triangle_areas

    rows = np.empty(2 * N, dtype=np.int32)
    cols = np.empty(2 * N, dtype=np.int32)
    vals = np.empty(2 * N, dtype=np.float64)

    t_plus = rwg_basis.t_plus
    t_minus = rwg_basis.t_minus
    edge_length = rwg_basis.edge_length

    for n in range(N):
        tp = int(t_plus[n])
        tm = int(t_minus[n])
        l_n = edge_length[n]

        rows[2 * n] = tp
        cols[2 * n] = n
        vals[2 * n] = l_n / areas[tp]

        rows[2 * n + 1] = tm
        cols[2 * n + 1] = n
        vals[2 * n + 1] = -l_n / areas[tm]

    return sparse.csc_matrix((vals, (rows, cols)), shape=(T, N))


def fill_scalar_green_matrix(
    mesh,
    k: float,
    quad_order: int = 4,
    near_threshold: float = 0.2,
    backend: str = 'auto',
    greens_fn=None,
) -> np.ndarray:
    """Assemble the T x T scalar Green's function matrix G_s.

    G_s[t, t'] = integral_t integral_t' G(r, r') dS dS'

    For free-space: G(r, r') = exp(-jkR) / (4*pi*R).
    For layered media: G = G_fs + (G_ML - G_fs), where the singular
    free-space part uses Graglia extraction and the smooth layered
    correction is added via the provided ``greens_fn``.

    Parameters
    ----------
    mesh : Mesh
    k : float
        Wavenumber.
    quad_order : int
        Quadrature order for triangle integration.
    near_threshold : float
        Near-field threshold for singularity extraction.
    backend : str
        ``'auto'`` (default), ``'cpp'``, or ``'numpy'``.
    greens_fn : GreensFunctionBase, optional
        Layered-medium Green's function whose ``scalar_G`` returns the
        smooth correction G_ML - G_fs.  If None, free-space G_s is
        assembled.

    Returns
    -------
    G_s : ndarray, shape (T, T), complex128
    """
    from ..greens.quadrature import triangle_quad_rule

    T = len(mesh.triangles)
    weights, bary = triangle_quad_rule(quad_order)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    bary = np.ascontiguousarray(bary, dtype=np.float64)

    # Resolve backend
    # For layered media, free-space G_s uses real k (free-space wavenumber).
    # The layered correction is added separately.
    k_real = float(np.real(k))
    k_has_imag = abs(np.imag(k)) > 1e-30 * abs(k_real) if k_real != 0 else abs(np.imag(k)) > 0

    if backend == 'auto':
        # C++ kernel only supports real k; use numpy for complex k without layered GF
        if _CPP_AVAILABLE and (greens_fn is not None or not k_has_imag):
            backend = 'cpp'
        elif _CPP_AVAILABLE and not k_has_imag:
            backend = 'cpp'
        else:
            backend = 'numpy'
    if backend == 'cpp' and not _CPP_AVAILABLE:
        raise RuntimeError("C++ backend requested but _cpp_kernels not available")

    G_s = np.zeros((T, T), dtype=np.complex128)

    if backend == 'cpp':
        verts = mesh.vertices.astype(np.float64, copy=False)
        tris = mesh.triangles.astype(np.int32, copy=False)

        # Precompute triangle geometry
        tri_centroids = np.empty((T, 3), dtype=np.float64)
        tri_twice_area = np.empty(T, dtype=np.float64)
        tri_mean_edge = np.empty(T, dtype=np.float64)

        for t in range(T):
            v0, v1, v2 = verts[tris[t]]
            tri_centroids[t] = (v0 + v1 + v2) / 3.0
            e0 = np.linalg.norm(v1 - v0)
            e1 = np.linalg.norm(v2 - v1)
            e2 = np.linalg.norm(v0 - v2)
            tri_mean_edge[t] = (e0 + e1 + e2) / 3.0
            tri_twice_area[t] = np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        tri_centroids = np.ascontiguousarray(tri_centroids)
        tri_mean_edge = np.ascontiguousarray(tri_mean_edge)
        tri_twice_area = np.ascontiguousarray(tri_twice_area)

        # C++ kernel uses real k (free-space part of singularity decomposition)
        _fill_scalar_green_cpp(
            G_s, verts, tris,
            tri_centroids, tri_mean_edge, tri_twice_area,
            weights, bary,
            k_real, float(near_threshold), int(quad_order),
            0,  # num_threads: OMP default
        )
    else:
        _fill_scalar_green_numpy(G_s, mesh, k, weights, bary, near_threshold)

    # Add layered-medium smooth correction if a GF is provided
    if greens_fn is not None:
        _add_layered_correction(G_s, mesh, weights, bary, greens_fn)

    return G_s


def _add_layered_correction(G_s, mesh, weights, bary, greens_fn):
    """Add the smooth layered correction (G_ML - G_fs) to G_s.

    The free-space G_s has already been assembled (with singularity
    extraction).  This adds the smooth correction from the layered GF
    backend using standard quadrature on all triangle pairs.

    For self-interactions (t_obs == t_src), coincident quadrature points
    (R=0) are skipped and their contribution is redistributed to avoid
    numerical blow-up from the backend subtracting two singular values.
    The correction G_ML - G_fs is theoretically smooth, but most layered
    GF backends evaluate G_ML and G_fs separately, so their difference
    is numerically unstable at R=0.
    """
    T = len(mesh.triangles)
    verts = mesh.vertices
    tris = mesh.triangles
    twice_area = 2.0 * mesh.triangle_areas
    Q = len(weights)

    for t_obs in range(T):
        v_obs = verts[tris[t_obs]]
        for t_src in range(t_obs, T):
            v_src = verts[tris[t_src]]
            is_self = (t_obs == t_src)

            correction = 0.0 + 0.0j
            for p in range(Q):
                r_obs = (bary[p, 0] * v_obs[0]
                         + bary[p, 1] * v_obs[1]
                         + bary[p, 2] * v_obs[2])

                # Batch inner integral over source quad points
                r_src_all = (bary[:, 0:1] * v_src[0]
                             + bary[:, 1:2] * v_src[1]
                             + bary[:, 2:3] * v_src[2])  # (Q, 3)
                r_obs_tiled = np.tile(r_obs, (len(weights), 1))  # (Q, 3)

                # scalar_G returns the smooth correction G_ML - G_fs
                g_corr = greens_fn.scalar_G(r_obs_tiled, r_src_all)  # (Q,)

                if is_self:
                    # Skip coincident point (R=0 → numerical blow-up)
                    # and redistribute its weight to the remaining points.
                    w_mod = weights.copy()
                    w_mod[p] = 0.0
                    w_sum_remaining = w_mod.sum()
                    if w_sum_remaining > 0:
                        w_mod *= weights.sum() / w_sum_remaining
                    g_int = np.dot(w_mod, g_corr) * twice_area[t_src]
                else:
                    g_int = np.dot(weights, g_corr) * twice_area[t_src]

                correction += weights[p] * g_int

            correction *= twice_area[t_obs]

            G_s[t_obs, t_src] += correction
            if t_obs != t_src:
                G_s[t_src, t_obs] += correction


def _fill_scalar_green_numpy(G_s, mesh, k, weights, bary, near_threshold):
    """NumPy fallback for G_s assembly."""
    from ..greens.singularity import integrate_green_singular

    T = len(mesh.triangles)
    verts = mesh.vertices
    tris = mesh.triangles
    areas = mesh.triangle_areas
    twice_area = 2.0 * areas

    for t_obs in range(T):
        v_obs = verts[tris[t_obs]]
        for t_src in range(t_obs, T):
            v_src = verts[tris[t_src]]

            val = 0.0 + 0.0j
            for p in range(len(weights)):
                r_obs = (bary[p, 0] * v_obs[0]
                         + bary[p, 1] * v_obs[1]
                         + bary[p, 2] * v_obs[2])
                g_int = integrate_green_singular(
                    k, r_obs,
                    v_src[0], v_src[1], v_src[2],
                    quad_order=len(weights),
                    near_threshold=near_threshold,
                )
                val += weights[p] * g_int
            val *= twice_area[t_obs]

            G_s[t_obs, t_src] = val
            if t_obs != t_src:
                G_s[t_src, t_obs] = val


def solve_aefie(
    Z_A: np.ndarray,
    G_s: np.ndarray,
    D: sparse.spmatrix,
    V: np.ndarray,
    k: float,
    eta: float,
) -> np.ndarray:
    """Solve A-EFIE with right-preconditioned GMRES.

    The effective impedance is Z_eff = Z_A + φ D^T G_s D, but forming
    this sum explicitly loses the Z_A contribution when Φ >> A (kD << 1).

    We use right preconditioning with M = Z_A to transform the system::

        Z_eff Z_A^{-1} y = V       (solve for y)
        (I + Φ Z_A^{-1}) y = V     (condition number ~ (kD)^{-2})
        I = Z_A^{-1} y             (recover current)

    The right-preconditioned operator (I + Φ Z_A^{-1}) has condition
    number O((kD)^{-2}), which is modest even at very low frequencies.
    GMRES converges in a few iterations without residual amplification.

    Parameters
    ----------
    Z_A : ndarray, shape (N, N), complex128
        Vector-potential impedance matrix **with jkη prefactor included**
        (as returned by ``fill_matrix(VectorPotentialOperator(), ...)``).
    G_s : ndarray, shape (T, T), complex128
        Triangle-to-triangle scalar Green's function matrix.
    D : sparse matrix, shape (T, N)
        Divergence matrix.
    V : ndarray, shape (N,) or (N, P), complex128
        RHS excitation vector(s).
    k : float
        Wavenumber.
    eta : float
        Wave impedance.

    Returns
    -------
    I : ndarray, shape (N,) or (N, P), complex128
        Current coefficients.
    """
    from scipy.sparse.linalg import gmres, LinearOperator
    import scipy.linalg

    D_dense = D.toarray()  # T x N, at most 2N nonzeros — cheap to densify
    N = Z_A.shape[0]

    phi_pf = -1j * eta / k  # scalar-potential prefactor

    # Precompute: D^T @ G_s (N x T) — used in every matvec
    DtGs = D_dense.T @ G_s  # N x T

    # LU-factorise Z_A once for right preconditioning
    lu_A, piv_A = scipy.linalg.lu_factor(Z_A)

    # Right-preconditioned operator: (I + Φ Z_A^{-1}) y = V
    # where Φ = phi_pf * D^T G_s D.
    # Matvec: y → y + phi_pf * D^T G_s D (Z_A^{-1} y)
    def matvec_right(y):
        z = scipy.linalg.lu_solve((lu_A, piv_A), y)  # Z_A^{-1} y
        return y + phi_pf * (DtGs @ (D_dense @ z))

    A_right = LinearOperator((N, N), matvec=matvec_right, dtype=np.complex128)

    single = V.ndim == 1
    if single:
        V = V.reshape(-1, 1)

    P = V.shape[1]
    I_all = np.zeros((N, P), dtype=np.complex128)

    for p in range(P):
        # Solve (I + Φ Z_A^{-1}) y = V
        y, info = gmres(
            A_right, V[:, p],
            rtol=1e-12, maxiter=200, restart=50,
        )
        # Recover current: I = Z_A^{-1} y
        I_all[:, p] = scipy.linalg.lu_solve((lu_A, piv_A), y)

    return I_all[:, 0] if single else I_all


def estimate_kD(mesh, k: float) -> float:
    """Estimate the electrical size kD of the mesh.

    Uses the bounding box diagonal as the characteristic dimension D.

    Parameters
    ----------
    mesh : Mesh
    k : float
        Wavenumber (real part used).

    Returns
    -------
    kD : float
    """
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    D = np.linalg.norm(bbox_max - bbox_min)
    return abs(k) * D
