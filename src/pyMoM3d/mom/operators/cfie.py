"""CFIE operator: Combined Field Integral Equation."""

import numpy as np

from .base import AbstractOperator
from .efie import EFIEOperator
from .mfie import MFIEOperator, compute_gram_matrix

# ---------------------------------------------------------------------------
# Optional fast backends
# ---------------------------------------------------------------------------
try:
    from .._cpp_kernels import (
        fill_impedance_cfie_cpp as _fill_impedance_cfie_cpp,
    )
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False

try:
    from ..numba_kernels import (
        NUMBA_AVAILABLE as _NUMBA_AVAILABLE,
        fill_Z_cfie_numba as _fill_Z_cfie_numba,
    )
except ImportError:
    _NUMBA_AVAILABLE = False

from ...greens.quadrature import triangle_quad_rule


class CFIEOperator(AbstractOperator):
    """CFIE impedance matrix operator (Mautz–Harrington combination).

    Assembles the matrix:

        Z^CFIE = α · Z^EFIE + (1−α) · η · Z^MFIE

    with excitation:

        V^CFIE = α · V^EFIE + (1−α) · η · V^MFIE

    This eliminates spurious interior resonances that appear in EFIE alone,
    while preserving well-posedness of the integral equation system.

    Parameters
    ----------
    alpha : float
        Combination parameter in (0, 1).  α=1 → pure EFIE; α=0 → pure
        (scaled) MFIE.  Default 0.5.  Values near 0 or 1 weaken
        resonance suppression; a warning is raised for α < 0.2 or α > 0.8.

    Notes
    -----
    - Valid only for **closed** PEC surfaces (no boundary edges).
    - The matrix is **not symmetric** due to the MFIE K-term.
    - Fast cpp/numba backends use a single fused kernel (one double-quadrature
      pass computes I_A, I_Phi, and I_K simultaneously).
    - Numpy fallback blends per-pair EFIE and MFIE contributions.
    """

    is_symmetric: bool = False

    def __init__(self, alpha: float = 0.5):
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"CFIEOperator: alpha must be in the open interval (0, 1), got {alpha}"
            )
        if alpha < 0.2 or alpha > 0.8:
            import warnings
            warnings.warn(
                f"CFIEOperator: alpha={alpha} is near 0 or 1; "
                "resonance suppression may be weakened.",
                UserWarning,
                stacklevel=2,
            )
        self.alpha = float(alpha)
        self._efie = EFIEOperator()
        self._mfie = MFIEOperator()

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

        near_order = min(quad_order + 3, 13)
        weights_near, bary_near = triangle_quad_rule(near_order)
        weights_near = np.ascontiguousarray(weights_near, dtype=np.float64)
        bary_near    = np.ascontiguousarray(bary_near,    dtype=np.float64)

        if backend == 'cpp':
            _fill_impedance_cfie_cpp(
                Z, verts, tris_arr, normals,
                t_plus, t_minus, fv_plus, fv_minus,
                a_plus, a_minus, elen, cents, medge, tarea,
                weights, bary, weights_near, bary_near,
                float(k), float(eta), float(self.alpha),
                float(near_threshold), int(quad_order),
                0,  # num_threads: 0 = OMP default
            )
        elif backend == 'numba':
            _fill_Z_cfie_numba(
                Z, verts, tris_arr, normals,
                t_plus, t_minus, fv_plus, fv_minus,
                a_plus, a_minus, elen, cents, medge, tarea,
                weights, bary, weights_near, bary_near,
                float(k), float(eta), float(self.alpha), float(near_threshold),
            )
        else:
            raise ValueError(f"CFIEOperator.fill_fast: unknown backend '{backend}'")

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
        """Per-pair numpy fallback: blend EFIE and MFIE contributions."""
        args = (
            k, eta, mesh,
            tri_test, tri_src, fv_test, fv_src,
            sign_test, sign_src, l_test, l_src,
            A_test, A_src,
            quad_order, near_threshold, weights, bary,
            twice_area_test, twice_area_src, is_near, n_hat_test,
        )
        z_efie = self._efie.compute_pair_numpy(*args)
        z_mfie = self._mfie.compute_pair_numpy(*args)
        return self.alpha * z_efie + (1.0 - self.alpha) * eta * z_mfie

    def post_assembly(self, Z, rwg_basis, mesh, k, eta) -> None:
        """Validate closed surface; add (1−α)·η·(1/2)·B Gram term."""
        if rwg_basis.num_boundary_edges > 0:
            raise ValueError(
                "CFIEOperator requires a closed surface "
                f"(found {rwg_basis.num_boundary_edges} boundary edge(s))."
            )
        B = compute_gram_matrix(rwg_basis, mesh)
        Z += 0.5 * (1.0 - self.alpha) * eta * B
