"""EFIE operator: Electric Field Integral Equation."""

import numpy as np

from ...greens.singularity import integrate_green_singular, integrate_rho_green_singular
from .base import AbstractOperator

# ---------------------------------------------------------------------------
# Optional fast backends
# ---------------------------------------------------------------------------
try:
    from .._cpp_kernels import fill_impedance_cpp as _fill_impedance_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False

try:
    from ..numba_kernels import NUMBA_AVAILABLE as _NUMBA_AVAILABLE, fill_Z_numba as _fill_Z_numba
except ImportError:
    _NUMBA_AVAILABLE = False


class EFIEOperator(AbstractOperator):
    """EFIE impedance matrix operator.

    Assembles the matrix:

        Z_mn = jk*eta * A_mn  -  (j*eta/k) * Phi_mn

    where A_mn and Phi_mn are the vector- and scalar-potential integrals
    from the standard EFIE for PEC surfaces.
    """

    def supports_backend(self, backend: str) -> bool:
        if backend == 'cpp':
            return _CPP_AVAILABLE
        if backend == 'numba':
            return _NUMBA_AVAILABLE
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
        tri_normals: np.ndarray,      # unused by EFIE
        weights: np.ndarray,
        bary: np.ndarray,
        quad_order: int,
        near_threshold: float,
    ) -> None:
        verts   = mesh.vertices.astype(np.float64, copy=False)
        tris    = mesh.triangles.astype(np.int32, copy=False)
        t_plus  = rwg_basis.t_plus.astype(np.int32, copy=False)
        t_minus = rwg_basis.t_minus.astype(np.int32, copy=False)
        fv_plus  = rwg_basis.free_vertex_plus.astype(np.int32, copy=False)
        fv_minus = rwg_basis.free_vertex_minus.astype(np.int32, copy=False)
        a_plus  = rwg_basis.area_plus.astype(np.float64, copy=False)
        a_minus = rwg_basis.area_minus.astype(np.float64, copy=False)
        elen    = rwg_basis.edge_length.astype(np.float64, copy=False)
        cents   = np.ascontiguousarray(tri_centroids, dtype=np.float64)
        medge   = np.ascontiguousarray(tri_mean_edge, dtype=np.float64)
        tarea   = np.ascontiguousarray(tri_twice_area, dtype=np.float64)

        if backend == 'cpp':
            _fill_impedance_cpp(
                Z, verts, tris, t_plus, t_minus, fv_plus, fv_minus,
                a_plus, a_minus, elen, cents, medge, tarea,
                weights, bary,
                float(k), float(eta), float(near_threshold), int(quad_order),
                0,  # num_threads: 0 = OMP default
            )
        elif backend == 'numba':
            _fill_Z_numba(
                Z, verts, tris, t_plus, t_minus, fv_plus, fv_minus,
                a_plus, a_minus, elen, cents, medge, tarea,
                weights, bary,
                float(k), float(eta), float(near_threshold), int(quad_order),
            )
        else:
            raise ValueError(f"EFIEOperator.fill_fast: unknown backend '{backend}'")

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
        n_hat_test,         # unused by EFIE
    ) -> complex:
        """Compute EFIE contribution from one triangle pair.

        Returns  prefactor_A * I_A  +  prefactor_Phi * I_Phi  for this pair.
        """
        verts_test = mesh.vertices[mesh.triangles[tri_test]]
        verts_src  = mesh.vertices[mesh.triangles[tri_src]]
        r_fv_test  = mesh.vertices[fv_test]
        r_fv_src   = mesh.vertices[fv_src]

        I_A_raw   = 0.0 + 0.0j
        I_Phi_raw = 0.0 + 0.0j

        for i in range(len(weights)):
            r_obs = (bary[i, 0] * verts_test[0]
                     + bary[i, 1] * verts_test[1]
                     + bary[i, 2] * verts_test[2])
            rho_test = r_obs - r_fv_test

            if is_near:
                g_int = integrate_green_singular(
                    k, r_obs, verts_src[0], verts_src[1], verts_src[2],
                    quad_order=quad_order, near_threshold=near_threshold,
                )
            else:
                g_int = 0.0 + 0.0j
                for j in range(len(weights)):
                    r_prime = (bary[j, 0] * verts_src[0]
                               + bary[j, 1] * verts_src[1]
                               + bary[j, 2] * verts_src[2])
                    R = np.linalg.norm(r_obs - r_prime)
                    g_int += weights[j] * np.exp(-1j * k * R) / (4.0 * np.pi * R)
                g_int *= twice_area_src

            I_Phi_raw += weights[i] * g_int

            if is_near:
                rho_src_g_int = integrate_rho_green_singular(
                    k, r_obs, verts_src[0], verts_src[1], verts_src[2],
                    r_fv_src, quad_order=quad_order, near_threshold=near_threshold,
                )
            else:
                rho_src_g_int = np.zeros(3, dtype=np.complex128)
                for j in range(len(weights)):
                    r_prime = (bary[j, 0] * verts_src[0]
                               + bary[j, 1] * verts_src[1]
                               + bary[j, 2] * verts_src[2])
                    rho_src = r_prime - r_fv_src
                    R = np.linalg.norm(r_obs - r_prime)
                    if R < 1e-30:
                        R = 1e-30
                    g_val = np.exp(-1j * k * R) / (4.0 * np.pi * R)
                    rho_src_g_int += weights[j] * rho_src * g_val
                rho_src_g_int *= twice_area_src

            I_A_raw += weights[i] * np.dot(rho_test, rho_src_g_int)

        I_A_raw   *= twice_area_test
        I_Phi_raw *= twice_area_test

        scale_A   = (sign_test * l_test / (2.0 * A_test)) * (sign_src * l_src / (2.0 * A_src))
        scale_Phi = (sign_test * l_test / A_test) * (sign_src * l_src / A_src)

        prefactor_A   = 1j * k * eta
        prefactor_Phi = -1j * eta / k

        return prefactor_A * I_A_raw * scale_A + prefactor_Phi * I_Phi_raw * scale_Phi
