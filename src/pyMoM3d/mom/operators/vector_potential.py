"""Vector-potential-only EFIE operator for inductance extraction.

At low frequencies (kD << 1), the standard EFIE impedance matrix is
dominated by the scalar-potential term (proportional to 1/omega), making
inductance extraction unreliable.  This operator assembles only the
vector-potential (A) term:

    Z_A[m,n] = j*k*eta * integral( f_m . f_n * G, dS dS' )

The resulting matrix is proportional to omega and represents the mutual
inductance between RWG basis functions.  Solving Z_A * I = V gives
currents whose terminal impedance yields frequency-independent
inductance L = Im(Z_in) / omega.

This is the standard approach used in partial-element equivalent circuit
(PEEC) methods and commercial inductance extractors (FastHenry, EMX).
"""

import numpy as np

from ...greens.singularity import integrate_rho_green_singular
from .efie import EFIEOperator


class VectorPotentialOperator(EFIEOperator):
    """EFIE operator returning only the vector-potential (A) contribution.

    Drops the scalar-potential (Phi) term entirely during assembly.
    The resulting impedance matrix Z_A scales as omega, enabling
    frequency-independent inductance extraction at low kD.
    """

    def supports_backend(self, backend: str) -> bool:
        if backend == 'cpp':
            from .efie import _CPP_AVAILABLE
            return _CPP_AVAILABLE
        return backend == 'numpy'

    def fill_fast(self, backend: str, Z, rwg_basis, mesh, k, eta,
                  tri_centroids, tri_mean_edge, tri_twice_area,
                  tri_normals, weights, bary, quad_order, near_threshold):
        if backend == 'cpp':
            from .efie import _fill_impedance_cpp
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
            _fill_impedance_cpp(
                Z, verts, tris, t_plus, t_minus, fv_plus, fv_minus,
                a_plus, a_minus, elen, cents, medge, tarea,
                weights, bary,
                float(k), float(eta), float(near_threshold), int(quad_order),
                0,      # num_threads
                True,   # a_only
            )
        else:
            raise ValueError(f"VectorPotentialOperator.fill_fast: unknown backend '{backend}'")

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
        """Compute only the vector-potential contribution from one triangle pair.

        Returns  prefactor_A * I_A  (no Phi term).
        """
        verts_test = mesh.vertices[mesh.triangles[tri_test]]
        verts_src  = mesh.vertices[mesh.triangles[tri_src]]
        r_fv_test  = mesh.vertices[fv_test]
        r_fv_src   = mesh.vertices[fv_src]

        I_A_raw = 0.0 + 0.0j

        for i in range(len(weights)):
            r_obs = (bary[i, 0] * verts_test[0]
                     + bary[i, 1] * verts_test[1]
                     + bary[i, 2] * verts_test[2])
            rho_test = r_obs - r_fv_test

            if is_near:
                rho_src_g_int = integrate_rho_green_singular(
                    k, r_obs, verts_src[0], verts_src[1], verts_src[2],
                    r_fv_src, quad_order=quad_order,
                    near_threshold=near_threshold,
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

        I_A_raw *= twice_area_test

        scale_A = (sign_test * l_test / (2.0 * A_test)) * \
                  (sign_src * l_src / (2.0 * A_src))
        prefactor_A = 1j * k * eta

        return prefactor_A * I_A_raw * scale_A
