"""EFIE impedance matrix fill.

Assembles the N x N complex impedance matrix Z for the Electric Field
Integral Equation (EFIE) using RWG basis functions.

Z_mn = jk*eta * [ A_mn - (1/k^2) * Phi_mn ]

where:
    A_mn   = integral integral f_m(r) . f_n(r') g(r,r') dS dS'
    Phi_mn = integral integral div(f_m(r)) div(f_n(r')) g(r,r') dS dS'
    g(r,r') = exp(-jkR) / (4*pi*R)

Each (m, n) pair involves 4 triangle-triangle interactions.

Convention: exp(-j*omega*t) time dependence.
"""

import numpy as np

from ..mesh.mesh_data import Mesh
from ..mesh.rwg_basis import RWGBasis
from ..greens.quadrature import triangle_quad_rule
from ..greens.singularity import integrate_green_singular, integrate_rho_green_singular


def _compute_triangle_pair(
    k: float,
    mesh: Mesh,
    tri_test: int,
    tri_src: int,
    fv_test: int,
    fv_src: int,
    sign_test: float,
    sign_src: float,
    l_test: float,
    l_src: float,
    A_test: float,
    A_src: float,
    quad_order: int,
    near_threshold: float,
):
    """Compute A-term and Phi-term contribution from one triangle pair.

    Returns (I_A, I_Phi) where:
        I_A   = (sign_test * l_test / 2A_test) * (sign_src * l_src / 2A_src) *
                double_integral (r - r_fv_test) . (r' - r_fv_src) * g(r,r') dS dS'
        I_Phi = (sign_test * l_test / A_test) * (sign_src * l_src / A_src) *
                double_integral g(r,r') dS dS'
    """
    verts_test = mesh.vertices[mesh.triangles[tri_test]]
    verts_src = mesh.vertices[mesh.triangles[tri_src]]
    r_fv_test = mesh.vertices[fv_test]
    r_fv_src = mesh.vertices[fv_src]

    weights_o, bary_o = triangle_quad_rule(quad_order)
    cross_test = np.cross(verts_test[1] - verts_test[0], verts_test[2] - verts_test[0])
    twice_area_test = np.linalg.norm(cross_test)

    weights_s, bary_s = triangle_quad_rule(quad_order)
    cross_src = np.cross(verts_src[1] - verts_src[0], verts_src[2] - verts_src[0])
    twice_area_src = np.linalg.norm(cross_src)

    # Check if triangles are the same or adjacent (need singularity handling)
    centroid_test = np.mean(verts_test, axis=0)
    centroid_src = np.mean(verts_src, axis=0)
    mean_edge = (np.linalg.norm(verts_src[1] - verts_src[0])
                 + np.linalg.norm(verts_src[2] - verts_src[1])
                 + np.linalg.norm(verts_src[0] - verts_src[2])) / 3.0
    dist = np.linalg.norm(centroid_test - centroid_src)
    is_near = dist < near_threshold * mean_edge * 3.0 if mean_edge > 1e-30 else True

    I_A_raw = 0.0 + 0.0j
    I_Phi_raw = 0.0 + 0.0j

    for i in range(len(weights_o)):
        r_obs = (bary_o[i, 0] * verts_test[0] + bary_o[i, 1] * verts_test[1]
                 + bary_o[i, 2] * verts_test[2])
        rho_test = r_obs - r_fv_test

        if is_near:
            # Use singularity extraction for the inner integral of g
            g_int = integrate_green_singular(
                k, r_obs, verts_src[0], verts_src[1], verts_src[2],
                quad_order=quad_order, near_threshold=near_threshold,
            )
        else:
            g_int = 0.0 + 0.0j
            for j in range(len(weights_s)):
                r_prime = (bary_s[j, 0] * verts_src[0] + bary_s[j, 1] * verts_src[1]
                           + bary_s[j, 2] * verts_src[2])
                R = np.linalg.norm(r_obs - r_prime)
                g_int += weights_s[j] * np.exp(-1j * k * R) / (4.0 * np.pi * R)
            g_int *= twice_area_src

        # Scalar potential: just needs integral of g
        I_Phi_raw += weights_o[i] * g_int

        # Vector potential: needs integral of rho_src * g
        # integral_src (r' - r_fv_src) * g(r_obs, r') dS'
        if is_near:
            rho_src_g_int = integrate_rho_green_singular(
                k, r_obs, verts_src[0], verts_src[1], verts_src[2],
                r_fv_src, quad_order=quad_order, near_threshold=near_threshold,
            )
        else:
            rho_src_g_int = np.zeros(3, dtype=np.complex128)
            for j in range(len(weights_s)):
                r_prime = (bary_s[j, 0] * verts_src[0] + bary_s[j, 1] * verts_src[1]
                           + bary_s[j, 2] * verts_src[2])
                rho_src = r_prime - r_fv_src
                R = np.linalg.norm(r_obs - r_prime)
                if R < 1e-30:
                    R = 1e-30
                g_val = np.exp(-1j * k * R) / (4.0 * np.pi * R)
                rho_src_g_int += weights_s[j] * rho_src * g_val
            rho_src_g_int *= twice_area_src

        I_A_raw += weights_o[i] * np.dot(rho_test, rho_src_g_int)

    I_A_raw *= twice_area_test
    I_Phi_raw *= twice_area_test

    # Apply basis function scaling
    scale_A = (sign_test * l_test / (2.0 * A_test)) * (sign_src * l_src / (2.0 * A_src))
    scale_Phi = (sign_test * l_test / A_test) * (sign_src * l_src / A_src)

    return I_A_raw * scale_A, I_Phi_raw * scale_Phi


def fill_impedance_matrix(
    rwg_basis: RWGBasis,
    mesh: Mesh,
    k: float,
    eta: float,
    quad_order: int = 4,
    near_threshold: float = 0.2,
    progress_callback=None,
) -> np.ndarray:
    """Assemble the EFIE impedance matrix.

    Parameters
    ----------
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
    progress_callback : callable, optional
        Called once per completed outer row with ``progress_callback(fraction)``
        where *fraction* is in [0, 1).

    Returns
    -------
    Z : ndarray, shape (N, N), complex128
        Impedance matrix.
    """
    N = rwg_basis.num_basis
    Z = np.zeros((N, N), dtype=np.complex128)

    prefactor_A = 1j * k * eta
    prefactor_Phi = -1j * eta / k

    total_pairs = N * (N + 1) // 2
    for m in range(N):
        if progress_callback is not None:
            done = m * (2 * N - m - 1) // 2
            progress_callback(done / total_pairs if total_pairs > 0 else 0.0)
        for n in range(m, N):
            I_A_total = 0.0 + 0.0j
            I_Phi_total = 0.0 + 0.0j

            for (tri_m, fv_m, sign_m, A_m) in [
                (rwg_basis.t_plus[m], rwg_basis.free_vertex_plus[m], +1.0, rwg_basis.area_plus[m]),
                (rwg_basis.t_minus[m], rwg_basis.free_vertex_minus[m], -1.0, rwg_basis.area_minus[m]),
            ]:
                for (tri_n, fv_n, sign_n, A_n) in [
                    (rwg_basis.t_plus[n], rwg_basis.free_vertex_plus[n], +1.0, rwg_basis.area_plus[n]),
                    (rwg_basis.t_minus[n], rwg_basis.free_vertex_minus[n], -1.0, rwg_basis.area_minus[n]),
                ]:
                    I_A, I_Phi = _compute_triangle_pair(
                        k, mesh, tri_m, tri_n, fv_m, fv_n,
                        sign_m, sign_n,
                        rwg_basis.edge_length[m], rwg_basis.edge_length[n],
                        A_m, A_n,
                        quad_order, near_threshold,
                    )
                    I_A_total += I_A
                    I_Phi_total += I_Phi

            Z[m, n] = prefactor_A * I_A_total + prefactor_Phi * I_Phi_total
            if m != n:
                Z[n, m] = Z[m, n]

    return Z
