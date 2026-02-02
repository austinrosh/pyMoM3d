"""Far-field computation from MoM surface currents.

The far-field electric field is computed from the radiation integral:

    E_far(r) ~ -jk*eta/(4*pi) * exp(-jkr)/r * L(theta, phi)

where L is the vector radiation function:

    L(theta, phi) = integral_S f_n(r') exp(+jk*r_hat.r') dS'

Note: the far-field uses exp(+jk*r_hat.r'), OPPOSITE sign from the
Green's function exp(-jkR). See CONVENTIONS.md.
"""

import numpy as np

from ..mesh.mesh_data import Mesh
from ..mesh.rwg_basis import RWGBasis
from ..greens.quadrature import triangle_quad_rule


def compute_far_field(
    I_coeffs: np.ndarray,
    rwg_basis: RWGBasis,
    mesh: Mesh,
    k: float,
    eta: float,
    theta: np.ndarray,
    phi: np.ndarray,
    quad_order: int = 4,
    progress_callback=None,
) -> tuple:
    """Compute far-field E_theta and E_phi components.

    Parameters
    ----------
    I_coeffs : ndarray, shape (N,), complex128
        Current expansion coefficients from MoM solve.
    rwg_basis : RWGBasis
    mesh : Mesh
    k : float
        Wavenumber (rad/m).
    eta : float
        Intrinsic impedance (Ohms).
    theta : ndarray, shape (M,)
        Elevation angles (radians), 0 = +z.
    phi : ndarray, shape (M,)
        Azimuth angles (radians).
    quad_order : int

    Returns
    -------
    E_theta : ndarray, shape (M,), complex128
    E_phi : ndarray, shape (M,), complex128

    Notes
    -----
    The returned fields are the far-field pattern functions, i.e., the
    E-field multiplied by r*exp(+jkr)/(jk*eta/(4*pi)).  To get the
    actual field at distance r, multiply by -jk*eta/(4*pi) * exp(-jkr)/r.
    For RCS computation, only the pattern is needed.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    M = len(theta)

    # Observation directions
    r_hat = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=-1)  # (M, 3)

    # Theta and phi unit vectors
    theta_hat = np.stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta),
    ], axis=-1)  # (M, 3)

    phi_hat = np.stack([
        -np.sin(phi),
        np.cos(phi),
        np.zeros_like(phi),
    ], axis=-1)  # (M, 3)

    weights, bary = triangle_quad_rule(quad_order)

    # Radiation vector N(r_hat) = sum_n I_n * integral f_n(r') exp(+jk r_hat.r') dS'
    N_vec = np.zeros((M, 3), dtype=np.complex128)

    _N_basis = rwg_basis.num_basis
    for n in range(_N_basis):
        if progress_callback is not None and _N_basis > 0:
            progress_callback(n / _N_basis)
        I_n = I_coeffs[n]
        if abs(I_n) < 1e-30:
            continue

        for (tri, fv, sign, area) in [
            (rwg_basis.t_plus[n], rwg_basis.free_vertex_plus[n], +1.0, rwg_basis.area_plus[n]),
            (rwg_basis.t_minus[n], rwg_basis.free_vertex_minus[n], -1.0, rwg_basis.area_minus[n]),
        ]:
            verts = mesh.vertices[mesh.triangles[tri]]
            r_fv = mesh.vertices[fv]
            cross = np.cross(verts[1] - verts[0], verts[2] - verts[0])
            twice_area = np.linalg.norm(cross)
            scale = sign * rwg_basis.edge_length[n] / (2.0 * area)

            for i in range(len(weights)):
                r_prime = (bary[i, 0] * verts[0] + bary[i, 1] * verts[1]
                           + bary[i, 2] * verts[2])
                rho = r_prime - r_fv
                f_val = scale * rho  # f_n at this point, shape (3,)

                # Phase: exp(+jk * r_hat . r'), for all observation directions
                phase = np.exp(1j * k * (r_hat @ r_prime))  # (M,)

                # Accumulate
                N_vec += (I_n * weights[i] * twice_area) * np.outer(phase, f_val)

    # Project onto theta and phi
    E_theta = -1j * k * eta / (4.0 * np.pi) * np.sum(N_vec * theta_hat, axis=-1)
    E_phi = -1j * k * eta / (4.0 * np.pi) * np.sum(N_vec * phi_hat, axis=-1)

    return E_theta, E_phi
