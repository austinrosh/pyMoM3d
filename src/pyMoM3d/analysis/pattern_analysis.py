"""Radiation pattern analysis: gain, directivity, beamwidth."""

import numpy as np


def compute_directivity(
    E_theta: np.ndarray,
    E_phi: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    eta: float,
) -> tuple:
    """Compute directivity from far-field pattern.

    Parameters
    ----------
    E_theta, E_phi : ndarray, shape (N_theta, N_phi), complex128
        Far-field components on a regular (theta, phi) grid.
    theta : ndarray, shape (N_theta,)
        Theta angles (radians).
    phi : ndarray, shape (N_phi,)
        Phi angles (radians).
    eta : float
        Intrinsic impedance (Ohms).

    Returns
    -------
    D : ndarray, shape (N_theta, N_phi)
        Directivity pattern.
    D_max : float
        Maximum directivity (linear).
    D_max_dBi : float
        Maximum directivity in dBi.
    """
    # Radiation intensity
    U = (np.abs(E_theta)**2 + np.abs(E_phi)**2) / (2.0 * eta)

    # Total radiated power via integration over solid angle
    dtheta = theta[1] - theta[0] if len(theta) > 1 else np.pi
    dphi = phi[1] - phi[0] if len(phi) > 1 else 2 * np.pi

    sin_theta = np.sin(theta)
    # U is (N_theta, N_phi), integrate
    P_rad = np.sum(U * sin_theta[:, np.newaxis]) * dtheta * dphi

    if P_rad < 1e-30:
        return np.zeros_like(U), 0.0, -np.inf

    D = 4.0 * np.pi * U / P_rad
    D_max = float(np.max(D))
    D_max_dBi = 10.0 * np.log10(max(D_max, 1e-30))

    return D, D_max, D_max_dBi


def compute_beamwidth_3dB(D: np.ndarray, theta: np.ndarray) -> float:
    """Estimate 3 dB beamwidth from directivity in principal plane.

    Parameters
    ----------
    D : ndarray, shape (N_theta,)
        Directivity in a principal plane cut.
    theta : ndarray, shape (N_theta,)

    Returns
    -------
    bw_deg : float
        3 dB beamwidth in degrees.
    """
    D_max = np.max(D)
    half_power = D_max / 2.0
    above = D >= half_power
    if not np.any(above):
        return 360.0

    indices = np.where(above)[0]
    theta_min = theta[indices[0]]
    theta_max = theta[indices[-1]]
    return float(np.degrees(theta_max - theta_min))
