"""Radar Cross Section (RCS) computation."""

import numpy as np


def compute_rcs(
    E_theta: np.ndarray,
    E_phi: np.ndarray,
    E_inc_mag: float = 1.0,
) -> np.ndarray:
    """Compute bistatic RCS from far-field components.

    sigma = 4*pi * (|E_theta|^2 + |E_phi|^2) / |E_inc|^2

    Parameters
    ----------
    E_theta, E_phi : ndarray, shape (M,), complex128
        Far-field theta and phi components.
    E_inc_mag : float
        Magnitude of incident electric field (V/m).

    Returns
    -------
    rcs_dBsm : ndarray, shape (M,)
        RCS in dBsm.
    """
    sigma = 4.0 * np.pi * (np.abs(E_theta)**2 + np.abs(E_phi)**2) / E_inc_mag**2
    # Avoid log of zero
    sigma = np.maximum(sigma, 1e-30)
    return 10.0 * np.log10(sigma)


def compute_monostatic_rcs(
    E_theta: complex,
    E_phi: complex,
    E_inc_mag: float = 1.0,
) -> float:
    """Compute monostatic RCS (single direction).

    Parameters
    ----------
    E_theta, E_phi : complex
        Far-field components in the backscatter direction.
    E_inc_mag : float
        Incident field magnitude.

    Returns
    -------
    rcs_dBsm : float
    """
    sigma = 4.0 * np.pi * (abs(E_theta)**2 + abs(E_phi)**2) / E_inc_mag**2
    return 10.0 * np.log10(max(sigma, 1e-30))
