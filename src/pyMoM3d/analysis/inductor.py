"""Inductor analysis utilities.

Provides the modified Wheeler formula for planar spiral inductance,
Z-parameter extraction (simple L = Im(Z)/omega), and Y-parameter
extraction (L = -1/(omega*Im(Y11))) used by commercial EM solvers
for flat L(f) below the self-resonant frequency.
"""

from __future__ import annotations

import numpy as np


def wheeler_inductance(n: float, d_out: float, d_in: float) -> float:
    """Inductance of a square planar spiral via the modified Wheeler formula.

    Parameters
    ----------
    n : float
        Number of turns.
    d_out : float
        Outer diameter (m).
    d_in : float
        Inner diameter (m).

    Returns
    -------
    L : float
        Inductance (H).

    References
    ----------
    S. S. Mohan, M. del Mar Hershenson, S. P. Boyd, and T. H. Lee,
    "Simple Accurate Expressions for Planar Spiral Inductances,"
    IEEE JSSC, vol. 34, no. 10, Oct. 1999, pp. 1419-1424.

    Notes
    -----
    Uses the modified Wheeler coefficients for a square spiral:
    K1 = 2.34, K2 = 2.75.
    """
    from ..utils.constants import mu0

    d_avg = 0.5 * (d_out + d_in)
    rho = (d_out - d_in) / (d_out + d_in)  # fill ratio

    K1 = 2.34
    K2 = 2.75

    L = (K1 * mu0 * n**2 * d_avg) / (1.0 + K2 * rho)

    return float(L)


def quality_factor(Z_in: complex) -> float:
    """Quality factor Q = Im(Z) / Re(Z).

    Parameters
    ----------
    Z_in : complex
        Input impedance (Ω).

    Returns
    -------
    Q : float
        Quality factor.  Returns 0 if Re(Z) ≈ 0.
    """
    if abs(Z_in.real) < 1e-30:
        return 0.0
    return abs(Z_in.imag / Z_in.real)


def inductance_from_z(Z_in: complex, freq: float) -> float:
    """Extract inductance from input impedance: L = Im(Z) / ω.

    Parameters
    ----------
    Z_in : complex
        Input impedance (Ω).
    freq : float
        Frequency (Hz).

    Returns
    -------
    L : float
        Inductance (H).  Negative if capacitive.
    """
    omega = 2.0 * np.pi * freq
    if omega < 1e-30:
        return 0.0
    return Z_in.imag / omega


# ---------------------------------------------------------------------------
# Y-parameter extraction (standard EDA method)
# ---------------------------------------------------------------------------

def inductance_from_y(Y11: complex, freq: float) -> float:
    """Extract inductance from Y-parameter: L = -1 / (omega * Im(Y11)).

    This is the standard method used in commercial EM solvers (EMX,
    Momentum, HFSS, Sonnet).  Unlike ``inductance_from_z``, it naturally
    separates parallel parasitic capacitance, giving flat L(f) below the
    self-resonant frequency.

    Parameters
    ----------
    Y11 : complex
        1-port admittance (S).
    freq : float
        Frequency (Hz).

    Returns
    -------
    L : float
        Inductance (H).  Positive when inductive (Im(Y11) < 0).
    """
    omega = 2.0 * np.pi * freq
    if omega < 1e-30 or abs(Y11.imag) < 1e-30:
        return 0.0
    return -1.0 / (omega * Y11.imag)


def quality_factor_y(Y11: complex) -> float:
    """Quality factor from Y-parameter: Q = -Im(Y11) / Re(Y11).

    Positive Q indicates inductive behavior (Im(Y11) < 0, Re(Y11) > 0).

    Parameters
    ----------
    Y11 : complex
        1-port admittance (S).

    Returns
    -------
    Q : float
        Quality factor.  Returns 0 if Re(Y11) ~ 0.
    """
    if abs(Y11.real) < 1e-30:
        return 0.0
    return -Y11.imag / Y11.real


def resistance_from_y(Y11: complex) -> float:
    """Series resistance from Y-parameter: R = Re(1/Y11).

    Parameters
    ----------
    Y11 : complex
        1-port admittance (S).

    Returns
    -------
    R : float
        Series resistance (Ohm).
    """
    if abs(Y11) < 1e-30:
        return 0.0
    return (1.0 / Y11).real


def self_resonant_frequency(
    frequencies: np.ndarray,
    Y11_array: np.ndarray,
) -> tuple:
    """Detect the self-resonant frequency where Im(Y11) crosses zero.

    Below SRF, Im(Y11) < 0 (inductive).  Above SRF, Im(Y11) > 0
    (capacitive).  The SRF is where L(f) extraction becomes invalid.

    Uses linear interpolation between adjacent frequency samples.

    Parameters
    ----------
    frequencies : ndarray, shape (N,)
        Frequency points (Hz), monotonically increasing.
    Y11_array : ndarray, shape (N,), complex128
        Y11 at each frequency.

    Returns
    -------
    f_srf : float
        Self-resonant frequency (Hz).  ``np.inf`` if no crossing found.
    index : int
        Index of the last inductive sample before the crossing.
        ``-1`` if no crossing found.
    """
    im_y = Y11_array.imag
    for i in range(len(im_y) - 1):
        if im_y[i] < 0 and im_y[i + 1] >= 0:
            # Linear interpolation
            f_srf = frequencies[i] + (
                -im_y[i] / (im_y[i + 1] - im_y[i])
                * (frequencies[i + 1] - frequencies[i])
            )
            return float(f_srf), i
    return float('inf'), -1


def differential_inductance_from_y(
    Y_matrix: np.ndarray,
    freq: float,
) -> float:
    """Differential inductance from 2-port Y-matrix.

    L_diff = -2 / (omega * Im(Y11 + Y22 - Y12 - Y21))

    Parameters
    ----------
    Y_matrix : ndarray, shape (2, 2), complex128
        2-port admittance matrix.
    freq : float
        Frequency (Hz).

    Returns
    -------
    L_diff : float
        Differential inductance (H).
    """
    omega = 2.0 * np.pi * freq
    Y_diff_imag = (Y_matrix[0, 0] + Y_matrix[1, 1]
                   - Y_matrix[0, 1] - Y_matrix[1, 0]).imag
    if omega < 1e-30 or abs(Y_diff_imag) < 1e-30:
        return 0.0
    return -2.0 / (omega * Y_diff_imag)


def differential_quality_factor_y(Y_matrix: np.ndarray) -> float:
    """Differential quality factor from 2-port Y-matrix.

    Q_diff = -Im(Y_diff) / Re(Y_diff)

    where Y_diff = Y11 + Y22 - Y12 - Y21.

    Parameters
    ----------
    Y_matrix : ndarray, shape (2, 2), complex128
        2-port admittance matrix.

    Returns
    -------
    Q_diff : float
        Differential quality factor.
    """
    Y_diff = (Y_matrix[0, 0] + Y_matrix[1, 1]
              - Y_matrix[0, 1] - Y_matrix[1, 0])
    if abs(Y_diff.real) < 1e-30:
        return 0.0
    return -Y_diff.imag / Y_diff.real
