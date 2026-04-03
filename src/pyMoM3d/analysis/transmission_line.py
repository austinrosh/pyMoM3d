"""Transmission line analysis utilities for multilayer benchmark validation.

Provides analytical formulas for microstrip and stripline characteristic
impedance, plus S-parameter-based extraction of propagation constant and
characteristic impedance from 2-port MoM results.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Analytical formulas
# ------------------------------------------------------------------

def microstrip_z0_hammerstad(W: float, h: float, eps_r: float) -> tuple:
    """Microstrip characteristic impedance via Hammerstad-Jensen formulas.

    Parameters
    ----------
    W : float
        Strip width (m).
    h : float
        Substrate height (m).
    eps_r : float
        Substrate relative permittivity.

    Returns
    -------
    Z0 : float
        Characteristic impedance (Ω).
    eps_eff : float
        Effective relative permittivity.

    References
    ----------
    E. Hammerstad and O. Jensen, "Accurate Models for Microstrip Computer-Aided
    Design," IEEE MTT-S Int. Microwave Symp. Dig., 1980, pp. 407-409.
    """
    u = W / h

    # Effective permittivity
    a = 1.0 + (1.0 / 49.0) * np.log(
        (u**4 + (u / 52.0)**2) / (u**4 + 0.432)
    ) + (1.0 / 18.7) * np.log(1.0 + (u / 18.1)**3)
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3.0))**0.053

    eps_eff = 0.5 * (eps_r + 1.0) + 0.5 * (eps_r - 1.0) * (1.0 + 10.0 / u)**(-a * b)

    # Characteristic impedance in free space (eps_r = 1)
    F = 6.0 + (2.0 * np.pi - 6.0) * np.exp(-(30.666 / u)**0.7528)
    Z0_air = 60.0 * np.log(F / u + np.sqrt(1.0 + (2.0 / u)**2))

    Z0 = Z0_air / np.sqrt(eps_eff)

    return float(Z0), float(eps_eff)


def stripline_z0_cohn(W: float, b: float, eps_r: float) -> float:
    """Stripline characteristic impedance via Cohn's elliptic-integral formula.

    Valid for a zero-thickness centered strip between two ground planes
    separated by distance b.

    Parameters
    ----------
    W : float
        Strip width (m).
    b : float
        Ground plane separation (m).  Strip is centered at b/2.
    eps_r : float
        Dielectric relative permittivity (fills entire cross-section).

    Returns
    -------
    Z0 : float
        Characteristic impedance (Ω).

    References
    ----------
    S. B. Cohn, "Problems in Strip Transmission Lines," IRE Trans. MTT,
    vol. 3, no. 2, pp. 119-126, March 1955.
    """
    from scipy.special import ellipk

    # Cohn formula: k = sech(π W / (2b)), Z0 = (30π/√ε_r) K(k')/K(k)
    k = 1.0 / np.cosh(np.pi * W / (2.0 * b))
    kp = np.sqrt(1.0 - k**2)
    Z0 = (30.0 * np.pi / np.sqrt(eps_r)) * (ellipk(kp**2) / ellipk(k**2))

    return float(Z0)


# ------------------------------------------------------------------
# S-parameter extraction
# ------------------------------------------------------------------

def s_to_abcd(S: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Convert 2×2 S-matrix to ABCD matrix.

    Parameters
    ----------
    S : (2, 2) complex128
        Scattering matrix.
    Z0 : float
        Reference impedance (Ω).

    Returns
    -------
    ABCD : (2, 2) complex128
        ABCD (transmission) matrix.
    """
    S11, S12, S21, S22 = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
    denom = 2.0 * S21

    A = ((1.0 + S11) * (1.0 - S22) + S12 * S21) / denom
    B = Z0 * ((1.0 + S11) * (1.0 + S22) - S12 * S21) / denom
    C = (1.0 / Z0) * ((1.0 - S11) * (1.0 - S22) - S12 * S21) / denom
    D = ((1.0 - S11) * (1.0 + S22) + S12 * S21) / denom

    return np.array([[A, B], [C, D]], dtype=np.complex128)


def extract_propagation_constant(
    S: np.ndarray, L: float, Z0_ref: float = 50.0
) -> complex:
    """Extract propagation constant γ from a 2-port S-matrix of a transmission line.

    Uses the ABCD matrix: for a uniform line of length L,
    ``cosh(γL) = A = D`` and ``γ = acosh(A) / L``.

    Parameters
    ----------
    S : (2, 2) complex128
        2-port scattering matrix.
    L : float
        Physical length of the line (m).
    Z0_ref : float
        Reference impedance used for S-parameter extraction (Ω).

    Returns
    -------
    gamma : complex
        Propagation constant α + jβ (Np/m + rad/m).
    """
    ABCD = s_to_abcd(S, Z0_ref)
    A = ABCD[0, 0]
    return np.arccosh(A) / L


def extract_z0_from_s(S: np.ndarray, Z0_ref: float = 50.0) -> complex:
    """Extract characteristic impedance Z0 from a 2-port S-matrix.

    Uses the ABCD matrix: ``Z0 = sqrt(B / C)`` for a uniform line.

    Parameters
    ----------
    S : (2, 2) complex128
        2-port scattering matrix.
    Z0_ref : float
        Reference impedance used for S-parameter extraction (Ω).

    Returns
    -------
    Z0 : complex
        Characteristic impedance (Ω).
    """
    ABCD = s_to_abcd(S, Z0_ref)
    B, C = ABCD[0, 1], ABCD[1, 0]
    if abs(C) < 1e-30:
        return complex(np.inf)
    return np.sqrt(B / C)
