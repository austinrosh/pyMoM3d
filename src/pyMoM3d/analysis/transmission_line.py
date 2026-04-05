"""Transmission line and patch antenna analysis utilities.

Provides analytical formulas for microstrip, stripline, and CPW characteristic
impedance, patch antenna cavity model, plus S-parameter-based extraction of
propagation constant and characteristic impedance from 2-port MoM results.
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

    # Cohn formula: k = sech(π W / (2b)), Z0 = (30π/√ε_r) K(k²)/K(k'²)
    k = 1.0 / np.cosh(np.pi * W / (2.0 * b))
    kp = np.sqrt(1.0 - k**2)
    Z0 = (30.0 * np.pi / np.sqrt(eps_r)) * (ellipk(k**2) / ellipk(kp**2))

    return float(Z0)


def patch_antenna_cavity_model(
    W: float, L: float, h: float, eps_r: float,
) -> tuple:
    """Rectangular patch antenna cavity model predictions.

    Uses the Hammerstad effective permittivity and fringing extension
    to predict the TM010 resonant frequency, and the Balanis approximation
    for edge input resistance.

    Parameters
    ----------
    W : float
        Patch width (m) — the radiating dimension.
    L : float
        Patch length (m) — the resonant dimension.
    h : float
        Substrate height (m).
    eps_r : float
        Substrate relative permittivity.

    Returns
    -------
    f_res : float
        Resonant frequency of the TM010 mode (Hz).
    R_in_edge : float
        Edge input resistance at resonance (Ohm), approximate.
    eps_eff : float
        Effective permittivity used in the calculation.
    delta_L : float
        Fringing extension (m).

    References
    ----------
    C. A. Balanis, "Antenna Theory: Analysis and Design," 4th ed.,
    Wiley, 2016, Ch. 14.
    """
    from ..utils.constants import c0

    _, eps_eff = microstrip_z0_hammerstad(W, h, eps_r)

    # Fringing extension (Hammerstad)
    delta_L = 0.412 * h * (
        (eps_eff + 0.3) * (W / h + 0.264)
        / ((eps_eff - 0.258) * (W / h + 0.8))
    )

    # TM010 resonant frequency
    f_res = c0 / (2.0 * (L + 2.0 * delta_L) * np.sqrt(eps_eff))

    # Edge input resistance (Balanis approximation)
    R_in_edge = 90.0 * eps_r**2 / (eps_r - 1.0) * (L / W)**2

    return float(f_res), float(R_in_edge), float(eps_eff), float(delta_L)


def cpw_z0_conformal(
    W: float, S: float, eps_r: float, h: float,
) -> tuple:
    """CPW characteristic impedance via conformal mapping.

    Computes the quasi-static characteristic impedance and effective
    permittivity for a coplanar waveguide on a substrate of finite
    thickness h, backed by a PEC ground plane.

    Parameters
    ----------
    W : float
        Center conductor width (m).
    S : float
        Gap between center conductor and each ground plane (m).
    eps_r : float
        Substrate relative permittivity.
    h : float
        Substrate thickness (m).

    Returns
    -------
    Z0 : float
        Characteristic impedance (Ohm).
    eps_eff : float
        Effective permittivity.

    References
    ----------
    R. N. Simons, "Coplanar Waveguide Circuits, Components, and Systems,"
    Wiley, 2001, Ch. 2.
    """
    from scipy.special import ellipk

    # Air-space elliptic modulus
    k0 = W / (W + 2.0 * S)
    k0p = np.sqrt(1.0 - k0**2)

    # Substrate-modified elliptic modulus (finite thickness h + PEC backing)
    k1 = np.sinh(np.pi * W / (4.0 * h)) / np.sinh(
        np.pi * (W + 2.0 * S) / (4.0 * h)
    )
    k1p = np.sqrt(1.0 - k1**2)

    # Effective permittivity
    K_k0 = ellipk(k0**2)
    K_k0p = ellipk(k0p**2)
    K_k1 = ellipk(k1**2)
    K_k1p = ellipk(k1p**2)

    eps_eff = 1.0 + (eps_r - 1.0) / 2.0 * (K_k1 / K_k1p) * (K_k0p / K_k0)

    # Characteristic impedance
    Z0 = 30.0 * np.pi / np.sqrt(eps_eff) * (K_k0p / K_k0)

    return float(Z0), float(eps_eff)


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
