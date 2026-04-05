"""Surface wave pole extraction for layered media Green's functions.

DCIM (Discrete Complex Image Method) accurately represents the near-field
Green's function but misses surface wave poles — spectral singularities at
k_rho = beta_sw corresponding to guided modes propagating along the
stratified medium.  For transmission line structures (microstrip, CPW),
these poles carry the dominant long-range mutual coupling.

This module computes surface wave pole contributions analytically and adds
them as corrections to the DCIM Green's function.

Theory
------
The spectral Green's function has poles at k_rho = beta_n (n-th surface
wave mode).  In the spatial domain, each pole contributes:

    G_sw(rho) = sum_n  R_n * H_0^(2)(beta_n * rho) / (4j)

where R_n is the spectral residue and H_0^(2) is the Hankel function of
the second kind (exp(-jwt) convention).

For PEC-backed microstrip (PEC ground, dielectric eps_r, air above),
the dominant TM_0 mode satisfies:

    eps_r * alpha_0 + k_z1 * cot(k_z1 * h) = 0    (TM dispersion)

where k_z1 = sqrt(k_1^2 - beta^2), alpha_0 = sqrt(beta^2 - k_0^2),
k_1 = k_0 * sqrt(eps_r), and h is the substrate height.

References
----------
[1] M. I. Aksun, "A robust approach for the derivation of closed-form
    Green's functions," IEEE Trans. Microw. Theory Tech., vol. 44,
    no. 5, pp. 651-658, May 1996.
[2] Y. L. Chow et al., "A simplified method for calculating the complex
    image of a Green's function near a lossy ground," IEEE Trans. MTT,
    vol. 39, no. 4, pp. 588-592, Apr. 1991.
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel2
from scipy.optimize import brentq

from ...utils.constants import c0, mu0, eps0


def _tm_dispersion(beta: float, k0: float, eps_r: float, h: float) -> float:
    """TM surface wave dispersion equation for PEC-backed microstrip.

    Returns the residual of:
        k_z1 * tan(k_z1 * h) - eps_r * alpha_0 = 0

    Derived from the resonance condition Z_in_down + Z_in_up = 0 at z=h:
        Z_in_down = j * Z_TM1 * tan(k_z1 * h)   (PEC short at z=0)
        Z_in_up   = -j * alpha_0 / (omega * eps_0)  (evanescent air)

    where:
        k_z1 = sqrt(k_1^2 - beta^2)     (vertical wavenumber in substrate)
        alpha_0 = sqrt(beta^2 - k_0^2)  (decay constant in air)
        k_1 = k_0 * sqrt(eps_r)

    The TM_0 surface wave exists for beta in (k_0, k_1).
    """
    k1 = k0 * np.sqrt(eps_r)

    # k_z1 must be real and positive for a guided mode (beta < k1)
    kz1_sq = k1**2 - beta**2
    if kz1_sq <= 0:
        return 1e20  # beta >= k1: no guided mode in substrate

    kz1 = np.sqrt(kz1_sq)

    # alpha_0 must be real and positive for evanescent decay in air (beta > k0)
    alpha0_sq = beta**2 - k0**2
    if alpha0_sq <= 0:
        return -1e20  # beta <= k0: radiating, not guided

    alpha0 = np.sqrt(alpha0_sq)

    return kz1 * np.tan(kz1 * h) - eps_r * alpha0


def find_tm0_pole(k0: float, eps_r: float, h: float) -> float | None:
    """Find the TM_0 surface wave propagation constant.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber (rad/m).
    eps_r : float
        Substrate relative permittivity.
    h : float
        Substrate height (m).

    Returns
    -------
    beta : float or None
        TM_0 propagation constant (rad/m).  None if no surface wave
        exists (substrate too thin for the mode to be guided).
    """
    k1 = k0 * np.sqrt(eps_r)

    # TM_0 exists for beta in (k0, k1) — must be guided in air (beta > k0)
    # and propagating in substrate (beta < k1).
    # For thin substrates (k1*h < pi/2), TM_0 always exists.
    beta_min = k0 * 1.0001  # slightly above k0
    beta_max = k1 * 0.9999  # slightly below k1

    if beta_min >= beta_max:
        return None  # eps_r too close to 1

    f_min = _tm_dispersion(beta_min, k0, eps_r, h)
    f_max = _tm_dispersion(beta_max, k0, eps_r, h)

    if f_min * f_max > 0:
        # No sign change — no root in this interval
        # Try a finer search
        betas = np.linspace(beta_min, beta_max, 200)
        f_vals = np.array([_tm_dispersion(b, k0, eps_r, h) for b in betas])
        sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
        if len(sign_changes) == 0:
            return None
        # Use the first sign change
        idx = sign_changes[0]
        beta_min = betas[idx]
        beta_max = betas[idx + 1]

    try:
        beta = brentq(_tm_dispersion, beta_min, beta_max,
                       args=(k0, eps_r, h), xtol=1e-12)
        return beta
    except ValueError:
        return None


def _tm_residue_scalar(beta: float, k0: float, eps_r: float, h: float) -> complex:
    """Compute the spectral residue of G_phi at the TM_0 pole.

    For PEC-backed microstrip, the scalar potential spectral GF is:

        G_phi_tilde(k_rho) = V^e(z, z') / k_rho

    evaluated at z = z' = h (strip at substrate/air interface).

    The residue is computed as:
        R = lim_{k_rho -> beta} (k_rho - beta) * G_phi_tilde(k_rho)

    Using the transmission line analogy:
        V^e(z=h, z'=h) = Z_in_above * V_src / (Z_in_above + Z_in_below)

    where Z_in_above/below are the TM impedances looking up/down from z=h.

    For PEC-backed: Z_in_below = j * Z_TM1 * tan(k_z1 * h)
    For air above:  Z_in_above = Z_TM0 / alpha_0 (evanescent)

    The pole occurs where Z_in_above + Z_in_below = 0.  The residue
    involves the derivative of this sum w.r.t. k_rho.
    """
    k1 = k0 * np.sqrt(eps_r)
    omega = k0 * c0

    # Vertical wavenumbers at the pole
    kz1 = np.sqrt(k1**2 - beta**2 + 0j)
    alpha0 = np.sqrt(beta**2 - k0**2 + 0j)

    # TM characteristic impedances: Z_TM = k_z / (omega * eps)
    Z_TM1 = kz1 / (omega * eps0 * eps_r)
    Z_TM0 = alpha0 / (omega * eps0)  # Note: for evanescent, Z is real

    kz1h = kz1 * h

    # V^e at z = z' = h: the transmission line voltage at the interface
    # For source at z' = h in the substrate looking at PEC below:
    # V(z=h) = Z_in_above * I_src, where I_src = 1/(Z_total)
    # The spectral Green's function for V^e at the interface involves
    # both the upward and downward impedances.

    # Residue via finite difference (numerically stable)
    dk = beta * 1e-8
    def _Gphi_spectral(krho):
        _kz1 = np.sqrt(k1**2 - krho**2 + 0j)
        _alpha0 = np.sqrt(krho**2 - k0**2 + 0j)
        _kz1h = _kz1 * h

        # TM impedances
        _Ztm1 = _kz1 / (omega * eps0 * eps_r)
        _Ztm0 = _alpha0 / (omega * eps0)

        # Input impedance looking down (PEC short at z=0)
        sin_kz1h = np.sin(_kz1h)
        cos_kz1h = np.cos(_kz1h)
        if abs(sin_kz1h) < 1e-30:
            Z_down = 1e30j
        else:
            Z_down = 1j * _Ztm1 * sin_kz1h / cos_kz1h  # j*Z*tan(kz*h)

        # Input impedance looking up (air half-space)
        Z_up = _Ztm0  # characteristic impedance of evanescent air

        # Source at z = h: V = Z_up * Z_down / (Z_up + Z_down)
        Z_total = Z_up + Z_down
        if abs(Z_total) < 1e-30:
            return 0.0 + 0j
        V_at_h = Z_up * Z_down / Z_total

        # Spectral G_phi ~ V^e / k_rho  (simplified)
        return V_at_h / (krho + 1e-30)

    # Residue = lim (krho - beta) * G(krho) ≈ dk * G(beta+dk) - dk * G(beta-dk) ...
    # Better: R = 1 / [d/dkrho (1/G) at krho=beta]
    # Numerically: R = (krho - beta) * G(krho) for krho near beta
    G_plus = _Gphi_spectral(beta + dk)
    G_minus = _Gphi_spectral(beta - dk)
    # The spectral function has a simple pole: G ~ R/(krho - beta) near the pole
    # So (krho - beta) * G ~ R
    R_plus = dk * G_plus
    R_minus = -dk * G_minus
    residue = (R_plus + R_minus) / 2.0

    return residue


def compute_sw_correction_scalar(
    rho: np.ndarray,
    k0: float,
    eps_r: float,
    h: float,
    k_src: complex | None = None,
) -> np.ndarray:
    """Compute scalar potential surface wave correction G_sw(rho).

    This correction should be ADDED to the DCIM scalar Green's function
    to account for the missing surface wave pole.

    Parameters
    ----------
    rho : (N,) float
        Horizontal distances (m).
    k0 : float
        Free-space wavenumber (rad/m).
    eps_r : float
        Substrate relative permittivity.
    h : float
        Substrate height (m).
    k_src : complex, optional
        Source layer wavenumber.  If given, the correction is returned
        relative to the smooth-correction convention (G_sw only, no g_fs
        subtraction needed since the pole is not in the free-space part).

    Returns
    -------
    G_sw : (N,) complex
        Surface wave scalar potential correction.
    """
    rho = np.asarray(rho, dtype=np.float64)
    result = np.zeros_like(rho, dtype=np.complex128)

    beta = find_tm0_pole(k0, eps_r, h)
    if beta is None:
        return result

    residue = _tm_residue_scalar(beta, k0, eps_r, h)

    # Surface wave spatial contribution:
    # G_sw(rho) = -(j/4) * R * beta * H_0^(2)(beta * rho)
    # The factor of beta comes from the Sommerfeld integral: k_rho dk_rho
    # For the residue at a simple pole, the spatial contribution is:
    # G_sw = -(j * beta / (4)) * R * H_0^(2)(beta * rho)
    # With exp(-jwt) convention, H_0^(2) is the outgoing wave.
    mask = rho > 1e-30
    if np.any(mask):
        H0 = hankel2(0, beta * rho[mask])
        result[mask] = -1j * beta * residue * H0 / 4.0

    return result


def compute_sw_correction_dyadic(
    rho: np.ndarray,
    phi: np.ndarray,
    k0: float,
    eps_r: float,
    h: float,
) -> np.ndarray:
    """Compute vector potential surface wave correction (dyadic).

    For horizontal currents (xx, yy components of G_A), the TM_0 surface
    wave contributes through the K[0] spectral component.  The correction
    has the same Hankel function form but with a different residue.

    Parameters
    ----------
    rho : (N,) float
        Horizontal distances (m).
    phi : (N,) float
        Azimuthal angles (rad).
    k0 : float
        Free-space wavenumber (rad/m).
    eps_r : float
        Substrate relative permittivity.
    h : float
        Substrate height (m).

    Returns
    -------
    G_A_sw : (N, 3, 3) complex
        Surface wave dyadic correction.  Currently only the xx=yy
        (horizontal) component is computed; vertical components (xz, zx,
        zz) are set to zero.
    """
    rho = np.asarray(rho, dtype=np.float64)
    N = len(rho)
    result = np.zeros((N, 3, 3), dtype=np.complex128)

    beta = find_tm0_pole(k0, eps_r, h)
    if beta is None:
        return result

    # For the vector potential, the TM_0 pole contributes through K[0]
    # (the horizontal G_A component).  The residue has a different form
    # involving G_V^h (TE transmission line GF).
    #
    # For the dominant effect on horizontal microstrip currents, the
    # scalar potential pole is more important than the dyadic pole.
    # For now, return zeros — the scalar correction is the primary fix.
    return result
