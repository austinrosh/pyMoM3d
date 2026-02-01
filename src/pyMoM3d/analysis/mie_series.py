"""PEC sphere Mie series for RCS validation.

Computes the exact bistatic and monostatic RCS of a perfectly
conducting sphere using the Mie series expansion.

References
----------
- Harrington, "Time-Harmonic Electromagnetic Fields", Ch. 6
- Balanis, "Advanced Engineering Electromagnetics", Ch. 11
- Bohren & Huffman, "Absorption and Scattering of Light by Small Particles"
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def _spherical_hn2(n, z, derivative=False):
    """Spherical Hankel function of the second kind h_n^(2)(z) = j_n - j*y_n.

    Convention: exp(-j*omega*t) time dependence uses h^(2).
    """
    return spherical_jn(n, z, derivative=derivative) - 1j * spherical_yn(n, z, derivative=derivative)


def mie_rcs_pec_sphere(
    ka: float,
    theta: np.ndarray,
    n_max: int = None,
) -> np.ndarray:
    """Compute bistatic RCS of a PEC sphere normalized by pi*a^2.

    For theta-polarized incident plane wave propagating in -z direction.

    Parameters
    ----------
    ka : float
        Electrical size (k * radius).
    theta : ndarray, shape (M,)
        Bistatic angles (radians). 0 = forward scatter, pi = backscatter.
    n_max : int, optional
        Number of series terms. Default: ceil(ka) + 10.

    Returns
    -------
    rcs_norm : ndarray, shape (M,)
        sigma / (pi * a^2).
    """
    theta = np.asarray(theta, dtype=np.float64)
    if n_max is None:
        n_max = max(int(np.ceil(ka)) + 10, 15)

    M = len(theta)
    cos_theta = np.cos(theta)

    # Scattering amplitudes S1 and S2
    # S1 = sum_n (2n+1)/(n(n+1)) * [a_n * pi_n + b_n * tau_n]
    # S2 = sum_n (2n+1)/(n(n+1)) * [a_n * tau_n + b_n * pi_n]
    #
    # where a_n, b_n are Mie coefficients for PEC sphere:
    #   a_n = j_n(ka) / h_n^(2)(ka)
    #   b_n = [ka*j_n(ka)]' / [ka*h_n^(2)(ka)]'
    #
    # pi_n and tau_n are angular functions from associated Legendre P_n^1.

    S1 = np.zeros(M, dtype=np.complex128)
    S2 = np.zeros(M, dtype=np.complex128)

    # pi_n and tau_n recurrence
    pi_nm1 = np.zeros(M)  # pi_{n-1}, starts as pi_0 = 0
    pi_n = np.ones(M)     # pi_1 = 1

    for n in range(1, n_max + 1):
        # Mie coefficients
        jn = spherical_jn(n, ka)
        jn_d = spherical_jn(n, ka, derivative=True)
        hn = _spherical_hn2(n, ka)
        hn_d = _spherical_hn2(n, ka, derivative=True)

        # a_n = j_n(ka) / h_n^(2)(ka)
        a_n = jn / hn

        # b_n = [d/dx (x*j_n(x))]_{x=ka} / [d/dx (x*h_n^(2)(x))]_{x=ka}
        # d/dx [x*f(x)] = f(x) + x*f'(x)
        b_n = (jn + ka * jn_d) / (hn + ka * hn_d)

        # tau_n = n*cos(theta)*pi_n - (n+1)*pi_{n-1}
        tau_n = n * cos_theta * pi_n - (n + 1) * pi_nm1

        pf = (2.0 * n + 1.0) / (n * (n + 1.0))

        S1 += pf * (a_n * pi_n + b_n * tau_n)
        S2 += pf * (a_n * tau_n + b_n * pi_n)

        # Recurrence for pi_{n+1}
        pi_next = ((2 * n + 1) * cos_theta * pi_n - (n + 1) * pi_nm1) / n
        pi_nm1 = pi_n
        pi_n = pi_next

    # Bistatic RCS: sigma = (lambda^2 / pi) * |S|^2
    # sigma / (pi*a^2) = (lambda^2 / (pi^2 * a^2)) * |S|^2
    #                   = (4 / (ka)^2) * |S|^2   since lambda = 2*pi/k, a = ka/k
    # Wait: lambda^2/(pi * pi * a^2) = (2*pi/k)^2 / (pi^2 * (ka/k)^2)
    #      = 4*pi^2/k^2 / (pi^2 * ka^2/k^2) = 4/ka^2

    # For theta-polarized incidence:
    # sigma_theta = (lambda^2/pi) * |S1(theta)|^2
    # sigma/(pi*a^2) = 4/ka^2 * |S1|^2

    # Actually the standard definition:
    # sigma = lim_{r->inf} 4*pi*r^2 * |E_s|^2 / |E_i|^2
    # For the Mie series: sigma = (lambda^2/pi) * |S|^2
    # sigma/(pi*a^2) = lambda^2/(pi^2*a^2) * |S|^2
    # = (2*pi/k)^2 / (pi^2 * a^2) * |S|^2
    # = 4/(k^2 * a^2) * |S|^2
    # = 4/ka^2 * |S|^2

    rcs_norm = (4.0 / ka**2) * np.abs(S1)**2

    return rcs_norm


def mie_monostatic_rcs_pec_sphere(ka: float, n_max: int = None) -> float:
    """Monostatic (backscatter) RCS of PEC sphere, normalized by pi*a^2.

    Uses the specialized backscatter formula:
    sigma/(pi*a^2) = (1/ka^2) * |sum_n (-1)^n (2n+1)(a_n - b_n)|^2
    """
    if n_max is None:
        n_max = max(int(np.ceil(ka)) + 10, 15)

    S_back = 0.0 + 0.0j

    for n in range(1, n_max + 1):
        jn = spherical_jn(n, ka)
        jn_d = spherical_jn(n, ka, derivative=True)
        hn = _spherical_hn2(n, ka)
        hn_d = _spherical_hn2(n, ka, derivative=True)

        a_n = jn / hn
        b_n = (jn + ka * jn_d) / (hn + ka * hn_d)

        S_back += (-1.0)**n * (2 * n + 1) * (a_n - b_n)

    # sigma/(pi*a^2) = (1/ka^2) * |S_back|^2
    rcs_norm = (1.0 / ka**2) * np.abs(S_back)**2

    return float(rcs_norm)
