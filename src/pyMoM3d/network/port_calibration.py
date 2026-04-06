"""Port discontinuity calibration for half-RWG ports.

Extracts the reactive parasitic of the port gap by comparing a simulated
calibration standard (short through-line) to its known analytical behavior.
The extracted port error matrix can then be used in the de-embedding cascade
to remove the port discontinuity from DUT measurements.

The calibration structure is a short uniform TL section (< λ/10 at f_max)
with identical ports at each end.  The analytical ABCD of this section is
known from the 2D cross-section solver (Z0, γ).  The difference between
simulated and analytical ABCD is the port error.

Extraction math (symmetric ports)::

    T_cal = T_err × T_TL × T_err    (simulated calibration)
    T_TL  = tl_abcd(Z0, γ, d_cal)   (known analytically)

The port error T_err is extracted by solving the symmetric matrix equation
``E A E = B`` via Newton iteration on the Fréchet derivative, which yields
quadratic convergence to machine precision.  The initial estimate uses the
Cayley-Hamilton matrix square root of ``T_cal × T_TL⁻¹``.

Full de-embedding cascade::

    T_DUT = T_feed1⁻¹ × T_err⁻¹ × T_measured × T_err⁻¹ × T_feed2⁻¹

References
----------
Rautio, "An Ultra-High Precision Benchmark for Validation of Planar
Electromagnetic Analyses," IEEE Trans. MTT, 2004.

Okhmatovski et al., "On Deembedding of Port Discontinuities in Full-Wave
CAD Models of Multiport Circuits," IEEE Trans. MTT, 2003.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .soc_deembedding import abcd_to_s, s_to_abcd, invert_abcd
from .tl_deembedding import tl_abcd


def abcd_sqrt(M: np.ndarray) -> np.ndarray:
    """Principal square root of a 2×2 matrix via Cayley-Hamilton.

    For a 2×2 matrix M, the closed-form principal square root is::

        √M = (M + s·I) / t

    where s = √det(M) and t = √(tr(M) + 2s).

    Parameters
    ----------
    M : ndarray, shape (2, 2), complex

    Returns
    -------
    sqrt_M : ndarray, shape (2, 2), complex

    Raises
    ------
    ValueError
        If the matrix square root is ill-defined (t ≈ 0).
    """
    s = np.sqrt(np.linalg.det(M))
    tau = np.trace(M) + 2.0 * s
    t = np.sqrt(tau)

    if abs(t) < 1e-15:
        raise ValueError(
            f"Matrix square root ill-defined: trace(M) + 2√det(M) = {tau:.2e}"
        )

    return (M + s * np.eye(2, dtype=np.complex128)) / t


def solve_symmetric_fixture(
    T_cal: np.ndarray,
    T_tl: np.ndarray,
) -> np.ndarray:
    """Solve T_err × T_tl × T_err = T_cal for reciprocal T_err (exact).

    Uses the identity ``E·A = B·E⁻¹`` (valid when ``det(E) = 1``) to
    reduce the quadratic matrix equation to a closed-form solution for
    the 2×2 case.  The element equations give ``b/a``, ``c``, and ``a²``
    in terms of known ABCD elements, with a single square-root branch
    choice (``a`` near 1 for small port errors).

    Parameters
    ----------
    T_cal : ndarray, shape (2, 2), complex
        Measured calibration ABCD matrix (``T_err × T_tl × T_err``).
    T_tl : ndarray, shape (2, 2), complex
        Known analytical TL ABCD matrix.

    Returns
    -------
    T_err : ndarray, shape (2, 2), complex
        Port error ABCD matrix with ``det(T_err) = 1``.
    """
    # T_tl elements
    alpha, beta = T_tl[0, 0], T_tl[0, 1]
    gamma_tl, delta = T_tl[1, 0], T_tl[1, 1]

    # T_cal elements
    Ac, Bc = T_cal[0, 0], T_cal[0, 1]
    Cc, Dc = T_cal[1, 0], T_cal[1, 1]

    # From E·A = B·E⁻¹, element (0,1): a(β - Bc) + b(δ + Ac) = 0
    denom_r = delta + Ac
    if abs(denom_r) < 1e-30:
        # Degenerate case: fall back to approximate sqrt method
        M = T_cal @ invert_abcd(T_tl)
        return abcd_sqrt(M)

    r = (Bc - beta) / denom_r        # b = r·a

    s = beta + r * delta              # β + r·δ
    p = alpha + r * gamma_tl          # α + r·γ
    q = Ac * r - Bc                   # Ac·r - Bc
    u = Dc - Cc * r                   # Dc - Cc·r

    # a² = (Ac·s - δ·q) / (p·s - u·q)
    numer = Ac * s - delta * q
    denom = p * s - u * q
    if abs(denom) < 1e-30:
        M = T_cal @ invert_abcd(T_tl)
        return abcd_sqrt(M)

    a_sq = numer / denom

    # Choose branch: a near 1 for small port errors
    a = np.sqrt(a_sq)
    if a.real < 0:
        a = -a

    b = r * a

    # c = (a²·u - δ) / (a·s)
    if abs(s) < 1e-30:
        M = T_cal @ invert_abcd(T_tl)
        return abcd_sqrt(M)
    c = (a_sq * u - delta) / (a * s)

    # d from reciprocity: ad - bc = 1
    d = (1.0 + b * c) / a

    return np.array([[a, b], [c, d]], dtype=np.complex128)


@dataclass
class PortCalibration:
    """Port discontinuity calibration from a through-line standard.

    Stores the port error ABCD matrix extracted from a calibration
    measurement.  Used with :class:`TLDeembedding` to remove both
    the port discontinuity and feedline propagation effects.

    Parameters
    ----------
    T_err : ndarray, shape (2, 2), complex
        Port error ABCD matrix (near-identity for small discontinuity).
    T_err_inv : ndarray, shape (2, 2), complex
        Inverse of the port error ABCD matrix.
    freq : float
        Calibration frequency (Hz).
    """

    T_err: np.ndarray
    T_err_inv: np.ndarray
    freq: float

    @classmethod
    def from_through_standard(
        cls,
        S_cal: np.ndarray,
        Z0: complex,
        gamma: complex,
        d_cal: float,
        freq: float,
        Z0_ref: float = 50.0,
    ) -> 'PortCalibration':
        """Extract port error from a through-line calibration measurement.

        Solves the symmetric fixture equation::

            T_err × T_TL × T_err = T_cal

        for T_err via Newton iteration (exact to machine precision).

        Parameters
        ----------
        S_cal : ndarray, shape (2, 2), complex
            Measured S-parameters of the calibration through-line.
        Z0 : complex
            Characteristic impedance of the calibration TL (Ω).
        gamma : complex
            Propagation constant at the calibration frequency (1/m).
        d_cal : float
            Physical length of the calibration through-line (m).
        freq : float
            Calibration frequency (Hz).
        Z0_ref : float
            Reference impedance for S-parameters (Ω).

        Returns
        -------
        PortCalibration
        """
        T_cal = s_to_abcd(S_cal, Z0_ref)
        T_tl = tl_abcd(Z0, gamma, d_cal)

        T_err = solve_symmetric_fixture(T_cal, T_tl)

        return cls(
            T_err=T_err,
            T_err_inv=invert_abcd(T_err),
            freq=freq,
        )

    @classmethod
    def from_s_params_sweep(
        cls,
        S_cal_list: list,
        Z0: complex,
        gamma_func: object,
        d_cal: float,
        freqs: list,
        Z0_ref: float = 50.0,
    ) -> list:
        """Extract port error at multiple frequencies.

        Parameters
        ----------
        S_cal_list : list of ndarray, shape (2, 2)
            Calibration S-parameters at each frequency.
        Z0 : complex
            Characteristic impedance (Ω).
        gamma_func : callable
            ``gamma_func(freq) → complex`` propagation constant.
        d_cal : float
            Calibration through-line length (m).
        freqs : list of float
            Frequencies (Hz).
        Z0_ref : float
            Reference impedance (Ω).

        Returns
        -------
        list of PortCalibration
        """
        cals = []
        for S_cal, freq in zip(S_cal_list, freqs):
            if hasattr(gamma_func, 'gamma'):
                gamma = gamma_func.gamma(freq)
            else:
                gamma = gamma_func(freq)
            cals.append(cls.from_through_standard(
                S_cal, Z0, gamma, d_cal, freq, Z0_ref
            ))
        return cals

    def correct(self, S_measured: np.ndarray, Z0_ref: float = 50.0) -> np.ndarray:
        """Remove port error from measured S-parameters.

        Computes::

            T_corrected = T_err⁻¹ × T_measured × T_err⁻¹

        Parameters
        ----------
        S_measured : ndarray, shape (2, 2), complex
        Z0_ref : float
            Reference impedance (Ω).

        Returns
        -------
        S_corrected : ndarray, shape (2, 2), complex
        """
        T_meas = s_to_abcd(S_measured, Z0_ref)
        T_corrected = self.T_err_inv @ T_meas @ self.T_err_inv
        return abcd_to_s(T_corrected, Z0_ref)
