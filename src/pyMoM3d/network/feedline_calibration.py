"""Feedline calibration for de-embedding port discontinuity effects.

In 3D planar MoM with layered Green's functions, the self-interaction
singularity (1/R) masks the dielectric-air interface effect, preventing
direct extraction of per-unit-length TL parameters.  This is a fundamental
property of 3D MoM, not a code bug.

The standard EDA approach (used by Keysight Momentum) is feedline
calibration: extend the DUT with feedlines of known Z₀ and γ, simulate
the full structure, then de-embed the feedline ABCD matrices analytically.

Usage
-----
1. Build the DUT mesh with feedline extensions at each port.
2. Simulate with ``NetworkExtractor`` to get raw 2-port S-parameters.
3. Create a ``FeedlineCalibration`` with the feedline TL parameters.
4. Call ``deembed`` to remove the feedline effects.

Example
-------
::

    from pyMoM3d.analysis.transmission_line import microstrip_z0_hammerstad
    from pyMoM3d.network.feedline_calibration import FeedlineCalibration

    Z0_cal, eps_eff_cal = microstrip_z0_hammerstad(W, H, eps_r)
    cal = FeedlineCalibration(Z0_cal, eps_eff_cal, L_feedline)
    S_dut = cal.deembed(S_raw, freq)

References
----------
[1] Keysight Momentum Theory of Operation, Section 5: Port Calibration.
"""

from __future__ import annotations

import numpy as np

from ..utils.constants import c0


def _tl_abcd(Z0: float, gamma: complex, L: float) -> np.ndarray:
    """ABCD matrix for a uniform transmission line.

    Parameters
    ----------
    Z0 : float
        Characteristic impedance (Ω).
    gamma : complex
        Propagation constant (α + jβ), units Np/m + rad/m.
    L : float
        Physical length (m).

    Returns
    -------
    ABCD : (2, 2) complex128
    """
    gL = gamma * L
    ch = np.cosh(gL)
    sh = np.sinh(gL)
    return np.array([
        [ch,       Z0 * sh],
        [sh / Z0,  ch     ],
    ], dtype=np.complex128)


def _abcd_to_s(ABCD: np.ndarray, Z0_ref: float = 50.0) -> np.ndarray:
    """Convert ABCD matrix to S-matrix.

    Parameters
    ----------
    ABCD : (2, 2) complex128
    Z0_ref : float
        Reference impedance for the S-parameters (Ω).

    Returns
    -------
    S : (2, 2) complex128
    """
    A, B, C, D = ABCD[0, 0], ABCD[0, 1], ABCD[1, 0], ABCD[1, 1]
    denom = A + B / Z0_ref + C * Z0_ref + D

    S11 = (A + B / Z0_ref - C * Z0_ref - D) / denom
    S12 = 2.0 * (A * D - B * C) / denom
    S21 = 2.0 / denom
    S22 = (-A + B / Z0_ref - C * Z0_ref + D) / denom

    return np.array([[S11, S12], [S21, S22]], dtype=np.complex128)


def _s_to_abcd(S: np.ndarray, Z0_ref: float = 50.0) -> np.ndarray:
    """Convert S-matrix to ABCD matrix."""
    S11, S12, S21, S22 = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
    denom = 2.0 * S21

    A = ((1.0 + S11) * (1.0 - S22) + S12 * S21) / denom
    B = Z0_ref * ((1.0 + S11) * (1.0 + S22) - S12 * S21) / denom
    C = (1.0 / Z0_ref) * ((1.0 - S11) * (1.0 - S22) - S12 * S21) / denom
    D = ((1.0 - S11) * (1.0 + S22) + S12 * S21) / denom

    return np.array([[A, B], [C, D]], dtype=np.complex128)


class FeedlineCalibration:
    """De-embed feedline effects from 2-port S-parameters.

    Parameters
    ----------
    Z0 : float
        Feedline characteristic impedance (Ω).
    eps_eff : float
        Feedline effective permittivity.
    feedline_length : float or tuple of float
        Physical length of each feedline extension (m).
        If scalar, same length for both ports.
        If tuple (L1, L2), different lengths for port 1 and port 2.
    alpha : float, optional
        Attenuation constant (Np/m).  Default 0 (lossless).
    Z0_ref : float, optional
        Reference impedance for S-parameter normalisation (Ω).  Default 50.
    """

    def __init__(
        self,
        Z0: float,
        eps_eff: float,
        feedline_length,
        alpha: float = 0.0,
        Z0_ref: float = 50.0,
    ):
        self.Z0 = float(Z0)
        self.eps_eff = float(eps_eff)
        self.alpha = float(alpha)
        self.Z0_ref = float(Z0_ref)

        if np.isscalar(feedline_length):
            self.L1 = self.L2 = float(feedline_length)
        else:
            self.L1, self.L2 = float(feedline_length[0]), float(feedline_length[1])

    def gamma(self, freq: float) -> complex:
        """Propagation constant at the given frequency.

        Parameters
        ----------
        freq : float
            Frequency (Hz).

        Returns
        -------
        gamma : complex
            α + jβ (Np/m + rad/m).
        """
        beta = 2.0 * np.pi * freq * np.sqrt(self.eps_eff) / c0
        return complex(self.alpha + 1j * beta)

    def deembed(self, S_raw: np.ndarray, freq: float) -> np.ndarray:
        """De-embed feedline effects from a 2-port S-matrix.

        Computes: ABCD_DUT = ABCD_line1_inv @ ABCD_raw @ ABCD_line2_inv

        Parameters
        ----------
        S_raw : (2, 2) complex128
            Raw (uncalibrated) 2-port S-matrix from MoM simulation.
        freq : float
            Frequency (Hz).

        Returns
        -------
        S_dut : (2, 2) complex128
            De-embedded S-matrix of the DUT.
        """
        g = self.gamma(freq)

        # Feedline ABCD matrices
        T1 = _tl_abcd(self.Z0, g, self.L1)
        T2 = _tl_abcd(self.Z0, g, self.L2)

        # Raw ABCD
        T_raw = _s_to_abcd(S_raw, self.Z0_ref)

        # De-embed: T_DUT = T1_inv @ T_raw @ T2_inv
        T1_inv = np.linalg.inv(T1)
        T2_inv = np.linalg.inv(T2)
        T_dut = T1_inv @ T_raw @ T2_inv

        return _abcd_to_s(T_dut, self.Z0_ref)

    def deembed_sweep(
        self, S_list, freqs
    ):
        """De-embed a frequency sweep of S-matrices.

        Parameters
        ----------
        S_list : list of (2, 2) complex128
            Raw S-matrices at each frequency.
        freqs : array-like
            Frequencies (Hz), same length as S_list.

        Returns
        -------
        S_dut_list : list of (2, 2) complex128
            De-embedded S-matrices.
        """
        return [self.deembed(S, f) for S, f in zip(S_list, freqs)]
