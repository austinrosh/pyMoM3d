"""Analytical transmission line de-embedding.

Removes uniform feedline sections from measured S-parameters using the
ABCD cascade representation.  The feedline's Z0 and propagation constant
γ come from the 2D cross-section solver (or analytical formulas), so the
de-embedding is always stable — no quarter-wave singularities, no
re-solving the EM problem.

This replaces SOC for feedline removal.  SOC (Okhmatovski 2003) failed
for wideband use because the VSOC extraction hits singularities whenever
the feed section is an odd multiple of λ/4.  Analytical TL de-embedding
has no such instability.

Usage
-----
1. Solve the full structure (DUT + feedlines) with the 3D solver.
2. Construct a ``TLDeembedding`` object with the feedline Z0, γ, and
   physical lengths.
3. Call ``deembed(S_measured, freq)`` to remove the feedlines.

The de-embedding cascade is::

    T_DUT = T_feed1^{-1} × T_measured × T_feed2^{-1}

where T_feed is the ABCD matrix of a uniform TL section.

References
----------
Pozar, *Microwave Engineering*, 4th ed., Ch. 4.
Rautio, "An Ultra-High Precision Benchmark for Validation of Planar
Electromagnetic Analyses," IEEE Trans. MTT, 2004.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .soc_deembedding import abcd_to_s, s_to_abcd, invert_abcd


def tl_abcd(Z0: complex, gamma: complex, length: float) -> np.ndarray:
    """Compute the ABCD matrix of a uniform transmission line section.

    Parameters
    ----------
    Z0 : complex
        Characteristic impedance (Ω).
    gamma : complex
        Propagation constant (1/m).  For lossless TL: γ = jβ.
    length : float
        Physical length of the TL section (m).

    Returns
    -------
    T : ndarray, shape (2, 2), complex
        ABCD transmission matrix.
    """
    gl = gamma * length
    cosh_gl = np.cosh(gl)
    sinh_gl = np.sinh(gl)
    return np.array([
        [cosh_gl,       Z0 * sinh_gl],
        [sinh_gl / Z0,  cosh_gl      ],
    ], dtype=np.complex128)


@dataclass
class TLDeembedding:
    """Analytical transmission line de-embedding.

    Removes one or two uniform feedline sections from measured 2-port
    S-parameters using ABCD cascade math.

    Parameters
    ----------
    Z0 : complex or float
        Characteristic impedance of the feedline (Ω).  Can be
        frequency-dependent if computed per-frequency.
    gamma_func : callable
        Function ``gamma_func(freq) -> complex`` returning the
        propagation constant at a given frequency (Hz).  For lossless
        TL: ``lambda f: 1j * 2*pi*f * sqrt(L_pul * C_pul)``.
        Can also be a ``CrossSectionResult`` object whose ``.gamma(f)``
        method is used automatically.
    d1 : float
        Length of feedline at port 1 (m).
    d2 : float, optional
        Length of feedline at port 2 (m).  Default: same as d1
        (symmetric feeds).
    Z0_ref : float
        Reference impedance for S-parameter normalization (Ω).
        Default 50 Ω.
    """

    Z0: complex
    gamma_func: object  # callable or CrossSectionResult
    d1: float
    d2: Optional[float] = None
    Z0_ref: float = 50.0

    def __post_init__(self):
        if self.d2 is None:
            self.d2 = self.d1

    def _get_gamma(self, freq: float) -> complex:
        """Evaluate propagation constant at a frequency."""
        if hasattr(self.gamma_func, 'gamma'):
            # CrossSectionResult object
            return self.gamma_func.gamma(freq)
        return self.gamma_func(freq)

    def feed_abcd(self, freq: float, port: int = 1) -> np.ndarray:
        """Compute the ABCD matrix of a feedline section.

        Parameters
        ----------
        freq : float
            Frequency (Hz).
        port : int
            1 or 2 — which port's feedline.

        Returns
        -------
        T_feed : ndarray, shape (2, 2), complex
        """
        gamma = self._get_gamma(freq)
        d = self.d1 if port == 1 else self.d2
        return tl_abcd(self.Z0, gamma, d)

    def deembed(
        self,
        S_measured: np.ndarray,
        freq: float,
        port_cal=None,
    ) -> np.ndarray:
        """De-embed feedlines from measured 2-port S-parameters.

        Without port calibration, computes::

            T_DUT = T_feed1⁻¹ × T_measured × T_feed2⁻¹

        With port calibration, removes the port discontinuity first::

            T_DUT = T_feed1⁻¹ × T_err⁻¹ × T_measured × T_err⁻¹ × T_feed2⁻¹

        Parameters
        ----------
        S_measured : ndarray, shape (2, 2), complex
            Raw S-parameters including feedline effects.
        freq : float
            Frequency (Hz).
        port_cal : PortCalibration, optional
            Port discontinuity calibration.  If provided, the port error
            is removed before the feedline de-embedding.

        Returns
        -------
        S_deembedded : ndarray, shape (2, 2), complex
            S-parameters with feedlines (and optionally port error) removed.
        """
        T_meas = s_to_abcd(S_measured, self.Z0_ref)

        # Remove port error (outermost layer in the cascade)
        if port_cal is not None:
            T_meas = port_cal.T_err_inv @ T_meas @ port_cal.T_err_inv

        # Remove feedlines
        T_feed1_inv = invert_abcd(self.feed_abcd(freq, port=1))
        T_feed2_inv = invert_abcd(self.feed_abcd(freq, port=2))
        T_dut = T_feed1_inv @ T_meas @ T_feed2_inv
        return abcd_to_s(T_dut, self.Z0_ref)

    def deembed_sweep(
        self,
        S_list: list,
        frequencies: list,
        port_cals: list = None,
    ) -> list:
        """De-embed feedlines from a frequency sweep of S-parameters.

        Parameters
        ----------
        S_list : list of ndarray, shape (2, 2)
            Raw S-parameters at each frequency.
        frequencies : list of float
            Frequencies (Hz), same length as S_list.
        port_cals : list of PortCalibration, optional
            Per-frequency port calibrations.  If provided, must be same
            length as S_list.

        Returns
        -------
        S_deembedded : list of ndarray, shape (2, 2)
        """
        if port_cals is None:
            return [self.deembed(S, f) for S, f in zip(S_list, frequencies)]
        return [
            self.deembed(S, f, pc)
            for S, f, pc in zip(S_list, frequencies, port_cals)
        ]

    @classmethod
    def from_cross_section(
        cls,
        tl_result,
        d1: float,
        d2: float = None,
        Z0_ref: float = 50.0,
    ) -> 'TLDeembedding':
        """Create from a CrossSectionResult.

        Parameters
        ----------
        tl_result : CrossSectionResult
            From ``extract_tl_params()`` or ``compute_reference_impedance()``.
        d1 : float
            Feedline length at port 1 (m).
        d2 : float, optional
            Feedline length at port 2 (m).  Default: same as d1.
        Z0_ref : float
            Reference impedance (Ω).

        Returns
        -------
        TLDeembedding
        """
        return cls(
            Z0=tl_result.Z0,
            gamma_func=tl_result,
            d1=d1,
            d2=d2,
            Z0_ref=Z0_ref,
        )
