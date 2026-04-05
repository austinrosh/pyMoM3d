"""Transmission line characterization from 2-port network extraction.

Wraps the frequency sweep → Z0, eps_eff, alpha, beta extraction pipeline
with outlier rejection and quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..analysis.transmission_line import extract_z0_from_s, extract_propagation_constant
from ..utils.constants import c0


@dataclass
class TLResult:
    """Transmission line characterization results over frequency.

    Attributes
    ----------
    freqs : ndarray, shape (N,)
        Frequencies (Hz).
    Z0 : ndarray, shape (N,)
        Characteristic impedance magnitude (Ohm).
    eps_eff : ndarray, shape (N,)
        Effective permittivity from beta/k0.
    alpha : ndarray, shape (N,)
        Attenuation constant (Np/m).
    beta : ndarray, shape (N,)
        Phase constant (rad/m).
    S21_dB : ndarray, shape (N,)
        Insertion loss (dB).
    mean_Z0 : float
        Frequency-averaged Z0 (Ohm).
    mean_eps_eff : float
        Frequency-averaged eps_eff.
    Z0_std : float
        Standard deviation of Z0 (measures frequency independence).
    """

    freqs: np.ndarray
    Z0: np.ndarray
    eps_eff: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    S21_dB: np.ndarray
    mean_Z0: float
    mean_eps_eff: float
    Z0_std: float


class TLCharacterization:
    """Extract transmission line parameters from 2-port S-parameter sweep.

    Parameters
    ----------
    extractor : NetworkExtractor
        Configured 2-port network extractor.
    port_separation : float
        Physical distance between port 1 and port 2 (m).
    Z0_ref : float
        Reference impedance for S-parameter extraction (Ohm).
    """

    def __init__(self, extractor, port_separation: float, Z0_ref: float = 50.0):
        self.extractor = extractor
        self.port_separation = port_separation
        self.Z0_ref = Z0_ref

    def extract(self, freqs: List[float]) -> TLResult:
        """Run frequency sweep and extract TL parameters.

        Parameters
        ----------
        freqs : list of float
            Frequencies (Hz).

        Returns
        -------
        TLResult
        """
        results = self.extractor.extract(freqs)
        freqs_arr = np.array(freqs)

        Z0_arr = np.empty(len(freqs))
        eps_eff_arr = np.empty(len(freqs))
        alpha_arr = np.empty(len(freqs))
        beta_arr = np.empty(len(freqs))
        s21_dB_arr = np.empty(len(freqs))

        for i, (freq, result) in enumerate(zip(freqs, results)):
            S = result.S_matrix
            z0_ext = extract_z0_from_s(S, self.Z0_ref)
            gamma = extract_propagation_constant(S, self.port_separation, self.Z0_ref)
            k0 = 2.0 * np.pi * freq / c0

            Z0_arr[i] = abs(z0_ext)
            alpha_arr[i] = gamma.real
            beta_arr[i] = gamma.imag
            eps_eff_arr[i] = (gamma.imag / k0) ** 2 if k0 > 0 else np.nan
            s21_dB_arr[i] = 20.0 * np.log10(max(abs(S[1, 0]), 1e-12))

        # Filter outliers for mean calculation (reject Z0 values > 3 sigma)
        z0_valid = Z0_arr[np.isfinite(Z0_arr)]
        if len(z0_valid) > 2:
            med = np.median(z0_valid)
            mad = np.median(np.abs(z0_valid - med))
            mask = np.abs(Z0_arr - med) < 3 * max(mad, 1.0)
            z0_for_mean = Z0_arr[mask & np.isfinite(Z0_arr)]
            ee_for_mean = eps_eff_arr[mask & np.isfinite(eps_eff_arr)]
        else:
            z0_for_mean = z0_valid
            ee_for_mean = eps_eff_arr[np.isfinite(eps_eff_arr)]

        mean_z0 = float(np.mean(z0_for_mean)) if len(z0_for_mean) > 0 else np.nan
        mean_ee = float(np.mean(ee_for_mean)) if len(ee_for_mean) > 0 else np.nan
        z0_std = float(np.std(z0_for_mean)) if len(z0_for_mean) > 1 else np.nan

        return TLResult(
            freqs=freqs_arr,
            Z0=Z0_arr,
            eps_eff=eps_eff_arr,
            alpha=alpha_arr,
            beta=beta_arr,
            S21_dB=s21_dB_arr,
            mean_Z0=mean_z0,
            mean_eps_eff=mean_ee,
            Z0_std=z0_std,
        )
