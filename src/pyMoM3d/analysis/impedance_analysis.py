"""Input impedance and S-parameter analysis."""

import numpy as np
from typing import List

from ..simulation import SimulationResult


def compute_s11(Z_in: complex, Z0: float = 50.0) -> complex:
    """Compute S11 reflection coefficient.

    Parameters
    ----------
    Z_in : complex
        Input impedance (Ohms).
    Z0 : float
        Reference impedance (Ohms).

    Returns
    -------
    S11 : complex
    """
    return (Z_in - Z0) / (Z_in + Z0)


def impedance_vs_frequency(results: List[SimulationResult]) -> tuple:
    """Extract input impedance vs frequency from sweep results.

    Parameters
    ----------
    results : list of SimulationResult
        From Simulation.sweep().

    Returns
    -------
    frequencies : ndarray
    Z_in : ndarray, complex128
    """
    freqs = np.array([r.frequency for r in results])
    Z_in = np.array([r.Z_input if r.Z_input is not None else np.nan + 0j
                      for r in results], dtype=np.complex128)
    return freqs, Z_in


def s11_vs_frequency(results: List[SimulationResult], Z0: float = 50.0) -> tuple:
    """Compute S11 vs frequency.

    Returns
    -------
    frequencies : ndarray
    s11_dB : ndarray
    """
    freqs, Z_in = impedance_vs_frequency(results)
    s11 = np.array([compute_s11(z, Z0) if np.isfinite(z) else np.nan + 0j
                     for z in Z_in])
    s11_dB = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-30))
    return freqs, s11_dB
