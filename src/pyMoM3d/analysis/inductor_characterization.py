"""EDA-grade inductor parameter extraction from MoM simulation results.

Wraps a frequency sweep of ``NetworkResult`` objects and provides
Y-parameter based L/Q/R extraction, self-resonant frequency detection,
and optional broadband pi-model fitting.

This follows the standard methodology used in commercial EM solvers
(EMX, Momentum, HFSS, Sonnet) for on-chip inductor characterization.

Typical usage
-------------
>>> results = extractor.extract(frequencies)
>>> char = InductorCharacterization(results)
>>> cr = char.characterize(fit_model=True)
>>> print(f"L_dc = {cr.L_dc * 1e9:.2f} nH, SRF = {cr.srf / 1e9:.2f} GHz")
>>> print(f"Q_peak = {cr.Q_peak:.1f} at {cr.f_Q_peak / 1e9:.2f} GHz")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from ..network.network_result import NetworkResult
from .inductor import (
    inductance_from_y,
    inductance_from_z,
    quality_factor,
    quality_factor_y,
    resistance_from_y,
    self_resonant_frequency,
    wheeler_inductance,
)
from .inductor_model import PiModelParams, fit_pi_model


@dataclass
class InductorParams:
    """Extracted inductor parameters at a single frequency point.

    Attributes
    ----------
    frequency : float
        Frequency (Hz).
    L : float
        Inductance from Y-parameter (H).
    Q : float
        Quality factor from Y-parameter.
    R : float
        Series resistance from Y-parameter (Ohm).
    L_z : float
        Inductance from Z-parameter (H), for comparison.
    Q_z : float
        Quality factor from Z-parameter, for comparison.
    Z11 : complex
        Raw impedance (Ohm).
    Y11 : complex
        Raw admittance (S).
    S11 : complex
        Reflection coefficient.
    """

    frequency: float
    L: float
    Q: float
    R: float
    L_z: float
    Q_z: float
    Z11: complex
    Y11: complex
    S11: complex


@dataclass
class CharacterizationResult:
    """Complete inductor characterization across frequency.

    Attributes
    ----------
    frequencies : ndarray, shape (N,)
        Frequency points (Hz).
    params : list of InductorParams
        Per-frequency extracted parameters.
    srf : float
        Self-resonant frequency (Hz).  ``inf`` if not in sweep range.
    srf_index : int
        Index of last inductive sample before SRF.  ``-1`` if not found.
    L_dc : float
        Low-frequency inductance estimate (H).
    Q_peak : float
        Peak quality factor.
    f_Q_peak : float
        Frequency of peak Q (Hz).
    pi_model : PiModelParams or None
        Fitted broadband model (if requested).
    pi_model_fit_info : dict or None
        Fitting diagnostics (if model was fitted).
    """

    frequencies: np.ndarray
    params: List[InductorParams]
    srf: float
    srf_index: int
    L_dc: float
    Q_peak: float
    f_Q_peak: float
    pi_model: Optional[PiModelParams] = None
    pi_model_fit_info: Optional[dict] = None

    @property
    def L(self) -> np.ndarray:
        """Inductance array from Y-parameter extraction (H)."""
        return np.array([p.L for p in self.params])

    @property
    def Q(self) -> np.ndarray:
        """Quality factor array from Y-parameter extraction."""
        return np.array([p.Q for p in self.params])

    @property
    def R(self) -> np.ndarray:
        """Series resistance array (Ohm)."""
        return np.array([p.R for p in self.params])

    @property
    def L_z(self) -> np.ndarray:
        """Inductance array from Z-parameter extraction (H), for comparison."""
        return np.array([p.L_z for p in self.params])

    @property
    def Q_z(self) -> np.ndarray:
        """Quality factor array from Z-parameter extraction, for comparison."""
        return np.array([p.Q_z for p in self.params])

    @property
    def Z11(self) -> np.ndarray:
        """Raw impedance array (Ohm)."""
        return np.array([p.Z11 for p in self.params])

    @property
    def Y11(self) -> np.ndarray:
        """Raw admittance array (S)."""
        return np.array([p.Y11 for p in self.params])

    @property
    def S11(self) -> np.ndarray:
        """Reflection coefficient array."""
        return np.array([p.S11 for p in self.params])

    def summary(self) -> str:
        """Human-readable characterization summary."""
        lines = [
            "Inductor Characterization Summary",
            "=" * 40,
            f"  Frequency range: {self.frequencies[0]/1e9:.2f} - {self.frequencies[-1]/1e9:.2f} GHz",
            f"  L_dc:            {self.L_dc * 1e9:.3f} nH",
            f"  SRF:             {self.srf / 1e9:.2f} GHz" if np.isfinite(self.srf) else "  SRF:             > sweep range",
            f"  Q_peak:          {self.Q_peak:.1f} at {self.f_Q_peak / 1e9:.2f} GHz",
        ]
        if self.pi_model is not None:
            lines.append("")
            lines.append(self.pi_model.summary())
        return "\n".join(lines)


class InductorCharacterization:
    """EDA-grade inductor parameter extraction from MoM results.

    Parameters
    ----------
    results : list of NetworkResult
        Frequency sweep results from ``NetworkExtractor.extract()``.
        Must be sorted by frequency.
    port_index : int
        Port index for 1-port extraction (default 0).
    mode : str
        ``'1port'`` (default).  ``'differential'`` for 2-port.
    """

    def __init__(
        self,
        results: List[NetworkResult],
        port_index: int = 0,
        mode: str = '1port',
    ):
        if not results:
            raise ValueError("results list must not be empty")
        self.results = sorted(results, key=lambda r: r.frequency)
        self.port_index = port_index
        self.mode = mode

        self.frequencies = np.array([r.frequency for r in self.results])
        self._extract_raw_params()

    def _extract_raw_params(self) -> None:
        """Extract Z11, Y11, S11 arrays from NetworkResult list."""
        p = self.port_index
        self._Z11 = np.array([r.Z_matrix[p, p] for r in self.results])
        self._Y11 = np.array([r.Y_matrix[p, p] for r in self.results])
        self._S11 = np.array([r.S_matrix[p, p] for r in self.results])

    def characterize(
        self,
        fit_model: bool = False,
        model_kwargs: Optional[dict] = None,
    ) -> CharacterizationResult:
        """Run full inductor characterization.

        1. Y-parameter based L(f), Q(f), R(f) extraction
        2. SRF detection
        3. Q peak identification
        4. Optionally fit broadband pi-model

        Parameters
        ----------
        fit_model : bool
            If True, fit a broadband pi-model to the Y-parameter data.
        model_kwargs : dict, optional
            Additional keyword arguments for ``fit_pi_model()``.

        Returns
        -------
        CharacterizationResult
        """
        params = []
        for i, freq in enumerate(self.frequencies):
            z11 = self._Z11[i]
            y11 = self._Y11[i]
            s11 = self._S11[i]

            params.append(InductorParams(
                frequency=freq,
                L=inductance_from_y(y11, freq),
                Q=quality_factor_y(y11),
                R=resistance_from_y(y11),
                L_z=inductance_from_z(z11, freq),
                Q_z=quality_factor(z11),
                Z11=z11,
                Y11=y11,
                S11=s11,
            ))

        # SRF detection
        srf, srf_idx = self_resonant_frequency(self.frequencies, self._Y11)

        # Low-frequency L estimate (first inductive point)
        L_vals = np.array([p.L for p in params])
        inductive = L_vals > 0
        L_dc = float(L_vals[inductive][0]) if np.any(inductive) else 0.0

        # Q peak
        Q_vals = np.array([p.Q for p in params])
        # Only consider points below SRF for Q peak
        if srf_idx >= 0:
            Q_below_srf = Q_vals[:srf_idx + 1]
        else:
            Q_below_srf = Q_vals
        if len(Q_below_srf) > 0:
            q_peak_idx = int(np.argmax(Q_below_srf))
            Q_peak = float(Q_below_srf[q_peak_idx])
            f_Q_peak = float(self.frequencies[q_peak_idx])
        else:
            Q_peak = 0.0
            f_Q_peak = 0.0

        # Optional pi-model fitting
        pi_model = None
        fit_info = None
        if fit_model:
            kw = model_kwargs or {}
            pi_model, fit_info = fit_pi_model(
                self.frequencies, self._Y11, mode=self.mode, **kw,
            )

        return CharacterizationResult(
            frequencies=self.frequencies,
            params=params,
            srf=srf,
            srf_index=srf_idx,
            L_dc=L_dc,
            Q_peak=Q_peak,
            f_Q_peak=f_Q_peak,
            pi_model=pi_model,
            pi_model_fit_info=fit_info,
        )

    def compare_with_wheeler(
        self,
        n: float,
        d_out: float,
        d_in: float,
    ) -> dict:
        """Compare extracted inductance with the Wheeler formula.

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
        dict
            ``'wheeler_L'``: analytical inductance (H).
            ``'L_dc'``: extracted low-frequency L (H).
            ``'error_pct'``: percentage error of L_dc vs Wheeler.
            ``'L_y_array'``: L(f) from Y extraction (H).
            ``'L_z_array'``: L(f) from Z extraction (H).
        """
        L_wheeler = wheeler_inductance(n, d_out, d_in)

        # Need characterization result for arrays
        cr = self.characterize(fit_model=False)

        error_pct = abs(cr.L_dc - L_wheeler) / L_wheeler * 100 if L_wheeler > 0 else float('inf')

        return {
            'wheeler_L': L_wheeler,
            'L_dc': cr.L_dc,
            'error_pct': error_pct,
            'L_y_array': cr.L,
            'L_z_array': cr.L_z,
        }
