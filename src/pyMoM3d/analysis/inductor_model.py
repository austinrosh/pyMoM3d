"""Broadband pi-model for on-chip inductor characterization.

Implements the standard symmetric pi equivalent circuit used in EDA tools
for SPICE-compatible inductor modeling.  The topology follows Cao et al.
and Niknejad & Meyer:

.. code-block:: text

    Port 1 ----[ R_s(f) + jωL_s ]---- Port 2
                   |--- C_s ---|
           C_ox                    C_ox
           R_sub || C_sub          R_sub || C_sub
           GND                    GND

For 1-port (single-ended) characterization, port 2 is grounded and the
circuit reduces to one side of the pi.

References
----------
* Y. Cao et al., "Frequency-independent equivalent-circuit model for
  on-chip spiral inductors," IEEE JSSC, vol. 38, no. 3, 2003.
* A. M. Niknejad and R. G. Meyer, "Analysis, design, and optimization
  of spiral inductors and transformers for Si RF ICs," IEEE JSSC,
  vol. 33, no. 10, 1998.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
from scipy.optimize import least_squares


@dataclass
class PiModelParams:
    """Broadband pi-model parameters for an on-chip inductor.

    Parameters
    ----------
    L_s : float
        Series inductance (H).
    R_s : float
        DC series resistance (Ohm).
    R_skin : float
        Skin-effect resistance coefficient.
        Frequency-dependent resistance: R(f) = R_s + R_skin * sqrt(f).
    C_s : float
        Inter-turn (series) capacitance (F).
    C_ox : float
        Oxide capacitance per side (F).
    R_sub : float
        Substrate resistance per side (Ohm).
    C_sub : float
        Substrate capacitance per side (F).
    """

    L_s: float
    R_s: float
    R_skin: float
    C_s: float
    C_ox: float
    R_sub: float
    C_sub: float

    def to_vector(self) -> np.ndarray:
        """Pack parameters into a 1-D array for optimization."""
        return np.array([
            self.L_s, self.R_s, self.R_skin,
            self.C_s, self.C_ox, self.R_sub, self.C_sub,
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> 'PiModelParams':
        """Unpack a 1-D array into PiModelParams."""
        return cls(
            L_s=float(v[0]), R_s=float(v[1]), R_skin=float(v[2]),
            C_s=float(v[3]), C_ox=float(v[4]),
            R_sub=float(v[5]), C_sub=float(v[6]),
        )

    def _series_impedance(self, omega: float) -> complex:
        """Series branch impedance: R(f) + jωL_s in parallel with 1/(jωC_s)."""
        freq = omega / (2.0 * np.pi)
        R_f = self.R_s + self.R_skin * np.sqrt(max(freq, 0.0))
        Z_rl = R_f + 1j * omega * self.L_s
        if self.C_s > 0 and omega > 0:
            Z_cs = 1.0 / (1j * omega * self.C_s)
            # Parallel combination
            return (Z_rl * Z_cs) / (Z_rl + Z_cs)
        return Z_rl

    def _shunt_admittance(self, omega: float) -> complex:
        """Shunt branch admittance (one side): C_ox in series with (R_sub || C_sub)."""
        if omega < 1e-30:
            return 0.0 + 0.0j
        Y_cox = 1j * omega * self.C_ox
        if self.R_sub > 0 or self.C_sub > 0:
            Y_sub = 1.0 / self.R_sub if self.R_sub > 0 else 0.0
            Y_sub += 1j * omega * self.C_sub
            Z_sub = 1.0 / Y_sub if abs(Y_sub) > 1e-30 else 1e30
            Z_total = 1.0 / Y_cox + Z_sub if abs(Y_cox) > 1e-30 else Z_sub
            return 1.0 / Z_total if abs(Z_total) > 1e-30 else 0.0 + 0.0j
        return Y_cox

    def Y_model_1port(self, freq: float) -> complex:
        """Compute 1-port Y11 with port 2 grounded.

        When port 2 is shorted to ground (V2 = 0), the port 2 shunt
        branch carries no current and the input admittance is simply:

            Y_in = Y_shunt1 + 1/Z_series

        Parameters
        ----------
        freq : float
            Frequency (Hz).

        Returns
        -------
        Y11 : complex
            1-port admittance (S).
        """
        omega = 2.0 * np.pi * freq
        Z_s = self._series_impedance(omega)
        Y_sh = self._shunt_admittance(omega)

        Y_series = 1.0 / Z_s if abs(Z_s) > 1e-30 else 1e30 + 0j
        return Y_sh + Y_series

    def Y_model_1port_array(self, frequencies: np.ndarray) -> np.ndarray:
        """Evaluate 1-port model at multiple frequencies.

        Parameters
        ----------
        frequencies : ndarray, shape (N,)
            Frequencies (Hz).

        Returns
        -------
        Y11 : ndarray, shape (N,), complex128
        """
        return np.array([self.Y_model_1port(f) for f in frequencies])

    def Y_model_2port(self, freq: float) -> np.ndarray:
        """Compute 2-port Y-matrix (symmetric pi).

        Parameters
        ----------
        freq : float
            Frequency (Hz).

        Returns
        -------
        Y : ndarray, shape (2, 2), complex128
        """
        omega = 2.0 * np.pi * freq
        Z_s = self._series_impedance(omega)
        Y_sh = self._shunt_admittance(omega)
        Y_s = 1.0 / Z_s if abs(Z_s) > 1e-30 else 1e30 + 0j

        Y = np.zeros((2, 2), dtype=np.complex128)
        Y[0, 0] = Y_s + Y_sh
        Y[1, 1] = Y_s + Y_sh
        Y[0, 1] = -Y_s
        Y[1, 0] = -Y_s
        return Y

    def summary(self) -> str:
        """Human-readable parameter summary."""
        lines = [
            "Pi-Model Parameters:",
            f"  L_s    = {self.L_s * 1e9:.3f} nH",
            f"  R_s    = {self.R_s:.3f} Ohm",
            f"  R_skin = {self.R_skin:.4e} Ohm/sqrt(Hz)",
            f"  C_s    = {self.C_s * 1e15:.2f} fF",
            f"  C_ox   = {self.C_ox * 1e15:.2f} fF",
            f"  R_sub  = {self.R_sub:.1f} Ohm",
            f"  C_sub  = {self.C_sub * 1e15:.2f} fF",
        ]
        return "\n".join(lines)


def estimate_initial_params(
    frequencies: np.ndarray,
    Y_data: np.ndarray,
    mode: str = '1port',
) -> PiModelParams:
    """Estimate initial pi-model parameters from Y-parameter data.

    Strategy:
    - L_s from low-frequency Im(Y11)
    - R_s from low-frequency Re(1/Y11)
    - C_s from SRF (if detectable) or small default
    - C_ox from high-frequency capacitive slope
    - R_sub, C_sub from reasonable defaults

    Parameters
    ----------
    frequencies : ndarray, shape (N,)
        Frequency points (Hz).
    Y_data : ndarray, shape (N,) or (N, 2, 2)
        Measured Y-parameter data.
    mode : str
        ``'1port'`` or ``'2port'``.

    Returns
    -------
    PiModelParams
        Initial parameter estimate.
    """
    from .inductor import self_resonant_frequency

    if mode == '2port':
        Y11 = Y_data[:, 0, 0]
    else:
        Y11 = np.asarray(Y_data)

    omega = 2.0 * np.pi * frequencies

    # L from lowest-frequency point where inductive
    inductive_mask = Y11.imag < 0
    if np.any(inductive_mask):
        idx_low = np.where(inductive_mask)[0][0]
        L_est = -1.0 / (omega[idx_low] * Y11[idx_low].imag)
        L_est = max(L_est, 1e-15)  # floor at ~1 fH
    else:
        L_est = 1e-9  # 1 nH default

    # R from low-frequency Re(1/Y11)
    idx_low = 0
    if abs(Y11[idx_low]) > 1e-30:
        R_est = (1.0 / Y11[idx_low]).real
        R_est = max(R_est, 0.01)
    else:
        R_est = 1.0

    # SRF and C_s
    f_srf, srf_idx = self_resonant_frequency(frequencies, Y11)
    if np.isfinite(f_srf) and f_srf > 0:
        omega_srf = 2.0 * np.pi * f_srf
        C_s_est = 1.0 / (omega_srf**2 * L_est)
    else:
        C_s_est = 1e-15  # 1 fF default

    # C_ox: estimate from high-frequency capacitive behavior
    # If above SRF, Im(Y11) > 0 and slope ~ omega*C_ox
    if srf_idx >= 0 and srf_idx + 2 < len(frequencies):
        idx_hf = min(srf_idx + 2, len(frequencies) - 1)
        C_ox_est = Y11[idx_hf].imag / omega[idx_hf] if omega[idx_hf] > 0 else 1e-15
        C_ox_est = max(C_ox_est, 1e-16)
    else:
        C_ox_est = 1e-14  # 10 fF default

    # Substrate: reasonable defaults for silicon
    R_sub_est = 200.0
    C_sub_est = 1e-14  # 10 fF

    return PiModelParams(
        L_s=L_est,
        R_s=R_est,
        R_skin=0.0,
        C_s=C_s_est,
        C_ox=C_ox_est,
        R_sub=R_sub_est,
        C_sub=C_sub_est,
    )


def _residual_1port_normalized(
    x_norm: np.ndarray,
    x_scale: np.ndarray,
    frequencies: np.ndarray,
    Y_data: np.ndarray,
    Y_scale: float,
) -> np.ndarray:
    """Residual vector for 1-port fitting with normalized parameters.

    Parameters are normalized: ``params = x_norm * x_scale``, so the
    optimizer works with O(1) variables regardless of physical units.
    """
    params_vec = x_norm * x_scale
    model = PiModelParams.from_vector(params_vec)
    Y_model = model.Y_model_1port_array(frequencies)
    diff = (Y_model - Y_data) / Y_scale
    return np.concatenate([diff.real, diff.imag])


def fit_pi_model(
    frequencies: np.ndarray,
    Y_data: np.ndarray,
    mode: str = '1port',
    initial_guess: Optional[PiModelParams] = None,
    max_iterations: int = 2000,
) -> tuple:
    """Fit broadband pi-model to Y-parameter data.

    Uses ``scipy.optimize.least_squares`` with trust-region reflective
    algorithm.  Parameters are internally normalized to O(1) for
    robust convergence across the 15-order-of-magnitude range of
    physical parameters (nH, fF, Ohm).

    Parameters
    ----------
    frequencies : ndarray, shape (N,)
        Frequency points (Hz).
    Y_data : ndarray, shape (N,) for 1-port, (N, 2, 2) for 2-port
        Measured/simulated Y-parameter data.
    mode : str
        ``'1port'`` (default) or ``'2port'``.
    initial_guess : PiModelParams, optional
        Initial parameter estimate.  If None, ``estimate_initial_params``
        is used.
    max_iterations : int
        Maximum function evaluations for the optimizer.

    Returns
    -------
    params : PiModelParams
        Fitted model parameters.
    fit_info : dict
        ``'residual_norm'``: final residual L2 norm.
        ``'Y_fitted'``: model Y at each frequency, shape matching Y_data.
        ``'cost'``: final cost value.
        ``'success'``: bool, whether optimizer converged.
        ``'message'``: optimizer status message.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)

    if initial_guess is None:
        initial_guess = estimate_initial_params(frequencies, Y_data, mode)

    x0 = initial_guess.to_vector()

    # Parameter bounds (physical)
    lower_phys = np.array([1e-15, 1e-4, 0.0, 1e-18, 1e-18, 1.0, 1e-18])
    upper_phys = np.array([1e-6, 1e4, 1e-1, 1e-10, 1e-10, 1e5, 1e-10])

    # Clamp initial guess within bounds
    x0 = np.clip(x0, lower_phys * 1.01, upper_phys * 0.99)

    # Normalize: x_norm = x / x_scale, so x_norm ~ O(1)
    x_scale = x0.copy()
    # For zero-valued params (e.g., R_skin=0), use the upper bound as scale
    for i in range(len(x_scale)):
        if x_scale[i] < 1e-30:
            x_scale[i] = upper_phys[i] if upper_phys[i] > 0 else 1.0
    x_norm_0 = x0 / x_scale

    lower_norm = lower_phys / x_scale
    upper_norm = upper_phys / x_scale

    # Y-data scale for residual normalization
    Y_mag = np.abs(Y_data)
    Y_scale_val = float(np.mean(Y_mag[Y_mag > 0])) if np.any(Y_mag > 0) else 1.0

    if mode == '1port':
        result = least_squares(
            _residual_1port_normalized, x_norm_0,
            args=(x_scale, frequencies, Y_data, Y_scale_val),
            bounds=(lower_norm, upper_norm),
            method='trf',
            max_nfev=max_iterations,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
    else:
        raise NotImplementedError("2-port pi-model fitting not yet implemented")

    fitted = PiModelParams.from_vector(result.x * x_scale)

    # Compute fitted Y for comparison
    if mode == '1port':
        Y_fitted = fitted.Y_model_1port_array(frequencies)
    else:
        Y_fitted = np.array([fitted.Y_model_2port(f) for f in frequencies])

    fit_info = {
        'residual_norm': float(np.linalg.norm(result.fun)),
        'Y_fitted': Y_fitted,
        'cost': float(result.cost),
        'success': bool(result.success),
        'message': result.message,
    }

    return fitted, fit_info
