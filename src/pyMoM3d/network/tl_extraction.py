"""Y11-based transmission line parameter extraction.

Extracts Z0 and eps_eff from 1-port self-admittance Y11, bypassing the
weak Y12 mutual coupling and port discontinuity problems that plague
2-port S-parameter extraction in planar MoM solvers.

Theory
------
For a shunt-terminated (open-circuit) microstrip stub of length L:

    Y11_TL = -j * Y0 * cot(β L)

where Y0 = 1/Z0 and β = ω √(ε_eff) / c₀.

The measured Y11 from MoM includes a port discontinuity capacitance C_port
(parasitic from the strip delta-gap excitation):

    Y11_measured(f) = -j / Z0 * cot(2π f √(ε_eff) L / c₀) + j 2π f C_port

This module fits the three unknowns (Z0, ε_eff, C_port) to Y11_measured(f)
at multiple frequencies using nonlinear least squares.

References
----------
[1] V. Okhmatovski et al., "On Deembedding of Port Discontinuities in
    Full-Wave CAD Models of Multiport Circuits," IEEE TMTT, Dec. 2003.
[2] Keysight Momentum Theory of Operation — port calibration methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import least_squares

from ..utils.constants import c0


@dataclass
class Y11ExtractionResult:
    """Results of Y11-based transmission line extraction.

    Attributes
    ----------
    Z0 : float
        Extracted characteristic impedance (Ω).
    eps_eff : float
        Extracted effective permittivity.
    C_port : float
        Extracted port discontinuity capacitance (F).
    freqs : ndarray, shape (N,)
        Frequencies used for extraction (Hz).
    Y11_measured : ndarray, shape (N,)
        Measured Y11 values (complex).
    Y11_fit : ndarray, shape (N,)
        Fitted Y11 values (complex).
    residual_norm : float
        Norm of the fit residual (quality metric).
    """

    Z0: float
    eps_eff: float
    C_port: float
    freqs: np.ndarray
    Y11_measured: np.ndarray
    Y11_fit: np.ndarray
    residual_norm: float


def _y11_model(freqs: np.ndarray, Z0: float, eps_eff: float,
               C_port: float, L: float) -> np.ndarray:
    """Compute modeled Y11 = Y11_TL + jωC_port.

    Parameters
    ----------
    freqs : (N,) float
        Frequencies (Hz).
    Z0 : float
        Characteristic impedance (Ω).
    eps_eff : float
        Effective permittivity.
    C_port : float
        Port discontinuity capacitance (F).
    L : float
        Stub length (m).

    Returns
    -------
    Y11 : (N,) complex
    """
    omega = 2.0 * np.pi * freqs
    beta = omega * np.sqrt(eps_eff) / c0
    Y0 = 1.0 / Z0

    # Y11_TL for an open-circuited stub: -j Y0 cot(βL)
    beta_L = beta * L
    # Avoid division by zero at β L = nπ
    sin_bL = np.sin(beta_L)
    cos_bL = np.cos(beta_L)
    # cot(βL) = cos(βL)/sin(βL)
    cot_bL = np.where(np.abs(sin_bL) > 1e-30,
                      cos_bL / sin_bL,
                      np.sign(cos_bL) * 1e30)

    Y11_TL = -1j * Y0 * cot_bL
    Y11_parasitic = 1j * omega * C_port

    return Y11_TL + Y11_parasitic


def _residual_func(params: np.ndarray, freqs: np.ndarray,
                   Y11_meas: np.ndarray, L: float) -> np.ndarray:
    """Residual for least-squares fit.

    Parameters are [Z0, eps_eff, C_port_pF] where C_port is in pF for
    better numerical scaling.

    Uses relative residuals weighted by 1/(1 + |Y11|) to prevent resonance
    spikes (where cot(βL) → ∞) from dominating the fit.
    """
    Z0, eps_eff, C_port_pF = params
    C_port = C_port_pF * 1e-12  # pF → F

    Y11_model = _y11_model(freqs, Z0, eps_eff, C_port, L)

    diff = Y11_model - Y11_meas
    # Weight: suppress contribution near cot resonances where |Y11| blows up
    weight = 1.0 / (1.0 + np.abs(Y11_meas))
    weighted_diff = diff * weight
    return np.concatenate([weighted_diff.real, weighted_diff.imag])


def _estimate_initial_params(
    freqs: np.ndarray, Y11: np.ndarray, L: float,
    Z0_guess: float, eps_eff_guess: float,
) -> list[tuple[float, float, float]]:
    """Generate multiple initial guesses for (Z0, eps_eff, C_port_pF).

    Uses several heuristics:
    1. Half-wave resonance detection (|Y11| peak) → eps_eff
    2. Low-frequency asymptote → Z0
    3. User-provided guesses as fallback
    """
    omega = 2.0 * np.pi * freqs
    B = Y11.imag
    guesses = []

    # --- Heuristic 1: Half-wave resonance detection ---
    # At f_hw, βL = nπ → cot(βL) → ∞ → |Y11| peaks
    # eps_eff = (n * c0 / (2 * f_hw * L))^2
    abs_Y11 = np.abs(Y11)
    peak_idx = np.argmax(abs_Y11)
    # Only trust this if the peak is well above the median
    if abs_Y11[peak_idx] > 3.0 * np.median(abs_Y11) and peak_idx > 0 and peak_idx < len(freqs) - 1:
        f_hw = freqs[peak_idx]
        for n in [1, 2]:  # try first and second harmonic
            eps_from_hw = (n * c0 / (2.0 * f_hw * L)) ** 2
            if 1.0 < eps_from_hw < 20.0:
                # Estimate Z0 from low-frequency data with this eps_eff
                z0_est = _estimate_z0_from_low_freq(freqs, B, L, eps_from_hw, Z0_guess)
                c_est = _estimate_cport(freqs, B, L, eps_from_hw, z0_est)
                guesses.append((z0_est, eps_from_hw, c_est))

    # --- Heuristic 2: Low-frequency capacitive slope ---
    # At low freq: B ≈ -c0/(Z0*ω*√ε*L) + ωC
    # If we have at least 2 low-freq points, fit a line: B = a/ω + b*ω
    if len(freqs) >= 4:
        # Use lowest quarter of frequencies
        n_low = max(3, len(freqs) // 4)
        f_low = freqs[:n_low]
        B_low = B[:n_low]
        omega_low = 2 * np.pi * f_low

        # Fit: B = a/ω + b*ω  where a = -c0/(Z0*√ε*L), b = C_port
        A_mat = np.column_stack([1.0 / omega_low, omega_low])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, B_low, rcond=None)
            a_coeff, b_coeff = coeffs
            # a = -c0/(Z0*√ε*L)
            # Try with user's eps_eff_guess
            for eps_try in [eps_eff_guess, 2.0, 4.0, 6.0]:
                z0_from_a = abs(-c0 / (a_coeff * np.sqrt(eps_try) * L))
                z0_from_a = np.clip(z0_from_a, 5.0, 400.0)
                c_pF = b_coeff * 1e12
                c_pF = np.clip(c_pF, -10.0, 50.0)
                guesses.append((float(z0_from_a), float(eps_try), float(c_pF)))
        except np.linalg.LinAlgError:
            pass

    # --- Heuristic 3: User-provided guess ---
    guesses.append((Z0_guess, eps_eff_guess, 1.0))

    return guesses


def _estimate_z0_from_low_freq(
    freqs: np.ndarray, B: np.ndarray, L: float,
    eps_eff: float, Z0_fallback: float,
) -> float:
    """Estimate Z0 from low-frequency Im(Y11)."""
    omega = 2.0 * np.pi * freqs[0]
    beta_L = omega * np.sqrt(eps_eff) * L / c0
    if beta_L < 0.3 and abs(B[0]) > 1e-15:
        # cot(βL) ≈ 1/(βL) for small βL
        z0_est = abs(1.0 / (B[0] * beta_L * np.sqrt(eps_eff) * L * omega / c0))
        # More precisely: B[0] ≈ -Y0/tan(βL) + ωC
        # For small βL: B[0] ≈ -1/(Z0*βL) + ωC ≈ -c0/(Z0*ω*√ε*L)
        z0_est = abs(-c0 / (B[0] * omega * np.sqrt(eps_eff) * L))
        return float(np.clip(z0_est, 5.0, 400.0))
    return Z0_fallback


def _estimate_cport(
    freqs: np.ndarray, B: np.ndarray, L: float,
    eps_eff: float, Z0: float,
) -> float:
    """Estimate C_port in pF from data-model residual at a safe frequency."""
    # Pick a frequency where βL is not near resonance
    omega = 2.0 * np.pi * freqs
    beta_L = omega * np.sqrt(eps_eff) * L / c0
    # Distance to nearest resonance (nπ)
    dist_to_res = np.abs(np.sin(beta_L))
    safe_idx = np.argmax(dist_to_res)

    cot_bL = np.cos(beta_L[safe_idx]) / np.sin(beta_L[safe_idx])
    Y11_TL = -cot_bL / Z0
    C_port = (B[safe_idx] - Y11_TL) / omega[safe_idx]
    return float(np.clip(C_port * 1e12, -10.0, 50.0))


def extract_tl_from_y11(
    freqs: np.ndarray,
    Y11_values: np.ndarray,
    stub_length: float,
    Z0_guess: float = 50.0,
    eps_eff_guess: float = 2.5,
    C_port_guess_pF: float = 1.0,
    termination: str = 'open',
) -> Y11ExtractionResult:
    """Extract Z0, eps_eff, C_port from measured Y11 at multiple frequencies.

    Parameters
    ----------
    freqs : (N,) float
        Frequencies (Hz). At least 3 frequencies required (3 unknowns).
    Y11_values : (N,) complex
        Measured Y11 self-admittance at each frequency.
    stub_length : float
        Physical length of the stub (m).
    Z0_guess : float
        Initial guess for Z0 (Ω).
    eps_eff_guess : float
        Initial guess for eps_eff.
    C_port_guess_pF : float
        Initial guess for port capacitance (pF).
    termination : str
        Stub termination: 'open' (default) or 'short'.

    Returns
    -------
    Y11ExtractionResult

    Raises
    ------
    ValueError
        If fewer than 3 frequencies are provided.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    Y11_values = np.asarray(Y11_values, dtype=np.complex128)

    if len(freqs) < 3:
        raise ValueError(
            f"Need at least 3 frequencies for 3-parameter fit, got {len(freqs)}"
        )

    if termination != 'open':
        raise NotImplementedError(
            f"Only 'open' termination supported, got '{termination}'"
        )

    # Generate multiple initial guesses using data heuristics
    guess_list = _estimate_initial_params(
        freqs, Y11_values, stub_length, Z0_guess, eps_eff_guess,
    )
    # Also add user's explicit guess
    guess_list.append((Z0_guess, eps_eff_guess, C_port_guess_pF))

    best_result = None
    best_cost = np.inf

    for (z0_g, ee_g, cp_g) in guess_list:
        x0 = np.array([z0_g, ee_g, cp_g])
        result = least_squares(
            _residual_func,
            x0,
            args=(freqs, Y11_values, stub_length),
            bounds=([1.0, 1.0, -10.0], [500.0, 20.0, 50.0]),
            method='trf',
            ftol=1e-12,
            xtol=1e-12,
        )
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result

    Z0_fit = best_result.x[0]
    eps_eff_fit = best_result.x[1]
    C_port_fit = best_result.x[2] * 1e-12  # pF → F

    Y11_fit = _y11_model(freqs, Z0_fit, eps_eff_fit, C_port_fit, stub_length)

    return Y11ExtractionResult(
        Z0=Z0_fit,
        eps_eff=eps_eff_fit,
        C_port=C_port_fit,
        freqs=freqs,
        Y11_measured=Y11_values,
        Y11_fit=Y11_fit,
        residual_norm=float(np.linalg.norm(best_result.fun)),
    )


def _y11_diff_model(
    freqs: np.ndarray, Z0: float, eps_eff: float,
    L1: float, L2: float,
) -> np.ndarray:
    """Model for Y11(L1) - Y11(L2) (port parasitic cancels).

    ΔY11 = -j/Z0 * [cot(βL1) - cot(βL2)]
    """
    omega = 2.0 * np.pi * freqs
    beta = omega * np.sqrt(eps_eff) / c0
    Y0 = 1.0 / Z0

    def safe_cot(x):
        s = np.sin(x)
        c = np.cos(x)
        return np.where(np.abs(s) > 1e-30, c / s, np.sign(c) * 1e30)

    cot1 = safe_cot(beta * L1)
    cot2 = safe_cot(beta * L2)

    return -1j * Y0 * (cot1 - cot2)


def _diff_residual(params, freqs, dY11, L1, L2):
    """Residual for dual-stub fit (2 unknowns: Z0, eps_eff)."""
    Z0, eps_eff = params
    model = _y11_diff_model(freqs, Z0, eps_eff, L1, L2)
    diff = model - dY11
    weight = 1.0 / (1.0 + np.abs(dY11))
    weighted = diff * weight
    return np.concatenate([weighted.real, weighted.imag])


def extract_tl_dual_stub(
    freqs: np.ndarray,
    Y11_L1: np.ndarray,
    Y11_L2: np.ndarray,
    stub_length_1: float,
    stub_length_2: float,
    Z0_guess: float = 50.0,
    eps_eff_guess: float = 3.0,
) -> Y11ExtractionResult:
    """Extract Z0 and eps_eff from differential Y11 of two stub lengths.

    By subtracting Y11 measured on two stubs with the same port model,
    the port discontinuity capacitance cancels:

        ΔY11 = Y11(L1) - Y11(L2) = Y_TL(L1) - Y_TL(L2)

    Only Z0 and eps_eff remain as unknowns (2-parameter fit).

    Parameters
    ----------
    freqs : (N,) float
        Frequencies (Hz).
    Y11_L1, Y11_L2 : (N,) complex
        Measured Y11 for stub lengths L1 and L2.
    stub_length_1, stub_length_2 : float
        Physical stub lengths (m).
    Z0_guess, eps_eff_guess : float
        Initial guesses.

    Returns
    -------
    Y11ExtractionResult
        Z0 and eps_eff extracted; C_port estimated from residual.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    dY11 = np.asarray(Y11_L1, dtype=np.complex128) - np.asarray(Y11_L2, dtype=np.complex128)

    if len(freqs) < 2:
        raise ValueError("Need at least 2 frequencies for 2-parameter fit")

    L1, L2 = stub_length_1, stub_length_2

    # Multi-start with different initial guesses
    guesses = [
        (Z0_guess, eps_eff_guess),
        (Z0_guess * 0.5, eps_eff_guess * 0.8),
        (Z0_guess * 2.0, eps_eff_guess * 1.5),
    ]

    # Detect resonance from |ΔY11| peak → eps_eff
    abs_dY = np.abs(dY11)
    peak_idx = np.argmax(abs_dY)
    if abs_dY[peak_idx] > 3.0 * np.median(abs_dY) and 0 < peak_idx < len(freqs) - 1:
        f_peak = freqs[peak_idx]
        for n in [1, 2]:
            for L_try in [L1, L2]:
                ee_try = (n * c0 / (2.0 * f_peak * L_try)) ** 2
                if 1.0 < ee_try < 20.0:
                    guesses.append((Z0_guess, ee_try))

    best_result = None
    best_cost = np.inf

    for z0_g, ee_g in guesses:
        x0 = np.array([z0_g, ee_g])
        result = least_squares(
            _diff_residual, x0,
            args=(freqs, dY11, L1, L2),
            bounds=([1.0, 1.0], [500.0, 20.0]),
            method='trf',
            ftol=1e-12, xtol=1e-12,
        )
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result

    Z0_fit = best_result.x[0]
    eps_eff_fit = best_result.x[1]

    # Estimate C_port from L1 measurement
    Y11_L1 = np.asarray(Y11_L1, dtype=np.complex128)
    Y11_TL_L1 = _y11_model(freqs, Z0_fit, eps_eff_fit, 0.0, L1)
    omega = 2.0 * np.pi * freqs
    C_port_arr = (Y11_L1.imag - Y11_TL_L1.imag) / omega
    C_port = float(np.median(C_port_arr))

    Y11_fit = _y11_model(freqs, Z0_fit, eps_eff_fit, C_port, L1)

    return Y11ExtractionResult(
        Z0=Z0_fit,
        eps_eff=eps_eff_fit,
        C_port=C_port,
        freqs=freqs,
        Y11_measured=Y11_L1,
        Y11_fit=Y11_fit,
        residual_norm=float(np.linalg.norm(best_result.fun)),
    )


def extract_tl_from_extractor(
    extractor,
    freqs: List[float],
    stub_length: float,
    port_index: int = 0,
    Z0_guess: float = 50.0,
    eps_eff_guess: float = 2.5,
    C_port_guess_pF: float = 1.0,
) -> Y11ExtractionResult:
    """Run NetworkExtractor at multiple frequencies and extract TL params.

    Convenience wrapper that runs the MoM extraction and feeds Y11 to
    the fitting routine.

    Parameters
    ----------
    extractor : NetworkExtractor
        Configured network extractor with at least one port.
    freqs : list of float
        Frequencies (Hz).
    stub_length : float
        Physical length of the stub (m).
    port_index : int
        Which port's Y11 to use (default 0).
    Z0_guess : float
        Initial guess for Z0 (Ω).
    eps_eff_guess : float
        Initial guess for eps_eff.
    C_port_guess_pF : float
        Initial guess for port capacitance (pF).

    Returns
    -------
    Y11ExtractionResult
    """
    results = extractor.extract(freqs)
    freqs_arr = np.array(freqs)

    Y11_arr = np.array([
        r.Y_matrix[port_index, port_index] for r in results
    ])

    return extract_tl_from_y11(
        freqs_arr, Y11_arr, stub_length,
        Z0_guess=Z0_guess,
        eps_eff_guess=eps_eff_guess,
        C_port_guess_pF=C_port_guess_pF,
    )
