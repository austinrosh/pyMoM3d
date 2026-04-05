"""Current-based transmission line parameter extraction.

Extracts propagation constant β and characteristic impedance Z0 directly
from the MoM current distribution along a transmission line, completely
bypassing port discontinuity effects.

Theory
------
For a transmission line excited at one end (x=0) with an open circuit
at the far end (x=L), the x-directed current has a standing wave pattern:

    I_x(x) = I_0 * sin(β (L - x))

where β = ω √(ε_eff) / c₀.

By fitting the spatial current distribution to this pattern at each
frequency, we extract β(f) and from that ε_eff = (β c₀ / ω)².

The characteristic impedance Z0 is extracted from the relationship
between the terminal voltage and current at the driven end:

    Z0 = V_port / (I_0 * sin(βL))  (for port at x=0)

Or equivalently from the maximum current magnitude:
    Z0 = V_port / |I_max|  (for a quarter-wave stub)

References
----------
[1] Momentum Theory of Operation, Keysight Technologies
[2] V. Okhmatovski et al., "On Deembedding of Port Discontinuities in
    Full-Wave CAD Models of Multiport Circuits," IEEE TMTT, 2003.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import least_squares

from ..utils.constants import c0


@dataclass
class CurrentExtractionResult:
    """Results of current-based TL extraction.

    Attributes
    ----------
    freqs : ndarray, shape (N,)
        Frequencies (Hz).
    beta : ndarray, shape (N,)
        Extracted propagation constant at each frequency (rad/m).
    eps_eff : ndarray, shape (N,)
        Extracted effective permittivity at each frequency.
    Z0 : ndarray, shape (N,)
        Extracted characteristic impedance at each frequency (Ω).
    mean_eps_eff : float
        Frequency-averaged eps_eff.
    mean_Z0 : float
        Frequency-averaged Z0 (Ω).
    fit_quality : ndarray, shape (N,)
        R² goodness of fit at each frequency (1.0 = perfect).
    """

    freqs: np.ndarray
    beta: np.ndarray
    eps_eff: np.ndarray
    Z0: np.ndarray
    mean_eps_eff: float
    mean_Z0: float
    fit_quality: np.ndarray


def _rwg_edge_center_x(mesh, basis) -> np.ndarray:
    """Compute x-coordinate of each RWG edge midpoint.

    Parameters
    ----------
    mesh : Mesh
    basis : RWGBasis

    Returns
    -------
    x_centers : (N_basis,) float
    """
    verts = mesh.vertices  # (V, 3)
    edges = mesh.edges     # (E, 2)

    x_centers = np.empty(basis.num_basis, dtype=np.float64)
    for n in range(basis.num_basis):
        edge_idx = basis.edge_index[n]
        v0, v1 = edges[edge_idx]
        x_centers[n] = 0.5 * (verts[v0, 0] + verts[v1, 0])

    return x_centers


def _rwg_edge_direction(mesh, basis) -> np.ndarray:
    """Compute direction vector (unit) of each RWG edge.

    Parameters
    ----------
    mesh : Mesh
    basis : RWGBasis

    Returns
    -------
    directions : (N_basis, 3) float
    """
    verts = mesh.vertices
    edges = mesh.edges

    directions = np.empty((basis.num_basis, 3), dtype=np.float64)
    for n in range(basis.num_basis):
        edge_idx = basis.edge_index[n]
        v0, v1 = edges[edge_idx]
        d = verts[v1] - verts[v0]
        length = np.linalg.norm(d)
        if length > 1e-30:
            directions[n] = d / length
        else:
            directions[n] = np.array([0.0, 0.0, 0.0])

    return directions


def extract_current_profile(
    mesh,
    basis,
    I_coeffs: np.ndarray,
    propagation_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the current amplitude profile along the propagation axis.

    For each RWG basis function, projects its contribution onto the
    propagation direction and accumulates by position. Returns sorted
    (x_positions, I_x_amplitudes).

    Parameters
    ----------
    mesh : Mesh
    basis : RWGBasis
    I_coeffs : (N_basis,) complex
        MoM current coefficients.
    propagation_axis : int
        0=x, 1=y, 2=z. Default 0 (x-propagation).

    Returns
    -------
    x_positions : (M,) float
        Sorted unique x-positions of RWG edges.
    I_x : (M,) complex
        x-directed current amplitude at each position.
    """
    x_centers = _rwg_edge_center_x(mesh, basis)
    directions = _rwg_edge_direction(mesh, basis)

    # Project each basis function onto the propagation axis
    # I_n * edge_length * cos(theta) gives the current contribution
    axis_proj = directions[:, propagation_axis]

    # Group by x-position (bin edges at similar x)
    # Sort by x
    sort_idx = np.argsort(x_centers)
    x_sorted = x_centers[sort_idx]

    # Bin edges within tolerance
    tol = np.median(np.diff(x_sorted)) * 0.3 if len(x_sorted) > 1 else 1e-6
    bins = []
    current_bin = [sort_idx[0]]
    current_x = x_sorted[0]

    for i in range(1, len(sort_idx)):
        if x_sorted[i] - current_x < tol:
            current_bin.append(sort_idx[i])
        else:
            bins.append(current_bin)
            current_bin = [sort_idx[i]]
            current_x = x_sorted[i]
    bins.append(current_bin)

    x_positions = np.empty(len(bins))
    I_x = np.empty(len(bins), dtype=np.complex128)

    for i, bin_indices in enumerate(bins):
        bin_idx = np.array(bin_indices)
        x_positions[i] = np.mean(x_centers[bin_idx])
        # Sum current contributions: I_n * edge_length_n * projection
        I_x[i] = np.sum(
            I_coeffs[bin_idx] * basis.edge_length[bin_idx] * axis_proj[bin_idx]
        )

    return x_positions, I_x


def _fit_standing_wave(
    x_positions: np.ndarray,
    I_x: np.ndarray,
    x_open: float,
    beta_guess: float,
) -> tuple[float, float, float]:
    """Fit current to standing wave pattern I(x) = A * sin(β(x_open - x)).

    Parameters
    ----------
    x_positions : (M,) float
    I_x : (M,) complex
    x_open : float
        x-position of the open end.
    beta_guess : float
        Initial guess for β (rad/m).

    Returns
    -------
    beta : float
        Fitted propagation constant (rad/m).
    amplitude : complex
        Fitted amplitude A.
    r_squared : float
        Goodness of fit (R²).
    """
    # Use magnitude of current for fitting (avoids phase issues)
    I_mag = np.abs(I_x)

    # Exclude points very close to the open end (near-zero current, noisy)
    mask = np.abs(x_open - x_positions) > 0.5e-3

    if np.sum(mask) < 3:
        mask = np.ones(len(x_positions), dtype=bool)

    x_fit = x_positions[mask]
    I_fit = I_mag[mask]

    def residual(params):
        beta, A = params
        model = np.abs(A) * np.abs(np.sin(beta * (x_open - x_fit)))
        return model - I_fit

    x0 = np.array([beta_guess, np.max(I_fit)])
    result = least_squares(
        residual, x0,
        bounds=([0.1, 0.0], [1000.0, 10.0 * np.max(I_fit) + 1e-10]),
    )

    beta_fit = result.x[0]
    A_fit = result.x[1]

    # R² metric
    model = A_fit * np.abs(np.sin(beta_fit * (x_open - x_fit)))
    ss_res = np.sum((I_fit - model) ** 2)
    ss_tot = np.sum((I_fit - np.mean(I_fit)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-30)

    return beta_fit, A_fit, r_squared


def extract_tl_from_current(
    extractor,
    freqs: List[float],
    strip_length: float,
    port_index: int = 0,
    propagation_axis: int = 0,
    x_port: float | None = None,
    x_open: float | None = None,
    eps_eff_guess: float = 3.0,
) -> CurrentExtractionResult:
    """Extract TL parameters from MoM current distribution.

    Runs the NetworkExtractor at each frequency with store_currents=True,
    extracts the current profile along the strip, and fits the standing
    wave pattern to get β and Z0.

    Parameters
    ----------
    extractor : NetworkExtractor
        Must have store_currents=True.
    freqs : list of float
        Frequencies (Hz).
    strip_length : float
        Total strip length (m).
    port_index : int
        Which port's excitation to analyze.
    propagation_axis : int
        0=x, 1=y, 2=z.
    x_port : float, optional
        x-position of the driven port (m). If None, auto-detected.
    x_open : float, optional
        x-position of the open end (m). If None, uses the strip extent.
    eps_eff_guess : float
        Initial guess for eps_eff (for β initial guess).

    Returns
    -------
    CurrentExtractionResult
    """
    # Ensure currents are stored
    extractor.store_currents = True

    results = extractor.extract(freqs)
    freqs_arr = np.array(freqs)
    mesh = extractor.sim.mesh
    basis = extractor.sim.basis

    # Auto-detect strip extent from mesh
    if x_open is None:
        x_max = mesh.vertices[:, propagation_axis].max()
        x_min = mesh.vertices[:, propagation_axis].min()
        # Open end is the far end from the port
        if x_port is not None:
            x_open = x_max if abs(x_port - x_min) < abs(x_port - x_max) else x_min
        else:
            x_open = x_max  # default: open end at max x

    N = len(freqs)
    beta_arr = np.empty(N)
    eps_eff_arr = np.empty(N)
    Z0_arr = np.empty(N)
    quality_arr = np.empty(N)

    for i, (freq, result) in enumerate(zip(freqs, results)):
        k0 = 2.0 * np.pi * freq / c0
        beta_guess = k0 * np.sqrt(eps_eff_guess)

        # Get current coefficients for this port's excitation
        I_coeffs = result.I_solutions[:, port_index]

        # Extract current profile
        x_pos, I_x = extract_current_profile(
            mesh, basis, I_coeffs, propagation_axis,
        )

        # Fit standing wave
        beta_fit, A_fit, r_sq = _fit_standing_wave(
            x_pos, I_x, x_open, beta_guess,
        )

        beta_arr[i] = beta_fit
        eps_eff_arr[i] = (beta_fit / k0) ** 2 if k0 > 0 else np.nan
        quality_arr[i] = r_sq

        # Z0 from V_port / I_max
        # V_port = 1.0 (unit excitation from NetworkExtractor)
        # I_max = max current along the strip
        V_port = 1.0
        I_max = np.max(np.abs(I_x))
        if I_max > 1e-30:
            Z0_arr[i] = abs(V_port / I_max)
        else:
            Z0_arr[i] = np.nan

    # Average (exclude poor fits)
    good = quality_arr > 0.5
    if np.any(good):
        mean_ee = float(np.mean(eps_eff_arr[good]))
        mean_z0 = float(np.mean(Z0_arr[good]))
    else:
        mean_ee = float(np.mean(eps_eff_arr))
        mean_z0 = float(np.mean(Z0_arr))

    return CurrentExtractionResult(
        freqs=freqs_arr,
        beta=beta_arr,
        eps_eff=eps_eff_arr,
        Z0=Z0_arr,
        mean_eps_eff=mean_ee,
        mean_Z0=mean_z0,
        fit_quality=quality_arr,
    )
