"""Characteristic Mode Analysis (CMA) for MoM impedance matrices.

This module implements Characteristic Mode Analysis on top of the existing
MoM solver. CMA decomposes the impedance matrix Z = R + jX into intrinsic
current modes that are independent of the excitation source.

The characteristic modes satisfy the generalized eigenvalue problem:
    X · J_n = λ_n · R · J_n

where:
    - J_n: nth characteristic current (eigenvector)
    - λ_n: nth characteristic eigenvalue (real scalar)
    - R = Re(Z): radiation resistance matrix
    - X = Im(Z): reactance matrix

References:
    - R. Harrington and J. Mautz, "Theory of characteristic modes for
      conducting bodies," IEEE Trans. AP, vol. 19, no. 5, 1971.
    - E. Cabedo-Fabrés et al., "The theory of characteristic modes revisited:
      A contribution to the design of antennas for modern applications,"
      IEEE Antennas Propag. Mag., vol. 49, no. 5, 2007.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import scipy.linalg


@dataclass
class CMAResult:
    """Result of Characteristic Mode Analysis at a single frequency.

    Parameters
    ----------
    frequency : float
        Frequency (Hz) at which CMA was computed.
    eigenvalues : ndarray, shape (N,)
        Characteristic eigenvalues λ_n (real).
    eigenvectors : ndarray, shape (N, N)
        Characteristic currents J_n as columns. Power-normalized so that
        J_n^H · R · J_n = 1.
    modal_significance : ndarray, shape (N,)
        Modal significance MS_n = |1 / (1 + j·λ_n)|.
    characteristic_angle : ndarray, shape (N,)
        Characteristic angle α_n = 180° - arctan(λ_n), in degrees.
    R_matrix : ndarray, shape (N, N)
        Radiation resistance matrix Re(Z).
    X_matrix : ndarray, shape (N, N)
        Reactance matrix Im(Z).
    sort_order : ndarray, shape (N,)
        Indices that sort modes by decreasing modal significance.
    """
    frequency: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    modal_significance: np.ndarray
    characteristic_angle: np.ndarray
    R_matrix: np.ndarray
    X_matrix: np.ndarray
    sort_order: np.ndarray

    def get_mode(self, n: int) -> np.ndarray:
        """Get the nth most significant characteristic current.

        Parameters
        ----------
        n : int
            Mode index (0 = most significant).

        Returns
        -------
        J_n : ndarray, shape (N,)
            Characteristic current vector.
        """
        return self.eigenvectors[:, self.sort_order[n]]

    def get_eigenvalue(self, n: int) -> float:
        """Get the eigenvalue of the nth most significant mode."""
        return self.eigenvalues[self.sort_order[n]]

    def get_modal_significance(self, n: int) -> float:
        """Get the modal significance of the nth most significant mode."""
        return self.modal_significance[self.sort_order[n]]

    def get_characteristic_angle(self, n: int) -> float:
        """Get the characteristic angle (degrees) of the nth most significant mode."""
        return self.characteristic_angle[self.sort_order[n]]


def compute_modal_significance(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute modal significance from characteristic eigenvalues.

    Modal significance indicates how well a mode can be excited:
        MS_n = |1 / (1 + j·λ_n)| = 1 / sqrt(1 + λ_n²)

    MS = 1 at resonance (λ = 0), MS → 0 as |λ| → ∞.

    Parameters
    ----------
    eigenvalues : ndarray
        Characteristic eigenvalues λ_n.

    Returns
    -------
    ms : ndarray
        Modal significance values in [0, 1].
    """
    return 1.0 / np.sqrt(1.0 + np.real(eigenvalues) ** 2)


def compute_characteristic_angle(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute characteristic angle from characteristic eigenvalues.

    The characteristic angle represents the phase of the modal current
    relative to the excitation:
        α_n = 180° - arctan(λ_n)

    α = 180° at resonance, α = 90° for inductive, α = 270° for capacitive.

    Parameters
    ----------
    eigenvalues : ndarray
        Characteristic eigenvalues λ_n.

    Returns
    -------
    alpha : ndarray
        Characteristic angles in degrees.
    """
    return 180.0 - np.degrees(np.arctan(np.real(eigenvalues)))


def solve_cma(
    Z: np.ndarray,
    regularization: float = 1e-10,
    check_conditioning: bool = True,
) -> CMAResult:
    """Solve the characteristic mode analysis eigenvalue problem.

    Decomposes Z = R + jX and solves the generalized eigenvalue problem:
        X · J_n = λ_n · R · J_n

    Parameters
    ----------
    Z : ndarray, shape (N, N), complex
        Impedance matrix from fill_impedance_matrix().
    regularization : float, default 1e-10
        Regularization factor for R matrix if nearly singular.
        Applied as R_reg = R + epsilon * max(||R||, ||X||) * I.
        This prevents division by near-zero when R is very small
        compared to X (which occurs for electrically small structures).
    check_conditioning : bool, default True
        If True, warn when R is ill-conditioned.

    Returns
    -------
    result : CMAResult
        Characteristic mode analysis results.

    Notes
    -----
    The eigenvectors are power-normalized such that J_n^H · R · J_n = 1,
    which corresponds to unit radiated power for each mode.

    For PEC structures with symmetric Z, all eigenvalues should be real.
    Complex eigenvalues indicate numerical issues or asymmetric Z.

    The regularization is applied using max(||R||, ||X||) as the scale
    factor to handle cases where R is very small (electrically small
    structures have weak radiation resistance).
    """
    N = Z.shape[0]

    # Extract R and X matrices
    R = np.real(Z)
    X = np.imag(Z)

    # Enforce symmetry (guards against numerical asymmetry)
    R = 0.5 * (R + R.T)
    X = 0.5 * (X + X.T)

    # Check R conditioning
    if check_conditioning:
        cond_R = np.linalg.cond(R)
        if cond_R > 1e12:
            warnings.warn(
                f"R matrix ill-conditioned (cond={cond_R:.2e}); "
                "CMA modes may be inaccurate for electrically small structures."
            )

    # Regularize R using max of R and X norms as scale
    # This handles cases where ||R|| << ||X|| for small structures
    R_norm = np.linalg.norm(R, ord='fro')
    X_norm = np.linalg.norm(X, ord='fro')
    scale = max(R_norm, X_norm)
    epsilon = regularization * scale
    R_reg = R + epsilon * np.eye(N)

    # Solve generalized eigenvalue problem: X @ J = λ * R @ J
    eigenvalues, eigenvectors = scipy.linalg.eig(X, R_reg)

    # Filter spurious modes with large imaginary parts
    imag_tol = 1e-8 * np.max(np.abs(eigenvalues))
    large_imag = np.abs(np.imag(eigenvalues)) > imag_tol
    if np.any(large_imag):
        warnings.warn(
            f"{np.sum(large_imag)} eigenvalues have significant imaginary parts; "
            "Z matrix may not be symmetric."
        )

    # Take real part of eigenvalues (should be real for symmetric R, X)
    eigenvalues = np.real(eigenvalues)

    # Power-normalize eigenvectors: J_n^H · R · J_n = 1
    for n in range(N):
        J_n = eigenvectors[:, n]
        power = np.real(np.conj(J_n) @ R @ J_n)
        if power > 0:
            eigenvectors[:, n] = J_n / np.sqrt(power)

    # Compute modal significance and characteristic angle
    ms = compute_modal_significance(eigenvalues)
    alpha = compute_characteristic_angle(eigenvalues)

    # Sort by decreasing modal significance
    sort_order = np.argsort(ms)[::-1]

    return CMAResult(
        frequency=0.0,  # Set by caller
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        modal_significance=ms,
        characteristic_angle=alpha,
        R_matrix=R,
        X_matrix=X,
        sort_order=sort_order,
    )


def compute_characteristic_modes(
    Z: np.ndarray,
    frequency: float = 0.0,
    num_modes: Optional[int] = None,
    regularization: float = 1e-12,
) -> CMAResult:
    """Compute characteristic modes from an impedance matrix.

    This is the main entry point for CMA. It wraps solve_cma() and
    optionally filters to retain only the top modes by significance.

    Parameters
    ----------
    Z : ndarray, shape (N, N), complex
        Impedance matrix from fill_impedance_matrix().
    frequency : float, default 0.0
        Frequency (Hz) for metadata.
    num_modes : int, optional
        Number of modes to retain (sorted by significance).
        If None, returns all N modes.
    regularization : float, default 1e-12
        Regularization factor for R matrix.

    Returns
    -------
    result : CMAResult
        Characteristic mode analysis results.

    Examples
    --------
    >>> from pyMoM3d import fill_impedance_matrix, compute_rwg_connectivity
    >>> from pyMoM3d.analysis.cma import compute_characteristic_modes
    >>> Z = fill_impedance_matrix(basis, mesh, k, eta0)
    >>> cma = compute_characteristic_modes(Z, frequency=1e9, num_modes=5)
    >>> J_mode1 = cma.get_mode(0)  # Most significant mode
    >>> print(f"Mode 1: MS={cma.get_modal_significance(0):.3f}")
    """
    result = solve_cma(Z, regularization=regularization)
    result.frequency = frequency

    if num_modes is not None and num_modes < len(result.eigenvalues):
        result.sort_order = result.sort_order[:num_modes]

    return result


def track_modes_across_frequency(
    cma_results: List[CMAResult],
    correlation_threshold: float = 0.7,
) -> List[np.ndarray]:
    """Track characteristic modes across frequency using eigenvector correlation.

    Modes are matched between consecutive frequencies by maximizing the
    R-weighted inner product (correlation) between eigenvectors.

    Parameters
    ----------
    cma_results : list of CMAResult
        CMA results at multiple frequencies, in increasing frequency order.
    correlation_threshold : float, default 0.7
        Minimum correlation to consider a mode match valid.
        Below this threshold, mode tracking may be unreliable.

    Returns
    -------
    tracked_indices : list of ndarray
        For each frequency, an array of indices mapping the original
        mode ordering to the tracked ordering. tracked_indices[f][n]
        gives the original mode index that corresponds to tracked mode n
        at frequency f.

    Notes
    -----
    Mode tracking is challenging when:
    - Modes cross (exchange significance rankings)
    - Modes are degenerate (e.g., on symmetric structures)
    - Large frequency steps cause discontinuous changes

    When correlation falls below threshold, a warning is issued.

    Examples
    --------
    >>> tracked = track_modes_across_frequency(cma_results)
    >>> # Get tracked mode 0 across all frequencies
    >>> mode0_eigenvalues = [cma.eigenvalues[tracked[i][0]]
    ...                      for i, cma in enumerate(cma_results)]
    """
    if len(cma_results) < 2:
        return [cma_results[0].sort_order] if cma_results else []

    tracked_indices = [cma_results[0].sort_order.copy()]
    N = len(cma_results[0].eigenvalues)

    for f_idx in range(1, len(cma_results)):
        prev_result = cma_results[f_idx - 1]
        curr_result = cma_results[f_idx]

        # Get eigenvectors in tracked order from previous frequency
        prev_order = tracked_indices[f_idx - 1]
        J_prev = prev_result.eigenvectors[:, prev_order]

        # Current eigenvectors in original order
        J_curr = curr_result.eigenvectors

        # Compute R-weighted correlation matrix
        R = curr_result.R_matrix
        # correlation[i, j] = |J_prev[:, i]^H @ R @ J_curr[:, j]|
        correlation = np.abs(np.conj(J_prev.T) @ R @ J_curr)

        # Use Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-correlation)
        except ImportError:
            # Fallback: greedy matching
            col_ind = np.zeros(N, dtype=int)
            used = set()
            for i in range(N):
                best_j = -1
                best_corr = -1
                for j in range(N):
                    if j not in used and correlation[i, j] > best_corr:
                        best_corr = correlation[i, j]
                        best_j = j
                col_ind[i] = best_j
                used.add(best_j)
            row_ind = np.arange(N)

        # Check for low correlations
        min_corr = min(correlation[row_ind[i], col_ind[i]] for i in range(N))
        if min_corr < correlation_threshold:
            warnings.warn(
                f"Low mode correlation ({min_corr:.3f}) at frequency index {f_idx}; "
                "mode tracking may be unreliable."
            )

        tracked_indices.append(col_ind)

    return tracked_indices


def verify_orthogonality(cma_result: CMAResult, tolerance: float = 1e-6) -> Tuple[bool, float]:
    """Verify that characteristic modes are R-orthogonal.

    Modes should satisfy J_m^H · R · J_n = δ_mn (Kronecker delta).

    Parameters
    ----------
    cma_result : CMAResult
        CMA results to verify.
    tolerance : float, default 1e-6
        Maximum allowed deviation from orthogonality.

    Returns
    -------
    is_orthogonal : bool
        True if modes are orthogonal within tolerance.
    max_error : float
        Maximum off-diagonal element of J^H @ R @ J.
    """
    J = cma_result.eigenvectors
    R = cma_result.R_matrix

    # Compute J^H @ R @ J, should be identity
    orthog_matrix = np.conj(J.T) @ R @ J
    identity = np.eye(len(cma_result.eigenvalues))

    error_matrix = np.abs(orthog_matrix - identity)
    max_error = np.max(error_matrix)

    return max_error < tolerance, max_error


def verify_eigenvalue_reality(
    cma_result: CMAResult,
    tolerance: float = 1e-10,
) -> Tuple[bool, float]:
    """Verify that eigenvalues are real (as expected for symmetric Z).

    Parameters
    ----------
    cma_result : CMAResult
        CMA results to verify.
    tolerance : float, default 1e-10
        Maximum allowed imaginary part relative to largest eigenvalue.

    Returns
    -------
    are_real : bool
        True if all eigenvalues are effectively real.
    max_imag : float
        Maximum absolute imaginary part.
    """
    # Note: eigenvalues in CMAResult are already real-valued,
    # but this can be used on raw scipy output
    eigenvalues = cma_result.eigenvalues
    if np.iscomplexobj(eigenvalues):
        max_imag = np.max(np.abs(np.imag(eigenvalues)))
        scale = np.max(np.abs(eigenvalues)) if np.max(np.abs(eigenvalues)) > 0 else 1.0
        return max_imag / scale < tolerance, max_imag
    return True, 0.0


def compute_modal_excitation_coefficient(
    cma_result: CMAResult,
    V: np.ndarray,
    mode_index: int,
) -> complex:
    """Compute the modal excitation coefficient for a given excitation.

    The modal excitation coefficient α_n represents how strongly mode n
    is excited by the given voltage vector V:
        α_n = V^T · J_n / (1 + j·λ_n)

    Parameters
    ----------
    cma_result : CMAResult
        CMA results.
    V : ndarray, shape (N,)
        Excitation voltage vector from PlaneWaveExcitation or DeltaGapExcitation.
    mode_index : int
        Mode index (0 = most significant).

    Returns
    -------
    alpha_n : complex
        Modal excitation coefficient.

    Notes
    -----
    The total current can be reconstructed as:
        I_total = Σ_n α_n · J_n
    """
    idx = cma_result.sort_order[mode_index]
    J_n = cma_result.eigenvectors[:, idx]
    lambda_n = cma_result.eigenvalues[idx]

    # Modal excitation coefficient
    alpha_n = np.dot(V, J_n) / (1.0 + 1j * lambda_n)
    return alpha_n


def expand_current_in_modes(
    cma_result: CMAResult,
    I: np.ndarray,
    num_modes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand a driven current in terms of characteristic modes.

    Computes the coefficients c_n such that I ≈ Σ_n c_n · J_n.

    Parameters
    ----------
    cma_result : CMAResult
        CMA results.
    I : ndarray, shape (N,)
        Driven current coefficients from MoM solve.
    num_modes : int, optional
        Number of modes to use in expansion.
        If None, uses all modes.

    Returns
    -------
    coefficients : ndarray, shape (M,)
        Mode coefficients c_n.
    reconstruction : ndarray, shape (N,)
        Reconstructed current I_approx = Σ_n c_n · J_n.

    Notes
    -----
    For power-normalized modes, c_n = J_n^H · R · I.
    """
    R = cma_result.R_matrix
    J = cma_result.eigenvectors

    M = num_modes if num_modes is not None else len(cma_result.eigenvalues)
    order = cma_result.sort_order[:M]

    coefficients = np.zeros(M, dtype=np.complex128)
    reconstruction = np.zeros_like(I)

    for i, idx in enumerate(order):
        J_n = J[:, idx]
        c_n = np.conj(J_n) @ R @ I
        coefficients[i] = c_n
        reconstruction += c_n * J_n

    return coefficients, reconstruction


def cma_frequency_sweep(
    basis,
    mesh,
    frequencies: np.ndarray,
    eta: float,
    quad_order: int = 4,
    near_threshold: float = 0.2,
    num_modes: Optional[int] = None,
    track_modes: bool = True,
    progress_callback=None,
) -> Tuple[List[CMAResult], Optional[List[np.ndarray]]]:
    """Perform CMA across a frequency sweep.

    Computes characteristic modes at each frequency and optionally
    tracks modes across the sweep.

    Parameters
    ----------
    basis : RWGBasis
        RWG basis function data.
    mesh : Mesh
        Surface mesh.
    frequencies : array-like
        Frequencies to analyze (Hz).
    eta : float
        Intrinsic impedance of the medium (Ohms).
    quad_order : int, default 4
        Quadrature order for Z-fill.
    near_threshold : float, default 0.2
        Near-field threshold for singularity extraction.
    num_modes : int, optional
        Number of modes to retain at each frequency.
    track_modes : bool, default True
        Whether to track modes across frequency.
    progress_callback : callable, optional
        Called with (frequency_index, total_frequencies) at each step.

    Returns
    -------
    cma_results : list of CMAResult
        CMA results at each frequency.
    tracked_indices : list of ndarray or None
        Mode tracking indices if track_modes=True, else None.

    Examples
    --------
    >>> from pyMoM3d.analysis.cma import cma_frequency_sweep
    >>> results, tracking = cma_frequency_sweep(
    ...     basis, mesh, frequencies=np.linspace(1e9, 2e9, 11),
    ...     eta=eta0, num_modes=5
    ... )
    >>> # Plot eigenvalue vs frequency for mode 0
    >>> eigenvalues_mode0 = [results[i].eigenvalues[tracking[i][0]]
    ...                      for i in range(len(results))]
    """
    from ..mom.impedance import fill_impedance_matrix
    from ..utils.constants import c0

    cma_results = []

    for i, freq in enumerate(frequencies):
        if progress_callback is not None:
            progress_callback(i, len(frequencies))

        k = 2.0 * np.pi * freq / c0
        Z = fill_impedance_matrix(
            basis, mesh, k, eta,
            quad_order=quad_order,
            near_threshold=near_threshold,
        )
        result = compute_characteristic_modes(
            Z, frequency=freq, num_modes=num_modes
        )
        cma_results.append(result)

    tracked_indices = None
    if track_modes and len(cma_results) > 1:
        tracked_indices = track_modes_across_frequency(cma_results)

    return cma_results, tracked_indices
