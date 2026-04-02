"""Matrix Pencil Method (GPOF) for complex exponential fitting.

Implements the Hua-Sarkar (1990) matrix pencil algorithm to extract K complex
exponentials from M uniformly-sampled signal values:

    f(t_n) ≈ Σᵢ aᵢ · exp(sᵢ · t_n),   t_n = n · Δt,  n = 0 … M-1

Applications in DCIM
--------------------
The layered-medium smooth Sommerfeld correction can be sampled along a
deformed path in the complex kρ plane.  GPOF fits these samples to a sum of
complex exponentials, which are then converted to complex image terms via the
Sommerfeld identity.  This yields a fast closed-form approximation that is
evaluated in O(K) operations per observation point.

Algorithm (Hua & Sarkar 1990)
------------------------------
1. Build the (M-L) × (L+1) data (Hankel) matrix Y from the M samples,
   where L ≈ M/3 is the pencil parameter.
2. SVD of Y; truncate to K significant singular values.
3. Form the matrix pencil Y₁† Y₂ (Y₁ = Y[:,0:L], Y₂ = Y[:,1:L+1]).
4. Eigenvalues of the pencil: zₖ = exp(sₖ Δt).
5. Amplitudes from overdetermined least-squares on the Vandermonde system.

References
----------
Y. Hua and T. K. Sarkar, "Matrix pencil method for estimating parameters of
exponentially damped/undamped sinusoids in noise," IEEE Trans. ASSP, 1990.

M. I. Aksun, "A robust approach for the derivation of closed-form Green's
functions," IEEE Trans. Microw. Theory Tech., 1996.
"""

from __future__ import annotations

import numpy as np


def matrix_pencil(
    samples: np.ndarray,
    dt: float,
    pencil_param: int | None = None,
    svd_threshold: float = 1e-3,
    K_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract complex exponential components from uniformly sampled data.

    Parameters
    ----------
    samples : (M,) complex array
        Signal values f(t_n) at t_n = n·dt for n = 0 … M-1.
    dt : float
        Uniform sampling interval.
    pencil_param : int, optional
        Matrix pencil parameter L.  Must satisfy M/3 ≤ L ≤ 2M/3.
        Default: L = M // 3.
    svd_threshold : float
        Relative singular value threshold for rank estimation.
        Singular values below svd_threshold · σ_max are discarded.
        Default 1e-3.
    K_max : int, optional
        Maximum number of components to extract.  None means no limit.

    Returns
    -------
    amplitudes : (K,) complex128
        Complex amplitudes aᵢ.
    exponents : (K,) complex128
        Complex exponents sᵢ such that f(t) ≈ Σᵢ aᵢ exp(sᵢ t).

    Raises
    ------
    ValueError
        If fewer than 4 samples are provided or pencil_param is invalid.
    """
    samples = np.asarray(samples, dtype=np.complex128)
    M = len(samples)
    if M < 4:
        raise ValueError(f"matrix_pencil requires at least 4 samples; got {M}.")

    # Pencil parameter
    L = pencil_param if pencil_param is not None else M // 3
    if not (M // 3 <= L <= 2 * M // 3):
        raise ValueError(
            f"pencil_param L={L} outside recommended range [{M//3}, {2*M//3}]."
        )

    # ------------------------------------------------------------------
    # Step 1: Build the (M-L) × (L+1) Hankel data matrix
    # ------------------------------------------------------------------
    Y = np.empty((M - L, L + 1), dtype=np.complex128)
    for col in range(L + 1):
        Y[:, col] = samples[col: col + M - L]

    # ------------------------------------------------------------------
    # Step 2: SVD and rank estimation
    # ------------------------------------------------------------------
    U, sigma, Vh = np.linalg.svd(Y, full_matrices=False)

    K_svd = int(np.sum(sigma >= svd_threshold * sigma[0]))
    K = K_svd if K_max is None else min(K_svd, K_max)
    K = max(K, 1)  # always extract at least one component

    # Truncated right singular vectors: (K, L+1)
    V_K = Vh[:K, :]

    # ------------------------------------------------------------------
    # Step 3: Matrix pencil eigenvalues
    # ------------------------------------------------------------------
    # Partition V_K: Y₁ uses columns 0..L-1, Y₂ uses columns 1..L
    V1 = V_K[:, :L]    # (K, L)
    V2 = V_K[:, 1:]    # (K, L)

    # Eigenvalues of V₁† V₂ (via pseudo-inverse) give z_k = exp(s_k dt)
    # Use least-squares: V₁ Z = V₂ → Z = V₁† V₂
    Z, _, _, _ = np.linalg.lstsq(V1.T, V2.T, rcond=None)
    z_poles = np.linalg.eigvals(Z.T)   # (K,)

    # Convert from z-domain to s-domain
    with np.errstate(divide='ignore', invalid='ignore'):
        exponents = np.where(
            np.abs(z_poles) > 0,
            np.log(z_poles) / dt,
            np.full_like(z_poles, -np.inf),
        )

    # ------------------------------------------------------------------
    # Step 4: Amplitudes from overdetermined least-squares
    # ------------------------------------------------------------------
    # Vandermonde system: V[n, k] = z_k^n,  n = 0..M-1
    n_idx = np.arange(M, dtype=np.float64)
    Vander = z_poles[np.newaxis, :] ** n_idx[:, np.newaxis]   # (M, K)
    amplitudes, _, _, _ = np.linalg.lstsq(Vander, samples, rcond=None)

    return amplitudes.astype(np.complex128), exponents.astype(np.complex128)


class GPOFSolver:
    """Stateful wrapper around matrix_pencil for repeated fitting tasks.

    Stores fitting parameters and exposes evaluate / residual helpers.

    Parameters
    ----------
    svd_threshold : float
        Relative singular value threshold (passed to matrix_pencil).
    K_max : int, optional
        Maximum number of exponential components.
    """

    def __init__(self, svd_threshold: float = 1e-3, K_max: int | None = None):
        self.svd_threshold = svd_threshold
        self.K_max = K_max
        self.amplitudes: np.ndarray | None = None
        self.exponents:  np.ndarray | None = None

    def fit(self, samples: np.ndarray, dt: float) -> None:
        """Fit complex exponential model to samples.

        Parameters
        ----------
        samples : (M,) complex
        dt : float
        """
        self.amplitudes, self.exponents = matrix_pencil(
            samples,
            dt,
            svd_threshold=self.svd_threshold,
            K_max=self.K_max,
        )

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the fitted model at arbitrary times t.

        Parameters
        ----------
        t : (...) float or complex

        Returns
        -------
        (...) complex128
        """
        if self.amplitudes is None:
            raise RuntimeError("Call fit() before evaluate().")
        t   = np.asarray(t)
        out = np.zeros(t.shape, dtype=np.complex128)
        for a, s in zip(self.amplitudes, self.exponents):
            out += a * np.exp(s * t)
        return out

    def residual_rms(self, samples: np.ndarray, dt: float) -> float:
        """Root-mean-square residual between fitted model and original samples."""
        t_vals = np.arange(len(samples), dtype=np.float64) * dt
        f_fit  = self.evaluate(t_vals)
        return float(np.sqrt(np.mean(np.abs(f_fit - samples) ** 2)))
