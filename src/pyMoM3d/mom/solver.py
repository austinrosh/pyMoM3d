"""Linear system solvers for MoM."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def solve_direct(Z: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Solve ZI = V using direct LU factorization.

    Parameters
    ----------
    Z : ndarray, shape (N, N), complex128
        Impedance matrix.
    V : ndarray, shape (N,), complex128
        Voltage vector.

    Returns
    -------
    I : ndarray, shape (N,), complex128
        Current coefficients.
    """
    cond = np.linalg.cond(Z)
    logger.info(f"Condition number: {cond:.2e}")

    I = np.linalg.solve(Z, V)

    residual = np.linalg.norm(Z @ I - V) / np.linalg.norm(V)
    logger.info(f"Relative residual: {residual:.2e}")

    return I


def solve_gmres(
    Z: np.ndarray,
    V: np.ndarray,
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> np.ndarray:
    """Solve ZI = V using GMRES with diagonal preconditioner.

    Parameters
    ----------
    Z : ndarray, shape (N, N), complex128
        Impedance matrix.
    V : ndarray, shape (N,), complex128
        Voltage vector.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    I : ndarray, shape (N,), complex128
        Current coefficients.
    """
    from scipy.sparse.linalg import gmres, LinearOperator

    N = len(V)

    # Diagonal preconditioner
    diag = np.diag(Z)
    diag_inv = np.where(np.abs(diag) > 1e-30, 1.0 / diag, 1.0)

    M = LinearOperator((N, N), matvec=lambda x: diag_inv * x)

    I, info = gmres(Z, V, M=M, rtol=tol, maxiter=maxiter)

    if info != 0:
        logger.warning(f"GMRES did not converge (info={info})")

    residual = np.linalg.norm(Z @ I - V) / np.linalg.norm(V)
    logger.info(f"GMRES relative residual: {residual:.2e}")

    return I
