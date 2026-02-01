"""Free-space scalar Green's function.

Convention: exp(-j*omega*t) time dependence.
g(r, r') = exp(-j*k*R) / (4*pi*R),  R = |r - r'|.
"""

import numpy as np


def scalar_green(
    k: float,
    r: np.ndarray,
    r_prime: np.ndarray,
) -> np.ndarray:
    """Evaluate the free-space scalar Green's function.

    Parameters
    ----------
    k : float
        Wavenumber (rad/m).
    r : ndarray, shape (..., 3)
        Observation point(s).
    r_prime : ndarray, shape (..., 3)
        Source point(s).

    Returns
    -------
    g : ndarray (complex128)
        Green's function value(s), same broadcast shape as inputs.
    """
    r = np.asarray(r, dtype=np.float64)
    r_prime = np.asarray(r_prime, dtype=np.float64)

    diff = r - r_prime
    R = np.sqrt(np.sum(diff**2, axis=-1))

    # Avoid division by zero; caller is responsible for singular cases
    R = np.maximum(R, 1e-30)

    g = np.exp(-1j * k * R) / (4.0 * np.pi * R)
    return g
