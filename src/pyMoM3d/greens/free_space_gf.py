"""Free-space Green's function implementing both GreensFunctionBase and GreensBackend.

Provides vectorised scalar_G, dyadic_G, and grad_G for the homogeneous
free-space (or homogeneous-medium) case.  Used as the default GF when no
LayerStack is provided, and as the reference term for multilayer singularity
decomposition.

Convention
----------
g(r, r') = exp(-j*k*R) / (4*pi*R),   R = |r - r'|
Time dependence: exp(-j*omega*t).
"""

from __future__ import annotations

import numpy as np

from .base import GreensBackend, GreensFunctionBase


class FreeSpaceGreensFunction(GreensFunctionBase, GreensBackend):
    """Free-space Green's function.

    Implements GreensFunctionBase (high-level) and GreensBackend (low-level)
    in one class for zero-overhead use in the free-space path.

    Parameters
    ----------
    k : complex
        Wavenumber (rad/m).
    eta : complex
        Intrinsic impedance (Ω).
    """

    def __init__(self, k: complex, eta: complex):
        self._k   = complex(k)
        self._eta = complex(eta)

    # ------------------------------------------------------------------
    # GreensFunctionBase interface
    # ------------------------------------------------------------------

    @property
    def wavenumber(self) -> complex:
        return self._k

    @property
    def wave_impedance(self) -> complex:
        return self._eta

    @property
    def backend(self) -> GreensBackend:
        return self   # self IS the backend

    # ------------------------------------------------------------------
    # GreensBackend interface — fully vectorised, no per-point loops
    # ------------------------------------------------------------------

    def scalar_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """g(r, r') = exp(-jkR)/(4*pi*R).

        Parameters
        ----------
        r, r_prime : (..., 3)

        Returns
        -------
        (...,) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        diff = r - r_prime
        R = np.sqrt(np.sum(diff ** 2, axis=-1))
        R = np.maximum(R, 1e-30)
        return np.exp(-1j * self._k * R) / (4.0 * np.pi * R)

    def grad_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """nabla_r g(r, r') = g(R) * (-jk - 1/R) * (r - r') / R.

        Parameters
        ----------
        r, r_prime : (..., 3)

        Returns
        -------
        (..., 3) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        diff = r - r_prime                            # (..., 3)
        R = np.sqrt(np.sum(diff ** 2, axis=-1))       # (...,)
        R = np.maximum(R, 1e-30)
        g = np.exp(-1j * self._k * R) / (4.0 * np.pi * R)  # (...,)
        factor = g * (-1j * self._k - 1.0 / R) / R         # (...,)
        return factor[..., np.newaxis] * diff               # (..., 3)

    def dyadic_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Dyadic G_bar = (I + nabla nabla / k^2) g, vectorised.

        For the free-space scalar Green's function:
            G_bar_ij = g(R) * [ (1 + jkR + (kR)^2) * delta_ij
                                 - (3 + 3jkR - (kR)^2) * R_i*R_j/R^2 ]
                        / (k^2 * R^2)  * k^2     <- after simplification

        In compact form:
            G_bar = A * I + B * R_hat R_hat
        where
            A = [1 + jkR - 1/(kR)^2] * g / (...)   <- derived below
            B = [-1 - 3jkR/... ] ...

        Derivation from nabla nabla g:
            nabla_i nabla_j g = g/R^2 * [
                ((-jk - 1/R)/R) * delta_ij
                + (3/R^2 + 3jk/R - k^2) * r_i r_j / R^2
                - (-jk - 1/R)/R * delta_ij        <- cancels with first
            ]

        Full analytical result (standard EM textbook):
            G_bar_ij = g * { [(1/R^2 - jk/R) * delta_ij]
                             + [(1 - 1/(kR)^2 + j/kR^3... ] }

        Using the standard dyadic form valid for R > 0:

            nabla nabla g = g * [
                  (-jk/R - 1/R^2) * I
                + (3/R^2 + 3jk/R - k^2) * r̂ r̂
            ] / 1    (times -1 from nabla_r, then sign from product rule)

        The correct expression is:
            nabla_r nabla_r' g(R) = -nabla_r nabla_r g(R)

        For EFIE the relevant dyadic is G_bar = (I + nabla nabla / k^2) g
        with nabla acting on r (observation):

            nabla_i nabla_j g = g(R)/R^2 * [
                (jkR + 1) * (hat_r_i * hat_r_j) * (-3/R^2 + jk^2/R ... )
            ]

        Using the well-known closed form:

            G_bar_ij = g(R) * {
                [1 + jkR + (kR)^2] / (kR)^2 * delta_ij / ...
            }

        Implemented as the standard dyadic GF (Chew, "Waves and Fields"):

            G_bar = g(R) * {
                [1 + 1/(jkR) - 1/(kR)^2] * I
              + [-1 - 3/(jkR) + 3/(kR)^2] * r̂ r̂
            }

        Parameters
        ----------
        r, r_prime : (..., 3)

        Returns
        -------
        (..., 3, 3) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        diff = r - r_prime          # (..., 3)
        R = np.sqrt(np.sum(diff ** 2, axis=-1))   # (...,)
        R = np.maximum(R, 1e-30)

        k  = self._k
        kR = k * R                  # (...,)
        g  = np.exp(-1j * k * R) / (4.0 * np.pi * R)  # (...,)

        r_hat = diff / R[..., np.newaxis]   # (..., 3)  unit vector

        # Scalar coefficients (from Chew, WAVES AND FIELDS, eq. for dyadic GF)
        #   A = g * (1 + 1/(jkR) - 1/(kR)^2)  multiplies I
        #   B = g * (-1 - 3/(jkR) + 3/(kR)^2)  multiplies r_hat r_hat
        inv_jkR   = 1.0 / (1j * kR)            # (...,)
        inv_kR2   = 1.0 / (kR ** 2)            # (...,)

        A = g * (1.0 + inv_jkR - inv_kR2)      # (...,)
        B = g * (-1.0 - 3.0 * inv_jkR + 3.0 * inv_kR2)  # (...,)

        # Build (..., 3, 3) tensor
        # A * I  +  B * r_hat ⊗ r_hat
        I3 = np.eye(3, dtype=np.complex128)
        # Outer product r_hat ⊗ r_hat:  (..., 3, 1) * (..., 1, 3)
        rr = r_hat[..., np.newaxis] * r_hat[..., np.newaxis, :]   # (..., 3, 3)

        return (A[..., np.newaxis, np.newaxis] * I3
                + B[..., np.newaxis, np.newaxis] * rr)
