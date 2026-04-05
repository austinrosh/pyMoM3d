"""Abstract base classes for Green's function backends and high-level GF objects.

Two-level design
----------------
GreensBackend (ABC)
    Low-level evaluation interface.  All backends (empymod, DCIM, tabulated,
    Strata) implement scalar_G, dyadic_G, and grad_G on fully-batched arrays.

GreensFunctionBase (ABC)
    High-level interface consumed by MoM operators.  Carries wavenumber and
    wave_impedance for the source layer.  Delegates evaluation calls to a
    GreensBackend instance.

Batching requirement
--------------------
All evaluation methods accept:
    r       : (..., 3) complex/float — observation points
    r_prime : (..., 3) complex/float — source points
and return arrays with the corresponding leading shape.  No per-point Python
loops are permitted inside any backend implementation.

Symmetry requirement
--------------------
All backends must satisfy G(r, r') = G(r', r) (reciprocity).  This is
enforced in the test suite and is required for MoM matrix symmetry Z_mn = Z_nm.

Why dyadic_G must be direct
---------------------------
The EFIE matrix element is
    Z_mn = j*omega*mu * integral( f_m · G_dyadic · f_n ) dS dS'
where G_dyadic = (I + nabla nabla / k^2) g.  The nabla-nabla term is large
and oscillatory; finite-difference approximation of scalar_G introduces O(h^2)
error that breaks Z_mn = Z_nm symmetry and degrades matrix conditioning.
Each backend must provide dyadic_G analytically or from its own spectral
representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GreensBackend(ABC):
    """Low-level Green's function evaluation backend.

    All multilayer GF backends implement this interface.  Operators never
    import concrete backend classes — they receive a GreensFunctionBase whose
    backend property holds a GreensBackend instance.
    """

    @abstractmethod
    def scalar_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Scalar Green's function G(r, r').

        Parameters
        ----------
        r, r_prime : ndarray, shape (..., 3)
            Observation and source points (m).  Must support arbitrary leading
            batch dimensions.

        Returns
        -------
        ndarray, shape (...,), complex128
        """

    @abstractmethod
    def dyadic_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Dyadic (tensor) Green's function G_bar(r, r') = (I + nabla nabla / k^2) g.

        Must be provided directly by each backend — NOT approximated by finite
        differences of scalar_G.  See module docstring for why.

        Parameters
        ----------
        r, r_prime : ndarray, shape (..., 3)

        Returns
        -------
        ndarray, shape (..., 3, 3), complex128
            Full 3x3 dyadic tensor at each observation/source pair.
        """

    @abstractmethod
    def grad_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Gradient of scalar G with respect to r.

        Parameters
        ----------
        r, r_prime : ndarray, shape (..., 3)

        Returns
        -------
        ndarray, shape (..., 3), complex128
        """


class GreensFunctionBase(ABC):
    """High-level Green's function object consumed by MoM operators.

    Carries the medium parameters (wavenumber, wave_impedance) for the source
    layer and delegates all evaluation calls to an underlying GreensBackend.
    """

    @property
    @abstractmethod
    def wavenumber(self) -> complex:
        """Wavenumber k in the source layer (rad/m)."""

    @property
    @abstractmethod
    def wave_impedance(self) -> complex:
        """Intrinsic wave impedance eta in the source layer (Ω)."""

    @property
    @abstractmethod
    def backend(self) -> GreensBackend:
        """Active evaluation backend."""

    # Convenience delegation — operators call these, not the backend directly.
    def scalar_G(self, r: np.ndarray, r_prime: np.ndarray, **kwargs) -> np.ndarray:
        return self.backend.scalar_G(r, r_prime, **kwargs)

    def dyadic_G(self, r: np.ndarray, r_prime: np.ndarray, **kwargs) -> np.ndarray:
        return self.backend.dyadic_G(r, r_prime, **kwargs)

    def grad_G(self, r: np.ndarray, r_prime: np.ndarray, **kwargs) -> np.ndarray:
        return self.backend.grad_G(r, r_prime, **kwargs)
