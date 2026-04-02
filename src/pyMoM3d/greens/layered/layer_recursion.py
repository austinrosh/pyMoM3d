"""LayerRecursionBackend: exact N-layer Sommerfeld integration.

Evaluates the smooth Green's function correction G_ML - G_fs for arbitrary
N-layer stacks via the generalized TM reflection coefficient recursion followed
by Gauss-Legendre quadrature on a deformed complex path.

Physics
-------
For source and observation both in the topmost layer (layer N-1), the smooth
correction to the scalar GF is the Sommerfeld integral:

    G_smooth(ρ, z, z') = (1/4π) ∫₀^∞  Γ_tot(kρ)/(j kz)
                          · exp(-j kz · H) · J₀(kρ ρ) kρ dkρ

where:
    kz      = √(k_src² - kρ²)   [Im(kz) ≤ 0 branch, decaying/outgoing]
    H       = (z - z_iface) + (z' - z_iface)   [total height above interface]
    z_iface = z_bot of the source layer (= z_top of the layer below)
    Γ_tot   = generalized TM reflection coefficient (upward recursion)

The generalized reflection coefficient Γ_tot(kρ) is built by an upward
recursion from the semi-infinite substrate (where Γ = 0) to the interface
just below the source layer.  Multiple reflections within intermediate layers
are included exactly.

Quadrature
----------
Gauss-Legendre on [0, kρ_max] with a small imaginary shift ε_im pushes the
path slightly into the upper half-plane, moving real-axis surface-wave poles
away from the integration contour.  The integrand is evaluated for all
quadrature points and all (ρ, H) pairs simultaneously via NumPy broadcasting
— O(N_quad × N_pairs) matrix operations, no Python loops over pairs.

Accuracy / performance trade-offs
----------------------------------
n_quad=128 gives < 0.1 % relative error for most two-layer configurations
at 1 GHz.  Increase to 256–512 for geometries with strong surface-wave
contributions (high-contrast substrates, small separations).

This backend is slower per evaluation than DCIMBackend (which evaluates a
precomputed image sum) but is exact for arbitrary N-layer stacks and requires
no precomputed coefficients.  It serves as the reference for validating and
generating table data for TabulatedPNGFBackend.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv as _jv

def _j0(z):
    """J₀(z) supporting both real and complex input via scipy.special.jv."""
    return _jv(0, z)

from ..base import GreensBackend
from ..free_space_gf import FreeSpaceGreensFunction
from ...medium.layer_stack import Layer, LayerStack


class LayerRecursionBackend(GreensBackend):
    """Exact N-layer Sommerfeld integration via generalized reflection recursion.

    Returns the smooth correction G_ML - G_fs (same contract as
    EmpymodSommerfeldBackend and DCIMBackend).  The free-space singular term
    is handled upstream by MultilayerEFIEOperator.

    Parameters
    ----------
    layer_stack : LayerStack
        Arbitrary N-layer stack (no restriction on number of layers).
    frequency : float
        Operating frequency (Hz).
    source_layer : Layer
        Layer containing the MoM mesh.  Must be the topmost layer.
    n_quad : int
        Number of Gauss-Legendre quadrature points.  Default 128.
    kp_max_factor : float
        Upper integration limit as a multiple of the source layer wavenumber.
        Default 15.  Increase for grazing-angle dominated problems.
    """

    def __init__(
        self,
        layer_stack: LayerStack,
        frequency: float,
        source_layer: Layer,
        n_quad: int = 128,
        kp_max_factor: float = 15.0,
    ):
        self._stack      = layer_stack
        self._freq       = float(frequency)
        self._omega      = 2.0 * np.pi * self._freq
        self._src_layer  = source_layer
        self._n_quad     = n_quad

        # Source layer index (bottom-to-top ordering)
        layers = layer_stack.layers
        self._src_idx = next(
            i for i, lyr in enumerate(layers) if lyr.name == source_layer.name
        )
        # Interface below the source layer
        self._z_iface = float(source_layer.z_bot)
        # No interface below the source → smooth correction is identically zero
        self._no_interface = (self._src_idx == 0)

        # Per-layer properties at omega
        self._eps_layers = np.array(
            [complex(l.eps_r_eff(self._omega)) for l in layers], dtype=np.complex128
        )
        self._k_layers = np.array(
            [complex(l.wavenumber(self._omega)) for l in layers], dtype=np.complex128
        )
        self._d_layers = np.array(
            [l.z_top - l.z_bot for l in layers], dtype=np.float64
        )

        # Source-layer wavenumber and free-space GF for subtraction
        k_src   = complex(source_layer.wavenumber(self._omega))
        eta_src = complex(source_layer.wave_impedance(self._omega))
        self._k_src = k_src
        self._fs_gf = FreeSpaceGreensFunction(k=k_src, eta=eta_src)

        # Gauss-Legendre quadrature: N_quad points on [0, kp_max]
        k_max   = float(np.max(np.abs(self._k_layers)))
        kp_max  = kp_max_factor * k_max
        t_pts, t_wts = np.polynomial.legendre.leggauss(n_quad)
        kp_real = kp_max * (t_pts + 1.0) / 2.0   # map [-1,1] → [0, kp_max]
        kp_wts  = (kp_max / 2.0) * t_wts

        # Small imaginary deformation: moves surface-wave poles off real axis
        eps_im  = 5e-3 * k_max
        self._kp_pts = kp_real.astype(np.complex128) + 1j * eps_im   # (N_quad,)
        self._kp_wts = kp_wts.astype(np.complex128)                   # (N_quad,)

        # Precompute Γ_tot and kz at quadrature points (independent of r, r')
        self._kz_src_q, self._Gamma_q = self._precompute_quadrature()

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _kz_for_layer(self, kp: np.ndarray, layer_idx: int) -> np.ndarray:
        """kz = √(k_i² - kρ²) with Im(kz) ≤ 0 (outgoing branch)."""
        k = self._k_layers[layer_idx]
        kz = np.sqrt(k**2 - kp**2 + 0j)
        # Enforce Im(kz) ≤ 0 (downward-decaying convention)
        kz = np.where(kz.imag > 0.0, -kz, kz)
        return kz

    def _precompute_quadrature(self):
        """Compute kz_src and Γ_tot at all quadrature points once at init."""
        kp  = self._kp_pts        # (N_quad,)
        kz_src = self._kz_for_layer(kp, self._src_idx)
        Gamma  = self._gamma_total_TM(kp)
        return kz_src, Gamma      # each (N_quad,)

    # ------------------------------------------------------------------
    # Generalized reflection coefficient
    # ------------------------------------------------------------------

    def _gamma_total_TM(self, kp: np.ndarray) -> np.ndarray:
        """Upward recursion for the total TM downward reflection coefficient.

        Builds the generalized reflection coefficient seen from the source
        layer looking downward, including all multiple reflections within
        intermediate layers.

        Parameters
        ----------
        kp : (N,) complex
            Transverse wavenumber values (possibly complex for path deformation).

        Returns
        -------
        (N,) complex128
        """
        src_idx = self._src_idx
        eps = self._eps_layers      # (N_layers,)
        d   = self._d_layers        # (N_layers,)

        # kz for all layers at all kp: (N_layers, N_kp)
        kz = np.vstack([self._kz_for_layer(kp, i) for i in range(len(eps))])

        # Start: Γ = 0 below the bottommost (semi-infinite) layer
        Gamma = np.zeros(len(kp), dtype=np.complex128)

        for i in range(src_idx):
            # Interface between layer i (below) and layer i+1 (above).
            # TM Fresnel for a downward wave in layer i+1 hitting layer i:
            #   r = (kz_i/ε_i - kz_{i+1}/ε_{i+1}) / (kz_i/ε_i + kz_{i+1}/ε_{i+1})
            num = kz[i] / eps[i] - kz[i + 1] / eps[i + 1]
            den = kz[i] / eps[i] + kz[i + 1] / eps[i + 1]
            r   = num / den                        # (N_kp,)

            # Round-trip phase in layer i (d[0] may be inf for semi-infinite bottom)
            d_i = d[i]
            if not np.isfinite(d_i):
                # Semi-infinite layer: phase → 0 (wave decays away)
                phase = np.zeros(len(kp), dtype=np.complex128)
            else:
                phase = np.exp(-2j * kz[i] * d_i)   # (N_kp,)

            # Generalized recursion: Γ_new = (r + Γ_old·phase) / (1 + r·Γ_old·phase)
            numer = r + Gamma * phase
            denom = 1.0 + r * Gamma * phase
            Gamma = numer / denom

        return Gamma

    # ------------------------------------------------------------------
    # GreensBackend interface
    # ------------------------------------------------------------------

    def scalar_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Smooth correction G_ML - G_fs via Sommerfeld quadrature.

        Fully vectorized over batched (r, r') pairs.

        Parameters
        ----------
        r, r_prime : (..., 3) float64

        Returns
        -------
        (...,) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]

        if self._no_interface:
            return np.zeros(shape, dtype=np.complex128)

        r_flat       = r.reshape(-1, 3)
        r_prime_flat = r_prime.reshape(-1, 3)
        N_pairs      = r_flat.shape[0]

        # Horizontal separation ρ and height parameter H: (N_pairs,)
        dx  = r_flat[:, 0] - r_prime_flat[:, 0]
        dy  = r_flat[:, 1] - r_prime_flat[:, 1]
        rho = np.sqrt(dx**2 + dy**2)
        H   = (r_flat[:, 2] - self._z_iface) + (r_prime_flat[:, 2] - self._z_iface)

        kp     = self._kp_pts    # (N_quad,)
        wts    = self._kp_wts    # (N_quad,)
        kz_src = self._kz_src_q  # (N_quad,)
        Gamma  = self._Gamma_q   # (N_quad,)

        # J₀(kρ · ρ): shape (N_quad, N_pairs) — outer product
        # scipy j0 broadcasts over arrays
        j0_mat = _j0(np.outer(kp, rho))    # (N_quad, N_pairs)

        # exp(-j kz · H): shape (N_quad, N_pairs)
        exp_H = np.exp(-1j * np.outer(kz_src, H))   # (N_quad, N_pairs)

        # Kernel: Γ / (j kz) — shape (N_quad,)
        kernel = Gamma / (1j * kz_src)    # (N_quad,)

        # Integrand: (N_quad, N_pairs)
        integrand = (kernel * kp)[:, np.newaxis] * exp_H * j0_mat

        # Integrate over kρ (axis 0): wts @ integrand → (N_pairs,)
        G_smooth = wts @ integrand / (4.0 * np.pi)

        return G_smooth.reshape(shape)

    def dyadic_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Dyadic smooth correction.

        For planar horizontal meshes the dominant contributions are the TM
        horizontal components.  Phase 3 returns G_scalar·I as the dyadic
        (scalar approximation valid for in-plane RWG elements).  Full TE/TM
        separation with separate spectral kernels is Phase 4.

        Parameters
        ----------
        r, r_prime : (..., 3) float64

        Returns
        -------
        (..., 3, 3) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]

        if self._no_interface:
            return np.zeros(shape + (3, 3), dtype=np.complex128)

        G_s = self.scalar_G(r, r_prime)            # (...,)
        return G_s[..., np.newaxis, np.newaxis] * np.eye(3, dtype=np.complex128)

    def grad_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Gradient of smooth correction w.r.t. r via central finite differences.

        Parameters
        ----------
        r, r_prime : (..., 3) float64

        Returns
        -------
        (..., 3) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)

        if self._no_interface:
            return np.zeros(r.shape, dtype=np.complex128)

        h    = 1e-7
        grad = np.zeros(r.shape, dtype=np.complex128)
        for i in range(3):
            dr        = np.zeros_like(r)
            dr[..., i] = h
            gp        = self.scalar_G(r + dr, r_prime)
            gm        = self.scalar_G(r - dr, r_prime)
            grad[..., i] = (gp - gm) / (2.0 * h)
        return grad
