"""DCIM (Discrete Complex Image Method) backend — Phase 2 simplified.

Approximates the smooth Sommerfeld correction G_ML - G_fs as a sum of complex
image terms.  Each image term has the same form as the free-space Green's
function but evaluated at a complex (mirrored) source point and scaled by a
reflection coefficient.

Phase 2 scope
-------------
Supports single-interface (two-layer) stacks via a single quasi-static image:

    G_smooth(r, r') ≈ Γ · G_fs(r, r'_img)

where r'_img = (x', y', 2·z_iface - z') is the mirror of the source through the
interface, and Γ = (ε₁ - ε₂) / (ε₁ + ε₂) is the quasi-static reflection
coefficient (ε₁ = source layer, ε₂ = adjacent layer).

This single-image approximation captures the dominant substrate effect with
O(1) evaluations per point in fully vectorized NumPy — no per-point Python loops.

For three-or-more layer stacks, ``DCIMBackend`` raises ``NotImplementedError``
and the ``LayeredGreensFunction`` dispatcher falls back to the empymod backend.

Future work
-----------
Phase 3 will replace the static-Γ single image with full GPOF coefficient
fitting over the Sommerfeld integrand sampled along a two-level deformation
path.  GPOF extracts K complex exponentials {aᵢ, sᵢ}, converted to complex
image distances via the Sommerfeld identity.  This will support arbitrary N-layer
stacks and achieve < 1e-4 relative error versus exact Sommerfeld integrals.
"""

from __future__ import annotations

import numpy as np

from ..base import GreensBackend
from ..free_space_gf import FreeSpaceGreensFunction
from ...medium.layer_stack import Layer, LayerStack


class DCIMBackend(GreensBackend):
    """Single-interface image approximation for two-layer stacks.

    Returns the smooth correction G_ML - G_fs (NOT the full G_ML).  The
    free-space singular term is handled upstream by the Graglia extraction
    and explicit G_fs quadrature in ``MultilayerEFIEOperator``.

    Parameters
    ----------
    layer_stack : LayerStack
        Must have exactly one or two layers.  Three-or-more raises
        ``NotImplementedError``.
    frequency : float
        Operating frequency (Hz).
    source_layer : Layer
        Layer containing the MoM mesh.

    Raises
    ------
    NotImplementedError
        If ``layer_stack`` has more than two layers (Phase 2 limitation).
    """

    def __init__(
        self,
        layer_stack: LayerStack,
        frequency: float,
        source_layer: Layer,
    ):
        n_layers = len(layer_stack.layers)
        if n_layers > 2:
            raise NotImplementedError(
                f"DCIMBackend Phase 2 supports only one or two layers; "
                f"got {n_layers}.  Use backend='empymod' for N-layer stacks."
            )

        self._freq  = float(frequency)
        self._omega = 2.0 * np.pi * self._freq
        self._src_layer = source_layer

        k   = source_layer.wavenumber(self._omega)
        eta = source_layer.wave_impedance(self._omega)
        self._k    = complex(k)
        self._fs_gf = FreeSpaceGreensFunction(k=k, eta=eta)

        if n_layers == 1:
            # Single unbounded layer — no interface, no correction
            self._gamma  = 0.0 + 0.0j
            self._z_iface = 0.0  # unused
        else:
            # Two layers: find the interface and reflection coefficient
            self._gamma, self._z_iface = self._compute_image_params(
                layer_stack, source_layer
            )

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _compute_image_params(
        self,
        layer_stack: LayerStack,
        source_layer: Layer,
    ):
        """Compute quasi-static reflection coefficient and interface z.

        The static image coefficient for a point source in medium 1 above a
        planar interface with medium 2 is:

            Γ = (ε₁ - ε₂) / (ε₁ + ε₂)

        where ε₁, ε₂ are complex effective permittivities.  This captures the
        dominant substrate polarization effect.  The dynamic correction (k·d
        dependent terms) is deferred to Phase 3 GPOF fitting.
        """
        layers = layer_stack.layers
        src_idx = next(
            i for i, lyr in enumerate(layers) if lyr.name == source_layer.name
        )

        # Find the nearest interface below (or above) the source layer
        if src_idx > 0:
            # Adjacent layer is below the source layer
            adj_layer = layers[src_idx - 1]
            z_iface   = source_layer.z_bot
        else:
            # Source is the bottommost layer; adjacent is above
            adj_layer = layers[src_idx + 1]
            z_iface   = source_layer.z_top

        eps1 = source_layer.eps_r_eff(self._omega)
        eps2 = adj_layer.eps_r_eff(self._omega)
        gamma = (eps1 - eps2) / (eps1 + eps2)
        return complex(gamma), float(z_iface)

    # ------------------------------------------------------------------
    # GreensBackend interface
    # ------------------------------------------------------------------

    def scalar_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Smooth correction G_smooth = Γ · G_fs(r, r'_img).

        For single-layer stacks returns zeros (no interface, no correction).

        Parameters
        ----------
        r, r_prime : (..., 3) float64
            Observation and source points (batched).

        Returns
        -------
        (...,) complex128
            Smooth correction G_ML - G_fs.
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)

        if self._gamma == 0.0:
            return np.zeros(r.shape[:-1], dtype=np.complex128)

        r_img = self._mirror(r_prime)
        return self._gamma * self._fs_gf.scalar_G(r, r_img)

    def dyadic_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Smooth correction for dyadic GF: Γ · G̅_fs(r, r'_img).

        Uses the same scalar reflection coefficient for all tensor components
        (single-coefficient approximation; adequate for Phase 2 validation).

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

        if self._gamma == 0.0:
            return np.zeros(shape + (3, 3), dtype=np.complex128)

        r_img = self._mirror(r_prime)
        return self._gamma * self._fs_gf.dyadic_G(r, r_img)

    def grad_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Gradient of smooth correction w.r.t. r: Γ · ∇G_fs(r, r'_img).

        Parameters
        ----------
        r, r_prime : (..., 3) float64

        Returns
        -------
        (..., 3) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]

        if self._gamma == 0.0:
            return np.zeros(shape + (3,), dtype=np.complex128)

        r_img = self._mirror(r_prime)
        return self._gamma * self._fs_gf.grad_G(r, r_img)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mirror(self, r_prime: np.ndarray) -> np.ndarray:
        """Mirror source points through the interface plane.

        For interface at z = z_iface:
            r'_img = (x', y', 2·z_iface - z')

        Parameters
        ----------
        r_prime : (..., 3) float64

        Returns
        -------
        (..., 3) float64
        """
        r_img = r_prime.copy()
        r_img[..., 2] = 2.0 * self._z_iface - r_prime[..., 2]
        return r_img

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> complex:
        """Quasi-static reflection coefficient."""
        return self._gamma

    @property
    def z_interface(self) -> float:
        """Interface z-coordinate used for image mirroring."""
        return self._z_iface
