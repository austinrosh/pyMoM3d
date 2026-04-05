"""Static (k→0) Green's function for PEC-backed dielectric substrates.

Provides the smooth correction (G_static - G_free_space) needed by the
singularity-decomposition assembly framework.  The free-space 1/(4πR)
singularity is handled by the existing Graglia extraction; this module
adds only the PEC ground-plane image terms.

Physics
-------
For a horizontal current source at z = z_cond above a PEC ground at z = z_pec:

**Vector potential G_A (inductance):**
    G_A_total = [1/R₀ + 1/R_img] / (4π)
    correction = +1/(4πR_img)

    The PEC image for horizontal current has the SAME sign (horizontal
    component preserved by reflection).  In the static limit, the
    dielectric interface does NOT modify G_A because the TE reflection
    coefficient vanishes as ω → 0.

**Scalar potential G_φ (capacitance, MPIE Formulation C):**
    G_φ_total = [1/R₀ - 1/R_img] / (4π)
    correction = -1/(4πR_img)

    The PEC image for charge has OPPOSITE sign.  The dielectric interface
    creates additional image terms via reflection coefficient
    k_c = (ε_r - 1)/(ε_r + 1), added as a convergent series.

where R_img = |r - r'_image|, and r'_image is the reflection of the
source through the PEC ground plane: r'_image = (x', y', 2·z_pec - z').

Note: This is the PEC image only.  The dielectric enhancement of the
capacitance (ε_eff correction) requires additional image series terms,
which can be added later for improved Z₀ accuracy.

References
----------
* Michalski, K. A. & Zheng, D. (1990). "Electromagnetic scattering and
  radiation by surfaces of arbitrary shape in layered media."
  IEEE Trans. Antennas Propag., 38(3), 335–344.
* Aksun, M. I. (1996). "A robust approach for the derivation of
  closed-form Green's functions." IEEE Trans. MTT, 44(5), 651–658.
"""

from __future__ import annotations

import numpy as np

from ..base import GreensBackend, GreensFunctionBase
from ...utils.constants import eta0


class StaticPECImageBackend(GreensBackend):
    """Backend providing PEC ground-plane image corrections.

    Parameters
    ----------
    z_pec : float
        z-coordinate of the PEC ground plane (m).
    z_conductor : float
        z-coordinate of the conductor surface (m).  Both source and
        observation are assumed to be at this height.
    eps_r : float
        Substrate relative permittivity.  Used for dielectric image
        series in G_φ.  Default 1.0 (no dielectric correction).
    n_dielectric_images : int
        Number of dielectric image pairs for G_φ.  Default 0 (PEC only).
        Set to ~10 for ε_r > 1 to include dielectric enhancement.
    """

    def __init__(
        self,
        z_pec: float,
        z_conductor: float,
        eps_r: float = 1.0,
        n_dielectric_images: int = 0,
    ):
        self.z_pec = float(z_pec)
        self.z_conductor = float(z_conductor)
        self.eps_r = float(eps_r)
        self.h_sub = abs(z_conductor - z_pec)
        self.n_diel = int(n_dielectric_images)

        # Dielectric reflection coefficient for image series
        if eps_r > 1.0 and self.n_diel > 0:
            self.k_c = (eps_r - 1.0) / (eps_r + 1.0)
        else:
            self.k_c = 0.0

    def _image_R(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Distance from observation point to PEC image of source.

        r'_image = (x', y', 2*z_pec - z')
        """
        diff = r - r_prime
        # z-component changes: z_obs - z_img = z_obs - (2*z_pec - z_src)
        diff[..., 2] = r[..., 2] - (2.0 * self.z_pec - r_prime[..., 2])
        return np.sqrt(np.sum(diff**2, axis=-1))

    def _image_R_at_depth(
        self, r: np.ndarray, r_prime: np.ndarray, z_img: float,
    ) -> np.ndarray:
        """Distance from observation to image at specified z."""
        dx = r[..., 0] - r_prime[..., 0]
        dy = r[..., 1] - r_prime[..., 1]
        dz = r[..., 2] - z_img
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def scalar_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Smooth correction for G_φ: PEC image + optional dielectric series.

        Returns G_φ_static - G_fs = correction terms only.
        """
        # PEC image: charge at z_img = 2*z_pec - z_src, sign = -1
        R_img = self._image_R(r, r_prime)
        R_img = np.maximum(R_img, 1e-30)
        result = -1.0 / (4.0 * np.pi * R_img)

        # Dielectric image series for G_φ
        # Images come in pairs at distances 2*m*h from the source plane.
        # Net contribution per pair: -k_c^{m-1} * (1-k_c) / (4π * R_m)
        # For m=1 (PEC image only): this gives -(1-k_c)/(4πR₁) = -(1-0)/(4πR₁)
        # which matches the PEC-only result when k_c=0.
        if self.k_c != 0.0 and self.n_diel > 0:
            h = self.h_sub
            rho_sq = ((r[..., 0] - r_prime[..., 0])**2
                      + (r[..., 1] - r_prime[..., 1])**2)

            # The PEC-only result is -1/(4πR₁) with R₁ = √(ρ² + 4h²).
            # The dielectric series replaces this with:
            # -(1-k_c) * Σ_{m=1}^M k_c^{m-1} / (4π √(ρ² + (2mh)²))
            #
            # So the additional correction beyond PEC is:
            # [-(1-k_c)*1/R₁ + 1/R₁] + -(1-k_c)*Σ_{m=2}^M k_c^{m-1}/R_m
            # = k_c/R₁ - (1-k_c)*Σ_{m=2}^M k_c^{m-1}/R_m
            #
            # But it's cleaner to compute the full series and subtract the
            # free-space term, rather than adding to the PEC-only result.

            # Recompute: full correction = -(1-k_c) * Σ_{m=1}^M k_c^{m-1}/R_m
            result = np.zeros_like(R_img, dtype=np.complex128)
            kc_pow = 1.0  # k_c^{m-1}
            for m in range(1, self.n_diel + 1):
                d_m = 2.0 * m * h
                R_m = np.sqrt(rho_sq + d_m**2)
                R_m = np.maximum(R_m, 1e-30)
                result -= (1.0 - self.k_c) * kc_pow / (4.0 * np.pi * R_m)
                kc_pow *= self.k_c

        return result.astype(np.complex128) if not np.iscomplexobj(result) else result

    def dyadic_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Smooth correction for G_A: +1/(4πR_img) · I for horizontal components.

        PEC image for horizontal current has the same sign.
        In the static limit, TE reflection vanishes → no dielectric
        correction needed for G_A.
        """
        R_img = self._image_R(r, r_prime)
        R_img = np.maximum(R_img, 1e-30)
        g_img = 1.0 / (4.0 * np.pi * R_img)

        shape = r.shape[:-1] + (3, 3)
        result = np.zeros(shape, dtype=np.complex128)
        # Horizontal components: same sign as direct (PEC preserves horizontal)
        result[..., 0, 0] = g_img
        result[..., 1, 1] = g_img
        # Vertical component: opposite sign (PEC flips vertical current)
        result[..., 2, 2] = -g_img
        return result

    def grad_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Gradient of scalar G correction (not used in quasi-static mode)."""
        R_img = self._image_R(r, r_prime)
        R_img = np.maximum(R_img, 1e-30)
        diff = r.copy()
        diff[..., 2] = r[..., 2] - (2.0 * self.z_pec - r_prime[..., 2])
        diff[..., 0] = r[..., 0] - r_prime[..., 0]
        diff[..., 1] = r[..., 1] - r_prime[..., 1]
        return diff / (4.0 * np.pi * R_img[..., np.newaxis]**3)


class DualPECImageBackend(GreensBackend):
    """Backend for stripline: two parallel PEC ground planes.

    Uses the method of images for a source between PEC at z = z_bot and
    z = z_top.

    **G_φ (scalar potential / capacitance):**
        Full dual-PEC image series — converges because charge images
        alternate in sign (Dirichlet Green's function).  Truncated after
        ``n_orders`` image pairs per side.

        Image positions and signs:
        - Type A: z = z_bot + 2m·d + h, sign = +1  (m ≠ 0)
        - Type B: z = z_bot + 2m·d - h, sign = -1  (all m)

    **G_A (vector potential / inductance, horizontal components):**
        Uses only the single nearest-PEC-image term.  The full Neumann
        image series (all signs +1) diverges logarithmically between
        infinite parallel plates — its zero-mode grows without bound.
        The nearest-image approximation is finite and gives a physically
        reasonable partial inductance, with the understanding that the
        exact stripline inductance is set by the 2D cross-section solver,
        not the 3D MoM.

    Parameters
    ----------
    z_bot : float
        z-coordinate of the bottom PEC plane.
    z_top : float
        z-coordinate of the top PEC plane.
    z_conductor : float
        z-coordinate of the strip conductor (between z_bot and z_top).
    eps_r : float
        Substrate relative permittivity.
    n_orders : int
        Number of image orders per side for G_φ. Default 20.
    """

    def __init__(
        self,
        z_bot: float,
        z_top: float,
        z_conductor: float,
        eps_r: float = 1.0,
        n_orders: int = 20,
    ):
        self.z_bot = float(z_bot)
        self.z_top = float(z_top)
        self.z_conductor = float(z_conductor)
        self.eps_r = float(eps_r)
        self.d = self.z_top - self.z_bot
        self.h = self.z_conductor - self.z_bot

        # Nearest PEC plane for G_A (single image)
        dist_bot = abs(self.z_conductor - self.z_bot)
        dist_top = abs(self.z_top - self.z_conductor)
        self._z_pec_nearest = self.z_bot if dist_bot <= dist_top else self.z_top

        # Precompute G_φ image z-positions and charge signs analytically
        img_z = []
        img_sign_phi = []

        d = self.d
        h = self.h
        z_b = self.z_bot

        for m in range(-n_orders, n_orders + 1):
            # Type A: z = z_bot + 2m*d + h, charge sign = +1
            if m != 0:  # m=0 is the source, excluded
                img_z.append(z_b + 2.0 * m * d + h)
                img_sign_phi.append(+1.0)

            # Type B: z = z_bot + 2m*d - h, charge sign = -1
            img_z.append(z_b + 2.0 * m * d - h)
            img_sign_phi.append(-1.0)

        self._img_z = np.array(img_z)
        self._img_sign_phi = np.array(img_sign_phi)
        self._n_images = len(img_z)

    def scalar_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """G_φ correction: full dual-PEC image series (converges)."""
        result = np.zeros(r.shape[:-1], dtype=np.complex128)
        dx = r[..., 0] - r_prime[..., 0]
        dy = r[..., 1] - r_prime[..., 1]
        rho_sq = dx**2 + dy**2

        for i in range(self._n_images):
            dz = r[..., 2] - self._img_z[i]
            R = np.sqrt(rho_sq + dz**2)
            R = np.maximum(R, 1e-30)
            result += self._img_sign_phi[i] / (4.0 * np.pi * R)

        return result

    def dyadic_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """G_A correction: nearest-PEC single image (finite approximation).

        The full dual-PEC image series for horizontal current diverges
        (all +1 signs → Neumann zero-mode).  Use only the nearest ground
        plane image, same as the single-PEC microstrip case.
        """
        # Single image in nearest PEC plane
        z_img = 2.0 * self._z_pec_nearest - r_prime[..., 2]
        dx = r[..., 0] - r_prime[..., 0]
        dy = r[..., 1] - r_prime[..., 1]
        dz = r[..., 2] - z_img
        R_img = np.sqrt(dx**2 + dy**2 + dz**2)
        R_img = np.maximum(R_img, 1e-30)
        g_img = 1.0 / (4.0 * np.pi * R_img)

        shape = r.shape[:-1] + (3, 3)
        result = np.zeros(shape, dtype=np.complex128)
        result[..., 0, 0] = g_img
        result[..., 1, 1] = g_img
        result[..., 2, 2] = -g_img  # vertical: opposite sign
        return result

    def grad_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Gradient of scalar G correction (G_φ series)."""
        result = np.zeros(r.shape, dtype=np.complex128)
        dx = r[..., 0] - r_prime[..., 0]
        dy = r[..., 1] - r_prime[..., 1]
        rho_sq = dx**2 + dy**2

        for i in range(self._n_images):
            dz = r[..., 2] - self._img_z[i]
            R = np.sqrt(rho_sq + dz**2)
            R = np.maximum(R, 1e-30)
            coeff = self._img_sign_phi[i] / (4.0 * np.pi * R**3)
            result[..., 0] += coeff * dx
            result[..., 1] += coeff * dy
            result[..., 2] += coeff * dz

        return result


class StaticLayeredGF(GreensFunctionBase):
    """Static Green's function for PEC-backed dielectric.

    Wraps :class:`StaticPECImageBackend` and presents the
    :class:`GreensFunctionBase` interface expected by MoM operators.

    The ``wavenumber`` is set to a very small value (not zero) to avoid
    division-by-zero in assembly prefactors.  The assembly with this
    tiny k produces the correct static integrals because
    exp(-jkR) ≈ 1 when k → 0.

    Parameters
    ----------
    layer_stack : LayerStack
        Defines the PEC ground, dielectric, and air layers.
    source_layer_name : str, optional
        Name of the source layer.  Default: first non-PEC layer.
    n_dielectric_images : int
        Number of dielectric image terms for G_φ.  Default 0.
    """

    # Very small wavenumber for assembly (exp(-jkR) ≈ 1)
    _K_REF = 1e-10

    def __init__(
        self,
        layer_stack,
        source_layer_name: str = None,
        n_dielectric_images: int = 0,
    ):
        self._stack = layer_stack

        # Find ALL PEC layers and source layer
        pec_layers = []
        src_layer = None
        for layer in layer_stack.layers:
            if getattr(layer, 'is_pec', False):
                pec_layers.append(layer)
            if source_layer_name and layer.name == source_layer_name:
                src_layer = layer

        if not pec_layers:
            raise ValueError("StaticLayeredGF requires a PEC ground layer")

        if src_layer is None:
            # Default: first non-PEC layer
            for layer in layer_stack.layers:
                if not getattr(layer, 'is_pec', False):
                    src_layer = layer
                    break

        # Conductor is at the interface between source layer and the
        # next layer above it (typically: top of substrate)
        z_conductor = src_layer.z_top  # strip at top of dielectric
        eps_r = src_layer.eps_r

        # Classify PEC planes relative to the conductor
        pec_below = [
            L for L in pec_layers
            if np.isfinite(L.z_top) and L.z_top <= z_conductor + 1e-15
        ]
        pec_above = [
            L for L in pec_layers
            if np.isfinite(L.z_bot) and L.z_bot >= z_conductor - 1e-15
        ]

        if pec_below and pec_above:
            # Stripline: PEC on both sides → dual image series
            z_bot = max(L.z_top for L in pec_below)
            z_top = min(L.z_bot for L in pec_above)
            self._backend = DualPECImageBackend(
                z_bot=z_bot,
                z_top=z_top,
                z_conductor=z_conductor,
                eps_r=eps_r,
                n_orders=20,
            )
        elif pec_below:
            # Microstrip: single PEC below
            z_pec = max(L.z_top for L in pec_below)
            self._backend = StaticPECImageBackend(
                z_pec=z_pec,
                z_conductor=z_conductor,
                eps_r=eps_r,
                n_dielectric_images=n_dielectric_images,
            )
        else:
            # PEC above only — unusual but handle it
            z_pec = min(L.z_bot for L in pec_above)
            self._backend = StaticPECImageBackend(
                z_pec=z_pec,
                z_conductor=z_conductor,
                eps_r=eps_r,
                n_dielectric_images=n_dielectric_images,
            )

    @property
    def wavenumber(self) -> complex:
        return self._K_REF

    @property
    def wave_impedance(self) -> complex:
        return eta0

    @property
    def backend(self) -> GreensBackend:
        return self._backend
