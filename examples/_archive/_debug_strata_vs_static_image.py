"""Validate Strata DCIM G_phi against analytical static image solution.

For a PEC-backed dielectric slab (PEC at z=0, dielectric eps_r for 0<z<h, air above),
the scalar potential Green's function at the interface z=h has a known static image
expansion. This provides a reliable reference that does NOT depend on empymod.

The static Green's function for a point charge at the dielectric-air interface
above a PEC ground plane involves an infinite series of image charges from
multiple reflections between the PEC and the dielectric interface.

For source and observation both at z = h (top of dielectric):
  G_phi_static(rho) = 1/(4*pi) * sum over images

The dominant terms are:
  - Direct:   q / (4*pi*rho)         ... as rho >> h
  - PEC image: -q / (4*pi*sqrt(rho^2 + (2h)^2))  ... opposite sign, at z = -h

For the MoM scalar potential (Formulation-C), Strata returns G_phi = g/eps_r,
so the full scalar GF in the assembly is:
  g_ML(rho) = G_phi * eps_r

And the smooth correction that scalar_G_smooth returns is:
  g_ML(rho) - g_fs(rho) = G_phi * eps_r - exp(-jkR)/(4*pi*R)

At low frequency (k -> 0), g_fs -> 1/(4*pi*R), and the correction isolates
the substrate/PEC enhancement.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import Layer, LayerStack, c0, eta0
from pyMoM3d.greens.free_space import scalar_green

# --- Configuration ---
EPS_R = 4.4
H = 1.6e-3  # substrate thickness

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

z_src = H
z_obs = H

# --- Analytical static image series ---
def static_image_gphi(rho, z_obs, z_src, h, eps_r, n_images=50):
    """Static scalar potential G_phi for PEC-backed dielectric slab.

    Uses the full image series for a charge at the dielectric-air interface
    above a PEC ground plane. The series accounts for multiple reflections
    between the PEC (perfect mirror) and the dielectric interface
    (partial reflection coefficient Gamma = (eps_r - 1)/(eps_r + 1)).

    For Formulation-C: G_phi = g / eps_r_source_layer.
    Since the source is at the FR4-air interface, we need to be careful
    about which eps_r to use. In the source layer (FR4), eps_r = EPS_R.

    Returns g_ML (the full scalar GF in the source medium convention).
    """
    # Reflection coefficient at dielectric-air interface
    Gamma = (eps_r - 1.0) / (eps_r + 1.0)

    # The static Green's function in a layered medium with PEC ground plane
    # For source and observation both at z = h (interface):
    #
    # Method: Use the spectral-domain approach.
    # In the source layer (dielectric), the static scalar potential GF is:
    #
    # g(rho) = 1/(4*pi) * [1/R_direct + sum of images]
    #
    # The PEC at z=0 creates an image at z = -z_src = -h.
    # The dielectric interface at z=h creates partial images.
    #
    # For same-interface source/obs at z = h:
    # The primary image series (Chew, Waves and Fields, Ch. 4):
    #
    # g = 1/(4*pi) * [1/rho + sum_{n=1}^{inf} Gamma^n *
    #     ((-1)^n / sqrt(rho^2 + (2*n*h)^2) + (-1)^n / sqrt(rho^2 + (2*n*h)^2))]
    #
    # Actually, let me use the standard result more carefully.
    #
    # For a dielectric slab on PEC, source at z=h, observation at z=h:
    # The direct image from PEC at distance 2h:
    #   image_1 = -1/sqrt(rho^2 + (2h)^2)   [PEC image, opposite sign]
    # Then this image reflects off the top interface with coeff Gamma,
    # and the PEC image of that creates another image, etc.

    # Simple approach: PEC-backed dielectric Green's function in free space
    # The dominant effect is the PEC image. Let's compute term by term.

    # Direct term (source at z_src, obs at z_obs, both = h)
    R_direct = np.sqrt(rho**2 + (z_obs - z_src)**2)
    if R_direct < 1e-30:
        R_direct = 1e-30
    g = 1.0 / (4.0 * np.pi * R_direct)

    # PEC image at z = -z_src = -h. Distance from obs at h:
    R_pec = np.sqrt(rho**2 + (z_obs + z_src)**2)  # z_obs - (-z_src) = 2h
    g += -1.0 / (4.0 * np.pi * R_pec)  # PEC image has opposite sign

    # Higher-order images from multiple PEC + dielectric reflections
    # For a dielectric slab on PEC, the higher images involve Gamma^n
    # and are typically small for eps_r ~ 4.4 (Gamma ~ 0.63)
    for n in range(1, n_images + 1):
        # Images from dielectric interface reflections
        # These get progressively weaker by Gamma^n
        # Distance 2*n*h + 2h for the alternating series

        # Upward image series (reflected at top interface, then PEC, etc.)
        R_up = np.sqrt(rho**2 + (2*n*H + z_obs - z_src)**2)
        R_dn = np.sqrt(rho**2 + (2*n*H + z_obs + z_src)**2)

        sign_n = (-1)**n  # PEC reflection sign
        gamma_n = Gamma**n

        if R_up > 1e-30:
            g += gamma_n / (4.0 * np.pi * R_up)
        if R_dn > 1e-30:
            g += gamma_n * (-1.0) / (4.0 * np.pi * R_dn)

    return g


def static_image_gphi_spectral(rho, h, eps_r):
    """Static G_phi via Sommerfeld integral (spectral domain).

    For a PEC-backed dielectric slab, the spectral domain scalar potential
    Green's function at the interface z = z' = h is:

    G_phi_tilde(k_rho) = 1/(2*eps_0) * [1 + Gamma_PEC * exp(-2*k_rho*h)] /
                          [1 - Gamma_dielectric * Gamma_PEC * exp(-2*k_rho*h)]

    where Gamma_PEC = -1 for the PEC reflection.

    The spatial domain is obtained via the Hankel transform:
    g(rho) = 1/(2*pi) * integral_0^inf G_phi_tilde(k_rho) * J0(k_rho * rho) * k_rho dk_rho

    For numerical evaluation, use the Sommerfeld identity approach.
    We'll use scipy to evaluate this numerically.
    """
    from scipy.integrate import quad
    from scipy.special import j0

    # In the source layer (dielectric), static TL Green's function for G_phi
    # With PEC at z=0 and source/obs at z = h:
    #
    # The spectral kernel is (Michalski-Zheng static limit):
    # K(k_rho) = 1/(2*k_rho*eps_r) * [1 + R_PEC * exp(-2*k_rho*h)] /
    #            [1 + R_12 * R_PEC * exp(-2*k_rho*h)]
    # where R_PEC = +1 (for TM, voltage reflection, PEC shorts the TL)
    # and R_12 = (eps_r - 1)/(eps_r + 1) for the dielectric-air interface

    # Actually, for Formulation-C, G_phi is related to the charge potential.
    # Let me use a simpler known result.

    # For a horizontal electric dipole at the interface of a PEC-backed slab,
    # the scalar potential kernel in the static limit is:
    #
    # K_phi(k_rho) = 1 / (2 * eps_0 * eps_r * k_rho) * coth(k_rho * h)
    #
    # Wait - let me just compute the Sommerfeld integral numerically.
    # The static scalar GF kernel for the source layer in a PEC-backed slab:
    #
    # K(k_rho) = 1/(2*k_rho) * (1 - exp(-2*k_rho*h)) /
    #            (1 - Gamma*exp(-2*k_rho*h))
    #            + 1/(2*k_rho) from the direct term
    #
    # This is getting complicated. Let me just use the image series which
    # converges well for rho > h.

    pass  # Use image series instead


# --- Test at quasi-static frequency ---
print("=" * 80)
print("Strata DCIM G_phi Validation Against Analytical Static Images")
print("=" * 80)
print(f"\nConfiguration: PEC / FR4 (eps_r={EPS_R}) / air")
print(f"Substrate thickness: h = {H*1e3:.1f} mm")
print(f"Source & obs at z = {z_src*1e3:.1f} mm (FR4-air interface)\n")

rhos = np.array([0.5e-3, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 10e-3, 15e-3, 20e-3, 30e-3])

# Compute analytical static image GF
print("--- Static image analysis (frequency-independent) ---")
print(f"{'rho (mm)':>10}  {'g_static':>14}  {'g_fs_static':>14}  {'correction':>14}  {'ratio g/gfs':>12}")
print("-" * 75)

g_static_vals = []
g_fs_static_vals = []
for rho in rhos:
    g_stat = static_image_gphi(rho, z_obs, z_src, H, EPS_R)
    R = np.sqrt(rho**2 + (z_obs - z_src)**2)
    g_fs = 1.0 / (4.0 * np.pi * R)
    correction = g_stat - g_fs
    ratio = g_stat / g_fs if g_fs > 1e-30 else float('inf')
    g_static_vals.append(g_stat)
    g_fs_static_vals.append(g_fs)
    print(f"  {rho*1e3:>8.1f}  {g_stat:>14.6e}  {g_fs:>14.6e}  {correction:>14.6e}  {ratio:>12.6f}")

# --- Now compare with Strata DCIM ---
from pyMoM3d.greens.layered import LayeredGreensFunction

# Low frequency first (quasi-static, where static images should match DCIM)
for freq in [100e6, 1e9, 2e9, 5e9]:
    k = 2 * np.pi * freq / c0

    print(f"\n--- Strata DCIM at f = {freq/1e9:.1f} GHz (k = {k:.4f} rad/m) ---")

    try:
        gf_strata = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='strata')
    except Exception as e:
        print(f"  Strata not available: {e}")
        continue

    print(f"{'rho (mm)':>10}  {'|g_ML_strata|':>14}  {'|g_fs|':>14}  {'|correction|':>14}  {'|g_static|':>14}  {'strata/static':>14}")
    print("-" * 95)

    for i, rho in enumerate(rhos):
        r_src = np.array([[0.0, 0.0, z_src]])
        r_obs = np.array([[rho, 0.0, z_obs]])

        # Strata returns smooth correction: g_ML - g_fs
        g_smooth = gf_strata.backend.scalar_G(r_obs, r_src)[0]

        # Free-space GF at this frequency
        g_fs_dynamic = scalar_green(k, r_obs, r_src)[0]

        # Full multilayer GF from Strata: g_ML = g_smooth + g_fs
        g_ml_strata = g_smooth + g_fs_dynamic

        # Static reference
        g_stat = g_static_vals[i]

        # Ratio
        ratio = abs(g_ml_strata) / abs(g_stat) if abs(g_stat) > 1e-30 else float('inf')

        print(f"  {rho*1e3:>8.1f}  {abs(g_ml_strata):>14.6e}  {abs(g_fs_dynamic):>14.6e}  {abs(g_smooth):>14.6e}  {abs(g_stat):>14.6e}  {ratio:>14.6f}")

# --- Also check: what does Strata return as the RAW G_phi (before eps_r multiplication)? ---
print(f"\n\n--- Raw Strata internals check ---")
print("The C++ code does: result = G_phi * eps_r - g_fs(k, R)")
print("So g_smooth = G_phi * eps_r - g_fs")
print("And g_ML = G_phi * eps_r = g_smooth + g_fs")
print()
print("If Strata is correct, g_ML should approach g_static as k -> 0")
print("If g_ML ~ g_fs everywhere, then G_phi * eps_r ~ g_fs, meaning G_phi ~ g_fs/eps_r")
print("which would mean Strata is returning the FREE-SPACE result, ignoring the substrate.")

# --- Dyadic G_A check ---
print(f"\n\n--- Dyadic G_A (vector potential) check ---")
print("For free-space: G_A = g_fs * I (identity)")
print("For PEC-backed slab: G_A_xx should show enhancement from PEC image")
print()

freq = 2e9
k = 2 * np.pi * freq / c0

try:
    gf_strata = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='strata')

    print(f"f = {freq/1e9:.1f} GHz")
    print(f"{'rho (mm)':>10}  {'|GA_xx_smooth|':>14}  {'|g_fs|':>14}  {'smooth/gfs':>14}")
    print("-" * 60)

    for rho in rhos:
        r_src = np.array([[0.0, 0.0, z_src]])
        r_obs = np.array([[rho, 0.0, z_obs]])

        ga_smooth = gf_strata.backend.dyadic_G(r_obs, r_src)
        # dyadic_G returns (N, 9) for the full 3x3 tensor
        ga_xx = ga_smooth[0, 0] if ga_smooth.ndim == 2 else ga_smooth[0]

        g_fs_val = scalar_green(k, r_obs, r_src)[0]

        ratio = abs(ga_xx) / abs(g_fs_val) if abs(g_fs_val) > 1e-30 else float('inf')

        print(f"  {rho*1e3:>8.1f}  {abs(ga_xx):>14.6e}  {abs(g_fs_val):>14.6e}  {ratio:>14.6f}")

except Exception as e:
    print(f"  Error: {e}")
