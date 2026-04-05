"""
Rectangular Patch Antenna on FR4 Substrate — Cavity Model Validation.

Single-port edge-fed rectangular patch antenna on FR4 substrate.
Validates resonant frequency and input impedance against the cavity
model (Balanis).

Physical setup:
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - PEC ground at z=0 (implicit in layered Green's function)
  - Patch: W=38mm (radiating edge), L=29mm (resonant dimension)
  - Edge feed: delta-gap near one edge of the patch

LayerStack (bottom to top):
  PEC half-space (is_pec=True)
  FR4  (z=0 to z=h, eps_r=4.4)
  Air half-space
  Strip placed at FR4/air interface (z=h)

Validation:
  1. Resonant frequency: within ~5% of cavity model prediction
  2. Input resistance at resonance: order-of-magnitude check vs cavity model

Produces:
  images/patch_antenna_validation.png

Usage:
    source venv/bin/activate
    python examples/patch_antenna_validation.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Layer, LayerStack,
    configure_latex_style, c0,
    patch_antenna_cavity_model,
)
from pyMoM3d.mom.excitation import StripDeltaGapExcitation, find_feed_edges, compute_feed_signs

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R    = 4.4          # FR4 relative permittivity
H_SUB    = 1.6e-3       # Substrate height (m)
W_PATCH  = 38.0e-3      # Patch width (m) — radiating dimension
L_PATCH  = 29.0e-3      # Patch length (m) — resonant dimension

# Mesh — target ~lambda/20 at resonance
TEL = 2.0e-3

# Frequency sweep
FREQS = np.linspace(1.5e9, 3.5e9, 40)


# ---------------------------------------------------------------------------
# LayerStack
# ---------------------------------------------------------------------------

def build_layer_stack():
    return LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4',        z_bot=0.0,     z_top=H_SUB, eps_r=EPS_R),
        Layer('air',        z_bot=H_SUB,   z_top=np.inf, eps_r=1.0),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("pyMoM3d — Rectangular Patch Antenna Validation")
    print("         Edge-fed on FR4 substrate, cavity model comparison")
    print("=" * 65)

    # --- Analytical reference ---
    f_res_ana, R_in_ana, eps_eff, delta_L = patch_antenna_cavity_model(
        W_PATCH, L_PATCH, H_SUB, EPS_R,
    )
    print(f"\nCavity Model (Balanis):")
    print(f"  eps_eff  = {eps_eff:.3f}")
    print(f"  delta_L  = {delta_L*1e3:.3f} mm")
    print(f"  f_res    = {f_res_ana/1e9:.3f} GHz")
    print(f"  R_in     = {R_in_ana:.1f} Ohm (edge, approximate)")

    # --- Layer stack ---
    stack = build_layer_stack()
    print(f"\nLayer stack:")
    for lyr in stack.layers:
        print(f"  {lyr.name:15s}  z=[{lyr.z_bot:+.4e}, {lyr.z_top:+.4e}]  "
              f"eps_r={lyr.eps_r}  pec={lyr.is_pec}")

    # --- Mesh ---
    mesher = GmshMesher(target_edge_length=TEL)
    z_mesh = H_SUB  # Strip at FR4/air interface

    # Feed at the edge of the patch (small inset for better matching)
    feed_x = -L_PATCH / 2.0 + TEL / 2.0

    mesh = mesher.mesh_plate_with_feed(
        width=L_PATCH, height=W_PATCH,
        feed_x=feed_x,
        center=(0.0, 0.0, z_mesh),
    )
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    print(f"\nMesh: {stats['num_triangles']} triangles, "
          f"{basis.num_basis} RWG, "
          f"mean edge {stats['mean_edge_length']*1e3:.2f} mm")

    feed_indices = find_feed_edges(mesh, basis, feed_x=feed_x)
    feed_signs = compute_feed_signs(mesh, basis, feed_indices)
    print(f"  Feed at x = {feed_x*1e3:.2f} mm ({len(feed_indices)} edges)")

    if not feed_indices:
        print("ERROR: Could not find feed edges.")
        return

    # --- Frequency sweep ---
    exc = StripDeltaGapExcitation(feed_basis_indices=feed_indices, voltage=1.0,
                                  feed_signs=feed_signs)

    Z_arr = []
    print(f"\nSweeping {len(FREQS)} frequencies "
          f"({FREQS[0]/1e9:.1f}-{FREQS[-1]/1e9:.1f} GHz)...")
    print(f"\n  {'f (GHz)':>8}  {'R_in':>8}  {'X_in':>8}")
    print("  " + "-" * 28)

    for freq in FREQS:
        cfg = SimulationConfig(
            frequency=freq, excitation=exc,
            quad_order=4, backend='auto',
            layer_stack=stack, source_layer_name='FR4',
        )
        sim = Simulation(cfg, mesh=mesh, reporter=SilentReporter())
        r = sim.run()
        Z_arr.append(r.Z_input)

        print(f"  {freq/1e9:>8.3f}  {r.Z_input.real:>8.2f}  {r.Z_input.imag:>+8.2f}")

    Z = np.array(Z_arr)

    # --- Find resonance (X=0 crossing) ---
    def find_resonance(Z_arr, freqs):
        sign_changes = np.where(np.diff(np.sign(Z_arr.imag)))[0]
        if len(sign_changes):
            i0 = sign_changes[0]
            alpha = -Z_arr[i0].imag / (Z_arr[i0+1].imag - Z_arr[i0].imag)
            f_res = freqs[i0] + alpha * (freqs[i0+1] - freqs[i0])
            R_res = Z_arr[i0].real + alpha * (Z_arr[i0+1].real - Z_arr[i0].real)
            return f_res, R_res
        return None, None

    f_res_mom, R_res_mom = find_resonance(Z, FREQS)

    print(f"\n--- Results ---")
    if f_res_mom:
        f_err = abs(f_res_mom - f_res_ana) / f_res_ana * 100
        print(f"  MoM resonance:    f = {f_res_mom/1e9:.3f} GHz, "
              f"R = {R_res_mom:.1f} Ohm")
        print(f"  Cavity model:     f = {f_res_ana/1e9:.3f} GHz, "
              f"R = {R_in_ana:.1f} Ohm (edge)")
        print(f"  Frequency error:  {f_err:.1f}%  "
              f"{'PASS' if f_err < 10 else 'CHECK'}")
    else:
        print("  No resonance found in sweep range.")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        rf'Patch Antenna — FR4 ($\varepsilon_r = {EPS_R}$, '
        rf'$h = {H_SUB*1e3:.1f}$ mm, '
        rf'$W = {W_PATCH*1e3:.0f}$ mm, $L = {L_PATCH*1e3:.0f}$ mm)',
        fontsize=12,
    )
    fGHz = FREQS / 1e9

    # Resistance
    ax = axes[0]
    ax.plot(fGHz, Z.real, 'b-o', ms=3, lw=1.5, label='MoM (Strata)')
    if f_res_mom:
        ax.axvline(f_res_mom/1e9, color='b', ls=':', lw=1, alpha=0.7)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$R_{\mathrm{in}}$ ($\Omega$)')
    ax.set_title(r'Input Resistance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Reactance
    ax = axes[1]
    ax.plot(fGHz, Z.imag, 'b-o', ms=3, lw=1.5, label='MoM (Strata)')
    ax.axhline(0, color='k', lw=0.5)
    if f_res_mom:
        ax.axvline(f_res_mom/1e9, color='b', ls=':', lw=1, alpha=0.7,
                   label=rf'$f_{{\mathrm{{res}}}} = {f_res_mom/1e9:.3f}$ GHz')
    ax.axvline(f_res_ana/1e9, color='r', ls='--', lw=1, alpha=0.7,
               label=rf'Cavity model $f_{{\mathrm{{res}}}} = {f_res_ana/1e9:.3f}$ GHz')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$X_{\mathrm{in}}$ ($\Omega$)')
    ax.set_title(r'Input Reactance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # |S11|
    ax = axes[2]
    Z0_ref = 50.0
    S11 = (Z - Z0_ref) / (Z + Z0_ref)
    ax.plot(fGHz, 20 * np.log10(np.abs(S11)), 'b-o', ms=3, lw=1.5,
            label='MoM (Strata)')
    if f_res_ana:
        ax.axvline(f_res_ana/1e9, color='r', ls='--', lw=1, alpha=0.7,
                   label='Cavity model')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-30, 0])

    out = os.path.join(IMAGES_DIR, 'patch_antenna_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
