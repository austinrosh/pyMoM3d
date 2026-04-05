"""
Microstrip Multilayer Validation: Free-Space vs PEC Ground + FR4 Substrate.

End-to-end multilayer pipeline validation demonstrating the Strata C++
Green's function on a microstrip geometry.  A center-fed strip is simulated
in free space and then with a PEC ground plane under an FR4 substrate.

Physical setup:
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - PEC ground at z=0
  - Strip: width W~3.06mm (50 Ohm design), length L=50mm, at z=h
  - Center-fed (delta gap at x=0)

LayerStack (bottom to top):
  PEC half-space (is_pec=True)
  FR4  (z=0 to z=h, eps_r=4.4)
  Air half-space
  Strip placed at FR4/air interface (z=h)

Validation:
  1. Multilayer shifts resonance downward: eps_eff from resonance shift
     should approximate Hammerstad-Jensen prediction
  2. Input impedance at resonance: substrate loading changes R_res
  3. Reactance slope at resonance differs between free-space and multilayer

Note: This example uses 1-port center-fed antenna-mode analysis.
  For proper quasi-TEM Z0 extraction, a 2-port model with explicit
  ground-to-strip ports would be needed (requires vertical port connections
  or ground plane meshing, which is not yet implemented).

Produces:
  images/microstrip_multilayer_validation.png

Usage:
    source venv/bin/activate
    python examples/microstrip_z0_validation.py
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
    Layer, LayerStack, configure_latex_style, c0,
    microstrip_z0_hammerstad,
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
W_STRIP  = 3.06e-3      # Strip width (m) — approx 50 Ohm on FR4 1.6mm
L_STRIP  = 50.0e-3      # Strip length (m) — ~lambda/2 at resonance

# Mesh
TEL = 2.0e-3

# Frequency sweep
FREQS = np.linspace(0.5e9, 5.0e9, 30)


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
    print("pyMoM3d — Microstrip Multilayer Validation")
    print("         Free-space vs PEC ground + FR4 substrate")
    print("=" * 65)

    # Analytical reference
    Z0_ana, eps_eff_ana = microstrip_z0_hammerstad(W_STRIP, H_SUB, EPS_R)
    f_res_fs = c0 / (2.0 * L_STRIP)     # free-space half-wave
    f_res_ml = f_res_fs / np.sqrt(eps_eff_ana)   # substrate-loaded
    print(f"\nAnalytical (Hammerstad-Jensen):")
    print(f"  Z0      = {Z0_ana:.1f} Ohm")
    print(f"  eps_eff = {eps_eff_ana:.3f}")
    print(f"  Expected resonance: free-space ~{f_res_fs/1e9:.2f} GHz, "
          f"multilayer ~{f_res_ml/1e9:.2f} GHz")

    # Layer stack
    stack = build_layer_stack()
    print(f"\nLayer stack:")
    for lyr in stack.layers:
        print(f"  {lyr.name:15s}  z=[{lyr.z_bot:+.4e}, {lyr.z_top:+.4e}]  "
              f"eps_r={lyr.eps_r}  pec={lyr.is_pec}")

    # Mesh — strip at z=H for multilayer, z=0 for free-space
    mesher = GmshMesher(target_edge_length=TEL)

    # Place strip at FR4/air interface
    z_mesh = H_SUB
    mesh_ml = mesher.mesh_plate_with_feed(
        width=L_STRIP, height=W_STRIP, feed_x=0.0,
        center=(0.0, 0.0, z_mesh),
    )
    basis_ml = compute_rwg_connectivity(mesh_ml)
    feed_ml = find_feed_edges(mesh_ml, basis_ml, feed_x=0.0)
    signs_ml = compute_feed_signs(mesh_ml, basis_ml, feed_ml)

    mesh_fs = mesher.mesh_plate_with_feed(
        width=L_STRIP, height=W_STRIP, feed_x=0.0,
        center=(0.0, 0.0, 0.0),
    )
    basis_fs = compute_rwg_connectivity(mesh_fs)
    feed_fs = find_feed_edges(mesh_fs, basis_fs, feed_x=0.0)
    signs_fs = compute_feed_signs(mesh_fs, basis_fs, feed_fs)

    stats = mesh_ml.get_statistics()
    print(f"\nMesh: {stats['num_triangles']} triangles, "
          f"{basis_ml.num_basis} RWG, "
          f"mean edge {stats['mean_edge_length']*1e3:.2f} mm")
    print(f"Feed edges: {len(feed_ml)}")

    exc_ml = StripDeltaGapExcitation(feed_basis_indices=feed_ml, voltage=1.0,
                                     feed_signs=signs_ml)
    exc_fs = StripDeltaGapExcitation(feed_basis_indices=feed_fs, voltage=1.0,
                                     feed_signs=signs_fs)

    # Frequency sweep
    Z_fs_arr = []
    Z_ml_arr = []

    print(f"\nSweeping {len(FREQS)} frequencies "
          f"({FREQS[0]/1e9:.1f}-{FREQS[-1]/1e9:.1f} GHz)...")
    print(f"\n  {'f (GHz)':>8}  {'R_fs':>8}  {'X_fs':>8}  "
          f"{'R_ml':>8}  {'X_ml':>8}")
    print("  " + "-" * 48)

    for freq in FREQS:
        # Free-space
        cfg_fs = SimulationConfig(
            frequency=freq, excitation=exc_fs,
            quad_order=4, backend='auto',
        )
        sim_fs = Simulation(cfg_fs, mesh=mesh_fs, reporter=SilentReporter())
        r_fs = sim_fs.run()
        Z_fs_arr.append(r_fs.Z_input)

        # Multilayer
        cfg_ml = SimulationConfig(
            frequency=freq, excitation=exc_ml,
            quad_order=4, backend='auto',
            layer_stack=stack, source_layer_name='FR4',
        )
        sim_ml = Simulation(cfg_ml, mesh=mesh_ml, reporter=SilentReporter())
        r_ml = sim_ml.run()
        Z_ml_arr.append(r_ml.Z_input)

        zf, zm = r_fs.Z_input, r_ml.Z_input
        print(f"  {freq/1e9:>8.2f}  {zf.real:>8.2f}  {zf.imag:>8.2f}  "
              f"{zm.real:>8.2f}  {zm.imag:>8.2f}")

    Z_fs = np.array(Z_fs_arr)
    Z_ml = np.array(Z_ml_arr)

    # Find resonance (X=0 crossings)
    def find_resonance(Z, freqs):
        sign_changes = np.where(np.diff(np.sign(Z.imag)))[0]
        if len(sign_changes):
            i0 = sign_changes[0]
            alpha = -Z[i0].imag / (Z[i0+1].imag - Z[i0].imag)
            f_res = freqs[i0] + alpha * (freqs[i0+1] - freqs[i0])
            R_res = Z[i0].real + alpha * (Z[i0+1].real - Z[i0].real)
            return f_res, R_res
        return None, None

    f_res_fs_mom, R_res_fs = find_resonance(Z_fs, FREQS)
    f_res_ml_mom, R_res_ml = find_resonance(Z_ml, FREQS)

    print(f"\n--- Resonance (X=0) ---")
    if f_res_fs_mom:
        print(f"  Free-space: f_res = {f_res_fs_mom/1e9:.3f} GHz, "
              f"R = {R_res_fs:.1f} Ohm")
    if f_res_ml_mom:
        print(f"  Multilayer: f_res = {f_res_ml_mom/1e9:.3f} GHz, "
              f"R = {R_res_ml:.1f} Ohm")
        if f_res_fs_mom:
            eps_eff_mom = (f_res_fs_mom / f_res_ml_mom)**2
            eps_eff_err = abs(eps_eff_mom - eps_eff_ana) / eps_eff_ana * 100
            print(f"  (f_fs/f_ml)^2 = {eps_eff_mom:.3f}  "
                  f"(Hammerstad eps_eff = {eps_eff_ana:.3f}, "
                  f"error = {eps_eff_err:.1f}%)")
            print(f"  Substrate loading: {'PASS' if f_res_ml_mom < f_res_fs_mom else 'FAIL'} "
                  f"(f_ml < f_fs)")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        rf'Microstrip Multilayer Validation — FR4 ($\varepsilon_r = {EPS_R}$, '
        rf'$h = {H_SUB*1e3:.1f}$ mm)',
        fontsize=12,
    )
    fGHz = FREQS / 1e9

    # Resistance
    ax = axes[0]
    ax.plot(fGHz, Z_fs.real, 'b-o', ms=4, lw=1.5, label='Free-space')
    ax.plot(fGHz, Z_ml.real, 'r-s', ms=4, lw=1.5, label='PEC + FR4 (Strata)')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$R_{\mathrm{in}}$ ($\Omega$)')
    ax.set_title(r'Input Resistance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Reactance
    ax = axes[1]
    ax.plot(fGHz, Z_fs.imag, 'b-o', ms=4, lw=1.5, label='Free-space')
    ax.plot(fGHz, Z_ml.imag, 'r-s', ms=4, lw=1.5, label='PEC + FR4 (Strata)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$X_{\mathrm{in}}$ ($\Omega$)')
    ax.set_title(r'Input Reactance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate resonances
    if f_res_fs_mom:
        ax.axvline(f_res_fs_mom/1e9, color='b', ls=':', lw=1, alpha=0.7)
        ax.annotate(rf'$f_{{\mathrm{{res}}}}^{{\mathrm{{fs}}}}={f_res_fs_mom/1e9:.2f}$',
                    xy=(f_res_fs_mom/1e9, 0), fontsize=8, color='b',
                    xytext=(5, 15), textcoords='offset points')
    if f_res_ml_mom:
        ax.axvline(f_res_ml_mom/1e9, color='r', ls=':', lw=1, alpha=0.7)
        ax.annotate(rf'$f_{{\mathrm{{res}}}}^{{\mathrm{{ml}}}}={f_res_ml_mom/1e9:.2f}$',
                    xy=(f_res_ml_mom/1e9, 0), fontsize=8, color='r',
                    xytext=(5, -20), textcoords='offset points')

    # |Z| comparison
    ax = axes[2]
    ax.plot(fGHz, np.abs(Z_fs), 'b-o', ms=4, lw=1.5, label='Free-space')
    ax.plot(fGHz, np.abs(Z_ml), 'r-s', ms=4, lw=1.5, label='PEC + FR4 (Strata)')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|Z_{\mathrm{in}}|$ ($\Omega$)')
    ax.set_title(r'Impedance Magnitude')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'microstrip_multilayer_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
