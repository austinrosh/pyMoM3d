"""
Microstrip 2-Port Z0 and eps_eff Validation.

Extracts characteristic impedance Z0 and effective permittivity eps_eff
from 2-port S-parameters of a microstrip transmission line using the
Strata C++ multilayer Green's function.

Physical setup:
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - PEC ground at z=0 (implicit in layered Green's function)
  - Strip: width W~3.06mm (50 Ohm design), length L=40mm, at z=h
  - 2-port extraction: delta-gap at each end

LayerStack (bottom to top):
  PEC half-space (is_pec=True)
  FR4  (z=0 to z=h, eps_r=4.4)
  Air half-space
  Strip placed at FR4/air interface (z=h)

Known limitations:
  The surface EFIE with a single layered Green's function uses the source
  layer wavenumber for both vector and scalar potential contributions.
  For conductors at a dielectric interface (microstrip), the effective
  propagation constant differs from the quasi-TEM mode value, leading to
  inaccurate Z0 and eps_eff extraction.  Accurate TL extraction requires
  an MPIE formulation with separate G_A / G_Phi Sommerfeld integrals.

  The Z0 extraction accuracy varies with frequency — some frequencies
  yield good results while others are sensitive to port discontinuity
  effects.  Results are most reliable when |S21| > -3 dB.

Produces:
  images/microstrip_z0_2port_validation.png

Usage:
    source venv/bin/activate
    python examples/microstrip_z0_2port_validation.py
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
    Port, NetworkExtractor,
    Layer, LayerStack,
    configure_latex_style, c0,
    microstrip_z0_hammerstad,
    extract_z0_from_s, extract_propagation_constant,
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
L_STRIP  = 40.0e-3      # Strip length (m)

# Mesh — need ~4 elements across strip width for accurate current distribution
TEL = 1.0e-3

# Frequency sweep
FREQS_GHZ = np.linspace(1.0, 10.0, 8)
FREQS     = FREQS_GHZ * 1e9


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
    print("pyMoM3d — Microstrip 2-Port Z0 / eps_eff Validation")
    print("=" * 65)

    # --- Analytical reference ---
    Z0_ana, eps_eff_ana = microstrip_z0_hammerstad(W_STRIP, H_SUB, EPS_R)
    print(f"\nAnalytical (Hammerstad-Jensen):")
    print(f"  Z0      = {Z0_ana:.2f} Ohm")
    print(f"  eps_eff = {eps_eff_ana:.3f}")

    # --- Layer stack ---
    stack = build_layer_stack()
    print(f"\nLayer stack:")
    for lyr in stack.layers:
        print(f"  {lyr.name:15s}  z=[{lyr.z_bot:+.4e}, {lyr.z_top:+.4e}]  "
              f"eps_r={lyr.eps_r}  pec={lyr.is_pec}")

    # --- Mesh with two feed lines ---
    mesher = GmshMesher(target_edge_length=TEL)
    z_mesh = H_SUB  # Strip at FR4/air interface

    margin = 3.0e-3
    port1_x = -L_STRIP / 2.0 + margin
    port2_x = +L_STRIP / 2.0 - margin
    L_port = port2_x - port1_x

    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[port1_x, port2_x],
        center=(0.0, 0.0, z_mesh),
    )
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    print(f"\nMesh: {stats['num_triangles']} triangles, "
          f"{basis.num_basis} RWG, "
          f"mean edge {stats['mean_edge_length']*1e3:.2f} mm")

    # --- Port definition ---
    feed1 = find_feed_edges(mesh, basis, feed_x=port1_x)
    feed2 = find_feed_edges(mesh, basis, feed_x=port2_x)
    print(f"  Port 1 at x = {port1_x*1e3:.1f} mm ({len(feed1)} edges)")
    print(f"  Port 2 at x = {port2_x*1e3:.1f} mm ({len(feed2)} edges)")
    print(f"  Port separation = {L_port*1e3:.1f} mm")

    if not feed1 or not feed2:
        print("ERROR: Could not find feed edges at port locations.")
        return

    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    # --- Build Simulation ---
    exc_dummy = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=FREQS[0],
        excitation=exc_dummy,
        quad_order=4,
        backend='auto',
        layer_stack=stack,
        source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # --- Frequency sweep ---
    extractor = NetworkExtractor(sim, [port1, port2])
    print(f"\nSweeping {len(FREQS)} frequencies "
          f"({FREQS[0]/1e9:.1f}-{FREQS[-1]/1e9:.1f} GHz)...")

    results = extractor.extract(FREQS.tolist())

    # --- Extract Z0 and eps_eff ---
    Z0_mom = []
    eps_eff_mom = []

    print(f"\n  {'f (GHz)':>8}  {'Z0 (Ohm)':>10}  {'Z0_err':>8}  "
          f"{'eps_eff':>8}  {'ee_err':>8}  {'|S21| dB':>9}")
    print("  " + "-" * 60)

    for freq, result in zip(FREQS, results):
        S = result.S_matrix
        z0_ext = extract_z0_from_s(S, Z0_ref=50.0)
        gamma = extract_propagation_constant(S, L_port, Z0_ref=50.0)
        k0 = 2.0 * np.pi * freq / c0
        ee = (gamma.imag / k0)**2 if k0 > 0 else float('nan')
        s21_dB = 20.0 * np.log10(max(abs(S[1, 0]), 1e-12))

        Z0_mom.append(abs(z0_ext))
        eps_eff_mom.append(ee)

        z0_err = (abs(z0_ext) - Z0_ana) / Z0_ana * 100
        ee_err = (ee - eps_eff_ana) / eps_eff_ana * 100

        print(f"  {freq/1e9:>8.2f}  {abs(z0_ext):>10.2f}  {z0_err:>+7.1f}%  "
              f"{ee:>8.3f}  {ee_err:>+7.1f}%  {s21_dB:>9.2f}")

    Z0_mom = np.array(Z0_mom)
    eps_eff_mom = np.array(eps_eff_mom)

    # --- Summary ---
    mean_z0_err = np.mean(np.abs(Z0_mom - Z0_ana) / Z0_ana * 100)
    mean_ee_err = np.mean(np.abs(eps_eff_mom - eps_eff_ana) / eps_eff_ana * 100)
    print(f"\n  Mean Z0 error:      {mean_z0_err:.1f}%  "
          f"{'PASS' if mean_z0_err < 15 else 'CHECK'}")
    print(f"  Mean eps_eff error: {mean_ee_err:.1f}%  "
          f"{'PASS' if mean_ee_err < 15 else 'CHECK'}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        rf'Microstrip 2-Port Validation — FR4 ($\varepsilon_r = {EPS_R}$, '
        rf'$h = {H_SUB*1e3:.1f}$ mm, $W = {W_STRIP*1e3:.2f}$ mm)',
        fontsize=12,
    )

    # Z0
    ax1.plot(FREQS_GHZ, Z0_mom, 'bo-', ms=6, lw=1.5, label=r'MoM (Strata)')
    ax1.axhline(Z0_ana, color='r', ls='--', lw=1.5,
                label=rf'Hammerstad ($Z_0 = {Z0_ana:.1f}\,\Omega$)')
    ax1.set_xlabel(r'Frequency $f$ (GHz)')
    ax1.set_ylabel(r'$Z_0$ ($\Omega$)')
    ax1.set_title(r'Characteristic Impedance')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(100, Z0_ana * 2)])

    # eps_eff
    ax2.plot(FREQS_GHZ, eps_eff_mom, 'bo-', ms=6, lw=1.5, label=r'MoM (Strata)')
    ax2.axhline(eps_eff_ana, color='r', ls='--', lw=1.5,
                label=rf'Hammerstad ($\varepsilon_{{\mathrm{{eff}}}} = {eps_eff_ana:.3f}$)')
    ax2.set_xlabel(r'Frequency $f$ (GHz)')
    ax2.set_ylabel(r'$\varepsilon_{\mathrm{eff}}$')
    ax2.set_title(r'Effective Permittivity')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(EPS_R + 1, eps_eff_ana * 2)])

    out = os.path.join(IMAGES_DIR, 'microstrip_z0_2port_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
