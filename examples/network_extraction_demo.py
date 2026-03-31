"""
Network Extraction Demo: Lumped Port Modeling & Multi-Port S/Z Parameters.

Demonstrates the new Port / NetworkExtractor API on two scenarios:

  1. Single-port dipole (validation)
       - Z[0,0] from NetworkExtractor  vs  Z_input from Simulation → must agree
       - Frequency sweep of S11

  2. Two-port coupled dipoles
       - Parallel half-wave dipoles separated by lambda/3
       - Full 2x2 Z-matrix and S-matrix vs frequency
       - Verify reciprocity  |Z12 - Z21| / |Z11| ≪ 1
       - Verify symmetry      Z11 ≈ Z22

Produces:
  images/network_single_port_sweep.png  — Z11, S11 vs frequency
  images/network_two_port_sweep.png     — Z11, Z12, S11, S21 vs frequency

Usage:
    source venv/bin/activate
    python examples/network_extraction_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    combine_meshes,
    Port, NetworkExtractor,
    configure_latex_style, c0,
)
from pyMoM3d.mom.excitation import StripDeltaGapExcitation, find_feed_edges

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

DIPOLE_LENGTH = 0.15     # m  →  resonance near 1 GHz (half-wave)
DIPOLE_WIDTH  = 0.01     # m  strip width
TEL           = 0.008    # m  target edge length (coarser = faster demo)

def make_dipole_mesh(center_x=0.0):
    """Mesh a strip dipole centred at (center_x, 0, 0), feed at center_x."""
    mesher = GmshMesher(target_edge_length=TEL)
    return mesher.mesh_plate_with_feed(
        width=DIPOLE_LENGTH,
        height=DIPOLE_WIDTH,
        feed_x=center_x,
        center=(center_x, 0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Part 1 — Single-port dipole: validation + S11 sweep
# ---------------------------------------------------------------------------

def demo_single_port():
    print("\n" + "=" * 60)
    print("PART 1 — Single-port dipole (validation + S11 sweep)")
    print("=" * 60)

    # --- Mesh + basis (built once) ---
    print("\nMeshing single strip dipole...")
    mesh = make_dipole_mesh(center_x=0.0)
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    print(f"  Triangles: {stats['num_triangles']},  "
          f"RWG basis: {basis.num_basis},  "
          f"mean edge: {stats['mean_edge_length']*100:.2f} cm")

    # --- Port definition ---
    feed_indices = find_feed_edges(mesh, basis, feed_x=0.0)
    print(f"  Feed edges at x=0: {len(feed_indices)}")
    port = Port(name='P1', feed_basis_indices=feed_indices)

    # --- Single-frequency validation ---
    f_val = 1.0e9   # 1 GHz
    exc_sim = StripDeltaGapExcitation(feed_basis_indices=feed_indices, voltage=1.0)
    config = SimulationConfig(
        frequency=f_val, excitation=exc_sim, quad_order=4,
        backend='auto',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # Traditional route
    result_sim = sim.run()
    Z_sim = result_sim.Z_input

    # NetworkExtractor route
    extractor = NetworkExtractor(sim, [port])
    [nr] = extractor.extract(f_val)
    Z_net = nr.Z_matrix[0, 0]

    print(f"\n  Validation at f = {f_val/1e9:.1f} GHz")
    print(f"  Simulation.Z_input  = {Z_sim.real:+.4f} {Z_sim.imag:+.4f}j Ω")
    print(f"  NetworkExtractor Z  = {Z_net.real:+.4f} {Z_net.imag:+.4f}j Ω")
    rel_err = abs(Z_net - Z_sim) / abs(Z_sim) if abs(Z_sim) > 0 else 0.0
    print(f"  Relative error      = {rel_err:.2e}  {'✓ PASS' if rel_err < 1e-6 else '✗ FAIL'}")

    # --- Frequency sweep via NetworkExtractor ---
    freqs = np.linspace(0.5e9, 1.5e9, 21)
    print(f"\n  Sweeping {len(freqs)} frequencies ({freqs[0]/1e9:.1f}–{freqs[-1]/1e9:.1f} GHz)...")
    results = extractor.extract(freqs.tolist())

    Z11 = np.array([r.Z_matrix[0, 0] for r in results])
    S11 = np.array([r.S_matrix[0, 0] for r in results])
    S11_dB = 20 * np.log10(np.abs(S11).clip(1e-9))

    # Print table
    print(f"\n  {'f (GHz)':>8}  {'R (Ω)':>10}  {'X (Ω)':>10}  {'|S11| (dB)':>12}")
    print("  " + "-" * 46)
    for i, f in enumerate(freqs[::2]):   # every other point
        idx = i * 2
        print(f"  {f/1e9:>8.3f}  {Z11[idx].real:>10.2f}  {Z11[idx].imag:>10.2f}  {S11_dB[idx]:>12.2f}")

    # Find resonance (X=0 crossing)
    sign_changes = np.where(np.diff(np.sign(Z11.imag)))[0]
    if len(sign_changes):
        i0 = sign_changes[0]
        alpha = -Z11[i0].imag / (Z11[i0+1].imag - Z11[i0].imag)
        f_res = freqs[i0] + alpha * (freqs[i0+1] - freqs[i0])
        R_res = Z11[i0].real + alpha * (Z11[i0+1].real - Z11[i0].real)
        print(f"\n  Resonance (X=0):   f_res = {f_res/1e9:.3f} GHz,  R = {R_res:.1f} Ω")
        print(f"  (Half-wave resonance expected near {c0/(2*DIPOLE_LENGTH)/1e9:.2f} GHz)")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(r'Single-Port Strip Dipole — NetworkExtractor', fontsize=12)
    fGHz = freqs / 1e9

    ax = axes[0]
    ax.plot(fGHz, Z11.real, 'b-o', ms=4, label=r'$R_{\mathrm{in}}$')
    ax.plot(fGHz, Z11.imag, 'r--s', ms=4, label=r'$X_{\mathrm{in}}$')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'Impedance ($\Omega$)')
    ax.set_title(r'Input Impedance $Z_{11}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(fGHz, S11_dB, 'g-^', ms=4)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss $|S_{11}|$')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(Z11.real, Z11.imag, 'mo-', ms=5, lw=1.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(50, color='b', lw=0.5, ls='--', label=r'$Z_0 = 50\,\Omega$')
    ax.set_xlabel(r'$R_{\mathrm{in}}$ ($\Omega$)')
    ax.set_ylabel(r'$X_{\mathrm{in}}$ ($\Omega$)')
    ax.set_title(r'Impedance Locus')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'network_single_port_sweep.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved → {out}")

    return sim, port, freqs


# ---------------------------------------------------------------------------
# Part 2 — Two-port coupled dipoles: Z-matrix + S-matrix
# ---------------------------------------------------------------------------

def demo_two_port(freqs):
    print("\n" + "=" * 60)
    print("PART 2 — Two-port coupled dipoles (Z-matrix + reciprocity)")
    print("=" * 60)

    separation = 0.10    # m  ≈ lambda/3 at 1 GHz — visible coupling

    # --- Mesh each dipole separately, then combine ---
    print(f"\n  Dipole separation: {separation*100:.0f} cm "
          f"({separation/(c0/1e9):.2f} λ at 1 GHz)")
    print("  Meshing dipole 1 (x = 0)...")
    mesh1 = make_dipole_mesh(center_x=0.0)
    print("  Meshing dipole 2 (x = {:.2f} m)...".format(separation))
    mesh2 = make_dipole_mesh(center_x=separation)

    combined, _ = combine_meshes([mesh1, mesh2])
    basis = compute_rwg_connectivity(combined)
    stats = combined.get_statistics()
    print(f"  Combined mesh: {stats['num_triangles']} triangles, "
          f"{basis.num_basis} RWG basis functions")

    # --- Port definitions in the combined mesh ---
    feed1 = find_feed_edges(combined, basis, feed_x=0.0)
    feed2 = find_feed_edges(combined, basis, feed_x=separation)
    print(f"  Port 1: {len(feed1)} feed edges at x=0")
    print(f"  Port 2: {len(feed2)} feed edges at x={separation}")

    if not feed1 or not feed2:
        print("  ERROR: could not find feed edges — aborting two-port demo")
        return

    port1 = Port(name='P1', feed_basis_indices=feed1)
    port2 = Port(name='P2', feed_basis_indices=feed2)

    # --- Build a Simulation around the combined mesh ---
    exc_dummy = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=freqs[0], excitation=exc_dummy, quad_order=4, backend='auto',
    )
    sim = Simulation(config, mesh=combined, reporter=SilentReporter())

    # --- Frequency sweep via NetworkExtractor ---
    extractor = NetworkExtractor(sim, [port1, port2])
    print(f"\n  Sweeping {len(freqs)} frequencies...")
    results = extractor.extract(freqs.tolist())

    Z11 = np.array([r.Z_matrix[0, 0] for r in results])
    Z12 = np.array([r.Z_matrix[0, 1] for r in results])
    Z21 = np.array([r.Z_matrix[1, 0] for r in results])
    Z22 = np.array([r.Z_matrix[1, 1] for r in results])

    S11 = np.array([r.S_matrix[0, 0] for r in results])
    S21 = np.array([r.S_matrix[1, 0] for r in results])
    S11_dB = 20 * np.log10(np.abs(S11).clip(1e-9))
    S21_dB = 20 * np.log10(np.abs(S21).clip(1e-9))

    # --- Diagnostics ---
    recip_err  = np.abs(Z12 - Z21) / np.abs(Z11).clip(1e-9)
    sym_err    = np.abs(Z11 - Z22) / np.abs(Z11).clip(1e-9)

    print(f"\n  {'f (GHz)':>8}  {'Z11 (Ω)':>18}  {'Z12 (Ω)':>18}  "
          f"{'|Z12-Z21|/|Z11|':>16}  {'|Z11-Z22|/|Z11|':>16}")
    print("  " + "-" * 86)
    for i, f in enumerate(freqs[::3]):
        idx = i * 3
        z11 = Z11[idx]; z12 = Z12[idx]
        print(f"  {f/1e9:>8.3f}  "
              f"{z11.real:+7.1f}{z11.imag:+7.1f}j  "
              f"{z12.real:+7.1f}{z12.imag:+7.1f}j  "
              f"{recip_err[idx]:>16.2e}  {sym_err[idx]:>16.2e}")

    max_recip = float(np.max(recip_err))
    max_sym   = float(np.max(sym_err))
    print(f"\n  Max reciprocity error  |Z12-Z21|/|Z11| = {max_recip:.2e}  "
          f"{'✓ PASS' if max_recip < 1e-4 else '! CHECK'}")
    print(f"  Max symmetry error     |Z11-Z22|/|Z11| = {max_sym:.2e}  "
          f"{'✓ PASS' if max_sym < 1e-2 else '! CHECK'}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        rf'Two-Port Coupled Dipoles (separation = {separation*100:.0f} cm '
        rf'$\approx\lambda/3$)',
        fontsize=12,
    )
    fGHz = freqs / 1e9

    ax = axes[0, 0]
    ax.plot(fGHz, Z11.real, 'b-',  lw=1.5, label=r'$R_{11}$')
    ax.plot(fGHz, Z11.imag, 'b--', lw=1.5, label=r'$X_{11}$')
    ax.plot(fGHz, Z22.real, 'r:',  lw=1.0, label=r'$R_{22}$')
    ax.plot(fGHz, Z22.imag, 'r:', lw=1.0, label=r'$X_{22}$')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'Impedance ($\Omega$)')
    ax.set_title(r'Self-impedance $Z_{11}$ and $Z_{22}$')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(fGHz, Z12.real, 'g-',  lw=1.5, label=r'$R_{12}$')
    ax.plot(fGHz, Z12.imag, 'g--', lw=1.5, label=r'$X_{12}$')
    ax.plot(fGHz, Z21.real, 'm:',  lw=1.0, label=r'$R_{21}$')
    ax.plot(fGHz, Z21.imag, 'm:', lw=1.0, label=r'$X_{21}$')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'Mutual Impedance ($\Omega$)')
    ax.set_title(r'Mutual impedance $Z_{12}$ and $Z_{21}$ (reciprocity check)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(fGHz, S11_dB, 'b-o', ms=4, lw=1.5, label=r'$|S_{11}|$')
    ax.plot(fGHz, S21_dB, 'r-s', ms=4, lw=1.5, label=r'$|S_{21}|$')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'(dB)')
    ax.set_title(r'S-parameters')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(fGHz, recip_err, 'k-', lw=1.5, label=r'$|Z_{12}-Z_{21}|/|Z_{11}|$')
    ax.plot(fGHz, sym_err,   'b--', lw=1.5, label=r'$|Z_{11}-Z_{22}|/|Z_{11}|$')
    ax.set_yscale('log')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'Relative error')
    ax.set_title(r'Reciprocity \& symmetry verification')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'network_two_port_sweep.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("pyMoM3d  —  Network Extraction Demo")
    print(f"Dipole: {DIPOLE_LENGTH*100:.0f} cm × {DIPOLE_WIDTH*100:.1f} cm strip")
    print(f"Target edge length: {TEL*1000:.1f} mm")
    print("=" * 60)

    sim, port, freqs = demo_single_port()
    demo_two_port(freqs)

    print("\nDone.")
