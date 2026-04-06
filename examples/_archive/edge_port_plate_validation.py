"""Edge-fed vertical plate port validation: microstrip through-line.

Validates the edge-port framework by extracting S-parameters for a 50-ohm
microstrip through-line on FR4 using vertical plate ports at both ends.

The edge-fed plate provides a conductive return path to the PEC ground
plane at all frequencies, eliminating the low-frequency failure of strip
delta-gap ports.

This script:
1. Meshes a microstrip with vertical plates at both ends
2. Creates edge ports at the strip-plate junctions
3. Extracts S-parameters via full-wave MPIE (MultilayerEFIEOperator)
4. Compares against the 2D cross-section reference impedance
5. Optionally compares against QS probe results in the overlap band

Usage
-----
    python examples/microstrip_edge_port_validation.py
    python examples/microstrip_edge_port_validation.py --with-qs
"""

import argparse
import numpy as np
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor,
    Layer, LayerStack,
    configure_latex_style, c0, eta0,
)
from pyMoM3d.cross_section import compute_reference_impedance

try:
    configure_latex_style()
except Exception:
    configure_latex_style(use_tex=False)

IMG_DIR = Path(__file__).resolve().parent.parent / 'images' / 'edge_port'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Physical parameters
# ============================================================================

EPS_R = 4.4           # FR4
H_SUB = 1.6e-3        # substrate height (m)
W_STRIP = 3.06e-3     # strip width for ~50 ohm (m)
L_STRIP = 10.0e-3     # strip length (m)
TEL = 0.7e-3          # target edge length (m)
PLATE_Z_OFFSET = H_SUB / 10.0   # keep plate bottom above PEC ground

# Frequency sweep
FREQS = np.linspace(1.0e9, 12.0e9, 23)


def setup():
    """Build layer stack, mesh with edge ports, compute basis and ports."""
    stack = LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])

    # 2D reference impedance
    tl = compute_reference_impedance(
        stack, strip_width=W_STRIP, source_layer_name='FR4', base_cells=300,
    )
    print(f"2D solver: Z0 = {tl.Z0:.2f} Ohm, eps_eff = {tl.eps_eff:.3f}")

    # Mesh with vertical plates at both ends
    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_microstrip_with_edge_ports(
        width=W_STRIP,
        length=L_STRIP,
        substrate_height=H_SUB,
        port_edges=['left', 'right'],
        center=(0.0, 0.0, H_SUB),
        plate_z_offset=PLATE_Z_OFFSET,
    )
    basis = compute_rwg_connectivity(mesh)
    print(f"Mesh: {len(mesh.triangles)} triangles, {basis.num_basis} basis functions")

    # Edge ports at strip-plate junctions
    x_left = -L_STRIP / 2.0
    x_right = +L_STRIP / 2.0
    port1 = Port.from_edge_port(mesh, basis, port_x=x_left, strip_z=H_SUB, name='P1')
    port2 = Port.from_edge_port(mesh, basis, port_x=x_right, strip_z=H_SUB, name='P2')
    print(f"Port 1: {len(port1.feed_basis_indices)} junction edges at x={x_left*1e3:.1f} mm")
    print(f"Port 2: {len(port2.feed_basis_indices)} junction edges at x={x_right*1e3:.1f} mm")

    return stack, tl, mesh, basis, port1, port2


def run_edge_port(stack, tl, mesh, basis, port1, port2):
    """Full-wave MPIE extraction with edge-port excitation."""
    print("\n--- Full-wave MPIE (edge-fed vertical plate) ---")

    from pyMoM3d.mom.excitation import StripDeltaGapExcitation
    # Excitation object is only needed for SimulationConfig; actual port
    # excitation is handled by NetworkExtractor via Port objects.
    dummy_exc = StripDeltaGapExcitation(
        feed_basis_indices=port1.feed_basis_indices, voltage=1.0,
        feed_signs=port1.feed_signs,
    )
    config = SimulationConfig(
        frequency=1e9,
        excitation=dummy_exc,
        quad_order=4,
        backend='auto',
        layer_stack=stack,
        source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    extractor = NetworkExtractor(sim, [port1, port2], Z0=tl.Z0)

    t0 = time.time()
    results = extractor.extract(FREQS.tolist())
    dt = time.time() - t0
    print(f"  {len(FREQS)} frequencies in {dt:.1f}s "
          f"({dt/len(FREQS)*1e3:.0f} ms/freq)")

    return results


def run_qs_probe(stack, tl, mesh_strip, basis_strip, freqs_qs):
    """QS probe solver for comparison (uses strip-only mesh)."""
    from pyMoM3d.mom.excitation import (
        StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
    )
    from pyMoM3d.mom.quasi_static import QuasiStaticSolver

    # QS needs ports on the strip mesh (no plates)
    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[-L_STRIP/2.0 + 1.0e-3, L_STRIP/2.0 - 1.0e-3],
        center=(0.0, 0.0, H_SUB),
    )
    basis = compute_rwg_connectivity(mesh)

    port1_x = -L_STRIP/2.0 + 1.0e-3
    port2_x = L_STRIP/2.0 - 1.0e-3
    feed1 = find_feed_edges(mesh, basis, feed_x=port1_x)
    feed2 = find_feed_edges(mesh, basis, feed_x=port2_x)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    p1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    p2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=1e9, excitation=exc, quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    qs = QuasiStaticSolver(sim, [p1, p2], Z0=tl.Z0, probe_feeds=True)

    print("\n--- QS Solver (probe feeds, comparison) ---")
    t0 = time.time()
    results = qs.extract(freqs_qs.tolist())
    dt = time.time() - t0
    print(f"  {len(freqs_qs)} frequencies in {dt:.1f}s")
    return results


def validate(results):
    """Check pass criteria and print diagnostics."""
    print("\n=== Validation Checks ===")
    all_pass = True

    for r in results:
        S = r.S_matrix
        f_ghz = r.frequency / 1e9

        # Reciprocity
        recip_err = abs(S[0, 1] - S[1, 0])
        if recip_err > 1e-6:
            print(f"  WARN: reciprocity error |S12-S21| = {recip_err:.2e} at {f_ghz:.1f} GHz")

        # Port symmetry
        sym_err = abs(S[0, 0] - S[1, 1])
        if sym_err > 1e-6:
            print(f"  WARN: symmetry error |S11-S22| = {sym_err:.2e} at {f_ghz:.1f} GHz")

    # S21 at ~5 GHz
    idx_5g = np.argmin(np.abs(FREQS - 5e9))
    s21_5g_dB = 20*np.log10(abs(results[idx_5g].S_matrix[1, 0]) + 1e-30)
    s11_5g_dB = 20*np.log10(abs(results[idx_5g].S_matrix[0, 0]) + 1e-30)
    print(f"\n  At {FREQS[idx_5g]/1e9:.1f} GHz:")
    print(f"    |S21| = {s21_5g_dB:.2f} dB  (target: > -3.0 dB)")
    print(f"    |S11| = {s11_5g_dB:.2f} dB  (target: < -5.0 dB)")
    if s21_5g_dB < -3.0:
        print("    FAIL: S21 too low")
        all_pass = False
    else:
        print("    PASS")

    # S21 at lowest frequency (~1 GHz) — should not collapse
    s21_low_dB = 20*np.log10(abs(results[0].S_matrix[1, 0]) + 1e-30)
    print(f"\n  At {FREQS[0]/1e9:.1f} GHz:")
    print(f"    |S21| = {s21_low_dB:.2f} dB  (target: > -10.0 dB, no collapse)")
    if s21_low_dB < -10.0:
        print("    FAIL: low-freq S21 collapse (port return path may be broken)")
        all_pass = False
    else:
        print("    PASS")

    # Overall reciprocity
    max_recip = max(abs(r.S_matrix[0, 1] - r.S_matrix[1, 0]) for r in results)
    print(f"\n  Max |S12 - S21| = {max_recip:.2e}  (target: < 1e-6)")
    if max_recip > 1e-6:
        print("    WARN: reciprocity check")
    else:
        print("    PASS")

    # Overall symmetry
    max_sym = max(abs(r.S_matrix[0, 0] - r.S_matrix[1, 1]) for r in results)
    print(f"  Max |S11 - S22| = {max_sym:.2e}  (target: < 1e-6)")
    if max_sym > 1e-6:
        print("    WARN: symmetry check")
    else:
        print("    PASS")

    if all_pass:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED — investigate")

    return all_pass


def plot_results(results, tl, qs_results=None):
    """Plot S-parameters vs frequency with 2D reference."""
    f_ghz = FREQS / 1e9
    s21 = np.array([r.S_matrix[1, 0] for r in results])
    s11 = np.array([r.S_matrix[0, 0] for r in results])
    z11 = np.array([r.Z_matrix[0, 0] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        rf'Edge-Port Validation — Microstrip Through-Line '
        rf'($\varepsilon_r={EPS_R}$, $h={H_SUB*1e3:.1f}$ mm, '
        rf'$W={W_STRIP*1e3:.2f}$ mm)',
        fontsize=12,
    )

    # --- S21 magnitude ---
    ax = axes[0, 0]
    ax.plot(f_ghz, 20*np.log10(np.abs(s21) + 1e-30),
            'b-o', linewidth=1.5, markersize=4, label='Edge port (full-wave)')
    if qs_results:
        f_qs = np.array([r.frequency for r in qs_results]) / 1e9
        s21_qs = np.array([r.S_matrix[1, 0] for r in qs_results])
        ax.plot(f_qs, 20*np.log10(np.abs(s21_qs) + 1e-30),
                'g-', linewidth=1.5, label='QS probe')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{21}|$ (dB)')
    ax.set_title(r'Insertion Loss $|S_{21}|$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-30, 5)

    # --- S11 magnitude ---
    ax = axes[0, 1]
    ax.plot(f_ghz, 20*np.log10(np.abs(s11) + 1e-30),
            'r-o', linewidth=1.5, markersize=4, label='Edge port (full-wave)')
    if qs_results:
        s11_qs = np.array([r.S_matrix[0, 0] for r in qs_results])
        ax.plot(f_qs, 20*np.log10(np.abs(s11_qs) + 1e-30),
                'g-', linewidth=1.5, label='QS probe')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss $|S_{11}|$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- S21 phase ---
    ax = axes[1, 0]
    ax.plot(f_ghz, np.degrees(np.angle(s21)),
            'b-o', linewidth=1.5, markersize=4, label='Edge port (full-wave)')
    # 2D reference phase
    L_eff = L_STRIP  # port-to-port distance (at strip ends)
    beta_ref = 2 * np.pi * FREQS * np.sqrt(tl.eps_eff) / c0
    s21_ref_phase = np.degrees(-beta_ref * L_eff)
    # Wrap to [-180, 180]
    s21_ref_phase = (s21_ref_phase + 180) % 360 - 180
    ax.plot(f_ghz, s21_ref_phase,
            'k--', linewidth=1.5, label=rf'2D ref ($Z_0={tl.Z0:.1f}\,\Omega$)')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$\angle S_{21}$ (deg)')
    ax.set_title(r'$S_{21}$ Phase')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Z11 real and imaginary ---
    ax = axes[1, 1]
    ax.plot(f_ghz, np.real(z11), 'b-o', linewidth=1.5, markersize=4,
            label=r'$\mathrm{Re}(Z_{11})$')
    ax.plot(f_ghz, np.imag(z11), 'r-s', linewidth=1.5, markersize=4,
            label=r'$\mathrm{Im}(Z_{11})$')
    ax.axhline(tl.Z0, color='gray', linestyle='--', alpha=0.7,
               label=rf'$Z_0 = {tl.Z0:.1f}\,\Omega$')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$Z_{11}$ ($\Omega$)')
    ax.set_title(r'Port 1 Impedance $Z_{11}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = IMG_DIR / 'edge_port_validation.png'
    fig.savefig(path, dpi=150)
    print(f"\nPlot saved to {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Edge-port validation")
    parser.add_argument("--with-qs", action="store_true",
                        help="Include QS probe comparison (slower)")
    args = parser.parse_args()

    print("=" * 70)
    print("Edge-Fed Vertical Plate Port — Microstrip Validation")
    print("=" * 70)

    stack, tl, mesh, basis, port1, port2 = setup()

    # Full-wave with edge ports
    results = run_edge_port(stack, tl, mesh, basis, port1, port2)

    # Validate
    validate(results)

    # Optional QS comparison
    qs_results = None
    if args.with_qs:
        freqs_qs = np.linspace(0.5e9, 5.0e9, 30)
        qs_results = run_qs_probe(stack, tl, mesh, basis, freqs_qs)

    # Plot
    plot_results(results, tl, qs_results)


if __name__ == '__main__':
    main()
