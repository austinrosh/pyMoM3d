"""
Quasi-Static Solver Validation — Microstrip Through-Line.

Compares the quasi-static solver (one matrix fill, algebraic sweep)
against the full-wave MPIE solver on a microstrip through-line.

Key validation criteria:
  - S21 near 0 dB at ALL frequencies (including low freq where full-wave fails)
  - S21 monotonically decreasing (or flat) with frequency
  - S11 well below S21 at all frequencies
  - Passivity: |S11|² + |S21|² ≤ 1
  - Reciprocity: S21 ≈ S12

Usage:
    source venv/bin/activate
    python examples/microstrip_quasi_static_validation.py
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
    plot_structure_with_ports,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.network.soc_deembedding import SOCDeembedding

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R    = 4.4          # FR4 relative permittivity
H_SUB    = 1.6e-3       # Substrate height (m)
W_STRIP  = 3.06e-3      # Strip width (m) — approx 50 Ohm on FR4 1.6mm
L_STRIP  = 40.0e-3      # Total strip length (m)

TEL = 0.7e-3            # Mesh target edge length

# Port and reference plane locations
FEED_LEN  = 15.0e-3
PORT1_X   = -L_STRIP / 2.0 + 2.0e-3
PORT2_X   = +L_STRIP / 2.0 - 2.0e-3
REF1_X    = PORT1_X + FEED_LEN
REF2_X    = PORT2_X - FEED_LEN

# Frequency sweep — full range including low frequencies
FREQS_GHZ = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
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
# Helpers
# ---------------------------------------------------------------------------

def print_s_table(label, results, freqs):
    print(f"\n--- {label} ---")
    print(f"  {'f (GHz)':>8}  {'|S11| dB':>9}  {'|S21| dB':>9}  "
          f"{'|S12| dB':>9}  {'|S22| dB':>9}  "
          f"{'|S21-S12|':>9}  {'|S11-S22|':>9}  {'passive':>7}")
    print("  " + "-" * 80)
    for freq, result in zip(freqs, results):
        S = result.S_matrix
        s11 = abs(S[0, 0])
        s21 = abs(S[1, 0])
        s12 = abs(S[0, 1])
        s22 = abs(S[1, 1])
        recip_err = abs(S[1, 0] - S[0, 1])
        symm_err = abs(S[0, 0] - S[1, 1])
        passive = s11**2 + s21**2 <= 1.001  # small tolerance
        print(f"  {freq/1e9:>8.1f}  "
              f"{20*np.log10(max(s11, 1e-15)):>9.2f}  "
              f"{20*np.log10(max(s21, 1e-15)):>9.2f}  "
              f"{20*np.log10(max(s12, 1e-15)):>9.2f}  "
              f"{20*np.log10(max(s22, 1e-15)):>9.2f}  "
              f"{recip_err:>9.2e}  {symm_err:>9.2e}  "
              f"{'OK' if passive else 'FAIL':>7}")


def extract_db(results, i, j):
    return np.array([
        20 * np.log10(max(abs(r.S_matrix[i, j]), 1e-15)) for r in results
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("pyMoM3d — Quasi-Static Solver Validation")
    print("=" * 70)

    print(f"\nGeometry:")
    print(f"  Strip: W={W_STRIP*1e3:.2f} mm, L={L_STRIP*1e3:.1f} mm")
    print(f"  Substrate: eps_r={EPS_R}, h={H_SUB*1e3:.1f} mm")
    print(f"  Port 1 at x = {PORT1_X*1e3:.1f} mm")
    print(f"  Port 2 at x = {PORT2_X*1e3:.1f} mm")
    print(f"  Ref plane 1 at x = {REF1_X*1e3:.1f} mm")
    print(f"  Ref plane 2 at x = {REF2_X*1e3:.1f} mm")

    stack = build_layer_stack()

    # --- Mesh ---
    mesher = GmshMesher(target_edge_length=TEL)
    z_mesh = H_SUB

    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, REF1_X, REF2_X, PORT2_X],
        center=(0.0, 0.0, z_mesh),
    )
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    print(f"\nMesh: {stats['num_triangles']} triangles, "
          f"{basis.num_basis} RWG, "
          f"mean edge {stats['mean_edge_length']*1e3:.2f} mm")

    # --- Port definition ---
    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
    print(f"  Port 1: {len(feed1)} edges at x = {PORT1_X*1e3:.1f} mm")
    print(f"  Port 2: {len(feed2)} edges at x = {PORT2_X*1e3:.1f} mm")

    if not feed1 or not feed2:
        print("ERROR: Could not find feed edges.")
        return

    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    # --- Build Simulation (needed for config/mesh/basis) ---
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

    # ==================================================================
    # A) Full-wave MPIE + SOC (existing solver)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"A) Full-wave MPIE + SOC")
    print(f"{'='*70}")

    extractor_fw = NetworkExtractor(
        sim, [port1, port2], store_currents=True,
    )
    results_fw = extractor_fw.extract(FREQS.tolist())
    print_s_table("Full-wave raw", results_fw, FREQS)

    soc = SOCDeembedding(
        sim, [port1, port2],
        reference_plane_x=[REF1_X, REF2_X],
        port_x=[PORT1_X, PORT2_X],
        symmetric=False,
    )
    results_fw_soc = []
    for result in results_fw:
        try:
            results_fw_soc.append(soc.deembed(result))
        except Exception as e:
            print(f"  WARNING at {result.frequency/1e9:.1f} GHz: {e}")
            results_fw_soc.append(result)
    print_s_table("Full-wave + SOC", results_fw_soc, FREQS)

    # ==================================================================
    # B) Quasi-static solver (PEC images only)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"B) Quasi-static solver (PEC image, no dielectric correction)")
    print(f"{'='*70}")

    import time
    t0 = time.perf_counter()
    qs_solver = QuasiStaticSolver(
        sim, [port1, port2],
        store_currents=True,
        n_dielectric_images=0,
    )
    t_fill = time.perf_counter() - t0
    print(f"  Matrix fill: {t_fill:.2f} s (one-time)")

    t0 = time.perf_counter()
    results_qs = qs_solver.extract(FREQS.tolist())
    t_sweep = time.perf_counter() - t0
    print(f"  Frequency sweep ({len(FREQS)} pts): {t_sweep:.3f} s")
    print_s_table("Quasi-static (PEC only)", results_qs, FREQS)

    # ==================================================================
    # C) Z-matrix comparison (QS vs full-wave, key validation)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"C) Z-matrix comparison — QS vs Full-wave (raw)")
    print(f"{'='*70}")
    print(f"\n  The quasi-static Z-matrix should match full-wave at low kD.")
    print(f"  Port S-parameters are limited by the strip delta-gap port model")
    print(f"  (same for both solvers), so Z-matrix agreement is the key metric.")

    print(f"\n  {'f (GHz)':>8}  {'kD':>5}  {'|Z11_QS/Z11_FW|':>16}  "
          f"{'Z11_QS':>20}  {'Z11_FW':>20}")
    print(f"  {'-'*80}")
    for freq, rq, rf in zip(FREQS, results_qs, results_fw):
        kD = 2 * np.pi * freq / c0 * L_STRIP
        zq = rq.Z_matrix[0, 0]
        zf = rf.Z_matrix[0, 0]
        ratio = abs(zq) / max(abs(zf), 1e-30)
        print(f"  {freq/1e9:>8.1f}  {kD:>5.2f}  {ratio:>16.3f}  "
              f"{zq.real:>+9.2f}{zq.imag:>+9.2f}j  "
              f"{zf.real:>+9.2f}{zf.imag:>+9.2f}j")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    datasets = [
        ("Full-wave raw", results_fw),
        ("Full-wave + SOC", results_fw_soc),
        ("QS (PEC only)", results_qs),
    ]
    for label, results in datasets:
        s21 = extract_db(results, 1, 0)
        s11 = extract_db(results, 0, 0)
        diffs = np.diff(s21)
        monotonic = np.all(diffs <= 0.5)
        print(f"\n  {label}:")
        print(f"    S21 range: [{s21.min():.1f}, {s21.max():.1f}] dB")
        print(f"    S11 range: [{s11.min():.1f}, {s11.max():.1f}] dB")
        print(f"    S21 monotonically decreasing: {'YES' if monotonic else 'NO'}")
        if not monotonic:
            print(f"    Worst S21 increase: {diffs.max():.2f} dB")

    # ==================================================================
    # Plot
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        r'Quasi-Static vs Full-Wave — Microstrip Through-Line'
        rf' ($\varepsilon_r = {EPS_R}$, $h = {H_SUB*1e3:.1f}$ mm)',
        fontsize=12,
    )

    # Panel 1: S21 comparison
    ax = axes[0]
    ax.plot(FREQS_GHZ, extract_db(results_fw, 1, 0),
            'ks--', ms=4, lw=0.8, label=r'Full-wave (raw)')
    ax.plot(FREQS_GHZ, extract_db(results_fw_soc, 1, 0),
            'rs-', ms=5, lw=1, label=r'Full-wave + SOC')
    ax.plot(FREQS_GHZ, extract_db(results_qs, 1, 0),
            'b^--', ms=5, lw=1, label=r'Quasi-static')
    ax.axhline(0.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{21}|$ (dB)')
    ax.set_title(r'Insertion Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: S11 comparison
    ax = axes[1]
    ax.plot(FREQS_GHZ, extract_db(results_fw, 0, 0),
            'ks--', ms=4, lw=0.8, label=r'Full-wave (raw)')
    ax.plot(FREQS_GHZ, extract_db(results_fw_soc, 0, 0),
            'rs-', ms=5, lw=1, label=r'Full-wave + SOC')
    ax.plot(FREQS_GHZ, extract_db(results_qs, 0, 0),
            'b^--', ms=5, lw=1, label=r'Quasi-static')
    ax.axhline(-10.0, color='gray', ls='--', lw=0.8, alpha=0.5, label=r'$-10$ dB')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Z-matrix ratio (key validation)
    ax = axes[2]
    z_ratio = np.array([
        abs(rq.Z_matrix[0, 0]) / max(abs(rf.Z_matrix[0, 0]), 1e-30)
        for rq, rf in zip(results_qs, results_fw)
    ])
    kD = 2 * np.pi * FREQS / c0 * L_STRIP
    ax.plot(kD, z_ratio, 'b^-', ms=5, lw=1.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$kD$')
    ax.set_ylabel(r'$|Z_{11}^{\mathrm{QS}}| \,/\, |Z_{11}^{\mathrm{FW}}|$')
    ax.set_title(r'Z-matrix agreement (target $= 1$)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)

    out = os.path.join(IMAGES_DIR, 'microstrip_quasi_static_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
