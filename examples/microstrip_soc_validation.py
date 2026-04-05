"""
Microstrip SOC De-embedding Validation.

Validates the Short-Open Calibration (SOC) de-embedding by simulating
a microstrip through-line and comparing raw vs. de-embedded S-parameters.

Physical setup:
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - PEC ground at z=0 (implicit in layered Green's function)
  - Strip: width W=3.06mm (~50 Ohm), total length = 40mm
  - Port 1 at x = -17mm, Port 2 at x = +17mm
  - Reference planes at x = -10mm and x = +10mm
  - DUT = 20mm through-line between reference planes
  - Feed networks = 7mm sections on each side

Expected results:
  - Raw: S21 = -30 to -55 dB (port discontinuity dominated)
  - After SOC: S21 > -3 dB, S11 < -10 dB (calibrated)

Usage:
    source venv/bin/activate
    python examples/microstrip_soc_validation.py
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

# Mesh density — ~4-5 cells across strip width
TEL = 0.7e-3

# Port and reference plane locations
FEED_LEN  = 15.0e-3     # Feed network length (m) on each side
PORT1_X   = -L_STRIP / 2.0 + 2.0e-3   # Port 1 position (near strip end)
PORT2_X   = +L_STRIP / 2.0 - 2.0e-3   # Port 2 position
REF1_X    = PORT1_X + FEED_LEN         # Reference plane 1
REF2_X    = PORT2_X - FEED_LEN         # Reference plane 2

# Frequency sweep — internal ungrounded port works best above ~3 GHz
FREQS_GHZ = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
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
    print("=" * 70)
    print("pyMoM3d — Microstrip SOC De-embedding Validation")
    print("=" * 70)

    print(f"\nGeometry:")
    print(f"  Strip: W={W_STRIP*1e3:.2f} mm, L={L_STRIP*1e3:.1f} mm")
    print(f"  Substrate: eps_r={EPS_R}, h={H_SUB*1e3:.1f} mm")
    print(f"  Port 1 at x = {PORT1_X*1e3:.1f} mm")
    print(f"  Port 2 at x = {PORT2_X*1e3:.1f} mm")
    print(f"  Ref plane 1 at x = {REF1_X*1e3:.1f} mm")
    print(f"  Ref plane 2 at x = {REF2_X*1e3:.1f} mm")
    print(f"  DUT length = {(REF2_X - REF1_X)*1e3:.1f} mm")
    print(f"  Feed length = {FEED_LEN*1e3:.1f} mm per side")

    # --- Layer stack ---
    stack = build_layer_stack()

    # --- Mesh ---
    mesher = GmshMesher(target_edge_length=TEL)
    z_mesh = H_SUB  # Strip at FR4/air interface

    # Include reference plane locations in feed_x_list so the mesher
    # creates conformal transverse edges there (needed for SOC mirroring)
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

    # ------------------------------------------------------------------
    # Helper: print S-parameter table with diagnostics
    # ------------------------------------------------------------------
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
            passive = s11**2 + s21**2 <= 1.0
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

    # ------------------------------------------------------------------
    # Raw extraction
    # ------------------------------------------------------------------
    extractor_raw = NetworkExtractor(
        sim, [port1, port2], store_currents=True,
    )
    results_raw = extractor_raw.extract(FREQS.tolist())
    print_s_table("Raw (uncalibrated)", results_raw, FREQS)

    # ------------------------------------------------------------------
    # SOC de-embedding
    # ------------------------------------------------------------------
    soc = SOCDeembedding(
        sim, [port1, port2],
        reference_plane_x=[REF1_X, REF2_X],
        port_x=[PORT1_X, PORT2_X],
        symmetric=False,
    )
    results_cal = []
    for result in results_raw:
        try:
            results_cal.append(soc.deembed(result))
        except Exception as e:
            print(f"  WARNING at {result.frequency/1e9:.1f} GHz: {e}")
            results_cal.append(result)
    print_s_table("SOC de-embedded", results_cal, FREQS)

    # ------------------------------------------------------------------
    # Summary & physics checks
    # ------------------------------------------------------------------
    print(f"\n--- Summary ---")
    for label, results in [("Raw", results_raw), ("SOC", results_cal)]:
        s21 = extract_db(results, 1, 0)
        s11 = extract_db(results, 0, 0)
        diffs = np.diff(s21)
        monotonic = np.all(diffs <= 0.5)
        print(f"\n  {label}:")
        print(f"    S21 range: [{s21.min():.1f}, {s21.max():.1f}] dB")
        print(f"    S11 range: [{s11.min():.1f}, {s11.max():.1f}] dB")
        print(f"    S21 monotonically decreasing: {'YES' if monotonic else 'NO'}")

    # ------------------------------------------------------------------
    # Plot: 3 panels — structure, S21, S11
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        r'SOC De-embedding — Microstrip Through-Line'
        rf' ($\varepsilon_r = {EPS_R}$, $h = {H_SUB*1e3:.1f}$ mm)',
        fontsize=12,
    )

    # Panel 1: Structure
    ax = axes[0]
    plot_structure_with_ports(
        mesh,
        port_x_list=[PORT1_X, PORT2_X],
        port_labels=['P1', 'P2'],
        reference_plane_x=[REF1_X, REF2_X],
        ax=ax,
        title=rf'$W = {W_STRIP*1e3:.1f}$ mm, $L = {L_STRIP*1e3:.0f}$ mm',
    )

    # Panel 2: S21
    ax = axes[1]
    ax.plot(FREQS_GHZ, extract_db(results_raw, 1, 0),
            'r^--', ms=6, lw=1.5, label=r'Raw $|S_{21}|$')
    ax.plot(FREQS_GHZ, extract_db(results_cal, 1, 0),
            'bo-', ms=6, lw=1.5, label=r'SOC $|S_{21}|$')
    ax.axhline(0.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(-3.0, color='gray', ls='--', lw=0.8, alpha=0.5, label=r'$-3$ dB')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{21}|$ (dB)')
    ax.set_title(r'Insertion Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: S11
    ax = axes[2]
    ax.plot(FREQS_GHZ, extract_db(results_raw, 0, 0),
            'r^--', ms=6, lw=1.5, label=r'Raw $|S_{11}|$')
    ax.plot(FREQS_GHZ, extract_db(results_cal, 0, 0),
            'bo-', ms=6, lw=1.5, label=r'SOC $|S_{11}|$')
    ax.axhline(-10.0, color='gray', ls='--', lw=0.8, alpha=0.5, label=r'$-10$ dB')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'microstrip_soc_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
