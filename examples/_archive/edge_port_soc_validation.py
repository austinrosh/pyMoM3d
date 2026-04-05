"""Edge port + SOC de-embedding validation.

Tests whether SOC de-embedding can correct edge port S-parameters to
physical values across the full frequency range.  Uses a longer strip
(40mm) to provide adequate feed network length for calibration.

Expected:
  - Raw edge port S21: -10 to -30 dB (port discontinuity dominated)
  - After SOC: S21 near 0 dB, monotonically decreasing with frequency
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
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_edge_port_feed_edges,
    find_feed_edges, compute_feed_signs,
)
from pyMoM3d.network.soc_deembedding import SOCDeembedding

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R = 4.4
H_SUB = 1.6e-3       # m
W_STRIP = 3.06e-3    # m
L_STRIP = 40.0e-3    # m (longer strip for SOC feed networks)
TEL = 0.7e-3

# Port and reference plane locations
# Edge ports at strip ends (x = +/- L/2)
PORT1_X = -L_STRIP / 2.0       # left edge
PORT2_X = +L_STRIP / 2.0       # right edge
FEED_LEN = 15.0e-3             # feed network length
REF1_X = PORT1_X + FEED_LEN    # reference plane 1
REF2_X = PORT2_X - FEED_LEN    # reference plane 2

# Frequency sweep
FREQS_GHZ = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
FREQS = FREQS_GHZ * 1e9

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])


def main():
    print("=" * 70)
    print("pyMoM3d -- Edge Port + SOC De-embedding Validation")
    print("=" * 70)

    print(f"\nGeometry:")
    print(f"  Strip: W={W_STRIP*1e3:.2f} mm, L={L_STRIP*1e3:.1f} mm")
    print(f"  Substrate: eps_r={EPS_R}, h={H_SUB*1e3:.1f} mm")
    print(f"  Edge port 1 at x = {PORT1_X*1e3:.1f} mm")
    print(f"  Edge port 2 at x = {PORT2_X*1e3:.1f} mm")
    print(f"  Ref plane 1 at x = {REF1_X*1e3:.1f} mm")
    print(f"  Ref plane 2 at x = {REF2_X*1e3:.1f} mm")
    print(f"  DUT length = {(REF2_X - REF1_X)*1e3:.1f} mm")
    print(f"  Feed length = {FEED_LEN*1e3:.1f} mm per side")

    # --- Edge port mesh ---
    mesher = GmshMesher(target_edge_length=TEL)

    # Include reference plane x-coordinates in the mesh so we get
    # conformal transverse edges there for SOC seam identification.
    # The mesher needs to know about these x-coordinates.
    mesh = mesher.mesh_microstrip_with_edge_ports(
        width=W_STRIP,
        length=L_STRIP,
        substrate_height=H_SUB,
        port_edges=['left', 'right'],
        feed_x_list=[REF1_X, REF2_X],
    )
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    print(f"\nMesh: {stats['num_triangles']} triangles, "
          f"{basis.num_basis} RWG, "
          f"mean edge {stats['mean_edge_length']*1e3:.2f} mm")

    # z-range check
    z_vals = mesh.vertices[:, 2]
    print(f"  z range: [{z_vals.min()*1e3:.3f}, {z_vals.max()*1e3:.3f}] mm")

    # --- Port definition (edge port: junction edges at strip_z) ---
    feed1 = find_edge_port_feed_edges(mesh, basis,
                                       port_x=PORT1_X, strip_z=H_SUB)
    feed2 = find_edge_port_feed_edges(mesh, basis,
                                       port_x=PORT2_X, strip_z=H_SUB)
    print(f"  Port 1: {len(feed1)} edge port feed edges")
    print(f"  Port 2: {len(feed2)} edge port feed edges")

    if not feed1 or not feed2:
        print("ERROR: Could not find edge port feed edges.")
        return

    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    # --- Simulation ---
    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=FREQS[0],
        excitation=exc,
        quad_order=4,
        backend='auto',
        layer_stack=stack,
        source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def print_s_table(label, results, freqs):
        print(f"\n--- {label} ---")
        print(f"  {'f (GHz)':>8}  {'|S11| dB':>9}  {'|S21| dB':>9}  "
              f"{'|S12| dB':>9}  {'|S22| dB':>9}  "
              f"{'|S21-S12|':>9}  {'passive':>7}  {'cond':>10}")
        print("  " + "-" * 85)
        for freq, result in zip(freqs, results):
            S = result.S_matrix
            s11, s21 = abs(S[0, 0]), abs(S[1, 0])
            s12, s22 = abs(S[0, 1]), abs(S[1, 1])
            recip_err = abs(S[1, 0] - S[0, 1])
            passive = s11**2 + s21**2 <= 1.0
            cond = getattr(result, 'condition_number', 0)
            print(f"  {freq/1e9:>8.1f}  "
                  f"{20*np.log10(max(s11, 1e-15)):>9.2f}  "
                  f"{20*np.log10(max(s21, 1e-15)):>9.2f}  "
                  f"{20*np.log10(max(s12, 1e-15)):>9.2f}  "
                  f"{20*np.log10(max(s22, 1e-15)):>9.2f}  "
                  f"{recip_err:>9.2e}  "
                  f"{'OK' if passive else 'FAIL':>7}  "
                  f"{cond:>10.1f}")

    def extract_db(results, i, j):
        return np.array([
            20 * np.log10(max(abs(r.S_matrix[i, j]), 1e-15)) for r in results
        ])

    # ------------------------------------------------------------------
    # Raw extraction
    # ------------------------------------------------------------------
    extractor = NetworkExtractor(
        sim, [port1, port2], store_currents=True,
    )
    results_raw = extractor.extract(FREQS.tolist())
    print_s_table("Raw Edge Port (uncalibrated)", results_raw, FREQS)

    # ------------------------------------------------------------------
    # SOC de-embedding with edge port support
    # ------------------------------------------------------------------
    print("\n  Setting up SOC de-embedding (strip_z for edge port)...")
    soc = SOCDeembedding(
        sim, [port1, port2],
        reference_plane_x=[REF1_X, REF2_X],
        port_x=[PORT1_X, PORT2_X],
        symmetric=False,
        strip_z=H_SUB,
    )
    results_cal = []
    for result in results_raw:
        try:
            cal = soc.deembed(result)
            results_cal.append(cal)
        except Exception as e:
            print(f"  WARNING at {result.frequency/1e9:.1f} GHz: {e}")
            results_cal.append(result)
    print_s_table("SOC De-embedded Edge Port", results_cal, FREQS)

    # ------------------------------------------------------------------
    # Flat mesh comparison (baseline)
    # ------------------------------------------------------------------
    print("\n  Building flat mesh baseline for comparison...")
    PORT1_FLAT_X = PORT1_X + 2.0e-3   # 2mm inset
    PORT2_FLAT_X = PORT2_X - 2.0e-3

    mesh_flat = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_FLAT_X, REF1_X, REF2_X, PORT2_FLAT_X],
        center=(0.0, 0.0, H_SUB),
    )
    basis_flat = compute_rwg_connectivity(mesh_flat)

    feed1_flat = find_feed_edges(mesh_flat, basis_flat, feed_x=PORT1_FLAT_X)
    feed2_flat = find_feed_edges(mesh_flat, basis_flat, feed_x=PORT2_FLAT_X)
    signs1_flat = compute_feed_signs(mesh_flat, basis_flat, feed1_flat)
    signs2_flat = compute_feed_signs(mesh_flat, basis_flat, feed2_flat)
    port1_flat = Port(name='P1', feed_basis_indices=feed1_flat,
                      feed_signs=signs1_flat)
    port2_flat = Port(name='P2', feed_basis_indices=feed2_flat,
                      feed_signs=signs2_flat)

    exc_flat = StripDeltaGapExcitation(feed_basis_indices=feed1_flat, voltage=1.0)
    config_flat = SimulationConfig(
        frequency=FREQS[0], excitation=exc_flat, quad_order=4,
        backend='auto', layer_stack=stack, source_layer_name='FR4',
    )
    sim_flat = Simulation(config_flat, mesh=mesh_flat, reporter=SilentReporter())
    print(f"  Flat mesh: {len(mesh_flat.triangles)} tri, "
          f"{basis_flat.num_basis} RWG")

    ext_flat = NetworkExtractor(sim_flat, [port1_flat, port2_flat],
                                store_currents=True)
    results_flat_raw = ext_flat.extract(FREQS.tolist())
    print_s_table("Raw Flat Mesh (uncalibrated)", results_flat_raw, FREQS)

    soc_flat = SOCDeembedding(
        sim_flat, [port1_flat, port2_flat],
        reference_plane_x=[REF1_X, REF2_X],
        port_x=[PORT1_FLAT_X, PORT2_FLAT_X],
        symmetric=False,
    )
    results_flat_cal = []
    for result in results_flat_raw:
        try:
            results_flat_cal.append(soc_flat.deembed(result))
        except Exception as e:
            print(f"  WARNING at {result.frequency/1e9:.1f} GHz: {e}")
            results_flat_cal.append(result)
    print_s_table("SOC Flat Mesh (calibrated)", results_flat_cal, FREQS)

    # ------------------------------------------------------------------
    # Summary comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY: S21 (dB)")
    print(f"{'='*70}")
    print(f"{'f (GHz)':>8}  {'Edge Raw':>10}  {'Edge SOC':>10}  "
          f"{'Flat Raw':>10}  {'Flat SOC':>10}")
    print("-" * 55)
    for i, f in enumerate(FREQS):
        s21_edge_raw = 20*np.log10(max(abs(results_raw[i].S_matrix[1,0]), 1e-15))
        s21_edge_cal = 20*np.log10(max(abs(results_cal[i].S_matrix[1,0]), 1e-15))
        s21_flat_raw = 20*np.log10(max(abs(results_flat_raw[i].S_matrix[1,0]), 1e-15))
        s21_flat_cal = 20*np.log10(max(abs(results_flat_cal[i].S_matrix[1,0]), 1e-15))
        print(f"{f/1e9:>8.1f}  {s21_edge_raw:>10.2f}  {s21_edge_cal:>10.2f}  "
              f"{s21_flat_raw:>10.2f}  {s21_flat_cal:>10.2f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        r'Edge Port + SOC vs Flat Mesh + SOC'
        rf' ($\varepsilon_r = {EPS_R}$, $h = {H_SUB*1e3:.1f}$ mm)',
        fontsize=12,
    )

    # S21
    ax = axes[0]
    ax.plot(FREQS_GHZ, extract_db(results_raw, 1, 0),
            'r^--', ms=5, lw=1, label=r'Edge Raw')
    ax.plot(FREQS_GHZ, extract_db(results_cal, 1, 0),
            'ro-', ms=6, lw=1.5, label=r'Edge SOC')
    ax.plot(FREQS_GHZ, extract_db(results_flat_raw, 1, 0),
            'bs--', ms=5, lw=1, label=r'Flat Raw')
    ax.plot(FREQS_GHZ, extract_db(results_flat_cal, 1, 0),
            'bo-', ms=6, lw=1.5, label=r'Flat SOC')
    ax.axhline(0.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(-3.0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{21}|$ (dB)')
    ax.set_title(r'Insertion Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # S11
    ax = axes[1]
    ax.plot(FREQS_GHZ, extract_db(results_cal, 0, 0),
            'ro-', ms=6, lw=1.5, label=r'Edge SOC')
    ax.plot(FREQS_GHZ, extract_db(results_flat_cal, 0, 0),
            'bo-', ms=6, lw=1.5, label=r'Flat SOC')
    ax.axhline(-10.0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss (SOC)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'edge_port_soc_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
