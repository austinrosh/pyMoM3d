"""
Coupled Microstrip Lines SOC De-embedding Validation.

Validates SOC de-embedding on a pair of coupled microstrip lines.
Two parallel strips are meshed independently and combined. Ports are
placed on strip 1, and SOC removes the feed network artifacts.

Physical setup:
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - Strip 1 and Strip 2: W=3.06mm each, L=40mm
  - Gap S=1.0mm between inner edges
  - Port 1 at x=-17mm on strip 1, Port 2 at x=+17mm on strip 1
  - Reference planes at x=-10mm and x=+10mm

Expected results:
  - After SOC: S21 near 0 dB (through on strip 1)
  - After SOC: S11 < -10 dB (matched)

Usage:
    source venv/bin/activate
    python examples/microstrip_coupled_soc_validation.py
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
from pyMoM3d.mesh.mirror import combine_meshes
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
W_STRIP  = 3.06e-3      # Strip width — ~50 Ohm
L_STRIP  = 40.0e-3      # Strip length (m)
S_GAP    = 1.0e-3       # Gap between inner edges (m)

TEL = 0.7e-3            # Mesh target edge length

# Strip y-centers
Y_STRIP1 = +(W_STRIP + S_GAP) / 2.0   # Upper strip
Y_STRIP2 = -(W_STRIP + S_GAP) / 2.0   # Lower strip

# Port and reference plane locations (on strip 1)
PORT1_X  = -L_STRIP / 2.0 + 3.0e-3
PORT2_X  = +L_STRIP / 2.0 - 3.0e-3
REF1_X   = -10.0e-3
REF2_X   = +10.0e-3

# Frequency sweep
FREQS_GHZ = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
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
    print("pyMoM3d — Coupled Microstrip Lines SOC Validation")
    print("=" * 70)

    print(f"\nGeometry:")
    print(f"  Strip width: W = {W_STRIP*1e3:.2f} mm")
    print(f"  Strip length: L = {L_STRIP*1e3:.1f} mm")
    print(f"  Gap: S = {S_GAP*1e3:.1f} mm")
    print(f"  Strip 1 center y = {Y_STRIP1*1e3:.2f} mm")
    print(f"  Strip 2 center y = {Y_STRIP2*1e3:.2f} mm")
    print(f"  Port 1 at x = {PORT1_X*1e3:.1f} mm (strip 1)")
    print(f"  Port 2 at x = {PORT2_X*1e3:.1f} mm (strip 1)")
    print(f"  Ref plane 1 at x = {REF1_X*1e3:.1f} mm")
    print(f"  Ref plane 2 at x = {REF2_X*1e3:.1f} mm")

    # --- Layer stack ---
    stack = build_layer_stack()

    # --- Mesh two strips independently ---
    mesher = GmshMesher(target_edge_length=TEL)
    z_mesh = H_SUB

    mesh1 = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, REF1_X, REF2_X, PORT2_X],
        center=(0.0, Y_STRIP1, z_mesh),
    )
    mesh2 = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, REF1_X, REF2_X, PORT2_X],
        center=(0.0, Y_STRIP2, z_mesh),
    )

    # Combine meshes (strips don't share vertices — just concatenation)
    mesh = combine_meshes(mesh1, mesh2)
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    print(f"\nMesh: {stats['num_triangles']} triangles, "
          f"{basis.num_basis} RWG, "
          f"mean edge {stats['mean_edge_length']*1e3:.2f} mm")

    # --- Port definition on strip 1 only ---
    # Use y_range to select edges only on strip 1
    y_lo_s1 = Y_STRIP1 - W_STRIP
    y_hi_s1 = Y_STRIP1 + W_STRIP
    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X,
                            y_range=(y_lo_s1, y_hi_s1))
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X,
                            y_range=(y_lo_s1, y_hi_s1))
    print(f"  Port 1: {len(feed1)} edges at x = {PORT1_X*1e3:.1f} mm")
    print(f"  Port 2: {len(feed2)} edges at x = {PORT2_X*1e3:.1f} mm")

    if not feed1 or not feed2:
        print("ERROR: Could not find feed edges on strip 1.")
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

    # --- Raw extraction ---
    print(f"\n--- Raw (uncalibrated) extraction ---")
    extractor_raw = NetworkExtractor(sim, [port1, port2], store_currents=True)
    results_raw = extractor_raw.extract(FREQS.tolist())

    print(f"  {'f (GHz)':>8}  {'|S11| dB':>9}  {'|S21| dB':>9}")
    print("  " + "-" * 30)
    for freq, result in zip(FREQS, results_raw):
        S = result.S_matrix
        print(f"  {freq/1e9:>8.1f}  "
              f"{20*np.log10(max(abs(S[0,0]), 1e-15)):>9.2f}  "
              f"{20*np.log10(max(abs(S[1,0]), 1e-15)):>9.2f}")

    # --- SOC de-embedding ---
    print(f"\n--- SOC de-embedding ---")
    soc = SOCDeembedding(
        sim, [port1, port2],
        reference_plane_x=[REF1_X, REF2_X],
        port_x=[PORT1_X, PORT2_X],
        symmetric=False,
    )

    results_cal = []
    for result in results_raw:
        try:
            cal = soc.deembed(result)
            results_cal.append(cal)
        except Exception as e:
            print(f"  WARNING at {result.frequency/1e9:.1f} GHz: {e}")
            results_cal.append(result)

    print(f"\n  {'f (GHz)':>8}  {'|S11| dB':>9}  {'|S21| dB':>9}  "
          f"{'|S12| dB':>9}  {'|S22| dB':>9}")
    print("  " + "-" * 50)
    for freq, result in zip(FREQS, results_cal):
        S = result.S_matrix
        print(f"  {freq/1e9:>8.1f}  "
              f"{20*np.log10(max(abs(S[0,0]), 1e-15)):>9.2f}  "
              f"{20*np.log10(max(abs(S[1,0]), 1e-15)):>9.2f}  "
              f"{20*np.log10(max(abs(S[0,1]), 1e-15)):>9.2f}  "
              f"{20*np.log10(max(abs(S[1,1]), 1e-15)):>9.2f}")

    # --- Summary ---
    s21_raw = np.array([
        20 * np.log10(max(abs(r.S_matrix[1, 0]), 1e-15)) for r in results_raw
    ])
    s21_cal = np.array([
        20 * np.log10(max(abs(r.S_matrix[1, 0]), 1e-15)) for r in results_cal
    ])
    s11_raw = np.array([
        20 * np.log10(max(abs(r.S_matrix[0, 0]), 1e-15)) for r in results_raw
    ])
    s11_cal = np.array([
        20 * np.log10(max(abs(r.S_matrix[0, 0]), 1e-15)) for r in results_cal
    ])

    print(f"\n--- Summary ---")
    print(f"  Cal S21 range:  [{s21_cal.min():.1f}, {s21_cal.max():.1f}] dB")
    print(f"  Cal S11 range:  [{s11_cal.min():.1f}, {s11_cal.max():.1f}] dB")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        r'SOC De-embedding — Coupled Microstrip Lines'
        rf' ($\varepsilon_r = {EPS_R}$, $S = {S_GAP*1e3:.1f}$ mm)',
        fontsize=12,
    )

    # Panel 1: Structure visualization
    ax = axes[0]
    plot_structure_with_ports(
        mesh,
        port_x_list=[PORT1_X, PORT2_X],
        port_labels=['P1', 'P2'],
        reference_plane_x=[REF1_X, REF2_X],
        ax=ax,
        title=rf'$W = {W_STRIP*1e3:.1f}$ mm, gap $= {S_GAP*1e3:.1f}$ mm',
    )

    # Panel 2: S21
    ax = axes[1]
    ax.plot(FREQS_GHZ, s21_raw, 'r^--', ms=6, lw=1.5, label=r'Raw $|S_{21}|$')
    ax.plot(FREQS_GHZ, s21_cal, 'bo-', ms=6, lw=1.5, label=r'SOC $|S_{21}|$')
    ax.axhline(0.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(-3.0, color='gray', ls='--', lw=0.8, alpha=0.5, label=r'$-3$ dB')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{21}|$ (dB)')
    ax.set_title(r'Insertion Loss (Strip 1)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: S11
    ax = axes[2]
    ax.plot(FREQS_GHZ, s11_raw, 'r^--', ms=6, lw=1.5, label=r'Raw $|S_{11}|$')
    ax.plot(FREQS_GHZ, s11_cal, 'bo-', ms=6, lw=1.5, label=r'SOC $|S_{11}|$')
    ax.axhline(-10.0, color='gray', ls='--', lw=0.8, alpha=0.5, label=r'$-10$ dB')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss (Strip 1)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'microstrip_coupled_soc_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
