"""
Microstrip Open Stub SOC De-embedding Validation.

Validates SOC de-embedding on a microstrip T-junction with an open stub.
The stub acts as a notch filter — S21 has a null at the quarter-wave
resonance frequency f_res = c0 / (4 * L_stub * sqrt(eps_eff)).

Physical setup:
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - Main strip: W=3.06mm (~50 Ohm), L=40mm
  - Open stub: W=3.06mm, L_stub~5.9mm, perpendicular at x=0
  - Port 1 at x=-17mm, Port 2 at x=+17mm
  - Reference planes at x=-5mm and x=+5mm

Expected results:
  - After SOC: S21 notch near f_res ~ 7 GHz
  - After SOC: S11 peak near f_res

Usage:
    source venv/bin/activate
    python examples/microstrip_stub_soc_validation.py
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
from pyMoM3d.analysis.transmission_line import microstrip_z0_hammerstad

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R    = 4.4          # FR4 relative permittivity
H_SUB    = 1.6e-3       # Substrate height (m)
W_STRIP  = 3.06e-3      # Strip width — ~50 Ohm
L_MAIN   = 40.0e-3      # Main strip total length (m)

# Stub design: quarter-wave resonance at ~7 GHz
_, EPS_EFF = microstrip_z0_hammerstad(W_STRIP, H_SUB, EPS_R)
F_RES_TARGET = 7.0e9
L_STUB   = c0 / (4.0 * F_RES_TARGET * np.sqrt(EPS_EFF))
W_STUB   = W_STRIP      # Same width as main strip

TEL = 0.7e-3            # Mesh target edge length

# Port and reference plane locations
PORT1_X  = -L_MAIN / 2.0 + 3.0e-3
PORT2_X  = +L_MAIN / 2.0 - 3.0e-3
REF1_X   = -5.0e-3
REF2_X   = +5.0e-3

# Frequency sweep — finer around expected resonance
FREQS_GHZ = np.array([4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0])
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
    print("pyMoM3d — Microstrip Open Stub SOC Validation")
    print("=" * 70)

    f_res = c0 / (4.0 * L_STUB * np.sqrt(EPS_EFF))
    print(f"\nStub design:")
    print(f"  eps_eff = {EPS_EFF:.2f} (Hammerstad)")
    print(f"  L_stub = {L_STUB*1e3:.2f} mm (quarter-wave at {f_res/1e9:.1f} GHz)")
    print(f"  W_stub = {W_STUB*1e3:.2f} mm")

    print(f"\nGeometry:")
    print(f"  Main strip: W={W_STRIP*1e3:.2f} mm, L={L_MAIN*1e3:.1f} mm")
    print(f"  Open stub at x = 0, extending in +y, L = {L_STUB*1e3:.2f} mm")
    print(f"  Port 1 at x = {PORT1_X*1e3:.1f} mm")
    print(f"  Port 2 at x = {PORT2_X*1e3:.1f} mm")
    print(f"  Ref plane 1 at x = {REF1_X*1e3:.1f} mm")
    print(f"  Ref plane 2 at x = {REF2_X*1e3:.1f} mm")

    # --- Layer stack ---
    stack = build_layer_stack()

    # --- Mesh ---
    mesher = GmshMesher(target_edge_length=TEL)
    z_mesh = H_SUB

    mesh = mesher.mesh_t_junction(
        main_width=L_MAIN, main_height=W_STRIP,
        stub_width=W_STUB, stub_height=L_STUB,
        stub_x=0.0,
        feed_x_list=[PORT1_X, REF1_X, REF2_X, PORT2_X],
        center=(0.0, 0.0, z_mesh),
    )
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    print(f"\nMesh: {stats['num_triangles']} triangles, "
          f"{basis.num_basis} RWG, "
          f"mean edge {stats['mean_edge_length']*1e3:.2f} mm")

    # --- Port definition (on main strip only) ---
    # Use y_range to ensure feed edges are on the main strip, not the stub
    y_range_main = (-W_STRIP, +W_STRIP)
    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X, y_range=y_range_main)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X, y_range=y_range_main)
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
    print(f"  Expected notch at f_res = {f_res/1e9:.1f} GHz")
    print(f"  Cal S21 range:  [{s21_cal.min():.1f}, {s21_cal.max():.1f}] dB")
    print(f"  Cal S11 range:  [{s11_cal.min():.1f}, {s11_cal.max():.1f}] dB")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        r'SOC De-embedding — Microstrip Open Stub'
        rf' ($\varepsilon_r = {EPS_R}$, $h = {H_SUB*1e3:.1f}$ mm)',
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
        title=rf'$L_{{\mathrm{{stub}}}} = {L_STUB*1e3:.1f}$ mm',
    )

    # Panel 2: S21
    ax = axes[1]
    ax.plot(FREQS_GHZ, s21_raw, 'r^--', ms=6, lw=1.5, label=r'Raw $|S_{21}|$')
    ax.plot(FREQS_GHZ, s21_cal, 'bo-', ms=6, lw=1.5, label=r'SOC $|S_{21}|$')
    ax.axvline(f_res / 1e9, color='green', ls='-', lw=2.0, alpha=0.7,
               label=rf'$f_{{\mathrm{{res}}}} = {f_res/1e9:.1f}$ GHz')
    ax.axhline(0.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{21}|$ (dB)')
    ax.set_title(r'Insertion Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: S11
    ax = axes[2]
    ax.plot(FREQS_GHZ, s11_raw, 'r^--', ms=6, lw=1.5, label=r'Raw $|S_{11}|$')
    ax.plot(FREQS_GHZ, s11_cal, 'bo-', ms=6, lw=1.5, label=r'SOC $|S_{11}|$')
    ax.axvline(f_res / 1e9, color='green', ls='-', lw=2.0, alpha=0.7,
               label=rf'$f_{{\mathrm{{res}}}} = {f_res/1e9:.1f}$ GHz')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'microstrip_stub_soc_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
