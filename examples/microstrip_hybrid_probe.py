"""
Microstrip Z0 Extraction with Wire-Surface Hybrid Probe Feed.

Uses vertical wire probe feeds (wire-surface hybrid MoM) to properly
excite the quasi-TEM mode of a microstrip transmission line and extract
characteristic impedance Z0 and effective permittivity eps_eff.

The hybrid MoM assembles a block-structured impedance matrix:

    Z = [ Z_ss   Z_sw ]
        [ Z_ws   Z_ww ]

where Z_ss uses the multilayer Green's function (Strata EFIE) and the
wire blocks use free-space thin-wire EFIE kernels.

Physical setup:
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - PEC ground at z=0
  - Strip at z=h, width W=3.06mm (≈50 Ohm), length L=30mm
  - Vertical probe wires from z=0 to z=h at each end

Produces:
  images/microstrip_hybrid_probe.png

Usage:
    source venv/bin/activate
    python examples/microstrip_hybrid_probe.py
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
    configure_latex_style, c0, eta0,
    microstrip_z0_hammerstad, extract_z0_from_s,
    WireMesh, compute_wire_connectivity,
    HybridBasis,
)
from pyMoM3d.wire.probe_port import create_probe_port
from pyMoM3d.mom.excitation import StripDeltaGapExcitation

configure_latex_style(use_tex=False)

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R = 4.4          # FR4
H_SUB = 1.6e-3       # Substrate height (m)
W_STRIP = 3.06e-3    # Strip width (m) — ~50 Ohm on FR4
L_STRIP = 30.0e-3    # Strip length (m)
WIRE_RADIUS = 0.25e-3  # Probe wire radius (m)
N_WIRE_SEG = 3        # Wire segments per probe

# Mesh
TEL = 2.0e-3          # Target edge length (m)

# Frequency sweep
FREQS_GHZ = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
FREQS = FREQS_GHZ * 1e9


def build_layer_stack():
    """Microstrip: PEC ground -> FR4 substrate -> air."""
    return LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])


def main():
    print("=" * 65)
    print("pyMoM3d — Microstrip Z0 with Hybrid Wire Probe Feed")
    print("=" * 65)

    # Analytical reference
    Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W_STRIP, H_SUB, EPS_R)
    print(f"\nHammerstad-Jensen: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

    # --- Build surface mesh (strip at z = H) ---
    stack = build_layer_stack()

    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[-L_STRIP / 2.0, L_STRIP / 2.0],
        center=(0.0, 0.0, H_SUB),
    )
    basis = compute_rwg_connectivity(mesh)
    N_s = basis.num_basis
    stats = mesh.get_statistics()
    print(f"\nSurface mesh: {stats['num_triangles']} tris, {N_s} RWG basis")
    print(f"  Strip: {L_STRIP*1e3:.1f} mm x {W_STRIP*1e3:.2f} mm at z = {H_SUB*1e3:.1f} mm")

    # --- Build probe wires ---
    # Probe 1 at left end, probe 2 at right end
    x1 = -L_STRIP / 2.0
    x2 = L_STRIP / 2.0

    wire1 = WireMesh.vertical_probe(x1, 0.0, 0.0, H_SUB, WIRE_RADIUS, N_WIRE_SEG)
    wire2 = WireMesh.vertical_probe(x2, 0.0, 0.0, H_SUB, WIRE_RADIUS, N_WIRE_SEG)

    # Merge both wires into a single WireMesh
    # Wire 2 nodes are offset by wire1.num_nodes
    n1 = wire1.num_nodes
    merged_nodes = np.vstack([wire1.nodes, wire2.nodes])
    merged_segments = list(wire1.segments)
    for seg in wire2.segments:
        from pyMoM3d.wire.wire_basis import WireSegment
        merged_segments.append(WireSegment(
            node_start=seg.node_start + n1,
            node_end=seg.node_end + n1,
            length=seg.length,
            direction=seg.direction.copy(),
            radius=seg.radius,
        ))
    wire_mesh = WireMesh(nodes=merged_nodes, segments=merged_segments)
    wire_basis = compute_wire_connectivity(wire_mesh)
    N_w = wire_basis.num_basis

    print(f"\nWire mesh: {wire_mesh.num_segments} segments, {N_w} wire basis")
    print(f"  Probe 1 at x = {x1*1e3:.1f} mm")
    print(f"  Probe 2 at x = {x2*1e3:.1f} mm")
    print(f"  Wire radius: {WIRE_RADIUS*1e3:.2f} mm, {N_WIRE_SEG} segments each")

    # --- Hybrid basis ---
    hybrid = HybridBasis(rwg_basis=basis, wire_basis=wire_basis, wire_mesh=wire_mesh)
    print(f"\nHybrid system: {hybrid.num_total} total unknowns "
          f"({N_s} surface + {N_w} wire)")

    # --- Ports at wire bases ---
    # Wire 1 has basis functions 0..N_w1-1 (N_w1 = N_WIRE_SEG - 1)
    # Wire 2 has basis functions N_w1..N_w-1
    N_w1 = N_WIRE_SEG - 1  # basis functions for wire 1

    # Port 1: first wire basis function (base of wire 1)
    port1 = Port(
        name='P1',
        feed_basis_indices=[N_s + 0],
        feed_signs=[+1],
    )

    # Port 2: first wire basis function of wire 2 (base of wire 2)
    port2 = Port(
        name='P2',
        feed_basis_indices=[N_s + N_w1],
        feed_signs=[+1],
    )

    print(f"  Port 1: wire basis index {0} (global {N_s})")
    print(f"  Port 2: wire basis index {N_w1} (global {N_s + N_w1})")

    # --- Simulation config ---
    config = SimulationConfig(
        frequency=FREQS[0],
        excitation=StripDeltaGapExcitation([], voltage=1.0),
        quad_order=4,
        layer_stack=stack,
        source_layer_name='FR4',
        backend='auto',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # --- Extraction ---
    extractor = NetworkExtractor(
        sim,
        [port1, port2],
        Z0=50.0,
        hybrid_basis=hybrid,
    )

    print(f"\n{'f (GHz)':>8}  {'Z_in (Ohm)':>12}  {'Z0 (Ohm)':>10}  {'Z0_err':>8}  "
          f"{'|S21| dB':>10}  {'|S11| dB':>10}  {'cond':>10}")
    print("  " + "-" * 80)

    Z0_arr = []
    S21_dB_arr = []
    S11_dB_arr = []

    for freq in FREQS:
        results = extractor.extract([freq])
        r = results[0]

        Z_in = r.Z_matrix[0, 0]
        S = r.S_matrix
        S11_dB = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
        S21_dB = 20 * np.log10(max(abs(S[1, 0]), 1e-30))

        try:
            Z0_ext = abs(extract_z0_from_s(S, Z0_ref=50.0))
        except Exception:
            Z0_ext = float('nan')

        Z0_arr.append(Z0_ext)
        S21_dB_arr.append(S21_dB)
        S11_dB_arr.append(S11_dB)

        err = (Z0_ext - Z0_ref) / Z0_ref * 100 if Z0_ref > 0 else float('nan')
        cond = getattr(extractor, '_last_cond', float('nan'))
        print(f"  {freq/1e9:>6.2f}  {abs(Z_in):>12.2f}  {Z0_ext:>10.2f}  "
              f"{err:>+7.1f}%  {S21_dB:>10.2f}  {S11_dB:>10.2f}  {cond:>10.1f}")

    Z0_arr = np.array(Z0_arr)
    S21_dB_arr = np.array(S21_dB_arr)
    S11_dB_arr = np.array(S11_dB_arr)

    # Summary
    valid = np.isfinite(Z0_arr) & (Z0_arr > 0)
    if np.any(valid):
        mean_Z0 = np.mean(Z0_arr[valid])
        mean_err = abs(mean_Z0 - Z0_ref) / Z0_ref * 100
        print(f"\n  Mean Z0: {mean_Z0:.2f} Ohm  (ref: {Z0_ref:.2f} Ohm)")
        print(f"  Mean error: {mean_err:.1f}%  "
              f"{'PASS (<15%)' if mean_err < 15 else 'FAIL'}")
    else:
        mean_Z0 = float('nan')
        print("\n  No valid Z0 values extracted")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(FREQS_GHZ, Z0_arr, 'bo-', ms=6, lw=1.5, label='MoM (hybrid probe)')
    ax1.axhline(Z0_ref, color='r', ls='--', lw=1.5,
                label=f'Hammerstad (Z0 = {Z0_ref:.1f} Ohm)')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Z0 (Ohm)')
    ax1.set_title('Characteristic Impedance')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    if np.any(valid):
        ax1.set_ylim(0, max(200, 2 * Z0_ref))

    ax2.plot(FREQS_GHZ, S21_dB_arr, 'go-', ms=6, lw=1.5, label='|S21|')
    ax2.plot(FREQS_GHZ, S11_dB_arr, 'rs-', ms=6, lw=1.5, label='|S11|')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('S-Parameters')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Microstrip Hybrid Probe Feed '
                 f'(eps_r={EPS_R}, h={H_SUB*1e3:.1f}mm, W={W_STRIP*1e3:.2f}mm)')
    plt.tight_layout()

    out = os.path.join(IMAGES_DIR, 'microstrip_hybrid_probe.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
