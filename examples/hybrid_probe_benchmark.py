"""Hybrid probe benchmark: full-wave with vertical wire probes across 0.5–12 GHz.

Compares three extraction methods on a 10mm FR4 microstrip through-line:
1. QS solver with probe feeds (valid kD << 1)
2. Full-wave hybrid: Strata DCIM surface + free-space wire probes
3. Full-wave strip delta-gap (no ground return, baseline)

Also plots 2D TL reference phase prediction.

Saves to images/benchmark/.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor,
    Layer, LayerStack,
    configure_latex_style, c0,
    WireMesh, compute_wire_connectivity,
    HybridBasis, detect_junctions,
)
from pyMoM3d.wire.wire_basis import WireSegment
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.cross_section import compute_reference_impedance

try:
    configure_latex_style()
except Exception:
    configure_latex_style(use_tex=False)

IMG_DIR = Path(__file__).resolve().parent.parent / 'images' / 'benchmark'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Physical parameters
# ============================================================================

EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
L_STRIP = 10.0e-3
TEL = 0.7e-3

# Probe wire parameters
WIRE_RADIUS = 0.25e-3
N_WIRE_SEG = 3

# Port positions — at strip ends
X1 = -L_STRIP / 2.0
X2 = +L_STRIP / 2.0

# For QS and delta-gap: ports 1mm inboard
PORT1_X = X1 + 1.0e-3
PORT2_X = X2 - 1.0e-3

# Frequency sweeps
FREQS_QS = np.linspace(0.1e9, 4.0e9, 20)
FREQS_FW = np.linspace(0.5e9, 12.0e9, 24)


# ============================================================================
# Setup
# ============================================================================

def build_stack():
    return LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])


def build_mesh_and_basis(stack):
    """Build surface mesh with feed lines for all port types."""
    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, PORT2_X],
        center=(0.0, 0.0, H_SUB),
    )
    basis = compute_rwg_connectivity(mesh)
    return mesh, basis


def build_wire_probes(mesh):
    """Build vertical probe wires at strip ends, centered in y."""
    # Probe positions: strip x-ends, y=0 (center of strip width)
    wire1 = WireMesh.vertical_probe(X1, 0.0, 0.0, H_SUB, WIRE_RADIUS, N_WIRE_SEG)
    wire2 = WireMesh.vertical_probe(X2, 0.0, 0.0, H_SUB, WIRE_RADIUS, N_WIRE_SEG)

    # Merge wires
    n1 = wire1.num_nodes
    merged_nodes = np.vstack([wire1.nodes, wire2.nodes])
    merged_segments = list(wire1.segments)
    for seg in wire2.segments:
        merged_segments.append(WireSegment(
            node_start=seg.node_start + n1,
            node_end=seg.node_end + n1,
            length=seg.length,
            direction=seg.direction.copy(),
            radius=seg.radius,
        ))
    wire_mesh = WireMesh(nodes=merged_nodes, segments=merged_segments)
    wire_basis = compute_wire_connectivity(wire_mesh)
    return wire_mesh, wire_basis


# ============================================================================
# Solver runs
# ============================================================================

def run_qs_probe(stack, tl, mesh, basis):
    """QS solver with probe feeds."""
    print("\n--- QS Solver (probe feeds) ---")
    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=1e9, excitation=exc, quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    qs = QuasiStaticSolver(sim, [port1, port2], Z0=tl.Z0, probe_feeds=True)

    t0 = time.time()
    results = qs.extract(FREQS_QS.tolist())
    dt = time.time() - t0
    print(f"  {len(FREQS_QS)} frequencies in {dt:.1f}s")
    return results


def run_hybrid_probe(stack, tl, mesh, basis):
    """Full-wave hybrid: Strata surface + free-space wire probes."""
    print("\n--- Full-wave Hybrid Probe ---")
    N_s = basis.num_basis

    wire_mesh, wire_basis = build_wire_probes(mesh)
    N_w = wire_basis.num_basis
    N_w1 = N_WIRE_SEG - 1  # basis functions per wire

    junctions = detect_junctions(wire_mesh, mesh)
    hybrid = HybridBasis(
        rwg_basis=basis, wire_basis=wire_basis, wire_mesh=wire_mesh,
        junctions=junctions,
    )
    N_j = hybrid.num_junctions
    print(f"  {hybrid.num_total} unknowns ({N_s} surface + {N_w} wire + {N_j} junction)")
    for j in junctions:
        pos = wire_mesh.nodes[j.wire_node_idx]
        sv = mesh.vertices[j.surface_vertex_idx]
        print(f"    Junction: wire node {j.wire_node_idx} ({pos}) "
              f"→ surface vertex {j.surface_vertex_idx} ({sv})")

    # Ports at wire bases (ground end, first basis function of each wire)
    port1 = Port(name='P1', feed_basis_indices=[N_s + 0], feed_signs=[+1])
    port2 = Port(name='P2', feed_basis_indices=[N_s + N_w1], feed_signs=[+1])

    exc = StripDeltaGapExcitation(feed_basis_indices=[], voltage=1.0)
    config = SimulationConfig(
        frequency=1e9, excitation=exc, quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    extractor = NetworkExtractor(
        sim, [port1, port2], Z0=tl.Z0, hybrid_basis=hybrid,
    )

    t0 = time.time()
    results = extractor.extract(FREQS_FW.tolist())
    dt = time.time() - t0
    print(f"  {len(FREQS_FW)} frequencies in {dt:.1f}s "
          f"({dt/len(FREQS_FW)*1e3:.0f} ms/freq)")
    return results


def run_delta_gap(stack, tl, mesh, basis):
    """Full-wave strip delta-gap (no ground return)."""
    print("\n--- Full-wave Strip Delta-Gap ---")
    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=1e9, excitation=exc, quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    extractor = NetworkExtractor(sim, [port1, port2], Z0=tl.Z0)

    t0 = time.time()
    results = extractor.extract(FREQS_FW.tolist())
    dt = time.time() - t0
    print(f"  {len(FREQS_FW)} frequencies in {dt:.1f}s "
          f"({dt/len(FREQS_FW)*1e3:.0f} ms/freq)")
    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_benchmark(qs_results, hybrid_results, dg_results, tl):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    f_qs = FREQS_QS / 1e9
    f_fw = FREQS_FW / 1e9
    eps_eff = tl.eps_eff

    # Extract S-params
    s21_qs = np.array([r.S_matrix[1, 0] for r in qs_results])
    s11_qs = np.array([r.S_matrix[0, 0] for r in qs_results])
    s21_hyb = np.array([r.S_matrix[1, 0] for r in hybrid_results])
    s11_hyb = np.array([r.S_matrix[0, 0] for r in hybrid_results])
    s21_dg = np.array([r.S_matrix[1, 0] for r in dg_results])
    s11_dg = np.array([r.S_matrix[0, 0] for r in dg_results])

    # kD markers
    f_kd05 = 0.5 * c0 / (2 * np.pi * np.sqrt(eps_eff) * L_STRIP) / 1e9

    # 2D reference phase
    L_eff = abs(X2 - X1)
    beta_2d = np.array([tl.beta(f) for f in FREQS_FW])
    phase_2d = np.degrees(-beta_2d * L_eff)

    # --- S21 magnitude ---
    ax = axes[0, 0]
    ax.plot(f_qs, 20*np.log10(np.abs(s21_qs)+1e-30),
            'b-', linewidth=1.5, label='QS probe', alpha=0.7)
    ax.plot(f_fw, 20*np.log10(np.abs(s21_hyb)+1e-30),
            'g-s', linewidth=2, markersize=4, label='Hybrid probe (full-wave)')
    ax.plot(f_fw, 20*np.log10(np.abs(s21_dg)+1e-30),
            'r--o', linewidth=1, markersize=3, alpha=0.6, label='Strip delta-gap')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(-1, color='orange', linestyle=':', alpha=0.5, label='$-1$ dB target')
    ax.axvline(f_kd05, color='orange', linestyle='--', alpha=0.4,
               label=f'$kD = 0.5$ ({f_kd05:.1f} GHz)')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{21}|$ (dB)')
    ax.set_title(r'Insertion Loss $|S_{21}|$')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 12)

    # --- S11 magnitude ---
    ax = axes[0, 1]
    ax.plot(f_qs, 20*np.log10(np.abs(s11_qs)+1e-30),
            'b-', linewidth=1.5, label='QS probe', alpha=0.7)
    ax.plot(f_fw, 20*np.log10(np.abs(s11_hyb)+1e-30),
            'g-s', linewidth=2, markersize=4, label='Hybrid probe')
    ax.plot(f_fw, 20*np.log10(np.abs(s11_dg)+1e-30),
            'r--o', linewidth=1, markersize=3, alpha=0.6, label='Strip delta-gap')
    ax.axvline(f_kd05, color='orange', linestyle='--', alpha=0.4)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss $|S_{11}|$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 12)

    # --- S21 phase ---
    ax = axes[1, 0]
    ax.plot(f_qs, np.degrees(np.angle(s21_qs)),
            'b-', linewidth=1.5, label='QS probe', alpha=0.7)
    ax.plot(f_fw, np.degrees(np.angle(s21_hyb)),
            'g-s', linewidth=2, markersize=4, label='Hybrid probe')
    ax.plot(f_fw, np.degrees(np.angle(s21_dg)),
            'r--o', linewidth=1, markersize=3, alpha=0.6, label='Strip delta-gap')
    ax.plot(f_fw, phase_2d,
            'k--', linewidth=1.5, label=r'2D: $-\beta L$')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$\angle S_{21}$ (deg)')
    ax.set_title(r'Phase $\angle S_{21}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 12)

    # --- Passivity ---
    ax = axes[1, 1]
    power_qs = np.abs(s11_qs)**2 + np.abs(s21_qs)**2
    power_hyb = np.abs(s11_hyb)**2 + np.abs(s21_hyb)**2
    power_dg = np.abs(s11_dg)**2 + np.abs(s21_dg)**2
    ax.plot(f_qs, power_qs, 'b-', linewidth=1.5, label='QS probe', alpha=0.7)
    ax.plot(f_fw, power_hyb, 'g-s', linewidth=2, markersize=4, label='Hybrid probe')
    ax.plot(f_fw, power_dg, 'r--o', linewidth=1, markersize=3, alpha=0.6,
            label='Strip delta-gap')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|^2 + |S_{21}|^2$')
    ax.set_title('Passivity Check')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.1)
    ax.set_xlim(0, 12)

    fig.suptitle(
        rf'Microstrip Benchmark: $W={W_STRIP*1e3:.1f}$ mm, '
        rf'$L={L_STRIP*1e3:.0f}$ mm, $\varepsilon_r={EPS_R}$, '
        rf'$h={H_SUB*1e3:.1f}$ mm, $Z_0 = {49:.0f}\,\Omega$ (2D)',
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fname = IMG_DIR / 'hybrid_probe_benchmark.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {fname}")


def print_table(label, results, freqs):
    print(f"\n{label}:")
    print(f"  {'f(GHz)':>7} {'|S21| dB':>9} {'|S11| dB':>9} "
          f"{'∠S21 (°)':>9} {'|S11|²+|S21|²':>14}")
    print("  " + "-" * 55)
    for r, f in zip(results, freqs):
        S = r.S_matrix
        s21, s11 = S[1, 0], S[0, 0]
        power = abs(s11)**2 + abs(s21)**2
        print(f"  {f/1e9:>7.1f} "
              f"{20*np.log10(abs(s21)+1e-30):>9.2f} "
              f"{20*np.log10(abs(s11)+1e-30):>9.2f} "
              f"{np.degrees(np.angle(s21)):>9.1f} "
              f"{power:>14.4f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("HYBRID PROBE BENCHMARK — Full Frequency Range")
    print("=" * 70)

    stack = build_stack()
    tl = compute_reference_impedance(
        stack, strip_width=W_STRIP, source_layer_name='FR4', base_cells=300,
    )
    print(f"2D solver: Z0 = {tl.Z0:.2f} Ohm, eps_eff = {tl.eps_eff:.3f}")

    mesh, basis = build_mesh_and_basis(stack)
    print(f"Mesh: {len(mesh.triangles)} triangles, {basis.num_basis} basis functions")

    # Run all three solvers
    qs_results = run_qs_probe(stack, tl, mesh, basis)
    hybrid_results = run_hybrid_probe(stack, tl, mesh, basis)
    dg_results = run_delta_gap(stack, tl, mesh, basis)

    # Print tables
    print_table("Hybrid Probe (full-wave)", hybrid_results, FREQS_FW)

    # Plot
    plot_benchmark(qs_results, hybrid_results, dg_results, tl)
    print("\nDone!")
