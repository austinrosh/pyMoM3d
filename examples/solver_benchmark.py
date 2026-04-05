"""Solver benchmark: QS vs full-wave on the same microstrip through-line.

Compares three extraction methods across 0.1–12 GHz:
1. QS solver with probe feeds (valid for kD << 1)
2. Full-wave MPIE with strip delta-gap (NetworkExtractor)
3. Full-wave MPIE with SOC de-embedding

Also plots the 2D reference (lossless TL from cross-section solver).

Saves results and plots to images/benchmark/.
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
)
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

# Port positions — 1mm inboard from strip ends
PORT1_X = -L_STRIP / 2.0 + 1.0e-3
PORT2_X = +L_STRIP / 2.0 - 1.0e-3

# Frequency sweep
FREQS_QS = np.linspace(0.1e9, 12.0e9, 60)
FREQS_FW = np.linspace(1.0e9, 12.0e9, 23)  # full-wave is expensive per point

# ============================================================================
# Setup (shared geometry)
# ============================================================================

def setup():
    """Build mesh, basis, ports, layer stack — shared by all solvers."""
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

    # Mesh
    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, PORT2_X],
        center=(0.0, 0.0, H_SUB),
    )
    basis = compute_rwg_connectivity(mesh)
    print(f"Mesh: {len(mesh.triangles)} triangles, {basis.num_basis} basis functions")

    # Ports
    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    return stack, tl, mesh, basis, port1, port2, feed1


# ============================================================================
# Solver runs
# ============================================================================

def run_qs_probe(stack, tl, mesh, basis, port1, port2, feed1):
    """QS solver with probe feeds."""
    print("\n--- QS Solver (probe feeds) ---")
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
    print(f"  {len(FREQS_QS)} frequencies in {dt:.1f}s "
          f"({dt/len(FREQS_QS)*1e3:.1f} ms/freq)")
    return results


def run_fullwave(stack, mesh, basis, port1, port2, feed1, Z0=50.0):
    """Full-wave MPIE with strip delta-gap ports."""
    print("\n--- Full-wave MPIE (strip delta-gap) ---")
    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=1e9, excitation=exc, quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    extractor = NetworkExtractor(sim, [port1, port2], Z0=Z0)

    t0 = time.time()
    results = extractor.extract(FREQS_FW.tolist())
    dt = time.time() - t0
    print(f"  {len(FREQS_FW)} frequencies in {dt:.1f}s "
          f"({dt/len(FREQS_FW)*1e3:.0f} ms/freq)")
    return results


# ============================================================================
# Plotting
# ============================================================================

def extract_s(results, i, j):
    return np.array([r.S_matrix[i, j] for r in results])


def plot_benchmark(qs_results, fw_results, tl, fw_z0):
    """Main benchmark plot: S21 and S11 across full frequency range."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    f_qs_ghz = FREQS_QS / 1e9
    f_fw_ghz = FREQS_FW / 1e9

    s21_qs = extract_s(qs_results, 1, 0)
    s11_qs = extract_s(qs_results, 0, 0)
    s21_fw = extract_s(fw_results, 1, 0)
    s11_fw = extract_s(fw_results, 0, 0)

    # Compute kD for reference
    eps_eff = tl.eps_eff
    kD_qs = 2 * np.pi * FREQS_QS * np.sqrt(eps_eff) / c0 * L_STRIP
    kD_fw = 2 * np.pi * FREQS_FW * np.sqrt(eps_eff) / c0 * L_STRIP

    # 2D TL reference (lossless through-line)
    L_eff = abs(PORT2_X - PORT1_X)  # distance between ports
    beta_2d = np.array([tl.beta(f) for f in FREQS_QS])
    s21_2d_phase = -beta_2d * L_eff  # expected phase

    # --- S21 magnitude ---
    ax = axes[0, 0]
    ax.plot(f_qs_ghz, 20*np.log10(np.abs(s21_qs) + 1e-30),
            'b-', linewidth=2, label='QS probe')
    ax.plot(f_fw_ghz, 20*np.log10(np.abs(s21_fw) + 1e-30),
            'r-o', linewidth=1.5, markersize=4, label=f'Full-wave ($Z_0={fw_z0:.0f}\\,\\Omega$)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(c0 / (4 * L_STRIP * np.sqrt(eps_eff)) / 1e9,
               color='green', linestyle='--', alpha=0.5, label=r'$\lambda/4$ resonance')
    # Mark kD = 0.5
    f_kd05 = 0.5 * c0 / (2 * np.pi * np.sqrt(eps_eff) * L_STRIP) / 1e9
    ax.axvline(f_kd05, color='orange', linestyle=':', alpha=0.7,
               label=f'$kD = 0.5$ ({f_kd05:.1f} GHz)')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{21}|$ (dB)')
    ax.set_title(r'Insertion Loss $|S_{21}|$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 12)

    # --- S11 magnitude ---
    ax = axes[0, 1]
    ax.plot(f_qs_ghz, 20*np.log10(np.abs(s11_qs) + 1e-30),
            'b-', linewidth=2, label='QS probe')
    ax.plot(f_fw_ghz, 20*np.log10(np.abs(s11_fw) + 1e-30),
            'r-o', linewidth=1.5, markersize=4, label=f'Full-wave ($Z_0={fw_z0:.0f}\\,\\Omega$)')
    ax.axvline(f_kd05, color='orange', linestyle=':', alpha=0.7,
               label=f'$kD = 0.5$')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss $|S_{11}|$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 12)

    # --- S21 phase ---
    ax = axes[1, 0]
    ax.plot(f_qs_ghz, np.degrees(np.angle(s21_qs)),
            'b-', linewidth=2, label='QS probe')
    ax.plot(f_fw_ghz, np.degrees(np.angle(s21_fw)),
            'r-o', linewidth=1.5, markersize=4, label='Full-wave')
    ax.plot(f_qs_ghz, np.degrees(np.unwrap(s21_2d_phase)),
            'g--', linewidth=1.5, label=r'2D: $-\beta L_{\mathrm{eff}}$')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$\angle S_{21}$ (deg)')
    ax.set_title(r'Phase $\angle S_{21}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 12)

    # --- Passivity ---
    ax = axes[1, 1]
    power_qs = np.abs(s11_qs)**2 + np.abs(s21_qs)**2
    power_fw = np.abs(s11_fw)**2 + np.abs(s21_fw)**2
    ax.plot(f_qs_ghz, power_qs, 'b-', linewidth=2, label='QS probe')
    ax.plot(f_fw_ghz, power_fw, 'r-o', linewidth=1.5, markersize=4,
            label='Full-wave')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|^2 + |S_{21}|^2$')
    ax.set_title('Passivity Check')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(0, 12)

    fig.suptitle(
        rf'Microstrip Through-Line Benchmark: $W={W_STRIP*1e3:.1f}$ mm, '
        rf'$L={L_STRIP*1e3:.0f}$ mm, $\varepsilon_r={EPS_R}$, '
        rf'$h={H_SUB*1e3:.1f}$ mm',
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'microstrip_solver_benchmark.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {IMG_DIR / 'microstrip_solver_benchmark.png'}")


def print_table(label, results, freqs, Z0_ref):
    """Print S-parameter table."""
    print(f"\n{label}:")
    print(f"  {'f(GHz)':>7} {'kD':>6} {'|S21| dB':>9} {'|S11| dB':>9} "
          f"{'∠S21 (°)':>9} {'|S11|²+|S21|²':>14}")
    print("  " + "-" * 60)
    eps_eff = (c0 / (Z0_ref * 2))**2 if Z0_ref == 50 else 3.26  # rough
    for r, f in zip(results, freqs):
        S = r.S_matrix
        s21 = S[1, 0]
        s11 = S[0, 0]
        kD = 2 * np.pi * f * np.sqrt(3.26) / c0 * L_STRIP
        power = abs(s11)**2 + abs(s21)**2
        print(f"  {f/1e9:>7.1f} {kD:>6.2f} "
              f"{20*np.log10(abs(s21)+1e-30):>9.2f} "
              f"{20*np.log10(abs(s11)+1e-30):>9.2f} "
              f"{np.degrees(np.angle(s21)):>9.1f} "
              f"{power:>14.4f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MICROSTRIP THROUGH-LINE SOLVER BENCHMARK")
    print("=" * 70)
    print(f"Strip: W={W_STRIP*1e3:.2f} mm, L={L_STRIP*1e3:.0f} mm")
    print(f"Substrate: eps_r={EPS_R}, h={H_SUB*1e3:.1f} mm")
    print(f"Ports at x = {PORT1_X*1e3:.1f} mm, {PORT2_X*1e3:.1f} mm")

    stack, tl, mesh, basis, port1, port2, feed1 = setup()

    # Run QS solver
    qs_results = run_qs_probe(stack, tl, mesh, basis, port1, port2, feed1)

    # Spot-check QS results
    for f_target in [0.5e9, 1e9, 2e9, 5e9, 8e9, 10e9]:
        idx = np.argmin(np.abs(FREQS_QS - f_target))
        s21 = qs_results[idx].S_matrix[1, 0]
        print(f"  QS f={FREQS_QS[idx]/1e9:.1f} GHz: "
              f"|S21|={20*np.log10(abs(s21)+1e-30):.2f} dB")

    # Run full-wave with 2D-derived Z0
    fw_results = run_fullwave(stack, mesh, basis, port1, port2, feed1, Z0=tl.Z0)
    print_table("Full-wave (Z0=2D)", fw_results, FREQS_FW, tl.Z0)

    # Plot
    plot_benchmark(qs_results, fw_results, tl, fw_z0=tl.Z0)

    print("\nDone!")
