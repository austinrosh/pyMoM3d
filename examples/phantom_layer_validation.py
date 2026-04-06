"""Phantom air layer validation for microstrip S-parameter extraction.

Compares 3-layer (no phantom) vs 4-layer (with phantom air) microstrip
stacks across four port/extraction methods.  The phantom air layer
ensures Strata's DCIM fitting point is close to the mesh z-coordinate,
eliminating the z-mismatch that degrades Green's function accuracy.

Physical setup
--------------
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - Strip: W=3.06mm (~50 Ohm), 20mm DUT
  - Ports at DUT edges (mesh extends 3mm beyond for mode coupling)
  - Four methods: DG+Dir, DG+Var, FW+Dir, FW+Var

Produces
--------
  images/phantom-layer/structure.png       -- Mesh with layer boundaries
  images/phantom-layer/s21_comparison.png  -- S21 vs frequency
  images/phantom-layer/s11_comparison.png  -- S11 vs frequency

Usage
-----
    venv/bin/python examples/phantom_layer_validation.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Port,
    Layer, LayerStack,
    c0,
)
from pyMoM3d.mom.excitation import find_feed_edges, compute_feed_signs
from pyMoM3d.cross_section.extraction import compute_reference_impedance
from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator
from pyMoM3d.greens.layered.sommerfeld import LayeredGreensFunction
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.visualization.mesh_plot import plot_structure_with_ports
from pyMoM3d.visualization.plot_style import configure_latex_style


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
IMG_DIR = Path(__file__).resolve().parent.parent / 'images' / 'phantom-layer'
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
Z_STRIP = H_SUB

L_DUT = 20e-3       # 20mm DUT between ports
STUB_LEN = 3.0e-3   # Mesh extension beyond each port

TEL = 0.35e-3
DELTA = H_SUB * 0.01  # 16um phantom layer

FREQS = np.array([3, 5, 7, 8, 9, 10, 12, 15]) * 1e9

METHOD_NAMES = ['DG+Dir', 'DG+Var', 'FW+Dir', 'FW+Var']
METHOD_KEYS = ['dg_direct', 'dg_var', 'fw_direct', 'fw_var']
METHOD_COLORS = ['C0', 'C1', 'C2', 'C3']


# ---------------------------------------------------------------------------
# Layer stacks
# ---------------------------------------------------------------------------

def build_stack_3layer():
    """Standard 3-layer microstrip (no phantom)."""
    return LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])


def build_stack_4layer():
    """4-layer microstrip with phantom air layer."""
    return LayerStack.make_microstrip_stack(H_SUB, EPS_R, delta=DELTA)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def y_to_s(Y, Z0_ref):
    """Convert Y-matrix to S-matrix."""
    P = Y.shape[0]
    I_eye = np.eye(P)
    Z_net = np.linalg.inv(Y)
    return (Z_net / Z0_ref - I_eye) @ np.linalg.inv(Z_net / Z0_ref + I_eye)


def extract_all_methods(basis, ports, freq, Z0_ref, mesh, stack, src_layer):
    """Extract S-params using all four methods at one frequency."""
    k = 2 * np.pi * freq / c0

    gf = LayeredGreensFunction(
        stack, frequency=freq, source_layer_name=src_layer,
    )
    op = MultilayerEFIEOperator(gf)
    Z_sys = fill_matrix(
        op, basis, mesh, k, gf.wave_impedance,
        quad_order=4, backend='auto',
    )

    P = len(ports)
    results = {}

    for port_mode in ['dg', 'fw']:
        if port_mode == 'fw':
            V_all = np.column_stack([
                p.build_excitation_vector(basis, mesh=mesh)
                for p in ports
            ])
        else:
            V_all = np.column_stack([
                p.build_excitation_vector(basis) for p in ports
            ])

        I_all = np.linalg.solve(Z_sys, V_all)

        for y_mode in ['direct', 'var']:
            if y_mode == 'var':
                ZI = Z_sys @ I_all
                Y = np.zeros((P, P), dtype=complex)
                for q in range(P):
                    V_q = ports[q].V_ref
                    for p in range(P):
                        V_p = ports[p].V_ref
                        t1 = 2.0 * (V_all[:, q] @ I_all[:, p])
                        t2 = I_all[:, q] @ ZI[:, p]
                        Y[q, p] = (t1 - t2) / (V_q * V_p)
            else:
                Y = np.zeros((P, P), dtype=complex)
                for q in range(P):
                    for p in range(P):
                        Y[q, p] = (
                            ports[q].terminal_current(I_all[:, p], basis)
                            / ports[p].V_ref
                        )

            S = y_to_s(Y, Z0_ref)
            results[f'{port_mode}_{y_mode}'] = S

    return results


def run_sweep(basis, ports, mesh, stack, src_layer, Z0_ref):
    """Run frequency sweep for one stack."""
    all_results = {k: [] for k in METHOD_KEYS}

    for freq in FREQS:
        res = extract_all_methods(
            basis, ports, freq, Z0_ref, mesh, stack, src_layer,
        )
        for key in METHOD_KEYS:
            all_results[key].append(res[key])

    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def dB(x):
    return 20 * np.log10(max(abs(x), 1e-30))


def plot_structure(mesh, basis, x_port1, x_port2, mesh_half_len):
    """Two-panel structure figure: xy mesh + xz cross-section."""
    fig, (ax_top, ax_side) = plt.subplots(
        2, 1, figsize=(10, 6),
        gridspec_kw={'height_ratios': [2, 1]},
    )

    # Top: xy mesh with ports
    plot_structure_with_ports(
        mesh,
        port_x_list=[x_port1, x_port2],
        port_labels=['P1', 'P2'],
        ax=ax_top, mm_units=True,
    )
    ax_top.set_title(
        rf'Microstrip mesh ($N_t = {mesh.get_num_triangles()}$, '
        rf'$N_b = {basis.num_basis}$)',
    )

    # Bottom: xz cross-section showing layer boundaries
    x_lo = -mesh_half_len * 1.1e3  # mm, with padding
    x_hi = +mesh_half_len * 1.1e3

    # PEC ground
    ax_side.fill_between(
        [x_lo, x_hi], -0.3, 0,
        color='0.5', alpha=0.6, hatch='///',
    )
    ax_side.text(0, -0.15, r'PEC ground', ha='center', va='center',
                 fontsize=8, color='white', fontweight='bold')

    # FR4 substrate
    ax_side.fill_between(
        [x_lo, x_hi], 0, H_SUB * 1e3,
        color='#90EE90', alpha=0.5,
    )
    ax_side.text(
        0, H_SUB * 1e3 / 2,
        rf'FR4 ($\varepsilon_r = {EPS_R}$)',
        ha='center', va='center', fontsize=9,
    )

    # Phantom air layer (exaggerated for visibility)
    phant_top_vis = H_SUB * 1e3 + 0.15
    ax_side.fill_between(
        [x_lo, x_hi], H_SUB * 1e3, phant_top_vis,
        color='#FFFFCC', alpha=0.8,
    )
    ax_side.text(
        x_hi * 0.95, (H_SUB * 1e3 + phant_top_vis) / 2,
        r'phantom ($\varepsilon_r = 1.001$)',
        ha='right', va='center', fontsize=7, style='italic',
    )

    # Air
    ax_side.fill_between(
        [x_lo, x_hi], phant_top_vis, phant_top_vis + 0.5,
        color='white', alpha=0.3,
    )
    ax_side.text(x_hi * 0.95, phant_top_vis + 0.25, 'air',
                 ha='right', va='center', fontsize=8, color='0.4')

    # Strip conductor (full mesh extent)
    ax_side.plot(
        [-mesh_half_len * 1e3, mesh_half_len * 1e3],
        [H_SUB * 1e3, H_SUB * 1e3],
        'r-', linewidth=3, label='strip conductor',
    )

    # Stub regions (shaded)
    for xlo_stub, xhi_stub in [
        (-mesh_half_len * 1e3, x_port1 * 1e3),
        (x_port2 * 1e3, mesh_half_len * 1e3),
    ]:
        ax_side.axvspan(xlo_stub, xhi_stub, alpha=0.15, color='blue',
                        zorder=0)
        mid = (xlo_stub + xhi_stub) / 2
        ax_side.text(
            mid, phant_top_vis + 0.35,
            'stub', ha='center', va='bottom', fontsize=7, color='blue',
        )

    # Port arrows
    for px, label in [(x_port1, 'P1'), (x_port2, 'P2')]:
        px_mm = px * 1e3
        ax_side.annotate(
            '', xy=(px_mm, H_SUB * 1e3),
            xytext=(px_mm, -0.05),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        )
        ax_side.text(px_mm, -0.20, label, ha='center', va='top',
                     fontsize=9, color='red', fontweight='bold')

    ax_side.set_xlim(x_lo, x_hi)
    ax_side.set_ylim(-0.35, phant_top_vis + 0.6)
    ax_side.set_xlabel(r'$x$ (mm)')
    ax_side.set_ylabel(r'$z$ (mm)')
    ax_side.set_title(r'Cross-section (layer stack)')
    ax_side.set_aspect('auto')

    fig.tight_layout()
    fig.savefig(IMG_DIR / 'structure.png', dpi=200, bbox_inches='tight')
    print(f"  Saved {IMG_DIR / 'structure.png'}")
    plt.close(fig)


def plot_s_comparison(results_3, results_4, idx, ylabel, filename, ylim=None):
    """Plot S-parameter comparison: 3-layer vs 4-layer, all methods."""
    f_ghz = FREQS / 1e9

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (key, name, color) in enumerate(
            zip(METHOD_KEYS, METHOD_NAMES, METHOD_COLORS)):
        vals_3 = [dB(S[idx]) for S in results_3[key]]
        vals_4 = [dB(S[idx]) for S in results_4[key]]

        ax.plot(f_ghz, vals_3, color=color, linestyle='--', marker='x',
                markersize=5, linewidth=1.2,
                label=rf'{name} (3-layer)')
        ax.plot(f_ghz, vals_4, color=color, linestyle='-', marker='o',
                markersize=4, linewidth=1.8,
                label=rf'{name} (4-layer)')

    ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(ylabel)
    ax.set_title(
        r'Microstrip through-line: 3-layer vs 4-layer (phantom air)'
    )
    ax.legend(fontsize=8, ncol=2, loc='lower left')
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(IMG_DIR / filename, dpi=200, bbox_inches='tight')
    print(f"  Saved {IMG_DIR / filename}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configure_latex_style()

    print("=" * 72)
    print("Phantom Air Layer Validation — Microstrip S-Parameter Extraction")
    print("=" * 72)
    print()

    # --- 2D cross-section ---
    stack_3 = build_stack_3layer()
    tl_result = compute_reference_impedance(
        stack_3, W_STRIP, source_layer_name='FR4',
    )
    Z0_ref = tl_result.Z0
    print(f"2D cross-section: Z0 = {Z0_ref:.2f} Ohm, "
          f"eps_eff = {tl_result.eps_eff:.3f}")
    print()

    # --- Mesh — ports at DUT edges ---
    # Mesh extends STUB_LEN beyond each port so feed edges are interior
    # (two adjacent triangles → valid RWG basis functions).
    mesher = GmshMesher(target_edge_length=TEL)
    x_port1 = -L_DUT / 2
    x_port2 = +L_DUT / 2

    mesh_total_len = L_DUT + 2 * STUB_LEN
    mesh = mesher.mesh_plate_with_feeds(
        width=mesh_total_len, height=W_STRIP,
        feed_x_list=[x_port1, x_port2],
        center=(0, 0, Z_STRIP),
    )
    basis = compute_rwg_connectivity(mesh)

    # --- Ports ---
    feed1 = find_feed_edges(mesh, basis, feed_x=x_port1)
    feed2 = find_feed_edges(mesh, basis, feed_x=x_port2)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)

    ports = [
        Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1,
             gap_width=TEL),
        Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2,
             gap_width=TEL),
    ]

    print(f"DUT: {L_DUT*1e3:.0f}mm between ports")
    print(f"Total mesh: {mesh_total_len*1e3:.1f}mm "
          f"({STUB_LEN*1e3:.1f}mm stubs), "
          f"{mesh.get_num_triangles()} tris, {basis.num_basis} basis")
    print(f"  Port 1: {len(feed1)} edges at x={x_port1*1e3:.1f}mm "
          f"(DUT edge)")
    print(f"  Port 2: {len(feed2)} edges at x={x_port2*1e3:.1f}mm "
          f"(DUT edge)")
    print(f"  Gap width = {TEL*1e3:.2f}mm, "
          f"phantom delta = {DELTA*1e6:.0f}um")
    print()

    # --- Structure plot ---
    print("Generating plots...")
    plot_structure(mesh, basis, x_port1, x_port2, mesh_total_len / 2)

    # --- 3-layer sweep ---
    print("\nRunning 3-layer (no phantom) sweep...")
    results_3 = run_sweep(
        basis, ports, mesh, stack_3, 'FR4', Z0_ref,
    )

    # --- 4-layer sweep ---
    stack_4 = build_stack_4layer()
    print("Running 4-layer (phantom air) sweep...")
    results_4 = run_sweep(
        basis, ports, mesh, stack_4, 'phantom_air', Z0_ref,
    )

    # --- Comparison table ---
    print()
    print("S21 (dB) — edge ports on 20mm through-line (target: 0 dB)")
    print("-" * 100)
    header = (f"{'Freq':>6} |"
              + "".join(f" {'3L '+n:>11} {'4L '+n:>11} |"
                        for n in METHOD_NAMES))
    print(header)
    print("-" * len(header))

    for fi, freq in enumerate(FREQS):
        parts = [f"{freq/1e9:5.0f}G |"]
        for key in METHOD_KEYS:
            s21_3 = dB(results_3[key][fi][0, 1])
            s21_4 = dB(results_4[key][fi][0, 1])
            parts.append(f" {s21_3:11.2f} {s21_4:11.2f} |")
        print("".join(parts))

    # --- S-parameter plots ---
    plot_s_comparison(
        results_3, results_4, (0, 1),
        r'$|S_{21}|$ (dB)', 's21_comparison.png',
        ylim=(-15, 1),
    )
    plot_s_comparison(
        results_3, results_4, (0, 0),
        r'$|S_{11}|$ (dB)', 's11_comparison.png',
        ylim=(-30, 0),
    )

    # --- Passivity and reciprocity ---
    print()
    print("Physics checks (4-layer, FW+Var):")
    print(f"  {'Freq':>6} | {'Passivity':>10} | {'Reciprocity':>12} | {'Pass':>6}")
    print(f"  {'-'*50}")
    all_pass = True
    for fi, freq in enumerate(FREQS):
        S = results_4['fw_var'][fi]
        passivity = abs(S[0, 0])**2 + abs(S[1, 0])**2
        recip = abs(S[0, 1] - S[1, 0])
        ok = passivity <= 1.01 and recip < 1e-10
        if not ok:
            all_pass = False
        print(f"  {freq/1e9:5.0f}G | {passivity:10.6f} | {recip:12.2e} | "
              f"{'OK' if ok else 'FAIL':>6}")

    # --- Impact summary ---
    print()
    print("=" * 72)
    print("Impact Summary (4-layer phantom improvement over 3-layer):")
    print("=" * 72)
    for fi, freq in enumerate(FREQS):
        best_3 = max(dB(results_3[k][fi][0, 1]) for k in METHOD_KEYS)
        best_4 = max(dB(results_4[k][fi][0, 1]) for k in METHOD_KEYS)
        delta = best_4 - best_3
        print(f"  {freq/1e9:5.0f} GHz: 3-layer {best_3:+6.1f} dB -> "
              f"4-layer {best_4:+6.1f} dB  ({delta:+5.1f} dB)")

    # Overall summary
    mid_freqs = [fi for fi, f in enumerate(FREQS) if 5e9 <= f <= 12e9]
    avg_3 = np.mean([max(dB(results_3[k][fi][0, 1]) for k in METHOD_KEYS)
                      for fi in mid_freqs])
    avg_4 = np.mean([max(dB(results_4[k][fi][0, 1]) for k in METHOD_KEYS)
                      for fi in mid_freqs])
    print()
    print(f"  5-12 GHz average: 3-layer {avg_3:+.1f} dB, "
          f"4-layer {avg_4:+.1f} dB")
    print(f"  Phantom layer improvement: {avg_4 - avg_3:+.1f} dB")
    print()
    if all_pass:
        print("  All passivity/reciprocity checks PASSED.")
    else:
        print("  WARNING: Some passivity/reciprocity checks FAILED.")


if __name__ == '__main__':
    main()
