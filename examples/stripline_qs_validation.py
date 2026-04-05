"""Stripline & microstrip QS validation: 2D Z₀ → 3D QS → S-parameters.

Validates the full pipeline for through-lines with probe-fed ports:
1. 2D FDM solver: Z₀, ε_eff (cross-section analysis)
2. 3D QS MoM with vertical probe feeds (signal-to-ground)
3. S-parameter extraction with 2D-derived reference impedance

Saves structure visualization and S-parameter plots to images/.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, Layer, LayerStack,
    configure_latex_style,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.cross_section import compute_reference_impedance

IMG_DIR = Path(__file__).resolve().parent.parent / 'images'


def setup_style():
    try:
        configure_latex_style()
    except Exception:
        configure_latex_style(use_tex=False)


# ============================================================================
# Microstrip through-line
# ============================================================================

def run_microstrip_pipeline():
    """Microstrip through-line with probe feeds."""
    print("=" * 60)
    print("MICROSTRIP THROUGH-LINE (Probe feeds)")
    print("=" * 60)

    EPS_R = 4.4
    H_SUB = 1.6e-3
    W_STRIP = 3.06e-3
    L_STRIP = 10.0e-3
    TEL = 0.7e-3
    PORT1_X = -L_STRIP / 2.0 + 1.0e-3
    PORT2_X = +L_STRIP / 2.0 - 1.0e-3

    stack = LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])

    tl = compute_reference_impedance(
        stack, strip_width=W_STRIP, source_layer_name='FR4', base_cells=300,
    )
    print(f"2D solver: Z0 = {tl.Z0:.2f} Ohm, eps_eff = {tl.eps_eff:.3f}")

    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, PORT2_X],
        center=(0.0, 0.0, H_SUB),
    )
    basis = compute_rwg_connectivity(mesh)
    print(f"Mesh: {len(mesh.triangles)} triangles, {basis.num_basis} basis functions")

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

    freqs = np.linspace(0.1e9, 3.0e9, 30)
    results = qs.extract(freqs.tolist())

    return {
        'name': 'Microstrip',
        'tl': tl, 'mesh': mesh, 'basis': basis, 'qs': qs,
        'ports': [port1, port2], 'results': results, 'freqs': freqs,
        'stack': stack, 'params': {
            'eps_r': EPS_R, 'h': H_SUB, 'W': W_STRIP, 'L': L_STRIP,
            'port_x': [PORT1_X, PORT2_X],
        },
    }


# ============================================================================
# Stripline through-line
# ============================================================================

def run_stripline_pipeline():
    """Stripline through-line with probe feeds."""
    print("\n" + "=" * 60)
    print("STRIPLINE THROUGH-LINE (Probe feeds)")
    print("=" * 60)

    b = 3.0e-3
    EPS_R = 2.2
    W_STRIP = 1.5e-3
    L_STRIP = 10e-3
    h_strip = b / 2
    TEL = 0.7e-3
    PORT1_X = -L_STRIP / 2.0 + 1.0e-3
    PORT2_X = +L_STRIP / 2.0 - 1.0e-3

    stack = LayerStack([
        Layer('pec_bot', z_bot=-np.inf, z_top=0, eps_r=1.0, is_pec=True),
        Layer('diel', z_bot=0, z_top=b, eps_r=EPS_R),
        Layer('pec_top', z_bot=b, z_top=np.inf, eps_r=1.0, is_pec=True),
    ])

    tl = compute_reference_impedance(
        stack, strip_width=W_STRIP, strip_z=h_strip, base_cells=200,
    )
    print(f"2D solver: Z0 = {tl.Z0:.2f} Ohm, eps_eff = {tl.eps_eff:.3f}")

    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, PORT2_X],
        center=(0.0, 0.0, h_strip),
    )
    basis = compute_rwg_connectivity(mesh)
    print(f"Mesh: {len(mesh.triangles)} triangles, {basis.num_basis} basis functions")

    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=1e9, excitation=exc, quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='diel',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    qs = QuasiStaticSolver(sim, [port1, port2], Z0=tl.Z0, probe_feeds=True)

    freqs = np.linspace(0.1e9, 3.0e9, 30)
    results = qs.extract(freqs.tolist())

    return {
        'name': 'Stripline',
        'tl': tl, 'mesh': mesh, 'basis': basis, 'qs': qs,
        'ports': [port1, port2], 'results': results, 'freqs': freqs,
        'stack': stack, 'params': {
            'eps_r': EPS_R, 'b': b, 'h': h_strip,
            'W': W_STRIP, 'L': L_STRIP,
            'port_x': [PORT1_X, PORT2_X],
        },
    }


# ============================================================================
# Plotting — structure with dielectric and probes
# ============================================================================

def plot_structure(data, save_dir):
    """3D structure visualization with dielectric layers and probe feeds."""
    mesh = data['mesh']
    name = data['name']
    params = data['params']
    qs = data['qs']
    stack = data['stack']

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    verts = mesh.vertices
    tris = mesh.triangles

    # --- Dielectric layers (translucent boxes) ---
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15

    for layer in stack.layers:
        if layer.is_pec:
            continue
        z_bot = layer.z_bot if np.isfinite(layer.z_bot) else 0
        z_top = layer.z_top if np.isfinite(layer.z_top) else params.get('h', params.get('b', 2e-3))
        eps_r = float(np.real(layer.eps_r))
        if eps_r < 1.01:
            continue  # skip air

        # Draw dielectric as translucent box
        bx = [x_min - x_pad, x_max + x_pad]
        by = [y_min - y_pad, y_max + y_pad]
        bz = [z_bot, z_top]
        faces = [
            # bottom
            [[bx[0], by[0], bz[0]], [bx[1], by[0], bz[0]],
             [bx[1], by[1], bz[0]], [bx[0], by[1], bz[0]]],
            # top
            [[bx[0], by[0], bz[1]], [bx[1], by[0], bz[1]],
             [bx[1], by[1], bz[1]], [bx[0], by[1], bz[1]]],
            # front
            [[bx[0], by[0], bz[0]], [bx[1], by[0], bz[0]],
             [bx[1], by[0], bz[1]], [bx[0], by[0], bz[1]]],
            # back
            [[bx[0], by[1], bz[0]], [bx[1], by[1], bz[0]],
             [bx[1], by[1], bz[1]], [bx[0], by[1], bz[1]]],
            # left
            [[bx[0], by[0], bz[0]], [bx[0], by[1], bz[0]],
             [bx[0], by[1], bz[1]], [bx[0], by[0], bz[1]]],
            # right
            [[bx[1], by[0], bz[0]], [bx[1], by[1], bz[0]],
             [bx[1], by[1], bz[1]], [bx[1], by[0], bz[1]]],
        ]
        poly = Poly3DCollection(faces, alpha=0.12, facecolor='green',
                                edgecolor='green', linewidth=0.3)
        ax.add_collection3d(poly)

    # --- PEC ground planes ---
    for layer in stack.layers:
        if not layer.is_pec:
            continue
        if np.isfinite(layer.z_top) and layer.z_top <= verts[:, 2].max() + 1e-6:
            z_gnd = layer.z_top
        elif np.isfinite(layer.z_bot) and layer.z_bot >= verts[:, 2].min() - 1e-6:
            z_gnd = layer.z_bot
        else:
            continue

        gnd_verts = [
            [x_min - x_pad, y_min - y_pad, z_gnd],
            [x_max + x_pad, y_min - y_pad, z_gnd],
            [x_max + x_pad, y_max + y_pad, z_gnd],
            [x_min - x_pad, y_max + y_pad, z_gnd],
        ]
        poly = Poly3DCollection([gnd_verts], alpha=0.25, facecolor='#C0A060',
                                edgecolor='#806030', linewidth=0.5)
        ax.add_collection3d(poly)

    # --- Strip mesh ---
    tri_verts = verts[tris]
    poly = Poly3DCollection(tri_verts, alpha=0.6, facecolor='gold',
                            edgecolor='k', linewidth=0.3)
    ax.add_collection3d(poly)

    # --- Probe feeds (vertical lines from ground to strip) ---
    z_strip = qs._z_strip
    z_ground = z_strip - qs._h_via
    probe_vertices = qs._probe_vertices
    port_names = [p.name for p in data['ports']]

    for i, vi in enumerate(probe_vertices):
        px, py, pz = verts[vi]
        ax.plot([px, px], [py, py], [z_ground, z_strip],
                color='red', linewidth=3, solid_capstyle='round')
        ax.scatter([px], [py], [z_ground], color='red', s=30, zorder=10)
        ax.scatter([px], [py], [z_strip], color='red', s=30, zorder=10)
        ax.text(px, py, z_strip * 1.15 + 0.0002,
                port_names[i], color='red', fontsize=11,
                fontweight='bold', ha='center')

    # --- Labels and limits ---
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    ax.set_zlabel(r'$z$ (m)')

    eps_str = f"$\\varepsilon_r = {params['eps_r']:.1f}$"
    h_str = f"$h = {params['h']*1e3:.1f}$ mm" if 'h' in params else ''
    title_parts = [f'Probe-Fed {name}']
    if eps_str:
        title_parts.append(eps_str)
    if h_str:
        title_parts.append(h_str)
    ax.set_title(', '.join(title_parts))

    x_range = x_max - x_min + 2 * x_pad
    y_range = y_max - y_min + 2 * y_pad
    z_lo = min(0, z_ground) - 0.0003
    z_hi = max(z_strip, params.get('b', z_strip)) + 0.0005
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    half = max(x_range, y_range) / 2
    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    ax.set_zlim(z_lo, z_hi)

    fig.tight_layout()
    fname = save_dir / f'{name.lower()}_structure.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_s_parameters(data, save_dir):
    """S-parameter magnitude and phase plots."""
    results = data['results']
    freqs = data['freqs']
    tl = data['tl']
    name = data['name']

    f_ghz = freqs / 1e9
    s11_db = [20 * np.log10(abs(r.S_matrix[0, 0]) + 1e-30) for r in results]
    s21_db = [20 * np.log10(abs(r.S_matrix[1, 0]) + 1e-30) for r in results]
    s21_phase = [np.degrees(np.angle(r.S_matrix[1, 0])) for r in results]
    power = [abs(r.S_matrix[0, 0])**2 + abs(r.S_matrix[1, 0])**2 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(f_ghz, s21_db, 'b-', linewidth=2, label=r'$|S_{21}|$')
    ax.plot(f_ghz, s11_db, 'r--', linewidth=1.5, label=r'$|S_{11}|$')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'Magnitude (dB)')
    ax.set_title(f'{name}: S-parameter Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-40, 2)

    ax = axes[0, 1]
    ax.plot(f_ghz, s21_phase, 'b-', linewidth=2, label=r'$\angle S_{21}$')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'Phase (deg)')
    ax.set_title(f'{name}: S21 Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(f_ghz, power, 'k-', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|^2 + |S_{21}|^2$')
    ax.set_title(f'{name}: Passivity Check')
    ax.set_ylim(0.99, 1.01)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    L_eff = data['params']['L'] * 0.8
    beta_2d = [tl.beta(f) for f in freqs]
    expected_phase = [-np.degrees(b * L_eff) for b in beta_2d]
    ax.plot(f_ghz, s21_phase, 'b-', linewidth=2, label=r'QS $\angle S_{21}$')
    ax.plot(f_ghz, expected_phase, 'g--', linewidth=1.5,
            label=rf'$-\beta L_\mathrm{{eff}}$ (2D)')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'Phase (deg)')
    ax.set_title(f'{name}: Phase vs 2D Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf'{name}: $Z_0 = {tl.Z0:.1f}\,\Omega$, '
        rf'$\varepsilon_{{\mathrm{{eff}}}} = {tl.eps_eff:.2f}$',
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fname = save_dir / f'{name.lower()}_s_parameters.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_comparison(ms_data, sl_data, save_dir):
    """Side-by-side comparison of microstrip vs stripline."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for data, ax in [(ms_data, axes[0]), (sl_data, axes[1])]:
        results = data['results']
        freqs = data['freqs']
        f_ghz = freqs / 1e9
        s21_db = [20 * np.log10(abs(r.S_matrix[1, 0]) + 1e-30) for r in results]
        s11_db = [20 * np.log10(abs(r.S_matrix[0, 0]) + 1e-30) for r in results]

        ax.plot(f_ghz, s21_db, 'b-', linewidth=2, label=r'$|S_{21}|$')
        ax.plot(f_ghz, s11_db, 'r--', linewidth=1.5, label=r'$|S_{11}|$')
        ax.set_xlabel(r'Frequency $f$ (GHz)')
        ax.set_ylabel(r'Magnitude (dB)')
        ax.set_title(
            rf"{data['name']}: $Z_0 = {data['tl'].Z0:.1f}\,\Omega$"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-40, 2)

    fig.suptitle(
        'Microstrip vs Stripline: Through-Line S-Parameters (QS + Probe Feeds)',
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fname = save_dir / 'comparison_s_parameters.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    setup_style()

    ms_dir = IMG_DIR / 'transmission-line'
    sl_dir = IMG_DIR / 'stripline'
    ms_dir.mkdir(parents=True, exist_ok=True)
    sl_dir.mkdir(parents=True, exist_ok=True)

    ms_data = run_microstrip_pipeline()
    sl_data = run_stripline_pipeline()

    for data in [ms_data, sl_data]:
        print(f"\n{data['name']} summary:")
        for r in data['results']:
            s21_db = 20 * np.log10(abs(r.S_matrix[1, 0]) + 1e-30)
            if abs(r.frequency - 0.5e9) < 1e6 or abs(r.frequency - 1.0e9) < 1e6 or abs(r.frequency - 2.0e9) < 1e6:
                print(f"  f={r.frequency/1e9:.1f} GHz: S21={s21_db:.2f} dB")

    print("\nGenerating plots...")
    plot_structure(ms_data, ms_dir)
    plot_s_parameters(ms_data, ms_dir)
    plot_structure(sl_data, sl_dir)
    plot_s_parameters(sl_data, sl_dir)
    plot_comparison(ms_data, sl_data, sl_dir)
    print("\nDone!")
