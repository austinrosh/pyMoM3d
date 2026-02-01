"""
Example: Thin wire dipole input impedance vs frequency.

Approximates a thin-wire dipole as a narrow rectangular plate
with a delta-gap feed at the center edge. Sweeps frequency to
find the resonant input impedance.

Produces:
  Figure 1: Input impedance (R_in, X_in, S11) vs frequency
  Figure 2: Current distribution along the dipole at resonance
  Figure 3: Surface current density heatmap at resonance
  Figure 4: Radiation pattern with directivity at resonance

Expected result for a half-wave dipole:
  Z_in ~ 73 + j42 ohms at resonance

Note: a flat plate is a crude approximation to a cylindrical wire.
The impedance will differ from the canonical 73+j42 value, but
the resonance behavior (real part peak, reactance zero-crossing)
should be visible.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from pyMoM3d import (
    RectangularPlate,
    GmshMesher,
    compute_rwg_connectivity,
    Simulation,
    SimulationConfig,
    plot_surface_current,
    compute_far_field,
    c0,
    eta0,
)
from pyMoM3d.mom.excitation import DeltaGapExcitation, find_nearest_edge
from pyMoM3d.analysis.pattern_analysis import compute_directivity


def main():
    print("=" * 60)
    print("Dipole Impedance Sweep")
    print("=" * 60)

    # --- Parameters ---
    dipole_length = 0.15   # m (half-wave at ~1 GHz)
    dipole_width = 0.01    # m (strip width)
    target_edge_length = 0.005  # fine mesh for accurate feed

    print(f"\nDipole length: {dipole_length} m")
    print(f"Dipole width:  {dipole_width} m")
    print(f"Expected resonance: ~{c0 / (2 * dipole_length) / 1e9:.2f} GHz")

    # --- Mesh ---
    print("\n--- Meshing ---")
    plate = RectangularPlate(dipole_length, dipole_width, center=(0, 0, 0))
    mesher = GmshMesher(target_edge_length=target_edge_length)
    mesh = mesher.mesh_from_geometry(plate)
    basis = compute_rwg_connectivity(mesh)

    stats = mesh.get_statistics()
    print(f"Vertices:     {stats['num_vertices']}")
    print(f"Triangles:    {stats['num_triangles']}")
    print(f"RWG basis:    {basis.num_basis}")

    # --- Find feed edge ---
    feed_point = np.array([0.0, 0.0, 0.0])
    feed_idx = find_nearest_edge(mesh, basis, feed_point)
    feed_edge = mesh.edges[basis.edge_index[feed_idx]]
    feed_midpoint = 0.5 * (mesh.vertices[feed_edge[0]] + mesh.vertices[feed_edge[1]])
    print(f"Feed basis index: {feed_idx}")
    print(f"Feed edge midpoint: ({feed_midpoint[0]:.4f}, {feed_midpoint[1]:.4f}, {feed_midpoint[2]:.4f})")

    # --- Frequency sweep ---
    freq_start = 0.5e9
    freq_stop = 1.5e9
    n_freqs = 21
    frequencies = np.linspace(freq_start, freq_stop, n_freqs)

    print(f"\n--- Frequency sweep: {freq_start/1e9:.1f} - {freq_stop/1e9:.1f} GHz, {n_freqs} points ---")

    exc = DeltaGapExcitation(basis_index=feed_idx, voltage=1.0)
    config = SimulationConfig(frequency=frequencies[0], excitation=exc, quad_order=4)
    sim = Simulation(config, mesh=mesh)

    results = sim.sweep(frequencies.tolist())

    # --- Extract impedance ---
    Z_in = np.array([r.Z_input if r.Z_input is not None else np.nan + 0j
                      for r in results])
    R_in = np.real(Z_in)
    X_in = np.imag(Z_in)

    print("\nFrequency (GHz)   R_in (Ohm)      X_in (Ohm)")
    print("-" * 50)
    for i, f in enumerate(frequencies):
        print(f"  {f/1e9:8.3f}       {R_in[i]:12.4f}   {X_in[i]:12.4f}")

    # Find resonance
    valid = np.isfinite(X_in) & np.isfinite(R_in)
    idx_res = 0
    if np.any(valid):
        valid_X = X_in[valid]
        sign_changes = np.where(np.diff(np.sign(valid_X.real)))[0]
        if len(sign_changes) > 0:
            idx_res = np.where(valid)[0][sign_changes[0]]
            if abs(X_in[idx_res]) > abs(X_in[idx_res + 1]):
                idx_res = idx_res + 1
        else:
            idx_res = np.where(valid)[0][np.argmax(R_in[valid])]
    f_res = frequencies[idx_res]
    Z_res = Z_in[idx_res]
    print(f"\nNearest to resonance: f = {f_res/1e9:.3f} GHz")
    print(f"Z_in at resonance:   {Z_res.real:.4f} + j{Z_res.imag:.4f} Ohm")

    I_res = results[idx_res].I_coefficients

    # --- S11 ---
    Z0 = 50.0
    S11 = (Z_in - Z0) / (Z_in + Z0)
    S11_dB = 20.0 * np.log10(np.maximum(np.abs(S11), 1e-30))

    # --- Basis function position along dipole ---
    edge_x_pos = np.zeros(basis.num_basis)
    edge_is_transverse = np.zeros(basis.num_basis, dtype=bool)
    for n in range(basis.num_basis):
        edge_verts = mesh.edges[basis.edge_index[n]]
        v_a = mesh.vertices[edge_verts[0]]
        v_b = mesh.vertices[edge_verts[1]]
        edge_x_pos[n] = 0.5 * (v_a[0] + v_b[0])
        edge_dir = v_b - v_a
        edge_dir /= np.linalg.norm(edge_dir) + 1e-30
        edge_is_transverse[n] = abs(edge_dir[1]) > abs(edge_dir[0])

    trans_idx = np.where(edge_is_transverse)[0]
    if len(trans_idx) == 0:
        trans_idx = np.arange(basis.num_basis)
    sort_idx = trans_idx[np.argsort(edge_x_pos[trans_idx])]

    # --- Far-field radiation pattern at resonance ---
    print("\n--- Radiation pattern at resonance ---")
    k_res = 2.0 * np.pi * f_res / c0

    theta_half = np.linspace(0.001, np.pi - 0.001, 181)

    # E-plane (xz-plane, phi=0)
    E_th_e, E_ph_e = compute_far_field(I_res, basis, mesh, k_res, eta0,
                                        theta_half, np.zeros_like(theta_half))
    gain_e = np.abs(E_th_e)**2 + np.abs(E_ph_e)**2

    # H-plane (yz-plane, phi=pi/2)
    E_th_h, E_ph_h = compute_far_field(I_res, basis, mesh, k_res, eta0,
                                        theta_half, np.full_like(theta_half, np.pi / 2))
    gain_h = np.abs(E_th_h)**2 + np.abs(E_ph_h)**2

    # --- Compute directivity on a full (theta, phi) grid ---
    n_th_grid = 91
    n_ph_grid = 72
    theta_grid = np.linspace(0.001, np.pi - 0.001, n_th_grid)
    phi_grid = np.linspace(0.0, 2.0 * np.pi - 2.0 * np.pi / n_ph_grid, n_ph_grid)
    E_th_2d = np.zeros((n_th_grid, n_ph_grid), dtype=np.complex128)
    E_ph_2d = np.zeros((n_th_grid, n_ph_grid), dtype=np.complex128)
    for j in range(n_ph_grid):
        E_th_2d[:, j], E_ph_2d[:, j] = compute_far_field(
            I_res, basis, mesh, k_res, eta0,
            theta_grid, np.full_like(theta_grid, phi_grid[j]))

    D, D_max, D_max_dBi = compute_directivity(E_th_2d, E_ph_2d, theta_grid, phi_grid, eta0)

    print(f"Peak directivity: {D_max:.2f} ({D_max_dBi:.2f} dBi)")
    print(f"E-plane peak at theta = {np.degrees(theta_half[np.argmax(gain_e)]):.1f} deg")
    print(f"H-plane peak at theta = {np.degrees(theta_half[np.argmax(gain_h)]):.1f} deg")

    # Normalize patterns to directivity
    gain_max = max(gain_e.max(), gain_h.max())
    gain_e_dBi = 10.0 * np.log10(np.maximum(gain_e / gain_max, 1e-30)) + D_max_dBi
    gain_h_dBi = 10.0 * np.log10(np.maximum(gain_h / gain_max, 1e-30)) + D_max_dBi

    # --- Plots ---
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Figure 1: Input impedance (R, X, S11)
    fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
    freq_GHz = frequencies / 1e9

    ax = axes[0]
    ax.plot(freq_GHz, R_in, 'b-o', linewidth=1.5, markersize=3)
    ax.axvline(f_res / 1e9, color='gray', linestyle='--', alpha=0.5, label='Resonance')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'$R_{in}$ ($\Omega$)')
    ax.set_title('Input Resistance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(freq_GHz, X_in, 'r-o', linewidth=1.5, markersize=3)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.axvline(f_res / 1e9, color='gray', linestyle='--', alpha=0.5, label='Resonance')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'$X_{in}$ ($\Omega$)')
    ax.set_title('Input Reactance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(freq_GHz, S11_dB, 'g-o', linewidth=1.5, markersize=3)
    ax.axhline(-10, color='k', linestyle='--', alpha=0.5, label='-10 dB')
    ax.axvline(f_res / 1e9, color='gray', linestyle='--', alpha=0.5, label='Resonance')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|S11| (dB)')
    ax.set_title(f'Return Loss (Z0={Z0:.0f} $\\Omega$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig1.suptitle(f'Dipole: {dipole_length*100:.1f} cm x {dipole_width*1000:.1f} mm strip',
                  fontsize=13)
    fig1.tight_layout()
    output_file = os.path.join(images_dir, 'dipole_impedance_sweep.png')
    fig1.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")

    # Figure 2: Current distribution along dipole at resonance
    fig2, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=(14, 5))

    x_sorted = edge_x_pos[sort_idx] * 1000  # mm
    I_sorted = I_res[sort_idx]

    ax_mag.plot(x_sorted, np.abs(I_sorted), 'b-o', linewidth=1.5, markersize=3)
    ax_mag.set_xlabel('Position along dipole (mm)')
    ax_mag.set_ylabel('|I_n| (A)')
    ax_mag.set_title(f'Current magnitude at resonance ({f_res/1e9:.2f} GHz)')
    ax_mag.grid(True, alpha=0.3)

    ax_phase.plot(x_sorted, np.degrees(np.angle(I_sorted)), 'r-o', linewidth=1.5, markersize=3)
    ax_phase.set_xlabel('Position along dipole (mm)')
    ax_phase.set_ylabel('Phase (deg)')
    ax_phase.set_title('Current phase at resonance')
    ax_phase.grid(True, alpha=0.3)

    fig2.suptitle(f'Current distribution at resonance, f = {f_res/1e9:.2f} GHz', fontsize=13)
    fig2.tight_layout()
    output_file = os.path.join(images_dir, 'dipole_current_distribution.png')
    fig2.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    # Figure 3: Surface current heatmap at resonance
    fig3 = plt.figure(figsize=(12, 5))
    ax_curr = fig3.add_subplot(111, projection='3d')
    plot_surface_current(I_res, basis, mesh, ax=ax_curr, cmap='hot',
                         edge_color='gray', edge_width=0.2,
                         title=f'|J| on dipole at resonance, f = {f_res/1e9:.2f} GHz')
    ax_curr.view_init(elev=25, azim=-60)
    output_file = os.path.join(images_dir, 'dipole_surface_current.png')
    fig3.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    # Figure 4: Radiation pattern with directivity
    theta_deg = np.degrees(theta_half)

    fig4, (ax_rect, ax_polar_e, ax_polar_h) = plt.subplots(
        1, 3, figsize=(18, 5),
        subplot_kw={'projection': None})

    # Rectangular pattern cuts (dBi)
    ax_rect.plot(theta_deg, gain_e_dBi, 'b-', linewidth=1.5, label='E-plane (xz)')
    ax_rect.plot(theta_deg, gain_h_dBi, 'r--', linewidth=1.5, label='H-plane (yz)')
    ax_rect.set_xlabel('Theta (deg)')
    ax_rect.set_ylabel('Directivity (dBi)')
    ax_rect.set_title(f'Radiation pattern, D_max = {D_max_dBi:.1f} dBi')
    ax_rect.legend()
    ax_rect.grid(True, alpha=0.3)
    ax_rect.set_xlim([0, 180])
    ax_rect.set_ylim([max(gain_e_dBi.min(), -40), D_max_dBi + 3])

    # E-plane polar (linear scale)
    theta_full = np.concatenate([theta_half, 2 * np.pi - theta_half[::-1]])
    gain_e_full = np.concatenate([gain_e, gain_e[::-1]])
    gain_e_linear_norm = np.maximum(gain_e_full / gain_max, 0)

    ax_polar_e.set_visible(False)
    ax_pe = fig4.add_subplot(132, polar=True)
    ax_pe.plot(theta_full, gain_e_linear_norm, 'b-', linewidth=1.5)
    ax_pe.set_theta_zero_location('N')
    ax_pe.set_theta_direction(-1)
    ax_pe.set_title(f'E-plane (xz)\nD_max = {D_max_dBi:.1f} dBi', pad=15)

    # H-plane polar
    gain_h_full = np.concatenate([gain_h, gain_h[::-1]])
    gain_h_linear_norm = np.maximum(gain_h_full / gain_max, 0)

    ax_polar_h.set_visible(False)
    ax_ph = fig4.add_subplot(133, polar=True)
    ax_ph.plot(theta_full, gain_h_linear_norm, 'r-', linewidth=1.5)
    ax_ph.set_theta_zero_location('N')
    ax_ph.set_theta_direction(-1)
    ax_ph.set_title('H-plane (yz)', pad=15)

    fig4.suptitle(f'Radiation pattern at resonance, f = {f_res/1e9:.2f} GHz', fontsize=13)
    fig4.tight_layout()
    output_file = os.path.join(images_dir, 'dipole_radiation_pattern.png')
    fig4.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    plt.show()

    print("\n" + "=" * 60)
    print("Dipole impedance sweep complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
