"""
Example: Thin wire dipole input impedance vs frequency.

Approximates a thin-wire dipole as a narrow rectangular plate
with a delta-gap feed at the center edge. Sweeps frequency to
find the resonant input impedance.

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
    PythonMesher,
    compute_rwg_connectivity,
    Simulation,
    SimulationConfig,
    plot_surface_current,
    compute_far_field,
    compute_rcs,
    c0,
    eta0,
)
from pyMoM3d.mom.excitation import DeltaGapExcitation, find_nearest_edge


def main():
    print("=" * 60)
    print("Dipole Impedance Sweep")
    print("=" * 60)

    # --- Parameters ---
    # Dipole approximated as a narrow plate
    dipole_length = 0.15   # m (half-wave at ~1 GHz)
    dipole_width = 0.01    # m (strip width)
    subdivisions = 6

    print(f"\nDipole length: {dipole_length} m")
    print(f"Dipole width:  {dipole_width} m")
    print(f"Expected resonance: ~{c0 / (2 * dipole_length) / 1e9:.2f} GHz")

    # --- Mesh ---
    print("\n--- Meshing ---")
    plate = RectangularPlate(dipole_length, dipole_width, center=(0, 0, 0))
    trimesh_obj = plate.to_trimesh(subdivisions=subdivisions)
    mesher = PythonMesher()
    mesh = mesher.mesh_from_geometry(trimesh_obj)
    basis = compute_rwg_connectivity(mesh)

    stats = mesh.get_statistics()
    print(f"Vertices:     {stats['num_vertices']}")
    print(f"Triangles:    {stats['num_triangles']}")
    print(f"RWG basis:    {basis.num_basis}")

    # --- Find feed edge ---
    # The feed is at the center of the dipole
    feed_point = np.array([0.0, 0.0, 0.0])
    feed_idx = find_nearest_edge(mesh, basis, feed_point)
    feed_edge = mesh.edges[basis.edge_index[feed_idx]]
    feed_midpoint = 0.5 * (mesh.vertices[feed_edge[0]] + mesh.vertices[feed_edge[1]])
    print(f"Feed basis index: {feed_idx}")
    print(f"Feed edge midpoint: ({feed_midpoint[0]:.4f}, {feed_midpoint[1]:.4f}, {feed_midpoint[2]:.4f})")

    # --- Frequency sweep ---
    freq_start = 0.5e9
    freq_stop = 1.5e9
    n_freqs = 11
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

    # Find resonance: prefer where reactance crosses zero; fall back to peak resistance
    valid = np.isfinite(X_in) & np.isfinite(R_in)
    idx_res = 0
    f_res = frequencies[0]
    if np.any(valid):
        valid_X = X_in[valid]
        # Check for zero crossing in reactance
        sign_changes = np.where(np.diff(np.sign(valid_X.real)))[0]
        if len(sign_changes) > 0:
            # Interpolate to find zero crossing
            idx_res = np.where(valid)[0][sign_changes[0]]
            # Pick the frequency closer to zero
            if abs(X_in[idx_res]) > abs(X_in[idx_res + 1]):
                idx_res = idx_res + 1
        else:
            # No zero crossing: use peak resistance as proxy for resonance
            idx_res = np.where(valid)[0][np.argmax(R_in[valid])]
        f_res = frequencies[idx_res]
        Z_res = Z_in[idx_res]
        print(f"\nNearest to resonance: f = {f_res/1e9:.3f} GHz")
        print(f"Z_in at resonance:   {Z_res.real:.4f} + j{Z_res.imag:.4f} Ohm")

    # Resonance current coefficients
    I_res = results[idx_res].I_coefficients

    # --- S11 ---
    Z0 = 50.0
    S11 = (Z_in - Z0) / (Z_in + Z0)
    S11_dB = 20.0 * np.log10(np.maximum(np.abs(S11), 1e-30))

    # --- Far-field radiation pattern at resonance ---
    print("\n--- Radiation pattern at resonance ---")
    k_res = 2.0 * np.pi * f_res / c0

    # E-plane (xz-plane, phi=0): contains dipole axis (x) and z
    theta_e = np.linspace(0.001, 2 * np.pi - 0.001, 361)
    phi_e = np.zeros_like(theta_e)
    # Use theta 0->pi for forward, and pi->2pi by symmetry
    theta_half = np.linspace(0.001, np.pi - 0.001, 181)
    E_th_e, E_ph_e = compute_far_field(I_res, basis, mesh, k_res, eta0,
                                        theta_half, np.zeros_like(theta_half))
    gain_e = np.abs(E_th_e)**2 + np.abs(E_ph_e)**2
    gain_e_dB = 10.0 * np.log10(np.maximum(gain_e / gain_e.max(), 1e-30))

    # H-plane (yz-plane, phi=pi/2): perpendicular to dipole axis through broadside
    E_th_h, E_ph_h = compute_far_field(I_res, basis, mesh, k_res, eta0,
                                        theta_half, np.full_like(theta_half, np.pi / 2))
    gain_h = np.abs(E_th_h)**2 + np.abs(E_ph_h)**2
    # Normalize to same peak as E-plane for comparison
    gain_max = max(gain_e.max(), gain_h.max())
    gain_h_dB = 10.0 * np.log10(np.maximum(gain_h / gain_max, 1e-30))
    gain_e_dB = 10.0 * np.log10(np.maximum(gain_e / gain_max, 1e-30))

    print(f"E-plane peak at theta = {np.degrees(theta_half[np.argmax(gain_e)]):.1f} deg")
    print(f"H-plane peak at theta = {np.degrees(theta_half[np.argmax(gain_h)]):.1f} deg")

    # --- Basis function position along dipole ---
    # Compute the centroid position (along x) and orientation of each edge
    edge_x_pos = np.zeros(basis.num_basis)
    edge_is_transverse = np.zeros(basis.num_basis, dtype=bool)
    for n in range(basis.num_basis):
        edge_verts = mesh.edges[basis.edge_index[n]]
        v_a = mesh.vertices[edge_verts[0]]
        v_b = mesh.vertices[edge_verts[1]]
        edge_x_pos[n] = 0.5 * (v_a[0] + v_b[0])
        edge_dir = v_b - v_a
        edge_dir /= np.linalg.norm(edge_dir) + 1e-30
        # Transverse edges are primarily along y (across the strip width)
        edge_is_transverse[n] = abs(edge_dir[1]) > abs(edge_dir[0])

    # For the current vs position plot, show transverse (y-directed) edges
    # which carry the dominant current for an x-directed dipole
    trans_idx = np.where(edge_is_transverse)[0]
    if len(trans_idx) == 0:
        trans_idx = np.arange(basis.num_basis)
    sort_idx = trans_idx[np.argsort(edge_x_pos[trans_idx])]

    # --- Plots ---
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Figure 1: Input impedance (R, X, S11)
    fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
    freq_GHz = frequencies / 1e9

    ax = axes[0]
    ax.plot(freq_GHz, R_in, 'b-o', linewidth=1.5, markersize=4)
    ax.axvline(f_res / 1e9, color='gray', linestyle='--', alpha=0.5, label='Resonance')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'$R_{in}$ ($\Omega$)')
    ax.set_title('Input Resistance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(freq_GHz, X_in, 'r-o', linewidth=1.5, markersize=4)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.axvline(f_res / 1e9, color='gray', linestyle='--', alpha=0.5, label='Resonance')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'$X_{in}$ ($\Omega$)')
    ax.set_title('Input Reactance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(freq_GHz, S11_dB, 'g-o', linewidth=1.5, markersize=4)
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
    # Adjust view for a thin dipole (look along y to see length)
    ax_curr.view_init(elev=25, azim=-60)
    output_file = os.path.join(images_dir, 'dipole_surface_current.png')
    fig3.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    # Figure 4: Radiation pattern (E-plane and H-plane)
    fig4, (ax_e, ax_h, ax_polar) = plt.subplots(1, 3, figsize=(18, 5))

    theta_deg = np.degrees(theta_half)

    ax_e.plot(theta_deg, gain_e_dB, 'b-', linewidth=1.5, label='E-plane (xz)')
    ax_e.plot(theta_deg, gain_h_dB, 'r--', linewidth=1.5, label='H-plane (yz)')
    ax_e.set_xlabel('Theta (deg)')
    ax_e.set_ylabel('Normalized gain (dB)')
    ax_e.set_title('Radiation pattern cuts')
    ax_e.legend()
    ax_e.grid(True, alpha=0.3)
    ax_e.set_xlim([0, 180])
    ax_e.set_ylim([max(gain_e_dB.min(), -40), 3])

    # E-plane polar plot
    # Mirror to get full 360 deg
    theta_full = np.concatenate([theta_half, 2 * np.pi - theta_half[::-1]])
    gain_e_full = np.concatenate([gain_e, gain_e[::-1]])
    gain_e_full_norm = np.maximum(gain_e_full / gain_max, 1e-30)
    gain_e_linear = np.maximum(10.0**(10.0 * np.log10(gain_e_full_norm) / 10.0), 0)

    ax_h_polar = fig4.add_subplot(132, polar=True)
    ax_h.set_visible(False)
    ax_h_polar.plot(theta_full, gain_e_linear, 'b-', linewidth=1.5, label='E-plane')
    ax_h_polar.set_theta_zero_location('N')
    ax_h_polar.set_theta_direction(-1)
    ax_h_polar.set_title('E-plane (xz)', pad=15)

    # H-plane polar plot
    gain_h_full = np.concatenate([gain_h, gain_h[::-1]])
    gain_h_linear = np.maximum(gain_h_full / gain_max, 0)

    ax_polar.set_visible(False)
    ax_p2 = fig4.add_subplot(133, polar=True)
    ax_p2.plot(theta_full, gain_h_linear, 'r-', linewidth=1.5, label='H-plane')
    ax_p2.set_theta_zero_location('N')
    ax_p2.set_theta_direction(-1)
    ax_p2.set_title('H-plane (yz)', pad=15)

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
