"""
Example: PEC Sphere RCS — MoM vs Mie Series Validation

Produces:
  Figure 1 Left:  Monostatic RCS (dBsm) vs frequency
  Figure 1 Right: Bistatic RCS (dBsm) vs elevation angle at a single frequency
  Figure 2: Induced surface current density on the sphere (scalar heatmap)
  Figure 3: Surface current vectors on the sphere (3D arrows)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from pyMoM3d import (
    Sphere,
    GmshMesher,
    compute_rwg_connectivity,
    fill_impedance_matrix,
    PlaneWaveExcitation,
    solve_direct,
    compute_far_field,
    compute_rcs,
    plot_mesh_3d,
    plot_surface_current,
    plot_surface_current_vectors,
    configure_latex_style,
    eta0,
    c0,
)
from pyMoM3d.analysis.mie_series import mie_rcs_pec_sphere, mie_monostatic_rcs_pec_sphere

# Configure LaTeX-style plotting
configure_latex_style()


def main():
    print("=" * 60)
    print("PEC Sphere RCS: MoM vs Mie Series")
    print("=" * 60)

    # --- Parameters ---
    radius = 0.1           # meters
    target_edge_length = 0.02  # ~lambda/10 at 1.5 GHz

    # Frequencies for monostatic sweep
    frequencies = np.linspace(0.5e9, 2.0e9, 16)

    # Fixed frequency for bistatic plot
    f_bistatic = 1.0e9

    sphere = Sphere(radius=radius)

    print(f"\nRadius:             {radius} m")
    print(f"Target edge length: {target_edge_length} m")
    print(f"Monostatic sweep:   {frequencies[0]/1e9:.1f} - {frequencies[-1]/1e9:.1f} GHz ({len(frequencies)} points)")
    print(f"Bistatic frequency: {f_bistatic/1e9:.1f} GHz")

    # --- Mesh (single mesh used for all frequencies) ---
    print("\n--- Meshing ---")
    mesher = GmshMesher(target_edge_length=target_edge_length)
    mesh = mesher.mesh_from_geometry(sphere)
    basis = compute_rwg_connectivity(mesh)

    stats = mesh.get_statistics()
    print(f"Vertices:     {stats['num_vertices']}")
    print(f"Triangles:    {stats['num_triangles']}")
    print(f"RWG basis:    {basis.num_basis}")
    print(f"Mean edge:    {stats['mean_edge_length']:.4f} m")

    # --- Excitation (same for all frequencies) ---
    exc = PlaneWaveExcitation(
        E0=np.array([1.0, 0.0, 0.0]),      # x-polarized
        k_hat=np.array([0.0, 0.0, -1.0]),   # propagating in -z
    )

    # ============================================================
    # Monostatic RCS vs frequency
    # ============================================================
    print("\n--- Monostatic RCS sweep ---")
    rcs_mono_mom_dBsm = np.zeros(len(frequencies))
    rcs_mono_mie_dBsm = np.zeros(len(frequencies))

    for i, freq in enumerate(frequencies):
        k = 2.0 * np.pi * freq / c0
        ka = k * radius
        wavelength = c0 / freq

        # MoM solve
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
        V = exc.compute_voltage_vector(basis, mesh, k)
        I = solve_direct(Z, V)

        # Backscatter: +z direction (theta=0 in spherical coords)
        # Wave propagates in -z, so backscatter is +z = theta~0
        theta_back = np.array([0.001])
        phi_back = np.array([0.0])
        E_th, E_ph = compute_far_field(I, basis, mesh, k, eta0, theta_back, phi_back)
        rcs_mono_mom_dBsm[i] = compute_rcs(E_th, E_ph, E_inc_mag=1.0)[0]

        # Mie monostatic (returns sigma / (pi*a^2), convert to dBsm)
        mie_norm = mie_monostatic_rcs_pec_sphere(ka)
        sigma_mie = mie_norm * np.pi * radius**2
        rcs_mono_mie_dBsm[i] = 10.0 * np.log10(max(sigma_mie, 1e-30))

        print(f"  f={freq/1e9:5.2f} GHz, ka={ka:.2f}: "
              f"MoM={rcs_mono_mom_dBsm[i]:7.2f} dBsm, "
              f"Mie={rcs_mono_mie_dBsm[i]:7.2f} dBsm, "
              f"err={abs(rcs_mono_mom_dBsm[i]-rcs_mono_mie_dBsm[i]):.2f} dB")

    # ============================================================
    # Bistatic RCS vs angle at f_bistatic
    # ============================================================
    print(f"\n--- Bistatic RCS at {f_bistatic/1e9:.1f} GHz ---")
    k_bi = 2.0 * np.pi * f_bistatic / c0
    ka_bi = k_bi * radius

    Z_bi = fill_impedance_matrix(basis, mesh, k_bi, eta0, quad_order=4)
    V_bi = exc.compute_voltage_vector(basis, mesh, k_bi)
    I_bi = solve_direct(Z_bi, V_bi)

    residual = np.linalg.norm(Z_bi @ I_bi - V_bi) / np.linalg.norm(V_bi)
    print(f"Cond(Z):        {np.linalg.cond(Z_bi):.2e}")
    print(f"||ZI-V||/||V||: {residual:.2e}")

    theta_bi = np.linspace(0.001, np.pi - 0.001, 181)
    phi_bi = np.zeros_like(theta_bi)
    E_th_bi, E_ph_bi = compute_far_field(I_bi, basis, mesh, k_bi, eta0, theta_bi, phi_bi)
    rcs_bi_mom_dBsm = compute_rcs(E_th_bi, E_ph_bi, E_inc_mag=1.0)

    # Mie bistatic (sigma/(pi*a^2), convert to dBsm)
    # Mie convention: theta=0 is forward scatter (-z), theta=pi is backscatter (+z)
    # MoM convention: standard spherical coords, theta=0 is +z, theta=pi is -z
    # So Mie theta = pi - MoM theta
    mie_bi_norm = mie_rcs_pec_sphere(ka_bi, np.pi - theta_bi)
    rcs_bi_mie_dBsm = 10.0 * np.log10(np.maximum(mie_bi_norm * np.pi * radius**2, 1e-30))

    # ============================================================
    # Plots
    # ============================================================
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Monostatic RCS vs frequency
    freq_GHz = frequencies / 1e9
    ax1.plot(freq_GHz, rcs_mono_mie_dBsm, 'k-', linewidth=2, label='Mie (exact)')
    ax1.plot(freq_GHz, rcs_mono_mom_dBsm, 'ro--', linewidth=1.5, markersize=5,
             label=rf'MoM ($N={basis.num_basis}$)')
    ax1.set_xlabel(r'Frequency $f$ (GHz)')
    ax1.set_ylabel(r'Monostatic RCS $\sigma$ (dBsm)')
    ax1.set_title(rf'Monostatic RCS, PEC Sphere $a = {radius*100:.0f}$ cm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Bistatic RCS vs elevation angle
    theta_deg = np.degrees(theta_bi)
    ax2.plot(theta_deg, rcs_bi_mie_dBsm, 'k-', linewidth=2, label='Mie (exact)')
    ax2.plot(theta_deg, rcs_bi_mom_dBsm, 'r--', linewidth=1.5,
             label=rf'MoM ($N={basis.num_basis}$)')
    ax2.set_xlabel(r'Bistatic angle $\theta$ (deg)')
    ax2.set_ylabel(r'Bistatic RCS $\sigma$ (dBsm)')
    ax2.set_title(rf'Bistatic RCS at $f = {f_bistatic/1e9:.1f}$ GHz ($ka = {ka_bi:.2f}$)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 180])

    fig.tight_layout()
    output_file = os.path.join(images_dir, 'sphere_rcs_validation.png')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")

    # Figure 2: Induced surface current at bistatic frequency
    fig2 = plt.figure(figsize=(10, 8))
    ax_curr = fig2.add_subplot(111, projection='3d')
    plot_surface_current(I_bi, basis, mesh, ax=ax_curr, cmap='hot',
                         edge_color='gray', edge_width=0.3,
                         title=(rf'Induced Surface Current $|\mathbf{{J}}|$ on PEC Sphere'
                                '\n'
                                rf'$f = {f_bistatic/1e9:.1f}$ GHz ($ka = {ka_bi:.2f}$), '
                                rf'$N = {basis.num_basis}$'))
    ax_curr.view_init(elev=30, azim=-60)
    output_file = os.path.join(images_dir, 'sphere_surface_current.png')
    fig2.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    # Figure 3: Surface current vectors
    fig3 = plt.figure(figsize=(10, 8))
    ax_vec = fig3.add_subplot(111, projection='3d')
    ax_vec, sm = plot_surface_current_vectors(
        I_bi, basis, mesh, ax=ax_vec,
        subsample=300,
        subsample_method='magnitude',
        cmap='plasma',
        scale=1.5,
        title=(rf'Surface Current Vectors $\mathrm{{Re}}(\mathbf{{J}})$ on PEC Sphere'
               '\n'
               rf'$f = {f_bistatic/1e9:.1f}$ GHz, $t = 0$'),
    )
    plt.colorbar(sm, ax=ax_vec, label=r'$|\mathbf{J}|$ (A/m)', shrink=0.6)
    ax_vec.view_init(elev=30, azim=-60)
    output_file = os.path.join(images_dir, 'sphere_surface_current_vectors.png')
    fig3.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    plt.show()

    print("\n" + "=" * 60)
    print("Sphere RCS validation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
