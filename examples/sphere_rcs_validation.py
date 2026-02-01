"""
Example: PEC Sphere RCS — MoM vs Mie Series Validation

This is the critical validation example for the solver. It:
1. Meshes a PEC sphere
2. Illuminates with a plane wave
3. Solves the EFIE for surface currents
4. Computes bistatic RCS
5. Compares against the exact Mie series solution

A match to within a few dB at ka ~ 1 with lambda/10 meshing
confirms the solver is working correctly.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from pyMoM3d import (
    Sphere,
    PythonMesher,
    compute_rwg_connectivity,
    fill_impedance_matrix,
    PlaneWaveExcitation,
    solve_direct,
    compute_far_field,
    compute_rcs,
    plot_mesh_3d,
    plot_surface_current,
    eta0,
    c0,
)
from pyMoM3d.analysis.mie_series import mie_rcs_pec_sphere


def main():
    print("=" * 60)
    print("PEC Sphere RCS: MoM vs Mie Series")
    print("=" * 60)

    # --- Parameters ---
    radius = 0.1           # meters
    frequency = 1.5e9      # Hz  (ka ~ pi at this freq/radius)
    subdivisions = 2       # icosphere refinement

    k = 2.0 * np.pi * frequency / c0
    wavelength = c0 / frequency
    ka = k * radius

    print(f"\nRadius:       {radius} m")
    print(f"Frequency:    {frequency/1e9:.2f} GHz")
    print(f"Wavelength:   {wavelength:.4f} m")
    print(f"ka:           {ka:.3f}")
    print(f"lambda/10:    {wavelength/10:.4f} m")

    # --- Mesh ---
    print("\n--- Meshing ---")
    sphere = Sphere(radius=radius)
    trimesh_obj = sphere.to_trimesh(subdivisions=subdivisions)
    mesher = PythonMesher()
    mesh = mesher.mesh_from_geometry(trimesh_obj)
    basis = compute_rwg_connectivity(mesh)

    stats = mesh.get_statistics()
    print(f"Vertices:     {stats['num_vertices']}")
    print(f"Triangles:    {stats['num_triangles']}")
    print(f"RWG basis:    {basis.num_basis}")
    print(f"Mean edge:    {stats['mean_edge_length']:.4f} m")
    print(f"Edges/lambda: {wavelength / stats['mean_edge_length']:.1f}")

    mesh.check_density(frequency)

    # --- Z-matrix fill ---
    print("\n--- Impedance matrix fill ---")
    Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
    print(f"Z shape:      {Z.shape}")
    print(f"Z symmetric:  {np.allclose(Z, Z.T, atol=1e-10)}")
    print(f"Cond(Z):      {np.linalg.cond(Z):.2e}")

    # --- Excitation + solve ---
    print("\n--- Plane wave excitation + solve ---")
    exc = PlaneWaveExcitation(
        E0=np.array([1.0, 0.0, 0.0]),      # x-polarized
        k_hat=np.array([0.0, 0.0, -1.0]),   # propagating in -z
    )
    V = exc.compute_voltage_vector(basis, mesh, k)
    I = solve_direct(Z, V)

    residual = np.linalg.norm(Z @ I - V) / np.linalg.norm(V)
    print(f"||ZI-V||/||V||: {residual:.2e}")

    # --- Far-field / RCS ---
    print("\n--- Far-field computation ---")
    theta = np.linspace(0.001, np.pi - 0.001, 181)
    phi = np.zeros_like(theta)

    E_theta, E_phi = compute_far_field(I, basis, mesh, k, eta0, theta, phi)
    rcs_mom_dBsm = compute_rcs(E_theta, E_phi, E_inc_mag=1.0)

    # Normalize to sigma / (pi*a^2) for comparison with Mie
    rcs_mom_linear = 10.0**(rcs_mom_dBsm / 10.0)
    rcs_mom_norm = rcs_mom_linear / (np.pi * radius**2)

    # --- Mie series ---
    print("--- Mie series reference ---")
    rcs_mie_norm = mie_rcs_pec_sphere(ka, theta)

    # --- Comparison ---
    # Backscatter (theta = pi)
    idx_back = -1
    mom_back_dB = 10.0 * np.log10(max(rcs_mom_norm[idx_back], 1e-30))
    mie_back_dB = 10.0 * np.log10(max(rcs_mie_norm[idx_back], 1e-30))
    print(f"\nBackscatter RCS / (pi*a^2):")
    print(f"  MoM:  {mom_back_dB:.2f} dB")
    print(f"  Mie:  {mie_back_dB:.2f} dB")
    print(f"  Error: {abs(mom_back_dB - mie_back_dB):.2f} dB")

    # --- Plot ---
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Figure 1: RCS comparison
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    theta_deg = np.degrees(theta)
    rcs_mie_dB = 10.0 * np.log10(np.maximum(rcs_mie_norm, 1e-30))
    rcs_mom_dB = 10.0 * np.log10(np.maximum(rcs_mom_norm, 1e-30))

    ax1.plot(theta_deg, rcs_mie_dB, 'k-', linewidth=2, label='Mie (exact)')
    ax1.plot(theta_deg, rcs_mom_dB, 'r--', linewidth=1.5, label=f'MoM (N={basis.num_basis})')
    ax1.set_xlabel('Bistatic angle (deg)')
    ax1.set_ylabel(r'$\sigma / (\pi a^2)$ (dB)')
    ax1.set_title(f'PEC Sphere Bistatic RCS, ka = {ka:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 180])

    I_mag = np.abs(I)
    ax2.bar(range(basis.num_basis), I_mag, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Basis function index')
    ax2.set_ylabel('|I_n| (A)')
    ax2.set_title('Surface current coefficients')
    ax2.grid(True, alpha=0.3)

    fig1.tight_layout()
    output_file = os.path.join(images_dir, 'sphere_rcs_validation.png')
    fig1.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")

    # Figure 2: Mesh visualization
    fig2 = plt.figure(figsize=(10, 8))
    ax_mesh = fig2.add_subplot(111, projection='3d')
    plot_mesh_3d(mesh, ax=ax_mesh, color='lightblue', alpha=0.6,
                 edge_color='navy', edge_width=0.4)
    ax_mesh.set_title(f'PEC Sphere Mesh: {stats["num_triangles"]} triangles, '
                      f'{basis.num_basis} RWG basis functions')
    output_file = os.path.join(images_dir, 'sphere_mesh.png')
    fig2.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    # Figure 3: Surface current heatmap
    fig3 = plt.figure(figsize=(11, 8))
    ax_curr = fig3.add_subplot(111, projection='3d')
    plot_surface_current(I, basis, mesh, ax=ax_curr, cmap='hot',
                         title=f'|J| on PEC Sphere, f = {frequency/1e9:.2f} GHz')
    output_file = os.path.join(images_dir, 'sphere_surface_current.png')
    fig3.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    plt.show()

    print("\n" + "=" * 60)
    print("Sphere RCS validation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
