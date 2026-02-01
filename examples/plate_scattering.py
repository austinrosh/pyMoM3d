"""
Example: Plane wave scattering from a PEC rectangular plate.

Demonstrates the full solver pipeline on an open surface:
1. Mesh a rectangular plate
2. Illuminate with a broadside plane wave
3. Solve for surface currents
4. Compute bistatic RCS in the principal plane
5. Compare current magnitude against physical optics (PO) estimate

For electrically large plates, PO predicts J ~ 2(n_hat x H_inc).
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
    fill_impedance_matrix,
    PlaneWaveExcitation,
    solve_direct,
    compute_far_field,
    compute_rcs,
    eta0,
    c0,
)


def main():
    print("=" * 60)
    print("PEC Plate Scattering Example")
    print("=" * 60)

    # --- Parameters ---
    width = 0.3       # x-dimension (m)
    height = 0.3      # y-dimension (m)
    frequency = 1e9   # Hz
    subdivisions = 5  # mesh density

    k = 2.0 * np.pi * frequency / c0
    wavelength = c0 / frequency

    print(f"\nPlate:        {width} x {height} m")
    print(f"Frequency:    {frequency/1e9:.2f} GHz")
    print(f"Wavelength:   {wavelength:.4f} m")
    print(f"Plate size:   {width/wavelength:.2f} x {height/wavelength:.2f} lambda")

    # --- Mesh ---
    print("\n--- Meshing ---")
    plate = RectangularPlate(width, height, center=(0, 0, 0))
    trimesh_obj = plate.to_trimesh(subdivisions=subdivisions)
    mesher = PythonMesher()
    mesh = mesher.mesh_from_geometry(trimesh_obj)
    basis = compute_rwg_connectivity(mesh)

    stats = mesh.get_statistics()
    print(f"Vertices:     {stats['num_vertices']}")
    print(f"Triangles:    {stats['num_triangles']}")
    print(f"RWG basis:    {basis.num_basis}")
    print(f"  Interior:   {basis.num_basis}")
    print(f"  Boundary:   {basis.num_boundary_edges}")
    print(f"Mean edge:    {stats['mean_edge_length']:.4f} m")

    mesh.check_density(frequency)

    # --- Solve ---
    print("\n--- Impedance matrix fill ---")
    Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
    print(f"Z shape:      {Z.shape}")
    print(f"Cond(Z):      {np.linalg.cond(Z):.2e}")

    print("\n--- Plane wave excitation + solve ---")
    # Broadside incidence: propagating in -z, x-polarized
    exc = PlaneWaveExcitation(
        E0=np.array([1.0, 0.0, 0.0]),
        k_hat=np.array([0.0, 0.0, -1.0]),
    )
    V = exc.compute_voltage_vector(basis, mesh, k)
    I = solve_direct(Z, V)

    residual = np.linalg.norm(Z @ I - V) / np.linalg.norm(V)
    print(f"||ZI-V||/||V||: {residual:.2e}")

    # --- Physical optics comparison ---
    # PO: J = 2 * (n_hat x H_inc) on the illuminated side
    # For broadside plane wave: H_inc = (1/eta0) * y_hat
    # n_hat = z_hat for a plate in the xy-plane
    # J_PO = 2 * (z_hat x (1/eta0) * y_hat) = 2/eta0 * (-x_hat)
    # |J_PO| = 2/eta0
    J_PO_mag = 2.0 / eta0
    print(f"\nPO current:   |J_PO| = 2/eta0 = {J_PO_mag:.6f} A/m")
    print(f"Mean |I_n|:   {np.mean(np.abs(I)):.6f}")

    # --- Far-field / RCS ---
    print("\n--- Far-field computation ---")
    # RCS in xz-plane (phi=0)
    theta = np.linspace(0.001, np.pi - 0.001, 181)
    phi = np.zeros_like(theta)
    E_theta, E_phi = compute_far_field(I, basis, mesh, k, eta0, theta, phi)
    rcs_dBsm = compute_rcs(E_theta, E_phi, E_inc_mag=1.0)

    # Specular direction for broadside = backscatter (theta=pi for -z incidence)
    idx_spec = np.argmax(rcs_dBsm)
    print(f"Peak RCS:     {rcs_dBsm[idx_spec]:.2f} dBsm at theta={np.degrees(theta[idx_spec]):.1f} deg")
    print(f"Backscatter:  {rcs_dBsm[-1]:.2f} dBsm")

    # PO backscatter RCS for a rectangular plate:
    # sigma_PO = 4*pi*A^2/lambda^2
    A = width * height
    rcs_po = 4 * np.pi * A**2 / wavelength**2
    rcs_po_dBsm = 10 * np.log10(rcs_po)
    print(f"PO backscatter: {rcs_po_dBsm:.2f} dBsm")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Bistatic RCS
    ax = axes[0]
    theta_deg = np.degrees(theta)
    ax.plot(theta_deg, rcs_dBsm, 'b-', linewidth=1.5, label='MoM')
    ax.axhline(rcs_po_dBsm, color='r', linestyle='--', alpha=0.7, label='PO backscatter')
    ax.set_xlabel('Theta (deg)')
    ax.set_ylabel('RCS (dBsm)')
    ax.set_title(f'Bistatic RCS, {width/wavelength:.1f}$\\lambda$ plate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 180])

    # Current magnitude distribution
    ax = axes[1]
    ax.bar(range(basis.num_basis), np.abs(I), color='steelblue', alpha=0.7)
    ax.axhline(J_PO_mag * stats['mean_edge_length'], color='r', linestyle='--',
               alpha=0.7, label='PO estimate')
    ax.set_xlabel('Basis function index')
    ax.set_ylabel('|I_n|')
    ax.set_title('Current coefficients')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Current phase
    ax = axes[2]
    ax.bar(range(basis.num_basis), np.degrees(np.angle(I)), color='orange', alpha=0.7)
    ax.set_xlabel('Basis function index')
    ax.set_ylabel('Phase(I_n) (deg)')
    ax.set_title('Current phase')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    output_file = os.path.join(images_dir, 'plate_scattering.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")

    plt.show()

    print("\n" + "=" * 60)
    print("Plate scattering example complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
