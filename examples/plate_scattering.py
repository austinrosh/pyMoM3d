"""
Example: Plane wave scattering from a PEC rectangular plate.

Demonstrates the full solver pipeline on an open surface:
1. Mesh a rectangular plate
2. Illuminate with a broadside plane wave
3. Solve for surface currents
4. Visualize induced surface current density on the plate
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
    fill_impedance_matrix,
    PlaneWaveExcitation,
    solve_direct,
    plot_surface_current,
    configure_latex_style,
    eta0,
    c0,
)

# Configure LaTeX-style plotting
configure_latex_style()


def main():
    print("=" * 60)
    print("PEC Plate Scattering Example")
    print("=" * 60)

    # --- Parameters ---
    width = 0.3       # x-dimension (m)
    height = 0.3      # y-dimension (m)
    frequency = 1e9   # Hz
    target_edge_length = 0.02  # ~lambda/15 for good resolution

    k = 2.0 * np.pi * frequency / c0
    wavelength = c0 / frequency

    print(f"\nPlate:        {width} x {height} m")
    print(f"Frequency:    {frequency/1e9:.2f} GHz")
    print(f"Wavelength:   {wavelength:.4f} m")
    print(f"Plate size:   {width/wavelength:.2f} x {height/wavelength:.2f} lambda")

    # --- Mesh ---
    print("\n--- Meshing ---")
    plate = RectangularPlate(width, height, center=(0, 0, 0))
    mesher = GmshMesher(target_edge_length=target_edge_length)
    mesh = mesher.mesh_from_geometry(plate)
    basis = compute_rwg_connectivity(mesh)

    stats = mesh.get_statistics()
    print(f"Vertices:     {stats['num_vertices']}")
    print(f"Triangles:    {stats['num_triangles']}")
    print(f"RWG basis:    {basis.num_basis}")
    print(f"  Boundary:   {basis.num_boundary_edges}")
    print(f"Mean edge:    {stats['mean_edge_length']:.4f} m")
    print(f"Edges/lambda: {wavelength / stats['mean_edge_length']:.1f}")

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

    # --- Plot: Surface current visualization ---
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    plot_surface_current(I, basis, mesh, ax=ax, cmap='hot',
                         edge_color='gray', edge_width=0.3,
                         title=(rf'Induced Surface Current $|\mathbf{{J}}|$ on '
                                rf'${width/wavelength:.1f}\lambda \times {height/wavelength:.1f}\lambda$ PEC Plate'
                                '\n'
                                rf'$f = {frequency/1e9:.1f}$ GHz, $N = {basis.num_basis}$'))
    ax.view_init(elev=45, azim=-60)

    output_file = os.path.join(images_dir, 'plate_scattering.png')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")

    plt.show()

    print("\n" + "=" * 60)
    print("Plate scattering example complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
