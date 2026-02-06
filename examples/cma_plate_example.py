"""
Example: Characteristic Mode Analysis of a Rectangular Plate

Demonstrates CMA on a simple PEC rectangular plate:
  - Computes characteristic modes at a single frequency
  - Visualizes the first few modal currents
  - Shows modal significance and characteristic angles
  - Performs frequency sweep to track eigenvalue vs frequency

This is a canonical structure for CMA with well-understood modal behavior:
  - Mode 1: Half-wave dipole-like (current max at center)
  - Mode 2: Full-wave (two lobes with null at center)
  - Higher modes: Increasingly complex current distributions
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
    compute_characteristic_modes,
    verify_orthogonality,
    plot_surface_current,
    configure_latex_style,
    eta0,
    c0,
)

# Configure LaTeX-style plotting
configure_latex_style()


def main():
    print("=" * 60)
    print("Characteristic Mode Analysis: Rectangular Plate")
    print("=" * 60)

    # --- Parameters ---
    # Plate dimensions: approximately half-wavelength at 1 GHz
    plate_width = 0.15   # meters (x-direction)
    plate_height = 0.10  # meters (y-direction)
    target_edge_length = 0.015  # ~lambda/20 at 1 GHz for good resolution

    # Analysis frequency
    f_analysis = 1.0e9  # 1 GHz
    wavelength = c0 / f_analysis

    print(f"\nPlate dimensions: {plate_width*100:.1f} x {plate_height*100:.1f} cm")
    print(f"Analysis frequency: {f_analysis/1e9:.2f} GHz")
    print(f"Wavelength: {wavelength*100:.1f} cm")
    print(f"Plate width / wavelength: {plate_width/wavelength:.2f}")

    # --- Create geometry and mesh ---
    print("\n--- Meshing ---")
    plate = RectangularPlate(width=plate_width, height=plate_height)
    mesher = GmshMesher(target_edge_length=target_edge_length)
    mesh = mesher.mesh_from_geometry(plate)
    basis = compute_rwg_connectivity(mesh)

    stats = mesh.get_statistics()
    print(f"Vertices:     {stats['num_vertices']}")
    print(f"Triangles:    {stats['num_triangles']}")
    print(f"RWG basis:    {basis.num_basis}")
    print(f"Mean edge:    {stats['mean_edge_length']*100:.2f} cm")
    print(f"Edges per wavelength: {wavelength / stats['mean_edge_length']:.1f}")

    # --- Compute impedance matrix ---
    print("\n--- Computing impedance matrix ---")
    k = 2.0 * np.pi * f_analysis / c0
    Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
    print(f"Z matrix size: {Z.shape[0]} x {Z.shape[1]}")
    print(f"Z condition number: {np.linalg.cond(Z):.2e}")

    # --- Characteristic Mode Analysis ---
    print("\n--- Characteristic Mode Analysis ---")
    cma = compute_characteristic_modes(Z, frequency=f_analysis)

    # Verify orthogonality
    is_orthog, max_error = verify_orthogonality(cma)
    print(f"Modes R-orthogonal: {is_orthog} (max error: {max_error:.2e})")

    # Display top modes
    num_display = min(6, basis.num_basis)
    print(f"\nTop {num_display} modes by modal significance:")
    print("-" * 50)
    print(f"{'Mode':>4} {'λ_n':>10} {'MS':>8} {'α (deg)':>10} {'Status':>12}")
    print("-" * 50)

    for n in range(num_display):
        lambda_n = cma.get_eigenvalue(n)
        ms_n = cma.get_modal_significance(n)
        alpha_n = cma.get_characteristic_angle(n)

        # Determine mode status
        if abs(lambda_n) < 0.1:
            status = "resonant"
        elif lambda_n > 0:
            status = "inductive"
        else:
            status = "capacitive"

        print(f"{n+1:>4} {lambda_n:>10.3f} {ms_n:>8.3f} {alpha_n:>10.1f} {status:>12}")

    # --- Visualization ---
    print("\n--- Generating plots ---")
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Plot first 4 modal currents
    num_plot = min(4, basis.num_basis)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    for n in range(num_plot):
        J_n = cma.get_mode(n)
        ms_n = cma.get_modal_significance(n)
        lambda_n = cma.get_eigenvalue(n)

        ax = axes[n]
        plot_surface_current(
            J_n, basis, mesh, ax=ax, cmap='hot',
            title=rf'Mode {n+1}: $\lambda = {lambda_n:.2f}$, MS $= {ms_n:.3f}$',
        )
        ax.view_init(elev=60, azim=-45)

    fig.suptitle(rf'Characteristic Modes of ${plate_width*100:.0f} \times {plate_height*100:.0f}$ cm Plate'
                 '\n'
                 rf'$f = {f_analysis/1e9:.2f}$ GHz, $N = {basis.num_basis}$',
                 fontsize=14)
    fig.tight_layout()

    output_file = os.path.join(images_dir, 'cma_plate_modes.png')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved modal currents to: {output_file}")

    # --- Frequency sweep for eigenvalue spectrum ---
    print("\n--- Frequency sweep for eigenvalue spectrum ---")
    frequencies = np.linspace(0.5e9, 2.0e9, 21)
    num_track = 5  # Track first 5 modes

    eigenvalues_vs_freq = np.zeros((len(frequencies), num_track))
    ms_vs_freq = np.zeros((len(frequencies), num_track))

    for i, freq in enumerate(frequencies):
        k_f = 2.0 * np.pi * freq / c0
        Z_f = fill_impedance_matrix(basis, mesh, k_f, eta0, quad_order=4)
        cma_f = compute_characteristic_modes(Z_f, frequency=freq, num_modes=num_track)

        for n in range(num_track):
            eigenvalues_vs_freq[i, n] = cma_f.get_eigenvalue(n)
            ms_vs_freq[i, n] = cma_f.get_modal_significance(n)

        print(f"  f={freq/1e9:.2f} GHz: λ_1={cma_f.get_eigenvalue(0):>7.2f}, "
              f"MS_1={cma_f.get_modal_significance(0):.3f}")

    # Plot eigenvalue spectrum
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    freq_ghz = frequencies / 1e9
    colors = plt.cm.tab10(np.linspace(0, 1, num_track))

    for n in range(num_track):
        ax1.plot(freq_ghz, eigenvalues_vs_freq[:, n], 'o-',
                 color=colors[n], label=rf'Mode {n+1}', markersize=4)

    ax1.axhline(0, color='k', linestyle='--', alpha=0.3, label=r'Resonance ($\lambda = 0$)')
    ax1.set_xlabel(r'Frequency $f$ (GHz)')
    ax1.set_ylabel(r'Eigenvalue $\lambda_n$')
    ax1.set_title(r'Characteristic Eigenvalues vs Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([freq_ghz[0], freq_ghz[-1]])

    for n in range(num_track):
        ax2.plot(freq_ghz, ms_vs_freq[:, n], 'o-',
                 color=colors[n], label=rf'Mode {n+1}', markersize=4)

    ax2.axhline(0.707, color='k', linestyle='--', alpha=0.3, label=r'MS $= 0.707$')
    ax2.set_xlabel(r'Frequency $f$ (GHz)')
    ax2.set_ylabel(r'Modal Significance MS$_n$')
    ax2.set_title(r'Modal Significance vs Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([freq_ghz[0], freq_ghz[-1]])
    ax2.set_ylim([0, 1.05])

    fig2.suptitle(rf'CMA Frequency Sweep: ${plate_width*100:.0f} \times {plate_height*100:.0f}$ cm Plate',
                  fontsize=14)
    fig2.tight_layout()

    output_file = os.path.join(images_dir, 'cma_plate_spectrum.png')
    fig2.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved eigenvalue spectrum to: {output_file}")

    plt.show()

    print("\n" + "=" * 60)
    print("CMA plate analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
