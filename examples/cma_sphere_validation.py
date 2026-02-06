"""
Example: Characteristic Mode Analysis of a PEC Sphere — Validation

Demonstrates CMA on a PEC sphere and validates against theoretical expectations:
  - Sphere modes correspond to spherical harmonics (TM and TE modes)
  - Mode eigenvalues cross zero at frequencies determined by spherical Bessel functions
  - Modes are degenerate due to spherical symmetry

For a PEC sphere of radius a, the characteristic eigenvalues λ_n cross zero
(resonance) at frequencies where:
  - TM modes: j_n(ka) = 0 (spherical Bessel function zeros)
  - TE modes: [ka * j_n(ka)]' = 0

The first TM mode (dipole, n=1) resonates at ka ≈ 2.744 (first zero of j_1).

Note: Due to mesh discretization and numerical errors, computed resonances
will differ slightly from theoretical values.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

from pyMoM3d import (
    Sphere,
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


def find_bessel_zeros(n, num_zeros=3):
    """Find zeros of spherical Bessel function j_n(x).

    Uses simple bisection search in intervals.
    """
    zeros = []
    x = 0.1
    dx = 0.01
    prev_sign = np.sign(spherical_jn(n, x))

    while len(zeros) < num_zeros and x < 50:
        x += dx
        curr_sign = np.sign(spherical_jn(n, x))
        if curr_sign != prev_sign and prev_sign != 0:
            # Found sign change, refine with bisection
            a, b = x - dx, x
            for _ in range(50):
                mid = (a + b) / 2
                if np.sign(spherical_jn(n, mid)) == np.sign(spherical_jn(n, a)):
                    a = mid
                else:
                    b = mid
            zeros.append((a + b) / 2)
        prev_sign = curr_sign

    return np.array(zeros)


def main():
    print("=" * 60)
    print("Characteristic Mode Analysis: PEC Sphere Validation")
    print("=" * 60)

    # --- Parameters ---
    radius = 0.1  # meters
    target_edge_length = 0.025  # Coarser mesh for faster computation

    # Frequency range for sweep
    # ka ranges from ~0.5 to ~4 to capture first few resonances
    ka_min, ka_max = 0.5, 4.0
    num_freqs = 31
    ka_values = np.linspace(ka_min, ka_max, num_freqs)
    frequencies = ka_values * c0 / (2 * np.pi * radius)

    print(f"\nSphere radius: {radius*100:.1f} cm")
    print(f"ka range: {ka_min:.1f} to {ka_max:.1f}")
    print(f"Frequency range: {frequencies[0]/1e9:.2f} to {frequencies[-1]/1e9:.2f} GHz")

    # Theoretical resonances: zeros of j_n(ka) for TM modes
    print("\n--- Theoretical TM Mode Resonances ---")
    print("(Zeros of spherical Bessel functions j_n(ka))")
    for n in [1, 2, 3]:
        zeros = find_bessel_zeros(n, num_zeros=2)
        print(f"  TM{n} (dipole n={n}): ka = {zeros}")

    # --- Create geometry and mesh ---
    print("\n--- Meshing ---")
    sphere = Sphere(radius=radius)
    mesher = GmshMesher(target_edge_length=target_edge_length)
    mesh = mesher.mesh_from_geometry(sphere)
    basis = compute_rwg_connectivity(mesh)

    stats = mesh.get_statistics()
    print(f"Vertices:     {stats['num_vertices']}")
    print(f"Triangles:    {stats['num_triangles']}")
    print(f"RWG basis:    {basis.num_basis}")
    print(f"Mean edge:    {stats['mean_edge_length']*100:.2f} cm")

    # --- Single-frequency CMA for visualization ---
    print("\n--- Single-frequency CMA at ka=2.5 ---")
    ka_single = 2.5
    k_single = ka_single / radius
    f_single = k_single * c0 / (2 * np.pi)

    Z_single = fill_impedance_matrix(basis, mesh, k_single, eta0, quad_order=4)
    cma_single = compute_characteristic_modes(Z_single, frequency=f_single)

    is_orthog, max_error = verify_orthogonality(cma_single)
    print(f"Modes R-orthogonal: {is_orthog} (max error: {max_error:.2e})")

    num_display = min(8, basis.num_basis)
    print(f"\nTop {num_display} modes at ka={ka_single}:")
    print("-" * 50)
    print(f"{'Mode':>4} {'λ_n':>10} {'MS':>8} {'α (deg)':>10}")
    print("-" * 50)
    for n in range(num_display):
        lambda_n = cma_single.get_eigenvalue(n)
        ms_n = cma_single.get_modal_significance(n)
        alpha_n = cma_single.get_characteristic_angle(n)
        print(f"{n+1:>4} {lambda_n:>10.3f} {ms_n:>8.3f} {alpha_n:>10.1f}")

    # Note about degeneracy
    print("\nNote: Due to spherical symmetry, modes appear in degenerate groups.")
    print("For example, the dipole mode (n=1) is 3-fold degenerate (m=-1,0,+1).")

    # --- Frequency sweep ---
    print("\n--- Frequency sweep for eigenvalue spectrum ---")
    num_track = min(10, basis.num_basis)  # Track more modes for sphere

    eigenvalues_vs_ka = np.zeros((len(frequencies), num_track))
    ms_vs_ka = np.zeros((len(frequencies), num_track))

    for i, freq in enumerate(frequencies):
        ka = ka_values[i]
        k = 2.0 * np.pi * freq / c0
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
        cma = compute_characteristic_modes(Z, frequency=freq, num_modes=num_track)

        for n in range(num_track):
            eigenvalues_vs_ka[i, n] = cma.get_eigenvalue(n)
            ms_vs_ka[i, n] = cma.get_modal_significance(n)

        if i % 5 == 0:
            print(f"  ka={ka:.2f}: λ_1={cma.get_eigenvalue(0):>7.2f}, "
                  f"λ_2={cma.get_eigenvalue(1):>7.2f}, "
                  f"MS_1={cma.get_modal_significance(0):.3f}")

    # --- Visualization ---
    print("\n--- Generating plots ---")
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Plot eigenvalue spectrum vs ka
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, num_track))

    for n in range(num_track):
        ax1.plot(ka_values, eigenvalues_vs_ka[:, n], '-',
                 color=colors[n], label=f'Mode {n+1}', linewidth=1.5, alpha=0.8)

    ax1.axhline(0, color='k', linestyle='--', alpha=0.5, label=r'Resonance ($\lambda = 0$)')

    # Mark theoretical TM mode resonances
    tm_colors = ['red', 'green', 'purple']
    for n_tm, color in zip([1, 2, 3], tm_colors):
        zeros = find_bessel_zeros(n_tm, num_zeros=1)
        for zero in zeros:
            if ka_min < zero < ka_max:
                ax1.axvline(zero, color=color, linestyle=':', alpha=0.7,
                            label=rf'TM$_{n_tm}$ theory ($ka = {zero:.2f}$)')

    ax1.set_xlabel(r'$ka$ (electrical size)')
    ax1.set_ylabel(r'Eigenvalue $\lambda_n$')
    ax1.set_title(r'Characteristic Eigenvalues vs Electrical Size')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([ka_min, ka_max])
    ax1.set_ylim([-10, 10])

    for n in range(num_track):
        ax2.plot(ka_values, ms_vs_ka[:, n], '-',
                 color=colors[n], label=rf'Mode {n+1}', linewidth=1.5, alpha=0.8)

    ax2.axhline(0.707, color='k', linestyle='--', alpha=0.5, label=r'MS $= 0.707$')
    ax2.set_xlabel(r'$ka$ (electrical size)')
    ax2.set_ylabel(r'Modal Significance MS$_n$')
    ax2.set_title(r'Modal Significance vs Electrical Size')
    ax2.legend(loc='lower right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([ka_min, ka_max])
    ax2.set_ylim([0, 1.05])

    fig.suptitle(rf'CMA of PEC Sphere ($a = {radius*100:.0f}$ cm, $N = {basis.num_basis}$)',
                 fontsize=14)
    fig.tight_layout()

    output_file = os.path.join(images_dir, 'cma_sphere_spectrum.png')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved eigenvalue spectrum to: {output_file}")

    # Plot first 4 modal currents at ka=2.5
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10),
                              subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    for n in range(4):
        J_n = cma_single.get_mode(n)
        ms_n = cma_single.get_modal_significance(n)
        lambda_n = cma_single.get_eigenvalue(n)

        ax = axes[n]
        plot_surface_current(
            J_n, basis, mesh, ax=ax, cmap='hot',
            title=rf'Mode {n+1}: $\lambda = {lambda_n:.2f}$, MS $= {ms_n:.3f}$',
        )
        ax.view_init(elev=30, azim=-60)

    fig2.suptitle(rf'Characteristic Modes of PEC Sphere at $ka = {ka_single}$'
                  '\n'
                  rf'$a = {radius*100:.0f}$ cm, $N = {basis.num_basis}$',
                  fontsize=14)
    fig2.tight_layout()

    output_file = os.path.join(images_dir, 'cma_sphere_modes.png')
    fig2.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved modal currents to: {output_file}")

    # --- Find resonances from computed data ---
    print("\n--- Detected resonances (λ=0 crossings) ---")
    for n in range(min(5, num_track)):
        eigenvals = eigenvalues_vs_ka[:, n]
        # Find zero crossings
        crossings = []
        for i in range(len(eigenvals) - 1):
            if eigenvals[i] * eigenvals[i + 1] < 0:
                # Linear interpolation to find ka at crossing
                ka_cross = ka_values[i] - eigenvals[i] * (ka_values[i + 1] - ka_values[i]) / (
                    eigenvals[i + 1] - eigenvals[i])
                crossings.append(ka_cross)
        if crossings:
            print(f"  Mode {n+1}: ka = {[f'{c:.2f}' for c in crossings]}")

    plt.show()

    print("\n" + "=" * 60)
    print("CMA sphere validation complete!")
    print("=" * 60)
    print("\nValidation notes:")
    print("- TM1 (dipole) mode should resonate near ka ≈ 2.74")
    print("- Due to mesh discretization, computed values may differ by ~5-10%")
    print("- Degenerate modes may not be perfectly separated numerically")


if __name__ == '__main__':
    main()
