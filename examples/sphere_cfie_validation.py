"""
Example: CFIE vs EFIE — Interior Resonance Comparison for PEC Sphere

Demonstrates that EFIE exhibits a spurious resonance spike near ka ≈ 2.74
(first TE111 interior resonance) while CFIE (α = 0.5) remains well-conditioned
and matches the Mie series throughout.

Produces a single figure with three panels:
  Panel 1: Monostatic RCS (dBsm) vs ka — EFIE, CFIE, Mie
  Panel 2: Condition number cond(Z) vs ka — EFIE vs CFIE (semilogy)
  Panel 3: Bistatic RCS (dBsm) vs angle at resonant ka ≈ 2.74

Output: images/sphere_cfie_validation.png
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
    fill_matrix,
    EFIEOperator,
    CFIEOperator,
    PlaneWaveExcitation,
    solve_direct,
    compute_far_field,
    compute_rcs,
    configure_latex_style,
    eta0,
    c0,
)
from pyMoM3d.analysis.mie_series import mie_rcs_pec_sphere, mie_monostatic_rcs_pec_sphere

configure_latex_style()


def main():
    print("=" * 65)
    print("CFIE vs EFIE: Interior Resonance Study, PEC Sphere")
    print("=" * 65)

    # ----------------------------------------------------------------
    # Geometry and mesh
    # ----------------------------------------------------------------
    radius = 0.1                 # meters
    target_edge_length = 0.025   # ~lambda/10 at 1.2 GHz; N_RWG ≈ 486

    sphere = Sphere(radius=radius)
    print(f"\nRadius:             {radius} m")
    print(f"Target edge length: {target_edge_length} m")

    print("\n--- Meshing ---")
    mesher = GmshMesher(target_edge_length=target_edge_length)
    mesh = mesher.mesh_from_geometry(sphere)
    basis = compute_rwg_connectivity(mesh)

    stats = mesh.get_statistics()
    print(f"Vertices:   {stats['num_vertices']}")
    print(f"Triangles:  {stats['num_triangles']}")
    print(f"RWG basis:  {basis.num_basis}")
    print(f"Mean edge:  {stats['mean_edge_length']:.4f} m")

    # ----------------------------------------------------------------
    # ka sweep: 1.0 → 3.5, ~22 points (dense near resonance at 2.74)
    # ----------------------------------------------------------------
    ka_coarse = np.linspace(1.0, 2.5, 10)
    ka_fine   = np.linspace(2.5, 3.0, 8)    # dense around resonance
    ka_tail   = np.linspace(3.0, 3.5, 6)[1:]
    ka_values = np.concatenate([ka_coarse, ka_fine, ka_tail])

    # Resonant ka: target near 2.74; will be updated to peak-κ after sweep
    ka_res = ka_values[np.argmin(np.abs(ka_values - 2.74))]
    f_res  = ka_res * c0 / (2.0 * np.pi * radius)

    print(f"\nka sweep:   {ka_values[0]:.2f} → {ka_values[-1]:.2f}  ({len(ka_values)} points)")
    print(f"Resonance:  ka={ka_res:.3f}, f={f_res/1e9:.3f} GHz")

    exc = PlaneWaveExcitation(
        E0=np.array([1.0, 0.0, 0.0]),
        k_hat=np.array([0.0, 0.0, -1.0]),
    )

    op_efie = EFIEOperator()
    op_cfie = CFIEOperator(alpha=0.5)

    rcs_efie = np.zeros(len(ka_values))
    rcs_cfie = np.zeros(len(ka_values))
    rcs_mie  = np.zeros(len(ka_values))
    cond_efie = np.zeros(len(ka_values))
    cond_cfie = np.zeros(len(ka_values))

    # ----------------------------------------------------------------
    # Frequency sweep
    # ----------------------------------------------------------------
    print("\n--- Frequency sweep ---")
    print(f"{'ka':>6}  {'f (GHz)':>8}  {'EFIE':>8}  {'CFIE':>8}  {'Mie':>8}  "
          f"{'κ(EFIE)':>10}  {'κ(CFIE)':>10}")
    print("-" * 72)

    for i, ka in enumerate(ka_values):
        freq = ka * c0 / (2.0 * np.pi * radius)
        k    = 2.0 * np.pi * freq / c0

        Z_efie = fill_matrix(op_efie, basis, mesh, k, eta0, backend='auto')
        Z_cfie = fill_matrix(op_cfie, basis, mesh, k, eta0, backend='auto')
        V_efie = exc.compute_voltage_vector(basis, mesh, k)
        V_mfie = exc.compute_mfie_voltage_vector(basis, mesh, k, eta0)
        alpha  = op_cfie.alpha
        V_cfie = alpha * V_efie + (1.0 - alpha) * eta0 * V_mfie

        I_efie = solve_direct(Z_efie, V_efie)
        I_cfie = solve_direct(Z_cfie, V_cfie)

        theta_back = np.array([0.001])
        phi_back   = np.array([0.0])

        E_th, E_ph = compute_far_field(I_efie, basis, mesh, k, eta0, theta_back, phi_back)
        rcs_efie[i] = compute_rcs(E_th, E_ph, E_inc_mag=1.0)[0]

        E_th, E_ph = compute_far_field(I_cfie, basis, mesh, k, eta0, theta_back, phi_back)
        rcs_cfie[i] = compute_rcs(E_th, E_ph, E_inc_mag=1.0)[0]

        mie_norm   = mie_monostatic_rcs_pec_sphere(ka)
        rcs_mie[i] = 10.0 * np.log10(max(mie_norm * np.pi * radius**2, 1e-30))

        cond_efie[i] = np.linalg.cond(Z_efie)
        cond_cfie[i] = np.linalg.cond(Z_cfie)

        print(f"{ka:6.3f}  {freq/1e9:8.4f}  {rcs_efie[i]:8.2f}  {rcs_cfie[i]:8.2f}  "
              f"{rcs_mie[i]:8.2f}  {cond_efie[i]:10.2e}  {cond_cfie[i]:10.2e}")

    # ----------------------------------------------------------------
    # Bistatic RCS at the ka where EFIE is most ill-conditioned
    # ----------------------------------------------------------------
    # Pick ka with maximum κ(EFIE) — this is where resonance suppression matters most
    ka_res = ka_values[np.argmax(cond_efie)]
    f_res  = ka_res * c0 / (2.0 * np.pi * radius)
    print(f"\n--- Bistatic RCS at ka={ka_res:.3f} (peak κ(EFIE)={cond_efie.max():.1f}), f={f_res/1e9:.3f} GHz ---")
    k_res = 2.0 * np.pi * f_res / c0

    Z_efie_res = fill_matrix(op_efie, basis, mesh, k_res, eta0, backend='auto')
    Z_cfie_res = fill_matrix(op_cfie, basis, mesh, k_res, eta0, backend='auto')
    V_efie_res = exc.compute_voltage_vector(basis, mesh, k_res)
    V_mfie_res = exc.compute_mfie_voltage_vector(basis, mesh, k_res, eta0)
    alpha      = op_cfie.alpha
    V_cfie_res = alpha * V_efie_res + (1.0 - alpha) * eta0 * V_mfie_res

    I_efie_res = solve_direct(Z_efie_res, V_efie_res)
    I_cfie_res = solve_direct(Z_cfie_res, V_cfie_res)

    theta_bi = np.linspace(0.001, np.pi - 0.001, 181)
    phi_bi   = np.zeros_like(theta_bi)

    E_th, E_ph = compute_far_field(I_efie_res, basis, mesh, k_res, eta0, theta_bi, phi_bi)
    bi_efie    = compute_rcs(E_th, E_ph, E_inc_mag=1.0)

    E_th, E_ph = compute_far_field(I_cfie_res, basis, mesh, k_res, eta0, theta_bi, phi_bi)
    bi_cfie    = compute_rcs(E_th, E_ph, E_inc_mag=1.0)

    mie_bi_norm = mie_rcs_pec_sphere(ka_res, np.pi - theta_bi)
    bi_mie      = 10.0 * np.log10(np.maximum(mie_bi_norm * np.pi * radius**2, 1e-30))

    print(f"κ(EFIE) at resonance: {np.linalg.cond(Z_efie_res):.2e}")
    print(f"κ(CFIE) at resonance: {np.linalg.cond(Z_cfie_res):.2e}")

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Monostatic RCS vs ka ---
    ax = axes[0]
    ax.plot(ka_values, rcs_mie,  'k-',  linewidth=2,   label='Mie (exact)')
    ax.plot(ka_values, rcs_cfie, 'b-o', linewidth=1.5, markersize=4,
            label=rf'CFIE ($\alpha=0.5$, $N={basis.num_basis}$)')
    ax.plot(ka_values, rcs_efie, 'r--s', linewidth=1.5, markersize=4,
            label=rf'EFIE ($N={basis.num_basis}$)')
    ax.axvline(x=ka_res, color='gray', linestyle=':', linewidth=1, alpha=0.7,
               label=rf'$ka={ka_res:.2f}$ (resonance)')
    ax.set_xlabel(r'$ka$')
    ax.set_ylabel(r'Monostatic RCS $\sigma$ (dBsm)')
    ax.set_title(rf'Monostatic RCS, PEC Sphere $a={radius*100:.0f}$ cm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Condition number vs ka ---
    ax = axes[1]
    ax.semilogy(ka_values, cond_cfie, 'b-o', linewidth=1.5, markersize=4,
                label=rf'CFIE ($\alpha=0.5$)')
    ax.semilogy(ka_values, cond_efie, 'r--s', linewidth=1.5, markersize=4,
                label='EFIE')
    ax.axvline(x=ka_res, color='gray', linestyle=':', linewidth=1, alpha=0.7,
               label=rf'$ka={ka_res:.2f}$')
    ax.set_xlabel(r'$ka$')
    ax.set_ylabel(r'Condition number $\kappa(Z)$')
    ax.set_title(r'Impedance Matrix Conditioning')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # --- Panel 3: Bistatic RCS at resonance ---
    ax = axes[2]
    theta_deg = np.degrees(theta_bi)
    ax.plot(theta_deg, bi_mie,  'k-',  linewidth=2,   label='Mie (exact)')
    ax.plot(theta_deg, bi_cfie, 'b--', linewidth=1.5, label=rf'CFIE ($\alpha=0.5$)')
    ax.plot(theta_deg, bi_efie, 'r:',  linewidth=1.5, label='EFIE')
    ax.set_xlabel(r'Bistatic angle $\theta$ (deg)')
    ax.set_ylabel(r'Bistatic RCS $\sigma$ (dBsm)')
    ax.set_title(rf'Bistatic RCS at $ka={ka_res:.2f}$ ($f={f_res/1e9:.3f}$ GHz)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 180])

    fig.suptitle(
        rf'CFIE vs EFIE: Interior Resonance Suppression, PEC Sphere $a={radius*100:.0f}$ cm',
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    output_file = os.path.join(images_dir, 'sphere_cfie_validation.png')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")

    plt.show()

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)


if __name__ == '__main__':
    main()
