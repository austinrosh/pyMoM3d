"""
Example: 8-element linear dipole array with beam steering.

Validates pyMoM3d's antenna array support using a canonical
8-element half-wave dipole array at 5 GHz. Demonstrates broadside
and steered beam patterns with full mutual coupling via MoM.

Scenario:
  - N = 8 z-directed half-wave strip dipoles arrayed along x
  - Frequency: 5 GHz (lambda = 60 mm)
  - Spacing: d = 0.5 lambda = 30 mm
  - Strip width: 2 mm
  - Three beam configurations: broadside, 30 deg, 45 deg off broadside

  Coordinate convention:
  - Dipoles along z, array along x
  - Element pattern peaks at theta=90 (equatorial plane), null at theta=0,180
  - Array factor modulates the phi pattern at theta=90
  - Broadside = +y direction (theta=90, phi=90)
  - Pattern cut showing array factor: phi sweep at theta=90

Expected results:
  - Broadside D_max within 1.5 dB of 10*log10(N * D_element) ~ 11.2 dBi
  - HPBW within 30% of theoretical 12.8 deg
  - MoM pattern agrees with AF x element pattern within 1 dB at main lobe
  - Steered beams peak within 5 deg of commanded scan angle
  - Symmetric element currents for uniform excitation

Produces:
  Figure 1: Array layout (x-z plane)
  Figure 2: Broadside array-plane pattern (rectangular + polar)
  Figure 3: Steered beam patterns (broadside + 30 + 45 deg)
  Figure 4: Element currents and active impedances
  Figure 5: 3D surface current on all 8 elements
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from pyMoM3d import (
    LinearDipoleArray,
    uniform_excitation,
    progressive_phase_excitation,
    compute_far_field,
    plot_surface_current,
    plot_array_layout,
    configure_latex_style,
    c0,
    eta0,
)
from pyMoM3d.analysis.pattern_analysis import compute_directivity, compute_beamwidth_3dB

configure_latex_style()


def main():
    print("=" * 65)
    print("8-Element Linear Dipole Array Validation")
    print("=" * 65)

    # --- Physical parameters ---
    freq = 5.0e9
    lam = c0 / freq
    k = 2.0 * np.pi * freq / c0
    N = 8
    d = 0.5 * lam          # spacing = lambda/2
    L = lam / 2             # dipole length
    w = 2e-3                # strip width
    mesh_edge = lam / 15    # ~4 mm

    print(f"\nFrequency:      {freq/1e9:.1f} GHz")
    print(f"Wavelength:     {lam*1e3:.1f} mm")
    print(f"Elements:       {N}")
    print(f"Spacing:        {d*1e3:.1f} mm (0.5 lambda)")
    print(f"Dipole length:  {L*1e3:.1f} mm (lambda/2)")
    print(f"Strip width:    {w*1e3:.1f} mm")
    print(f"Mesh edge:      {mesh_edge*1e3:.1f} mm (lambda/15)")

    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    # ===================================================================
    # Part 0: Build array and fill Z-matrix
    # ===================================================================
    print("\n" + "-" * 65)
    print("Part 0: Building array and filling Z-matrix")
    print("-" * 65)

    array = LinearDipoleArray(
        n_elements=N,
        spacing=d,
        frequency=freq,
        dipole_length=L,
        strip_width=w,
        dipole_axis='z',
        array_axis='x',
        mesh_edge_length=mesh_edge,
    )

    stats = array.mesh.get_statistics()
    print(f"  Triangles: {stats['num_triangles']}, "
          f"Vertices: {stats['num_vertices']}, "
          f"Basis: {array.basis.num_basis}")

    for n in range(N):
        print(f"  Element {n}: {len(array.element_feed_indices[n])} feed edges, "
              f"pos = ({array.element_positions[n][0]*1e3:.1f}, "
              f"{array.element_positions[n][1]*1e3:.1f}, "
              f"{array.element_positions[n][2]*1e3:.1f}) mm")

    print("\n  Filling Z-matrix...")
    Z = array.fill_impedance_matrix()
    print(f"  Z-matrix shape: {Z.shape}")

    # ===================================================================
    # Part 1: Broadside pattern (uniform excitation)
    # ===================================================================
    print("\n" + "-" * 65)
    print("Part 1: Broadside Pattern (uniform excitation)")
    print("-" * 65)

    weights_bs = uniform_excitation(N, voltage=1.0)
    I_bs = array.solve(weights_bs)

    # Array-plane cut: theta=90 (equatorial plane), sweep phi
    # For z-dipoles arrayed along x, the array factor modulates the phi pattern
    # Broadside = phi=90 (+y direction)
    phi_cut = np.linspace(0.0, 2 * np.pi - 0.001, 721)
    theta_eq = np.full_like(phi_cut, np.pi / 2)

    E_th_arr, E_ph_arr = array.compute_far_field(I_bs, theta_eq, phi_cut)
    gain_arr = np.abs(E_th_arr)**2 + np.abs(E_ph_arr)**2

    # Directivity
    D, D_max, D_max_dBi = array.compute_directivity(I_bs)
    print(f"  D_max = {D_max:.3f} ({D_max_dBi:.2f} dBi)")

    # Normalize array-plane pattern
    gain_arr_max = gain_arr.max()
    if gain_arr_max > 0:
        gain_arr_norm_dB = 10.0 * np.log10(np.maximum(gain_arr / gain_arr_max, 1e-30))
    else:
        gain_arr_norm_dB = np.full_like(gain_arr, -300.0)

    # HPBW in array plane (around phi=90 broadside)
    gain_arr_linear = np.maximum(gain_arr / gain_arr_max, 1e-30) * D_max
    # Extract the half centered on phi=90 (0 to pi)
    half_mask = phi_cut <= np.pi
    hpbw = compute_beamwidth_3dB(gain_arr_linear[half_mask], phi_cut[half_mask])
    print(f"  HPBW (array plane): {hpbw:.1f} deg")

    # Analytical AF x element pattern for comparison
    AF_bs = array.compute_array_factor(theta_eq, phi_cut, weights_bs)

    # Element pattern at theta=90 is ~constant (sin(90)=1), so AF alone gives shape
    # But the element pattern has no phi variation for a z-dipole, so
    # the overall pattern = AF * sin(theta) and at theta=90: pattern ~ |AF|
    af_mag = np.abs(AF_bs)
    af_max = af_mag.max() if af_mag.max() > 0 else 1.0
    af_norm_dB = 20.0 * np.log10(np.maximum(af_mag / af_max, 1e-30))

    # Element currents
    currents_bs = array.compute_element_currents(I_bs)
    print("\n  Element terminal currents (broadside):")
    for n, I_term in enumerate(currents_bs):
        print(f"    Element {n}: |I| = {np.abs(I_term):.4f} A, "
              f"phase = {np.degrees(np.angle(I_term)):.1f} deg")

    # ===================================================================
    # Part 2: Steered beams
    # ===================================================================
    print("\n" + "-" * 65)
    print("Part 2: Steered Beams")
    print("-" * 65)

    scan_angles_deg = [30.0, 45.0]
    steered_data = []

    for scan_deg in scan_angles_deg:
        # Scan angle from broadside (phi=90) toward endfire (phi=0)
        # Target phi = 90 - scan_deg (in the first quadrant)
        target_phi = np.radians(90.0 - scan_deg)

        # Phase shift: beta = -k*d*sin(theta)*cos(phi) at theta=90
        # beta = -k*d*cos(target_phi) = -k*d*sin(scan_deg_rad)
        beta = -k * d * np.sin(np.radians(scan_deg))

        weights_scan = progressive_phase_excitation(N, beta, voltage=1.0)
        I_scan = array.solve(weights_scan)

        E_th_scan, E_ph_scan = array.compute_far_field(I_scan, theta_eq, phi_cut)
        gain_scan = np.abs(E_th_scan)**2 + np.abs(E_ph_scan)**2
        gain_scan_max = gain_scan.max()

        # Find peak phi (look only in first half 0 to pi for the main lobe)
        gain_first_half = gain_scan[half_mask]
        phi_first_half = phi_cut[half_mask]
        peak_idx = np.argmax(gain_first_half)
        peak_phi_deg = np.degrees(phi_first_half[peak_idx])
        expected_phi = 90.0 - scan_deg
        peak_error = abs(peak_phi_deg - expected_phi)

        print(f"\n  Scan {scan_deg} deg from broadside:")
        print(f"    Phase shift: {np.degrees(beta):.1f} deg/element")
        print(f"    Peak at phi = {peak_phi_deg:.1f} deg "
              f"(expected {expected_phi:.1f} deg, error {peak_error:.1f} deg)")

        # AF for comparison
        AF_scan = array.compute_array_factor(theta_eq, phi_cut, weights_scan)
        af_scan_mag = np.abs(AF_scan)
        af_scan_max = af_scan_mag.max() if af_scan_mag.max() > 0 else 1.0

        steered_data.append({
            'scan_deg': scan_deg,
            'gain': gain_scan,
            'gain_max': gain_scan_max,
            'peak_phi': peak_phi_deg,
            'expected_phi': expected_phi,
            'peak_error': peak_error,
            'beta': beta,
            'weights': weights_scan,
            'I': I_scan,
            'af_mag': af_scan_mag,
            'af_max': af_scan_max,
        })

    # ===================================================================
    # Part 3: Element current analysis
    # ===================================================================
    print("\n" + "-" * 65)
    print("Part 3: Element Current Analysis")
    print("-" * 65)

    # Active impedances (broadside)
    Z_active = array.compute_element_impedances(I_bs, weights_bs)
    print("\n  Active impedances (broadside):")
    for n, Z_in in enumerate(Z_active):
        print(f"    Element {n}: Z_in = {Z_in.real:.2f} + j{Z_in.imag:.2f} Ohm")

    # ===================================================================
    # Part 4: Validation summary
    # ===================================================================
    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    # Check 1: Broadside D_max
    expected_D_dBi = 10.0 * np.log10(N * 1.64)  # ~11.17 dBi
    d_err = abs(D_max_dBi - expected_D_dBi)
    d_pass = d_err < 1.5
    print(f"\n  Broadside D_max: {D_max_dBi:.2f} dBi "
          f"(expected ~{expected_D_dBi:.1f} dBi, error {d_err:.2f} dB) "
          f"{'PASS' if d_pass else 'FAIL'}")

    # Check 2: HPBW
    hpbw_expected = 12.8
    hpbw_err = abs(hpbw - hpbw_expected) / hpbw_expected
    hpbw_pass = hpbw_err < 0.30
    print(f"  HPBW: {hpbw:.1f} deg "
          f"(expected ~{hpbw_expected:.1f} deg, error {hpbw_err*100:.1f}%) "
          f"{'PASS' if hpbw_pass else 'FAIL'}")

    # Check 3: AF agreement at main lobe
    # At broadside peak (phi=90), MoM and AF should agree
    idx_bs = np.argmin(np.abs(phi_cut - np.pi / 2))
    mom_at_bs_dB = gain_arr_norm_dB[idx_bs]
    af_at_bs_dB = af_norm_dB[idx_bs]
    af_err = abs(mom_at_bs_dB - af_at_bs_dB)
    af_pass = af_err < 1.0
    print(f"  AF agreement at broadside: error = {af_err:.2f} dB "
          f"{'PASS' if af_pass else 'FAIL'}")

    # Check 4: Beam steering accuracy
    steer_pass = True
    for sd in steered_data:
        if sd['peak_error'] > 5.0:
            steer_pass = False
        print(f"  Beam steer {sd['scan_deg']} deg: peak error = {sd['peak_error']:.1f} deg "
              f"{'PASS' if sd['peak_error'] <= 5.0 else 'FAIL'}")

    # Check 5: Element symmetry (uniform excitation → symmetric currents)
    I_mags = [np.abs(c) for c in currents_bs]
    # For uniform excitation, the array is symmetric: element n ↔ element N-1-n
    sym_pass = True
    max_sym_err = 0.0
    for n in range(N // 2):
        err = abs(I_mags[n] - I_mags[N - 1 - n]) / max(I_mags[n], 1e-30)
        max_sym_err = max(max_sym_err, err)
        if err > 0.05:
            sym_pass = False
    print(f"  Element symmetry: max asymmetry = {max_sym_err*100:.1f}% "
          f"{'PASS' if sym_pass else 'FAIL'}")

    all_pass = d_pass and hpbw_pass and af_pass and steer_pass and sym_pass
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    # ===================================================================
    # Part 5: Plots
    # ===================================================================

    phi_deg = np.degrees(phi_cut)

    # --- Figure 1: Array layout ---
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    plot_array_layout(
        array.element_positions,
        dipole_length=L,
        strip_width=w,
        dipole_axis=np.array([0, 0, 1]),
        array_axis=np.array([1, 0, 0]),
        ax=ax1,
        title=rf'8-Element Dipole Array, $f = {freq/1e9:.1f}$ GHz, '
              rf'$d = \lambda/2 = {d*1e3:.1f}$ mm',
    )
    fig1.tight_layout()
    out1 = os.path.join(images_dir, 'array_layout.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out1}")

    # --- Figure 2: Broadside array-plane pattern ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5),
                                       subplot_kw={'projection': None})

    # Left: rectangular plot (phi cut at theta=90)
    ax2a.plot(phi_deg, gain_arr_norm_dB, 'b-', linewidth=1.5, label='MoM')
    ax2a.plot(phi_deg, af_norm_dB, 'r--', linewidth=1.5, label=r'AF')
    ax2a.set_xlabel(r'$\phi$ (deg)')
    ax2a.set_ylabel(r'Normalized pattern (dB)')
    ax2a.set_title(rf'Array plane ($\theta = 90^\circ$), broadside')
    ax2a.annotate(rf'$D_{{\max}} = {D_max_dBi:.1f}$ dBi'
                  rf', HPBW $= {hpbw:.1f}^\circ$',
                  xy=(0.02, 0.02), xycoords='axes fraction',
                  fontsize=9, ha='left', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))
    ax2a.legend(fontsize=9)
    ax2a.grid(True, alpha=0.3)
    ax2a.set_xlim([0, 360])
    ax2a.set_ylim([-40, 3])

    # Right: polar plot
    ax2b.set_visible(False)
    ax2b_polar = fig2.add_subplot(122, polar=True)
    gain_norm = np.maximum(gain_arr / gain_arr_max, 0)
    ax2b_polar.plot(phi_cut, gain_norm, 'b-', linewidth=1.5)
    ax2b_polar.set_theta_zero_location('E')
    ax2b_polar.set_title(rf'Array plane polar, $D_{{\max}} = {D_max_dBi:.1f}$ dBi',
                         pad=15)

    fig2.suptitle(rf'Broadside Pattern, $N = {N}$, $d = \lambda/2$', fontsize=13)
    fig2.tight_layout()
    out2 = os.path.join(images_dir, 'array_pattern_broadside.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")

    # --- Figure 3: Steered beams ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MoM patterns
    ax3a.plot(phi_deg, gain_arr_norm_dB, 'b-', linewidth=1.5,
              label='Broadside')
    colors_steer = ['green', 'orange']
    for i, sd in enumerate(steered_data):
        g = sd['gain']
        g_dB = 10.0 * np.log10(np.maximum(g / sd['gain_max'], 1e-30))
        ax3a.plot(phi_deg, g_dB, '-', color=colors_steer[i], linewidth=1.5,
                  label=rf'Steered ${sd["scan_deg"]:.0f}^\circ$')
    ax3a.set_xlabel(r'$\phi$ (deg)')
    ax3a.set_ylabel(r'Normalized pattern (dB)')
    ax3a.set_title(r'MoM patterns ($\theta = 90^\circ$)')
    ax3a.legend(fontsize=9)
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xlim([0, 360])
    ax3a.set_ylim([-40, 3])

    # Right: MoM vs AF
    ax3b.plot(phi_deg, gain_arr_norm_dB, 'b-', linewidth=1.5,
              label='Broadside (MoM)')
    ax3b.plot(phi_deg, af_norm_dB, 'b--', linewidth=1, alpha=0.7,
              label='Broadside (AF)')
    for i, sd in enumerate(steered_data):
        g = sd['gain']
        g_dB = 10.0 * np.log10(np.maximum(g / sd['gain_max'], 1e-30))
        af_dB = 20.0 * np.log10(
            np.maximum(sd['af_mag'] / sd['af_max'], 1e-30))
        ax3b.plot(phi_deg, g_dB, '-', color=colors_steer[i], linewidth=1.5,
                  label=rf'${sd["scan_deg"]:.0f}^\circ$ (MoM)')
        ax3b.plot(phi_deg, af_dB, '--', color=colors_steer[i], linewidth=1,
                  alpha=0.7, label=rf'${sd["scan_deg"]:.0f}^\circ$ (AF)')
    ax3b.set_xlabel(r'$\phi$ (deg)')
    ax3b.set_ylabel(r'Normalized pattern (dB)')
    ax3b.set_title(r'MoM vs AF ($\theta = 90^\circ$)')
    ax3b.legend(fontsize=8, ncol=2)
    ax3b.grid(True, alpha=0.3)
    ax3b.set_xlim([0, 360])
    ax3b.set_ylim([-40, 3])

    fig3.suptitle(rf'Steered Beams, $N = {N}$, $d = \lambda/2$', fontsize=13)
    fig3.tight_layout()
    out3 = os.path.join(images_dir, 'array_pattern_steered.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out3}")

    # --- Figure 4: Element currents and active impedance ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: |I_terminal| per element for all excitations
    x_elem = np.arange(N)
    bar_width = 0.25
    I_mags_bs = [np.abs(c) for c in currents_bs]
    ax4a.bar(x_elem - bar_width, I_mags_bs, bar_width, label='Broadside',
             color='steelblue')

    for i, sd in enumerate(steered_data):
        currents_scan = array.compute_element_currents(sd['I'])
        I_mags_scan = [np.abs(c) for c in currents_scan]
        ax4a.bar(x_elem + i * bar_width, I_mags_scan, bar_width,
                 label=rf'${sd["scan_deg"]:.0f}^\circ$',
                 color=colors_steer[i])

    ax4a.set_xlabel(r'Element index')
    ax4a.set_ylabel(r'$|I_{\mathrm{terminal}}|$ (A)')
    ax4a.set_title(r'Terminal current per element')
    ax4a.set_xticks(x_elem)
    ax4a.legend(fontsize=9)
    ax4a.grid(True, alpha=0.3, axis='y')

    # Right: Active impedance (broadside)
    R_in = [Z_in.real for Z_in in Z_active]
    X_in = [Z_in.imag for Z_in in Z_active]
    ax4b.bar(x_elem - 0.15, R_in, 0.3, label=r'$R_{\mathrm{in}}$',
             color='steelblue')
    ax4b.bar(x_elem + 0.15, X_in, 0.3, label=r'$X_{\mathrm{in}}$',
             color='coral')
    ax4b.set_xlabel(r'Element index')
    ax4b.set_ylabel(r'Impedance ($\Omega$)')
    ax4b.set_title(r'Active impedance (broadside)')
    ax4b.set_xticks(x_elem)
    ax4b.legend(fontsize=9)
    ax4b.grid(True, alpha=0.3, axis='y')
    ax4b.axhline(0, color='black', linewidth=0.5)

    fig4.suptitle(rf'Element Analysis, $N = {N}$, $d = \lambda/2$', fontsize=13)
    fig4.tight_layout()
    out4 = os.path.join(images_dir, 'array_element_currents.png')
    fig4.savefig(out4, dpi=150, bbox_inches='tight')
    print(f"Saved: {out4}")

    # --- Figure 5: Surface current ---
    fig5 = plt.figure(figsize=(14, 6))
    ax5 = fig5.add_subplot(111, projection='3d')
    plot_surface_current(I_bs, array.basis, array.mesh, ax=ax5, cmap='hot',
                         edge_color='gray', edge_width=0.2,
                         title=(rf'$|\mathbf{{J}}|$ on {N}-element array, '
                                rf'broadside, $f = {freq/1e9:.1f}$ GHz'))
    ax5.view_init(elev=25, azim=-60)
    out5 = os.path.join(images_dir, 'array_surface_current.png')
    fig5.savefig(out5, dpi=150, bbox_inches='tight')
    print(f"Saved: {out5}")

    plt.show()

    print("\n" + "=" * 65)
    print("Dipole array validation complete!")
    print("=" * 65)


if __name__ == '__main__':
    main()
