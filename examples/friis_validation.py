"""
Example: Friis transmission equation validation.

Validates pyMoM3d's ability to predict received power between two
half-wave dipole antennas in free space. Compares MoM-simulated
received power against the analytical Friis transmission equation,
sweeping both separation distance and polarization alignment.

Scenario:
  - TX and RX are identical lambda/2 strip dipoles at 5 GHz
  - Link direction: +z (broadside to both dipoles)
  - Distance sweep: 5-15 lambda at co-polarization
  - Polarization sweep: 0-90 deg at fixed R = 10 lambda

Expected results:
  - Distance sweep: MoM matches Friis within 1.0 dB
  - Polarization sweep: MoM matches Friis * cos^2(psi) within 1.5 dB
  - Broadside gain: ~2.15 dBi (half-wave dipole)
  - Cross-pol (psi=90): received power < -60 dB below co-pol

Produces:
  Figure 1: Dipole characterization (radiation pattern, parameters, polar)
  Figure 2: Distance sweep P_rx/P_tx vs R/lambda
  Figure 3: Polarization sweep P_rx/P_tx vs psi
  Figure 4: Surface current on TX+RX pair
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from pyMoM3d import (
    GmshMesher,
    compute_rwg_connectivity,
    fill_matrix,
    EFIEOperator,
    solve_direct,
    compute_far_field,
    plot_surface_current,
    configure_latex_style,
    c0,
    eta0,
)
from pyMoM3d.mom.excitation import StripDeltaGapExcitation, find_feed_edges
from pyMoM3d.analysis.pattern_analysis import compute_directivity
from pyMoM3d.mesh.mesh_data import Mesh

# Configure LaTeX-style plotting
configure_latex_style()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_two_antenna_mesh(mesh_tx, mesh_rx):
    """Combine two disjoint antenna meshes into one.

    Parameters
    ----------
    mesh_tx, mesh_rx : Mesh
        Individual antenna meshes (may overlap in vertex indices).

    Returns
    -------
    combined : Mesh
        Single mesh containing both antennas.
    n_verts_tx : int
        Number of vertices from the TX mesh (offset for RX indices).
    """
    n_verts_tx = len(mesh_tx.vertices)
    combined_verts = np.vstack([mesh_tx.vertices, mesh_rx.vertices])
    combined_tris = np.vstack([
        mesh_tx.triangles,
        mesh_rx.triangles + n_verts_tx,
    ])
    return Mesh(combined_verts, combined_tris), n_verts_tx


def find_feed_edges_at_z(mesh, basis, feed_x, z_center, tol_z=None):
    """Find feed edges at a specific z-plane.

    Wraps find_feed_edges with z-coordinate filtering to disambiguate
    TX and RX feeds when both have feed_x=0.

    Parameters
    ----------
    mesh : Mesh
    basis : RWGBasis
    feed_x : float
        x-coordinate of the feed line.
    z_center : float
        z-coordinate of the antenna center.
    tol_z : float, optional
        z-tolerance. Defaults to 10% of min edge length.

    Returns
    -------
    indices : list of int
        Basis function indices for feed edges at the specified z.
    """
    all_feed = find_feed_edges(mesh, basis, feed_x)

    if tol_z is None:
        lengths = []
        for n in all_feed[:50]:
            e = mesh.edges[basis.edge_index[n]]
            lengths.append(np.linalg.norm(
                mesh.vertices[e[1]] - mesh.vertices[e[0]]))
        tol_z = 0.1 * min(lengths) if lengths else 1e-3

    filtered = []
    for n in all_feed:
        e = mesh.edges[basis.edge_index[n]]
        mid_z = 0.5 * (mesh.vertices[e[0]][2] + mesh.vertices[e[1]][2])
        if abs(mid_z - z_center) <= tol_z:
            filtered.append(n)
    return filtered


def find_rotated_feed_edges(mesh, basis, z_center, psi, tol_z=None, tol_pos=None):
    """Find feed edges on a rotated dipole at a specific z-plane.

    For a dipole rotated by angle psi about z, the feed edges are no
    longer y-directed. This function checks edge direction alignment
    with the rotated transverse direction.

    Parameters
    ----------
    mesh : Mesh
    basis : RWGBasis
    z_center : float
        z-coordinate of the antenna center.
    psi : float
        Rotation angle about z-axis (radians).
    tol_z : float, optional
    tol_pos : float, optional
        Positional tolerance for midpoint proximity to feed center.

    Returns
    -------
    indices : list of int
    """
    # Transverse direction for rotated dipole
    transverse = np.array([-np.sin(psi), np.cos(psi), 0.0])
    # Dipole axis direction
    dipole_axis = np.array([np.cos(psi), np.sin(psi), 0.0])

    if tol_z is None or tol_pos is None:
        lengths = []
        for n in range(min(basis.num_basis, 50)):
            e = mesh.edges[basis.edge_index[n]]
            lengths.append(np.linalg.norm(
                mesh.vertices[e[1]] - mesh.vertices[e[0]]))
        min_len = min(lengths) if lengths else 1e-3
        if tol_z is None:
            tol_z = 0.1 * min_len
        if tol_pos is None:
            tol_pos = 0.5 * min_len

    indices = []
    for n in range(basis.num_basis):
        e = mesh.edges[basis.edge_index[n]]
        va = mesh.vertices[e[0]]
        vb = mesh.vertices[e[1]]
        mid = 0.5 * (va + vb)

        # Check z-plane
        if abs(mid[2] - z_center) > tol_z:
            continue

        # Check midpoint is near feed center (projection onto dipole axis ~ 0)
        r_mid = mid - np.array([0.0, 0.0, z_center])
        proj = abs(np.dot(r_mid, dipole_axis))
        if proj > tol_pos:
            continue

        # Check edge direction alignment with transverse
        edge_dir = vb - va
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-30:
            continue
        edge_dir /= edge_len
        if abs(np.dot(edge_dir, transverse)) > abs(np.dot(edge_dir, dipole_axis)):
            indices.append(n)

    return indices


def create_rotated_dipole_mesh(mesher, L, w, center, psi):
    """Create a dipole mesh rotated by psi about z, then translated.

    Parameters
    ----------
    mesher : GmshMesher
    L : float
        Dipole length (along x before rotation).
    w : float
        Strip width (along y before rotation).
    center : tuple
        (x, y, z) center after translation.
    psi : float
        Rotation angle about z-axis (radians).

    Returns
    -------
    mesh : Mesh
    """
    mesh = mesher.mesh_plate_with_feed(
        width=L, height=w, feed_x=0.0, center=(0, 0, 0))

    # Rotation matrix about z
    c, s = np.cos(psi), np.sin(psi)
    R_z = np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

    rotated_verts = (R_z @ mesh.vertices.T).T
    rotated_verts += np.array(center)

    return Mesh(rotated_verts, mesh.triangles.copy())


def compute_terminal_current(I_coeffs, basis, feed_indices):
    """Compute terminal current from RWG coefficients at feed edges.

    I_terminal = sum_n I_n * l_n  for n in feed_indices.

    Parameters
    ----------
    I_coeffs : ndarray, shape (N,)
    basis : RWGBasis
    feed_indices : list of int

    Returns
    -------
    I_terminal : complex
    """
    I_terminal = 0.0 + 0.0j
    for idx in feed_indices:
        I_terminal += I_coeffs[idx] * basis.edge_length[idx]
    return I_terminal


def compute_broadside_gain(I, basis, mesh, k, eta):
    """Compute broadside gain and peak directivity.

    Evaluates far-field on a (theta, phi) grid and computes directivity.
    For an x-directed dipole, broadside is theta=0 (+z direction).

    Parameters
    ----------
    I : ndarray, shape (N,)
    basis : RWGBasis
    mesh : Mesh
    k : float
    eta : float

    Returns
    -------
    G_broadside : float
        Broadside directivity (linear).
    G_broadside_dBi : float
    D_max : float
        Peak directivity (linear).
    D_max_dBi : float
    D : ndarray
        Full directivity pattern.
    theta_grid : ndarray
    phi_grid : ndarray
    """
    n_th = 91
    n_ph = 72
    theta_grid = np.linspace(0.001, np.pi - 0.001, n_th)
    phi_grid = np.linspace(0.0, 2.0 * np.pi - 2.0 * np.pi / n_ph, n_ph)

    E_th_2d = np.zeros((n_th, n_ph), dtype=np.complex128)
    E_ph_2d = np.zeros((n_th, n_ph), dtype=np.complex128)
    for j in range(n_ph):
        E_th_2d[:, j], E_ph_2d[:, j] = compute_far_field(
            I, basis, mesh, k, eta,
            theta_grid, np.full_like(theta_grid, phi_grid[j]))

    D, D_max, D_max_dBi = compute_directivity(E_th_2d, E_ph_2d,
                                                theta_grid, phi_grid, eta)

    # Broadside gain: D at theta~0, average over phi (azimuthally symmetric)
    G_broadside = float(np.mean(D[0, :]))
    G_broadside_dBi = 10.0 * np.log10(max(G_broadside, 1e-30))

    return G_broadside, G_broadside_dBi, D_max, D_max_dBi, D, theta_grid, phi_grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Friis Transmission Equation Validation")
    print("=" * 65)

    # --- Physical parameters ---
    freq = 5.0e9
    lam = c0 / freq
    k = 2.0 * np.pi * freq / c0
    L = lam / 2           # dipole length (30 mm)
    w = 2e-3              # strip width (2 mm)
    mesh_edge = lam / 15  # ~4 mm

    print(f"\nFrequency:      {freq/1e9:.1f} GHz")
    print(f"Wavelength:     {lam*1e3:.1f} mm")
    print(f"Dipole length:  {L*1e3:.1f} mm (lambda/2)")
    print(f"Strip width:    {w*1e3:.1f} mm")
    print(f"Mesh edge:      {mesh_edge*1e3:.1f} mm (lambda/15)")

    mesher = GmshMesher(target_edge_length=mesh_edge)
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    # ===================================================================
    # PART 1: Isolated dipole characterization
    # ===================================================================
    print("\n" + "-" * 65)
    print("Part 1: Isolated Dipole Characterization")
    print("-" * 65)

    mesh_single = mesher.mesh_plate_with_feed(
        width=L, height=w, feed_x=0.0, center=(0, 0, 0))
    basis_single = compute_rwg_connectivity(mesh_single)
    stats = mesh_single.get_statistics()
    print(f"  Triangles: {stats['num_triangles']}, "
          f"Vertices: {stats['num_vertices']}, "
          f"Basis: {basis_single.num_basis}")

    feed_single = find_feed_edges(mesh_single, basis_single, feed_x=0.0)
    print(f"  Feed edges: {len(feed_single)}")

    exc_single = StripDeltaGapExcitation(feed_basis_indices=feed_single, voltage=1.0)
    V_single = exc_single.compute_voltage_vector(basis_single, mesh_single, k)
    Z_mat = fill_matrix(EFIEOperator(),basis_single, mesh_single, k, eta0)
    I_single = solve_direct(Z_mat, V_single)

    Z_self = exc_single.compute_input_impedance(I_single, basis_single, mesh_single)
    print(f"  Z_self = {Z_self.real:.2f} + j{Z_self.imag:.2f} Ohm")

    # Compute broadside gain
    (G_bs, G_bs_dBi, D_max, D_max_dBi,
     D_pattern, theta_grid, phi_grid) = compute_broadside_gain(
        I_single, basis_single, mesh_single, k, eta0)

    print(f"  D_max = {D_max:.3f} ({D_max_dBi:.2f} dBi)")
    print(f"  G_broadside = {G_bs:.3f} ({G_bs_dBi:.2f} dBi)")

    # Radiation pattern cuts for plotting
    theta_cut = np.linspace(0.001, np.pi - 0.001, 181)

    # E-plane (xz-plane, phi=0)
    E_th_e, E_ph_e = compute_far_field(I_single, basis_single, mesh_single,
                                        k, eta0, theta_cut, np.zeros_like(theta_cut))
    gain_e = np.abs(E_th_e)**2 + np.abs(E_ph_e)**2

    # H-plane (yz-plane, phi=pi/2)
    E_th_h, E_ph_h = compute_far_field(I_single, basis_single, mesh_single,
                                        k, eta0, theta_cut,
                                        np.full_like(theta_cut, np.pi / 2))
    gain_h = np.abs(E_th_h)**2 + np.abs(E_ph_h)**2

    # Normalize to directivity
    gain_max = max(gain_e.max(), gain_h.max())
    gain_e_dBi = 10.0 * np.log10(np.maximum(gain_e / gain_max, 1e-30)) + D_max_dBi
    gain_h_dBi = 10.0 * np.log10(np.maximum(gain_h / gain_max, 1e-30)) + D_max_dBi

    # ===================================================================
    # PART 2: Distance sweep (co-polarized)
    # ===================================================================
    print("\n" + "-" * 65)
    print("Part 2: Distance Sweep (co-polarized)")
    print("-" * 65)

    R_over_lam = np.linspace(5, 15, 8)
    R_vals = R_over_lam * lam

    Prx_Ptx_mom = np.zeros(len(R_vals))
    Prx_Ptx_friis = np.zeros(len(R_vals))

    print(f"\n  {'R/lam':>8s}  {'R (m)':>8s}  {'MoM (dB)':>10s}  "
          f"{'Friis (dB)':>10s}  {'Error (dB)':>10s}")
    print("  " + "-" * 55)

    for i, R in enumerate(R_vals):
        # TX at origin, RX at (0, 0, R)
        mesh_tx = mesher.mesh_plate_with_feed(
            width=L, height=w, feed_x=0.0, center=(0, 0, 0))
        mesh_rx = mesher.mesh_plate_with_feed(
            width=L, height=w, feed_x=0.0, center=(0, 0, R))

        combined, n_verts_tx = build_two_antenna_mesh(mesh_tx, mesh_rx)
        basis_comb = compute_rwg_connectivity(combined)

        # Find feed edges for TX (z~0) and RX (z~R)
        feed_tx = find_feed_edges_at_z(combined, basis_comb, 0.0, z_center=0.0)
        feed_rx = find_feed_edges_at_z(combined, basis_comb, 0.0, z_center=R)

        # Excite TX only (RX short-circuited: V=0)
        exc_tx = StripDeltaGapExcitation(feed_basis_indices=feed_tx, voltage=1.0)
        V_comb = exc_tx.compute_voltage_vector(basis_comb, combined, k)
        Z_comb = fill_matrix(EFIEOperator(),basis_comb, combined, k, eta0)
        I_comb = solve_direct(Z_comb, V_comb)

        # TX input power: P_tx = 0.5 * Re(V_0 * conj(I_tx_terminal))
        I_tx_term = compute_terminal_current(I_comb, basis_comb, feed_tx)
        P_tx = 0.5 * np.real(1.0 * np.conj(I_tx_term))

        # RX short-circuit current
        I_rx_sc = compute_terminal_current(I_comb, basis_comb, feed_rx)

        # RX open-circuit voltage and power to conjugate-matched load
        V_oc = -Z_self * I_rx_sc
        P_rx = np.abs(V_oc)**2 / (8.0 * np.real(Z_self))

        ratio = P_rx / P_tx if P_tx > 0 else 0.0
        Prx_Ptx_mom[i] = ratio

        # Friis prediction: G_bs^2 * (lam / 4*pi*R)^2
        friis = G_bs**2 * (lam / (4.0 * np.pi * R))**2
        Prx_Ptx_friis[i] = friis

        mom_dB = 10.0 * np.log10(max(ratio, 1e-30))
        friis_dB = 10.0 * np.log10(max(friis, 1e-30))
        err_dB = abs(mom_dB - friis_dB)

        print(f"  {R/lam:8.1f}  {R:8.4f}  {mom_dB:10.2f}  {friis_dB:10.2f}  {err_dB:10.2f}")

    Prx_Ptx_mom_dB = 10.0 * np.log10(np.maximum(Prx_Ptx_mom, 1e-30))
    Prx_Ptx_friis_dB = 10.0 * np.log10(np.maximum(Prx_Ptx_friis, 1e-30))
    dist_error_dB = np.abs(Prx_Ptx_mom_dB - Prx_Ptx_friis_dB)

    # ===================================================================
    # PART 3: Polarization sweep
    # ===================================================================
    print("\n" + "-" * 65)
    print("Part 3: Polarization Sweep (R = 10 lambda)")
    print("-" * 65)

    R_pol = 10.0 * lam
    psi_deg = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    psi_rad = np.radians(psi_deg)

    Prx_Ptx_pol_mom = np.zeros(len(psi_deg))
    Prx_Ptx_pol_friis = np.zeros(len(psi_deg))

    print(f"\n  {'psi (deg)':>10s}  {'MoM (dB)':>10s}  "
          f"{'Friis (dB)':>10s}  {'Error (dB)':>10s}")
    print("  " + "-" * 45)

    for i, psi in enumerate(psi_rad):
        # TX at origin (unrotated)
        mesh_tx = mesher.mesh_plate_with_feed(
            width=L, height=w, feed_x=0.0, center=(0, 0, 0))

        # RX rotated by psi, translated to (0, 0, R_pol)
        mesh_rx = create_rotated_dipole_mesh(
            mesher, L, w, center=(0, 0, R_pol), psi=psi)

        combined, n_verts_tx = build_two_antenna_mesh(mesh_tx, mesh_rx)
        basis_comb = compute_rwg_connectivity(combined)

        # TX feed edges (z~0, unrotated)
        feed_tx = find_feed_edges_at_z(combined, basis_comb, 0.0, z_center=0.0)

        # RX feed edges (z~R_pol, rotated by psi)
        if abs(psi) < 1e-6:
            feed_rx = find_feed_edges_at_z(combined, basis_comb, 0.0, z_center=R_pol)
        else:
            feed_rx = find_rotated_feed_edges(
                combined, basis_comb, z_center=R_pol, psi=psi)

        if len(feed_rx) == 0:
            print(f"  {psi_deg[i]:10.1f}  WARNING: no RX feed edges found, skipping")
            Prx_Ptx_pol_mom[i] = 1e-30
            Prx_Ptx_pol_friis[i] = 1e-30
            continue

        # Excite TX only
        exc_tx = StripDeltaGapExcitation(feed_basis_indices=feed_tx, voltage=1.0)
        V_comb = exc_tx.compute_voltage_vector(basis_comb, combined, k)
        Z_comb = fill_matrix(EFIEOperator(),basis_comb, combined, k, eta0)
        I_comb = solve_direct(Z_comb, V_comb)

        # Power extraction
        I_tx_term = compute_terminal_current(I_comb, basis_comb, feed_tx)
        P_tx = 0.5 * np.real(1.0 * np.conj(I_tx_term))

        I_rx_sc = compute_terminal_current(I_comb, basis_comb, feed_rx)
        V_oc = -Z_self * I_rx_sc
        P_rx = np.abs(V_oc)**2 / (8.0 * np.real(Z_self))

        ratio = P_rx / P_tx if P_tx > 0 else 0.0
        Prx_Ptx_pol_mom[i] = ratio

        # Friis with polarization mismatch: cos^2(psi)
        cos2_psi = np.cos(psi)**2
        friis = G_bs**2 * (lam / (4.0 * np.pi * R_pol))**2 * cos2_psi
        Prx_Ptx_pol_friis[i] = friis

        mom_dB = 10.0 * np.log10(max(ratio, 1e-30))
        friis_dB = 10.0 * np.log10(max(friis, 1e-30))
        err_dB = abs(mom_dB - friis_dB) if friis > 1e-30 else float('nan')

        print(f"  {psi_deg[i]:10.1f}  {mom_dB:10.2f}  {friis_dB:10.2f}  {err_dB:10.2f}")

    Prx_Ptx_pol_mom_dB = 10.0 * np.log10(np.maximum(Prx_Ptx_pol_mom, 1e-30))
    Prx_Ptx_pol_friis_dB = 10.0 * np.log10(np.maximum(Prx_Ptx_pol_friis, 1e-30))

    # ===================================================================
    # Summary / pass-fail
    # ===================================================================
    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    # Distance sweep check
    max_dist_err = np.max(dist_error_dB)
    dist_pass = max_dist_err < 1.0
    print(f"\n  Distance sweep max error:  {max_dist_err:.3f} dB "
          f"{'PASS' if dist_pass else 'FAIL'} (threshold: 1.0 dB)")

    # Polarization sweep check (exclude psi=90)
    pol_mask = psi_deg < 89.0
    if np.any(pol_mask):
        pol_err = np.abs(Prx_Ptx_pol_mom_dB[pol_mask] -
                         Prx_Ptx_pol_friis_dB[pol_mask])
        max_pol_err = np.max(pol_err)
    else:
        max_pol_err = 0.0
    pol_pass = max_pol_err < 1.5
    print(f"  Polarization sweep max error: {max_pol_err:.3f} dB "
          f"{'PASS' if pol_pass else 'FAIL'} (threshold: 1.5 dB)")

    # Broadside gain check
    gain_err = abs(G_bs_dBi - 2.15)
    gain_pass = gain_err < 0.5
    print(f"  Broadside gain: {G_bs_dBi:.2f} dBi, error: {gain_err:.2f} dB "
          f"{'PASS' if gain_pass else 'FAIL'} (threshold: 0.5 dB from 2.15 dBi)")

    # Cross-pol check
    if len(psi_deg) > 0 and psi_deg[-1] >= 89.0:
        copol_dB = Prx_Ptx_pol_mom_dB[0]
        xpol_dB = Prx_Ptx_pol_mom_dB[-1]
        xpol_isolation = copol_dB - xpol_dB
        xpol_pass = xpol_isolation > 60.0
        print(f"  Cross-pol isolation: {xpol_isolation:.1f} dB "
              f"{'PASS' if xpol_pass else 'FAIL'} (threshold: 60 dB)")
    else:
        xpol_pass = True

    all_pass = dist_pass and pol_pass and gain_pass and xpol_pass
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    # ===================================================================
    # PLOTS
    # ===================================================================

    # --- Figure 1: Dipole characterization ---
    fig1, (ax1a, ax1b, ax1c) = plt.subplots(1, 3, figsize=(18, 5),
                                              subplot_kw={'projection': None})

    theta_deg = np.degrees(theta_cut)

    # Left: rectangular pattern cuts
    ax1a.plot(theta_deg, gain_e_dBi, 'b-', linewidth=1.5, label=r'E-plane ($xz$)')
    ax1a.plot(theta_deg, gain_h_dBi, 'r--', linewidth=1.5, label=r'H-plane ($yz$)')
    ax1a.set_xlabel(r'$\theta$ (deg)')
    ax1a.set_ylabel(r'Directivity $D$ (dBi)')
    ax1a.set_title(r'Radiation Pattern')
    ax1a.annotate(rf'$G_{{\mathrm{{bs}}}} = {G_bs_dBi:.2f}$ dBi',
                  xy=(0, G_bs_dBi), xycoords='data',
                  xytext=(30, -15), textcoords='offset points',
                  fontsize=9, ha='left',
                  bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8),
                  arrowprops=dict(arrowstyle='->', color='black'))
    ax1a.legend(fontsize=9)
    ax1a.grid(True, alpha=0.3)
    ax1a.set_xlim([0, 180])
    ax1a.set_ylim([max(gain_e_dBi.min(), -40), D_max_dBi + 3])

    # Center: text box with parameters
    ax1b.axis('off')
    param_text = (
        r'$\bf{Dipole\ Parameters}$' + '\n\n'
        rf'$f = {freq/1e9:.1f}$ GHz,  $\lambda = {lam*1e3:.1f}$ mm' + '\n'
        rf'$L = \lambda/2 = {L*1e3:.1f}$ mm' + '\n'
        rf'$w = {w*1e3:.1f}$ mm' + '\n\n'
        rf'$Z_{{\mathrm{{in}}}} = {Z_self.real:.1f} + j{Z_self.imag:.1f}$ $\Omega$' + '\n'
        rf'$D_{{\max}} = {D_max_dBi:.2f}$ dBi' + '\n'
        rf'$G_{{\mathrm{{bs}}}} = {G_bs_dBi:.2f}$ dBi' + '\n\n'
        rf'Mesh: $N = {basis_single.num_basis}$ basis fn' + '\n'
        rf'Edge: ${mesh_edge*1e3:.1f}$ mm $(\lambda/15)$'
    )
    ax1b.text(0.5, 0.5, param_text, transform=ax1b.transAxes,
              fontsize=11, verticalalignment='center', horizontalalignment='center',
              bbox=dict(boxstyle='round,pad=0.8', fc='lightyellow', ec='gray'))

    # Right: E-plane polar
    ax1c.set_visible(False)
    ax1c_polar = fig1.add_subplot(133, polar=True)
    theta_full = np.concatenate([theta_cut, 2 * np.pi - theta_cut[::-1]])
    gain_e_full = np.concatenate([gain_e, gain_e[::-1]])
    gain_e_norm = np.maximum(gain_e_full / gain_max, 0)
    ax1c_polar.plot(theta_full, gain_e_norm, 'b-', linewidth=1.5)
    ax1c_polar.set_theta_zero_location('N')
    ax1c_polar.set_theta_direction(-1)
    ax1c_polar.set_title(rf'E-plane ($xz$)' + '\n'
                         rf'$D_{{\max}} = {D_max_dBi:.1f}$ dBi', pad=15)

    fig1.suptitle(rf'Half-Wave Dipole at $f = {freq/1e9:.1f}$ GHz', fontsize=13)
    fig1.tight_layout()
    out1 = os.path.join(images_dir, 'friis_dipole_characterization.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out1}")

    # --- Figure 2: Distance sweep ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5))

    ax2a.plot(R_over_lam, Prx_Ptx_friis_dB, 'k-', linewidth=2, label='Friis (theory)')
    ax2a.plot(R_over_lam, Prx_Ptx_mom_dB, 'ro', markersize=8, label='MoM')
    ax2a.set_xlabel(r'Separation $R / \lambda$')
    ax2a.set_ylabel(r'$P_{\mathrm{rx}} / P_{\mathrm{tx}}$ (dB)')
    ax2a.set_title(r'Received Power vs Distance (co-pol)')
    ax2a.legend()
    ax2a.grid(True, alpha=0.3)

    ax2b.plot(R_over_lam, dist_error_dB, 'bs-', linewidth=1.5, markersize=6)
    ax2b.axhline(1.0, color='r', linestyle='--', alpha=0.7, label='1.0 dB threshold')
    ax2b.set_xlabel(r'Separation $R / \lambda$')
    ax2b.set_ylabel(r'$|\mathrm{Error}|$ (dB)')
    ax2b.set_title(r'Agreement: MoM vs Friis')
    ax2b.legend()
    ax2b.grid(True, alpha=0.3)
    ax2b.set_ylim(bottom=0)

    fig2.suptitle(rf'Friis Distance Sweep at $f = {freq/1e9:.1f}$ GHz', fontsize=13)
    fig2.tight_layout()
    out2 = os.path.join(images_dir, 'friis_distance_sweep.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")

    # --- Figure 3: Polarization sweep ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))

    ax3a.plot(psi_deg, Prx_Ptx_pol_friis_dB, 'k-', linewidth=2, label='Friis (theory)')
    ax3a.plot(psi_deg, Prx_Ptx_pol_mom_dB, 'ro', markersize=8, label='MoM')
    ax3a.set_xlabel(r'Polarization angle $\psi$ (deg)')
    ax3a.set_ylabel(r'$P_{\mathrm{rx}} / P_{\mathrm{tx}}$ (dB)')
    ax3a.set_title(rf'Received Power vs Polarization ($R = 10\lambda$)')
    ax3a.legend()
    ax3a.grid(True, alpha=0.3)

    # Polarization loss factor relative to co-pol
    psi_smooth = np.linspace(0, 90, 91)
    cos2_smooth = np.cos(np.radians(psi_smooth))**2
    plf_theory_dB = 10.0 * np.log10(np.maximum(cos2_smooth, 1e-30))

    # MoM normalized to psi=0
    if Prx_Ptx_pol_mom[0] > 0:
        plf_mom_dB = 10.0 * np.log10(
            np.maximum(Prx_Ptx_pol_mom / Prx_Ptx_pol_mom[0], 1e-30))
    else:
        plf_mom_dB = np.full_like(Prx_Ptx_pol_mom_dB, -300.0)

    ax3b.plot(psi_smooth, plf_theory_dB, 'k-', linewidth=2,
              label=r'$\cos^2(\psi)$')
    ax3b.plot(psi_deg, plf_mom_dB, 'ro', markersize=8, label='MoM (normalized)')
    ax3b.set_xlabel(r'Polarization angle $\psi$ (deg)')
    ax3b.set_ylabel(r'Polarization loss factor (dB)')
    ax3b.set_title(r'Polarization Mismatch Factor')
    ax3b.legend()
    ax3b.grid(True, alpha=0.3)
    ax3b.set_ylim(bottom=-40)

    fig3.suptitle(rf'Friis Polarization Sweep at $f = {freq/1e9:.1f}$ GHz', fontsize=13)
    fig3.tight_layout()
    out3 = os.path.join(images_dir, 'friis_polarization_sweep.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out3}")

    # --- Figure 4: Surface current at representative distance ---
    R_vis = 10.0 * lam
    mesh_tx_vis = mesher.mesh_plate_with_feed(
        width=L, height=w, feed_x=0.0, center=(0, 0, 0))
    mesh_rx_vis = mesher.mesh_plate_with_feed(
        width=L, height=w, feed_x=0.0, center=(0, 0, R_vis))

    combined_vis, _ = build_two_antenna_mesh(mesh_tx_vis, mesh_rx_vis)
    basis_vis = compute_rwg_connectivity(combined_vis)

    feed_tx_vis = find_feed_edges_at_z(combined_vis, basis_vis, 0.0, z_center=0.0)
    exc_vis = StripDeltaGapExcitation(feed_basis_indices=feed_tx_vis, voltage=1.0)
    V_vis = exc_vis.compute_voltage_vector(basis_vis, combined_vis, k)
    Z_vis = fill_matrix(EFIEOperator(),basis_vis, combined_vis, k, eta0)
    I_vis = solve_direct(Z_vis, V_vis)

    fig4 = plt.figure(figsize=(12, 6))
    ax4 = fig4.add_subplot(111, projection='3d')
    plot_surface_current(I_vis, basis_vis, combined_vis, ax=ax4, cmap='hot',
                         edge_color='gray', edge_width=0.2,
                         title=(rf'$|\mathbf{{J}}|$ on TX + RX dipoles, '
                                rf'$R = 10\lambda$, $f = {freq/1e9:.1f}$ GHz'))
    ax4.view_init(elev=20, azim=-60)
    out4 = os.path.join(images_dir, 'friis_surface_current.png')
    fig4.savefig(out4, dpi=150, bbox_inches='tight')
    print(f"Saved: {out4}")

    plt.show()

    print("\n" + "=" * 65)
    print("Friis validation complete!")
    print("=" * 65)


if __name__ == '__main__':
    main()
