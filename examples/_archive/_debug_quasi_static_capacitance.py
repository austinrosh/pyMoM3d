"""Extract per-unit-length capacitance from the quasi-static scalar potential.

The AEFIE scalar Green's function matrix G_s gives the potential at each
triangle due to charge at each triangle:

    phi_i = sum_j G_s[i,j] * sigma_j * A_j

For a constant-potential strip (phi = 1 everywhere):
    G_s @ diag(A) @ sigma = 1

Total charge Q = sum(sigma_j * A_j), capacitance C = Q/V = Q.
Per-unit-length C_pul = C / L.
eps_eff = C_pul(eps_r) / C_pul(air).

This bypasses port models entirely — pure electrostatic computation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Layer, LayerStack, c0,
    microstrip_z0_hammerstad,
)
from pyMoM3d.mom.aefie import fill_scalar_green_matrix
from pyMoM3d.greens.layered import LayeredGreensFunction

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 10e-3
TEL = 0.75e-3
FREQ = 100e6  # quasi-static (low freq)

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f}, eps_eff = {eps_eff_ref:.3f}")

# Analytical parallel-plate capacitance for reference
C_pp = 8.854e-12 * EPS_R * W * L / H
C_pp_air = 8.854e-12 * W * L / H
print(f"Parallel plate: C(FR4) = {C_pp*1e12:.3f} pF, C(air) = {C_pp_air*1e12:.3f} pF")
print(f"PP ratio: {EPS_R:.1f}, MS ratio (Hammerstad): {eps_eff_ref:.3f}")


def compute_C_from_Gs(eps_r_sub, tel, freq=FREQ):
    """Compute strip-to-ground capacitance from scalar Green's function matrix."""
    stack = LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('sub', z_bot=0.0, z_top=H, eps_r=eps_r_sub),
        Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
    ])

    mesher = GmshMesher(target_edge_length=tel)
    mesh = mesher.mesh_plate(width=L, height=W, center=(0.0, 0.0, H))

    T = mesh.get_statistics()['num_triangles']
    areas = mesh.triangle_areas

    # Source-layer wavenumber
    omega = 2 * np.pi * freq
    src_layer = stack.get_layer('sub')

    # Build layered GF for correction
    gf = LayeredGreensFunction(stack, freq, source_layer_name='sub')
    k = complex(gf.wavenumber)

    # G_s: T x T scalar Green's function including layered correction
    G_s = fill_scalar_green_matrix(
        mesh, k, quad_order=4,
        near_threshold=0.2,
        backend='auto',
        greens_fn=gf,
    )

    # Solve: G_s @ diag(A) @ sigma = 1
    # Let q = sigma * A (charge per element), then:
    # G_s @ q = 1
    # But G_s[i,j] includes the *source* area integration (from fill_scalar_green_matrix),
    # so the equation is: (G_s / diag(A_src)) @ (sigma * A) = phi
    # Wait, need to check how fill_scalar_green_matrix is defined.

    # fill_scalar_green_matrix computes:
    # G_s[t_obs, t_src] = integral_{obs} integral_{src} G(r,r') dS dS'
    # So G_s already includes both area integrals.
    # The potential at triangle t_obs due to charge sigma on all triangles:
    # phi(t_obs) = sum_j G_s[t_obs, j] * sigma_j
    # (where sigma_j is charge per unit area on triangle j)
    # Wait, that would make G_s * sigma give integrated potential, not potential at a point.

    # Actually, the EFIE relationship is:
    # Z_Phi = -j/(omega*eps_0) * integral_obs (div f_m)(r) * integral_src G(r,r') (div f_n)(r') dS dS'
    # And in AEFIE: G_s gives triangle-to-triangle integral.

    # For our purpose, the potential at the centroid of triangle i due to
    # a unit charge DENSITY sigma_j on triangle j is:
    # phi_i = G_s[i,j] / A_i * sigma_j
    # because G_s includes the observation area integral.

    # No wait, the convention might be:
    # G_s[i,j] = A_i * A_j * G_avg(i,j)
    # where G_avg is the double-averaged Green's function.

    # For constant sigma: phi_i = sum_j G_avg(i,j) * A_j * sigma_j
    # = sum_j (G_s[i,j] / (A_i * A_j)) * A_j * sigma_j
    # = sum_j G_s[i,j] / A_i * sigma_j
    # = (1/A_i) * G_s @ sigma

    # Setting phi_i = 1 for all i:
    # (1/A_i) * G_s @ sigma = 1
    # diag(1/A) @ G_s @ sigma = 1
    # sigma = (diag(1/A) @ G_s)^{-1} @ 1

    # Actually, I think G_s already includes the area weighting such that
    # G_s @ sigma gives the potential integrated over the obs triangle.
    # Let me check...

    # From the code: G_s is assembled as:
    # G_s[t_obs, t_src] += w_obs * w_src * G(r_obs, r_src) * 2*A_obs * 2*A_src

    # Wait, looking at the code more carefully:
    # correction *= twice_area[t_obs]  <-- at line 259
    # The inner integral already has * twice_area[t_src]

    # So G_s[i,j] ≈ A_i * A_j * <G(r,r')>_{i,j} (area-weighted average)

    # The potential at centroid of triangle i:
    # phi_i = sum_j <G(r,r')>_{i,j} * sigma_j * A_j
    # = sum_j G_s[i,j] / A_i * sigma_j

    # For phi = 1:
    # sum_j G_s[i,j] / A_i * sigma_j = 1
    # G_s @ sigma = A (vector of triangle areas)

    # The scalar potential prefactor is 1/(4*pi*eps_0) in free-space.
    # But fill_scalar_green_matrix just computes exp(-jkR)/(4piR).
    # The AEFIE prefactor is applied separately.

    # For quasi-static capacitance:
    # phi = (1/eps_0) * G_s_QS @ sigma  (QS = quasi-static, k→0: exp(-jkR) → 1)
    # Setting phi = 1: (1/eps_0) * G_s @ sigma = A
    # sigma = eps_0 * G_s^{-1} @ A
    # Q = sigma . A = eps_0 * A^T @ G_s^{-1} @ A

    # But wait, for the LAYERED case, the Green's function already accounts
    # for the layer structure. The scalar potential in MPIE is:
    # phi(r) = (1/eps_0) * integral G_phi^C(r,r') * sigma(r') dS'
    # But with Formulation-C, the relationship is more complex.

    # Actually, in the AEFIE code, G_s is the RAW Green's function integral
    # (not scaled by any eps factor). The EFIE scalar potential term is:
    # Z_Phi = (-j*eta/k) * D^T * G_s * D
    # where D is the divergence matrix.
    # eta/k = 1/(omega*eps_0*eps_r) for the source-layer.

    # For our capacitance extraction, the potential due to charge is:
    # phi = (1/eps_0) * G_s @ sigma (approximately, with the right conventions)

    # For the LAYERED case, G_s includes the CORRECTION G_ML - G_fs + G_fs = G_ML.
    # G_ML here is G_phi^C * eps_r (from the strata code).

    # So phi = (1/eps_0) * G_ML @ sigma where G_ML = G_phi^C * eps_r = G_V

    # For a constant potential strip:
    # G_V @ sigma = eps_0 * A (setting phi = V = 1 everywhere)
    # sigma = eps_0 * G_V^{-1} @ A
    # Q = sigma . A = eps_0 * A^T @ G_V^{-1} @ A

    # Hmm, I'm getting confused about the conventions. Let me just compute
    # the ratio numerically. If I do:
    # G_s_FR4 @ x = A → x is proportional to the charge for constant potential
    # C_FR4 = sum(x * A)
    # G_s_air @ y = A → y is proportional to charge for constant potential
    # C_air = sum(y * A)
    # ratio = C_FR4 / C_air (the eps_0 cancels)

    A_vec = areas.copy()

    # Solve G_s @ x = A
    x = np.linalg.solve(G_s, A_vec)

    # Total "charge" * area
    Q = np.dot(x, A_vec)

    return Q.real, T, G_s


print(f"\n{'TEL(mm)':>8} {'#tri':>6} {'Q_sub':>12} {'Q_air':>12} "
      f"{'Q ratio':>8} {'eps_eff':>10} {'err%':>8}")
print("-" * 80)

for tel in [1.0e-3, 0.75e-3, 0.5e-3, 0.35e-3, 0.25e-3]:
    Q_sub, T1, G_sub = compute_C_from_Gs(EPS_R, tel)
    Q_air, T2, G_air = compute_C_from_Gs(1.0, tel)

    # C(FR4)/C(air) = eps_r * Q_sub / Q_air
    # because the EFIE prefactor has 1/eps_r, and C = eps_0*eps_r / <G_V>
    q_ratio = Q_sub / Q_air
    eps_eff_ext = EPS_R * q_ratio
    err = abs(eps_eff_ext - eps_eff_ref) / eps_eff_ref * 100

    print(f"  {tel*1e3:>5.2f}  {T1:>6d}  {Q_sub:>12.4e}  {Q_air:>12.4e}  "
          f"{q_ratio:>8.4f}  {eps_eff_ext:>10.3f}  {err:>7.1f}")

# Also look at the G_s diagonal ratio
print(f"\n--- G_s diagonal comparison (TEL = 0.75mm) ---")
Q_sub, _, G_sub = compute_C_from_Gs(EPS_R, 0.75e-3)
Q_air, _, G_air = compute_C_from_Gs(1.0, 0.75e-3)

diag_ratio = np.abs(np.diag(G_sub)) / np.abs(np.diag(G_air))
print(f"  G_s diagonal ratio: mean={np.mean(diag_ratio):.3f}, "
      f"min={np.min(diag_ratio):.3f}, max={np.max(diag_ratio):.3f}")
print(f"  Expected for eps_r: {1/EPS_R:.3f}")
print(f"  Expected for eps_eff: {1/eps_eff_ref:.3f}")

# Off-diagonal ratio
mask = ~np.eye(G_sub.shape[0], dtype=bool)
offdiag_ratio = np.abs(G_sub[mask]) / np.abs(G_air[mask])
print(f"  G_s off-diagonal ratio: mean={np.mean(offdiag_ratio):.3f}, "
      f"median={np.median(offdiag_ratio):.3f}")
