"""Debug Z₀ extraction: test SP prefactor conventions.

The QS solver uses Z(ω) = jωμ₀·L + sp_pf·P.
Currently sp_pf = 1/(jωε₀ε_r), but this may double-count the dielectric
when dielectric images are also included in G_φ.

Tests different conventions to find which gives Z₀ ≈ 50 Ω.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, Layer, LayerStack,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.utils.constants import c0, mu0, eps0, eta0
from pyMoM3d.network.network_result import NetworkResult


def hammerstad_z0(w, h, eps_r):
    u = w / h
    a = 1 + (1/49) * np.log((u**4 + (u/52)**2)/(u**4 + 0.432)) + (1/18.7) * np.log(1 + (u/18.1)**3)
    b = 0.564 * ((eps_r - 0.9)/(eps_r + 3))**0.053
    eps_eff = 0.5*(eps_r + 1) + 0.5*(eps_r - 1)*(1 + 10/u)**(-a*b)
    f_u = 6 + (2*np.pi - 6) * np.exp(-(30.666/u)**0.7528)
    z0 = (eta0/(2*np.pi)) * np.log(f_u/u + np.sqrt(1 + (2/u)**2)) / np.sqrt(eps_eff)
    return z0, eps_eff


EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
L_STRIP = 10.0e-3
TEL = 0.7e-3

z0_ref, eps_eff_ref = hammerstad_z0(W_STRIP, H_SUB, EPS_R)
print(f"Hammerstad: Z₀ = {z0_ref:.2f} Ω, ε_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

mesher = GmshMesher(target_edge_length=TEL)

# Flat mesh with interior ports
PORT1_X = -L_STRIP / 2.0 + 1.0e-3
PORT2_X = +L_STRIP / 2.0 - 1.0e-3

mesh = mesher.mesh_plate_with_feeds(
    width=L_STRIP, height=W_STRIP,
    feed_x_list=[PORT1_X, PORT2_X],
    center=(0.0, 0.0, H_SUB),
)
basis = compute_rwg_connectivity(mesh)

feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
signs1 = compute_feed_signs(mesh, basis, feed1)
signs2 = compute_feed_signs(mesh, basis, feed2)
port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
config = SimulationConfig(
    frequency=1e9, excitation=exc, quad_order=4, backend='auto',
    layer_stack=stack, source_layer_name='FR4',
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

print(f"Mesh: {len(mesh.triangles)} tri, {basis.num_basis} RWG")

# ── Build QS solver to access L_raw, P, G_s ──────────────────────
# Test with n_diel=0 and n_diel=10
for n_diel in [0, 10]:
    qs = QuasiStaticSolver(sim, [port1, port2],
                           probe_feeds=True, n_dielectric_images=n_diel)

    freq = 0.5e9  # low freq where QS is valid
    omega = 2 * np.pi * freq

    N_s = qs._N_surface
    N_p = qs._N_probes
    N_total = N_s + N_p

    # Test different SP prefactors
    print(f"\n=== n_diel = {n_diel} ===")

    for label, sp_scale in [
        ("1/(jωε₀ε_r)", 1.0),           # current code
        ("1/(jωε₀)",     EPS_R),         # no ε_r correction
        ("1/(jωε₀√ε_r)", np.sqrt(EPS_R)), # geometric mean
    ]:
        vp_pf = 1j * omega * mu0
        sp_pf = 1.0 / (1j * omega * eps0 * EPS_R) * sp_scale

        Z_hybrid = vp_pf * qs.L_raw_hybrid + sp_pf * qs.P_hybrid

        try:
            I_all = np.linalg.solve(Z_hybrid, qs.V_all)
        except np.linalg.LinAlgError:
            print(f"  {label}: SINGULAR")
            continue

        Y_mat = np.zeros((N_p, N_p), dtype=np.complex128)
        for q in range(N_p):
            for p in range(N_p):
                Y_mat[q, p] = I_all[N_s + q, p] / qs.ports[p].V_ref

        try:
            Z_mat = np.linalg.inv(Y_mat)
        except:
            Z_mat = np.full((N_p, N_p), np.inf + 0j)

        # S-parameters
        Z0 = 50.0
        I_P = np.eye(N_p)
        S = (Z_mat - Z0 * I_P) @ np.linalg.inv(Z_mat + Z0 * I_P)
        s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
        s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))

        # Z₀ from Z21 at low frequency
        beta = 2 * np.pi * freq * np.sqrt(eps_eff_ref) / c0
        bl = beta * L_STRIP
        z0_ext = abs(Z_mat[1, 0] * np.sin(bl))

        print(f"  {label:16s}: S21={s21:+7.2f} dB, S11={s11:+7.2f} dB, "
              f"Z₀≈{z0_ext:.1f} Ω, Z11={Z_mat[0,0].imag:+.1f}j")

# ── Also try: compute C_air (no images at all) ───────────────────
print("\n=== Reference: C_air (free-space G_φ, no images) ===")

# Build a QS solver with a layer stack that has no PEC ground
# (just air everywhere) to get C_air
stack_air = LayerStack([
    Layer('air_below', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('air_sub', z_bot=0.0, z_top=H_SUB, eps_r=1.0),
    Layer('air_above', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])
config_air = SimulationConfig(
    frequency=1e9, excitation=exc, quad_order=4, backend='auto',
    layer_stack=stack_air, source_layer_name='air_sub',
)
sim_air = Simulation(config_air, mesh=mesh, reporter=SilentReporter())
qs_air = QuasiStaticSolver(sim_air, [port1, port2],
                           probe_feeds=True, n_dielectric_images=0)

# Extract at low frequency with sp_pf = 1/(jωε₀) (no ε_r since air)
omega = 2 * np.pi * 0.5e9
vp_pf = 1j * omega * mu0
# For air stack, ε_r = 1, so sp_pf = 1/(jωε₀)
sp_pf_air = 1.0 / (1j * omega * eps0 * 1.0)  # ε_r = 1

Z_air = vp_pf * qs_air.L_raw_hybrid + sp_pf_air * qs_air.P_hybrid
I_air = np.linalg.solve(Z_air, qs_air.V_all)
Y_air = np.zeros((2, 2), dtype=np.complex128)
for q in range(2):
    for p in range(2):
        Y_air[q, p] = I_air[qs_air._N_surface + q, p] / qs_air.ports[p].V_ref
Z_mat_air = np.linalg.inv(Y_air)

beta_air = 2 * np.pi * 0.5e9 / c0  # free space
bl_air = beta_air * L_STRIP
z0_air = abs(Z_mat_air[1, 0] * np.sin(bl_air))
print(f"  Z₀_air ≈ {z0_air:.1f} Ω (free-space, should be ~Z₀/√ε_eff)")
print(f"  Z₀_air/Z₀_hammerstad = {z0_air/z0_ref:.3f}")
print(f"  Expected ratio: 1/√ε_eff = ... wait, Z₀_air should be higher than Z₀")
print(f"  Z₀_air = Z₀_microstrip × √ε_eff = {z0_ref * np.sqrt(eps_eff_ref):.1f} Ω expected")

# Compute ε_eff from the ratio
if z0_air > 0:
    eps_eff_extracted = (z0_air / z0_ref)**2 if z0_air > z0_ref else 0
    print(f"  ε_eff from Z₀ ratio: (Z₀_air/Z₀)² = ({z0_air:.1f}/{z0_ref:.1f})² = {eps_eff_extracted:.2f}")

print("\nDone.")
