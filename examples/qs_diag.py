"""Diagnostic: extract Z₀ and ε_eff self-consistently from QS Z-matrix.

Uses β-independent formulas:
  Z₀ = sqrt(Z11² - Z21²)
  cos(βl) = Z11 / Z21
  ε_eff = (β·c₀ / (2πf))²
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


def hammerstad_z0(w, h, eps_r):
    u = w / h
    a = 1 + (1/49)*np.log((u**4 + (u/52)**2)/(u**4 + 0.432)) + (1/18.7)*np.log(1 + (u/18.1)**3)
    b = 0.564 * ((eps_r - 0.9)/(eps_r + 3))**0.053
    eps_eff = 0.5*(eps_r + 1) + 0.5*(eps_r - 1)*(1 + 10/u)**(-a*b)
    f_u = 6 + (2*np.pi - 6) * np.exp(-(30.666/u)**0.7528)
    z0 = (eta0/(2*np.pi)) * np.log(f_u/u + np.sqrt(1 + (2/u)**2)) / np.sqrt(eps_eff)
    return z0, eps_eff


EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
L_STRIP = 10.0e-3
TEL = 0.5e-3  # finer mesh

z0_ref, eps_eff_ref = hammerstad_z0(W_STRIP, H_SUB, EPS_R)
print(f"Hammerstad: Z₀ = {z0_ref:.2f} Ω, ε_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

mesher = GmshMesher(target_edge_length=TEL)

PORT1_X = -L_STRIP / 2.0 + 1.0e-3
PORT2_X = +L_STRIP / 2.0 - 1.0e-3
L_eff = PORT2_X - PORT1_X  # effective line length between ports

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
print(f"Effective line length: {L_eff*1e3:.1f} mm")

freqs = [0.05e9, 0.1e9, 0.2e9, 0.5e9, 1e9]

for n_diel in [0, 10, 30]:
    print(f"\n=== n_diel = {n_diel} ===")
    qs = QuasiStaticSolver(sim, [port1, port2],
                           probe_feeds=True, n_dielectric_images=n_diel)

    results = qs.extract(freqs)

    print(f"{'f (GHz)':>8} {'Z11 imag':>10} {'Z21 imag':>10} {'Z₀':>8} {'ε_eff':>8} {'β':>10}")
    print("-" * 60)
    for r in results:
        Z = r.Z_matrix
        z11 = Z[0, 0]
        z21 = Z[1, 0]

        # Z₀ = sqrt(Z11² - Z21²)
        z0_sq = z11**2 - z21**2
        z0_ext = np.sqrt(z0_sq).real if z0_sq.real > 0 else np.sqrt(abs(z0_sq))

        # cos(βl) = Z11/Z21
        ratio = z11 / z21
        cos_bl = ratio.real if abs(ratio.imag) < 0.1 * abs(ratio.real) else ratio
        bl = np.arccos(np.clip(np.real(cos_bl), -1, 1))
        beta = bl / L_eff if L_eff > 0 else 0
        eps_eff = (beta * c0 / (2 * np.pi * r.frequency))**2

        print(f"{r.frequency/1e9:8.2f} {z11.imag:10.2f} {z21.imag:10.2f} "
              f"{z0_ext:8.2f} {eps_eff:8.3f} {beta:10.2f}")

    # Also compute L_pul and C_pul from lowest-frequency Z-matrix
    r_low = results[0]
    Z = r_low.Z_matrix
    f_low = r_low.frequency
    omega = 2 * np.pi * f_low
    z11, z21 = Z[0, 0], Z[1, 0]

    # Low-frequency approximation: Z11 ≈ jωL/2 - j/(ωC), Z21 ≈ -j/(ωC)
    # So Z11 - Z21 ≈ jωL/2
    L_total = 2 * (z11 - z21).imag / omega
    C_total = -1 / (omega * z21.imag)
    L_pul = L_total / L_eff
    C_pul = C_total / L_eff

    print(f"\n  L_pul = {L_pul*1e9:.2f} nH/m, C_pul = {C_pul*1e12:.2f} pF/m")
    print(f"  Z₀ = √(L/C) = {np.sqrt(L_pul/C_pul):.2f} Ω")
    print(f"  v_p = 1/√(LC) = {1/np.sqrt(L_pul*C_pul)/1e8:.3f} × 10⁸ m/s")
    print(f"  ε_eff = (c₀/v_p)² = {c0**2 * L_pul * C_pul:.3f}")

# Also check with air substrate (ε_r = 1)
print(f"\n=== Air substrate (ε_r=1) ===")
stack_air = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('air_sub', z_bot=0.0, z_top=H_SUB, eps_r=1.0),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])
config_air = SimulationConfig(
    frequency=1e9, excitation=exc, quad_order=4, backend='auto',
    layer_stack=stack_air, source_layer_name='air_sub',
)
sim_air = Simulation(config_air, mesh=mesh, reporter=SilentReporter())
qs_air = QuasiStaticSolver(sim_air, [port1, port2],
                           probe_feeds=True, n_dielectric_images=0)
results_air = qs_air.extract(freqs)

print(f"{'f (GHz)':>8} {'Z₀':>8} {'ε_eff':>8}")
print("-" * 28)
for r in results_air:
    Z = r.Z_matrix
    z0_sq = Z[0,0]**2 - Z[1,0]**2
    z0_ext = np.sqrt(z0_sq).real if z0_sq.real > 0 else np.sqrt(abs(z0_sq))
    ratio = Z[0,0] / Z[1,0]
    bl = np.arccos(np.clip(np.real(ratio), -1, 1))
    beta = bl / L_eff
    eps_eff = (beta * c0 / (2 * np.pi * r.frequency))**2
    print(f"{r.frequency/1e9:8.2f} {z0_ext:8.2f} {eps_eff:8.3f}")

r_low = results_air[0]
Z = r_low.Z_matrix
omega = 2 * np.pi * r_low.frequency
L_total = 2 * (Z[0,0] - Z[1,0]).imag / omega
C_total = -1 / (omega * Z[1,0].imag)
L_pul_air = L_total / L_eff
C_pul_air = C_total / L_eff
print(f"\n  L_pul = {L_pul_air*1e9:.2f} nH/m, C_pul = {C_pul_air*1e12:.2f} pF/m")
print(f"  Z₀ = √(L/C) = {np.sqrt(L_pul_air/C_pul_air):.2f} Ω")
print(f"  ε_eff = (c₀/v_p)² = {c0**2 * L_pul_air * C_pul_air:.3f}")
print(f"  Expected: Z₀ ≈ {hammerstad_z0(W_STRIP, H_SUB, 1.0)[0]:.2f} Ω, ε_eff ≈ 1.000")

print(f"\nReference: Z₀ = {z0_ref:.2f} Ω, ε_eff = {eps_eff_ref:.3f}")
print("Done.")
