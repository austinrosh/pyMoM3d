"""QS solver validation for microstrip through-line.

Tests:
1. Edge port mesh + probe feeds at the edge
2. Z₀ extraction vs Hammerstad formula
3. Effect of dielectric images on accuracy
4. Frequency sweep to find QS validity range
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
    StripDeltaGapExcitation, find_feed_edges,
    find_edge_port_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.utils.constants import c0, eta0


def hammerstad_z0(w, h, eps_r):
    """Hammerstad-Jensen microstrip Z₀ and ε_eff.

    References: Hammerstad & Jensen, "Accurate Models for Microstrip
    Computer-Aided Design", IEEE MTT-S 1980.
    """
    u = w / h
    # Effective dielectric constant
    a = 1.0 + (1.0/49.0) * np.log(
        (u**4 + (u/52.0)**2) / (u**4 + 0.432)
    ) + (1.0/18.7) * np.log(1.0 + (u/18.1)**3)
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3.0))**0.053
    eps_eff = 0.5 * (eps_r + 1.0) + 0.5 * (eps_r - 1.0) * (
        1.0 + 10.0/u
    )**(-a * b)

    # Characteristic impedance (Hammerstad)
    f_u = 6.0 + (2.0*np.pi - 6.0) * np.exp(-(30.666/u)**0.7528)
    z0 = (eta0 / (2.0*np.pi)) * np.log(f_u/u + np.sqrt(1.0 + (2.0/u)**2)) / np.sqrt(eps_eff)
    return z0, eps_eff


# ── Geometry ──────────────────────────────────────────────────────
EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
L_STRIP = 10.0e-3
TEL = 0.7e-3

z0_ham, eps_eff_ham = hammerstad_z0(W_STRIP, H_SUB, EPS_R)
print(f"Hammerstad: Z₀ = {z0_ham:.2f} Ω, ε_eff = {eps_eff_ham:.3f}")
print(f"  λ/4 at 1 GHz: {c0/(4*1e9*np.sqrt(eps_eff_ham))*1e3:.1f} mm")

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

mesher = GmshMesher(target_edge_length=TEL)

# ── Edge port mesh ────────────────────────────────────────────────
mesh_edge = mesher.mesh_microstrip_with_edge_ports(
    width=W_STRIP, length=L_STRIP, substrate_height=H_SUB,
    port_edges=['left', 'right'],
    plate_z_offset=0.0,
)
basis_edge = compute_rwg_connectivity(mesh_edge)

x_left = -L_STRIP / 2.0
x_right = +L_STRIP / 2.0

feed1 = find_edge_port_feed_edges(mesh_edge, basis_edge, port_x=x_left, strip_z=H_SUB)
feed2 = find_edge_port_feed_edges(mesh_edge, basis_edge, port_x=x_right, strip_z=H_SUB)
signs1 = compute_feed_signs(mesh_edge, basis_edge, feed1)
signs2 = compute_feed_signs(mesh_edge, basis_edge, feed2)
port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
config = SimulationConfig(
    frequency=1e9, excitation=exc, quad_order=4, backend='auto',
    layer_stack=stack, source_layer_name='FR4',
)
sim_edge = Simulation(config, mesh=mesh_edge, reporter=SilentReporter())

print(f"\nEdge port mesh: {len(mesh_edge.triangles)} tri, {basis_edge.num_basis} RWG")

# ── Flat mesh (interior ports) for comparison ─────────────────────
PORT1_X = -L_STRIP / 2.0 + 1.0e-3
PORT2_X = +L_STRIP / 2.0 - 1.0e-3

mesh_flat = mesher.mesh_plate_with_feeds(
    width=L_STRIP, height=W_STRIP,
    feed_x_list=[PORT1_X, PORT2_X],
    center=(0.0, 0.0, H_SUB),
)
basis_flat = compute_rwg_connectivity(mesh_flat)

feed1f = find_feed_edges(mesh_flat, basis_flat, feed_x=PORT1_X)
feed2f = find_feed_edges(mesh_flat, basis_flat, feed_x=PORT2_X)
signs1f = compute_feed_signs(mesh_flat, basis_flat, feed1f)
signs2f = compute_feed_signs(mesh_flat, basis_flat, feed2f)
port1f = Port(name='P1', feed_basis_indices=feed1f, feed_signs=signs1f)
port2f = Port(name='P2', feed_basis_indices=feed2f, feed_signs=signs2f)

exc_f = StripDeltaGapExcitation(feed_basis_indices=feed1f, voltage=1.0)
config_f = SimulationConfig(
    frequency=1e9, excitation=exc_f, quad_order=4, backend='auto',
    layer_stack=stack, source_layer_name='FR4',
)
sim_flat = Simulation(config_f, mesh=mesh_flat, reporter=SilentReporter())

print(f"Flat mesh: {len(mesh_flat.triangles)} tri, {basis_flat.num_basis} RWG")

# ── Frequency sweep ───────────────────────────────────────────────
freqs = [0.1e9, 0.2e9, 0.5e9, 1e9, 2e9, 3e9, 5e9]

# 1. Edge port + probe feeds (no dielectric images)
print("\n=== Edge port + probe feeds (n_diel=0) ===")
qs_edge = QuasiStaticSolver(sim_edge, [port1, port2],
                             probe_feeds=True, n_dielectric_images=0)
results_edge = qs_edge.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} "
      f"{'|S|^2':>8} {'Z11 imag':>10} {'Z21 imag':>10}")
print("-" * 62)
for r in results_edge:
    S = r.S_matrix
    Z = r.Z_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    pwr = abs(S[0,0])**2 + abs(S[1,0])**2
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f} "
          f"{pwr:8.4f} {Z[0,0].imag:10.2f} {Z[1,0].imag:10.2f}")

# 2. Edge port + probe feeds + dielectric images
print("\n=== Edge port + probe feeds (n_diel=10) ===")
qs_edge_diel = QuasiStaticSolver(sim_edge, [port1, port2],
                                  probe_feeds=True, n_dielectric_images=10)
results_edge_diel = qs_edge_diel.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} "
      f"{'|S|^2':>8} {'Z11 imag':>10} {'Z21 imag':>10}")
print("-" * 62)
for r in results_edge_diel:
    S = r.S_matrix
    Z = r.Z_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    pwr = abs(S[0,0])**2 + abs(S[1,0])**2
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f} "
          f"{pwr:8.4f} {Z[0,0].imag:10.2f} {Z[1,0].imag:10.2f}")

# 3. Flat mesh + probe feeds (reference)
print("\n=== Flat mesh + probe feeds (n_diel=0) ===")
qs_flat = QuasiStaticSolver(sim_flat, [port1f, port2f],
                             probe_feeds=True, n_dielectric_images=0)
results_flat = qs_flat.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} "
      f"{'|S|^2':>8} {'Z11 imag':>10} {'Z21 imag':>10}")
print("-" * 62)
for r in results_flat:
    S = r.S_matrix
    Z = r.Z_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    pwr = abs(S[0,0])**2 + abs(S[1,0])**2
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f} "
          f"{pwr:8.4f} {Z[0,0].imag:10.2f} {Z[1,0].imag:10.2f}")

# ── Z₀ extraction ────────────────────────────────────────────────
# For a lossless matched line: Z11 = -jZ0·cot(βl/2), Z21 = -jZ0·csc(βl)
# At low frequency (βl << 1): Z21 ≈ -jZ0/(βl), Z11 ≈ -jZ0/(βl/2)
# So Z0 ≈ |Z21| × βl ≈ |Z11| × βl/2
print("\n=== Z₀ extraction ===")
print(f"Hammerstad Z₀ = {z0_ham:.2f} Ω, ε_eff = {eps_eff_ham:.3f}")
print(f"{'f (GHz)':>8} {'Z0 edge':>10} {'Z0 edge+diel':>12} {'Z0 flat':>10}")
print("-" * 44)
for re, rd, rf in zip(results_edge, results_edge_diel, results_flat):
    f = re.frequency
    beta = 2 * np.pi * f * np.sqrt(eps_eff_ham) / c0
    bl = beta * L_STRIP

    def extract_z0(r):
        Z = r.Z_matrix
        # Use Z21: Z0 = |Z21 * sin(βl)|
        z0_21 = abs(Z[1, 0] * np.sin(bl))
        # Use Z11: Z0 = |Z11 * tan(βl/2)|
        z0_11 = abs(Z[0, 0] * np.tan(bl / 2.0))
        return z0_21, z0_11

    z0e_21, z0e_11 = extract_z0(re)
    z0d_21, z0d_11 = extract_z0(rd)
    z0f_21, z0f_11 = extract_z0(rf)
    print(f"{f/1e9:8.1f} {z0e_21:10.2f} {z0d_21:12.2f} {z0f_21:10.2f}")

print("\nDone.")
