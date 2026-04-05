"""QS solver with edge ports using vertical current signs.

Tests the fix: edge port junction edges use z-directed signs instead of
x-directed signs, so the port correctly measures vertical current from
plate to strip (ground return path).
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
    StripDeltaGapExcitation, find_edge_port_feed_edges,
    compute_feed_signs, compute_feed_signs_along_direction,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.utils.constants import c0

# ── Geometry ──────────────────────────────────────────────────────
EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
L_STRIP = 10.0e-3
TEL = 0.7e-3

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

# ── Edge port mesh (plate goes to z=0) ───────────────────────────
mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_microstrip_with_edge_ports(
    width=W_STRIP, length=L_STRIP, substrate_height=H_SUB,
    port_edges=['left', 'right'],
    plate_z_offset=0.0,
)
basis = compute_rwg_connectivity(mesh)

print(f"Mesh: {len(mesh.triangles)} tri, {basis.num_basis} RWG")

# ── Port setup with z-directed signs ─────────────────────────────
x_left = -L_STRIP / 2.0
x_right = +L_STRIP / 2.0

feed1 = find_edge_port_feed_edges(mesh, basis, port_x=x_left, strip_z=H_SUB)
feed2 = find_edge_port_feed_edges(mesh, basis, port_x=x_right, strip_z=H_SUB)

# OLD: x-directed signs (wrong for junction edges)
signs1_x = compute_feed_signs(mesh, basis, feed1)
signs2_x = compute_feed_signs(mesh, basis, feed2)

# NEW: z-directed signs (vertical current from plate to strip)
signs1_z = compute_feed_signs_along_direction(mesh, basis, feed1, [0, 0, 1])
signs2_z = compute_feed_signs_along_direction(mesh, basis, feed2, [0, 0, 1])

print(f"\nPort1 feed indices: {feed1}")
print(f"  x-signs: {signs1_x}")
print(f"  z-signs: {signs1_z}")
print(f"Port2 feed indices: {feed2}")
print(f"  x-signs: {signs2_x}")
print(f"  z-signs: {signs2_z}")

# Use z-directed signs
port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1_z)
port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2_z)

exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
config = SimulationConfig(
    frequency=1e9, excitation=exc, quad_order=4, backend='auto',
    layer_stack=stack, source_layer_name='FR4',
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

# ── QS extraction with z-directed signs ───────────────────────────
print("\n=== QS + edge port, z-directed signs (no probes) ===")
qs = QuasiStaticSolver(sim, [port1, port2], probe_feeds=False)
freqs = [0.5e9, 1e9, 2e9, 3e9, 5e9, 7e9, 10e9]
results = qs.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} "
      f"{'Z11':>20} {'Z21':>20} {'|S|^2':>8}")
print("-" * 82)
for r in results:
    S = r.S_matrix
    Z = r.Z_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    pwr = abs(S[0,0])**2 + abs(S[1,0])**2
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f} "
          f"{Z[0,0].real:+9.2f}{Z[0,0].imag:+9.2f}j "
          f"{Z[1,0].real:+9.2f}{Z[1,0].imag:+9.2f}j "
          f"{pwr:8.4f}")

# ── Comparison: probe feeds at edge ───────────────────────────────
print("\n=== QS + edge port, probe feeds (reference) ===")
qs_probe = QuasiStaticSolver(sim, [port1, port2], probe_feeds=True)
results_probe = qs_probe.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} "
      f"{'Z11':>20} {'Z21':>20} {'|S|^2':>8}")
print("-" * 82)
for r in results_probe:
    S = r.S_matrix
    Z = r.Z_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    pwr = abs(S[0,0])**2 + abs(S[1,0])**2
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f} "
          f"{Z[0,0].real:+9.2f}{Z[0,0].imag:+9.2f}j "
          f"{Z[1,0].real:+9.2f}{Z[1,0].imag:+9.2f}j "
          f"{pwr:8.4f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n=== Summary: S21 (dB) ===")
print(f"{'f (GHz)':>8} {'z-signs':>10} {'probe':>10}")
print("-" * 32)
for rz, rp in zip(results, results_probe):
    s21_z = 20 * np.log10(max(abs(rz.S_matrix[1, 0]), 1e-30))
    s21_p = 20 * np.log10(max(abs(rp.S_matrix[1, 0]), 1e-30))
    print(f"{rz.frequency/1e9:8.1f} {s21_z:10.2f} {s21_p:10.2f}")

print("\nDone.")
