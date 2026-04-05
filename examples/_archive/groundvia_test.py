"""Test GroundVia for broadband microstrip port model.

Adds GroundVia at each port to provide the missing ground return path
for the full-wave MPIE solver. Compares:
1. Full-wave, no GroundVia (baseline — fails at low freq)
2. Full-wave + GroundVia (should fix low-freq S21)
3. Full-wave + GroundVia + SOC de-embedding
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, GroundVia, NetworkExtractor, Layer, LayerStack,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges,
    find_edge_port_feed_edges, compute_feed_signs,
)
from pyMoM3d.utils.constants import c0

# ── Geometry ──────────────────────────────────────────────────────
EPS_R = 4.4
H_SUB = 1.6e-3       # m
W_STRIP = 3.06e-3    # m
L_STRIP = 10.0e-3    # m
TEL = 0.7e-3

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

# ── Flat mesh with interior ports (known-working baseline) ────────
mesher = GmshMesher(target_edge_length=TEL)
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

# GroundVia at port feed edges
via1 = GroundVia('via1', basis_indices=feed1)
via2 = GroundVia('via2', basis_indices=feed2)

exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
config = SimulationConfig(
    frequency=1e9, excitation=exc, quad_order=4, backend='auto',
    layer_stack=stack, source_layer_name='FR4',
    gf_backend='auto',
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

print(f"Mesh: {len(mesh.triangles)} tri, {basis.num_basis} RWG")
print(f"Feed1 indices: {feed1} ({len(feed1)} edges)")
print(f"Feed2 indices: {feed2} ({len(feed2)} edges)")

# ── Sweep ─────────────────────────────────────────────────────────
freqs = [1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 8e9, 10e9]

# 1. Without GroundVia (baseline)
print("\n=== Full-Wave, NO GroundVia (baseline) ===")
ext_novia = NetworkExtractor(sim, [port1, port2], store_currents=False)
results_novia = ext_novia.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} {'cond':>10}")
print("-" * 42)
for r in results_novia:
    S = r.S_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f} {r.condition_number:10.1f}")

# 2. With GroundVia
print("\n=== Full-Wave + GroundVia ===")
ext_via = NetworkExtractor(
    sim, [port1, port2],
    ground_vias=[via1, via2],
    store_currents=False,
)
results_via = ext_via.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} {'cond':>10}")
print("-" * 42)
for r in results_via:
    S = r.S_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f} {r.condition_number:10.1f}")

# 3. Summary comparison
print("\n=== Summary: S21 (dB) ===")
print(f"{'f (GHz)':>8} {'No Via':>10} {'Via':>10} {'Delta':>10}")
print("-" * 42)
for rn, rv in zip(results_novia, results_via):
    s21_n = 20 * np.log10(max(abs(rn.S_matrix[1, 0]), 1e-30))
    s21_v = 20 * np.log10(max(abs(rv.S_matrix[1, 0]), 1e-30))
    print(f"{rn.frequency/1e9:8.1f} {s21_n:10.2f} {s21_v:10.2f} {s21_v-s21_n:10.2f}")

# 4. Passivity check
print("\n=== Passivity Check (with GroundVia) ===")
print(f"{'f (GHz)':>8} {'|S11|^2+|S21|^2':>18} {'passive?':>10}")
print("-" * 40)
for r in results_via:
    S = r.S_matrix
    pwr = abs(S[0,0])**2 + abs(S[1,0])**2
    ok = "YES" if pwr <= 1.001 else "NO"
    print(f"{r.frequency/1e9:8.1f} {pwr:18.6f} {ok:>10}")

print("\nDone.")
