"""Full-wave MPIE extraction with edge-fed vertical plate ports.

Tests the edge port mesh with the full-wave NetworkExtractor (Strata DCIM
backend).  Compares against:
1. QS + probe feeds (baseline, known working at low freq)
2. Full-wave with old interior strip delta-gap ports (baseline at high freq)

Expected: the edge port gives the full-wave solver a better port model,
pushing the working range down from ~5 GHz toward lower frequencies.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor, Layer, LayerStack,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges,
    find_edge_port_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.utils.constants import c0

# ── Geometry ──────────────────────────────────────────────────────
EPS_R = 4.4
H_SUB = 1.6e-3       # m
W_STRIP = 3.06e-3    # m
L_STRIP = 10.0e-3    # m
TEL = 0.7e-3          # mesh edge length

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

# ── Edge port mesh ────────────────────────────────────────────────
mesher = GmshMesher(target_edge_length=TEL)
# Offset plate bottom from z=0 to avoid PEC ground singularity
# in the layered Green's function
PLATE_Z_OFFSET = H_SUB / 10.0  # 0.16 mm above ground

mesh_edge = mesher.mesh_microstrip_with_edge_ports(
    width=W_STRIP,
    length=L_STRIP,
    substrate_height=H_SUB,
    port_edges=['left', 'right'],
    plate_z_offset=PLATE_Z_OFFSET,
)
basis_edge = compute_rwg_connectivity(mesh_edge)

x_left = -L_STRIP / 2.0
x_right = +L_STRIP / 2.0

feed1_edge = find_edge_port_feed_edges(mesh_edge, basis_edge,
                                        port_x=x_left, strip_z=H_SUB)
feed2_edge = find_edge_port_feed_edges(mesh_edge, basis_edge,
                                        port_x=x_right, strip_z=H_SUB)
signs1_edge = compute_feed_signs(mesh_edge, basis_edge, feed1_edge)
signs2_edge = compute_feed_signs(mesh_edge, basis_edge, feed2_edge)
port1_edge = Port(name='P1', feed_basis_indices=feed1_edge,
                  feed_signs=signs1_edge)
port2_edge = Port(name='P2', feed_basis_indices=feed2_edge,
                  feed_signs=signs2_edge)

exc_edge = StripDeltaGapExcitation(feed_basis_indices=feed1_edge, voltage=1.0)
config_edge = SimulationConfig(
    frequency=1e9, excitation=exc_edge, quad_order=4, backend='auto',
    layer_stack=stack, source_layer_name='FR4',
    gf_backend='auto',
)
sim_edge = Simulation(config_edge, mesh=mesh_edge, reporter=SilentReporter())

print(f"Edge port mesh: {len(mesh_edge.triangles)} tri, "
      f"{basis_edge.num_basis} RWG")

# ── Full-wave extraction with edge ports ──────────────────────────
print("\n=== Full-Wave MPIE + Edge Ports ===")
freqs = [1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 8e9, 10e9]

extractor_edge = NetworkExtractor(
    sim_edge, [port1_edge, port2_edge],
    store_currents=False,
)
results_fw_edge = extractor_edge.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} "
      f"{'|S12-S21|':>10} {'cond':>10}")
print("-" * 54)
for r in results_fw_edge:
    S = r.S_matrix
    s21_dB = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11_dB = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    recip = abs(S[0, 1] - S[1, 0])
    print(f"{r.frequency/1e9:8.1f} {s21_dB:10.2f} {s11_dB:10.2f} "
          f"{recip:10.2e} {r.condition_number:10.1f}")

# ── QS + probe feeds for comparison ──────────────────────────────
print("\n=== QS + Probe Feeds (edge port mesh, reference) ===")
qs = QuasiStaticSolver(sim_edge, [port1_edge, port2_edge], probe_feeds=True)
results_qs = qs.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10}")
print("-" * 32)
for r in results_qs:
    S = r.S_matrix
    s21_dB = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11_dB = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    print(f"{r.frequency/1e9:8.1f} {s21_dB:10.2f} {s11_dB:10.2f}")

# ── Summary comparison ───────────────────────────────────────────
print("\n=== Summary: S21 (dB) ===")
print(f"{'f (GHz)':>8} {'FW Edge':>10} {'QS Probe':>10}")
print("-" * 32)
for rfw, rqs in zip(results_fw_edge, results_qs):
    s21_fw = 20 * np.log10(max(abs(rfw.S_matrix[1, 0]), 1e-30))
    s21_qs = 20 * np.log10(max(abs(rqs.S_matrix[1, 0]), 1e-30))
    print(f"{rfw.frequency/1e9:8.1f} {s21_fw:10.2f} {s21_qs:10.2f}")

# ── Baseline: old flat mesh with interior ports ───────────────────
print("\n=== Baseline: Full-Wave with Interior Strip Delta-Gap ===")
PORT1_X = -L_STRIP / 2.0 + 1.0e-3
PORT2_X = +L_STRIP / 2.0 - 1.0e-3

mesh_flat = mesher.mesh_plate_with_feeds(
    width=L_STRIP, height=W_STRIP,
    feed_x_list=[PORT1_X, PORT2_X],
    center=(0.0, 0.0, H_SUB),
)
basis_flat = compute_rwg_connectivity(mesh_flat)

feed1_flat = find_feed_edges(mesh_flat, basis_flat, feed_x=PORT1_X)
feed2_flat = find_feed_edges(mesh_flat, basis_flat, feed_x=PORT2_X)
signs1_flat = compute_feed_signs(mesh_flat, basis_flat, feed1_flat)
signs2_flat = compute_feed_signs(mesh_flat, basis_flat, feed2_flat)
port1_flat = Port(name='P1', feed_basis_indices=feed1_flat,
                  feed_signs=signs1_flat)
port2_flat = Port(name='P2', feed_basis_indices=feed2_flat,
                  feed_signs=signs2_flat)

exc_flat = StripDeltaGapExcitation(feed_basis_indices=feed1_flat, voltage=1.0)
config_flat = SimulationConfig(
    frequency=1e9, excitation=exc_flat, quad_order=4, backend='auto',
    layer_stack=stack, source_layer_name='FR4',
)
sim_flat = Simulation(config_flat, mesh=mesh_flat, reporter=SilentReporter())

print(f"Flat mesh: {len(mesh_flat.triangles)} tri, {basis_flat.num_basis} RWG")

extractor_flat = NetworkExtractor(sim_flat, [port1_flat, port2_flat])
results_fw_flat = extractor_flat.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} {'cond':>10}")
print("-" * 42)
for r in results_fw_flat:
    S = r.S_matrix
    s21_dB = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11_dB = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    print(f"{r.frequency/1e9:8.1f} {s21_dB:10.2f} {s11_dB:10.2f} "
          f"{r.condition_number:10.1f}")

print("\nDone.")
