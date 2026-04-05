"""Debug QS solver with edge port mesh (no probe feeds).

Investigate why QS + edge port without probes gives oscillating S21.
Check L_raw, G_s, P matrices and the solve path.
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
    StripDeltaGapExcitation, find_edge_port_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.utils.constants import c0, mu0, eps0, eta0

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

# ── Edge port mesh ────────────────────────────────────────────────
mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_microstrip_with_edge_ports(
    width=W_STRIP, length=L_STRIP, substrate_height=H_SUB,
    port_edges=['left', 'right'],
    plate_z_offset=0.0,  # plate goes all the way to z=0
)
basis = compute_rwg_connectivity(mesh)

print(f"Mesh: {len(mesh.triangles)} tri, {basis.num_basis} RWG")
z_vals = mesh.vertices[:, 2]
print(f"z range: [{z_vals.min():.6f}, {z_vals.max():.6f}]")

# Classify basis functions
n_strip = n_plate = n_junction = 0
strip_idx = []
plate_idx = []
junc_idx = []
for n in range(basis.num_basis):
    tp, tm = basis.t_plus[n], basis.t_minus[n]
    z_tp = mesh.vertices[mesh.triangles[tp], 2]
    z_tm = mesh.vertices[mesh.triangles[tm], 2]
    tp_strip = np.allclose(z_tp, H_SUB, atol=1e-8)
    tm_strip = np.allclose(z_tm, H_SUB, atol=1e-8)
    if tp_strip and tm_strip:
        n_strip += 1
        strip_idx.append(n)
    elif tp_strip != tm_strip:
        n_junction += 1
        junc_idx.append(n)
    else:
        n_plate += 1
        plate_idx.append(n)

print(f"Strip: {n_strip}, Plate: {n_plate}, Junction: {n_junction}")

# ── Port setup ────────────────────────────────────────────────────
x_left = -L_STRIP / 2.0
x_right = +L_STRIP / 2.0

feed1 = find_edge_port_feed_edges(mesh, basis, port_x=x_left, strip_z=H_SUB)
feed2 = find_edge_port_feed_edges(mesh, basis, port_x=x_right, strip_z=H_SUB)
signs1 = compute_feed_signs(mesh, basis, feed1)
signs2 = compute_feed_signs(mesh, basis, feed2)
port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

print(f"\nPort1 feed indices: {feed1}, signs: {signs1}")
print(f"Port2 feed indices: {feed2}, signs: {signs2}")

# Check what type of basis functions the feed edges are
for idx in feed1:
    tp, tm = basis.t_plus[idx], basis.t_minus[idx]
    z_tp = mesh.vertices[mesh.triangles[tp], 2]
    z_tm = mesh.vertices[mesh.triangles[tm], 2]
    print(f"  Feed1 idx={idx}: T+z={z_tp}, T-z={z_tm}")

# ── Build QS solver manually to inspect matrices ──────────────────
exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
config = SimulationConfig(
    frequency=1e9, excitation=exc, quad_order=4, backend='auto',
    layer_stack=stack, source_layer_name='FR4',
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

# Build QS solver (no probe feeds)
qs = QuasiStaticSolver(sim, [port1, port2], probe_feeds=False)

print(f"\n=== Matrix diagnostics ===")
print(f"L_raw: shape={qs.L_raw.shape}, symmetric={np.allclose(qs.L_raw, qs.L_raw.T, atol=1e-12)}")
print(f"  |L_raw| range: [{np.abs(qs.L_raw).min():.4e}, {np.abs(qs.L_raw).max():.4e}]")
print(f"  L_raw diagonal: mean={np.mean(np.abs(np.diag(qs.L_raw))):.4e}")

# Check L_raw by type
L_diag = np.abs(np.diag(qs.L_raw))
if strip_idx:
    print(f"  L_diag strip:    mean={np.mean(L_diag[strip_idx]):.4e}, "
          f"range=[{np.min(L_diag[strip_idx]):.4e}, {np.max(L_diag[strip_idx]):.4e}]")
if plate_idx:
    print(f"  L_diag plate:    mean={np.mean(L_diag[plate_idx]):.4e}, "
          f"range=[{np.min(L_diag[plate_idx]):.4e}, {np.max(L_diag[plate_idx]):.4e}]")
if junc_idx:
    print(f"  L_diag junction: mean={np.mean(L_diag[junc_idx]):.4e}, "
          f"range=[{np.min(L_diag[junc_idx]):.4e}, {np.max(L_diag[junc_idx]):.4e}]")

print(f"\nG_s: shape={qs.G_s.shape}, symmetric={np.allclose(qs.G_s, qs.G_s.T, atol=1e-12)}")
print(f"  |G_s| range: [{np.abs(qs.G_s).min():.4e}, {np.abs(qs.G_s).max():.4e}]")
print(f"  G_s diagonal: mean={np.mean(np.abs(np.diag(qs.G_s))):.4e}")

# Check G_s for triangles near ground (z=0)
tri_z = np.array([mesh.vertices[mesh.triangles[t], 2].mean() for t in range(len(mesh.triangles))])
ground_tri = np.where(tri_z < H_SUB / 2)[0]
strip_tri = np.where(tri_z > H_SUB / 2)[0]
print(f"\n  Triangles near ground (z<{H_SUB/2:.4f}): {len(ground_tri)}")
print(f"  Triangles on strip (z>{H_SUB/2:.4f}): {len(strip_tri)}")

if len(ground_tri) > 0:
    gs_ground = np.abs(qs.G_s[np.ix_(ground_tri, ground_tri)])
    print(f"  G_s ground-ground: mean={gs_ground.mean():.4e}, max={gs_ground.max():.4e}")
    gs_ground_strip = np.abs(qs.G_s[np.ix_(ground_tri, strip_tri)])
    print(f"  G_s ground-strip:  mean={gs_ground_strip.mean():.4e}, max={gs_ground_strip.max():.4e}")

D_dense = qs.D.toarray()
P = D_dense.T @ qs.G_s @ D_dense
print(f"\nP matrix: shape={P.shape}")
P_diag = np.abs(np.diag(P))
if strip_idx:
    print(f"  P_diag strip:    mean={np.mean(P_diag[strip_idx]):.4e}")
if plate_idx:
    print(f"  P_diag plate:    mean={np.mean(P_diag[plate_idx]):.4e}")
if junc_idx:
    print(f"  P_diag junction: mean={np.mean(P_diag[junc_idx]):.4e}")

# ── Excitation vector ─────────────────────────────────────────────
V1 = port1.build_excitation_vector(basis)
V2 = port2.build_excitation_vector(basis)
print(f"\nV1: nnz={np.count_nonzero(V1)}, nonzero indices={np.where(np.abs(V1)>0)[0].tolist()}")
print(f"V1 values: {V1[np.abs(V1)>0]}")

# ── Solve at a few frequencies ────────────────────────────────────
print("\n=== QS extraction (no probe feeds) ===")
freqs = [0.5e9, 1e9, 2e9, 3e9, 5e9]
results = qs.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} "
      f"{'Z11 real':>10} {'Z11 imag':>10} {'Z21 real':>10} {'Z21 imag':>10}")
print("-" * 72)
for r in results:
    S = r.S_matrix
    Z = r.Z_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f} "
          f"{Z[0,0].real:10.2f} {Z[0,0].imag:10.2f} "
          f"{Z[1,0].real:10.2f} {Z[1,0].imag:10.2f}")

# ── For comparison: flat mesh with same solver ────────────────────
print("\n=== Flat mesh comparison ===")
PORT1_X = -L_STRIP / 2.0 + 1.0e-3
PORT2_X = +L_STRIP / 2.0 - 1.0e-3

from pyMoM3d.mom.excitation import find_feed_edges

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
sim_f = Simulation(config_f, mesh=mesh_flat, reporter=SilentReporter())

qs_flat = QuasiStaticSolver(sim_f, [port1f, port2f], probe_feeds=False)
results_flat = qs_flat.extract(freqs)

print(f"Flat mesh: {len(mesh_flat.triangles)} tri, {basis_flat.num_basis} RWG")
print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10}")
print("-" * 32)
for r in results_flat:
    S = r.S_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f}")

# ── Comparison: edge port WITH probe feeds ────────────────────────
print("\n=== Edge port + probe feeds (reference) ===")
qs_probe = QuasiStaticSolver(sim, [port1, port2], probe_feeds=True)
results_probe = qs_probe.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10}")
print("-" * 32)
for r in results_probe:
    S = r.S_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f}")

print("\nDone.")
