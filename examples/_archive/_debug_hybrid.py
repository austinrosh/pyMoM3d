"""Debug: examine hybrid matrix block structure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Layer, LayerStack, c0, eta0,
    WireMesh, compute_wire_connectivity,
    HybridBasis,
)
from pyMoM3d.wire.hybrid import fill_hybrid_matrix, HybridBasisAdapter
from pyMoM3d.wire.kernels import fill_wire_wire, fill_wire_surface
from pyMoM3d.mom.excitation import StripDeltaGapExcitation
from pyMoM3d.wire.wire_basis import WireSegment

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 30e-3
TEL = 2.0e-3
WIRE_RADIUS = 0.25e-3

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_plate_with_feeds(
    width=L, height=W,
    feed_x_list=[-L/2, L/2],
    center=(0.0, 0.0, H),
)
basis = compute_rwg_connectivity(mesh)
N_s = basis.num_basis

# Two probes
x1, x2 = -L/2, L/2
wire1 = WireMesh.vertical_probe(x1, 0.0, 0.0, H, WIRE_RADIUS, 3)
wire2 = WireMesh.vertical_probe(x2, 0.0, 0.0, H, WIRE_RADIUS, 3)

n1 = wire1.num_nodes
merged_nodes = np.vstack([wire1.nodes, wire2.nodes])
merged_segments = list(wire1.segments)
for seg in wire2.segments:
    merged_segments.append(WireSegment(
        node_start=seg.node_start + n1, node_end=seg.node_end + n1,
        length=seg.length, direction=seg.direction.copy(), radius=seg.radius,
    ))
wire_mesh = WireMesh(nodes=merged_nodes, segments=merged_segments)
wire_basis = compute_wire_connectivity(wire_mesh)
N_w = wire_basis.num_basis

print(f"N_s = {N_s}, N_w = {N_w}")

freq = 1e9
k = 2 * np.pi * freq / c0
eta = eta0

# --- Assemble individual blocks ---
print(f"\nk = {k:.4f}, eta = {eta:.2f}")

# Z_ww
Z_ww = np.zeros((N_w, N_w), dtype=np.complex128)
fill_wire_wire(Z_ww, wire_basis, wire_mesh, k, eta)
print(f"\nZ_ww block ({N_w}x{N_w}):")
print(f"  |Z_ww|_max = {np.abs(Z_ww).max():.4e}")
print(f"  |Z_ww|_min = {np.abs(Z_ww[Z_ww != 0]).min():.4e}")
print(f"  Z_ww diagonal: {[f'{z:.4e}' for z in np.diag(Z_ww)]}")

# Z_ws
Z_ws = np.zeros((N_w, N_s), dtype=np.complex128)
fill_wire_surface(Z_ws, wire_basis, wire_mesh, basis, mesh, k, eta)
print(f"\nZ_ws block ({N_w}x{N_s}):")
print(f"  |Z_ws|_max = {np.abs(Z_ws).max():.4e}")
print(f"  |Z_ws|_min (nonzero) = {np.abs(Z_ws[np.abs(Z_ws) > 1e-30]).min():.4e}")
print(f"  mean |Z_ws| = {np.abs(Z_ws).mean():.4e}")
print(f"  nonzero count: {np.count_nonzero(np.abs(Z_ws) > 1e-20)}/{N_w*N_s}")

# Z_ss via fill_matrix (using the layered operator)
config = SimulationConfig(
    frequency=freq,
    excitation=StripDeltaGapExcitation([], voltage=1.0),
    quad_order=4, layer_stack=stack, source_layer_name='FR4', backend='auto',
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
op = sim._make_operator(freq)
from pyMoM3d.mom.assembly import fill_matrix
Z_ss = fill_matrix(op, basis, mesh, k, eta, quad_order=4, backend='auto')
print(f"\nZ_ss block ({N_s}x{N_s}):")
print(f"  |Z_ss|_max = {np.abs(Z_ss).max():.4e}")
print(f"  |Z_ss| diagonal: min={np.abs(np.diag(Z_ss)).min():.4e}, max={np.abs(np.diag(Z_ss)).max():.4e}")
print(f"  cond(Z_ss) = {np.linalg.cond(Z_ss):.4e}")

# Full hybrid matrix
N = N_s + N_w
Z = np.zeros((N, N), dtype=np.complex128)
Z[:N_s, :N_s] = Z_ss
Z[N_s:, N_s:] = Z_ww
Z[N_s:, :N_s] = Z_ws
Z[:N_s, N_s:] = Z_ws.T

print(f"\nFull hybrid Z ({N}x{N}):")
print(f"  cond(Z) = {np.linalg.cond(Z):.4e}")

# Block ratio
ratio_ws_ss = np.abs(Z_ws).max() / np.abs(Z_ss).max()
ratio_ww_ss = np.abs(Z_ww).max() / np.abs(Z_ss).max()
print(f"  |Z_ws|/|Z_ss| = {ratio_ws_ss:.4e}")
print(f"  |Z_ww|/|Z_ss| = {ratio_ww_ss:.4e}")

# Check the RHS (excitation vector)
adapter = HybridBasisAdapter(HybridBasis(basis, wire_basis, wire_mesh))
print(f"\nAdapter edge_length: min={adapter.edge_length.min():.4e}, max={adapter.edge_length.max():.4e}")
print(f"  Surface edge lengths: min={basis.edge_length.min():.4e}, max={basis.edge_length.max():.4e}")
print(f"  Wire effective lengths: {adapter.edge_length[N_s:]}")

# Check proximity: wire tip to nearest triangle
wire_tip1 = wire1.nodes[-1]  # top of wire 1
wire_tip2 = wire2.nodes[-1]  # top of wire 2
for i, tip in enumerate([wire_tip1, wire_tip2]):
    # Find nearest triangle centroid
    centroids = np.mean(mesh.vertices[mesh.triangles], axis=1)
    dists = np.linalg.norm(centroids - tip, axis=1)
    print(f"\nWire {i+1} tip at {tip}")
    print(f"  Nearest triangle centroid: dist = {dists.min()*1e3:.3f} mm")
    # Nearest vertex
    vdists = np.linalg.norm(mesh.vertices - tip, axis=1)
    print(f"  Nearest mesh vertex: dist = {vdists.min()*1e3:.3f} mm")
