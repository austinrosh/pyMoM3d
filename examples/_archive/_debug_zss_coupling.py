"""Debug: check if Z_ss supports TL mode propagation.

Examines how Z_ss coupling decays with distance along the strip.
If coupling dies off too fast, TL mode can't propagate regardless of port model.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Layer, LayerStack, c0, eta0,
)
from pyMoM3d.mom.excitation import StripDeltaGapExcitation
from pyMoM3d.mom.assembly import fill_matrix

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 30e-3
TEL = 2.0e-3

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
N = basis.num_basis

freq = 2e9
k = 2 * np.pi * freq / c0

config = SimulationConfig(
    frequency=freq,
    excitation=StripDeltaGapExcitation([], voltage=1.0),
    quad_order=4, layer_stack=stack, source_layer_name='FR4', backend='auto',
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

# Also build a free-space Z matrix for comparison
from pyMoM3d.mom.operators.efie import EFIEOperator
op_fs = EFIEOperator()
Z_fs = fill_matrix(op_fs, basis, mesh, k, eta0, quad_order=4, backend='numpy')

# Get layered Z matrix via the operator factory
from pyMoM3d.greens.layered import LayeredGreensFunction
from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator

gf = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='auto')
op_ml = MultilayerEFIEOperator(gf)
Z_ml = fill_matrix(op_ml, basis, mesh, k, eta0, quad_order=4, backend='auto')

# Compute edge midpoints for distance calculation
verts = mesh.vertices
edges = mesh.edges
edge_centers = np.empty((N, 3))
for n in range(N):
    ei = basis.edge_index[n]
    v0, v1 = edges[ei]
    edge_centers[n] = 0.5 * (verts[v0] + verts[v1])

# Pick a reference basis function near the center of the strip
# (find one with x near 0)
dists_to_center = np.abs(edge_centers[:, 0])
ref_idx = np.argmin(dists_to_center)
print(f"Reference basis function {ref_idx} at x={edge_centers[ref_idx, 0]*1e3:.2f} mm")

# For each other basis function, compute |Z[ref, n]| vs distance
x_dists = np.abs(edge_centers[:, 0] - edge_centers[ref_idx, 0])

# Sort by distance
sort_idx = np.argsort(x_dists)

print(f"\n{'dist (mm)':>10}  {'|Z_ml|':>12}  {'|Z_fs|':>12}  {'ratio':>8}  {'x (mm)':>8}")
print("-" * 65)

# Print Z coupling decay
prev_d = -1
for i in sort_idx:
    d = x_dists[i] * 1e3  # mm
    if d - prev_d < 0.5:  # skip closely spaced entries
        continue
    prev_d = d
    z_ml = abs(Z_ml[ref_idx, i])
    z_fs = abs(Z_fs[ref_idx, i])
    ratio = z_ml / z_fs if z_fs > 1e-30 else float('inf')
    print(f"  {d:>8.2f}  {z_ml:>12.4e}  {z_fs:>12.4e}  {ratio:>8.3f}  {edge_centers[i,0]*1e3:>8.2f}")

# Also compare the full solution: excite at one end, see current distribution
print("\n\n--- Current distribution from delta-gap at x = -L/2 ---")

from pyMoM3d.mom.excitation import find_feed_edges, compute_feed_signs

margin = TEL / 2
port_x = -L/2 + margin
feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
signs = compute_feed_signs(mesh, basis, feed_edges)

# Build excitation vector
V = np.zeros(N, dtype=np.complex128)
for idx, sign in zip(feed_edges, signs):
    V[idx] = sign / basis.edge_length[idx]

# Solve with multilayer Z
I_ml = np.linalg.solve(Z_ml, V)
# Solve with free-space Z
I_fs = np.linalg.solve(Z_fs, V)

# Get x-directed current at each edge (project onto x-axis)
directions = np.empty((N, 3))
for n in range(N):
    ei = basis.edge_index[n]
    v0, v1 = edges[ei]
    d = verts[v1] - verts[v0]
    ln = np.linalg.norm(d)
    directions[n] = d / ln if ln > 1e-30 else 0

x_proj = np.abs(directions[:, 0])
x_mask = x_proj > 0.3  # x-directed edges

print(f"\n{'x (mm)':>8}  {'|I_ml| (mA)':>14}  {'|I_fs| (mA)':>14}  {'ratio':>8}")
print("-" * 55)

# Bin by x position
x_vals = edge_centers[x_mask, 0]
I_ml_vals = np.abs(I_ml[x_mask]) * 1e3
I_fs_vals = np.abs(I_fs[x_mask]) * 1e3

sort_x = np.argsort(x_vals)
prev_x = -999
for j in sort_x:
    x = x_vals[j] * 1e3
    if abs(x - prev_x) < 1.5:
        continue
    prev_x = x
    ratio = I_ml_vals[j] / I_fs_vals[j] if I_fs_vals[j] > 1e-10 else float('inf')
    print(f"  {x:>6.1f}  {I_ml_vals[j]:>14.3f}  {I_fs_vals[j]:>14.3f}  {ratio:>8.3f}")
