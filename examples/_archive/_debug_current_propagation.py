"""Debug: Does current propagate along the strip with multilayer Z?

Solve Z*I = V with port at one end, check if current reaches the other end.
Compare multilayer vs free-space Green's function.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor,
    Layer, LayerStack, c0, eta0,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators.efie import EFIEOperator
from pyMoM3d.greens.layered import LayeredGreensFunction
from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 30e-3
TEL = 1.5e-3  # finer mesh

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

margin = TEL / 2.0
port_x = -L / 2.0 + margin

mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_plate_with_feeds(
    width=L, height=W,
    feed_x_list=[port_x],
    center=(0.0, 0.0, H),
)
basis = compute_rwg_connectivity(mesh)
N = basis.num_basis
stats = mesh.get_statistics()
print(f"Mesh: {stats['num_triangles']} tris, {N} RWG basis, TEL={TEL*1e3:.1f} mm")

# Find port edges at the feed line position
feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
signs = compute_feed_signs(mesh, basis, feed_edges)
print(f"Port at x={port_x*1e3:.2f} mm, {len(feed_edges)} feed edges")

# Build excitation vector
V = np.zeros(N, dtype=np.complex128)
for idx, sign in zip(feed_edges, signs):
    V[idx] = sign / basis.edge_length[idx]

print(f"  V: {np.count_nonzero(V)} nonzero entries, max |V|={np.abs(V).max():.4e}")

freq = 2e9
k = 2 * np.pi * freq / c0

# Free-space Z
print(f"\nAssembling free-space Z... ", end='', flush=True)
op_fs = EFIEOperator()
Z_fs = fill_matrix(op_fs, basis, mesh, k, eta0, quad_order=4, backend='numpy')
print(f"done. cond = {np.linalg.cond(Z_fs):.2e}")

# Multilayer Z
print(f"Assembling multilayer Z... ", end='', flush=True)
gf = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='auto')
op_ml = MultilayerEFIEOperator(gf)
Z_ml = fill_matrix(op_ml, basis, mesh, k, eta0, quad_order=4, backend='auto')
print(f"done. cond = {np.linalg.cond(Z_ml):.2e}")

# Solve both
I_fs = np.linalg.solve(Z_fs, V)
I_ml = np.linalg.solve(Z_ml, V)

# Compute edge centers
verts = mesh.vertices
edges = mesh.edges
edge_x = np.empty(N)
for n in range(N):
    ei = basis.edge_index[n]
    v0, v1 = edges[ei]
    edge_x[n] = 0.5 * (verts[v0, 0] + verts[v1, 0])

# Get edge direction (x-component) to identify x-directed edges
edge_dir_x = np.empty(N)
for n in range(N):
    ei = basis.edge_index[n]
    v0, v1 = edges[ei]
    d = verts[v1] - verts[v0]
    ln = np.linalg.norm(d)
    edge_dir_x[n] = abs(d[0] / ln) if ln > 1e-30 else 0

# Show ALL basis function currents sorted by x, with direction info
print(f"\n{'x (mm)':>8}  {'|I_ml|':>12}  {'|I_fs|':>12}  {'ratio ml/fs':>12}  {'x-dir':>6}")
print("-" * 60)

sort_idx = np.argsort(edge_x)
for i in sort_idx:
    x = edge_x[i] * 1e3
    i_ml = abs(I_ml[i])
    i_fs = abs(I_fs[i])
    ratio = i_ml / i_fs if i_fs > 1e-30 else float('inf')
    print(f"  {x:>6.1f}  {i_ml:>12.4e}  {i_fs:>12.4e}  {ratio:>12.4f}  {edge_dir_x[i]:>6.2f}")

# Summary: max current at each distance bin
print(f"\n\n--- Binned max |I| along strip ---")
bins = np.arange(-16, 16, 2)
print(f"{'x_bin (mm)':>10}  {'max |I_ml|':>12}  {'max |I_fs|':>12}  {'ratio':>8}")
print("-" * 50)
for xb in bins:
    mask = (edge_x * 1e3 >= xb) & (edge_x * 1e3 < xb + 2)
    if np.any(mask):
        max_ml = np.abs(I_ml[mask]).max()
        max_fs = np.abs(I_fs[mask]).max()
        ratio = max_ml / max_fs if max_fs > 1e-30 else float('inf')
        print(f"  {xb:>8.0f}  {max_ml:>12.4e}  {max_fs:>12.4e}  {ratio:>8.4f}")
