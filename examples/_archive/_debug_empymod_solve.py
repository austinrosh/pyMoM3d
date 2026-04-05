"""Compare Strata vs Empymod Z matrices and current propagation.

If Empymod gives correct Z and current propagates, Strata DCIM is the bug.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Layer, LayerStack, c0, eta0,
    microstrip_z0_hammerstad,
)
from pyMoM3d.mom.excitation import find_feed_edges, compute_feed_signs
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.greens.layered import LayeredGreensFunction
from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 20e-3  # Shorter strip for speed with empymod
TEL = 3.0e-3  # Coarser mesh for speed

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

margin = TEL / 2
port_x = -L/2 + margin

mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_plate_with_feeds(
    width=L, height=W,
    feed_x_list=[port_x],
    center=(0.0, 0.0, H),
)
basis = compute_rwg_connectivity(mesh)
N = basis.num_basis
print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {N} RWG")

feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
signs = compute_feed_signs(mesh, basis, feed_edges)
print(f"Port at x={port_x*1e3:.2f} mm, {len(feed_edges)} feed edges")

V = np.zeros(N, dtype=np.complex128)
for idx, sign in zip(feed_edges, signs):
    V[idx] = sign / basis.edge_length[idx]

freq = 2e9
k = 2 * np.pi * freq / c0

# --- Compute edge x-positions ---
verts = mesh.vertices
edges = mesh.edges
edge_x = np.empty(N)
for n in range(N):
    ei = basis.edge_index[n]
    v0, v1 = edges[ei]
    edge_x[n] = 0.5 * (verts[v0, 0] + verts[v1, 0])

# --- Strata DCIM ---
print(f"\nAssembling Z with Strata DCIM...", flush=True)
gf_st = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='strata')
op_st = MultilayerEFIEOperator(gf_st)
Z_st = fill_matrix(op_st, basis, mesh, k, eta0, quad_order=4, backend='auto')
I_st = np.linalg.solve(Z_st, V)
print(f"  cond(Z_st) = {np.linalg.cond(Z_st):.2e}")

# --- Empymod Sommerfeld ---
print(f"Assembling Z with Empymod Sommerfeld...", flush=True)
gf_em = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='empymod')
op_em = MultilayerEFIEOperator(gf_em)
Z_em = fill_matrix(op_em, basis, mesh, k, eta0, quad_order=4, backend='numpy')
I_em = np.linalg.solve(Z_em, V)
print(f"  cond(Z_em) = {np.linalg.cond(Z_em):.2e}")

# --- Compare Z matrices ---
print(f"\nZ matrix comparison:")
print(f"  |Z_st|_max = {np.abs(Z_st).max():.4e}")
print(f"  |Z_em|_max = {np.abs(Z_em).max():.4e}")
print(f"  |Z_st - Z_em| / |Z_em| = {np.linalg.norm(Z_st - Z_em) / np.linalg.norm(Z_em):.4e}")

# Diagonal comparison
diag_st = np.diag(Z_st)
diag_em = np.diag(Z_em)
print(f"\n  Z diagonal comparison:")
print(f"    |diag_st| range: {np.abs(diag_st).min():.4e} to {np.abs(diag_st).max():.4e}")
print(f"    |diag_em| range: {np.abs(diag_em).min():.4e} to {np.abs(diag_em).max():.4e}")
print(f"    ratio |diag_st/diag_em|: {np.abs(diag_st/diag_em).min():.4f} to {np.abs(diag_st/diag_em).max():.4f}")

# --- Current comparison ---
print(f"\n--- Current distribution comparison ---")

# Sort by x
sort_idx = np.argsort(edge_x)
bins = np.arange(-12, 12, 2)

print(f"{'x_bin (mm)':>10}  {'max|I_st|':>12}  {'max|I_em|':>12}  {'ratio st/em':>12}")
print("-" * 55)
for xb in bins:
    mask = (edge_x * 1e3 >= xb) & (edge_x * 1e3 < xb + 2)
    if np.any(mask):
        max_st = np.abs(I_st[mask]).max()
        max_em = np.abs(I_em[mask]).max()
        ratio = max_st / max_em if max_em > 1e-30 else float('inf')
        print(f"  {xb:>8.0f}  {max_st:>12.4e}  {max_em:>12.4e}  {ratio:>12.4f}")

# Terminal current at port
I_term_st = sum(I_st[idx] * basis.edge_length[idx] * sign for idx, sign in zip(feed_edges, signs))
I_term_em = sum(I_em[idx] * basis.edge_length[idx] * sign for idx, sign in zip(feed_edges, signs))
print(f"\nTerminal current at port:")
print(f"  I_term_st = {I_term_st:.4e}")
print(f"  I_term_em = {I_term_em:.4e}")
print(f"  Y11_st = {I_term_st:.4e} / 1V = {I_term_st:.4e}")
print(f"  Y11_em = {I_term_em:.4e} / 1V = {I_term_em:.4e}")
print(f"  Z_in_st = {1/I_term_st:.4e}")
print(f"  Z_in_em = {1/I_term_em:.4e}")
