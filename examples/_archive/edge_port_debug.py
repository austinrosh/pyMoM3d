"""Debug edge port full-wave assembly.

Compare Z-matrix properties for flat mesh vs edge port mesh to identify
where the conditioning breaks down.
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
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator
from pyMoM3d.greens.layered import LayeredGreensFunction
from pyMoM3d.utils.constants import c0, eta0

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

FREQ = 5e9  # where flat mesh works well

# ── Flat mesh (baseline) ─────────────────────────────────────────
mesher = GmshMesher(target_edge_length=TEL)
mesh_flat = mesher.mesh_plate_with_feeds(
    width=L_STRIP, height=W_STRIP,
    feed_x_list=[-L_STRIP/2+1e-3, L_STRIP/2-1e-3],
    center=(0, 0, H_SUB),
)
basis_flat = compute_rwg_connectivity(mesh_flat)

print("=== Flat mesh ===")
print(f"  {len(mesh_flat.triangles)} tri, {basis_flat.num_basis} RWG")
z_flat = mesh_flat.vertices[:, 2]
print(f"  z range: [{z_flat.min():.6f}, {z_flat.max():.6f}]")

gf = LayeredGreensFunction(stack, FREQ, source_layer_name='FR4')
k = complex(gf.wavenumber)
eta = complex(gf.wave_impedance)

op = MultilayerEFIEOperator(gf)
Z_flat = fill_matrix(op, basis_flat, mesh_flat, k, eta,
                     quad_order=4, backend='numpy')

print(f"  Z_flat: shape={Z_flat.shape}, cond={np.linalg.cond(Z_flat):.2e}")
print(f"  |Z_flat| range: [{np.abs(Z_flat).min():.4e}, {np.abs(Z_flat).max():.4e}]")
print(f"  Z_flat diagonal: mean|diag|={np.mean(np.abs(np.diag(Z_flat))):.4e}")
print(f"  Symmetry: |Z-Z^T|/|Z|={np.linalg.norm(Z_flat-Z_flat.T)/np.linalg.norm(Z_flat):.2e}")

# ── Edge port mesh ───────────────────────────────────────────────
mesh_edge = mesher.mesh_microstrip_with_edge_ports(
    width=W_STRIP, length=L_STRIP, substrate_height=H_SUB,
    port_edges=['left', 'right'],
)
basis_edge = compute_rwg_connectivity(mesh_edge)

print("\n=== Edge port mesh ===")
print(f"  {len(mesh_edge.triangles)} tri, {basis_edge.num_basis} RWG")
z_edge = mesh_edge.vertices[:, 2]
print(f"  z range: [{z_edge.min():.6f}, {z_edge.max():.6f}]")

# Count basis functions by type
n_strip_basis = 0
n_plate_basis = 0
n_junction_basis = 0
for n in range(basis_edge.num_basis):
    tp = basis_edge.t_plus[n]
    tm = basis_edge.t_minus[n]
    z_tp = mesh_edge.vertices[mesh_edge.triangles[tp], 2]
    z_tm = mesh_edge.vertices[mesh_edge.triangles[tm], 2]
    tp_strip = np.allclose(z_tp, H_SUB, atol=1e-8)
    tm_strip = np.allclose(z_tm, H_SUB, atol=1e-8)
    if tp_strip and tm_strip:
        n_strip_basis += 1
    elif tp_strip != tm_strip:
        n_junction_basis += 1
    else:
        n_plate_basis += 1

print(f"  Strip basis: {n_strip_basis}, Plate basis: {n_plate_basis}, "
      f"Junction basis: {n_junction_basis}")

Z_edge = fill_matrix(op, basis_edge, mesh_edge, k, eta,
                     quad_order=4, backend='numpy')

print(f"  Z_edge: shape={Z_edge.shape}, cond={np.linalg.cond(Z_edge):.2e}")
print(f"  |Z_edge| range: [{np.abs(Z_edge).min():.4e}, {np.abs(Z_edge).max():.4e}]")
print(f"  Z_edge diagonal: mean|diag|={np.mean(np.abs(np.diag(Z_edge))):.4e}")
print(f"  Symmetry: |Z-Z^T|/|Z|={np.linalg.norm(Z_edge-Z_edge.T)/np.linalg.norm(Z_edge):.2e}")

# Analyze diagonal by basis type
diag = np.abs(np.diag(Z_edge))
strip_diag = []
plate_diag = []
junc_diag = []
for n in range(basis_edge.num_basis):
    tp = basis_edge.t_plus[n]
    tm = basis_edge.t_minus[n]
    z_tp = mesh_edge.vertices[mesh_edge.triangles[tp], 2]
    z_tm = mesh_edge.vertices[mesh_edge.triangles[tm], 2]
    tp_strip = np.allclose(z_tp, H_SUB, atol=1e-8)
    tm_strip = np.allclose(z_tm, H_SUB, atol=1e-8)
    if tp_strip and tm_strip:
        strip_diag.append(diag[n])
    elif tp_strip != tm_strip:
        junc_diag.append(diag[n])
    else:
        plate_diag.append(diag[n])

if strip_diag:
    print(f"\n  Strip |Z_nn|: mean={np.mean(strip_diag):.4e}, "
          f"range=[{min(strip_diag):.4e}, {max(strip_diag):.4e}]")
if plate_diag:
    print(f"  Plate |Z_nn|: mean={np.mean(plate_diag):.4e}, "
          f"range=[{min(plate_diag):.4e}, {max(plate_diag):.4e}]")
if junc_diag:
    print(f"  Junc  |Z_nn|: mean={np.mean(junc_diag):.4e}, "
          f"range=[{min(junc_diag):.4e}, {max(junc_diag):.4e}]")

# ── Check with auto (C++) backend ─────────────────────────────────
print("\n=== C++ backend comparison ===")
Z_edge_cpp = fill_matrix(op, basis_edge, mesh_edge, k, eta,
                         quad_order=4, backend='auto')
print(f"  Z_edge (auto): cond={np.linalg.cond(Z_edge_cpp):.2e}")
print(f"  |Z_cpp - Z_np| / |Z_np| = "
      f"{np.linalg.norm(Z_edge_cpp - Z_edge) / np.linalg.norm(Z_edge):.2e}")

# ── Test solve with edge port ─────────────────────────────────────
x_left = -L_STRIP / 2.0
x_right = +L_STRIP / 2.0
feed1 = find_edge_port_feed_edges(mesh_edge, basis_edge,
                                   port_x=x_left, strip_z=H_SUB)
feed2 = find_edge_port_feed_edges(mesh_edge, basis_edge,
                                   port_x=x_right, strip_z=H_SUB)
signs1 = compute_feed_signs(mesh_edge, basis_edge, feed1)
signs2 = compute_feed_signs(mesh_edge, basis_edge, feed2)
port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

exc1 = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
V1 = exc1.compute_voltage_vector(basis_edge, mesh_edge, float(np.real(k)))
print(f"\nV1: nnz={np.count_nonzero(V1)}, |V1|={np.linalg.norm(V1):.4e}")
print(f"  V1 nonzero indices: {np.where(np.abs(V1) > 0)[0].tolist()}")
print(f"  Feed1 indices: {feed1}")

# Solve with numpy backend Z-matrix
I = np.linalg.solve(Z_edge, V1)
print(f"\n  I solution norm: {np.linalg.norm(I):.4e}")

# Terminal current
I_term1 = port1.terminal_current(I, basis_edge)
I_term2 = port2.terminal_current(I, basis_edge)
print(f"  I_term1 = {I_term1:.4e}")
print(f"  I_term2 = {I_term2:.4e}")
print(f"  |I_term2/I_term1| = {abs(I_term2)/max(abs(I_term1),1e-30):.4e}")

# Y and S parameters
Y11 = I_term1 / 1.0
Y21 = I_term2 / 1.0
print(f"\n  Y11 = {Y11:.4e}")
print(f"  Y21 = {Y21:.4e}")
if abs(Y21) > 1e-30:
    print(f"  S21 ~ {20*np.log10(abs(2*Y21*50/(1+Y11*50+Y21*50))):.1f} dB (approx)")

print("\nDone.")
