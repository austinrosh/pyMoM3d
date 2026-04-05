"""Compare Y11 from different Green's function backends.

If layer recursion gives correct TL behavior but Strata DCIM doesn't,
the DCIM accuracy at moderate rho is the problem.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor,
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
L = 20e-3  # shorter for speed
TEL = 1.5e-3

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

# Build mesh
mesher = GmshMesher(target_edge_length=TEL)
margin = TEL / 2
port_x = -L/2 + margin
mesh = mesher.mesh_plate_with_feeds(
    width=L, height=W, feed_x_list=[port_x],
    center=(0.0, 0.0, H),
)
basis = compute_rwg_connectivity(mesh)
feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
signs = compute_feed_signs(mesh, basis, feed_edges)
N = basis.num_basis
stub_length = L/2 - port_x

print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {N} RWG")
print(f"Stub: {stub_length*1e3:.1f} mm, {len(feed_edges)} feed edges")

# Build excitation vector
V = np.zeros(N, dtype=np.complex128)
for idx, sign in zip(feed_edges, signs):
    V[idx] = sign / basis.edge_length[idx]

freq = 2e9
k = 2 * np.pi * freq / c0

# --- Compare backends ---
backends_to_try = ['strata', 'layer_recursion']

for backend_name in backends_to_try:
    print(f"\n--- Backend: {backend_name} ---")
    try:
        gf = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend=backend_name)
    except Exception as e:
        print(f"  Not available: {e}")
        continue

    op = MultilayerEFIEOperator(gf)

    # Use numpy backend for layer_recursion (no C++ kernel for it)
    asm_backend = 'auto' if backend_name == 'strata' else 'numpy'
    Z = fill_matrix(op, basis, mesh, k, eta0, quad_order=4, backend=asm_backend)

    I = np.linalg.solve(Z, V)
    I_term = sum(I[idx] * basis.edge_length[idx] * sign for idx, sign in zip(feed_edges, signs))
    Y11 = I_term
    Z_in = 1.0 / Y11

    print(f"  cond(Z) = {np.linalg.cond(Z):.2e}")
    print(f"  Y11 = {Y11.real:.6f} + j {Y11.imag:.6f}")
    print(f"  Z_in = {Z_in.real:.2f} + j {Z_in.imag:.2f} Ohm")

    # Check Z matrix structure
    diag = np.abs(np.diag(Z))
    print(f"  |Z_diag| range: [{diag.min():.2e}, {diag.max():.2e}]")

    # Check off-diagonal coupling decay
    # Find two basis functions at different x-positions
    edge_x = np.empty(N)
    for n in range(N):
        ei = basis.edge_index[n]
        v0, v1 = mesh.edges[ei]
        edge_x[n] = 0.5 * (mesh.vertices[v0, 0] + mesh.vertices[v1, 0])

    ref_idx = np.argmin(np.abs(edge_x - port_x))
    dists = np.abs(edge_x - edge_x[ref_idx]) * 1e3  # mm

    print(f"\n  Off-diagonal coupling from ref edge at x={edge_x[ref_idx]*1e3:.1f}mm:")
    print(f"  {'dist(mm)':>10}  {'|Z_mn|':>12}  {'|Z_mn|/|Z_nn|':>14}")
    for dist_bin in [0, 1, 2, 5, 8, 10, 15]:
        mask = (dists >= dist_bin) & (dists < dist_bin + 1)
        if np.any(mask):
            z_vals = np.abs(Z[ref_idx, mask])
            ratio = z_vals / diag[ref_idx]
            print(f"  {dist_bin:>8}-{dist_bin+1}  {z_vals.mean():>12.4e}  {ratio.mean():>14.6f}")

# --- Also try empymod with lossy ground (sigma=1e8 instead of PEC) ---
print(f"\n--- Empymod with lossy ground (sigma=1e8) ---")
stack_lossy = LayerStack([
    Layer('ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, sigma=1e8),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

try:
    gf_emp = LayeredGreensFunction(stack_lossy, freq, source_layer_name='FR4', backend='empymod')
    op_emp = MultilayerEFIEOperator(gf_emp)
    Z_emp = fill_matrix(op_emp, basis, mesh, k, eta0, quad_order=4, backend='numpy')

    I_emp = np.linalg.solve(Z_emp, V)
    I_term_emp = sum(I_emp[idx] * basis.edge_length[idx] * sign
                     for idx, sign in zip(feed_edges, signs))
    Y11_emp = I_term_emp
    Z_in_emp = 1.0 / Y11_emp

    print(f"  cond(Z) = {np.linalg.cond(Z_emp):.2e}")
    print(f"  Y11 = {Y11_emp.real:.6f} + j {Y11_emp.imag:.6f}")
    print(f"  Z_in = {Z_in_emp.real:.2f} + j {Z_in_emp.imag:.2f} Ohm")

    diag_emp = np.abs(np.diag(Z_emp))
    print(f"  |Z_diag| range: [{diag_emp.min():.2e}, {diag_emp.max():.2e}]")

    # Compare Z matrices
    diff = np.linalg.norm(Z - Z_emp) / np.linalg.norm(Z_emp)
    print(f"\n  |Z_strata - Z_emp| / |Z_emp| = {diff:.4e}")

    diag_ratio = np.abs(np.diag(Z)) / np.abs(np.diag(Z_emp))
    print(f"  Z diagonal ratio (strata/empymod): [{diag_ratio.min():.4f}, {diag_ratio.max():.4f}]")

except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
