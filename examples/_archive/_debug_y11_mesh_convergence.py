"""Check Y11 convergence with mesh refinement.

If Y11 sign/magnitude changes dramatically with mesh density,
the coarse mesh is the problem. If it stays wrong even with fine mesh,
the formulation or Green's function is the issue.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor,
    Layer, LayerStack, c0,
    microstrip_z0_hammerstad,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 30e-3

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

# Test at a single frequency first
freq = 1e9
k = 2*np.pi*freq / c0
beta_ref = 2*np.pi*freq * np.sqrt(eps_eff_ref) / c0

# Sweep mesh density
tels = [2.0e-3, 1.5e-3, 1.0e-3, 0.75e-3, 0.5e-3]

print(f"\nf = {freq/1e9:.1f} GHz, k = {k:.2f} rad/m")
print(f"beta_ref = {beta_ref:.2f} rad/m")
print(f"lambda_eff = {2*np.pi/beta_ref*1e3:.1f} mm")
print(f"\n{'TEL(mm)':>8} {'N_tri':>6} {'N_rwg':>6} {'N_feed':>6} "
      f"{'Re(Y11)':>12} {'Im(Y11)':>12} {'Im_ref':>12} {'err':>8}")
print("-" * 85)

for tel in tels:
    mesher = GmshMesher(target_edge_length=tel)
    margin = tel / 2
    port_x = -L/2 + margin

    mesh = mesher.mesh_plate_with_feeds(
        width=L, height=W, feed_x_list=[port_x],
        center=(0.0, 0.0, H),
    )
    basis = compute_rwg_connectivity(mesh)
    feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
    signs = compute_feed_signs(mesh, basis, feed_edges)
    port = Port(name='P1', feed_basis_indices=feed_edges, feed_signs=signs)
    exc = StripDeltaGapExcitation(feed_basis_indices=feed_edges, voltage=1.0)

    config = SimulationConfig(
        frequency=freq, excitation=exc,
        source_layer_name='FR4', backend='auto', quad_order=4,
        layer_stack=stack,
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    ext = NetworkExtractor(sim, [port])

    stub_len = L/2 - port_x
    bL = beta_ref * stub_len
    Y11_ref = -1j / Z0_ref * np.cos(bL) / np.sin(bL)

    results = ext.extract([freq])
    Y11 = results[0].Y_matrix[0, 0]

    n_tri = mesh.get_statistics()['num_triangles']
    err = abs(Y11.imag - Y11_ref.imag) / max(abs(Y11_ref.imag), 1e-12) * 100

    print(f"  {tel*1e3:>6.2f} {n_tri:>6} {basis.num_basis:>6} {len(feed_edges):>6} "
          f"{Y11.real:>12.6f} {Y11.imag:>12.6f} {Y11_ref.imag:>12.6f} {err:>7.1f}%")

# --- Also check the excitation vector ---
print(f"\n--- Excitation vector analysis ---")
tel = 1.0e-3
mesher = GmshMesher(target_edge_length=tel)
margin = tel / 2
port_x = -L/2 + margin

mesh = mesher.mesh_plate_with_feeds(
    width=L, height=W, feed_x_list=[port_x],
    center=(0.0, 0.0, H),
)
basis = compute_rwg_connectivity(mesh)
feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
signs = compute_feed_signs(mesh, basis, feed_edges)

print(f"Feed edges: {feed_edges}")
print(f"Feed signs: {signs}")
print(f"Feed edge lengths: {basis.edge_length[feed_edges]}")

# Check which direction the feed edges run
verts = mesh.vertices
edges = mesh.edges
for idx in feed_edges:
    ei = basis.edge_index[idx]
    v0, v1 = edges[ei]
    p0, p1 = verts[v0], verts[v1]
    direction = p1 - p0
    midpoint = 0.5 * (p0 + p1)
    print(f"  Edge {idx}: ({p0[0]*1e3:.2f},{p0[1]*1e3:.2f},{p0[2]*1e3:.2f}) -> "
          f"({p1[0]*1e3:.2f},{p1[1]*1e3:.2f},{p1[2]*1e3:.2f}), "
          f"dir=({direction[0]*1e3:.2f},{direction[1]*1e3:.2f},{direction[2]*1e3:.2f})")

# Check the V vector
port_obj = Port(name='P1', feed_basis_indices=feed_edges, feed_signs=signs)
V = port_obj.build_excitation_vector(basis.num_basis, basis.edge_length)
nonzero = np.nonzero(V)[0]
print(f"\nExcitation vector: {len(nonzero)} nonzero entries")
for idx in nonzero:
    print(f"  V[{idx}] = {V[idx]:.6f}")
