"""Test Y11 extraction using Port.from_vertex (probe feed) vs strip delta-gap.

Port.from_vertex models a coaxial probe at a single mesh vertex, exciting
all RWG basis functions connected to that vertex. This should give a
physically better excitation for microstrip than strip delta-gap.
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
TEL = 0.75e-3

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

# Build mesh with feed constraint
mesher = GmshMesher(target_edge_length=TEL)
margin = TEL / 2
port_x = -L/2 + margin

mesh = mesher.mesh_plate_with_feeds(
    width=L, height=W, feed_x_list=[port_x],
    center=(0.0, 0.0, H),
)
basis = compute_rwg_connectivity(mesh)
print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {basis.num_basis} RWG")

stub_length = L/2 - port_x

# --- Strip delta-gap port ---
feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
signs = compute_feed_signs(mesh, basis, feed_edges)
port_sdg = Port(name='SDG', feed_basis_indices=feed_edges, feed_signs=signs)
print(f"\nStrip delta-gap: {len(feed_edges)} edges")

# --- Vertex probe port (at center of feed line) ---
probe_pos = np.array([port_x, 0.0, H])
port_probe = Port.from_vertex(mesh, basis, probe_pos, name='Probe')
print(f"Vertex probe: {len(port_probe.feed_basis_indices)} edges connected")
print(f"  Indices: {port_probe.feed_basis_indices}")
print(f"  Signs:   {port_probe.feed_signs}")

# Print probe edge details
verts = mesh.vertices
edges = mesh.edges
for idx, sign in zip(port_probe.feed_basis_indices, port_probe.feed_signs):
    ei = basis.edge_index[idx]
    v0, v1 = edges[ei]
    p0, p1 = verts[v0], verts[v1]
    fvp = verts[basis.free_vertex_plus[idx]]
    fvm = verts[basis.free_vertex_minus[idx]]
    print(f"  Edge {idx}: edge ({p0[0]*1e3:.2f},{p0[1]*1e3:.2f}) -> ({p1[0]*1e3:.2f},{p1[1]*1e3:.2f}), "
          f"fvp=({fvp[0]*1e3:.2f},{fvp[1]*1e3:.2f}), "
          f"fvm=({fvm[0]*1e3:.2f},{fvm[1]*1e3:.2f}), sign={sign}")

# --- Frequency sweep ---
freqs = np.linspace(0.5, 6.0, 12) * 1e9

# SDG extraction
exc_sdg = StripDeltaGapExcitation(feed_basis_indices=feed_edges, voltage=1.0)
config_sdg = SimulationConfig(
    frequency=1e9, excitation=exc_sdg,
    source_layer_name='FR4', backend='auto', quad_order=4,
    layer_stack=stack,
)
sim_sdg = Simulation(config_sdg, mesh=mesh, reporter=SilentReporter())
ext_sdg = NetworkExtractor(sim_sdg, [port_sdg])

# Probe extraction
config_probe = SimulationConfig(
    frequency=1e9, excitation=exc_sdg,  # excitation doesn't matter for NetworkExtractor
    source_layer_name='FR4', backend='auto', quad_order=4,
    layer_stack=stack,
)
sim_probe = Simulation(config_probe, mesh=mesh, reporter=SilentReporter())
ext_probe = NetworkExtractor(sim_probe, [port_probe])

print(f"\nSweeping strip delta-gap...")
results_sdg = ext_sdg.extract(freqs.tolist())
Y11_sdg = np.array([r.Y_matrix[0, 0] for r in results_sdg])

print(f"Sweeping vertex probe...")
results_probe = ext_probe.extract(freqs.tolist())
Y11_probe = np.array([r.Y_matrix[0, 0] for r in results_probe])

# Expected Y11 for OC stub
def y11_ref(f, Z0, eps_eff, L):
    beta = 2*np.pi*f * np.sqrt(eps_eff) / c0
    bL = beta * L
    s = np.sin(bL)
    if abs(s) < 1e-10:
        return complex(0, np.sign(np.cos(bL)) * 1e10)
    return -1j / Z0 * np.cos(bL) / s

print(f"\n{'f(GHz)':>8}  {'Im(Y11_sdg)':>12}  {'Im(Y11_probe)':>13}  "
      f"{'Im(Y11_ref)':>12}  {'sdg_err%':>9}  {'probe_err%':>10}")
print("-" * 75)
for i, f in enumerate(freqs):
    y_ref = y11_ref(f, Z0_ref, eps_eff_ref, stub_length)
    err_sdg = (Y11_sdg[i].imag - y_ref.imag) / abs(y_ref.imag) * 100 if abs(y_ref.imag) > 1e-12 else float('nan')
    err_probe = (Y11_probe[i].imag - y_ref.imag) / abs(y_ref.imag) * 100 if abs(y_ref.imag) > 1e-12 else float('nan')
    print(f"  {f/1e9:>6.2f}  {Y11_sdg[i].imag:>12.6f}  {Y11_probe[i].imag:>13.6f}  "
          f"{y_ref.imag:>12.6f}  {err_sdg:>8.1f}%  {err_probe:>9.1f}%")

# Also compute Z_in for probe
print(f"\n--- Z_in from probe port ---")
print(f"{'f(GHz)':>8}  {'Re(Z_in)':>12}  {'Im(Z_in)':>12}  {'Z_in_ref':>12}")
print("-" * 52)
for i, f in enumerate(freqs):
    Z_sdg = results_sdg[i].Z_matrix[0, 0]
    Z_probe = results_probe[i].Z_matrix[0, 0]
    Z_ref = -1j * Z0_ref * np.cos(2*np.pi*f*np.sqrt(eps_eff_ref)/c0*stub_length) / \
            max(abs(np.sin(2*np.pi*f*np.sqrt(eps_eff_ref)/c0*stub_length)), 1e-10)
    print(f"  {f/1e9:>6.2f}  {Z_probe.real:>12.2f}  {Z_probe.imag:>12.2f}  {Z_ref.imag:>12.2f}")
