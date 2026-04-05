"""Test Y11 extraction with AEFIE stabilization.

The standard EFIE has low-frequency breakdown: for ka << 1, the scalar
potential dominates and the vector potential (inductive TL contribution)
is lost in numerical noise. AEFIE separates the two potentials, avoiding
this issue.
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
L = 20e-3
TEL = 1.0e-3

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

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
port = Port(name='P1', feed_basis_indices=feed_edges, feed_signs=signs)
exc = StripDeltaGapExcitation(feed_basis_indices=feed_edges, voltage=1.0)

stub_length = L/2 - port_x
N = basis.num_basis

# kD estimate
from pyMoM3d.mom.aefie import estimate_kD
k_test = 2*np.pi*2e9/c0
kD = estimate_kD(mesh, k_test)
print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {N} RWG")
print(f"Stub: {stub_length*1e3:.1f} mm, {len(feed_edges)} feed edges")
print(f"kD at 2 GHz: {kD:.4f} (AEFIE threshold: 0.5)")

config = SimulationConfig(
    frequency=1e9, excitation=exc,
    source_layer_name='FR4', backend='auto', quad_order=4,
    layer_stack=stack,
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

# --- Standard EFIE ---
ext_efie = NetworkExtractor(sim, [port], low_freq_stabilization='none')

# --- AEFIE ---
ext_aefie = NetworkExtractor(sim, [port], low_freq_stabilization='aefie')

freqs = [0.5e9, 1e9, 2e9, 3e9, 5e9]

print(f"\n{'f(GHz)':>8}  {'Im(Y11_efie)':>13}  {'Im(Y11_aefie)':>14}  "
      f"{'Im_ref':>12}  {'efie_err%':>10}  {'aefie_err%':>11}")
print("-" * 80)

for freq in freqs:
    beta = 2*np.pi*freq * np.sqrt(eps_eff_ref) / c0
    bL = beta * stub_length
    s = np.sin(bL)
    Y11_ref = -1j / Z0_ref * np.cos(bL) / s if abs(s) > 1e-10 else 0

    results_efie = ext_efie.extract([freq])
    Y11_efie = results_efie[0].Y_matrix[0, 0]

    results_aefie = ext_aefie.extract([freq])
    Y11_aefie = results_aefie[0].Y_matrix[0, 0]

    err_efie = (Y11_efie.imag - Y11_ref.imag) / abs(Y11_ref.imag) * 100 if abs(Y11_ref.imag) > 1e-12 else float('nan')
    err_aefie = (Y11_aefie.imag - Y11_ref.imag) / abs(Y11_ref.imag) * 100 if abs(Y11_ref.imag) > 1e-12 else float('nan')

    Z_aefie = results_aefie[0].Z_matrix[0, 0]

    print(f"  {freq/1e9:>6.1f}  {Y11_efie.imag:>13.6f}  {Y11_aefie.imag:>14.6f}  "
          f"{Y11_ref.imag:>12.6f}  {err_efie:>9.1f}%  {err_aefie:>10.1f}%")
    print(f"         Z_in_aefie = {Z_aefie.real:.2f} + j{Z_aefie.imag:.2f}")
