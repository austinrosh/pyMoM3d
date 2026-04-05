"""Test dual-stub Y11 extraction for microstrip Z0 and eps_eff.

By subtracting Y11 from two stubs of different length but identical port
geometry, the port discontinuity capacitance cancels:

    dY11 = Y11(L1) - Y11(L2) = Y_TL(L1) - Y_TL(L2)

Only Z0 and eps_eff remain as unknowns.
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
from pyMoM3d.mom.excitation import find_feed_edges, compute_feed_signs, StripDeltaGapExcitation
from pyMoM3d.network.tl_extraction import extract_tl_from_y11, extract_tl_dual_stub

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
TEL = 1.0e-3

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])


def simulate_stub(L, freqs):
    """Simulate Y11 for a microstrip stub of length L."""
    mesher = GmshMesher(target_edge_length=TEL)
    margin = TEL / 2
    port_x = -L / 2 + margin

    mesh = mesher.mesh_plate_with_feeds(
        width=L, height=W, feed_x_list=[port_x],
        center=(0.0, 0.0, H),
    )
    basis = compute_rwg_connectivity(mesh)
    feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
    signs = compute_feed_signs(mesh, basis, feed_edges)
    port = Port(name='P1', feed_basis_indices=feed_edges, feed_signs=signs)
    exc = StripDeltaGapExcitation(feed_basis_indices=feed_edges, voltage=1.0)

    stub_length = L / 2 - port_x  # distance from port to far end
    N = basis.num_basis
    print(f"  L={L*1e3:.0f}mm: {mesh.get_statistics()['num_triangles']} tris, {N} RWG, "
          f"{len(feed_edges)} feeds, stub={stub_length*1e3:.1f}mm")

    config = SimulationConfig(
        frequency=1e9, excitation=exc,
        source_layer_name='FR4', backend='auto', quad_order=4,
        layer_stack=stack,
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    ext = NetworkExtractor(sim, [port])

    results = ext.extract(freqs.tolist())
    Y11 = np.array([r.Y_matrix[0, 0] for r in results])
    Z11 = np.array([r.Z_matrix[0, 0] for r in results])
    return Y11, Z11, stub_length


# Use frequencies that avoid resonances for both stubs
freqs = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]) * 1e9

# --- Simulate two stubs ---
L1_total = 20e-3
L2_total = 30e-3

print("\nSimulating stubs...")
Y11_1, Z11_1, stub_1 = simulate_stub(L1_total, freqs)
Y11_2, Z11_2, stub_2 = simulate_stub(L2_total, freqs)

# --- Print raw Y11 data ---
print(f"\n{'f(GHz)':>8}  {'Im(Y11_L1)':>12}  {'Im(Y11_L2)':>12}  {'Im(dY11)':>12}  {'Im(Y_ref_L1)':>14}  {'Im(Y_ref_L2)':>14}")
print("-" * 85)

for i, f in enumerate(freqs):
    beta = 2 * np.pi * f * np.sqrt(eps_eff_ref) / c0
    Y_ref_1 = -1j / Z0_ref * np.cos(beta * stub_1) / np.sin(beta * stub_1) if abs(np.sin(beta * stub_1)) > 1e-10 else 0
    Y_ref_2 = -1j / Z0_ref * np.cos(beta * stub_2) / np.sin(beta * stub_2) if abs(np.sin(beta * stub_2)) > 1e-10 else 0

    print(f"  {f/1e9:>5.1f}  {Y11_1[i].imag:>12.6f}  {Y11_2[i].imag:>12.6f}  "
          f"{(Y11_1[i]-Y11_2[i]).imag:>12.6f}  {Y_ref_1.imag:>14.6f}  {Y_ref_2.imag:>14.6f}")

# --- Single-stub 3-parameter fit ---
print(f"\n--- Single-stub extraction (L1 = {stub_1*1e3:.1f}mm) ---")
result_single = extract_tl_from_y11(
    freqs, Y11_1, stub_1,
    Z0_guess=50.0, eps_eff_guess=3.3,
)
print(f"  Z0 = {result_single.Z0:.2f} Ohm (ref: {Z0_ref:.2f}, err: {abs(result_single.Z0-Z0_ref)/Z0_ref*100:.1f}%)")
print(f"  eps_eff = {result_single.eps_eff:.3f} (ref: {eps_eff_ref:.3f}, err: {abs(result_single.eps_eff-eps_eff_ref)/eps_eff_ref*100:.1f}%)")
print(f"  C_port = {result_single.C_port*1e12:.3f} pF")
print(f"  Residual = {result_single.residual_norm:.4e}")

# --- Dual-stub 2-parameter fit ---
print(f"\n--- Dual-stub extraction (L1={stub_1*1e3:.1f}mm, L2={stub_2*1e3:.1f}mm) ---")
result_dual = extract_tl_dual_stub(
    freqs, Y11_1, Y11_2,
    stub_1, stub_2,
    Z0_guess=50.0, eps_eff_guess=3.3,
)
print(f"  Z0 = {result_dual.Z0:.2f} Ohm (ref: {Z0_ref:.2f}, err: {abs(result_dual.Z0-Z0_ref)/Z0_ref*100:.1f}%)")
print(f"  eps_eff = {result_dual.eps_eff:.3f} (ref: {eps_eff_ref:.3f}, err: {abs(result_dual.eps_eff-eps_eff_ref)/eps_eff_ref*100:.1f}%)")
print(f"  C_port = {result_dual.C_port*1e12:.3f} pF")
print(f"  Residual = {result_dual.residual_norm:.4e}")

# --- Also try with analytical reference to verify the fitting works ---
print(f"\n--- Verification: fit with analytical Y11 + synthetic C_port ---")
C_port_syn = 1.0e-12  # 1 pF synthetic parasitic
Y11_syn = np.zeros(len(freqs), dtype=np.complex128)
for i, f in enumerate(freqs):
    omega = 2 * np.pi * f
    beta = omega * np.sqrt(eps_eff_ref) / c0
    bL = beta * stub_1
    s = np.sin(bL)
    Y_TL = -1j / Z0_ref * np.cos(bL) / s if abs(s) > 1e-10 else 0
    Y11_syn[i] = Y_TL + 1j * omega * C_port_syn

result_syn = extract_tl_from_y11(
    freqs, Y11_syn, stub_1,
    Z0_guess=50.0, eps_eff_guess=3.3,
)
print(f"  Z0 = {result_syn.Z0:.2f} Ohm (ref: {Z0_ref:.2f}, err: {abs(result_syn.Z0-Z0_ref)/Z0_ref*100:.1f}%)")
print(f"  eps_eff = {result_syn.eps_eff:.3f} (ref: {eps_eff_ref:.3f}, err: {abs(result_syn.eps_eff-eps_eff_ref)/eps_eff_ref*100:.1f}%)")
print(f"  C_port = {result_syn.C_port*1e12:.3f} pF (expected: {C_port_syn*1e12:.3f} pF)")
