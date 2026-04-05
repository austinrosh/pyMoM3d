"""Debug: Single-stub Y11 extraction with detailed diagnostics."""
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
from pyMoM3d.network.tl_extraction import extract_tl_from_y11

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 30e-3
TEL = 1.0e-3

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

# Build 1-port stub
mesher = GmshMesher(target_edge_length=TEL)
margin = TEL / 2.0
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
    frequency=1e9, excitation=exc,
    source_layer_name='FR4', backend='auto', quad_order=4,
    layer_stack=stack,
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
extractor = NetworkExtractor(sim, [port])

stub_length = L/2 - port_x  # port at -L/2+margin, open end at +L/2
print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {basis.num_basis} RWG")
print(f"Stub length: {stub_length*1e3:.1f} mm")
print(f"Feed edges: {len(feed_edges)}")

# Dense frequency sweep
freqs = np.linspace(0.5, 8.0, 20) * 1e9
print(f"\nSweeping {len(freqs)} frequencies ({freqs[0]/1e9:.1f} to {freqs[-1]/1e9:.1f} GHz)...")
results = extractor.extract(freqs.tolist())

Y11 = np.array([r.Y_matrix[0, 0] for r in results])

# Print raw Y11
print(f"\n{'f(GHz)':>8}  {'Re(Y11)':>12}  {'Im(Y11)':>12}  {'|Y11|':>12}")
print("-" * 50)
for i, f in enumerate(freqs):
    print(f"  {f/1e9:>6.2f}  {Y11[i].real:>12.6f}  {Y11[i].imag:>12.6f}  {abs(Y11[i]):>12.6f}")

# Compute expected Y11_TL at each freq for reference
print(f"\n--- Expected vs measured Im(Y11) ---")
print(f"{'f(GHz)':>8}  {'Im(Y11)_meas':>12}  {'Y11_TL(ref)':>12}  {'diff(~wC)':>12}  {'C(pF)':>12}")
print("-" * 65)
for i, f in enumerate(freqs):
    omega = 2*np.pi*f
    beta = omega * np.sqrt(eps_eff_ref) / c0
    bL = beta * stub_length
    sin_bL = np.sin(bL)
    if abs(sin_bL) > 1e-6:
        Y11_tl = -1/(Z0_ref) * np.cos(bL)/sin_bL
    else:
        Y11_tl = float('nan')
    diff = Y11[i].imag - Y11_tl
    C_pF = diff / omega * 1e12 if abs(omega) > 0 else 0
    print(f"  {f/1e9:>6.2f}  {Y11[i].imag:>12.6f}  {Y11_tl:>12.6f}  {diff:>12.6f}  {C_pF:>12.3f}")

# Single-stub fit
print(f"\n--- Single-stub 3-parameter fit ---")
result = extract_tl_from_y11(
    freqs, Y11, stub_length,
    Z0_guess=Z0_ref, eps_eff_guess=eps_eff_ref,
)
print(f"  Z0       = {result.Z0:.2f} Ohm  (ref: {Z0_ref:.2f})")
print(f"  eps_eff  = {result.eps_eff:.3f}  (ref: {eps_eff_ref:.3f})")
print(f"  C_port   = {result.C_port*1e12:.3f} pF")
print(f"  residual = {result.residual_norm:.2e}")
z0_err = abs(result.Z0 - Z0_ref) / Z0_ref * 100
ee_err = abs(result.eps_eff - eps_eff_ref) / eps_eff_ref * 100
print(f"  Z0 error:     {z0_err:.1f}%")
print(f"  eps_eff error: {ee_err:.1f}%")
