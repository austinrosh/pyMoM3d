"""Compare Y11 from Strata DCIM vs Layer Recursion backends via NetworkExtractor.

Layer recursion maintains much stronger mutual coupling at moderate ρ.
If it produces correct TL-like Y11 where Strata doesn't, the DCIM is the problem.
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
L = 20e-3  # shorter for layer recursion speed
TEL = 1.0e-3  # fine enough for clean feed edges

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
print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {N} RWG")
print(f"Stub: {stub_length*1e3:.1f} mm, {len(feed_edges)} feed edges")

# Test at a few key frequencies
freqs = [0.5e9, 1e9, 2e9, 3e9]

for gf_backend_name in ['strata', 'layer_recursion']:
    print(f"\n{'='*60}")
    print(f"GF Backend: {gf_backend_name}")
    print(f"{'='*60}")

    config = SimulationConfig(
        frequency=1e9, excitation=exc,
        source_layer_name='FR4',
        backend='auto',
        gf_backend=gf_backend_name,
        quad_order=4,
        layer_stack=stack,
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    ext = NetworkExtractor(sim, [port])

    print(f"\n{'f(GHz)':>8}  {'Re(Z_in)':>12}  {'Im(Z_in)':>12}  "
          f"{'Im(Y11)':>12}  {'Im_ref':>12}  {'err%':>8}")
    print("-" * 75)

    for freq in freqs:
        results = ext.extract([freq])
        Z_in = results[0].Z_matrix[0, 0]
        Y11 = results[0].Y_matrix[0, 0]

        beta = 2*np.pi*freq * np.sqrt(eps_eff_ref) / c0
        bL = beta * stub_length
        s = np.sin(bL)
        Y11_ref = -1j / Z0_ref * np.cos(bL) / s if abs(s) > 1e-10 else 0

        err = (Y11.imag - Y11_ref.imag) / abs(Y11_ref.imag) * 100 if abs(Y11_ref.imag) > 1e-12 else float('nan')

        print(f"  {freq/1e9:>6.1f}  {Z_in.real:>12.2f}  {Z_in.imag:>12.2f}  "
              f"{Y11.imag:>12.6f}  {Y11_ref.imag:>12.6f}  {err:>7.1f}%")
