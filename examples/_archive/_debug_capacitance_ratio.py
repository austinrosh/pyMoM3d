"""Test: does mesh resolution affect eps_eff = C(eps_r)/C(air)?

The Hammerstad eps_eff comes from the ratio of per-unit-length capacitance
with dielectric to without. If the mesh can't resolve edge charge
singularities, C(eps_r)/C(air) ≈ eps_r (parallel plate).

At low frequency, Y11 ≈ jωC, so C = Im(Y11)/(2πf).
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
from pyMoM3d.mom.excitation import find_feed_edges, compute_feed_signs

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 10e-3   # short stub to avoid TL effects
FREQ = 100e6  # 100 MHz — quasi-static

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")
print(f"Expected C ratio: {eps_eff_ref:.3f}")
print(f"Parallel plate C ratio: {EPS_R:.1f}")


def compute_capacitance(eps_r_sub, tel, freq=FREQ):
    """Compute Y11 → C for a microstrip stub with given substrate eps_r."""
    stack = LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('sub', z_bot=0.0, z_top=H, eps_r=eps_r_sub),
        Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
    ])

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

    stats = mesh.get_statistics()
    n_tris = stats['num_triangles']
    n_rwg = basis.num_basis

    config = SimulationConfig(
        frequency=freq, excitation=None,
        source_layer_name='sub', backend='auto', quad_order=4,
        layer_stack=stack,
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    ext = NetworkExtractor(sim, [port])

    result = ext.extract([freq])[0]
    Y11 = result.Y_matrix[0, 0]
    C = Y11.imag / (2 * np.pi * freq)

    return C, n_tris, n_rwg


print(f"\n{'TEL(mm)':>8} {'#tri':>6} {'#RWG':>6} {'C(FR4)(pF)':>12} {'C(air)(pF)':>12} "
      f"{'ratio':>8} {'eps_eff':>10} {'err%':>8}")
print("-" * 85)

for tel in [1.5e-3, 1.0e-3, 0.75e-3, 0.5e-3, 0.35e-3, 0.25e-3]:
    C_sub, n1, r1 = compute_capacitance(EPS_R, tel)
    C_air, n2, r2 = compute_capacitance(1.0, tel)

    if C_air > 0 and C_sub > 0:
        ratio = C_sub / C_air
        err = abs(ratio - eps_eff_ref) / eps_eff_ref * 100
    else:
        ratio = np.nan
        err = np.nan

    print(f"  {tel*1e3:>5.2f}  {n1:>6d} {r1:>6d}  {C_sub*1e12:>12.4f}  {C_air*1e12:>12.4f}  "
          f"{ratio:>8.3f}  {ratio:>10.3f}  {err:>7.1f}")
