"""Quick mesh convergence test for QS Z₀ extraction."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, Layer, LayerStack,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.utils.constants import c0, eta0


def hammerstad_z0(w, h, eps_r):
    u = w / h
    a = 1 + (1/49)*np.log((u**4 + (u/52)**2)/(u**4 + 0.432)) + (1/18.7)*np.log(1 + (u/18.1)**3)
    b = 0.564 * ((eps_r - 0.9)/(eps_r + 3))**0.053
    eps_eff = 0.5*(eps_r + 1) + 0.5*(eps_r - 1)*(1 + 10/u)**(-a*b)
    f_u = 6 + (2*np.pi - 6) * np.exp(-(30.666/u)**0.7528)
    z0 = (eta0/(2*np.pi)) * np.log(f_u/u + np.sqrt(1 + (2/u)**2)) / np.sqrt(eps_eff)
    return z0, eps_eff


EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

z0_ref, eps_eff_ref = hammerstad_z0(W_STRIP, H_SUB, EPS_R)
print(f"Hammerstad: Z₀ = {z0_ref:.2f} Ω, ε_eff = {eps_eff_ref:.3f}")

freq = 0.1e9

# Test different strip lengths and mesh densities
print(f"\n{'L (mm)':>8} {'TEL (mm)':>10} {'#tri':>6} {'#RWG':>6} {'Z₀ (Ω)':>8} {'ε_eff':>8}")
print("-" * 52)

for L_STRIP in [10e-3, 20e-3, 40e-3]:
    for TEL in [0.7e-3, 0.5e-3, 0.3e-3]:
        mesher = GmshMesher(target_edge_length=TEL)
        PORT1_X = -L_STRIP / 2.0 + 1.0e-3
        PORT2_X = +L_STRIP / 2.0 - 1.0e-3
        L_eff = PORT2_X - PORT1_X

        mesh = mesher.mesh_plate_with_feeds(
            width=L_STRIP, height=W_STRIP,
            feed_x_list=[PORT1_X, PORT2_X],
            center=(0.0, 0.0, H_SUB),
        )
        basis = compute_rwg_connectivity(mesh)

        feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
        feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
        signs1 = compute_feed_signs(mesh, basis, feed1)
        signs2 = compute_feed_signs(mesh, basis, feed2)
        port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
        port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

        exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
        config = SimulationConfig(
            frequency=1e9, excitation=exc, quad_order=4, backend='auto',
            layer_stack=stack, source_layer_name='FR4',
        )
        sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

        qs = QuasiStaticSolver(sim, [port1, port2],
                               probe_feeds=True, n_dielectric_images=0)
        [result] = qs.extract([freq])

        Z = result.Z_matrix
        z0_sq = Z[0, 0]**2 - Z[1, 0]**2
        z0_ext = np.sqrt(z0_sq).real if z0_sq.real > 0 else np.sqrt(abs(z0_sq))

        ratio = Z[0, 0] / Z[1, 0]
        bl = np.arccos(np.clip(np.real(ratio), -1, 1))
        beta = bl / L_eff
        eps_eff = (beta * c0 / (2 * np.pi * freq))**2

        print(f"{L_STRIP*1e3:8.0f} {TEL*1e3:10.1f} {len(mesh.triangles):6d} "
              f"{basis.num_basis:6d} {z0_ext:8.2f} {eps_eff:8.3f}")

print(f"\nReference: Z₀ = {z0_ref:.2f} Ω, ε_eff = {eps_eff_ref:.3f}")
