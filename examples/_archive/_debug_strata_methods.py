"""Compare Strata DCIM vs quasistatic vs integrate for C ratio.

DCIM fitted at midpoint gives eps_eff ≈ eps_r because it doesn't handle
the source at the dielectric-air interface. Test if quasistatic or
integrate methods give the correct interface behavior.
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

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 10e-3
TEL = 1.0e-3  # coarser for speed (just checking ratio, not convergence)
FREQ = 2.0e9

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f}, eps_eff = {eps_eff_ref:.3f}")

mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_plate(width=L, height=W, center=(0.0, 0.0, H))
basis = compute_rwg_connectivity(mesh)
port = Port.from_vertex(mesh, basis, vertex_pos=np.array([-L/2, 0.0, H]),
                         name='P1', tol=TEL * 1.5)
print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {basis.num_basis} RWG")


def compute_C(eps_r_sub, method='auto'):
    stack = LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('sub', z_bot=0.0, z_top=H, eps_r=eps_r_sub),
        Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
    ])
    config = SimulationConfig(
        frequency=FREQ, excitation=None,
        source_layer_name='sub', backend='auto', quad_order=4,
        layer_stack=stack, gf_backend=method,
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    ext = NetworkExtractor(sim, [port])
    result = ext.extract([FREQ])[0]
    Y11 = result.Y_matrix[0, 0]
    return Y11.imag / (2 * np.pi * FREQ)


print(f"\n{'Method':>15} {'C_sub(fF)':>11} {'C_air(fF)':>11} {'ratio':>8} {'err%':>8}")
print("-" * 60)

for method in ['auto', 'dcim', 'quasistatic']:
    try:
        C_sub = compute_C(EPS_R, method)
        C_air = compute_C(1.0, method)
        ratio = C_sub / C_air if C_air > 0 else np.nan
        err = abs(ratio - eps_eff_ref) / eps_eff_ref * 100 if np.isfinite(ratio) else np.nan
        print(f"  {method:>13}  {C_sub*1e15:>11.2f}  {C_air*1e15:>11.2f}  "
              f"{ratio:>8.3f}  {err:>7.1f}")
    except Exception as e:
        print(f"  {method:>13}  FAILED: {e}")

# Also try layer recursion backend
print("\n--- Python-side backends ---")
for method in ['layer_recursion']:
    try:
        C_sub = compute_C(EPS_R, method)
        C_air = compute_C(1.0, method)
        ratio = C_sub / C_air if C_air > 0 else np.nan
        err = abs(ratio - eps_eff_ref) / eps_eff_ref * 100 if np.isfinite(ratio) else np.nan
        print(f"  {method:>13}  {C_sub*1e15:>11.2f}  {C_air*1e15:>11.2f}  "
              f"{ratio:>8.3f}  {err:>7.1f}")
    except Exception as e:
        print(f"  {method:>13}  FAILED: {e}")

# --- Direct comparison: raw G_phi from Strata at a single point ---
print("\n--- Raw G_phi comparison ---")
from pyMoM3d.greens.layered import LayeredGreensFunction

for eps_r in [4.4, 1.0]:
    stack = LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('sub', z_bot=0.0, z_top=H, eps_r=eps_r),
        Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
    ])
    gf = LayeredGreensFunction(stack, FREQ, source_layer_name='sub')

    # Evaluate G at various rho
    for rho in [0.5e-3, 1.0e-3, 2.0e-3, 5.0e-3, 10.0e-3]:
        r_obs = np.array([[rho, 0.0, H]])
        r_src = np.array([[0.0, 0.0, H]])

        # Full G (includes free-space)
        G_scalar = gf.backend.scalar_G(r_obs, r_src)

        # Correction only (ML - FS)
        G_smooth = gf.backend.scalar_G(r_obs, r_src, return_correction=True)

        print(f"  eps_r={eps_r:.1f}, rho={rho*1e3:.1f}mm: "
              f"G_scalar={abs(G_scalar[0]):.4e}, G_smooth={abs(G_smooth[0]):.4e}")
