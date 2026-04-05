"""Test: eps_eff = C(eps_r)/C(air) with AEFIE and consistent meshing.

Previous version had issues:
1. Standard EFIE at 100 MHz is severely ill-conditioned (kD ~ 0.01)
2. mesh_plate_with_feeds creates different topologies at different TEL

Fix: use AEFIE, use mesh_plate (no feed cuts), probe feed, higher frequency.
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
L = 10e-3   # short stub

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

FREQ = 500e6  # 500 MHz — better conditioned than 100 MHz


def compute_capacitance(eps_r_sub, tel, freq=FREQ):
    """Compute C from slope of Im(Y11) vs frequency."""
    stack = LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('sub', z_bot=0.0, z_top=H, eps_r=eps_r_sub),
        Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
    ])

    mesher = GmshMesher(target_edge_length=tel)
    mesh = mesher.mesh_plate(width=L, height=W, center=(0.0, 0.0, H))
    basis = compute_rwg_connectivity(mesh)

    # Probe feed at center of left edge
    port = Port.from_vertex(mesh, basis,
                             vertex_pos=np.array([-L/2, 0.0, H]),
                             name='P1', tol=tel * 1.5)

    stats = mesh.get_statistics()
    n_tris = stats['num_triangles']
    n_rwg = basis.num_basis

    config = SimulationConfig(
        frequency=freq, excitation=None,
        source_layer_name='sub', backend='auto', quad_order=4,
        layer_stack=stack,
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    ext = NetworkExtractor(sim, [port], low_freq_stabilization='aefie')

    # Extract at 3 frequencies, compute C from slope
    freqs = np.array([0.8, 1.0, 1.2]) * freq
    results = ext.extract(freqs.tolist())

    Y11_arr = np.array([r.Y_matrix[0, 0] for r in results])
    omega_arr = 2 * np.pi * freqs

    # C from slope: Im(Y11) = omega*C  → C = d(Im(Y11))/d(omega)
    slope, intercept = np.polyfit(omega_arr, Y11_arr.imag, 1)
    C = slope

    # Also single-point
    C_single = Y11_arr[1].imag / omega_arr[1]

    return C, C_single, n_tris, n_rwg, len(port.feed_basis_indices)


print(f"\n{'TEL(mm)':>8} {'#tri':>6} {'#RWG':>6} {'#port':>6} "
      f"{'C_sub(fF)':>11} {'C_air(fF)':>11} {'ratio':>8} {'err%':>8}")
print("-" * 80)

for tel in [1.0e-3, 0.75e-3, 0.5e-3, 0.35e-3, 0.25e-3]:
    try:
        C_sub, C_s1, n1, r1, p1 = compute_capacitance(EPS_R, tel)
        C_air, C_a1, n2, r2, p2 = compute_capacitance(1.0, tel)

        if C_air > 0 and C_sub > 0:
            ratio = C_sub / C_air
            err = abs(ratio - eps_eff_ref) / eps_eff_ref * 100
        else:
            ratio = err = np.nan

        print(f"  {tel*1e3:>5.2f}  {n1:>6d} {r1:>6d} {p1:>6d}  "
              f"{C_sub*1e15:>11.2f}  {C_air*1e15:>11.2f}  {ratio:>8.3f}  {err:>7.1f}")
    except Exception as e:
        print(f"  {tel*1e3:>5.2f}  FAILED: {e}")

# --- Also try direct EFIE at 2 GHz (better conditioned) ---
print(f"\n--- Direct EFIE at 2 GHz ---")
FREQ2 = 2e9

print(f"\n{'TEL(mm)':>8} {'#tri':>6} {'#RWG':>6} {'#port':>6} "
      f"{'C_sub(fF)':>11} {'C_air(fF)':>11} {'ratio':>8} {'err%':>8}")
print("-" * 80)


def compute_capacitance_efie(eps_r_sub, tel, freq=FREQ2):
    """Compute C from Y11 using standard EFIE at higher frequency."""
    stack = LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('sub', z_bot=0.0, z_top=H, eps_r=eps_r_sub),
        Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
    ])

    mesher = GmshMesher(target_edge_length=tel)
    mesh = mesher.mesh_plate(width=L, height=W, center=(0.0, 0.0, H))
    basis = compute_rwg_connectivity(mesh)

    port = Port.from_vertex(mesh, basis,
                             vertex_pos=np.array([-L/2, 0.0, H]),
                             name='P1', tol=tel * 1.5)

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
    omega = 2 * np.pi * freq
    C = Y11.imag / omega

    return C, n_tris, n_rwg, len(port.feed_basis_indices)


for tel in [1.0e-3, 0.75e-3, 0.5e-3, 0.35e-3, 0.25e-3]:
    try:
        C_sub, n1, r1, p1 = compute_capacitance_efie(EPS_R, tel)
        C_air, n2, r2, p2 = compute_capacitance_efie(1.0, tel)

        if C_air > 0 and C_sub > 0:
            ratio = C_sub / C_air
            err = abs(ratio - eps_eff_ref) / eps_eff_ref * 100
        else:
            ratio = err = np.nan

        print(f"  {tel*1e3:>5.2f}  {n1:>6d} {r1:>6d} {p1:>6d}  "
              f"{C_sub*1e15:>11.2f}  {C_air*1e15:>11.2f}  {ratio:>8.3f}  {err:>7.1f}")
    except Exception as e:
        print(f"  {tel*1e3:>5.2f}  FAILED: {e}")
