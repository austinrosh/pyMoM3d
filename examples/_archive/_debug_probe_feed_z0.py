"""Microstrip Z0 extraction using vertex probe feeds.

Strip delta-gap creates ~1.77 pF parasitic capacitance that dominates Y11
and makes TL extraction impossible. Vertex probe feeds (Port.from_vertex)
inject current at a single mesh vertex — like a coaxial probe — avoiding
the transverse gap discontinuity entirely.

Strategy: 2-port extraction with probe feeds at the center of each short end.
Z0 and eps_eff from ABCD parameters of the 2-port S-matrix.
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
from pyMoM3d.analysis.transmission_line import (
    extract_z0_from_s, extract_propagation_constant,
)

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 20e-3    # 20mm microstrip line
TEL = 0.5e-3  # finer mesh for more basis functions per port

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

# --- Mesh ---
mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_plate(width=L, height=W, center=(0.0, 0.0, H))
basis = compute_rwg_connectivity(mesh)
stats = mesh.get_statistics()
print(f"Mesh: {stats['num_triangles']} tris, {basis.num_basis} RWG")

# --- Probe ports at center of each short end ---
# Port 1: center of left end (x = -L/2, y = 0)
# Port 2: center of right end (x = +L/2, y = 0)
port1 = Port.from_vertex(mesh, basis, vertex_pos=np.array([-L/2, 0.0, H]),
                          name='P1', tol=TEL)
port2 = Port.from_vertex(mesh, basis, vertex_pos=np.array([+L/2, 0.0, H]),
                          name='P2', tol=TEL)

print(f"Port 1: {len(port1.feed_basis_indices)} basis functions")
print(f"Port 2: {len(port2.feed_basis_indices)} basis functions")

# --- Simulation ---
config = SimulationConfig(
    frequency=1e9,
    excitation=None,   # NetworkExtractor handles excitation
    source_layer_name='FR4',
    backend='auto',
    quad_order=4,
    layer_stack=stack,
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
ext = NetworkExtractor(sim, [port1, port2], store_currents=True)

# --- Frequency sweep ---
freqs = np.linspace(0.5, 6.0, 24) * 1e9
print(f"\nFrequency sweep ({len(freqs)} points)...")
results = ext.extract(freqs.tolist())

# --- Extract Z0 and eps_eff ---
print(f"\n{'f(GHz)':>8}  {'|S11|dB':>10}  {'|S21|dB':>10}  {'Z0(Ohm)':>12}  "
      f"{'eps_eff':>10}  {'Re(Z11)':>10}  {'Im(Z11)':>10}")
print("-" * 85)

Z0_vals = []
eps_eff_vals = []

for i, (f, r) in enumerate(zip(freqs, results)):
    S = r.S_matrix
    Z = r.Z_matrix
    Y = r.Y_matrix

    s11_db = 20 * np.log10(max(abs(S[0, 0]), 1e-15))
    s21_db = 20 * np.log10(max(abs(S[1, 0]), 1e-15))

    try:
        z0_ext = extract_z0_from_s(S, Z0_ref=50.0)
        gamma = extract_propagation_constant(S, L, Z0_ref=50.0)
        beta = gamma.imag
        eps_eff_ext = (beta * c0 / (2 * np.pi * f)) ** 2
    except Exception:
        z0_ext = complex(np.nan)
        eps_eff_ext = np.nan

    Z0_vals.append(z0_ext)
    eps_eff_vals.append(eps_eff_ext)

    print(f"  {f/1e9:>5.1f}  {s11_db:>10.2f}  {s21_db:>10.2f}  "
          f"{z0_ext.real:>8.1f}+j{z0_ext.imag:>5.1f}  {eps_eff_ext:>10.3f}  "
          f"{Z[0,0].real:>10.2f}  {Z[0,0].imag:>10.2f}")

# --- 1-port Y11 comparison ---
print(f"\n--- 1-port Y11 behavior ---")
ext1 = NetworkExtractor(sim, [port1])
results1 = ext1.extract(freqs.tolist())

print(f"{'f(GHz)':>8}  {'Re(Y11)':>12}  {'Im(Y11)':>12}  {'Re(Z11)':>12}  {'Im(Z11)':>12}")
print("-" * 65)
for f, r in zip(freqs, results1):
    Y11 = r.Y_matrix[0, 0]
    Z11 = r.Z_matrix[0, 0]
    print(f"  {f/1e9:>5.1f}  {Y11.real:>12.6f}  {Y11.imag:>12.6f}  "
          f"{Z11.real:>12.4f}  {Z11.imag:>12.2f}")

# --- Summary ---
print(f"\n{'='*60}")
Z0_arr = np.array(Z0_vals)
eps_arr = np.array(eps_eff_vals)

# Filter out NaN and unreasonable values
mask = np.isfinite(Z0_arr.real) & (Z0_arr.real > 5) & (Z0_arr.real < 500)
if np.any(mask):
    Z0_mean = np.mean(Z0_arr[mask].real)
    eps_mean = np.mean(eps_arr[mask])
    Z0_err = abs(Z0_mean - Z0_ref) / Z0_ref * 100
    eps_err = abs(eps_mean - eps_eff_ref) / eps_eff_ref * 100
    print(f"Mean Z0 = {Z0_mean:.1f} Ohm (ref: {Z0_ref:.1f}, err: {Z0_err:.1f}%)")
    print(f"Mean eps_eff = {eps_mean:.3f} (ref: {eps_eff_ref:.3f}, err: {eps_err:.1f}%)")
else:
    print("No valid Z0 values extracted")
print(f"{'='*60}")
