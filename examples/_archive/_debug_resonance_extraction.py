"""Extract eps_eff from microstrip half-wave resonance frequency.

The 3D MoM CAN detect resonance frequencies accurately (validated:
patch antenna f_res within 5.8%). This script finds the half-wave
resonance of a microstrip strip and extracts eps_eff from it.

f_res = c0 / (2 * L * sqrt(eps_eff))
=> eps_eff = (c0 / (2 * L * f_res))^2
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

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
TEL = 1.0e-3

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

# For half-wave resonance at ~2 GHz:
# L = c0 / (2 * f_res * sqrt(eps_eff)) = 3e8 / (2 * 2e9 * sqrt(3.3)) = 41 mm
# Try L = 40 mm to get f_res near 2 GHz
L = 40e-3

# Expected resonance:
f_res_expected = c0 / (2 * L * np.sqrt(eps_eff_ref))
print(f"Expected half-wave resonance: {f_res_expected/1e9:.3f} GHz")

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

N = basis.num_basis
print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {N} RWG, {len(feed_edges)} feed edges")

config = SimulationConfig(
    frequency=1e9, excitation=exc,
    source_layer_name='FR4', backend='auto', quad_order=4,
    layer_stack=stack,
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
ext = NetworkExtractor(sim, [port])

# Coarse sweep to find resonance region
print("\nCoarse frequency sweep...")
freqs_coarse = np.linspace(1.0, 4.0, 16) * 1e9
results_coarse = ext.extract(freqs_coarse.tolist())

Y11_coarse = np.array([r.Y_matrix[0, 0] for r in results_coarse])
Z11_coarse = np.array([r.Z_matrix[0, 0] for r in results_coarse])

print(f"\n{'f(GHz)':>8}  {'Re(Z_in)':>12}  {'Im(Z_in)':>12}  {'|Y11|':>12}")
print("-" * 50)
for i, f in enumerate(freqs_coarse):
    print(f"  {f/1e9:>6.2f}  {Z11_coarse[i].real:>12.4f}  {Z11_coarse[i].imag:>12.2f}  "
          f"{abs(Y11_coarse[i]):>12.6f}")

# Find resonance: peak in Re(Z_in) or |Y11|
# At resonance, Im(Z_in) = 0 (or crosses through zero)
abs_Y11 = np.abs(Y11_coarse)
re_Z = Z11_coarse.real
peak_idx = np.argmax(abs_Y11)
f_res_measured = freqs_coarse[peak_idx]
print(f"\nPeak |Y11| at f = {f_res_measured/1e9:.3f} GHz (index {peak_idx})")

# Also find where Im(Z) changes sign
imag_Z = Z11_coarse.imag
for i in range(len(freqs_coarse)-1):
    if imag_Z[i] * imag_Z[i+1] < 0:
        # Linear interpolation
        f1, f2 = freqs_coarse[i], freqs_coarse[i+1]
        z1, z2 = imag_Z[i], imag_Z[i+1]
        f_cross = f1 - z1 * (f2 - f1) / (z2 - z1)
        print(f"Im(Z) zero crossing at f = {f_cross/1e9:.3f} GHz")

# Fine sweep around resonance
f_center = f_res_measured
f_range = 0.3e9
freqs_fine = np.linspace(f_center - f_range, f_center + f_range, 21)
freqs_fine = np.clip(freqs_fine, 0.5e9, 5.0e9)

print(f"\nFine sweep around {f_center/1e9:.2f} GHz...")
results_fine = ext.extract(freqs_fine.tolist())
Y11_fine = np.array([r.Y_matrix[0, 0] for r in results_fine])
Z11_fine = np.array([r.Z_matrix[0, 0] for r in results_fine])

print(f"\n{'f(GHz)':>8}  {'Re(Z_in)':>12}  {'Im(Z_in)':>12}  {'|Y11|':>12}")
print("-" * 50)
for i, f in enumerate(freqs_fine):
    print(f"  {f/1e9:>6.2f}  {Z11_fine[i].real:>12.4f}  {Z11_fine[i].imag:>12.2f}  "
          f"{abs(Y11_fine[i]):>12.6f}")

# Find peak in fine sweep
abs_Y11_fine = np.abs(Y11_fine)
peak_fine = np.argmax(abs_Y11_fine)
f_res_fine = freqs_fine[peak_fine]

# Better f_res from Im(Z) zero crossing in fine sweep
imag_Z_fine = Z11_fine.imag
f_res_interp = f_res_fine  # fallback
for i in range(len(freqs_fine)-1):
    if imag_Z_fine[i] * imag_Z_fine[i+1] < 0 and imag_Z_fine[i] < 0:
        # First inductive-to-capacitive crossing = series resonance
        f1, f2 = freqs_fine[i], freqs_fine[i+1]
        z1, z2 = imag_Z_fine[i], imag_Z_fine[i+1]
        f_res_interp = f1 - z1 * (f2 - f1) / (z2 - z1)
        break

# Open-end fringing correction (Hammerstad-Bekkadal)
# ΔL/h = 0.412 * (eps_eff+0.3)(W/h+0.264) / ((eps_eff-0.258)(W/h+0.8))
wh = W / H
dL_over_h = 0.412 * (eps_eff_ref + 0.3) * (wh + 0.264) / ((eps_eff_ref - 0.258) * (wh + 0.8))
dL = dL_over_h * H
L_eff = L + 2 * dL  # two open ends

# Extract eps_eff with and without fringing
eps_eff_raw = (c0 / (2 * L * f_res_interp))**2
eps_eff_corrected = (c0 / (2 * L_eff * f_res_interp))**2

print(f"\n{'='*55}")
print(f"Resonance-based extraction:")
print(f"  f_res (|Y11| peak) = {f_res_fine/1e9:.3f} GHz")
print(f"  f_res (Im(Z)=0)    = {f_res_interp/1e9:.3f} GHz")
print(f"  f_res expected      = {f_res_expected/1e9:.3f} GHz")
print(f"  Open-end ΔL         = {dL*1e3:.3f} mm (each end)")
print(f"  L_eff               = {L_eff*1e3:.2f} mm (vs L = {L*1e3:.1f} mm)")
print(f"  eps_eff (raw)       = {eps_eff_raw:.3f}  (ref: {eps_eff_ref:.3f}, err: {abs(eps_eff_raw - eps_eff_ref)/eps_eff_ref*100:.1f}%)")
print(f"  eps_eff (corrected) = {eps_eff_corrected:.3f}  (ref: {eps_eff_ref:.3f}, err: {abs(eps_eff_corrected - eps_eff_ref)/eps_eff_ref*100:.1f}%)")
print(f"{'='*55}")
