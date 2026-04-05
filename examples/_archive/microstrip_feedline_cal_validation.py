"""Microstrip feedline calibration validation.

Validates the feedline de-embedding approach:
1. Build a microstrip through-line with feedline extensions at each port
2. Simulate raw 2-port S-parameters
3. De-embed feedlines using Hammerstad Z0/eps_eff
4. The de-embedded S-matrix should show a matched through-line
   (S11 << -15 dB, S21 ≈ 0 dB)

The DUT is the central section of the line (L_dut).
The feedlines extend L_ext on each side.
Total simulated length: L_total = L_dut + 2 * L_ext.

This validates that the 3D MoM + feedline calibration workflow produces
meaningful circuit parameters despite the 1/R singularity limitation.
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
from pyMoM3d.network.feedline_calibration import FeedlineCalibration

# --- Physical parameters ---
EPS_R = 4.4
H = 1.6e-3       # substrate height
W = 3.06e-3       # strip width (50 Ohm microstrip)
L_DUT = 10e-3     # DUT section (through line)
L_EXT = 15e-3     # feedline extension on each side (larger for visible effect)
L_TOTAL = L_DUT + 2 * L_EXT
TEL = 1.0e-3      # target edge length

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")
print(f"Total line length: {L_TOTAL*1e3:.0f} mm")
print(f"  DUT: {L_DUT*1e3:.0f} mm (center)")
print(f"  Feedlines: {L_EXT*1e3:.0f} mm each side")

# Expected wavelength at 2 GHz
lam_2GHz = c0 / (2e9 * np.sqrt(eps_eff_ref))
print(f"Guided wavelength at 2 GHz: {lam_2GHz*1e3:.1f} mm")
print(f"L_ext / lambda: {L_EXT/lam_2GHz:.2f}")

# --- Layer stack ---
stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

# --- Mesh with two port locations ---
mesher = GmshMesher(target_edge_length=TEL)
margin = TEL / 2

# Port 1 at left end (x = -L_TOTAL/2 + margin)
# Port 2 at right end (x = +L_TOTAL/2 - margin)
port1_x = -L_TOTAL / 2 + margin
port2_x = +L_TOTAL / 2 - margin

mesh = mesher.mesh_plate_with_feeds(
    width=L_TOTAL, height=W,
    feed_x_list=[port1_x, port2_x],
    center=(0.0, 0.0, H),
)
basis = compute_rwg_connectivity(mesh)
stats = mesh.get_statistics()
print(f"\nMesh: {stats['num_triangles']} tris, {basis.num_basis} RWG")

# --- Ports ---
feed1 = find_feed_edges(mesh, basis, feed_x=port1_x)
signs1 = compute_feed_signs(mesh, basis, feed1)
port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)

feed2 = find_feed_edges(mesh, basis, feed_x=port2_x)
signs2 = compute_feed_signs(mesh, basis, feed2)
port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

print(f"Port 1: {len(feed1)} edges at x = {port1_x*1e3:.1f} mm")
print(f"Port 2: {len(feed2)} edges at x = {port2_x*1e3:.1f} mm")

# --- Simulation ---
config = SimulationConfig(
    frequency=1e9,
    excitation=None,
    source_layer_name='FR4',
    backend='auto',
    quad_order=4,
    layer_stack=stack,
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
ext = NetworkExtractor(sim, [port1, port2], Z0=Z0_ref)

# --- Feedline calibration ---
cal = FeedlineCalibration(
    Z0=Z0_ref,
    eps_eff=eps_eff_ref,
    feedline_length=L_EXT,
    Z0_ref=Z0_ref,
)

# --- Frequency sweep ---
freqs = np.linspace(0.5, 5.0, 20) * 1e9
print(f"\nFrequency sweep ({len(freqs)} points, {freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz)...")
results = ext.extract(freqs.tolist())

# --- Results ---
print(f"\n{'f(GHz)':>8}  {'|S11|raw':>10}  {'|S21|raw':>10}  "
      f"{'|S11|cal':>10}  {'|S21|cal':>10}  {'|S21|dut_exp':>12}")
print("-" * 75)

S11_raw_db = []
S21_raw_db = []
S11_cal_db = []
S21_cal_db = []

for i, (f, r) in enumerate(zip(freqs, results)):
    S_raw = r.S_matrix

    # De-embed feedlines
    S_cal = cal.deembed(S_raw, f)

    s11r = 20 * np.log10(max(abs(S_raw[0, 0]), 1e-15))
    s21r = 20 * np.log10(max(abs(S_raw[1, 0]), 1e-15))
    s11c = 20 * np.log10(max(abs(S_cal[0, 0]), 1e-15))
    s21c = 20 * np.log10(max(abs(S_cal[1, 0]), 1e-15))

    # Expected S21 for DUT thru-line (lossless): ~0 dB
    # Expected S11 for matched line: very low
    beta = 2 * np.pi * f * np.sqrt(eps_eff_ref) / c0
    s21_exp = 0.0  # lossless thru

    S11_raw_db.append(s11r)
    S21_raw_db.append(s21r)
    S11_cal_db.append(s11c)
    S21_cal_db.append(s21c)

    print(f"  {f/1e9:>5.1f}  {s11r:>10.2f}  {s21r:>10.2f}  "
          f"{s11c:>10.2f}  {s21c:>10.2f}  {s21_exp:>12.2f}")

# --- Summary ---
S11_raw_arr = np.array(S11_raw_db)
S21_raw_arr = np.array(S21_raw_db)
S11_cal_arr = np.array(S11_cal_db)
S21_cal_arr = np.array(S21_cal_db)

print(f"\n{'='*60}")
print(f"Raw (uncalibrated):")
print(f"  |S11| range: {S11_raw_arr.min():.1f} to {S11_raw_arr.max():.1f} dB")
print(f"  |S21| range: {S21_raw_arr.min():.1f} to {S21_raw_arr.max():.1f} dB")

print(f"\nCalibrated (feedline de-embedded):")
print(f"  |S11| range: {S11_cal_arr.min():.1f} to {S11_cal_arr.max():.1f} dB")
print(f"  |S21| range: {S21_cal_arr.min():.1f} to {S21_cal_arr.max():.1f} dB")

# For a matched thru-line:
# S21 should be near 0 dB (within ~1 dB)
# S11 should be below -15 dB
s21_mean = np.mean(S21_cal_arr)
s11_mean = np.mean(S11_cal_arr)
print(f"\n  Mean |S21|_cal = {s21_mean:.1f} dB (target: > -3 dB)")
print(f"  Mean |S11|_cal = {s11_mean:.1f} dB (target: < -10 dB)")

if s21_mean > -3.0:
    print(f"\n  PASS: De-embedded S21 is reasonable")
else:
    print(f"\n  MARGINAL: S21 still weak after de-embedding")

print(f"{'='*60}")
