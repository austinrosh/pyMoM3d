"""QS solver + SOC de-embedding validation for microstrip through-line.

Uses edge port mesh with probe feeds and QS-mode SOC de-embedding.
Extracts Z₀ and ε_eff from de-embedded S-parameters.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, Layer, LayerStack,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges,
    find_edge_port_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.network.soc_deembedding import SOCDeembedding
from pyMoM3d.utils.constants import c0, eta0


def hammerstad_z0(w, h, eps_r):
    u = w / h
    a = 1 + (1/49)*np.log((u**4 + (u/52)**2)/(u**4 + 0.432)) + (1/18.7)*np.log(1 + (u/18.1)**3)
    b = 0.564 * ((eps_r - 0.9)/(eps_r + 3))**0.053
    eps_eff = 0.5*(eps_r + 1) + 0.5*(eps_r - 1)*(1 + 10/u)**(-a*b)
    f_u = 6 + (2*np.pi - 6) * np.exp(-(30.666/u)**0.7528)
    z0 = (eta0/(2*np.pi)) * np.log(f_u/u + np.sqrt(1 + (2/u)**2)) / np.sqrt(eps_eff)
    return z0, eps_eff


# ── Geometry ──────────────────────────────────────────────────────
EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
L_STRIP = 20.0e-3   # longer strip for SOC (need room for reference planes)
TEL = 0.7e-3

z0_ref, eps_eff_ref = hammerstad_z0(W_STRIP, H_SUB, EPS_R)
print(f"Hammerstad: Z₀ = {z0_ref:.2f} Ω, ε_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

mesher = GmshMesher(target_edge_length=TEL)

# ── Flat mesh with interior ports + reference plane edges ─────────
PORT1_X = -L_STRIP / 2.0 + 1.0e-3
PORT2_X = +L_STRIP / 2.0 - 1.0e-3
REF1_X = -L_STRIP / 2.0 + 3.0e-3   # reference plane 3mm from edge
REF2_X = +L_STRIP / 2.0 - 3.0e-3

mesh = mesher.mesh_plate_with_feeds(
    width=L_STRIP, height=W_STRIP,
    feed_x_list=[PORT1_X, PORT2_X, REF1_X, REF2_X],
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

print(f"Mesh: {len(mesh.triangles)} tri, {basis.num_basis} RWG")
print(f"DUT length: {(REF2_X - REF1_X)*1e3:.1f} mm")

# ── QS extraction (raw) ──────────────────────────────────────────
freqs = [0.1e9, 0.2e9, 0.5e9, 1e9, 2e9, 3e9]

print("\n=== Raw QS + probe feeds ===")
qs = QuasiStaticSolver(sim, [port1, port2],
                       probe_feeds=True, n_dielectric_images=0,
                       store_currents=True)
results_raw = qs.extract(freqs)

print(f"{'f (GHz)':>8} {'S21 (dB)':>10} {'S11 (dB)':>10} {'|S|^2':>8}")
print("-" * 40)
for r in results_raw:
    S = r.S_matrix
    s21 = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s11 = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
    pwr = abs(S[0,0])**2 + abs(S[1,0])**2
    print(f"{r.frequency/1e9:8.1f} {s21:10.2f} {s11:10.2f} {pwr:8.4f}")

# ── SOC de-embedding with QS solver ──────────────────────────────
print("\n=== QS SOC de-embedding ===")
try:
    soc = SOCDeembedding(
        sim, [port1, port2],
        reference_plane_x=[REF1_X, REF2_X],
        Z0=50.0,
        use_qs=True,
        probe_feeds=True,
        n_dielectric_images=0,
    )

    results_soc = []
    print(f"{'f (GHz)':>8} {'S21 raw':>10} {'S21 SOC':>10} {'S11 SOC':>10} {'|S|^2':>8}")
    print("-" * 52)
    for r_raw in results_raw:
        try:
            r_cal = soc.deembed(r_raw)
            results_soc.append(r_cal)
            S_raw = r_raw.S_matrix
            S_cal = r_cal.S_matrix
            s21_raw = 20 * np.log10(max(abs(S_raw[1, 0]), 1e-30))
            s21_cal = 20 * np.log10(max(abs(S_cal[1, 0]), 1e-30))
            s11_cal = 20 * np.log10(max(abs(S_cal[0, 0]), 1e-30))
            pwr = abs(S_cal[0,0])**2 + abs(S_cal[1,0])**2
            print(f"{r_raw.frequency/1e9:8.1f} {s21_raw:10.2f} {s21_cal:10.2f} "
                  f"{s11_cal:10.2f} {pwr:8.4f}")
        except Exception as e:
            print(f"{r_raw.frequency/1e9:8.1f}  ERROR: {e}")
            results_soc.append(None)

    # ── Z₀ extraction from de-embedded Z-matrix ──────────────────
    print("\n=== Z₀ extraction from de-embedded results ===")
    l_dut = REF2_X - REF1_X  # DUT length
    print(f"DUT length = {l_dut*1e3:.1f} mm")
    print(f"{'f (GHz)':>8} {'Z₀ (Ω)':>10} {'ε_eff':>8} {'β (rad/m)':>10}")
    print("-" * 40)
    for r in results_soc:
        if r is None:
            continue
        Z = r.Z_matrix
        # For a lossless uniform line: Z11 = Z22 = -jZ₀cot(βl)
        # Z21 = Z12 = -jZ₀csc(βl)
        # Z₀² = Z11² - Z21² × ... no, use: Z₀ = sqrt(Z11² - Z21²)
        # For symmetric: Z₀ = sqrt((Z11 + Z21)(Z11 - Z21))
        z11, z21 = Z[0,0], Z[1,0]
        z0_sq = z11**2 - z21**2
        if z0_sq.real > 0:
            z0_ext = np.sqrt(z0_sq).real
        else:
            z0_ext = np.sqrt(abs(z0_sq))

        # β from Z11/Z21 = cos(βl): cot(βl)/csc(βl) = cos(βl)
        cos_bl = z11 / z21 if abs(z21) > 1e-30 else 0
        bl = np.arccos(np.clip(cos_bl.real, -1, 1))
        beta = bl / l_dut if l_dut > 0 else 0
        eps_eff = (beta * c0 / (2 * np.pi * r.frequency))**2 if r.frequency > 0 else 0

        print(f"{r.frequency/1e9:8.1f} {z0_ext:10.2f} {eps_eff:8.3f} {beta:10.2f}")

except Exception as e:
    import traceback
    print(f"SOC failed: {e}")
    traceback.print_exc()

print(f"\nReference: Z₀ = {z0_ref:.2f} Ω, ε_eff = {eps_eff_ref:.3f}")
print("Done.")
