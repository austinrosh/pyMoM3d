"""End-to-end QS pipeline: probe feeds + SOC de-embedding.

Validates:
1. Raw QS S-parameters (magnitude and phase)
2. SOC de-embedded S-parameters
3. Z₀ extraction from raw and de-embedded Z-matrices
4. Self-consistency checks (reciprocity, passivity, symmetry)
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
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
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


EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
L_STRIP = 20.0e-3
TEL = 0.7e-3

z0_ref, eps_eff_ref = hammerstad_z0(W_STRIP, H_SUB, EPS_R)
print(f"Hammerstad: Z₀ = {z0_ref:.2f} Ω, ε_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

mesher = GmshMesher(target_edge_length=TEL)

PORT1_X = -L_STRIP / 2.0 + 1.0e-3
PORT2_X = +L_STRIP / 2.0 - 1.0e-3
REF1_X = -L_STRIP / 2.0 + 3.0e-3
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
print(f"Port-to-port: {(PORT2_X - PORT1_X)*1e3:.1f} mm")
print(f"DUT (ref-to-ref): {(REF2_X - REF1_X)*1e3:.1f} mm")
print(f"Feed section: {(REF1_X - PORT1_X)*1e3:.1f} mm")

# ── RAW QS extraction ──────────────────────────────────────────────
freqs = [0.1e9, 0.2e9, 0.5e9, 1e9, 2e9]

qs = QuasiStaticSolver(sim, [port1, port2],
                       probe_feeds=True, n_dielectric_images=0,
                       store_currents=True)
results_raw = qs.extract(freqs)

print("\n=== Raw QS (probe feeds, n_diel=0) ===")
print(f"{'f':>6} {'|S21|':>8} {'∠S21':>8} {'|S11|':>8} {'Z₀':>8} {'Recip':>8} {'Sym':>8}")
print("-" * 56)
for r in results_raw:
    S = r.S_matrix
    Z = r.Z_matrix
    s21_db = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
    s21_phase = np.angle(S[1, 0], deg=True)
    s11_db = 20 * np.log10(max(abs(S[0, 0]), 1e-30))

    # Z₀ from Z-matrix (β-independent)
    z0_sq = Z[0, 0]**2 - Z[1, 0]**2
    z0_ext = np.sqrt(z0_sq).real if z0_sq.real > 0 else np.sqrt(abs(z0_sq))

    # Reciprocity: |S21 - S12|
    recip = abs(S[1, 0] - S[0, 1])
    # Symmetry: |S11 - S22|
    sym = abs(S[0, 0] - S[1, 1])

    print(f"{r.frequency/1e9:6.1f} {s21_db:8.2f} {s21_phase:8.1f} {s11_db:8.2f} "
          f"{z0_ext:8.2f} {recip:8.2e} {sym:8.2e}")

# ── SOC de-embedding ────────────────────────────────────────────────
print("\n=== SOC de-embedding ===")
soc = SOCDeembedding(
    sim, [port1, port2],
    reference_plane_x=[REF1_X, REF2_X],
    Z0=50.0,
    use_qs=True,
    probe_feeds=True,
    n_dielectric_images=0,
)

# Print error box ABCD at a few frequencies
print("\nError box ABCD (port 1):")
for f in [0.1e9, 0.5e9, 1e9]:
    T_err = soc.compute_error_abcd(0, f)
    A, B, C, D = T_err[0, 0], T_err[0, 1], T_err[1, 0], T_err[1, 1]
    det = A*D - B*C
    print(f"  {f/1e9:.1f} GHz: A={A:.4f}, B={B:.4f}, C={C:.6f}, D={D:.4f}, det={det:.4f}")

print(f"\n{'f':>6} {'S21 raw':>8} {'S21 SOC':>8} {'S11 SOC':>8} {'Z₀ raw':>8} {'Z₀ SOC':>8}")
print("-" * 48)
results_soc = []
for r_raw in results_raw:
    try:
        r_cal = soc.deembed(r_raw)
        results_soc.append(r_cal)
        S_raw = r_raw.S_matrix
        S_cal = r_cal.S_matrix
        Z_raw = r_raw.Z_matrix
        Z_cal = r_cal.Z_matrix

        s21_raw = 20 * np.log10(max(abs(S_raw[1, 0]), 1e-30))
        s21_cal = 20 * np.log10(max(abs(S_cal[1, 0]), 1e-30))
        s11_cal = 20 * np.log10(max(abs(S_cal[0, 0]), 1e-30))

        z0_raw = np.sqrt(Z_raw[0,0]**2 - Z_raw[1,0]**2)
        z0_cal = np.sqrt(Z_cal[0,0]**2 - Z_cal[1,0]**2)
        z0_raw = z0_raw.real if z0_raw.real > 0 else abs(z0_raw)
        z0_cal = z0_cal.real if z0_cal.real > 0 else abs(z0_cal)

        print(f"{r_raw.frequency/1e9:6.1f} {s21_raw:8.2f} {s21_cal:8.2f} {s11_cal:8.2f} "
              f"{z0_raw:8.2f} {z0_cal:8.2f}")
    except Exception as e:
        print(f"{r_raw.frequency/1e9:6.1f}  ERROR: {e}")
        results_soc.append(None)

print(f"\nReference: Z₀ = {z0_ref:.2f} Ω, ε_eff = {eps_eff_ref:.3f}")
print("Done.")
