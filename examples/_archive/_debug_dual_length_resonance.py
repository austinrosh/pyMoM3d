"""Dual-length resonance extraction for eps_eff.

Measure half-wave resonance of two different strip lengths.
Fringing extends effective length by the same 2*dL for both:

    f_A = c0 / (2*(L_A + 2*dL)*sqrt(eps_eff))
    f_B = c0 / (2*(L_B + 2*dL)*sqrt(eps_eff))

From the ratio, solve for dL, then eps_eff.
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

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])


def find_resonance(L, freq_range, n_coarse=20, n_fine=15):
    """Find the fundamental half-wave resonance of a microstrip strip.

    Returns the Im(Z)=0 crossing frequency (series resonance).
    """
    mesher = GmshMesher(target_edge_length=TEL)
    margin = TEL / 2
    port_x = -L / 2 + margin

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
    ntri = mesh.get_statistics()['num_triangles']
    print(f"  L={L*1e3:.0f}mm: {ntri} tris, {N} RWG, {len(feed_edges)} feeds")

    config = SimulationConfig(
        frequency=1e9, excitation=exc,
        source_layer_name='FR4', backend='auto', quad_order=4,
        layer_stack=stack,
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    ext = NetworkExtractor(sim, [port])

    # Coarse sweep
    freqs_c = np.linspace(freq_range[0], freq_range[1], n_coarse)
    results_c = ext.extract(freqs_c.tolist())
    Z11_c = np.array([r.Z_matrix[0, 0] for r in results_c])

    # Find Im(Z)=0 crossings (negative→positive = series resonance)
    crossings = []
    for i in range(len(freqs_c) - 1):
        if Z11_c[i].imag < 0 and Z11_c[i+1].imag > 0:
            f1, f2 = freqs_c[i], freqs_c[i+1]
            z1, z2 = Z11_c[i].imag, Z11_c[i+1].imag
            f_cross = f1 - z1 * (f2 - f1) / (z2 - z1)
            crossings.append(f_cross)

    if not crossings:
        # Also check Re(Z) peaks as fallback
        re_Z = Z11_c.real
        peak_idx = np.argmax(re_Z)
        if re_Z[peak_idx] > 3 * np.median(re_Z):
            f_approx = freqs_c[peak_idx]
            print(f"    No Im(Z)=0 crossing; using Re(Z) peak at {f_approx/1e9:.3f} GHz")
            crossings = [f_approx]
        else:
            print(f"    No resonance found!")
            return None, Z11_c, freqs_c

    f_res_coarse = crossings[0]  # take fundamental
    print(f"    Coarse resonance at {f_res_coarse/1e9:.3f} GHz")

    # Fine sweep around resonance
    df = (freq_range[1] - freq_range[0]) / n_coarse * 2
    freqs_f = np.linspace(f_res_coarse - df, f_res_coarse + df, n_fine)
    results_f = ext.extract(freqs_f.tolist())
    Z11_f = np.array([r.Z_matrix[0, 0] for r in results_f])

    # Refined crossing
    for i in range(len(freqs_f) - 1):
        if Z11_f[i].imag < 0 and Z11_f[i+1].imag > 0:
            f1, f2 = freqs_f[i], freqs_f[i+1]
            z1, z2 = Z11_f[i].imag, Z11_f[i+1].imag
            f_res_fine = f1 - z1 * (f2 - f1) / (z2 - z1)
            print(f"    Fine resonance at {f_res_fine/1e9:.4f} GHz")
            return f_res_fine, Z11_f, freqs_f

    print(f"    Fine sweep: no crossing found, using coarse value")
    return f_res_coarse, Z11_f, freqs_f


# --- Two strip lengths ---
# Choose lengths to have well-separated resonances within 1-4 GHz

L_A = 40e-3   # expect ~2 GHz
L_B = 30e-3   # expect ~2.7 GHz
L_C = 50e-3   # expect ~1.6 GHz (optional 3rd length)

print("\n=== Finding resonances ===")

print(f"\nStrip A (L={L_A*1e3:.0f}mm):")
f_res_A, _, _ = find_resonance(L_A, [1.0e9, 3.5e9])

print(f"\nStrip B (L={L_B*1e3:.0f}mm):")
f_res_B, _, _ = find_resonance(L_B, [1.5e9, 4.5e9])

print(f"\nStrip C (L={L_C*1e3:.0f}mm):")
f_res_C, _, _ = find_resonance(L_C, [0.8e9, 3.0e9])

# --- Dual-length extraction ---
print(f"\n{'='*60}")
print(f"Dual-length resonance extraction")
print(f"{'='*60}")

if f_res_A and f_res_B:
    # f_A = c0 / (2*(L_A + 2*dL)*sqrt(eps_eff))
    # f_B = c0 / (2*(L_B + 2*dL)*sqrt(eps_eff))
    # Dividing: f_A/f_B = (L_B + 2*dL)/(L_A + 2*dL)
    # => (L_A + 2*dL)*f_A = (L_B + 2*dL)*f_B
    # => L_A*f_A - L_B*f_B = 2*dL*(f_B - f_A)
    # => dL = (L_A*f_A - L_B*f_B) / (2*(f_B - f_A))

    dL_AB = (L_A * f_res_A - L_B * f_res_B) / (2.0 * (f_res_B - f_res_A))
    L_A_eff = L_A + 2 * dL_AB
    eps_eff_AB = (c0 / (2 * L_A_eff * f_res_A)) ** 2

    print(f"\n  A-B pair:")
    print(f"    f_A = {f_res_A/1e9:.4f} GHz (L={L_A*1e3:.0f}mm)")
    print(f"    f_B = {f_res_B/1e9:.4f} GHz (L={L_B*1e3:.0f}mm)")
    print(f"    Extracted dL = {dL_AB*1e3:.3f} mm")
    print(f"    L_A_eff = {L_A_eff*1e3:.2f} mm")
    print(f"    eps_eff = {eps_eff_AB:.3f} (ref: {eps_eff_ref:.3f}, err: {abs(eps_eff_AB-eps_eff_ref)/eps_eff_ref*100:.1f}%)")

    # Single-length extraction for comparison (no fringing correction)
    eps_eff_A_raw = (c0 / (2 * L_A * f_res_A)) ** 2
    eps_eff_B_raw = (c0 / (2 * L_B * f_res_B)) ** 2
    print(f"\n  Single-length (raw, no fringing correction):")
    print(f"    eps_eff from A = {eps_eff_A_raw:.3f} (err: {abs(eps_eff_A_raw-eps_eff_ref)/eps_eff_ref*100:.1f}%)")
    print(f"    eps_eff from B = {eps_eff_B_raw:.3f} (err: {abs(eps_eff_B_raw-eps_eff_ref)/eps_eff_ref*100:.1f}%)")

if f_res_A and f_res_C:
    dL_AC = (L_A * f_res_A - L_C * f_res_C) / (2.0 * (f_res_C - f_res_A))
    L_A_eff_ac = L_A + 2 * dL_AC
    eps_eff_AC = (c0 / (2 * L_A_eff_ac * f_res_A)) ** 2

    print(f"\n  A-C pair:")
    print(f"    f_A = {f_res_A/1e9:.4f} GHz (L={L_A*1e3:.0f}mm)")
    print(f"    f_C = {f_res_C/1e9:.4f} GHz (L={L_C*1e3:.0f}mm)")
    print(f"    Extracted dL = {dL_AC*1e3:.3f} mm")
    print(f"    eps_eff = {eps_eff_AC:.3f} (ref: {eps_eff_ref:.3f}, err: {abs(eps_eff_AC-eps_eff_ref)/eps_eff_ref*100:.1f}%)")

if f_res_B and f_res_C:
    dL_BC = (L_B * f_res_B - L_C * f_res_C) / (2.0 * (f_res_C - f_res_B))
    L_B_eff_bc = L_B + 2 * dL_BC
    eps_eff_BC = (c0 / (2 * L_B_eff_bc * f_res_B)) ** 2

    print(f"\n  B-C pair:")
    print(f"    f_B = {f_res_B/1e9:.4f} GHz (L={L_B*1e3:.0f}mm)")
    print(f"    f_C = {f_res_C/1e9:.4f} GHz (L={L_C*1e3:.0f}mm)")
    print(f"    Extracted dL = {dL_BC*1e3:.3f} mm")
    print(f"    eps_eff = {eps_eff_BC:.3f} (ref: {eps_eff_ref:.3f}, err: {abs(eps_eff_BC-eps_eff_ref)/eps_eff_ref*100:.1f}%)")

# Expected fringing correction (Hammerstad)
wh = W / H
dL_expected = 0.412 * H * (eps_eff_ref + 0.3) * (wh + 0.264) / ((eps_eff_ref - 0.258) * (wh + 0.8))
print(f"\n  Expected dL (Hammerstad) = {dL_expected*1e3:.3f} mm")
