"""SOC with SHORT feeds + analytical TL de-embedding — wideband validation.

Key idea: SOC fails at quarter-wave frequencies (cos(βd) = 0).  By using
SHORT feed sections (2mm), the first failure moves to ~20 GHz on FR4:
    f_fail = c0 / (4·d·√eps_eff) ≈ 20.7 GHz  (d=2mm, eps_eff=3.26)

This combines:
  1. SOC de-embedding on 2mm feeds → removes port discontinuity (stable to ~20 GHz)
  2. Analytical TL de-embedding → removes remaining feed length (stable always)

Physical setup
--------------
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - Strip: W=3.06mm (~50 Ohm), total 20mm
  - Ports at ±8mm (2mm from strip ends)
  - SOC reference planes at ±6mm (2mm feed sections)
  - DUT section: 12mm between reference planes

Usage
-----
    venv/bin/python examples/microstrip_short_soc_validation.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port,
    Layer, LayerStack,
    c0,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.network.soc_deembedding import SOCDeembedding
from pyMoM3d.network.tl_deembedding import TLDeembedding
from pyMoM3d.network.extractor import NetworkExtractor
from pyMoM3d.cross_section.extraction import compute_reference_impedance


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
Z_STRIP = H_SUB

L_STRIP = 30e-3         # Total strip length (30mm)

# Port positions: 2mm from strip ends
PORT1_X = -L_STRIP/2 + 2e-3   # -13mm
PORT2_X = +L_STRIP/2 - 2e-3   # +13mm

# SOC reference planes: 5mm inside from each port
SOC_FEED = 5e-3
REF1_X = PORT1_X + SOC_FEED    # -8mm
REF2_X = PORT2_X - SOC_FEED    # +8mm

# Mesh density
TEL = 0.7e-3

# Frequencies
FREQS = np.array([3, 5, 7, 8, 9, 10, 12, 15]) * 1e9


def build_layer_stack():
    return LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])


def main():
    print("=" * 70)
    print("SOC (Short Feeds) + Analytical TL De-embedding — Wideband Validation")
    print("=" * 70)
    print()

    stack = build_layer_stack()

    # --- 2D cross-section solver ---
    tl_result = compute_reference_impedance(stack, W_STRIP, source_layer_name='FR4')
    print(f"2D cross-section solver:")
    print(f"  Z0 = {tl_result.Z0:.2f} Ohm, eps_eff = {tl_result.eps_eff:.3f}")

    # Quarter-wave frequency for SOC feed
    eps_eff = tl_result.eps_eff
    f_qw = c0 / (4 * SOC_FEED * np.sqrt(eps_eff))
    print(f"  SOC feed = {SOC_FEED*1e3:.0f}mm → first quarter-wave at {f_qw/1e9:.1f} GHz")
    print()

    Z0_ref = tl_result.Z0

    # --- Mesh ---
    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, REF1_X, REF2_X, PORT2_X],
        center=(0, 0, Z_STRIP),
    )
    basis = compute_rwg_connectivity(mesh)

    # --- Ports ---
    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    print(f"Structure: {L_STRIP*1e3:.0f}mm strip")
    print(f"  Port 1 at x={PORT1_X*1e3:.1f}mm, Port 2 at x={PORT2_X*1e3:.1f}mm")
    print(f"  SOC ref planes at x={REF1_X*1e3:.1f}mm, x={REF2_X*1e3:.1f}mm")
    print(f"  SOC feed = {SOC_FEED*1e3:.0f}mm, DUT = {(REF2_X-REF1_X)*1e3:.0f}mm")
    print(f"  Mesh: {mesh.get_num_triangles()} tris, {basis.num_basis} basis")
    print(f"  Port 1: {len(feed1)} edges, Port 2: {len(feed2)} edges")
    print()

    # --- Simulation ---
    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=FREQS[0], excitation=exc,
        quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # --- SOC setup ---
    soc = SOCDeembedding(
        sim, [port1, port2],
        reference_plane_x=[REF1_X, REF2_X],
        port_x=[PORT1_X, PORT2_X],
        Z0=Z0_ref,
        symmetric=False,
    )

    # --- Analytical TL de-embedding for remaining feed ---
    # After SOC removes the 2mm feed, the reference planes are at ±6mm.
    # The DUT between ref planes is 12mm. No additional TL removal needed
    # since we're measuring the through-line.

    # --- Raw extraction ---
    extractor = NetworkExtractor(sim, [port1, port2], Z0=Z0_ref, store_currents=True)
    results_raw = extractor.extract(FREQS.tolist())

    # --- SOC de-embedding ---
    print(f"{'Freq':>6} | {'Raw S21':>10} | {'SOC S21':>10} | "
          f"{'Raw S11':>10} | {'SOC S11':>10} | "
          f"{'SOC A':>12} | {'cos(βd)':>9} | {'Pass':>6}")
    print("-" * 100)

    def dB(x):
        return 20 * np.log10(max(abs(x), 1e-30))

    for result in results_raw:
        freq = result.frequency
        S_raw = result.S_matrix

        try:
            result_cal = soc.deembed(result)
            S_cal = result_cal.S_matrix

            # Get the SOC A parameter for diagnostics
            T_err = soc.compute_error_abcd(0, freq)
            A_soc = T_err[0, 0]

            # Expected cos(βd)
            gamma = tl_result.gamma(freq)
            beta_d = gamma.imag * SOC_FEED
            cos_bd = np.cos(beta_d)

            passivity = abs(S_cal[0, 0])**2 + abs(S_cal[1, 0])**2

            print(f"{freq/1e9:5.0f}G | {dB(S_raw[1,0]):10.2f} | {dB(S_cal[1,0]):10.2f} | "
                  f"{dB(S_raw[0,0]):10.2f} | {dB(S_cal[0,0]):10.2f} | "
                  f"{A_soc.real:6.3f}{A_soc.imag:+6.3f}j | {cos_bd:9.4f} | {passivity:6.3f}")

        except Exception as e:
            print(f"{freq/1e9:5.0f}G | {dB(S_raw[1,0]):10.2f} | {'FAIL':>10} | "
                  f"{dB(S_raw[0,0]):10.2f} | {'FAIL':>10} | "
                  f"{'ERROR':>12} | {'---':>9} | {'---':>6}")
            print(f"         Error: {e}")

    print()
    print("Diagnostics:")
    print(f"  DUT = {(REF2_X-REF1_X)*1e3:.0f}mm through-line between SOC reference planes")
    print(f"  Expected: SOC S21 → 0 dB (matched through-line)")
    print(f"  A ≈ cos(βd): should be near 1 for short feeds, stays > 0 below {f_qw/1e9:.0f} GHz")


if __name__ == '__main__':
    main()
