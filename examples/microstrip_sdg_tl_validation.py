"""Strip delta-gap + analytical TL de-embedding validation.

Tests the production pipeline for microstrip S-parameter extraction:
  1. Strip delta-gap ports (proven coupling to microstrip TEM mode)
  2. Analytical TL de-embedding using Z0/γ from 2D cross-section solver
  3. Port calibration via short through-standard

Physical setup
--------------
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - PEC ground at z=0 (layered Green's function via Strata)
  - Strip: W=3.06mm (~50 Ohm), at z=1.6mm (FR4/air interface)
  - DUT structure: |<-feed1->|<---DUT--->|<-feed2->|
  - Strip delta-gap ports at feed boundaries

Expected results
----------------
  For a matched through-line: S21 → 0 dB, S11 → -∞ dB.
  The delta-gap port artifact is a small reactive discontinuity that
  the port calibration should remove.

Usage
-----
    venv/bin/python examples/microstrip_sdg_tl_validation.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor,
    Layer, LayerStack,
    c0,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.network.tl_deembedding import TLDeembedding
from pyMoM3d.network.port_calibration import PortCalibration
from pyMoM3d.cross_section.extraction import compute_reference_impedance


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R = 4.4            # FR4 relative permittivity
H_SUB = 1.6e-3         # Substrate height (m)
W_STRIP = 3.06e-3      # Strip width (m)
Z_STRIP = H_SUB        # Strip at FR4/air interface

# Structure lengths
D_FEED = 5e-3          # 5mm feed sections
D_DUT = 10e-3          # 10mm DUT
L_TOTAL = 2 * D_FEED + D_DUT  # 20mm total

# Mesh density
TEL = 0.7e-3           # ~4 cells across strip width

# Calibration standard
D_CAL = 2e-3           # 2mm through-line between ports
CAL_MARGIN = 1e-3      # 1mm margin outside each port

# Frequencies
FREQS = np.array([3, 5, 7, 8, 9, 10, 12, 15]) * 1e9


def build_layer_stack():
    return LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])


def build_structure(total_length, port_x_list, stack, tel=TEL):
    """Mesh a strip and set up strip delta-gap ports."""
    mesher = GmshMesher(target_edge_length=tel)
    mesh = mesher.mesh_plate_with_feeds(
        width=total_length,
        height=W_STRIP,
        feed_x_list=port_x_list,
        center=(0, 0, Z_STRIP),
    )
    basis = compute_rwg_connectivity(mesh)

    ports = []
    for i, px in enumerate(port_x_list):
        feed_idx = find_feed_edges(mesh, basis, feed_x=px)
        signs = compute_feed_signs(mesh, basis, feed_idx)
        p = Port(name=f'P{i+1}', feed_basis_indices=feed_idx, feed_signs=signs)
        ports.append(p)

    # Build Simulation
    exc_dummy = StripDeltaGapExcitation(
        feed_basis_indices=ports[0].feed_basis_indices, voltage=1.0
    )
    config = SimulationConfig(
        frequency=FREQS[0],
        excitation=exc_dummy,
        quad_order=4,
        backend='auto',
        layer_stack=stack,
        source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    return sim, basis, ports


def extract_s_params(sim, basis, ports, freq, Z0_ref=50.0):
    """Extract 2-port S-params at a single frequency using full-wave MPIE."""
    from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator
    from pyMoM3d.greens.layered.sommerfeld import LayeredGreensFunction
    from pyMoM3d.mom.assembly import fill_matrix

    k = 2 * np.pi * freq / c0

    gf = LayeredGreensFunction(
        sim.config.layer_stack,
        frequency=freq,
        source_layer_name=sim.config.source_layer_name,
    )

    op = MultilayerEFIEOperator(gf)
    Z = fill_matrix(
        op, basis, sim.mesh, k, gf.wave_impedance,
        quad_order=4, backend='auto',
    )

    V_all = np.column_stack([p.build_excitation_vector(basis) for p in ports])
    I_all = np.linalg.solve(Z, V_all)

    P = len(ports)
    Y = np.zeros((P, P), dtype=complex)
    for q in range(P):
        for p in range(P):
            Y[q, p] = ports[q].terminal_current(I_all[:, p], basis) / ports[p].V_ref

    Z_net = np.linalg.inv(Y)
    I_eye = np.eye(P)
    S = (Z_net / Z0_ref - I_eye) @ np.linalg.inv(Z_net / Z0_ref + I_eye)
    return S


def main():
    print("=" * 70)
    print("Strip Delta-Gap + Analytical TL De-embedding Validation")
    print("=" * 70)
    print()

    stack = build_layer_stack()

    # --- 2D cross-section solver: get Z0, γ ---
    print("2D cross-section solver:")
    tl_result = compute_reference_impedance(stack, W_STRIP, source_layer_name='FR4')
    print(f"  Z0 = {tl_result.Z0:.2f} Ohm")
    print(f"  eps_eff = {tl_result.eps_eff:.3f}")
    print(f"  v_phase = {tl_result.v_phase/c0:.4f} c0")
    print()

    Z0_ref = tl_result.Z0

    # --- DUT structure ---
    x_left = -L_TOTAL / 2
    x_port1 = x_left + D_FEED
    x_port2 = x_left + D_FEED + D_DUT

    print(f"DUT structure: {L_TOTAL*1e3:.0f}mm total")
    print(f"  Feed1: {D_FEED*1e3:.0f}mm | DUT: {D_DUT*1e3:.0f}mm | Feed2: {D_FEED*1e3:.0f}mm")
    print(f"  Port 1 at x = {x_port1*1e3:.1f}mm, Port 2 at x = {x_port2*1e3:.1f}mm")

    sim_dut, basis_dut, ports_dut = build_structure(
        L_TOTAL, [x_port1, x_port2], stack
    )
    print(f"  Mesh: {sim_dut.mesh.get_num_triangles()} tris, {basis_dut.num_basis} basis")
    print(f"  Port 1: {len(ports_dut[0].feed_basis_indices)} edges")
    print(f"  Port 2: {len(ports_dut[1].feed_basis_indices)} edges")
    print()

    # --- Calibration structure ---
    cal_total = D_CAL + 2 * CAL_MARGIN
    cal_px = [-D_CAL / 2, D_CAL / 2]
    sim_cal, basis_cal, ports_cal = build_structure(
        cal_total, cal_px, stack
    )
    print(f"Cal structure: {D_CAL*1e3:.0f}mm through-line ({cal_total*1e3:.0f}mm total)")
    print(f"  Mesh: {sim_cal.mesh.get_num_triangles()} tris, {basis_cal.num_basis} basis")
    print()

    # --- TL de-embedding setup ---
    deemb = TLDeembedding.from_cross_section(tl_result, d1=D_FEED, d2=D_FEED, Z0_ref=Z0_ref)

    # --- Frequency sweep ---
    print(f"{'Freq':>6} | {'Raw S21':>10} | {'TL S21':>10} | {'Cal+TL S21':>12} | "
          f"{'Raw S11':>10} | {'TL S11':>10} | {'Cal+TL S11':>12} | {'Pass':>6}")
    print("-" * 100)

    def dB(x):
        return 20 * np.log10(max(abs(x), 1e-30))

    for freq in FREQS:
        # Calibration
        S_cal = extract_s_params(sim_cal, basis_cal, ports_cal, freq, Z0_ref)
        gamma = tl_result.gamma(freq)
        cal = PortCalibration.from_through_standard(
            S_cal, tl_result.Z0, gamma, D_CAL, freq, Z0_ref=Z0_ref
        )

        # DUT measurement
        S_raw = extract_s_params(sim_dut, basis_dut, ports_dut, freq, Z0_ref)

        # De-embed: TL only
        S_tl = deemb.deembed(S_raw, freq)

        # De-embed: calibration + TL
        S_cal_tl = deemb.deembed(S_raw, freq, port_cal=cal)

        passivity = abs(S_cal_tl[0, 0])**2 + abs(S_cal_tl[0, 1])**2

        print(f"{freq/1e9:5.0f}G | {dB(S_raw[0,1]):10.2f} | {dB(S_tl[0,1]):10.2f} | "
              f"{dB(S_cal_tl[0,1]):12.2f} | {dB(S_raw[0,0]):10.2f} | "
              f"{dB(S_tl[0,0]):10.2f} | {dB(S_cal_tl[0,0]):12.2f} | {passivity:6.3f}")

    # --- Detailed diagnostics at 10 GHz ---
    print()
    print("Diagnostics at 10 GHz:")
    freq = 10e9

    S_cal = extract_s_params(sim_cal, basis_cal, ports_cal, freq, Z0_ref)
    cal = PortCalibration.from_through_standard(
        S_cal, tl_result.Z0, tl_result.gamma(freq), D_CAL, freq, Z0_ref=Z0_ref
    )

    S_raw = extract_s_params(sim_dut, basis_dut, ports_dut, freq, Z0_ref)
    S_tl = deemb.deembed(S_raw, freq)
    S_both = deemb.deembed(S_raw, freq, port_cal=cal)

    print(f"  Port error T_err (should be near-identity):")
    print(f"    diag: [{cal.T_err[0,0]:.4f}, {cal.T_err[1,1]:.4f}]")
    print(f"    off:  [{cal.T_err[0,1]:.4f}, {cal.T_err[1,0]:.4f}]")
    print(f"    det = {np.linalg.det(cal.T_err):.6f}")
    print()
    print(f"  Raw:     |S21| = {abs(S_raw[0,1]):.4f} ({dB(S_raw[0,1]):.1f} dB)")
    print(f"  TL:      |S21| = {abs(S_tl[0,1]):.4f} ({dB(S_tl[0,1]):.1f} dB)")
    print(f"  Cal+TL:  |S21| = {abs(S_both[0,1]):.4f} ({dB(S_both[0,1]):.1f} dB)")
    print(f"  Reciprocity: |S12-S21| = {abs(S_both[0,1] - S_both[1,0]):.2e}")


if __name__ == '__main__':
    main()
