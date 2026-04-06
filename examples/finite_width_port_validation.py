"""Finite-width port + variational Y extraction validation.

Compares four port extraction methods on a microstrip through-line:
  1. Delta-gap + direct Y (baseline)
  2. Delta-gap + variational Y
  3. Finite-width + direct Y
  4. Finite-width + variational Y

All methods include analytical TL de-embedding using Z0/gamma from
the 2D cross-section solver.

Expected result: S21 closer to 0 dB with finite-width and/or
variational extraction compared to the delta-gap baseline.

Physical setup
--------------
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - Strip: W=3.06mm (~50 Ohm), total 20mm
  - Ports at +/-5mm from center
  - TL de-embedding removes 5mm feeds

Usage
-----
    venv/bin/python examples/finite_width_port_validation.py
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
from pyMoM3d.network.tl_deembedding import TLDeembedding
from pyMoM3d.cross_section.extraction import compute_reference_impedance
from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator
from pyMoM3d.greens.layered.sommerfeld import LayeredGreensFunction
from pyMoM3d.mom.assembly import fill_matrix


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
Z_STRIP = H_SUB

D_FEED = 5e-3
D_DUT = 10e-3
L_TOTAL = 2 * D_FEED + D_DUT  # 20mm

TEL = 0.7e-3

FREQS = np.array([3, 5, 7, 8, 9, 10, 12, 15]) * 1e9


def build_layer_stack():
    return LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])


def extract_all_methods(sim, basis, ports, freq, Z0_ref, mesh):
    """Extract S-params using all four methods at one frequency.

    Returns dict with keys: 'dg_direct', 'dg_var', 'fw_direct', 'fw_var'
    """
    k = 2 * np.pi * freq / c0

    gf = LayeredGreensFunction(
        sim.config.layer_stack,
        frequency=freq,
        source_layer_name=sim.config.source_layer_name,
    )
    op = MultilayerEFIEOperator(gf)
    Z_sys = fill_matrix(
        op, basis, mesh, k, gf.wave_impedance,
        quad_order=4, backend='auto',
    )

    P = len(ports)
    results = {}

    for port_mode in ['dg', 'fw']:
        # Build RHS
        if port_mode == 'fw':
            V_all = np.column_stack([
                p.build_excitation_vector(basis, mesh=mesh)
                for p in ports
            ])
        else:
            V_all = np.column_stack([
                p.build_excitation_vector(basis)
                for p in ports
            ])

        I_all = np.linalg.solve(Z_sys, V_all)

        for y_mode in ['direct', 'var']:
            if y_mode == 'var':
                # Variational: Y[q,p] = 2*<V_q,I_p>/(V_q*V_p) + <I_q,Z*I_p>/(V_q*V_p)
                ZI = Z_sys @ I_all
                Y = np.zeros((P, P), dtype=complex)
                for q in range(P):
                    for p in range(P):
                        V_q = ports[q].V_ref
                        V_p = ports[p].V_ref
                        t1 = 2.0 * (V_all[:, q] @ I_all[:, p])
                        t2 = I_all[:, q] @ ZI[:, p]
                        Y[q, p] = (t1 - t2) / (V_q * V_p)
            else:
                # Direct: Y[q,p] = I_term[q,p] / V_ref[p]
                Y = np.zeros((P, P), dtype=complex)
                for q in range(P):
                    for p in range(P):
                        Y[q, p] = (
                            ports[q].terminal_current(I_all[:, p], basis)
                            / ports[p].V_ref
                        )

            Z_net = np.linalg.inv(Y)
            I_eye = np.eye(P)
            S = (Z_net / Z0_ref - I_eye) @ np.linalg.inv(Z_net / Z0_ref + I_eye)
            results[f'{port_mode}_{y_mode}'] = S

    return results


def main():
    print("=" * 70)
    print("Finite-Width Port + Variational Y Extraction — Validation")
    print("=" * 70)
    print()

    stack = build_layer_stack()

    # --- 2D cross-section solver ---
    tl_result = compute_reference_impedance(stack, W_STRIP, source_layer_name='FR4')
    Z0_ref = tl_result.Z0
    print(f"2D cross-section: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {tl_result.eps_eff:.3f}")
    print()

    # --- Mesh ---
    mesher = GmshMesher(target_edge_length=TEL)
    x_left = -L_TOTAL / 2
    x_port1 = x_left + D_FEED
    x_port2 = x_left + D_FEED + D_DUT

    mesh = mesher.mesh_plate_with_feeds(
        width=L_TOTAL, height=W_STRIP,
        feed_x_list=[x_port1, x_port2],
        center=(0, 0, Z_STRIP),
    )
    basis = compute_rwg_connectivity(mesh)

    # --- Ports (delta-gap and finite-width share the same feed edges) ---
    feed1 = find_feed_edges(mesh, basis, feed_x=x_port1)
    feed2 = find_feed_edges(mesh, basis, feed_x=x_port2)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)

    # Create ports with gap_width = TEL for finite-width mode
    ports = [
        Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1,
             gap_width=TEL),
        Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2,
             gap_width=TEL),
    ]

    print(f"Structure: {L_TOTAL*1e3:.0f}mm strip, {mesh.get_num_triangles()} tris, "
          f"{basis.num_basis} basis")
    print(f"  Port 1: {len(feed1)} edges at x={x_port1*1e3:.1f}mm")
    print(f"  Port 2: {len(feed2)} edges at x={x_port2*1e3:.1f}mm")
    print(f"  Gap width = {TEL*1e3:.1f}mm (= target_edge_length)")
    print()

    # --- Simulation setup ---
    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=FREQS[0], excitation=exc,
        quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # --- TL de-embedding ---
    deemb = TLDeembedding.from_cross_section(
        tl_result, d1=D_FEED, d2=D_FEED, Z0_ref=Z0_ref,
    )

    # --- Frequency sweep ---
    def dB(x):
        return 20 * np.log10(max(abs(x), 1e-30))

    header = (f"{'Freq':>6} | {'DG+Dir S21':>12} | {'DG+Var S21':>12} | "
              f"{'FW+Dir S21':>12} | {'FW+Var S21':>12} | "
              f"{'DG+Dir S11':>12} | {'FW+Var S11':>12}")
    print(header)
    print("-" * len(header))

    for freq in FREQS:
        S_all = extract_all_methods(sim, basis, ports, freq, Z0_ref, mesh)

        # De-embed all
        S_de = {}
        for key, S_raw in S_all.items():
            S_de[key] = deemb.deembed(S_raw, freq)

        print(
            f"{freq/1e9:5.0f}G | "
            f"{dB(S_de['dg_direct'][0,1]):12.2f} | "
            f"{dB(S_de['dg_var'][0,1]):12.2f} | "
            f"{dB(S_de['fw_direct'][0,1]):12.2f} | "
            f"{dB(S_de['fw_var'][0,1]):12.2f} | "
            f"{dB(S_de['dg_direct'][0,0]):12.2f} | "
            f"{dB(S_de['fw_var'][0,0]):12.2f}"
        )

    print()
    print("Legend:")
    print("  DG = Delta-gap (standard), FW = Finite-width (Lo/Jiang/Chew 2013)")
    print("  Dir = Direct Y extraction, Var = Variational Y (second-order)")
    print("  All include analytical TL de-embedding (5mm feeds removed)")
    print()
    print("Expected: S21 closer to 0 dB with FW and/or Var methods")


if __name__ == '__main__':
    main()
