"""Analytical TL de-embedding validation.

End-to-end test: mesh a microstrip through-line with feedline sections,
extract S-parameters with half-RWG ports, then de-embed the feedlines
analytically.  After de-embedding, S21 of the DUT section should
approach 0 dB (ideal through-line in matched system).

Structure
---------
    |<-- feed1 -->|<------ DUT ------>|<-- feed2 -->|
    |   d_feed    |     d_dut         |   d_feed    |
    P1            ref1               ref2            P2

P1, P2 = half-RWG port gaps
ref1, ref2 = reference planes (port locations)
The DUT is a uniform through-line between the reference planes.

De-embedding removes the feedline sections analytically using Z0 and γ
from the 2D cross-section solver.

Usage
-----
    venv/bin/python examples/tl_deembedding_validation.py
"""

import sys
import numpy as np

sys.path.insert(0, 'src')

from pyMoM3d.mesh import (
    GmshMesher,
    compute_rwg_connectivity,
    split_mesh_at_ports,
    add_half_rwg_basis,
)
from pyMoM3d.network.port import Port
from pyMoM3d.network.tl_deembedding import TLDeembedding, tl_abcd
from pyMoM3d.network.soc_deembedding import abcd_to_s, s_to_abcd
from pyMoM3d.mom.operators.efie import EFIEOperator
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.utils.constants import c0, eta0


def extract_2port(mesh, rwg_basis, ports, freq, Z0_ref=50.0):
    """Extract 2-port S-parameters at a single frequency."""
    k = 2 * np.pi * freq / c0

    op = EFIEOperator()
    Z = fill_matrix(op, rwg_basis, mesh, k, eta0, backend='numpy')

    V_all = np.column_stack([p.build_excitation_vector(rwg_basis) for p in ports])
    I_all = np.linalg.solve(Z, V_all)

    P = len(ports)
    Y = np.zeros((P, P), dtype=complex)
    for q in range(P):
        for p in range(P):
            Y[q, p] = ports[q].terminal_current(I_all[:, p], rwg_basis) / ports[p].V_ref

    Z_net = np.linalg.inv(Y)
    I_eye = np.eye(P)
    S = (Z_net / Z0_ref - I_eye) @ np.linalg.inv(Z_net / Z0_ref + I_eye)
    return S


def main():
    print("=" * 70)
    print("Analytical TL De-embedding Validation")
    print("=" * 70)
    print()

    # Structure parameters
    strip_width = 0.003    # 3mm width (y)
    d_feed = 5e-3          # 5mm feedlines
    d_dut = 10e-3          # 10mm DUT
    total_length = 2 * d_feed + d_dut  # 20mm total

    x_left = -total_length / 2
    x_port1 = x_left + d_feed      # left port
    x_port2 = x_left + d_feed + d_dut  # right port

    print(f"Strip: {total_length*1e3:.0f}mm x {strip_width*1e3:.1f}mm")
    print(f"Feed: {d_feed*1e3:.0f}mm | DUT: {d_dut*1e3:.0f}mm | Feed: {d_feed*1e3:.0f}mm")
    print(f"Port 1 at x = {x_port1*1e3:.1f}mm, Port 2 at x = {x_port2*1e3:.1f}mm")
    print()

    # Mesh
    target_lc = 1e-3
    mesher = GmshMesher(target_edge_length=target_lc)
    base_mesh = mesher.mesh_plate_with_feeds(
        width=total_length,
        height=strip_width,
        feed_x_list=[x_port1, x_port2],
        center=(0, 0, 0),
    )

    split_mesh, _ = split_mesh_at_ports(base_mesh, [x_port1, x_port2])
    rwg = compute_rwg_connectivity(split_mesh)
    ext_basis, _ = add_half_rwg_basis(split_mesh, rwg, x_port1)
    ext_basis, _ = add_half_rwg_basis(split_mesh, ext_basis, x_port2)

    p1 = Port.from_nonradiating_gap(split_mesh, ext_basis, x_port1, name='P1')
    p2 = Port.from_nonradiating_gap(split_mesh, ext_basis, x_port2, name='P2')

    print(f"Mesh: {split_mesh.get_num_triangles()} tris, {ext_basis.num_basis} basis")
    print(f"Port 1: {len(p1.feed_basis_indices)} half-RWG edges")
    print(f"Port 2: {len(p2.feed_basis_indices)} half-RWG edges")
    print()

    # TL de-embedding parameters (free-space strip, Z0 ≈ analytical)
    # For a free-space strip, use free-space propagation
    Z0_tl = 120.0  # approximate Z0 for free-space strip
    def gamma_func(freq):
        return 1j * 2 * np.pi * freq / c0

    deemb = TLDeembedding(Z0=Z0_tl, gamma_func=gamma_func, d1=d_feed, d2=d_feed)

    # Frequency sweep
    freqs = np.array([2, 5, 8, 10, 12, 15]) * 1e9

    print(f"{'Freq (GHz)':>10} | {'Raw S21 (dB)':>12} | {'De-emb S21 (dB)':>15} | "
          f"{'Raw S11 (dB)':>12} | {'De-emb S11 (dB)':>15} | {'Passivity':>10}")
    print("-" * 95)

    for freq in freqs:
        S_raw = extract_2port(split_mesh, ext_basis, [p1, p2], freq)
        S_de = deemb.deembed(S_raw, freq)

        s21_raw = 20 * np.log10(max(abs(S_raw[0, 1]), 1e-30))
        s21_de = 20 * np.log10(max(abs(S_de[0, 1]), 1e-30))
        s11_raw = 20 * np.log10(max(abs(S_raw[0, 0]), 1e-30))
        s11_de = 20 * np.log10(max(abs(S_de[0, 0]), 1e-30))
        passivity = abs(S_de[0, 0])**2 + abs(S_de[0, 1])**2

        print(f"{freq/1e9:10.1f} | {s21_raw:12.2f} | {s21_de:15.2f} | "
              f"{s11_raw:12.2f} | {s11_de:15.2f} | {passivity:10.4f}")

    # Detailed diagnostics at 10 GHz
    print()
    print("Diagnostics at 10 GHz:")
    freq = 10e9
    S_raw = extract_2port(split_mesh, ext_basis, [p1, p2], freq)
    S_de = deemb.deembed(S_raw, freq)

    print(f"  Raw:      S11 = {S_raw[0,0]:.4f}, S21 = {S_raw[0,1]:.4f}")
    print(f"  De-embed: S11 = {S_de[0,0]:.4f}, S21 = {S_de[0,1]:.4f}")
    print(f"  |S21_raw| = {abs(S_raw[0,1]):.4f}, |S21_de| = {abs(S_de[0,1]):.4f}")
    print(f"  Reciprocity raw:  |S12-S21| = {abs(S_raw[0,1]-S_raw[1,0]):.2e}")
    print(f"  Reciprocity de:   |S12-S21| = {abs(S_de[0,1]-S_de[1,0]):.2e}")


if __name__ == '__main__':
    main()
