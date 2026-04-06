"""Half-RWG non-radiating port validation.

Compares the non-radiating (half-RWG) port model against the standard
strip delta-gap port on a microstrip through-line with layered media.

The non-radiating port (Liu et al. 2018) eliminates the spurious radiation
artifact of the delta-gap model by using half-RWG basis functions across
the port gap.  No current flows in the gap itself.

Test structure
--------------
- 50 Ohm microstrip on FR4 (eps_r = 4.4, h = 1.6mm)
- Strip width: 3.06mm (for ~50 Ohm Z0)
- Strip length: 30mm with ports at +/-10mm from center

Expected behavior
-----------------
- Both port models should give similar S21 at low frequencies
- Half-RWG should show better convergence with mesh refinement
- Both should satisfy reciprocity (S12 = S21) and passivity

Usage
-----
    venv/bin/python examples/half_rwg_port_validation.py
"""

import sys
import numpy as np

# Ensure src/ is on path
sys.path.insert(0, 'src')

from pyMoM3d.mesh import (
    GmshMesher,
    compute_rwg_connectivity,
    split_mesh_at_ports,
    add_half_rwg_basis,
)
from pyMoM3d.network.port import Port
from pyMoM3d.mom.operators.efie import EFIEOperator
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.utils.constants import c0, eta0


def extract_2port_sparams(mesh, rwg_basis, ports, freq, Z0=50.0):
    """Extract 2-port S-parameters at a single frequency."""
    k = 2 * np.pi * freq / c0
    eta = eta0

    op = EFIEOperator()
    Z = fill_matrix(op, rwg_basis, mesh, k, eta, backend='numpy')

    V_all = np.column_stack([p.build_excitation_vector(rwg_basis) for p in ports])
    I_all = np.linalg.solve(Z, V_all)

    P = len(ports)
    Y = np.zeros((P, P), dtype=complex)
    for q in range(P):
        for p in range(P):
            Y[q, p] = ports[q].terminal_current(I_all[:, p], rwg_basis) / ports[p].V_ref

    Z_net = np.linalg.inv(Y)
    I_eye = np.eye(P)
    S = (Z_net / Z0 - I_eye) @ np.linalg.inv(Z_net / Z0 + I_eye)
    return S, Z_net


def setup_half_rwg_ports(target_edge_length=1e-3):
    """Set up a strip with non-radiating (half-RWG) ports."""
    mesher = GmshMesher(target_edge_length=target_edge_length)
    port_x1, port_x2 = -0.010, 0.010

    # Create mesh with conformal edges at port locations
    base_mesh = mesher.mesh_plate_with_feeds(
        width=0.030,    # 30mm strip length (x)
        height=0.003,   # ~3mm strip width (y)
        feed_x_list=[port_x1, port_x2],
        center=(0, 0, 0),
    )

    # Split mesh to create port gaps
    split_mesh, _ = split_mesh_at_ports(base_mesh, [port_x1, port_x2])

    # Compute standard RWG and add half-RWG at ports
    rwg = compute_rwg_connectivity(split_mesh)
    ext_basis, _ = add_half_rwg_basis(split_mesh, rwg, port_x1)
    ext_basis, _ = add_half_rwg_basis(split_mesh, ext_basis, port_x2)

    # Create ports
    p1 = Port.from_nonradiating_gap(split_mesh, ext_basis, port_x1, name='P1')
    p2 = Port.from_nonradiating_gap(split_mesh, ext_basis, port_x2, name='P2')

    return split_mesh, ext_basis, [p1, p2]


def setup_delta_gap_ports(target_edge_length=1e-3):
    """Set up a strip with standard strip delta-gap ports."""
    from pyMoM3d.mom.excitation import find_feed_edges, compute_feed_signs

    mesher = GmshMesher(target_edge_length=target_edge_length)
    port_x1, port_x2 = -0.010, 0.010

    mesh = mesher.mesh_plate_with_feeds(
        width=0.030,
        height=0.003,
        feed_x_list=[port_x1, port_x2],
        center=(0, 0, 0),
    )

    rwg = compute_rwg_connectivity(mesh)

    # Standard delta-gap ports
    idx1 = find_feed_edges(mesh, rwg, feed_x=port_x1)
    idx2 = find_feed_edges(mesh, rwg, feed_x=port_x2)
    signs1 = compute_feed_signs(mesh, rwg, idx1)
    signs2 = compute_feed_signs(mesh, rwg, idx2)

    p1 = Port(name='P1', feed_basis_indices=idx1, feed_signs=signs1)
    p2 = Port(name='P2', feed_basis_indices=idx2, feed_signs=signs2)

    return mesh, rwg, [p1, p2]


def main():
    print("=" * 70)
    print("Half-RWG Non-Radiating Port Validation")
    print("=" * 70)
    print()

    target_lc = 1e-3  # 1mm mesh

    # Set up both port models
    print("Setting up half-RWG (non-radiating) ports...")
    hr_mesh, hr_basis, hr_ports = setup_half_rwg_ports(target_lc)
    print(f"  Mesh: {hr_mesh.get_num_triangles()} tris, "
          f"{hr_basis.num_basis} basis functions")
    print(f"  Port 1: {len(hr_ports[0].feed_basis_indices)} half-RWG edges")
    print(f"  Port 2: {len(hr_ports[1].feed_basis_indices)} half-RWG edges")

    print()
    print("Setting up standard strip delta-gap ports...")
    dg_mesh, dg_basis, dg_ports = setup_delta_gap_ports(target_lc)
    print(f"  Mesh: {dg_mesh.get_num_triangles()} tris, "
          f"{dg_basis.num_basis} basis functions")
    print(f"  Port 1: {len(dg_ports[0].feed_basis_indices)} delta-gap edges")
    print(f"  Port 2: {len(dg_ports[1].feed_basis_indices)} delta-gap edges")

    # Frequency sweep
    freqs = np.array([2, 5, 8, 10, 12, 15]) * 1e9
    print()
    print(f"{'Freq (GHz)':>10} | {'Half-RWG S21 (dB)':>18} | {'Delta-Gap S21 (dB)':>18} | "
          f"{'HR Passivity':>12} | {'DG Passivity':>12}")
    print("-" * 90)

    for freq in freqs:
        # Half-RWG extraction
        S_hr, _ = extract_2port_sparams(hr_mesh, hr_basis, hr_ports, freq)
        s21_hr = 20 * np.log10(max(abs(S_hr[0, 1]), 1e-30))
        passive_hr = np.sum(np.abs(S_hr[:, 0]) ** 2)

        # Delta-gap extraction
        S_dg, _ = extract_2port_sparams(dg_mesh, dg_basis, dg_ports, freq)
        s21_dg = 20 * np.log10(max(abs(S_dg[0, 1]), 1e-30))
        passive_dg = np.sum(np.abs(S_dg[:, 0]) ** 2)

        print(f"{freq/1e9:10.1f} | {s21_hr:18.2f} | {s21_dg:18.2f} | "
              f"{passive_hr:12.4f} | {passive_dg:12.4f}")

    # Final diagnostics
    print()
    print("Diagnostics at 10 GHz:")
    S_hr, Z_hr = extract_2port_sparams(hr_mesh, hr_basis, hr_ports, 10e9)
    S_dg, Z_dg = extract_2port_sparams(dg_mesh, dg_basis, dg_ports, 10e9)

    print(f"  Half-RWG:  Z11 = {Z_hr[0,0]:.2f}, Z21 = {Z_hr[0,1]:.2f}")
    print(f"  Delta-gap: Z11 = {Z_dg[0,0]:.2f}, Z21 = {Z_dg[0,1]:.2f}")
    print(f"  Half-RWG:  S11 = {20*np.log10(abs(S_hr[0,0])):.2f} dB, "
          f"S21 = {20*np.log10(abs(S_hr[0,1])):.2f} dB")
    print(f"  Delta-gap: S11 = {20*np.log10(abs(S_dg[0,0])):.2f} dB, "
          f"S21 = {20*np.log10(abs(S_dg[0,1])):.2f} dB")
    print(f"  Half-RWG reciprocity:  |S12-S21| = {abs(S_hr[0,1]-S_hr[1,0]):.2e}")
    print(f"  Delta-gap reciprocity: |S12-S21| = {abs(S_dg[0,1]-S_dg[1,0]):.2e}")


if __name__ == '__main__':
    main()
