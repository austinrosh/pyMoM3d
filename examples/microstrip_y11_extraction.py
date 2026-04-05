"""
Microstrip Z0/eps_eff Extraction via Dual-Stub Y11 Differential Method.

Extracts characteristic impedance Z0 and effective permittivity eps_eff
from the DIFFERENCE of Y11 measured on two different stub lengths.
The port discontinuity capacitance cancels in the subtraction:

    ΔY11 = Y11(L1) - Y11(L2) = Y_TL(L1) - Y_TL(L2)

This avoids the dominant port parasitic (~5-10 pF from strip delta-gap)
that prevents direct Y11-based or S-parameter extraction.

Physical setup:
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - PEC ground at z=0
  - Two strip lengths: L1=30mm and L2=15mm
  - 1-port strip delta-gap at one end of each

Produces:
  images/microstrip_dual_stub_extraction.png

Usage:
    source venv/bin/activate
    python examples/microstrip_y11_extraction.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor,
    Layer, LayerStack,
    configure_latex_style, c0,
    microstrip_z0_hammerstad,
)
from pyMoM3d.network.tl_extraction import extract_tl_dual_stub
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)


def build_microstrip_1port(L, W, H, eps_r, tel, stack):
    """Build a 1-port microstrip stub with port at one end."""
    mesher = GmshMesher(target_edge_length=tel)
    margin = tel / 2.0
    port_x = -L / 2.0 + margin

    mesh = mesher.mesh_plate_with_feeds(
        width=L, height=W,
        feed_x_list=[port_x],
        center=(0.0, 0.0, H),
    )
    basis = compute_rwg_connectivity(mesh)

    feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
    signs = compute_feed_signs(mesh, basis, feed_edges)
    port = Port(name='P1', feed_basis_indices=feed_edges, feed_signs=signs)

    exc = StripDeltaGapExcitation(feed_basis_indices=feed_edges, voltage=1.0)
    config = SimulationConfig(
        frequency=1e9,
        excitation=exc,
        source_layer_name='FR4',
        backend='auto',
        quad_order=4,
        layer_stack=stack,
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
    extractor = NetworkExtractor(sim, [port])

    stub_length = L / 2.0 - port_x  # port to open end

    return extractor, stub_length, mesh, basis


def main():
    configure_latex_style(use_tex=False)

    # --- Physical parameters ---
    eps_r = 4.4
    H = 1.6e-3
    W = 3.06e-3
    tel = 1.0e-3

    L1 = 30e-3  # 30 mm stub
    L2 = 15e-3  # 15 mm stub

    Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, eps_r)
    print(f"Hammerstad-Jensen reference: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

    # --- Layer stack ---
    stack = LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H, eps_r=eps_r),
        Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
    ])

    # --- Build two stubs ---
    print(f"\nBuilding stub 1: L = {L1*1e3:.0f} mm")
    ext1, sL1, mesh1, basis1 = build_microstrip_1port(L1, W, H, eps_r, tel, stack)
    print(f"  {mesh1.get_statistics()['num_triangles']} triangles, "
          f"{basis1.num_basis} RWG, stub = {sL1*1e3:.1f} mm")

    print(f"Building stub 2: L = {L2*1e3:.0f} mm")
    ext2, sL2, mesh2, basis2 = build_microstrip_1port(L2, W, H, eps_r, tel, stack)
    print(f"  {mesh2.get_statistics()['num_triangles']} triangles, "
          f"{basis2.num_basis} RWG, stub = {sL2*1e3:.1f} mm")

    # --- Frequency sweep ---
    freqs = np.linspace(1.0, 8.0, 10)
    freqs_hz = (freqs * 1e9).tolist()
    print(f"\nFrequencies: {[f'{f:.1f}' for f in freqs]} GHz")

    # --- Sweep both stubs ---
    print("\nSweeping stub 1...")
    results1 = ext1.extract(freqs_hz)
    Y11_L1 = np.array([r.Y_matrix[0, 0] for r in results1])

    print("Sweeping stub 2...")
    results2 = ext2.extract(freqs_hz)
    Y11_L2 = np.array([r.Y_matrix[0, 0] for r in results2])

    # --- Print raw data ---
    dY11 = Y11_L1 - Y11_L2
    print(f"\n  {'f':>6}  {'Im(Y11_L1)':>12}  {'Im(Y11_L2)':>12}  {'Im(dY11)':>12}")
    for i, f in enumerate(freqs):
        print(f"  {f:6.1f}  {Y11_L1[i].imag:12.6f}  {Y11_L2[i].imag:12.6f}  {dY11[i].imag:12.6f}")

    # --- Dual-stub extraction ---
    print("\nRunning dual-stub extraction...")
    result = extract_tl_dual_stub(
        np.array(freqs_hz), Y11_L1, Y11_L2, sL1, sL2,
        Z0_guess=Z0_ref, eps_eff_guess=eps_eff_ref,
    )

    print(f"\n{'='*55}")
    print(f"Dual-Stub Extraction Results")
    print(f"{'='*55}")
    print(f"  Z0       = {result.Z0:.2f} Ohm  (ref: {Z0_ref:.2f} Ohm)")
    print(f"  eps_eff  = {result.eps_eff:.3f}  (ref: {eps_eff_ref:.3f})")
    print(f"  C_port   = {result.C_port*1e12:.3f} pF  (estimated)")
    print(f"  Residual = {result.residual_norm:.2e}")

    z0_err = abs(result.Z0 - Z0_ref) / Z0_ref * 100
    ee_err = abs(result.eps_eff - eps_eff_ref) / eps_eff_ref * 100
    print(f"\n  Z0 error:     {z0_err:.1f}%")
    print(f"  eps_eff error: {ee_err:.1f}%")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    freq_GHz = np.array(freqs)

    ax = axes[0]
    ax.plot(freq_GHz, Y11_L1.imag * 1e3, 'bo-', markersize=5,
            label=f'Y11 (L={L1*1e3:.0f}mm)')
    ax.plot(freq_GHz, Y11_L2.imag * 1e3, 'rs-', markersize=5,
            label=f'Y11 (L={L2*1e3:.0f}mm)')
    ax.plot(freq_GHz, dY11.imag * 1e3, 'k^-', markersize=5, linewidth=2,
            label='dY11 = Y11(L1) - Y11(L2)')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Im(Y11) (mS)')
    ax.set_title('Raw Y11 and Differential')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    categories = ['Z0 (Ohm)', 'eps_eff']
    extracted = [result.Z0, result.eps_eff]
    reference = [Z0_ref, eps_eff_ref]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, extracted, width, label='Dual-Stub', color='steelblue')
    ax.bar(x + width/2, reference, width, label='Hammerstad-Jensen', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_title(f'Z0 err = {z0_err:.1f}%, eps_eff err = {ee_err:.1f}%')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Microstrip Dual-Stub Extraction (W={W*1e3:.2f}mm, h={H*1e3:.1f}mm, er={eps_r})')
    plt.tight_layout()

    os.makedirs('images', exist_ok=True)
    plt.savefig('images/microstrip_dual_stub_extraction.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to images/microstrip_dual_stub_extraction.png")

    return result


if __name__ == '__main__':
    result = main()
