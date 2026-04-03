"""PEEC spiral inductor characterization.

Demonstrates the PEEC extraction module for square spiral inductors
with two configurations:
1. Free-space (no substrate) — shows flat L(f) and monotonically increasing Q
2. On-chip with oxide/substrate — shows SRF, Q peak, and pi-model fitting

Usage
-----
    venv/bin/python examples/peec_spiral_validation.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    ConductorProperties,
    wheeler_inductance,
    InductorCharacterization,
)
from pyMoM3d.peec import PEECExtractor, Trace, TraceNetwork, PEECPort
from pyMoM3d.visualization import (
    configure_latex_style,
    plot_inductor_characterization,
)


def demo_free_space():
    """Free-space spiral: flat L(f), monotonically increasing Q."""
    print("\n" + "=" * 60)
    print("Case 1: Free-Space Spiral Inductor")
    print("=" * 60)

    N_TURNS = 2.5
    D_OUT = 2e-3
    W_TRACE = 100e-6
    S_SPACE = 100e-6
    THICKNESS = 2e-6

    copper = ConductorProperties(sigma=5.8e7, thickness=THICKNESS, name='Cu')

    trace = Trace.rectangular_spiral(
        n_turns=N_TURNS, d_out=D_OUT, w_trace=W_TRACE, s_space=S_SPACE,
        thickness=THICKNESS, conductor=copper,
    )

    d_in = max(D_OUT - 2 * N_TURNS * (W_TRACE + S_SPACE), W_TRACE)
    L_wheeler = wheeler_inductance(N_TURNS, D_OUT, d_in)

    port = PEECPort('P1', positive_segment_idx=0)
    network = TraceNetwork([trace], [port])
    extractor = PEECExtractor(network)

    print(f"  Geometry: {N_TURNS} turns, D_out={D_OUT*1e3:.0f} mm, "
          f"w={W_TRACE*1e6:.0f} um, s={S_SPACE*1e6:.0f} um")
    print(f"  Segments: {trace.num_segments}")
    print(f"  Wheeler L: {L_wheeler*1e9:.3f} nH")
    print(f"  PEEC L:    {extractor.total_inductance_estimate*1e9:.3f} nH")
    print(f"  DC R:      {extractor.dc_resistance:.4f} Ohm")

    freqs = np.linspace(0.1e9, 20e9, 40)
    results = extractor.extract(freqs)
    char = InductorCharacterization(results)
    cr = char.characterize()

    print(f"\n{cr.summary()}")

    # L(f) flatness
    L_arr = cr.L[cr.L > 0]
    variation = (L_arr.max() - L_arr.min()) / L_arr.mean() * 100
    print(f"\n  L(f) variation: {variation:.2f}%")

    # Plot
    fig = plot_inductor_characterization(
        cr, wheeler_L=L_wheeler,
        title=rf'Free-Space Spiral: {N_TURNS} turns, '
              rf'$D_{{\mathrm{{out}}}} = {D_OUT*1e3:.0f}$ mm',
        show=False,
    )
    fig.savefig('images/peec_free_space.png', dpi=150, bbox_inches='tight')
    print("  Saved: images/peec_free_space.png")
    return cr


def demo_on_chip():
    """On-chip spiral: SRF, Q peak, realistic frequency response."""
    print("\n" + "=" * 60)
    print("Case 2: On-Chip Spiral Inductor (SiO2/Si substrate)")
    print("=" * 60)

    # Smaller inductor typical of CMOS RF designs
    N_TURNS = 1.5
    D_OUT = 500e-6      # 500 um
    W_TRACE = 20e-6     # 20 um
    S_SPACE = 20e-6     # 20 um
    THICKNESS = 3e-6    # 3 um top metal
    H_OX = 10e-6        # 10 um oxide to substrate

    copper = ConductorProperties(sigma=5.8e7, thickness=THICKNESS, name='Cu')

    trace = Trace.rectangular_spiral(
        n_turns=N_TURNS, d_out=D_OUT, w_trace=W_TRACE, s_space=S_SPACE,
        thickness=THICKNESS, conductor=copper,
    )

    d_in = max(D_OUT - 2 * N_TURNS * (W_TRACE + S_SPACE), W_TRACE)
    L_wheeler = wheeler_inductance(N_TURNS, D_OUT, d_in)

    port = PEECPort('P1', positive_segment_idx=0)
    network = TraceNetwork([trace], [port])
    extractor = PEECExtractor(
        network,
        oxide_thickness=H_OX,
        oxide_eps_r=3.9,
    )

    print(f"  Geometry: {N_TURNS} turns, D_out={D_OUT*1e6:.0f} um, "
          f"w={W_TRACE*1e6:.0f} um, s={S_SPACE*1e6:.0f} um")
    print(f"  Segments: {trace.num_segments}")
    print(f"  Oxide:    {H_OX*1e6:.0f} um SiO2 (eps_r=3.9)")
    print(f"  Wheeler L: {L_wheeler*1e9:.3f} nH")
    print(f"  PEEC L:    {extractor.total_inductance_estimate*1e9:.3f} nH")
    print(f"  DC R:      {extractor.dc_resistance:.4f} Ohm")

    freqs = np.linspace(0.5e9, 30e9, 60)
    results = extractor.extract(freqs)
    char = InductorCharacterization(results)
    cr = char.characterize()

    print(f"\n{cr.summary()}")

    # Plot
    fig = plot_inductor_characterization(
        cr, wheeler_L=L_wheeler,
        title=rf'On-Chip Spiral: {N_TURNS} turns, '
              rf'$D_{{\mathrm{{out}}}} = {D_OUT*1e6:.0f}\,\mu$m, '
              rf'$h_{{\mathrm{{ox}}}} = {H_OX*1e6:.0f}\,\mu$m',
        show=False,
    )
    fig.savefig('images/peec_on_chip.png', dpi=150, bbox_inches='tight')
    print("  Saved: images/peec_on_chip.png")
    return cr


def demo_convergence():
    """Convergence study: L vs number of segments."""
    print("\n" + "=" * 60)
    print("Convergence Study: L vs Segment Count")
    print("=" * 60)

    copper = ConductorProperties(sigma=5.8e7, thickness=2e-6)
    N_TURNS, D_OUT, W_TRACE, S_SPACE = 2.5, 2e-3, 100e-6, 100e-6
    d_in = max(D_OUT - 2 * N_TURNS * (W_TRACE + S_SPACE), W_TRACE)
    L_w = wheeler_inductance(N_TURNS, D_OUT, d_in)

    print(f"\n  {'sps':>6}  {'segs':>5}  {'L (nH)':>9}  {'time (s)':>9}")
    print("  " + "-" * 40)

    import time
    for sps in [1, 2, 5, 10, None]:
        t0 = time.time()
        trace = Trace.rectangular_spiral(
            n_turns=N_TURNS, d_out=D_OUT, w_trace=W_TRACE, s_space=S_SPACE,
            thickness=2e-6, conductor=copper, segments_per_section=sps,
        )
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)
        r = ext.extract([1e9])[0]
        dt = time.time() - t0
        Z = r.Z_matrix[0, 0]
        L = Z.imag / (2 * np.pi * 1e9) * 1e9
        label = f'{sps}' if sps else 'auto'
        print(f"  {label:>6}  {trace.num_segments:>5}  {L:>9.3f}  {dt:>9.3f}")

    print(f"\n  Wheeler reference: {L_w*1e9:.3f} nH")


def main():
    os.makedirs('images', exist_ok=True)

    try:
        configure_latex_style()
    except Exception:
        pass

    demo_free_space()
    demo_on_chip()
    demo_convergence()

    print("\nDone.")


if __name__ == '__main__':
    main()
