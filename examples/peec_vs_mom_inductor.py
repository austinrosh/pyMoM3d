"""
PEEC vs MoM Inductor Cross-Validation.

Compares inductance extraction from PEEC (partial element equivalent circuit)
and MoM EFIE (surface integral equation) for the same spiral inductor geometry.

PEEC constrains current along filaments, giving frequency-independent L.
MoM EFIE uses 2D surface currents, which suffer from low-frequency breakdown
(scalar potential dominance) but provide physically correct current distribution.

This example validates both solvers against each other and the Wheeler formula.

Produces:
  images/peec_vs_mom_inductor.png
  images/peec_inductor_current.png

Usage:
    source venv/bin/activate
    python examples/peec_vs_mom_inductor.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyMoM3d import (
    configure_latex_style, c0,
    wheeler_inductance, inductance_from_z, quality_factor,
    plot_peec_currents,
)
from pyMoM3d.peec.trace import Trace, TraceNetwork, PEECPort
from pyMoM3d.peec.extractor import PEECExtractor
from pyMoM3d.mom.surface_impedance import ConductorProperties

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

N_TURNS  = 2.5
W_TRACE  = 100e-6      # Trace width (m)
S_SPACE  = 100e-6      # Spacing (m)
D_OUT    = 2e-3         # Outer dimension (m)
T_COND   = 2e-6         # Conductor thickness (m)
SIGMA    = 5.8e7        # Copper conductivity (S/m)

FREQS_GHZ = np.linspace(1.0, 10.0, 10)
FREQS     = FREQS_GHZ * 1e9


def main():
    print("=" * 65)
    print("pyMoM3d — PEEC vs MoM Inductor Cross-Validation")
    print("=" * 65)

    # --- Analytical reference ---
    d_in = D_OUT - 2 * N_TURNS * (W_TRACE + S_SPACE)
    d_in = max(d_in, W_TRACE)
    L_wheeler = wheeler_inductance(N_TURNS, D_OUT, d_in)
    print(f"\nAnalytical (Wheeler): L = {L_wheeler*1e9:.3f} nH")

    # --- PEEC extraction ---
    print("\n--- PEEC Extraction ---")
    conductor = ConductorProperties(sigma=SIGMA, thickness=T_COND)
    spiral = Trace.rectangular_spiral(
        n_turns=N_TURNS, d_out=D_OUT, w_trace=W_TRACE, s_space=S_SPACE,
        thickness=T_COND, conductor=conductor,
    )
    port = PEECPort('P1', positive_segment_idx=0)
    network = TraceNetwork([spiral], [port])
    ext = PEECExtractor(network)

    print(f"  Segments: {network.num_segments}")
    print(f"  Total L estimate: {ext.total_inductance_estimate*1e9:.3f} nH")
    print(f"  DC resistance: {ext.dc_resistance*1e3:.3f} mOhm")

    results_peec = ext.extract(FREQS.tolist(), store_currents=True)

    Z_peec = np.array([r.Z_matrix[0, 0] for r in results_peec])
    L_peec = np.array([inductance_from_z(z, f) for z, f in zip(Z_peec, FREQS)])
    Q_peec = np.array([quality_factor(z) for z in Z_peec])

    # --- Print table ---
    print(f"\n  {'f (GHz)':>8}  {'L_PEEC (nH)':>12}  {'Q_PEEC':>8}  "
          f"{'R_PEEC':>10}  {'L_err':>8}")
    print("  " + "-" * 55)

    for i, freq in enumerate(FREQS):
        L_err = abs(L_peec[i] - L_wheeler) / L_wheeler * 100
        print(f"  {freq/1e9:>8.2f}  {L_peec[i]*1e9:>12.3f}  {Q_peec[i]:>8.2f}  "
              f"{Z_peec[i].real:>10.4f}  {L_err:>7.1f}%")

    # --- Summary ---
    L_pos = L_peec[L_peec > 0]
    if len(L_pos) > 1:
        ratio = np.max(L_pos) / np.min(L_pos)
    else:
        ratio = float('inf')

    mean_L = np.mean(L_pos) if len(L_pos) > 0 else float('nan')
    mean_err = abs(mean_L - L_wheeler) / L_wheeler * 100

    print(f"\n  PEEC Results:")
    print(f"    Mean L:        {mean_L*1e9:.3f} nH")
    print(f"    Wheeler L:     {L_wheeler*1e9:.3f} nH")
    print(f"    Error vs Wheeler: {mean_err:.1f}%")
    print(f"    L(f) variation: {ratio:.3f}x  "
          f"{'PASS' if ratio < 1.05 else 'CHECK'}")

    # --- PEEC current visualization ---
    mid_result = results_peec[len(FREQS) // 2]
    mid_freq = FREQS[len(FREQS) // 2]

    if mid_result.I_solutions is not None:
        I_seg = mid_result.I_solutions[:, 0]
        segs = network.all_segments

        fig_c, ax_c = plot_peec_currents(
            segs, I_seg, normalize=True, cmap='viridis',
            title=rf'PEEC Segment Currents at $f = {mid_freq/1e9:.1f}$ GHz',
        )
        out_c = os.path.join(IMAGES_DIR, 'peec_inductor_current.png')
        fig_c.savefig(out_c, dpi=150, bbox_inches='tight')
        plt.close(fig_c)
        print(f"\n  Saved -> {out_c}")

    # --- Comparison plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        rf'PEEC Inductor Validation — {N_TURNS} turns, '
        rf'$d_{{\mathrm{{out}}}} = {D_OUT*1e3:.1f}$ mm, '
        rf'$w = {W_TRACE*1e6:.0f}\,\mu$m',
        fontsize=12,
    )

    # L vs frequency
    ax1.plot(FREQS_GHZ, L_peec * 1e9, 'rs-', ms=5, lw=1.5, label='PEEC')
    ax1.axhline(L_wheeler * 1e9, color='k', ls='--', lw=1.5,
                label=rf'Wheeler ($L = {L_wheeler*1e9:.2f}$ nH)')
    ax1.set_xlabel(r'Frequency $f$ (GHz)')
    ax1.set_ylabel(r'$L$ (nH)')
    ax1.set_title(r'Inductance $L(f)$')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Q vs frequency
    ax2.plot(FREQS_GHZ, Q_peec, 'rs-', ms=5, lw=1.5, label='PEEC')
    ax2.set_xlabel(r'Frequency $f$ (GHz)')
    ax2.set_ylabel(r'$Q$')
    ax2.set_title(r'Quality Factor $Q(f)$')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'peec_vs_mom_inductor.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {out}")


if __name__ == '__main__':
    main()
