"""
Strip Delta-Gap + SOC De-embedding — Wideband Validation (2-20 GHz).

Determines whether SOC de-embedding produces physically meaningful results
across a wide frequency range when applied to strip delta-gap ports on a
layered microstrip structure.

Key questions:
  1. Where does the raw strip delta-gap produce usable results (S21 > -3 dB)?
  2. Does SOC improve, degrade, or produce non-physical results?
  3. At what frequencies does SOC error-box extraction become ill-conditioned?
  4. Are there passivity violations or other non-physical artifacts?

Physical setup:
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - PEC ground at z=0
  - Strip: W=3.06mm (~50 Ohm), L=40mm
  - Feed sections: 15mm per side (electrically significant at high freq)
  - DUT = 6mm between reference planes

The 40mm structure is used (not 10mm) because:
  - The delta-gap port stays better-matched over a wider band on longer strips
  - SOC is already validated at 2-10 GHz on this structure (S21 = -0.09 dB at 10 GHz)
  - We extend to 20 GHz to see wideband behavior

NOTE: This is purely a full-wave experiment. QS-Probe is not compared here
because it is invalid on a 40mm structure above ~300 MHz (kD >> 0.5).
The QS/FW crossover must be characterized on a shorter structure separately.

Usage:
    source venv/bin/activate
    python examples/strip_delta_gap_soc_gap_validation.py
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
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.network.soc_deembedding import SOCDeembedding

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R    = 4.4          # FR4 relative permittivity
H_SUB    = 1.6e-3       # Substrate height (m)
W_STRIP  = 3.06e-3      # Strip width (m) — approx 50 Ohm on FR4 1.6mm
L_STRIP  = 40.0e-3      # Total strip length (m)

TEL = 0.7e-3            # Mesh target edge length

# Port and reference plane locations (same as validated SOC setup)
FEED_LEN  = 15.0e-3     # Feed section per side
PORT1_X   = -L_STRIP / 2.0 + 2.0e-3   # 2mm inset from strip end
PORT2_X   = +L_STRIP / 2.0 - 2.0e-3
REF1_X    = PORT1_X + FEED_LEN         # Reference plane 1
REF2_X    = PORT2_X - FEED_LEN         # Reference plane 2

# Effective wavelength info for diagnostics
EPS_EFF_APPROX = 3.3  # approximate for this geometry

# Frequency sweep — 2-20 GHz, dense enough to see structure
FREQS_GHZ = np.array([
    2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
])
FREQS = FREQS_GHZ * 1e9


# ---------------------------------------------------------------------------
# LayerStack
# ---------------------------------------------------------------------------

def build_layer_stack():
    return LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4',        z_bot=0.0,     z_top=H_SUB, eps_r=EPS_R),
        Layer('air',        z_bot=H_SUB,   z_top=np.inf, eps_r=1.0),
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_db(results, i, j):
    return np.array([
        20.0 * np.log10(max(abs(r.S_matrix[i, j]), 1e-15)) for r in results
    ])


def print_s_table(label, results, freqs):
    print(f"\n--- {label} ---")
    print(f"  {'f (GHz)':>8}  {'|S11| dB':>9}  {'|S21| dB':>9}  "
          f"{'|S21-S12|':>9}  {'|S11|²+|S21|²':>14}  {'passive':>7}")
    print("  " + "-" * 65)
    for freq, result in zip(freqs, results):
        S = result.S_matrix
        s11 = abs(S[0, 0])
        s21 = abs(S[1, 0])
        power_sum = s11**2 + s21**2
        passive = power_sum <= 1.0 + 1e-6
        print(f"  {freq/1e9:>8.1f}  "
              f"{20*np.log10(max(s11, 1e-15)):>9.2f}  "
              f"{20*np.log10(max(s21, 1e-15)):>9.2f}  "
              f"{abs(S[1,0] - S[0,1]):>9.2e}  "
              f"{power_sum:>14.6f}  "
              f"{'OK' if passive else 'FAIL':>7}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Strip Delta-Gap + SOC — Wideband Validation (2-20 GHz)")
    print("=" * 70)

    dut_len = REF2_X - REF1_X
    feed_len = REF1_X - PORT1_X
    print(f"\nGeometry:")
    print(f"  Strip: W={W_STRIP*1e3:.2f} mm, L={L_STRIP*1e3:.1f} mm")
    print(f"  Substrate: eps_r={EPS_R}, h={H_SUB*1e3:.1f} mm")
    print(f"  Port 1 at x = {PORT1_X*1e3:.1f} mm")
    print(f"  Port 2 at x = {PORT2_X*1e3:.1f} mm")
    print(f"  Ref plane 1 at x = {REF1_X*1e3:.1f} mm")
    print(f"  Ref plane 2 at x = {REF2_X*1e3:.1f} mm")
    print(f"  DUT length = {dut_len*1e3:.1f} mm")
    print(f"  Feed length = {feed_len*1e3:.1f} mm per side")

    # Electrical size table
    print(f"\n  Electrical sizes (eps_eff �� {EPS_EFF_APPROX}):")
    print(f"  {'f (GHz)':>8}  {'λ_eff (mm)':>10}  {'feed/λ':>8}  {'DUT/λ':>8}  {'strip/λ':>9}")
    print("  " + "-" * 50)
    for f_ghz in [2, 5, 10, 15, 20]:
        lam_eff = c0 / (f_ghz * 1e9 * np.sqrt(EPS_EFF_APPROX))
        print(f"  {f_ghz:>8}  {lam_eff*1e3:>8.1f}    {feed_len/lam_eff:>8.3f}  "
              f"{dut_len/lam_eff:>8.3f}  {L_STRIP/lam_eff:>9.3f}")

    # Quarter-wave singularity frequencies for the feed section
    f_qw = c0 / (4 * feed_len * np.sqrt(EPS_EFF_APPROX))
    print(f"\n  Feed section quarter-wave singularities:")
    for n in range(1, 5):
        fn = (2*n - 1) * f_qw
        if fn <= 25e9:
            print(f"    n={n}: {fn/1e9:.2f} GHz (feed = {2*n-1}λ/4)")

    stack = build_layer_stack()
    z_mesh = H_SUB

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, REF1_X, REF2_X, PORT2_X],
        center=(0.0, 0.0, z_mesh),
    )
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    mel = stats['mean_edge_length']
    print(f"\nMesh: {stats['num_triangles']} triangles, "
          f"{basis.num_basis} RWG, "
          f"mean edge {mel*1e3:.2f} mm")

    # Check mesh density at highest frequency
    lam_min = c0 / (FREQS[-1] * np.sqrt(EPS_EFF_APPROX))
    elements_per_lambda = lam_min / mel
    print(f"  At {FREQS[-1]/1e9:.0f} GHz: {elements_per_lambda:.1f} elements/λ_eff "
          f"({'OK' if elements_per_lambda >= 8 else 'WARNING: < 10/λ'})")

    # ------------------------------------------------------------------
    # Port definition
    # ------------------------------------------------------------------
    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)
    print(f"  Port 1: {len(feed1)} edges, Port 2: {len(feed2)} edges")

    if not feed1 or not feed2:
        print("ERROR: Could not find feed edges.")
        return

    # ------------------------------------------------------------------
    # Build Simulation
    # ------------------------------------------------------------------
    exc_dummy = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=FREQS[0],
        excitation=exc_dummy,
        quad_order=4,
        backend='auto',
        layer_stack=stack,
        source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # ==================================================================
    # Dataset 1: FW-Raw
    # ==================================================================
    print("\n" + "=" * 70)
    print("Dataset 1: FW-Raw (strip delta-gap, uncalibrated)")
    print("=" * 70)

    extractor = NetworkExtractor(
        sim, [port1, port2], store_currents=False,
    )
    results_raw = extractor.extract(FREQS.tolist())
    print_s_table("FW-Raw", results_raw, FREQS)

    # ==================================================================
    # Dataset 2: FW-SOC
    # ==================================================================
    print("\n" + "=" * 70)
    print("Dataset 2: FW-SOC (strip delta-gap + SOC de-embedding)")
    print("=" * 70)

    soc = SOCDeembedding(
        sim, [port1, port2],
        reference_plane_x=[REF1_X, REF2_X],
        port_x=[PORT1_X, PORT2_X],
        symmetric=False,
    )
    results_soc = []
    for result in results_raw:
        freq = result.frequency
        try:
            cal = soc.deembed(result)
            results_soc.append(cal)
        except Exception as e:
            print(f"  WARNING at {freq/1e9:.1f} GHz: SOC failed — {e}")
            results_soc.append(result)
    print_s_table("FW-SOC", results_soc, FREQS)

    # ==================================================================
    # SOC error-box diagnostics
    # ==================================================================
    print("\n" + "=" * 70)
    print("SOC Error-Box Diagnostics (Port 1)")
    print("=" * 70)
    print(f"  {'f (GHz)':>8}  {'Re(A)':>8}  {'Im(A)':>8}  {'|A|':>6}  "
          f"{'|B| (Ω)':>10}  {'|C| (S)':>10}  {'|det-1|':>9}  "
          f"{'feed/λ':>8}")
    print("  " + "-" * 78)
    for freq in FREQS:
        lam_eff = c0 / (freq * np.sqrt(EPS_EFF_APPROX))
        try:
            T = soc.compute_error_abcd(0, freq)
            A, B, C, D = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
            det_err = abs(A * D - B * C - 1.0)
            print(f"  {freq/1e9:>8.1f}  {A.real:>8.4f}  {A.imag:>8.4f}  "
                  f"{abs(A):>6.3f}  {abs(B):>10.2f}  {abs(C):>10.4e}  "
                  f"{det_err:>9.2e}  {feed_len/lam_eff:>8.3f}")
        except Exception as e:
            print(f"  {freq/1e9:>8.1f}  FAILED: {e}")

    # ==================================================================
    # Comparison table
    # ==================================================================
    print("\n" + "=" * 70)
    print("Comparison: Raw vs SOC")
    print("=" * 70)

    s21_raw = extract_db(results_raw, 1, 0)
    s21_soc = extract_db(results_soc, 1, 0)
    s11_raw = extract_db(results_raw, 0, 0)
    s11_soc = extract_db(results_soc, 0, 0)

    print(f"\n  {'f (GHz)':>8}  {'Raw S21':>9}  {'SOC S21':>9}  "
          f"{'Δ S21':>8}  {'Raw S11':>9}  {'SOC S11':>9}  "
          f"{'SOC pass':>8}")
    print("  " + "-" * 65)
    for i, freq in enumerate(FREQS):
        delta = s21_soc[i] - s21_raw[i]
        S = results_soc[i].S_matrix
        power = abs(S[0,0])**2 + abs(S[1,0])**2
        passive = power <= 1.0 + 1e-6
        print(f"  {freq/1e9:>8.1f}  {s21_raw[i]:>9.2f}  {s21_soc[i]:>9.2f}  "
              f"{delta:>+7.1f}  {s11_raw[i]:>9.2f}  {s11_soc[i]:>9.2f}  "
              f"{'OK' if passive else 'FAIL':>8}")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # Where is raw usable?
    raw_usable = s21_raw > -3.0
    if np.any(raw_usable):
        raw_min_f = FREQS_GHZ[raw_usable][0]
        print(f"\n  Raw delta-gap usable (S21 > -3 dB): {raw_min_f:.0f}+ GHz")
    else:
        print(f"\n  Raw delta-gap: never reaches S21 > -3 dB in this range")

    # Where does SOC improve?
    soc_better = s21_soc > s21_raw + 1.0  # SOC improves by > 1 dB
    soc_worse = s21_soc < s21_raw - 1.0   # SOC degrades by > 1 dB
    if np.any(soc_better):
        print(f"  SOC improves (>1 dB): "
              f"{FREQS_GHZ[soc_better][0]:.0f}-{FREQS_GHZ[soc_better][-1]:.0f} GHz")
    if np.any(soc_worse):
        print(f"  SOC DEGRADES (>1 dB): "
              f"{FREQS_GHZ[soc_worse][0]:.0f}-{FREQS_GHZ[soc_worse][-1]:.0f} GHz")

    # Passivity
    n_fail = sum(
        1 for r in results_soc
        if abs(r.S_matrix[0,0])**2 + abs(r.S_matrix[1,0])**2 > 1.0 + 1e-6
    )
    print(f"  Passivity violations: {n_fail}/{len(results_soc)} frequencies")

    # Best SOC operating band
    soc_usable = s21_soc > -1.0
    if np.any(soc_usable):
        soc_good_freqs = FREQS_GHZ[soc_usable]
        print(f"  SOC S21 > -1 dB: {soc_good_freqs[0]:.0f}-{soc_good_freqs[-1]:.0f} GHz")

    # ==================================================================
    # Plot: 3 panels
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        r'Strip Delta-Gap + SOC — Wideband Validation'
        '\n'
        rf'FR4 ($\varepsilon_r = {EPS_R}$, $h = {H_SUB*1e3:.1f}$ mm), '
        rf'$W = {W_STRIP*1e3:.2f}$ mm, $L = {L_STRIP*1e3:.0f}$ mm, '
        rf'feed = {feed_len*1e3:.0f} mm',
        fontsize=11,
    )

    # Panel 1: S21
    ax = axes[0]
    ax.plot(FREQS_GHZ, s21_raw, 'r^--', ms=5, lw=1.5, label=r'Raw $|S_{21}|$')
    ax.plot(FREQS_GHZ, s21_soc, 'bo-', ms=5, lw=1.5, label=r'SOC $|S_{21}|$')
    ax.axhline(0.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(-1.0, color='green', ls='--', lw=0.8, alpha=0.5, label=r'$-1$ dB')
    ax.axhline(-3.0, color='gray', ls='--', lw=0.8, alpha=0.5, label=r'$-3$ dB')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{21}|$ (dB)')
    ax.set_title(r'Insertion Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(min(s21_raw) - 3, -50))

    # Panel 2: S11
    ax = axes[1]
    ax.plot(FREQS_GHZ, s11_raw, 'r^--', ms=5, lw=1.5, label=r'Raw $|S_{11}|$')
    ax.plot(FREQS_GHZ, s11_soc, 'bo-', ms=5, lw=1.5, label=r'SOC $|S_{11}|$')
    ax.axhline(-10.0, color='gray', ls='--', lw=0.8, alpha=0.5, label=r'$-10$ dB')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: SOC improvement
    ax = axes[2]
    improvement = s21_soc - s21_raw
    colors = ['steelblue' if d >= 0 else 'coral' for d in improvement]
    ax.bar(FREQS_GHZ, improvement, width=0.7, color=colors, alpha=0.8)
    ax.axhline(0.0, color='k', ls='-', lw=0.8)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$\Delta|S_{21}|$ (dB)')
    ax.set_title(r'SOC Improvement (blue) / Degradation (red)')
    ax.grid(True, alpha=0.3, axis='y')

    out = os.path.join(IMAGES_DIR, 'soc_gap_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
