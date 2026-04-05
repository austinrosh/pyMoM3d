"""
Microstrip Z0 Extraction with Probe Feed.

Uses a surface probe feed model (Port.from_vertex) to extract microstrip
characteristic impedance.  The probe excites RWG basis functions radially
from a vertex point, modeling a coaxial probe attachment.  The PEC ground
plane is handled by the layered Green's function.

Physical setup:
  - FR4 substrate: eps_r=4.4, height H=1.6mm
  - PEC ground at z=0 (implicit in layered GF)
  - Strip at z=H, width W≈3mm (≈50 Ω), length L=30mm
  - Probe feeds at each end of the strip

Known limitations:
  Surface probe feeds couple weakly to the guided quasi-TEM mode because
  the surface EFIE with a single layered Green's function does not
  correctly separate vector and scalar potential contributions at the
  dielectric interface.  The Z0 extraction accuracy varies with frequency.
  See microstrip_z0_2port_validation.py for further discussion.

Produces:
  images/microstrip_z0_probe.png

Usage:
    source venv/bin/activate
    python examples/microstrip_z0_probe_validation.py
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
    configure_latex_style, c0, eta0,
    microstrip_z0_hammerstad, extract_z0_from_s,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R     = 4.4          # FR4
H_SUB     = 1.6e-3       # Substrate height (m)
W_STRIP   = 3.06e-3      # Strip width (m) — ≈50 Ω on FR4
L_STRIP   = 30.0e-3      # Strip length (m)

# Mesh
TEL = 2.0e-3             # Target edge length (m)

# Frequency sweep
FREQS_GHZ = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
FREQS     = FREQS_GHZ * 1e9


# ---------------------------------------------------------------------------
# LayerStack
# ---------------------------------------------------------------------------

def build_layer_stack():
    """Microstrip: PEC ground -> FR4 substrate -> air."""
    return LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("pyMoM3d — Microstrip Z0 with Probe Feed")
    print("=" * 65)

    # Analytical reference
    Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W_STRIP, H_SUB, EPS_R)
    print(f"\nHammerstad-Jensen: Z0 = {Z0_ref:.2f} Ω, eps_eff = {eps_eff_ref:.3f}")

    # --- Build mesh ---
    stack = build_layer_stack()
    z_mesh = H_SUB

    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[-L_STRIP / 2.0, L_STRIP / 2.0],
        center=(0.0, 0.0, z_mesh),
    )
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    N = basis.num_basis
    print(f"\nMesh: {stats['num_triangles']} tris, {N} RWG basis functions")
    print(f"  Strip: {L_STRIP*1e3:.1f} mm × {W_STRIP*1e3:.2f} mm at z = {z_mesh*1e3:.1f} mm")

    # --- Create probe feed ports ---
    # Probe at the center of each short edge (x = ±L/2, y = 0)
    port1 = Port.from_vertex(
        mesh, basis,
        vertex_pos=np.array([-L_STRIP / 2.0, 0.0, z_mesh]),
        name='P1',
    )
    port2 = Port.from_vertex(
        mesh, basis,
        vertex_pos=np.array([L_STRIP / 2.0, 0.0, z_mesh]),
        name='P2',
    )

    print(f"  Port 1: {len(port1.feed_basis_indices)} edges at x = {-L_STRIP/2*1e3:.1f} mm")
    print(f"  Port 2: {len(port2.feed_basis_indices)} edges at x = {L_STRIP/2*1e3:.1f} mm")

    # --- Simulation config ---
    config = SimulationConfig(
        frequency=FREQS[0],
        excitation=StripDeltaGapExcitation([], voltage=1.0),  # placeholder
        quad_order=4,
        layer_stack=stack,
        source_layer_name='FR4',
        gf_backend='strata',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # --- Extraction ---
    extractor = NetworkExtractor(
        sim,
        [port1, port2],
        Z0=50.0,
        store_currents=True,
    )

    print(f"\n{'f (GHz)':>8}  {'Z_in (Ω)':>12}  {'Z0 (Ω)':>10}  {'Z0_err':>8}  "
          f"{'|S21| dB':>10}  {'|S11| dB':>10}")
    print("  " + "-" * 70)

    Z0_arr = []
    S21_dB_arr = []
    S11_dB_arr = []
    Z_in_arr = []

    for freq in FREQS:
        results = extractor.extract([freq])
        r = results[0]

        Z_in = r.Z_matrix[0, 0]
        S = r.S_matrix
        S11_dB = 20 * np.log10(max(abs(S[0, 0]), 1e-30))
        S21_dB = 20 * np.log10(max(abs(S[1, 0]), 1e-30))

        # Extract Z0 from S-parameters
        try:
            Z0_ext = abs(extract_z0_from_s(S, Z0_ref=50.0))
        except Exception:
            Z0_ext = float('nan')

        Z0_arr.append(Z0_ext)
        S21_dB_arr.append(S21_dB)
        S11_dB_arr.append(S11_dB)
        Z_in_arr.append(Z_in)

        err = (Z0_ext - Z0_ref) / Z0_ref * 100 if Z0_ref > 0 else float('nan')
        print(f"  {freq/1e9:>6.2f}  {abs(Z_in):>12.2f}  {Z0_ext:>10.2f}  "
              f"{err:>+7.1f}%  {S21_dB:>10.2f}  {S11_dB:>10.2f}")

    Z0_arr = np.array(Z0_arr)
    S21_dB_arr = np.array(S21_dB_arr)
    S11_dB_arr = np.array(S11_dB_arr)

    # Filter valid values
    valid = np.isfinite(Z0_arr) & (Z0_arr > 0)
    if np.any(valid):
        mean_Z0 = np.mean(Z0_arr[valid])
        mean_err = abs(mean_Z0 - Z0_ref) / Z0_ref * 100
        print(f"\n  Mean Z0: {mean_Z0:.2f} Ω")
        print(f"  Hammerstad Z0: {Z0_ref:.2f} Ω")
        print(f"  Mean error: {mean_err:.1f}%  "
              f"{'PASS' if mean_err < 15 else 'CHECK'}")
    else:
        mean_Z0 = float('nan')
        mean_err = float('nan')
        print("\n  No valid Z0 values extracted")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        rf'Microstrip Z0 (Probe Feed) — $\varepsilon_r = {EPS_R}$, '
        rf'$h = {H_SUB*1e3:.1f}$ mm, $W = {W_STRIP*1e3:.2f}$ mm',
        fontsize=12,
    )

    ax1.plot(FREQS_GHZ, Z0_arr, 'bo-', ms=6, lw=1.5, label='MoM (probe)')
    ax1.axhline(Z0_ref, color='r', ls='--', lw=1.5,
                label=rf'Hammerstad ($Z_0 = {Z0_ref:.1f}\,\Omega$)')
    ax1.set_xlabel(r'Frequency $f$ (GHz)')
    ax1.set_ylabel(r'$Z_0$ ($\Omega$)')
    ax1.set_title(r'Characteristic Impedance')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(FREQS_GHZ, S21_dB_arr, 'go-', ms=6, lw=1.5, label=r'$|S_{21}|$')
    ax2.plot(FREQS_GHZ, S11_dB_arr, 'rs-', ms=6, lw=1.5, label=r'$|S_{11}|$')
    ax2.set_xlabel(r'Frequency $f$ (GHz)')
    ax2.set_ylabel(r'Magnitude (dB)')
    ax2.set_title(r'S-Parameters')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'microstrip_z0_probe.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
