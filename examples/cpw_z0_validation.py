"""
Coplanar Waveguide (CPW) Z0 Validation.

Three-conductor CPW on FR4 substrate: center strip + two coplanar ground
planes, with a PEC backing ground.  Extracts Z0 from 2-port S-parameters
and compares against the conformal mapping formula (Simons).

Physical setup:
  - FR4 substrate: eps_r=4.4, h=1.6mm
  - PEC ground at z=0 (implicit in layered Green's function)
  - Center strip: width W=1.0mm, length L=30mm, at z=h
  - Ground planes: width W_gnd=5mm each side, gap S=0.2mm

LayerStack (bottom to top):
  PEC half-space (is_pec=True)
  FR4  (z=0 to z=h, eps_r=4.4)
  Air half-space
  Strip placed at FR4/air interface (z=h)

Mesh:
  Three separate plates (center + left ground + right ground) combined
  into a single mesh via combine_meshes().  2-port extraction on the
  center strip only.

Validation:
  Z0 vs conformal mapping (Simons) — expect <15% error

Produces:
  images/cpw_z0_validation.png

Usage:
    source venv/bin/activate
    python examples/cpw_z0_validation.py
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
    Port, NetworkExtractor, combine_meshes,
    Layer, LayerStack,
    configure_latex_style, c0,
    cpw_z0_conformal, extract_z0_from_s,
)
from pyMoM3d.mom.excitation import StripDeltaGapExcitation, find_feed_edges, compute_feed_signs

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R    = 4.4           # FR4 relative permittivity
H_SUB    = 1.6e-3        # Substrate height (m)
W_CENTER = 1.0e-3        # Center strip width (m)
S_GAP    = 0.2e-3        # Gap between center strip and ground (m)
W_GND    = 5.0e-3        # Ground plane width (m), each side
L_LINE   = 30.0e-3       # Line length (m)

# Mesh
TEL = 0.5e-3             # Finer mesh for CPW gaps

# Frequency sweep
FREQS_GHZ = np.linspace(1.0, 10.0, 8)
FREQS     = FREQS_GHZ * 1e9


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
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("pyMoM3d — CPW Z0 Validation (Conformal Mapping)")
    print("=" * 65)

    # --- Analytical reference ---
    Z0_ana, eps_eff_ana = cpw_z0_conformal(W_CENTER, S_GAP, EPS_R, H_SUB)
    print(f"\nConformal Mapping (Simons):")
    print(f"  Z0      = {Z0_ana:.2f} Ohm")
    print(f"  eps_eff = {eps_eff_ana:.3f}")

    # --- Layer stack ---
    stack = build_layer_stack()
    print(f"\nLayer stack:")
    for lyr in stack.layers:
        print(f"  {lyr.name:15s}  z=[{lyr.z_bot:+.4e}, {lyr.z_top:+.4e}]  "
              f"eps_r={lyr.eps_r}  pec={lyr.is_pec}")

    # --- Mesh construction ---
    z_mesh = H_SUB  # Strip at FR4/air interface
    mesher = GmshMesher(target_edge_length=TEL)

    # Port locations on center strip
    margin = 2.0e-3
    port1_x = -L_LINE / 2.0 + margin
    port2_x = +L_LINE / 2.0 - margin
    L_port = port2_x - port1_x

    # Center strip with two feed lines
    center_mesh = mesher.mesh_plate_with_feeds(
        width=L_LINE, height=W_CENTER,
        feed_x_list=[port1_x, port2_x],
        center=(0.0, 0.0, z_mesh),
    )

    # Left ground plane (negative y)
    y_left = -(W_CENTER / 2.0 + S_GAP + W_GND / 2.0)
    left_gnd = mesher.mesh_plate(
        width=L_LINE, height=W_GND,
        center=(0.0, y_left, z_mesh),
    )

    # Right ground plane (positive y)
    y_right = +(W_CENTER / 2.0 + S_GAP + W_GND / 2.0)
    right_gnd = mesher.mesh_plate(
        width=L_LINE, height=W_GND,
        center=(0.0, y_right, z_mesh),
    )

    # Combine into single mesh
    combined_mesh, offsets = combine_meshes([center_mesh, left_gnd, right_gnd])
    basis = compute_rwg_connectivity(combined_mesh)

    stats = combined_mesh.get_statistics()
    print(f"\nMesh: {stats['num_triangles']} triangles, "
          f"{basis.num_basis} RWG, "
          f"mean edge {stats['mean_edge_length']*1e3:.3f} mm")
    print(f"  Center strip: {len(center_mesh.triangles)} tris")
    print(f"  Left ground:  {len(left_gnd.triangles)} tris")
    print(f"  Right ground: {len(right_gnd.triangles)} tris")

    # --- Port definition (center strip only, filtered by y-range) ---
    y_min = -W_CENTER / 2.0 - TEL
    y_max = +W_CENTER / 2.0 + TEL
    feed1 = find_feed_edges(combined_mesh, basis, feed_x=port1_x,
                            y_range=(y_min, y_max))
    feed2 = find_feed_edges(combined_mesh, basis, feed_x=port2_x,
                            y_range=(y_min, y_max))

    print(f"\n  Port 1 at x = {port1_x*1e3:.1f} mm ({len(feed1)} edges)")
    print(f"  Port 2 at x = {port2_x*1e3:.1f} mm ({len(feed2)} edges)")
    print(f"  Port separation = {L_port*1e3:.1f} mm")

    if not feed1 or not feed2:
        print("ERROR: Could not find feed edges at port locations.")
        return

    signs1 = compute_feed_signs(combined_mesh, basis, feed1)
    signs2 = compute_feed_signs(combined_mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    # --- Build Simulation ---
    exc_dummy = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=FREQS[0],
        excitation=exc_dummy,
        quad_order=4,
        backend='auto',
        layer_stack=stack,
        source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=combined_mesh, reporter=SilentReporter())

    # --- Frequency sweep ---
    extractor = NetworkExtractor(sim, [port1, port2])
    print(f"\nSweeping {len(FREQS)} frequencies "
          f"({FREQS[0]/1e9:.1f}-{FREQS[-1]/1e9:.1f} GHz)...")

    results = extractor.extract(FREQS.tolist())

    # --- Extract Z0 ---
    Z0_mom = []

    print(f"\n  {'f (GHz)':>8}  {'Z0 (Ohm)':>10}  {'Z0_err (%)':>10}  "
          f"{'|S21| (dB)':>10}")
    print("  " + "-" * 45)

    for freq, result in zip(FREQS, results):
        S = result.S_matrix
        z0_ext = extract_z0_from_s(S, Z0_ref=50.0)
        s21_dB = 20.0 * np.log10(max(abs(S[1, 0]), 1e-12))

        Z0_mom.append(abs(z0_ext))
        z0_err = abs(abs(z0_ext) - Z0_ana) / Z0_ana * 100

        print(f"  {freq/1e9:>8.2f}  {abs(z0_ext):>10.2f}  {z0_err:>10.1f}  "
              f"{s21_dB:>10.2f}")

    Z0_mom = np.array(Z0_mom)

    # --- Summary ---
    mean_z0_err = np.mean(np.abs(Z0_mom - Z0_ana) / Z0_ana * 100)
    z0_std = np.std(Z0_mom)
    print(f"\n  Mean Z0 error:  {mean_z0_err:.1f}%  "
          f"{'PASS' if mean_z0_err < 15 else 'CHECK'}")
    print(f"  Z0 std dev:     {z0_std:.2f} Ohm  "
          f"(freq independence: {'PASS' if z0_std < 10 else 'CHECK'})")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(
        rf'CPW $Z_0$ — FR4 ($\varepsilon_r = {EPS_R}$, '
        rf'$h = {H_SUB*1e3:.1f}$ mm, $W = {W_CENTER*1e3:.1f}$ mm, '
        rf'$S = {S_GAP*1e3:.1f}$ mm)',
        fontsize=12,
    )

    ax.plot(FREQS_GHZ, Z0_mom, 'bo-', ms=6, lw=1.5, label=r'MoM (Strata)')
    ax.axhline(Z0_ana, color='r', ls='--', lw=1.5,
               label=rf'Conformal mapping ($Z_0 = {Z0_ana:.1f}\,\Omega$)')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$Z_0$ ($\Omega$)')
    ax.set_title(r'Characteristic Impedance vs Frequency')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(100, Z0_ana * 2)])

    out = os.path.join(IMAGES_DIR, 'cpw_z0_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
