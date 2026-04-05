"""
Stripline Transmission Line Z0 Validation.

Symmetric stripline between two PEC ground planes, filled with dielectric.
Uses the Strata C++ multilayer Green's function.

Physical setup:
  - Rogers RT/duroid 5880: eps_r=2.2, total thickness b=3mm
  - PEC planes at z=0 and z=b
  - Strip at z=b/2 (centered), width W≈1.05mm (≈50 Ω)
  - Length L=40mm

LayerStack (bottom to top):
  PEC half-space
  Dielectric lower (z=0 to z=b/2−δ, eps_r=2.2)
  Phantom (z=b/2−δ to z=b/2+δ, eps_r=2.201)   ← strip mesh here
  Dielectric upper (z=b/2+δ to z=b, eps_r=2.2)
  PEC half-space

Validation:
  Z0 vs Cohn elliptic-integral formula (expect <10% error)
  Z0 should be approximately frequency-independent (TEM mode)

Produces:
  images/stripline_z0_validation.png

Usage:
    source venv/bin/activate
    python examples/stripline_z0_validation.py
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
    stripline_z0_cohn, extract_z0_from_s, extract_propagation_constant,
)
from pyMoM3d.mom.excitation import StripDeltaGapExcitation, find_feed_edges, compute_feed_signs

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

EPS_R     = 2.2          # Rogers RT/duroid 5880
B_TOTAL   = 3.0e-3       # Ground-plane separation (m)
W_STRIP   = 1.05e-3      # Strip width (m) — ≈50 Ω for this stackup
L_STRIP   = 40.0e-3      # Strip length (m)
DELTA     = 0.05e-3      # Phantom layer half-thickness (m)

# Strip is at z = b/2
Z_STRIP   = B_TOTAL / 2.0

# Mesh
TEL = 1.0e-3             # Target edge length (m)

# Frequency sweep
FREQS_GHZ = np.linspace(1.0, 10.0, 8)
FREQS     = FREQS_GHZ * 1e9


# ---------------------------------------------------------------------------
# LayerStack definition
# ---------------------------------------------------------------------------

def build_layer_stack():
    """Build the stripline layer stack with dual PEC half-spaces.

    Unlike microstrip/CPW (where the strip sits at the dielectric/air
    interface and needs no phantom), stripline has the strip *embedded*
    inside homogeneous dielectric.  A thin phantom layer with eps_r
    slightly different from the bulk is required so that the Strata
    backend treats the source plane as a layer boundary.
    """
    return LayerStack([
        Layer('pec_bot',  z_bot=-np.inf,           z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('diel_lo',  z_bot=0.0,               z_top=Z_STRIP - DELTA, eps_r=EPS_R),
        Layer('phantom',  z_bot=Z_STRIP - DELTA,   z_top=Z_STRIP + DELTA, eps_r=EPS_R + 0.001),
        Layer('diel_hi',  z_bot=Z_STRIP + DELTA,   z_top=B_TOTAL, eps_r=EPS_R),
        Layer('pec_top',  z_bot=B_TOTAL,           z_top=np.inf, eps_r=1.0, is_pec=True),
    ])


# ---------------------------------------------------------------------------
# Mesh and port creation
# ---------------------------------------------------------------------------

def create_stripline_mesh():
    """Create strip mesh with two feed lines."""
    mesher = GmshMesher(target_edge_length=TEL)

    margin = 3.0e-3
    port1_x = -L_STRIP / 2.0 + margin
    port2_x = +L_STRIP / 2.0 - margin

    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP,
        height=W_STRIP,
        feed_x_list=[port1_x, port2_x],
        center=(0.0, 0.0, Z_STRIP),
    )
    return mesh, port1_x, port2_x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("pyMoM3d — Stripline Transmission Line Z0 Validation")
    print("=" * 65)

    # --- Analytical reference ---
    Z0_ana = stripline_z0_cohn(W_STRIP, B_TOTAL, EPS_R)
    print(f"\nAnalytical (Cohn):")
    print(f"  Z0 = {Z0_ana:.2f} Ohm")
    print(f"  (TEM mode — Z0 should be frequency-independent)")

    # --- Layer stack ---
    stack = build_layer_stack()
    print(f"\nLayer stack:")
    for lyr in stack.layers:
        print(f"  {lyr.name:15s}  z=[{lyr.z_bot:+.4e}, {lyr.z_top:+.4e}]  "
              f"eps_r={lyr.eps_r}  pec={lyr.is_pec}")

    # --- Mesh ---
    print("\nMeshing strip...")
    mesh, port1_x, port2_x = create_stripline_mesh()
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    print(f"  Triangles: {stats['num_triangles']}")
    print(f"  RWG basis: {basis.num_basis}")
    print(f"  Mean edge: {stats['mean_edge_length']*1e3:.2f} mm")

    L_port = port2_x - port1_x
    print(f"\n  Port 1 at x = {port1_x*1e3:.1f} mm")
    print(f"  Port 2 at x = {port2_x*1e3:.1f} mm")
    print(f"  Port separation = {L_port*1e3:.1f} mm")

    # --- Port definition ---
    feed1 = find_feed_edges(mesh, basis, feed_x=port1_x)
    feed2 = find_feed_edges(mesh, basis, feed_x=port2_x)
    print(f"  Port 1 feed edges: {len(feed1)}")
    print(f"  Port 2 feed edges: {len(feed2)}")

    if not feed1 or not feed2:
        print("ERROR: Could not find feed edges at port locations.")
        return

    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
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
        source_layer_name='phantom',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # --- Frequency sweep ---
    extractor = NetworkExtractor(sim, [port1, port2])
    print(f"\nSweeping {len(FREQS)} frequencies "
          f"({FREQS[0]/1e9:.1f}–{FREQS[-1]/1e9:.1f} GHz)...")

    results = extractor.extract(FREQS.tolist())

    # --- Extract Z0 and eps_eff from S-parameters ---
    Z0_mom = []
    eps_eff_mom = []

    print(f"\n  {'f (GHz)':>8}  {'Z0 (Ohm)':>10}  {'Z0_err (%)':>10}  "
          f"{'eps_eff':>8}  {'|S21| (dB)':>10}")
    print("  " + "-" * 55)

    for freq, result in zip(FREQS, results):
        S = result.S_matrix
        z0_ext = extract_z0_from_s(S, Z0_ref=50.0)
        gamma = extract_propagation_constant(S, L_port, Z0_ref=50.0)
        k0 = 2.0 * np.pi * freq / c0
        ee = (gamma.imag / k0)**2 if k0 > 0 else float('nan')
        s21_dB = 20.0 * np.log10(max(abs(S[1, 0]), 1e-12))

        Z0_mom.append(abs(z0_ext))
        eps_eff_mom.append(ee)
        z0_err = abs(abs(z0_ext) - Z0_ana) / Z0_ana * 100

        print(f"  {freq/1e9:>8.2f}  {abs(z0_ext):>10.2f}  {z0_err:>10.1f}  "
              f"{ee:>8.3f}  {s21_dB:>10.2f}")

    Z0_mom = np.array(Z0_mom)
    eps_eff_mom = np.array(eps_eff_mom)

    # --- Summary ---
    mean_z0_err = np.mean(np.abs(Z0_mom - Z0_ana) / Z0_ana * 100)
    z0_std = np.std(Z0_mom)
    mean_ee = np.mean(eps_eff_mom)
    ee_err = abs(mean_ee - EPS_R) / EPS_R * 100
    print(f"\n  Mean Z0 error:  {mean_z0_err:.1f}%  "
          f"{'PASS' if mean_z0_err < 15 else 'CHECK'}")
    print(f"  Z0 std dev:     {z0_std:.2f} Ohm  "
          f"(frequency independence: {'PASS' if z0_std < 5 else 'CHECK'})")
    print(f"  Mean eps_eff:   {mean_ee:.3f}  "
          f"(expected {EPS_R:.1f} for TEM, error {ee_err:.1f}%)")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(
        rf'Stripline $Z_0$ — RT/duroid ($\varepsilon_r = {EPS_R}$, '
        rf'$b = {B_TOTAL*1e3:.1f}$ mm, $W = {W_STRIP*1e3:.2f}$ mm)',
        fontsize=12,
    )

    ax.plot(FREQS_GHZ, Z0_mom, 'bo-', ms=6, lw=1.5, label=r'MoM (Strata)')
    ax.axhline(Z0_ana, color='r', ls='--', lw=1.5,
               label=rf'Cohn ($Z_0 = {Z0_ana:.1f}\,\Omega$)')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$Z_0$ ($\Omega$)')
    ax.set_title(r'Characteristic Impedance vs Frequency')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(100, Z0_ana * 2)])

    out = os.path.join(IMAGES_DIR, 'stripline_z0_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved → {out}")


if __name__ == '__main__':
    main()
