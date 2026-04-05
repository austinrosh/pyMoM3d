"""
A-EFIE Spiral Inductor Validation (1-10 GHz).

Compares standard EFIE vs A-EFIE inductance extraction for a square planar
spiral inductor on a lossy silicon substrate.  At low kD, standard EFIE
suffers from low-frequency breakdown (L(f) varies wildly), while A-EFIE
maintains stable extraction across all frequencies.

Physical setup:
  Same as spiral_inductor_silicon.py — 2.5-turn spiral on CMOS stack.

Produces:
  images/aefie_spiral_validation.png

Usage:
    source venv/bin/activate
    python examples/aefie_spiral_validation.py
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
    wheeler_inductance, quality_factor, inductance_from_z,
)
from pyMoM3d.mesh.mesh_data import Mesh
from pyMoM3d.mom.excitation import StripDeltaGapExcitation, compute_feed_signs_along_direction

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters (same as spiral_inductor_silicon.py)
# ---------------------------------------------------------------------------

H_SI     = 300e-6
EPS_SI   = 11.7
SIGMA_SI = 10.0
H_OX     = 5e-6
EPS_OX   = 3.9
N_TURNS  = 2.5
W_TRACE  = 100e-6
S_SPACE  = 100e-6
D_OUT    = 2e-3

Z_SPIRAL = H_SI + H_OX
TEL = 80e-6

# 1-10 GHz sweep (user-specified range)
FREQS_GHZ = np.linspace(1.0, 10.0, 10)
FREQS     = FREQS_GHZ * 1e9


# ---------------------------------------------------------------------------
# Reuse helpers from spiral_inductor_silicon.py
# ---------------------------------------------------------------------------

def build_layer_stack():
    z_si_top = H_SI
    z_ox_top = H_SI + H_OX
    return LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('silicon',    z_bot=0.0,     z_top=z_si_top, eps_r=EPS_SI,
              conductivity=SIGMA_SI),
        Layer('SiO2',       z_bot=z_si_top, z_top=z_ox_top, eps_r=EPS_OX),
        Layer('air',        z_bot=z_ox_top, z_top=np.inf, eps_r=1.0),
    ])


def create_spiral_mesh():
    """Build a square spiral as a connected Gmsh surface."""
    try:
        import gmsh
    except ImportError:
        raise ImportError("gmsh is required for spiral mesh generation")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("spiral")

    segments = []
    pitch = W_TRACE + S_SPACE
    half = D_OUT / 2.0
    x, y = -half, -half

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    n_full_turns = int(N_TURNS)
    has_half = (N_TURNS - n_full_turns) >= 0.45

    for turn in range(n_full_turns):
        inset = turn * pitch
        for d_idx, (dx, dy) in enumerate(directions):
            side = D_OUT - 2 * inset
            if d_idx >= 2:
                side = D_OUT - 2 * inset - pitch
            length = side
            if length <= 0:
                break
            x_end = x + dx * length
            y_end = y + dy * length
            segments.append((x, y, x_end, y_end))
            x, y = x_end, y_end

    if has_half:
        inset = n_full_turns * pitch
        for d_idx in range(2):
            dx, dy = directions[d_idx]
            side = D_OUT - 2 * inset
            if d_idx >= 2:
                side -= pitch
            length = side
            if length <= 0:
                break
            x_end = x + dx * length
            y_end = y + dy * length
            segments.append((x, y, x_end, y_end))
            x, y = x_end, y_end

    if not segments:
        raise ValueError("No spiral segments generated")

    feed_offset = W_TRACE
    x0_s, y0_s, x1_s, y1_s = segments[0]
    dx_s, dy_s = x1_s - x0_s, y1_s - y0_s
    seg_len = np.sqrt(dx_s**2 + dy_s**2)
    if seg_len > feed_offset + 1e-12:
        frac = feed_offset / seg_len
        x_cut = x0_s + frac * dx_s
        y_cut = y0_s + frac * dy_s
        segments = [
            (x0_s, y0_s, x_cut, y_cut),
            (x_cut, y_cut, x1_s, y1_s),
        ] + segments[1:]
        feed_x = x_cut
    else:
        feed_x = x0_s + feed_offset

    x0_e, y0_e, x1_e, y1_e = segments[-1]
    dx_e, dy_e = x1_e - x0_e, y1_e - y0_e
    seg_len_e = np.sqrt(dx_e**2 + dy_e**2)
    if seg_len_e > feed_offset + 1e-12:
        frac_e = (seg_len_e - feed_offset) / seg_len_e
        x_cut_e = x0_e + frac_e * dx_e
        y_cut_e = y0_e + frac_e * dy_e
        segments = segments[:-1] + [
            (x0_e, y0_e, x_cut_e, y_cut_e),
            (x_cut_e, y_cut_e, x1_e, y1_e),
        ]
        ret_x = x_cut_e
        ret_y = y_cut_e
        ret_seg_dir = np.array([dx_e, dy_e, 0.0])
        ret_seg_dir /= np.linalg.norm(ret_seg_dir)
    else:
        ret_x = x1_e
        ret_y = y1_e
        ret_seg_dir = np.array([dx_e, dy_e, 0.0])
        ret_seg_dir /= np.linalg.norm(ret_seg_dir)

    surface_tags = []
    for x0, y0, x1, y1 in segments:
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-12:
            continue
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        hw = W_TRACE / 2.0
        corners = [
            (x0 - nx * hw, y0 - ny * hw, Z_SPIRAL),
            (x1 - nx * hw, y1 - ny * hw, Z_SPIRAL),
            (x1 + nx * hw, y1 + ny * hw, Z_SPIRAL),
            (x0 + nx * hw, y0 + ny * hw, Z_SPIRAL),
        ]
        pts = [gmsh.model.occ.addPoint(*c, meshSize=TEL) for c in corners]
        lines = [
            gmsh.model.occ.addLine(pts[0], pts[1]),
            gmsh.model.occ.addLine(pts[1], pts[2]),
            gmsh.model.occ.addLine(pts[2], pts[3]),
            gmsh.model.occ.addLine(pts[3], pts[0]),
        ]
        loop = gmsh.model.occ.addCurveLoop(lines)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        surface_tags.append(surf)

    if len(surface_tags) > 1:
        tool = [(2, t) for t in surface_tags[1:]]
        gmsh.model.occ.fragment([(2, surface_tags[0])], tool)

    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", TEL)
    gmsh.option.setNumber("Mesh.MeshSizeMin", TEL * 0.3)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    vertices = coords.reshape(-1, 3)
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    triangles_list = []
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        n_nodes_per = len(enodes) // len(etags)
        if n_nodes_per == 3:
            for j in range(len(etags)):
                tri = [tag_to_idx[int(enodes[3*j + kk])] for kk in range(3)]
                triangles_list.append(tri)

    gmsh.finalize()

    if not triangles_list:
        raise RuntimeError("Gmsh produced no triangles for the spiral")

    mesh = Mesh(
        vertices=np.array(vertices, dtype=np.float64),
        triangles=np.array(triangles_list, dtype=np.int32),
    )
    return mesh, feed_x, ret_x, ret_y, ret_seg_dir


def _find_transverse_edges(mesh, basis, x_coord, y_coord, seg_direction, tol_frac=0.3):
    is_y_seg = abs(seg_direction[1]) > abs(seg_direction[0])
    indices = []
    for n in range(basis.num_basis):
        eidx = basis.edge_index[n]
        e = mesh.edges[eidx]
        va = mesh.vertices[e[0]]
        vb = mesh.vertices[e[1]]
        mid = 0.5 * (va + vb)
        edge_vec = vb - va
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        edge_dir = edge_vec / edge_len
        if is_y_seg:
            if abs(mid[1] - y_coord) > W_TRACE * tol_frac:
                continue
            if abs(mid[0] - x_coord) > W_TRACE * tol_frac:
                continue
            if abs(edge_dir[0]) < 0.8:
                continue
        else:
            if abs(mid[0] - x_coord) > W_TRACE * tol_frac:
                continue
            if abs(edge_dir[1]) < 0.8:
                continue
        indices.append(n)
    return indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("pyMoM3d -- A-EFIE Spiral Inductor Validation (1-10 GHz)")
    print("=" * 70)

    # --- Analytical reference ---
    d_in = D_OUT - 2 * N_TURNS * (W_TRACE + S_SPACE)
    d_in = max(d_in, W_TRACE)
    L_wheeler = wheeler_inductance(N_TURNS, D_OUT, d_in)
    print(f"\nAnalytical (Wheeler): L = {L_wheeler*1e9:.3f} nH")

    # --- Layer stack ---
    stack = build_layer_stack()

    # --- Mesh ---
    print("\nMeshing spiral...")
    mesh, feed_x, ret_x, ret_y, ret_seg_dir = create_spiral_mesh()
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    print(f"  Triangles: {stats['num_triangles']}")
    print(f"  RWG basis: {basis.num_basis}")
    print(f"  Mean edge: {stats['mean_edge_length']*1e6:.1f} um")

    # --- Port ---
    feed_indices = _find_transverse_edges(
        mesh, basis, feed_x, -D_OUT/2.0, np.array([1, 0, 0]))
    ret_indices = _find_transverse_edges(
        mesh, basis, ret_x, ret_y, ret_seg_dir)
    print(f"  Signal edges: {len(feed_indices)}")
    print(f"  Return edges: {len(ret_indices)}")

    if not feed_indices:
        print("ERROR: No feed edges found.")
        return

    # Compute feed signs for correct RWG orientation
    feed_signs = compute_feed_signs_along_direction(
        mesh, basis, feed_indices, np.array([1, 0, 0]))

    port = Port(name='P1', feed_basis_indices=feed_indices,
                feed_signs=feed_signs,
                return_basis_indices=ret_indices if ret_indices else [])

    # --- Build Simulation ---
    exc_dummy = StripDeltaGapExcitation(feed_basis_indices=feed_indices, voltage=1.0)
    config = SimulationConfig(
        frequency=FREQS[0],
        excitation=exc_dummy,
        quad_order=4,
        backend='auto',
        layer_stack=stack,
        source_layer_name='SiO2',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # --- Standard EFIE sweep ---
    print(f"\n--- Standard EFIE ({FREQS[0]/1e9:.0f}-{FREQS[-1]/1e9:.0f} GHz) ---")
    ext_efie = NetworkExtractor(
        sim, [port], low_freq_stabilization='none')
    results_efie = ext_efie.extract(FREQS.tolist())

    Z_efie = np.array([r.Z_matrix[0, 0] for r in results_efie])
    L_efie = np.array([inductance_from_z(z, f) for z, f in zip(Z_efie, FREQS)])
    Q_efie = np.array([quality_factor(z) for z in Z_efie])
    cond_efie = np.array([r.condition_number for r in results_efie])

    # --- A-EFIE sweep ---
    print(f"\n--- A-EFIE ({FREQS[0]/1e9:.0f}-{FREQS[-1]/1e9:.0f} GHz) ---")
    ext_aefie = NetworkExtractor(
        sim, [port], low_freq_stabilization='aefie')
    results_aefie = ext_aefie.extract(FREQS.tolist())

    Z_aefie = np.array([r.Z_matrix[0, 0] for r in results_aefie])
    L_aefie = np.array([inductance_from_z(z, f) for z, f in zip(Z_aefie, FREQS)])
    Q_aefie = np.array([quality_factor(z) for z in Z_aefie])
    cond_aefie = np.array([r.condition_number for r in results_aefie])

    # --- Print comparison table ---
    kD_arr = 2 * np.pi * FREQS / c0 * D_OUT

    print(f"\n  {'f(GHz)':>7}  {'kD':>7}  "
          f"{'L_EFIE':>9}  {'L_AEFIE':>9}  "
          f"{'cond_EFIE':>10}  {'cond_AEFIE':>11}  "
          f"{'L_err_E':>8}  {'L_err_A':>8}")
    print("  " + "-" * 80)

    for i, freq in enumerate(FREQS):
        kD = kD_arr[i]
        Le = L_efie[i] * 1e9
        La = L_aefie[i] * 1e9
        ce = cond_efie[i]
        ca = cond_aefie[i]
        Lw = L_wheeler * 1e9
        err_e = abs(Le - Lw) / Lw * 100 if Lw > 0 else float('nan')
        err_a = abs(La - Lw) / Lw * 100 if Lw > 0 else float('nan')
        print(f"  {freq/1e9:>7.1f}  {kD:>7.4f}  "
              f"{Le:>9.3f}  {La:>9.3f}  "
              f"{ce:>10.2e}  {ca:>11.2e}  "
              f"{err_e:>7.1f}%  {err_a:>7.1f}%")

    # --- L(f) flatness metrics ---
    L_finite_e = L_efie[np.isfinite(L_efie) & (L_efie > 0)]
    L_finite_a = L_aefie[np.isfinite(L_aefie) & (L_aefie > 0)]

    if len(L_finite_e) > 1:
        ratio_e = np.max(L_finite_e) / np.min(L_finite_e)
    else:
        ratio_e = float('inf')

    if len(L_finite_a) > 1:
        ratio_a = np.max(L_finite_a) / np.min(L_finite_a)
    else:
        ratio_a = float('inf')

    print(f"\n  L(f) variation (max/min):")
    print(f"    Standard EFIE:  {ratio_e:.1f}x")
    print(f"    A-EFIE:         {ratio_a:.1f}x")
    print(f"    Target:         < 2x (ideally < 1.5x)")

    if ratio_a < ratio_e:
        print(f"\n  A-EFIE L(f) is more stable than EFIE  [PASS]")
    else:
        print(f"\n  A-EFIE L(f) is NOT more stable  [CHECK]")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        rf'A-EFIE vs EFIE: Spiral Inductor on Si '
        rf'($\varepsilon_r={EPS_SI}$, $\sigma={SIGMA_SI}$ S/m) '
        rf'$-$ {N_TURNS} turns, $d_{{\mathrm{{out}}}} = {D_OUT*1e3:.1f}$ mm',
        fontsize=12,
    )

    # (0,0) Inductance comparison
    ax = axes[0, 0]
    ax.plot(FREQS_GHZ, L_efie * 1e9, 'b--o', ms=4, lw=1.2, label='EFIE')
    ax.plot(FREQS_GHZ, L_aefie * 1e9, 'r-s', ms=5, lw=1.5, label='A-EFIE')
    ax.axhline(L_wheeler * 1e9, color='k', ls=':', lw=1.5,
               label=rf'Wheeler ($L = {L_wheeler*1e9:.2f}$ nH)')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$L$ (nH)')
    ax.set_title(r'Inductance $L(f)$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1) Condition number comparison
    ax = axes[0, 1]
    ax.semilogy(FREQS_GHZ, cond_efie, 'b--o', ms=4, lw=1.2, label='EFIE')
    ax.semilogy(FREQS_GHZ, cond_aefie, 'r-s', ms=5, lw=1.5, label='A-EFIE')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'Condition number')
    ax.set_title(r'System Conditioning')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,0) Quality factor
    ax = axes[1, 0]
    ax.plot(FREQS_GHZ, Q_efie, 'b--o', ms=4, lw=1.2, label='EFIE')
    ax.plot(FREQS_GHZ, Q_aefie, 'r-s', ms=5, lw=1.5, label='A-EFIE')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$Q$')
    ax.set_title(r'Quality Factor $Q(f)$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,1) Input impedance
    ax = axes[1, 1]
    ax.plot(FREQS_GHZ, Z_efie.real, 'b--', lw=1, alpha=0.7, label=r'$R$ (EFIE)')
    ax.plot(FREQS_GHZ, Z_efie.imag, 'b:', lw=1, alpha=0.7, label=r'$X$ (EFIE)')
    ax.plot(FREQS_GHZ, Z_aefie.real, 'r-', lw=1.5, label=r'$R$ (A-EFIE)')
    ax.plot(FREQS_GHZ, Z_aefie.imag, 'r--', lw=1.5, label=r'$X$ (A-EFIE)')
    X_wheeler = 2 * np.pi * FREQS * L_wheeler
    ax.plot(FREQS_GHZ, X_wheeler, 'k:', lw=1, alpha=0.5,
            label=r'$\omega L_{\mathrm{Wheeler}}$')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$Z_{\mathrm{in}}$ ($\Omega$)')
    ax.set_title(r'Input Impedance')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'aefie_spiral_validation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
