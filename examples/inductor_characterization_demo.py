"""
EDA-Grade Inductor Characterization Demo.

Demonstrates Y-parameter based inductor characterization following the
methodology used in commercial EM solvers (EMX, Momentum, HFSS, Sonnet).

Shows:
  1. Y-parameter L(f), Q(f), R(f) extraction (flat L below SRF)
  2. Comparison of Z-based vs Y-based extraction
  3. Broadband pi-model fitting
  4. Port parasitic de-embedding
  5. Wheeler formula comparison

Uses the same CMOS spiral geometry as spiral_inductor_silicon.py and
aefie_spiral_validation.py.

Produces:
  images/inductor_characterization.png
  images/inductor_model_fit.png
  images/inductor_z_vs_y.png

Usage:
    source venv/bin/activate
    python examples/inductor_characterization_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor,
    Layer, LayerStack,
    configure_latex_style, c0,
    wheeler_inductance,
    InductorCharacterization,
    plot_inductor_characterization,
    plot_model_fit,
    plot_z_vs_y_comparison,
)
from pyMoM3d.mesh.mesh_data import Mesh
from pyMoM3d.mom.excitation import StripDeltaGapExcitation

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
H_PHANT  = 5e-6

N_TURNS  = 2.5
W_TRACE  = 100e-6
S_SPACE  = 100e-6
D_OUT    = 2e-3

Z_SPIRAL = H_SI + H_OX
TEL = 80e-6

# Sweep range: 0.5 - 10 GHz, 20 points
FREQS_GHZ = np.linspace(0.5, 10.0, 20)
FREQS     = FREQS_GHZ * 1e9


# ---------------------------------------------------------------------------
# Geometry helpers (reused from aefie_spiral_validation.py)
# ---------------------------------------------------------------------------

def build_layer_stack():
    z_si_top = H_SI
    z_ox_top = H_SI + H_OX
    z_ph_top = z_ox_top + H_PHANT
    return LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('silicon',    z_bot=0.0,     z_top=z_si_top, eps_r=EPS_SI,
              conductivity=SIGMA_SI),
        Layer('SiO2',       z_bot=z_si_top, z_top=z_ox_top, eps_r=EPS_OX),
        Layer('phantom',    z_bot=z_ox_top, z_top=z_ph_top, eps_r=1.001),
        Layer('air',        z_bot=z_ph_top, z_top=np.inf, eps_r=1.0),
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


def _find_transverse_edges(mesh, basis, x_coord, y_coord, seg_direction,
                           tol_frac=0.3):
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
    print("pyMoM3d -- EDA-Grade Inductor Characterization")
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

    port = Port(name='P1', feed_basis_indices=feed_indices,
                return_basis_indices=ret_indices if ret_indices else [])

    # --- Build Simulation ---
    exc_dummy = StripDeltaGapExcitation(
        feed_basis_indices=feed_indices, voltage=1.0)
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
    print(f"\n--- NetworkExtractor sweep "
          f"({FREQS[0]/1e9:.1f}-{FREQS[-1]/1e9:.1f} GHz, "
          f"{len(FREQS)} points) ---")
    extractor = NetworkExtractor(sim, [port], low_freq_stabilization='none')
    results = extractor.extract(FREQS.tolist())

    # --- InductorCharacterization ---
    print("\n--- Y-parameter characterization ---")
    char = InductorCharacterization(results)
    cr = char.characterize(fit_model=True)

    print(cr.summary())

    # --- Wheeler comparison ---
    comp = char.compare_with_wheeler(N_TURNS, D_OUT, d_in)
    print(f"\n  Wheeler L:         {comp['wheeler_L']*1e9:.3f} nH")
    print(f"  Extracted L_dc:    {comp['L_dc']*1e9:.3f} nH")
    print(f"  Error:             {comp['error_pct']:.1f}%")

    # --- L(f) variation comparison ---
    L_z = cr.L_z
    L_y = cr.L
    L_z_pos = L_z[L_z > 0]
    L_y_pos = L_y[L_y > 0]
    z_var = L_z_pos.max() / L_z_pos.min() if len(L_z_pos) > 0 else float('inf')
    y_var = L_y_pos.max() / L_y_pos.min() if len(L_y_pos) > 0 else float('inf')
    print(f"\n  L(f) variation (max/min):")
    print(f"    Z-parameter:   {z_var:.1f}x")
    print(f"    Y-parameter:   {y_var:.1f}x")

    # --- Print frequency table ---
    print(f"\n  {'f(GHz)':>8s}  {'L_z(nH)':>8s}  {'L_y(nH)':>8s}  "
          f"{'Q_z':>6s}  {'Q_y':>6s}  {'R(Ohm)':>8s}")
    print("  " + "-" * 56)
    for p in cr.params:
        print(f"  {p.frequency/1e9:8.2f}  {p.L_z*1e9:8.3f}  {p.L*1e9:8.3f}  "
              f"{p.Q_z:6.1f}  {p.Q:6.1f}  {p.R:8.3f}")

    # --- Plot 1: Full characterization ---
    title = (
        rf'Spiral Inductor on Si '
        rf'($\varepsilon_r={EPS_SI}$, $\sigma={SIGMA_SI}$ S/m) '
        rf'$-$ {N_TURNS} turns, '
        rf'$d_{{\mathrm{{out}}}} = {D_OUT*1e3:.1f}$ mm'
    )
    path1 = os.path.join(IMAGES_DIR, 'inductor_characterization.png')
    plot_inductor_characterization(
        cr, wheeler_L=L_wheeler, title=title,
        save_path=path1, show=False,
    )
    print(f"\n  Saved -> {path1}")

    # --- Plot 2: Z vs Y comparison ---
    path2 = os.path.join(IMAGES_DIR, 'inductor_z_vs_y.png')
    plot_z_vs_y_comparison(cr, wheeler_L=L_wheeler,
                           save_path=path2, show=False)
    print(f"  Saved -> {path2}")

    # --- Plot 3: Model fit ---
    if cr.pi_model is not None:
        path3 = os.path.join(IMAGES_DIR, 'inductor_model_fit.png')
        plot_model_fit(cr, save_path=path3, show=False)
        print(f"  Saved -> {path3}")

    # --- Demonstrate de-embedding ---
    print("\n--- De-embedding demo ---")
    omega_mid = 2 * np.pi * FREQS[len(FREQS)//2]
    Z_gap_series = 1j * omega_mid * 1e-12  # ~1 pH series inductance
    Y_gap_shunt = 1j * omega_mid * 0.1e-15  # ~0.1 fF shunt capacitance
    result_mid = results[len(FREQS)//2]
    result_deemb = result_mid.correct_port_parasitics(
        series_Z=[Z_gap_series],
        shunt_Y=[Y_gap_shunt],
    )
    Z_raw = result_mid.Z_matrix[0, 0]
    Z_deemb = result_deemb.Z_matrix[0, 0]
    print(f"  At f = {result_mid.frequency/1e9:.1f} GHz:")
    print(f"    Raw    Z11 = {Z_raw.real:.4f} + j{Z_raw.imag:.4f}")
    print(f"    De-emb Z11 = {Z_deemb.real:.4f} + j{Z_deemb.imag:.4f}")
    print(f"    Delta: {abs(Z_raw - Z_deemb):.6f} Ohm")

    print("\nDone.")


if __name__ == '__main__':
    main()
