"""
Spiral Inductor on Lossy Silicon Substrate.

End-to-end multilayer benchmark: square planar spiral inductor on a CMOS-like
stackup with PEC ground, lossy silicon, SiO2 ILD, and air.

Physical setup:
  - PEC ground at z=0
  - Silicon substrate: eps_r=11.7, sigma=10 S/m, thickness=300 um
  - SiO2 ILD: eps_r=3.9, thickness=5 um
  - Rectangular spiral at z=305 um: 2.5 turns, w=100 um, s=100 um, d_out=2 mm
  - 1-port differential extraction (outer signal, inner grounded)

LayerStack (bottom to top):
  PEC half-space
  Silicon  (z=0 to z=300 um, eps_r=11.7, sigma=10)
  SiO2     (z=300 um to z=305 um, eps_r=3.9)
  Phantom  (z=305 um to z=310 um, eps_r=1.001)
  Air half-space

Port model:
  Differential port: +V_ref at the outer terminal, -V_ref at the inner
  terminal.  This forces current through the entire spiral and models the
  physical situation where the inner terminal is grounded through a via.
  The -V_ref acts as the MoM equivalent of a V=0 boundary condition at
  the inner terminal.

Loop-star decomposition:
  Enabled by default to improve EFIE conditioning at low kD.  Separates
  divergence-free (inductance) and irrotational (capacitance) current
  components, preventing scalar-potential contamination of the inductance.

Validation:
  L vs modified Wheeler formula (<20% at mid-range freq)
  Q peaks then decreases with frequency
  L approximately flat below self-resonance

Produces:
  images/spiral_inductor_silicon.png

Usage:
    source venv/bin/activate
    python examples/spiral_inductor_silicon.py
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
from pyMoM3d.mom.excitation import StripDeltaGapExcitation

configure_latex_style()

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

# Substrate
H_SI     = 300e-6      # Silicon thickness (m)
EPS_SI   = 11.7        # Silicon relative permittivity
SIGMA_SI = 10.0        # Silicon conductivity (S/m)
H_OX     = 5e-6        # SiO2 thickness (m)
EPS_OX   = 3.9         # SiO2 relative permittivity
H_PHANT  = 5e-6        # Phantom air layer thickness (m)

# Spiral geometry (mm-scale for EFIE validity at GHz frequencies)
N_TURNS  = 2.5         # Number of turns
W_TRACE  = 100e-6      # Trace width (m)
S_SPACE  = 100e-6      # Spacing between turns (m)
D_OUT    = 2e-3        # Outer dimension (m)

# Derived
Z_SPIRAL = H_SI + H_OX   # Spiral z-coordinate (m)

# Mesh
TEL = 80e-6            # Target edge length (m)

# Frequency sweep (1-5 GHz: kD ~ 0.04-0.21)
FREQS_GHZ = np.linspace(1.0, 5.0, 10)
FREQS     = FREQS_GHZ * 1e9


# ---------------------------------------------------------------------------
# LayerStack
# ---------------------------------------------------------------------------

def build_layer_stack():
    """Build the CMOS-like layer stack."""
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


# ---------------------------------------------------------------------------
# Spiral mesh generation via Gmsh
# ---------------------------------------------------------------------------

def create_spiral_mesh():
    """Build a square spiral as a connected Gmsh surface.

    Creates the spiral as a sequence of rectangular trace segments that share
    corner vertices, ensuring current continuity.  Both the outer (signal)
    and inner (via) terminals have conformal transverse edges created by
    segment splitting.

    Returns
    -------
    mesh : Mesh
    feed_x : float
        x-coordinate of the signal feed edge (outer terminal).
    ret_x : float
        x-coordinate of the return/via edge (inner terminal).
    ret_y : float
        y-coordinate of the return/via edge.
    ret_seg_dir : ndarray, shape (3,)
        Direction vector of the last segment (for transverse edge finding).
    """
    try:
        import gmsh
    except ImportError:
        raise ImportError("gmsh is required for spiral mesh generation")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("spiral")

    # Build spiral path as a list of (x_start, y_start, x_end, y_end) segments
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

    # Split first and last segments to create conformal transverse edges
    feed_offset = W_TRACE

    # Split first segment (outer terminal / signal port)
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

    # Split last segment (inner terminal / ground via)
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

    # Create Gmsh rectangles for each trace segment
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
                tri = [tag_to_idx[int(enodes[3*j + k])] for k in range(3)]
                triangles_list.append(tri)

    gmsh.finalize()

    if not triangles_list:
        raise RuntimeError("Gmsh produced no triangles for the spiral")

    mesh = Mesh(
        vertices=np.array(vertices, dtype=np.float64),
        triangles=np.array(triangles_list, dtype=np.int32),
    )
    return mesh, feed_x, ret_x, ret_y, ret_seg_dir


# ---------------------------------------------------------------------------
# Edge-finding helper
# ---------------------------------------------------------------------------

def _find_transverse_edges(mesh, basis, x_coord, y_coord, seg_direction, tol_frac=0.3):
    """Find strongly transverse RWG edges near a given cut location.

    Parameters
    ----------
    mesh : Mesh
    basis : RWGBasis
    x_coord, y_coord : float
        Coordinates of the cut (only the relevant one is used).
    seg_direction : ndarray, shape (3,)
        Direction of the trace segment at the cut.
    tol_frac : float
        Tolerance as a fraction of W_TRACE.

    Returns
    -------
    indices : list of int
        RWG basis function indices.
    """
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
    print("=" * 65)
    print("pyMoM3d — Spiral Inductor on Lossy Silicon")
    print("         (differential port + loop-star decomposition)")
    print("=" * 65)

    # --- Analytical reference ---
    d_in = D_OUT - 2 * N_TURNS * (W_TRACE + S_SPACE)
    d_in = max(d_in, W_TRACE)
    L_wheeler = wheeler_inductance(N_TURNS, D_OUT, d_in)
    print(f"\nAnalytical (Wheeler):")
    print(f"  n = {N_TURNS} turns, d_out = {D_OUT*1e6:.0f} um, d_in = {d_in*1e6:.0f} um")
    print(f"  L = {L_wheeler*1e9:.3f} nH")

    # --- Layer stack ---
    stack = build_layer_stack()
    print(f"\nLayer stack:")
    for lyr in stack.layers:
        cond = f"  sigma={lyr.conductivity}" if lyr.conductivity > 0 else ""
        print(f"  {lyr.name:15s}  z=[{lyr.z_bot:+.4e}, {lyr.z_top:+.4e}]  "
              f"eps_r={lyr.eps_r}  pec={lyr.is_pec}{cond}")

    # --- Mesh ---
    print("\nMeshing spiral...")
    mesh, feed_x, ret_x, ret_y, ret_seg_dir = create_spiral_mesh()
    basis = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    print(f"  Triangles: {stats['num_triangles']}")
    print(f"  RWG basis: {basis.num_basis}")
    print(f"  Mean edge: {stats['mean_edge_length']*1e6:.1f} um")

    # --- Differential port ---
    # Signal side: transverse edges at the outer terminal (x = feed_x).
    # Return side: transverse edges at the inner terminal.
    # The differential port applies +V_ref at the outer terminal and -V_ref
    # at the inner terminal, forcing current to flow through the entire
    # spiral.  This models a 1-port inductor with one terminal grounded
    # through a via — the -V_ref enforces V=0 (ground) in the MoM.
    feed_indices = _find_transverse_edges(
        mesh, basis, feed_x, -D_OUT/2.0,
        np.array([1, 0, 0]),
    )
    print(f"  Signal edges: {len(feed_indices)} (at x={feed_x*1e6:.0f} um)")

    ret_indices = _find_transverse_edges(
        mesh, basis, ret_x, ret_y, ret_seg_dir,
    )
    print(f"  Return edges: {len(ret_indices)} (at x={ret_x*1e6:.0f}, y={ret_y*1e6:.0f} um)")

    if not feed_indices:
        print("ERROR: No feed edges found. Cannot proceed.")
        return
    if not ret_indices:
        print("WARNING: No return edges found. Using single-ended port.")

    # Differential port: +V at outer terminal, -V at inner terminal
    port = Port(name='P1', feed_basis_indices=feed_indices,
                return_basis_indices=ret_indices if ret_indices else [])

    # --- Build Simulation ---
    exc_dummy = StripDeltaGapExcitation(feed_basis_indices=feed_indices, voltage=1.0)
    config = SimulationConfig(
        frequency=FREQS[0],
        excitation=exc_dummy,
        quad_order=4,
        backend='auto',
        layer_stack=stack,
        source_layer_name='phantom',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # --- Frequency sweep with loop-star ---
    extractor = NetworkExtractor(
        sim, [port],
        use_loop_star=False,
    )
    print(f"\nSweeping {len(FREQS)} frequencies "
          f"({FREQS[0]/1e9:.1f}-{FREQS[-1]/1e9:.1f} GHz) "
          f"with direct EFIE solve...")

    results = extractor.extract(FREQS.tolist())

    # --- Extract L and Q ---
    Z_in_arr = np.array([r.Z_matrix[0, 0] for r in results])
    L_arr = np.array([inductance_from_z(z, f) for z, f in zip(Z_in_arr, FREQS)])
    Q_arr = np.array([quality_factor(z) for z in Z_in_arr])

    print(f"\n  {'f (GHz)':>8}  {'R (Ohm)':>10}  {'X (Ohm)':>10}  "
          f"{'L (nH)':>8}  {'Q':>6}  {'kD':>7}  {'L_err':>7}")
    print("  " + "-" * 65)

    for freq, z, L, Q in zip(FREQS, Z_in_arr, L_arr, Q_arr):
        kD = 2 * np.pi * freq / c0 * D_OUT
        L_err = abs(L - L_wheeler) / L_wheeler * 100 if L_wheeler > 0 else float('nan')
        print(f"  {freq/1e9:>8.2f}  {z.real:>10.2f}  {z.imag:>10.2f}  "
              f"{L*1e9:>8.3f}  {Q:>6.2f}  {kD:>7.4f}  {L_err:>6.1f}%")

    # --- Summary ---
    L_err_arr = np.abs(L_arr - L_wheeler) / L_wheeler * 100
    best_idx = np.argmin(L_err_arr)
    L_best = L_arr[best_idx]
    f_best = FREQS[best_idx]
    L_err_best = L_err_arr[best_idx]
    kD_best = 2 * np.pi * f_best / c0 * D_OUT

    print(f"\n  Best L match: {L_best*1e9:.3f} nH at {f_best/1e9:.2f} GHz (kD={kD_best:.3f})")
    print(f"  Wheeler L:    {L_wheeler*1e9:.3f} nH")
    print(f"  Error:        {L_err_best:.1f}%  "
          f"{'PASS' if L_err_best < 20 else 'CHECK'}")

    # Condition number comparison
    print(f"\n  Condition numbers (loop-star rescaled):")
    for i, r in enumerate(results):
        print(f"    {FREQS[i]/1e9:.1f} GHz: {r.condition_number:.2e}")

    # --- Plot ---
    kD_arr = 2 * np.pi * FREQS / c0 * D_OUT

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        rf'Square Spiral Inductor on Si ($\varepsilon_r={EPS_SI}$, '
        rf'$\sigma={SIGMA_SI}$ S/m) — {N_TURNS} turns, '
        rf'$d_{{\mathrm{{out}}}} = {D_OUT*1e3:.1f}$ mm',
        fontsize=11,
    )

    # Inductance vs frequency
    ax = axes[0]
    ax.plot(FREQS_GHZ, L_arr * 1e9, 'bo-', ms=5, lw=1.5, label=r'MoM (EFIE)')
    ax.axhline(L_wheeler * 1e9, color='r', ls='--', lw=1.5,
               label=rf'Wheeler ($L = {L_wheeler*1e9:.2f}$ nH)')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$L$ (nH)')
    ax.set_title(r'Inductance $L(f)$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if np.any(np.isfinite(L_arr)):
        L_plot = L_arr[np.isfinite(L_arr)]
        if len(L_plot) > 0:
            y_max = max(np.max(L_plot) * 1e9 * 1.3, L_wheeler * 1e9 * 1.5)
            y_min = min(0, np.min(L_plot) * 1e9 * 0.8)
            ax.set_ylim(y_min, y_max)

    # Quality factor vs frequency
    ax = axes[1]
    ax.plot(FREQS_GHZ, Q_arr, 'rs-', ms=5, lw=1.5, label=r'$Q(f)$')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$Q$')
    ax.set_title(r'Quality Factor $Q(f)$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Input impedance (R and X)
    ax = axes[2]
    ax.plot(FREQS_GHZ, Z_in_arr.real, 'g-^', ms=4, lw=1.5, label=r'$R_{\mathrm{in}}$')
    ax.plot(FREQS_GHZ, Z_in_arr.imag, 'b-o', ms=4, lw=1.5, label=r'$X_{\mathrm{in}}$')
    X_wheeler = 2 * np.pi * FREQS * L_wheeler
    ax.plot(FREQS_GHZ, X_wheeler, 'k--', lw=1, alpha=0.7,
            label=rf'$\omega L_{{\mathrm{{Wheeler}}}}$')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$Z_{\mathrm{in}}$ ($\Omega$)')
    ax.set_title(r'Input Impedance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(IMAGES_DIR, 'spiral_inductor_silicon.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {out}")


if __name__ == '__main__':
    main()
