"""Wire-wire and wire-surface EFIE interaction kernels.

Implements the Pocklington thin-wire EFIE kernel for wire-wire interactions
and mixed 1D-2D quadrature for wire-surface coupling.

Formulation
-----------
The EFIE impedance between two basis functions (wire or surface) is:

    Z_mn = jkη ∫∫ f_m · f_n g(R) dτ dτ'  -  (jη/k) ∫∫ div(f_m) div(f_n) g(R) dτ dτ'

where g(R) = exp(-jkR) / (4πR) is the free-space Green's function.

For wire-wire interactions, the reduced kernel uses R = sqrt((s-s')² + a²)
to avoid the 1/R singularity (thin-wire approximation, wire radius a).

For wire-surface interactions, no singularity extraction is needed because
wire interior and triangle interior occupy disjoint spatial regions.
"""

from __future__ import annotations

import numpy as np

from .wire_basis import WireBasis, WireMesh


def wire_quad_rule(n_points: int) -> tuple:
    """Gauss-Legendre quadrature on [0, 1].

    Parameters
    ----------
    n_points : int
        Number of quadrature points.

    Returns
    -------
    weights : ndarray, shape (n_points,)
    points : ndarray, shape (n_points,)
        Quadrature points in [0, 1].
    """
    # scipy Gauss-Legendre on [-1, 1], mapped to [0, 1]
    from numpy.polynomial.legendre import leggauss
    xi, wi = leggauss(n_points)
    points = 0.5 * (xi + 1.0)
    weights = 0.5 * wi
    return weights, points


def _green_reduced(k: complex, R_sq: np.ndarray) -> np.ndarray:
    """Reduced-kernel Green's function g(R) = exp(-jkR) / (4πR).

    Parameters
    ----------
    k : complex
        Wavenumber.
    R_sq : ndarray
        Squared distances (already includes wire radius: R² = Δs² + a²).

    Returns
    -------
    g : ndarray, complex
    """
    R = np.sqrt(R_sq)
    R = np.maximum(R, 1e-30)
    return np.exp(-1j * k * R) / (4.0 * np.pi * R)


def _segment_segment_integral(
    k: complex,
    node_start_m: np.ndarray,
    node_end_m: np.ndarray,
    dl_m: float,
    dir_m: np.ndarray,
    sign_m: int,
    node_start_n: np.ndarray,
    node_end_n: np.ndarray,
    dl_n: float,
    dir_n: np.ndarray,
    sign_n: int,
    radius: float,
    w_quad: np.ndarray,
    t_quad: np.ndarray,
) -> tuple:
    """Compute vector and scalar potential integrals between two wire segments.

    Returns (I_A, I_Phi) contributions for one segment-segment pair.

    The rooftop basis on a segment:
      If sign = +1 (seg_plus): f(t) = t * dl * d̂  (rising from 0 to dl)
      If sign = -1 (seg_minus): f(t) = (1-t) * dl * d̂  (falling from dl to 0)

    Divergence:
      sign = +1: div(f) = +1/dl
      sign = -1: div(f) = -1/dl

    Parameters
    ----------
    node_start_m, node_end_m : ndarray (3,)
        Endpoints of test segment.
    dl_m : float
        Length of test segment.
    dir_m : ndarray (3,)
        Unit direction of test segment.
    sign_m : int
        +1 for seg_plus, -1 for seg_minus.
    (same for _n = source segment)
    radius : float
        Wire radius for reduced kernel.
    w_quad, t_quad : ndarray
        Quadrature weights and points on [0, 1].

    Returns
    -------
    I_A : complex
        Vector potential integral: ∫∫ f_m(s) · f_n(s') g(R) ds ds'
    I_Phi : complex
        Scalar potential integral: ∫∫ div(f_m) div(f_n) g(R) ds ds'
    """
    Q = len(w_quad)

    # Quadrature points on test segment: r_m(t) = start_m + t * dl_m * dir_m
    r_m = node_start_m[np.newaxis, :] + t_quad[:, np.newaxis] * dl_m * dir_m[np.newaxis, :]  # (Q, 3)
    # Quadrature points on source segment
    r_n = node_start_n[np.newaxis, :] + t_quad[:, np.newaxis] * dl_n * dir_n[np.newaxis, :]  # (Q, 3)

    # Rooftop values at quadrature points
    if sign_m == 1:
        f_m_scalar = t_quad  # rising: f(t) = t
    else:
        f_m_scalar = 1.0 - t_quad  # falling: f(t) = 1-t

    if sign_n == 1:
        f_n_scalar = t_quad
    else:
        f_n_scalar = 1.0 - t_quad

    # Divergence (constant on each segment)
    div_m = float(sign_m) / dl_m
    div_n = float(sign_n) / dl_n

    # Dot product of segment directions (constant)
    dot_dir = np.dot(dir_m, dir_n)

    # Double integral via outer product of quadrature
    # R²[i,j] = |r_m[i] - r_n[j]|² + a²
    diff = r_m[:, np.newaxis, :] - r_n[np.newaxis, :, :]  # (Q, Q, 3)
    R_sq = np.sum(diff * diff, axis=-1) + radius * radius  # (Q, Q)

    g_vals = _green_reduced(k, R_sq)  # (Q, Q)

    # Vector potential: ∫∫ f_m · f_n * g * ds ds'
    # f_m(s) = f_m_scalar * dl_m * dir_m, ds = dl_m * dt
    # f_m · f_n = f_m_scalar * f_n_scalar * dl_m * dl_n * dot_dir
    # ds ds' = dl_m * dl_n * dt dt'
    # Total: ∫∫ = dl_m² * dl_n² * dot_dir * ΣΣ w_i w_j f_m[i] f_n[j] g[i,j]
    f_outer = f_m_scalar[:, np.newaxis] * f_n_scalar[np.newaxis, :]  # (Q, Q)
    w_outer = w_quad[:, np.newaxis] * w_quad[np.newaxis, :]  # (Q, Q)

    I_A = dl_m * dl_m * dl_n * dl_n * dot_dir * np.sum(w_outer * f_outer * g_vals)

    # Scalar potential: ∫∫ div(f_m) div(f_n) g * ds ds'
    # div(f_m) = sign_m/dl_m (constant), ds = dl_m * dt
    # Total: div_m * div_n * dl_m * dl_n * ΣΣ w_i w_j g[i,j]
    I_Phi = div_m * div_n * dl_m * dl_n * np.sum(w_outer * g_vals)

    return complex(I_A), complex(I_Phi)


def fill_wire_wire(
    Z_ww: np.ndarray,
    wire_basis: WireBasis,
    wire_mesh: WireMesh,
    k: complex,
    eta: complex,
    n_quad: int = 8,
) -> None:
    """Fill the wire-wire impedance block using Pocklington's thin-wire EFIE.

    Parameters
    ----------
    Z_ww : ndarray, shape (N_w, N_w), complex128
        Output matrix, filled in-place.
    wire_basis : WireBasis
    wire_mesh : WireMesh
    k : complex
        Wavenumber (rad/m).
    eta : complex
        Wave impedance (Ohm).
    n_quad : int
        Number of 1D Gauss-Legendre quadrature points per segment.
    """
    N_w = wire_basis.num_basis
    w_quad, t_quad = wire_quad_rule(n_quad)

    prefactor_A = 1j * k * eta
    prefactor_Phi = -1j * eta / k

    nodes = wire_mesh.nodes
    segs = wire_mesh.segments

    for m in range(N_w):
        for n in range(m, N_w):
            z_mn = 0.0 + 0.0j

            # Each basis function has 2 segments: seg_plus and seg_minus
            m_segs = [
                (wire_basis.seg_plus[m], +1),
                (wire_basis.seg_minus[m], -1),
            ]
            n_segs = [
                (wire_basis.seg_plus[n], +1),
                (wire_basis.seg_minus[n], -1),
            ]

            for si_m, sign_m in m_segs:
                seg_m = segs[si_m]
                for si_n, sign_n in n_segs:
                    seg_n = segs[si_n]

                    I_A, I_Phi = _segment_segment_integral(
                        k,
                        nodes[seg_m.node_start], nodes[seg_m.node_end],
                        seg_m.length, seg_m.direction, sign_m,
                        nodes[seg_n.node_start], nodes[seg_n.node_end],
                        seg_n.length, seg_n.direction, sign_n,
                        seg_m.radius,
                        w_quad, t_quad,
                    )

                    z_mn += prefactor_A * I_A + prefactor_Phi * I_Phi

            Z_ww[m, n] = z_mn
            if m != n:
                Z_ww[n, m] = z_mn  # reciprocity


def _segment_triangle_integral(
    k: complex,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    dl_seg: float,
    dir_seg: np.ndarray,
    sign_seg: int,
    tri_verts: np.ndarray,
    fv_rwg: np.ndarray,
    sign_rwg: int,
    l_rwg: float,
    A_tri: float,
    radius: float,
    w_wire: np.ndarray,
    t_wire: np.ndarray,
    w_tri: np.ndarray,
    bary_tri: np.ndarray,
    twice_area: float,
) -> tuple:
    """Compute vector and scalar potential integrals between a wire segment
    and a triangle (one half of an RWG basis function).

    Returns (I_A, I_Phi).

    Wire basis on segment:
      f_wire(t) = f_scalar(t) * dl * d̂_seg,  where f_scalar = t (plus) or 1-t (minus)
      div(f_wire) = sign_seg / dl

    RWG basis on triangle:
      f_rwg(r') = (l_rwg / (2*A_tri)) * ρ_src,  where ρ_src = r' - r_fv (if T+) or r_fv - r' (if T-)
      div(f_rwg) = sign_rwg * l_rwg / A_tri
    """
    Q_w = len(w_wire)
    Q_t = len(w_tri)

    # Wire quadrature points
    r_wire = seg_start[np.newaxis, :] + t_wire[:, np.newaxis] * dl_seg * dir_seg[np.newaxis, :]  # (Q_w, 3)

    # Wire rooftop scalar values
    if sign_seg == 1:
        f_w_scalar = t_wire  # (Q_w,)
    else:
        f_w_scalar = 1.0 - t_wire  # (Q_w,)

    # Triangle quadrature points
    r_tri = (bary_tri[:, 0:1] * tri_verts[0]
             + bary_tri[:, 1:2] * tri_verts[1]
             + bary_tri[:, 2:3] * tri_verts[2])  # (Q_t, 3)

    # RWG ρ vectors at triangle quad points
    if sign_rwg == 1:
        rho_src = r_tri - fv_rwg[np.newaxis, :]  # (Q_t, 3)
    else:
        rho_src = fv_rwg[np.newaxis, :] - r_tri  # (Q_t, 3)

    # Distances: R[i,j] = |r_wire[i] - r_tri[j]|
    diff = r_wire[:, np.newaxis, :] - r_tri[np.newaxis, :, :]  # (Q_w, Q_t, 3)
    R_sq = np.sum(diff * diff, axis=-1)  # (Q_w, Q_t)
    # No wire radius for wire-surface (no singularity)
    R = np.sqrt(np.maximum(R_sq, 1e-60))

    g_vals = np.exp(-1j * k * R) / (4.0 * np.pi * R)  # (Q_w, Q_t)

    # Vector potential integral:
    # ∫_wire ∫_tri f_wire(s) · f_rwg(r') g(R) ds dS'
    #
    # f_wire(s) = f_w_scalar * dl * d̂
    # f_rwg(r') = (l_rwg / (2*A_tri)) * ρ_src
    # f_wire · f_rwg = f_w_scalar * dl * (l_rwg/(2*A_tri)) * (d̂ · ρ_src)
    # ds = dl * dt, dS' = twice_area * (triangle quad)
    #
    # I_A = dl² * (l_rwg/(2*A_tri)) * twice_area * ΣΣ w_i w_j f_w[i] (d̂·ρ[j]) g[i,j]

    dot_dir_rho = np.dot(rho_src, dir_seg)  # (Q_t,) — d̂ · ρ_src at each triangle quad point

    # Weighted sum
    # For each wire point i: sum_j w_j * dot_dir_rho[j] * g[i,j]
    inner_A = np.dot(g_vals, w_tri * dot_dir_rho)  # (Q_w,)
    I_A = dl_seg * dl_seg * (l_rwg / (2.0 * A_tri)) * twice_area * np.dot(w_wire * f_w_scalar, inner_A)

    # Scalar potential integral:
    # ∫_wire ∫_tri div(f_wire) div(f_rwg) g(R) ds dS'
    #
    # div(f_wire) = sign_seg / dl (constant)
    # div(f_rwg) = sign_rwg * l_rwg / A_tri (constant)
    # ds = dl * dt, dS' = twice_area * (triangle quad)
    #
    # I_Phi = (sign_seg/dl) * (sign_rwg * l_rwg / A_tri) * dl * twice_area * ΣΣ w_i w_j g[i,j]

    div_wire = float(sign_seg) / dl_seg
    div_rwg = float(sign_rwg) * l_rwg / A_tri

    inner_Phi = np.dot(g_vals, w_tri)  # (Q_w,)
    I_Phi = div_wire * div_rwg * dl_seg * twice_area * np.dot(w_wire, inner_Phi)

    return complex(I_A), complex(I_Phi)


def fill_wire_wire_static(
    wire_basis: WireBasis,
    wire_mesh: WireMesh,
    n_quad: int = 8,
) -> tuple:
    """Compute frequency-independent VP and SP integrals for wire-wire pairs.

    Uses the static Green's function g(R) = 1/(4πR).

    Parameters
    ----------
    wire_basis : WireBasis
    wire_mesh : WireMesh
    n_quad : int
        Number of 1D Gauss-Legendre quadrature points per segment.

    Returns
    -------
    L_raw_ww : ndarray, shape (N_w, N_w), float64
        Vector-potential integral ∫∫ f_m · f_n g(R) dτ dτ' (frequency-independent).
    P_ww : ndarray, shape (N_w, N_w), float64
        Scalar-potential integral ∫∫ div(f_m) div(f_n) g(R) dτ dτ'.
    """
    N_w = wire_basis.num_basis
    w_quad, t_quad = wire_quad_rule(n_quad)

    L_raw = np.zeros((N_w, N_w), dtype=np.float64)
    P_mat = np.zeros((N_w, N_w), dtype=np.float64)

    k_static = 1e-15  # effectively zero
    nodes = wire_mesh.nodes
    segs = wire_mesh.segments

    for m in range(N_w):
        for n in range(m, N_w):
            i_a_total = 0.0
            i_phi_total = 0.0

            m_segs = [
                (wire_basis.seg_plus[m], +1),
                (wire_basis.seg_minus[m], -1),
            ]
            n_segs = [
                (wire_basis.seg_plus[n], +1),
                (wire_basis.seg_minus[n], -1),
            ]

            for si_m, sign_m in m_segs:
                seg_m = segs[si_m]
                for si_n, sign_n in n_segs:
                    seg_n = segs[si_n]
                    I_A, I_Phi = _segment_segment_integral(
                        k_static,
                        nodes[seg_m.node_start], nodes[seg_m.node_end],
                        seg_m.length, seg_m.direction, sign_m,
                        nodes[seg_n.node_start], nodes[seg_n.node_end],
                        seg_n.length, seg_n.direction, sign_n,
                        seg_m.radius,
                        w_quad, t_quad,
                    )
                    i_a_total += I_A.real
                    i_phi_total += I_Phi.real

            L_raw[m, n] = i_a_total
            L_raw[n, m] = i_a_total
            P_mat[m, n] = i_phi_total
            P_mat[n, m] = i_phi_total

    return L_raw, P_mat


def fill_wire_surface_static(
    wire_basis: WireBasis,
    wire_mesh: WireMesh,
    rwg_basis,
    mesh,
    n_quad_wire: int = 8,
    n_quad_tri: int = 4,
) -> tuple:
    """Compute frequency-independent VP and SP integrals for wire-surface pairs.

    Uses the static Green's function g(R) = 1/(4πR).

    Parameters
    ----------
    wire_basis : WireBasis
    wire_mesh : WireMesh
    rwg_basis : RWGBasis
    mesh : Mesh
    n_quad_wire : int
    n_quad_tri : int

    Returns
    -------
    L_raw_ws : ndarray, shape (N_w, N_s), float64
        VP integral (typically ~0 for vertical wire + horizontal strip).
    P_ws : ndarray, shape (N_w, N_s), float64
        SP integral (the dominant coupling term).
    """
    from ..greens.quadrature import triangle_quad_rule

    N_w = wire_basis.num_basis
    N_s = rwg_basis.num_basis

    w_wire, t_wire = wire_quad_rule(n_quad_wire)
    w_tri, bary_tri = triangle_quad_rule(n_quad_tri)

    L_raw = np.zeros((N_w, N_s), dtype=np.float64)
    P_mat = np.zeros((N_w, N_s), dtype=np.float64)

    k_static = 1e-15
    nodes = wire_mesh.nodes
    w_segs = wire_mesh.segments
    verts = mesh.vertices
    tris = mesh.triangles

    for m in range(N_w):
        wire_halves = [
            (wire_basis.seg_plus[m], +1),
            (wire_basis.seg_minus[m], -1),
        ]

        for n in range(N_s):
            i_a_total = 0.0
            i_phi_total = 0.0

            rwg_halves = [
                (rwg_basis.t_plus[n], rwg_basis.free_vertex_plus[n], +1),
                (rwg_basis.t_minus[n], rwg_basis.free_vertex_minus[n], -1),
            ]
            l_rwg = rwg_basis.edge_length[n]

            for si_w, sign_w in wire_halves:
                seg = w_segs[si_w]
                for tri_idx, fv_idx, sign_rwg in rwg_halves:
                    tri_v = verts[tris[tri_idx]]
                    fv = verts[fv_idx]
                    if sign_rwg == 1:
                        A_tri = rwg_basis.area_plus[n]
                    else:
                        A_tri = rwg_basis.area_minus[n]
                    twice_area = 2.0 * A_tri

                    I_A, I_Phi = _segment_triangle_integral(
                        k_static,
                        nodes[seg.node_start], nodes[seg.node_end],
                        seg.length, seg.direction, sign_w,
                        tri_v, fv, sign_rwg, l_rwg, A_tri, seg.radius,
                        w_wire, t_wire,
                        w_tri, bary_tri, twice_area,
                    )
                    i_a_total += I_A.real
                    i_phi_total += I_Phi.real

            L_raw[m, n] = i_a_total
            P_mat[m, n] = i_phi_total

    return L_raw, P_mat


def fill_wire_surface(
    Z_ws: np.ndarray,
    wire_basis: WireBasis,
    wire_mesh: WireMesh,
    rwg_basis,
    mesh,
    k: complex,
    eta: complex,
    n_quad_wire: int = 8,
    n_quad_tri: int = 4,
) -> None:
    """Fill the wire-surface coupling block Z_ws.

    Parameters
    ----------
    Z_ws : ndarray, shape (N_w, N_s), complex128
        Output matrix, filled in-place.
    wire_basis : WireBasis
    wire_mesh : WireMesh
    rwg_basis : RWGBasis
    mesh : Mesh
    k : complex
        Wavenumber.
    eta : complex
        Wave impedance.
    n_quad_wire : int
        1D quadrature points per wire segment.
    n_quad_tri : int
        2D quadrature points per triangle (uses triangle_quad_rule).
    """
    from ..greens.quadrature import triangle_quad_rule

    N_w = wire_basis.num_basis
    N_s = rwg_basis.num_basis

    w_wire, t_wire = wire_quad_rule(n_quad_wire)
    w_tri, bary_tri = triangle_quad_rule(n_quad_tri)

    prefactor_A = 1j * k * eta
    prefactor_Phi = -1j * eta / k

    nodes = wire_mesh.nodes
    w_segs = wire_mesh.segments
    verts = mesh.vertices
    tris = mesh.triangles

    for m in range(N_w):
        # Wire basis m: seg_plus and seg_minus
        wire_halves = [
            (wire_basis.seg_plus[m], +1),
            (wire_basis.seg_minus[m], -1),
        ]

        for n in range(N_s):
            z_mn = 0.0 + 0.0j

            # RWG basis n: T+ and T-
            rwg_halves = [
                (rwg_basis.t_plus[n], rwg_basis.free_vertex_plus[n], +1),
                (rwg_basis.t_minus[n], rwg_basis.free_vertex_minus[n], -1),
            ]

            l_rwg = rwg_basis.edge_length[n]

            for si_w, sign_w in wire_halves:
                seg = w_segs[si_w]
                for tri_idx, fv_idx, sign_rwg in rwg_halves:
                    tri_v = verts[tris[tri_idx]]  # (3, 3)
                    fv = verts[fv_idx]  # (3,)
                    if sign_rwg == 1:
                        A_tri = rwg_basis.area_plus[n]
                    else:
                        A_tri = rwg_basis.area_minus[n]
                    twice_area = 2.0 * A_tri

                    I_A, I_Phi = _segment_triangle_integral(
                        k,
                        nodes[seg.node_start], nodes[seg.node_end],
                        seg.length, seg.direction, sign_w,
                        tri_v, fv, sign_rwg, l_rwg, A_tri, seg.radius,
                        w_wire, t_wire,
                        w_tri, bary_tri, twice_area,
                    )

                    z_mn += prefactor_A * I_A + prefactor_Phi * I_Phi

            Z_ws[m, n] = z_mn
