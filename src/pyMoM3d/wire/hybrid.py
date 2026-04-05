"""Hybrid wire-surface MoM assembly.

Builds the block-structured impedance matrix for combined wire and surface
basis functions:

    Z = [ Z_ss   Z_sw ]
        [ Z_ws   Z_ww ]

Z_ss is assembled by the existing ``fill_matrix()`` for surface-surface
interactions.  Z_ww and Z_ws are computed by the wire kernels.  Z_sw = Z_ws^T
by EFIE reciprocity.

Junction Enforcement
--------------------
Wire rooftop basis functions go to zero at the wire endpoints, so no current
flows from the wire tip to the strip mesh by default.  To enforce current
continuity at wire-surface junctions, the ``junctions`` parameter in
``HybridBasis`` maps wire tip nodes to surface mesh vertices.  At each
junction, a **junction basis function** is created: it combines the last
wire segment's half-rooftop with the RWG attachment mode on the surface
(charge distributed uniformly over the strip triangles at the junction
vertex).  This is the standard Rao (1980) / Rao-Wilton (1991) junction
treatment for hybrid wire-surface MoM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from .wire_basis import WireBasis, WireMesh
from .kernels import fill_wire_wire, fill_wire_surface


@dataclass
class WireSurfaceJunction:
    """Describes a junction between a wire tip and a surface mesh vertex.

    Parameters
    ----------
    wire_segment_idx : int
        Index of the wire segment whose endpoint touches the surface.
    wire_node_idx : int
        Index of the wire node at the junction (tip of wire).
    surface_vertex_idx : int
        Index of the surface mesh vertex at the junction.
    is_wire_start : bool
        If True, the junction is at the start of the segment (node_start).
        If False, at the end (node_end).
    """

    wire_segment_idx: int
    wire_node_idx: int
    surface_vertex_idx: int
    is_wire_start: bool = False


@dataclass
class HybridBasis:
    """Combined surface (RWG) + wire (rooftop) basis.

    The global index space is [0..N_s-1] for surface basis functions
    and [N_s..N_s+N_w-1] for wire basis functions.
    If junctions are specified, [N_s+N_w..N_s+N_w+N_j-1] for junction basis.

    Parameters
    ----------
    rwg_basis : RWGBasis
        Surface basis functions.
    wire_basis : WireBasis
        Wire basis functions.
    wire_mesh : WireMesh
        Wire geometry.
    junctions : list of WireSurfaceJunction, optional
        Wire-surface junction definitions.  Each junction creates one
        additional DOF connecting the wire tip to the surface mesh.
    """

    rwg_basis: object  # RWGBasis (avoid circular import)
    wire_basis: WireBasis
    wire_mesh: WireMesh
    junctions: List[WireSurfaceJunction] = field(default_factory=list)

    @property
    def num_surface(self) -> int:
        return self.rwg_basis.num_basis

    @property
    def num_wire(self) -> int:
        return self.wire_basis.num_basis

    @property
    def num_junctions(self) -> int:
        return len(self.junctions)

    @property
    def num_total(self) -> int:
        return self.num_surface + self.num_wire + self.num_junctions


class HybridBasisAdapter:
    """Duck-types as RWGBasis for Port / NetworkExtractor compatibility.

    Concatenates ``edge_length`` from the RWG and wire bases so that
    ``Port.build_excitation_vector()`` and ``Port.terminal_current()``
    work unchanged with global indices.

    The "edge length" for a wire basis function is the average of its
    two segment lengths, analogous to the RWG edge length.

    Parameters
    ----------
    hybrid_basis : HybridBasis
    """

    def __init__(self, hybrid_basis: HybridBasis):
        self._hb = hybrid_basis
        self.num_basis = hybrid_basis.num_total

        # Wire effective edge lengths
        wb = hybrid_basis.wire_basis
        wire_lengths = 0.5 * (wb.length_plus + wb.length_minus)

        parts = [hybrid_basis.rwg_basis.edge_length, wire_lengths]

        # Junction basis: effective edge length = last wire segment length
        for junc in hybrid_basis.junctions:
            seg = hybrid_basis.wire_mesh.segments[junc.wire_segment_idx]
            parts.append(np.array([seg.length]))

        self.edge_length = np.concatenate(parts)


def detect_junctions(
    wire_mesh: WireMesh,
    mesh,
    tol: float = 1e-8,
) -> List[WireSurfaceJunction]:
    """Auto-detect wire-surface junctions by proximity.

    For each wire endpoint, check if it coincides with a surface mesh
    vertex.  If so, create a junction.

    Parameters
    ----------
    wire_mesh : WireMesh
        Wire geometry.
    mesh : Mesh
        Surface mesh.
    tol : float
        Distance tolerance for coincidence.

    Returns
    -------
    list of WireSurfaceJunction
    """
    junctions = []
    verts = mesh.vertices

    for si, seg in enumerate(wire_mesh.segments):
        # Check segment start
        start_pos = wire_mesh.nodes[seg.node_start]
        dists = np.linalg.norm(verts - start_pos[np.newaxis, :], axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] < tol:
            # Check if this is a wire endpoint (not an interior node)
            is_endpoint = True
            if si > 0:
                prev_seg = wire_mesh.segments[si - 1]
                if prev_seg.node_end == seg.node_start:
                    is_endpoint = False
            if is_endpoint:
                junctions.append(WireSurfaceJunction(
                    wire_segment_idx=si,
                    wire_node_idx=seg.node_start,
                    surface_vertex_idx=idx,
                    is_wire_start=True,
                ))

        # Check segment end
        end_pos = wire_mesh.nodes[seg.node_end]
        dists = np.linalg.norm(verts - end_pos[np.newaxis, :], axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] < tol:
            # Check if this is a wire endpoint (not an interior node)
            is_endpoint = True
            if si < wire_mesh.num_segments - 1:
                next_seg = wire_mesh.segments[si + 1]
                if next_seg.node_start == seg.node_end:
                    is_endpoint = False
            if is_endpoint:
                junctions.append(WireSurfaceJunction(
                    wire_segment_idx=si,
                    wire_node_idx=seg.node_end,
                    surface_vertex_idx=idx,
                    is_wire_start=False,
                ))

    return junctions


def fill_hybrid_matrix(
    operator,
    rwg_basis,
    mesh,
    wire_basis: WireBasis,
    wire_mesh: WireMesh,
    k: complex,
    eta: complex,
    quad_order: int = 4,
    near_threshold: float = 0.2,
    backend: str = 'auto',
    n_quad_wire: int = 8,
    n_quad_tri: int = 4,
    junctions: Optional[List[WireSurfaceJunction]] = None,
) -> np.ndarray:
    """Assemble the full hybrid (surface + wire) impedance matrix.

    When ``junctions`` is provided, junction basis functions are added to
    enforce current continuity between wire tips and the surface mesh.
    Each junction DOF combines the wire tip's half-rooftop with an
    attachment mode on the adjacent surface triangles (uniform charge
    distribution), following Rao-Wilton (1991).

    Parameters
    ----------
    operator : AbstractOperator
        Operator for the Z_ss (surface-surface) block.
    rwg_basis : RWGBasis
        Surface basis functions.
    mesh : Mesh
        Surface mesh.
    wire_basis : WireBasis
        Wire basis functions.
    wire_mesh : WireMesh
        Wire geometry.
    k : complex
        Wavenumber (rad/m).
    eta : complex
        Wave impedance (Ohm).
    quad_order : int
        Quadrature order for surface-surface integrals.
    near_threshold : float
        Near-field threshold for surface-surface singularity extraction.
    backend : str
        Backend for surface-surface assembly ('auto', 'cpp', 'numpy').
    n_quad_wire : int
        1D quadrature points per wire segment.
    n_quad_tri : int
        2D quadrature points per triangle for wire-surface coupling.
    junctions : list of WireSurfaceJunction, optional
        Wire-surface junction definitions.

    Returns
    -------
    Z : ndarray, shape (N_s + N_w + N_j, N_s + N_w + N_j), complex128
        Full hybrid impedance matrix with junction enforcement.
    """
    from ..mom.assembly import fill_matrix

    N_s = rwg_basis.num_basis
    N_w = wire_basis.num_basis
    N_j = len(junctions) if junctions else 0
    N = N_s + N_w + N_j

    Z = np.zeros((N, N), dtype=np.complex128)

    # Block (0,0): surface-surface — existing assembly
    Z[:N_s, :N_s] = fill_matrix(
        operator, rwg_basis, mesh, k, eta,
        quad_order=quad_order,
        near_threshold=near_threshold,
        backend=backend,
    )

    # Block (1,1): wire-wire
    fill_wire_wire(
        Z[N_s:N_s+N_w, N_s:N_s+N_w],
        wire_basis, wire_mesh, k, eta,
        n_quad=n_quad_wire,
    )

    # Block (1,0): wire-surface coupling
    Z_ws = np.zeros((N_w, N_s), dtype=np.complex128)
    fill_wire_surface(
        Z_ws,
        wire_basis, wire_mesh,
        rwg_basis, mesh,
        k, eta,
        n_quad_wire=n_quad_wire,
        n_quad_tri=n_quad_tri,
    )
    Z[N_s:N_s+N_w, :N_s] = Z_ws

    # Block (0,1): Z_sw = Z_ws^T (EFIE reciprocity)
    Z[:N_s, N_s:N_s+N_w] = Z_ws.T

    # --- Junction basis functions ---
    if junctions:
        _fill_junction_blocks(
            Z, junctions, rwg_basis, mesh, wire_basis, wire_mesh,
            k, eta, N_s, N_w, n_quad_wire, n_quad_tri, operator,
        )

    return Z


def _get_attachment_info(vi, mesh, tol=1e-6):
    """Find attachment triangles for a junction vertex.

    Returns (attach_tris, A_total) where attach_tris is a list of
    triangle indices sharing vertex vi on the strip (same z-plane),
    and A_total is their total area.
    """
    verts = mesh.vertices
    tris = mesh.triangles
    T = len(tris)
    z_strip = verts[vi, 2]

    attach_tris = []
    for t in range(T):
        if vi in tris[t]:
            z_c = verts[tris[t], 2].mean()
            if abs(z_c - z_strip) < tol:
                attach_tris.append(t)

    if not attach_tris:
        attach_tris = [t for t in range(T) if vi in tris[t]]

    A_total = float(sum(mesh.triangle_areas[t] for t in attach_tris))
    return attach_tris, A_total


def _junction_wire_params(junc, wire_mesh):
    """Extract wire segment parameters for a junction's half-rooftop.

    Returns (node_a, node_b, dl, direction, sign_wire) where the
    half-rooftop is on the segment from node_a to node_b.
    """
    seg = wire_mesh.segments[junc.wire_segment_idx]
    node_a = wire_mesh.nodes[seg.node_start]
    node_b = wire_mesh.nodes[seg.node_end]
    dl = seg.length
    direction = seg.direction
    # sign = +1 means rooftop rises toward node_end (junction at end)
    # sign = -1 means rooftop falls toward node_start (junction at start)
    sign_wire = -1 if junc.is_wire_start else +1
    return node_a, node_b, dl, direction, sign_wire


def _integrate_green_layered(
    k, r_obs, v0, v1, v2, quad_order, gf_backend=None,
):
    """Integrate layered-medium Green's function over a source triangle.

    Uses singularity extraction for the free-space 1/(4πR) part plus a
    smooth correction from the layered GF backend (if available).

    G_ML(r,r') = G_fs(r,r') + [G_ML(r,r') - G_fs(r,r')]

    Parameters
    ----------
    k : complex
        Wavenumber.
    r_obs : ndarray (3,)
        Observation point.
    v0, v1, v2 : ndarray (3,)
        Source triangle vertices.
    quad_order : int
        Quadrature order.
    gf_backend : GreensBackend, optional
        Layered GF backend for smooth correction. If None, free-space only.

    Returns
    -------
    complex
        ∫_T G_ML(r_obs, r') dS'
    """
    from ..greens.quadrature import triangle_quad_rule
    from ..greens.singularity import integrate_green_singular

    # Free-space part with singularity extraction
    I_fs = integrate_green_singular(
        k.real, r_obs, v0, v1, v2, quad_order=quad_order,
    )

    if gf_backend is None:
        return I_fs

    # Smooth correction: ∫_T [G_ML - G_fs](r_obs, r') dS'
    w, bary = triangle_quad_rule(quad_order)
    cross = np.cross(v1 - v0, v2 - v0)
    twice_area = np.linalg.norm(cross)

    r_obs_arr = np.asarray(r_obs, dtype=np.float64)
    I_smooth = 0.0 + 0.0j
    for i in range(len(w)):
        r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
        correction = gf_backend.scalar_G(
            r_obs_arr[np.newaxis, :],
            np.asarray(r_prime, dtype=np.float64)[np.newaxis, :],
        )
        I_smooth += w[i] * complex(correction.ravel()[0])

    I_smooth *= twice_area
    return I_fs + I_smooth


def _fill_junction_blocks(
    Z, junctions, rwg_basis, mesh, wire_basis, wire_mesh,
    k, eta, N_s, N_w, n_quad_wire, n_quad_tri, operator=None,
):
    """Fill the junction rows/columns of the hybrid impedance matrix.

    Each junction basis function j has two parts:

    1. **Wire half:** half-rooftop on the tip segment (VP + SP).
       Current flows from wire interior toward the junction vertex.
       div_wire = sign/dl on the segment.

    2. **Surface half:** attachment mode on strip triangles at the
       junction vertex (SP only, no in-plane VP current).
       div_surface = 1/A_total on each attachment triangle.
       This distributes charge uniformly, ensuring current continuity
       at the junction: wire charge (+1) is balanced by attachment
       charge flowing into the surrounding RWG functions.

    The junction DOF's coupling to every other DOF (surface, wire,
    junction) includes contributions from BOTH halves.
    """
    from ..greens.quadrature import triangle_quad_rule
    from .kernels import (
        wire_quad_rule, _segment_segment_integral,
        _segment_triangle_integral,
    )

    N_j = len(junctions)
    verts = mesh.vertices
    tris = mesh.triangles

    prefactor_A = 1j * k * eta
    prefactor_Phi = -1j * eta / k

    # Extract layered GF backend for surface-half SP coupling
    gf_backend = None
    if operator is not None and hasattr(operator, '_gf'):
        gf = operator._gf
        if hasattr(gf, 'backend'):
            gf_backend = gf.backend

    w_wire, t_wire = wire_quad_rule(n_quad_wire)
    w_tri, bary_tri = triangle_quad_rule(n_quad_tri)

    # Pre-compute attachment info and wire params for each junction
    junc_info = []
    for junc in junctions:
        seg = wire_mesh.segments[junc.wire_segment_idx]
        attach_tris, A_total = _get_attachment_info(
            junc.surface_vertex_idx, mesh,
        )
        node_a, node_b, dl_j, dir_j, sign_j = _junction_wire_params(
            junc, wire_mesh,
        )
        junc_info.append({
            'seg': seg,
            'attach_tris': attach_tris,
            'A_total': A_total,
            'node_a': node_a, 'node_b': node_b,
            'dl': dl_j, 'dir': dir_j, 'sign': sign_j,
        })

    for ji, junc in enumerate(junctions):
        j_col = N_s + N_w + ji
        info = junc_info[ji]
        seg = info['seg']
        attach_tris = info['attach_tris']
        A_total = info['A_total']
        node_a, node_b = info['node_a'], info['node_b']
        dl_j, dir_j, sign_j = info['dl'], info['dir'], info['sign']
        div_j_surface = 1.0 / A_total

        # ---------------------------------------------------------------
        # Junction-to-surface coupling (Z[j, 0:N_s] and Z[0:N_s, j])
        # ---------------------------------------------------------------
        # Two contributions:
        # A) Wire half (VP + SP) via _segment_triangle_integral
        # B) Surface half (SP only) — attachment mode divergence ×
        #    RWG divergence × g(R), using singularity extraction for
        #    overlapping/adjacent triangle pairs.

        for n in range(rwg_basis.num_basis):
            z_jn = 0.0 + 0.0j
            l_rwg = rwg_basis.edge_length[n]

            # --- A) Wire half VP + SP ---
            rwg_halves = [
                (rwg_basis.t_plus[n], rwg_basis.free_vertex_plus[n], +1),
                (rwg_basis.t_minus[n], rwg_basis.free_vertex_minus[n], -1),
            ]
            for tri_idx, fv_idx, sign_rwg in rwg_halves:
                tri_v = verts[tris[tri_idx]]
                fv = verts[fv_idx]
                A_tri = (rwg_basis.area_plus[n] if sign_rwg == 1
                         else rwg_basis.area_minus[n])
                twice_area = 2.0 * A_tri

                I_A, I_Phi = _segment_triangle_integral(
                    k,
                    node_a, node_b, dl_j, dir_j, sign_j,
                    tri_v, fv, sign_rwg, l_rwg, A_tri, seg.radius,
                    w_wire, t_wire, w_tri, bary_tri, twice_area,
                )
                z_jn += prefactor_A * I_A + prefactor_Phi * I_Phi

            # --- B) Surface half SP (attachment mode) ---
            # div_j = 1/A_total on each attachment triangle
            # div_n = ±l/A on T+/T- of RWG n
            # Uses singularity extraction for the inner integral to
            # handle overlapping/adjacent triangle pairs correctly.
            for t_idx in attach_tris:
                tri_v_j = verts[tris[t_idx]]
                A_j = mesh.triangle_areas[t_idx]
                twice_A_j = 2.0 * A_j

                # Outer quadrature over the attachment triangle
                for iq_j in range(len(w_tri)):
                    r_obs = (bary_tri[iq_j, 0] * tri_v_j[0]
                             + bary_tri[iq_j, 1] * tri_v_j[1]
                             + bary_tri[iq_j, 2] * tri_v_j[2])

                    # Inner integral over each RWG half-triangle
                    # using singularity extraction
                    for t_rwg, sign_rwg in [(rwg_basis.t_plus[n], +1),
                                            (rwg_basis.t_minus[n], -1)]:
                        tri_v_n = verts[tris[t_rwg]]
                        A_n = (rwg_basis.area_plus[n] if sign_rwg == 1
                               else rwg_basis.area_minus[n])
                        div_n = sign_rwg * l_rwg / A_n

                        # ∫ g(r_obs, r') dS' with singularity extraction
                        I_g = integrate_green_singular(
                            k.real, r_obs,
                            tri_v_n[0], tri_v_n[1], tri_v_n[2],
                            quad_order=n_quad_tri,
                        )

                        z_jn += (prefactor_Phi * div_j_surface * div_n
                                 * twice_A_j * w_tri[iq_j] * I_g)

            Z[j_col, n] = z_jn
            Z[n, j_col] = z_jn  # reciprocity

        # ---------------------------------------------------------------
        # Junction-to-wire coupling (Z[j, N_s:N_s+N_w] and reverse)
        # ---------------------------------------------------------------
        # A) Wire half VP+SP coupling to interior wire basis (seg-seg)
        # B) Surface half SP coupling: attachment mode div × wire div

        for m in range(wire_basis.num_basis):
            z_jm = 0.0 + 0.0j

            # --- A) Wire half VP+SP ---
            for si_m, sign_m in [(wire_basis.seg_plus[m], +1),
                                 (wire_basis.seg_minus[m], -1)]:
                seg_m = wire_mesh.segments[si_m]
                I_A, I_Phi = _segment_segment_integral(
                    k,
                    node_a, node_b, dl_j, dir_j, sign_j,
                    wire_mesh.nodes[seg_m.node_start],
                    wire_mesh.nodes[seg_m.node_end],
                    seg_m.length, seg_m.direction, sign_m,
                    seg.radius,
                    w_wire, t_wire,
                )
                z_jm += prefactor_A * I_A + prefactor_Phi * I_Phi

            # --- B) Surface half SP ---
            # No singularity issue (wire and surface are disjoint)
            for si_m, sign_m in [(wire_basis.seg_plus[m], +1),
                                 (wire_basis.seg_minus[m], -1)]:
                seg_m = wire_mesh.segments[si_m]
                div_m = float(sign_m) / seg_m.length

                r_wire_m = (wire_mesh.nodes[seg_m.node_start][np.newaxis, :]
                            + t_wire[:, np.newaxis] * seg_m.length
                            * seg_m.direction[np.newaxis, :])

                for t_idx in attach_tris:
                    tri_v = verts[tris[t_idx]]
                    r_tri = (bary_tri[:, 0:1] * tri_v[0]
                             + bary_tri[:, 1:2] * tri_v[1]
                             + bary_tri[:, 2:3] * tri_v[2])
                    A_t = mesh.triangle_areas[t_idx]

                    diff = r_wire_m[:, np.newaxis, :] - r_tri[np.newaxis, :, :]
                    R = np.sqrt(np.maximum(np.sum(diff**2, axis=-1), 1e-60))
                    g_vals = np.exp(-1j * k * R) / (4.0 * np.pi * R)
                    integral = seg_m.length * float(2*A_t) * np.sum(
                        w_wire[:, np.newaxis] * w_tri[np.newaxis, :] * g_vals
                    )
                    z_jm += prefactor_Phi * div_j_surface * div_m * integral

            Z[j_col, N_s + m] = z_jm
            Z[N_s + m, j_col] = z_jm  # reciprocity

        # ---------------------------------------------------------------
        # Junction-to-junction coupling (Z[j, j'] and self)
        # ---------------------------------------------------------------
        for ji2 in range(ji, N_j):
            junc2 = junctions[ji2]
            j_col2 = N_s + N_w + ji2
            info2 = junc_info[ji2]
            seg2 = info2['seg']
            node_a2, node_b2 = info2['node_a'], info2['node_b']
            dl_j2, dir_j2, sign_j2 = info2['dl'], info2['dir'], info2['sign']
            attach_tris2 = info2['attach_tris']
            A_total2 = info2['A_total']
            div_j2_surface = 1.0 / A_total2

            z_jj = 0.0 + 0.0j

            # Wire-half × wire-half VP+SP
            I_A, I_Phi = _segment_segment_integral(
                k,
                node_a, node_b, dl_j, dir_j, sign_j,
                node_a2, node_b2, dl_j2, dir_j2, sign_j2,
                seg.radius,
                w_wire, t_wire,
            )
            z_jj += prefactor_A * I_A + prefactor_Phi * I_Phi

            # Wire-half(j) × surface-half(j2) SP
            div_wire_j = float(sign_j) / dl_j
            for t2 in attach_tris2:
                tri_v2 = verts[tris[t2]]
                A2 = mesh.triangle_areas[t2]
                # Observation points on wire segment j
                r_wire_j = (node_a[np.newaxis, :]
                            + t_wire[:, np.newaxis] * dl_j
                            * dir_j[np.newaxis, :])
                r_tri2 = (bary_tri[:, 0:1] * tri_v2[0]
                          + bary_tri[:, 1:2] * tri_v2[1]
                          + bary_tri[:, 2:3] * tri_v2[2])
                diff = r_wire_j[:, np.newaxis, :] - r_tri2[np.newaxis, :, :]
                R = np.sqrt(np.maximum(np.sum(diff**2, axis=-1), 1e-60))
                g_vals = np.exp(-1j * k * R) / (4.0 * np.pi * R)
                integral = dl_j * float(2*A2) * np.sum(
                    w_wire[:, np.newaxis] * w_tri[np.newaxis, :] * g_vals
                )
                z_jj += prefactor_Phi * div_wire_j * div_j2_surface * integral

            # Surface-half(j) × wire-half(j2) SP
            div_wire_j2 = float(sign_j2) / dl_j2
            for t1 in attach_tris:
                tri_v1 = verts[tris[t1]]
                A1 = mesh.triangle_areas[t1]
                r_tri1 = (bary_tri[:, 0:1] * tri_v1[0]
                          + bary_tri[:, 1:2] * tri_v1[1]
                          + bary_tri[:, 2:3] * tri_v1[2])
                r_wire_j2 = (node_a2[np.newaxis, :]
                             + t_wire[:, np.newaxis] * dl_j2
                             * dir_j2[np.newaxis, :])
                diff = r_tri1[:, np.newaxis, :] - r_wire_j2[np.newaxis, :, :]
                R = np.sqrt(np.maximum(np.sum(diff**2, axis=-1), 1e-60))
                g_vals = np.exp(-1j * k * R) / (4.0 * np.pi * R)
                integral = float(2*A1) * dl_j2 * np.sum(
                    w_tri[:, np.newaxis] * w_wire[np.newaxis, :] * g_vals
                )
                z_jj += prefactor_Phi * div_j_surface * div_wire_j2 * integral

            # Surface-half × surface-half SP (with singularity extraction)
            for t1 in attach_tris:
                tri_v1 = verts[tris[t1]]
                A1 = mesh.triangle_areas[t1]
                twice_A1 = 2.0 * A1
                for iq in range(len(w_tri)):
                    r_obs = (bary_tri[iq, 0] * tri_v1[0]
                             + bary_tri[iq, 1] * tri_v1[1]
                             + bary_tri[iq, 2] * tri_v1[2])
                    for t2 in attach_tris2:
                        tri_v2 = verts[tris[t2]]
                        I_g = integrate_green_singular(
                            k.real, r_obs,
                            tri_v2[0], tri_v2[1], tri_v2[2],
                            quad_order=n_quad_tri,
                        )
                        z_jj += (prefactor_Phi * div_j_surface
                                 * div_j2_surface
                                 * twice_A1 * w_tri[iq] * I_g)

            Z[j_col, j_col2] = z_jj
            if ji2 != ji:
                Z[j_col2, j_col] = z_jj  # reciprocity
