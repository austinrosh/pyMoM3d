"""Port mesh utilities for non-radiating (half-RWG) port gaps.

Provides functions to create transverse gaps in a surface mesh and
identify matched boundary edge pairs for half-RWG basis functions.

The non-radiating port model (Liu et al. 2018) places a narrow gap
across the conductor mesh at each port location.  No current flows in
the gap — instead, half-RWG basis functions bridge the gap with
constrained current continuity.  This eliminates the spurious port
radiation artifact of standard delta-gap models.

References
----------
Liu et al., "A Nonradiating Finite-Gap Lumped-Port Model for Full-Wave
EM Solvers," IEEE AWPL, vol. 17, no. 7, July 2018.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .mesh_data import Mesh
from .rwg_basis import RWGBasis


@dataclass
class PortCut:
    """Metadata for a single port gap cut in the mesh.

    Parameters
    ----------
    x : float
        x-coordinate of the cut.
    left_boundary_edges : list of tuple
        (edge_idx, tri_idx) pairs for boundary edges on the left side.
    right_boundary_edges : list of tuple
        (edge_idx, tri_idx) pairs for boundary edges on the right side.
    half_rwg_pairs : list of tuple
        (left_edge_idx, left_tri, right_edge_idx, right_tri) for matched
        boundary edge pairs.
    """

    x: float
    left_boundary_edges: List[Tuple[int, int]] = field(default_factory=list)
    right_boundary_edges: List[Tuple[int, int]] = field(default_factory=list)
    half_rwg_pairs: List[Tuple[int, int, int, int]] = field(default_factory=list)


def split_mesh_at_x(
    mesh: Mesh,
    x_split: float,
    tol: float = None,
) -> Tuple[Mesh, dict]:
    """Split a mesh at x = x_split by duplicating shared vertices.

    Creates a transverse gap by duplicating vertices at x_split so that
    triangles on opposite sides no longer share vertices.  The mesh must
    have conformal edges at x_split (use ``feed_x_list`` in meshing).

    Parameters
    ----------
    mesh : Mesh
        Input mesh with conformal transverse edges at x = x_split.
    x_split : float
        x-coordinate of the gap.
    tol : float, optional
        Tolerance for vertex identification.  Default: 1% of mean edge
        length.

    Returns
    -------
    new_mesh : Mesh
        Mesh with duplicated vertices at the gap.
    vertex_remap : dict
        Mapping from original vertex index to new (right-side) vertex
        index for vertices on the split line.
    """
    vertices = mesh.vertices.copy()
    triangles = mesh.triangles.copy()

    if tol is None:
        tol = 0.01 * float(np.mean(mesh.edge_lengths))

    # Find vertices on the split line
    on_split = np.abs(vertices[:, 0] - x_split) < tol
    split_indices = np.where(on_split)[0]

    if len(split_indices) == 0:
        raise ValueError(
            f"split_mesh_at_x: no vertices found at x = {x_split} "
            f"(tol = {tol:.4g}).  Ensure the mesh has conformal edges "
            f"at this coordinate."
        )

    # Classify triangles as left or right based on centroid
    tri_centroids_x = vertices[triangles].mean(axis=1)[:, 0]
    is_right = tri_centroids_x > x_split + tol * 0.1

    # For each split vertex, create a duplicate for right-side triangles
    vertex_remap = {}
    new_verts = [vertices]
    next_idx = len(vertices)

    for vi in split_indices:
        new_verts.append(vertices[vi:vi + 1].copy())
        vertex_remap[int(vi)] = next_idx
        next_idx += 1

    new_vertices = np.vstack(new_verts)

    # Remap right-side triangles to use duplicated vertices
    new_triangles = triangles.copy()
    for ti in range(len(triangles)):
        if is_right[ti]:
            for j in range(3):
                vi = int(triangles[ti, j])
                if vi in vertex_remap:
                    new_triangles[ti, j] = vertex_remap[vi]

    new_mesh = Mesh(new_vertices, new_triangles)
    return new_mesh, vertex_remap


def split_mesh_at_ports(
    mesh: Mesh,
    port_x_list: List[float],
    tol: float = None,
) -> Tuple[Mesh, List[dict]]:
    """Split a mesh at multiple port x-coordinates.

    Applies :func:`split_mesh_at_x` sequentially for each port location.

    Parameters
    ----------
    mesh : Mesh
        Input mesh with conformal edges at each port x-coordinate.
    port_x_list : list of float
        x-coordinates of port gaps.
    tol : float, optional
        Tolerance for vertex identification.

    Returns
    -------
    new_mesh : Mesh
        Mesh with gaps at all port locations.
    remaps : list of dict
        One vertex_remap per port, in the same order as port_x_list.
    """
    current_mesh = mesh
    remaps = []
    for x in sorted(port_x_list):
        current_mesh, remap = split_mesh_at_x(current_mesh, x, tol=tol)
        remaps.append(remap)
    return current_mesh, remaps


def find_half_rwg_pairs(
    mesh: Mesh,
    rwg_basis: RWGBasis,
    port_x: float,
    tol: float = None,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Find matched boundary edge pairs at a port gap for half-RWG creation.

    After splitting, each side of the gap has boundary edges (edges with
    only one adjacent triangle).  This function matches boundary edges by
    geometric position and returns the data needed to create half-RWG
    basis functions.

    Parameters
    ----------
    mesh : Mesh
        Mesh that has been split at x = port_x.
    rwg_basis : RWGBasis
        Standard RWG basis (interior edges only).
    port_x : float
        x-coordinate of the port gap.
    tol : float, optional
        Tolerance for edge matching.  Default: 1% of mean edge length.

    Returns
    -------
    pairs : list of (left_edge_idx, left_tri, left_free_vertex,
                      right_edge_idx, right_tri, right_free_vertex)
        Each entry describes one matched pair of boundary edges suitable
        for creating a half-RWG basis function.
    """
    if tol is None:
        tol = 0.01 * float(np.mean(mesh.edge_lengths))

    # Find boundary edges at x ≈ port_x
    left_edges = []   # (edge_idx, tri_idx, y_mid)
    right_edges = []  # (edge_idx, tri_idx, y_mid)

    for edge_idx, tri_list in mesh.edge_to_triangles.items():
        if len(tri_list) != 1:
            continue  # not a boundary edge

        edge = mesh.edges[edge_idx]
        va = mesh.vertices[edge[0]]
        vb = mesh.vertices[edge[1]]

        # Both vertices must be at x ≈ port_x
        if abs(va[0] - port_x) > tol or abs(vb[0] - port_x) > tol:
            continue

        # Must be approximately y-directed (transverse)
        edge_dir = vb - va
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-30:
            continue
        edge_dir_n = edge_dir / edge_len
        if abs(edge_dir_n[1]) < abs(edge_dir_n[0]):
            continue  # more longitudinal than transverse

        tri_idx = tri_list[0]
        tri_centroid_x = float(mesh.vertices[mesh.triangles[tri_idx]].mean(axis=0)[0])
        y_mid = 0.5 * (va[1] + vb[1])

        if tri_centroid_x < port_x:
            left_edges.append((edge_idx, tri_idx, y_mid, edge_len))
        else:
            right_edges.append((edge_idx, tri_idx, y_mid, edge_len))

    # Match left and right edges by y-midpoint and edge length
    pairs = []
    used_right = set()

    for le_idx, le_tri, le_y, le_len in left_edges:
        best_match = None
        best_dist = float('inf')

        for ri, (re_idx, re_tri, re_y, re_len) in enumerate(right_edges):
            if ri in used_right:
                continue
            # Check y-midpoint proximity and edge length match
            dy = abs(le_y - re_y)
            dl = abs(le_len - re_len)
            if dy < tol and dl < tol * 0.1:
                dist = dy + dl
                if dist < best_dist:
                    best_dist = dist
                    best_match = ri

        if best_match is not None:
            used_right.add(best_match)
            re_idx, re_tri, _, _ = right_edges[best_match]

            # Find free vertices (vertex in triangle NOT on the boundary edge)
            le_edge_verts = set(int(v) for v in mesh.edges[le_idx])
            le_tri_verts = set(int(v) for v in mesh.triangles[le_tri])
            le_free = (le_tri_verts - le_edge_verts).pop()

            re_edge_verts = set(int(v) for v in mesh.edges[re_idx])
            re_tri_verts = set(int(v) for v in mesh.triangles[re_tri])
            re_free = (re_tri_verts - re_edge_verts).pop()

            pairs.append((le_idx, le_tri, le_free, re_idx, re_tri, re_free))

    return pairs


def add_half_rwg_basis(
    mesh: Mesh,
    rwg_basis: RWGBasis,
    port_x: float,
    tol: float = None,
) -> Tuple[RWGBasis, List[int]]:
    """Extend an RWGBasis with half-RWG basis functions at a port gap.

    Finds matched boundary edge pairs at x = port_x and appends
    half-RWG entries to the existing RWGBasis arrays.  The half-RWG
    basis functions have the same mathematical form as standard RWGs
    but with T+ and T- on opposite sides of the gap.

    The T+/T- assignment follows the standard RWG convention: current
    flows in the +x direction across the gap (from left to right).

    Parameters
    ----------
    mesh : Mesh
        Mesh that has been split at x = port_x.
    rwg_basis : RWGBasis
        Standard RWG basis (interior edges only).
    port_x : float
        x-coordinate of the port gap.
    tol : float, optional
        Tolerance for edge matching.

    Returns
    -------
    extended_basis : RWGBasis
        New RWGBasis with half-RWG entries appended.
    half_rwg_indices : list of int
        Basis function indices of the new half-RWG entries.
    """
    pairs = find_half_rwg_pairs(mesh, rwg_basis, port_x, tol=tol)

    if not pairs:
        raise ValueError(
            f"add_half_rwg_basis: no matched boundary edge pairs found "
            f"at x = {port_x}.  Ensure the mesh was split at this "
            f"coordinate with split_mesh_at_x()."
        )

    n_old = rwg_basis.num_basis
    n_new = len(pairs)
    n_total = n_old + n_new

    # Preallocate extended arrays
    edge_index = np.empty(n_total, dtype=np.int32)
    edge_length = np.empty(n_total, dtype=np.float64)
    t_plus = np.empty(n_total, dtype=np.int32)
    t_minus = np.empty(n_total, dtype=np.int32)
    free_vertex_plus = np.empty(n_total, dtype=np.int32)
    free_vertex_minus = np.empty(n_total, dtype=np.int32)
    area_plus = np.empty(n_total, dtype=np.float64)
    area_minus = np.empty(n_total, dtype=np.float64)

    # Copy existing basis data
    edge_index[:n_old] = rwg_basis.edge_index
    edge_length[:n_old] = rwg_basis.edge_length
    t_plus[:n_old] = rwg_basis.t_plus
    t_minus[:n_old] = rwg_basis.t_minus
    free_vertex_plus[:n_old] = rwg_basis.free_vertex_plus
    free_vertex_minus[:n_old] = rwg_basis.free_vertex_minus
    area_plus[:n_old] = rwg_basis.area_plus
    area_minus[:n_old] = rwg_basis.area_minus

    half_rwg_indices = []

    for i, (le_idx, le_tri, le_free, re_idx, re_tri, re_free) in enumerate(pairs):
        idx = n_old + i
        half_rwg_indices.append(idx)

        # Determine T+/T- based on current direction convention.
        # For the half-RWG, we want current flowing from left to right
        # (in the +x direction across the gap).
        #
        # On T+ (left triangle): current flows AWAY from free vertex
        # toward the gap edge.  The free vertex is to the left of the
        # gap edge, so current flows rightward (+x).  This is correct
        # for T+.
        #
        # On T- (right triangle): current flows from the gap edge
        # TOWARD the free vertex.  The free vertex is to the right of
        # the gap edge, so current flows rightward (+x).  Correct for T-.
        #
        # Check: if free_vertex_plus is to the LEFT of the gap, then
        # rho+ = r - r_free points rightward (toward the gap).  Good.
        le_free_x = mesh.vertices[le_free, 0]
        re_free_x = mesh.vertices[re_free, 0]

        if le_free_x < re_free_x:
            # Left free vertex is further left → left triangle is T+
            t_plus[idx] = le_tri
            t_minus[idx] = re_tri
            free_vertex_plus[idx] = le_free
            free_vertex_minus[idx] = re_free
            edge_index[idx] = le_idx
        else:
            # Right free vertex is further left → right triangle is T+
            t_plus[idx] = re_tri
            t_minus[idx] = le_tri
            free_vertex_plus[idx] = re_free
            free_vertex_minus[idx] = le_free
            edge_index[idx] = re_idx

        edge_length[idx] = mesh.edge_lengths[le_idx]
        area_plus[idx] = mesh.triangle_areas[t_plus[idx]]
        area_minus[idx] = mesh.triangle_areas[t_minus[idx]]

    extended_basis = RWGBasis(
        num_basis=n_total,
        edge_index=edge_index,
        edge_length=edge_length,
        t_plus=t_plus,
        t_minus=t_minus,
        free_vertex_plus=free_vertex_plus,
        free_vertex_minus=free_vertex_minus,
        area_plus=area_plus,
        area_minus=area_minus,
        num_boundary_edges=rwg_basis.num_boundary_edges,
    )

    return extended_basis, half_rwg_indices
