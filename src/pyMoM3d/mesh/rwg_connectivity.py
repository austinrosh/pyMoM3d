"""RWG basis function connectivity computation."""

import warnings
import numpy as np

from .mesh_data import Mesh
from .rwg_basis import RWGBasis


def compute_rwg_connectivity(mesh: Mesh) -> RWGBasis:
    """
    Compute RWG basis functions from mesh connectivity.

    Each interior edge (shared by exactly two triangles) produces one RWG
    basis function.  Boundary edges are counted but excluded from the basis
    set.

    The result is stored on ``mesh.rwg_basis`` and the legacy
    ``mesh.rwg_pairs`` is also updated for backward compatibility.

    Parameters
    ----------
    mesh : Mesh
        Mesh with computed edges and edge_to_triangles.

    Returns
    -------
    rwg_basis : RWGBasis
        Enriched RWG basis data.
    """
    edge_indices = []
    edge_lengths = []
    t_plus_list = []
    t_minus_list = []
    free_vertex_plus_list = []
    free_vertex_minus_list = []
    area_plus_list = []
    area_minus_list = []

    rwg_pairs = []  # legacy
    num_boundary = 0

    for edge_idx in range(len(mesh.edges)):
        triangles = mesh.edge_to_triangles[edge_idx]

        if len(triangles) == 1:
            # Boundary edge
            num_boundary += 1
            rwg_pairs.append([triangles[0], -1])
            continue

        if len(triangles) != 2:
            raise ValueError(
                f"Edge {edge_idx} has {len(triangles)} triangles (expected 1 or 2)"
            )

        # Interior edge — create basis function
        edge = mesh.edges[edge_idx]
        ev = set(int(v) for v in edge)

        t1, t2 = triangles[0], triangles[1]

        # Identify free vertices (the vertex in each triangle NOT on the shared edge)
        tri1_verts = set(int(v) for v in mesh.triangles[t1])
        tri2_verts = set(int(v) for v in mesh.triangles[t2])
        free1 = (tri1_verts - ev).pop()
        free2 = (tri2_verts - ev).pop()

        # Determine T+ and T-:
        # In T+, the edge traversal in the triangle's winding goes v_a -> v_b
        # where (v_a, v_b) are the shared edge vertices in the triangle's
        # cyclic order.  T+ is defined so that the current flows away from
        # the free vertex.
        #
        # Convention: look at the cyclic order of vertices in t1.
        # If the shared edge appears as (..., v_a, v_b, free1, ...) in
        # cyclic order, then the RWG vector rho = r - r_free points away
        # from free1 across the edge, which is the T+ definition.
        #
        # We assign T+ = t1 when the edge (v0, v1) appears in the cyclic
        # order of t1 (i.e., v0 immediately precedes v1).
        tri1 = mesh.triangles[t1]
        idx_v0 = int(np.where(tri1 == edge[0])[0][0])
        idx_v1 = int(np.where(tri1 == edge[1])[0][0])

        if (idx_v0 + 1) % 3 == idx_v1:
            # Edge goes v0->v1 in t1's winding => t1 is T+
            tp, tm = t1, t2
            fvp, fvm = free1, free2
        else:
            # Edge goes v1->v0 in t1's winding => t1 is T-
            tp, tm = t2, t1
            fvp, fvm = free2, free1

        edge_indices.append(edge_idx)
        edge_lengths.append(float(mesh.edge_lengths[edge_idx]))
        t_plus_list.append(tp)
        t_minus_list.append(tm)
        free_vertex_plus_list.append(fvp)
        free_vertex_minus_list.append(fvm)
        area_plus_list.append(float(mesh.triangle_areas[tp]))
        area_minus_list.append(float(mesh.triangle_areas[tm]))

        rwg_pairs.append([tp, tm])

    num_basis = len(edge_indices)

    rwg_basis = RWGBasis(
        num_basis=num_basis,
        edge_index=np.array(edge_indices, dtype=np.int32),
        edge_length=np.array(edge_lengths, dtype=np.float64),
        t_plus=np.array(t_plus_list, dtype=np.int32),
        t_minus=np.array(t_minus_list, dtype=np.int32),
        free_vertex_plus=np.array(free_vertex_plus_list, dtype=np.int32),
        free_vertex_minus=np.array(free_vertex_minus_list, dtype=np.int32),
        area_plus=np.array(area_plus_list, dtype=np.float64),
        area_minus=np.array(area_minus_list, dtype=np.float64),
        num_boundary_edges=num_boundary,
    )

    # Store on mesh
    mesh.rwg_basis = rwg_basis

    # Legacy: store rwg_pairs (interior pairs first, then boundary)
    mesh.rwg_pairs = np.array(rwg_pairs, dtype=np.int32) if rwg_pairs else None

    return rwg_basis
