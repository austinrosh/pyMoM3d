"""Mesh mirroring utilities for SOC de-embedding.

Mirrors a triangular mesh about a plane, used to create the symmetric
calibration structures required by Short-Open Calibration (SOC).

References
----------
[1] Okhmatovski et al., "On Deembedding of Port Discontinuities in
    Full-Wave CAD Models of Multiport Circuits," IEEE Trans. MTT, 2003.
"""

from __future__ import annotations

import numpy as np

from .mesh_data import Mesh
from .rwg_connectivity import compute_rwg_connectivity


def mirror_mesh_x(mesh: Mesh, x_plane: float) -> Mesh:
    """Mirror a mesh about a plane perpendicular to the x-axis.

    Creates a new mesh that is the reflection of the input mesh
    about the plane x = x_plane.  Triangle winding order is reversed
    so that normals remain consistently oriented.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    x_plane : float
        x-coordinate of the mirror plane.

    Returns
    -------
    mirrored : Mesh
        New mesh with reflected vertices and reversed winding.
    """
    verts = mesh.vertices.copy()
    verts[:, 0] = 2.0 * x_plane - verts[:, 0]

    # Reverse winding to keep normals consistent after reflection
    tris = mesh.triangles[:, ::-1].copy()

    return Mesh(vertices=verts, triangles=tris)


def combine_meshes(mesh_a: Mesh, mesh_b: Mesh, merge_tol: float = 1e-10) -> Mesh:
    """Combine two meshes, merging coincident vertices.

    Vertices closer than *merge_tol* are identified and shared.
    This is essential for creating a connected mesh at the mirror plane
    so that interior RWG edges form across the seam.

    Parameters
    ----------
    mesh_a, mesh_b : Mesh
        Two meshes to combine.
    merge_tol : float
        Distance below which two vertices are considered identical.

    Returns
    -------
    combined : Mesh
        Unified mesh with shared vertices at the seam.
    """
    n_a = len(mesh_a.vertices)

    # Build mapping: for each vertex in mesh_b, either map to an existing
    # mesh_a vertex (if close enough) or create a new vertex.
    verts_list = list(mesh_a.vertices)
    b_to_combined = np.empty(len(mesh_b.vertices), dtype=np.int32)

    for i, vb in enumerate(mesh_b.vertices):
        # Check distance to all mesh_a vertices
        dists = np.linalg.norm(mesh_a.vertices - vb, axis=1)
        j_min = np.argmin(dists)
        if dists[j_min] < merge_tol:
            b_to_combined[i] = j_min
        else:
            b_to_combined[i] = len(verts_list)
            verts_list.append(vb)

    combined_verts = np.array(verts_list, dtype=np.float64)

    # Remap mesh_b triangles
    tris_b_remapped = b_to_combined[mesh_b.triangles]
    combined_tris = np.vstack([mesh_a.triangles, tris_b_remapped])

    return Mesh(vertices=combined_verts, triangles=combined_tris)


def extract_submesh(mesh: Mesh, x_min: float, x_max: float,
                    tol: float = 1e-10) -> tuple[Mesh, np.ndarray]:
    """Extract triangles whose centroids lie within an x-range.

    Parameters
    ----------
    mesh : Mesh
        Source mesh.
    x_min, x_max : float
        x-coordinate bounds (inclusive).
    tol : float
        Tolerance added to the bounds.

    Returns
    -------
    submesh : Mesh
        Extracted submesh with reindexed vertices.
    tri_mask : ndarray of bool, shape (N_t,)
        Boolean mask indicating which original triangles were kept.
    """
    centroids_x = mesh.vertices[mesh.triangles].mean(axis=1)[:, 0]
    mask = (centroids_x >= x_min - tol) & (centroids_x <= x_max + tol)
    kept_tris = mesh.triangles[mask]

    # Reindex vertices
    unique_verts, inverse = np.unique(kept_tris.ravel(), return_inverse=True)
    new_verts = mesh.vertices[unique_verts]
    new_tris = inverse.reshape(-1, 3).astype(np.int32)

    return Mesh(vertices=new_verts, triangles=new_tris), mask
