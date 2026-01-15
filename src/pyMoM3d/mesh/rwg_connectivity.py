"""RWG basis function connectivity computation."""

import numpy as np
from typing import Tuple

from .mesh_data import Mesh


def compute_rwg_connectivity(mesh: Mesh) -> np.ndarray:
    """
    Compute RWG basis function pairs from mesh connectivity.
    
    RWG (Rao-Wilton-Glisson) basis functions are defined on pairs of
    triangles sharing a common edge. Each interior edge gives rise to
    one basis function. Boundary edges (edges belonging to only one
    triangle) also give rise to basis functions.
    
    Parameters
    ----------
    mesh : Mesh
        Mesh object with computed edges and edge_to_triangles mapping
    
    Returns
    -------
    rwg_pairs : ndarray, shape (N_basis, 2)
        Array where each row [t1, t2] represents a basis function.
        t1 and t2 are triangle indices. For boundary edges, t2 = -1.
        The basis function is defined with positive direction from t1 to t2.
    """
    rwg_pairs = []
    
    for edge_idx, edge in enumerate(mesh.edges):
        triangles = mesh.edge_to_triangles[edge_idx]
        
        if len(triangles) == 2:
            # Interior edge: two triangles share this edge
            t1, t2 = triangles[0], triangles[1]
            
            # Determine orientation: ensure edge direction is consistent
            # RWG basis function direction: from t1 to t2
            # We need to check edge orientation in each triangle
            edge_orientation = _get_edge_orientation(mesh, edge_idx, t1, t2)
            
            if edge_orientation:
                rwg_pairs.append([t1, t2])
            else:
                rwg_pairs.append([t2, t1])
        
        elif len(triangles) == 1:
            # Boundary edge: only one triangle
            t1 = triangles[0]
            rwg_pairs.append([t1, -1])  # -1 indicates boundary
        
        else:
            # Should not happen for a valid mesh
            raise ValueError(f"Edge {edge_idx} has {len(triangles)} triangles (expected 1 or 2)")
    
    rwg_pairs = np.array(rwg_pairs, dtype=np.int32)
    
    # Update mesh with computed RWG pairs
    mesh.rwg_pairs = rwg_pairs
    
    return rwg_pairs


def _get_edge_orientation(mesh: Mesh, edge_idx: int, t1: int, t2: int) -> bool:
    """
    Determine the orientation of an edge relative to two triangles.
    
    For RWG basis functions, we need to ensure consistent orientation:
    - In triangle t1, the edge should point from first vertex to second
    - In triangle t2, the edge should point from second vertex to first
    - This ensures the basis function flows from t1 to t2
    
    Parameters
    ----------
    mesh : Mesh
        Mesh object
    edge_idx : int
        Index of the edge
    t1 : int
        First triangle index
    t2 : int
        Second triangle index
    
    Returns
    -------
    orientation : bool
        True if edge orientation in t1 matches mesh edge direction,
        False otherwise
    """
    edge = mesh.edges[edge_idx]
    v0, v1 = edge[0], edge[1]
    
    tri1 = mesh.triangles[t1]
    tri2 = mesh.triangles[t2]
    
    # Find position of edge vertices in triangle t1
    idx_v0_t1 = np.where(tri1 == v0)[0][0]
    idx_v1_t1 = np.where(tri1 == v1)[0][0]
    
    # Check if vertices are consecutive in triangle (cyclic)
    # If v0 comes before v1 in the triangle's vertex order, orientation is positive
    next_idx = (idx_v0_t1 + 1) % 3
    if next_idx == idx_v1_t1:
        # v0 -> v1 is the natural order in triangle t1
        return True
    else:
        # v1 -> v0 is the natural order in triangle t1
        return False
