"""Mesh generation utilities."""

import numpy as np
from scipy.spatial import Delaunay
from typing import Optional

from .mesh_data import Mesh


def create_mesh_from_vertices(
    vertices: np.ndarray,
    triangles: Optional[np.ndarray] = None
) -> Mesh:
    """
    Create a Mesh object from vertices and optional triangle connectivity.
    
    If triangles are not provided, Delaunay triangulation is performed
    on the projection of vertices to the xy-plane.
    
    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Vertex coordinates
    triangles : ndarray, shape (M, 3), optional
        Triangle connectivity. If None, Delaunay triangulation is used.
    
    Returns
    -------
    mesh : Mesh
        Mesh object with computed connectivity
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    
    if vertices.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3)")
    
    if triangles is None:
        # Perform Delaunay triangulation on xy-projection
        # This works well for flat surfaces like rectangular plates
        points_2d = vertices[:, :2]  # Project to xy-plane
        tri = Delaunay(points_2d)
        triangles = tri.simplices
    else:
        triangles = np.asarray(triangles, dtype=np.int32)
    
    return Mesh(vertices, triangles)
