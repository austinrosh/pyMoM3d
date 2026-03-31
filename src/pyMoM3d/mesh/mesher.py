"""Mesh generation utilities."""

import numpy as np
from typing import Optional

from .mesh_data import Mesh


def create_mesh_from_vertices(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> Mesh:
    """Create a Mesh object from vertices and triangle connectivity.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Vertex coordinates.
    triangles : ndarray, shape (M, 3)
        Triangle connectivity (vertex indices).

    Returns
    -------
    mesh : Mesh
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3)")

    return Mesh(vertices, triangles)


def create_rectangular_mesh(
    width: float,
    height: float,
    nx: int,
    ny: int,
    center: tuple = (0.0, 0.0, 0.0)
) -> Mesh:
    """Create a mesh for a rectangular plate with specified mesh density.

    Parameters
    ----------
    width : float
        Width of the plate along x-axis.
    height : float
        Height of the plate along y-axis.
    nx : int
        Number of vertices along x-axis.
    ny : int
        Number of vertices along y-axis.
    center : tuple of float, optional
        Centre point (x, y, z) of the plate.  Default (0, 0, 0).

    Returns
    -------
    mesh : Mesh
    """
    from ..geometry.primitives import RectangularPlate

    plate = RectangularPlate(width, height, center=center)
    vertices = plate.get_vertex_grid(nx, ny)

    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            idx = i * nx + j
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])

    faces = np.array(faces, dtype=np.int32)
    return Mesh(vertices, faces)
