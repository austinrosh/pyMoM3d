"""Mesh generation utilities."""

import numpy as np
import trimesh
from typing import Optional

from .mesh_data import Mesh
from .trimesh_mesher import PythonMesher


def create_mesh_from_vertices(
    vertices: np.ndarray,
    triangles: Optional[np.ndarray] = None
) -> Mesh:
    """
    Create a Mesh object from vertices and optional triangle connectivity.
    
    If triangles are not provided, trimesh is used to create a mesh
    from the point cloud. For flat surfaces, a 2D projection is used.
    
    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Vertex coordinates
    triangles : ndarray, shape (M, 3), optional
        Triangle connectivity. If None, trimesh is used to generate triangles.
    
    Returns
    -------
    mesh : Mesh
        Mesh object with computed connectivity
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    
    if vertices.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3)")
    
    mesher = PythonMesher()
    
    if triangles is None:
        # Use trimesh to create mesh from point cloud
        # For flat surfaces (like plates), we can use a 2D projection
        # Check if vertices are approximately coplanar (e.g., z is constant)
        z_values = vertices[:, 2]
        z_std = np.std(z_values)
        
        if z_std < 1e-10:
            # Flat surface - use trimesh's convex hull for triangulation
            # This works well for arbitrary point sets on a plane
            point_cloud = trimesh.PointCloud(vertices)
            mesh_obj = point_cloud.convex_hull
            
            # If convex hull doesn't work or gives poor results,
            # we can try other trimesh methods
            if mesh_obj is None or len(mesh_obj.faces) == 0:
                # Fallback: create a simple mesh by detecting grid pattern
                # For regular grids, we can triangulate directly
                # This is a heuristic - may not work for all cases
                # In practice, users should provide triangles for complex cases
                raise ValueError(
                    "Could not automatically triangulate flat surface. "
                    "Please provide triangles explicitly or use a primitive's to_trimesh() method."
                )
        else:
            # 3D surface - use trimesh's surface reconstruction
            # For now, use convex hull as a simple method
            # In production, you might want to use more sophisticated methods
            point_cloud = trimesh.PointCloud(vertices)
            mesh_obj = point_cloud.convex_hull
    else:
        triangles = np.asarray(triangles, dtype=np.int32)
        mesh_obj = trimesh.Trimesh(vertices=vertices, faces=triangles)
    
    return mesher.mesh_from_geometry(mesh_obj)


def create_rectangular_mesh(
    width: float,
    height: float,
    nx: int,
    ny: int,
    center: tuple = (0.0, 0.0, 0.0)
) -> Mesh:
    """
    Create a mesh for a rectangular plate with specified mesh density.
    
    Convenience function that combines geometry creation, vertex grid
    generation, and mesh creation using trimesh.
    
    Parameters
    ----------
    width : float
        Width of the plate along x-axis
    height : float
        Height of the plate along y-axis
    nx : int
        Number of points along x-axis (controls mesh density)
    ny : int
        Number of points along y-axis (controls mesh density)
    center : tuple of float, optional
        Center point (x, y, z) of the plate. Default is (0, 0, 0).
    
    Returns
    -------
    mesh : Mesh
        Mesh object with computed connectivity
    """
    from ..geometry.primitives import RectangularPlate
    
    plate = RectangularPlate(width, height, center=center)
    
    # Use trimesh-based approach: create grid and triangulate
    vertices = plate.get_vertex_grid(nx, ny)
    
    # Create regular grid triangulation
    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            # Two triangles per quad
            idx = i * nx + j
            faces.append([idx, idx + 1, idx + nx])
            faces.append([idx + 1, idx + nx + 1, idx + nx])
    
    faces = np.array(faces, dtype=np.int32)
    
    mesher = PythonMesher()
    return mesher.mesh_from_geometry(vertices, triangles=faces)
