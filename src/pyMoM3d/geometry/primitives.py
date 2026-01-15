"""Programmatic geometry generation primitives."""

import numpy as np
from typing import Tuple


class RectangularPlate:
    """
    Generate a rectangular plate geometry.
    
    The plate is defined in the xy-plane (z=0) with specified dimensions.
    Vertices are ordered counter-clockwise when viewed from +z direction.
    
    Parameters
    ----------
    width : float
        Width of the plate along x-axis
    height : float
        Height of the plate along y-axis
    center : tuple of float, optional
        Center point (x, y, z) of the plate. Default is (0, 0, 0).
    """
    
    def __init__(self, width: float, height: float, center: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        
        self.width = width
        self.height = height
        self.center = np.array(center, dtype=np.float64)
    
    def get_vertices(self) -> np.ndarray:
        """
        Get the four corner vertices of the rectangular plate.
        
        Returns
        -------
        vertices : ndarray, shape (4, 3)
            Array of vertex coordinates. Vertices are ordered:
            [bottom-left, bottom-right, top-right, top-left]
        """
        w = self.width / 2.0
        h = self.height / 2.0
        
        # Define vertices in local coordinates (centered at origin)
        local_vertices = np.array([
            [-w, -h, 0.0],  # bottom-left
            [ w, -h, 0.0],  # bottom-right
            [ w,  h, 0.0],  # top-right
            [-w,  h, 0.0],  # top-left
        ], dtype=np.float64)
        
        # Translate to center position
        vertices = local_vertices + self.center
        
        return vertices
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the axis-aligned bounding box of the plate.
        
        Returns
        -------
        min_corner : ndarray, shape (3,)
            Minimum corner coordinates (x_min, y_min, z_min)
        max_corner : ndarray, shape (3,)
            Maximum corner coordinates (x_max, y_max, z_max)
        """
        vertices = self.get_vertices()
        min_corner = np.min(vertices, axis=0)
        max_corner = np.max(vertices, axis=0)
        return min_corner, max_corner
