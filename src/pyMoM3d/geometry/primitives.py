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
    
    def get_vertex_grid(self, nx: int, ny: int) -> np.ndarray:
        """
        Get a refined grid of vertices for mesh generation.
        
        Parameters
        ----------
        nx : int
            Number of points along x-axis (width direction)
        ny : int
            Number of points along y-axis (height direction)
        
        Returns
        -------
        vertices : ndarray, shape (nx * ny, 3)
            Grid of vertex coordinates covering the plate
        """
        if nx < 2 or ny < 2:
            raise ValueError("nx and ny must be at least 2")
        
        w = self.width / 2.0
        h = self.height / 2.0
        
        # Generate grid in local coordinates
        x = np.linspace(-w, w, nx)
        y = np.linspace(-h, h, ny)
        X, Y = np.meshgrid(x, y)
        
        # Flatten and add z-coordinate
        vertices = np.column_stack([
            X.ravel(),
            Y.ravel(),
            np.zeros(nx * ny)
        ])
        
        # Translate to center position
        vertices = vertices + self.center
        
        return vertices.astype(np.float64)
    


class Sphere:
    """
    Generate a sphere geometry.
    
    Parameters
    ----------
    radius : float
        Radius of the sphere
    center : tuple of float, optional
        Center point (x, y, z) of the sphere. Default is (0, 0, 0).
    """
    
    def __init__(self, radius: float, center: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        if radius <= 0:
            raise ValueError("Radius must be positive")
        
        self.radius = radius
        self.center = np.array(center, dtype=np.float64)
    
    def get_vertices(self, n_theta: int = 20, n_phi: int = 10) -> np.ndarray:
        """
        Get vertices on the sphere surface using spherical coordinates.
        
        Parameters
        ----------
        n_theta : int, default 20
            Number of points in azimuthal direction (longitude)
        n_phi : int, default 10
            Number of points in polar direction (latitude)
        
        Returns
        -------
        vertices : ndarray, shape (n_theta * n_phi, 3)
            Vertex coordinates on sphere surface
        """
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)  # Azimuthal
        phi = np.linspace(0, np.pi, n_phi)  # Polar (0 to pi)
        
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Convert to Cartesian
        x = self.radius * np.sin(PHI) * np.cos(THETA)
        y = self.radius * np.sin(PHI) * np.sin(THETA)
        z = self.radius * np.cos(PHI)
        
        vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        vertices = vertices + self.center
        
        return vertices.astype(np.float64)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        r = self.radius
        min_corner = self.center - np.array([r, r, r])
        max_corner = self.center + np.array([r, r, r])
        return min_corner, max_corner
    


class Cylinder:
    """
    Generate a cylinder geometry (open-ended, no caps).
    
    Parameters
    ----------
    radius : float
        Radius of the cylinder
    height : float
        Height of the cylinder along z-axis
    center : tuple of float, optional
        Center point (x, y, z) of the cylinder. Default is (0, 0, 0).
    """
    
    def __init__(self, radius: float, height: float, center: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        if radius <= 0 or height <= 0:
            raise ValueError("Radius and height must be positive")
        
        self.radius = radius
        self.height = height
        self.center = np.array(center, dtype=np.float64)
    
    def get_vertices(self, n_theta: int = 20, n_z: int = 10) -> np.ndarray:
        """
        Get vertices on the cylinder surface.
        
        Parameters
        ----------
        n_theta : int, default 20
            Number of points around the circumference
        n_z : int, default 10
            Number of points along the height
        
        Returns
        -------
        vertices : ndarray, shape (n_theta * n_z, 3)
            Vertex coordinates on cylinder surface
        """
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        z = np.linspace(-self.height / 2.0, self.height / 2.0, n_z)
        
        THETA, Z = np.meshgrid(theta, z)
        
        x = self.radius * np.cos(THETA)
        y = self.radius * np.sin(THETA)
        
        vertices = np.column_stack([x.ravel(), y.ravel(), Z.ravel()])
        vertices = vertices + self.center
        
        return vertices.astype(np.float64)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        r = self.radius
        h = self.height / 2.0
        min_corner = self.center + np.array([-r, -r, -h])
        max_corner = self.center + np.array([r, r, h])
        return min_corner, max_corner
    


class Cube:
    """
    Generate a cube geometry.
    
    Parameters
    ----------
    side_length : float
        Length of each side
    center : tuple of float, optional
        Center point (x, y, z) of the cube. Default is (0, 0, 0).
    """
    
    def __init__(self, side_length: float, center: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        if side_length <= 0:
            raise ValueError("Side length must be positive")
        
        self.side_length = side_length
        self.center = np.array(center, dtype=np.float64)
    
    def get_vertices(self) -> np.ndarray:
        """
        Get the 8 corner vertices of the cube.
        
        Returns
        -------
        vertices : ndarray, shape (8, 3)
            Corner vertex coordinates
        """
        s = self.side_length / 2.0
        
        # All 8 corners of a cube
        vertices = np.array([
            [-s, -s, -s],  # 0: bottom-left-back
            [ s, -s, -s],  # 1: bottom-right-back
            [ s,  s, -s],  # 2: top-right-back
            [-s,  s, -s],  # 3: top-left-back
            [-s, -s,  s],  # 4: bottom-left-front
            [ s, -s,  s],  # 5: bottom-right-front
            [ s,  s,  s],  # 6: top-right-front
            [-s,  s,  s],  # 7: top-left-front
        ], dtype=np.float64)
        
        vertices = vertices + self.center
        return vertices
    
    def get_vertex_grid(self, nx: int, ny: int, nz: int) -> np.ndarray:
        """
        Get a refined grid of vertices on the cube surface.
        
        Parameters
        ----------
        nx : int
            Points along x-axis per face
        ny : int
            Points along y-axis per face
        nz : int
            Points along z-axis per face
        
        Returns
        -------
        vertices : ndarray, shape (N, 3)
            Grid of vertex coordinates on cube surface
        """
        if nx < 2 or ny < 2 or nz < 2:
            raise ValueError("nx, ny, nz must be at least 2")
        
        s = self.side_length / 2.0
        vertices_list = []
        
        # Generate vertices for each of the 6 faces
        # Face 1: z = -s (back face)
        x = np.linspace(-s, s, nx)
        y = np.linspace(-s, s, ny)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, -s)
        vertices_list.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        
        # Face 2: z = s (front face)
        Z = np.full_like(X, s)
        vertices_list.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        
        # Face 3: y = -s (bottom face)
        x = np.linspace(-s, s, nx)
        z = np.linspace(-s, s, nz)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, -s)
        vertices_list.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        
        # Face 4: y = s (top face)
        Y = np.full_like(X, s)
        vertices_list.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        
        # Face 5: x = -s (left face)
        y = np.linspace(-s, s, ny)
        z = np.linspace(-s, s, nz)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, -s)
        vertices_list.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        
        # Face 6: x = s (right face)
        X = np.full_like(Y, s)
        vertices_list.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        
        # Combine all faces
        vertices = np.vstack(vertices_list)
        
        # Translate to center
        vertices = vertices + self.center
        
        return vertices.astype(np.float64)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        s = self.side_length / 2.0
        min_corner = self.center - s
        max_corner = self.center + s
        return min_corner, max_corner
    


class Pyramid:
    """
    Generate a square pyramid geometry.
    
    Parameters
    ----------
    base_size : float
        Size of the square base
    height : float
        Height of the pyramid
    center : tuple of float, optional
        Center point (x, y, z) of the base. Default is (0, 0, 0).
    """
    
    def __init__(self, base_size: float, height: float, center: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        if base_size <= 0 or height <= 0:
            raise ValueError("Base size and height must be positive")
        
        self.base_size = base_size
        self.height = height
        self.center = np.array(center, dtype=np.float64)
    
    def get_vertices(self) -> np.ndarray:
        """
        Get the 5 vertices of the pyramid (4 base corners + 1 apex).
        
        Returns
        -------
        vertices : ndarray, shape (5, 3)
            Vertex coordinates
        """
        s = self.base_size / 2.0
        h = self.height
        
        # Base vertices (z = 0)
        base_vertices = np.array([
            [-s, -s, 0.0],  # bottom-left
            [ s, -s, 0.0],  # bottom-right
            [ s,  s, 0.0],  # top-right
            [-s,  s, 0.0],  # top-left
        ], dtype=np.float64)
        
        # Apex (at center of base, height h above)
        apex = np.array([0.0, 0.0, h], dtype=np.float64)
        
        vertices = np.vstack([base_vertices, apex])
        vertices = vertices + self.center
        
        return vertices
    
    def get_vertex_grid(self, n_base: int = 10, n_height: int = 5) -> np.ndarray:
        """
        Get a refined grid of vertices on the pyramid surface.
        
        Parameters
        ----------
        n_base : int, default 10
            Number of points along each base edge
        n_height : int, default 5
            Number of points along height (for triangular faces)
        
        Returns
        -------
        vertices : ndarray, shape (N, 3)
            Grid of vertex coordinates on pyramid surface
        """
        if n_base < 2 or n_height < 2:
            raise ValueError("n_base and n_height must be at least 2")
        
        s = self.base_size / 2.0
        h = self.height
        vertices_list = []
        
        # Base face (z = 0)
        x = np.linspace(-s, s, n_base)
        y = np.linspace(-s, s, n_base)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        vertices_list.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        
        # Four triangular faces
        # For each face, create a grid from base edge to apex
        base_edges = [
            (np.linspace(-s, s, n_base), -s, 0),  # bottom edge
            (s, np.linspace(-s, s, n_base), 0),   # right edge
            (np.linspace(s, -s, n_base), s, 0),    # top edge
            (-s, np.linspace(s, -s, n_base), 0),  # left edge
        ]
        
        for edge_x, edge_y, edge_z in base_edges:
            # Create grid from edge to apex
            if isinstance(edge_x, np.ndarray):
                # Horizontal edge
                for i, x_val in enumerate(edge_x):
                    y_val = edge_y
                    z_vals = np.linspace(edge_z, h, n_height)
                    x_vals = np.full(n_height, x_val)
                    y_vals = np.full(n_height, y_val)
                    vertices_list.append(np.column_stack([x_vals, y_vals, z_vals]))
            else:
                # Vertical edge
                for i, y_val in enumerate(edge_y):
                    x_val = edge_x
                    z_vals = np.linspace(edge_z, h, n_height)
                    x_vals = np.full(n_height, x_val)
                    y_vals = np.full(n_height, y_val)
                    vertices_list.append(np.column_stack([x_vals, y_vals, z_vals]))
        
        vertices = np.vstack(vertices_list)
        vertices = vertices + self.center
        
        return vertices.astype(np.float64)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        s = self.base_size / 2.0
        min_corner = self.center + np.array([-s, -s, 0.0])
        max_corner = self.center + np.array([s, s, self.height])
        return min_corner, max_corner
    
