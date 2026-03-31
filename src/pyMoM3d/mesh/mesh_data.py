"""Core mesh data structures."""

import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple


class Mesh:
    """
    Triangular surface mesh with RWG basis function connectivity.
    
    This class stores all geometric and topological information needed
    for Method of Moments (MoM) computations with RWG basis functions.
    
    Attributes
    ----------
    vertices : ndarray, shape (N_v, 3)
        Vertex coordinates. Each row is a 3D point.
    triangles : ndarray, shape (N_t, 3)
        Triangle connectivity. Each row contains indices into vertices array.
    edges : ndarray, shape (N_e, 2)
        Edge connectivity. Each row contains vertex indices for an edge.
    rwg_pairs : ndarray, shape (N_basis, 2)
        RWG basis function pairs. Each row contains triangle indices.
        For boundary edges, the second index is -1.
    edge_to_triangles : dict
        Mapping from edge index to list of triangle indices sharing that edge.
    triangle_normals : ndarray, shape (N_t, 3)
        Outward-pointing normal vectors for each triangle.
    triangle_areas : ndarray, shape (N_t,)
        Area of each triangle.
    edge_lengths : ndarray, shape (N_e,)
        Length of each edge.
    """
    
    def __init__(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        edges: Optional[np.ndarray] = None,
        rwg_pairs: Optional[np.ndarray] = None,
        edge_to_triangles: Optional[Dict[int, List[int]]] = None,
        triangle_normals: Optional[np.ndarray] = None,
        triangle_areas: Optional[np.ndarray] = None,
        edge_lengths: Optional[np.ndarray] = None
    ):
        """
        Initialize a Mesh object.
        
        Parameters
        ----------
        vertices : ndarray, shape (N_v, 3)
            Vertex coordinates
        triangles : ndarray, shape (N_t, 3)
            Triangle connectivity (vertex indices)
        edges : ndarray, shape (N_e, 2), optional
            Edge connectivity. Computed if not provided.
        rwg_pairs : ndarray, shape (N_basis, 2), optional
            RWG basis function pairs. Computed if not provided.
        edge_to_triangles : dict, optional
            Edge to triangles mapping. Computed if not provided.
        triangle_normals : ndarray, shape (N_t, 3), optional
            Triangle normal vectors. Computed if not provided.
        triangle_areas : ndarray, shape (N_t,), optional
            Triangle areas. Computed if not provided.
        edge_lengths : ndarray, shape (N_e,), optional
            Edge lengths. Computed if not provided.
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int32)
        
        if self.vertices.shape[1] != 3:
            raise ValueError("Vertices must have shape (N, 3)")
        if self.triangles.shape[1] != 3:
            raise ValueError("Triangles must have shape (N, 3)")
        if np.any(self.triangles < 0) or np.any(self.triangles >= len(self.vertices)):
            raise ValueError("Triangle indices out of bounds")
        
        # Compute derived quantities if not provided
        if triangle_normals is None:
            self.triangle_normals = self._compute_triangle_normals()
        else:
            self.triangle_normals = np.asarray(triangle_normals, dtype=np.float64)
        
        if triangle_areas is None:
            self.triangle_areas = self._compute_triangle_areas()
        else:
            self.triangle_areas = np.asarray(triangle_areas, dtype=np.float64)
        
        if edges is None:
            self.edges, self.edge_to_triangles = self._compute_edges()
        else:
            self.edges = np.asarray(edges, dtype=np.int32)
            if edge_to_triangles is None:
                _, self.edge_to_triangles = self._compute_edges()
            else:
                self.edge_to_triangles = edge_to_triangles
        
        if edge_lengths is None:
            self.edge_lengths = self._compute_edge_lengths()
        else:
            self.edge_lengths = np.asarray(edge_lengths, dtype=np.float64)
        
        if rwg_pairs is None:
            # Will be computed by RWG connectivity module
            self.rwg_pairs = None
        else:
            self.rwg_pairs = np.asarray(rwg_pairs, dtype=np.int32)

        # Will be populated by compute_rwg_connectivity()
        self.rwg_basis = None
    
    def _compute_triangle_normals(self) -> np.ndarray:
        """Compute outward-pointing normal vectors for each triangle."""
        normals = np.zeros((len(self.triangles), 3), dtype=np.float64)
        
        for i, tri in enumerate(self.triangles):
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            
            # Two edge vectors
            e1 = v1 - v0
            e2 = v2 - v0
            
            # Cross product gives normal (not normalized)
            normal = np.cross(e1, e2)
            
            # Normalize
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normals[i] = normal / norm
            else:
                # Degenerate triangle
                normals[i] = [0, 0, 1]  # Default to z-direction
        
        return normals
    
    def _compute_triangle_areas(self) -> np.ndarray:
        """Compute area of each triangle."""
        areas = np.zeros(len(self.triangles), dtype=np.float64)
        
        for i, tri in enumerate(self.triangles):
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            
            # Area = 0.5 * ||(v1 - v0) × (v2 - v0)||
            e1 = v1 - v0
            e2 = v2 - v0
            cross = np.cross(e1, e2)
            areas[i] = 0.5 * np.linalg.norm(cross)
        
        return areas
    
    def _compute_edges(self) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """
        Compute edge connectivity and edge-to-triangle mapping.
        
        Returns
        -------
        edges : ndarray, shape (N_e, 2)
            Unique edges (sorted by vertex indices)
        edge_to_triangles : dict
            Mapping from edge index to list of triangle indices
        """
        edge_set = set()
        edge_to_triangles = {}
        
        # Collect all edges from triangles
        for tri_idx, tri in enumerate(self.triangles):
            # Three edges per triangle
            edges_in_tri = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]]))
            ]
            
            for edge in edges_in_tri:
                edge_set.add(edge)
                # Track which triangles share this edge
                # We'll convert to index later
                if edge not in edge_to_triangles:
                    edge_to_triangles[edge] = []
                edge_to_triangles[edge].append(tri_idx)
        
        # Convert to sorted array
        edges_list = sorted(edge_set)
        edges = np.array(edges_list, dtype=np.int32)
        
        # Convert edge_to_triangles to use indices
        edge_to_triangles_idx = {}
        for edge_idx, edge in enumerate(edges_list):
            edge_to_triangles_idx[edge_idx] = edge_to_triangles[edge]
        
        return edges, edge_to_triangles_idx
    
    def _compute_edge_lengths(self) -> np.ndarray:
        """Compute length of each edge."""
        lengths = np.zeros(len(self.edges), dtype=np.float64)
        
        for i, edge in enumerate(self.edges):
            v0 = self.vertices[edge[0]]
            v1 = self.vertices[edge[1]]
            lengths[i] = np.linalg.norm(v1 - v0)
        
        return lengths
    
    def get_num_vertices(self) -> int:
        """Return number of vertices."""
        return len(self.vertices)
    
    def get_num_triangles(self) -> int:
        """Return number of triangles."""
        return len(self.triangles)
    
    def get_num_edges(self) -> int:
        """Return number of edges."""
        return len(self.edges)
    
    def get_num_basis_functions(self) -> int:
        """Return number of RWG basis functions."""
        if self.rwg_pairs is None:
            return 0
        return len(self.rwg_pairs)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get mesh statistics.
        
        Returns
        -------
        stats : dict
            Dictionary containing mesh statistics
        """
        stats = {
            'num_vertices': self.get_num_vertices(),
            'num_triangles': self.get_num_triangles(),
            'num_edges': self.get_num_edges(),
            'num_basis_functions': self.get_num_basis_functions(),
            'min_triangle_area': np.min(self.triangle_areas),
            'max_triangle_area': np.max(self.triangle_areas),
            'mean_triangle_area': np.mean(self.triangle_areas),
            'min_edge_length': np.min(self.edge_lengths),
            'max_edge_length': np.max(self.edge_lengths),
            'mean_edge_length': np.mean(self.edge_lengths),
        }
        return stats
    
    def check_density(self, frequency: float) -> bool:
        """Check if mesh density is adequate for the given frequency.

        Warns if the mean edge length exceeds lambda/10.

        Parameters
        ----------
        frequency : float
            Operating frequency in Hz.

        Returns
        -------
        adequate : bool
            True if mesh density is sufficient.
        """
        from ..utils.constants import c0

        wavelength = c0 / frequency
        mean_edge = float(np.mean(self.edge_lengths))
        threshold = wavelength / 10.0

        if mean_edge > threshold:
            warnings.warn(
                f"Mesh too coarse: mean edge length {mean_edge:.4g} m "
                f"> lambda/10 = {threshold:.4g} m at {frequency:.4g} Hz. "
                f"Consider refining the mesh.",
                stacklevel=2,
            )
            return False
        return True

    def validate(self) -> Dict[str, any]:
        """
        Validate mesh topology and geometry.
        
        Returns
        -------
        validation_results : dict
            Dictionary containing validation results with keys:
            - 'is_valid': bool, overall validity
            - 'has_duplicate_vertices': bool
            - 'num_duplicate_vertices': int
            - 'has_degenerate_triangles': bool
            - 'num_degenerate_triangles': int
            - 'has_non_manifold_edges': bool
            - 'num_non_manifold_edges': int
            - 'has_inconsistent_orientation': bool
            - 'warnings': list of str, validation warnings
        """
        results = {
            'is_valid': True,
            'has_duplicate_vertices': False,
            'num_duplicate_vertices': 0,
            'has_degenerate_triangles': False,
            'num_degenerate_triangles': 0,
            'has_non_manifold_edges': False,
            'num_non_manifold_edges': 0,
            'has_inconsistent_orientation': False,
            'warnings': []
        }
        
        # Check for duplicate vertices
        # Use a tolerance for floating point comparison
        tolerance = 1e-10
        unique_vertices = []
        vertex_map = {}
        for i, v in enumerate(self.vertices):
            is_duplicate = False
            for j, uv in enumerate(unique_vertices):
                if np.linalg.norm(v - uv) < tolerance:
                    vertex_map[i] = j
                    is_duplicate = True
                    results['num_duplicate_vertices'] += 1
                    break
            if not is_duplicate:
                unique_vertices.append(v)
                vertex_map[i] = len(unique_vertices) - 1
        
        if results['num_duplicate_vertices'] > 0:
            results['has_duplicate_vertices'] = True
            results['is_valid'] = False
            results['warnings'].append(
                f"Found {results['num_duplicate_vertices']} duplicate vertices"
            )
        
        # Check for degenerate triangles
        degenerate_threshold = 1e-12
        for i, area in enumerate(self.triangle_areas):
            if area < degenerate_threshold:
                results['num_degenerate_triangles'] += 1
        
        if results['num_degenerate_triangles'] > 0:
            results['has_degenerate_triangles'] = True
            results['is_valid'] = False
            results['warnings'].append(
                f"Found {results['num_degenerate_triangles']} degenerate triangles"
            )
        
        # Check for non-manifold edges (edges shared by more than 2 triangles)
        for edge_idx, triangles in self.edge_to_triangles.items():
            if len(triangles) > 2:
                results['num_non_manifold_edges'] += 1
        
        if results['num_non_manifold_edges'] > 0:
            results['has_non_manifold_edges'] = True
            results['is_valid'] = False
            results['warnings'].append(
                f"Found {results['num_non_manifold_edges']} non-manifold edges"
            )
        
        # Check for consistent face orientation
        # For each interior edge, check that triangles have opposite normals
        # (simplified check - in practice, need to check edge orientation)
        for edge_idx, triangles in self.edge_to_triangles.items():
            if len(triangles) == 2:
                t1, t2 = triangles[0], triangles[1]
                n1 = self.triangle_normals[t1]
                n2 = self.triangle_normals[t2]
                # Normals should point in opposite directions for shared edge
                # This is a simplified check
                dot_product = np.dot(n1, n2)
                if dot_product > 0.9:  # Too similar (should be ~-1 for opposite)
                    results['has_inconsistent_orientation'] = True
                    results['warnings'].append(
                        f"Edge {edge_idx} has inconsistent triangle orientation"
                    )
                    break
        
        return results
    
