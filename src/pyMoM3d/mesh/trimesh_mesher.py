"""Trimesh-based mesh generation utilities."""

import numpy as np
import trimesh
from typing import Optional, Union

from .mesh_data import Mesh


class PythonMesher:
    """
    Mesh generator using trimesh for high-quality surface meshing.
    
    This class replaces Delaunay-based triangulation with trimesh,
    which provides better topology and triangle quality for 3D surfaces.
    """
    
    def __init__(self, merge_vertices: bool = True, remove_degenerate: bool = True):
        """
        Initialize the mesher.
        
        Parameters
        ----------
        merge_vertices : bool, default True
            Merge duplicate vertices in generated meshes
        remove_degenerate : bool, default True
            Remove degenerate triangles from generated meshes
        """
        self.merge_vertices = merge_vertices
        self.remove_degenerate = remove_degenerate
    
    def mesh_from_geometry(
        self,
        geometry: Union[trimesh.Trimesh, np.ndarray],
        triangles: Optional[np.ndarray] = None
    ) -> Mesh:
        """
        Create a Mesh object from geometry.
        
        Parameters
        ----------
        geometry : trimesh.Trimesh or ndarray
            Either a trimesh object or vertices array (N, 3)
        triangles : ndarray, shape (M, 3), optional
            Triangle connectivity. Only used if geometry is vertices array.
        
        Returns
        -------
        mesh : Mesh
            Mesh object with cleaned topology
        """
        if isinstance(geometry, trimesh.Trimesh):
            mesh_obj = geometry.copy()
        elif isinstance(geometry, np.ndarray):
            # Create trimesh from vertices and triangles
            if triangles is None:
                raise ValueError("triangles must be provided when geometry is vertices array")
            mesh_obj = trimesh.Trimesh(vertices=geometry, faces=triangles)
        else:
            raise TypeError(f"geometry must be trimesh.Trimesh or ndarray, got {type(geometry)}")
        
        # Clean the mesh
        mesh_obj = self._clean_mesh(mesh_obj)
        
        # Convert to internal Mesh format
        return Mesh.from_trimesh(mesh_obj)
    
    def _clean_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Clean mesh: merge vertices, remove degenerate faces, fix normals.
        
        Parameters
        ----------
        mesh : trimesh.Trimesh
            Input mesh
        
        Returns
        -------
        cleaned_mesh : trimesh.Trimesh
            Cleaned mesh
        """
        # Merge duplicate vertices
        if self.merge_vertices:
            mesh.merge_vertices()
        
        # Remove degenerate faces
        if self.remove_degenerate:
            # Remove faces with zero area
            areas = mesh.area_faces
            valid_faces = areas > 1e-12
            if not np.all(valid_faces):
                mesh.update_faces(valid_faces)
        
        # Fix normals to be consistent
        mesh.fix_normals()
        
        # Ensure we have a valid mesh
        if not mesh.is_volume:
            # For surface meshes, ensure winding is consistent
            mesh.fix_normals()
        
        return mesh


def create_mesh_from_trimesh(trimesh_obj: trimesh.Trimesh) -> Mesh:
    """
    Convenience function to create Mesh from trimesh object.
    
    Parameters
    ----------
    trimesh_obj : trimesh.Trimesh
        Trimesh object
    
    Returns
    -------
    mesh : Mesh
        Mesh object
    """
    mesher = PythonMesher()
    return mesher.mesh_from_geometry(trimesh_obj)
