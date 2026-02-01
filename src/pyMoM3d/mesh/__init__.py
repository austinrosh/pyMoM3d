"""Mesh module for triangular surface meshing and RWG connectivity."""

from .mesh_data import Mesh
from .mesher import create_mesh_from_vertices, create_rectangular_mesh
from .trimesh_mesher import PythonMesher, create_mesh_from_trimesh
from .rwg_connectivity import compute_rwg_connectivity
from .rwg_basis import RWGBasis

__all__ = [
    'Mesh',
    'RWGBasis',
    'create_mesh_from_vertices',
    'create_rectangular_mesh',
    'PythonMesher',
    'create_mesh_from_trimesh',
    'compute_rwg_connectivity',
]
