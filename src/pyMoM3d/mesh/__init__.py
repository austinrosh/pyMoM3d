"""Mesh module for triangular surface meshing and RWG connectivity."""

from .mesh_data import Mesh
from .mesher import create_mesh_from_vertices, create_rectangular_mesh
from .gmsh_mesher import GmshMesher
from .rwg_connectivity import compute_rwg_connectivity
from .rwg_basis import RWGBasis
from .mirror import mirror_mesh_x, combine_meshes, extract_submesh

__all__ = [
    'Mesh',
    'RWGBasis',
    'create_mesh_from_vertices',
    'create_rectangular_mesh',
    'GmshMesher',
    'compute_rwg_connectivity',
    'mirror_mesh_x',
    'combine_meshes',
    'extract_submesh',
]
