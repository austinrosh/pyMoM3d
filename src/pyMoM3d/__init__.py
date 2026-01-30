"""pyMoM3d: 3D Method of Moments solver with RWG basis functions."""

from .geometry import RectangularPlate, Sphere, Cylinder, Cube, Pyramid
from .mesh import (
    Mesh,
    create_mesh_from_vertices,
    create_rectangular_mesh,
    PythonMesher,
    create_mesh_from_trimesh,
    compute_rwg_connectivity,
)
from .visualization import plot_mesh, plot_mesh_3d

__version__ = '0.1.0'

__all__ = [
    'RectangularPlate',
    'Sphere',
    'Cylinder',
    'Cube',
    'Pyramid',
    'Mesh',
    'create_mesh_from_vertices',
    'create_rectangular_mesh',
    'PythonMesher',
    'create_mesh_from_trimesh',
    'compute_rwg_connectivity',
    'plot_mesh',
    'plot_mesh_3d',
]
