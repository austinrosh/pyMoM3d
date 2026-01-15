"""pyMoM3d: 3D Method of Moments solver with RWG basis functions."""

from .geometry import RectangularPlate
from .mesh import Mesh, create_mesh_from_vertices, compute_rwg_connectivity
from .visualization import plot_mesh, plot_mesh_3d

__version__ = '0.1.0'

__all__ = [
    'RectangularPlate',
    'Mesh',
    'create_mesh_from_vertices',
    'compute_rwg_connectivity',
    'plot_mesh',
    'plot_mesh_3d',
]
