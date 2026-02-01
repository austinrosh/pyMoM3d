"""High-level simulation driver.

Provides SimulationConfig, SimulationResult, and Simulation classes
that orchestrate the full MoM pipeline: mesh -> basis -> Z-fill -> solve -> post-process.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .mesh.mesh_data import Mesh
from .mesh.rwg_basis import RWGBasis
from .mesh.rwg_connectivity import compute_rwg_connectivity
from .mesh.trimesh_mesher import PythonMesher
from .mom.impedance import fill_impedance_matrix
from .mom.excitation import Excitation, PlaneWaveExcitation
from .mom.solver import solve_direct, solve_gmres
from .fields.far_field import compute_far_field
from .fields.rcs import compute_rcs
from .utils.constants import c0, eta0

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for a MoM simulation.

    Parameters
    ----------
    frequency : float
        Operating frequency (Hz).
    excitation : Excitation
        Excitation source.
    solver_type : str
        'direct' or 'gmres'.
    quad_order : int
        Quadrature order for integration.
    near_threshold : float
        Near-field threshold for singularity extraction.
    """
    frequency: float
    excitation: Excitation
    solver_type: str = 'direct'
    quad_order: int = 4
    near_threshold: float = 0.2


@dataclass
class SimulationResult:
    """Result of a MoM simulation at a single frequency.

    Parameters
    ----------
    frequency : float
        Frequency (Hz).
    I_coefficients : ndarray, shape (N,), complex128
        Current expansion coefficients.
    Z_input : complex, optional
        Input impedance (for delta-gap excitation).
    condition_number : float, optional
        Condition number of Z matrix.
    """
    frequency: float
    I_coefficients: np.ndarray
    Z_input: Optional[complex] = None
    condition_number: Optional[float] = None

    def save(self, path: str) -> None:
        """Save result to .npz file."""
        data = {
            'frequency': self.frequency,
            'I_coefficients': self.I_coefficients,
        }
        if self.Z_input is not None:
            data['Z_input'] = self.Z_input
        if self.condition_number is not None:
            data['condition_number'] = self.condition_number
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> 'SimulationResult':
        """Load result from .npz file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            frequency=float(data['frequency']),
            I_coefficients=data['I_coefficients'],
            Z_input=complex(data['Z_input']) if 'Z_input' in data else None,
            condition_number=float(data['condition_number']) if 'condition_number' in data else None,
        )


class Simulation:
    """MoM simulation orchestrator.

    Parameters
    ----------
    config : SimulationConfig
    geometry : object
        Geometry primitive (RectangularPlate, Sphere, etc.).
    mesh : Mesh, optional
        Pre-built mesh. If None, mesh is generated from geometry.
    subdivisions : int
        Subdivision level for trimesh mesh generation.
    mesher : str
        Mesher backend: 'trimesh' (default) or 'gmsh'.
    target_edge_length : float, optional
        Target edge length in meters (used with mesher='gmsh').
    """

    def __init__(
        self,
        config: SimulationConfig,
        geometry=None,
        mesh: Mesh = None,
        subdivisions: int = 2,
        mesher: str = 'trimesh',
        target_edge_length: Optional[float] = None,
    ):
        self.config = config
        self.geometry = geometry
        self.subdivisions = subdivisions

        if mesh is not None:
            self.mesh = mesh
        elif geometry is not None:
            if mesher == 'gmsh':
                from .mesh.gmsh_mesher import GmshMesher
                gmsh_mesher = GmshMesher(target_edge_length=target_edge_length)
                self.mesh = gmsh_mesher.mesh_from_geometry(geometry)
            else:
                trimesh_mesher = PythonMesher()
                trimesh_obj = geometry.to_trimesh(subdivisions=subdivisions)
                self.mesh = trimesh_mesher.mesh_from_geometry(trimesh_obj)
        else:
            raise ValueError("Either geometry or mesh must be provided")

        self.basis = compute_rwg_connectivity(self.mesh)
        self._Z_cache = {}

    def run(self) -> SimulationResult:
        """Run the simulation at the configured frequency.

        Returns
        -------
        result : SimulationResult
        """
        return self._solve_at_frequency(self.config.frequency)

    def sweep(self, frequencies: List[float]) -> List[SimulationResult]:
        """Run simulation at multiple frequencies.

        The mesh and basis are computed once; Z-fill and solve run per frequency.

        Parameters
        ----------
        frequencies : list of float
            Frequencies to sweep (Hz).

        Returns
        -------
        results : list of SimulationResult
        """
        return [self._solve_at_frequency(f) for f in frequencies]

    def _solve_at_frequency(self, frequency: float) -> SimulationResult:
        k = 2.0 * np.pi * frequency / c0

        logger.info(f"Solving at f={frequency:.4g} Hz (k={k:.4g} rad/m), "
                     f"N={self.basis.num_basis} unknowns")

        self.mesh.check_density(frequency)

        Z = fill_impedance_matrix(
            self.basis, self.mesh, k, eta0,
            quad_order=self.config.quad_order,
            near_threshold=self.config.near_threshold,
        )

        V = self.config.excitation.compute_voltage_vector(self.basis, self.mesh, k)

        cond = float(np.linalg.cond(Z))

        if self.config.solver_type == 'gmres':
            I = solve_gmres(Z, V)
        else:
            I = solve_direct(Z, V)

        # Compute input impedance for delta-gap
        from .mom.excitation import DeltaGapExcitation
        Z_in = None
        if isinstance(self.config.excitation, DeltaGapExcitation):
            idx = self.config.excitation.basis_index
            if abs(I[idx]) > 0:
                Z_in = self.config.excitation.voltage / I[idx]

        return SimulationResult(
            frequency=frequency,
            I_coefficients=I,
            Z_input=Z_in,
            condition_number=cond,
        )


def load_stl(path: str, mesher: str = 'trimesh') -> Mesh:
    """Load a mesh from an STL file.

    Parameters
    ----------
    path : str
        Path to .stl file.
    mesher : str
        Mesher backend: 'trimesh' (default) or 'gmsh'.

    Returns
    -------
    mesh : Mesh
    """
    if mesher == 'gmsh':
        from .mesh.gmsh_mesher import GmshMesher
        return GmshMesher().mesh_from_file(path)
    import trimesh
    trimesh_obj = trimesh.load(path)
    m = PythonMesher()
    return m.mesh_from_geometry(trimesh_obj)
