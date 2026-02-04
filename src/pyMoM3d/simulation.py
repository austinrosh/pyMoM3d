"""High-level simulation driver.

Provides SimulationConfig, SimulationResult, and Simulation classes
that orchestrate the full MoM pipeline: mesh -> basis -> Z-fill -> solve -> post-process.
"""

import logging
import os
import time as _time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from .utils.reporter import TerminalReporter, SilentReporter, RecordingReporter
from .utils.report_writer import write_report

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
    enable_report: bool = False
    report_dir: str = 'results/simulation_info'


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
    reporter : object, optional
        Progress reporter. Defaults to ``TerminalReporter``.
        Pass ``SilentReporter()`` to suppress output.
    """

    def __init__(
        self,
        config: SimulationConfig,
        geometry=None,
        mesh: Mesh = None,
        subdivisions: int = 2,
        mesher: str = 'trimesh',
        target_edge_length: Optional[float] = None,
        reporter=None,
    ):
        self.config = config
        self.geometry = geometry
        self.subdivisions = subdivisions
        self._start_time = _time.monotonic()

        inner_reporter = reporter if reporter is not None else TerminalReporter()
        if config.enable_report:
            self.reporter = RecordingReporter(inner_reporter)
            # Seed metadata with config info
            wavelength = c0 / config.frequency if config.frequency > 0 else None
            exc = config.excitation
            exc_str = type(exc).__name__
            if hasattr(exc, 'E0') and hasattr(exc, 'k_hat'):
                exc_str += f" (E0={list(exc.E0)}, k_hat={list(exc.k_hat)})"
            elif hasattr(exc, 'voltage'):
                exc_str += f" (V={exc.voltage})"
            self.reporter.metadata["config"] = {
                "frequency": config.frequency,
                "wavelength": wavelength,
                "excitation": exc_str,
                "solver_type": config.solver_type,
                "quad_order": config.quad_order,
                "near_threshold": config.near_threshold,
            }
            geom_type = type(geometry).__name__ if geometry else "custom"
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.reporter.metadata["run_id"] = f"{geom_type.lower()}_{ts}"
            self.reporter.metadata["timestamp"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            from . import __version__
            self.reporter.metadata["version"] = __version__
        else:
            self.reporter = inner_reporter

        # --- Mesh generation ---
        if mesh is not None:
            self.mesh = mesh
        elif geometry is not None:
            geom_type = type(geometry).__name__
            self.reporter.stage_start("mesh", geometry_type=geom_type)
            try:
                if mesher == 'gmsh':
                    from .mesh.gmsh_mesher import GmshMesher
                    gmsh_mesher = GmshMesher(target_edge_length=target_edge_length)
                    self.mesh = gmsh_mesher.mesh_from_geometry(geometry)
                else:
                    trimesh_mesher = PythonMesher()
                    trimesh_obj = geometry.to_trimesh(subdivisions=subdivisions)
                    self.mesh = trimesh_mesher.mesh_from_geometry(trimesh_obj)
            except Exception:
                self.reporter.error("Mesh generation failed")
                raise
            stats = self.mesh.get_statistics()
            self.reporter.stage_end(
                "mesh",
                num_triangles=stats['num_triangles'],
                num_vertices=stats['num_vertices'],
                mean_edge=stats['mean_edge_length'],
            )
            if isinstance(self.reporter, RecordingReporter):
                self.reporter.metadata.setdefault("mesh", {})
                self.reporter.metadata["mesh"]["mesher"] = mesher
                self.reporter.metadata["mesh"]["target_edge_length"] = target_edge_length
        else:
            raise ValueError("Either geometry or mesh must be provided")

        # --- RWG connectivity ---
        self.reporter.stage_start("rwg")
        self.basis = compute_rwg_connectivity(self.mesh)
        num_interior = self.basis.num_basis
        # Count boundary edges
        num_boundary = 0
        if self.mesh.rwg_pairs is not None:
            num_boundary = int(np.sum(self.mesh.rwg_pairs[:, 1] == -1))
        self.reporter.stage_end("rwg", num_interior=num_interior, num_boundary=num_boundary)

        if num_interior == 0:
            self.reporter.error("No RWG basis functions found. Check mesh connectivity.")

        self._Z_cache = {}

    def run(self) -> SimulationResult:
        """Run the simulation at the configured frequency.

        Returns
        -------
        result : SimulationResult
        """
        status = "COMPLETED"
        result = None
        try:
            result = self._solve_at_frequency(self.config.frequency)
            return result
        except KeyboardInterrupt:
            status = "INTERRUPTED"
            raise
        except Exception:
            status = "FAILED"
            raise
        finally:
            self._write_report(status, [result] if result else [])

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
        K = len(frequencies)
        f_min_ghz = frequencies[0] / 1e9 if K > 0 else 0
        f_max_ghz = frequencies[-1] / 1e9 if K > 0 else 0
        self.reporter.stage_start(
            "sweep", num_freqs=K, f_min_ghz=f"{f_min_ghz:.2f}",
            f_max_ghz=f"{f_max_ghz:.2f}",
        )
        results = []
        status = "COMPLETED"
        try:
            for i, f in enumerate(frequencies):
                self.reporter.stage_start(
                    "sweep_freq", index=i + 1, total=K,
                    freq_ghz=f"{f/1e9:.2f}",
                )
                results.append(self._solve_at_frequency(f))
        except KeyboardInterrupt:
            status = f"INTERRUPTED ({len(results)}/{K} completed)"
            self.reporter.warning("Sweep interrupted by user")
            self.reporter.finish()
            self._write_report(status, results)
            raise
        except Exception:
            status = "FAILED"
            self._write_report(status, results)
            raise
        self.reporter.stage_end("sweep", num_freqs=K)
        self._write_report(status, results)
        return results

    def compute_cma(
        self,
        frequency: Optional[float] = None,
        num_modes: Optional[int] = None,
    ):
        """Compute Characteristic Mode Analysis at a single frequency.

        This method performs CMA using the existing mesh and RWG basis.
        The impedance matrix is computed (or retrieved from cache), then
        decomposed into characteristic modes.

        Parameters
        ----------
        frequency : float, optional
            Frequency (Hz). If None, uses config.frequency.
        num_modes : int, optional
            Number of modes to retain (sorted by significance).
            If None, returns all N modes.

        Returns
        -------
        cma_result : CMAResult
            Characteristic mode analysis results containing eigenvalues,
            eigenvectors (modal currents), modal significance, and
            characteristic angles.

        Examples
        --------
        >>> sim = Simulation(config, geometry=plate)
        >>> cma = sim.compute_cma(frequency=1e9, num_modes=5)
        >>> J_mode1 = cma.get_mode(0)  # Most significant mode
        >>> plot_surface_current(J_mode1, sim.basis, sim.mesh)
        """
        from .analysis.cma import compute_characteristic_modes

        if frequency is None:
            frequency = self.config.frequency

        k = 2.0 * np.pi * frequency / c0
        N = self.basis.num_basis

        self.reporter.stage_start("cma", frequency_ghz=f"{frequency/1e9:.3f}", N=N)

        # --- Z-fill ---
        total_pairs = N * (N + 1) // 2
        self.reporter.stage_start(
            "z_fill", N=N, total_pairs=total_pairs,
            quad_order=self.config.quad_order,
        )

        def _z_progress(fraction):
            self.reporter.stage_progress("z_fill", fraction, row=int(fraction * N), N=N)

        try:
            Z = fill_impedance_matrix(
                self.basis, self.mesh, k, eta0,
                quad_order=self.config.quad_order,
                near_threshold=self.config.near_threshold,
                progress_callback=_z_progress,
            )
        except Exception:
            self.reporter.error("Z-fill failed")
            raise

        self.reporter.stage_end("z_fill", N=N)

        # --- CMA eigensolve ---
        self.reporter.stage_start("cma_eigensolve", N=N)
        cma_result = compute_characteristic_modes(Z, frequency=frequency, num_modes=num_modes)
        self.reporter.stage_end(
            "cma_eigensolve",
            num_modes=len(cma_result.sort_order),
            top_ms=f"{cma_result.get_modal_significance(0):.3f}",
        )

        self.reporter.stage_end("cma", num_modes=len(cma_result.sort_order))

        return cma_result

    def cma_sweep(
        self,
        frequencies: List[float],
        num_modes: Optional[int] = None,
        track_modes: bool = True,
    ) -> tuple:
        """Perform CMA across a frequency sweep.

        The mesh and RWG basis are computed once; Z-fill and CMA are
        performed at each frequency.

        Parameters
        ----------
        frequencies : list of float
            Frequencies to sweep (Hz).
        num_modes : int, optional
            Number of modes to retain at each frequency.
        track_modes : bool, default True
            Whether to track modes across frequency using eigenvector
            correlation.

        Returns
        -------
        cma_results : list of CMAResult
            CMA results at each frequency.
        tracked_indices : list of ndarray or None
            Mode tracking indices if track_modes=True. tracked_indices[f][n]
            gives the original mode index corresponding to tracked mode n
            at frequency index f.

        Examples
        --------
        >>> results, tracking = sim.cma_sweep(frequencies, num_modes=5)
        >>> # Get mode 0 eigenvalue across frequency
        >>> lambda_mode0 = [results[i].eigenvalues[tracking[i][0]]
        ...                 for i in range(len(results))]
        """
        from .analysis.cma import track_modes_across_frequency

        K = len(frequencies)
        f_min_ghz = frequencies[0] / 1e9 if K > 0 else 0
        f_max_ghz = frequencies[-1] / 1e9 if K > 0 else 0
        self.reporter.stage_start(
            "cma_sweep", num_freqs=K, f_min_ghz=f"{f_min_ghz:.2f}",
            f_max_ghz=f"{f_max_ghz:.2f}",
        )

        cma_results = []
        for i, freq in enumerate(frequencies):
            self.reporter.stage_start(
                "cma_sweep_freq", index=i + 1, total=K,
                freq_ghz=f"{freq/1e9:.3f}",
            )
            cma_result = self.compute_cma(frequency=freq, num_modes=num_modes)
            cma_results.append(cma_result)
            self.reporter.stage_end("cma_sweep_freq")

        tracked_indices = None
        if track_modes and len(cma_results) > 1:
            self.reporter.stage_start("mode_tracking", num_freqs=K)
            tracked_indices = track_modes_across_frequency(cma_results)
            self.reporter.stage_end("mode_tracking")

        self.reporter.stage_end("cma_sweep", num_freqs=K)

        return cma_results, tracked_indices

    def _write_report(self, status, results):
        """Write report file if reporting is enabled."""
        if not isinstance(self.reporter, RecordingReporter):
            return
        md = self.reporter.metadata
        md["status"] = status
        md["total_time"] = _time.monotonic() - self._start_time
        if results:
            last = results[-1]
            if last is not None and last.Z_input is not None:
                md.setdefault("results", {})["Z_input"] = last.Z_input
        run_id = md.get("run_id", "unknown")
        path = os.path.join(self.config.report_dir, f"{run_id}.txt")
        write_report(md, path)

    def _solve_at_frequency(self, frequency: float) -> SimulationResult:
        k = 2.0 * np.pi * frequency / c0
        N = self.basis.num_basis

        logger.info(f"Solving at f={frequency:.4g} Hz (k={k:.4g} rad/m), "
                     f"N={N} unknowns")

        # Mesh density check via reporter
        from .utils.constants import c0 as _c0
        wavelength = _c0 / frequency
        mean_edge = float(np.mean(self.mesh.edge_lengths))
        if mean_edge > wavelength / 10.0:
            self.reporter.warning(
                f"Mesh too coarse: mean edge {mean_edge:.4g} m "
                f"> lambda/10 = {wavelength/10:.4g} m at {frequency:.4g} Hz"
            )

        # --- Z-fill ---
        total_pairs = N * (N + 1) // 2
        self.reporter.stage_start(
            "z_fill", N=N, total_pairs=total_pairs,
            quad_order=self.config.quad_order,
        )

        def _z_progress(fraction):
            self.reporter.stage_progress("z_fill", fraction, row=int(fraction * N), N=N)

        try:
            Z = fill_impedance_matrix(
                self.basis, self.mesh, k, eta0,
                quad_order=self.config.quad_order,
                near_threshold=self.config.near_threshold,
                progress_callback=_z_progress,
            )
        except Exception:
            self.reporter.error("Z-fill failed")
            raise

        # Sanity check
        if not np.isfinite(Z).all():
            self.reporter.error("Impedance matrix contains NaN/Inf")

        if isinstance(self.reporter, RecordingReporter):
            self.reporter.metadata.setdefault("z_fill", {})["z_memory_mb"] = Z.nbytes / 1e6
        z_fill_rate = total_pairs  # will be divided by elapsed in stage_end
        self.reporter.stage_end("z_fill", N=N)

        # --- Excitation ---
        V = self.config.excitation.compute_voltage_vector(self.basis, self.mesh, k)

        # --- Solve ---
        cond = float(np.linalg.cond(Z))

        if self.config.solver_type == 'gmres':
            self.reporter.stage_start("solve_gmres", N=N, tol=1e-6)

            def _gmres_progress(iteration, residual):
                self.reporter.stage_progress(
                    "solve_gmres", min(iteration / 100, 0.999),
                    iteration=iteration, residual=residual,
                )

            I = solve_gmres(Z, V, progress_callback=_gmres_progress)
            residual = float(np.linalg.norm(Z @ I - V) / np.linalg.norm(V))
            self.reporter.stage_end(
                "solve_gmres", iterations=None, residual=residual,
            )
        else:
            self.reporter.stage_start("solve_direct", N=N)
            I = solve_direct(Z, V)
            residual = float(np.linalg.norm(Z @ I - V) / np.linalg.norm(V))
            self.reporter.stage_end("solve_direct", cond=cond, residual=residual)

        if cond > 1e12:
            self.reporter.warning(f"cond={cond:.2e}, results may be unreliable")

        # Compute input impedance for delta-gap
        from .mom.excitation import DeltaGapExcitation, StripDeltaGapExcitation
        Z_in = None
        if isinstance(self.config.excitation, StripDeltaGapExcitation):
            Z_in = self.config.excitation.compute_input_impedance(
                I, self.basis, self.mesh)
        elif isinstance(self.config.excitation, DeltaGapExcitation):
            idx = self.config.excitation.basis_index
            l_n = self.basis.edge_length[idx]
            I_terminal = I[idx] * l_n
            if abs(I_terminal) > 0:
                Z_in = self.config.excitation.voltage / I_terminal

        return SimulationResult(
            frequency=frequency,
            I_coefficients=I,
            Z_input=Z_in,
            condition_number=cond,
        )


def _load_trimesh(path: str):
    """Load an STL or OBJ file and return a single trimesh.Trimesh.

    OBJ files with multiple material groups or objects are returned as
    a ``trimesh.Scene`` by ``trimesh.load()``.  This helper collapses
    the scene into one ``Trimesh`` via concatenation so downstream code
    always receives a single mesh.
    """
    import trimesh as _trimesh

    loaded = _trimesh.load(path)
    if isinstance(loaded, _trimesh.Trimesh):
        return loaded
    if isinstance(loaded, _trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()
                  if isinstance(g, _trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle meshes found in {path}")
        return _trimesh.util.concatenate(meshes)
    raise TypeError(
        f"Unexpected type from trimesh.load(): {type(loaded).__name__}"
    )


def load_stl(path: str, mesher: str = 'trimesh') -> Mesh:
    """Load a mesh from an STL or OBJ file.

    Parameters
    ----------
    path : str
        Path to .stl or .obj file.
    mesher : str
        Mesher backend: 'trimesh' (default) or 'gmsh'.

    Returns
    -------
    mesh : Mesh
    """
    if mesher == 'gmsh':
        from .mesh.gmsh_mesher import GmshMesher
        return GmshMesher().mesh_from_file(path)
    trimesh_obj = _load_trimesh(path)
    m = PythonMesher()
    return m.mesh_from_geometry(trimesh_obj)
