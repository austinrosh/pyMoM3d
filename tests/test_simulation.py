"""Tests for simulation driver and analysis modules."""

import numpy as np
import pytest
import tempfile

from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
from pyMoM3d.mom.excitation import PlaneWaveExcitation, DeltaGapExcitation
from pyMoM3d.simulation import SimulationConfig, SimulationResult, Simulation
from pyMoM3d.analysis.impedance_analysis import compute_s11
from pyMoM3d.analysis.pattern_analysis import compute_directivity
from pyMoM3d.utils.constants import eta0


def _make_small_plate_mesh():
    """Simple 2-triangle mesh for quick tests."""
    vertices = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.0, 0.5, 0.0],
    ])
    triangles = np.array([[0, 1, 2], [2, 1, 3]])
    return Mesh(vertices, triangles)


class TestSimulationConfig:
    def test_create(self):
        exc = PlaneWaveExcitation(
            E0=np.array([1.0, 0.0, 0.0]),
            k_hat=np.array([0.0, 0.0, -1.0]),
        )
        config = SimulationConfig(frequency=1e9, excitation=exc)
        assert config.frequency == 1e9
        assert config.solver_type == 'direct'


class TestSimulation:
    def test_run_with_mesh(self):
        mesh = _make_small_plate_mesh()
        exc = PlaneWaveExcitation(
            E0=np.array([1.0, 0.0, 0.0]),
            k_hat=np.array([0.0, 0.0, -1.0]),
        )
        config = SimulationConfig(frequency=3e8, excitation=exc)
        sim = Simulation(config, mesh=mesh)
        result = sim.run()

        assert isinstance(result, SimulationResult)
        assert result.frequency == 3e8
        assert len(result.I_coefficients) > 0
        assert np.all(np.isfinite(result.I_coefficients))
        assert result.condition_number is not None

    def test_delta_gap_impedance(self):
        mesh = _make_small_plate_mesh()
        basis = compute_rwg_connectivity(mesh)
        exc = DeltaGapExcitation(basis_index=0, voltage=1.0)
        config = SimulationConfig(frequency=3e8, excitation=exc)
        sim = Simulation(config, mesh=mesh)
        result = sim.run()

        assert result.Z_input is not None
        assert np.isfinite(result.Z_input)

    def test_sweep(self):
        mesh = _make_small_plate_mesh()
        exc = PlaneWaveExcitation(
            E0=np.array([1.0, 0.0, 0.0]),
            k_hat=np.array([0.0, 0.0, -1.0]),
        )
        config = SimulationConfig(frequency=3e8, excitation=exc)
        sim = Simulation(config, mesh=mesh)
        results = sim.sweep([2e8, 3e8])
        assert len(results) == 2
        assert results[0].frequency == 2e8
        assert results[1].frequency == 3e8


class TestSimulationResult:
    def test_save_load(self):
        result = SimulationResult(
            frequency=1e9,
            I_coefficients=np.array([1 + 2j, 3 + 4j]),
            Z_input=50 + 25j,
            condition_number=100.0,
        )
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            result.save(f.name)
            loaded = SimulationResult.load(f.name)

        assert loaded.frequency == 1e9
        assert np.allclose(loaded.I_coefficients, result.I_coefficients)


class TestAnalysis:
    def test_s11(self):
        # Matched load
        assert np.isclose(abs(compute_s11(50.0 + 0j, Z0=50.0)), 0.0, atol=1e-10)
        # Open circuit
        assert np.isclose(abs(compute_s11(1e10 + 0j, Z0=50.0)), 1.0, atol=1e-5)
        # Short circuit
        assert np.isclose(abs(compute_s11(0.0 + 0j, Z0=50.0)), 1.0, atol=1e-10)

    def test_directivity(self):
        theta = np.linspace(0.01, np.pi - 0.01, 90)
        phi = np.array([0.0])
        # Isotropic pattern
        E_theta = np.ones((90, 1), dtype=np.complex128)
        E_phi = np.zeros((90, 1), dtype=np.complex128)
        D, D_max, D_dBi = compute_directivity(E_theta, E_phi, theta, phi, eta0)
        # For isotropic: D_max should be close to 1 (0 dBi)
        assert D_max > 0
        assert np.isfinite(D_dBi)
