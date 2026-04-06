"""Tests for automatic QS/FW source selection (DualBandExtractor).

Tests cover:
- Frequency partitioning by kD
- Electrical size computation
- Mesh bounding box dimension
- DualBandExtractor dispatch logic
- Crossover frequency calculation
- Edge cases (all QS, all FW, single frequency)
"""

import numpy as np
import pytest

from pyMoM3d.network.auto_extract import (
    DualBandExtractor,
    partition_frequencies,
    compute_kD,
    mesh_max_dimension,
)
from pyMoM3d.network.network_result import NetworkResult
from pyMoM3d.utils.constants import c0


# ------------------------------------------------------------------ #
# compute_kD tests
# ------------------------------------------------------------------ #

class TestComputeKD:

    def test_known_value(self):
        """kD at 3 GHz for 10mm structure."""
        freq = 3e9
        D = 10e-3
        k = 2 * np.pi * freq / c0
        expected = k * D
        assert abs(compute_kD(freq, D) - expected) < 1e-15

    def test_zero_frequency(self):
        """kD = 0 at DC."""
        assert compute_kD(0.0, 0.01) == 0.0

    def test_scales_linearly(self):
        """kD should scale linearly with both freq and D."""
        kD1 = compute_kD(1e9, 0.01)
        kD2 = compute_kD(2e9, 0.01)
        assert abs(kD2 / kD1 - 2.0) < 1e-12

        kD3 = compute_kD(1e9, 0.02)
        assert abs(kD3 / kD1 - 2.0) < 1e-12


# ------------------------------------------------------------------ #
# partition_frequencies tests
# ------------------------------------------------------------------ #

class TestPartitionFrequencies:

    def test_all_qs(self):
        """All frequencies below threshold → all QS."""
        freqs = [0.1e9, 0.5e9, 1e9]
        D = 10e-3  # kD = 0.02, 0.1, 0.2 at these freqs
        qs, fw, qs_idx, fw_idx = partition_frequencies(freqs, D, kD_threshold=0.5)
        assert len(qs) == 3
        assert len(fw) == 0
        assert qs_idx == [0, 1, 2]

    def test_all_fw(self):
        """All frequencies above threshold → all FW."""
        freqs = [10e9, 15e9, 20e9]
        D = 20e-3  # kD = 4.2, 6.3, 8.4
        qs, fw, qs_idx, fw_idx = partition_frequencies(freqs, D, kD_threshold=0.5)
        assert len(qs) == 0
        assert len(fw) == 3
        assert fw_idx == [0, 1, 2]

    def test_mixed_partition(self):
        """Mixed frequencies split correctly."""
        D = 20e-3
        # kD = 2π f D / c0 = 2π × f × 0.02 / 3e8
        # At 1 GHz: kD ≈ 0.42 (QS)
        # At 2 GHz: kD ≈ 0.84 (FW)
        # At 5 GHz: kD ≈ 2.09 (FW)
        freqs = [1e9, 2e9, 5e9]
        qs, fw, qs_idx, fw_idx = partition_frequencies(freqs, D, kD_threshold=0.5)
        assert len(qs) == 1
        assert len(fw) == 2
        assert qs_idx == [0]
        assert fw_idx == [1, 2]
        assert qs[0] == 1e9

    def test_preserves_order(self):
        """Indices map back to original frequency order."""
        freqs = [10e9, 0.1e9, 5e9, 0.5e9]  # mixed order
        D = 20e-3
        qs, fw, qs_idx, fw_idx = partition_frequencies(freqs, D, kD_threshold=0.5)

        # Verify indices point to correct frequencies
        for i in qs_idx:
            assert compute_kD(freqs[i], D) < 0.5
        for i in fw_idx:
            assert compute_kD(freqs[i], D) >= 0.5

    def test_empty_input(self):
        """Empty frequency list."""
        qs, fw, qi, fi = partition_frequencies([], 0.01)
        assert qs == [] and fw == []

    def test_exact_threshold(self):
        """Frequency exactly at threshold goes to FW (≥ threshold)."""
        D = 0.01
        # kD = 0.5 at f = 0.5 * c0 / (2π * D)
        f_cross = 0.5 * c0 / (2 * np.pi * D)
        qs, fw, _, _ = partition_frequencies([f_cross], D, kD_threshold=0.5)
        # kD = 0.5, which is NOT < 0.5, so goes to FW
        assert len(fw) == 1
        assert len(qs) == 0


# ------------------------------------------------------------------ #
# mesh_max_dimension tests
# ------------------------------------------------------------------ #

class TestMeshMaxDimension:

    def test_unit_cube(self):
        """Bounding box diagonal of a unit cube = sqrt(3)."""
        from pyMoM3d.mesh.mesh_data import Mesh

        # Minimal mesh: 2 triangles forming a unit square on z=0 plane
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1],  # extend to 3D bounding box
        ], dtype=float)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        mesh = Mesh(vertices=verts, triangles=tris)
        D = mesh_max_dimension(mesh)
        expected = np.sqrt(1**2 + 1**2 + 1**2)
        assert abs(D - expected) < 1e-12

    def test_flat_plate(self):
        """Flat plate: diagonal = sqrt(w² + h²)."""
        from pyMoM3d.mesh.mesh_data import Mesh

        w, h = 0.02, 0.003  # 20mm x 3mm
        verts = np.array([
            [-w/2, -h/2, 0], [w/2, -h/2, 0],
            [w/2, h/2, 0], [-w/2, h/2, 0],
        ], dtype=float)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        mesh = Mesh(vertices=verts, triangles=tris)
        D = mesh_max_dimension(mesh)
        expected = np.sqrt(w**2 + h**2)
        assert abs(D - expected) < 1e-12


# ------------------------------------------------------------------ #
# Mock solvers for DualBandExtractor tests
# ------------------------------------------------------------------ #

class MockQSSolver:
    """Mock quasi-static solver that returns synthetic NetworkResults."""

    def __init__(self, Z0=50.0):
        self.Z0 = Z0
        self.call_log = []

    def extract(self, frequencies):
        self.call_log.append(list(frequencies))
        results = []
        for f in frequencies:
            Z = np.array([[self.Z0]], dtype=complex)
            results.append(NetworkResult(
                frequency=f,
                Z_matrix=Z,
                port_names=['P1'],
                Z0=self.Z0,
            ))
        return results


class MockFWExtractor:
    """Mock full-wave extractor that returns synthetic NetworkResults."""

    def __init__(self, Z0=50.0):
        self.Z0 = Z0
        self.call_log = []

    def extract(self, frequencies):
        self.call_log.append(list(frequencies))
        results = []
        for f in frequencies:
            Z = np.array([[self.Z0 * 1.1]], dtype=complex)  # slightly different
            results.append(NetworkResult(
                frequency=f,
                Z_matrix=Z,
                port_names=['P1'],
                Z0=self.Z0,
            ))
        return results


# ------------------------------------------------------------------ #
# DualBandExtractor tests
# ------------------------------------------------------------------ #

class TestDualBandExtractor:

    @pytest.fixture
    def dual(self):
        """DualBandExtractor with mock solvers."""
        qs = MockQSSolver()
        fw = MockFWExtractor()
        return DualBandExtractor(qs, fw, max_dimension=0.02, kD_threshold=0.5)

    def test_all_qs(self, dual):
        """Low frequencies → all dispatched to QS."""
        results = dual.extract([0.1e9, 0.5e9])
        assert len(results) == 2
        assert all(r is not None for r in results)
        # QS returns Z0 = 50, FW returns Z0 * 1.1
        for r in results:
            assert abs(r.Z_matrix[0, 0] - 50.0) < 1e-10

    def test_all_fw(self, dual):
        """High frequencies → all dispatched to FW."""
        results = dual.extract([10e9, 15e9])
        assert len(results) == 2
        for r in results:
            assert abs(r.Z_matrix[0, 0] - 55.0) < 1e-10  # 50 * 1.1

    def test_mixed_dispatch(self, dual):
        """Mixed frequencies dispatched to correct solvers."""
        # kD at 1 GHz, D=20mm: 2π×1e9×0.02/3e8 ≈ 0.42 → QS
        # kD at 5 GHz, D=20mm: 2π×5e9×0.02/3e8 ≈ 2.09 → FW
        results = dual.extract([1e9, 5e9])
        assert abs(results[0].Z_matrix[0, 0] - 50.0) < 1e-10   # QS
        assert abs(results[1].Z_matrix[0, 0] - 55.0) < 1e-10   # FW

    def test_frequency_order_preserved(self, dual):
        """Results are in the same order as input frequencies."""
        freqs = [10e9, 0.1e9, 5e9, 0.5e9]
        results = dual.extract(freqs)
        assert len(results) == 4
        for r, f in zip(results, freqs):
            assert r.frequency == f

    def test_single_frequency(self, dual):
        """Single scalar frequency works."""
        results = dual.extract(1e9)
        assert len(results) == 1

    def test_solver_at(self, dual):
        """solver_at() returns correct solver label."""
        assert dual.solver_at(0.1e9) == 'qs'
        assert dual.solver_at(10e9) == 'fw'

    def test_crossover_frequency(self, dual):
        """Crossover frequency is consistent with threshold."""
        f_cross = dual.crossover_frequency()
        # At crossover, kD should equal threshold
        kD_at_cross = compute_kD(f_cross, dual.max_dim)
        assert abs(kD_at_cross - dual.kD_threshold) < 1e-12

    def test_solvers_called_with_correct_freqs(self, dual):
        """Verify mock solvers receive only their assigned frequencies."""
        freqs = [0.1e9, 10e9, 0.5e9, 15e9]
        dual.extract(freqs)

        qs_calls = dual.qs.call_log
        fw_calls = dual.fw.call_log

        # QS should get the low frequencies
        assert len(qs_calls) == 1
        assert set(qs_calls[0]) == {0.1e9, 0.5e9}

        # FW should get the high frequencies
        assert len(fw_calls) == 1
        assert set(fw_calls[0]) == {10e9, 15e9}

    def test_empty_qs_band(self):
        """When all freqs are FW, QS solver not called."""
        qs = MockQSSolver()
        fw = MockFWExtractor()
        dual = DualBandExtractor(qs, fw, max_dimension=0.02, kD_threshold=0.5)
        dual.extract([10e9, 15e9])
        assert len(qs.call_log) == 0
        assert len(fw.call_log) == 1

    def test_empty_fw_band(self):
        """When all freqs are QS, FW solver not called."""
        qs = MockQSSolver()
        fw = MockFWExtractor()
        dual = DualBandExtractor(qs, fw, max_dimension=0.02, kD_threshold=0.5)
        dual.extract([0.1e9, 0.5e9])
        assert len(qs.call_log) == 1
        assert len(fw.call_log) == 0

    def test_custom_threshold(self):
        """Custom kD threshold changes the partition."""
        qs = MockQSSolver()
        fw = MockFWExtractor()
        # With threshold=2.0, more frequencies go to QS
        dual = DualBandExtractor(qs, fw, max_dimension=0.02, kD_threshold=2.0)
        # kD at 5 GHz: ~2.09 → FW (just barely)
        # kD at 4 GHz: ~1.68 → QS
        results = dual.extract([4e9, 5e9])
        assert abs(results[0].Z_matrix[0, 0] - 50.0) < 1e-10   # QS
        assert abs(results[1].Z_matrix[0, 0] - 55.0) < 1e-10   # FW
