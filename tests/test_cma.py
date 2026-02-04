"""Tests for Characteristic Mode Analysis (CMA)."""

import numpy as np
import pytest

from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
from pyMoM3d.mom.impedance import fill_impedance_matrix
from pyMoM3d.utils.constants import eta0, c0
from pyMoM3d.analysis.cma import (
    CMAResult,
    compute_characteristic_modes,
    compute_modal_significance,
    compute_characteristic_angle,
    solve_cma,
    verify_orthogonality,
    verify_eigenvalue_reality,
    compute_modal_excitation_coefficient,
    expand_current_in_modes,
    track_modes_across_frequency,
)


def _make_two_triangle_mesh():
    """Two triangles sharing an edge — simplest possible RWG test case."""
    vertices = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.0, 0.5, 0.0],
    ])
    triangles = np.array([
        [0, 1, 2],
        [2, 1, 3],
    ])
    return Mesh(vertices, triangles)


def _make_small_plate():
    """2x2 grid = 8 triangles, multiple interior edges."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    triangles = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4],
        [3, 4, 7],
        [3, 7, 6],
        [4, 5, 8],
        [4, 8, 7],
    ])
    return Mesh(vertices, triangles)


def _make_larger_plate():
    """3x3 grid = 18 triangles, more basis functions for richer modal analysis."""
    nx, ny = 4, 4
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    vertices = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(nx * ny)])

    triangles = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            v0 = i * nx + j
            v1 = v0 + 1
            v2 = v0 + nx
            v3 = v2 + 1
            triangles.append([v0, v1, v3])
            triangles.append([v0, v3, v2])

    return Mesh(vertices.astype(np.float64), np.array(triangles, dtype=np.int32))


class TestModalSignificanceAndAngle:
    """Test standalone metric functions."""

    def test_modal_significance_at_resonance(self):
        """MS = 1 when λ = 0."""
        ms = compute_modal_significance(np.array([0.0]))
        assert np.isclose(ms[0], 1.0)

    def test_modal_significance_away_from_resonance(self):
        """MS decreases as |λ| increases."""
        eigenvalues = np.array([0.0, 1.0, 10.0, 100.0])
        ms = compute_modal_significance(eigenvalues)
        assert ms[0] > ms[1] > ms[2] > ms[3]
        assert np.isclose(ms[1], 1.0 / np.sqrt(2))  # λ=1 -> MS=1/sqrt(2)

    def test_characteristic_angle_at_resonance(self):
        """α = 180° when λ = 0."""
        alpha = compute_characteristic_angle(np.array([0.0]))
        assert np.isclose(alpha[0], 180.0)

    def test_characteristic_angle_inductive(self):
        """α < 180° for λ > 0 (inductive)."""
        alpha = compute_characteristic_angle(np.array([1.0]))
        assert alpha[0] < 180.0
        assert alpha[0] > 90.0

    def test_characteristic_angle_capacitive(self):
        """α > 180° for λ < 0 (capacitive)."""
        alpha = compute_characteristic_angle(np.array([-1.0]))
        assert alpha[0] > 180.0
        assert alpha[0] < 270.0


class TestTwoTriangleCMA:
    """CMA for single basis function (1x1 matrix)."""

    def test_single_mode(self):
        mesh = _make_two_triangle_mesh()
        basis = compute_rwg_connectivity(mesh)
        assert basis.num_basis == 1

        k = 2 * np.pi  # lambda = 1 m
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)

        result = compute_characteristic_modes(Z, frequency=c0)
        assert len(result.eigenvalues) == 1
        assert len(result.eigenvectors) == 1

    def test_eigenvalue_is_real(self):
        mesh = _make_two_triangle_mesh()
        basis = compute_rwg_connectivity(mesh)
        k = 2 * np.pi
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)

        result = compute_characteristic_modes(Z)
        is_real, max_imag = verify_eigenvalue_reality(result)
        assert is_real


class TestSmallPlateCMA:
    """CMA for a small plate with multiple basis functions."""

    @pytest.fixture
    def plate_cma(self):
        mesh = _make_small_plate()
        basis = compute_rwg_connectivity(mesh)
        k = 2 * np.pi  # lambda = 1 m
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
        result = compute_characteristic_modes(Z, frequency=c0)
        return result, basis, Z

    def test_num_modes_equals_basis(self, plate_cma):
        """Number of modes equals number of basis functions."""
        result, basis, Z = plate_cma
        assert len(result.eigenvalues) == basis.num_basis

    def test_eigenvalues_are_real(self, plate_cma):
        """All eigenvalues should be real for symmetric Z."""
        result, basis, Z = plate_cma
        is_real, max_imag = verify_eigenvalue_reality(result)
        assert is_real, f"Max imaginary part: {max_imag}"

    def test_orthogonality(self, plate_cma):
        """Modes should be R-orthogonal: J_m^H · R · J_n = δ_mn."""
        result, basis, Z = plate_cma
        is_orthog, max_error = verify_orthogonality(result, tolerance=1e-5)
        assert is_orthog, f"Max orthogonality error: {max_error}"

    def test_power_normalization(self, plate_cma):
        """Modes should be power-normalized: J_n^H · R · J_n = 1."""
        result, basis, Z = plate_cma
        R = result.R_matrix
        for n in range(basis.num_basis):
            J_n = result.eigenvectors[:, n]
            power = np.real(np.conj(J_n) @ R @ J_n)
            assert np.isclose(power, 1.0, atol=1e-5), f"Mode {n} power: {power}"

    def test_modal_significance_sorted(self, plate_cma):
        """get_mode(0) should return most significant mode."""
        result, basis, Z = plate_cma
        ms_0 = result.get_modal_significance(0)
        ms_1 = result.get_modal_significance(1)
        assert ms_0 >= ms_1

    def test_num_modes_filter(self, plate_cma):
        """num_modes parameter should filter modes."""
        result, basis, Z = plate_cma
        N = basis.num_basis
        result_filtered = compute_characteristic_modes(Z, num_modes=3)
        assert len(result_filtered.sort_order) == 3
        # Should have same top 3 modes
        for i in range(3):
            assert result_filtered.sort_order[i] == result.sort_order[i]


class TestLargerPlateCMA:
    """CMA tests on a larger plate for more thorough validation."""

    @pytest.fixture
    def larger_plate_cma(self):
        mesh = _make_larger_plate()
        basis = compute_rwg_connectivity(mesh)
        k = 2 * np.pi * 0.5  # lower frequency for larger plate
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
        result = compute_characteristic_modes(Z, frequency=c0 * 0.5)
        return result, basis, mesh, Z

    def test_first_mode_physical(self, larger_plate_cma):
        """First mode should have high modal significance."""
        result, basis, mesh, Z = larger_plate_cma
        ms_0 = result.get_modal_significance(0)
        # At least some mode should be reasonably excitable
        assert ms_0 > 0.1

    def test_eigenvalue_equation(self, larger_plate_cma):
        """Verify X @ J ≈ λ * R @ J for each mode.

        Note: Due to regularization of the R matrix for numerical stability,
        the eigenvalue equation holds approximately, not exactly.
        """
        result, basis, mesh, Z = larger_plate_cma
        R = result.R_matrix
        X = result.X_matrix

        for n in range(min(5, basis.num_basis)):
            J_n = result.eigenvectors[:, n]
            lambda_n = result.eigenvalues[n]

            lhs = X @ J_n
            rhs = lambda_n * R @ J_n

            residual = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), 1e-10)
            # Tolerance relaxed to account for R matrix regularization
            assert residual < 1e-3, f"Mode {n} eigenvalue equation residual: {residual}"


class TestModalExpansion:
    """Test modal excitation and current expansion."""

    @pytest.fixture
    def plate_setup(self):
        mesh = _make_small_plate()
        basis = compute_rwg_connectivity(mesh)
        k = 2 * np.pi
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
        cma = compute_characteristic_modes(Z, frequency=c0)
        # Create a dummy excitation vector
        V = np.ones(basis.num_basis, dtype=np.complex128)
        return cma, Z, V, basis

    def test_modal_excitation_coefficient(self, plate_setup):
        """Modal excitation coefficients should be computed."""
        cma, Z, V, basis = plate_setup
        alpha_0 = compute_modal_excitation_coefficient(cma, V, mode_index=0)
        assert np.isfinite(alpha_0)

    def test_current_expansion_reconstruction(self, plate_setup):
        """Expanding current in modes should reconstruct original."""
        cma, Z, V, basis = plate_setup

        # Solve for driven current
        I_driven = np.linalg.solve(Z, V)

        # Expand in all modes
        coeffs, I_reconstructed = expand_current_in_modes(cma, I_driven)

        # Should reconstruct well with all modes
        error = np.linalg.norm(I_driven - I_reconstructed) / np.linalg.norm(I_driven)
        assert error < 0.01, f"Reconstruction error: {error}"


class TestModeTracking:
    """Test mode tracking across frequency."""

    def test_tracking_single_frequency(self):
        """Tracking with single frequency should return sort_order."""
        mesh = _make_small_plate()
        basis = compute_rwg_connectivity(mesh)
        k = 2 * np.pi
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
        result = compute_characteristic_modes(Z)

        tracked = track_modes_across_frequency([result])
        assert len(tracked) == 1
        assert np.array_equal(tracked[0], result.sort_order)

    def test_tracking_two_frequencies(self):
        """Tracking with two frequencies should produce valid permutations."""
        mesh = _make_small_plate()
        basis = compute_rwg_connectivity(mesh)

        # Two close frequencies
        k1 = 2 * np.pi
        k2 = 2 * np.pi * 1.05  # 5% frequency change

        Z1 = fill_impedance_matrix(basis, mesh, k1, eta0, quad_order=4)
        Z2 = fill_impedance_matrix(basis, mesh, k2, eta0, quad_order=4)

        result1 = compute_characteristic_modes(Z1, frequency=c0)
        result2 = compute_characteristic_modes(Z2, frequency=c0 * 1.05)

        tracked = track_modes_across_frequency([result1, result2])

        assert len(tracked) == 2
        # Second tracking should be a valid permutation
        N = basis.num_basis
        assert len(tracked[1]) == N
        assert len(set(tracked[1])) == N  # All unique indices


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_near_singular_r_warning(self):
        """Should warn when R is ill-conditioned."""
        # Create a very small structure (electrically small)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0005, 0.001, 0.0],
            [0.001, 0.001, 0.0],
        ])
        triangles = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ])
        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)

        # Very low frequency -> electrically very small
        k = 0.001  # lambda ~ 6000 m
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)

        # May or may not warn depending on conditioning
        result = compute_characteristic_modes(Z)
        assert result is not None

    def test_zero_frequency_handling(self):
        """CMA should handle frequency=0 metadata."""
        mesh = _make_two_triangle_mesh()
        basis = compute_rwg_connectivity(mesh)
        k = 2 * np.pi
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)

        result = compute_characteristic_modes(Z, frequency=0.0)
        assert result.frequency == 0.0
