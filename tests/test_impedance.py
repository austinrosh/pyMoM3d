"""Tests for EFIE impedance matrix fill."""

import numpy as np
import pytest

from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators import EFIEOperator
from pyMoM3d.utils.constants import eta0, c0


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


class TestTwoTriangle:
    """Z matrix for single basis function (1x1 matrix)."""

    def test_z11_is_scalar(self):
        mesh = _make_two_triangle_mesh()
        basis = compute_rwg_connectivity(mesh)
        assert basis.num_basis == 1

        k = 2 * np.pi  # lambda = 1 m
        Z = fill_matrix(EFIEOperator(), basis, mesh, k, eta0, quad_order=4)
        assert Z.shape == (1, 1)
        assert np.isfinite(Z[0, 0])

    def test_z11_positive_real_part(self):
        """Diagonal should have positive real part (radiation resistance)."""
        mesh = _make_two_triangle_mesh()
        basis = compute_rwg_connectivity(mesh)
        k = 2 * np.pi
        Z = fill_matrix(EFIEOperator(), basis, mesh, k, eta0, quad_order=4)
        assert Z[0, 0].real > 0


class TestSmallPlate:
    """Z matrix for a small plate with multiple basis functions."""

    @pytest.fixture
    def plate_result(self):
        mesh = _make_small_plate()
        basis = compute_rwg_connectivity(mesh)
        k = 2 * np.pi  # lambda = 1 m
        Z = fill_matrix(EFIEOperator(), basis, mesh, k, eta0, quad_order=4)
        return Z, basis

    def test_symmetric(self, plate_result):
        """Z_mn == Z_nm to machine precision."""
        Z, basis = plate_result
        assert np.allclose(Z, Z.T, atol=1e-12)

    def test_diagonal_positive_real(self, plate_result):
        """All diagonal elements should have positive real part."""
        Z, basis = plate_result
        for i in range(basis.num_basis):
            assert Z[i, i].real > 0, f"Z[{i},{i}] = {Z[i,i]}"

    def test_finite(self, plate_result):
        Z, basis = plate_result
        assert np.all(np.isfinite(Z))


class TestQuadConvergence:
    """Higher quad order should converge matrix elements."""

    def test_convergence(self):
        mesh = _make_two_triangle_mesh()
        basis = compute_rwg_connectivity(mesh)
        k = 2 * np.pi

        Z4 = fill_matrix(EFIEOperator(), basis, mesh, k, eta0, quad_order=4)
        Z7 = fill_matrix(EFIEOperator(), basis, mesh, k, eta0, quad_order=7)

        # Should be close but not identical
        assert np.isfinite(Z4[0, 0])
        assert np.isfinite(Z7[0, 0])
        # With singularity extraction, convergence should be much tighter
        rdiff = abs(Z4[0, 0] - Z7[0, 0]) / abs(Z7[0, 0])
        assert rdiff < 1.0, f"Quad convergence too slow: rdiff={rdiff}"


class TestConditionNumber:
    """Impedance matrix condition number should be reasonable."""

    def test_small_plate_condition_number(self):
        """Condition number of small plate Z should be < 1e6."""
        mesh = _make_small_plate()
        basis = compute_rwg_connectivity(mesh)
        k = 2 * np.pi  # lambda = 1 m
        Z = fill_matrix(EFIEOperator(), basis, mesh, k, eta0, quad_order=4)
        cond = np.linalg.cond(Z)
        assert cond < 1e6, f"Condition number too high: {cond:.2e}"
