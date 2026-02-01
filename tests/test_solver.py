"""Tests for excitation, solver, and surface current modules."""

import numpy as np
import pytest

from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
from pyMoM3d.mom.impedance import fill_impedance_matrix
from pyMoM3d.mom.excitation import PlaneWaveExcitation, DeltaGapExcitation, find_nearest_edge
from pyMoM3d.mom.solver import solve_direct, solve_gmres
from pyMoM3d.mom.surface_current import evaluate_surface_current
from pyMoM3d.utils.constants import eta0, c0


def _make_small_plate():
    """2x2 grid plate."""
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
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6],
        [4, 5, 8], [4, 8, 7],
    ])
    mesh = Mesh(vertices, triangles)
    basis = compute_rwg_connectivity(mesh)
    return mesh, basis


class TestPlaneWaveExcitation:
    def test_voltage_vector_finite(self):
        mesh, basis = _make_small_plate()
        k = 2 * np.pi
        exc = PlaneWaveExcitation(
            E0=np.array([1.0, 0.0, 0.0]),
            k_hat=np.array([0.0, 0.0, -1.0]),
        )
        V = exc.compute_voltage_vector(basis, mesh, k)
        assert len(V) == basis.num_basis
        assert np.all(np.isfinite(V))

    def test_nonzero_voltage(self):
        mesh, basis = _make_small_plate()
        k = 2 * np.pi
        exc = PlaneWaveExcitation(
            E0=np.array([1.0, 0.0, 0.0]),
            k_hat=np.array([0.0, 0.0, -1.0]),
        )
        V = exc.compute_voltage_vector(basis, mesh, k)
        assert np.linalg.norm(V) > 0

    def test_invalid_e0_k_hat(self):
        with pytest.raises(ValueError):
            PlaneWaveExcitation(
                E0=np.array([1.0, 0.0, 0.0]),
                k_hat=np.array([1.0, 0.0, 0.0]),  # Parallel — invalid
            )


class TestDeltaGapExcitation:
    def test_single_nonzero(self):
        mesh, basis = _make_small_plate()
        exc = DeltaGapExcitation(basis_index=0, voltage=1.0)
        V = exc.compute_voltage_vector(basis, mesh, k=1.0)
        assert V[0] == 1.0
        assert np.count_nonzero(V) == 1


class TestSolve:
    @pytest.fixture
    def plate_system(self):
        mesh, basis = _make_small_plate()
        k = 2 * np.pi
        Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
        exc = PlaneWaveExcitation(
            E0=np.array([1.0, 0.0, 0.0]),
            k_hat=np.array([0.0, 0.0, -1.0]),
        )
        V = exc.compute_voltage_vector(basis, mesh, k)
        return Z, V, mesh, basis

    def test_direct_solve_residual(self, plate_system):
        Z, V, mesh, basis = plate_system
        I = solve_direct(Z, V)
        residual = np.linalg.norm(Z @ I - V) / np.linalg.norm(V)
        assert residual < 1e-10

    def test_gmres_solve(self, plate_system):
        Z, V, mesh, basis = plate_system
        I_direct = solve_direct(Z, V)
        I_gmres = solve_gmres(Z, V)
        # GMRES should produce similar result
        assert np.allclose(I_direct, I_gmres, rtol=1e-4)

    def test_currents_finite(self, plate_system):
        Z, V, mesh, basis = plate_system
        I = solve_direct(Z, V)
        assert np.all(np.isfinite(I))


class TestSurfaceCurrent:
    def test_evaluate(self):
        mesh, basis = _make_small_plate()
        I_coeffs = np.ones(basis.num_basis, dtype=np.complex128)
        centroid = np.mean(mesh.vertices[mesh.triangles[0]], axis=0)
        J = evaluate_surface_current(I_coeffs, basis, mesh, centroid.reshape(1, 3))
        assert J.shape == (1, 3)
        assert np.all(np.isfinite(J))


class TestFindNearestEdge:
    def test_finds_edge(self):
        mesh, basis = _make_small_plate()
        # Pick the midpoint of the first basis edge
        edge = mesh.edges[basis.edge_index[0]]
        mid = 0.5 * (mesh.vertices[edge[0]] + mesh.vertices[edge[1]])
        idx = find_nearest_edge(mesh, basis, mid)
        assert idx == 0
