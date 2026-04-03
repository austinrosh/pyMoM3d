"""Tests for loop-star decomposition: topology, divergence-free, conditioning."""

import numpy as np
import pytest

from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
from pyMoM3d.mesh.gmsh_mesher import GmshMesher
from pyMoM3d.mom.loop_star import (
    build_loop_star_basis,
    solve_loop_star,
    verify_divergence_free,
)
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators.efie import EFIEOperator
from pyMoM3d.mom.excitation import StripDeltaGapExcitation, find_feed_edges
from pyMoM3d.utils.constants import c0, eta0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plate_mesh():
    """Small rectangular plate mesh for topology tests."""
    mesher = GmshMesher(target_edge_length=0.3)
    mesh = mesher.mesh_plate(width=1.0, height=1.0)
    basis = compute_rwg_connectivity(mesh)
    return mesh, basis


@pytest.fixture
def feed_plate_mesh():
    """Plate with a feed line for impedance tests."""
    mesher = GmshMesher(target_edge_length=0.05)
    mesh = mesher.mesh_plate_with_feed(width=0.5, height=0.02, feed_x=0.0)
    basis = compute_rwg_connectivity(mesh)
    return mesh, basis


# ---------------------------------------------------------------------------
# Topology tests
# ---------------------------------------------------------------------------

class TestTopology:
    """Verify P matrix dimensions, rank, and Euler relations."""

    def test_counts_match_euler(self, plate_mesh):
        """n_loops + n_stars = N_basis, n_stars = N_t - 1."""
        mesh, basis = plate_mesh
        P, n_loops = build_loop_star_basis(basis, mesh)

        N = basis.num_basis
        N_t = len(mesh.triangles)
        n_stars = N - n_loops

        assert n_stars == N_t - 1, (
            f"n_stars={n_stars} should equal N_t-1={N_t - 1}"
        )
        assert n_loops == N - N_t + 1, (
            f"n_loops={n_loops} should equal N-N_t+1={N - N_t + 1}"
        )

    def test_P_shape_square(self, plate_mesh):
        """P is a square N x N matrix."""
        mesh, basis = plate_mesh
        P, _ = build_loop_star_basis(basis, mesh)
        N = basis.num_basis
        assert P.shape == (N, N)

    def test_P_full_rank(self, plate_mesh):
        """P must be invertible (full rank)."""
        mesh, basis = plate_mesh
        P, _ = build_loop_star_basis(basis, mesh)
        rank = np.linalg.matrix_rank(P.toarray())
        assert rank == basis.num_basis, (
            f"P rank {rank} != N_basis {basis.num_basis}"
        )

    def test_P_columns_normalized(self, plate_mesh):
        """All columns of P should have unit L2 norm."""
        mesh, basis = plate_mesh
        P, _ = build_loop_star_basis(basis, mesh)
        from scipy.sparse.linalg import norm as sp_norm
        col_norms = sp_norm(P, axis=0)
        np.testing.assert_allclose(col_norms, 1.0, atol=1e-12)

    def test_disconnected_mesh_raises(self):
        """Disconnected mesh (dual graph not connected) should raise."""
        # Two separate triangles sharing no edges
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [5, 0, 0], [6, 0, 0], [5, 1, 0],
        ], dtype=np.float64)
        triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)

        if basis.num_basis == 0:
            pytest.skip("No RWG basis functions (no interior edges)")
        with pytest.raises(ValueError, match="disconnected"):
            build_loop_star_basis(basis, mesh)


# ---------------------------------------------------------------------------
# Divergence-free tests
# ---------------------------------------------------------------------------

class TestDivergenceFree:
    """Loop columns of P must be divergence-free."""

    def test_divergence_free_plate(self, plate_mesh):
        mesh, basis = plate_mesh
        P, n_loops = build_loop_star_basis(basis, mesh)
        assert verify_divergence_free(P, n_loops, basis, mesh), (
            "Loop functions are not divergence-free"
        )

    def test_divergence_free_fine_mesh(self):
        """Divergence-free on a finer mesh with more basis functions."""
        mesher = GmshMesher(target_edge_length=0.15)
        mesh = mesher.mesh_plate(width=1.0, height=1.0)
        basis = compute_rwg_connectivity(mesh)
        P, n_loops = build_loop_star_basis(basis, mesh)
        assert verify_divergence_free(P, n_loops, basis, mesh)


# ---------------------------------------------------------------------------
# Solve consistency tests
# ---------------------------------------------------------------------------

class TestSolveConsistency:
    """Loop-star solve must match standard solve at all frequencies."""

    def test_halfwave_dipole(self, feed_plate_mesh):
        """At half-wave resonance (kD ~ pi), both solves agree."""
        mesh, basis = feed_plate_mesh
        feed = find_feed_edges(mesh, basis, feed_x=0.0)
        if not feed:
            pytest.skip("No feed edges found")

        freq = c0 / (2.0 * 0.5)  # f for lambda = 1m, L = 0.5m
        k = 2 * np.pi * freq / c0

        op = EFIEOperator()
        Z = fill_matrix(op, basis, mesh, k, eta0, quad_order=4, backend='auto')
        exc = StripDeltaGapExcitation(feed_basis_indices=feed, voltage=1.0)
        V = exc.compute_voltage_vector(basis, mesh, k)

        I_std = np.linalg.solve(Z, V)
        P, n_loops = build_loop_star_basis(basis, mesh)
        I_ls = solve_loop_star(Z, V, P, n_loops)

        # Current coefficients should match to high precision
        np.testing.assert_allclose(I_ls, I_std, rtol=1e-8)

    def test_low_frequency(self, feed_plate_mesh):
        """At low frequency (kD << 1), loop-star gives same Z_in."""
        mesh, basis = feed_plate_mesh
        feed = find_feed_edges(mesh, basis, feed_x=0.0)
        if not feed:
            pytest.skip("No feed edges found")

        freq = 1e6  # Very low frequency
        k = 2 * np.pi * freq / c0

        op = EFIEOperator()
        Z = fill_matrix(op, basis, mesh, k, eta0, quad_order=4, backend='auto')
        exc = StripDeltaGapExcitation(feed_basis_indices=feed, voltage=1.0)
        V = exc.compute_voltage_vector(basis, mesh, k)

        I_std = np.linalg.solve(Z, V)
        Z_in_std = 1.0 / sum(
            I_std[m] * basis.edge_length[m] for m in feed
        )

        P, n_loops = build_loop_star_basis(basis, mesh)
        I_ls = solve_loop_star(Z, V, P, n_loops)
        Z_in_ls = 1.0 / sum(
            I_ls[m] * basis.edge_length[m] for m in feed
        )

        assert abs(Z_in_ls - Z_in_std) / abs(Z_in_std) < 1e-6, (
            f"Z_in mismatch: std={Z_in_std}, loop-star={Z_in_ls}"
        )


# ---------------------------------------------------------------------------
# Conditioning tests
# ---------------------------------------------------------------------------

class TestConditioning:
    """Loop-star should improve conditioning at low kD."""

    def test_conditioning_improvement_low_freq(self):
        """Condition number should improve at kD << 1."""
        mesher = GmshMesher(target_edge_length=0.2e-3)
        mesh = mesher.mesh_plate(width=1e-3, height=0.5e-3)
        basis = compute_rwg_connectivity(mesh)

        freq = 1e6
        k = 2 * np.pi * freq / c0

        op = EFIEOperator()
        Z = fill_matrix(op, basis, mesh, k, eta0, quad_order=4, backend='auto')
        P, n_loops = build_loop_star_basis(basis, mesh)

        from pyMoM3d.mom.loop_star import _compute_block_rescaling

        Z_ls = np.asarray(P.T @ Z @ P)
        D = _compute_block_rescaling(Z_ls, n_loops)
        Z_sc = D[:, None] * Z_ls * D[None, :]

        cond_std = np.linalg.cond(Z)
        cond_ls = np.linalg.cond(Z_sc)

        assert cond_ls < cond_std, (
            f"Loop-star cond ({cond_ls:.2e}) should be less than "
            f"standard cond ({cond_std:.2e}) at low kD"
        )

    def test_no_rescale_still_works(self, feed_plate_mesh):
        """solve_loop_star with rescale=False gives correct result."""
        mesh, basis = feed_plate_mesh
        feed = find_feed_edges(mesh, basis, feed_x=0.0)
        if not feed:
            pytest.skip("No feed edges found")

        freq = 300e6
        k = 2 * np.pi * freq / c0

        op = EFIEOperator()
        Z = fill_matrix(op, basis, mesh, k, eta0, quad_order=4, backend='auto')
        exc = StripDeltaGapExcitation(feed_basis_indices=feed, voltage=1.0)
        V = exc.compute_voltage_vector(basis, mesh, k)

        I_std = np.linalg.solve(Z, V)
        P, n_loops = build_loop_star_basis(basis, mesh)
        I_ls = solve_loop_star(Z, V, P, n_loops, rescale=False)

        np.testing.assert_allclose(I_ls, I_std, rtol=1e-8)


# ---------------------------------------------------------------------------
# Multi-RHS test
# ---------------------------------------------------------------------------

class TestMultiRHS:
    """solve_loop_star handles (N, P) RHS matrices correctly."""

    def test_multi_rhs(self, feed_plate_mesh):
        mesh, basis = feed_plate_mesh
        freq = 300e6
        k = 2 * np.pi * freq / c0

        op = EFIEOperator()
        Z = fill_matrix(op, basis, mesh, k, eta0, quad_order=4, backend='auto')

        N = basis.num_basis
        V_all = np.random.randn(N, 3) + 1j * np.random.randn(N, 3)
        I_std = np.linalg.solve(Z, V_all)

        P, n_loops = build_loop_star_basis(basis, mesh)
        I_ls = solve_loop_star(Z, V_all, P, n_loops)

        np.testing.assert_allclose(I_ls, I_std, rtol=1e-8)
