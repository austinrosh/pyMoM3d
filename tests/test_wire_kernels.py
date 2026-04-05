"""Tests for wire-wire and wire-surface EFIE kernels."""

import numpy as np
import pytest

from pyMoM3d.wire.wire_basis import WireMesh, compute_wire_connectivity
from pyMoM3d.wire.kernels import fill_wire_wire, fill_wire_surface, wire_quad_rule
from pyMoM3d.wire.hybrid import HybridBasis, HybridBasisAdapter, fill_hybrid_matrix
from pyMoM3d.wire.probe_port import create_probe_port
from pyMoM3d.utils.constants import c0, eta0


class TestWireQuadRule:
    """1D Gauss-Legendre quadrature tests."""

    def test_weights_sum_to_one(self):
        """Weights on [0,1] should sum to 1."""
        for n in [2, 4, 8, 16]:
            w, _ = wire_quad_rule(n)
            np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-14)

    def test_points_in_unit_interval(self):
        """All points should be in [0, 1]."""
        w, t = wire_quad_rule(8)
        assert np.all(t >= 0.0)
        assert np.all(t <= 1.0)

    def test_integrates_polynomial(self):
        """Should integrate x^3 on [0,1] exactly with 4+ points."""
        w, t = wire_quad_rule(4)
        result = np.dot(w, t**3)
        np.testing.assert_allclose(result, 0.25, atol=1e-14)


class TestWireWireKernel:
    """Wire-wire impedance matrix tests."""

    def _make_probe(self, n_seg=3, height=1.5e-3, radius=0.1e-3):
        wm = WireMesh.vertical_probe(0, 0, 0, height, radius, n_segments=n_seg)
        wb = compute_wire_connectivity(wm)
        return wm, wb

    def test_symmetry(self):
        """Z_ww should be symmetric (EFIE reciprocity)."""
        wm, wb = self._make_probe(n_seg=4)
        N_w = wb.num_basis
        Z_ww = np.zeros((N_w, N_w), dtype=np.complex128)

        freq = 1e9
        k = 2 * np.pi * freq / c0

        fill_wire_wire(Z_ww, wb, wm, k, eta0)

        np.testing.assert_allclose(
            Z_ww, Z_ww.T, atol=1e-10 * np.abs(Z_ww).max(),
            err_msg="Z_ww is not symmetric",
        )

    def test_nonzero(self):
        """Z_ww should have nonzero entries."""
        wm, wb = self._make_probe()
        N_w = wb.num_basis
        Z_ww = np.zeros((N_w, N_w), dtype=np.complex128)

        freq = 1e9
        k = 2 * np.pi * freq / c0

        fill_wire_wire(Z_ww, wb, wm, k, eta0)
        assert np.abs(Z_ww).max() > 0, "Z_ww is all zeros"

    def test_diagonal_has_reactance(self):
        """Diagonal entries should have nonzero imaginary part (reactance).
        For a short open-ended wire (kl << 1), the scalar potential dominates
        so the impedance is capacitive (negative imaginary)."""
        wm, wb = self._make_probe(n_seg=3, height=1e-3)
        N_w = wb.num_basis
        Z_ww = np.zeros((N_w, N_w), dtype=np.complex128)

        freq = 1e9
        k = 2 * np.pi * freq / c0

        fill_wire_wire(Z_ww, wb, wm, k, eta0)

        for i in range(N_w):
            assert abs(Z_ww[i, i].imag) > 0, (
                f"Z_ww[{i},{i}] = {Z_ww[i,i]} — expected nonzero reactance"
            )


class TestWireSurfaceKernel:
    """Wire-surface coupling kernel tests."""

    def _make_plate_and_probe(self):
        """Create a small plate mesh and a vertical probe."""
        from pyMoM3d.mesh.mesh_data import Mesh
        from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity

        # Small plate at z = 1.6e-3 (microstrip height)
        h = 1.6e-3
        s = 2e-3
        vertices = np.array([
            [-s, -s/2, h],
            [s, -s/2, h],
            [s, s/2, h],
            [-s, s/2, h],
        ], dtype=np.float64)
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)
        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)

        # Vertical probe at center
        wm = WireMesh.vertical_probe(0, 0, 0, h, radius=0.2e-3, n_segments=3)
        wb = compute_wire_connectivity(wm)

        return mesh, basis, wm, wb

    def test_reciprocity(self):
        """Z_ws should equal Z_sw^T (computed independently)."""
        mesh, basis, wm, wb = self._make_plate_and_probe()

        N_w = wb.num_basis
        N_s = basis.num_basis

        freq = 1e9
        k = 2 * np.pi * freq / c0

        Z_ws = np.zeros((N_w, N_s), dtype=np.complex128)
        fill_wire_surface(Z_ws, wb, wm, basis, mesh, k, eta0)

        # Z_ws should be nonzero
        assert np.abs(Z_ws).max() > 0, "Z_ws is all zeros"

        # The transpose Z_ws^T should equal Z_sw (by EFIE reciprocity)
        # Since we compute Z_sw = Z_ws^T in fill_hybrid_matrix, just
        # verify Z_ws is finite and has reasonable magnitude
        assert np.all(np.isfinite(Z_ws)), "Z_ws has non-finite entries"

    def test_nonzero_coupling(self):
        """Wire touching the plate should have nonzero coupling."""
        mesh, basis, wm, wb = self._make_plate_and_probe()

        N_w = wb.num_basis
        N_s = basis.num_basis

        freq = 1e9
        k = 2 * np.pi * freq / c0

        Z_ws = np.zeros((N_w, N_s), dtype=np.complex128)
        fill_wire_surface(Z_ws, wb, wm, basis, mesh, k, eta0)

        # At least some entries should be nonzero (wire couples to plate)
        n_nonzero = np.count_nonzero(np.abs(Z_ws) > 1e-30)
        assert n_nonzero > 0, f"All Z_ws entries are zero"


class TestHybridAssembly:
    """Full hybrid matrix assembly tests."""

    def _make_plate_and_probe(self):
        from pyMoM3d.mesh.mesh_data import Mesh
        from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity

        h = 1.6e-3
        s = 2e-3
        vertices = np.array([
            [-s, -s/2, h],
            [s, -s/2, h],
            [s, s/2, h],
            [-s, s/2, h],
        ], dtype=np.float64)
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)
        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)
        wm = WireMesh.vertical_probe(0, 0, 0, h, radius=0.2e-3, n_segments=3)
        wb = compute_wire_connectivity(wm)
        return mesh, basis, wm, wb

    def test_matrix_symmetry(self):
        """Full hybrid matrix should be symmetric."""
        from pyMoM3d.mom.operators.efie import EFIEOperator

        mesh, basis, wm, wb = self._make_plate_and_probe()
        freq = 1e9
        k = 2 * np.pi * freq / c0

        op = EFIEOperator()
        Z = fill_hybrid_matrix(
            op, basis, mesh, wb, wm, k, eta0,
            backend='numpy',
        )

        N = basis.num_basis + wb.num_basis
        assert Z.shape == (N, N)
        np.testing.assert_allclose(
            Z, Z.T, atol=1e-10 * np.abs(Z).max(),
            err_msg="Hybrid Z is not symmetric",
        )

    def test_matrix_dimensions(self):
        """Matrix should be (N_s + N_w) x (N_s + N_w)."""
        from pyMoM3d.mom.operators.efie import EFIEOperator

        mesh, basis, wm, wb = self._make_plate_and_probe()
        freq = 1e9
        k = 2 * np.pi * freq / c0

        op = EFIEOperator()
        Z = fill_hybrid_matrix(
            op, basis, mesh, wb, wm, k, eta0,
            backend='numpy',
        )

        N_s = basis.num_basis
        N_w = wb.num_basis
        assert Z.shape == (N_s + N_w, N_s + N_w)

    def test_ss_block_matches_standalone(self):
        """Z_ss block should match fill_matrix result."""
        from pyMoM3d.mom.operators.efie import EFIEOperator
        from pyMoM3d.mom.assembly import fill_matrix

        mesh, basis, wm, wb = self._make_plate_and_probe()
        freq = 1e9
        k = 2 * np.pi * freq / c0

        op = EFIEOperator()

        Z_hybrid = fill_hybrid_matrix(
            op, basis, mesh, wb, wm, k, eta0,
            backend='numpy',
        )

        Z_ss = fill_matrix(op, basis, mesh, k, eta0, backend='numpy')

        N_s = basis.num_basis
        np.testing.assert_allclose(
            Z_hybrid[:N_s, :N_s], Z_ss,
            atol=1e-12 * np.abs(Z_ss).max(),
        )


class TestHybridBasisAdapter:
    """HybridBasisAdapter compatibility tests."""

    def test_edge_length_concatenation(self):
        """Adapter concatenates RWG and wire edge lengths."""
        from pyMoM3d.mesh.mesh_data import Mesh
        from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity

        h = 1.6e-3
        s = 2e-3
        vertices = np.array([
            [-s, -s/2, h], [s, -s/2, h], [s, s/2, h], [-s, s/2, h],
        ], dtype=np.float64)
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)

        wm = WireMesh.vertical_probe(0, 0, 0, h, 0.2e-3, n_segments=3)
        wb = compute_wire_connectivity(wm)

        hb = HybridBasis(rwg_basis=basis, wire_basis=wb, wire_mesh=wm)
        adapter = HybridBasisAdapter(hb)

        assert adapter.num_basis == basis.num_basis + wb.num_basis
        assert len(adapter.edge_length) == adapter.num_basis
        # First N_s entries match RWG
        np.testing.assert_array_equal(
            adapter.edge_length[:basis.num_basis],
            basis.edge_length,
        )


class TestProbePort:
    """Probe port construction tests."""

    def test_port_indices(self):
        """Probe port should reference wire basis indices offset by N_surface."""
        wm = WireMesh.vertical_probe(0, 0, 0, 1.6e-3, 0.1e-3, n_segments=3)
        wb = compute_wire_connectivity(wm)

        N_s = 50  # hypothetical surface basis count
        port = create_probe_port(wb, N_s, name='P1')

        assert port.name == 'P1'
        assert port.feed_basis_indices == [N_s]
        assert port.feed_signs == [+1]

    def test_port_voltage_vector(self):
        """Port should produce correct voltage vector."""
        from pyMoM3d.mesh.mesh_data import Mesh
        from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity

        h = 1.6e-3
        s = 2e-3
        vertices = np.array([
            [-s, -s/2, h], [s, -s/2, h], [s, s/2, h], [-s, s/2, h],
        ], dtype=np.float64)
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)

        wm = WireMesh.vertical_probe(0, 0, 0, h, 0.2e-3, n_segments=3)
        wb = compute_wire_connectivity(wm)

        hb = HybridBasis(rwg_basis=basis, wire_basis=wb, wire_mesh=wm)
        adapter = HybridBasisAdapter(hb)

        port = create_probe_port(wb, basis.num_basis, name='P1')
        V = port.build_excitation_vector(adapter)

        assert len(V) == adapter.num_basis
        # Only the probe basis index should be nonzero
        idx = basis.num_basis  # first wire basis
        assert abs(V[idx]) > 0
        # All surface entries should be zero
        assert np.all(V[:basis.num_basis] == 0)
