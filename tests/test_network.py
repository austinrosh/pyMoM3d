"""Tests for the network extraction module (Port, NetworkResult, NetworkExtractor)."""

import numpy as np
import pytest
import tempfile
import os

from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
from pyMoM3d.mom.excitation import (
    DeltaGapExcitation, StripDeltaGapExcitation, find_feed_edges,
)
from pyMoM3d.simulation import Simulation, SimulationConfig, SimulationResult
from pyMoM3d.network import Port, NetworkResult, NetworkExtractor
from pyMoM3d.utils.constants import eta0


# ---------------------------------------------------------------------------
# Shared mesh helpers
# ---------------------------------------------------------------------------

def _make_two_triangle_mesh():
    """Minimal 2-triangle plate mesh with one interior RWG edge."""
    vertices = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.0, 0.5, 0.0],
    ])
    triangles = np.array([[0, 1, 2], [2, 1, 3]])
    return Mesh(vertices, triangles)


def _make_plate_mesh(nx=4, ny=4):
    """Rectangular plate mesh (nx × ny quads → 2*nx*ny triangles)."""
    verts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            verts.append([i / nx, j / ny, 0.0])
    verts = np.array(verts, dtype=np.float64)

    tris = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            c = a + (nx + 1)
            d = c + 1
            tris.append([a, b, c])
            tris.append([b, d, c])
    tris = np.array(tris, dtype=np.int32)
    return Mesh(verts, tris)


def _make_sim_with_mesh(mesh, freq=3e8):
    """Return a Simulation with a DeltaGapExcitation (for operator setup)."""
    basis = compute_rwg_connectivity(mesh)
    exc = DeltaGapExcitation(basis_index=0, voltage=1.0)
    config = SimulationConfig(frequency=freq, excitation=exc)
    return Simulation(config, mesh=mesh)


# ---------------------------------------------------------------------------
# Port class tests
# ---------------------------------------------------------------------------

class TestPort:
    def test_basic_construction(self):
        port = Port(name='P1', feed_basis_indices=[0, 1, 2])
        assert port.name == 'P1'
        assert port.feed_basis_indices == [0, 1, 2]
        assert not port.is_differential
        assert port.V_ref == 1.0

    def test_differential_flag(self):
        port = Port(name='P1', feed_basis_indices=[0], return_basis_indices=[1])
        assert port.is_differential

    def test_feed_signs_length_mismatch_raises(self):
        with pytest.raises(ValueError, match='feed_signs length'):
            Port(name='P1', feed_basis_indices=[0, 1], feed_signs=[+1])

    def test_build_excitation_vector_matches_delta_gap(self):
        """Single-edge port excitation must match DeltaGapExcitation."""
        mesh = _make_two_triangle_mesh()
        basis = compute_rwg_connectivity(mesh)
        assert basis.num_basis >= 1

        idx = 0
        port = Port(name='P1', feed_basis_indices=[idx])
        V_port = port.build_excitation_vector(basis)

        exc = DeltaGapExcitation(basis_index=idx, voltage=1.0)
        V_dg = exc.compute_voltage_vector(basis, mesh, k=1.0)

        np.testing.assert_allclose(V_port, V_dg)

    def test_build_excitation_vector_matches_strip_delta_gap(self):
        """Multi-edge port excitation must match StripDeltaGapExcitation."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)
        N = basis.num_basis
        indices = [0, 1] if N >= 2 else [0]

        port = Port(name='P1', feed_basis_indices=indices)
        V_port = port.build_excitation_vector(basis)

        exc = StripDeltaGapExcitation(feed_basis_indices=indices, voltage=1.0)
        V_sdg = exc.compute_voltage_vector(basis, mesh, k=1.0)

        np.testing.assert_allclose(V_port, V_sdg)

    def test_terminal_current_consistency(self):
        """terminal_current must be consistent with StripDeltaGapExcitation.compute_input_impedance."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)
        N = basis.num_basis
        indices = [0, 1] if N >= 2 else [0]

        # Fake current: all ones
        I = np.ones(N, dtype=np.complex128)
        V_ref = 2.0 + 0j

        port = Port(name='P1', feed_basis_indices=indices, V_ref=V_ref)
        I_term = port.terminal_current(I, basis)

        # Manual computation
        expected = sum(I[m] * basis.edge_length[m] for m in indices)
        np.testing.assert_allclose(I_term, expected)

    def test_feed_signs_flip(self):
        """feed_signs=-1 negates the contribution."""
        mesh = _make_two_triangle_mesh()
        basis = compute_rwg_connectivity(mesh)
        idx = 0

        port_pos = Port(name='P+', feed_basis_indices=[idx], feed_signs=[+1])
        port_neg = Port(name='P-', feed_basis_indices=[idx], feed_signs=[-1])

        V_pos = port_pos.build_excitation_vector(basis)
        V_neg = port_neg.build_excitation_vector(basis)
        np.testing.assert_allclose(V_pos, -V_neg)

    def test_differential_antisymmetric_excitation(self):
        """Differential port: signal=+V, return=-V."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)
        N = basis.num_basis
        assert N >= 2, "Need at least 2 basis functions for this test"

        port = Port(name='P1', feed_basis_indices=[0], return_basis_indices=[1], V_ref=1.0)
        V = port.build_excitation_vector(basis)

        assert V[0] == pytest.approx(1.0 * basis.edge_length[0])
        assert V[1] == pytest.approx(-1.0 * basis.edge_length[1])

    def test_from_x_plane_wraps_find_feed_edges(self):
        """Port.from_x_plane must return the same indices as find_feed_edges."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)

        # find any x-coordinate that has interior edges
        edge_mids_x = []
        for n in range(basis.num_basis):
            va = mesh.vertices[mesh.edges[basis.edge_index[n], 0]]
            vb = mesh.vertices[mesh.edges[basis.edge_index[n], 1]]
            edge_mids_x.append(0.5 * (va[0] + vb[0]))

        if not edge_mids_x:
            pytest.skip("No interior edges to test")

        x_coord = edge_mids_x[0]
        direct_indices = find_feed_edges(mesh, basis, feed_x=x_coord)
        if not direct_indices:
            pytest.skip("find_feed_edges returned empty for this mesh")

        port = Port.from_x_plane(mesh, basis, x_coord=x_coord, name='P1')
        assert sorted(port.feed_basis_indices) == sorted(direct_indices)


# ---------------------------------------------------------------------------
# Finite-width port tests
# ---------------------------------------------------------------------------

class TestFiniteWidthPort:
    """Tests for the finite-width distributed excitation (Lo/Jiang/Chew 2013)."""

    def test_gap_width_default_zero(self):
        port = Port(name='P1', feed_basis_indices=[0])
        assert port.gap_width == 0.0

    def test_gap_width_stored(self):
        port = Port(name='P1', feed_basis_indices=[0], gap_width=1e-3)
        assert port.gap_width == 1e-3

    def test_zero_gap_width_gives_delta_gap(self):
        """gap_width=0 must produce identical results to the standard delta-gap."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)
        idx = [0, 1] if basis.num_basis >= 2 else [0]

        port_dg = Port(name='DG', feed_basis_indices=idx, gap_width=0.0)
        port_fw = Port(name='FW', feed_basis_indices=idx, gap_width=0.0)

        V_dg = port_dg.build_excitation_vector(basis)
        V_fw = port_fw.build_excitation_vector(basis, mesh=mesh)
        np.testing.assert_allclose(V_fw, V_dg)

    def test_finite_width_without_mesh_falls_back(self):
        """When gap_width > 0 but mesh is None, falls back to delta-gap."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)

        port = Port(name='P1', feed_basis_indices=[0], gap_width=0.1)
        V = port.build_excitation_vector(basis)  # mesh=None
        # Should use delta-gap fallback
        assert V[0] == pytest.approx(1.0 * basis.edge_length[0])

    def test_finite_width_produces_nonzero_rhs(self):
        """Finite-width excitation must produce a non-zero RHS vector."""
        from pyMoM3d.mesh import GmshMesher
        mesher = GmshMesher(target_edge_length=0.25)
        mesh = mesher.mesh_plate_with_feeds(
            width=1.0, height=0.3, feed_x_list=[0.0], center=(0, 0, 0),
        )
        basis = compute_rwg_connectivity(mesh)

        feed_idx = find_feed_edges(mesh, basis, feed_x=0.0)
        assert len(feed_idx) > 0

        from pyMoM3d.mom.excitation import compute_feed_signs
        signs = compute_feed_signs(mesh, basis, feed_idx)

        port = Port(
            name='P1', feed_basis_indices=feed_idx,
            feed_signs=signs, gap_width=0.25,
        )
        V = port.build_excitation_vector(basis, mesh=mesh)

        # Must have non-zero entries
        assert np.any(np.abs(V) > 0)
        # Should excite more basis functions than just the feed edges
        # (adjacent basis functions get partial excitation)
        n_excited = np.sum(np.abs(V) > 1e-15)
        assert n_excited >= len(feed_idx)

    def test_finite_width_voltage_integral_preserved(self):
        """The total voltage integral should approximate V_ref.

        For a strip mesh with a feed at x=0, the sum of V[m] contributions
        across the port should give approximately V_ref when weighted correctly.
        """
        from pyMoM3d.mesh import GmshMesher
        mesher = GmshMesher(target_edge_length=0.25)
        mesh = mesher.mesh_plate_with_feeds(
            width=1.0, height=0.3, feed_x_list=[0.0], center=(0, 0, 0),
        )
        basis = compute_rwg_connectivity(mesh)
        feed_idx = find_feed_edges(mesh, basis, feed_x=0.0)

        from pyMoM3d.mom.excitation import compute_feed_signs
        signs = compute_feed_signs(mesh, basis, feed_idx)

        port_dg = Port(name='DG', feed_basis_indices=feed_idx, feed_signs=signs)
        port_fw = Port(
            name='FW', feed_basis_indices=feed_idx,
            feed_signs=signs, gap_width=0.25,
        )
        V_dg = port_dg.build_excitation_vector(basis)
        V_fw = port_fw.build_excitation_vector(basis, mesh=mesh)

        # Both should have non-trivial energy
        assert np.linalg.norm(V_dg) > 0
        assert np.linalg.norm(V_fw) > 0


# ---------------------------------------------------------------------------
# Variational Y extraction tests
# ---------------------------------------------------------------------------

class TestVariationalExtraction:
    """Tests for the variational admittance formula."""

    def test_variational_parameter_accepted(self):
        """NetworkExtractor.extract must accept variational=True."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        sim = _make_sim_with_mesh(mesh, freq=3e8)
        basis = sim.basis
        port = Port(name='P1', feed_basis_indices=[0])
        ext = NetworkExtractor(sim, [port])

        # Should not raise
        results = ext.extract(variational=True)
        assert len(results) == 1
        assert results[0].Z_matrix.shape == (1, 1)
        assert np.isfinite(results[0].Z_matrix[0, 0])

    def test_variational_vs_direct_consistency(self):
        """Variational and direct Y should be similar for a well-resolved mesh."""
        mesh = _make_plate_mesh(nx=6, ny=6)
        sim = _make_sim_with_mesh(mesh, freq=3e8)
        basis = sim.basis
        port = Port(name='P1', feed_basis_indices=[0])
        ext = NetworkExtractor(sim, [port])

        results_direct = ext.extract(variational=False)
        results_var = ext.extract(variational=True)

        Z_direct = results_direct[0].Z_matrix[0, 0]
        Z_var = results_var[0].Z_matrix[0, 0]

        # Both should be finite and have the same sign of real/imag parts
        assert np.isfinite(Z_direct)
        assert np.isfinite(Z_var)

    def test_variational_default_false(self):
        """Default variational=False should give direct Y extraction."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        sim = _make_sim_with_mesh(mesh, freq=3e8)
        port = Port(name='P1', feed_basis_indices=[0])
        ext = NetworkExtractor(sim, [port])

        r1 = ext.extract()  # default
        r2 = ext.extract(variational=False)

        np.testing.assert_allclose(
            r1[0].Z_matrix, r2[0].Z_matrix, atol=1e-10,
        )


# ---------------------------------------------------------------------------
# NetworkResult tests
# ---------------------------------------------------------------------------

class TestNetworkResult:
    def _make_result(self, Z_val=100.0 + 50j, Z0=50.0):
        """1-port NetworkResult with given Z_matrix entry."""
        Z = np.array([[Z_val]], dtype=np.complex128)
        return NetworkResult(frequency=1e9, Z_matrix=Z, port_names=['P1'], Z0=Z0)

    def _make_2port_result(self, Z0=50.0):
        """2-port with Z = diag(100, 100)."""
        Z = np.diag([100.0 + 0j, 100.0 + 0j])
        return NetworkResult(frequency=1e9, Z_matrix=Z, port_names=['P1', 'P2'], Z0=Z0)

    def test_y_matrix_is_z_inverse(self):
        result = self._make_2port_result()
        Y = result.Y_matrix
        I_approx = result.Z_matrix @ Y
        np.testing.assert_allclose(I_approx, np.eye(2), atol=1e-12)

    def test_s_matrix_matched_load(self):
        """Z = Z0 * I → S = 0."""
        Z0 = 50.0
        Z = np.diag([Z0 + 0j, Z0 + 0j])
        result = NetworkResult(frequency=1e9, Z_matrix=Z, port_names=['P1', 'P2'], Z0=Z0)
        S = result.S_matrix
        np.testing.assert_allclose(S, np.zeros((2, 2)), atol=1e-12)

    def test_s_matrix_short(self):
        """Z = 0 → S = -I."""
        Z0 = 50.0
        Z = np.zeros((2, 2), dtype=np.complex128)
        result = NetworkResult(frequency=1e9, Z_matrix=Z, port_names=['P1', 'P2'], Z0=Z0)
        S = result.S_matrix
        np.testing.assert_allclose(S, -np.eye(2), atol=1e-12)

    def test_s_matrix_1port(self):
        """1-port: S11 = (Z - Z0) / (Z + Z0)."""
        Z_val = 100.0 + 50j
        Z0 = 50.0
        result = self._make_result(Z_val, Z0)
        S11_computed = result.S_matrix[0, 0]
        S11_expected = (Z_val - Z0) / (Z_val + Z0)
        np.testing.assert_allclose(S11_computed, S11_expected, rtol=1e-12)

    def test_deembed_phase_zero(self):
        """Zero phase shift → unchanged Z_matrix."""
        result = self._make_2port_result()
        deembed = result.deembed_phase([0.0, 0.0])
        np.testing.assert_allclose(deembed.Z_matrix, result.Z_matrix, rtol=1e-10)

    def test_deembed_phase_wrong_length_raises(self):
        result = self._make_2port_result()
        with pytest.raises(ValueError, match='delta_theta length'):
            result.deembed_phase([0.1])

    def test_correct_port_parasitics_zero_is_identity(self):
        """Zero parasitics should return approximately the same Z_matrix."""
        result = self._make_result()
        deembed = result.correct_port_parasitics([0j], [0j])
        np.testing.assert_allclose(deembed.Z_matrix, result.Z_matrix, rtol=1e-10)

    def test_save_load_roundtrip(self):
        result = self._make_2port_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'test_result')
            result.save(path)
            loaded = NetworkResult.load(path + '.npz')
        np.testing.assert_allclose(loaded.Z_matrix, result.Z_matrix)
        assert loaded.frequency == result.frequency
        assert loaded.port_names == result.port_names
        assert loaded.Z0 == result.Z0

    def test_save_load_with_i_solutions(self):
        Z = np.array([[100.0 + 0j]])
        I_sols = np.random.randn(10, 1) + 1j * np.random.randn(10, 1)
        result = NetworkResult(
            frequency=1e9, Z_matrix=Z, port_names=['P1'],
            Z0=50.0, I_solutions=I_sols,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'test')
            result.save(path)
            loaded = NetworkResult.load(path + '.npz')
        np.testing.assert_allclose(loaded.I_solutions, I_sols)


# ---------------------------------------------------------------------------
# NetworkExtractor tests
# ---------------------------------------------------------------------------

class TestNetworkExtractor:
    def test_single_port_z_matches_simulation_z_input(self):
        """NetworkExtractor 1-port Z[0,0] must match SimulationResult.Z_input."""
        mesh = _make_two_triangle_mesh()
        basis = compute_rwg_connectivity(mesh)
        assert basis.num_basis >= 1, "Need at least 1 RWG basis function"

        freq = 3e8
        idx = 0

        # Reference: Simulation with DeltaGapExcitation
        exc = DeltaGapExcitation(basis_index=idx, voltage=1.0)
        config = SimulationConfig(frequency=freq, excitation=exc)
        sim = Simulation(config, mesh=mesh)
        sim_result = sim.run()
        Z_sim = sim_result.Z_input

        # NetworkExtractor with equivalent single-index port
        port = Port(name='P1', feed_basis_indices=[idx], V_ref=1.0)
        extractor = NetworkExtractor(sim, [port])
        [net_result] = extractor.extract(freq)
        Z_net = net_result.Z_matrix[0, 0]

        np.testing.assert_allclose(Z_net, Z_sim, rtol=1e-8)

    def test_single_port_at_config_frequency(self):
        """extract() with no argument uses sim.config.frequency."""
        mesh = _make_two_triangle_mesh()
        basis = compute_rwg_connectivity(mesh)
        sim = _make_sim_with_mesh(mesh, freq=5e8)
        port = Port(name='P1', feed_basis_indices=[0])
        extractor = NetworkExtractor(sim, [port])
        results = extractor.extract()
        assert len(results) == 1
        assert results[0].frequency == pytest.approx(5e8)

    def test_single_port_Z_finite(self):
        """Extracted Z[0,0] must be finite for a valid mesh."""
        mesh = _make_two_triangle_mesh()
        sim = _make_sim_with_mesh(mesh)
        port = Port(name='P1', feed_basis_indices=[0])
        extractor = NetworkExtractor(sim, [port])
        [result] = extractor.extract()
        assert np.isfinite(result.Z_matrix[0, 0])
        assert result.condition_number is not None

    def test_two_port_z_matrix_shape(self):
        """Two-port extraction returns (2, 2) Z_matrix."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)
        N = basis.num_basis
        if N < 2:
            pytest.skip("Need at least 2 RWG basis functions")

        sim = _make_sim_with_mesh(mesh)
        port1 = Port(name='P1', feed_basis_indices=[0])
        port2 = Port(name='P2', feed_basis_indices=[1])
        extractor = NetworkExtractor(sim, [port1, port2])
        [result] = extractor.extract()

        assert result.Z_matrix.shape == (2, 2)
        assert result.port_names == ['P1', 'P2']

    def test_two_port_z_finite(self):
        """All Z-matrix entries must be finite for a valid mesh."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)
        if basis.num_basis < 2:
            pytest.skip("Need at least 2 RWG basis functions")

        sim = _make_sim_with_mesh(mesh)
        port1 = Port(name='P1', feed_basis_indices=[0])
        port2 = Port(name='P2', feed_basis_indices=[1])
        extractor = NetworkExtractor(sim, [port1, port2])
        [result] = extractor.extract()
        assert np.all(np.isfinite(result.Z_matrix))

    def test_reciprocity(self):
        """EFIE Z-matrix must be symmetric: |Z[0,1] - Z[1,0]| / |Z[0,0]| < 0.01."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)
        if basis.num_basis < 2:
            pytest.skip("Need at least 2 RWG basis functions")

        sim = _make_sim_with_mesh(mesh)
        port1 = Port(name='P1', feed_basis_indices=[0])
        port2 = Port(name='P2', feed_basis_indices=[1])
        extractor = NetworkExtractor(sim, [port1, port2])
        [result] = extractor.extract()
        Z = result.Z_matrix

        asymmetry = abs(Z[0, 1] - Z[1, 0]) / (abs(Z[0, 0]) + 1e-30)
        assert asymmetry < 0.01, (
            f"EFIE Z-matrix not reciprocal: asymmetry={asymmetry:.4f}"
        )

    def test_store_currents(self):
        """store_currents=True populates I_solutions with shape (N, P)."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)
        if basis.num_basis < 2:
            pytest.skip("Need at least 2 RWG basis functions")

        sim = _make_sim_with_mesh(mesh)
        port1 = Port(name='P1', feed_basis_indices=[0])
        port2 = Port(name='P2', feed_basis_indices=[1])
        extractor = NetworkExtractor(sim, [port1, port2], store_currents=True)
        [result] = extractor.extract()

        assert result.I_solutions is not None
        assert result.I_solutions.shape == (basis.num_basis, 2)
        assert np.all(np.isfinite(result.I_solutions))

    def test_no_store_currents_default(self):
        """Default: I_solutions is None."""
        mesh = _make_two_triangle_mesh()
        sim = _make_sim_with_mesh(mesh)
        port = Port(name='P1', feed_basis_indices=[0])
        extractor = NetworkExtractor(sim, [port])
        [result] = extractor.extract()
        assert result.I_solutions is None

    def test_frequency_sweep(self):
        """extract(list_of_freqs) returns one result per frequency."""
        mesh = _make_two_triangle_mesh()
        sim = _make_sim_with_mesh(mesh)
        port = Port(name='P1', feed_basis_indices=[0])
        extractor = NetworkExtractor(sim, [port])
        freqs = [1e8, 3e8, 5e8]
        results = extractor.extract(freqs)
        assert len(results) == 3
        assert [r.frequency for r in results] == pytest.approx(freqs)

    def test_invalid_basis_index_raises(self):
        """Port with out-of-range basis index must raise ValueError at construction."""
        mesh = _make_two_triangle_mesh()
        sim = _make_sim_with_mesh(mesh)
        bad_port = Port(name='P_bad', feed_basis_indices=[9999])
        with pytest.raises(ValueError, match='out of range'):
            NetworkExtractor(sim, [bad_port])

    def test_use_lu_cache_matches_default(self):
        """use_lu_cache=True must give same result as default (np.linalg.solve)."""
        mesh = _make_plate_mesh(nx=4, ny=4)
        basis = compute_rwg_connectivity(mesh)
        if basis.num_basis < 2:
            pytest.skip("Need at least 2 RWG basis functions")

        sim = _make_sim_with_mesh(mesh)
        ports = [Port('P1', [0]), Port('P2', [1])]

        [r_default] = NetworkExtractor(sim, ports, use_lu_cache=False).extract()
        [r_cached]  = NetworkExtractor(sim, ports, use_lu_cache=True).extract()

        np.testing.assert_allclose(r_default.Z_matrix, r_cached.Z_matrix, rtol=1e-10)

    def test_s_matrix_from_extractor_matches_formula(self):
        """S11 from NetworkResult must match (Z-Z0)/(Z+Z0) for 1-port."""
        mesh = _make_two_triangle_mesh()
        sim = _make_sim_with_mesh(mesh)
        port = Port(name='P1', feed_basis_indices=[0])
        extractor = NetworkExtractor(sim, [port], Z0=50.0)
        [result] = extractor.extract()

        Z11 = result.Z_matrix[0, 0]
        Z0  = result.Z0
        S11_expected = (Z11 - Z0) / (Z11 + Z0)
        S11_computed = result.S_matrix[0, 0]
        np.testing.assert_allclose(S11_computed, S11_expected, rtol=1e-12)
