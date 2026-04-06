"""Tests for the non-radiating (half-RWG) port model.

Tests cover:
- Mesh splitting at port gaps
- Half-RWG boundary edge pair detection
- Half-RWG basis function extension
- Port creation from half-RWG basis
- Z-matrix assembly with half-RWG basis functions
- S-parameter extraction reciprocity and passivity
"""

import numpy as np
import pytest

from pyMoM3d.mesh import (
    GmshMesher,
    compute_rwg_connectivity,
    split_mesh_at_x,
    split_mesh_at_ports,
    add_half_rwg_basis,
)
from pyMoM3d.mesh.port_mesh import find_half_rwg_pairs
from pyMoM3d.network.port import Port


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def strip_mesh():
    """A 30mm x 3mm strip with conformal edges at x = -10mm and x = +10mm."""
    mesher = GmshMesher(target_edge_length=1e-3)
    mesh = mesher.mesh_plate_with_feeds(
        width=0.030,
        height=0.003,
        feed_x_list=[-0.010, 0.010],
        center=(0, 0, 0),
    )
    return mesh


@pytest.fixture
def split_strip(strip_mesh):
    """Strip mesh split at both port locations."""
    mesh, _ = split_mesh_at_ports(strip_mesh, [-0.010, 0.010])
    return mesh


# ------------------------------------------------------------------ #
# Mesh splitting tests
# ------------------------------------------------------------------ #

class TestMeshSplitting:

    def test_split_adds_vertices(self, strip_mesh):
        """Splitting at a port x-coordinate should add duplicate vertices."""
        n_orig = strip_mesh.get_num_vertices()
        split_mesh, remap = split_mesh_at_x(strip_mesh, -0.010)
        n_new = split_mesh.get_num_vertices()
        assert n_new > n_orig
        assert len(remap) > 0

    def test_split_preserves_triangles(self, strip_mesh):
        """Splitting should not change the number of triangles."""
        split_mesh, _ = split_mesh_at_x(strip_mesh, -0.010)
        assert split_mesh.get_num_triangles() == strip_mesh.get_num_triangles()

    def test_split_creates_boundary_edges(self, strip_mesh):
        """Splitting should create additional boundary edges at the cut."""
        rwg_before = compute_rwg_connectivity(strip_mesh)
        split_mesh, _ = split_mesh_at_x(strip_mesh, -0.010)
        rwg_after = compute_rwg_connectivity(split_mesh)
        # Splitting converts interior edges to boundary pairs
        assert rwg_after.num_boundary_edges > rwg_before.num_boundary_edges

    def test_split_reduces_interior_edges(self, strip_mesh):
        """Splitting should reduce interior edges (some become boundary)."""
        rwg_before = compute_rwg_connectivity(strip_mesh)
        split_mesh, _ = split_mesh_at_x(strip_mesh, -0.010)
        rwg_after = compute_rwg_connectivity(split_mesh)
        assert rwg_after.num_basis < rwg_before.num_basis

    def test_split_at_invalid_x_raises(self, strip_mesh):
        """Splitting at an x with no vertices should raise ValueError."""
        with pytest.raises(ValueError, match="no vertices found"):
            split_mesh_at_x(strip_mesh, 0.999)

    def test_split_at_multiple_ports(self, strip_mesh):
        """Splitting at two ports should work sequentially."""
        split_mesh, remaps = split_mesh_at_ports(strip_mesh, [-0.010, 0.010])
        assert len(remaps) == 2
        assert split_mesh.get_num_vertices() > strip_mesh.get_num_vertices()
        assert split_mesh.get_num_triangles() == strip_mesh.get_num_triangles()

    def test_duplicated_vertices_colocated(self, strip_mesh):
        """Duplicated vertices should be at the same geometric position."""
        split_mesh, remap = split_mesh_at_x(strip_mesh, -0.010)
        for old_idx, new_idx in remap.items():
            np.testing.assert_allclose(
                split_mesh.vertices[old_idx],
                split_mesh.vertices[new_idx],
                atol=1e-15,
            )


# ------------------------------------------------------------------ #
# Half-RWG pair detection tests
# ------------------------------------------------------------------ #

class TestHalfRWGPairs:

    def test_finds_pairs_at_port(self, split_strip):
        """Should find matched boundary edge pairs at each port gap."""
        rwg = compute_rwg_connectivity(split_strip)
        pairs = find_half_rwg_pairs(split_strip, rwg, -0.010)
        assert len(pairs) > 0

    def test_pairs_have_matching_edge_lengths(self, split_strip):
        """Left and right boundary edges in a pair should have equal length."""
        rwg = compute_rwg_connectivity(split_strip)
        pairs = find_half_rwg_pairs(split_strip, rwg, -0.010)
        for le_idx, le_tri, le_fv, re_idx, re_tri, re_fv in pairs:
            le_len = split_strip.edge_lengths[le_idx]
            re_len = split_strip.edge_lengths[re_idx]
            np.testing.assert_allclose(le_len, re_len, rtol=1e-10)

    def test_pairs_triangles_on_opposite_sides(self, split_strip):
        """T+ and T- should be on opposite sides of the port gap."""
        rwg = compute_rwg_connectivity(split_strip)
        pairs = find_half_rwg_pairs(split_strip, rwg, -0.010)
        for le_idx, le_tri, le_fv, re_idx, re_tri, re_fv in pairs:
            le_cx = split_strip.vertices[
                split_strip.triangles[le_tri]
            ].mean(axis=0)[0]
            re_cx = split_strip.vertices[
                split_strip.triangles[re_tri]
            ].mean(axis=0)[0]
            assert le_cx < -0.010 + 1e-6
            assert re_cx > -0.010 - 1e-6

    def test_no_pairs_away_from_port(self, split_strip):
        """Should find no pairs at an x-coordinate with no gap."""
        rwg = compute_rwg_connectivity(split_strip)
        pairs = find_half_rwg_pairs(split_strip, rwg, 0.0)
        assert len(pairs) == 0


# ------------------------------------------------------------------ #
# Half-RWG basis extension tests
# ------------------------------------------------------------------ #

class TestAddHalfRWG:

    def test_extends_basis_count(self, split_strip):
        """Adding half-RWGs should increase the basis count."""
        rwg = compute_rwg_connectivity(split_strip)
        n_before = rwg.num_basis
        ext, indices = add_half_rwg_basis(split_strip, rwg, -0.010)
        assert ext.num_basis == n_before + len(indices)
        assert len(indices) > 0

    def test_half_rwg_indices_are_appended(self, split_strip):
        """Half-RWG indices should start after existing basis functions."""
        rwg = compute_rwg_connectivity(split_strip)
        n_before = rwg.num_basis
        ext, indices = add_half_rwg_basis(split_strip, rwg, -0.010)
        for idx in indices:
            assert idx >= n_before

    def test_preserves_existing_basis(self, split_strip):
        """Existing RWG data should be unchanged after extension."""
        rwg = compute_rwg_connectivity(split_strip)
        ext, _ = add_half_rwg_basis(split_strip, rwg, -0.010)
        n = rwg.num_basis
        np.testing.assert_array_equal(ext.t_plus[:n], rwg.t_plus)
        np.testing.assert_array_equal(ext.t_minus[:n], rwg.t_minus)
        np.testing.assert_array_equal(ext.edge_length[:n], rwg.edge_length)

    def test_half_rwg_triangles_disconnected(self, split_strip):
        """T+ and T- of half-RWG should NOT share any vertices."""
        rwg = compute_rwg_connectivity(split_strip)
        ext, indices = add_half_rwg_basis(split_strip, rwg, -0.010)
        for idx in indices:
            tp_verts = set(int(v) for v in split_strip.triangles[ext.t_plus[idx]])
            tm_verts = set(int(v) for v in split_strip.triangles[ext.t_minus[idx]])
            assert tp_verts.isdisjoint(tm_verts), (
                f"Half-RWG {idx}: T+ and T- share vertices {tp_verts & tm_verts}"
            )

    def test_half_rwg_positive_areas(self, split_strip):
        """Half-RWG area_plus and area_minus should be positive."""
        rwg = compute_rwg_connectivity(split_strip)
        ext, indices = add_half_rwg_basis(split_strip, rwg, -0.010)
        for idx in indices:
            assert ext.area_plus[idx] > 0
            assert ext.area_minus[idx] > 0

    def test_sequential_extension_at_two_ports(self, split_strip):
        """Adding half-RWGs at two ports sequentially should work."""
        rwg = compute_rwg_connectivity(split_strip)
        ext1, idx1 = add_half_rwg_basis(split_strip, rwg, -0.010)
        ext2, idx2 = add_half_rwg_basis(split_strip, ext1, 0.010)
        assert ext2.num_basis == rwg.num_basis + len(idx1) + len(idx2)
        # Indices should not overlap
        assert set(idx1).isdisjoint(set(idx2))


# ------------------------------------------------------------------ #
# Port creation tests
# ------------------------------------------------------------------ #

class TestNonRadiatingPort:

    def test_creates_port(self, split_strip):
        """Should create a port with feed_basis_indices and feed_signs."""
        rwg = compute_rwg_connectivity(split_strip)
        ext, _ = add_half_rwg_basis(split_strip, rwg, -0.010)
        port = Port.from_nonradiating_gap(split_strip, ext, -0.010)
        assert len(port.feed_basis_indices) > 0
        assert len(port.feed_signs) == len(port.feed_basis_indices)

    def test_port_signs_consistent(self, split_strip):
        """All port signs should be +1 for +x current convention."""
        rwg = compute_rwg_connectivity(split_strip)
        ext, _ = add_half_rwg_basis(split_strip, rwg, -0.010)
        port = Port.from_nonradiating_gap(split_strip, ext, -0.010)
        # For left port, T+ is on the left side → sign should be +1
        for s in port.feed_signs:
            assert s == +1

    def test_port_at_wrong_location_raises(self, split_strip):
        """Port creation at a location without half-RWGs should raise."""
        rwg = compute_rwg_connectivity(split_strip)
        ext, _ = add_half_rwg_basis(split_strip, rwg, -0.010)
        with pytest.raises(ValueError, match="no half-RWG"):
            Port.from_nonradiating_gap(split_strip, ext, 0.005)

    def test_two_ports(self, split_strip):
        """Should create independent ports at both gap locations."""
        rwg = compute_rwg_connectivity(split_strip)
        ext, _ = add_half_rwg_basis(split_strip, rwg, -0.010)
        ext, _ = add_half_rwg_basis(split_strip, ext, 0.010)
        p1 = Port.from_nonradiating_gap(split_strip, ext, -0.010, name='P1')
        p2 = Port.from_nonradiating_gap(split_strip, ext, 0.010, name='P2')
        assert set(p1.feed_basis_indices).isdisjoint(set(p2.feed_basis_indices))


# ------------------------------------------------------------------ #
# Assembly and S-parameter tests
# ------------------------------------------------------------------ #

class TestHalfRWGAssembly:

    @pytest.fixture
    def two_port_system(self, split_strip):
        """Set up a complete 2-port extraction with half-RWG ports."""
        from pyMoM3d.mom.operators.efie import EFIEOperator
        from pyMoM3d.mom.assembly import fill_matrix
        from pyMoM3d.utils.constants import c0, eta0

        rwg = compute_rwg_connectivity(split_strip)
        ext, _ = add_half_rwg_basis(split_strip, rwg, -0.010)
        ext, _ = add_half_rwg_basis(split_strip, ext, 0.010)

        p1 = Port.from_nonradiating_gap(split_strip, ext, -0.010, name='P1')
        p2 = Port.from_nonradiating_gap(split_strip, ext, 0.010, name='P2')

        freq = 10e9
        k = 2 * np.pi * freq / c0

        op = EFIEOperator()
        Z = fill_matrix(op, ext, split_strip, k, eta0, backend='numpy')

        ports = [p1, p2]
        V_all = np.column_stack([p.build_excitation_vector(ext) for p in ports])
        I_all = np.linalg.solve(Z, V_all)

        P = 2
        Y = np.zeros((P, P), dtype=complex)
        for q in range(P):
            for p in range(P):
                Y[q, p] = ports[q].terminal_current(I_all[:, p], ext) / ports[p].V_ref

        Z_net = np.linalg.inv(Y)
        Z0 = 50.0
        I_eye = np.eye(P)
        S = (Z_net / Z0 - I_eye) @ np.linalg.inv(Z_net / Z0 + I_eye)

        return {'Z': Z, 'S': S, 'Z_net': Z_net, 'ext': ext}

    def test_z_matrix_symmetric(self, two_port_system):
        """EFIE impedance matrix should be symmetric."""
        Z = two_port_system['Z']
        err = np.max(np.abs(Z - Z.T)) / np.max(np.abs(Z))
        assert err < 1e-12

    def test_reciprocity(self, two_port_system):
        """S12 should equal S21 (reciprocity)."""
        S = two_port_system['S']
        assert abs(S[0, 1] - S[1, 0]) < 1e-10

    def test_passivity(self, two_port_system):
        """Sum of |S_ij|^2 per column should not exceed 1."""
        S = two_port_system['S']
        for p in range(2):
            col_power = np.sum(np.abs(S[:, p]) ** 2)
            assert col_power <= 1.0 + 1e-10, (
                f"Passivity violation at port {p}: sum = {col_power}"
            )

    def test_s_parameters_finite(self, two_port_system):
        """All S-parameters should be finite."""
        S = two_port_system['S']
        assert np.all(np.isfinite(S))
