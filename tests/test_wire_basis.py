"""Tests for wire mesh and basis function data structures."""

import numpy as np
import pytest

from pyMoM3d.wire.wire_basis import (
    WireSegment, WireMesh, WireBasis, compute_wire_connectivity,
)


class TestWireMesh:
    """WireMesh construction tests."""

    def test_vertical_probe_nodes(self):
        """Vertical probe has correct node positions."""
        wm = WireMesh.vertical_probe(
            x=1.0, y=2.0, z_bot=0.0, z_top=1.5e-3,
            radius=0.1e-3, n_segments=3,
        )
        assert wm.num_nodes == 4
        assert wm.num_segments == 3
        # All nodes at (1.0, 2.0, z_i)
        np.testing.assert_allclose(wm.nodes[:, 0], 1.0)
        np.testing.assert_allclose(wm.nodes[:, 1], 2.0)
        np.testing.assert_allclose(wm.nodes[0, 2], 0.0)
        np.testing.assert_allclose(wm.nodes[-1, 2], 1.5e-3)

    def test_vertical_probe_segments(self):
        """Vertical probe segments are uniform and z-directed."""
        wm = WireMesh.vertical_probe(
            x=0.0, y=0.0, z_bot=0.0, z_top=3.0e-3,
            radius=0.2e-3, n_segments=3,
        )
        for seg in wm.segments:
            np.testing.assert_allclose(seg.length, 1.0e-3, rtol=1e-12)
            np.testing.assert_allclose(seg.direction, [0, 0, 1])
            assert seg.radius == 0.2e-3

    def test_from_nodes(self):
        """WireMesh.from_nodes creates correct segments."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        wm = WireMesh.from_nodes(nodes, radius=0.01)
        assert wm.num_nodes == 3
        assert wm.num_segments == 2
        np.testing.assert_allclose(wm.segments[0].length, 1.0)
        np.testing.assert_allclose(wm.segments[0].direction, [1, 0, 0])
        np.testing.assert_allclose(wm.segments[1].length, 1.0)
        np.testing.assert_allclose(wm.segments[1].direction, [0, 1, 0])

    def test_invalid_zero_length(self):
        """Zero-length segment raises ValueError."""
        nodes = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        with pytest.raises(ValueError, match="Zero-length"):
            WireMesh.from_nodes(nodes, radius=0.01)

    def test_invalid_z_top_le_z_bot(self):
        """z_top <= z_bot raises ValueError."""
        with pytest.raises(ValueError, match="z_top"):
            WireMesh.vertical_probe(0, 0, z_bot=1.0, z_top=0.5, radius=0.1)


class TestWireBasis:
    """Wire basis function connectivity tests."""

    def test_three_segment_probe(self):
        """3-segment probe gives 2 basis functions."""
        wm = WireMesh.vertical_probe(0, 0, 0, 1.5e-3, 0.1e-3, n_segments=3)
        wb = compute_wire_connectivity(wm)

        assert wb.num_basis == 2
        # First basis: segments 0 (plus) and 1 (minus), node 1
        assert wb.seg_plus[0] == 0
        assert wb.seg_minus[0] == 1
        assert wb.node_index[0] == 1
        # Second basis: segments 1 (plus) and 2 (minus), node 2
        assert wb.seg_plus[1] == 1
        assert wb.seg_minus[1] == 2
        assert wb.node_index[1] == 2

    def test_four_segment_probe(self):
        """4-segment probe gives 3 basis functions."""
        wm = WireMesh.vertical_probe(0, 0, 0, 2.0e-3, 0.1e-3, n_segments=4)
        wb = compute_wire_connectivity(wm)
        assert wb.num_basis == 3

    def test_lengths_correct(self):
        """Basis function segment lengths match wire segments."""
        wm = WireMesh.vertical_probe(0, 0, 0, 3.0e-3, 0.1e-3, n_segments=3)
        wb = compute_wire_connectivity(wm)
        dl = 1.0e-3
        np.testing.assert_allclose(wb.length_plus, [dl, dl])
        np.testing.assert_allclose(wb.length_minus, [dl, dl])

    def test_single_segment_raises(self):
        """1 segment has no interior nodes — should raise."""
        wm = WireMesh.vertical_probe(0, 0, 0, 1e-3, 0.1e-3, n_segments=1)
        with pytest.raises(ValueError, match="at least 2"):
            compute_wire_connectivity(wm)

    def test_two_segments(self):
        """2 segments gives 1 basis function."""
        wm = WireMesh.vertical_probe(0, 0, 0, 1e-3, 0.1e-3, n_segments=2)
        wb = compute_wire_connectivity(wm)
        assert wb.num_basis == 1
