"""Tests for mesh mirroring utilities."""

import numpy as np
import pytest

from pyMoM3d.mesh.mesh_data import Mesh
from pyMoM3d.mesh.mirror import mirror_mesh_x, combine_meshes, extract_submesh
from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity


def _simple_plate_mesh():
    """A 2-triangle plate spanning x=[0, 2], y=[0, 1], z=0."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [2, 1, 0],
    ], dtype=np.float64)
    tris = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4],
    ], dtype=np.int32)
    return Mesh(vertices=verts, triangles=tris)


class TestMirrorMeshX:
    def test_vertices_reflected(self):
        mesh = _simple_plate_mesh()
        mirrored = mirror_mesh_x(mesh, x_plane=0.0)
        # x-coordinates should be negated
        np.testing.assert_allclose(mirrored.vertices[:, 0], -mesh.vertices[:, 0])
        # y and z unchanged
        np.testing.assert_allclose(mirrored.vertices[:, 1], mesh.vertices[:, 1])
        np.testing.assert_allclose(mirrored.vertices[:, 2], mesh.vertices[:, 2])

    def test_winding_reversed(self):
        mesh = _simple_plate_mesh()
        mirrored = mirror_mesh_x(mesh, x_plane=0.0)
        # Winding should be reversed
        np.testing.assert_array_equal(mirrored.triangles[:, 0], mesh.triangles[:, 2])
        np.testing.assert_array_equal(mirrored.triangles[:, 2], mesh.triangles[:, 0])

    def test_triangle_count_preserved(self):
        mesh = _simple_plate_mesh()
        mirrored = mirror_mesh_x(mesh, x_plane=1.0)
        assert len(mirrored.triangles) == len(mesh.triangles)

    def test_areas_preserved(self):
        mesh = _simple_plate_mesh()
        mirrored = mirror_mesh_x(mesh, x_plane=0.0)
        np.testing.assert_allclose(mirrored.triangle_areas, mesh.triangle_areas)


class TestCombineMeshes:
    def test_shared_vertices_at_seam(self):
        """Two plates sharing an edge at x=1 should merge seam vertices."""
        mesh = _simple_plate_mesh()
        # Mirror about x=0 gives plate at x=[-2, 0]
        mirrored = mirror_mesh_x(mesh, x_plane=0.0)
        combined = combine_meshes(mesh, mirrored)
        # Original has 6 verts, mirror has 6 verts, 2 shared at x=0
        assert len(combined.vertices) == 10  # 6 + 6 - 2

    def test_all_triangles_present(self):
        mesh = _simple_plate_mesh()
        mirrored = mirror_mesh_x(mesh, x_plane=0.0)
        combined = combine_meshes(mesh, mirrored)
        assert len(combined.triangles) == 8  # 4 + 4

    def test_rwg_edges_across_seam(self):
        """Combined mesh should have interior RWG edges across the seam."""
        mesh = _simple_plate_mesh()
        mirrored = mirror_mesh_x(mesh, x_plane=0.0)
        combined = combine_meshes(mesh, mirrored)
        basis = compute_rwg_connectivity(combined)
        # With 8 triangles and shared seam, should have interior edges
        assert basis.num_basis > 0
        # More interior edges than either sub-mesh alone
        basis_a = compute_rwg_connectivity(mesh)
        assert basis.num_basis > basis_a.num_basis


class TestExtractSubmesh:
    def test_extract_left_half(self):
        mesh = _simple_plate_mesh()
        sub, mask = extract_submesh(mesh, x_min=0.0, x_max=1.0)
        # Triangles with centroids in [0, 1] — first 2 triangles
        assert len(sub.triangles) == 2
        assert mask.sum() == 2

    def test_extract_right_half(self):
        mesh = _simple_plate_mesh()
        sub, mask = extract_submesh(mesh, x_min=1.0, x_max=2.0)
        assert len(sub.triangles) == 2

    def test_extract_full_mesh(self):
        mesh = _simple_plate_mesh()
        sub, mask = extract_submesh(mesh, x_min=-1.0, x_max=3.0)
        assert len(sub.triangles) == 4
