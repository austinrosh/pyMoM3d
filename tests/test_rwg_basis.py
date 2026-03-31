"""Tests for enriched RWG basis data structure."""

import numpy as np
import pytest

from pyMoM3d.mesh import Mesh, compute_rwg_connectivity, RWGBasis
from pyMoM3d.utils.constants import c0


# ---------- helper: two-triangle bowtie ----------

def _make_bowtie():
    """Two triangles sharing edge (1,2), free vertices 0 and 3.

    Triangle 0: [0, 1, 2]  (vertices CCW in xy-plane, looking down z)
    Triangle 1: [2, 1, 3]  (CCW, free vertex 3 on opposite side of edge)

    Shared edge is between vertices 1 and 2.
    Free vertex for tri 0 is vertex 0 (left of edge).
    Free vertex for tri 1 is vertex 3 (right of edge).
    """
    vertices = np.array([
        [0.0, 0.5, 0.0],   # 0 — free vertex of tri 0 (left)
        [0.5, 0.0, 0.0],   # 1 — shared
        [0.5, 1.0, 0.0],   # 2 — shared
        [1.0, 0.5, 0.0],   # 3 — free vertex of tri 1 (right)
    ])
    triangles = np.array([
        [0, 1, 2],
        [2, 1, 3],
    ])
    return Mesh(vertices, triangles)


class TestBowtie:
    """Hand-verified RWG fields on a two-triangle mesh."""

    def test_one_interior_basis(self):
        mesh = _make_bowtie()
        basis = compute_rwg_connectivity(mesh)
        assert basis.num_basis == 1

    def test_four_boundary_edges(self):
        mesh = _make_bowtie()
        basis = compute_rwg_connectivity(mesh)
        assert basis.num_boundary_edges == 4

    def test_edge_length(self):
        mesh = _make_bowtie()
        basis = compute_rwg_connectivity(mesh)
        # Shared edge between vertices 1=(0.5,0,0) and 2=(0.5,1,0)
        expected_len = 1.0
        assert np.isclose(basis.edge_length[0], expected_len)

    def test_free_vertices_not_on_edge(self):
        mesh = _make_bowtie()
        basis = compute_rwg_connectivity(mesh)
        edge = mesh.edges[basis.edge_index[0]]
        ev = set(edge)
        assert basis.free_vertex_plus[0] not in ev
        assert basis.free_vertex_minus[0] not in ev

    def test_free_vertices_are_0_and_3(self):
        mesh = _make_bowtie()
        basis = compute_rwg_connectivity(mesh)
        fv = {int(basis.free_vertex_plus[0]), int(basis.free_vertex_minus[0])}
        assert fv == {0, 3}

    def test_areas(self):
        mesh = _make_bowtie()
        basis = compute_rwg_connectivity(mesh)
        # Both triangles: right triangles with legs 0.5 and 1.0
        # area = 0.5 * 0.5 * 1.0 = 0.25
        assert np.isclose(basis.area_plus[0], 0.25)
        assert np.isclose(basis.area_minus[0], 0.25)

    def test_rho_continuity_at_edge(self):
        """rho_plus and rho_minus should point in the SAME direction at the
        shared edge, ensuring current continuity across the edge.

        rho_plus = r - r_free_plus  (away from free vertex on T+)
        rho_minus = r_free_minus - r (toward free vertex on T-)

        Both should point from T+ free vertex through the edge toward T-
        free vertex, giving a continuous current flow.
        """
        mesh = _make_bowtie()
        basis = compute_rwg_connectivity(mesh)

        edge = mesh.edges[basis.edge_index[0]]
        midpoint = 0.5 * (mesh.vertices[edge[0]] + mesh.vertices[edge[1]])

        r_free_plus = mesh.vertices[basis.free_vertex_plus[0]]
        r_free_minus = mesh.vertices[basis.free_vertex_minus[0]]

        rho_plus = midpoint - r_free_plus    # away from free vertex on T+
        rho_minus = r_free_minus - midpoint   # toward free vertex on T-

        # Both should point in the same general direction (current continuity)
        assert np.dot(rho_plus, rho_minus) > 0

    def test_validate_passes(self):
        mesh = _make_bowtie()
        basis = compute_rwg_connectivity(mesh)
        basis.validate(mesh)  # Should not raise


class TestClosedSphere:
    """RWG basis on a closed icosphere — Euler formula checks."""

    @pytest.fixture
    def sphere_mesh(self):
        from pyMoM3d import Sphere, GmshMesher
        sphere = Sphere(radius=1.0)
        mesh = GmshMesher(target_edge_length=0.5).mesh_from_geometry(sphere)
        return mesh

    def test_zero_boundary_edges(self, sphere_mesh):
        basis = compute_rwg_connectivity(sphere_mesh)
        assert basis.num_boundary_edges == 0

    def test_num_basis_equals_interior_edges(self, sphere_mesh):
        basis = compute_rwg_connectivity(sphere_mesh)
        # All edges should be interior for a closed surface
        assert basis.num_basis == sphere_mesh.get_num_edges()

    def test_euler_formula(self, sphere_mesh):
        V = sphere_mesh.get_num_vertices()
        E = sphere_mesh.get_num_edges()
        F = sphere_mesh.get_num_triangles()
        assert V - E + F == 2

    def test_validate_passes(self, sphere_mesh):
        basis = compute_rwg_connectivity(sphere_mesh)
        basis.validate(sphere_mesh)


class TestCheckDensity:
    def test_coarse_warns(self):
        mesh = _make_bowtie()
        # At 3 GHz, lambda ~ 0.1 m, lambda/10 ~ 0.01 m
        # Edge lengths are ~1 m, so this should warn
        with pytest.warns(UserWarning, match="Mesh too coarse"):
            result = mesh.check_density(3e9)
        assert result is False

    def test_fine_ok(self):
        mesh = _make_bowtie()
        # At 1 Hz, lambda ~ 3e8 m, lambda/10 ~ 3e7 m — edges are fine
        result = mesh.check_density(1.0)
        assert result is True


class TestLegacyCompat:
    """Ensure rwg_pairs is still populated for backward compatibility."""

    def test_rwg_pairs_populated(self):
        mesh = _make_bowtie()
        compute_rwg_connectivity(mesh)
        assert mesh.rwg_pairs is not None
        assert mesh.rwg_pairs.shape[1] == 2
