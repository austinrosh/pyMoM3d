"""Tests for Gmsh-based mesh generation."""

import numpy as np
import pytest

from pyMoM3d import (
    GmshMesher,
    Mesh,
    RectangularPlate,
    Sphere,
    Cylinder,
    Cube,
    Pyramid,
    compute_rwg_connectivity,
)


@pytest.fixture
def mesher():
    return GmshMesher(target_edge_length=0.3)


class TestGmshMesherSphere:
    def test_basic_sphere(self, mesher):
        mesh = mesher.mesh_sphere(radius=1.0)
        assert isinstance(mesh, Mesh)
        assert mesh.get_num_vertices() > 10
        assert mesh.get_num_triangles() > 10

    def test_sphere_surface_area(self, mesher):
        r = 1.0
        mesh = mesher.mesh_sphere(radius=r, target_edge_length=0.2)
        total_area = np.sum(mesh.triangle_areas)
        expected = 4 * np.pi * r**2
        assert abs(total_area - expected) / expected < 0.05  # within 5%

    def test_sphere_with_center(self, mesher):
        center = (1.0, 2.0, 3.0)
        mesh = mesher.mesh_sphere(radius=0.5, center=center)
        centroid = np.mean(mesh.vertices, axis=0)
        np.testing.assert_allclose(centroid, center, atol=0.1)

    def test_sphere_rwg(self, mesher):
        mesh = mesher.mesh_sphere(radius=1.0)
        basis = compute_rwg_connectivity(mesh)
        # Closed surface: all edges should be interior (no boundary)
        assert np.all(mesh.rwg_pairs[:, 1] != -1)
        assert basis.num_basis > 0


class TestGmshMesherPlate:
    def test_basic_plate(self, mesher):
        mesh = mesher.mesh_plate(width=1.0, height=0.5)
        assert isinstance(mesh, Mesh)
        assert mesh.get_num_triangles() >= 2

    def test_plate_area(self, mesher):
        w, h = 2.0, 1.0
        mesh = mesher.mesh_plate(width=w, height=h, target_edge_length=0.2)
        total_area = np.sum(mesh.triangle_areas)
        assert abs(total_area - w * h) / (w * h) < 0.01

    def test_plate_is_planar(self, mesher):
        mesh = mesher.mesh_plate(width=1.0, height=1.0)
        # All z-coordinates should be 0
        np.testing.assert_allclose(mesh.vertices[:, 2], 0.0, atol=1e-12)

    def test_plate_rwg_has_boundary(self, mesher):
        mesh = mesher.mesh_plate(width=1.0, height=1.0)
        compute_rwg_connectivity(mesh)
        # Open surface: some edges should be boundary
        assert np.any(mesh.rwg_pairs[:, 1] == -1)


class TestGmshMesherCylinder:
    def test_basic_cylinder(self, mesher):
        mesh = mesher.mesh_cylinder(radius=0.5, height=1.0)
        assert isinstance(mesh, Mesh)
        assert mesh.get_num_triangles() > 10

    def test_cylinder_surface_area(self, mesher):
        r, h = 0.5, 1.0
        mesh = mesher.mesh_cylinder(radius=r, height=h, target_edge_length=0.15)
        total_area = np.sum(mesh.triangle_areas)
        # Closed cylinder: lateral + 2 caps
        expected = 2 * np.pi * r * h + 2 * np.pi * r**2
        assert abs(total_area - expected) / expected < 0.05


class TestGmshMesherCube:
    def test_basic_cube(self, mesher):
        mesh = mesher.mesh_cube(side_length=1.0)
        assert isinstance(mesh, Mesh)
        assert mesh.get_num_triangles() >= 12  # at least 2 per face

    def test_cube_surface_area(self, mesher):
        s = 1.0
        mesh = mesher.mesh_cube(side_length=s, target_edge_length=0.2)
        total_area = np.sum(mesh.triangle_areas)
        expected = 6 * s**2
        assert abs(total_area - expected) / expected < 0.02


class TestGmshMesherPyramid:
    def test_basic_pyramid(self, mesher):
        mesh = mesher.mesh_pyramid(base_size=1.0, height=1.0)
        assert isinstance(mesh, Mesh)
        assert mesh.get_num_triangles() >= 6

    def test_pyramid_surface_area(self, mesher):
        s, h = 1.0, 1.0
        mesh = mesher.mesh_pyramid(
            base_size=s, height=h, target_edge_length=0.15
        )
        total_area = np.sum(mesh.triangle_areas)
        # Base + 4 triangular faces
        slant = np.sqrt((s / 2) ** 2 + h**2)
        expected = s**2 + 4 * 0.5 * s * slant
        assert abs(total_area - expected) / expected < 0.05


class TestMeshFromGeometry:
    def test_sphere_dispatch(self, mesher):
        geom = Sphere(radius=1.0)
        mesh = mesher.mesh_from_geometry(geom)
        assert mesh.get_num_triangles() > 10

    def test_plate_dispatch(self, mesher):
        geom = RectangularPlate(width=1.0, height=0.5)
        mesh = mesher.mesh_from_geometry(geom)
        assert mesh.get_num_triangles() >= 2

    def test_cylinder_dispatch(self, mesher):
        geom = Cylinder(radius=0.5, height=1.0)
        mesh = mesher.mesh_from_geometry(geom)
        assert mesh.get_num_triangles() > 10

    def test_cube_dispatch(self, mesher):
        geom = Cube(side_length=1.0)
        mesh = mesher.mesh_from_geometry(geom)
        assert mesh.get_num_triangles() >= 12

    def test_pyramid_dispatch(self, mesher):
        geom = Pyramid(base_size=1.0, height=1.0)
        mesh = mesher.mesh_from_geometry(geom)
        assert mesh.get_num_triangles() >= 6

    def test_unsupported_type(self, mesher):
        with pytest.raises(TypeError):
            mesher.mesh_from_geometry("not a geometry")


class TestEdgeLengthControl:
    def test_finer_mesh_has_more_elements(self):
        coarse = GmshMesher(target_edge_length=0.5)
        fine = GmshMesher(target_edge_length=0.15)

        mesh_coarse = coarse.mesh_sphere(radius=1.0)
        mesh_fine = fine.mesh_sphere(radius=1.0)

        assert mesh_fine.get_num_triangles() > mesh_coarse.get_num_triangles()
