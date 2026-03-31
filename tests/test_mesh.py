"""Tests for mesh data structures and RWG connectivity."""

import numpy as np
import pytest

from pyMoM3d import (
    RectangularPlate,
    create_mesh_from_vertices,
    compute_rwg_connectivity,
    Mesh
)


def test_mesh_creation():
    """Test basic mesh creation."""
    plate = RectangularPlate(1.0, 0.5)
    vertices = plate.get_vertices()
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = create_mesh_from_vertices(vertices, triangles=triangles)

    assert mesh.get_num_vertices() == 4
    assert mesh.get_num_triangles() == 2
    assert mesh.get_num_edges() == 5  # 4 boundary + 1 interior (diagonal)


def test_mesh_properties():
    """Test that mesh computes all required properties."""
    plate = RectangularPlate(1.0, 1.0)
    vertices = plate.get_vertices()
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = create_mesh_from_vertices(vertices, triangles=triangles)
    
    # Check that all properties are computed
    assert mesh.triangle_normals.shape == (mesh.get_num_triangles(), 3)
    assert mesh.triangle_areas.shape == (mesh.get_num_triangles(),)
    assert mesh.edge_lengths.shape == (mesh.get_num_edges(),)
    assert len(mesh.edge_to_triangles) == mesh.get_num_edges()
    
    # Check that triangle areas are positive
    assert np.all(mesh.triangle_areas > 0)
    
    # Check that edge lengths are positive
    assert np.all(mesh.edge_lengths > 0)
    
    # Check that normals are unit vectors
    norms = np.linalg.norm(mesh.triangle_normals, axis=1)
    assert np.allclose(norms, 1.0)


def test_rwg_connectivity():
    """Test RWG connectivity computation."""
    plate = RectangularPlate(1.0, 0.5)
    vertices = plate.get_vertices()
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = create_mesh_from_vertices(vertices, triangles=triangles)
    basis = compute_rwg_connectivity(mesh)

    # compute_rwg_connectivity returns an RWGBasis, not an ndarray
    # For a 2-triangle plate with 1 interior edge:
    # - 1 RWG basis function (interior edge)
    # - 4 boundary edges
    assert basis.num_basis == 1
    assert basis.num_boundary_edges == 4

    # Interior edge connects the two triangles (t_plus and t_minus are 0 and 1)
    assert basis.t_plus[0] in [0, 1]
    assert basis.t_minus[0] in [0, 1]
    assert basis.t_plus[0] != basis.t_minus[0]


def test_mesh_statistics():
    """Test mesh statistics computation."""
    plate = RectangularPlate(1.0, 0.5)
    vertices = plate.get_vertices()
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = create_mesh_from_vertices(vertices, triangles=triangles)
    
    stats = mesh.get_statistics()
    
    assert 'num_vertices' in stats
    assert 'num_triangles' in stats
    assert 'num_edges' in stats
    assert 'num_basis_functions' in stats
    assert 'min_triangle_area' in stats
    assert 'max_triangle_area' in stats
    assert 'mean_triangle_area' in stats
    
    assert stats['num_vertices'] == 4
    assert stats['num_triangles'] == 2
    assert stats['num_edges'] == 5


def test_mesh_validation():
    """Test that invalid meshes raise errors."""
    # Invalid vertices shape
    with pytest.raises(ValueError):
        Mesh(np.array([[1, 2]]), np.array([[0, 1, 2]]))
    
    # Invalid triangle shape
    with pytest.raises(ValueError):
        Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1]]))
    
    # Triangle indices out of bounds
    with pytest.raises(ValueError):
        Mesh(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 10]])  # Index 10 doesn't exist
        )


def test_mesh_with_provided_triangles():
    """Test mesh creation with explicitly provided triangles."""
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ])
    
    triangles = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])
    
    mesh = Mesh(vertices, triangles)
    
    assert mesh.get_num_vertices() == 4
    assert mesh.get_num_triangles() == 2
