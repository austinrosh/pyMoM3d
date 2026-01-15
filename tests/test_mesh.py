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
    mesh = create_mesh_from_vertices(vertices)
    
    assert mesh.get_num_vertices() == 4
    assert mesh.get_num_triangles() == 2
    assert mesh.get_num_edges() == 5  # 4 boundary + 1 interior (diagonal)


def test_mesh_properties():
    """Test that mesh computes all required properties."""
    plate = RectangularPlate(1.0, 1.0)
    vertices = plate.get_vertices()
    mesh = create_mesh_from_vertices(vertices)
    
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
    mesh = create_mesh_from_vertices(vertices)
    rwg_pairs = compute_rwg_connectivity(mesh)
    
    assert rwg_pairs.shape[1] == 2  # Each pair has 2 triangle indices
    assert len(rwg_pairs) == mesh.get_num_edges()
    
    # Check that boundary edges have -1 as second index
    boundary_edges = rwg_pairs[:, 1] == -1
    interior_edges = rwg_pairs[:, 1] != -1
    
    # For a rectangular plate with 2 triangles, we should have:
    # - 4 boundary edges (one per side)
    # - 1 interior edge (the diagonal)
    assert np.sum(boundary_edges) == 4
    assert np.sum(interior_edges) == 1
    
    # Check that interior edge connects the two triangles
    interior_pair = rwg_pairs[interior_edges][0]
    assert interior_pair[0] in [0, 1]
    assert interior_pair[1] in [0, 1]
    assert interior_pair[0] != interior_pair[1]


def test_mesh_statistics():
    """Test mesh statistics computation."""
    plate = RectangularPlate(1.0, 0.5)
    vertices = plate.get_vertices()
    mesh = create_mesh_from_vertices(vertices)
    
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
