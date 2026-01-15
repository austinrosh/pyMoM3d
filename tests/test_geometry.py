"""Tests for geometry primitives."""

import numpy as np
import pytest

from pyMoM3d import RectangularPlate


def test_rectangular_plate_creation():
    """Test basic rectangular plate creation."""
    plate = RectangularPlate(1.0, 0.5)
    assert plate.width == 1.0
    assert plate.height == 0.5
    assert np.allclose(plate.center, [0, 0, 0])


def test_rectangular_plate_vertices():
    """Test vertex generation for rectangular plate."""
    plate = RectangularPlate(2.0, 1.0, center=(0, 0, 0))
    vertices = plate.get_vertices()
    
    assert vertices.shape == (4, 3)
    
    # Check that vertices form a rectangle
    # Width should be 2.0, height should be 1.0
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    z_coords = vertices[:, 2]
    
    assert np.allclose(z_coords, 0)  # All z should be 0
    assert np.isclose(np.max(x_coords) - np.min(x_coords), 2.0)  # Width
    assert np.isclose(np.max(y_coords) - np.min(y_coords), 1.0)  # Height


def test_rectangular_plate_centered():
    """Test rectangular plate with custom center."""
    plate = RectangularPlate(1.0, 1.0, center=(1, 2, 3))
    vertices = plate.get_vertices()
    
    # Center of vertices should match plate center
    center_computed = np.mean(vertices, axis=0)
    assert np.allclose(center_computed, [1, 2, 3])


def test_rectangular_plate_bounding_box():
    """Test bounding box computation."""
    plate = RectangularPlate(2.0, 1.0, center=(0, 0, 0))
    min_corner, max_corner = plate.get_bounding_box()
    
    assert np.allclose(min_corner, [-1, -0.5, 0])
    assert np.allclose(max_corner, [1, 0.5, 0])


def test_rectangular_plate_invalid_dimensions():
    """Test that invalid dimensions raise errors."""
    with pytest.raises(ValueError):
        RectangularPlate(-1.0, 1.0)
    
    with pytest.raises(ValueError):
        RectangularPlate(1.0, 0.0)
