"""
Regression test: Validate trimesh-based meshing pipeline.

This script tests:
1. Mesh generation using trimesh primitives
2. Mesh validation (topology, geometry)
3. RWG connectivity computation
4. Mesh statistics and quality metrics
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from pyMoM3d import (
    RectangularPlate,
    Sphere,
    Cylinder,
    Cube,
    Pyramid,
    PythonMesher,
    compute_rwg_connectivity,
    plot_mesh_3d,
)


def test_rectangular_plate():
    """Test rectangular plate mesh generation."""
    print("=" * 60)
    print("Testing Rectangular Plate")
    print("=" * 60)
    
    plate = RectangularPlate(width=1.0, height=0.5, center=(0, 0, 0))
    
    # Test with trimesh
    mesher = PythonMesher()
    trimesh_obj = plate.to_trimesh(subdivisions=10)
    mesh = mesher.mesh_from_geometry(trimesh_obj)
    
    # Validate mesh
    validation = mesh.validate()
    print(f"\nValidation results:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Warnings: {validation['warnings']}")
    
    # Compute RWG connectivity
    rwg_pairs = compute_rwg_connectivity(mesh)
    
    # Print statistics
    stats = mesh.get_statistics()
    print(f"\nMesh statistics:")
    print(f"  Vertices: {stats['num_vertices']}")
    print(f"  Triangles: {stats['num_triangles']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  RWG basis functions: {stats['num_basis_functions']}")
    print(f"  Mean triangle area: {stats['mean_triangle_area']:.6e}")
    print(f"  Min triangle area: {stats['min_triangle_area']:.6e}")
    
    # Verify RWG count matches expected
    # For a closed mesh: E = 3*T/2 (approximately, for interior edges)
    # For an open mesh: E = 3*T - B, where B is boundary edges
    num_boundary = np.sum(rwg_pairs[:, 1] == -1)
    num_interior = np.sum(rwg_pairs[:, 1] != -1)
    expected_edges = num_interior + num_boundary
    
    print(f"\nRWG connectivity:")
    print(f"  Interior edges: {num_interior}")
    print(f"  Boundary edges: {num_boundary}")
    print(f"  Total edges: {len(mesh.edges)}")
    
    assert validation['is_valid'], "Mesh validation failed!"
    assert stats['num_triangles'] > 0, "No triangles generated!"
    assert stats['num_basis_functions'] > 0, "No RWG basis functions!"
    assert num_interior + num_boundary == len(mesh.edges), "Edge count mismatch!"
    
    print("✓ Rectangular plate test passed!")
    return mesh


def test_sphere():
    """Test sphere mesh generation."""
    print("\n" + "=" * 60)
    print("Testing Sphere")
    print("=" * 60)
    
    sphere = Sphere(radius=0.5, center=(0, 0, 0))
    
    mesher = PythonMesher()
    trimesh_obj = sphere.to_trimesh(subdivisions=2)
    mesh = mesher.mesh_from_geometry(trimesh_obj)
    
    validation = mesh.validate()
    print(f"\nValidation results:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Warnings: {validation['warnings']}")
    
    rwg_pairs = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    
    print(f"\nMesh statistics:")
    print(f"  Vertices: {stats['num_vertices']}")
    print(f"  Triangles: {stats['num_triangles']}")
    print(f"  RWG basis functions: {stats['num_basis_functions']}")
    
    assert validation['is_valid'], "Mesh validation failed!"
    assert stats['num_basis_functions'] > 0, "No RWG basis functions!"
    
    print("✓ Sphere test passed!")
    return mesh


def test_cube():
    """Test cube mesh generation."""
    print("\n" + "=" * 60)
    print("Testing Cube")
    print("=" * 60)
    
    cube = Cube(side_length=1.0, center=(0, 0, 0))
    
    mesher = PythonMesher()
    trimesh_obj = cube.to_trimesh()
    mesh = mesher.mesh_from_geometry(trimesh_obj)
    
    validation = mesh.validate()
    print(f"\nValidation results:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Warnings: {validation['warnings']}")
    
    rwg_pairs = compute_rwg_connectivity(mesh)
    stats = mesh.get_statistics()
    
    print(f"\nMesh statistics:")
    print(f"  Vertices: {stats['num_vertices']}")
    print(f"  Triangles: {stats['num_triangles']}")
    print(f"  RWG basis functions: {stats['num_basis_functions']}")
    
    assert validation['is_valid'], "Mesh validation failed!"
    assert stats['num_basis_functions'] > 0, "No RWG basis functions!"
    
    print("✓ Cube test passed!")
    return mesh


def visualize_meshes(meshes, names):
    """Visualize test meshes."""
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    n = len(meshes)
    fig = plt.figure(figsize=(5 * n, 5))
    
    for i, (mesh, name) in enumerate(zip(meshes, names)):
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        plot_mesh_3d(mesh, ax=ax, show_edges=True, show_normals=False)
        ax.set_title(name)
    
    plt.tight_layout()
    
    # Save figure
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    output_file = os.path.join(images_dir, 'trimesh_regression_test.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    
    plt.show()


def main():
    """Run all regression tests."""
    print("=" * 60)
    print("Trimesh Regression Test Suite")
    print("=" * 60)
    
    meshes = []
    names = []
    
    # Test rectangular plate
    try:
        mesh = test_rectangular_plate()
        meshes.append(mesh)
        names.append("Rectangular Plate")
    except Exception as e:
        print(f"✗ Rectangular plate test failed: {e}")
        raise
    
    # Test sphere
    try:
        mesh = test_sphere()
        meshes.append(mesh)
        names.append("Sphere")
    except Exception as e:
        print(f"✗ Sphere test failed: {e}")
        raise
    
    # Test cube
    try:
        mesh = test_cube()
        meshes.append(mesh)
        names.append("Cube")
    except Exception as e:
        print(f"✗ Cube test failed: {e}")
        raise
    
    # Visualize results
    visualize_meshes(meshes, names)
    
    print("\n" + "=" * 60)
    print("All regression tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
