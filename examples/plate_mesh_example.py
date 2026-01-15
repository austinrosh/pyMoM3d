"""
Simple example: Generate and visualize a rectangular plate mesh.

This demonstrates Phase 1 functionality:
- Geometry primitive (rectangular plate)
- Mesh generation with Delaunay triangulation
- RWG connectivity computation
- Mesh visualization
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from pyMoM3d import (
    RectangularPlate,
    create_mesh_from_vertices,
    compute_rwg_connectivity,
    plot_mesh_3d,
    plot_mesh
)


def main():
    """Generate and visualize a rectangular plate mesh."""
    
    print("=" * 60)
    print("Rectangular Plate Mesh Example")
    print("=" * 60)
    
    # Create a rectangular plate
    width = 1.0
    height = 0.5
    plate = RectangularPlate(width, height, center=(0, 0, 0))
    
    print(f"\nPlate dimensions: {width} × {height}")
    print(f"Plate center: {plate.center}")
    
    # Get vertices
    vertices = plate.get_vertices()
    print(f"\nVertices shape: {vertices.shape}")
    print(f"Vertices:\n{vertices}")
    
    # Create mesh using Delaunay triangulation
    print("\nCreating mesh with Delaunay triangulation...")
    mesh = create_mesh_from_vertices(vertices)
    
    # Compute RWG connectivity
    print("Computing RWG connectivity...")
    rwg_pairs = compute_rwg_connectivity(mesh)
    
    # Print mesh statistics
    stats = mesh.get_statistics()
    print("\n" + "=" * 60)
    print("Mesh Statistics:")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.6f}")
        else:
            print(f"  {key:25s}: {value}")
    
    print(f"\nRWG pairs shape: {rwg_pairs.shape}")
    print(f"Number of boundary edges: {np.sum(rwg_pairs[:, 1] == -1)}")
    print(f"Number of interior edges: {np.sum(rwg_pairs[:, 1] != -1)}")
    
    # Visualize mesh
    print("\nGenerating visualizations...")
    
    # 3D plot
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    plot_mesh_3d(mesh, ax=ax1, show_edges=True, show_normals=False)
    
    # 2D projection (xy-plane)
    ax2 = fig.add_subplot(132)
    plot_mesh(mesh, projection='xy', ax=ax2, show_edges=True)
    
    # 2D projection with normals
    ax3 = fig.add_subplot(133, projection='3d')
    plot_mesh_3d(mesh, ax=ax3, show_edges=True, show_normals=True, normal_scale=0.05)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'plate_mesh_example.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    
    # Show plot
    plt.show()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
