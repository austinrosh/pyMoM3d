"""
Example: Generate and visualize a cylinder mesh.

This demonstrates:
- Cylinder geometry primitive
- Mesh generation with trimesh
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
    Cylinder,
    PythonMesher,
    compute_rwg_connectivity,
    plot_mesh_3d,
    plot_mesh
)


def main():
    """Generate and visualize a cylinder mesh."""
    
    print("=" * 60)
    print("Cylinder Mesh Example")
    print("=" * 60)
    
    # Create a cylinder
    radius = 0.3
    height = 1.0
    cylinder = Cylinder(radius, height, center=(0, 0, 0))
    
    print(f"\nCylinder radius: {radius}")
    print(f"Cylinder height: {height}")
    print(f"Cylinder center: {cylinder.center}")
    
    # Create mesh using trimesh
    sections = 32  # Number of sections around cylinder
    print(f"\nMesh sections: {sections}")
    print("\nCreating mesh with trimesh...")
    mesher = PythonMesher()
    trimesh_obj = cylinder.to_trimesh(sections=sections)
    mesh = mesher.mesh_from_geometry(trimesh_obj)
    
    # Validate mesh
    validation = mesh.validate()
    print(f"\nMesh validation:")
    print(f"  Valid: {validation['is_valid']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
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
    ax1.set_title('3D View')
    
    # 2D projection (xy-plane)
    ax2 = fig.add_subplot(132)
    plot_mesh(mesh, projection='xy', ax=ax2, show_edges=True)
    ax2.set_title('XY Projection')
    
    # 2D projection with normals
    ax3 = fig.add_subplot(133, projection='3d')
    plot_mesh_3d(mesh, ax=ax3, show_edges=True, show_normals=True, normal_scale=0.1)
    ax3.set_title('With Normals')
    
    plt.tight_layout()
    
    # Save figure to images directory
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    output_file = os.path.join(images_dir, 'cylinder_example.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    
    # Show plot
    plt.show()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
