# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyMoM3d is a Python library implementing a 3D Method of Moments (MoM) electromagnetic solver using RWG (Rao-Wilton-Glisson) basis functions on triangular surface meshes.

## Commands

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_geometry.py

# Run a specific test
pytest tests/test_mesh.py::test_rwg_connectivity

# Run examples (each is standalone)
python examples/plate_mesh_example.py
python examples/sphere_example.py
```

## Architecture

The library follows a pipeline: **Geometry → Mesh → RWG Basis → (future: MoM solver)**

### Core Modules

**`geometry/primitives.py`** - 5 geometry primitives (RectangularPlate, Sphere, Cylinder, Cube, Pyramid)
- All support `to_trimesh()` conversion for mesh generation
- Use `get_vertex_grid(subdivisions)` for mesh refinement control

**`mesh/mesh_data.py`** - Central `Mesh` class storing:
- `vertices` (N×3), `triangles` (N×3), `edges` (N×2)
- `rwg_pairs` (N×2): RWG basis function pairs where interior edges have [t1, t2] and boundary edges have [t1, -1]
- `edge_to_triangles`: Dict mapping edge index to triangle indices

**`mesh/trimesh_mesher.py`** - `PythonMesher` class for high-quality mesh generation via trimesh

**`mesh/rwg_connectivity.py`** - `compute_rwg_connectivity()` computes RWG basis pairs from mesh topology

**`visualization/mesh_plot.py`** - `plot_mesh_3d()` for 3D surface rendering, `plot_mesh()` for 2D

### Typical Workflow

```python
from pyMoM3d import Sphere, PythonMesher, compute_rwg_connectivity, plot_mesh_3d

# 1. Create geometry
sphere = Sphere(radius=1.0)

# 2. Generate mesh
trimesh_obj = sphere.to_trimesh(subdivisions=3)
mesher = PythonMesher()
mesh = mesher.mesh_from_geometry(trimesh_obj)

# 3. Validate and compute RWG basis
mesh.validate()
compute_rwg_connectivity(mesh)

# 4. Visualize
plot_mesh_3d(mesh, show_edges=True)
```

## Code Conventions

From `.cursorrules`:

- **Types**: Use `np.float64` for numerical arrays, `np.int32` for indices
- **Docstrings**: NumPy convention with Parameters, Returns, Examples sections
- **Units**: Meters for length, Hz for frequency
- **MoM guidance**: Validate mesh density (warn if < 10 elements/λ)
- **Constants**: Use physical constants from `utils/constants.py`

## Project Status

**Implemented**: Geometry primitives, mesh data structures, RWG connectivity, trimesh-based meshing, visualization

**Placeholder modules** (empty, reserved for future): `analysis/`, `fields/`, `greens/`, `mom/`, `utils/`
