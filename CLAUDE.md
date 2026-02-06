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
python examples/sphere_rcs_validation.py
python examples/dipole_impedance_sweep.py
python examples/plate_scattering.py
python examples/simulation_driver_demo.py
python examples/solver_performance.py
python examples/stl_rcs_example.py    # requires tkinter
python examples/cma_plate_example.py
python examples/cma_sphere_validation.py
```

## Architecture

The library follows a pipeline: **Geometry → Mesh → RWG Basis → Z-Fill → Solve → Post-Process**

### Core Modules

**`geometry/primitives.py`** - 5 geometry primitives (RectangularPlate, Sphere, Cylinder, Cube, Pyramid)
- All support `to_trimesh()` conversion for trimesh meshing (legacy)
- All supported by `GmshMesher.mesh_from_geometry()` for Gmsh meshing (recommended)
- Use `get_vertex_grid(subdivisions)` for mesh refinement control

**`mesh/mesh_data.py`** - Central `Mesh` class storing:
- `vertices` (N×3), `triangles` (N×3), `edges` (N×2)
- `rwg_pairs` (N×2): RWG basis function pairs where interior edges have [t1, t2] and boundary edges have [t1, -1]
- `edge_to_triangles`: Dict mapping edge index to triangle indices

**`mesh/gmsh_mesher.py`** - `GmshMesher` class for high-quality mesh generation via Gmsh with `target_edge_length` control, curvature adaptation, and CAD import support. Recommended mesher.

**`mesh/trimesh_mesher.py`** - `PythonMesher` class for mesh generation via trimesh (legacy fallback)

**`mesh/rwg_connectivity.py`** - `compute_rwg_connectivity()` computes RWG basis pairs from mesh topology

**`mom/impedance.py`** - `fill_impedance_matrix()` assembles the EFIE impedance matrix with singularity extraction

**`mom/excitation.py`** - Excitation sources: `PlaneWaveExcitation`, `DeltaGapExcitation`, `StripDeltaGapExcitation`, `find_feed_edges`

**`mom/solver.py`** - `solve_direct()` (LU) and `solve_gmres()` (iterative with diagonal preconditioner)

**`fields/far_field.py`** - `compute_far_field()` computes E_theta, E_phi from radiation integral

**`fields/rcs.py`** - `compute_rcs()` and `compute_monostatic_rcs()` for RCS in dBsm

**`simulation.py`** - `Simulation` class orchestrating the full pipeline; `load_stl()` for STL/OBJ loading; `compute_cma()` and `cma_sweep()` for Characteristic Mode Analysis

**`analysis/cma.py`** - `compute_characteristic_modes()` for CMA eigenvalue decomposition, `CMAResult` dataclass, modal significance/angle computation, `track_modes_across_frequency()` for mode tracking, `verify_orthogonality()` for validation

**`visualization/mesh_plot.py`** - `plot_mesh_3d()` for 3D surface rendering, `plot_mesh()` for 2D, `plot_surface_current()` for current density heatmaps (works with both driven currents and CMA modal currents)

**`visualization/plot_style.py`** - `configure_latex_style()` for LaTeX rendering configuration, label formatters (`format_frequency_label()`, `format_rcs_label()`, etc.), and title helpers for consistent scientific notation

**`utils/reporter.py`** - `TerminalReporter`, `SilentReporter`, `RecordingReporter` for progress reporting

**`utils/report_writer.py`** - `write_report()` generates structured text simulation reports

### Typical Workflow (Gmsh — recommended)

```python
from pyMoM3d import Sphere, GmshMesher, compute_rwg_connectivity, plot_mesh_3d

# 1. Create geometry
sphere = Sphere(radius=1.0)

# 2. Generate mesh with target edge length
mesher = GmshMesher(target_edge_length=0.1)
mesh = mesher.mesh_from_geometry(sphere)

# 3. Validate and compute RWG basis
mesh.validate()
compute_rwg_connectivity(mesh)

# 4. Visualize
plot_mesh_3d(mesh, show_edges=True)
```

### Typical Workflow (trimesh — legacy)

```python
from pyMoM3d import Sphere, PythonMesher, compute_rwg_connectivity, plot_mesh_3d

sphere = Sphere(radius=1.0)
trimesh_obj = sphere.to_trimesh(subdivisions=3)
mesh = PythonMesher().mesh_from_geometry(trimesh_obj)
mesh.validate()
compute_rwg_connectivity(mesh)
plot_mesh_3d(mesh, show_edges=True)
```

### Characteristic Mode Analysis Workflow

```python
from pyMoM3d import (
    RectangularPlate, GmshMesher, compute_rwg_connectivity,
    fill_impedance_matrix, compute_characteristic_modes,
    plot_surface_current, eta0, c0,
)
import numpy as np

# 1. Create geometry and mesh
plate = RectangularPlate(width=0.15, height=0.10)
mesh = GmshMesher(target_edge_length=0.015).mesh_from_geometry(plate)
basis = compute_rwg_connectivity(mesh)

# 2. Compute impedance matrix
k = 2 * np.pi * 1e9 / c0
Z = fill_impedance_matrix(basis, mesh, k, eta0)

# 3. Perform CMA
cma = compute_characteristic_modes(Z, frequency=1e9, num_modes=5)

# 4. Access modes by significance rank
J_mode1 = cma.get_mode(0)  # Most significant mode
lambda_1 = cma.get_eigenvalue(0)
ms_1 = cma.get_modal_significance(0)

# 5. Visualize modal current
plot_surface_current(J_mode1, basis, mesh, title=f'Mode 1: λ={lambda_1:.2f}')
```

## Code Conventions

From `.cursorrules`:

- **Types**: Use `np.float64` for numerical arrays, `np.int32` for indices
- **Docstrings**: NumPy convention with Parameters, Returns, Examples sections
- **Units**: Meters for length, Hz for frequency
- **MoM guidance**: Validate mesh density (warn if < 10 elements/λ)
- **Constants**: Use physical constants from `utils/constants.py`

## LaTeX Plotting Style

pyMoM3d uses publication-quality LaTeX-rendered text in all plots. The centralized configuration is in `visualization/plot_style.py`.

### Configuration

Enable LaTeX rendering at the start of any plotting script:

```python
from pyMoM3d import configure_latex_style

# Auto-detect LaTeX installation (falls back to mathtext if unavailable)
configure_latex_style()

# Or explicitly disable full LaTeX rendering
configure_latex_style(use_tex=False)
```

### Dependencies

- **Required**: matplotlib (included in requirements.txt)
- **Optional**: Full LaTeX installation (texlive, mactex, etc.) for `use_tex=True`
- **Fallback**: If LaTeX is not installed, matplotlib's built-in mathtext renderer is used automatically

### Notation Conventions

All plots follow consistent scientific notation:

| Element | Convention | Example |
|---------|------------|---------|
| Vectors | Boldface | `$\mathbf{J}$`, `$\mathbf{E}$` |
| Scalars | Italic | `$f$`, `$k$`, `$\lambda$` |
| Units | Roman font | `$\mathrm{A/m}$`, `$\mathrm{Hz}$` |
| Subscripts (labels) | Roman | `$R_{\mathrm{in}}$`, `$f_{\mathrm{res}}$` |
| Subscripts (indices) | Italic | `$\lambda_n$`, `$J_m$` |

### Label Helpers

Use the provided label formatters for consistency:

```python
from pyMoM3d import (
    format_frequency_label,    # → 'Frequency $f$ (GHz)'
    format_rcs_label,          # → 'RCS $\sigma$ (dBsm)'
    format_impedance_label,    # → '$R_{\mathrm{in}}$ ($\Omega$)'
    format_current_label,      # → '$|\mathbf{J}|$ (A/m)'
    format_eigenvalue_label,   # → 'Eigenvalue $\lambda_n$'
    format_angle_label,        # → '$\theta$ (deg)'
)

ax.set_xlabel(format_frequency_label())
ax.set_ylabel(format_rcs_label())
```

### Examples of Proper LaTeX Usage

```python
# Axis labels
ax.set_xlabel(r'Frequency $f$ (GHz)')
ax.set_ylabel(r'RCS $\sigma$ (dBsm)')
ax.set_zlabel(r'$z$ (m)')

# Titles
ax.set_title(rf'Bistatic RCS at $f = {freq/1e9:.1f}$ GHz ($ka = {ka:.2f}$)')

# Legend entries
ax.plot(x, y, label=rf'MoM ($N = {N}$)')

# Colorbar labels
cbar.set_label(r'$|\mathbf{J}|$ (A/m)')

# Annotations
ax.annotate(rf'$f_{{\mathrm{{res}}}} = {f_res:.3f}$ GHz', xy=(x, y), ...)
```

### Common Mistakes to Avoid

```python
# WRONG: Plain text in mathematical context
ax.set_ylabel('|J| (A/m)')          # Should use LaTeX
ax.set_xlabel('Frequency (GHz)')    # Missing symbol

# CORRECT: LaTeX formatting
ax.set_ylabel(r'$|\mathbf{J}|$ (A/m)')
ax.set_xlabel(r'Frequency $f$ (GHz)')

# WRONG: Italic subscript labels
ax.set_ylabel(r'$R_{in}$')          # 'in' should be roman

# CORRECT: Roman subscript for labels
ax.set_ylabel(r'$R_{\mathrm{in}}$')
```

## Project Status

**Implemented**: Geometry primitives, mesh data structures, RWG connectivity, Gmsh-based meshing (recommended) with feed-line support, trimesh-based meshing (legacy fallback), STL/OBJ file import with quality assessment and configurable resolution presets (coarse/medium/fine), uniform mesh refinement for STL/OBJ remeshing, OBJ multi-material Scene concatenation, EFIE impedance matrix with singularity extraction, excitation sources (plane wave, delta-gap, strip delta-gap), direct and iterative solvers with solver recommendation for large meshes, far-field/RCS computation, Mie series validation, surface current visualization (linear and dB scale) with plane wave info on plots, high-level simulation driver with reporting, interactive STL/OBJ example, frequency finder utility, **Characteristic Mode Analysis (CMA)** with modal significance, characteristic angles, frequency sweeps, and mode tracking
