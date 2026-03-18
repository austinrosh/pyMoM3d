# Getting Started

## What is pyMoM3d?

pyMoM3d is a Python library that solves electromagnetic scattering and radiation problems on 3D PEC (Perfect Electric Conductor) surfaces using the Method of Moments (MoM) with RWG (Rao-Wilton-Glisson) basis functions on triangular surface meshes.

Given a conducting surface geometry and an excitation (plane wave or delta-gap feed), the solver computes the induced surface currents by solving the Electric Field Integral Equation (EFIE), then post-processes them into far-field radiation patterns, radar cross section (RCS), and input impedance.

## Prerequisites

- Python 3.10+

## Installation

```bash
# Clone and enter the repo
cd pyMoM3d

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install pyMoM3d in editable mode (required once after cloning)
pip install -e .

# Verify the import works
python -c "import pyMoM3d; print(pyMoM3d.__version__)"
# Should print: 0.2.0
```

The runtime dependencies are listed in `requirements.txt`:
- **numpy** — array computation
- **scipy** — spherical Bessel functions (Mie series), GMRES solver
- **matplotlib** — plotting and visualization
- **gmsh** — high-quality surface meshing with element size control (recommended mesher)
- **trimesh** — geometry meshing fallback and STL/OBJ file loading

**Optional:**
- **tkinter** — required for the file-selection dialog in `stl_rcs_example.py`. On macOS with Homebrew Python: `brew install python-tk@3.13` (match your Python version). On Ubuntu/Debian: `sudo apt install python3-tk`. On Windows, tkinter is included with the standard Python installer.

## Quick Start: Scatter a Plane Wave off a Sphere

```python
import numpy as np
from pyMoM3d import (
    Sphere, GmshMesher, compute_rwg_connectivity,
    fill_impedance_matrix, PlaneWaveExcitation, solve_direct,
    compute_far_field, compute_rcs,
    plot_mesh_3d, plot_surface_current,
    eta0, c0,
)

# 1. Define geometry and frequency
radius = 0.1           # meters
frequency = 1.5e9      # Hz
k = 2.0 * np.pi * frequency / c0

# 2. Create mesh (Gmsh gives element size control)
sphere = Sphere(radius=radius)
mesher = GmshMesher(target_edge_length=0.02)  # ~lambda/10 at 1.5 GHz
mesh = mesher.mesh_from_geometry(sphere)
basis = compute_rwg_connectivity(mesh)

# 3. Fill impedance matrix and solve
Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
exc = PlaneWaveExcitation(
    E0=np.array([1.0, 0.0, 0.0]),
    k_hat=np.array([0.0, 0.0, -1.0]),
)
V = exc.compute_voltage_vector(basis, mesh, k)
I = solve_direct(Z, V)

# 4. Compute far-field RCS
theta = np.linspace(0.001, np.pi - 0.001, 181)
phi = np.zeros_like(theta)
E_theta, E_phi = compute_far_field(I, basis, mesh, k, eta0, theta, phi)
rcs_dBsm = compute_rcs(E_theta, E_phi)

# 5. Visualize
plot_mesh_3d(mesh, show_edges=True)
plot_surface_current(I, basis, mesh, cmap='hot')
```

## Quick Start: Load an STL/OBJ File

You can load external mesh files and solve directly — no geometry primitive needed:

```python
import numpy as np
from pyMoM3d import (
    load_stl, GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, PlaneWaveExcitation,
    compute_far_field, compute_rcs,
    eta0, c0,
)

# Load STL (preserving original triangulation)
mesh = load_stl("my_object.stl")

# Or remesh with Gmsh for element size control
mesh = GmshMesher(target_edge_length=0.02).mesh_from_file("my_object.stl")

# Solve using the high-level driver with a pre-built mesh
exc = PlaneWaveExcitation(E0=np.array([1, 0, 0]), k_hat=np.array([0, 0, -1]))
config = SimulationConfig(frequency=1e9, excitation=exc, enable_report=True)
sim = Simulation(config, mesh=mesh)
result = sim.run()
```

For a fully interactive workflow with mesh quality assessment and solver recommendations, see `examples/stl_rcs_example.py`.

## Quick Start: High-Level Simulation Driver

For convenience, the `Simulation` class wraps the full pipeline. Pass `mesher='gmsh'` to use the Gmsh backend:

```python
import numpy as np
from pyMoM3d import (
    Sphere, Simulation, SimulationConfig,
    PlaneWaveExcitation, compute_far_field, compute_rcs,
    eta0, c0,
)

# Configure
exc = PlaneWaveExcitation(
    E0=np.array([1.0, 0.0, 0.0]),
    k_hat=np.array([0.0, 0.0, -1.0]),
)
config = SimulationConfig(frequency=1.5e9, excitation=exc)

# Create and run (using Gmsh mesher)
sim = Simulation(
    config,
    geometry=Sphere(radius=0.1),
    mesher='gmsh',
    target_edge_length=0.02,
)
result = sim.run()

print(f"Condition number: {result.condition_number:.2e}")
print(f"Max |I|: {np.max(np.abs(result.I_coefficients)):.4e}")

# Frequency sweep
results = sim.sweep([1.0e9, 1.5e9, 2.0e9])
for r in results:
    print(f"f={r.frequency/1e9:.1f} GHz, cond={r.condition_number:.2e}")
```

## Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_impedance.py

# Specific test class or method
pytest tests/test_greens.py::TestSingularity::test_green_self_term_finite

# Verbose output
pytest tests/ -v
```

The test suite covers geometry, meshing, RWG basis computation, Green's function quadrature, singularity extraction, impedance matrix assembly, solvers, far-field computation, and the simulation driver.

## Running Examples

Each example in `examples/` is a standalone script:

```bash
# PEC sphere bistatic RCS vs Mie series (primary validation)
python examples/sphere_rcs_validation.py

# Strip dipole input impedance vs frequency
python examples/dipole_impedance_sweep.py

# Rectangular plate plane-wave scattering
python examples/plate_scattering.py

# High-level simulation driver demo
python examples/simulation_driver_demo.py

# Solver performance benchmark
python examples/solver_performance.py

# Interactive STL/OBJ → RCS + surface current (requires tkinter)
python examples/stl_rcs_example.py
```

Each example prints diagnostic output to the terminal and saves plots to the `images/` directory.

## Project Layout

```
pyMoM3d/
  src/pyMoM3d/
    geometry/       Parametric geometry primitives
    mesh/           Mesh data structures, meshing, RWG basis
    greens/         Green's function, quadrature, singularity extraction
    mom/            Impedance matrix, excitation, solvers
    fields/         Far-field computation, RCS
    analysis/       Mie series, convergence studies, impedance analysis
    visualization/  3D mesh and surface current plotting
    utils/          Physical constants, progress reporting, report generation
    simulation.py   High-level simulation driver with STL/OBJ loading
  tests/            Unit and integration tests
  examples/         Standalone example scripts
  docs/             This documentation
  images/           Plot output from examples
```

## Next Steps

- [Examples Guide](examples-guide.md) — detailed walkthrough of each example
