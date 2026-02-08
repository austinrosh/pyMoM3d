# Examples Guide

Detailed walkthrough of each example script. All examples are standalone and can be run from the repository root:

```bash
PYTHONPATH=src python examples/<example_name>.py
```

Each example prints progress to the terminal and saves plots to `images/`.

---

## 1. Sphere RCS Validation (`sphere_rcs_validation.py`)

**Purpose:** Primary solver validation. Computes bistatic RCS of a PEC sphere illuminated by a plane wave and compares against the exact Mie series solution.

**What it does:**

1. Creates a sphere mesh using `GmshMesher(target_edge_length=0.02)` for element size control
2. Sweeps monostatic RCS over 0.5–2.0 GHz (16 frequencies)
3. At a fixed bistatic frequency (1.0 GHz), fills the impedance matrix and solves with a -z propagating, x-polarized plane wave
4. Computes bistatic RCS in the xz-plane (phi=0, theta from 0 to 180 deg)
5. Compares both monostatic and bistatic RCS against exact Mie series solutions

**Key output:**
- Impedance matrix condition number (should be ~10-100)
- Backscatter RCS comparison: MoM vs Mie in dB

**Generated plots** (`images/`):

| File | Content |
|---|---|
| `sphere_rcs_validation.png` | Monostatic RCS vs frequency (left) and bistatic RCS vs angle (right), both MoM vs Mie |
| `sphere_surface_current.png` | Surface current density heatmap |

**Relevant code pattern — manual pipeline (Gmsh):**

```python
# Mesh (using Gmsh for element size control)
sphere = Sphere(radius=0.1)
mesher = GmshMesher(target_edge_length=0.02)
mesh = mesher.mesh_from_geometry(sphere)
basis = compute_rwg_connectivity(mesh)

# Solve
k = 2 * np.pi * frequency / c0
Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
V = PlaneWaveExcitation(E0, k_hat).compute_voltage_vector(basis, mesh, k)
I = solve_direct(Z, V)

# Far-field
E_theta, E_phi = compute_far_field(I, basis, mesh, k, eta0, theta, phi)
rcs_dBsm = compute_rcs(E_theta, E_phi)

# Mie reference
from pyMoM3d.analysis.mie_series import mie_rcs_pec_sphere
rcs_mie = mie_rcs_pec_sphere(ka, theta)
```

**Tips:**
- Decrease `target_edge_length` for better Mie agreement at the cost of runtime
- ka ~ 1 is the simplest validation regime; larger ka requires finer meshes

---

## 2. Dipole Impedance Sweep (`dipole_impedance_sweep.py`)

**Purpose:** Simulates a thin-wire dipole approximated as a narrow rectangular strip with a delta-gap feed. Sweeps frequency to find resonance and plots input impedance, surface currents, and radiation patterns.

**What it does:**

1. Creates a narrow plate (15 cm x 10 mm) centered at the origin, meshed into ~50 triangles
2. Locates the feed edge nearest to the origin using `find_nearest_edge()`
3. Runs a frequency sweep from 0.5 to 1.5 GHz (11 points) using the `Simulation` driver
4. Identifies resonance (peak input resistance or reactance zero-crossing)
5. At resonance: computes surface currents, current distribution along the dipole, and E-plane/H-plane radiation patterns

**Key output:**
- Input impedance (R_in, X_in) vs frequency table
- Resonance frequency and impedance

**Generated plots** (`images/`):

| File | Content |
|---|---|
| `dipole_impedance_sweep.png` | R_in, X_in, and S11 vs frequency |
| `dipole_current_distribution.png` | Current magnitude and phase vs position along the dipole |
| `dipole_surface_current.png` | 3D surface current density heatmap |
| `dipole_radiation_pattern.png` | E-plane and H-plane cuts (rectangular + polar) |

**Relevant code pattern — delta-gap feed with Simulation driver:**

```python
from pyMoM3d.mom.excitation import DeltaGapExcitation, find_nearest_edge

# Find the feed edge closest to the origin
feed_idx = find_nearest_edge(mesh, basis, np.array([0.0, 0.0, 0.0]))

# Create excitation and simulation
exc = DeltaGapExcitation(basis_index=feed_idx, voltage=1.0)
config = SimulationConfig(frequency=1e9, excitation=exc, quad_order=4)
sim = Simulation(config, mesh=mesh)

# Frequency sweep
results = sim.sweep([0.5e9, 0.7e9, 0.9e9, 1.0e9, 1.1e9, 1.3e9, 1.5e9])

# Extract impedance
for r in results:
    print(f"f={r.frequency/1e9:.1f} GHz, Z_in={r.Z_input}")
```

**Relevant code pattern — radiation pattern cuts:**

```python
# E-plane (xz-plane, phi=0) — contains dipole axis
theta = np.linspace(0.001, np.pi - 0.001, 181)
E_th, E_ph = compute_far_field(I, basis, mesh, k, eta0, theta, np.zeros_like(theta))
gain_e = np.abs(E_th)**2 + np.abs(E_ph)**2

# H-plane (yz-plane, phi=pi/2) — perpendicular to dipole
E_th, E_ph = compute_far_field(I, basis, mesh, k, eta0, theta, np.full_like(theta, np.pi/2))
gain_h = np.abs(E_th)**2 + np.abs(E_ph)**2
```

**Tips:**
- The strip dipole is a crude approximation of a wire dipole. Absolute impedance values may differ from the canonical 73+j42 Ohms, but the resonant behavior (R_in peak, X_in trend) should be qualitatively correct.
- Increase `subdivisions` for better accuracy.

---

## 3. Plate Scattering (`plate_scattering.py`)

**Purpose:** Broadside plane-wave scattering from a flat PEC rectangular plate. Compares MoM surface current against the Physical Optics (PO) approximation.

**What it does:**

1. Creates a 1-wavelength square plate at 1 GHz (0.3 m x 0.3 m)
2. Illuminates with a plane wave incident from +z (broadside)
3. Solves the EFIE for surface currents
4. Computes bistatic RCS in the xz-plane
5. Compares the mean surface current magnitude against PO: `|J_PO| = 2/eta0`

**Key output:**
- Backscatter RCS vs PO estimate (should agree within a few dB for electrically large plates)
- Peak RCS and its angular location

**Generated plots** (`images/`):

| File | Content |
|---|---|
| `plate_scattering.png` | Bistatic RCS, current magnitude, and current phase |

**Relevant code pattern — open surface (plate):**

```python
# Using Gmsh
plate = RectangularPlate(width, height)
mesh = GmshMesher(target_edge_length=0.02).mesh_from_geometry(plate)
basis = compute_rwg_connectivity(mesh)
# Note: basis.num_boundary_edges > 0 for an open surface
```

**Tips:**
- Plates have boundary edges that don't produce RWG basis functions. The ratio of basis functions to edges is lower than for closed surfaces.
- PO accuracy improves for electrically larger plates.

---

## 4. Simulation Driver Demo (`simulation_driver_demo.py`)

**Purpose:** Demonstrates the high-level `Simulation` API, which wraps the entire pipeline into a few lines of code. Shows single-frequency solve, frequency sweep, and result save/load.

**What it does:**

1. Creates a sphere simulation using `Simulation(config, geometry=Sphere(...), subdivisions=2)`
2. Runs a single-frequency solve and prints diagnostics
3. Runs a 3-frequency sweep (0.8, 1.0, 1.2 GHz) and prints RCS at each frequency
4. Saves a result to `.npz` and reloads it to verify persistence

**Generated plots** (`images/`):

| File | Content |
|---|---|
| `simulation_driver_demo.png` | Bistatic RCS at 1 GHz |

**Relevant code pattern — the simplest possible simulation:**

```python
from pyMoM3d import Sphere, Simulation, SimulationConfig, PlaneWaveExcitation
import numpy as np

exc = PlaneWaveExcitation(
    E0=np.array([1.0, 0.0, 0.0]),
    k_hat=np.array([0.0, 0.0, -1.0]),
)
config = SimulationConfig(frequency=1e9, excitation=exc)
sim = Simulation(config, geometry=Sphere(radius=0.1), subdivisions=2)

# Single solve
result = sim.run()
print(f"Cond(Z) = {result.condition_number:.2e}")

# Frequency sweep
results = sim.sweep([0.8e9, 1.0e9, 1.2e9])

# Save and reload
result.save("my_result.npz")
loaded = SimulationResult.load("my_result.npz")
```

---

## 5. Solver Performance Benchmark (`solver_performance.py`)

**Purpose:** Benchmarks impedance matrix assembly and solver execution time across different mesh sizes to characterize scaling behavior.

```bash
PYTHONPATH=src python examples/solver_performance.py
```

---

## 6. STL/OBJ RCS Example (`stl_rcs_example.py`)

**Purpose:** Interactive workflow for loading an external mesh file (`.stl` or `.obj`), inspecting it, assessing mesh quality, optionally remeshing, and computing bistatic RCS and surface current.

**Requires:** `tkinter` (for the file selection dialog).

**What it does:**

1. Opens a native file dialog to select a `.stl` or `.obj` file
2. Loads the mesh via trimesh (preserving the original triangulation)
3. Prints mesh statistics: triangle/vertex/edge counts, edge lengths, elements per wavelength
4. Displays the imported mesh interactively via `plot_mesh_3d` (close window to continue)
5. Prompts for mesh resolution:
   - **coarse** — ~8 elements/wavelength (fast, lower accuracy)
   - **medium** — ~15 elements/wavelength (balanced, default)
   - **fine** — ~25 elements/wavelength (slow, higher accuracy)
6. Assesses mesh quality against the selected resolution:
   - Elements/wavelength below target → **must remesh** (automatic)
   - Edge ratio > 5.0 → **recommend remesh** (prompts user)
   - Degenerate triangles → **must remesh**
   - Non-manifold edges → **warn** (cannot fix by remeshing)
7. If remeshing: uses `GmshMesher.mesh_from_file()` with `target_edge_length = min(lambda/epw, mean_edge)`, then shows the remeshed model
8. For large meshes (>2000 estimated basis functions), recommends GMRES over direct LU and prompts for solver choice
9. Runs the solve via `Simulation(config, mesh=mesh)` with `enable_report=True`
10. Computes bistatic RCS (361 points, theta 0→pi, phi=0) and saves:
    - Polar RCS plot to `images/<name>_rcs_bistatic.png`
    - 3D surface current plot (linear) to `images/<name>_surface_current.png`
    - 3D surface current plot (dB scale) to `images/<name>_surface_current_dB.png`
11. Plot titles include plane wave configuration (polarization and propagation direction) when applicable

**Generated plots** (`images/`):

| File | Content |
|---|---|
| `<name>_rcs_bistatic.png` | Polar bistatic RCS plot with plane wave info |
| `<name>_surface_current.png` | 3D surface current density heatmap (linear scale) |
| `<name>_surface_current_dB.png` | 3D surface current density heatmap (dB scale) |

**Relevant code pattern — solving with a pre-built mesh:**

```python
from pyMoM3d import Simulation, SimulationConfig, PlaneWaveExcitation

# Load and optionally remesh
mesh = GmshMesher(target_edge_length=0.02).mesh_from_file("object.stl")

# Solve — pass mesh directly, no geometry needed
exc = PlaneWaveExcitation(E0=np.array([1, 0, 0]), k_hat=np.array([0, 0, -1]))
config = SimulationConfig(frequency=1e9, excitation=exc, enable_report=True)
sim = Simulation(config, mesh=mesh)
result = sim.run()
basis = sim.basis  # access basis functions after construction
```

---

## Writing Your Own Simulations

### Template: Scattering Problem

```python
import numpy as np
from pyMoM3d import (
    Sphere,  # or any geometry primitive
    GmshMesher, compute_rwg_connectivity,
    fill_impedance_matrix, PlaneWaveExcitation, solve_direct,
    compute_far_field, compute_rcs,
    plot_surface_current,
    eta0, c0,
)

# Parameters
frequency = 1e9
k = 2 * np.pi * frequency / c0

# Mesh (Gmsh gives direct control over element size)
geometry = Sphere(radius=0.1)
mesh = GmshMesher(target_edge_length=0.02).mesh_from_geometry(geometry)
basis = compute_rwg_connectivity(mesh)
mesh.check_density(frequency)

# Solve
Z = fill_impedance_matrix(basis, mesh, k, eta0)
exc = PlaneWaveExcitation(E0=np.array([1,0,0]), k_hat=np.array([0,0,-1]))
V = exc.compute_voltage_vector(basis, mesh, k)
I = solve_direct(Z, V)

# Post-process
theta = np.linspace(0.001, np.pi - 0.001, 181)
phi = np.zeros_like(theta)
E_th, E_ph = compute_far_field(I, basis, mesh, k, eta0, theta, phi)
rcs = compute_rcs(E_th, E_ph)

# Visualize
plot_surface_current(I, basis, mesh, cmap='hot')
```

### Template: Antenna Problem

```python
import numpy as np
from pyMoM3d import (
    RectangularPlate, GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig,
    compute_far_field, eta0, c0,
)
from pyMoM3d.mom.excitation import DeltaGapExcitation, find_nearest_edge

# Mesh the antenna
plate = RectangularPlate(0.15, 0.01)
mesh = GmshMesher(target_edge_length=0.005).mesh_from_geometry(plate)
basis = compute_rwg_connectivity(mesh)

# Feed at center
feed_idx = find_nearest_edge(mesh, basis, np.array([0, 0, 0]))
exc = DeltaGapExcitation(basis_index=feed_idx, voltage=1.0)

# Sweep
config = SimulationConfig(frequency=1e9, excitation=exc)
sim = Simulation(config, mesh=mesh)
results = sim.sweep(np.linspace(0.5e9, 1.5e9, 21).tolist())

# Extract impedance
Z_in = np.array([r.Z_input for r in results])
frequencies = np.array([r.frequency for r in results])

# Radiation pattern at a single frequency
result = results[10]  # middle frequency
k = 2 * np.pi * result.frequency / c0
theta = np.linspace(0.001, np.pi, 181)
E_th, E_ph = compute_far_field(
    result.I_coefficients, basis, mesh, k, eta0,
    theta, np.zeros_like(theta),
)
```

### Template: Loading External Geometry

```python
from pyMoM3d import load_stl, GmshMesher, compute_rwg_connectivity

# STL or OBJ via trimesh (preserves original triangulation)
mesh = load_stl("my_antenna.stl")
mesh = load_stl("my_antenna.obj")

# STL/OBJ via Gmsh (with remeshing control)
mesh = load_stl("my_antenna.stl", mesher='gmsh')

# STEP/IGES via Gmsh
mesh = GmshMesher(target_edge_length=0.01).mesh_from_file("antenna.step")

basis = compute_rwg_connectivity(mesh)
# ... proceed with fill_impedance_matrix, etc.
```

---

## Unit Tests

The test suite is organized by module:

| Test file | What it tests |
|---|---|
| `test_geometry.py` | Geometry primitives (vertices, bounding boxes) |
| `test_gmsh_mesher.py` | Gmsh mesher: all 5 primitives, surface area, RWG, edge length control |
| `test_mesh.py` | Mesh creation, edge extraction, statistics, validation |
| `test_rwg_basis.py` | RWG basis computation, boundary edges, mesh density |
| `test_greens.py` | Quadrature rules, Green's function, singularity extraction |
| `test_impedance.py` | Z-matrix symmetry, diagonal properties, convergence, conditioning |
| `test_solver.py` | Excitation vectors, direct/GMRES solvers, surface current eval |
| `test_far_field.py` | Far-field computation, RCS, Mie series validation |
| `test_simulation.py` | Simulation driver, frequency sweep, save/load, analysis utilities |

Run all tests:

```bash
PYTHONPATH=src pytest tests/ -v
```

Run a single test file:

```bash
PYTHONPATH=src pytest tests/test_impedance.py -v
```

Run a single test:

```bash
PYTHONPATH=src pytest tests/test_greens.py::TestSingularity::test_green_self_term_finite -v
```
