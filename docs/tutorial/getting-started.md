# Getting Started

## What is pyMoM3d?

pyMoM3d is a Python library that solves electromagnetic scattering and radiation problems on 3D PEC (Perfect Electric Conductor) surfaces using the Method of Moments (MoM) with RWG (Rao-Wilton-Glisson) basis functions on triangular surface meshes.

Given a conducting surface geometry and an excitation (plane wave or delta-gap feed), the solver:

1. Generates a conformal triangular mesh via [Gmsh](https://gmsh.info/)
2. Assigns RWG basis functions to interior edges
3. Assembles the dense N×N impedance matrix by evaluating pairwise triangle interactions with singularity extraction
4. Solves the linear system for surface current coefficients
5. Post-processes into far-field patterns, RCS, input impedance, or multi-port Z/Y/S network parameters

The solver supports three integral equation formulations:

- **EFIE** (Electric Field Integral Equation) — general open and closed surfaces
- **MFIE** (Magnetic Field Integral Equation) — closed surfaces only
- **CFIE** (Combined Field Integral Equation) — closed surfaces; eliminates spurious interior resonances

Performance-critical matrix assembly is implemented in C++17 with OpenMP parallelism (100–500× faster than pure Python). Numba JIT and NumPy reference backends are also available.

---

## Prerequisites

- Python 3.10+
- **Optional (strongly recommended):** C++17 compiler and CMake 3.18+ for the native acceleration backend

---

## Installation

```bash
# Clone and enter the repo
cd pyMoM3d

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install pyMoM3d in editable mode
pip install -e .

# Verify
python -c "import pyMoM3d; print(pyMoM3d.__version__)"
# → 0.2.0
```

### Runtime dependencies (`requirements.txt`)

| Package | Purpose |
|---|---|
| `numpy` | Array computation, LAPACK solvers |
| `scipy` | Mie series (spherical Bessel), GMRES |
| `matplotlib` | Plotting and visualization |
| `gmsh` | Surface meshing |
| `pybind11` | C++ extension build support |
| `numba` | JIT-compiled fallback backend |

**Optional:**
- `tkinter` — file selection dialog in `stl_rcs_example.py`. macOS: `brew install python-tk@3.13`. Ubuntu/Debian: `sudo apt install python3-tk`. Windows: included with standard Python installer.

---

## Building the C++ Backend

The C++ acceleration backend provides the fastest matrix assembly and is strongly recommended for any mesh with more than ~100 basis functions.

```bash
# Requires: C++17 compiler, CMake 3.18+, pybind11 (pip install pybind11)
# OpenMP is optional but recommended for multi-core parallelism

python build_cpp.py build_ext --inplace
```

On macOS, OpenMP is not bundled with the system compiler. Install via Homebrew:

```bash
brew install libomp
python build_cpp.py build_ext --inplace
```

After building, the extension is auto-detected. You can check at runtime:

```python
from pyMoM3d.mom import _cpp_kernels
print(_cpp_kernels.OMP_ENABLED)   # True if OpenMP was found at build time
```

If the C++ backend is not built, pyMoM3d falls back to Numba (if installed), then to pure NumPy. All three backends produce identical numerical results.

---

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

# 2. Generate mesh (~lambda/10 elements)
sphere = Sphere(radius=radius)
mesher = GmshMesher(target_edge_length=0.02)
mesh = mesher.mesh_from_geometry(sphere)
basis = compute_rwg_connectivity(mesh)

print(f"Triangles: {mesh.get_num_triangles()},  RWG basis: {basis.num_basis}")

# 3. Assemble impedance matrix and solve
Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
exc = PlaneWaveExcitation(
    E0=np.array([1.0, 0.0, 0.0]),
    k_hat=np.array([0.0, 0.0, -1.0]),
)
V = exc.compute_voltage_vector(basis, mesh, k)
I = solve_direct(Z, V)

# 4. Compute bistatic RCS in the xz-plane
theta = np.linspace(0.001, np.pi - 0.001, 181)
phi   = np.zeros_like(theta)
E_theta, E_phi = compute_far_field(I, basis, mesh, k, eta0, theta, phi)
rcs_dBsm = compute_rcs(E_theta, E_phi)

# 5. Visualize
plot_mesh_3d(mesh, show_edges=True)
plot_surface_current(I, basis, mesh, cmap='hot')
```

---

## Quick Start: CFIE for Closed Surfaces

EFIE on a closed PEC surface produces singular or poorly conditioned systems near the interior cavity resonances of the enclosed volume. Use CFIE when simulating closed objects (spheres, closed shells, IC packages) at frequencies at or above the first resonance:

```python
import numpy as np
from pyMoM3d import (
    Sphere, Simulation, SimulationConfig, PlaneWaveExcitation, c0,
)

exc = PlaneWaveExcitation(
    E0=np.array([1.0, 0.0, 0.0]),
    k_hat=np.array([0.0, 0.0, -1.0]),
)

# CFIE: combines EFIE and MFIE via blending parameter alpha ∈ (0, 1)
# alpha=0.5 is the standard choice; any value strictly between 0 and 1 suppresses
# interior resonances.
config = SimulationConfig(
    frequency=1.5e9,
    excitation=exc,
    formulation='CFIE',
    cfie_alpha=0.5,
)
sim = Simulation(
    config,
    geometry=Sphere(radius=0.1),
    target_edge_length=0.02,
)
result = sim.run()

print(f"Condition number: {result.condition_number:.2e}")
```

**When to use CFIE vs EFIE:**

| Surface type | Recommended formulation |
|---|---|
| Open (plate, patch antenna, strip) | EFIE — symmetric, cheaper assembly |
| Closed (sphere, cube, enclosure) at low frequency | EFIE or CFIE both work |
| Closed surface at or above first interior resonance | **CFIE required** |

---

## Quick Start: Multi-Port Network Extraction (Z/Y/S Parameters)

The `Port` and `NetworkExtractor` API extracts Z/Y/S network parameters directly from a MoM simulation. The mesh and RWG basis are built once; the system matrix is assembled once per frequency and solved for all P ports simultaneously.

### Single-Port Impedance Sweep

```python
import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig,
    Port, NetworkExtractor, c0,
)
from pyMoM3d.mom.excitation import StripDeltaGapExcitation, find_feed_edges

# 1. Mesh a strip dipole with a conformal feed edge at x=0
mesher = GmshMesher(target_edge_length=0.008)
mesh = mesher.mesh_plate_with_feed(
    width=0.15,       # dipole length (m)
    height=0.01,      # strip width (m)
    feed_x=0.0,       # forced mesh line at the feed gap
)
basis = compute_rwg_connectivity(mesh)

# 2. Define the port: all transverse RWG edges crossing x=0
feed_indices = find_feed_edges(mesh, basis, feed_x=0.0)
port = Port(name='P1', feed_basis_indices=feed_indices)

# 3. Initialize simulation (excitation only needed to build Simulation object)
exc = StripDeltaGapExcitation(feed_basis_indices=feed_indices, voltage=1.0)
config = SimulationConfig(frequency=1e9, excitation=exc)
sim = Simulation(config, mesh=mesh)

# 4. Extract network parameters over a frequency sweep
extractor = NetworkExtractor(sim, [port], Z0=50.0)
freqs = np.linspace(0.5e9, 1.5e9, 21)
results = extractor.extract(freqs.tolist())

# 5. Read out Z11 and S11
for r in results:
    Z11 = r.Z_matrix[0, 0]
    S11 = r.S_matrix[0, 0]
    print(f"f={r.frequency/1e9:.2f} GHz  "
          f"Z11={Z11.real:+.1f}{Z11.imag:+.1f}j Ω  "
          f"|S11|={20*np.log10(abs(S11)):.1f} dB")
```

### Two-Port Mutual Impedance

```python
from pyMoM3d import combine_meshes

# Two identical dipoles at x=0 and x=0.10 m
mesh1 = mesher.mesh_plate_with_feed(width=0.15, height=0.01, feed_x=0.00,
                                    center=(0.00, 0, 0))
mesh2 = mesher.mesh_plate_with_feed(width=0.15, height=0.01, feed_x=0.10,
                                    center=(0.10, 0, 0))
combined, _ = combine_meshes([mesh1, mesh2])
basis = compute_rwg_connectivity(combined)

# Ports: one per dipole, located at the respective feed planes
port1 = Port.from_x_plane(combined, basis, x_coord=0.00, name='P1')
port2 = Port.from_x_plane(combined, basis, x_coord=0.10, name='P2')

config = SimulationConfig(frequency=1e9,
                          excitation=StripDeltaGapExcitation(port1.feed_basis_indices))
sim = Simulation(config, mesh=combined)

extractor = NetworkExtractor(sim, [port1, port2])
[nr] = extractor.extract(1e9)   # single frequency

print(f"Z11 = {nr.Z_matrix[0,0]:.1f} Ω")
print(f"Z12 = {nr.Z_matrix[0,1]:.1f} Ω  (mutual impedance)")
print(f"Reciprocity: |Z12-Z21|/|Z11| = "
      f"{abs(nr.Z_matrix[0,1]-nr.Z_matrix[1,0])/abs(nr.Z_matrix[0,0]):.2e}")

# S-parameters (50 Ω reference)
S = nr.S_matrix
print(f"S11 = {20*np.log10(abs(S[0,0])):.1f} dB")
print(f"S21 = {20*np.log10(abs(S[1,0])):.1f} dB")
```

### Reference Plane De-embedding

```python
# Shift the reference plane by electrical lengths [theta1, theta2] at each port
import numpy as np
f = 1e9
k = 2 * np.pi * f / c0
delta_l = 0.01   # 1 cm offset from the physical port plane

deembedded = nr.deembed_phase([k * delta_l, k * delta_l])
print(f"De-embedded Z11 = {deembedded.Z_matrix[0,0]:.1f} Ω")
```

---

## Quick Start: Load an STL/OBJ File

```python
from pyMoM3d import GmshMesher, compute_rwg_connectivity, load_stl
from pyMoM3d import Simulation, SimulationConfig, PlaneWaveExcitation
import numpy as np

# Load with Gmsh for element size control
mesh = GmshMesher(target_edge_length=0.02).mesh_from_file("my_object.stl")
# or: mesh = load_stl("my_object.stl")  # uses default Gmsh settings

# Solve using the high-level driver
exc = PlaneWaveExcitation(E0=np.array([1, 0, 0]), k_hat=np.array([0, 0, -1]))
config = SimulationConfig(frequency=1e9, excitation=exc)
sim = Simulation(config, mesh=mesh)
result = sim.run()
```

---

## Quick Start: High-Level Simulation Driver

The `Simulation` class orchestrates the full pipeline. It accepts a geometry primitive and calls Gmsh automatically:

```python
import numpy as np
from pyMoM3d import (
    Sphere, Simulation, SimulationConfig,
    PlaneWaveExcitation,
)

exc = PlaneWaveExcitation(
    E0=np.array([1.0, 0.0, 0.0]),
    k_hat=np.array([0.0, 0.0, -1.0]),
)
config = SimulationConfig(frequency=1.5e9, excitation=exc)
sim = Simulation(config, geometry=Sphere(radius=0.1), target_edge_length=0.02)
result = sim.run()

print(f"Condition number: {result.condition_number:.2e}")

# Frequency sweep — mesh and RWG basis built once, Z-fill and solve per frequency
results = sim.sweep([1.0e9, 1.5e9, 2.0e9])
for r in results:
    print(f"f={r.frequency/1e9:.1f} GHz  cond={r.condition_number:.2e}")
```

---

## Running Tests

```bash
# All tests
pytest tests/

# Specific file or class
pytest tests/test_impedance.py
pytest tests/test_network.py::TestNetworkExtractor

# Verbose
pytest tests/ -v
```

The test suite covers: geometry, meshing, RWG basis, Green's function quadrature, singularity extraction, impedance matrix assembly (EFIE/MFIE/CFIE), solvers, far-field computation, the Simulation driver, and the full network extraction pipeline (Port, NetworkExtractor, NetworkResult).

---

## Running Examples

Each script in `examples/` is standalone:

```bash
# PEC sphere bistatic RCS vs Mie series
python examples/sphere_rcs_validation.py

# CFIE vs EFIE resonance comparison on a sphere
python examples/sphere_cfie_validation.py

# Strip dipole impedance sweep — Z11, S11 vs frequency
python examples/dipole_impedance_sweep.py

# Network extraction: single-port S11 + two-port Z/S-matrix
python examples/network_extraction_demo.py

# Backend performance comparison (C++ vs Numba vs NumPy)
python examples/backend_comparison.py

# Rectangular plate plane-wave scattering
python examples/plate_scattering.py

# 8-element linear array with mutual coupling and beam steering
python examples/dipole_array.py

# Interactive STL/OBJ → RCS + surface current (requires tkinter)
python examples/stl_rcs_example.py
```

Each example saves plots to the `images/` directory and prints diagnostic output to the terminal.

---

## Project Layout

```
pyMoM3d/
  src/
    cpp/
      singularity.hpp       Header-only analytical + quadrature kernels
      mom_kernel.cpp        OpenMP EFIE / MFIE / CFIE fill (pybind11)
      CMakeLists.txt        C++ build configuration
    pyMoM3d/
      geometry/             Parametric geometry primitives
      mesh/                 Mesh data structures, Gmsh mesher, RWG basis
      greens/               Green's function, quadrature, singularity extraction
      mom/
        operators/          EFIE / MFIE / CFIE operator strategy classes
        assembly.py         Unified matrix fill with backend dispatch
        excitation.py       Plane wave, delta-gap, multi-port excitations
        numba_kernels.py    Numba JIT kernels (EFIE / MFIE / CFIE)
        impedance.py        EFIE fill (backward-compatible wrapper)
        solver.py           Direct (LU) and GMRES solvers
      network/
        port.py             Port dataclass and factory methods
        extractor.py        NetworkExtractor — batched multi-port solve
        network_result.py   NetworkResult — Z/Y/S properties, de-embedding
      arrays/               LinearDipoleArray, combine_meshes, array factor
      fields/               Far-field computation, RCS
      analysis/             Mie series, convergence studies, pattern analysis
      visualization/        3D mesh and surface current plotting
      utils/                Constants, progress reporter, report writer
      simulation.py         High-level simulation driver
  build_cpp.py              C++ extension build script (setuptools + pybind11)
  tests/                    Unit and integration tests
  examples/                 Standalone example scripts
  docs/                     Documentation
  images/                   Plot output from examples
```

---

## Next Steps

- [Architecture](architecture.md) — operator pattern, backend dispatch, module design
- [API Reference](api-reference.md) — complete public API
- [Examples Guide](examples-guide.md) — detailed walkthrough of each example
