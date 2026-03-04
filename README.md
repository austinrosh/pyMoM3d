# pyMoM3d

![Version](https://img.shields.io/badge/version-0.2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Python implementation of the 3D Method of Moments (MoM) electromagnetic solver using RWG (Rao-Wilton-Glisson) basis functions on triangular surface meshes.

pyMoM3d solves the Electric Field Integral Equation (EFIE) for induced surface currents on perfect electric conductors (PEC), then computes far-field radiation patterns, radar cross section (RCS), and input impedance.

## Features

- **Geometry primitives**: Rectangular plate, sphere, cylinder, cube, pyramid for canonical scattering analysis and solver benchmarking
- **STL/OBJ import**: Load external mesh files (`.stl`, `.obj`) with automatic quality assessment and optional remeshing
- **Automatic meshing**: Triangular surface meshes via [Gmsh](https://gmsh.info/) with configurable refinement
- **RWG basis functions**: Automatic detection of interior edges and basis function assignment
- **EFIE impedance matrix**: Full dense Z-matrix assembly with singularity extraction (Wilton 1984, Graglia 1993)
- **Excitation sources**: Plane wave, single-edge delta-gap, strip delta-gap, and multi-port excitation for arrays
- **Solvers**: Direct (LU) and iterative (GMRES with diagonal preconditioner)
- **Post-processing**: Far-field computation, bistatic/monostatic RCS, input impedance, S11, directivity, beamwidth
- **Validation**: Built-in Mie series for PEC sphere RCS comparison
- **Visualization**: 3D mesh rendering and surface current density heatmaps 
- **Reporting**: Automatic generates simulation report metadata with terminal progress and text-file summaries

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
python examples/sphere_rcs_validation.py
```

```python
import numpy as np
from pyMoM3d import (
    Sphere, GmshMesher, compute_rwg_connectivity,
    fill_impedance_matrix, PlaneWaveExcitation, solve_direct,
    compute_far_field, compute_rcs, plot_surface_current,
    eta0, c0,
)

# Create mesh (using Gmsh for mesh quality control)
sphere = Sphere(radius=0.1)
mesher = GmshMesher(target_edge_length=0.02)
mesh = mesher.mesh_from_geometry(sphere)
basis = compute_rwg_connectivity(mesh)

# Solve
frequency = 1.5e9
k = 2 * np.pi * frequency / c0
Z = fill_impedance_matrix(basis, mesh, k, eta0)
exc = PlaneWaveExcitation(E0=np.array([1, 0, 0]), k_hat=np.array([0, 0, -1]))
I = solve_direct(Z, exc.compute_voltage_vector(basis, mesh, k))

# Far-field RCS
theta = np.linspace(0.001, np.pi - 0.001, 181)
E_th, E_ph = compute_far_field(I, basis, mesh, k, eta0, theta, np.zeros_like(theta))
rcs_dBsm = compute_rcs(E_th, E_ph)

# Visualize surface currents
plot_surface_current(I, basis, mesh, cmap='hot')
```

Or use the high-level driver (with Gmsh mesher):

```python
from pyMoM3d import Sphere, Simulation, SimulationConfig, PlaneWaveExcitation

exc = PlaneWaveExcitation(E0=np.array([1, 0, 0]), k_hat=np.array([0, 0, -1]))
sim = Simulation(
    SimulationConfig(frequency=1.5e9, excitation=exc),
    geometry=Sphere(radius=0.1),
    mesher='gmsh',
    target_edge_length=0.02,
)
result = sim.run()
```

The trimesh mesher is still available (`mesher='trimesh'`, the default) for backward compatibility.

## Examples

| Example | Description |
|---|---|
| `sphere_rcs_validation.py` | PEC sphere bistatic RCS vs exact Mie series |
| `dipole_impedance_sweep.py` | Strip dipole input impedance, current distribution, radiation pattern |
| `plate_scattering.py` | Rectangular plate plane-wave scattering vs Physical Optics |
| `simulation_driver_demo.py` | High-level `Simulation` API with frequency sweep and save/load |
| `solver_performance.py` | Benchmarks Z-fill and solve time vs mesh size |
| `friis_validation.py` | Two-antenna Friis transmission equation validation: distance and polarization sweeps |
| `dipole_array.py` | 8-element linear dipole array with beam steering, mutual coupling, and pattern validation |
| `stl_rcs_example.py` | Load an STL/OBJ file via file dialog, choose mesh resolution (coarse/medium/fine), assess quality, remesh if needed, compute bistatic RCS and surface current (linear + dB) |

```bash
python examples/sphere_rcs_validation.py
python examples/dipole_impedance_sweep.py
python examples/plate_scattering.py
python examples/simulation_driver_demo.py
python examples/friis_validation.py
python examples/dipole_array.py
python examples/stl_rcs_example.py
```

> **Note:** `stl_rcs_example.py` uses `tkinter` for the file selection dialog. On macOS, the default Homebrew Python may not include tkinter. To install it:
> ```bash
> brew install python-tk@3.13   # match your Python version
> ```
> On Ubuntu/Debian: `sudo apt install python3-tk`. On Windows, tkinter is included with the standard Python installer.

## Project Structure

```
src/pyMoM3d/
    geometry/       Parametric geometry primitives
    mesh/           Mesh data structures, meshing, RWG basis
    greens/         Green's function, quadrature, singularity extraction
    arrays/         Antenna array abstractions (LinearDipoleArray)
    mom/            Impedance matrix, excitation, solvers
    fields/         Far-field computation, RCS
    analysis/       Mie series, convergence studies, impedance analysis
    visualization/  3D mesh and surface current plotting
    utils/          Physical constants, progress reporting, report generation
    simulation.py   High-level simulation driver with STL/OBJ loading
tests/              Unit and integration tests
examples/           Standalone example scripts
docs/               Documentation
```

## Testing

```bash
pytest tests/ -v
```

## Conventions

- **Time dependence**: $\exp(-j*\omega*t)$
- **Units**: meters (length), Hz (frequency), V/m (E-field), Ohms (impedance)
- **Green's function**: $\frac{\exp(-jkR)}{(4*\pi*R)}$
- **Numerical types**: `float64` for coordinates, `int32` for indices, `complex128` for fields


## Documentation

- [Getting Started](docs/getting-started.md) - Installation, quick start, running tests and examples
- [Architecture Guide](docs/architecture.md) - Solver pipeline, EM theory, singularity extraction
- [Examples Guide](docs/examples-guide.md) - Detailed walkthrough of each example with code patterns

## References

- S.M. Rao, D.R. Wilton, A.W. Glisson, "Electromagnetic scattering by surfaces of arbitrary shape," IEEE Trans. AP-30(3), 1982.
- D.R. Wilton et al., "Potential integrals for uniform and linear source distributions on polygonal and polyhedral domains," IEEE Trans. AP-32(3), 1984.
- R.D. Graglia, "On the numerical integration of the linear shape functions times the 3-D Green's function or its gradient on a plane triangle," IEEE Trans. AP-41(10), 1993.

## License

MIT
