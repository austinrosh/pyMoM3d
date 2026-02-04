# Architecture Guide

This document explains the solver pipeline, the electromagnetic theory behind each stage, and how the code modules map to that theory.

## Solver Pipeline Overview

```
Geometry  -->  Mesh  -->  RWG Basis  -->  Z-matrix Fill  -->  Solve  -->  Post-process
(primitives)  (triangles)  (basis fns)   (EFIE assembly)    (ZI=V)    (far-field, RCS)
                                              |
                                              v
                                    Characteristic Mode Analysis
                                    (modal decomposition of Z)
```

Each stage is implemented in a dedicated module. Data flows left to right; the output of each stage is the input to the next.

## Stage 1: Geometry (`geometry/`)

Five parametric primitives are available:

| Primitive | Constructor Parameters | Surface Type |
|---|---|---|
| `RectangularPlate` | `width`, `height`, `center` | Open flat surface |
| `Sphere` | `radius`, `center` | Closed surface |
| `Cylinder` | `radius`, `height`, `center` | Closed surface (with caps via Gmsh) |
| `Cube` | `side_length`, `center` | Closed surface |
| `Pyramid` | `base_size`, `height`, `center` | Closed surface |

### Meshing with Gmsh (recommended)

`GmshMesher` generates meshes directly from geometry primitives using the Gmsh CAD kernel. This provides control over element size, curvature adaptation, and mesh quality:

```python
from pyMoM3d import Sphere, GmshMesher

mesher = GmshMesher(target_edge_length=0.02)
mesh = mesher.mesh_from_geometry(Sphere(radius=0.1))
```

Individual meshing methods are also available: `mesh_sphere()`, `mesh_plate()`, `mesh_cylinder()`, `mesh_cube()`, `mesh_pyramid()`.

### Meshing with trimesh (legacy)

Each primitive also has a `to_trimesh(subdivisions=...)` method that returns a `trimesh.Trimesh` object for use with the `PythonMesher`:

```python
from pyMoM3d import Sphere, PythonMesher
mesh = PythonMesher().mesh_from_geometry(Sphere(radius=0.1).to_trimesh(subdivisions=2))
```

### Feed-Line Meshing

For strip dipole antennas, `GmshMesher.mesh_plate_with_feed()` creates a plate mesh with a forced transverse mesh line at the feed location. This ensures conformal edges exist at the feed point for accurate delta-gap excitation:

```python
mesh = GmshMesher(target_edge_length=0.005).mesh_plate_with_feed(
    width=0.15, height=0.01, feed_x=0.0,
)
```

### Loading from File

STL and OBJ files (and with Gmsh: STEP, IGES) can be loaded directly:

```python
from pyMoM3d import load_stl
mesh = load_stl("my_model.stl")                  # via trimesh (preserves original triangulation)
mesh = load_stl("my_model.stl", mesher='gmsh')   # via Gmsh (remeshes with size control)
```

With `GmshMesher`, you can also load STEP/IGES files and control element size:

```python
from pyMoM3d import GmshMesher
mesh = GmshMesher(target_edge_length=0.01).mesh_from_file("antenna.step")
```

Both `.stl` and `.obj` formats are supported by `trimesh.load()` and `gmsh.merge()`. OBJ files with multiple material groups are automatically concatenated into a single mesh. When remeshing STL/OBJ files, `GmshMesher.mesh_from_file()` uses uniform mesh refinement (`gmsh.model.mesh.refine()`) to reach the target edge length, which is fast and reliable on complex discrete meshes.

The interactive example `stl_rcs_example.py` provides automatic mesh quality assessment, configurable resolution presets (coarse/medium/fine), and remesh recommendations when loading external files.

## Stage 2: Mesh (`mesh/`)

### Mesh Data Structure

The `Mesh` class (`mesh/mesh_data.py`) stores the triangulated surface:

| Attribute | Shape | Description |
|---|---|---|
| `vertices` | `(N_v, 3)` | Vertex coordinates (float64) |
| `triangles` | `(N_t, 3)` | Triangle vertex indices (int32) |
| `edges` | `(N_e, 2)` | Edge vertex indices |
| `triangle_normals` | `(N_t, 3)` | Outward unit normals |
| `triangle_areas` | `(N_t,)` | Triangle areas |
| `edge_lengths` | `(N_e,)` | Edge lengths |
| `edge_to_triangles` | `dict` | Edge index -> list of triangle indices |

Edges and derived quantities are computed automatically when the mesh is created. Both `GmshMesher` and `PythonMesher` produce `Mesh` objects. `GmshMesher` generates clean meshes directly from the Gmsh kernel; `PythonMesher` handles conversion from `trimesh` objects with mesh cleaning (merging duplicate vertices, removing degenerate faces, fixing winding orientation).

### Mesh Quality

The mesh should satisfy lambda/10 density at the operating frequency for accurate results. The `mesh.check_density(frequency)` method warns if the mean edge length exceeds lambda/10:

```python
mesh.check_density(1.5e9)  # Warns if too coarse for 1.5 GHz
stats = mesh.get_statistics()
print(f"Mean edge: {stats['mean_edge_length']:.4f} m")
```

The `mesh.validate()` method checks topological integrity (duplicate vertices, degenerate triangles, non-manifold edges, consistent orientation).

## Stage 3: RWG Basis Functions (`mesh/`)

### Background

RWG (Rao-Wilton-Glisson) basis functions are the standard choice for MoM on triangular meshes. Each basis function is associated with one **interior edge** (shared by exactly two triangles). Boundary edges (on open surfaces like plates) do not produce basis functions.

For basis function `n` with shared edge of length `l_n`:

```
         T+ (area A+)              T- (area A-)
        /    \                    /    \
       / rho+ \                  / rho- \
      /   ->   \   shared edge  /   <-   \
     /          \--------------/          \
    v_free+      edge v_a--v_b      v_free-
```

- **T+**: triangle where current flows *away* from the free vertex
  - `f_n(r) = (l_n / 2A+) * (r - r_free+)`
  - Divergence: `+l_n / A+`

- **T-**: triangle where current flows *toward* the free vertex
  - `f_n(r) = (l_n / 2A-) * (r_free- - r)`
  - Divergence: `-l_n / A-`

### Code

`compute_rwg_connectivity(mesh)` walks the mesh topology to:
1. Identify interior edges (shared by exactly 2 triangles)
2. Assign T+ and T- for each edge
3. Find the free vertex (opposite the shared edge) in each triangle
4. Compute edge lengths and triangle areas

The result is an `RWGBasis` object with per-basis-function arrays:

```python
basis = compute_rwg_connectivity(mesh)
print(f"{basis.num_basis} RWG basis functions")
print(f"{basis.num_boundary_edges} boundary edges (excluded)")
```

## Stage 4: Impedance Matrix Assembly (`mom/impedance.py`)

### EFIE Formulation

The solver implements the Electric Field Integral Equation (EFIE) for PEC surfaces. Testing with RWG basis functions produces the linear system `ZI = V`:

```
Z_mn = jk*eta * [ A_mn - (1/k^2) * Phi_mn ]
```

where:

- **A_mn** (vector potential term): `integral integral f_m(r) . f_n(r') g(r,r') dS dS'`
- **Phi_mn** (scalar potential term): `integral integral div(f_m(r)) div(f_n(r')) g(r,r') dS dS'`
- **g(r,r')** = `exp(-jkR) / (4*pi*R)` is the free-space Green's function
- **eta** = intrinsic impedance of free space (~377 Ohms)
- **k** = wavenumber = `2*pi*f/c`

Each (m, n) entry involves 4 triangle-triangle interactions (T+_m with T+_n, T+_m with T-_n, T-_m with T+_n, T-_m with T-_n).

### Singularity Handling

When the observation point is on or near the source triangle, the Green's function kernel `1/R` is singular. The code uses **singularity extraction** (Wilton et al. 1984, Graglia 1993):

```
g(R) = 1/(4*pi*R) + [g(R) - 1/(4*pi*R)]
        ^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^
        analytical    smooth (quadrature)
```

The `1/(4*pi*R)` part is integrated analytically over the triangle using the Graglia edge-based formulae. The smooth remainder `[exp(-jkR)/(4*pi*R) - 1/(4*pi*R)]` has a finite limit as R->0 (`-jk/4*pi`) and is handled by standard Gauss quadrature.

This decomposition is applied to both the scalar potential term (`integrate_green_singular`) and the vector potential term (`integrate_rho_green_singular`).

### Quadrature

Triangle integration uses symmetric Gauss quadrature rules (Dunavant 1985) in barycentric coordinates. Available orders: 1, 3, 4, 7, 13 points. The default `quad_order=4` is sufficient for most cases; use 7 or 13 for higher accuracy at the cost of computation time.

### Matrix Properties

The impedance matrix Z is:
- **Square**: N x N (N = number of RWG basis functions)
- **Complex symmetric**: Z_mn = Z_nm (reciprocity)
- **Dense**: every pair of basis functions interacts
- **Well-conditioned** when singularity extraction is working correctly (condition number ~10-1000 for typical meshes)

```python
Z = fill_impedance_matrix(basis, mesh, k, eta0, quad_order=4)
print(f"Shape: {Z.shape}")
print(f"Symmetric: {np.allclose(Z, Z.T)}")
print(f"Condition: {np.linalg.cond(Z):.2e}")
```

## Stage 5: Excitation and Solve (`mom/excitation.py`, `mom/solver.py`)

### Excitation Sources

**Plane wave** — models an incoming electromagnetic wave:

```python
exc = PlaneWaveExcitation(
    E0=np.array([1.0, 0.0, 0.0]),      # x-polarized, 1 V/m
    k_hat=np.array([0.0, 0.0, -1.0]),   # propagating in -z
)
V = exc.compute_voltage_vector(basis, mesh, k)
```

The voltage vector entry is: `V_m = integral f_m(r) . E_inc(r) dS`

**Delta-gap feed** — models a voltage source across a single edge (for antennas):

```python
from pyMoM3d.mom.excitation import DeltaGapExcitation, find_nearest_edge

feed_idx = find_nearest_edge(mesh, basis, np.array([0.0, 0.0, 0.0]))
exc = DeltaGapExcitation(basis_index=feed_idx, voltage=1.0)
V = exc.compute_voltage_vector(basis, mesh, k)
# V[feed_idx] = 1.0, all others = 0
```

**Strip delta-gap feed** — distributes the voltage across all transverse edges at the feed location. More accurate for strip dipoles:

```python
from pyMoM3d.mom.excitation import StripDeltaGapExcitation, find_feed_edges

# Find all basis functions whose shared edge crosses x=0 transversely
feed_indices = find_feed_edges(mesh, basis, feed_x=0.0)
exc = StripDeltaGapExcitation(feed_basis_indices=feed_indices, voltage=1.0)
V = exc.compute_voltage_vector(basis, mesh, k)

# After solving, compute input impedance
Z_in = exc.compute_input_impedance(I, basis, mesh)
```

### Solvers

**Direct (LU)** — exact solution, O(N^3):

```python
I = solve_direct(Z, V)
```

**GMRES** — iterative, useful for large problems:

```python
I = solve_gmres(Z, V, tol=1e-6, maxiter=1000)
```

Both return the current coefficient vector `I` (shape `(N,)`, complex128).

### Input Impedance (Delta-Gap)

For delta-gap excitation, the input impedance is:

```
Z_in = V_gap / I[feed_index]
```

The `Simulation` class computes this automatically and stores it in `result.Z_input`.

## Stage 6: Characteristic Mode Analysis (`analysis/cma.py`)

### Background

Characteristic Mode Analysis (CMA) decomposes the impedance matrix into intrinsic current modes that are independent of the excitation. This provides physical insight into how a structure naturally supports currents at different frequencies.

### Theory

The impedance matrix Z = R + jX is decomposed into the radiation resistance matrix R and reactance matrix X. The characteristic modes are eigenvectors of the generalized eigenvalue problem:

```
X · J_n = λ_n · R · J_n
```

where λ_n is the characteristic eigenvalue and J_n is the characteristic current.

### Key Quantities

- **Modal significance**: MS_n = 1 / sqrt(1 + λ_n²) — measures excitability (MS=1 at resonance)
- **Characteristic angle**: α_n = 180° - arctan(λ_n) — phase relative to excitation

### Code

```python
from pyMoM3d import compute_characteristic_modes, verify_orthogonality

# After computing Z matrix
cma = compute_characteristic_modes(Z, frequency=1e9, num_modes=5)

# Access modes by significance rank
J_mode1 = cma.get_mode(0)  # Most significant mode
lambda_1 = cma.get_eigenvalue(0)
ms_1 = cma.get_modal_significance(0)

# Verify orthogonality
is_orthog, error = verify_orthogonality(cma)

# Use with Simulation class
sim = Simulation(config, geometry=Sphere(radius=0.1), mesher='gmsh')
cma = sim.compute_cma(frequency=1e9, num_modes=10)

# Frequency sweep with mode tracking
results, tracking = sim.cma_sweep(frequencies, num_modes=5, track_modes=True)
```

### Numerical Considerations

- The R matrix can be ill-conditioned for electrically small structures (weak radiation)
- Regularization is applied automatically: R_reg = R + ε·max(||R||, ||X||)·I
- Mode tracking across frequency uses eigenvector correlation

---

## Stage 7: Post-Processing (`fields/`, `analysis/`, `visualization/`)

### Far-Field Radiation

The far-field electric field is computed from the radiation integral:

```
E_far(r) ~ -jk*eta/(4*pi) * exp(-jkr)/r * N(theta, phi)
```

where `N` is the vector radiation function summing contributions from all basis functions with phase `exp(+jk * r_hat . r')`.

```python
theta = np.linspace(0.001, np.pi - 0.001, 181)
phi = np.zeros_like(theta)
E_theta, E_phi = compute_far_field(I, basis, mesh, k, eta0, theta, phi)
```

### Radar Cross Section

Bistatic RCS in dBsm:

```python
rcs_dBsm = compute_rcs(E_theta, E_phi, E_inc_mag=1.0)
# sigma = 4*pi * (|E_theta|^2 + |E_phi|^2) / |E_inc|^2
```

### Mie Series Validation

For PEC spheres, the exact Mie series solution is available for validation:

```python
from pyMoM3d.analysis.mie_series import mie_rcs_pec_sphere
rcs_mie = mie_rcs_pec_sphere(ka, theta)  # normalized by pi*a^2
```

### Surface Current Visualization

Map RWG coefficients to per-triangle current density and plot as a 3D heatmap:

```python
from pyMoM3d import plot_surface_current, compute_triangle_current_density

# Heatmap on 3D mesh
plot_surface_current(I, basis, mesh, cmap='hot')

# Raw per-triangle |J| values
J_mag = compute_triangle_current_density(I, basis, mesh)
```

### Analysis Utilities

```python
from pyMoM3d.analysis import (
    compute_s11,               # S11 from Z_in
    impedance_vs_frequency,    # Extract Z_in from sweep results
    s11_vs_frequency,          # S11 in dB from sweep results
    compute_directivity,       # Directivity from far-field pattern
    compute_beamwidth_3dB,     # 3dB beamwidth
    mesh_convergence_study,    # Automated convergence study
)
```

## Reporting System (`utils/`)

The simulation pipeline supports progress reporting and post-run report generation.

**Progress reporters** (`utils/reporter.py`):
- `TerminalReporter` — writes human-readable progress to stderr with TTY-based in-place updates. Used by default.
- `SilentReporter` — no-op reporter for tests or batch runs.
- `RecordingReporter` — wraps another reporter while accumulating metadata for report generation.

**Report generation** (`utils/report_writer.py`):
When `SimulationConfig(enable_report=True)` is set, the `Simulation` class uses a `RecordingReporter` to track all stages and writes a structured text report to `results/simulation_info/` after the run completes. Reports include configuration, mesh statistics, RWG basis info, matrix assembly timing, solver results, warnings, and errors.

```python
config = SimulationConfig(
    frequency=1e9, excitation=exc,
    enable_report=True,          # enable report generation
    report_dir='results/simulation_info',  # output directory (default)
)
```

## Conventions

All conventions are documented in `CONVENTIONS.md` at the repo root:

| Convention | Value |
|---|---|
| Time dependence | `exp(-j*omega*t)` |
| Green's function | `exp(-jkR) / (4*pi*R)` |
| Far-field phase | `exp(+jk * r_hat . r')` (opposite sign) |
| Length units | meters |
| Frequency units | Hz |
| Coordinate arrays | `np.float64` |
| Index arrays | `np.int32` |
| Complex fields | `np.complex128` |

## References

- D.R. Wilton, S.M. Rao, A.W. Glisson, D.H. Schaubert, O.M. Al-Bundak, C.M. Butler, "Potential integrals for uniform and linear source distributions on polygonal and polyhedral domains," IEEE Trans. AP-32(3), pp. 276-281, 1984.
- R.D. Graglia, "On the numerical integration of the linear shape functions times the 3-D Green's function or its gradient on a plane triangle," IEEE Trans. AP-41(10), 1993.
- S.M. Rao, D.R. Wilton, A.W. Glisson, "Electromagnetic scattering by surfaces of arbitrary shape," IEEE Trans. AP-30(3), pp. 409-418, 1982.
- D.A. Dunavant, "High degree efficient symmetrical Gaussian quadrature rules for the triangle," Int. J. Numer. Methods Eng. 21, pp. 1129-1148, 1985.
