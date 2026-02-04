# API Reference

Complete reference for all public modules, classes, and functions.

---

## `pyMoM3d.geometry` — Geometry Primitives

All primitives share a common pattern: construct with physical dimensions, then pass to `GmshMesher.mesh_from_geometry()` (recommended) or call `to_trimesh()` for the legacy trimesh mesher.

### `RectangularPlate(width, height, center=(0,0,0))`

Flat rectangular plate in the z=0 plane.

- `width` (float): Extent along x (meters)
- `height` (float): Extent along y (meters)
- `center` (tuple): Center point (x, y, z)

**Methods:**
- `to_trimesh(subdivisions=None)` -> `trimesh.Trimesh`
- `get_vertices()` -> `ndarray (4, 3)` — corner vertices
- `get_vertex_grid(nx, ny)` -> `ndarray` — refined grid
- `get_bounding_box()` -> `(min_corner, max_corner)`

### `Sphere(radius, center=(0,0,0))`

Icosphere (subdivided icosahedron). Closed surface.

- `to_trimesh(subdivisions=2)` -> `trimesh.Trimesh`
  - `subdivisions=1`: 80 triangles
  - `subdivisions=2`: 320 triangles
  - `subdivisions=3`: 1280 triangles

### `Cylinder(radius, height, center=(0,0,0))`

Cylindrical surface (no end caps). Open surface.

- `to_trimesh(sections=32)` -> `trimesh.Trimesh`

### `Cube(side_length, center=(0,0,0))`

Axis-aligned box. Closed surface.

- `to_trimesh()` -> `trimesh.Trimesh`
- `get_vertex_grid(nx, ny, nz)` -> surface grid on all 6 faces

### `Pyramid(base_size, height, center=(0,0,0))`

Square-base pyramid. Closed surface.

- `to_trimesh()` -> `trimesh.Trimesh`

---

## `pyMoM3d.mesh` — Mesh and Basis Functions

### `Mesh(vertices, triangles)`

Central mesh data structure.

**Constructor parameters:**
- `vertices` (`ndarray (N_v, 3)`, float64): Vertex coordinates
- `triangles` (`ndarray (N_t, 3)`, int32): Triangle connectivity

Edges, normals, areas, and edge lengths are computed automatically.

**Key attributes:**
- `vertices`, `triangles`, `edges`, `triangle_normals`, `triangle_areas`, `edge_lengths`
- `edge_to_triangles` (dict): Edge index -> list of triangle indices
- `rwg_basis` (RWGBasis or None): Populated after `compute_rwg_connectivity()`

**Methods:**
- `get_num_vertices()`, `get_num_triangles()`, `get_num_edges()` -> int
- `get_num_basis_functions()` -> int — number of RWG basis functions (0 if not computed)
- `get_statistics()` -> dict — mesh quality statistics:
  - `num_vertices`, `num_triangles`, `num_edges`
  - `min_area`, `max_area`, `mean_area`
  - `min_edge_length`, `max_edge_length`, `mean_edge_length`
- `check_density(frequency)` -> bool — warns if mesh too coarse for lambda/10
- `validate()` -> dict — topological validation report
- `Mesh.from_trimesh(trimesh_obj)` (classmethod) -> Mesh

### `GmshMesher(target_edge_length=None, min_edge_length=None, max_edge_length=None, curvature_adapt=True, verbosity=0)`

Recommended mesher. Uses the Gmsh Python API for high-quality surface meshing with element size control.

- `target_edge_length` (float, optional): Target edge length in meters. Controls global mesh density.
- `min_edge_length` (float, optional): Minimum edge length. Defaults to `target_edge_length / 5`.
- `max_edge_length` (float, optional): Maximum edge length. Defaults to `target_edge_length * 2`.
- `curvature_adapt` (bool): Enable curvature-based refinement. Default True.
- `verbosity` (int): Gmsh output verbosity (0=silent). Default 0.

**Methods:**
- `mesh_from_geometry(geometry, **kwargs)` -> Mesh — auto-dispatches based on geometry primitive type
- `mesh_sphere(radius, center, target_edge_length=None)` -> Mesh
- `mesh_plate(width, height, center, target_edge_length=None)` -> Mesh
- `mesh_cylinder(radius, height, center, target_edge_length=None)` -> Mesh
- `mesh_cube(side_length, center, target_edge_length=None)` -> Mesh
- `mesh_pyramid(base_size, height, center, target_edge_length=None)` -> Mesh
- `mesh_plate_with_feed(width, height, feed_x=0.0, center=(0,0,0), target_edge_length=None)` -> Mesh — plate with forced transverse mesh line at `feed_x` for conformal delta-gap feed edges
- `mesh_from_file(path)` -> Mesh — load and mesh STL/OBJ/STEP/IGES files

Each method accepts an optional `target_edge_length` override that takes precedence over the instance-level setting.

### `PythonMesher(merge_vertices=True, remove_degenerate=True)`

Legacy mesher. Converts `trimesh.Trimesh` objects into `Mesh` objects with cleaning.

**Methods:**
- `mesh_from_geometry(geometry, triangles=None)` -> Mesh

### `create_mesh_from_trimesh(trimesh_obj)` -> Mesh

Convenience wrapper around `PythonMesher`.

### `create_mesh_from_vertices(vertices, triangles=None)` -> Mesh

Create mesh from raw vertices. If `triangles` is None, uses convex hull.

### `create_rectangular_mesh(width, height, nx, ny, center=(0,0,0))` -> Mesh

Generate a rectangular plate mesh directly (without trimesh).

### `compute_rwg_connectivity(mesh)` -> RWGBasis

Compute RWG basis functions from mesh topology. Stores result in `mesh.rwg_basis`.

Returns an `RWGBasis` object.

### `RWGBasis` (dataclass)

Per-basis-function data for RWG basis functions.

**Attributes:**

| Attribute | Shape | Description |
|---|---|---|
| `num_basis` | int | Number of basis functions |
| `edge_index` | `(N,)` | Index into `mesh.edges` |
| `edge_length` | `(N,)` | Shared edge length l_n |
| `t_plus` | `(N,)` | Triangle index for T+ |
| `t_minus` | `(N,)` | Triangle index for T- |
| `free_vertex_plus` | `(N,)` | Free vertex index in T+ |
| `free_vertex_minus` | `(N,)` | Free vertex index in T- |
| `area_plus` | `(N,)` | Area of T+ |
| `area_minus` | `(N,)` | Area of T- |
| `num_boundary_edges` | int | Boundary edges found |

**Methods:**
- `get_free_vertex_plus_coords(mesh)` -> `ndarray (N, 3)`
- `get_free_vertex_minus_coords(mesh)` -> `ndarray (N, 3)`
- `validate(mesh)` -> None (raises ValueError on inconsistency)

---

## `pyMoM3d.mom` — Method of Moments

### `fill_impedance_matrix(rwg_basis, mesh, k, eta, quad_order=4, near_threshold=0.2, progress_callback=None)` -> `ndarray (N,N) complex128`

Assemble the EFIE impedance matrix Z.

- `rwg_basis` (RWGBasis)
- `mesh` (Mesh)
- `k` (float): Wavenumber (rad/m). Compute as `2*pi*frequency/c0`.
- `eta` (float): Intrinsic impedance. Use `eta0` (~377 Ohms) for free space.
- `quad_order` (int): Gauss quadrature order. Valid: 1, 3, 4, 7, 13. Default 4.
- `near_threshold` (float): Controls when singularity extraction activates. Default 0.2.
- `progress_callback` (callable, optional): Called with `(fraction)` during assembly for progress reporting.

### `PlaneWaveExcitation(E0, k_hat)`

Uniform plane wave: `E_inc(r) = E0 * exp(-jk * k_hat . r)`

- `E0` (`ndarray (3,)`): Polarization vector (V/m). Must be perpendicular to `k_hat`.
- `k_hat` (`ndarray (3,)`): Unit propagation direction. Normalized automatically.

**Methods:**
- `compute_voltage_vector(rwg_basis, mesh, k)` -> `ndarray (N,) complex128`

### `DeltaGapExcitation(basis_index, voltage=1.0)`

Delta-gap voltage source at a single edge.

- `basis_index` (int): Which basis function to excite.
- `voltage` (complex): Applied voltage (V).

**Methods:**
- `compute_voltage_vector(rwg_basis, mesh, k)` -> `ndarray (N,) complex128`

### `StripDeltaGapExcitation(feed_basis_indices, voltage=1.0)`

Delta-gap excitation distributed across multiple transverse edges at the feed location. More physical than a single-edge delta gap for strip dipoles.

- `feed_basis_indices` (list of int): Basis function indices crossing the feed line.
- `voltage` (complex): Applied voltage (V).

**Methods:**
- `compute_voltage_vector(rwg_basis, mesh, k)` -> `ndarray (N,) complex128`
- `compute_input_impedance(I_coeffs, rwg_basis, mesh)` -> complex — computes Z_in from the solved current coefficients

### `find_feed_edges(mesh, rwg_basis, feed_x, tol=None)` -> list of int

Find all interior basis functions whose shared edge crosses `x=feed_x` transversely. Used with `StripDeltaGapExcitation`. The `tol` parameter defaults to twice the minimum edge length.

### `find_nearest_edge(mesh, rwg_basis, point)` -> int

Find the basis function whose shared edge midpoint is closest to `point`. Useful for locating feed edges.

### `solve_direct(Z, V)` -> `ndarray (N,) complex128`

Solve `ZI = V` via LU factorization (`numpy.linalg.solve`).

### `solve_gmres(Z, V, tol=1e-6, maxiter=1000, progress_callback=None)` -> `ndarray (N,) complex128`

Solve `ZI = V` via GMRES with diagonal preconditioner (`scipy.sparse.linalg.gmres`).

- `progress_callback` (callable, optional): Called with `(iteration, residual)` per iteration.

### `evaluate_surface_current(I_coeffs, rwg_basis, mesh, points)` -> `ndarray (M,3) complex128`

Reconstruct surface current density J(r) at arbitrary points on the surface.

---

## `pyMoM3d.greens` — Green's Functions and Quadrature

### `scalar_green(k, r, r_prime)` -> complex or ndarray

Free-space scalar Green's function: `exp(-jkR) / (4*pi*R)`. Supports vectorized inputs.

### `triangle_quad_rule(order)` -> `(weights, barycentric_coords)`

Symmetric Gauss quadrature on the unit triangle.

- `order` (int): 1, 3, 4, 7, or 13. Raises ValueError for unsupported orders.
- Returns `(weights (N,), bary (N, 3))`. Weights sum to 0.5.

### `integrate_over_triangle(func, v0, v1, v2, quad_order=4)` -> complex

Integrate a scalar/complex function `func(r)` over a triangle.

### `integrate_green_singular(k, r_obs, v0, v1, v2, quad_order=4, near_threshold=0.2)` -> complex

Integrate `g(r_obs, r')` over a source triangle with singularity extraction for the scalar potential term.

### `integrate_rho_green_singular(k, r_obs, v0, v1, v2, r_free_vertex, quad_order=4, near_threshold=0.2)` -> `ndarray (3,) complex128`

Integrate `(r' - r_free) * g(r_obs, r')` over a source triangle with singularity extraction for the vector potential term.

---

## `pyMoM3d.fields` — Far-Field and RCS

### `compute_far_field(I_coeffs, rwg_basis, mesh, k, eta, theta, phi, quad_order=4)` -> `(E_theta, E_phi)`

Compute far-field E_theta and E_phi from the radiation integral.

- `theta`, `phi` (`ndarray (M,)`): Observation angles in radians. `theta=0` is +z.
- Returns two `ndarray (M,) complex128` arrays.

### `compute_rcs(E_theta, E_phi, E_inc_mag=1.0)` -> `ndarray (M,)` (dBsm)

Bistatic RCS: `sigma = 4*pi*(|E_theta|^2 + |E_phi|^2) / |E_inc|^2`

### `compute_monostatic_rcs(E_theta, E_phi, E_inc_mag=1.0)` -> float (dBsm)

Single-direction monostatic RCS.

---

## `pyMoM3d.analysis` — Validation and Analysis

### `mie_rcs_pec_sphere(ka, theta, n_max=None)` -> `ndarray (M,)`

Exact Mie series bistatic RCS for a PEC sphere, normalized by `pi*a^2`.

- `ka` (float): Electrical size (k * radius)
- `theta` (`ndarray`): Bistatic angles (radians)

### `mie_monostatic_rcs_pec_sphere(ka, n_max=None)` -> float

Monostatic Mie RCS (normalized by `pi*a^2`).

### `compute_s11(Z_in, Z0=50.0)` -> complex

Reflection coefficient: `(Z_in - Z0) / (Z_in + Z0)`

### `impedance_vs_frequency(results)` -> `(frequencies, Z_in)`

Extract Z_in array from a list of `SimulationResult`.

### `s11_vs_frequency(results, Z0=50.0)` -> `(frequencies, s11_dB)`

Compute `|S11|` in dB from sweep results.

### `mesh_convergence_study(geometry, frequency, subdivisions_list, ...)` -> `[(N_unknowns, rcs_dBsm), ...]`

Run automated mesh convergence study.

### `compute_directivity(E_theta, E_phi, theta, phi, eta)` -> `(D, D_max, D_max_dBi)`

Compute directivity from a full-sphere far-field pattern.

### `compute_beamwidth_3dB(D, theta)` -> float (degrees)

Estimate 3dB beamwidth from a principal-plane cut.

---

## `pyMoM3d.analysis.cma` — Characteristic Mode Analysis

### `CMAResult` (dataclass)

Result container for Characteristic Mode Analysis at a single frequency.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `frequency` | float | Frequency (Hz) at which CMA was computed |
| `eigenvalues` | `ndarray (N,)` | Characteristic eigenvalues λ_n (real) |
| `eigenvectors` | `ndarray (N, N)` | Characteristic currents J_n as columns, power-normalized |
| `modal_significance` | `ndarray (N,)` | Modal significance MS_n = \|1/(1+jλ_n)\| |
| `characteristic_angle` | `ndarray (N,)` | Characteristic angle α_n in degrees |
| `R_matrix` | `ndarray (N, N)` | Radiation resistance matrix Re(Z) |
| `X_matrix` | `ndarray (N, N)` | Reactance matrix Im(Z) |
| `sort_order` | `ndarray (N,)` | Indices that sort modes by decreasing modal significance |

**Methods:**
- `get_mode(n)` -> `ndarray (N,)` — Get the nth most significant characteristic current
- `get_eigenvalue(n)` -> float — Get eigenvalue of nth most significant mode
- `get_modal_significance(n)` -> float — Get modal significance of nth most significant mode
- `get_characteristic_angle(n)` -> float — Get characteristic angle (degrees) of nth most significant mode

### `compute_characteristic_modes(Z, frequency=0.0, num_modes=None, regularization=1e-10)` -> CMAResult

Main entry point for CMA. Computes characteristic modes from an impedance matrix.

- `Z` (`ndarray (N, N)`, complex): Impedance matrix from `fill_impedance_matrix()`
- `frequency` (float): Frequency for metadata
- `num_modes` (int, optional): Number of modes to retain (sorted by significance). If None, returns all N modes.
- `regularization` (float): Regularization factor for R matrix

### `solve_cma(Z, regularization=1e-10, check_conditioning=True)` -> CMAResult

Low-level CMA solver. Decomposes Z = R + jX and solves the generalized eigenvalue problem X·J = λ·R·J.

### `compute_modal_significance(eigenvalues)` -> `ndarray`

Compute modal significance: MS_n = 1 / sqrt(1 + λ_n²)

### `compute_characteristic_angle(eigenvalues)` -> `ndarray`

Compute characteristic angle: α_n = 180° - arctan(λ_n)

### `track_modes_across_frequency(cma_results, correlation_threshold=0.7)` -> list of ndarray

Track characteristic modes across frequency using eigenvector correlation.

- `cma_results` (list of CMAResult): CMA results at multiple frequencies
- `correlation_threshold` (float): Minimum correlation for valid mode match
- Returns list of index arrays for mode tracking

### `verify_orthogonality(cma_result, tolerance=1e-6)` -> `(bool, float)`

Verify R-orthogonality: J_m^H · R · J_n = δ_mn. Returns (is_orthogonal, max_error).

### `verify_eigenvalue_reality(cma_result, tolerance=1e-10)` -> `(bool, float)`

Verify eigenvalues are real. Returns (are_real, max_imaginary_part).

### `compute_modal_excitation_coefficient(cma_result, V, mode_index)` -> complex

Compute modal excitation coefficient α_n = V^T · J_n / (1 + j·λ_n).

### `expand_current_in_modes(cma_result, I, num_modes=None)` -> `(coefficients, reconstruction)`

Expand a driven current in terms of characteristic modes.

- Returns `coefficients` (mode weights) and `reconstruction` (approximated current)

### `cma_frequency_sweep(basis, mesh, frequencies, eta, quad_order=4, near_threshold=0.2, num_modes=None, track_modes=True, progress_callback=None)` -> `(list of CMAResult, tracked_indices or None)`

Perform CMA across a frequency sweep with optional mode tracking.

---

## `pyMoM3d.visualization` — Plotting

### `plot_mesh_3d(mesh, ax=None, show_edges=True, show_normals=False, ...)`  -> Axes3D

3D visualization of triangular mesh using matplotlib Poly3DCollection.

Optional parameters: `normal_scale`, `color`, `alpha`, `edge_color`, `edge_width`.

### `plot_mesh(mesh, projection='xy', ax=None, ...)`  -> Axes

2D projection of mesh. `projection` is `'xy'`, `'xz'`, or `'yz'`.

### `plot_surface_current(I, basis, mesh, ax=None, cmap='hot', log_scale=False, ...)` -> `(ax, ScalarMappable)`

Plot surface current density `|J|` as a 3D heatmap on the mesh.

- `I` (`ndarray (N,) complex128`): RWG coefficients from the solve
- `cmap` (str): Any matplotlib colormap name
- `log_scale` (bool): If True, plot in dB
- `clim` (tuple): `(vmin, vmax)` for color limits
- Returns the axes and a ScalarMappable (for custom colorbar placement)

### `compute_triangle_current_density(I, basis, mesh)` -> `ndarray (N_t,) float64`

Compute `|J|` at each triangle centroid. Useful for exporting current data without plotting.

### `compute_triangle_current_vectors(I, basis, mesh, component='real')` -> `(J_vectors, J_mag, centroids)`

Compute surface current density vectors J(r) at each triangle centroid.

- `I` (`ndarray (N,) complex128`): RWG coefficients from the solve
- `component` (str): `'real'` or `'imag'` — which part of complex current to return
- Returns tuple of:
  - `J_vectors` (`ndarray (N_t, 3) float64`): Current vectors at each centroid
  - `J_mag` (`ndarray (N_t,) float64`): Current magnitudes
  - `centroids` (`ndarray (N_t, 3) float64`): Centroid positions

### `plot_surface_current_vectors(I, basis, mesh, ax=None, ...)` -> `(ax, ScalarMappable or None)`

Plot surface current density as 3D vector arrows on the mesh using matplotlib quiver.

**Parameters:**
- `I` (`ndarray (N,) complex128`): RWG coefficients from the solve
- `component` (str): `'real'` or `'imag'` — which part of complex J to show. Default `'real'`.
- `scale` (float): Arrow length multiplier (auto-scaled to ~5% of mesh size). Default 1.0.
- `normalize` (bool): If True, all arrows have same length (direction only). Default False.
- `subsample` (int): Maximum number of arrows. If None, plots all triangles.
- `subsample_method` (str): `'magnitude'` keeps highest |J|, `'uniform'` picks random. Default `'magnitude'`.
- `color_by_magnitude` (bool): If True, color arrows by |J| using colormap. Default True.
- `cmap` (str): Matplotlib colormap name. Default `'viridis'`.
- `arrow_color` (str): Uniform arrow color (used if `color_by_magnitude=False`). Default `'black'`.
- `arrow_width` (float): Line width for arrows. Default 1.5.
- `show_mesh` (bool): Whether to render underlying mesh surface. Default True.
- `mesh_alpha` (float): Transparency of mesh surface. Default 0.3.
- `mesh_color` (str): Color of mesh surface. Default `'lightgray'`.
- `title` (str): Plot title. Auto-generated if None.
- `clim` (tuple): `(vmin, vmax)` for color limits. Auto-scaled if None.

**Returns:** The axes and a ScalarMappable (for custom colorbar placement), or None if `color_by_magnitude=False`.

---

## `pyMoM3d.simulation` — High-Level Driver

### `SimulationConfig(frequency, excitation, solver_type='direct', quad_order=4, near_threshold=0.2, enable_report=False, report_dir='results/simulation_info')`

Configuration dataclass.

- `enable_report` (bool): If True, generate a text report after the simulation completes. Default False.
- `report_dir` (str): Directory for report output. Default `'results/simulation_info'`.

### `SimulationResult(frequency, I_coefficients, Z_input=None, condition_number=None)`

Result dataclass.

**Methods:**
- `save(path)` — save to `.npz`
- `SimulationResult.load(path)` (classmethod) — load from `.npz`

### `Simulation(config, geometry=None, mesh=None, subdivisions=2, mesher='trimesh', target_edge_length=None, reporter=None)`

Provide either `geometry` (a primitive) or a pre-built `mesh`. The constructor meshes the geometry and computes RWG basis functions once.

- `mesher` (str): `'trimesh'` (default) or `'gmsh'`. Selects the meshing backend.
- `target_edge_length` (float, optional): Target edge length in meters (used with `mesher='gmsh'`).
- `subdivisions` (int): Subdivision level (used with `mesher='trimesh'`).
- `reporter` (object, optional): Progress reporter. Defaults to `TerminalReporter`. Pass `SilentReporter()` to suppress output.

**Methods:**
- `run()` -> SimulationResult — single-frequency solve
- `sweep(frequencies)` -> list of SimulationResult — multi-frequency sweep (mesh reused)
- `compute_cma(frequency=None, num_modes=None)` -> CMAResult — Characteristic Mode Analysis at a single frequency
- `cma_sweep(frequencies, num_modes=None, track_modes=True)` -> `(list of CMAResult, tracked_indices)` — CMA frequency sweep with mode tracking

### `load_stl(path, mesher='trimesh')` -> Mesh

Load a mesh from an `.stl` or `.obj` file. Pass `mesher='gmsh'` to use Gmsh instead of trimesh. With Gmsh, STEP and IGES files can also be loaded via `GmshMesher.mesh_from_file()`.

---

## `pyMoM3d.utils` — Constants and Reporting

### Constants

| Name | Value | Description |
|---|---|---|
| `c0` | 299792458.0 | Speed of light (m/s) |
| `mu0` | 4*pi*1e-7 | Permeability of free space (H/m) |
| `eps0` | 1/(mu0*c0^2) | Permittivity of free space (F/m) |
| `eta0` | mu0*c0 ≈ 376.73 | Intrinsic impedance (Ohms) |

### Progress Reporters (`utils/reporter.py`)

#### `TerminalReporter(stream=sys.stderr)`

Default reporter. Writes human-readable progress to a stream with TTY-based in-place line updates.

**Methods:**
- `stage_start(name, **kwargs)` — announce the start of a stage (mesh, rwg, z_fill, solve, etc.)
- `stage_progress(name, fraction, **kwargs)` — update in-place progress for a stage
- `stage_end(name, **kwargs)` — announce completion with summary stats
- `warning(msg)`, `error(msg)` — print warnings/errors
- `finish()` — print final summary

#### `SilentReporter()`

No-op reporter. All methods are no-ops. Use for tests or batch runs.

#### `RecordingReporter(inner_reporter)`

Wraps another reporter, forwarding all calls while accumulating a `metadata` dict. Used internally by `Simulation` when `enable_report=True`.

- `metadata` (dict): Accumulated data from all stages, used by `write_report()`.

### Report Writer (`utils/report_writer.py`)

#### `write_report(metadata, path)`

Write a structured plain-text simulation report to `path`. Called automatically by `Simulation` when reporting is enabled. The report includes: configuration, mesh statistics, RWG basis info, Z-matrix assembly timing and memory, solver results, condition number, warnings, and errors.
