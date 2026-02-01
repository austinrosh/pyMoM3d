# API Reference

Complete reference for all public modules, classes, and functions.

---

## `pyMoM3d.geometry` — Geometry Primitives

All primitives share a common pattern: construct with physical dimensions, then call `to_trimesh()` to get a triangulated surface.

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

### `PythonMesher(merge_vertices=True, remove_degenerate=True)`

Converts `trimesh.Trimesh` objects into `Mesh` objects with cleaning.

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

### `fill_impedance_matrix(rwg_basis, mesh, k, eta, quad_order=4, near_threshold=0.2)` -> `ndarray (N,N) complex128`

Assemble the EFIE impedance matrix Z.

- `rwg_basis` (RWGBasis)
- `mesh` (Mesh)
- `k` (float): Wavenumber (rad/m). Compute as `2*pi*frequency/c0`.
- `eta` (float): Intrinsic impedance. Use `eta0` (~377 Ohms) for free space.
- `quad_order` (int): Gauss quadrature order. Valid: 1, 3, 4, 7, 13. Default 4.
- `near_threshold` (float): Controls when singularity extraction activates. Default 0.2.

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

### `find_nearest_edge(mesh, rwg_basis, point)` -> int

Find the basis function whose shared edge midpoint is closest to `point`. Useful for locating feed edges.

### `solve_direct(Z, V)` -> `ndarray (N,) complex128`

Solve `ZI = V` via LU factorization (`numpy.linalg.solve`).

### `solve_gmres(Z, V, tol=1e-6, maxiter=1000)` -> `ndarray (N,) complex128`

Solve `ZI = V` via GMRES with diagonal preconditioner (`scipy.sparse.linalg.gmres`).

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

---

## `pyMoM3d.simulation` — High-Level Driver

### `SimulationConfig(frequency, excitation, solver_type='direct', quad_order=4, near_threshold=0.2)`

Configuration dataclass.

### `SimulationResult(frequency, I_coefficients, Z_input=None, condition_number=None)`

Result dataclass.

**Methods:**
- `save(path)` — save to `.npz`
- `SimulationResult.load(path)` (classmethod) — load from `.npz`

### `Simulation(config, geometry=None, mesh=None, subdivisions=2)`

Provide either `geometry` (a primitive with `to_trimesh()`) or a pre-built `mesh`. The constructor meshes the geometry and computes RWG basis functions once.

**Methods:**
- `run()` -> SimulationResult — single-frequency solve
- `sweep(frequencies)` -> list of SimulationResult — multi-frequency sweep (mesh reused)

### `load_stl(path)` -> Mesh

Load a mesh from a `.stl` file via trimesh.

---

## `pyMoM3d.utils` — Constants

| Name | Value | Description |
|---|---|---|
| `c0` | 299792458.0 | Speed of light (m/s) |
| `mu0` | 4*pi*1e-7 | Permeability of free space (H/m) |
| `eps0` | 1/(mu0*c0^2) | Permittivity of free space (F/m) |
| `eta0` | mu0*c0 ≈ 376.73 | Intrinsic impedance (Ohms) |
