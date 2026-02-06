"""
Example: Interactive STL/OBJ Upload → Bistatic RCS + Surface Current

Loads an arbitrary mesh file (.stl or .obj), analyzes mesh quality,
optionally remeshes, then runs a plane-wave MoM solve to compute
bistatic RCS and induced surface current density.

Usage:
    python examples/stl_rcs_example.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from pyMoM3d import (
    GmshMesher,
    PythonMesher,
    compute_rwg_connectivity,
    compute_far_field,
    compute_rcs,
    plot_mesh_3d,
    plot_surface_current,
    PlaneWaveExcitation,
    Simulation,
    SimulationConfig,
    configure_latex_style,
    eta0,
    c0,
)

# Configure LaTeX-style plotting
configure_latex_style()


def load_mesh_from_file(path):
    """Load a mesh file via trimesh, preserving original triangulation.

    Parameters
    ----------
    path : str
        Path to .stl or .obj file.

    Returns
    -------
    mesh : Mesh
    """
    from pyMoM3d.simulation import _load_trimesh
    trimesh_obj = _load_trimesh(path)
    mesher = PythonMesher()
    return mesher.mesh_from_geometry(trimesh_obj)


def assess_mesh_quality(stats, validation, wavelength, required_epw=10):
    """Assess mesh quality and decide on remeshing.

    Parameters
    ----------
    stats : dict
        From mesh.get_statistics().
    validation : dict
        From mesh.validate().
    wavelength : float
        Wavelength in meters.
    required_epw : float
        Minimum required elements per wavelength. Meshes below this
        threshold must be remeshed.

    Returns
    -------
    must_remesh : bool
    recommend_remesh : bool
    reasons : list of str
    """
    must_remesh = False
    recommend_remesh = False
    reasons = []

    mean_edge = stats['mean_edge_length']
    elements_per_lambda = wavelength / mean_edge

    # Check elements per wavelength
    if elements_per_lambda < required_epw:
        must_remesh = True
        reasons.append(
            f"Elements per wavelength: {elements_per_lambda:.1f} "
            f"(< {required_epw} required)"
        )
    else:
        reasons.append(
            f"Elements per wavelength: {elements_per_lambda:.1f} "
            f"(>= {required_epw} OK)"
        )

    # Check edge length ratio
    edge_ratio = stats['max_edge_length'] / stats['min_edge_length']
    if edge_ratio > 5.0:
        recommend_remesh = True
        reasons.append(
            f"Edge ratio {edge_ratio:.1f} > 5.0 — remeshing recommended for uniform elements"
        )
    else:
        reasons.append(f"Edge ratio {edge_ratio:.1f} (<= 5.0 OK)")

    # Check degenerate triangles
    if validation.get('has_degenerate_triangles', False):
        must_remesh = True
        n_degen = validation.get('num_degenerate_triangles', 0)
        reasons.append(f"{n_degen} degenerate triangle(s) found — must remesh")

    # Check non-manifold edges (warn only, can't fix by remeshing)
    if validation.get('has_non_manifold_edges', False):
        n_nm = validation.get('num_non_manifold_edges', 0)
        reasons.append(
            f"WARNING: {n_nm} non-manifold edge(s) — cannot fix by remeshing"
        )

    return must_remesh, recommend_remesh, reasons


def remesh_file(path, target_edge_length):
    """Remesh a file via GmshMesher.

    Parameters
    ----------
    path : str
        Path to .stl or .obj file.
    target_edge_length : float
        Target edge length in meters.

    Returns
    -------
    mesh : Mesh
    """
    mesher = GmshMesher(target_edge_length=target_edge_length)
    return mesher.mesh_from_file(path)


def _format_vec(v):
    """Format a 3-vector as a compact string like '(0,0,-1)'."""
    parts = []
    for x in v:
        if x == int(x):
            parts.append(str(int(x)))
        else:
            parts.append(f'{x:.2g}')
    return '(' + ','.join(parts) + ')'


def _exc_subtitle(exc):
    """Return a subtitle string describing the excitation, or ''."""
    if isinstance(exc, PlaneWaveExcitation):
        pol = _format_vec(exc.E0)
        prop = _format_vec(exc.k_hat)
        return rf'Plane wave: $\mathbf{{E}}_0 = {pol}$, $\hat{{\mathbf{{k}}}} = {prop}$'
    return ''


def plot_bistatic_rcs(theta_deg, rcs_dBsm, freq, filename, num_basis,
                      output_path, exc=None):
    """Create and save a polar plot of bistatic RCS.

    Parameters
    ----------
    theta_deg : ndarray
        Observation angles in degrees (0 to 180).
    rcs_dBsm : ndarray
        RCS values in dBsm.
    freq : float
        Frequency in Hz.
    filename : str
        Name of the input file.
    num_basis : int
        Number of RWG basis functions.
    output_path : str
        Path to save the figure.
    exc : Excitation, optional
        Excitation object; if PlaneWaveExcitation, its info is shown.
    """
    # Mirror for full polar plot: 0->360
    theta_rad_full = np.concatenate([
        np.radians(theta_deg),
        np.radians(360.0 - theta_deg[::-1]),
    ])
    rcs_full = np.concatenate([rcs_dBsm, rcs_dBsm[::-1]])

    title_lines = [
        rf'Bistatic RCS $\sigma$ — {os.path.basename(filename)}',
        rf'$f = {freq/1e9:.3f}$ GHz, $N = {num_basis}$',
    ]
    sub = _exc_subtitle(exc) if exc is not None else ''
    if sub:
        title_lines.append(sub)

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.plot(theta_rad_full, rcs_full, 'b-', linewidth=1.5)
    ax.set_title('\n'.join(title_lines), pad=20)
    ax.set_xlabel(r'RCS $\sigma$ (dBsm)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saving plots to {output_path}")


def plot_current(I_coeffs, basis, mesh, freq, filename, output_path,
                  log_scale=False, exc=None):
    """Create and save a 3D surface current plot.

    Parameters
    ----------
    I_coeffs : ndarray
        Current expansion coefficients.
    basis : RWGBasis
        RWG basis object.
    mesh : Mesh
        Mesh object.
    freq : float
        Frequency in Hz.
    filename : str
        Input file name.
    output_path : str
        Path to save the figure.
    log_scale : bool
        If True, plot |J| in dB scale.
    exc : Excitation, optional
        Excitation object; if PlaneWaveExcitation, its info is shown.
    """
    scale_label = "dB" if log_scale else "linear"
    title_lines = [
        rf'Induced Surface Current $|\mathbf{{J}}|$ ({scale_label}) — '
        f'{os.path.basename(filename)}',
        rf'$f = {freq/1e9:.3f}$ GHz, $N = {basis.num_basis}$',
    ]
    sub = _exc_subtitle(exc) if exc is not None else ''
    if sub:
        title_lines.append(sub)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_surface_current(
        I_coeffs, basis, mesh, ax=ax, cmap='hot',
        edge_color='gray', edge_width=0.3,
        log_scale=log_scale,
        title='\n'.join(title_lines),
    )
    ax.view_init(elev=30, azim=-60)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saving plots to {output_path}")


def select_mesh_file():
    """Open a native file dialog to select an STL or OBJ file.

    Returns
    -------
    path : str or None
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()          # hide the root window
    root.attributes('-topmost', True)  # dialog on top
    path = filedialog.askopenfilename(
        title="Select mesh file",
        filetypes=[
            ("Mesh files", "*.stl *.obj"),
            ("STL files", "*.stl"),
            ("OBJ files", "*.obj"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path if path else None


def main():
    # --- User input ---
    print("\nOpening file dialog...")
    path = select_mesh_file()
    if not path:
        print("No file selected. Exiting.")
        return

    ext = os.path.splitext(path)[1].lower()
    if ext not in ('.stl', '.obj'):
        print(f"Unsupported file format '{ext}'. Only .stl and .obj are supported.")
        return

    freq_str = input("Enter frequency in GHz [default: 1.0]: ").strip()
    freq_ghz = float(freq_str) if freq_str else 1.0
    frequency = freq_ghz * 1e9
    wavelength = c0 / frequency

    # --- Mesh resolution ---
    # Controls the target elements-per-wavelength used for remeshing.
    #   coarse  : lambda/8   (~8 elements/lambda)  — fast, lower accuracy
    #   medium  : lambda/15  (~15 elements/lambda)  — balanced (default)
    #   fine    : lambda/25  (~25 elements/lambda)  — slow, higher accuracy
    RESOLUTION_MAP = {
        'coarse': (8,  'fast, lower accuracy'),
        'medium': (15, 'balanced'),
        'fine':   (25, 'slow, higher accuracy'),
    }
    print("\nMesh resolution:")
    for key, (epw, desc) in RESOLUTION_MAP.items():
        tag = " (default)" if key == 'medium' else ""
        print(f"  {key:8s} — ~{epw} elements/lambda, {desc}{tag}")
    res_str = input("Select resolution [coarse/medium/fine, default: medium]: ").strip().lower()
    if res_str not in RESOLUTION_MAP:
        res_str = 'medium'
    elements_per_wave, _ = RESOLUTION_MAP[res_str]
    print(f"  Using '{res_str}' resolution ({elements_per_wave} elements/lambda)")

    # --- Load mesh ---
    print(f"\nLoading: {path}")
    try:
        mesh = load_mesh_from_file(path)
    except Exception as e:
        print(f"Failed to load mesh: {e}")
        return

    stats = mesh.get_statistics()
    validation = mesh.validate()

    mean_edge = stats['mean_edge_length']
    min_edge = stats['min_edge_length']
    max_edge = stats['max_edge_length']
    edge_ratio = max_edge / min_edge if min_edge > 0 else float('inf')
    elements_per_lambda = wavelength / mean_edge

    print(f"  Triangles: {stats['num_triangles']}   "
          f"Vertices: {stats['num_vertices']}   "
          f"Edges: {stats['num_edges']}")
    print(f"  Mean edge: {mean_edge:.4f} m   "
          f"Min: {min_edge:.4f} m   Max: {max_edge:.4f} m")
    print(f"  Edge length ratio (max/min): {edge_ratio:.1f}")
    print(f"  lambda at {freq_ghz:.3f} GHz = {wavelength:.3f} m "
          f"-> elements/lambda = {elements_per_lambda:.1f}")

    # --- Display imported mesh ---
    print("\nDisplaying imported mesh...")
    fig_import = plt.figure(figsize=(10, 8))
    ax_import = fig_import.add_subplot(111, projection='3d')
    plot_mesh_3d(mesh, show_edges=True, ax=ax_import)
    ax_import.set_title(
        f"Imported mesh: {os.path.basename(path)} "
        f"({stats['num_triangles']} triangles, "
        f"{stats['num_vertices']} vertices)"
    )
    print("Close the plot window to continue.")
    plt.show(block=True)

    # --- Assess mesh quality ---
    print("\nMesh quality assessment:")
    must_remesh, recommend_remesh, reasons = assess_mesh_quality(
        stats, validation, wavelength, required_epw=elements_per_wave
    )
    for r in reasons:
        marker = "!" if "must" in r.lower() or "WARNING" in r else \
                 "~" if "recommend" in r.lower() else "+"
        print(f"  {marker} {r}")

    # Decide on remeshing
    do_remesh = False
    target_el = min(wavelength / elements_per_wave, mean_edge)

    if must_remesh:
        print(f"\nRemeshing required (target_edge_length={target_el:.4f} m)...")
        do_remesh = True
    elif recommend_remesh:
        ans = input(
            f"\n  Remesh with target edge length = {target_el:.4f} m? [Y/n]: "
        ).strip().lower()
        do_remesh = ans != 'n'

    if do_remesh:
        print(f"\nRemeshing with Gmsh (target_edge_length={target_el:.4f} m)...")
        try:
            mesh = remesh_file(path, target_el)
        except Exception as e:
            print(f"Remeshing failed: {e}")
            print("Continuing with original mesh.")
            do_remesh = False

        if do_remesh:
            stats = mesh.get_statistics()
            print(f"  Done: {stats['num_triangles']} triangles, "
                  f"{stats['num_vertices']} vertices")

            # Display remeshed model
            print("\nDisplaying remeshed model...")
            fig_remesh = plt.figure(figsize=(10, 8))
            ax_remesh = fig_remesh.add_subplot(111, projection='3d')
            plot_mesh_3d(mesh, show_edges=True, ax=ax_remesh)
            ax_remesh.set_title(
                f"Remeshed: {os.path.basename(path)} "
                f"({stats['num_triangles']} triangles, "
                f"target edge = {target_el:.4f} m)"
            )
            print("Close the plot window to continue.")
            plt.show(block=True)

    # --- Solver selection ---
    # Estimate basis function count (~1.5x triangle count for closed surfaces)
    estimated_basis = int(stats['num_triangles'] * 1.5)
    LARGE_THRESHOLD = 2000

    if estimated_basis > LARGE_THRESHOLD:
        print(f"\nNote: ~{estimated_basis} basis functions estimated.")
        print(f"  Direct (LU) solves an NxN dense system — memory ~ N^2, time ~ N^3.")
        print(f"  GMRES (iterative) is faster for large N but may not converge.")
        print(f"  Recommended: gmres (N > {LARGE_THRESHOLD}).")
        solver_ans = input("  Solver [direct / gmres, default: gmres]: ").strip().lower()
        solver_type = 'gmres' if solver_ans != 'direct' else 'direct'
    else:
        solver_type = 'direct'

    exc = PlaneWaveExcitation(
        E0=np.array([0.0, 0.0, 1.0]),      # z-polarized
        k_hat=np.array([0.0, 1, 0.0]),   # propagating in +y
    )

    config = SimulationConfig(
        frequency=frequency,
        excitation=exc,
        solver_type=solver_type,
        enable_report=True,
    )

    print()
    sim = Simulation(config, mesh=mesh)
    result = sim.run()

    basis = sim.basis
    I = result.I_coefficients
    k = 2.0 * np.pi * frequency / c0

    # --- Far-field sweep ---
    theta = np.linspace(0.001, np.pi - 0.001, 361)
    phi = np.zeros_like(theta)
    E_th, E_ph = compute_far_field(I, basis, mesh, k, eta0, theta, phi)
    rcs_dBsm = compute_rcs(E_th, E_ph, E_inc_mag=1.0)
    theta_deg = np.degrees(theta)

    # --- Save plots ---
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(path))[0]
    plot_bistatic_rcs(
        theta_deg, rcs_dBsm, frequency, path, basis.num_basis,
        os.path.join(images_dir, f'{basename}_rcs_bistatic.png'),
    )
    plot_current(
        I, basis, mesh, frequency, path,
        os.path.join(images_dir, f'{basename}_surface_current.png'),
    )
    plot_current(
        I, basis, mesh, frequency, path,
        os.path.join(images_dir, f'{basename}_surface_current_dB.png'),
        log_scale=True,
    )

    plt.show()

    print("\nDone!")


if __name__ == '__main__':
    main()
