"""Backend fill-time comparison: numpy (pre-speedup) vs numba vs cpp.

Measures impedance matrix fill time for all three backends across a range
of mesh densities.  Because the numpy backend is O(N^2 * Q^2) pure Python
it is only run up to a configurable N limit; larger sizes are extrapolated
from an N^2 fit.

Usage
-----
    python examples/backend_comparison.py
    python examples/backend_comparison.py --plot
    python examples/backend_comparison.py --numpy-max-N 300
"""

import argparse
import os
import time

import numpy as np

from pyMoM3d import Sphere, GmshMesher, compute_rwg_connectivity, c0, eta0
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators import EFIEOperator

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RADIUS = 0.1        # m
FREQUENCY = 1.0e9   # Hz
QUAD_ORDER = 4
NEAR_THRESHOLD = 0.2

# Edge lengths to test — covers N from ~78 to ~3400
EDGE_LENGTHS = [0.10, 0.065, 0.045, 0.035, 0.025, 0.020, 0.015, 0.012]

# numpy is extremely slow — skip it above this N
DEFAULT_NUMPY_MAX_N = 400


def build_mesh(edge_len):
    import gmsh
    mesher = GmshMesher(target_edge_length=edge_len, curvature_adapt=False)
    mesher._init_gmsh()
    gmsh.model.occ.addSphere(0, 0, 0, RADIUS)
    gmsh.model.occ.synchronize()
    mesher._set_mesh_sizes()
    pts = gmsh.model.getEntities(0)
    gmsh.model.mesh.setSize(pts, edge_len)
    gmsh.model.mesh.generate(2)
    mesh = mesher._extract_surface_mesh()
    mesher._finalize_gmsh()
    mesh.validate()
    basis = compute_rwg_connectivity(mesh)
    return mesh, basis


def time_fill(basis, mesh, k, eta, backend):
    t0 = time.perf_counter()
    fill_matrix(EFIEOperator(), basis, mesh, k, eta,
                quad_order=QUAD_ORDER,
                near_threshold=NEAR_THRESHOLD,
                backend=backend)
    return time.perf_counter() - t0


def fit_n2(Ns, ts):
    """Fit t = a * N^2 through measured (N, t) points; return coefficient a."""
    Ns = np.array(Ns, dtype=float)
    ts = np.array(ts, dtype=float)
    return np.sum(ts * Ns**2) / np.sum(Ns**4)


def main():
    parser = argparse.ArgumentParser(description="Backend fill-time comparison")
    parser.add_argument("--plot", action="store_true", help="Save comparison plot")
    parser.add_argument("--numpy-max-N", type=int, default=DEFAULT_NUMPY_MAX_N,
                        help=f"Skip numpy above this N (default {DEFAULT_NUMPY_MAX_N})")
    args = parser.parse_args()

    k = 2 * np.pi * FREQUENCY / c0
    eta = eta0

    rows = []

    print(f"\nBackend Fill-Time Comparison  (r={RADIUS} m, f={FREQUENCY/1e9:.1f} GHz, quad={QUAD_ORDER})")
    print("=" * 90)
    print(f"{'edge_len':>8}  {'N':>5}  {'numpy (s)':>12}  {'numba (s)':>12}  {'cpp (s)':>10}"
          f"  {'numba/np':>9}  {'cpp/np':>8}")
    print("-" * 90)

    numpy_measured_Ns = []
    numpy_measured_ts = []

    for edge_len in EDGE_LENGTHS:
        mesh, basis = build_mesh(edge_len)
        N = basis.num_basis

        # --- C++ (always run) ---
        t_cpp = time_fill(basis, mesh, k, eta, 'cpp')

        # --- Numba (always run; warm up JIT on first call) ---
        t_numba = time_fill(basis, mesh, k, eta, 'numba')

        # --- NumPy (only for small N) ---
        if N <= args.numpy_max_N:
            t_numpy = time_fill(basis, mesh, k, eta, 'numpy')
            numpy_measured_Ns.append(N)
            numpy_measured_ts.append(t_numpy)
            numpy_str = f"{t_numpy:>12.3f}"
            speedup_numba = t_numpy / t_numba if t_numba > 0 else float('nan')
            speedup_cpp   = t_numpy / t_cpp   if t_cpp   > 0 else float('nan')
            su_numba_str = f"{speedup_numba:>9.0f}x"
            su_cpp_str   = f"{speedup_cpp:>8.0f}x"
        else:
            t_numpy = None
            numpy_str = f"{'(extrap)':>12}"
            speedup_numba = float('nan')
            speedup_cpp   = float('nan')
            su_numba_str  = f"{'---':>9}"
            su_cpp_str    = f"{'---':>8}"

        rows.append(dict(edge_len=edge_len, N=N,
                         t_numpy=t_numpy, t_numba=t_numba, t_cpp=t_cpp,
                         speedup_numba=speedup_numba, speedup_cpp=speedup_cpp))

        print(f"{edge_len:>8.4f}  {N:>5}  {numpy_str}  {t_numba:>12.3f}  {t_cpp:>10.4f}"
              f"  {su_numba_str}  {su_cpp_str}")

    # --- Extrapolate numpy for large N using N^2 fit ---
    if numpy_measured_Ns:
        a = fit_n2(numpy_measured_Ns, numpy_measured_ts)
        print()
        print("NumPy N^2 extrapolation (fitted on measured points):")
        print(f"  t_numpy ≈ {a:.3e} * N^2")
        print()
        for row in rows:
            if row['t_numpy'] is None:
                N = row['N']
                row['t_numpy_extrap'] = a * N**2
                su_numba = row['t_numpy_extrap'] / row['t_numba']
                su_cpp   = row['t_numpy_extrap'] / row['t_cpp']
                print(f"  N={N:5d}: numpy≈{row['t_numpy_extrap']:8.1f}s  "
                      f"numba≈{row['t_numba']:.3f}s ({su_numba:.0f}x)  "
                      f"cpp≈{row['t_cpp']:.4f}s ({su_cpp:.0f}x)")

    # --- Summary: speedup at measured points ---
    measured = [r for r in rows if r['t_numpy'] is not None]
    if measured:
        print()
        print("Average speedup at measured N values:")
        print(f"  numba vs numpy : {np.mean([r['speedup_numba'] for r in measured]):.0f}x")
        print(f"  cpp   vs numpy : {np.mean([r['speedup_cpp']   for r in measured]):.0f}x")
        print(f"  cpp   vs numba : {np.mean([r['t_numba']/r['t_cpp'] for r in measured]):.0f}x")

    # --- Optional plot ---
    if args.plot:
        _make_plot(rows, numpy_measured_Ns, a if numpy_measured_Ns else None)


def _make_plot(rows, numpy_measured_Ns, numpy_a):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyMoM3d import configure_latex_style
        configure_latex_style()
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    Ns       = [r['N'] for r in rows]
    t_numba  = [r['t_numba'] for r in rows]
    t_cpp    = [r['t_cpp']   for r in rows]
    t_numpy_meas = [(r['N'], r['t_numpy']) for r in rows if r['t_numpy'] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute fill times
    if t_numpy_meas:
        ax1.loglog([x[0] for x in t_numpy_meas], [x[1] for x in t_numpy_meas],
                   'o--', color='C0', label='NumPy (measured)')
    if numpy_a is not None:
        N_extrap = np.array([r['N'] for r in rows if r['t_numpy'] is None], dtype=float)
        if len(N_extrap):
            ax1.loglog(N_extrap, numpy_a * N_extrap**2,
                       'o:', color='C0', alpha=0.5, label=r'NumPy ($N^2$ extrap.)')

    ax1.loglog(Ns, t_numba, 's-', color='C1', label='Numba')
    ax1.loglog(Ns, t_cpp,   '^-', color='C2', label='C++ (OpenMP)')

    N_arr = np.array(Ns, dtype=float)
    ax1.loglog(N_arr, t_cpp[-1] * (N_arr / N_arr[-1])**2,
               '--', color='gray', alpha=0.4, label=r'$\sim N^2$')

    ax1.set_xlabel(r'$N$ (basis functions)')
    ax1.set_ylabel(r'Fill time $t$ (s)')
    ax1.set_title(r'Impedance Matrix Fill Time')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    # Right: speedup vs numpy
    measured = [(r['N'], r['speedup_numba'], r['speedup_cpp'])
                for r in rows if r['t_numpy'] is not None]
    if measured:
        Nm = [x[0] for x in measured]
        su_numba = [x[1] for x in measured]
        su_cpp   = [x[2] for x in measured]
        ax2.semilogy(Nm, su_numba, 's-', color='C1', label='Numba / NumPy')
        ax2.semilogy(Nm, su_cpp,   '^-', color='C2', label='C++ / NumPy')
        ax2.set_xlabel(r'$N$ (basis functions)')
        ax2.set_ylabel(r'Speedup (×)')
        ax2.set_title(r'Speedup vs.\ NumPy Baseline')
        ax2.legend()
        ax2.grid(True, which='both', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No numpy measurements\n(increase --numpy-max-N)',
                 ha='center', va='center', transform=ax2.transAxes)

    fig.tight_layout()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(root, "images")
    os.makedirs(images_dir, exist_ok=True)
    path = os.path.join(images_dir, "backend_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"\nPlot saved to {path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
