"""Performance & scaling analysis for the pyMoM3d solver.

Benchmarks PEC sphere plane-wave scattering across a range of mesh
densities, measuring impedance-matrix fill time, direct/GMRES solve
time, memory usage, and condition number.

Usage
-----
    python examples/solver_performance.py
    python examples/solver_performance.py --plot
"""

import argparse
import csv
import os
import time

import numpy as np

from pyMoM3d import (
    Sphere,
    GmshMesher,
    PlaneWaveExcitation,
    compute_rwg_connectivity,
    c0,
    eta0,
)
from pyMoM3d.mesh.rwg_basis import RWGBasis
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators import EFIEOperator
from pyMoM3d.mom.solver import solve_direct, solve_gmres


# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
RADIUS = 0.1          # m
FREQUENCY = 1.0e9     # Hz
QUAD_ORDER = 4
NEAR_THRESHOLD = 0.2
COND_MAX_N = 1200     # skip expensive SVD cond number above this N

# NOTE: 0.012 gives N≈2200 and may take several minutes for LU.
# Comment it out for a quicker run.
EDGE_LENGTHS = [0.10, 0.065, 0.045, 0.035, 0.025, 0.020, 0.015, 0.012]


def timed_mesh(edge_len, radius):
    """Mesh a sphere and compute RWG connectivity, returning timing."""
    import gmsh

    t0 = time.monotonic()
    mesher = GmshMesher(target_edge_length=edge_len, curvature_adapt=False)
    mesher._init_gmsh()
    gmsh.model.occ.addSphere(0, 0, 0, radius)
    gmsh.model.occ.synchronize()
    mesher._set_mesh_sizes()
    # Set explicit size on geometry points so Gmsh respects target_edge_length
    pts = gmsh.model.getEntities(0)
    gmsh.model.mesh.setSize(pts, edge_len)
    gmsh.model.mesh.generate(2)
    mesh = mesher._extract_surface_mesh()
    mesher._finalize_gmsh()
    mesh.validate()
    basis = compute_rwg_connectivity(mesh)
    elapsed = time.monotonic() - t0
    return mesh, basis, elapsed


def timed_fill(basis, mesh, k, eta):
    """Fill impedance matrix, returning Z and timing."""
    t0 = time.monotonic()
    Z = fill_matrix(
        EFIEOperator(), basis, mesh, k, eta,
        quad_order=QUAD_ORDER,
        near_threshold=NEAR_THRESHOLD,
        progress_callback=None,
    )
    elapsed = time.monotonic() - t0
    return Z, elapsed


def timed_solve_direct(Z, V):
    """Direct solve, returning current vector, timing, and residual."""
    t0 = time.monotonic()
    I = np.linalg.solve(Z, V)
    elapsed = time.monotonic() - t0
    residual = np.linalg.norm(Z @ I - V) / np.linalg.norm(V)
    return I, elapsed, residual


def timed_solve_gmres(Z, V):
    """GMRES solve, returning current vector, timing, residual, and iteration count."""
    from scipy.sparse.linalg import gmres, LinearOperator

    N = len(V)
    diag = np.diag(Z)
    diag_inv = np.where(np.abs(diag) > 1e-30, 1.0 / diag, 1.0)
    M = LinearOperator((N, N), matvec=lambda x: diag_inv * x)

    iter_count = [0]

    def callback(pr_norm):
        iter_count[0] += 1

    t0 = time.monotonic()
    I, info = gmres(Z, V, M=M, rtol=1e-6, maxiter=1000,
                    callback=callback, callback_type='pr_norm')
    elapsed = time.monotonic() - t0

    residual = np.linalg.norm(Z @ I - V) / np.linalg.norm(V)
    return I, elapsed, residual, iter_count[0]


def make_plot(results, root):
    """Generate and save the 3-panel scaling figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pyMoM3d import configure_latex_style
        configure_latex_style()

        Ns = [r['N'] for r in results]
        t_fills = [r['t_fill_s'] for r in results]
        t_directs = [r['t_solve_direct_s'] for r in results]
        t_gmres_list = [r['t_solve_gmres_s'] for r in results]
        Z_MBs = [r['Z_memory_MB'] for r in results]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        # Left: timing vs N
        ax1.loglog(Ns, t_fills, 'o-', label=r'$\mathbf{Z}$ fill')
        ax1.loglog(Ns, t_directs, 's-', label='Direct (LU)')
        ax1.loglog(Ns, t_gmres_list, '^-', label='GMRES')
        N_ref = np.array(Ns, dtype=float)
        ax1.loglog(N_ref, t_fills[-1] * (N_ref / N_ref[-1])**2,
                   '--', color='gray', alpha=0.5, label=r'$\sim N^2$')
        ax1.loglog(N_ref, t_directs[-1] * (N_ref / N_ref[-1])**3,
                   ':', color='gray', alpha=0.5, label=r'$\sim N^3$')
        ax1.set_xlabel(r'$N$ (basis functions)')
        ax1.set_ylabel(r'Time $t$ (s)')
        ax1.set_title(r'Solver Timing')
        ax1.legend()
        ax1.grid(True, which='both', alpha=0.3)

        # Middle: memory vs N
        ax2.loglog(Ns, Z_MBs, 'o-', label=r'$\mathbf{Z}$ matrix')
        ax2.loglog(N_ref, Z_MBs[-1] * (N_ref / N_ref[-1])**2,
                   '--', color='gray', alpha=0.5, label=r'$\sim N^2$')
        ax2.set_xlabel(r'$N$ (basis functions)')
        ax2.set_ylabel(r'Memory (MB)')
        ax2.set_title(r'$\mathbf{Z}$ Matrix Memory')
        ax2.legend()
        ax2.grid(True, which='both', alpha=0.3)

        # Right: GMRES iterations vs N
        gmres_iters_list = [r['gmres_iters'] for r in results]
        ax3.plot(Ns, gmres_iters_list, 'D-', color='C3', label=r'GMRES iterations')
        ax3.set_xlabel(r'$N$ (basis functions)')
        ax3.set_ylabel(r'Iterations')
        ax3.set_title(r'GMRES Convergence')
        ax3.legend()
        ax3.grid(True, which='both', alpha=0.3)

        fig.tight_layout()
        images_dir = os.path.join(root, "images")
        os.makedirs(images_dir, exist_ok=True)
        plot_path = os.path.join(images_dir, "solver_performance.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)

    except ImportError:
        print("matplotlib not available — skipping plot")


def load_results_from_csv(csv_path):
    """Load benchmark results from a previously written CSV."""
    results = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            results.append({
                'N': int(row['N']),
                't_fill_s': float(row['t_fill_s']),
                't_solve_direct_s': float(row['t_solve_direct_s']),
                't_solve_gmres_s': float(row['t_solve_gmres_s']),
                'gmres_iters': int(row['gmres_iters']),
                'Z_memory_MB': float(row['Z_memory_MB']),
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="pyMoM3d solver performance analysis")
    parser.add_argument("--plot", action="store_true", help="Generate scaling plot after simulation")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation; plot from existing results/solver_performance.csv")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root, "results", "solver_performance.csv")

    if args.plot_only:
        if not os.path.exists(csv_path):
            print(f"No CSV found at {csv_path}. Run without --plot-only first.")
            return
        results = load_results_from_csv(csv_path)
        make_plot(results, root)
        return

    k = 2 * np.pi * FREQUENCY / c0
    eta = eta0
    exc = PlaneWaveExcitation(E0=[1, 0, 0], k_hat=[0, 0, -1])

    # Header
    print(f"PEC Sphere Performance Analysis (r={RADIUS}m, f={FREQUENCY/1e9:.1f}GHz, quad={QUAD_ORDER})")
    print("=" * 100)
    print(f"{'Mesh':>4}  {'edge_len':>8}  {'tri':>5}  {'N':>5}  {'t_mesh':>7}  {'t_fill':>8}  "
          f"{'t_LU':>8}  {'t_GMRES':>8}  {'GMRES_it':>8}  {'Z_MB':>7}  {'cond':>10}")
    print("-" * 100)

    results = []

    for idx, edge_len in enumerate(EDGE_LENGTHS, 1):
        # Mesh
        mesh, basis, t_mesh = timed_mesh(edge_len, RADIUS)
        N = basis.num_basis
        num_tri = mesh.get_num_triangles()
        stats = mesh.get_statistics()
        mean_edge = stats.get('mean_edge_length', edge_len)

        # Fill
        Z, t_fill = timed_fill(basis, mesh, k, eta)
        Z_MB = Z.nbytes / 1e6
        fill_rate = N * (N + 1) / 2 / t_fill if t_fill > 0 else 0

        # Excitation
        V = exc.compute_voltage_vector(basis, mesh, k)

        # Solve direct
        I_direct, t_direct, res_direct = timed_solve_direct(Z, V)

        # Solve GMRES
        I_gmres, t_gmres, res_gmres, gmres_iters = timed_solve_gmres(Z, V)

        # Condition number (skip expensive SVD for large N)
        cond = np.linalg.cond(Z) if N <= COND_MAX_N else float('nan')

        row = {
            'target_edge_length_m': edge_len,
            'num_triangles': num_tri,
            'N': N,
            'mean_edge_length': mean_edge,
            't_mesh_s': t_mesh,
            't_fill_s': t_fill,
            't_solve_direct_s': t_direct,
            't_solve_gmres_s': t_gmres,
            'gmres_iters': gmres_iters,
            'residual_direct': res_direct,
            'residual_gmres': res_gmres,
            'cond': cond,
            'Z_memory_MB': Z_MB,
            'fill_rate': fill_rate,
        }
        results.append(row)

        print(f"{idx:>4}  {edge_len:>8.4f}  {num_tri:>5}  {N:>5}  {t_mesh:>6.1f}s  {t_fill:>7.1f}s  "
              f"{t_direct:>7.2f}s  {t_gmres:>7.2f}s  {gmres_iters:>8}  {Z_MB:>6.2f}  {cond:>10.2e}")

    # Write CSV
    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    fieldnames = [
        'target_edge_length_m', 'num_triangles', 'N', 'mean_edge_length',
        't_mesh_s', 't_fill_s', 't_solve_direct_s', 't_solve_gmres_s',
        'gmres_iters', 'residual_direct', 'residual_gmres', 'cond',
        'Z_memory_MB', 'fill_rate',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV written to {csv_path}")

    if args.plot:
        make_plot(results, root)


if __name__ == '__main__':
    main()
