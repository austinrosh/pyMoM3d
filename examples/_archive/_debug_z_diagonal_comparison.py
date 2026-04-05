"""Compare Z-matrix diagonal entries: free-space EFIE vs multilayer MPIE.

The free-space EFIE is validated (sphere RCS, dipole impedance pass).
If the multilayer Z diagonal is similar to free-space, the MPIE assembly
is producing correctly-scaled values and the TL issue is physics-related.
If they're very different, there's a normalization problem.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Layer, LayerStack, c0, eta0,
)
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators.efie import EFIEOperator
from pyMoM3d.greens.layered import LayeredGreensFunction
from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 10e-3  # short strip
TEL = 1.0e-3

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_plate(width=L, height=W, center=(0.0, 0.0, H))
basis = compute_rwg_connectivity(mesh)
N = basis.num_basis
print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {N} RWG")

freq = 2e9
k = 2*np.pi*freq / c0

# --- Free-space EFIE (NumPy) ---
print("\n--- Free-space EFIE (NumPy backend) ---")
op_fs = EFIEOperator()
Z_fs = fill_matrix(op_fs, basis, mesh, k, eta0, quad_order=4, backend='numpy')
diag_fs = np.diag(Z_fs)
print(f"|Z_diag|: [{np.abs(diag_fs).min():.4e}, {np.abs(diag_fs).max():.4e}]")
print(f"Re(Z_diag): [{diag_fs.real.min():.4e}, {diag_fs.real.max():.4e}]")
print(f"Im(Z_diag): [{diag_fs.imag.min():.4e}, {diag_fs.imag.max():.4e}]")

# --- Free-space EFIE (C++) ---
print("\n--- Free-space EFIE (C++ backend) ---")
Z_fs_cpp = fill_matrix(op_fs, basis, mesh, k, eta0, quad_order=4, backend='cpp')
diag_fs_cpp = np.diag(Z_fs_cpp)
print(f"|Z_diag|: [{np.abs(diag_fs_cpp).min():.4e}, {np.abs(diag_fs_cpp).max():.4e}]")
diff_fs = np.linalg.norm(Z_fs - Z_fs_cpp) / np.linalg.norm(Z_fs)
print(f"NumPy vs C++ relative error: {diff_fs:.2e}")

# --- Multilayer MPIE (Strata C++) ---
print("\n--- Multilayer MPIE (Strata C++ backend) ---")
gf_st = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='strata')
op_ml = MultilayerEFIEOperator(gf_st)
Z_ml_cpp = fill_matrix(op_ml, basis, mesh, k, eta0, quad_order=4, backend='cpp')
diag_ml_cpp = np.diag(Z_ml_cpp)
print(f"|Z_diag|: [{np.abs(diag_ml_cpp).min():.4e}, {np.abs(diag_ml_cpp).max():.4e}]")
print(f"Re(Z_diag): [{diag_ml_cpp.real.min():.4e}, {diag_ml_cpp.real.max():.4e}]")
print(f"Im(Z_diag): [{diag_ml_cpp.imag.min():.4e}, {diag_ml_cpp.imag.max():.4e}]")

# --- Multilayer MPIE (NumPy) ---
print("\n--- Multilayer MPIE (NumPy backend) ---")
Z_ml_np = fill_matrix(op_ml, basis, mesh, k, eta0, quad_order=4, backend='numpy')
diag_ml_np = np.diag(Z_ml_np)
print(f"|Z_diag|: [{np.abs(diag_ml_np).min():.4e}, {np.abs(diag_ml_np).max():.4e}]")
print(f"Re(Z_diag): [{diag_ml_np.real.min():.4e}, {diag_ml_np.real.max():.4e}]")
print(f"Im(Z_diag): [{diag_ml_np.imag.min():.4e}, {diag_ml_np.imag.max():.4e}]")

# --- Multilayer MPIE (layer recursion, NumPy) ---
print("\n--- Multilayer MPIE (layer recursion, NumPy) ---")
gf_lr = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='layer_recursion')
op_lr = MultilayerEFIEOperator(gf_lr)
Z_ml_lr = fill_matrix(op_lr, basis, mesh, k, eta0, quad_order=4, backend='numpy')
diag_ml_lr = np.diag(Z_ml_lr)
print(f"|Z_diag|: [{np.abs(diag_ml_lr).min():.4e}, {np.abs(diag_ml_lr).max():.4e}]")
print(f"Re(Z_diag): [{diag_ml_lr.real.min():.4e}, {diag_ml_lr.real.max():.4e}]")
print(f"Im(Z_diag): [{diag_ml_lr.imag.min():.4e}, {diag_ml_lr.imag.max():.4e}]")

# --- Ratios ---
print("\n--- Diagonal ratios ---")
ratio_ml_fs = np.abs(diag_ml_cpp) / np.abs(diag_fs_cpp)
print(f"|Z_ml_strata / Z_fs|: [{ratio_ml_fs.min():.4f}, {ratio_ml_fs.max():.4f}], median={np.median(ratio_ml_fs):.4f}")

ratio_lr_fs = np.abs(diag_ml_lr) / np.abs(diag_fs_cpp)
print(f"|Z_ml_lr / Z_fs|:     [{ratio_lr_fs.min():.4f}, {ratio_lr_fs.max():.4f}], median={np.median(ratio_lr_fs):.4f}")

# C++ vs NumPy for multilayer
diff_ml = np.linalg.norm(Z_ml_cpp - Z_ml_np) / np.linalg.norm(Z_ml_np)
print(f"\nMultilayer C++ vs NumPy relative error: {diff_ml:.2e}")

# Layer recursion vs strata (both NumPy)
diff_st_lr = np.linalg.norm(Z_ml_np - Z_ml_lr) / np.linalg.norm(Z_ml_lr)
print(f"Strata vs Layer Recursion relative error: {diff_st_lr:.2e}")

# Print a few diagonal entries
print(f"\n--- Sample diagonal entries ---")
print(f"{'idx':>5} {'Z_fs':>20} {'Z_ml_strata':>20} {'Z_ml_lr':>20} {'st/fs':>8} {'lr/fs':>8}")
for n in [0, N//4, N//2, 3*N//4, N-1]:
    print(f"  {n:>3} {diag_fs_cpp[n].real:>9.4e}+j{diag_fs_cpp[n].imag:>9.4e} "
          f"{diag_ml_cpp[n].real:>9.4e}+j{diag_ml_cpp[n].imag:>9.4e} "
          f"{diag_ml_lr[n].real:>9.4e}+j{diag_ml_lr[n].imag:>9.4e} "
          f"{ratio_ml_fs[n]:>7.4f} {ratio_lr_fs[n]:>7.4f}")
