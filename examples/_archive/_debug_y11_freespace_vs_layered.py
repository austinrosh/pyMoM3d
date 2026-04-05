"""Compare Y11 from free-space EFIE vs layered MPIE for same strip geometry.

If free-space Y11 resonances match analytical TL model but layered doesn't,
the problem is in the layered Green's function / MPIE assembly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor,
    Layer, LayerStack, c0, eta0,
    microstrip_z0_hammerstad,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 30e-3
TEL = 1.5e-3  # slightly coarser for speed

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad (layered): Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

# --- Build mesh (shared for both) ---
mesher = GmshMesher(target_edge_length=TEL)
margin = TEL / 2.0
port_x = -L/2 + margin

mesh = mesher.mesh_plate_with_feeds(
    width=L, height=W, feed_x_list=[port_x],
    center=(0.0, 0.0, H),
)
basis = compute_rwg_connectivity(mesh)
feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
signs = compute_feed_signs(mesh, basis, feed_edges)
port = Port(name='P1', feed_basis_indices=feed_edges, feed_signs=signs)
exc = StripDeltaGapExcitation(feed_basis_indices=feed_edges, voltage=1.0)

N = basis.num_basis
stub_length = L/2 - port_x
print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {N} RWG, stub = {stub_length*1e3:.1f} mm")

# --- Layer stack for multilayer ---
stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

# --- Free-space EFIE (no substrate) ---
config_fs = SimulationConfig(
    frequency=1e9, excitation=exc,
    formulation='EFIE',
    backend='auto', quad_order=4,
)
sim_fs = Simulation(config_fs, mesh=mesh, reporter=SilentReporter())
ext_fs = NetworkExtractor(sim_fs, [port])

# --- Layered MPIE ---
config_ml = SimulationConfig(
    frequency=1e9, excitation=exc,
    source_layer_name='FR4',
    backend='auto', quad_order=4,
    layer_stack=stack,
)
sim_ml = Simulation(config_ml, mesh=mesh, reporter=SilentReporter())
ext_ml = NetworkExtractor(sim_ml, [port])

# --- Frequency sweep ---
freqs = np.linspace(0.5, 8.0, 16) * 1e9

print(f"\nSweeping free-space EFIE...")
results_fs = ext_fs.extract(freqs.tolist())
Y11_fs = np.array([r.Y_matrix[0, 0] for r in results_fs])

print(f"Sweeping multilayer MPIE...")
results_ml = ext_ml.extract(freqs.tolist())
Y11_ml = np.array([r.Y_matrix[0, 0] for r in results_ml])

# --- Analytical TL models ---
def y11_tl(f, Z0, eps_eff, L):
    beta = 2*np.pi*f * np.sqrt(eps_eff) / c0
    bL = beta * L
    s = np.sin(bL)
    if abs(s) < 1e-10:
        return -1j * (1/Z0) * np.sign(np.cos(bL)) * 1e10
    return -1j * (1/Z0) * np.cos(bL) / s

# For free-space strip above PEC ground, eps_eff ≈ 1 (very approximately)
# and Z0 ≈ 120*pi * h / W (parallel-plate approximation) = 197 Ohm
# More accurately for a strip: Z0_fs ≈ 60 * ln(8h/W + W/(4h)) for W < h...
# Actually W >> h here, so Z0_fs ≈ eta0 * h / W = 377 * 1.6/3.06 = 197 Ohm
Z0_fs_approx = eta0 * H / W
print(f"\nFree-space Z0 approx (parallel-plate): {Z0_fs_approx:.1f} Ohm")

# --- Print comparison ---
print(f"\n{'f(GHz)':>8}  {'Im(Y11_fs)':>12}  {'Im(Y11_ml)':>12}  {'Y11_TL_fs':>12}  {'Y11_TL_ml':>12}  {'ml/fs':>8}")
print("-" * 75)
for i, f in enumerate(freqs):
    y_tl_fs = y11_tl(f, Z0_fs_approx, 1.0, stub_length)
    y_tl_ml = y11_tl(f, Z0_ref, eps_eff_ref, stub_length)
    ratio = Y11_ml[i].imag / Y11_fs[i].imag if abs(Y11_fs[i].imag) > 1e-12 else float('nan')
    print(f"  {f/1e9:>6.2f}  {Y11_fs[i].imag:>12.6f}  {Y11_ml[i].imag:>12.6f}  "
          f"{y_tl_fs.imag:>12.6f}  {y_tl_ml.imag:>12.6f}  {ratio:>8.3f}")

# --- Check Z matrix diagonal for both ---
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators.efie import EFIEOperator
from pyMoM3d.greens.free_space_gf import FreeSpaceGreensFunction
from pyMoM3d.greens.layered import LayeredGreensFunction
from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator

freq_test = 2e9
k = 2*np.pi*freq_test / c0

print(f"\n--- Z matrix diagnostics at f = {freq_test/1e9:.1f} GHz ---")

# Free-space Z
gf_fs = FreeSpaceGreensFunction(k, eta0)
op_fs = EFIEOperator(gf_fs)
Z_fs = fill_matrix(op_fs, basis, mesh, k, eta0, quad_order=4, backend='auto')
print(f"Free-space: |Z_diag| range = [{np.abs(np.diag(Z_fs)).min():.2e}, {np.abs(np.diag(Z_fs)).max():.2e}]")
print(f"  cond(Z) = {np.linalg.cond(Z_fs):.2e}")

# Multilayer Z
gf_ml = LayeredGreensFunction(stack, freq_test, source_layer_name='FR4', backend='strata')
op_ml = MultilayerEFIEOperator(gf_ml)
Z_ml = fill_matrix(op_ml, basis, mesh, k, eta0, quad_order=4, backend='auto')
print(f"Multilayer: |Z_diag| range = [{np.abs(np.diag(Z_ml)).min():.2e}, {np.abs(np.diag(Z_ml)).max():.2e}]")
print(f"  cond(Z) = {np.linalg.cond(Z_ml):.2e}")

# Ratio
diag_ratio = np.abs(np.diag(Z_ml)) / np.abs(np.diag(Z_fs))
print(f"  |Z_ml/Z_fs| diagonal ratio: [{diag_ratio.min():.4f}, {diag_ratio.max():.4f}], median={np.median(diag_ratio):.4f}")

# Off-diagonal comparison (first row)
offdiag_fs = np.abs(Z_fs[0, 1:])
offdiag_ml = np.abs(Z_ml[0, 1:])
ratio_off = offdiag_ml / np.maximum(offdiag_fs, 1e-30)
print(f"  Off-diag ratio (row 0): median={np.median(ratio_off):.4f}, "
      f"range=[{ratio_off.min():.4f}, {ratio_off.max():.4f}]")
