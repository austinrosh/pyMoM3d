"""Debug QS SOC de-embedding: check ABCD extraction."""

import numpy as np
import sys
import logging
sys.path.insert(0, 'src')

logging.basicConfig(level=logging.INFO)

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, Layer, LayerStack,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.mesh.mirror import mirror_mesh_x, combine_meshes, extract_submesh
from pyMoM3d.network.soc_deembedding import _find_seam_edges, _seam_current
from pyMoM3d.utils.constants import c0, mu0, eps0

EPS_R = 4.4
H_SUB = 1.6e-3
W_STRIP = 3.06e-3
L_STRIP = 20.0e-3
TEL = 0.7e-3

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

mesher = GmshMesher(target_edge_length=TEL)
PORT1_X = -L_STRIP / 2.0 + 1.0e-3
REF1_X = -L_STRIP / 2.0 + 3.0e-3

mesh = mesher.mesh_plate_with_feeds(
    width=L_STRIP, height=W_STRIP,
    feed_x_list=[PORT1_X, -PORT1_X, REF1_X, -REF1_X],
    center=(0.0, 0.0, H_SUB),
)
basis = compute_rwg_connectivity(mesh)
print(f"Full mesh: {len(mesh.triangles)} tri, {basis.num_basis} RWG")

# ── Extract and mirror feed submesh ───────────────────────────────
stats = mesh.get_statistics()
margin = 2.0 * stats['mean_edge_length']
x_lo = PORT1_X - margin
x_hi = REF1_X

feed_sub, _ = extract_submesh(mesh, x_min=x_lo, x_max=x_hi)
mirrored = mirror_mesh_x(feed_sub, x_plane=REF1_X)
combined = combine_meshes(feed_sub, mirrored)
basis_comb = compute_rwg_connectivity(combined)

print(f"Combined mesh: {len(combined.triangles)} tri, {basis_comb.num_basis} RWG")

# Port setup on combined mesh
x_mirror_port = 2.0 * REF1_X - PORT1_X
feed_orig = find_feed_edges(combined, basis_comb, feed_x=PORT1_X)
feed_mirr = find_feed_edges(combined, basis_comb, feed_x=x_mirror_port)
signs_orig = compute_feed_signs(combined, basis_comb, feed_orig)
signs_mirr = compute_feed_signs(combined, basis_comb, feed_mirr)
port_orig = Port(name='orig', feed_basis_indices=feed_orig, feed_signs=signs_orig)
port_mirr = Port(name='mirror', feed_basis_indices=feed_mirr, feed_signs=signs_mirr)

print(f"Port orig: {len(feed_orig)} edges at x={PORT1_X*1e3:.1f} mm")
print(f"Port mirr: {len(feed_mirr)} edges at x={x_mirror_port*1e3:.1f} mm")

# Seam edges at reference plane
seam_idx, seam_sgn = _find_seam_edges(combined, basis_comb, REF1_X, x_orig_side=PORT1_X)
print(f"Seam edges: {len(seam_idx)} at x={REF1_X*1e3:.1f} mm")

# ── Solve with QS + probe feeds ──────────────────────────────────
exc = StripDeltaGapExcitation(feed_basis_indices=feed_orig, voltage=1.0)
cfg = SimulationConfig(
    frequency=1e9, excitation=exc, quad_order=4, backend='auto',
    layer_stack=stack, source_layer_name='FR4',
)
sim_comb = Simulation(cfg, mesh=combined, reporter=SilentReporter())

qs = QuasiStaticSolver(
    sim_comb, [port_orig, port_mirr],
    probe_feeds=True, n_dielectric_images=0,
    store_currents=True,
)

freq = 0.5e9
omega = 2 * np.pi * freq
vp_pf = 1j * omega * mu0
sp_pf = 1.0 / (1j * omega * eps0 * qs._eps_r)

Z_hyb = vp_pf * qs.L_raw_hybrid + sp_pf * qs.P_hybrid
I_full = np.linalg.solve(Z_hyb, qs.V_all)

N_s = qs._N_surface
N_p = qs._N_probes
print(f"\nN_surface={N_s}, N_probes={N_p}, N_total={N_s+N_p}")
print(f"basis_comb.num_basis={basis_comb.num_basis}")

# Probe currents
print(f"\nProbe currents:")
print(f"  I_full[N_s+0, 0] = {I_full[N_s+0, 0]:.6e} (orig excited)")
print(f"  I_full[N_s+1, 0] = {I_full[N_s+1, 0]:.6e} (mirr response to orig)")
print(f"  I_full[N_s+0, 1] = {I_full[N_s+0, 1]:.6e} (orig response to mirr)")
print(f"  I_full[N_s+1, 1] = {I_full[N_s+1, 1]:.6e} (mirr excited)")

# Symmetric and antisymmetric
I_sym_full = I_full[:, 0] + I_full[:, 1]
I_anti_full = I_full[:, 0] - I_full[:, 1]

# Terminal currents from probes
I_in_s = I_sym_full[N_s + 0]
I_in_o = I_anti_full[N_s + 0]

# Seam current from surface DOFs
I_ref_s = _seam_current(I_sym_full[:N_s], basis_comb, seam_idx, seam_sgn)

print(f"\nI_in_s  = {I_in_s:.6e} (input current, symmetric)")
print(f"I_in_o  = {I_in_o:.6e} (input current, antisymmetric)")
print(f"I_ref_s = {I_ref_s:.6e} (seam current, symmetric)")
print(f"dI = I_in_o - I_in_s = {I_in_o - I_in_s:.6e}")

# ABCD
dI = I_in_o - I_in_s
if abs(dI) > 1e-30 and abs(I_ref_s) > 1e-30:
    A = -I_ref_s / dI
    B = 1.0 / I_ref_s
    C = -I_in_o * I_ref_s / dI
    D = I_in_s / I_ref_s

    print(f"\nABCD error matrix:")
    print(f"  A = {A:.6e}")
    print(f"  B = {B:.6e}")
    print(f"  C = {C:.6e}")
    print(f"  D = {D:.6e}")
    print(f"  det = {A*D - B*C:.6e}")
    print(f"  Expected: A≈cos(βd), B≈jZ₀sin(βd), C≈jsin(βd)/Z₀, D≈cos(βd)")

    # For a short feed section (d = REF1_X - PORT1_X = 2mm):
    d_feed = abs(REF1_X - PORT1_X)
    print(f"\n  Feed section length: {d_feed*1e3:.1f} mm")

    # Also compare: terminal current from edges vs probes
    I_in_s_edge = port_orig.terminal_current(I_sym_full[:N_s], basis_comb)
    I_in_o_edge = port_orig.terminal_current(I_anti_full[:N_s], basis_comb)
    print(f"\n  Edge-based I_in_s = {I_in_s_edge:.6e}")
    print(f"  Edge-based I_in_o = {I_in_o_edge:.6e}")
    print(f"  Probe-based I_in_s = {I_in_s:.6e}")
    print(f"  Probe-based I_in_o = {I_in_o:.6e}")
else:
    print(f"\nCannot compute ABCD: dI={dI:.2e}, I_ref_s={I_ref_s:.2e}")

print("\nDone.")
