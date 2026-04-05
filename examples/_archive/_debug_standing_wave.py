"""Debug: fit standing wave to current distribution away from port.

If I(x) = A*sin(beta*(L_open - x)) fits well for x > port region,
we can extract beta -> eps_eff and Z0 = V_port/I_max.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.optimize import curve_fit
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Layer, LayerStack, c0, eta0,
    microstrip_z0_hammerstad,
)
from pyMoM3d.mom.excitation import find_feed_edges, compute_feed_signs
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators.efie import EFIEOperator
from pyMoM3d.greens.layered import LayeredGreensFunction
from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator
from pyMoM3d.network.current_extraction import extract_current_profile

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 40e-3  # Longer strip for better standing wave
TEL = 1.5e-3

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

margin = TEL / 2
port_x = -L/2 + margin

mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_plate_with_feeds(
    width=L, height=W,
    feed_x_list=[port_x],
    center=(0.0, 0.0, H),
)
basis = compute_rwg_connectivity(mesh)
N = basis.num_basis
stats = mesh.get_statistics()
print(f"Mesh: {stats['num_triangles']} tris, {N} RWG, TEL={TEL*1e3:.1f} mm")

feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
signs = compute_feed_signs(mesh, basis, feed_edges)
print(f"Port at x={port_x*1e3:.2f} mm, {len(feed_edges)} feed edges")

# Build excitation
V = np.zeros(N, dtype=np.complex128)
for idx, sign in zip(feed_edges, signs):
    V[idx] = sign / basis.edge_length[idx]

x_open = L / 2  # open end at +L/2
print(f"Open end at x={x_open*1e3:.1f} mm")
print(f"Stub length from port to open end: {(x_open - port_x)*1e3:.1f} mm")

freqs = [1e9, 2e9, 3e9, 4e9, 5e9]

print(f"\n{'f(GHz)':>7}  {'beta(rad/m)':>12}  {'eps_eff':>8}  {'Z0(Ohm)':>8}  "
      f"{'R^2':>6}  {'beta_err%':>10}  {'Z0_err%':>8}")
print("-" * 75)

for freq in freqs:
    k = 2 * np.pi * freq / c0
    beta_ref = k * np.sqrt(eps_eff_ref)

    # Assemble and solve
    gf = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='auto')
    op_ml = MultilayerEFIEOperator(gf)
    Z = fill_matrix(op_ml, basis, mesh, k, eta0, quad_order=4, backend='auto')
    I = np.linalg.solve(Z, V)

    # Extract current profile
    x_pos, I_x = extract_current_profile(mesh, basis, I, propagation_axis=0)

    # Focus on region AWAY from port: x > port_x + 3*TEL
    exclude_near_port = 4 * TEL
    mask = x_pos > (port_x + exclude_near_port)
    # Also exclude very near open end
    mask &= x_pos < (x_open - TEL)

    x_fit = x_pos[mask]
    I_fit = np.abs(I_x[mask])

    if len(x_fit) < 4:
        print(f"  {freq/1e9:>5.1f}  (too few points after filtering)")
        continue

    # Fit I(x) = A * |sin(beta * (x_open - x))|
    def model(x, A, beta):
        return A * np.abs(np.sin(beta * (x_open - x)))

    try:
        popt, pcov = curve_fit(
            model, x_fit, I_fit,
            p0=[np.max(I_fit), beta_ref],
            bounds=([0, 0.1], [10*np.max(I_fit), 10*beta_ref]),
            maxfev=10000,
        )
        A_fit, beta_fit = popt
        eps_eff_fit = (beta_fit / k) ** 2

        # R^2
        I_model = model(x_fit, *popt)
        ss_res = np.sum((I_fit - I_model)**2)
        ss_tot = np.sum((I_fit - np.mean(I_fit))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-30)

        # Z0 from I_max and V_port=1
        # At x where sin(beta*(x_open-x)) = 1, I = A_fit
        # Z0 = V_port / I_max * |sin(beta*L_stub)| ≈ V/A for well-positioned port
        # More precisely: Z_in = V / I_port, and Z0 = Z_in * sin(beta*L_stub)
        # But I_port includes parasitic. Instead:
        # From the fitted amplitude A: for an open stub, I_max = V_port / Z0
        # at the current antinode. So Z0 = V_port / A_fit
        Z0_fit = 1.0 / A_fit  # V_port = 1.0

        beta_err = (beta_fit - beta_ref) / beta_ref * 100
        z0_err = (Z0_fit - Z0_ref) / Z0_ref * 100

        print(f"  {freq/1e9:>5.1f}  {beta_fit:>12.4f}  {eps_eff_fit:>8.4f}  "
              f"{Z0_fit:>8.2f}  {r2:>6.3f}  {beta_err:>+9.1f}%  {z0_err:>+7.1f}%")
    except Exception as e:
        print(f"  {freq/1e9:>5.1f}  fit failed: {e}")

    # Print current profile for visual inspection
    print(f"          x(mm)  |I_x|  model")
    for j in range(len(x_pos)):
        x = x_pos[j]
        ix = np.abs(I_x[j])
        in_fit = j < len(mask) and mask[j]
        if in_fit and 'popt' in dir():
            m_val = model(x, *popt)
            print(f"          {x*1e3:>6.1f}  {ix:>.2e}  {m_val:>.2e}  *fit")
        else:
            print(f"          {x*1e3:>6.1f}  {ix:>.2e}")
    print()
