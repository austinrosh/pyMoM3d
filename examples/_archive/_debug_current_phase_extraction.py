"""Extract eps_eff from current phase variation along microstrip.

Instead of extracting from port parameters (Y11, S21), which suffer from
port parasitic and PEC image cancellation issues, extract the propagation
constant beta directly from the spatial phase progression of the current
along the strip.

Method:
1. Excite strip at one end with probe feed
2. Compute Jx (x-component of surface current) at triangle centroids
3. Average |Jx| across the strip width at each x-position
4. Fit beta from the phase variation: angle(Jx(x)) ~ beta * x
5. eps_eff = (beta * c / (2*pi*f))^2
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor,
    Layer, LayerStack, c0,
    microstrip_z0_hammerstad,
)
from pyMoM3d.visualization.mesh_plot import compute_triangle_current_vectors

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 40e-3   # 40mm strip for clear phase progression
TEL = 0.75e-3

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f} Ohm, eps_eff = {eps_eff_ref:.3f}")
print(f"Expected beta at 2 GHz: {2*np.pi*2e9*np.sqrt(eps_eff_ref)/c0:.1f} rad/m")

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

# --- Mesh with feed at left end ---
mesher = GmshMesher(target_edge_length=TEL)
margin = TEL / 2
port_x = -L/2 + margin

mesh = mesher.mesh_plate_with_feeds(
    width=L, height=W, feed_x_list=[port_x],
    center=(0.0, 0.0, H),
)
basis = compute_rwg_connectivity(mesh)
stats = mesh.get_statistics()
print(f"Mesh: {stats['num_triangles']} tris, {basis.num_basis} RWG")

# --- Use strip delta-gap for strong excitation ---
from pyMoM3d.mom.excitation import find_feed_edges, compute_feed_signs, StripDeltaGapExcitation
feed_edges = find_feed_edges(mesh, basis, feed_x=port_x)
signs = compute_feed_signs(mesh, basis, feed_edges)
port = Port(name='P1', feed_basis_indices=feed_edges, feed_signs=signs)
exc = StripDeltaGapExcitation(feed_basis_indices=feed_edges, voltage=1.0)

print(f"Feed: {len(feed_edges)} edges at x = {port_x*1e3:.1f}mm")

config = SimulationConfig(
    frequency=1e9, excitation=exc,
    source_layer_name='FR4', backend='auto', quad_order=4,
    layer_stack=stack,
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())
ext = NetworkExtractor(sim, [port], store_currents=True)

# --- Triangle centroids ---
centroids = np.mean(mesh.vertices[mesh.triangles], axis=1)  # (T, 3)

# --- Extract beta at multiple frequencies ---
freqs = [1.5e9, 2.0e9, 2.5e9, 3.0e9]

print(f"\n{'='*70}")
print(f"Current phase extraction")
print(f"{'='*70}")

for freq in freqs:
    results = ext.extract([freq])
    I_coeffs = results[0].I_solutions[:, 0]  # current for port 1 excitation

    # Compute Jx at each triangle centroid
    T = mesh.get_statistics()['num_triangles']
    Jx = np.zeros(T, dtype=np.complex128)
    Jy = np.zeros(T, dtype=np.complex128)

    for n in range(basis.num_basis):
        l_n = basis.edge_length[n]

        # T+ contribution
        t_p = basis.t_plus[n]
        A_p = basis.area_plus[n]
        r_free_p = mesh.vertices[basis.free_vertex_plus[n]]
        rho_p = centroids[t_p] - r_free_p
        Jx[t_p] += I_coeffs[n] * (l_n / (2.0 * A_p)) * rho_p[0]
        Jy[t_p] += I_coeffs[n] * (l_n / (2.0 * A_p)) * rho_p[1]

        # T- contribution
        t_m = basis.t_minus[n]
        A_m = basis.area_minus[n]
        r_free_m = mesh.vertices[basis.free_vertex_minus[n]]
        rho_m = r_free_m - centroids[t_m]
        Jx[t_m] += I_coeffs[n] * (l_n / (2.0 * A_m)) * rho_m[0]
        Jy[t_m] += I_coeffs[n] * (l_n / (2.0 * A_m)) * rho_m[1]

    # --- Bin triangles by x-position and compute average Jx ---
    x_vals = centroids[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()
    n_bins = 30
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    Jx_avg = np.zeros(n_bins, dtype=np.complex128)
    Jx_count = np.zeros(n_bins)

    for i in range(T):
        bin_idx = int((x_vals[i] - x_min) / (x_max - x_min) * n_bins)
        bin_idx = min(bin_idx, n_bins - 1)
        Jx_avg[bin_idx] += Jx[i]
        Jx_count[bin_idx] += 1

    # Average and remove empty bins
    mask = Jx_count > 0
    Jx_avg[mask] /= Jx_count[mask]

    # --- Extract phase progression ---
    # Exclude bins near the feed (port parasitic) and near the far end (reflection)
    margin_bins = 3  # skip first and last few bins
    fit_mask = mask.copy()
    fit_mask[:margin_bins] = False
    fit_mask[-margin_bins:] = False

    if np.sum(fit_mask) < 5:
        print(f"  f={freq/1e9:.1f} GHz: not enough bins for fit")
        continue

    x_fit = x_centers[fit_mask]
    phase_fit = np.unwrap(np.angle(Jx_avg[fit_mask]))
    mag_fit = np.abs(Jx_avg[fit_mask])

    # Weighted linear fit: phase(x) = beta*x + phi_0
    # Weight by magnitude (ignore bins with weak current)
    weights = mag_fit / mag_fit.max()
    weights[weights < 0.01] = 0.01

    # Linear regression
    A_mat = np.column_stack([x_fit, np.ones(len(x_fit))])
    W_diag = np.diag(weights)
    sol = np.linalg.lstsq(W_diag @ A_mat, W_diag @ phase_fit, rcond=None)
    beta_fit = sol[0][0]

    # Expected beta
    beta_expected = 2 * np.pi * freq * np.sqrt(eps_eff_ref) / c0

    eps_eff_ext = (beta_fit * c0 / (2 * np.pi * freq)) ** 2

    print(f"\n  f = {freq/1e9:.1f} GHz:")
    print(f"    beta_fit = {beta_fit:.1f} rad/m (expected: {beta_expected:.1f})")
    print(f"    eps_eff = {eps_eff_ext:.3f} (ref: {eps_eff_ref:.3f}, "
          f"err: {abs(eps_eff_ext - eps_eff_ref)/eps_eff_ref*100:.1f}%)")

    # Print phase profile
    print(f"    x(mm)   |Jx|     phase(deg)   phase_fit(deg)")
    for j in range(n_bins):
        if not mask[j]:
            continue
        ph = np.angle(Jx_avg[j]) * 180 / np.pi
        ph_f = (sol[0][0] * x_centers[j] + sol[0][1]) * 180 / np.pi if fit_mask[j] else np.nan
        marker = '*' if fit_mask[j] else ' '
        print(f"    {x_centers[j]*1e3:>6.1f}  {abs(Jx_avg[j]):>8.1f}  {ph:>8.1f}  "
              f"{'':>8s}{marker}")
