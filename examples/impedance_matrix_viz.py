"""
Example: Impedance Matrix Visualisation

Uses a small strip dipole (coarse mesh, ~40-80 RWG basis functions) to
assemble and visualise the EFIE impedance matrix Z.

Produces:
  Figure 1 — Standalone |Z| magnitude heatmap (dB)  <-- primary plot
  Figure 2 — Four-panel heatmap: Re(Z), Im(Z), |Z| dB, arg(Z)
  Figure 3 — Structural properties: row norms, singular values,
              coupling decay, and diagonal self-terms
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

from pyMoM3d import (
    GmshMesher,
    compute_rwg_connectivity,
    fill_matrix,
    EFIEOperator,
    configure_latex_style,
    eta0,
    c0,
)
from pyMoM3d.mom.excitation import find_feed_edges

# --- Style: use mathtext (no external LaTeX needed), larger readable fonts ---
configure_latex_style(use_tex=False, font_size=13)
mpl.rcParams.update({
    'figure.figsize':   (9, 7),     # comfortable default for single panels
    'axes.grid':        False,      # grids fight imshow pixel grids
    'image.interpolation': 'none',  # keep matrix pixels sharp
    'image.aspect':     'auto',
})

# =============================================================================
# 1.  Geometry and mesh  (coarse — keeps N small enough to read each entry)
# =============================================================================
dipole_length = 0.15     # m  (half-wave at ~1 GHz)
dipole_width  = 0.01     # m

target_edge_length = 0.012   # ~12 mm  →  N ≈ 50-80 RWG basis functions

mesher = GmshMesher(target_edge_length=target_edge_length)
mesh   = mesher.mesh_plate_with_feed(
    width=dipole_length,
    height=dipole_width,
    feed_x=0.0,
    center=(0, 0, 0),
)
basis = compute_rwg_connectivity(mesh)
N     = basis.num_basis

print(f"Triangles : {mesh.get_num_triangles()}")
print(f"RWG basis : N = {N}")

# =============================================================================
# 2.  Assemble Z at near-resonant frequency
# =============================================================================
freq = 1.0e9                    # 1 GHz
k    = 2.0 * np.pi * freq / c0
lam  = c0 / freq

print(f"\nFrequency : {freq/1e9:.2f} GHz   lambda = {lam*100:.1f} cm")
print(f"h/lambda  : {target_edge_length/lam:.3f}")
print("Filling Z matrix ...")

Z = fill_matrix(EFIEOperator(), basis, mesh, k, eta0, quad_order=4)

sv = np.linalg.svd(Z, compute_uv=False)
print(f"Z shape   : {Z.shape}   dtype = {Z.dtype}")
print(f"|Z| range : {np.abs(Z).min():.4f} – {np.abs(Z).max():.4f}  Ohm")
print(f"Symmetry  : max|Z - Z^T| = {np.max(np.abs(Z - Z.T)):.2e}  Ohm")
print(f"kappa(Z)  = {sv[0]/sv[-1]:.2e}")

# Shared quantities
idx         = np.arange(N)
absZ_dB     = 20.0 * np.log10(np.maximum(np.abs(Z), 1e-12))
feed_indices = find_feed_edges(mesh, basis, feed_x=0.0)

# =============================================================================
# 3.  Figure 1 — Standalone |Z| dB heatmap  (primary plot)
# =============================================================================
vmin_dB = np.percentile(absZ_dB, 2)   # clip noisy floor at 2nd percentile
vmax_dB = absZ_dB.max()

fig1, ax1 = plt.subplots(figsize=(7, 6))

im1 = ax1.imshow(
    absZ_dB,
    cmap='inferno',
    vmin=vmin_dB, vmax=vmax_dB,
    extent=[0.5, N + 0.5, N + 0.5, 0.5],
    aspect='equal',
)

cb1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
cb1.set_label(r'$20\,\log_{10}|Z_{mn}|$ (dB $\Omega$)', fontsize=13)
cb1.ax.tick_params(labelsize=11)

ax1.set_xlabel(r'Source basis index $n$', fontsize=14)
ax1.set_ylabel(r'Test basis index $m$', fontsize=14)
ax1.tick_params(labelsize=11)
ax1.set_title(
    rf'$|\mathbf{{Z}}|$ — EFIE Impedance Matrix Magnitude'
    '\n'
    rf'Strip dipole, $N={N}$, $f={freq/1e9:.1f}$ GHz, '
    rf'$h/\lambda \approx {target_edge_length/lam:.2f}$',
    fontsize=13, pad=10,
)


fig1.tight_layout()


# =============================================================================
# 4.  Figure 2 — Four-panel heatmap
# =============================================================================
fig2, axes = plt.subplots(2, 2, figsize=(10, 8))
fig2.suptitle(
    rf'EFIE Impedance Matrix $\mathbf{{Z}}$ — Component Views'
    rf'  ($N={N}$, $f={freq/1e9:.1f}$ GHz)',
    fontsize=11,
)

def _imshow_panel(ax, data, cmap, vmin, vmax, cblabel, title):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0.5, N+0.5, N+0.5, 0.5], aspect='equal')
    cb = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cblabel, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel(r'Source index $n$', fontsize=9)
    ax.set_ylabel(r'Test index $m$', fontsize=9)
    ax.tick_params(labelsize=8)
    return im

ReZ = Z.real
ImZ = Z.imag
vmax_re = np.percentile(np.abs(ReZ), 98)
vmax_im = np.percentile(np.abs(ImZ), 98)
argZ    = np.degrees(np.angle(Z))

_imshow_panel(axes[0, 0], ReZ, 'RdBu_r', -vmax_re, vmax_re,
              r'$\mathrm{Re}(Z_{mn})$ ($\Omega$)',
              r'$\mathrm{Re}(\mathbf{Z})$ — resistive coupling')

_imshow_panel(axes[0, 1], ImZ, 'PuOr_r', -vmax_im, vmax_im,
              r'$\mathrm{Im}(Z_{mn})$ ($\Omega$)',
              r'$\mathrm{Im}(\mathbf{Z})$ — reactive coupling')

_imshow_panel(axes[1, 0], absZ_dB, 'inferno', vmin_dB, vmax_dB,
              r'$20\log_{10}|Z_{mn}|$ (dB $\Omega$)',
              r'$|\mathbf{Z}|$ (dB) — magnitude')

_imshow_panel(axes[1, 1], argZ, 'hsv', -180, 180,
              r'$\arg(Z_{mn})$ (deg)',
              r'$\arg(\mathbf{Z})$ — phase')

fig2.tight_layout()


# =============================================================================
# 5.  Figure 3 — Structural properties
# =============================================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(10, 8))
fig3.suptitle(
    rf'Structural Properties of $\mathbf{{Z}}$ ($N={N}$, $f={freq/1e9:.1f}$ GHz)',
    fontsize=11,
)

# --- Panel A: Row-norm (diagonal dominance) ---
ax = axes3[0, 0]
row_norms = np.abs(Z).sum(axis=1)
diag_abs  = np.abs(np.diag(Z))
off_sum   = row_norms - diag_abs
dom_ratio = diag_abs / np.where(off_sum > 0, off_sum, 1e-30)

ax.bar(idx + 1, diag_abs, width=0.8, label=r'$|Z_{mm}|$',
       color='steelblue', alpha=0.85)
ax.bar(idx + 1, off_sum, width=0.8, bottom=diag_abs,
       label=r'$\sum_{n \neq m}|Z_{mn}|$', color='coral', alpha=0.75)
ax.set_xlabel(r'Basis index $m$', fontsize=9)
ax.set_ylabel(r'$\Omega$', fontsize=9)
ax.set_title(r'Row $\ell^1$ norm: diagonal vs.\ off-diagonal', fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(0.5, N + 0.5)
ax.tick_params(labelsize=8)

ax_r = ax.twinx()
ax_r.plot(idx + 1, dom_ratio, 'k--', lw=1.2,
          label=r'$|Z_{mm}|/\!\sum_{n\neq m}|Z_{mn}|$')
ax_r.axhline(1.0, color='gray', lw=0.8, ls=':')
ax_r.set_ylabel("Dominance ratio", fontsize=9)
ax_r.legend(fontsize=8, loc='upper right')
ax_r.tick_params(labelsize=9)

# --- Panel B: Singular value spectrum ---
ax = axes3[0, 1]
ax.semilogy(np.arange(1, N + 1), sv, 'o-', ms=5, lw=1.4, color='steelblue')
ax.semilogy([1, N], [sv[0],  sv[0]],  'r--', lw=1.0,
            label=rf'$\sigma_1 = {sv[0]:.3f}\ \Omega$')
ax.semilogy([1, N], [sv[-1], sv[-1]], 'g--', lw=1.0,
            label=rf'$\sigma_N = {sv[-1]:.2e}\ \Omega$')
ax.set_xlabel('Singular value index', fontsize=9)
ax.set_ylabel(r'$\sigma_i$ ($\Omega$)', fontsize=9)
ax.set_title(rf'Singular value spectrum  ($\kappa = {sv[0]/sv[-1]:.2e}$)',
             fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)
ax.tick_params(labelsize=8)

# --- Panel C: Coupling decay vs offset ---
ax = axes3[1, 0]
max_offset = min(N - 1, N // 2)
offsets   = np.arange(0, max_offset + 1)
mean_diag = np.zeros(max_offset + 1)
for delta in offsets:
    if delta == 0:
        vals = np.abs(np.diag(Z))
    else:
        vals = np.concatenate([np.abs(np.diag(Z, k=delta)),
                               np.abs(np.diag(Z, k=-delta))])
    mean_diag[delta] = np.mean(vals)

ax.semilogy(offsets, mean_diag, 's-', ms=5, lw=1.4, color='darkorange')
ax.set_xlabel(r'Off-diagonal offset $\delta = |m - n|$', fontsize=9)
ax.set_ylabel(r'Mean $|Z_{m,\,m+\delta}|$ ($\Omega$)', fontsize=9)
ax.set_title('Coupling magnitude vs. basis-function separation', fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(-0.5, max_offset + 0.5)
ax.tick_params(labelsize=8)

# --- Panel D: Diagonal self-terms Re / Im ---
ax = axes3[1, 1]
sort_idx = np.argsort(basis.edge_length)
diag_re  = np.real(np.diag(Z))[sort_idx]
diag_im  = np.imag(np.diag(Z))[sort_idx]

ax.plot(idx + 1, diag_re, 'o-', ms=4, lw=1.2, color='steelblue',
        label=r'$\mathrm{Re}(Z_{mm})$')
ax.plot(idx + 1, diag_im, 's-', ms=4, lw=1.2, color='coral',
        label=r'$\mathrm{Im}(Z_{mm})$')
ax.axhline(0, color='gray', lw=0.7, ls='--')
ax.set_xlabel(r'Basis index (sorted by edge length $l_n$)', fontsize=9)
ax.set_ylabel(r'$\Omega$', fontsize=9)
ax.set_title(r'Diagonal self-coupling terms $Z_{mm}$', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, N + 0.5)
ax.tick_params(labelsize=8)

# Mark feed edges
if feed_indices:
    feed_sorted_pos = np.where(np.isin(sort_idx, feed_indices))[0]
    for pos in feed_sorted_pos:
        ax.axvline(pos + 1, color='green', lw=1.0, ls=':', alpha=0.8)
    ax.plot([], [], color='green', ls=':', lw=1.0, label='feed edges')
    ax.legend(fontsize=8)

fig3.tight_layout()

# =============================================================================
# 6.  Save
# =============================================================================
out_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
fig1.savefig(os.path.join(out_dir, 'Z_magnitude_dB.png'), dpi=180, bbox_inches='tight')
fig2.savefig(os.path.join(out_dir, 'Z_heatmap.png'),      dpi=150, bbox_inches='tight')
fig3.savefig(os.path.join(out_dir, 'Z_properties.png'),   dpi=150, bbox_inches='tight')

print("\nSaved:")
print("  docs/Z_magnitude_dB.png  — standalone |Z| dB heatmap")
print("  docs/Z_heatmap.png       — four-panel component heatmaps")
print("  docs/Z_properties.png    — structural properties")

plt.show()
