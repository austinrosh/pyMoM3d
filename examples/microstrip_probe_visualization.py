"""Visualization of probe-fed quasi-static microstrip simulation.

Generates four figures:
1. Top-down view of the mesh with port locations
2. 3D isometric view of the mesh on layered substrate
3. Surface current density at 1 GHz
4. S-parameters (S11, S21) vs frequency in dB
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
sys.path.insert(0, 'src')

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, Layer, LayerStack,
    configure_latex_style, plot_mesh_3d, plot_surface_current,
    compute_triangle_current_density,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.utils.constants import c0

configure_latex_style()

# ── Geometry ──────────────────────────────────────────────────────────
EPS_R = 4.4
H_SUB = 1.6e-3       # m
W_STRIP = 3.06e-3    # m
L_STRIP = 10.0e-3    # m
TEL = 0.7e-3          # mesh edge length
PORT1_X = -L_STRIP / 2.0 + 1.0e-3
PORT2_X = +L_STRIP / 2.0 - 1.0e-3

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_plate_with_feeds(
    width=L_STRIP, height=W_STRIP,
    feed_x_list=[PORT1_X, PORT2_X],
    center=(0.0, 0.0, H_SUB),
)
basis = compute_rwg_connectivity(mesh)

feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
signs1 = compute_feed_signs(mesh, basis, feed1)
signs2 = compute_feed_signs(mesh, basis, feed2)
port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
config = SimulationConfig(
    frequency=1e9, excitation=exc, quad_order=4, backend='auto',
    layer_stack=stack, source_layer_name='FR4',
)
sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

print(f"Mesh: {len(mesh.triangles)} triangles, {basis.num_basis} RWG basis")

# ── Build solver ──────────────────────────────────────────────────────
qs = QuasiStaticSolver(sim, [port1, port2], probe_feeds=True,
                       store_currents=True)

# ── Dense frequency sweep for S-parameter plot ───────────────────────
freqs = np.linspace(0.1e9, 12e9, 120)
results = qs.extract(freqs.tolist())

S11_dB = np.array([20 * np.log10(max(abs(r.S_matrix[0, 0]), 1e-30))
                    for r in results])
S21_dB = np.array([20 * np.log10(max(abs(r.S_matrix[1, 0]), 1e-30))
                    for r in results])
f_GHz = freqs / 1e9

# ── Get currents at 1 GHz for surface current plot ───────────────────
idx_1GHz = np.argmin(np.abs(freqs - 1e9))
I_1GHz = results[idx_1GHz].I_solutions[:, 0]  # port-1 excitation

# ── Figure 1: Top-down mesh view ─────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 4))
verts = mesh.vertices
for tri in mesh.triangles:
    pts = verts[tri]
    polygon = plt.Polygon(pts[:, :2] * 1e3, fill=False,
                          edgecolor='steelblue', linewidth=0.4)
    ax1.add_patch(polygon)

# Mark port locations
for px, label, color in [(PORT1_X, 'P1', 'C3'), (PORT2_X, 'P2', 'C0')]:
    ax1.axvline(px * 1e3, color=color, linewidth=1.5, linestyle='--',
                alpha=0.8, label=label)
    # Mark probe vertex
    vi = qs._probe_vertices[0 if label == 'P1' else 1]
    ax1.plot(verts[vi, 0] * 1e3, verts[vi, 1] * 1e3,
             'o', color=color, markersize=6, zorder=5)

ax1.set_xlim(verts[:, 0].min() * 1e3 - 0.3, verts[:, 0].max() * 1e3 + 0.3)
ax1.set_ylim(verts[:, 1].min() * 1e3 - 0.3, verts[:, 1].max() * 1e3 + 0.3)
ax1.set_aspect('equal')
ax1.set_xlabel(r'$x$ (mm)')
ax1.set_ylabel(r'$y$ (mm)')
ax1.set_title(rf'Microstrip Mesh — Top View ({len(mesh.triangles)} triangles, '
              rf'{basis.num_basis} RWG)')
ax1.legend(loc='upper right')
fig1.tight_layout()
fig1.savefig('microstrip_mesh_top.png', dpi=200)
print("Saved microstrip_mesh_top.png")

# ── Figure 2: 3D isometric with substrate ────────────────────────────
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')

# Draw substrate slab (dielectric)
x_lo, x_hi = verts[:, 0].min() - 0.5e-3, verts[:, 0].max() + 0.5e-3
y_lo, y_hi = verts[:, 1].min() - 0.5e-3, verts[:, 1].max() + 0.5e-3
substrate_corners = np.array([
    [x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi],
])
# Bottom face (ground plane, z=0)
ground_verts = [[*sc, 0.0] for sc in substrate_corners]
ground = Poly3DCollection([ground_verts], alpha=0.3, facecolor='#B87333',
                          edgecolor='#8B5E3C', linewidth=0.5)
ax2.add_collection3d(ground)

# Side faces
for i in range(4):
    j = (i + 1) % 4
    side = [
        [*substrate_corners[i], 0], [*substrate_corners[j], 0],
        [*substrate_corners[j], H_SUB], [*substrate_corners[i], H_SUB],
    ]
    face = Poly3DCollection([side], alpha=0.15, facecolor='#90EE90',
                            edgecolor='gray', linewidth=0.3)
    ax2.add_collection3d(face)

# Top face of substrate
top_verts = [[*sc, H_SUB] for sc in substrate_corners]
top_face = Poly3DCollection([top_verts], alpha=0.1, facecolor='#90EE90',
                            edgecolor='gray', linewidth=0.3)
ax2.add_collection3d(top_face)

# Draw strip mesh on top
strip_polys = []
for tri in mesh.triangles:
    pts = verts[tri]
    strip_polys.append(pts.tolist())
strip_coll = Poly3DCollection(strip_polys, alpha=0.85, facecolor='gold',
                              edgecolor='k', linewidth=0.3)
ax2.add_collection3d(strip_coll)

# Draw probe vias
for pi, vi in enumerate(qs._probe_vertices):
    vx, vy = verts[vi, 0], verts[vi, 1]
    ax2.plot([vx, vx], [vy, vy], [0, H_SUB],
             color='red', linewidth=2.5, zorder=10)
    ax2.scatter([vx], [vy], [0], color='red', s=20, zorder=10)
    ax2.scatter([vx], [vy], [H_SUB], color='red', s=20, zorder=10)
    ax2.text(vx, vy, H_SUB * 1.8,
             f'P{pi+1}', color='red', fontsize=10, ha='center')

# Labels
ax2.set_xlabel(r'$x$ (m)')
ax2.set_ylabel(r'$y$ (m)')
ax2.set_zlabel(r'$z$ (m)')
ax2.set_title(r'Probe-Fed Microstrip on FR4 ($\varepsilon_r = 4.4$, '
              r'$h = 1.6\,\mathrm{mm}$)')
ax2.view_init(elev=25, azim=-55)
# Set axis limits
margin = 0.5e-3
ax2.set_xlim(x_lo - margin, x_hi + margin)
ax2.set_ylim(y_lo - margin, y_hi + margin)
ax2.set_zlim(-0.2e-3, H_SUB * 3)
fig2.tight_layout()
fig2.savefig('microstrip_mesh_3d.png', dpi=200)
print("Saved microstrip_mesh_3d.png")

# ── Figure 3: Surface current density at 1 GHz ──────────────────────
fig3 = plt.figure(figsize=(10, 5))
ax3 = fig3.add_subplot(111, projection='3d')

ax3_out, mappable = plot_surface_current(
    I_1GHz, basis, mesh, ax=ax3,
    cmap='inferno', show_edges=True,
    edge_color='gray', edge_width=0.2, alpha=0.95,
    title=rf'$|\mathbf{{J}}|$ at $f = 1\,\mathrm{{GHz}}$ (probe feed, port 1 excited)',
)
fig3.colorbar(mappable, ax=ax3, shrink=0.6, pad=0.1,
              label=r'$|\mathbf{J}|$ (A/m)')
ax3.view_init(elev=60, azim=-60)
fig3.tight_layout()
fig3.savefig('microstrip_current_1GHz.png', dpi=200)
print("Saved microstrip_current_1GHz.png")

# ── Figure 3b: Top-down surface current (2D heatmap) ────────────────
J_mag = compute_triangle_current_density(I_1GHz, basis, mesh)
centroids = np.mean(mesh.vertices[mesh.triangles], axis=1)

fig3b, ax3b = plt.subplots(figsize=(9, 4))
from matplotlib.collections import PolyCollection
polys_2d = []
for tri in mesh.triangles:
    pts = mesh.vertices[tri, :2] * 1e3  # mm
    polys_2d.append(pts)
pc = PolyCollection(polys_2d, array=J_mag, cmap='inferno',
                    edgecolors='gray', linewidths=0.2)
ax3b.add_collection(pc)
ax3b.set_xlim(mesh.vertices[:, 0].min() * 1e3 - 0.2,
              mesh.vertices[:, 0].max() * 1e3 + 0.2)
ax3b.set_ylim(mesh.vertices[:, 1].min() * 1e3 - 0.2,
              mesh.vertices[:, 1].max() * 1e3 + 0.2)
ax3b.set_aspect('equal')
ax3b.set_xlabel(r'$x$ (mm)')
ax3b.set_ylabel(r'$y$ (mm)')
ax3b.set_title(r'Surface Current $|\mathbf{J}|$ at $f = 1\,\mathrm{GHz}$ — Top View')
cb = fig3b.colorbar(pc, ax=ax3b)
cb.set_label(r'$|\mathbf{J}|$ (A/m)')

# Mark probe positions
for pi, vi in enumerate(qs._probe_vertices):
    ax3b.plot(mesh.vertices[vi, 0] * 1e3, mesh.vertices[vi, 1] * 1e3,
              'v', color='cyan', markersize=8, markeredgecolor='white',
              markeredgewidth=0.8, zorder=5)
    ax3b.annotate(f'P{pi+1}',
                  (mesh.vertices[vi, 0] * 1e3, mesh.vertices[vi, 1] * 1e3),
                  textcoords='offset points', xytext=(0, -12),
                  color='cyan', fontsize=9, ha='center', fontweight='bold')

fig3b.tight_layout()
fig3b.savefig('microstrip_current_top.png', dpi=200)
print("Saved microstrip_current_top.png")

# ── Figure 4: S-parameters vs frequency ─────────────────────────────
fig4, ax4 = plt.subplots(figsize=(9, 5.5))

ax4.plot(f_GHz, S21_dB, 'C0-', linewidth=1.8, label=r'$S_{21}$ (insertion loss)')
ax4.plot(f_GHz, S11_dB, 'C3-', linewidth=1.8, label=r'$S_{11}$ (return loss)')

# Mark the quasi-static validity limit (kD ~ 0.5)
eps_eff = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 / np.sqrt(1 + 12 * H_SUB / W_STRIP)
f_qs_limit = 0.5 * c0 / (L_STRIP * np.sqrt(eps_eff)) / 1e9
ax4.axvline(f_qs_limit, color='gray', linestyle=':', linewidth=1,
            alpha=0.7)
ax4.text(f_qs_limit + 0.15, -1.5,
         rf'$kD \approx 0.5$ ({f_qs_limit:.1f} GHz)',
         color='gray', fontsize=9, rotation=90, va='bottom')

# Mark quarter-wave resonance
f_qw = c0 / (4 * (PORT2_X - PORT1_X) * np.sqrt(eps_eff)) / 1e9
ax4.axvline(f_qw, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax4.text(f_qw + 0.15, -1.5,
         rf'$\lambda/4$ ({f_qw:.1f} GHz)',
         color='gray', fontsize=9, rotation=90, va='bottom')

ax4.set_xlabel(r'Frequency $f$ (GHz)')
ax4.set_ylabel(r'$|S|$ (dB)')
ax4.set_title(r'Probe-Fed Microstrip S-Parameters (QS solver, '
              r'$L = 10\,\mathrm{mm}$, FR4)')
ax4.set_xlim(0, 12)
ax4.set_ylim(-40, 2)
ax4.legend(loc='lower left', fontsize=10)
ax4.grid(True, alpha=0.3)

# Annotate QS regime
ax4.annotate('', xy=(0.1, -38), xytext=(f_qs_limit, -38),
             arrowprops=dict(arrowstyle='<->', color='C2', lw=1.5))
ax4.text(f_qs_limit / 2, -37, 'quasi-static regime',
         ha='center', va='bottom', color='C2', fontsize=9)

fig4.tight_layout()
fig4.savefig('microstrip_sparams.png', dpi=200)
print("Saved microstrip_sparams.png")

plt.show()
