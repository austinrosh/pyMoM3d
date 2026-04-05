"""Visualization of edge-fed vertical plate port microstrip simulation.

Generates figures:
1. 3D isometric view of the mesh with vertical plates and substrate
2. Top-down view of the mesh with port locations
3. Surface current density at 1 GHz (3D)
4. Surface current density at 1 GHz (top-down heatmap)
5. S-parameters (S11, S21) vs frequency in dB
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
import sys
sys.path.insert(0, 'src')

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, Layer, LayerStack,
    configure_latex_style, plot_surface_current,
    compute_triangle_current_density,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_edge_port_feed_edges, compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.utils.constants import c0

configure_latex_style()

# ── Geometry ──────────────────────────────────────────────────────
EPS_R = 4.4
H_SUB = 1.6e-3       # m
W_STRIP = 3.06e-3    # m
L_STRIP = 10.0e-3    # m
TEL = 0.7e-3          # mesh edge length

stack = LayerStack([
    Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
    Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
])

# ── Mesh with edge ports ─────────────────────────────────────────
mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_microstrip_with_edge_ports(
    width=W_STRIP,
    length=L_STRIP,
    substrate_height=H_SUB,
    port_edges=['left', 'right'],
)
basis = compute_rwg_connectivity(mesh)

x_left = -L_STRIP / 2.0
x_right = +L_STRIP / 2.0

feed1 = find_edge_port_feed_edges(mesh, basis, port_x=x_left, strip_z=H_SUB)
feed2 = find_edge_port_feed_edges(mesh, basis, port_x=x_right, strip_z=H_SUB)
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

# Count strip vs plate triangles
z_centroids = np.mean(mesh.vertices[mesh.triangles, 2], axis=1)
n_strip = np.sum(np.abs(z_centroids - H_SUB) < 1e-6)
n_plate = len(mesh.triangles) - n_strip
print(f"  Strip: {n_strip} triangles, Plates: {n_plate} triangles")

# ── Build solver and extract ─────────────────────────────────────
qs = QuasiStaticSolver(sim, [port1, port2], probe_feeds=True,
                       store_currents=True)

freqs = np.linspace(0.1e9, 12e9, 120)
results = qs.extract(freqs.tolist())

S11_dB = np.array([20 * np.log10(max(abs(r.S_matrix[0, 0]), 1e-30))
                    for r in results])
S21_dB = np.array([20 * np.log10(max(abs(r.S_matrix[1, 0]), 1e-30))
                    for r in results])
f_GHz = freqs / 1e9

# Get currents at 1 GHz
idx_1GHz = np.argmin(np.abs(freqs - 1e9))
I_1GHz = results[idx_1GHz].I_solutions[:, 0]

verts = mesh.vertices

# ── Helper: classify triangles as strip or plate ─────────────────
def is_strip_tri(t):
    return np.allclose(verts[mesh.triangles[t], 2], H_SUB, atol=1e-8)

# ── Figure 1: 3D isometric with substrate and plates ─────────────
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')

# Substrate slab
x_lo, x_hi = verts[:, 0].min() - 0.5e-3, verts[:, 0].max() + 0.5e-3
y_lo, y_hi = verts[:, 1].min() - 0.5e-3, verts[:, 1].max() + 0.5e-3
sub_corners = np.array([
    [x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi],
])

# Ground plane (z=0)
ground_verts = [[*sc, 0.0] for sc in sub_corners]
ground = Poly3DCollection([ground_verts], alpha=0.3, facecolor='#B87333',
                          edgecolor='#8B5E3C', linewidth=0.5)
ax1.add_collection3d(ground)

# Substrate sides
for i in range(4):
    j = (i + 1) % 4
    side = [
        [*sub_corners[i], 0], [*sub_corners[j], 0],
        [*sub_corners[j], H_SUB], [*sub_corners[i], H_SUB],
    ]
    face = Poly3DCollection([side], alpha=0.15, facecolor='#90EE90',
                            edgecolor='gray', linewidth=0.3)
    ax1.add_collection3d(face)

# Substrate top
top_verts = [[*sc, H_SUB] for sc in sub_corners]
top_face = Poly3DCollection([top_verts], alpha=0.1, facecolor='#90EE90',
                            edgecolor='gray', linewidth=0.3)
ax1.add_collection3d(top_face)

# Draw mesh triangles — strip in gold, plates in red
strip_polys = []
plate_polys = []
for t_idx in range(len(mesh.triangles)):
    pts = verts[mesh.triangles[t_idx]].tolist()
    if is_strip_tri(t_idx):
        strip_polys.append(pts)
    else:
        plate_polys.append(pts)

if strip_polys:
    strip_coll = Poly3DCollection(strip_polys, alpha=0.85, facecolor='gold',
                                  edgecolor='k', linewidth=0.3)
    ax1.add_collection3d(strip_coll)

if plate_polys:
    plate_coll = Poly3DCollection(plate_polys, alpha=0.85, facecolor='#FF6B6B',
                                  edgecolor='k', linewidth=0.3)
    ax1.add_collection3d(plate_coll)

# Label ports
for px, label in [(x_left, 'P1'), (x_right, 'P2')]:
    ax1.text(px, 0, H_SUB * 2.2, label, color='red', fontsize=11,
             ha='center', fontweight='bold')

ax1.set_xlabel(r'$x$ (m)')
ax1.set_ylabel(r'$y$ (m)')
ax1.set_zlabel(r'$z$ (m)')
ax1.set_title(rf'Edge-Fed Microstrip on FR4 ($\varepsilon_r = {EPS_R}$, '
              rf'$h = {H_SUB*1e3:.1f}\,\mathrm{{mm}}$)')
ax1.view_init(elev=25, azim=-55)
margin = 0.5e-3
ax1.set_xlim(x_lo - margin, x_hi + margin)
ax1.set_ylim(y_lo - margin, y_hi + margin)
ax1.set_zlim(-0.2e-3, H_SUB * 3)
fig1.tight_layout()
fig1.savefig('edge_port_mesh_3d.png', dpi=200)
print("Saved edge_port_mesh_3d.png")

# ── Figure 2: Top-down mesh view ─────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
for t_idx in range(len(mesh.triangles)):
    pts = verts[mesh.triangles[t_idx]]
    color = 'steelblue' if is_strip_tri(t_idx) else '#FF6B6B'
    polygon = plt.Polygon(pts[:, :2] * 1e3, fill=False,
                          edgecolor=color, linewidth=0.4)
    ax2.add_patch(polygon)

# Mark port edge locations
for px, label, color in [(x_left, 'P1', 'C3'), (x_right, 'P2', 'C0')]:
    ax2.axvline(px * 1e3, color=color, linewidth=1.5, linestyle='--',
                alpha=0.8, label=label)

ax2.set_xlim(verts[:, 0].min() * 1e3 - 0.3, verts[:, 0].max() * 1e3 + 0.3)
ax2.set_ylim(verts[:, 1].min() * 1e3 - 0.3, verts[:, 1].max() * 1e3 + 0.3)
ax2.set_aspect('equal')
ax2.set_xlabel(r'$x$ (mm)')
ax2.set_ylabel(r'$y$ (mm)')
ax2.set_title(rf'Edge-Fed Microstrip — Top View ({len(mesh.triangles)} triangles, '
              rf'{basis.num_basis} RWG, plate triangles in red)')
ax2.legend(loc='upper right')
fig2.tight_layout()
fig2.savefig('edge_port_mesh_top.png', dpi=200)
print("Saved edge_port_mesh_top.png")

# ── Figure 3: 3D surface current at 1 GHz ────────────────────────
fig3 = plt.figure(figsize=(10, 5))
ax3 = fig3.add_subplot(111, projection='3d')

ax3_out, mappable = plot_surface_current(
    I_1GHz, basis, mesh, ax=ax3,
    cmap='inferno', show_edges=True,
    edge_color='gray', edge_width=0.2, alpha=0.95,
    title=rf'$|\mathbf{{J}}|$ at $f = 1\,\mathrm{{GHz}}$ (edge port, port 1 excited)',
)
fig3.colorbar(mappable, ax=ax3, shrink=0.6, pad=0.1,
              label=r'$|\mathbf{J}|$ (A/m)')
ax3.view_init(elev=60, azim=-60)
fig3.tight_layout()
fig3.savefig('edge_port_current_3d.png', dpi=200)
print("Saved edge_port_current_3d.png")

# ── Figure 4: Top-down surface current heatmap ───────────────────
J_mag = compute_triangle_current_density(I_1GHz, basis, mesh)

fig4, ax4 = plt.subplots(figsize=(9, 4))
polys_2d = []
for tri in mesh.triangles:
    pts = verts[tri, :2] * 1e3
    polys_2d.append(pts)
pc = PolyCollection(polys_2d, array=J_mag, cmap='inferno',
                    edgecolors='gray', linewidths=0.2)
ax4.add_collection(pc)
ax4.set_xlim(verts[:, 0].min() * 1e3 - 0.2, verts[:, 0].max() * 1e3 + 0.2)
ax4.set_ylim(verts[:, 1].min() * 1e3 - 0.2, verts[:, 1].max() * 1e3 + 0.2)
ax4.set_aspect('equal')
ax4.set_xlabel(r'$x$ (mm)')
ax4.set_ylabel(r'$y$ (mm)')
ax4.set_title(r'Surface Current $|\mathbf{J}|$ at $f = 1\,\mathrm{GHz}$ — Edge Ports')
cb = fig4.colorbar(pc, ax=ax4)
cb.set_label(r'$|\mathbf{J}|$ (A/m)')

# Mark port edges
for px, label in [(x_left, 'P1'), (x_right, 'P2')]:
    ax4.axvline(px * 1e3, color='cyan', linewidth=1.2, linestyle='--', alpha=0.7)
    ax4.text(px * 1e3, verts[:, 1].max() * 1e3 + 0.15, label,
             color='cyan', fontsize=9, ha='center', fontweight='bold')

fig4.tight_layout()
fig4.savefig('edge_port_current_top.png', dpi=200)
print("Saved edge_port_current_top.png")

# ── Figure 5: S-parameters — QS valid range only ─────────────────
eps_eff = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 / np.sqrt(1 + 12 * H_SUB / W_STRIP)
f_qs_limit = 0.5 * c0 / (L_STRIP * np.sqrt(eps_eff)) / 1e9

# Zoom to QS-valid range (0 to ~3 GHz)
f_max_plot = 3.0
mask = f_GHz <= f_max_plot

fig5, ax5 = plt.subplots(figsize=(9, 5.5))

ax5.plot(f_GHz[mask], S21_dB[mask], 'C0-', linewidth=1.8,
         label=r'$S_{21}$ (insertion loss)')
ax5.plot(f_GHz[mask], S11_dB[mask], 'C3-', linewidth=1.8,
         label=r'$S_{11}$ (return loss)')

ax5.set_xlabel(r'Frequency $f$ (GHz)')
ax5.set_ylabel(r'$|S|$ (dB)')
ax5.set_title(r'Edge-Fed Microstrip S-Parameters — QS Valid Range'
              '\n'
              r'($L = 10\,\mathrm{mm}$, FR4, $\varepsilon_r = 4.4$, '
              r'$h = 1.6\,\mathrm{mm}$)')
ax5.set_xlim(0, f_max_plot)
ax5.set_ylim(-40, 2)
ax5.legend(loc='lower left', fontsize=10)
ax5.grid(True, alpha=0.3)

fig5.tight_layout()
fig5.savefig('edge_port_sparams_qs.png', dpi=200)
print("Saved edge_port_sparams_qs.png")

# ── Figure 6: Full range showing QS breakdown ────────────────────
fig6, ax6 = plt.subplots(figsize=(9, 5.5))

ax6.plot(f_GHz, S21_dB, 'C0-', linewidth=1.8, label=r'$S_{21}$')
ax6.plot(f_GHz, S11_dB, 'C3-', linewidth=1.8, label=r'$S_{11}$')

# Shade invalid region
ax6.axvspan(f_qs_limit, 12, alpha=0.08, color='red')
ax6.axvline(f_qs_limit, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax6.text(f_qs_limit + 0.15, -2,
         rf'$kD \approx 0.5$ ({f_qs_limit:.1f} GHz)',
         color='gray', fontsize=9, rotation=90, va='bottom')
ax6.text(10, -3, 'QS invalid\n(non-physical)', color='red',
         fontsize=10, ha='center', alpha=0.6)

# Shade probe-parasitic region
ax6.axvspan(2.0, f_qs_limit, alpha=0.06, color='orange')
ax6.text(5.0, -36, 'probe parasitic\ndominates S11', color='orange',
         fontsize=9, ha='center', alpha=0.8)

ax6.set_xlabel(r'Frequency $f$ (GHz)')
ax6.set_ylabel(r'$|S|$ (dB)')
ax6.set_title(r'Edge-Fed Microstrip — Full Range (showing QS breakdown)')
ax6.set_xlim(0, 12)
ax6.set_ylim(-40, 2)
ax6.legend(loc='center left', fontsize=10)
ax6.grid(True, alpha=0.3)

fig6.tight_layout()
fig6.savefig('edge_port_sparams_full.png', dpi=200)
print("Saved edge_port_sparams_full.png")

plt.show()
