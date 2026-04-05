"""Compare DCIM vs Sommerfeld Green's function at various rho.

If the DCIM is losing accuracy at moderate rho, the Sommerfeld reference
should show significantly different (larger) values. If they agree, the
fast decay is correct physics.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d import Layer, LayerStack, c0, eta0

EPS_R = 4.4
H = 1.6e-3

stack = LayerStack([
    Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
    Layer('FR4', z_bot=0.0, z_top=H, eps_r=EPS_R),
    Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
])

freq = 2e9
k = 2 * np.pi * freq / c0

# Source and observation both at z = H (strip surface)
z_src = H
z_obs = H

# Test horizontal distances
rhos = [0.5e-3, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 10e-3, 15e-3, 20e-3]

# --- Strata DCIM ---
from pyMoM3d.greens.layered import LayeredGreensFunction

try:
    gf_strata = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='strata')
    has_strata = True
    print("Strata DCIM backend: loaded")
except Exception as e:
    has_strata = False
    print(f"Strata not available: {e}")

# --- Empymod Sommerfeld ---
try:
    gf_somm = LayeredGreensFunction(stack, freq, source_layer_name='FR4', backend='empymod')
    has_somm = True
    print("Empymod Sommerfeld backend: loaded")
except Exception as e:
    has_somm = False
    print(f"Empymod not available: {e}")

# --- Free-space reference ---
from pyMoM3d.greens.free_space import scalar_green as green_fs

print(f"\nfreq = {freq/1e9:.1f} GHz, k = {k:.4f}, eps_r = {EPS_R}")
print(f"z_src = z_obs = {z_src*1e3:.1f} mm (strip surface)")
print(f"\n{'rho (mm)':>10}  {'|G_fs|':>12}  ", end='')
if has_strata:
    print(f"{'|G_strata|':>12}  {'ratio_st/fs':>12}  ", end='')
if has_somm:
    print(f"{'|G_somm|':>12}  {'ratio_so/fs':>12}  ", end='')
if has_strata and has_somm:
    print(f"{'ratio_st/so':>12}  ", end='')
print()
print("-" * 100)

for rho in rhos:
    r_src = np.array([[0.0, 0.0, z_src]])
    r_obs = np.array([[rho, 0.0, z_obs]])

    # Free-space
    g_fs = green_fs(k, r_obs, r_src)[0]

    print(f"  {rho*1e3:>8.1f}  {abs(g_fs):>12.4e}  ", end='')

    if has_strata:
        g_st = gf_strata.backend.scalar_G(r_obs, r_src)[0]
        print(f"{abs(g_st):>12.4e}  {abs(g_st)/abs(g_fs):>12.6f}  ", end='')

    if has_somm:
        g_so = gf_somm.backend.scalar_G(r_obs, r_src)[0]
        print(f"{abs(g_so):>12.4e}  {abs(g_so)/abs(g_fs):>12.6f}  ", end='')

    if has_strata and has_somm:
        print(f"{abs(g_st)/abs(g_so):>12.6f}  ", end='')

    print()

# Also check dyadic G (vector potential)
print(f"\n\n--- Dyadic G_A (vector potential) ---")
print(f"{'rho (mm)':>10}  ", end='')
if has_strata:
    print(f"{'|GA_st_xx|':>12}  ", end='')
if has_somm:
    print(f"{'|GA_so_xx|':>12}  ", end='')
if has_strata and has_somm:
    print(f"{'ratio_st/so':>12}  ", end='')
print()
print("-" * 60)

for rho in rhos:
    r_src = np.array([[0.0, 0.0, z_src]])
    r_obs = np.array([[rho, 0.0, z_obs]])

    print(f"  {rho*1e3:>8.1f}  ", end='')

    if has_strata:
        ga_st = gf_strata.backend.dyadic_G(r_obs, r_src)
        # dyadic_G returns shape depends on backend
        if hasattr(ga_st, 'shape') and ga_st.ndim >= 2:
            ga_st_xx = ga_st[0, 0] if ga_st.ndim == 2 else ga_st[0]
        else:
            ga_st_xx = ga_st[0]
        print(f"{abs(ga_st_xx):>12.4e}  ", end='')

    if has_somm:
        ga_so = gf_somm.backend.dyadic_G(r_obs, r_src)
        if hasattr(ga_so, 'shape') and ga_so.ndim >= 2:
            ga_so_xx = ga_so[0, 0] if ga_so.ndim == 2 else ga_so[0]
        else:
            ga_so_xx = ga_so[0]
        print(f"{abs(ga_so_xx):>12.4e}  ", end='')

    if has_strata and has_somm:
        print(f"{abs(ga_st_xx)/abs(ga_so_xx):>12.6f}  ", end='')

    print()
