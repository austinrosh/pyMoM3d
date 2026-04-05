"""Test effect of z_src position on C(eps_r)/C(air) ratio.

The DCIM is fitted at z_src, but actual mesh is at z=H. If z_src is
at the midpoint H/2, the DCIM images are wrong for the interface.
Sweep z_src from H/2 to H to find the optimal position.
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

EPS_R = 4.4
H = 1.6e-3
W = 3.06e-3
L = 10e-3
TEL = 0.5e-3
FREQ = 2.0e9

Z0_ref, eps_eff_ref = microstrip_z0_hammerstad(W, H, EPS_R)
print(f"Hammerstad: Z0 = {Z0_ref:.2f}, eps_eff = {eps_eff_ref:.3f}")

# Pre-build meshes (same for all z_src)
mesher = GmshMesher(target_edge_length=TEL)
mesh = mesher.mesh_plate(width=L, height=W, center=(0.0, 0.0, H))
basis = compute_rwg_connectivity(mesh)
port = Port.from_vertex(mesh, basis, vertex_pos=np.array([-L/2, 0.0, H]),
                         name='P1', tol=TEL * 1.5)

print(f"Mesh: {mesh.get_statistics()['num_triangles']} tris, {basis.num_basis} RWG")


def compute_C_with_zsrc(eps_r_sub, z_src_override):
    """Compute C at a specific z_src by monkey-patching the model builder."""
    import math
    from pyMoM3d.greens.layered.strata import StrataBackend

    stack = LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('sub', z_bot=0.0, z_top=H, eps_r=eps_r_sub),
        Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
    ])

    config = SimulationConfig(
        frequency=FREQ, excitation=None,
        source_layer_name='sub', backend='auto', quad_order=4,
        layer_stack=stack,
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # Monkey-patch z_src in the GF
    original_build = StrataBackend._build_model

    def patched_build(self, layer_stack, frequency, source_layer):
        # Copy from original but override z_src
        import math as m
        interior = []
        top_hs = bot_hs = None
        for layer in layer_stack.layers:
            if not m.isfinite(layer.z_top):
                top_hs = layer
            elif not m.isfinite(layer.z_bot):
                bot_hs = layer
            else:
                interior.append(layer)

        layers_data = []
        for lyr in interior:
            eps = complex(lyr.eps_r)
            layers_data.append([
                float(lyr.z_bot), float(lyr.z_top),
                eps.real, eps.imag,
                float(complex(lyr.mu_r).real),
                float(lyr.conductivity), 0.0,
            ])

        if top_hs is not None:
            epsr_top = float(complex(top_hs.eps_r).real)
            mur_top = float(complex(top_hs.mu_r).real)
            sigma_top = float(top_hs.conductivity)
            pec_top = getattr(top_hs, 'is_pec', False)
        else:
            epsr_top, mur_top, sigma_top, pec_top = 1.0, 1.0, 0.0, False

        if bot_hs is not None:
            epsr_bot = float(complex(bot_hs.eps_r).real)
            mur_bot = float(complex(bot_hs.mu_r).real)
            sigma_bot = float(bot_hs.conductivity)
            pec_bot = getattr(bot_hs, 'is_pec', False)
        else:
            epsr_bot, mur_bot, sigma_bot, pec_bot = 1.0, 1.0, 0.0, False

        if self._pec_sigma_workaround:
            if pec_top:
                pec_top = False; sigma_top = 1e8
            if pec_bot:
                pec_bot = False; sigma_bot = 1e8

        z_src = z_src_override  # <-- the override

        omega = 2.0 * m.pi * frequency
        k_src = complex(source_layer.wavenumber(omega))
        eps_r_eff = complex(source_layer.eps_r_eff(omega))

        return self._sk.make_model(
            layers_data,
            epsr_top, mur_top, sigma_top, pec_top,
            epsr_bot, mur_bot, sigma_bot, pec_bot,
            float(frequency),
            z_src, z_src,
            k_src.real, k_src.imag,
            eps_r_eff.real, eps_r_eff.imag,
            self._method, self._rho_max,
        )

    StrataBackend._build_model = patched_build
    try:
        ext = NetworkExtractor(sim, [port])
        result = ext.extract([FREQ])[0]
        Y11 = result.Y_matrix[0, 0]
        C = Y11.imag / (2 * np.pi * FREQ)
    finally:
        StrataBackend._build_model = original_build

    return C


print(f"\n{'z_src/H':>10} {'z_src(um)':>10} {'C_sub(fF)':>11} {'C_air(fF)':>11} "
      f"{'ratio':>8} {'err%':>8}")
print("-" * 65)

for frac in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.999]:
    z_src = H * frac
    try:
        C_sub = compute_C_with_zsrc(EPS_R, z_src)
        C_air = compute_C_with_zsrc(1.0, z_src)
        if C_air > 0 and C_sub > 0:
            ratio = C_sub / C_air
            err = abs(ratio - eps_eff_ref) / eps_eff_ref * 100
        else:
            ratio = err = np.nan
        print(f"  {frac:>8.3f}  {z_src*1e6:>10.1f}  {C_sub*1e15:>11.2f}  {C_air*1e15:>11.2f}  "
              f"{ratio:>8.3f}  {err:>7.1f}")
    except Exception as e:
        print(f"  {frac:>8.3f}  {z_src*1e6:>10.1f}  FAILED: {e}")
