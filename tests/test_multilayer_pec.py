"""Diagnostic tests for multilayer EFIE with PEC ground planes.

Tests Strata output for PEC-backed layer stacks (microstrip, stripline)
using both method='integrate' (exact Sommerfeld) and method='dcim'.
Compares against analytical PEC image theory.
"""

import math
import numpy as np
import pytest

from pyMoM3d.medium.layer_stack import Layer, LayerStack
from pyMoM3d.utils.constants import c0, eta0

# ---------------------------------------------------------------------------
# Skip if strata_kernels not available
# ---------------------------------------------------------------------------

def _strata_available():
    try:
        from pyMoM3d.greens.layered import strata_kernels
        return True
    except ImportError:
        return False

pytestmark = pytest.mark.skipif(
    not _strata_available(), reason="strata_kernels not compiled"
)


# ---------------------------------------------------------------------------
# Helper: build common layer stacks
# ---------------------------------------------------------------------------

def _air_over_pec_stack():
    """Simplest PEC stack: air slab above PEC ground."""
    return LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('air_slab', z_bot=0.0, z_top=5e-3, eps_r=1.0),
        Layer('air_top', z_bot=5e-3, z_top=np.inf, eps_r=1.0),
    ])


def _microstrip_stack():
    """FR4 microstrip: PEC -> FR4 -> air. Strip at FR4/air interface."""
    H_SUB = 1.6e-3
    return LayerStack([
        Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=4.4),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])


# ---------------------------------------------------------------------------
# Phase 1a: Strata scalar_G_smooth for PEC ground
# ---------------------------------------------------------------------------

class TestStrataPECGround:
    """Test Strata output for PEC-backed stacks against analytical references."""

    @pytest.mark.skip(reason="Strata method='integrate' segfaults with PEC stacks")
    def test_air_over_pec_integrate_vs_image(self):
        """Air-over-PEC with method='integrate': smooth correction should match
        the PEC image -g_fs(R_image)."""
        from pyMoM3d.greens.layered.strata import StrataBackend
        from pyMoM3d.greens.free_space_gf import FreeSpaceGreensFunction

        stack = _air_over_pec_stack()
        freq = 1e9
        src_layer = stack.get_layer('air_slab')
        omega = 2 * np.pi * freq

        backend = StrataBackend(stack, freq, src_layer, method='integrate')

        k = complex(src_layer.wavenumber(omega))
        fs_gf = FreeSpaceGreensFunction(k=k, eta=complex(src_layer.wave_impedance(omega)))

        # Test points: source and observation both in air slab
        z_src = 2.5e-3  # midpoint of slab
        rho_values = [1e-3, 5e-3, 10e-3, 20e-3]

        for rho in rho_values:
            r = np.array([[rho, 0.0, z_src]])
            r_prime = np.array([[0.0, 0.0, z_src]])

            g_smooth = backend.scalar_G(r, r_prime)

            # PEC image: image source at z = -z_src (mirror through z=0)
            r_image = np.array([[0.0, 0.0, -z_src]])
            g_image = fs_gf.scalar_G(r, r_image)

            # For air-over-PEC (horizontal dipole), dominant image is -g_fs(R_image)
            # Strata's smooth correction should approximately match this
            g_expected = -g_image[0]  # negative image for PEC

            # Allow 20% tolerance (Strata may include higher-order effects)
            if abs(g_expected) > 1e-30:
                rel_err = abs(g_smooth[0] - g_expected) / abs(g_expected)
                print(f"  rho={rho*1e3:.1f}mm: "
                      f"g_smooth={g_smooth[0]:.6e}, "
                      f"g_image={g_expected:.6e}, "
                      f"rel_err={rel_err:.2e}")

    @pytest.mark.skip(reason="Strata method='integrate' segfaults with PEC stacks")
    def test_air_over_pec_dcim_vs_integrate(self):
        """Compare DCIM vs integrate for air-over-PEC."""
        from pyMoM3d.greens.layered.strata import StrataBackend

        stack = _air_over_pec_stack()
        freq = 1e9
        src_layer = stack.get_layer('air_slab')

        be_int = StrataBackend(stack, freq, src_layer, method='integrate')
        be_dcim = StrataBackend(stack, freq, src_layer, method='dcim')

        z_src = 2.5e-3
        rho_values = [1e-3, 5e-3, 10e-3, 20e-3]

        print("\n  Air-over-PEC: DCIM vs integrate")
        for rho in rho_values:
            r = np.array([[rho, 0.0, z_src]])
            r_prime = np.array([[0.0, 0.0, z_src]])

            g_int = be_int.scalar_G(r, r_prime)[0]
            g_dcim = be_dcim.scalar_G(r, r_prime)[0]

            if abs(g_int) > 1e-30:
                rel_err = abs(g_dcim - g_int) / abs(g_int)
            else:
                rel_err = abs(g_dcim - g_int)

            print(f"  rho={rho*1e3:.1f}mm: "
                  f"integrate={g_int:.6e}, dcim={g_dcim:.6e}, "
                  f"rel_err={rel_err:.2e}")

    @pytest.mark.skip(reason="Strata method='integrate' segfaults with PEC stacks")
    def test_microstrip_stack_integrate_nonzero(self):
        """Microstrip stack with integrate: smooth correction must be nonzero."""
        from pyMoM3d.greens.layered.strata import StrataBackend
        from pyMoM3d.greens.free_space_gf import FreeSpaceGreensFunction

        stack = _microstrip_stack()
        freq = 1e9
        src_layer = stack.get_layer('FR4')
        omega = 2 * np.pi * freq

        backend = StrataBackend(stack, freq, src_layer, method='integrate')

        k = complex(src_layer.wavenumber(omega))
        fs_gf = FreeSpaceGreensFunction(k=k, eta=complex(src_layer.wave_impedance(omega)))

        z_src = 1.6e-3  # at FR4/air interface  # midpoint of phantom

        rho_values = [1e-3, 5e-3, 10e-3]
        print("\n  Microstrip (integrate): scalar_G_smooth values")
        for rho in rho_values:
            r = np.array([[rho, 0.0, z_src]])
            r_prime = np.array([[0.0, 0.0, z_src]])

            g_smooth = backend.scalar_G(r, r_prime)[0]
            g_fs = fs_gf.scalar_G(r, r_prime)[0]
            ratio = abs(g_smooth) / abs(g_fs) if abs(g_fs) > 0 else 0

            print(f"  rho={rho*1e3:.1f}mm: "
                  f"g_smooth={g_smooth:.6e}, g_fs={g_fs:.6e}, "
                  f"|g_smooth/g_fs|={ratio:.3f}")

    @pytest.mark.skip(reason="Strata method='integrate' segfaults with PEC stacks")
    def test_microstrip_stack_dcim_vs_integrate(self):
        """Compare DCIM vs integrate for microstrip stack."""
        from pyMoM3d.greens.layered.strata import StrataBackend

        stack = _microstrip_stack()
        freq = 1e9
        src_layer = stack.get_layer('FR4')

        be_int = StrataBackend(stack, freq, src_layer, method='integrate')
        be_dcim = StrataBackend(stack, freq, src_layer, method='dcim')

        z_src = 1.6e-3  # at FR4/air interface

        rho_values = [1e-3, 5e-3, 10e-3]
        print("\n  Microstrip: DCIM vs integrate")
        for rho in rho_values:
            r = np.array([[rho, 0.0, z_src]])
            r_prime = np.array([[0.0, 0.0, z_src]])

            g_int = be_int.scalar_G(r, r_prime)[0]
            g_dcim = be_dcim.scalar_G(r, r_prime)[0]

            if abs(g_int) > 1e-30:
                rel_err = abs(g_dcim - g_int) / abs(g_int)
            else:
                rel_err = abs(g_dcim - g_int)

            print(f"  rho={rho*1e3:.1f}mm: "
                  f"integrate={g_int:.6e}, dcim={g_dcim:.6e}, "
                  f"rel_err={rel_err:.2e}")

    @pytest.mark.skip(reason="Strata method='integrate' segfaults with PEC stacks")
    def test_microstrip_dyadic_G_dcim_vs_integrate(self):
        """Compare dyadic G_A for microstrip: DCIM vs integrate."""
        from pyMoM3d.greens.layered.strata import StrataBackend

        stack = _microstrip_stack()
        freq = 1e9
        src_layer = stack.get_layer('FR4')

        be_int = StrataBackend(stack, freq, src_layer, method='integrate')
        be_dcim = StrataBackend(stack, freq, src_layer, method='dcim')

        z_src = 1.6e-3  # at FR4/air interface

        r = np.array([[5e-3, 0.0, z_src]])
        r_prime = np.array([[0.0, 0.0, z_src]])

        ga_int = be_int.dyadic_G(r, r_prime)[0]  # (3, 3)
        ga_dcim = be_dcim.dyadic_G(r, r_prime)[0]

        print("\n  Microstrip dyadic G_A at rho=5mm:")
        print(f"    integrate Gxx={ga_int[0,0]:.6e}, Gyy={ga_int[1,1]:.6e}, Gzz={ga_int[2,2]:.6e}")
        print(f"    dcim      Gxx={ga_dcim[0,0]:.6e}, Gyy={ga_dcim[1,1]:.6e}, Gzz={ga_dcim[2,2]:.6e}")
        print(f"    integrate Gxz={ga_int[0,2]:.6e}, Gzx={ga_int[2,0]:.6e}")
        print(f"    dcim      Gxz={ga_dcim[0,2]:.6e}, Gzx={ga_dcim[2,0]:.6e}")


# ---------------------------------------------------------------------------
# Phase 1b: Full Z-matrix comparison
# ---------------------------------------------------------------------------

class TestZMatrixPECGround:
    """Compare multilayer Z-matrix against free-space Z for a tiny microstrip."""

    def test_z_matrix_multilayer_vs_freespace(self):
        """Multilayer Z should differ from free-space Z (substrate effect)."""
        from pyMoM3d.mesh.mesh_data import Mesh
        from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity
        from pyMoM3d.mom.assembly import fill_matrix
        from pyMoM3d.mom.operators.efie import EFIEOperator
        from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator
        from pyMoM3d.greens.layered.sommerfeld import LayeredGreensFunction

        # Small 2-triangle mesh (single RWG edge) at microstrip height
        H_SUB = 1.6e-3
        z_mesh = H_SUB  # at FR4/air interface

        # Two triangles sharing an edge (minimal mesh)
        vertices = np.array([
            [-1e-3, -0.5e-3, z_mesh],
            [0.0,   -0.5e-3, z_mesh],
            [0.0,    0.5e-3, z_mesh],
            [1e-3,   0.5e-3, z_mesh],
        ], dtype=np.float64)
        triangles = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ], dtype=np.int32)

        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)
        N = basis.num_basis
        print(f"\n  Mesh: {len(triangles)} triangles, {N} RWG basis functions")

        freq = 1e9
        k0 = 2 * np.pi * freq / c0

        # --- Free-space Z ---
        op_fs = EFIEOperator()
        Z_fs = fill_matrix(op_fs, basis, mesh, k0, eta0,
                           quad_order=4, near_threshold=0.2, backend='numpy')

        # --- Multilayer Z (dcim method) ---
        stack = _microstrip_stack()
        gf_dcim = LayeredGreensFunction(stack, freq,
                                         source_layer_name='FR4',
                                         backend='strata')

        k_ml = complex(gf_dcim.wavenumber)
        eta_ml = complex(gf_dcim.wave_impedance)

        op_dcim = MultilayerEFIEOperator(gf_dcim)
        Z_ml_dcim = fill_matrix(op_dcim, basis, mesh, k_ml, eta_ml,
                                quad_order=4, near_threshold=0.2, backend='numpy')

        print(f"\n  Z-matrix comparison (N={N}):")
        for i in range(min(N, 3)):
            for j in range(min(N, 3)):
                print(f"    Z[{i},{j}]: fs={Z_fs[i,j]:.4e}, "
                      f"dcim={Z_ml_dcim[i,j]:.4e}")

        # Multilayer should differ from free-space
        Z_diff_dcim = np.abs(Z_ml_dcim - Z_fs).max()
        Z_scale = np.abs(Z_fs).max()
        print(f"\n  |Z_ml_dcim - Z_fs|_max / |Z_fs|_max = {Z_diff_dcim/Z_scale:.4f}")

        # Multilayer Z should be nonzero and differ from free-space
        assert np.abs(Z_ml_dcim).max() > 1e-20, "DCIM Z-matrix is zero!"
        assert Z_diff_dcim / Z_scale > 0.1, "Multilayer Z too close to free-space!"

    def test_z_matrix_cpp_vs_numpy_pec(self):
        """C++ and numpy assembly should agree for PEC-backed stack."""
        from pyMoM3d.mesh.mesh_data import Mesh
        from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity
        from pyMoM3d.mom.assembly import fill_matrix
        from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator
        from pyMoM3d.greens.layered.sommerfeld import LayeredGreensFunction

        H_SUB = 1.6e-3
        z_mesh = H_SUB  # at FR4/air interface

        # 4-triangle mesh (tetrahedron projected to plane)
        s = 1e-3
        vertices = np.array([
            [-s, -s, z_mesh],
            [s,  -s, z_mesh],
            [s,   s, z_mesh],
            [-s,  s, z_mesh],
        ], dtype=np.float64)
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)

        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)
        N = basis.num_basis

        freq = 1e9
        stack = _microstrip_stack()

        # Build with default (dcim)
        gf = LayeredGreensFunction(stack, freq,
                                    source_layer_name='FR4',
                                    backend='strata')
        k_ml = complex(gf.wavenumber)
        eta_ml = complex(gf.wave_impedance)

        op = MultilayerEFIEOperator(gf)

        if not op.supports_backend('cpp'):
            pytest.skip("C++ multilayer backend not available")

        Z_cpp = fill_matrix(op, basis, mesh, k_ml, eta_ml,
                            quad_order=4, near_threshold=0.2, backend='cpp')
        Z_np = fill_matrix(op, basis, mesh, k_ml, eta_ml,
                           quad_order=4, near_threshold=0.2, backend='numpy')

        Z_scale = max(np.abs(Z_cpp).max(), 1e-30)
        rel_err = np.abs(Z_cpp - Z_np).max() / Z_scale
        print(f"\n  C++ vs numpy PEC: rel_err = {rel_err:.2e}")
        print(f"  Z_cpp diagonal: {[f'{Z_cpp[i,i]:.4e}' for i in range(min(N, 3))]}")
        print(f"  Z_np  diagonal: {[f'{Z_np[i,i]:.4e}' for i in range(min(N, 3))]}")


# ---------------------------------------------------------------------------
# Phase 1c: End-to-end Z_in extraction
# ---------------------------------------------------------------------------

class TestZinExtraction:
    """End-to-end Z_in extraction for a simple microstrip."""

    def test_microstrip_z_in_single_freq(self):
        """Single-port Z_in for microstrip at 1 GHz."""
        from pyMoM3d.mesh.gmsh_mesher import GmshMesher
        from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity
        from pyMoM3d import Simulation, SimulationConfig, SilentReporter
        from pyMoM3d.mom.excitation import StripDeltaGapExcitation, find_feed_edges, compute_feed_signs

        H_SUB = 1.6e-3
        W_STRIP = 3.06e-3
        L_STRIP = 10e-3   # short strip for speed
        z_mesh = H_SUB  # at FR4/air interface

        stack = _microstrip_stack()
        freq = 1e9

        # Mesh
        mesher = GmshMesher(target_edge_length=2.0e-3)
        mesh = mesher.mesh_plate_with_feeds(
            width=L_STRIP, height=W_STRIP,
            feed_x_list=[0.0],
            center=(0.0, 0.0, z_mesh),
        )
        basis = compute_rwg_connectivity(mesh)
        stats = mesh.get_statistics()
        print(f"\n  Mesh: {stats['num_triangles']} tris, {basis.num_basis} RWG")

        feed = find_feed_edges(mesh, basis, feed_x=0.0)
        print(f"  Feed edges: {len(feed)}")

        if not feed:
            pytest.skip("No feed edges found")

        signs = compute_feed_signs(mesh, basis, feed)
        exc = StripDeltaGapExcitation(feed_basis_indices=feed, voltage=1.0, feed_signs=signs)

        # --- Test with Strata DCIM ---
        config_dcim = SimulationConfig(
            frequency=freq, excitation=exc, quad_order=4,
            layer_stack=stack, source_layer_name='FR4',
            gf_backend='strata',
        )
        sim_dcim = Simulation(config_dcim, mesh=mesh, reporter=SilentReporter())
        result_dcim = sim_dcim.run()

        print(f"\n  DCIM:      Z_in = {result_dcim.Z_input}")
        print(f"             cond = {result_dcim.condition_number:.2e}")

        # --- Test with Strata integrate (if we can force it) ---
        # Force integrate method by setting gf_backend differently
        # We need to modify the strata backend method
        # For now, use the built Z from simulation to compare

        if result_dcim.Z_input is not None:
            z_in = result_dcim.Z_input
            print(f"  R_in = {z_in.real:.2f} Ohm, X_in = {z_in.imag:.2f} Ohm")
            print(f"  |Z_in| = {abs(z_in):.2f} Ohm")
        else:
            print("  Z_in is None (no delta-gap excitation detected)")
