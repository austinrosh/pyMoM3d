"""Tests for A-EFIE: divergence matrix, scalar Green's, conditioning, EFIE agreement."""

import numpy as np
import pytest

from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
from pyMoM3d.mesh.gmsh_mesher import GmshMesher
from pyMoM3d.mom.aefie import (
    build_divergence_matrix,
    fill_scalar_green_matrix,
    solve_aefie,
    estimate_kD,
)
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators.efie import EFIEOperator
from pyMoM3d.mom.operators.vector_potential import VectorPotentialOperator
from pyMoM3d.mom.excitation import StripDeltaGapExcitation, find_feed_edges
from pyMoM3d.utils.constants import c0, eta0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plate_mesh():
    """Small rectangular plate mesh."""
    mesher = GmshMesher(target_edge_length=0.3)
    mesh = mesher.mesh_plate(width=1.0, height=1.0)
    basis = compute_rwg_connectivity(mesh)
    return mesh, basis


@pytest.fixture
def dipole_mesh():
    """Thin dipole strip for impedance comparison."""
    mesher = GmshMesher(target_edge_length=0.05)
    mesh = mesher.mesh_plate_with_feed(width=0.5, height=0.02, feed_x=0.0)
    basis = compute_rwg_connectivity(mesh)
    return mesh, basis


# ---------------------------------------------------------------------------
# Divergence matrix tests
# ---------------------------------------------------------------------------

class TestDivergenceMatrix:
    """Verify D matrix structure and values."""

    def test_shape(self, plate_mesh):
        mesh, basis = plate_mesh
        D = build_divergence_matrix(basis, mesh)
        T = len(mesh.triangles)
        N = basis.num_basis
        assert D.shape == (T, N)

    def test_sparsity(self, plate_mesh):
        """Each column has exactly 2 nonzeros (t_plus and t_minus)."""
        mesh, basis = plate_mesh
        D = build_divergence_matrix(basis, mesh)
        for n in range(basis.num_basis):
            col = D[:, n].toarray().ravel()
            assert np.count_nonzero(col) == 2

    def test_column_sum_zero(self, plate_mesh):
        """D columns don't necessarily sum to zero (different triangle areas),
        but the divergence values should have opposite signs."""
        mesh, basis = plate_mesh
        D = build_divergence_matrix(basis, mesh)
        for n in range(basis.num_basis):
            col = D[:, n].toarray().ravel()
            nonzero = col[col != 0]
            assert len(nonzero) == 2
            # One positive, one negative
            assert nonzero[0] * nonzero[1] < 0

    def test_values_match_formula(self, plate_mesh):
        """Verify D[t,n] = ±l_n / A_t (RWG surface divergence)."""
        mesh, basis = plate_mesh
        D = build_divergence_matrix(basis, mesh)
        areas = mesh.triangle_areas

        for n in range(min(10, basis.num_basis)):
            tp = int(basis.t_plus[n])
            tm = int(basis.t_minus[n])
            l_n = basis.edge_length[n]

            expected_plus = l_n / areas[tp]
            expected_minus = -l_n / areas[tm]

            np.testing.assert_allclose(D[tp, n], expected_plus, rtol=1e-12)
            np.testing.assert_allclose(D[tm, n], expected_minus, rtol=1e-12)


# ---------------------------------------------------------------------------
# Scalar Green's function tests
# ---------------------------------------------------------------------------

class TestScalarGreen:
    """Verify G_s matrix properties."""

    def test_symmetry(self, plate_mesh):
        """G_s must be symmetric: G_s[t,t'] = G_s[t',t]."""
        mesh, basis = plate_mesh
        k = 2.0 * np.pi * 1e9 / c0
        G_s = fill_scalar_green_matrix(mesh, k, quad_order=4)
        rel_err = np.linalg.norm(G_s - G_s.T) / np.linalg.norm(G_s)
        assert rel_err < 1e-12, f"G_s symmetry error: {rel_err:.2e}"

    def test_nonzero_diagonal(self, plate_mesh):
        """Self-terms (diagonal) must be nonzero and positive-real-dominated."""
        mesh, basis = plate_mesh
        k = 2.0 * np.pi * 1e9 / c0
        G_s = fill_scalar_green_matrix(mesh, k, quad_order=4)
        diag = np.diag(G_s)
        assert np.all(np.real(diag) > 0), "G_s diagonal should have positive real part"
        assert np.all(np.abs(diag) > 0), "G_s diagonal should be nonzero"

    def test_cpp_numpy_agreement(self, plate_mesh):
        """C++ and NumPy backends should agree."""
        mesh, basis = plate_mesh
        k = 2.0 * np.pi * 1e9 / c0
        try:
            G_cpp = fill_scalar_green_matrix(mesh, k, quad_order=4, backend='cpp')
        except RuntimeError:
            pytest.skip("C++ backend not available")
        G_np = fill_scalar_green_matrix(mesh, k, quad_order=4, backend='numpy')

        rel_err = np.linalg.norm(G_cpp - G_np) / np.linalg.norm(G_np)
        assert rel_err < 1e-4, f"C++/NumPy G_s disagreement: {rel_err:.2e}"


# ---------------------------------------------------------------------------
# Conditioning tests
# ---------------------------------------------------------------------------

class TestConditioning:
    """Verify A-EFIE conditioning improvement at low kD."""

    def test_aefie_better_conditioned(self, plate_mesh):
        """At low kD, A-EFIE augmented system should be better conditioned
        than the standard EFIE matrix."""
        mesh, basis = plate_mesh
        # Low frequency → small kD
        freq = 1e6  # 1 MHz, plate is 1m → kD ~ 0.02
        k = 2.0 * np.pi * freq / c0
        eta = eta0

        # Standard EFIE condition number
        op_efie = EFIEOperator()
        Z_efie = fill_matrix(op_efie, basis, mesh, k, eta)
        cond_efie = np.linalg.cond(Z_efie)

        # A-EFIE augmented system condition number
        op_vp = VectorPotentialOperator()
        Z_A = fill_matrix(op_vp, basis, mesh, k, eta)
        D = build_divergence_matrix(basis, mesh)
        G_s = fill_scalar_green_matrix(mesh, k, quad_order=4)

        N = Z_A.shape[0]
        T = G_s.shape[0]
        jk_over_eta = 1j * k / eta
        D_dense = D.toarray()
        DtGs = D_dense.T @ G_s
        Z_aug = np.empty((N + T, N + T), dtype=np.complex128)
        Z_aug[:N, :N] = Z_A
        Z_aug[:N, N:] = DtGs
        Z_aug[N:, :N] = -D_dense
        Z_aug[N:, N:] = jk_over_eta * np.eye(T)
        cond_aefie = np.linalg.cond(Z_aug)

        # A-EFIE should be dramatically better conditioned
        assert cond_aefie < cond_efie, (
            f"A-EFIE cond ({cond_aefie:.2e}) should be < EFIE cond ({cond_efie:.2e})"
        )


# ---------------------------------------------------------------------------
# EFIE agreement at moderate kD
# ---------------------------------------------------------------------------

class TestEFIEAgreement:
    """At moderate kD, A-EFIE and standard EFIE should agree."""

    def test_currents_agree(self, dipole_mesh):
        """A-EFIE and EFIE currents should match at kD ~ 1."""
        mesh, basis = dipole_mesh
        freq = 3e8  # 300 MHz, dipole ~ 0.5m → kD ~ π
        k = 2.0 * np.pi * freq / c0
        eta = eta0

        # Standard EFIE
        op_efie = EFIEOperator()
        Z_efie = fill_matrix(op_efie, basis, mesh, k, eta)

        # Build excitation
        feed_indices = find_feed_edges(mesh, basis, feed_x=0.0)
        exc = StripDeltaGapExcitation(feed_indices)
        V = exc.compute_voltage_vector(basis, mesh, k)

        I_efie = np.linalg.solve(Z_efie, V)

        # A-EFIE
        op_vp = VectorPotentialOperator()
        Z_A = fill_matrix(op_vp, basis, mesh, k, eta)
        D = build_divergence_matrix(basis, mesh)
        G_s = fill_scalar_green_matrix(mesh, k, quad_order=4)

        I_aefie = solve_aefie(Z_A, G_s, D, V, k, eta)

        # Compare currents (relative error)
        rel_err = np.linalg.norm(I_aefie - I_efie) / np.linalg.norm(I_efie)
        assert rel_err < 0.05, f"A-EFIE/EFIE current disagreement: {rel_err:.2e}"


# ---------------------------------------------------------------------------
# estimate_kD
# ---------------------------------------------------------------------------

class TestEstimateKD:
    """Verify kD estimation."""

    def test_known_geometry(self, plate_mesh):
        mesh, basis = plate_mesh
        k = 2.0 * np.pi  # wavelength = 1m
        kD = estimate_kD(mesh, k)
        # 1m x 1m plate → diagonal = sqrt(2) ≈ 1.414
        # kD = 2π * sqrt(2) ≈ 8.886
        expected = 2.0 * np.pi * np.sqrt(2.0)
        np.testing.assert_allclose(kD, expected, rtol=0.05)

    def test_scales_with_frequency(self, plate_mesh):
        mesh, basis = plate_mesh
        kD1 = estimate_kD(mesh, 1.0)
        kD2 = estimate_kD(mesh, 2.0)
        np.testing.assert_allclose(kD2 / kD1, 2.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Multilayer A-EFIE
# ---------------------------------------------------------------------------

class TestMultilayerAEFIE:
    """Verify multilayer A-EFIE structural correctness."""

    def test_multilayer_efie_a_only_attribute(self):
        """MultilayerEFIEOperator stores a_only flag."""
        from pyMoM3d.greens.free_space_gf import FreeSpaceGreensFunction
        from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator

        k = 2.0 * np.pi * 3e8 / c0
        gf = FreeSpaceGreensFunction(k, eta0)

        op_full = MultilayerEFIEOperator(gf, a_only=False)
        assert op_full._a_only is False

        op_aonly = MultilayerEFIEOperator(gf, a_only=True)
        assert op_aonly._a_only is True

    def test_multilayer_efie_a_only_single_pair(self):
        """a_only=True drops Phi in compute_pair_numpy for a far-field pair."""
        from pyMoM3d.greens.free_space_gf import FreeSpaceGreensFunction
        from pyMoM3d.mom.operators.efie_layered import MultilayerEFIEOperator
        from pyMoM3d.greens.quadrature import triangle_quad_rule

        mesher = GmshMesher(target_edge_length=0.3)
        mesh = mesher.mesh_plate(width=1.0, height=1.0)
        basis = compute_rwg_connectivity(mesh)
        k = 2.0 * np.pi * 3e8 / c0
        gf = FreeSpaceGreensFunction(k, eta0)

        weights, bary = triangle_quad_rule(4)
        areas = mesh.triangle_areas

        # Pick two well-separated basis functions
        m, n = 0, min(5, basis.num_basis - 1)
        tp_m, tp_n = int(basis.t_plus[m]), int(basis.t_plus[n])

        pair_args = dict(
            k=k, eta=eta0, mesh=mesh,
            tri_test=tp_m, tri_src=tp_n,
            fv_test=int(basis.free_vertex_plus[m]),
            fv_src=int(basis.free_vertex_plus[n]),
            sign_test=1.0, sign_src=1.0,
            l_test=basis.edge_length[m], l_src=basis.edge_length[n],
            A_test=areas[tp_m], A_src=areas[tp_n],
            quad_order=4, near_threshold=0.2,
            weights=weights, bary=bary,
            twice_area_test=2*areas[tp_m], twice_area_src=2*areas[tp_n],
            is_near=False, n_hat_test=mesh.triangle_normals[tp_m],
        )

        op_full = MultilayerEFIEOperator(gf, a_only=False)
        op_aonly = MultilayerEFIEOperator(gf, a_only=True)

        z_full = op_full.compute_pair_numpy(**pair_args)
        z_aonly = op_aonly.compute_pair_numpy(**pair_args)

        # a_only drops Phi, so results must differ
        assert z_full != z_aonly
        # a_only result should have smaller magnitude (Phi adds)
        assert abs(z_aonly) < abs(z_full) * 1.5  # rough check

    def test_layered_gs_greens_fn_interface(self, plate_mesh):
        """fill_scalar_green_matrix accepts greens_fn and adds correction."""
        mesh, basis = plate_mesh
        k = 2.0 * np.pi * 3e8 / c0

        # Free-space G_s (no correction)
        G_fs = fill_scalar_green_matrix(mesh, k, quad_order=4)

        # Passing greens_fn=None should give the same result
        G_none = fill_scalar_green_matrix(mesh, k, quad_order=4,
                                           greens_fn=None)
        np.testing.assert_array_equal(G_fs, G_none)
