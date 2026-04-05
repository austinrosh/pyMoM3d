"""Tests for SOC de-embedding module."""

import numpy as np
import pytest

from pyMoM3d.network.soc_deembedding import (
    abcd_to_s,
    s_to_abcd,
    y_to_abcd,
    abcd_to_y,
    invert_abcd,
)


# -----------------------------------------------------------------------
# ABCD / S / Y conversion round-trip tests
# -----------------------------------------------------------------------

class TestABCDConversions:
    """Verify ABCD ↔ S ↔ Y conversions are self-consistent."""

    def _random_reciprocal_abcd(self, rng):
        """Generate a random reciprocal ABCD (det = 1)."""
        A = rng.standard_normal() + 1j * rng.standard_normal()
        B = rng.standard_normal() + 1j * rng.standard_normal()
        C = rng.standard_normal() + 1j * rng.standard_normal()
        # Force det = 1: D = (1 + BC) / A
        if abs(A) < 1e-10:
            A = 1.0 + 0j
        D = (1.0 + B * C) / A
        return np.array([[A, B], [C, D]], dtype=np.complex128)

    def test_abcd_s_roundtrip(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            T = self._random_reciprocal_abcd(rng)
            S = abcd_to_s(T, Z0=50.0)
            T2 = s_to_abcd(S, Z0=50.0)
            np.testing.assert_allclose(T2, T, atol=1e-12)

    def test_abcd_y_roundtrip(self):
        rng = np.random.default_rng(123)
        for _ in range(20):
            T = self._random_reciprocal_abcd(rng)
            if abs(T[0, 1]) < 1e-10:
                continue  # B=0 makes Y singular
            Y = abcd_to_y(T)
            T2 = y_to_abcd(Y)
            np.testing.assert_allclose(T2, T, atol=1e-12)

    def test_invert_abcd_reciprocal(self):
        rng = np.random.default_rng(7)
        T = self._random_reciprocal_abcd(rng)
        T_inv = invert_abcd(T)
        product = T @ T_inv
        np.testing.assert_allclose(product, np.eye(2), atol=1e-12)

    def test_thru_line_abcd(self):
        """A perfect thru has ABCD = identity."""
        T = np.eye(2, dtype=np.complex128)
        S = abcd_to_s(T, Z0=50.0)
        # S11 = S22 = 0, S21 = S12 = 1
        np.testing.assert_allclose(S[0, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(S[1, 1], 0.0, atol=1e-14)
        np.testing.assert_allclose(S[0, 1], 1.0, atol=1e-14)
        np.testing.assert_allclose(S[1, 0], 1.0, atol=1e-14)

    def test_series_impedance_abcd(self):
        """Series impedance Z_s: T = [[1, Z_s], [0, 1]]."""
        Z_s = 25.0 + 10.0j
        T = np.array([[1, Z_s], [0, 1]], dtype=np.complex128)
        S = abcd_to_s(T, Z0=50.0)
        # S11 = Z_s / (2*Z0 + Z_s)
        expected_S11 = Z_s / (2 * 50.0 + Z_s)
        np.testing.assert_allclose(S[0, 0], expected_S11, atol=1e-14)
        # Reciprocity: S12 = S21
        np.testing.assert_allclose(S[0, 1], S[1, 0], atol=1e-14)

    def test_transmission_line_abcd(self):
        """Lossless TL: A=D=cos(βL), B=jZ_c*sin(βL), C=j*sin(βL)/Z_c."""
        Z_c = 75.0
        beta_L = np.pi / 6  # 30 degrees
        T = np.array([
            [np.cos(beta_L), 1j * Z_c * np.sin(beta_L)],
            [1j * np.sin(beta_L) / Z_c, np.cos(beta_L)],
        ], dtype=np.complex128)
        # det should be 1
        det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
        np.testing.assert_allclose(det, 1.0, atol=1e-14)
        # S-params should be passive (|S| <= 1)
        S = abcd_to_s(T, Z0=50.0)
        assert np.all(np.abs(S) <= 1.0 + 1e-14)


# -----------------------------------------------------------------------
# Synthetic de-embedding test
# -----------------------------------------------------------------------

class TestSyntheticDeembedding:
    """Test ABCD cascade de-embedding with known error boxes and DUT."""

    def test_known_dut_recovery(self):
        """err1 × DUT × err2 → de-embed → recover DUT exactly."""
        Z0 = 50.0

        # Known DUT: series 30 Ω resistor
        T_dut = np.array([[1, 30.0], [0, 1]], dtype=np.complex128)

        # Error box 1: 45-degree TL section, Z_c = 60 Ω
        beta_L = np.pi / 4
        Z_c = 60.0
        T_err1 = np.array([
            [np.cos(beta_L), 1j * Z_c * np.sin(beta_L)],
            [1j * np.sin(beta_L) / Z_c, np.cos(beta_L)],
        ], dtype=np.complex128)

        # Error box 2: 30-degree TL section, Z_c = 40 Ω
        beta_L2 = np.pi / 6
        Z_c2 = 40.0
        T_err2 = np.array([
            [np.cos(beta_L2), 1j * Z_c2 * np.sin(beta_L2)],
            [1j * np.sin(beta_L2) / Z_c2, np.cos(beta_L2)],
        ], dtype=np.complex128)

        # Cascade: T_total = T_err1 @ T_dut @ T_err2
        T_total = T_err1 @ T_dut @ T_err2

        # De-embed
        T_err1_inv = invert_abcd(T_err1)
        T_err2_inv = invert_abcd(T_err2)
        T_recovered = T_err1_inv @ T_total @ T_err2_inv

        np.testing.assert_allclose(T_recovered, T_dut, atol=1e-12)

    def test_symmetric_deembed(self):
        """Symmetric error boxes (same on both sides)."""
        # Error box: 60-degree TL
        beta_L = np.pi / 3
        Z_c = 50.0
        T_err = np.array([
            [np.cos(beta_L), 1j * Z_c * np.sin(beta_L)],
            [1j * np.sin(beta_L) / Z_c, np.cos(beta_L)],
        ], dtype=np.complex128)

        # DUT: shunt admittance (e.g., a stub)
        Y_shunt = 0.01 + 0.005j
        T_dut = np.array([[1, 0], [Y_shunt, 1]], dtype=np.complex128)

        T_total = T_err @ T_dut @ T_err
        T_err_inv = invert_abcd(T_err)
        T_recovered = T_err_inv @ T_total @ T_err_inv

        np.testing.assert_allclose(T_recovered, T_dut, atol=1e-12)

    def test_identity_error_box(self):
        """Identity error boxes should not change the DUT."""
        T_dut = np.array([
            [0.8 + 0.1j, 20.0 + 5j],
            [0.002 + 0.001j, 0.9 - 0.05j],
        ], dtype=np.complex128)
        T_err = np.eye(2, dtype=np.complex128)

        T_total = T_err @ T_dut @ T_err
        T_err_inv = invert_abcd(T_err)
        T_recovered = T_err_inv @ T_total @ T_err_inv

        np.testing.assert_allclose(T_recovered, T_dut, atol=1e-12)

    def test_s_parameter_deembed(self):
        """End-to-end: Y_total → ABCD → de-embed → S_DUT."""
        Z0 = 50.0

        # DUT: perfect thru (S21 = 1, S11 = 0)
        T_dut = np.eye(2, dtype=np.complex128)

        # Error box: 45-degree TL, Z_c = 50 Ω (matched, non-degenerate)
        beta_L = np.pi / 4
        T_err = np.array([
            [np.cos(beta_L), 1j * Z0 * np.sin(beta_L)],
            [1j * np.sin(beta_L) / Z0, np.cos(beta_L)],
        ], dtype=np.complex128)

        T_total = T_err @ T_dut @ T_err

        # Convert to Y (what NetworkExtractor would give)
        Y_total = abcd_to_y(T_total)

        # De-embed: Y → ABCD → remove error boxes → S
        T_total_recovered = y_to_abcd(Y_total)
        T_err_inv = invert_abcd(T_err)
        T_dut_recovered = T_err_inv @ T_total_recovered @ T_err_inv
        S_dut = abcd_to_s(T_dut_recovered, Z0=Z0)

        # DUT should be perfect thru
        np.testing.assert_allclose(abs(S_dut[0, 0]), 0.0, atol=1e-12)
        np.testing.assert_allclose(abs(S_dut[1, 0]), 1.0, atol=1e-12)


# -----------------------------------------------------------------------
# Seam-edge identification test
# -----------------------------------------------------------------------

class TestSeamEdges:
    """Test seam edge identification on combined meshes."""

    def test_seam_edges_found(self):
        """Combined mesh should have identifiable seam edges."""
        from pyMoM3d.mesh.mesh_data import Mesh
        from pyMoM3d.mesh.mirror import mirror_mesh_x, combine_meshes
        from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity
        from pyMoM3d.network.soc_deembedding import _find_seam_edges

        # Simple 4-triangle plate at x=[0, 2], y=[0, 1]
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [0, 1, 0], [1, 1, 0], [2, 1, 0],
        ], dtype=np.float64)
        tris = np.array([
            [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
        ], dtype=np.int32)
        mesh = Mesh(vertices=verts, triangles=tris)
        mirrored = mirror_mesh_x(mesh, x_plane=0.0)
        combined = combine_meshes(mesh, mirrored)
        basis = compute_rwg_connectivity(combined)

        seam_idx, seam_sgn = _find_seam_edges(
            combined, basis, x_ref=0.0, x_orig_side=1.0
        )
        assert len(seam_idx) > 0
        assert len(seam_idx) == len(seam_sgn)
        # All signs should be ±1
        assert all(s in (+1, -1) for s in seam_sgn)

    def test_seam_current_symmetric(self):
        """Symmetric excitation of mirrored plate should produce
        non-zero seam current."""
        from pyMoM3d.mesh.mesh_data import Mesh
        from pyMoM3d.mesh.mirror import mirror_mesh_x, combine_meshes
        from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity
        from pyMoM3d.network.soc_deembedding import (
            _find_seam_edges, _seam_current,
        )

        verts = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [0, 1, 0], [1, 1, 0], [2, 1, 0],
        ], dtype=np.float64)
        tris = np.array([
            [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
        ], dtype=np.int32)
        mesh = Mesh(vertices=verts, triangles=tris)
        mirrored = mirror_mesh_x(mesh, x_plane=0.0)
        combined = combine_meshes(mesh, mirrored)
        basis = compute_rwg_connectivity(combined)

        seam_idx, seam_sgn = _find_seam_edges(
            combined, basis, x_ref=0.0, x_orig_side=1.0
        )

        # With a fake symmetric current vector (all ones)
        I_fake = np.ones(basis.num_basis, dtype=np.complex128)
        I_ref = _seam_current(I_fake, basis, seam_idx, seam_sgn)
        # Should be non-zero
        assert abs(I_ref) > 0
