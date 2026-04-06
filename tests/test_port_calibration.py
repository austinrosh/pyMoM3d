"""Tests for port discontinuity calibration.

Tests cover:
- Matrix square root (Cayley-Hamilton closed-form)
- Port error extraction from through-line calibration
- Round-trip: embed with port error → calibrate → recover DUT
- Integration with TLDeembedding (combined calibration + de-embedding)
- Frequency sweep calibration
"""

import numpy as np
import pytest

from pyMoM3d.network.port_calibration import (
    PortCalibration, abcd_sqrt, solve_symmetric_fixture,
)
from pyMoM3d.network.tl_deembedding import TLDeembedding, tl_abcd
from pyMoM3d.network.soc_deembedding import abcd_to_s, s_to_abcd, invert_abcd


# ------------------------------------------------------------------ #
# Matrix square root tests
# ------------------------------------------------------------------ #

class TestABCDSqrt:

    def test_identity(self):
        """sqrt(I) = I."""
        I = np.eye(2, dtype=complex)
        np.testing.assert_allclose(abcd_sqrt(I), I, atol=1e-14)

    def test_known_diagonal(self):
        """sqrt of diagonal matrix with known eigenvalues."""
        M = np.diag([4.0 + 0j, 9.0 + 0j])
        sqrtM = abcd_sqrt(M)
        np.testing.assert_allclose(sqrtM @ sqrtM, M, atol=1e-12)

    def test_round_trip(self):
        """sqrt(M) @ sqrt(M) = M for a general 2x2 matrix."""
        M = np.array([[2.0 + 0.1j, 0.3 - 0.2j],
                       [0.1 + 0.05j, 1.8 - 0.1j]])
        sqrtM = abcd_sqrt(M)
        np.testing.assert_allclose(sqrtM @ sqrtM, M, atol=1e-12)

    def test_near_identity(self):
        """sqrt of near-identity matrix is also near-identity."""
        eps = 1e-4
        M = np.eye(2, dtype=complex) + eps * np.array(
            [[0.1j, 0.5], [0.02j, -0.1j]]
        )
        sqrtM = abcd_sqrt(M)
        np.testing.assert_allclose(sqrtM @ sqrtM, M, atol=1e-12)
        # sqrt should also be near identity
        assert np.allclose(sqrtM, np.eye(2), atol=0.1)

    def test_tl_abcd_sqrt(self):
        """sqrt of a half-wave TL ABCD squared should recover half-wave."""
        Z0 = 50.0
        freq = 10e9
        c0 = 3e8
        lam = c0 / freq
        length = lam / 2
        gamma = 1j * 2 * np.pi * freq / c0

        T_half = tl_abcd(Z0, gamma, length)
        M = T_half @ T_half  # full wave
        sqrtM = abcd_sqrt(M)
        # Should recover half-wave (up to branch choice)
        np.testing.assert_allclose(sqrtM @ sqrtM, M, atol=1e-10)

    def test_reciprocal_preserved(self):
        """sqrt of reciprocal (det=1) matrix has det=1."""
        T = tl_abcd(75.0, 0.1 + 1j * 200.0, 0.005)
        M = T @ T
        sqrtM = abcd_sqrt(M)
        det = np.linalg.det(sqrtM)
        np.testing.assert_allclose(abs(det), 1.0, atol=1e-12)


# ------------------------------------------------------------------ #
# Symmetric fixture solver tests
# ------------------------------------------------------------------ #

class TestSolveSymmetricFixture:

    def test_identity_error(self):
        """When T_cal = T_tl, T_err should be identity."""
        Z0 = 50.0
        gamma = 1j * 200.0
        T_tl = tl_abcd(Z0, gamma, 0.01)
        T_err = solve_symmetric_fixture(T_tl, T_tl)
        np.testing.assert_allclose(T_err, np.eye(2), atol=1e-12)

    def test_exact_shunt_c(self):
        """Exactly recover a shunt-C port error."""
        Z0 = 50.0
        gamma = 1j * 200.0
        d = 0.01
        T_tl = tl_abcd(Z0, gamma, d)

        # Port error: shunt capacitance
        Y = 0.01j
        T_err_true = np.array([[1, 0], [Y, 1]], dtype=complex)
        T_cal = T_err_true @ T_tl @ T_err_true

        T_err = solve_symmetric_fixture(T_cal, T_tl)
        np.testing.assert_allclose(T_err, T_err_true, atol=1e-12)

    def test_exact_series_z(self):
        """Exactly recover a series-Z port error."""
        Z0 = 50.0
        gamma = 0.1 + 1j * 200.0
        d = 0.005
        T_tl = tl_abcd(Z0, gamma, d)

        Z_port = 0.5 + 2.0j
        T_err_true = np.array([[1, Z_port], [0, 1]], dtype=complex)
        T_cal = T_err_true @ T_tl @ T_err_true

        T_err = solve_symmetric_fixture(T_cal, T_tl)
        np.testing.assert_allclose(T_err, T_err_true, atol=1e-12)

    def test_exact_general_error(self):
        """Recover a general (non-diagonal) port error."""
        Z0 = 75.0
        gamma = 1j * 300.0
        d = 0.008
        T_tl = tl_abcd(Z0, gamma, d)

        # General reciprocal port error (det=1)
        T_err_true = np.array([
            [1.01 + 0.005j,  0.3 - 0.1j],
            [0.002 + 0.01j,  (1 + 0.002*0.01 + 0.3*0.01) / (1.01 + 0.005j)],
        ], dtype=complex)
        # Force det = 1
        T_err_true[1, 1] = (1 + T_err_true[0, 1] * T_err_true[1, 0]) / T_err_true[0, 0]

        T_cal = T_err_true @ T_tl @ T_err_true
        T_err = solve_symmetric_fixture(T_cal, T_tl)
        np.testing.assert_allclose(T_err, T_err_true, atol=1e-10)


# ------------------------------------------------------------------ #
# Port error extraction tests
# ------------------------------------------------------------------ #

class TestPortErrorExtraction:

    def _make_port_error(self, C_parasitic=1e-15):
        """Create a shunt-capacitance port error ABCD matrix."""
        # Shunt admittance Y = jωC at 10 GHz
        omega = 2 * np.pi * 10e9
        Y = 1j * omega * C_parasitic
        return np.array([[1, 0], [Y, 1]], dtype=complex)

    def test_zero_error_gives_identity(self):
        """When cal matches analytical, port error should be identity."""
        Z0 = 50.0
        c0 = 3e8
        freq = 10e9
        gamma = 1j * 2 * np.pi * freq / c0
        d_cal = 2e-3  # 2mm cal standard

        # "Measured" cal = exact TL (no port error)
        T_tl = tl_abcd(Z0, gamma, d_cal)
        S_cal = abcd_to_s(T_tl, Z0)

        cal = PortCalibration.from_through_standard(
            S_cal, Z0, gamma, d_cal, freq, Z0_ref=Z0
        )

        np.testing.assert_allclose(cal.T_err, np.eye(2), atol=1e-10)

    def test_extract_known_error(self):
        """Extract a known shunt-C port error."""
        Z0 = 50.0
        c0 = 3e8
        freq = 10e9
        gamma = 1j * 2 * np.pi * freq / c0
        d_cal = 1e-3  # 1mm (very short, T_tl ≈ I)

        T_err_true = self._make_port_error(C_parasitic=5e-15)
        T_tl = tl_abcd(Z0, gamma, d_cal)

        # Simulated cal: T_err × T_tl × T_err
        T_cal = T_err_true @ T_tl @ T_err_true
        S_cal = abcd_to_s(T_cal, Z0)

        cal = PortCalibration.from_through_standard(
            S_cal, Z0, gamma, d_cal, freq, Z0_ref=Z0
        )

        # For very short d_cal, extracted T_err should be close to true
        np.testing.assert_allclose(cal.T_err, T_err_true, atol=1e-6)

    def test_extract_series_impedance_error(self):
        """Extract a series-impedance port error (small series R+jX)."""
        Z0 = 50.0
        c0 = 3e8
        freq = 10e9
        gamma = 1j * 2 * np.pi * freq / c0
        d_cal = 0.5e-3  # 0.5mm

        # Series impedance Z_port = 0.5 + j2.0
        Z_port = 0.5 + 2.0j
        T_err_true = np.array([[1, Z_port], [0, 1]], dtype=complex)
        T_tl = tl_abcd(Z0, gamma, d_cal)

        T_cal = T_err_true @ T_tl @ T_err_true
        S_cal = abcd_to_s(T_cal, Z0)

        cal = PortCalibration.from_through_standard(
            S_cal, Z0, gamma, d_cal, freq, Z0_ref=Z0
        )

        np.testing.assert_allclose(cal.T_err, T_err_true, atol=1e-5)


# ------------------------------------------------------------------ #
# Round-trip calibration + de-embedding tests
# ------------------------------------------------------------------ #

class TestRoundTrip:

    def _make_port_error(self, C_parasitic=1e-14, freq=10e9):
        """Shunt-C port error."""
        Y = 1j * 2 * np.pi * freq * C_parasitic
        return np.array([[1, 0], [Y, 1]], dtype=complex)

    def test_calibrate_then_correct(self):
        """Embed with port error, calibrate, correct → recover DUT."""
        Z0 = 50.0
        c0 = 3e8
        freq = 10e9
        gamma = 1j * 2 * np.pi * freq / c0
        d_cal = 1e-3

        T_err = self._make_port_error(C_parasitic=1e-14, freq=freq)

        # DUT: known S-params
        S_dut = np.array([[-0.1 + 0.05j, 0.8 - 0.1j],
                          [0.8 - 0.1j, -0.15 + 0.03j]])

        # Measured DUT: T_err × T_dut × T_err
        T_dut = s_to_abcd(S_dut, Z0)
        T_meas = T_err @ T_dut @ T_err
        S_meas = abcd_to_s(T_meas, Z0)

        # Calibration standard: T_err × T_tl × T_err
        T_tl = tl_abcd(Z0, gamma, d_cal)
        T_cal_sim = T_err @ T_tl @ T_err
        S_cal = abcd_to_s(T_cal_sim, Z0)

        # Extract calibration
        cal = PortCalibration.from_through_standard(
            S_cal, Z0, gamma, d_cal, freq, Z0_ref=Z0
        )

        # Correct DUT
        S_corrected = cal.correct(S_meas, Z0_ref=Z0)

        np.testing.assert_allclose(S_corrected, S_dut, atol=1e-5)

    def test_full_cascade_with_tl_deembedding(self):
        """Full cascade: port error + feeds → calibrate + de-embed → DUT."""
        Z0 = 50.0
        c0 = 3e8
        freq = 10e9

        def gamma_func(f):
            return 1j * 2 * np.pi * f / c0

        gamma = gamma_func(freq)
        d_feed = 5e-3
        d_cal = 1e-3

        T_err = self._make_port_error(C_parasitic=1e-14, freq=freq)

        # DUT S-params
        S_dut = np.array([[0.1 + 0.2j, 0.7 - 0.1j],
                          [0.7 - 0.1j, 0.05 + 0.1j]])

        # Full measured structure: T_err × T_feed1 × T_dut × T_feed2 × T_err
        T_dut = s_to_abcd(S_dut, Z0)
        T_feed1 = tl_abcd(Z0, gamma, d_feed)
        T_feed2 = tl_abcd(Z0, gamma, d_feed)
        T_total = T_err @ T_feed1 @ T_dut @ T_feed2 @ T_err
        S_total = abcd_to_s(T_total, Z0)

        # Calibration standard: T_err × T_tl_cal × T_err
        T_tl_cal = tl_abcd(Z0, gamma, d_cal)
        T_cal = T_err @ T_tl_cal @ T_err
        S_cal = abcd_to_s(T_cal, Z0)

        # Extract port calibration
        cal = PortCalibration.from_through_standard(
            S_cal, Z0, gamma, d_cal, freq, Z0_ref=Z0
        )

        # De-embed with TLDeembedding + port calibration
        deemb = TLDeembedding(Z0=Z0, gamma_func=gamma_func, d1=d_feed, d2=d_feed,
                              Z0_ref=Z0)
        S_recovered = deemb.deembed(S_total, freq, port_cal=cal)

        np.testing.assert_allclose(S_recovered, S_dut, atol=1e-4)

    def test_without_calibration_has_residual(self):
        """Without calibration, de-embedded result has port error residual."""
        Z0 = 50.0
        c0 = 3e8
        freq = 10e9

        def gamma_func(f):
            return 1j * 2 * np.pi * f / c0

        gamma = gamma_func(freq)
        d_feed = 5e-3

        # Significant port error
        T_err = self._make_port_error(C_parasitic=5e-14, freq=freq)

        # DUT: ideal through
        S_dut = np.array([[0, 1], [1, 0]], dtype=complex)

        T_dut = s_to_abcd(S_dut, Z0)
        T_feed1 = tl_abcd(Z0, gamma, d_feed)
        T_feed2 = tl_abcd(Z0, gamma, d_feed)
        T_total = T_err @ T_feed1 @ T_dut @ T_feed2 @ T_err
        S_total = abcd_to_s(T_total, Z0)

        # De-embed feeds only (no calibration)
        deemb = TLDeembedding(Z0=Z0, gamma_func=gamma_func, d1=d_feed, d2=d_feed,
                              Z0_ref=Z0)
        S_no_cal = deemb.deembed(S_total, freq)

        # Should NOT match DUT exactly (port error remains)
        assert not np.allclose(S_no_cal, S_dut, atol=1e-3), \
            "Without calibration, residual port error should be visible"

        # |S21| should be less than 1 (port error introduces mismatch)
        assert abs(S_no_cal[0, 1]) < 1.0


# ------------------------------------------------------------------ #
# Frequency sweep tests
# ------------------------------------------------------------------ #

class TestFrequencySweep:

    def test_sweep_extraction(self):
        """Extract port calibration across frequency sweep."""
        Z0 = 50.0
        c0 = 3e8
        d_cal = 1e-3

        def gamma_func(f):
            return 1j * 2 * np.pi * f / c0

        freqs = [2e9, 5e9, 10e9, 15e9]

        # Generate cal S-params (no port error → should give identity T_err)
        S_cal_list = []
        for f in freqs:
            T_tl = tl_abcd(Z0, gamma_func(f), d_cal)
            S_cal_list.append(abcd_to_s(T_tl, Z0))

        cals = PortCalibration.from_s_params_sweep(
            S_cal_list, Z0, gamma_func, d_cal, freqs, Z0_ref=Z0
        )

        assert len(cals) == len(freqs)
        for cal, f in zip(cals, freqs):
            assert cal.freq == f
            np.testing.assert_allclose(cal.T_err, np.eye(2), atol=1e-10)

    def test_sweep_with_tl_deembedding(self):
        """Full sweep: embed → calibrate + de-embed → recover at all freqs."""
        Z0 = 50.0
        c0 = 3e8
        d_feed = 5e-3
        d_cal = 1e-3

        def gamma_func(f):
            return 1j * 2 * np.pi * f / c0

        freqs = [2e9, 5e9, 10e9, 15e9, 20e9]

        # Fixed DUT
        S_dut = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=complex)

        # Port error varies with frequency (shunt C)
        C_par = 1e-14

        S_measured_list = []
        S_cal_list = []
        for f in freqs:
            gamma = gamma_func(f)
            Y_err = 1j * 2 * np.pi * f * C_par
            T_err = np.array([[1, 0], [Y_err, 1]], dtype=complex)

            # Measured DUT
            T_dut = s_to_abcd(S_dut, Z0)
            T_f1 = tl_abcd(Z0, gamma, d_feed)
            T_f2 = tl_abcd(Z0, gamma, d_feed)
            T_total = T_err @ T_f1 @ T_dut @ T_f2 @ T_err
            S_measured_list.append(abcd_to_s(T_total, Z0))

            # Cal standard
            T_tl_cal = tl_abcd(Z0, gamma, d_cal)
            T_cal = T_err @ T_tl_cal @ T_err
            S_cal_list.append(abcd_to_s(T_cal, Z0))

        # Extract calibrations
        cals = PortCalibration.from_s_params_sweep(
            S_cal_list, Z0, gamma_func, d_cal, freqs, Z0_ref=Z0
        )

        # De-embed sweep
        deemb = TLDeembedding(Z0=Z0, gamma_func=gamma_func, d1=d_feed, d2=d_feed,
                              Z0_ref=Z0)
        S_recovered = deemb.deembed_sweep(S_measured_list, freqs, port_cals=cals)

        for S_rec, f in zip(S_recovered, freqs):
            np.testing.assert_allclose(
                S_rec, S_dut, atol=1e-3,
                err_msg=f"Failed at {f/1e9:.1f} GHz"
            )


# ------------------------------------------------------------------ #
# Stability tests
# ------------------------------------------------------------------ #

class TestStability:

    def test_calibration_at_quarter_wave(self):
        """Calibration should be stable when cal section = λ/4."""
        Z0 = 50.0
        c0 = 3e8
        freq = 10e9
        lam = c0 / freq
        d_cal = lam / 4  # quarter-wave cal section

        gamma = 1j * 2 * np.pi * freq / c0

        # Cal standard: just a quarter-wave TL (no port error)
        T_tl = tl_abcd(Z0, gamma, d_cal)
        S_cal = abcd_to_s(T_tl, Z0)

        cal = PortCalibration.from_through_standard(
            S_cal, Z0, gamma, d_cal, freq, Z0_ref=Z0
        )

        assert np.all(np.isfinite(cal.T_err))
        np.testing.assert_allclose(cal.T_err, np.eye(2), atol=1e-10)

    def test_stable_wideband_calibration(self):
        """Calibration should be stable across all frequencies."""
        Z0 = 50.0
        c0 = 3e8
        d_cal = 2e-3

        def gamma_func(f):
            return 1j * 2 * np.pi * f / c0

        # Include quarter-wave frequencies for d_cal = 2mm
        freqs = np.linspace(1e9, 40e9, 40)

        for freq in freqs:
            gamma = gamma_func(freq)
            T_tl = tl_abcd(Z0, gamma, d_cal)
            S_cal = abcd_to_s(T_tl, Z0)

            cal = PortCalibration.from_through_standard(
                S_cal, Z0, gamma, d_cal, freq, Z0_ref=Z0
            )

            assert np.all(np.isfinite(cal.T_err)), \
                f"Non-finite at {freq/1e9:.1f} GHz"
            np.testing.assert_allclose(
                cal.T_err, np.eye(2), atol=1e-8,
                err_msg=f"Failed at {freq/1e9:.1f} GHz"
            )

    def test_large_port_error_recovery(self):
        """Even large port errors should be extractable and correctable."""
        Z0 = 50.0
        c0 = 3e8
        freq = 10e9
        gamma = 1j * 2 * np.pi * freq / c0
        d_cal = 0.5e-3

        # Large port error: shunt 100 fF at 10 GHz → Y = j6.28 mS
        C_par = 100e-15
        Y_err = 1j * 2 * np.pi * freq * C_par
        T_err = np.array([[1, 0], [Y_err, 1]], dtype=complex)

        # Cal measurement
        T_tl = tl_abcd(Z0, gamma, d_cal)
        T_cal = T_err @ T_tl @ T_err
        S_cal = abcd_to_s(T_cal, Z0)

        cal = PortCalibration.from_through_standard(
            S_cal, Z0, gamma, d_cal, freq, Z0_ref=Z0
        )

        # DUT
        S_dut = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        T_dut = s_to_abcd(S_dut, Z0)
        T_meas = T_err @ T_dut @ T_err
        S_meas = abcd_to_s(T_meas, Z0)

        S_corrected = cal.correct(S_meas, Z0_ref=Z0)

        # Should recover DUT within reasonable tolerance
        np.testing.assert_allclose(S_corrected, S_dut, atol=1e-3)


# ------------------------------------------------------------------ #
# CrossSectionResult integration
# ------------------------------------------------------------------ #

class TestCrossSectionIntegration:

    def test_sweep_with_cross_section_gamma(self):
        """from_s_params_sweep works with CrossSectionResult.gamma()."""
        from pyMoM3d.cross_section.extraction import CrossSectionResult

        tl = CrossSectionResult(
            Z0=50.0,
            eps_eff=4.0,
            v_phase=3e8 / np.sqrt(4.0),
            C_pul=1e-10,
            L_pul=1e-10 * 50.0**2,
        )

        d_cal = 1e-3
        freqs = [5e9, 10e9]

        S_cal_list = []
        for f in freqs:
            T_tl = tl_abcd(tl.Z0, tl.gamma(f), d_cal)
            S_cal_list.append(abcd_to_s(T_tl, tl.Z0))

        cals = PortCalibration.from_s_params_sweep(
            S_cal_list, tl.Z0, tl, d_cal, freqs, Z0_ref=tl.Z0
        )

        assert len(cals) == 2
        for cal in cals:
            np.testing.assert_allclose(cal.T_err, np.eye(2), atol=1e-10)
