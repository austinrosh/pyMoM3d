"""Tests for Y11-based transmission line parameter extraction."""

import numpy as np
import pytest

from pyMoM3d.network.tl_extraction import (
    _y11_model,
    extract_tl_from_y11,
    extract_tl_dual_stub,
)
from pyMoM3d.utils.constants import c0


class TestY11Model:
    """Test the analytical Y11 model."""

    def test_y11_tl_open_stub(self):
        """Y11 of a lossless open stub = -j Y0 cot(βL)."""
        Z0 = 50.0
        eps_eff = 2.5
        L = 0.01  # 10 mm
        C_port = 0.0  # no parasitic

        freqs = np.array([1e9, 2e9, 5e9])
        Y11 = _y11_model(freqs, Z0, eps_eff, C_port, L)

        # Manual check
        for i, f in enumerate(freqs):
            omega = 2 * np.pi * f
            beta = omega * np.sqrt(eps_eff) / c0
            Y0 = 1.0 / Z0
            expected = -1j * Y0 * np.cos(beta * L) / np.sin(beta * L)
            np.testing.assert_allclose(Y11[i], expected, rtol=1e-12)

    def test_parasitic_capacitance_adds_j_omega_C(self):
        """Adding C_port shifts Im(Y11) by +jωC_port."""
        Z0 = 50.0
        eps_eff = 2.5
        L = 0.01
        C_port = 1e-12  # 1 pF

        freqs = np.array([1e9, 2e9, 5e9])
        Y11_no_cap = _y11_model(freqs, Z0, eps_eff, 0.0, L)
        Y11_with_cap = _y11_model(freqs, Z0, eps_eff, C_port, L)

        omega = 2 * np.pi * freqs
        expected_shift = 1j * omega * C_port
        np.testing.assert_allclose(
            Y11_with_cap - Y11_no_cap, expected_shift, rtol=1e-12,
        )


class TestExtraction:
    """Test the nonlinear least-squares extraction."""

    def test_exact_recovery_no_noise(self):
        """Extract exact parameters from noise-free synthetic data."""
        Z0_true = 50.0
        eps_eff_true = 2.5
        C_port_true = 1.5e-12  # 1.5 pF
        L = 0.01  # 10 mm

        freqs = np.linspace(0.5e9, 10e9, 20)
        Y11_synth = _y11_model(freqs, Z0_true, eps_eff_true, C_port_true, L)

        result = extract_tl_from_y11(
            freqs, Y11_synth, L,
            Z0_guess=40.0, eps_eff_guess=3.0, C_port_guess_pF=2.0,
        )

        assert abs(result.Z0 - Z0_true) / Z0_true < 1e-6
        assert abs(result.eps_eff - eps_eff_true) / eps_eff_true < 1e-6
        assert abs(result.C_port - C_port_true) / C_port_true < 1e-4
        assert result.residual_norm < 1e-10

    def test_recovery_with_noise(self):
        """Extract parameters from noisy synthetic data within 5%."""
        Z0_true = 75.0
        eps_eff_true = 4.0
        C_port_true = 2e-12
        L = 0.015  # 15 mm

        freqs = np.linspace(1e9, 8e9, 15)
        Y11_synth = _y11_model(freqs, Z0_true, eps_eff_true, C_port_true, L)

        # Add 2% noise
        rng = np.random.default_rng(42)
        noise = 0.02 * np.abs(Y11_synth) * (rng.standard_normal(len(freqs))
                                              + 1j * rng.standard_normal(len(freqs)))
        Y11_noisy = Y11_synth + noise

        result = extract_tl_from_y11(
            freqs, Y11_noisy, L,
            Z0_guess=50.0, eps_eff_guess=3.0,
        )

        assert abs(result.Z0 - Z0_true) / Z0_true < 0.05
        assert abs(result.eps_eff - eps_eff_true) / eps_eff_true < 0.05

    def test_minimum_frequencies(self):
        """Reject fewer than 3 frequencies."""
        with pytest.raises(ValueError, match="at least 3"):
            extract_tl_from_y11(
                np.array([1e9, 2e9]), np.array([1j, 2j]), 0.01,
            )

    def test_different_Z0_values(self):
        """Extraction works for various Z0 values (20-200 Ω)."""
        L = 0.01
        eps_eff = 3.0
        C_port = 1e-12
        freqs = np.linspace(0.5e9, 5e9, 12)

        for Z0_true in [20.0, 50.0, 100.0, 200.0]:
            Y11 = _y11_model(freqs, Z0_true, eps_eff, C_port, L)
            result = extract_tl_from_y11(
                freqs, Y11, L,
                Z0_guess=50.0, eps_eff_guess=2.5,
            )
            assert abs(result.Z0 - Z0_true) / Z0_true < 1e-4, (
                f"Z0={Z0_true}: extracted {result.Z0}"
            )

    def test_zero_parasitic(self):
        """Extraction works when C_port ≈ 0."""
        Z0_true = 50.0
        eps_eff_true = 2.5
        C_port_true = 0.0
        L = 0.01

        freqs = np.linspace(1e9, 5e9, 10)
        Y11 = _y11_model(freqs, Z0_true, eps_eff_true, C_port_true, L)

        result = extract_tl_from_y11(
            freqs, Y11, L,
            Z0_guess=60.0, eps_eff_guess=3.0, C_port_guess_pF=0.5,
        )

        assert abs(result.Z0 - Z0_true) / Z0_true < 1e-4
        assert abs(result.C_port) < 1e-14  # essentially zero


class TestDualStubExtraction:
    """Test differential dual-stub extraction."""

    def test_cancels_port_parasitic(self):
        """Dual-stub correctly recovers Z0/eps_eff despite large C_port."""
        Z0_true = 50.0
        eps_eff_true = 3.0
        C_port = 10e-12  # 10 pF — very large parasitic
        L1 = 0.020  # 20 mm
        L2 = 0.010  # 10 mm

        freqs = np.linspace(1e9, 8e9, 15)
        Y11_L1 = _y11_model(freqs, Z0_true, eps_eff_true, C_port, L1)
        Y11_L2 = _y11_model(freqs, Z0_true, eps_eff_true, C_port, L2)

        result = extract_tl_dual_stub(
            freqs, Y11_L1, Y11_L2, L1, L2,
            Z0_guess=40.0, eps_eff_guess=2.5,
        )

        assert abs(result.Z0 - Z0_true) / Z0_true < 0.01, (
            f"Z0: {result.Z0:.2f} (expected {Z0_true})"
        )
        assert abs(result.eps_eff - eps_eff_true) / eps_eff_true < 0.01, (
            f"eps_eff: {result.eps_eff:.3f} (expected {eps_eff_true})"
        )

    def test_different_z0_values(self):
        """Dual-stub works for various Z0 values."""
        eps_eff = 4.0
        C_port = 5e-12
        L1, L2 = 0.015, 0.008
        freqs = np.linspace(0.5e9, 5e9, 12)

        for Z0_true in [20.0, 50.0, 100.0]:
            Y11_L1 = _y11_model(freqs, Z0_true, eps_eff, C_port, L1)
            Y11_L2 = _y11_model(freqs, Z0_true, eps_eff, C_port, L2)

            result = extract_tl_dual_stub(
                freqs, Y11_L1, Y11_L2, L1, L2,
                Z0_guess=50.0,
            )
            assert abs(result.Z0 - Z0_true) / Z0_true < 0.02, (
                f"Z0={Z0_true}: extracted {result.Z0:.2f}"
            )

    def test_with_noise(self):
        """Dual-stub handles noisy data within 10%."""
        Z0_true = 75.0
        eps_eff_true = 3.5
        C_port = 8e-12
        L1, L2 = 0.020, 0.012
        freqs = np.linspace(1e9, 6e9, 12)

        Y11_L1 = _y11_model(freqs, Z0_true, eps_eff_true, C_port, L1)
        Y11_L2 = _y11_model(freqs, Z0_true, eps_eff_true, C_port, L2)

        rng = np.random.default_rng(42)
        noise_scale = 0.02
        Y11_L1 += noise_scale * np.abs(Y11_L1) * (rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs)))
        Y11_L2 += noise_scale * np.abs(Y11_L2) * (rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs)))

        result = extract_tl_dual_stub(
            freqs, Y11_L1, Y11_L2, L1, L2,
        )

        assert abs(result.Z0 - Z0_true) / Z0_true < 0.10
        assert abs(result.eps_eff - eps_eff_true) / eps_eff_true < 0.10
