"""Tests for EDA-grade inductor characterization.

Covers Y-parameter extraction, SRF detection, pi-model fitting,
port parasitic de-embedding, and the InductorCharacterization orchestrator.
"""

import numpy as np
import pytest

from pyMoM3d.analysis.inductor import (
    inductance_from_y,
    inductance_from_z,
    quality_factor_y,
    resistance_from_y,
    self_resonant_frequency,
    differential_inductance_from_y,
    differential_quality_factor_y,
)
from pyMoM3d.analysis.inductor_model import (
    PiModelParams,
    fit_pi_model,
    estimate_initial_params,
)
from pyMoM3d.analysis.inductor_characterization import (
    InductorCharacterization,
    CharacterizationResult,
)
from pyMoM3d.network.network_result import NetworkResult


# ---------------------------------------------------------------------------
# Helpers: construct known circuits
# ---------------------------------------------------------------------------

def _series_rl_impedance(R, L, freq):
    """Z = R + jωL."""
    return R + 1j * 2 * np.pi * freq * L


def _parallel_rlc_impedance(R, L, C, freq):
    """Z of series R-L in parallel with C."""
    omega = 2 * np.pi * freq
    Z_rl = R + 1j * omega * L
    Z_c = 1.0 / (1j * omega * C) if omega > 0 else 1e30
    return (Z_rl * Z_c) / (Z_rl + Z_c)


def _make_network_results(Z11_array, frequencies, Z0=50.0):
    """Create list of NetworkResult from 1-port Z11 values."""
    results = []
    for z, f in zip(Z11_array, frequencies):
        Z_mat = np.array([[z]], dtype=np.complex128)
        results.append(NetworkResult(
            frequency=f,
            Z_matrix=Z_mat,
            port_names=['port1'],
            Z0=Z0,
        ))
    return results


# ---------------------------------------------------------------------------
# Y-parameter extraction tests
# ---------------------------------------------------------------------------

class TestYParameterExtraction:
    """Test Y-parameter based inductance and Q extraction."""

    def test_inductance_from_y_series_rl(self):
        """For Z = R + jωL, Y = 1/Z, verify L recovery."""
        R, L = 5.0, 10e-9  # 5 Ohm, 10 nH
        freq = 2e9
        Z = _series_rl_impedance(R, L, freq)
        Y = 1.0 / Z
        L_ext = inductance_from_y(Y, freq)
        # For pure series RL: L_from_Y should be very close to L
        # (not exact because Y extraction assumes parallel model)
        # But for high Q (ωL >> R), the error is small
        assert abs(L_ext - L) / L < 0.05

    def test_inductance_from_y_with_parallel_cap(self):
        """Y-extraction should give flat L despite parallel capacitance."""
        R, L, C = 2.0, 15e-9, 50e-15
        freqs = np.linspace(0.5e9, 5e9, 20)

        L_from_z = []
        L_from_y = []
        for f in freqs:
            Z = _parallel_rlc_impedance(R, L, C, f)
            Y = 1.0 / Z
            L_from_z.append(inductance_from_z(Z, f))
            L_from_y.append(inductance_from_y(Y, f))

        L_from_z = np.array(L_from_z)
        L_from_y = np.array(L_from_y)

        # Z-based extraction should vary significantly
        z_variation = L_from_z.max() / L_from_z.min()
        # Y-based should be much flatter
        y_variation = L_from_y.max() / L_from_y.min()

        assert y_variation < z_variation, "Y-extraction should be flatter than Z-extraction"
        # Y-based should be close to true L (within 20% for these frequencies)
        assert abs(L_from_y[0] - L) / L < 0.2

    def test_quality_factor_y(self):
        """Q = -Im(Y) / Re(Y) for inductive Y."""
        R, L = 3.0, 10e-9
        freq = 1e9
        omega = 2 * np.pi * freq
        Z = R + 1j * omega * L
        Y = 1.0 / Z
        Q = quality_factor_y(Y)
        Q_expected = omega * L / R
        # Not exact match (Y-based Q differs from simple ωL/R)
        # but should be positive and reasonable
        assert Q > 0
        assert abs(Q - Q_expected) / Q_expected < 0.1

    def test_resistance_from_y(self):
        """R = Re(1/Y) for series RL."""
        R, L = 5.0, 10e-9
        freq = 2e9
        Z = _series_rl_impedance(R, L, freq)
        Y = 1.0 / Z
        R_ext = resistance_from_y(Y)
        assert abs(R_ext - R) < 0.01

    def test_edge_cases(self):
        """Zero frequency and zero Y."""
        assert inductance_from_y(0.0 + 0.0j, 1e9) == 0.0
        assert inductance_from_y(1.0 + 0.0j, 0.0) == 0.0
        assert quality_factor_y(0.0 + 0.0j) == 0.0
        assert resistance_from_y(0.0 + 0.0j) == 0.0


# ---------------------------------------------------------------------------
# SRF detection tests
# ---------------------------------------------------------------------------

class TestSRFDetection:
    """Test self-resonant frequency detection."""

    def test_known_srf(self):
        """Parallel LC with known resonance."""
        L, C = 10e-9, 1e-12
        f_res_exact = 1.0 / (2 * np.pi * np.sqrt(L * C))

        freqs = np.linspace(0.1e9, 10e9, 200)
        Y11 = np.array([1.0 / _parallel_rlc_impedance(1.0, L, C, f) for f in freqs])

        f_srf, idx = self_resonant_frequency(freqs, Y11)
        assert abs(f_srf - f_res_exact) / f_res_exact < 0.02

    def test_no_srf_in_range(self):
        """Pure inductor with no capacitance — no SRF."""
        freqs = np.linspace(0.1e9, 5e9, 50)
        Y11 = np.array([1.0 / _series_rl_impedance(1.0, 10e-9, f) for f in freqs])
        f_srf, idx = self_resonant_frequency(freqs, Y11)
        assert f_srf == float('inf')
        assert idx == -1

    def test_srf_at_boundary(self):
        """SRF very close to last frequency point."""
        L, C = 10e-9, 1e-12
        f_res = 1.0 / (2 * np.pi * np.sqrt(L * C))
        freqs = np.linspace(0.1e9, f_res * 1.01, 100)
        Y11 = np.array([1.0 / _parallel_rlc_impedance(1.0, L, C, f) for f in freqs])
        f_srf, idx = self_resonant_frequency(freqs, Y11)
        assert np.isfinite(f_srf)
        assert abs(f_srf - f_res) / f_res < 0.05


# ---------------------------------------------------------------------------
# Differential extraction tests
# ---------------------------------------------------------------------------

class TestDifferentialExtraction:
    """Test 2-port differential inductance extraction."""

    def test_symmetric_2port(self):
        """Symmetric 2-port Y with known differential L."""
        L = 20e-9
        freq = 1e9
        omega = 2 * np.pi * freq

        # Simple symmetric Y-matrix for coupled inductor
        Y_s = 1.0 / (1j * omega * L)
        Y = np.array([
            [Y_s, -Y_s],
            [-Y_s, Y_s],
        ], dtype=np.complex128)

        L_diff = differential_inductance_from_y(Y, freq)
        # For this simple case, L_diff = 2*L / 4 * 2 = L
        # Y11+Y22-Y12-Y21 = 4*Y_s, L_diff = -2/(omega*Im(4*Y_s))
        L_expected = 2.0 / (omega * 4.0 / (omega * L))
        assert abs(L_diff - L / 2) / (L / 2) < 0.01

    def test_differential_q(self):
        """Differential Q from 2-port Y."""
        R, L = 2.0, 10e-9
        freq = 2e9
        omega = 2 * np.pi * freq
        Z_s = R + 1j * omega * L
        Y_s = 1.0 / Z_s
        Y = np.array([
            [Y_s, -Y_s],
            [-Y_s, Y_s],
        ], dtype=np.complex128)
        Q = differential_quality_factor_y(Y)
        assert Q > 0


# ---------------------------------------------------------------------------
# Pi-model fitting tests
# ---------------------------------------------------------------------------

class TestPiModelFitting:
    """Test broadband pi-model parameter fitting."""

    def test_model_evaluation(self):
        """PiModelParams produces reasonable Y at known frequency."""
        params = PiModelParams(
            L_s=10e-9, R_s=2.0, R_skin=0.0,
            C_s=50e-15, C_ox=100e-15,
            R_sub=200.0, C_sub=50e-15,
        )
        Y = params.Y_model_1port(2e9)
        assert np.isfinite(Y)
        # At 2 GHz with 10 nH, should be inductive (Im(Y) < 0)
        assert Y.imag < 0

    def test_round_trip_fit(self):
        """Generate Y from known params, fit with good initial guess, verify recovery."""
        true_params = PiModelParams(
            L_s=12e-9, R_s=3.0, R_skin=0.0,
            C_s=30e-15, C_ox=80e-15,
            R_sub=300.0, C_sub=40e-15,
        )
        freqs = np.linspace(0.5e9, 8e9, 40)
        Y_data = true_params.Y_model_1port_array(freqs)

        # Provide perturbed initial guess (±30%) to test optimizer convergence
        perturbed = PiModelParams(
            L_s=true_params.L_s * 1.3,
            R_s=true_params.R_s * 0.7,
            R_skin=0.0,
            C_s=true_params.C_s * 1.3,
            C_ox=true_params.C_ox * 0.7,
            R_sub=true_params.R_sub * 1.2,
            C_sub=true_params.C_sub * 0.8,
        )
        fitted, info = fit_pi_model(
            freqs, Y_data, mode='1port', initial_guess=perturbed,
        )
        assert info['success']

        # Check fitted Y matches data well
        Y_fit = info['Y_fitted']
        rel_err = np.linalg.norm(Y_fit - Y_data) / np.linalg.norm(Y_data)
        assert rel_err < 0.05

    def test_model_2port(self):
        """2-port Y-matrix is symmetric and well-formed."""
        params = PiModelParams(
            L_s=10e-9, R_s=2.0, R_skin=0.0,
            C_s=50e-15, C_ox=100e-15,
            R_sub=200.0, C_sub=50e-15,
        )
        Y = params.Y_model_2port(2e9)
        assert Y.shape == (2, 2)
        # Symmetric pi → symmetric Y
        assert abs(Y[0, 1] - Y[1, 0]) < 1e-15
        assert abs(Y[0, 0] - Y[1, 1]) < 1e-15

    def test_initial_guess_estimation(self):
        """estimate_initial_params gives physically reasonable values."""
        params = PiModelParams(
            L_s=10e-9, R_s=2.0, R_skin=0.0,
            C_s=50e-15, C_ox=100e-15,
            R_sub=200.0, C_sub=50e-15,
        )
        freqs = np.linspace(0.5e9, 8e9, 40)
        Y_data = params.Y_model_1port_array(freqs)

        guess = estimate_initial_params(freqs, Y_data)
        # L should be in the right ballpark
        assert 1e-10 < guess.L_s < 1e-7
        assert guess.R_s > 0
        assert guess.C_s > 0

    def test_summary_string(self):
        """PiModelParams.summary() produces readable output."""
        params = PiModelParams(
            L_s=10e-9, R_s=2.0, R_skin=0.0,
            C_s=50e-15, C_ox=100e-15,
            R_sub=200.0, C_sub=50e-15,
        )
        s = params.summary()
        assert 'nH' in s
        assert 'fF' in s
        assert 'Ohm' in s


# ---------------------------------------------------------------------------
# De-embedding tests
# ---------------------------------------------------------------------------

class TestDeembedding:
    """Test correct_port_parasitics de-embedding."""

    def test_round_trip_1port(self):
        """Add then remove parasitics recovers original Z."""
        Z_orig = np.array([[50.0 + 30j]], dtype=np.complex128)

        # Add parasitics: series Z and shunt Y
        Z_series = 2.0 + 1j * 5.0
        Y_shunt = 1j * 2e-3

        # Forward embedding: add series Z first, then shunt Y
        Z_with = Z_orig.copy()
        Z_with[0, 0] += Z_series
        Y_with = np.linalg.inv(Z_with)
        Y_with[0, 0] += Y_shunt
        Z_measured = np.linalg.inv(Y_with)
        nr_meas = NetworkResult(
            frequency=1e9,
            Z_matrix=Z_measured,
            port_names=['p1'],
        )

        # De-embed (reverse: remove shunt Y first, then series Z)
        nr_deemb = nr_meas.correct_port_parasitics(
            series_Z=[Z_series],
            shunt_Y=[Y_shunt],
        )

        np.testing.assert_allclose(
            nr_deemb.Z_matrix, Z_orig, rtol=1e-10,
        )

    def test_round_trip_2port(self):
        """2-port de-embedding round-trip."""
        Z_orig = np.array([
            [50 + 30j, 10 + 5j],
            [10 + 5j, 45 + 25j],
        ], dtype=np.complex128)

        Z_s = [1.0 + 0.5j, 2.0 + 1.0j]
        Y_sh = [1j * 1e-3, 1j * 2e-3]

        # Forward embedding: add series Z first, then shunt Y
        Z_temp = Z_orig.copy()
        for p in range(2):
            Z_temp[p, p] += Z_s[p]
        Y_temp = np.linalg.inv(Z_temp)
        for p in range(2):
            Y_temp[p, p] += Y_sh[p]
        Z_measured = np.linalg.inv(Y_temp)
        nr_meas = NetworkResult(
            frequency=2e9,
            Z_matrix=Z_measured,
            port_names=['p1', 'p2'],
        )

        # De-embed (reverse order)
        nr_deemb = nr_meas.correct_port_parasitics(
            series_Z=Z_s,
            shunt_Y=Y_sh,
        )

        np.testing.assert_allclose(
            nr_deemb.Z_matrix, Z_orig, rtol=1e-10,
        )

    def test_dimension_mismatch(self):
        """Wrong length series_Z or shunt_Y raises ValueError."""
        nr = NetworkResult(
            frequency=1e9,
            Z_matrix=np.array([[50.0 + 0j]]),
            port_names=['p1'],
        )
        with pytest.raises(ValueError):
            nr.correct_port_parasitics(series_Z=[1, 2], shunt_Y=[1e-3])


# ---------------------------------------------------------------------------
# InductorCharacterization orchestrator tests
# ---------------------------------------------------------------------------

class TestInductorCharacterization:
    """Test the high-level InductorCharacterization class."""

    def test_basic_characterization(self):
        """End-to-end with known series RL circuit."""
        R, L = 3.0, 10e-9
        freqs = np.linspace(0.5e9, 5e9, 20)
        Z_array = np.array([_series_rl_impedance(R, L, f) for f in freqs])
        results = _make_network_results(Z_array, freqs)

        char = InductorCharacterization(results)
        cr = char.characterize()

        assert len(cr.params) == 20
        assert cr.frequencies[0] == freqs[0]
        # L_dc should be close to true L
        assert abs(cr.L_dc - L) / L < 0.1
        # No SRF for pure RL
        assert cr.srf == float('inf')
        # Q should be positive
        assert cr.Q_peak > 0

    def test_with_parallel_cap(self):
        """Inductor with parallel cap: Y-extraction flatter below SRF.

        The Y-extraction advantage applies when the dominant parasitic is
        a capacitor in PARALLEL with the inductor (inter-turn cap).
        Z = (R + jωL) || (1/jωC_p), so Y = 1/(R+jωL) + jωC_p.
        The parallel C_p enters Im(Y) additively, giving nearly flat
        L_from_Y at mid-band frequencies.
        """
        R, L, C_p = 2.0, 10e-9, 100e-15  # 10 nH with 100 fF parallel cap
        # Stay well below SRF for comparison (SRF ~ 5 GHz)
        freqs = np.linspace(1e9, 4e9, 30)  # mid-band, below SRF
        Z_array = np.array([_parallel_rlc_impedance(R, L, C_p, f) for f in freqs])
        results = _make_network_results(Z_array, freqs)

        char = InductorCharacterization(results)
        cr = char.characterize()

        # Z-based L varies due to capacitive loading
        L_z = cr.L_z
        L_y = cr.L

        z_var = L_z.max() / L_z.min() if L_z.min() > 0 else float('inf')
        y_var = L_y.max() / L_y.min() if L_y.min() > 0 else float('inf')
        assert y_var < z_var, (
            f"Y-extraction variation ({y_var:.2f}x) should be less than "
            f"Z-extraction variation ({z_var:.2f}x)"
        )

    def test_with_model_fitting(self):
        """Characterization with pi-model fitting."""
        true_params = PiModelParams(
            L_s=10e-9, R_s=2.0, R_skin=0.0,
            C_s=40e-15, C_ox=80e-15,
            R_sub=200.0, C_sub=40e-15,
        )
        freqs = np.linspace(0.5e9, 8e9, 30)
        Y_data = true_params.Y_model_1port_array(freqs)
        Z_data = 1.0 / Y_data
        results = _make_network_results(Z_data, freqs)

        # Provide perturbed initial guess for reliable convergence
        perturbed = PiModelParams(
            L_s=true_params.L_s * 1.2,
            R_s=true_params.R_s * 0.8,
            R_skin=0.0,
            C_s=true_params.C_s * 1.2,
            C_ox=true_params.C_ox * 0.8,
            R_sub=true_params.R_sub * 1.2,
            C_sub=true_params.C_sub * 0.8,
        )
        char = InductorCharacterization(results)
        cr = char.characterize(
            fit_model=True,
            model_kwargs={'initial_guess': perturbed},
        )

        assert cr.pi_model is not None
        assert cr.pi_model_fit_info is not None
        assert cr.pi_model_fit_info['success']
        # Fitted Y should match data well
        Y_fit = cr.pi_model_fit_info['Y_fitted']
        Y_meas = cr.Y11
        rel_err = np.linalg.norm(Y_fit - Y_meas) / np.linalg.norm(Y_meas)
        assert rel_err < 0.1

    def test_summary(self):
        """CharacterizationResult.summary() produces readable output."""
        R, L = 3.0, 10e-9
        freqs = np.linspace(0.5e9, 5e9, 10)
        Z_array = np.array([_series_rl_impedance(R, L, f) for f in freqs])
        results = _make_network_results(Z_array, freqs)

        char = InductorCharacterization(results)
        cr = char.characterize()
        s = cr.summary()
        assert 'nH' in s
        assert 'GHz' in s

    def test_compare_with_wheeler(self):
        """Wheeler comparison returns reasonable dict."""
        R, L = 3.0, 10e-9
        freqs = np.linspace(0.5e9, 5e9, 10)
        Z_array = np.array([_series_rl_impedance(R, L, f) for f in freqs])
        results = _make_network_results(Z_array, freqs)

        char = InductorCharacterization(results)
        comp = char.compare_with_wheeler(n=2.5, d_out=2e-3, d_in=0.6e-3)
        assert 'wheeler_L' in comp
        assert 'L_dc' in comp
        assert 'error_pct' in comp
        assert comp['wheeler_L'] > 0

    def test_empty_results_raises(self):
        """Empty results list raises ValueError."""
        with pytest.raises(ValueError):
            InductorCharacterization([])

    def test_convenience_arrays(self):
        """CharacterizationResult array properties match params."""
        R, L = 3.0, 10e-9
        freqs = np.linspace(0.5e9, 2e9, 5)
        Z_array = np.array([_series_rl_impedance(R, L, f) for f in freqs])
        results = _make_network_results(Z_array, freqs)

        char = InductorCharacterization(results)
        cr = char.characterize()

        assert len(cr.L) == 5
        assert len(cr.Q) == 5
        assert len(cr.R) == 5
        assert len(cr.Z11) == 5
        assert len(cr.Y11) == 5
        assert len(cr.S11) == 5
        assert cr.L[0] == cr.params[0].L
