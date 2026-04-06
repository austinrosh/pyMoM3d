"""Tests for analytical TL de-embedding.

Tests cover:
- ABCD matrix construction for a uniform TL
- Round-trip: embed then de-embed recovers original S-params
- Known analytical cases (lossless through-line)
- Frequency sweep de-embedding
- Integration with CrossSectionResult
"""

import numpy as np
import pytest

from pyMoM3d.network.tl_deembedding import TLDeembedding, tl_abcd
from pyMoM3d.network.soc_deembedding import abcd_to_s, s_to_abcd, invert_abcd


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def lossless_50ohm():
    """A lossless 50 Ohm TL de-embedding setup."""
    Z0 = 50.0
    eps_eff = 4.0
    c0 = 3e8
    v_phase = c0 / np.sqrt(eps_eff)

    def gamma(freq):
        return 1j * 2 * np.pi * freq / v_phase

    return TLDeembedding(Z0=Z0, gamma_func=gamma, d1=5e-3, d2=5e-3)


# ------------------------------------------------------------------ #
# tl_abcd tests
# ------------------------------------------------------------------ #

class TestTLABCD:

    def test_identity_for_zero_length(self):
        """Zero-length TL should give identity ABCD."""
        T = tl_abcd(50.0, 1j * 100.0, 0.0)
        np.testing.assert_allclose(T, np.eye(2), atol=1e-14)

    def test_reciprocal(self):
        """TL ABCD should have det = 1 (reciprocal)."""
        T = tl_abcd(50.0, 0.1 + 1j * 200.0, 0.01)
        det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
        np.testing.assert_allclose(abs(det), 1.0, rtol=1e-12)

    def test_quarter_wave(self):
        """Quarter-wave TL: A = D = 0, B = jZ0, C = j/Z0."""
        Z0 = 75.0
        freq = 10e9
        c0 = 3e8
        lam = c0 / freq
        length = lam / 4
        gamma = 1j * 2 * np.pi * freq / c0

        T = tl_abcd(Z0, gamma, length)
        # A = cos(βl) = cos(π/2) ≈ 0
        assert abs(T[0, 0]) < 1e-10
        # B = jZ0·sin(βl) = jZ0
        np.testing.assert_allclose(T[0, 1], 1j * Z0, rtol=1e-10)
        # D = cos(βl) ≈ 0
        assert abs(T[1, 1]) < 1e-10

    def test_half_wave(self):
        """Half-wave TL should give -identity ABCD."""
        Z0 = 50.0
        freq = 10e9
        c0 = 3e8
        lam = c0 / freq
        length = lam / 2
        gamma = 1j * 2 * np.pi * freq / c0

        T = tl_abcd(Z0, gamma, length)
        np.testing.assert_allclose(T, -np.eye(2), atol=1e-10)

    def test_s21_magnitude_lossless(self):
        """Lossless TL in matched system: |S21| = 1."""
        Z0 = 50.0
        gamma = 1j * 200.0
        T = tl_abcd(Z0, gamma, 0.01)
        S = abcd_to_s(T, Z0=Z0)
        np.testing.assert_allclose(abs(S[0, 1]), 1.0, atol=1e-12)

    def test_lossy_s21_decreases(self):
        """Lossy TL: |S21| < 1."""
        Z0 = 50.0
        gamma = 0.5 + 1j * 200.0  # alpha = 0.5 Np/m
        T = tl_abcd(Z0, gamma, 0.01)
        S = abcd_to_s(T, Z0=Z0)
        assert abs(S[0, 1]) < 1.0


# ------------------------------------------------------------------ #
# Round-trip de-embedding tests
# ------------------------------------------------------------------ #

class TestRoundTrip:

    def test_embed_deembed_identity(self, lossless_50ohm):
        """Embed a DUT with feedlines, then de-embed → recover DUT."""
        # Known DUT: 3 dB attenuator (10 Ohm series, 100 Ohm shunt)
        S_dut = np.array([
            [-0.1 + 0.05j,  0.8 - 0.1j],
            [ 0.8 - 0.1j, -0.15 + 0.03j],
        ])

        freq = 10e9
        deemb = lossless_50ohm

        # Embed: T_total = T_feed1 × T_dut × T_feed2
        T_dut = s_to_abcd(S_dut, deemb.Z0_ref)
        T_feed1 = deemb.feed_abcd(freq, port=1)
        T_feed2 = deemb.feed_abcd(freq, port=2)
        T_total = T_feed1 @ T_dut @ T_feed2
        S_total = abcd_to_s(T_total, deemb.Z0_ref)

        # De-embed
        S_recovered = deemb.deembed(S_total, freq)

        np.testing.assert_allclose(S_recovered, S_dut, atol=1e-12)

    def test_embed_deembed_sweep(self, lossless_50ohm):
        """Round-trip over a frequency sweep."""
        freqs = [2e9, 5e9, 10e9, 15e9, 20e9]
        S_dut = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=complex)
        deemb = lossless_50ohm

        S_embedded = []
        for f in freqs:
            T_dut = s_to_abcd(S_dut, deemb.Z0_ref)
            T_f1 = deemb.feed_abcd(f, 1)
            T_f2 = deemb.feed_abcd(f, 2)
            S_embedded.append(abcd_to_s(T_f1 @ T_dut @ T_f2, deemb.Z0_ref))

        S_recovered = deemb.deembed_sweep(S_embedded, freqs)

        for S_rec in S_recovered:
            np.testing.assert_allclose(S_rec, S_dut, atol=1e-12)

    def test_asymmetric_feeds(self):
        """Different feed lengths at ports 1 and 2."""
        Z0 = 50.0
        c0 = 3e8

        def gamma(f):
            return 1j * 2 * np.pi * f / c0

        deemb = TLDeembedding(Z0=Z0, gamma_func=gamma, d1=3e-3, d2=7e-3)

        S_dut = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)  # through
        freq = 10e9

        T_dut = s_to_abcd(S_dut, 50.0)
        T_f1 = deemb.feed_abcd(freq, 1)
        T_f2 = deemb.feed_abcd(freq, 2)
        S_total = abcd_to_s(T_f1 @ T_dut @ T_f2, 50.0)

        S_rec = deemb.deembed(S_total, freq)
        np.testing.assert_allclose(S_rec, S_dut, atol=1e-12)


# ------------------------------------------------------------------ #
# Through-line analytical test
# ------------------------------------------------------------------ #

class TestThroughLine:

    def test_deembed_through_to_zero_phase(self):
        """De-embedding a through-line should give S21 ≈ 1, S11 ≈ 0."""
        Z0 = 50.0
        c0 = 3e8

        def gamma(f):
            return 1j * 2 * np.pi * f / c0

        # Structure: 5mm feed + 10mm DUT + 5mm feed = 20mm total
        d_feed = 5e-3
        d_total = 20e-3

        freq = 10e9

        # "Measured" S-params: just a 20mm through-line in Z0 system
        T_total = tl_abcd(Z0, gamma(freq), d_total)
        S_measured = abcd_to_s(T_total, Z0)

        # De-embed 5mm feeds
        deemb = TLDeembedding(Z0=Z0, gamma_func=gamma, d1=d_feed, d2=d_feed)
        S_dut = deemb.deembed(S_measured, freq)

        # DUT should be a 10mm through-line
        T_dut_expected = tl_abcd(Z0, gamma(freq), d_total - 2 * d_feed)
        S_expected = abcd_to_s(T_dut_expected, Z0)

        np.testing.assert_allclose(S_dut, S_expected, atol=1e-12)

    def test_deembed_full_length_gives_identity(self):
        """De-embedding the full line length leaves zero-length through."""
        Z0 = 50.0
        c0 = 3e8

        def gamma(f):
            return 1j * 2 * np.pi * f / c0

        d = 10e-3
        freq = 10e9

        T_total = tl_abcd(Z0, gamma(freq), 2 * d)
        S_measured = abcd_to_s(T_total, Z0)

        deemb = TLDeembedding(Z0=Z0, gamma_func=gamma, d1=d, d2=d)
        S_dut = deemb.deembed(S_measured, freq)

        # Zero-length through: S11 = S22 = 0, S21 = S12 = 1
        np.testing.assert_allclose(S_dut[0, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(S_dut[0, 1], 1.0, atol=1e-12)
        np.testing.assert_allclose(S_dut[1, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(S_dut[1, 1], 0.0, atol=1e-12)


# ------------------------------------------------------------------ #
# CrossSectionResult integration test
# ------------------------------------------------------------------ #

class TestCrossSectionIntegration:

    def test_from_cross_section(self):
        """TLDeembedding.from_cross_section should work with a result object."""
        from pyMoM3d.cross_section.extraction import CrossSectionResult

        tl = CrossSectionResult(
            Z0=50.0,
            eps_eff=4.0,
            v_phase=3e8 / np.sqrt(4.0),
            C_pul=1e-10,
            L_pul=1e-10 * 50.0**2,
        )

        deemb = TLDeembedding.from_cross_section(tl, d1=5e-3)
        assert deemb.Z0 == 50.0
        assert deemb.d1 == 5e-3
        assert deemb.d2 == 5e-3

        # Verify gamma works
        gamma = deemb._get_gamma(10e9)
        assert abs(gamma.imag) > 0

    def test_from_cross_section_round_trip(self):
        """Round-trip with CrossSectionResult gamma."""
        from pyMoM3d.cross_section.extraction import CrossSectionResult

        L_pul = 2.5e-7  # H/m
        C_pul = 1e-10    # F/m
        Z0 = np.sqrt(L_pul / C_pul)

        tl = CrossSectionResult(
            Z0=Z0,
            eps_eff=C_pul / (1e-10),
            v_phase=1.0 / np.sqrt(L_pul * C_pul),
            C_pul=C_pul,
            L_pul=L_pul,
        )

        deemb = TLDeembedding.from_cross_section(tl, d1=5e-3)

        S_dut = np.array([[0.1 + 0.2j, 0.7 - 0.1j],
                          [0.7 - 0.1j, 0.05 + 0.1j]])
        freq = 10e9

        # Embed
        T_dut = s_to_abcd(S_dut, deemb.Z0_ref)
        T_f1 = deemb.feed_abcd(freq, 1)
        T_f2 = deemb.feed_abcd(freq, 2)
        S_total = abcd_to_s(T_f1 @ T_dut @ T_f2, deemb.Z0_ref)

        # De-embed
        S_rec = deemb.deembed(S_total, freq)
        np.testing.assert_allclose(S_rec, S_dut, atol=1e-12)


# ------------------------------------------------------------------ #
# Stability tests (no quarter-wave singularity)
# ------------------------------------------------------------------ #

class TestStability:

    def test_no_singularity_at_quarter_wave(self):
        """De-embedding should be stable even when feed = λ/4."""
        Z0 = 50.0
        c0 = 3e8
        freq = 10e9
        lam = c0 / freq
        d_feed = lam / 4  # exactly quarter-wave

        def gamma(f):
            return 1j * 2 * np.pi * f / c0

        deemb = TLDeembedding(Z0=Z0, gamma_func=gamma, d1=d_feed, d2=d_feed)

        # Measured: through-line of 2 × λ/4 = λ/2
        T_total = tl_abcd(Z0, gamma(freq), 2 * d_feed)
        S_measured = abcd_to_s(T_total, Z0)

        # De-embed — this would blow up in SOC but should be fine here
        S_dut = deemb.deembed(S_measured, freq)

        # Should be zero-length through (S21 = 1)
        assert np.all(np.isfinite(S_dut))
        np.testing.assert_allclose(abs(S_dut[0, 1]), 1.0, atol=1e-10)

    def test_stable_across_wideband(self):
        """De-embedding should be stable at all frequencies including λ/4 multiples."""
        Z0 = 50.0
        c0 = 3e8
        d_feed = 15e-3  # 15mm feed (hits λ/4 at ~5 GHz)

        def gamma(f):
            return 1j * 2 * np.pi * f / c0

        deemb = TLDeembedding(Z0=Z0, gamma_func=gamma, d1=d_feed, d2=d_feed)

        # DUT: ideal through
        S_dut_true = np.array([[0, 1], [1, 0]], dtype=complex)

        # Sweep across frequencies including all quarter-wave multiples
        freqs = np.linspace(1e9, 20e9, 40)
        for freq in freqs:
            T_dut = s_to_abcd(S_dut_true, Z0)
            T_f1 = deemb.feed_abcd(freq, 1)
            T_f2 = deemb.feed_abcd(freq, 2)
            S_total = abcd_to_s(T_f1 @ T_dut @ T_f2, Z0)
            S_rec = deemb.deembed(S_total, freq)

            assert np.all(np.isfinite(S_rec)), f"Non-finite at {freq/1e9:.1f} GHz"
            np.testing.assert_allclose(S_rec, S_dut_true, atol=1e-10,
                                       err_msg=f"Failed at {freq/1e9:.1f} GHz")
