"""Tests for PEEC extraction module.

Validates partial inductance, conductor loss, circuit solve, and
end-to-end inductor characterization against analytical formulas.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pyMoM3d import ConductorProperties, mu0, wheeler_inductance
from pyMoM3d.peec import (
    TraceSegment, Trace, TraceNetwork, PEECPort, PEECExtractor,
)
from pyMoM3d.peec.partial_inductance import (
    self_inductance,
    mutual_inductance_filaments,
    partial_inductance_matrix,
    _gmd_rectangular,
)
from pyMoM3d.peec.conductor_loss import segment_resistance, resistance_vector


# Standard copper conductor for tests
COPPER = ConductorProperties(sigma=5.8e7, thickness=2e-6, name='Cu')


# ======================================================================
# Geometry tests
# ======================================================================

class TestTraceSegment:
    def test_length(self):
        seg = TraceSegment(np.array([0, 0, 0]), np.array([1e-3, 0, 0]),
                           100e-6, 2e-6, COPPER)
        assert abs(seg.length - 1e-3) < 1e-15

    def test_direction(self):
        seg = TraceSegment(np.array([0, 0, 0]), np.array([0, 1e-3, 0]),
                           100e-6, 2e-6, COPPER)
        np.testing.assert_allclose(seg.direction, [0, 1, 0], atol=1e-15)

    def test_midpoint(self):
        seg = TraceSegment(np.array([0, 0, 0]), np.array([2e-3, 0, 0]),
                           100e-6, 2e-6, COPPER)
        np.testing.assert_allclose(seg.midpoint, [1e-3, 0, 0])

    def test_cross_section_area(self):
        seg = TraceSegment(np.array([0, 0, 0]), np.array([1e-3, 0, 0]),
                           100e-6, 2e-6, COPPER)
        assert abs(seg.cross_section_area - 200e-12) < 1e-25


class TestTrace:
    def test_from_centerline_basic(self):
        points = np.array([[0, 0, 0], [1e-3, 0, 0]])
        trace = Trace.from_centerline(points, 100e-6, 2e-6, COPPER)
        assert trace.num_segments >= 1
        assert abs(trace.total_length - 1e-3) < 1e-12

    def test_from_centerline_multi_section(self):
        points = np.array([[0, 0, 0], [1e-3, 0, 0], [1e-3, 1e-3, 0]])
        trace = Trace.from_centerline(points, 100e-6, 2e-6, COPPER)
        assert abs(trace.total_length - 2e-3) < 1e-12

    def test_from_centerline_fixed_segments(self):
        points = np.array([[0, 0, 0], [1e-3, 0, 0]])
        trace = Trace.from_centerline(points, 100e-6, 2e-6, COPPER,
                                      segments_per_section=5)
        assert trace.num_segments == 5

    def test_from_centerline_connectivity(self):
        """Segments should be connected end-to-end."""
        points = np.array([[0, 0, 0], [1e-3, 0, 0]])
        trace = Trace.from_centerline(points, 100e-6, 2e-6, COPPER,
                                      segments_per_section=3)
        for i in range(trace.num_segments - 1):
            np.testing.assert_allclose(
                trace.segments[i].end, trace.segments[i + 1].start,
                atol=1e-15,
            )

    def test_from_centerline_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            Trace.from_centerline(np.array([[0, 0, 0]]), 100e-6, 2e-6, COPPER)

    def test_rectangular_spiral(self):
        trace = Trace.rectangular_spiral(
            n_turns=2.5, d_out=2e-3, w_trace=100e-6, s_space=100e-6,
            thickness=2e-6, conductor=COPPER,
        )
        assert trace.num_segments > 0
        assert trace.total_length > 0


class TestTraceNetwork:
    def test_all_segments(self):
        t1 = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]), 100e-6, 2e-6, COPPER,
            segments_per_section=3, name='t1')
        t2 = Trace.from_centerline(
            np.array([[0, 1e-3, 0], [1e-3, 1e-3, 0]]), 100e-6, 2e-6, COPPER,
            segments_per_section=2, name='t2')
        net = TraceNetwork([t1, t2], [])
        assert net.num_segments == 5

    def test_build_connectivity(self):
        t1 = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]), 100e-6, 2e-6, COPPER,
            segments_per_section=3, name='t1')
        net = TraceNetwork([t1], [])
        conn, num_nodes = net.build_connectivity()
        assert conn.shape == (3, 2)
        assert num_nodes == 4  # 3 segments = 4 nodes
        # Check connectivity: each segment's end node = next segment's start
        for i in range(len(conn) - 1):
            assert conn[i, 1] == conn[i + 1, 0]


# ======================================================================
# Partial inductance tests
# ======================================================================

class TestGMD:
    def test_square_cross_section(self):
        """GMD of a square is known: GMD/a ≈ 0.3378 (Grover exact)."""
        gmd = _gmd_rectangular(100e-6, 100e-6)
        ratio = gmd / 100e-6
        # Exact value from Grover formula: exp(ln(sqrt(2)) - 1/2 - pi/3 + ...)
        assert abs(ratio - 0.3379) < 0.01

    def test_thin_strip(self):
        """GMD of thin strip w >> t is ~ 0.2235*(w+t)."""
        gmd = _gmd_rectangular(100e-6, 1e-6)
        assert gmd > 0
        assert gmd < 100e-6  # Must be less than the width


class TestSelfInductance:
    def test_positive(self):
        seg = TraceSegment(np.array([0, 0, 0]), np.array([1e-3, 0, 0]),
                           100e-6, 2e-6, COPPER)
        L = self_inductance(seg)
        assert L > 0

    def test_scales_with_length(self):
        """Longer segment should have more inductance."""
        seg1 = TraceSegment(np.array([0, 0, 0]), np.array([0.5e-3, 0, 0]),
                            100e-6, 2e-6, COPPER)
        seg2 = TraceSegment(np.array([0, 0, 0]), np.array([1e-3, 0, 0]),
                            100e-6, 2e-6, COPPER)
        assert self_inductance(seg2) > self_inductance(seg1)

    def test_zero_length(self):
        seg = TraceSegment(np.array([0, 0, 0]), np.array([0, 0, 0]),
                           100e-6, 2e-6, COPPER)
        assert self_inductance(seg) == 0.0


class TestMutualInductance:
    def test_parallel_filaments_vs_analytical(self):
        """Validate against Neumann closed-form for parallel filaments."""
        l = 1e-3  # 1 mm
        d = 200e-6  # 200 um separation

        seg1 = TraceSegment(np.array([0, 0, 0]), np.array([l, 0, 0]),
                            100e-6, 2e-6, COPPER)
        seg2 = TraceSegment(np.array([0, d, 0]), np.array([l, d, 0]),
                            100e-6, 2e-6, COPPER)

        M_peec = mutual_inductance_filaments(seg1, seg2)

        # Analytical Neumann formula for parallel filaments
        r = l / d
        M_exact = (mu0 * l / (2 * np.pi)) * (
            np.log(r + np.sqrt(1 + r**2)) - np.sqrt(1 + 1/r**2) + 1/r
        )

        assert abs(M_peec - M_exact) / abs(M_exact) < 0.001  # < 0.1%

    def test_perpendicular_segments_zero(self):
        """Perpendicular segments should have zero mutual inductance."""
        seg1 = TraceSegment(np.array([0, 0, 0]), np.array([1e-3, 0, 0]),
                            100e-6, 2e-6, COPPER)
        seg2 = TraceSegment(np.array([0, 0.5e-3, 0]), np.array([0, 1.5e-3, 0]),
                            100e-6, 2e-6, COPPER)
        M = mutual_inductance_filaments(seg1, seg2)
        assert abs(M) < 1e-20

    def test_antiparallel_negative(self):
        """Anti-parallel segments should have negative mutual inductance."""
        l, d = 1e-3, 200e-6
        seg1 = TraceSegment(np.array([0, 0, 0]), np.array([l, 0, 0]),
                            100e-6, 2e-6, COPPER)
        seg2 = TraceSegment(np.array([l, d, 0]), np.array([0, d, 0]),
                            100e-6, 2e-6, COPPER)
        M = mutual_inductance_filaments(seg1, seg2)
        assert M < 0

    def test_symmetry(self):
        """M(i,j) should equal M(j,i)."""
        seg1 = TraceSegment(np.array([0, 0, 0]), np.array([1e-3, 0, 0]),
                            100e-6, 2e-6, COPPER)
        seg2 = TraceSegment(np.array([0.2e-3, 0.3e-3, 0]),
                            np.array([0.8e-3, 0.5e-3, 0]),
                            100e-6, 2e-6, COPPER)
        M_12 = mutual_inductance_filaments(seg1, seg2)
        M_21 = mutual_inductance_filaments(seg2, seg1)
        assert abs(M_12 - M_21) / max(abs(M_12), 1e-30) < 1e-12


class TestPartialInductanceMatrix:
    def test_symmetric(self):
        trace = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]),
            100e-6, 2e-6, COPPER, segments_per_section=5)
        Lp = partial_inductance_matrix(trace.segments)
        np.testing.assert_allclose(Lp, Lp.T, atol=1e-25)

    def test_positive_diagonal(self):
        trace = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]),
            100e-6, 2e-6, COPPER, segments_per_section=5)
        Lp = partial_inductance_matrix(trace.segments)
        assert np.all(np.diag(Lp) > 0)

    def test_shape(self):
        trace = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]),
            100e-6, 2e-6, COPPER, segments_per_section=3)
        Lp = partial_inductance_matrix(trace.segments)
        assert Lp.shape == (3, 3)


# ======================================================================
# Conductor loss tests
# ======================================================================

class TestConductorLoss:
    def test_dc_resistance(self):
        """At DC, R = l / (sigma * w * t)."""
        seg = TraceSegment(np.array([0, 0, 0]), np.array([1e-3, 0, 0]),
                           100e-6, 2e-6, COPPER)
        R_dc = segment_resistance(seg, 0.0)
        R_expected = 1e-3 / (5.8e7 * 100e-6 * 2e-6)
        assert abs(R_dc - R_expected) / R_expected < 1e-10

    def test_resistance_increases_with_frequency(self):
        """Skin effect increases resistance with frequency."""
        seg = TraceSegment(np.array([0, 0, 0]), np.array([1e-3, 0, 0]),
                           100e-6, 2e-6, COPPER)
        R_low = abs(segment_resistance(seg, 1e6))
        R_high = abs(segment_resistance(seg, 10e9))
        assert R_high > R_low

    def test_resistance_vector_shape(self):
        trace = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]),
            100e-6, 2e-6, COPPER, segments_per_section=5)
        R = resistance_vector(trace.segments, 1e9)
        assert R.shape == (5,)
        assert np.all(R.real > 0)


# ======================================================================
# Circuit solve tests
# ======================================================================

class TestPEECCircuit:
    def test_single_segment_impedance(self):
        """Single segment: Z = R + jwL."""
        seg = TraceSegment(np.array([0, 0, 0]), np.array([1e-3, 0, 0]),
                           100e-6, 2e-6, COPPER)
        trace = Trace(name='test', segments=[seg])
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)

        freq = 1e9
        omega = 2 * np.pi * freq
        results = ext.extract([freq])
        Z = results[0].Z_matrix[0, 0]

        # Should be approximately R + jwL
        L_self = self_inductance(seg)
        R = segment_resistance(seg, freq)
        Z_expected = R + 1j * omega * L_self

        # Allow some tolerance due to MNA ground node choice
        assert abs(Z.imag - Z_expected.imag) / abs(Z_expected.imag) < 0.01

    def test_frequency_sweep_returns_network_result(self):
        trace = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]),
            100e-6, 2e-6, COPPER, segments_per_section=3)
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)
        results = ext.extract([1e9, 5e9])
        assert len(results) == 2
        assert results[0].frequency == 1e9
        assert results[1].frequency == 5e9
        assert results[0].Z_matrix.shape == (1, 1)

    def test_network_result_has_y_and_s(self):
        """NetworkResult should provide Y and S matrices."""
        trace = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]),
            100e-6, 2e-6, COPPER, segments_per_section=3)
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)
        r = ext.extract([1e9])[0]
        Y = r.Y_matrix
        S = r.S_matrix
        assert Y.shape == (1, 1)
        assert S.shape == (1, 1)


# ======================================================================
# End-to-end inductor tests
# ======================================================================

class TestInductorCharacterization:
    def test_straight_trace_l_flat(self):
        """L(f) from a straight trace must be flat across frequency."""
        trace = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]),
            100e-6, 2e-6, COPPER, name='straight')
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)

        freqs = np.logspace(8, 10, 10)
        results = ext.extract(freqs)
        Ls = np.array([
            r.Z_matrix[0, 0].imag / (2 * np.pi * r.frequency)
            for r in results
        ])

        # L should vary less than 2%
        variation = (Ls.max() - Ls.min()) / Ls.mean()
        assert variation < 0.02

    def test_loop_l_flat(self):
        """L(f) from a rectangular loop must be flat across frequency."""
        a = 1e-3
        points = np.array([
            [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0], [0, 0, 0]
        ])
        trace = Trace.from_centerline(points, 100e-6, 2e-6, COPPER,
                                      name='loop')
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)

        freqs = np.logspace(8, 10, 10)
        results = ext.extract(freqs)
        Ls = np.array([
            r.Z_matrix[0, 0].imag / (2 * np.pi * r.frequency)
            for r in results
        ])

        variation = (Ls.max() - Ls.min()) / Ls.mean()
        assert variation < 0.02

    def test_spiral_l_positive(self):
        """Spiral inductor PEEC extraction gives positive L."""
        trace = Trace.rectangular_spiral(
            n_turns=2.5, d_out=2e-3, w_trace=100e-6, s_space=100e-6,
            thickness=2e-6, conductor=COPPER,
        )
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)
        r = ext.extract([1e9])[0]
        Z = r.Z_matrix[0, 0]
        L = Z.imag / (2 * np.pi * 1e9)
        assert L > 0

    def test_spiral_l_vs_wheeler_order_of_magnitude(self):
        """PEEC spiral L should be within 2x of Wheeler formula."""
        trace = Trace.rectangular_spiral(
            n_turns=2.5, d_out=2e-3, w_trace=100e-6, s_space=100e-6,
            thickness=2e-6, conductor=COPPER,
        )
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)
        r = ext.extract([1e9])[0]
        L_peec = r.Z_matrix[0, 0].imag / (2 * np.pi * 1e9)

        d_in = 2e-3 - 2 * 2.5 * (100e-6 + 100e-6)
        d_in = max(d_in, 100e-6)
        L_wheeler = wheeler_inductance(2.5, 2e-3, d_in)

        ratio = L_peec / L_wheeler
        assert 0.5 < ratio < 2.0

    def test_spiral_l_flat(self):
        """Spiral L(f) must be flat across frequency."""
        trace = Trace.rectangular_spiral(
            n_turns=2.5, d_out=2e-3, w_trace=100e-6, s_space=100e-6,
            thickness=2e-6, conductor=COPPER,
        )
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)

        freqs = np.logspace(8, 10, 10)
        results = ext.extract(freqs)
        Ls = np.array([
            r.Z_matrix[0, 0].imag / (2 * np.pi * r.frequency)
            for r in results
        ])

        variation = (Ls.max() - Ls.min()) / Ls.mean()
        assert variation < 0.02

    def test_dc_resistance_matches(self):
        """DC resistance from extractor matches R = l/(sigma*w*t)."""
        trace = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]),
            100e-6, 2e-6, COPPER, segments_per_section=5)
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)

        R_expected = 1e-3 / (5.8e7 * 100e-6 * 2e-6)
        assert abs(ext.dc_resistance - R_expected) / R_expected < 1e-10

    def test_total_inductance_estimate(self):
        """Total L estimate should be positive for a trace."""
        trace = Trace.from_centerline(
            np.array([[0, 0, 0], [1e-3, 0, 0]]),
            100e-6, 2e-6, COPPER)
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network)
        assert ext.total_inductance_estimate > 0

    def test_oxide_capacitance_lowers_srf(self):
        """Adding oxide capacitance should create an SRF."""
        from pyMoM3d.analysis.inductor import self_resonant_frequency

        trace = Trace.rectangular_spiral(
            n_turns=1.5, d_out=500e-6, w_trace=20e-6, s_space=20e-6,
            thickness=3e-6, conductor=COPPER,
        )
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])

        # With oxide capacitance
        ext = PEECExtractor(network, oxide_thickness=10e-6, oxide_eps_r=3.9)
        freqs = np.linspace(0.5e9, 30e9, 60)
        results = ext.extract(freqs)

        Y11 = np.array([r.Y_matrix[0, 0] for r in results])
        srf, srf_idx = self_resonant_frequency(freqs, Y11)

        # SRF should exist and be in the expected range
        assert np.isfinite(srf)
        assert 1e9 < srf < 30e9

    def test_oxide_capacitance_q_peak(self):
        """With oxide capacitance, Q should have a peak below SRF."""
        from pyMoM3d import InductorCharacterization

        trace = Trace.rectangular_spiral(
            n_turns=1.5, d_out=500e-6, w_trace=20e-6, s_space=20e-6,
            thickness=3e-6, conductor=COPPER,
        )
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([trace], [port])
        ext = PEECExtractor(network, oxide_thickness=10e-6, oxide_eps_r=3.9)

        freqs = np.linspace(0.5e9, 30e9, 60)
        results = ext.extract(freqs)
        char = InductorCharacterization(results)
        cr = char.characterize()

        # Q_peak should be positive and finite
        assert cr.Q_peak > 0
        assert np.isfinite(cr.Q_peak)
        # Q peak should be at a frequency below SRF
        if np.isfinite(cr.srf):
            assert cr.f_Q_peak < cr.srf
