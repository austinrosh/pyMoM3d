"""Tests for inductor parameter extraction pipelines.

Validates that PEEC extraction gives consistent, frequency-flat inductance
for a simple trace geometry, and that Y-parameter extraction formulas are
consistent with Z-parameter extraction.
"""

import numpy as np
import pytest

from pyMoM3d.analysis.inductor import (
    wheeler_inductance,
    inductance_from_z,
    inductance_from_y,
    quality_factor,
    quality_factor_y,
)


class TestWheelerConsistency:
    """Wheeler formula sanity checks."""

    def test_wheeler_positive(self):
        """Wheeler formula returns positive inductance."""
        L = wheeler_inductance(n=2.5, d_out=2e-3, d_in=1e-3)
        assert L > 0

    def test_wheeler_increases_with_turns(self):
        """More turns -> more inductance."""
        L1 = wheeler_inductance(n=1.5, d_out=2e-3, d_in=1.4e-3)
        L2 = wheeler_inductance(n=2.5, d_out=2e-3, d_in=1e-3)
        assert L2 > L1

    def test_wheeler_increases_with_size(self):
        """Larger spiral -> more inductance."""
        L1 = wheeler_inductance(n=2.5, d_out=1e-3, d_in=0.5e-3)
        L2 = wheeler_inductance(n=2.5, d_out=2e-3, d_in=1e-3)
        assert L2 > L1


class TestExtractionConsistency:
    """Z-parameter and Y-parameter extraction should give consistent results."""

    def test_l_from_z_and_y_agree(self):
        """L from Z and L from Y should agree for a purely inductive impedance."""
        freq = 2e9
        L_ref = 10e-9  # 10 nH
        omega = 2 * np.pi * freq
        R = 2.0  # small loss

        Z = R + 1j * omega * L_ref
        Y = 1.0 / Z

        L_z = inductance_from_z(Z, freq)
        L_y = inductance_from_y(Y, freq)

        assert abs(L_z - L_ref) / L_ref < 0.01, f"L_z={L_z*1e9:.3f} nH"
        assert abs(L_y - L_ref) / L_ref < 0.05, f"L_y={L_y*1e9:.3f} nH"

    def test_q_from_z_and_y_agree(self):
        """Q from Z and Y should agree for a series RL impedance."""
        freq = 2e9
        L = 10e-9
        R = 5.0
        omega = 2 * np.pi * freq

        Z = R + 1j * omega * L
        Y = 1.0 / Z

        Q_z = quality_factor(Z)
        Q_y = quality_factor_y(Y)

        # Both should be omega*L/R
        Q_expected = omega * L / R
        assert abs(Q_z - Q_expected) / Q_expected < 0.01
        assert abs(Q_y - Q_expected) / Q_expected < 0.01

    def test_l_frequency_independent_for_ideal(self):
        """L extraction from ideal Z = R + jwL should be constant across freq."""
        L_ref = 15e-9
        R = 3.0
        freqs = np.linspace(1e9, 10e9, 20)

        L_extracted = []
        for f in freqs:
            omega = 2 * np.pi * f
            Z = R + 1j * omega * L_ref
            L_extracted.append(inductance_from_z(Z, f))

        L_arr = np.array(L_extracted)
        assert np.all(np.abs(L_arr - L_ref) / L_ref < 1e-10)


class TestPEECInductance:
    """PEEC partial inductance extraction tests."""

    def _make_spiral_extractor(self):
        """Create a PEEC extractor for a 2.5-turn spiral."""
        from pyMoM3d.peec.trace import Trace, TraceNetwork, PEECPort
        from pyMoM3d.peec.extractor import PEECExtractor
        from pyMoM3d.mom.surface_impedance import ConductorProperties

        conductor = ConductorProperties(sigma=5.8e7, thickness=2e-6)
        spiral = Trace.rectangular_spiral(
            n_turns=2.5, d_out=2e-3, w_trace=100e-6, s_space=100e-6,
            thickness=2e-6, conductor=conductor,
        )
        port = PEECPort('P1', positive_segment_idx=0)
        network = TraceNetwork([spiral], [port])
        return PEECExtractor(network)

    def test_peec_spiral_positive_inductance(self):
        """PEEC spiral should have positive inductance."""
        ext = self._make_spiral_extractor()
        results = ext.extract([1e9])
        Z = results[0].Z_matrix[0, 0]
        L = Z.imag / (2 * np.pi * 1e9)
        assert L > 0

    def test_peec_spiral_flat_inductance(self):
        """PEEC spiral: L(f) should be flat (< 5% variation)."""
        ext = self._make_spiral_extractor()
        freqs = np.linspace(1e9, 10e9, 10)
        results = ext.extract(freqs.tolist())

        L_arr = np.array([
            r.Z_matrix[0, 0].imag / (2 * np.pi * r.frequency)
            for r in results
        ])

        # Filter positive values
        L_pos = L_arr[L_arr > 0]
        assert len(L_pos) > 5, f"Too few positive L values: {len(L_pos)}"

        # L should be flat: max/min ratio < 1.05
        ratio = np.max(L_pos) / np.min(L_pos)
        assert ratio < 1.05, f"L(f) variation ratio = {ratio:.3f} (want < 1.05)"

    def test_peec_wheeler_order_of_magnitude(self):
        """PEEC spiral L should be within 2x of Wheeler estimate."""
        ext = self._make_spiral_extractor()
        results = ext.extract([1e9])
        L_peec = results[0].Z_matrix[0, 0].imag / (2 * np.pi * 1e9)

        d_in = 2e-3 - 2 * 2.5 * (100e-6 + 100e-6)
        d_in = max(d_in, 100e-6)
        L_wheeler = wheeler_inductance(n=2.5, d_out=2e-3, d_in=d_in)

        # PEEC and Wheeler should agree to within 2x
        ratio = L_peec / L_wheeler
        assert 0.5 < ratio < 2.0, (
            f"PEEC={L_peec*1e9:.2f} nH, Wheeler={L_wheeler*1e9:.2f} nH, "
            f"ratio={ratio:.2f}"
        )

    def test_peec_quality_factor_positive(self):
        """PEEC spiral Q should be positive and increase with frequency."""
        ext = self._make_spiral_extractor()
        freqs = [1e9, 5e9, 10e9]
        results = ext.extract(freqs)

        Q_arr = np.array([quality_factor(r.Z_matrix[0, 0]) for r in results])
        assert np.all(Q_arr > 0), f"Q values: {Q_arr}"
        # Q should increase with frequency for lossless spiral (Q = wL/R)
        assert Q_arr[-1] > Q_arr[0], f"Q should increase: {Q_arr}"

    def test_peec_store_currents(self):
        """PEEC with store_currents=True stores per-segment currents."""
        ext = self._make_spiral_extractor()
        results = ext.extract([1e9], store_currents=True)

        r = results[0]
        assert r.I_solutions is not None, "I_solutions should be stored"
        M = len(ext.segments)
        assert r.I_solutions.shape == (M, 1), (
            f"Expected ({M}, 1), got {r.I_solutions.shape}"
        )
        # Currents should be nonzero
        assert np.max(np.abs(r.I_solutions)) > 0
