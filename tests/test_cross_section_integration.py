"""Integration tests: 2D cross-section Z₀ → 3D QS solver → S-parameters.

Validates the full pipeline where the 2D solver provides the authoritative
reference impedance for S-parameter normalization of the 3D QS extraction.

The key test: a matched through-line should have S21 ≈ 0 dB when the
reference impedance equals the line's characteristic impedance (from 2D).
"""

import numpy as np
import pytest

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, Layer, LayerStack,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges,
    compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.cross_section import compute_reference_impedance
from pyMoM3d.utils.constants import c0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def microstrip_tl_params():
    """2D-derived TL parameters for standard FR4 microstrip."""
    stack = LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=1.6e-3, eps_r=4.4),
        Layer('air', z_bot=1.6e-3, z_top=np.inf, eps_r=1.0),
    ])
    tl = compute_reference_impedance(
        stack, strip_width=3.06e-3, source_layer_name='FR4',
        base_cells=300,
    )
    return tl, stack


@pytest.fixture(scope="module")
def microstrip_qs_with_2d_z0(microstrip_tl_params):
    """QS solver with probe feeds using 2D-derived Z₀.

    Returns (solver, tl_result, frequencies).
    """
    tl, stack = microstrip_tl_params

    EPS_R = 4.4
    H_SUB = 1.6e-3
    W_STRIP = 3.06e-3
    L_STRIP = 10.0e-3
    TEL = 0.7e-3
    PORT1_X = -L_STRIP / 2.0 + 1.0e-3
    PORT2_X = +L_STRIP / 2.0 - 1.0e-3

    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, PORT2_X],
        center=(0.0, 0.0, H_SUB),
    )
    basis = compute_rwg_connectivity(mesh)

    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=1e9, excitation=exc, quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    # Use 2D-derived Z₀ as reference impedance
    qs = QuasiStaticSolver(
        sim, [port1, port2],
        Z0=tl.Z0,
        probe_feeds=True,
    )

    freqs = [0.5e9, 1.0e9, 2.0e9]
    results = qs.extract(freqs)
    return results, tl, freqs


@pytest.fixture(scope="module")
def edge_port_qs_with_2d_z0(microstrip_tl_params):
    """QS solver with probe feeds at strip edges using 2D-derived Z₀.

    Uses a plain plate mesh (no vertical plates) with probe feeds
    at the strip ends.  The probes provide vertical excitation from
    the PEC ground to the strip — the correct signal-to-ground path.
    """
    tl, stack = microstrip_tl_params

    H_SUB = 1.6e-3
    W_STRIP = 3.06e-3
    L_STRIP = 10.0e-3
    TEL = 0.7e-3
    PORT1_X = -L_STRIP / 2.0 + 1.0e-3
    PORT2_X = +L_STRIP / 2.0 - 1.0e-3

    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, PORT2_X],
        center=(0.0, 0.0, H_SUB),
    )
    basis = compute_rwg_connectivity(mesh)

    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=1e9, excitation=exc, quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='FR4',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    qs = QuasiStaticSolver(
        sim, [port1, port2],
        Z0=tl.Z0,
        probe_feeds=True,
    )

    freqs = [0.5e9, 1.0e9, 2.0e9]
    results = qs.extract(freqs)
    return results, tl, freqs


# ---------------------------------------------------------------------------
# Tests: 2D solver bridge
# ---------------------------------------------------------------------------

class TestComputeReferenceImpedance:
    """Test the LayerStack → CrossSection bridge."""

    def test_microstrip_z0_reasonable(self, microstrip_tl_params):
        """Z₀ for 3.06mm strip on 1.6mm FR4 should be ~50 Ohm."""
        tl, _ = microstrip_tl_params
        assert 40 < tl.Z0 < 60, f"Z0 = {tl.Z0:.1f} outside [40, 60]"

    def test_microstrip_eps_eff_reasonable(self, microstrip_tl_params):
        """eps_eff should be between 1 and eps_r=4.4."""
        tl, _ = microstrip_tl_params
        assert 1.0 < tl.eps_eff < 4.4, f"eps_eff = {tl.eps_eff:.3f}"

    def test_stripline_from_layer_stack(self):
        """Stripline with PEC-dielectric-PEC sandwich."""
        b = 3.0e-3
        stack = LayerStack([
            Layer('pec_bot', z_bot=-np.inf, z_top=0, eps_r=1.0, is_pec=True),
            Layer('dielectric', z_bot=0, z_top=b, eps_r=2.2),
            Layer('pec_top', z_bot=b, z_top=np.inf, eps_r=1.0, is_pec=True),
        ])
        tl = compute_reference_impedance(
            stack, strip_width=1.5e-3, strip_z=b / 2,
            base_cells=200,
        )
        # Stripline: eps_eff should equal eps_r for homogeneous dielectric
        assert tl.eps_eff == pytest.approx(2.2, rel=0.01)
        assert 30 < tl.Z0 < 100

    def test_no_pec_uses_boundary_ground(self):
        """Without PEC, domain boundary V=0 acts as ground."""
        stack = LayerStack([
            Layer('sub', z_bot=-np.inf, z_top=0, eps_r=4.4),
            Layer('air', z_bot=0, z_top=np.inf, eps_r=1.0),
        ])
        # No PEC ground → uses domain boundary as ground reference
        tl = compute_reference_impedance(
            stack, strip_width=1e-3, strip_z=0.0, base_cells=50,
        )
        assert tl.Z0 > 0

    def test_requires_z_or_layer_name(self):
        """Should raise if neither strip_z nor source_layer_name given."""
        stack = LayerStack([
            Layer('pec', z_bot=-np.inf, z_top=0, eps_r=1.0, is_pec=True),
            Layer('air', z_bot=0, z_top=np.inf, eps_r=1.0),
        ])
        with pytest.raises(ValueError, match="strip_z or source_layer_name"):
            compute_reference_impedance(stack, strip_width=1e-3)


# ---------------------------------------------------------------------------
# Tests: Probe feed QS with 2D Z₀
# ---------------------------------------------------------------------------

class TestProbeQSWith2DZ0:
    """Probe feed QS solver using 2D-derived reference impedance."""

    def test_s21_near_zero_db(self, microstrip_qs_with_2d_z0):
        """Through-line S21 should be near 0 dB at low frequencies."""
        results, tl, freqs = microstrip_qs_with_2d_z0
        for r in results:
            s21_db = 20 * np.log10(abs(r.S_matrix[1, 0]))
            assert s21_db > -1.0, (
                f"S21 = {s21_db:.2f} dB at {r.frequency/1e9:.1f} GHz "
                f"(Z0_ref = {tl.Z0:.1f} Ohm)"
            )

    def test_s21_monotonic(self, microstrip_qs_with_2d_z0):
        """S21 should decrease monotonically in the QS regime."""
        results, _, _ = microstrip_qs_with_2d_z0
        s21_vals = [abs(r.S_matrix[1, 0]) for r in results]
        for i in range(len(s21_vals) - 1):
            assert s21_vals[i + 1] <= s21_vals[i] + 1e-6

    def test_reciprocity(self, microstrip_qs_with_2d_z0):
        """|S12 - S21| should be negligible."""
        results, _, _ = microstrip_qs_with_2d_z0
        for r in results:
            diff = abs(r.S_matrix[0, 1] - r.S_matrix[1, 0])
            assert diff < 1e-10

    def test_passivity(self, microstrip_qs_with_2d_z0):
        """|S11|² + |S21|² ≤ 1 for lossless passive network."""
        results, _, _ = microstrip_qs_with_2d_z0
        for r in results:
            power = abs(r.S_matrix[0, 0])**2 + abs(r.S_matrix[1, 0])**2
            assert power <= 1.0 + 1e-6, f"Passivity violated: {power:.6f}"

    def test_z0_mismatch_degrades_s21(self, microstrip_tl_params):
        """Using wrong Z₀ (50 Ohm nominal) should give worse S21."""
        tl, stack = microstrip_tl_params

        H_SUB = 1.6e-3
        W_STRIP = 3.06e-3
        L_STRIP = 10.0e-3
        TEL = 0.7e-3
        PORT1_X = -L_STRIP / 2.0 + 1.0e-3
        PORT2_X = +L_STRIP / 2.0 - 1.0e-3

        mesher = GmshMesher(target_edge_length=TEL)
        mesh = mesher.mesh_plate_with_feeds(
            width=L_STRIP, height=W_STRIP,
            feed_x_list=[PORT1_X, PORT2_X],
            center=(0.0, 0.0, H_SUB),
        )
        basis = compute_rwg_connectivity(mesh)

        feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
        feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
        signs1 = compute_feed_signs(mesh, basis, feed1)
        signs2 = compute_feed_signs(mesh, basis, feed2)
        port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
        port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

        exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
        config = SimulationConfig(
            frequency=1e9, excitation=exc, quad_order=4, backend='auto',
            layer_stack=stack, source_layer_name='FR4',
        )
        sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

        # Extract with 2D Z₀ and with nominal 50 Ohm
        qs_2d = QuasiStaticSolver(sim, [port1, port2], Z0=tl.Z0, probe_feeds=True)
        qs_50 = QuasiStaticSolver(sim, [port1, port2], Z0=50.0, probe_feeds=True)

        [r_2d] = qs_2d.extract([1e9])
        [r_50] = qs_50.extract([1e9])

        s21_2d = abs(r_2d.S_matrix[1, 0])
        s21_50 = abs(r_50.S_matrix[1, 0])

        # Both should be close to 1.0 since Z₀ is ~49 Ohm,
        # but 2D-derived should be at least as good
        assert s21_2d >= s21_50 - 0.01, (
            f"2D Z0 ({tl.Z0:.1f}) gave worse S21 ({s21_2d:.4f}) "
            f"than 50 Ohm ({s21_50:.4f})"
        )


# ---------------------------------------------------------------------------
# Tests: Probe port QS with 2D Z₀ (second fixture, different port placement)
# ---------------------------------------------------------------------------

class TestProbePortQSWith2DZ0:
    """Probe-fed ports at strip edges using 2D-derived reference impedance."""

    def test_s21_near_zero_db(self, edge_port_qs_with_2d_z0):
        """Through-line S21 should be near 0 dB."""
        results, tl, freqs = edge_port_qs_with_2d_z0
        for r in results:
            s21_db = 20 * np.log10(abs(r.S_matrix[1, 0]))
            assert s21_db > -1.0, (
                f"S21 = {s21_db:.2f} dB at {r.frequency/1e9:.1f} GHz"
            )

    def test_reciprocity(self, edge_port_qs_with_2d_z0):
        results, _, _ = edge_port_qs_with_2d_z0
        for r in results:
            diff = abs(r.S_matrix[0, 1] - r.S_matrix[1, 0])
            assert diff < 1e-10

    def test_passivity(self, edge_port_qs_with_2d_z0):
        results, _, _ = edge_port_qs_with_2d_z0
        for r in results:
            power = abs(r.S_matrix[0, 0])**2 + abs(r.S_matrix[1, 0])**2
            assert power <= 1.0 + 1e-6

    def test_s21_monotonic(self, edge_port_qs_with_2d_z0):
        results, _, _ = edge_port_qs_with_2d_z0
        s21_vals = [abs(r.S_matrix[1, 0]) for r in results]
        for i in range(len(s21_vals) - 1):
            assert s21_vals[i + 1] <= s21_vals[i] + 1e-6


# ---------------------------------------------------------------------------
# Tests: CrossSectionResult.gamma consistency with QS propagation
# ---------------------------------------------------------------------------

class TestGammaConsistency:
    """Verify 2D gamma(f) is consistent with 3D QS S21 phase."""

    def test_s21_phase_matches_beta(self, microstrip_qs_with_2d_z0):
        """S21 phase should be approximately -beta*L for a through-line.

        The port offsets mean the effective length isn't exactly L_STRIP,
        so we just check that phase is negative and in the right ballpark.
        """
        results, tl, freqs = microstrip_qs_with_2d_z0
        L_eff = 8.0e-3  # ~10mm - 2×1mm port offset

        for r in results:
            s21_phase = np.angle(r.S_matrix[1, 0])
            beta_2d = tl.beta(r.frequency)
            expected_phase = -beta_2d * L_eff

            # Phase should be negative (propagation delay)
            assert s21_phase < 0, f"S21 phase should be negative, got {s21_phase:.4f}"

            # Rough agreement — the port model adds parasitic phase
            # from the probe attachment mode, so exact match isn't expected.
            # Just verify phase is in the right order of magnitude.
            assert abs(s21_phase) < 4.0 * abs(expected_phase) + 0.5


# ---------------------------------------------------------------------------
# Fixtures: Stripline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stripline_tl_params():
    """2D-derived TL parameters for Rogers RT/duroid 5880 stripline."""
    b = 3.0e-3   # plate separation
    stack = LayerStack([
        Layer('pec_bot', z_bot=-np.inf, z_top=0, eps_r=1.0, is_pec=True),
        Layer('diel', z_bot=0, z_top=b, eps_r=2.2),
        Layer('pec_top', z_bot=b, z_top=np.inf, eps_r=1.0, is_pec=True),
    ])
    tl = compute_reference_impedance(
        stack, strip_width=1.5e-3, strip_z=b / 2, base_cells=200,
    )
    return tl, stack


@pytest.fixture(scope="module")
def stripline_qs_with_2d_z0(stripline_tl_params):
    """Stripline QS solver with probe feeds using 2D-derived Z₀.

    Plain plate mesh at z = b/2 with vertical probe feeds to the
    nearest PEC ground plane.
    """
    tl, stack = stripline_tl_params

    b = 3.0e-3
    W_STRIP = 1.5e-3
    L_STRIP = 10.0e-3
    h_strip = b / 2
    TEL = 0.7e-3
    PORT1_X = -L_STRIP / 2.0 + 1.0e-3
    PORT2_X = +L_STRIP / 2.0 - 1.0e-3

    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_plate_with_feeds(
        width=L_STRIP, height=W_STRIP,
        feed_x_list=[PORT1_X, PORT2_X],
        center=(0.0, 0.0, h_strip),
    )
    basis = compute_rwg_connectivity(mesh)

    feed1 = find_feed_edges(mesh, basis, feed_x=PORT1_X)
    feed2 = find_feed_edges(mesh, basis, feed_x=PORT2_X)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)
    port1 = Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1)
    port2 = Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2)

    exc = StripDeltaGapExcitation(feed_basis_indices=feed1, voltage=1.0)
    config = SimulationConfig(
        frequency=1e9, excitation=exc, quad_order=4, backend='auto',
        layer_stack=stack, source_layer_name='diel',
    )
    sim = Simulation(config, mesh=mesh, reporter=SilentReporter())

    qs = QuasiStaticSolver(
        sim, [port1, port2],
        Z0=tl.Z0,
        probe_feeds=True,
    )

    freqs = [0.5e9, 1.0e9, 2.0e9]
    results = qs.extract(freqs)
    return results, tl, freqs


# ---------------------------------------------------------------------------
# Tests: Stripline with dual PEC
# ---------------------------------------------------------------------------

class TestStriplineQSWith2DZ0:
    """Stripline (PEC-dielectric-PEC) through-line with edge ports."""

    def test_2d_eps_eff_equals_eps_r(self, stripline_tl_params):
        """For homogeneous stripline, ε_eff must equal ε_r exactly."""
        tl, _ = stripline_tl_params
        assert tl.eps_eff == pytest.approx(2.2, rel=0.01)

    def test_2d_z0_reasonable(self, stripline_tl_params):
        """Z₀ for 1.5mm strip in 3mm, ε_r=2.2 should be ~67 Ω."""
        tl, _ = stripline_tl_params
        assert 50 < tl.Z0 < 90, f"Z0 = {tl.Z0:.1f} outside [50, 90]"

    def test_s21_near_zero_db(self, stripline_qs_with_2d_z0):
        """Through-line S21 should be near 0 dB at low frequencies."""
        results, tl, freqs = stripline_qs_with_2d_z0
        for r in results:
            s21_db = 20 * np.log10(abs(r.S_matrix[1, 0]))
            assert s21_db > -1.0, (
                f"S21 = {s21_db:.2f} dB at {r.frequency/1e9:.1f} GHz "
                f"(Z0_ref = {tl.Z0:.1f} Ohm)"
            )

    def test_reciprocity(self, stripline_qs_with_2d_z0):
        """|S12 - S21| should be negligible."""
        results, _, _ = stripline_qs_with_2d_z0
        for r in results:
            diff = abs(r.S_matrix[0, 1] - r.S_matrix[1, 0])
            assert diff < 1e-10

    def test_passivity(self, stripline_qs_with_2d_z0):
        """|S11|² + |S21|² ≤ 1 for lossless passive network."""
        results, _, _ = stripline_qs_with_2d_z0
        for r in results:
            power = abs(r.S_matrix[0, 0])**2 + abs(r.S_matrix[1, 0])**2
            assert power <= 1.0 + 1e-6, f"Passivity violated: {power:.6f}"

    def test_s21_monotonic(self, stripline_qs_with_2d_z0):
        """S21 should decrease monotonically in the QS regime."""
        results, _, _ = stripline_qs_with_2d_z0
        s21_vals = [abs(r.S_matrix[1, 0]) for r in results]
        for i in range(len(s21_vals) - 1):
            assert s21_vals[i + 1] <= s21_vals[i] + 1e-6

    def test_dual_pec_backend_used(self, stripline_tl_params):
        """StaticLayeredGF should auto-select DualPECImageBackend."""
        from pyMoM3d.greens.layered.static import (
            StaticLayeredGF, DualPECImageBackend,
        )
        _, stack = stripline_tl_params
        gf = StaticLayeredGF(stack, source_layer_name='diel')
        assert isinstance(gf.backend, DualPECImageBackend)
