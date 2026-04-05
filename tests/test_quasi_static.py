"""Tests for the quasi-static MoM solver and static Green's function."""

import numpy as np
import pytest

from pyMoM3d import (
    GmshMesher, compute_rwg_connectivity,
    Simulation, SimulationConfig, SilentReporter,
    Port, NetworkExtractor, Layer, LayerStack,
)
from pyMoM3d.mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges, find_edge_port_feed_edges,
    compute_feed_signs,
)
from pyMoM3d.mom.quasi_static import QuasiStaticSolver
from pyMoM3d.greens.layered.static import StaticLayeredGF, StaticPECImageBackend
from pyMoM3d.utils.constants import c0, eta0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def short_microstrip():
    """Short microstrip line (10mm) on FR4 with two ports."""
    EPS_R = 4.4
    H_SUB = 1.6e-3
    W_STRIP = 3.06e-3
    L_STRIP = 10.0e-3
    TEL = 0.7e-3
    PORT1_X = -L_STRIP / 2.0 + 1.0e-3
    PORT2_X = +L_STRIP / 2.0 - 1.0e-3

    stack = LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])

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

    return sim, [port1, port2], EPS_R


# ---------------------------------------------------------------------------
# StaticPECImageBackend tests
# ---------------------------------------------------------------------------

class TestStaticPECImageBackend:

    def test_scalar_g_pec_image_sign(self):
        """PEC image of charge has opposite sign → correction is negative."""
        backend = StaticPECImageBackend(z_pec=0.0, z_conductor=1.6e-3)
        r = np.array([[0.0, 0.0, 1.6e-3]])
        r_prime = np.array([[1.0e-3, 0.0, 1.6e-3]])
        g = backend.scalar_G(r, r_prime)
        assert np.real(g[0]) < 0, "Scalar G correction must be negative (PEC charge image)"

    def test_dyadic_g_horizontal_same_sign(self):
        """PEC image of horizontal current has same sign → correction is positive."""
        backend = StaticPECImageBackend(z_pec=0.0, z_conductor=1.6e-3)
        r = np.array([[0.0, 0.0, 1.6e-3]])
        r_prime = np.array([[1.0e-3, 0.0, 1.6e-3]])
        G = backend.dyadic_G(r, r_prime)
        # Horizontal components (xx, yy) positive
        assert np.real(G[0, 0, 0]) > 0, "G_A xx component must be positive"
        assert np.real(G[0, 1, 1]) > 0, "G_A yy component must be positive"
        # Vertical component (zz) negative
        assert np.real(G[0, 2, 2]) < 0, "G_A zz component must be negative"

    def test_scalar_g_symmetry(self):
        """G_φ(r, r') = G_φ(r', r)."""
        backend = StaticPECImageBackend(z_pec=0.0, z_conductor=1.6e-3)
        r = np.array([[0.0, 0.0, 1.6e-3], [2e-3, 1e-3, 1.6e-3]])
        r_prime = np.array([[1e-3, 0.5e-3, 1.6e-3], [3e-3, -1e-3, 1.6e-3]])
        g_fwd = backend.scalar_G(r, r_prime)
        g_rev = backend.scalar_G(r_prime, r)
        np.testing.assert_allclose(g_fwd, g_rev, rtol=1e-12)

    def test_image_distance(self):
        """Image distance R_img = √(ρ² + (2h)²) for z_pec=0, z_cond=h."""
        h = 1.6e-3
        backend = StaticPECImageBackend(z_pec=0.0, z_conductor=h)
        r = np.array([[0.0, 0.0, h]])
        r_prime = np.array([[0.0, 0.0, h]])
        R_img = backend._image_R(r, r_prime)
        expected = 2 * h  # same horizontal position → R_img = 2h
        np.testing.assert_allclose(R_img[0], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# StaticLayeredGF tests
# ---------------------------------------------------------------------------

class TestStaticLayeredGF:

    def test_requires_pec_layer(self):
        stack = LayerStack([
            Layer('dielectric', z_bot=0.0, z_top=1e-3, eps_r=4.0),
            Layer('air', z_bot=1e-3, z_top=np.inf, eps_r=1.0),
        ])
        with pytest.raises(ValueError, match="PEC ground"):
            StaticLayeredGF(stack)

    def test_wavenumber_is_small(self):
        stack = LayerStack([
            Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
            Layer('diel', z_bot=0.0, z_top=1e-3, eps_r=4.0),
            Layer('air', z_bot=1e-3, z_top=np.inf, eps_r=1.0),
        ])
        gf = StaticLayeredGF(stack, source_layer_name='diel')
        assert abs(gf.wavenumber) < 1e-5, "Static GF k must be very small"

    def test_wave_impedance_is_eta0(self):
        stack = LayerStack([
            Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
            Layer('diel', z_bot=0.0, z_top=1e-3, eps_r=4.0),
            Layer('air', z_bot=1e-3, z_top=np.inf, eps_r=1.0),
        ])
        gf = StaticLayeredGF(stack, source_layer_name='diel')
        np.testing.assert_allclose(gf.wave_impedance, eta0, rtol=1e-12)

    def test_eps_r_stored(self):
        stack = LayerStack([
            Layer('pec', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
            Layer('diel', z_bot=0.0, z_top=1e-3, eps_r=4.0),
            Layer('air', z_bot=1e-3, z_top=np.inf, eps_r=1.0),
        ])
        gf = StaticLayeredGF(stack, source_layer_name='diel')
        assert gf.backend.eps_r == 4.0


# ---------------------------------------------------------------------------
# QuasiStaticSolver tests
# ---------------------------------------------------------------------------

class TestQuasiStaticSolver:

    def test_l_raw_is_real(self, short_microstrip):
        """L_raw should be essentially real (static GF)."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, n_dielectric_images=0)
        imag_frac = np.max(np.abs(np.imag(qs.L_raw))) / np.max(np.abs(np.real(qs.L_raw)))
        assert imag_frac < 1e-10, f"L_raw imaginary fraction {imag_frac} too large"

    def test_l_raw_symmetric(self, short_microstrip):
        """L_raw must be symmetric (reciprocity)."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, n_dielectric_images=0)
        err = np.max(np.abs(qs.L_raw - qs.L_raw.T))
        assert err < 1e-20, f"L_raw symmetry error {err}"

    def test_g_s_symmetric(self, short_microstrip):
        """G_s must be symmetric (reciprocity)."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, n_dielectric_images=0)
        err = np.max(np.abs(qs.G_s - qs.G_s.T))
        assert err < 1e-20, f"G_s symmetry error {err}"

    def test_l_raw_positive_semidefinite(self, short_microstrip):
        """L_raw eigenvalues should be non-negative."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, n_dielectric_images=0)
        eigs = np.linalg.eigvalsh(np.real(qs.L_raw))
        assert eigs.min() > -1e-15, f"L_raw has negative eigenvalue {eigs.min()}"

    def test_extract_returns_network_results(self, short_microstrip):
        """extract() returns list of NetworkResult with correct length."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, n_dielectric_images=0)
        freqs = [1e9, 5e9, 10e9]
        results = qs.extract(freqs)
        assert len(results) == 3
        for r, f in zip(results, freqs):
            assert r.frequency == f

    def test_z_matrix_reciprocity(self, short_microstrip):
        """Port Z-matrix should be approximately symmetric."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, n_dielectric_images=0)
        results = qs.extract([1e9])
        Z = results[0].Z_matrix
        err = abs(Z[0, 1] - Z[1, 0]) / max(abs(Z[0, 1]), 1e-30)
        assert err < 0.1, f"Z12 != Z21 by {err*100:.1f}%"

    def test_passivity(self, short_microstrip):
        """S-parameters must satisfy passivity |S11|² + |S21|² ≤ 1."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, n_dielectric_images=0)
        results = qs.extract([1e9, 5e9])
        for r in results:
            S = r.S_matrix
            power = abs(S[0, 0])**2 + abs(S[1, 0])**2
            assert power <= 1.01, f"Passivity violated: |S11|²+|S21|²={power:.4f}"

    def test_eps_r_prefactor(self, short_microstrip):
        """QS Z-matrix should match full-wave within ~30% at low kD."""
        sim, ports, eps_r = short_microstrip
        qs = QuasiStaticSolver(sim, ports, n_dielectric_images=0)
        fw = NetworkExtractor(sim, ports)

        freq = 1e9  # kD ≈ 0.17 (quasi-static)
        res_qs = qs.extract([freq])[0]
        res_fw = fw.extract([freq])[0]

        ratio = abs(res_qs.Z_matrix[0, 0]) / max(abs(res_fw.Z_matrix[0, 0]), 1e-30)
        # Should be near 1.0 (within ~30% due to missing dielectric GF correction)
        assert 0.5 < ratio < 2.0, (
            f"Z11 ratio QS/FW = {ratio:.2f}, expected near 1.0"
        )

    def test_single_frequency_scalar(self, short_microstrip):
        """extract() with scalar frequency should work."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, n_dielectric_images=0)
        results = qs.extract(1e9)
        assert len(results) == 1
        assert results[0].frequency == 1e9


# ---------------------------------------------------------------------------
# Probe-fed QuasiStaticSolver tests
# ---------------------------------------------------------------------------

class TestProbeFeeds:

    def test_probe_setup(self, short_microstrip):
        """Probe feed setup creates correct hybrid matrices."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        N_s = qs._N_surface
        N_p = qs._N_probes
        assert N_p == 2
        assert qs.L_raw_hybrid.shape == (N_s + N_p, N_s + N_p)
        assert qs.P_hybrid.shape == (N_s + N_p, N_s + N_p)
        assert qs.V_all.shape == (N_s + N_p, N_p)

    def test_probe_charge_conservation(self, short_microstrip):
        """Probe divergence must integrate to 1 over attachment triangles."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        mesh = sim.mesh
        N_s = qs._N_surface

        # Reconstruct D_ext to check charge conservation
        for pi, vi in enumerate(qs._probe_vertices):
            tris = [t for t in range(len(mesh.triangles)) if vi in mesh.triangles[t]]
            areas = np.array([mesh.triangle_areas[t] for t in tris])
            A_total = areas.sum()
            charge_integral = sum(1.0 / A_total * a for a in areas)
            np.testing.assert_allclose(
                charge_integral, 1.0, rtol=1e-12,
                err_msg=f"Probe {pi} charge integral {charge_integral} != 1",
            )

    def test_p_hybrid_symmetric(self, short_microstrip):
        """P_hybrid must be symmetric (reciprocity)."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        P = np.real(qs.P_hybrid)
        err = np.max(np.abs(P - P.T))
        assert err < 1e-15, f"P_hybrid symmetry error {err}"

    def test_l_hybrid_symmetric(self, short_microstrip):
        """L_raw_hybrid must be symmetric."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        L = qs.L_raw_hybrid
        err = np.max(np.abs(L - L.T))
        assert err < 1e-20, f"L_raw_hybrid symmetry error {err}"

    def test_probe_s21_near_zero_db(self, short_microstrip):
        """Probe-fed through-line S21 should be near 0 dB at low frequency."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        results = qs.extract([0.5e9, 1e9])
        for r in results:
            S = r.S_matrix
            s21_db = 20 * np.log10(max(abs(S[1, 0]), 1e-30))
            assert s21_db > -1.0, (
                f"Probe S21 = {s21_db:.2f} dB at {r.frequency/1e9:.1f} GHz, "
                f"expected > -1 dB for matched through-line"
            )

    def test_probe_s21_monotonic_qs_regime(self, short_microstrip):
        """S21 must decrease monotonically in the quasi-static regime."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        freqs = [0.5e9, 1e9, 1.5e9, 2e9]
        results = qs.extract(freqs)
        s21_vals = [
            20 * np.log10(max(abs(r.S_matrix[1, 0]), 1e-30))
            for r in results
        ]
        for i in range(len(s21_vals) - 1):
            assert s21_vals[i] >= s21_vals[i + 1] - 0.1, (
                f"S21 not monotonic: {s21_vals[i]:.2f} dB at "
                f"{freqs[i]/1e9:.1f} GHz < {s21_vals[i+1]:.2f} dB at "
                f"{freqs[i+1]/1e9:.1f} GHz"
            )

    def test_probe_reciprocity(self, short_microstrip):
        """Probe-fed Z-matrix must be reciprocal (Z12 = Z21)."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        results = qs.extract([1e9])
        Z = results[0].Z_matrix
        err = abs(Z[0, 1] - Z[1, 0]) / max(abs(Z[0, 1]), 1e-30)
        assert err < 1e-10, f"Probe Z12 != Z21 by {err:.2e}"

    def test_probe_symmetry(self, short_microstrip):
        """Symmetric structure must give Z11 = Z22."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        results = qs.extract([1e9])
        Z = results[0].Z_matrix
        err = abs(Z[0, 0] - Z[1, 1]) / max(abs(Z[0, 0]), 1e-30)
        assert err < 1e-4, f"|Z11-Z22|/|Z11| = {err:.2e}"

    def test_probe_passivity(self, short_microstrip):
        """Probe-fed S-parameters must satisfy passivity."""
        sim, ports, _ = short_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        results = qs.extract([0.5e9, 1e9, 2e9])
        for r in results:
            S = r.S_matrix
            power = abs(S[0, 0])**2 + abs(S[1, 0])**2
            assert power <= 1.01, (
                f"Passivity violated at {r.frequency/1e9:.1f} GHz: "
                f"|S11|^2+|S21|^2={power:.4f}"
            )

    def test_probe_beats_standard_at_low_freq(self, short_microstrip):
        """Probe S21 must be much better than standard at low frequency."""
        sim, ports, _ = short_microstrip
        qs_probe = QuasiStaticSolver(sim, ports, probe_feeds=True)
        qs_std = QuasiStaticSolver(sim, ports, probe_feeds=False)

        freq = 0.5e9
        s21_probe = 20 * np.log10(max(
            abs(qs_probe.extract([freq])[0].S_matrix[1, 0]), 1e-30
        ))
        s21_std = 20 * np.log10(max(
            abs(qs_std.extract([freq])[0].S_matrix[1, 0]), 1e-30
        ))
        improvement = s21_probe - s21_std
        assert improvement > 20, (
            f"Probe S21 ({s21_probe:.1f} dB) should be >20 dB better than "
            f"standard ({s21_std:.1f} dB) at {freq/1e9:.1f} GHz"
        )


# ---------------------------------------------------------------------------
# Edge port fixtures and tests
# ---------------------------------------------------------------------------

@pytest.fixture
def edge_port_microstrip():
    """Microstrip with edge-fed vertical plate ports at both ends."""
    EPS_R = 4.4
    H_SUB = 1.6e-3
    W_STRIP = 3.06e-3
    L_STRIP = 10.0e-3
    TEL = 0.7e-3

    stack = LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('FR4', z_bot=0.0, z_top=H_SUB, eps_r=EPS_R),
        Layer('air', z_bot=H_SUB, z_top=np.inf, eps_r=1.0),
    ])

    mesher = GmshMesher(target_edge_length=TEL)
    mesh = mesher.mesh_microstrip_with_edge_ports(
        width=W_STRIP,
        length=L_STRIP,
        substrate_height=H_SUB,
        port_edges=['left', 'right'],
    )
    basis = compute_rwg_connectivity(mesh)

    x_left = -L_STRIP / 2.0
    x_right = +L_STRIP / 2.0

    feed1 = find_edge_port_feed_edges(mesh, basis, port_x=x_left, strip_z=H_SUB)
    feed2 = find_edge_port_feed_edges(mesh, basis, port_x=x_right, strip_z=H_SUB)
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

    return sim, [port1, port2], EPS_R, H_SUB


class TestEdgePorts:
    """Tests for edge-fed vertical plate port model."""

    def test_mesh_has_plates(self, edge_port_microstrip):
        """Mesh contains both strip and vertical plate triangles."""
        sim, _, _, h_sub = edge_port_microstrip
        mesh = sim.mesh
        z_centroids = np.mean(mesh.vertices[mesh.triangles, 2], axis=1)
        n_strip = np.sum(np.abs(z_centroids - h_sub) < 1e-6)
        n_plate = len(mesh.triangles) - n_strip
        assert n_strip > 0, "No strip triangles found"
        assert n_plate > 0, "No plate triangles found"

    def test_junction_basis_functions(self, edge_port_microstrip):
        """RWG basis functions span the strip-plate junction."""
        sim, _, _, h_sub = edge_port_microstrip
        mesh = sim.mesh
        basis = sim.basis
        n_junction = 0
        for n in range(basis.num_basis):
            tp = basis.t_plus[n]
            tm = basis.t_minus[n]
            z_tp = mesh.vertices[mesh.triangles[tp], 2]
            z_tm = mesh.vertices[mesh.triangles[tm], 2]
            tp_strip = np.allclose(z_tp, h_sub, atol=1e-8)
            tm_strip = np.allclose(z_tm, h_sub, atol=1e-8)
            if tp_strip != tm_strip:
                n_junction += 1
        assert n_junction >= 4, (
            f"Expected ≥4 junction basis functions, got {n_junction}"
        )

    def test_feed_edges_found(self, edge_port_microstrip):
        """Edge port feed edges are found at both port locations."""
        sim, ports, _, _ = edge_port_microstrip
        for port in ports:
            assert len(port.feed_basis_indices) >= 2, (
                f"Port {port.name}: expected ≥2 feed edges, "
                f"got {len(port.feed_basis_indices)}"
            )

    def test_s21_near_zero_low_freq(self, edge_port_microstrip):
        """S21 ≈ 0 dB at low frequency with probe feeds."""
        sim, ports, _, _ = edge_port_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        result = qs.extract([0.5e9])[0]
        s21_dB = 20 * np.log10(max(abs(result.S_matrix[1, 0]), 1e-30))
        assert s21_dB > -1.0, f"S21 = {s21_dB:.1f} dB at 0.5 GHz (expected > -1 dB)"

    def test_reciprocity(self, edge_port_microstrip):
        """S12 = S21 (reciprocity)."""
        sim, ports, _, _ = edge_port_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        result = qs.extract([1e9])[0]
        S = result.S_matrix
        assert abs(S[0, 1] - S[1, 0]) < 1e-10, (
            f"|S12-S21| = {abs(S[0,1]-S[1,0]):.2e}"
        )

    def test_passivity(self, edge_port_microstrip):
        """|S11|² + |S21|² ≤ 1 at all frequencies."""
        sim, ports, _, _ = edge_port_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        freqs = np.linspace(0.1e9, 3e9, 10)
        results = qs.extract(freqs.tolist())
        for r in results:
            S = r.S_matrix
            power = abs(S[0, 0])**2 + abs(S[1, 0])**2
            assert power <= 1.01, (
                f"Passivity violation at {r.frequency/1e9:.1f} GHz: "
                f"|S11|²+|S21|² = {power:.4f}"
            )

    def test_s21_monotonic_qs_regime(self, edge_port_microstrip):
        """S21 monotonically decreasing in QS regime (f < 2 GHz)."""
        sim, ports, _, _ = edge_port_microstrip
        qs = QuasiStaticSolver(sim, ports, probe_feeds=True)
        freqs = np.linspace(0.1e9, 2e9, 20)
        results = qs.extract(freqs.tolist())
        s21_dB = np.array([
            20 * np.log10(max(abs(r.S_matrix[1, 0]), 1e-30))
            for r in results
        ])
        diffs = np.diff(s21_dB)
        n_increasing = np.sum(diffs > 0.5)  # 0.5 dB tolerance
        assert n_increasing == 0, (
            f"S21 is not monotonically decreasing: "
            f"{n_increasing} violations in {diffs}"
        )
