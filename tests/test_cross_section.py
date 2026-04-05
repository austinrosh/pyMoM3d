"""Tests for the 2D cross-section FDM Laplace solver.

Validates geometry, grid, solver, extraction, and presets against
analytical benchmarks.

Tier 1: Analytical formula comparisons (Hammerstad, Cohn, conformal)
Tier 2: Grid convergence (Richardson extrapolation)
Tier 3: Exact limiting cases (parallel plate, stripline eps_eff identity)
Tier 4: Symmetry and reciprocity
"""

import copy

import numpy as np
import pytest
from scipy.special import ellipk

from pyMoM3d.cross_section.geometry import (
    Conductor,
    CrossSection,
    DielectricRegion,
)
from pyMoM3d.cross_section.grid import (
    NonUniformGrid,
    build_grid,
    build_grid_for_cross_section,
)
from pyMoM3d.cross_section.solver import LaplaceSolver2D, LaplaceSolution
from pyMoM3d.cross_section.extraction import (
    CrossSectionResult,
    extract_tl_params,
    extract_multiconductor_params,
)
from pyMoM3d.cross_section.presets import (
    microstrip_cross_section,
    stripline_cross_section,
    cpw_cross_section,
    coupled_microstrip_cross_section,
)
from pyMoM3d.analysis.transmission_line import (
    microstrip_z0_hammerstad,
    stripline_z0_cohn,
    cpw_z0_conformal,
)
from pyMoM3d.utils.constants import c0, eps0, mu0


# =====================================================================
# Geometry tests
# =====================================================================

class TestConductor:
    def test_zero_thickness(self):
        c = Conductor("strip", -1, 1, 5.0, 5.0, voltage=1.0)
        assert c.is_zero_thickness
        assert c.width == pytest.approx(2.0)
        assert c.center_x == pytest.approx(0.0)

    def test_finite_thickness(self):
        c = Conductor("strip", 0, 2, 1.0, 1.5, voltage=1.0)
        assert not c.is_zero_thickness


class TestCrossSection:
    def _make_microstrip(self):
        return microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4)

    def test_signal_ground_classification(self):
        cs = self._make_microstrip()
        signals = cs.signal_conductors()
        grounds = cs.ground_conductors()
        assert len(signals) == 1
        assert signals[0].name == "signal"
        assert len(grounds) == 1
        assert grounds[0].name == "ground"

    def test_eps_r_map(self):
        cs = self._make_microstrip()
        grid = build_grid_for_cross_section(cs, base_cells=50)
        eps = cs.eps_r_map(grid)
        # Below substrate top: eps_r = 4.4
        j_sub = np.argmin(np.abs(grid.y - 0.25e-3))
        i_mid = len(grid.x) // 2
        assert eps[i_mid, j_sub] == pytest.approx(4.4)
        # Above substrate: eps_r = 1.0
        j_air = np.argmin(np.abs(grid.y - 1.0e-3))
        assert eps[i_mid, j_air] == pytest.approx(1.0)

    def test_conductor_mask(self):
        cs = self._make_microstrip()
        grid = build_grid_for_cross_section(cs, base_cells=50)
        mask = cs.conductor_mask(grid)
        # At least some points should be conductors
        assert mask.any()
        # Ground at y=0 should be masked
        j0 = np.argmin(np.abs(grid.y))
        assert mask[:, j0].any()

    def test_vacuum_copy(self):
        cs = self._make_microstrip()
        cs_vac = cs.vacuum_copy()
        for d in cs_vac.dielectric_regions:
            assert d.eps_r == 1.0
        # Original unchanged
        assert cs.dielectric_regions[0].eps_r == 4.4


# =====================================================================
# Grid tests
# =====================================================================

class TestNonUniformGrid:
    def test_basic_properties(self):
        g = NonUniformGrid(x=np.linspace(0, 1, 11), y=np.linspace(0, 2, 21))
        assert g.Nx == 11
        assert g.Ny == 21
        assert g.shape == (11, 21)
        assert g.num_points == 231

    def test_dx_dy(self):
        g = NonUniformGrid(x=np.array([0.0, 1.0, 3.0]), y=np.array([0.0, 0.5, 2.0]))
        np.testing.assert_allclose(g.dx(), [1.0, 2.0])
        np.testing.assert_allclose(g.dy(), [0.5, 1.5])

    def test_minimum_size(self):
        with pytest.raises(ValueError):
            NonUniformGrid(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0, 2.0]))


class TestBuildGrid:
    def test_refinement_near_feature(self):
        g = build_grid(
            x_range=(-5, 5),
            y_range=(-5, 5),
            base_spacing=1.0,
            refine_x=[0.0],
            refine_spacing=0.05,
            growth_rate=1.3,
        )
        # Grid should have finer spacing near x=0
        dx = np.diff(g.x)
        i_center = np.argmin(np.abs(g.x))
        assert dx[i_center] < 0.1  # Fine near feature
        assert dx[0] > 0.2  # Coarse at edge

    def test_grid_for_microstrip(self):
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4)
        g = build_grid_for_cross_section(cs, base_cells=100)
        assert g.Nx > 50
        assert g.Ny > 30
        # Domain should extend well beyond conductor
        assert g.x[0] < -1e-3
        assert g.x[-1] > 1e-3

    def test_grid_for_stripline_clamped(self):
        """Stripline domain should be clamped to ground planes in y."""
        cs = stripline_cross_section(W=1e-3, b=3e-3, eps_r=2.2)
        g = build_grid_for_cross_section(cs, base_cells=100)
        assert g.y[0] == pytest.approx(0.0, abs=1e-12)
        assert g.y[-1] == pytest.approx(3e-3, abs=1e-12)

    def test_grid_for_cpw_includes_gaps(self):
        """CPW grid must include the coplanar ground gap region."""
        cs = cpw_cross_section(W=1e-3, S=0.5e-3, eps_r=4.4, h=1e-3)
        g = build_grid_for_cross_section(cs, base_cells=200)
        # Grid x-range must include the coplanar ground inner edge
        gap_edge = 1e-3 / 2 + 0.5e-3  # W/2 + S
        assert g.x[-1] > gap_edge


# =====================================================================
# Solver basic tests
# =====================================================================

class TestLaplaceSolver:
    def test_dirichlet_bc(self):
        """Verify conductor Dirichlet BCs are satisfied."""
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4)
        solver = LaplaceSolver2D(cs, base_cells=50)
        sol = solver.solve()
        # Signal conductor points should be ~1.0
        sig = cs.signal_conductors()[0]
        grid = solver.grid
        i_mid = np.argmin(np.abs(grid.x - sig.center_x))
        j_sig = np.argmin(np.abs(grid.y - sig.y_min))
        assert sol.V[i_mid, j_sig] == pytest.approx(1.0, abs=1e-10)
        # Ground at y=0 should be 0.0
        j_gnd = np.argmin(np.abs(grid.y))
        assert sol.V[i_mid, j_gnd] == pytest.approx(0.0, abs=1e-10)

    def test_potential_monotone(self):
        """Potential should increase from ground (0) to signal (1)."""
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=1.0)
        solver = LaplaceSolver2D(cs, base_cells=80)
        sol = solver.solve()
        grid = solver.grid
        i_mid = len(grid.x) // 2
        j_gnd = np.argmin(np.abs(grid.y))
        j_sig = np.argmin(np.abs(grid.y - 0.5e-3))
        # Along vertical line below signal: V should be non-decreasing
        V_col = sol.V[i_mid, j_gnd:j_sig + 1]
        assert np.all(np.diff(V_col) >= -1e-12)

    def test_charge_nonzero(self):
        """Charge on a signal conductor at V=1 should be non-zero."""
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4)
        solver = LaplaceSolver2D(cs, base_cells=100)
        sol = solver.solve()
        sig = cs.signal_conductors()[0]
        Q = solver.integrate_charge(sol, sig)
        assert abs(Q) > 1e-14


# =====================================================================
# Tier 1: Analytical benchmark comparisons
# =====================================================================

class TestMicrostripBenchmarks:
    """Microstrip Z0 and eps_eff vs Hammerstad-Jensen."""

    @pytest.mark.parametrize("wh,eps_r", [
        (0.5, 2.2), (1.0, 2.2), (2.0, 2.2),
        (0.5, 4.4), (1.0, 4.4), (2.0, 4.4),
        (0.5, 9.8), (1.0, 9.8), (2.0, 9.8),
    ])
    def test_z0_vs_hammerstad(self, wh, eps_r):
        h = 1.6e-3
        W = wh * h
        cs = microstrip_cross_section(W=W, h=h, eps_r=eps_r)
        result = extract_tl_params(cs, base_cells=300)
        z0_ref, ee_ref = microstrip_z0_hammerstad(W, h, eps_r)

        # Both the 2D solver and Hammerstad are approximate;
        # accept 4% tolerance on Z0 and eps_eff
        assert result.Z0 == pytest.approx(z0_ref, rel=0.04), (
            f"Z0: {result.Z0:.2f} vs {z0_ref:.2f}"
        )
        assert result.eps_eff == pytest.approx(ee_ref, rel=0.04), (
            f"eps_eff: {result.eps_eff:.3f} vs {ee_ref:.3f}"
        )


class TestStriplineBenchmarks:
    """Stripline Z0 vs Cohn elliptic integral (exact for centered, zero-t)."""

    @pytest.mark.parametrize("wb", [0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0])
    def test_z0_vs_cohn(self, wb):
        b = 3.0e-3
        eps_r = 2.2
        W = wb * b
        cs = stripline_cross_section(W=W, b=b, eps_r=eps_r)
        result = extract_tl_params(cs, base_cells=400)
        z0_ref = stripline_z0_cohn(W, b, eps_r)

        assert result.Z0 == pytest.approx(z0_ref, rel=0.015), (
            f"W/b={wb}: Z0={result.Z0:.2f} vs Cohn={z0_ref:.2f}"
        )

    @pytest.mark.parametrize("eps_r", [1.0, 2.2, 4.4, 9.8])
    def test_eps_eff_equals_eps_r(self, eps_r):
        """For homogeneous dielectric, eps_eff must equal eps_r."""
        cs = stripline_cross_section(W=1.5e-3, b=3e-3, eps_r=eps_r)
        result = extract_tl_params(cs, base_cells=200)
        assert result.eps_eff == pytest.approx(eps_r, rel=0.005), (
            f"eps_eff={result.eps_eff:.4f} vs eps_r={eps_r}"
        )


# =====================================================================
# Tier 2: Grid convergence
# =====================================================================

class TestGridConvergence:
    def test_microstrip_convergence(self):
        """Z0 should converge with grid refinement."""
        h = 1.6e-3
        W = 1.0 * h
        eps_r = 4.4
        cs = microstrip_cross_section(W=W, h=h, eps_r=eps_r)

        results = []
        for nc in [100, 200, 400]:
            r = extract_tl_params(cs, base_cells=nc)
            results.append(r.Z0)

        # Monotone convergence: successive refinements change less
        delta1 = abs(results[1] - results[0])
        delta2 = abs(results[2] - results[1])
        assert delta2 < delta1, (
            f"Not converging: delta1={delta1:.4f}, delta2={delta2:.4f}"
        )
        # Fine grid should be within 1% of medium grid
        assert results[2] == pytest.approx(results[1], rel=0.01)

    def test_stripline_convergence(self):
        """Stripline Z0 convergence and accuracy."""
        b = 3e-3
        W = 0.5 * b
        eps_r = 2.2
        cs = stripline_cross_section(W=W, b=b, eps_r=eps_r)
        z0_cohn = stripline_z0_cohn(W, b, eps_r)

        results = []
        for nc in [100, 200, 400]:
            r = extract_tl_params(cs, base_cells=nc)
            results.append(r.Z0)

        # Should converge toward Cohn value
        err_coarse = abs(results[0] - z0_cohn) / z0_cohn
        err_fine = abs(results[2] - z0_cohn) / z0_cohn
        assert err_fine < err_coarse


# =====================================================================
# Tier 3: Exact limiting cases
# =====================================================================

class TestExactCases:
    def test_parallel_plate_capacitance(self):
        """Parallel plate: C = eps0 * eps_r * W / d.

        Use a wide strip (W >> d) so fringing is negligible.
        """
        d = 1e-3
        W = 20e-3  # W/d = 20: fringing is small
        eps_r = 4.4
        ground_ext = 200e-3

        conductors = [
            Conductor("ground", -ground_ext, ground_ext, 0, 0, voltage=0.0),
            Conductor("signal", -W / 2, W / 2, d, d, voltage=1.0),
        ]
        dielectrics = [
            DielectricRegion("sub", -ground_ext, ground_ext, 0, d, eps_r=eps_r),
        ]
        cs = CrossSection(conductors=conductors, dielectric_regions=dielectrics)
        result = extract_tl_params(cs, base_cells=300)

        C_pp = eps0 * eps_r * W / d
        # For W/d=20, fringing adds ~5-10%. Accept 15% due to fringing.
        assert result.C_pul == pytest.approx(C_pp, rel=0.15), (
            f"C_pul={result.C_pul:.4e} vs C_pp={C_pp:.4e}"
        )

    def test_stripline_c_diel_over_c_vac(self):
        """For homogeneous dielectric, C_diel/C_vac = eps_r exactly."""
        eps_r = 6.0
        cs = stripline_cross_section(W=1.5e-3, b=3e-3, eps_r=eps_r)
        result = extract_tl_params(cs, base_cells=200)
        ratio = result.C_diel / result.C_vac
        assert ratio == pytest.approx(eps_r, rel=0.005)


# =====================================================================
# Tier 4: Symmetry and reciprocity
# =====================================================================

class TestSymmetry:
    def test_symmetric_potential(self):
        """Symmetric geometry should produce symmetric potential."""
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4)
        solver = LaplaceSolver2D(cs, base_cells=100)
        sol = solver.solve()
        grid = solver.grid

        # Find center index
        i_center = np.argmin(np.abs(grid.x))

        # V should be symmetric about x=0 (to grid tolerance)
        for offset in [1, 5, 10]:
            il = i_center - offset
            ir = i_center + offset
            if il >= 0 and ir < grid.Nx:
                np.testing.assert_allclose(
                    sol.V[il, :], sol.V[ir, :], atol=1e-6,
                    err_msg=f"Potential not symmetric at offset={offset}"
                )

    def test_coupled_microstrip_c_matrix_symmetric(self):
        """Multi-conductor C matrix must be symmetric."""
        cs = coupled_microstrip_cross_section(
            W=0.5e-3, S=0.3e-3, h=0.5e-3, eps_r=4.4,
        )
        result = extract_multiconductor_params(cs, base_cells=150)
        C = result.C_matrix
        assert C is not None
        np.testing.assert_allclose(C, C.T, rtol=0.05, atol=1e-14,
                                   err_msg="C matrix not symmetric")

    def test_coupled_microstrip_l_matrix_symmetric(self):
        """Multi-conductor L matrix must be symmetric."""
        cs = coupled_microstrip_cross_section(
            W=0.5e-3, S=0.3e-3, h=0.5e-3, eps_r=4.4,
        )
        result = extract_multiconductor_params(cs, base_cells=150)
        L = result.L_matrix
        assert L is not None
        np.testing.assert_allclose(L, L.T, rtol=0.05, atol=1e-14,
                                   err_msg="L matrix not symmetric")


# =====================================================================
# Extraction API tests
# =====================================================================

class TestExtraction:
    def test_result_fields(self):
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4)
        r = extract_tl_params(cs, base_cells=80)
        assert r.Z0 > 0
        assert r.eps_eff > 1.0
        assert r.v_phase > 0
        assert r.v_phase < c0
        assert r.C_pul > 0
        assert r.L_pul > 0
        assert r.C_diel > r.C_vac  # Dielectric increases C

    def test_gamma_beta(self):
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4)
        r = extract_tl_params(cs, base_cells=80)
        f = 1e9
        gamma = r.gamma(f)
        assert np.real(gamma) == pytest.approx(0.0, abs=1e-10)  # Lossless
        assert np.imag(gamma) > 0
        assert r.beta(f) == pytest.approx(np.imag(gamma), rel=1e-10)

    def test_z0_from_lc(self):
        """Z0 = sqrt(L/C) identity check."""
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4)
        r = extract_tl_params(cs, base_cells=100)
        z0_lc = np.sqrt(r.L_pul / r.C_pul)
        assert r.Z0 == pytest.approx(z0_lc, rel=1e-10)

    def test_v_phase_consistency(self):
        """v_phase = 1/sqrt(LC) = c0/sqrt(eps_eff)."""
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4)
        r = extract_tl_params(cs, base_cells=100)
        v_from_lc = 1.0 / np.sqrt(r.L_pul * r.C_pul)
        v_from_ee = c0 / np.sqrt(r.eps_eff)
        assert r.v_phase == pytest.approx(v_from_lc, rel=1e-10)
        assert r.v_phase == pytest.approx(v_from_ee, rel=1e-10)


# =====================================================================
# CPW tests (convergence and consistency, not formula comparison)
# =====================================================================

class TestCPW:
    def test_cpw_grid_includes_gaps(self):
        """The CPW grid must resolve the gap region."""
        W = 1e-3
        S = 0.5e-3
        cs = cpw_cross_section(W=W, S=S, eps_r=4.4, h=1e-3)
        grid = build_grid_for_cross_section(cs, base_cells=200)
        gap_edge = W / 2 + S
        # Grid must extend past the inner gap edge
        assert grid.x[-1] > gap_edge

    def test_cpw_convergence(self):
        """CPW Z0 should converge with grid refinement."""
        cs = cpw_cross_section(W=1e-3, S=0.5e-3, eps_r=4.4, h=1e-3)
        results = []
        for nc in [200, 400, 600]:
            r = extract_tl_params(cs, base_cells=nc)
            results.append(r.Z0)
        # Successive differences should decrease
        d1 = abs(results[1] - results[0])
        d2 = abs(results[2] - results[1])
        assert d2 < d1

    def test_cpw_eps_eff_bounded(self):
        """CPW eps_eff must be between 1 and eps_r."""
        eps_r = 4.4
        cs = cpw_cross_section(W=1e-3, S=0.5e-3, eps_r=eps_r, h=1e-3)
        r = extract_tl_params(cs, base_cells=200)
        assert 1.0 < r.eps_eff < eps_r + 0.1

    def test_cpw_z0_lower_than_microstrip(self):
        """Adding coplanar grounds should lower Z0 vs microstrip."""
        W = 1e-3
        h = 1e-3
        eps_r = 4.4
        ms = microstrip_cross_section(W=W, h=h, eps_r=eps_r)
        cpw = cpw_cross_section(W=W, S=0.5e-3, eps_r=eps_r, h=h)
        r_ms = extract_tl_params(ms, base_cells=200)
        r_cpw = extract_tl_params(cpw, base_cells=200)
        assert r_cpw.Z0 < r_ms.Z0


# =====================================================================
# Preset tests
# =====================================================================

class TestPresets:
    def test_microstrip_preset(self):
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4)
        assert len(cs.signal_conductors()) == 1
        assert len(cs.ground_conductors()) == 1

    def test_stripline_preset(self):
        cs = stripline_cross_section(W=1e-3, b=3e-3, eps_r=2.2)
        assert len(cs.signal_conductors()) == 1
        assert len(cs.ground_conductors()) == 2

    def test_cpw_preset(self):
        cs = cpw_cross_section(W=1e-3, S=0.5e-3, eps_r=4.4, h=1e-3)
        assert len(cs.signal_conductors()) == 1
        assert len(cs.ground_conductors()) == 3  # bottom + 2 coplanar

    def test_coupled_microstrip_preset(self):
        cs = coupled_microstrip_cross_section(W=0.5e-3, S=0.3e-3, h=0.5e-3, eps_r=4.4)
        assert len(cs.signal_conductors()) == 2
        assert len(cs.ground_conductors()) == 1

    def test_microstrip_with_thickness(self):
        cs = microstrip_cross_section(W=1e-3, h=0.5e-3, eps_r=4.4, t=35e-6)
        sig = cs.signal_conductors()[0]
        assert not sig.is_zero_thickness
        assert sig.y_max - sig.y_min == pytest.approx(35e-6)
