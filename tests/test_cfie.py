"""Tests for CFIEOperator: combination, post_assembly, API, backend consistency."""

import warnings
import numpy as np
import pytest

from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators.efie import EFIEOperator
from pyMoM3d.mom.operators.mfie import MFIEOperator
from pyMoM3d.mom.operators.cfie import CFIEOperator, _CPP_AVAILABLE, _NUMBA_AVAILABLE
from pyMoM3d.utils.constants import eta0, c0


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_closed_mesh():
    """Regular tetrahedron — simplest closed PEC surface."""
    s = 0.1
    vertices = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=np.float64) * s
    triangles = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ], dtype=np.int32)
    mesh = Mesh(vertices, triangles)
    from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity as _crwg
    basis = _crwg(mesh)
    return mesh, basis


# ---------------------------------------------------------------------------
# CFIEOperator construction
# ---------------------------------------------------------------------------

class TestCFIEOperatorInit:
    def test_default_alpha(self):
        op = CFIEOperator()
        assert op.alpha == 0.5

    def test_custom_alpha(self):
        op = CFIEOperator(alpha=0.3)
        assert op.alpha == 0.3

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError):
            CFIEOperator(alpha=0.0)

    def test_invalid_alpha_one(self):
        with pytest.raises(ValueError):
            CFIEOperator(alpha=1.0)

    def test_extreme_alpha_warns(self):
        with pytest.warns(UserWarning, match="near 0 or 1"):
            CFIEOperator(alpha=0.1)

    def test_is_symmetric_false(self):
        assert CFIEOperator.is_symmetric is False


# ---------------------------------------------------------------------------
# CFIE = α·EFIE + (1−α)·η·MFIE
# ---------------------------------------------------------------------------

class TestCFIECombination:
    """Verify entry-by-entry that Z_CFIE = α·Z_EFIE + (1−α)·η·Z_MFIE."""

    def _assemble_all(self, alpha):
        mesh, basis = _make_closed_mesh()
        k = 2.0 * np.pi * 1e9 / c0

        Z_efie = fill_matrix(EFIEOperator(), basis, mesh, k, eta0, backend='numpy')
        Z_mfie = fill_matrix(MFIEOperator(), basis, mesh, k, eta0, backend='numpy')
        Z_cfie = fill_matrix(CFIEOperator(alpha=alpha), basis, mesh, k, eta0, backend='numpy')
        return Z_efie, Z_mfie, Z_cfie

    def test_linear_combination_alpha_half(self):
        alpha = 0.5
        Z_efie, Z_mfie, Z_cfie = self._assemble_all(alpha)
        Z_expected = alpha * Z_efie + (1.0 - alpha) * eta0 * Z_mfie
        # Use atol=1e-12 to handle floating-point noise in near-zero entries
        assert np.allclose(Z_cfie, Z_expected, rtol=1e-6, atol=1e-12), (
            f"max error = {np.max(np.abs(Z_cfie - Z_expected)):.3e}"
        )

    def test_linear_combination_alpha_03(self):
        alpha = 0.3
        Z_efie, Z_mfie, Z_cfie = self._assemble_all(alpha)
        Z_expected = alpha * Z_efie + (1.0 - alpha) * eta0 * Z_mfie
        assert np.allclose(Z_cfie, Z_expected, rtol=1e-6, atol=1e-12)

    def test_matrix_finite(self):
        _, _, Z_cfie = self._assemble_all(0.5)
        assert np.all(np.isfinite(Z_cfie))

    def test_is_not_symmetric_by_design(self):
        """CFIEOperator.is_symmetric must be False."""
        assert CFIEOperator.is_symmetric is False


# ---------------------------------------------------------------------------
# Open surface raises
# ---------------------------------------------------------------------------

class TestCFIEOpenSurface:
    def test_raises_for_open_mesh(self):
        verts = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 0.5, 0.0],
        ])
        tris = np.array([[0, 1, 2], [2, 1, 3]])
        m = Mesh(verts, tris)
        from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity as _crwg
        b = _crwg(m)
        op = CFIEOperator(alpha=0.5)
        with pytest.raises(ValueError, match="closed surface"):
            fill_matrix(op, b, m, 1.0, eta0, backend='numpy')


# ---------------------------------------------------------------------------
# SimulationConfig CFIE wire-up
# ---------------------------------------------------------------------------

class TestSimulationConfigCFIE:
    def test_formulation_field_default(self):
        from pyMoM3d.simulation import SimulationConfig
        from pyMoM3d.mom.excitation import PlaneWaveExcitation
        exc = PlaneWaveExcitation(
            E0=[1.0, 0.0, 0.0],
            k_hat=[0.0, 0.0, -1.0],
        )
        cfg = SimulationConfig(frequency=1e9, excitation=exc)
        assert cfg.formulation == 'EFIE'
        assert cfg.cfie_alpha == 0.5
        assert cfg.backend == 'auto'

    def test_mfie_formulation_accepted(self):
        from pyMoM3d.simulation import SimulationConfig
        from pyMoM3d.mom.excitation import PlaneWaveExcitation
        exc = PlaneWaveExcitation(E0=[1.0, 0.0, 0.0], k_hat=[0.0, 0.0, -1.0])
        cfg = SimulationConfig(frequency=1e9, excitation=exc, formulation='MFIE')
        assert cfg.formulation == 'MFIE'

    def test_cfie_formulation_accepted(self):
        from pyMoM3d.simulation import SimulationConfig
        from pyMoM3d.mom.excitation import PlaneWaveExcitation
        exc = PlaneWaveExcitation(
            E0=[1.0, 0.0, 0.0],
            k_hat=[0.0, 0.0, -1.0],
        )
        cfg = SimulationConfig(
            frequency=1e9, excitation=exc,
            formulation='CFIE', cfie_alpha=0.4,
        )
        assert cfg.formulation == 'CFIE'
        assert cfg.cfie_alpha == 0.4


# ---------------------------------------------------------------------------
# CFIE backend consistency: cpp and numba must match numpy
# ---------------------------------------------------------------------------

class TestCFIEBackendConsistency:
    """Verify cpp and numba backends produce the same Z matrix as numpy.

    The fused CFIE kernel computes I_A, I_Phi, and I_K in a single pass;
    these tests confirm that the fused path gives the same result as the
    numpy per-pair blend reference.
    """

    RTOL = 1e-5
    ATOL = 1e-12

    def _numpy_ref(self, alpha=0.5):
        mesh, basis = _make_closed_mesh()
        k = 2.0 * np.pi * 3e9 / c0
        Z_np = fill_matrix(
            CFIEOperator(alpha=alpha), basis, mesh, k, eta0, backend='numpy'
        )
        return mesh, basis, k, Z_np

    @pytest.mark.skipif(not _CPP_AVAILABLE, reason="C++ backend not built")
    def test_cpp_matches_numpy(self):
        mesh, basis, k, Z_np = self._numpy_ref()
        Z_cpp = fill_matrix(CFIEOperator(), basis, mesh, k, eta0, backend='cpp')
        assert np.all(np.isfinite(Z_cpp)), "cpp result contains NaN/Inf"
        assert np.allclose(Z_cpp, Z_np, rtol=self.RTOL, atol=self.ATOL), (
            f"cpp vs numpy max error = {np.max(np.abs(Z_cpp - Z_np)):.3e}, "
            f"max relative = {np.max(np.abs(Z_cpp - Z_np) / (np.abs(Z_np) + 1e-30)):.3e}"
        )

    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba backend not available")
    def test_numba_matches_numpy(self):
        mesh, basis, k, Z_np = self._numpy_ref()
        Z_nb = fill_matrix(CFIEOperator(), basis, mesh, k, eta0, backend='numba')
        assert np.all(np.isfinite(Z_nb)), "numba result contains NaN/Inf"
        assert np.allclose(Z_nb, Z_np, rtol=self.RTOL, atol=self.ATOL), (
            f"numba vs numpy max error = {np.max(np.abs(Z_nb - Z_np)):.3e}, "
            f"max relative = {np.max(np.abs(Z_nb - Z_np) / (np.abs(Z_np) + 1e-30)):.3e}"
        )

    @pytest.mark.skipif(
        not (_CPP_AVAILABLE and _NUMBA_AVAILABLE),
        reason="Both cpp and numba backends required",
    )
    def test_cpp_matches_numba(self):
        mesh, basis = _make_closed_mesh()
        k = 2.0 * np.pi * 3e9 / c0
        Z_cpp = fill_matrix(CFIEOperator(), basis, mesh, k, eta0, backend='cpp')
        Z_nb  = fill_matrix(CFIEOperator(), basis, mesh, k, eta0, backend='numba')
        assert np.allclose(Z_cpp, Z_nb, rtol=self.RTOL, atol=self.ATOL), (
            f"cpp vs numba max error = {np.max(np.abs(Z_cpp - Z_nb)):.3e}"
        )

    @pytest.mark.skipif(not _CPP_AVAILABLE, reason="C++ backend not built")
    def test_cpp_alpha_sweep(self):
        """cpp backend gives correct linear combination for multiple alpha values."""
        mesh, basis = _make_closed_mesh()
        k = 2.0 * np.pi * 3e9 / c0
        for alpha in (0.3, 0.5, 0.7):
            Z_np  = fill_matrix(CFIEOperator(alpha=alpha), basis, mesh, k, eta0, backend='numpy')
            Z_cpp = fill_matrix(CFIEOperator(alpha=alpha), basis, mesh, k, eta0, backend='cpp')
            assert np.allclose(Z_cpp, Z_np, rtol=self.RTOL, atol=self.ATOL), (
                f"alpha={alpha}: cpp vs numpy max error = "
                f"{np.max(np.abs(Z_cpp - Z_np)):.3e}"
            )

    @pytest.mark.skipif(not _CPP_AVAILABLE, reason="C++ backend not built")
    def test_auto_backend_selects_cpp(self):
        """backend='auto' selects cpp when available."""
        mesh, basis = _make_closed_mesh()
        k = 2.0 * np.pi * 3e9 / c0
        Z_auto = fill_matrix(CFIEOperator(), basis, mesh, k, eta0, backend='auto')
        Z_cpp  = fill_matrix(CFIEOperator(), basis, mesh, k, eta0, backend='cpp')
        assert np.allclose(Z_auto, Z_cpp, rtol=1e-12, atol=1e-30)
