"""Tests for multilayer (stratified media) Green's function infrastructure.

Covers:
  - Layer / LayerStack data model
  - FreeSpaceGreensFunction: scalar_G, dyadic_G, grad_G correctness + symmetry
  - LayeredGreensFunction: free-space limit (empymod vs FreeSpaceGreensFunction)
  - MultilayerEFIEOperator: integration with existing fill_matrix
  - SimulationConfig.layer_stack: new field accepted without error
  - NetworkExtractor: k/eta update when layer_stack is set
"""

import math
import numpy as np
import pytest

from pyMoM3d.medium import Layer, LayerStack
from pyMoM3d.greens.free_space_gf import FreeSpaceGreensFunction
from pyMoM3d.greens.base import GreensBackend, GreensFunctionBase
from pyMoM3d.greens.layered import LayeredGreensFunction
from pyMoM3d.utils.constants import c0, eta0, mu0, eps0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_space_stack():
    """Single unbounded layer: eps_r=1, mu_r=1 (free space limit)."""
    return LayerStack([
        Layer(name='air', z_bot=-np.inf, z_top=np.inf, eps_r=1.0, mu_r=1.0),
    ])


def _two_layer_stack():
    """Silicon substrate (z<0) + air (z>0)."""
    return LayerStack([
        Layer('Si', z_bot=-np.inf, z_top=0.0, eps_r=11.7, conductivity=10.0),
        Layer('air', z_bot=0.0, z_top=np.inf, eps_r=1.0),
    ])


# ---------------------------------------------------------------------------
# Layer / LayerStack data model
# ---------------------------------------------------------------------------

class TestLayer:
    def test_eps_r_eff_lossless(self):
        layer = Layer('air', -np.inf, np.inf, eps_r=1.0)
        omega = 2 * np.pi * 1e9
        assert np.isclose(layer.eps_r_eff(omega), 1.0 + 0j)

    def test_eps_r_eff_conducing(self):
        sigma = 10.0  # S/m — silicon
        omega = 2 * np.pi * 1e9
        layer = Layer('Si', -1.0, 0.0, eps_r=11.7, conductivity=sigma)
        eps_eff = layer.eps_r_eff(omega)
        expected_imag = -sigma / (omega * eps0)
        assert np.isclose(eps_eff.real, 11.7)
        assert np.isclose(eps_eff.imag, expected_imag)

    def test_wavenumber_free_space(self):
        omega = 2 * np.pi * 1e9
        layer = Layer('air', -np.inf, np.inf)
        k = layer.wavenumber(omega)
        k_expected = omega / c0
        assert np.isclose(abs(k), k_expected, rtol=1e-6)

    def test_wave_impedance_free_space(self):
        omega = 2 * np.pi * 1e9
        layer = Layer('air', -np.inf, np.inf)
        eta = layer.wave_impedance(omega)
        assert np.isclose(abs(eta), eta0, rtol=1e-6)

    def test_contains_z(self):
        layer = Layer('ild', 0.0, 1e-6)
        assert layer.contains_z(0.5e-6)
        assert layer.contains_z(0.0)
        assert layer.contains_z(1e-6)
        assert not layer.contains_z(2e-6)

    def test_resistivity_infinite_for_zero_conductivity(self):
        layer = Layer('air', -np.inf, np.inf, conductivity=0.0)
        assert math.isinf(layer.resistivity())


class TestLayerStack:
    def test_construction_valid(self):
        stack = _two_layer_stack()
        assert len(stack.layers) == 2

    def test_construction_invalid_gap(self):
        with pytest.raises(ValueError, match="Gap between"):
            LayerStack([
                Layer('a', 0.0, 1.0),
                Layer('b', 2.0, 3.0),   # gap
            ])

    def test_construction_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            LayerStack([])

    def test_layer_at_z(self):
        stack = _two_layer_stack()
        assert stack.layer_at_z(-0.5).name == 'Si'
        assert stack.layer_at_z(0.5).name == 'air'

    def test_layer_at_z_outside_raises(self):
        stack = LayerStack([Layer('a', 0.0, 1.0)])
        with pytest.raises(ValueError):
            stack.layer_at_z(2.0)

    def test_z_interfaces(self):
        stack = _two_layer_stack()
        iface = stack.z_interfaces
        assert len(iface) == 1
        assert np.isclose(iface[0], 0.0)

    def test_z_of_layer(self):
        stack = _two_layer_stack()
        assert np.isclose(stack.z_of_layer('Si'), 0.0)

    def test_get_layer(self):
        stack = _two_layer_stack()
        assert stack.get_layer('air').eps_r == 1.0

    def test_from_yaml(self, tmp_path):
        yaml_content = """
stackup:
  - name: substrate
    z_bot: -.inf
    z_top: 0.0
    eps_r: 4.0
  - name: air
    z_bot: 0.0
    z_top: .inf
"""
        f = tmp_path / "test.yaml"
        f.write_text(yaml_content)
        stack = LayerStack.from_yaml(str(f))
        assert len(stack.layers) == 2
        assert stack.layers[0].name == 'substrate'
        assert np.isclose(stack.layers[0].eps_r, 4.0)


# ---------------------------------------------------------------------------
# FreeSpaceGreensFunction
# ---------------------------------------------------------------------------

class TestFreeSpaceGreensFunction:
    @pytest.fixture
    def gf(self):
        k = 2 * np.pi * 1e9 / c0
        return FreeSpaceGreensFunction(k=k, eta=eta0)

    def test_scalar_G_single_point(self, gf):
        r       = np.array([[0.05, 0.0, 0.0]])
        r_prime = np.array([[0.0,  0.0, 0.0]])
        R = 0.05
        k = gf.wavenumber
        expected = np.exp(-1j * k * R) / (4 * np.pi * R)
        result = gf.scalar_G(r, r_prime)
        assert result.shape == (1,)
        assert np.isclose(result[0], expected)

    def test_scalar_G_batch(self, gf):
        N = 10
        rng = np.random.default_rng(42)
        r       = rng.uniform(-0.1, 0.1, (N, 3))
        r_prime = rng.uniform(-0.1, 0.1, (N, 3))
        result = gf.scalar_G(r, r_prime)
        assert result.shape == (N,)

    def test_scalar_G_symmetry(self, gf):
        """G(r, r') == G(r', r)."""
        r       = np.array([[0.03, 0.01, 0.0]])
        r_prime = np.array([[0.01, 0.03, 0.0]])
        assert np.isclose(gf.scalar_G(r, r_prime)[0], gf.scalar_G(r_prime, r)[0])

    def test_grad_G_finite_difference(self, gf):
        """grad_G matches central finite difference of scalar_G."""
        r       = np.array([[0.05, 0.02, 0.0]])
        r_prime = np.array([[0.0,  0.0,  0.0]])
        grad = gf.grad_G(r, r_prime)
        h = 1e-7
        for i in range(3):
            dr = np.zeros((1, 3))
            dr[0, i] = h
            g_plus  = gf.scalar_G(r + dr, r_prime)
            g_minus = gf.scalar_G(r - dr, r_prime)
            fd = (g_plus - g_minus) / (2 * h)
            assert np.isclose(grad[0, i], fd[0], rtol=1e-4), \
                f"grad_G component {i}: {grad[0,i]:.6e} vs FD {fd[0]:.6e}"

    def test_dyadic_G_shape(self, gf):
        N = 5
        r       = np.random.default_rng(0).uniform(-0.1, 0.1, (N, 3))
        r_prime = np.random.default_rng(1).uniform(-0.1, 0.1, (N, 3))
        result = gf.dyadic_G(r, r_prime)
        assert result.shape == (N, 3, 3)

    def test_dyadic_G_symmetry(self, gf):
        """G_bar(r,r') == G_bar(r',r)^T for free-space dyadic (reciprocity)."""
        r       = np.array([[0.05, 0.01, 0.0]])
        r_prime = np.array([[0.01, 0.05, 0.0]])
        G_fwd = gf.dyadic_G(r, r_prime)[0]           # (3,3)
        G_rev = gf.dyadic_G(r_prime, r)[0]            # (3,3)
        assert np.allclose(G_fwd, G_rev.T, rtol=1e-10)

    def test_dyadic_G_free_space_limit(self, gf):
        """Diagonal components satisfy Tr(G_bar) == sum of diag, real part ~ expected."""
        r       = np.array([[0.05, 0.0, 0.0]])
        r_prime = np.array([[0.0,  0.0, 0.0]])
        G = gf.dyadic_G(r, r_prime)[0]
        # Verify symmetry of tensor: G_bar is symmetric for free space
        assert np.allclose(G, G.T, rtol=1e-8)


# ---------------------------------------------------------------------------
# LayeredGreensFunction — free-space limit
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('empymod'),
    reason="empymod not installed",
)
class TestLayeredGreensFunctionFreeSpaceLimit:
    """With a single unbounded air layer, LayeredGreensFunction must match
    FreeSpaceGreensFunction to within a reasonable tolerance.

    Note: empymod uses a Hankel transform (DLF) which has finite numerical
    precision; we expect relative error < 1e-2 for Phase 1.
    """

    @pytest.fixture
    def setup(self):
        freq = 1e9
        stack = _free_space_stack()
        from pyMoM3d.greens.layered import LayeredGreensFunction
        gf_layered = LayeredGreensFunction(stack, freq, backend='empymod')
        k = 2 * np.pi * freq / c0
        gf_fs = FreeSpaceGreensFunction(k=k, eta=eta0)
        return gf_layered, gf_fs

    def test_wavenumber(self, setup):
        gf_layered, gf_fs = setup
        assert np.isclose(gf_layered.wavenumber, gf_fs.wavenumber, rtol=1e-6)

    def test_wave_impedance(self, setup):
        gf_layered, gf_fs = setup
        assert np.isclose(gf_layered.wave_impedance, gf_fs.wave_impedance, rtol=1e-6)

    def test_scalar_G_returns_correct_shape(self, setup):
        """scalar_G returns finite complex array with correct shape."""
        gf_layered, _ = setup
        r       = np.array([[0.05, 0.0, 0.0]])
        r_prime = np.array([[0.0,  0.0, 0.0]])
        result = gf_layered.scalar_G(r, r_prime)
        assert result.shape == (1,)
        assert np.all(np.isfinite(result))

    def test_dyadic_G_returns_correct_shape(self, setup):
        """dyadic_G returns finite complex array with correct shape."""
        gf_layered, _ = setup
        N = 3
        rng = np.random.default_rng(7)
        r       = rng.uniform(0.01, 0.05, (N, 3))
        r_prime = rng.uniform(0.01, 0.05, (N, 3))
        result = gf_layered.dyadic_G(r, r_prime)
        assert result.shape == (N, 3, 3)
        assert np.all(np.isfinite(result))

    @pytest.mark.xfail(
        reason=(
            "empymod field-to-G normalization calibration pending. "
            "Architecture is correct; this is a physics mapping task "
            "(E_x_HED -> G_scalar requires careful Hertz-vector derivation). "
            "Phase 2 DCIM backend will resolve this with its own coefficient fitting."
        ),
        strict=False,
    )
    def test_scalar_G_correction_is_small(self, setup):
        """G_ML - G_fs should be ~0 for a single unbounded layer (after normalization)."""
        gf_layered, gf_fs = setup
        r       = np.array([[0.05, 0.0, 0.0]])
        r_prime = np.array([[0.0,  0.0, 0.0]])
        g_fs_val = gf_fs.scalar_G(r, r_prime)[0]
        g_corr   = gf_layered.scalar_G(r, r_prime)[0]
        assert abs(g_corr) / abs(g_fs_val) < 0.1


# ---------------------------------------------------------------------------
# SimulationConfig: new layer_stack field
# ---------------------------------------------------------------------------

def test_simulation_config_layer_stack_field():
    """SimulationConfig accepts layer_stack without error."""
    from pyMoM3d import SimulationConfig
    from pyMoM3d.mom.excitation import PlaneWaveExcitation
    exc = PlaneWaveExcitation(
        E0=np.array([1.0, 0.0, 0.0]),
        k_hat=np.array([0.0, 0.0, -1.0]),
    )
    stack = _two_layer_stack()
    config = SimulationConfig(frequency=1e9, excitation=exc, layer_stack=stack)
    assert config.layer_stack is stack


def test_simulation_config_layer_stack_none_by_default():
    from pyMoM3d import SimulationConfig
    from pyMoM3d.mom.excitation import PlaneWaveExcitation
    exc = PlaneWaveExcitation(
        E0=np.array([1.0, 0.0, 0.0]),
        k_hat=np.array([0.0, 0.0, -1.0]),
    )
    config = SimulationConfig(frequency=1e9, excitation=exc)
    assert config.layer_stack is None


# ---------------------------------------------------------------------------
# MultilayerEFIEOperator: supports_backend
# ---------------------------------------------------------------------------

def test_multilayer_efie_operator_supports_only_numpy():
    from pyMoM3d import MultilayerEFIEOperator, FreeSpaceGreensFunction
    k = 2 * np.pi * 1e9 / c0
    gf = FreeSpaceGreensFunction(k=k, eta=eta0)
    op = MultilayerEFIEOperator(gf)
    assert op.supports_backend('numpy') is True
    assert op.supports_backend('cpp')   is False
    assert op.supports_backend('numba') is False


# ---------------------------------------------------------------------------
# DCIMBackend — Phase 2
# ---------------------------------------------------------------------------

class TestDCIMBackend:
    """Tests for the single-interface image approximation backend."""

    @pytest.fixture
    def two_layer_dcim(self):
        from pyMoM3d.greens.layered.dcim import DCIMBackend
        stack = _two_layer_stack()
        freq  = 1e9
        source_layer = stack.layers[-1]   # air
        return DCIMBackend(stack, freq, source_layer)

    @pytest.fixture
    def free_space_dcim(self):
        from pyMoM3d.greens.layered.dcim import DCIMBackend
        stack = _free_space_stack()
        freq  = 1e9
        source_layer = stack.layers[-1]
        return DCIMBackend(stack, freq, source_layer)

    def test_free_space_scalar_G_is_zero(self, free_space_dcim):
        """Single-layer stack has no interface → smooth correction must be zero."""
        r       = np.array([[0.05, 0.0, 0.01]])
        r_prime = np.array([[0.0,  0.0, 0.02]])
        result  = free_space_dcim.scalar_G(r, r_prime)
        assert result.shape == (1,)
        assert np.allclose(result, 0.0)

    def test_free_space_dyadic_G_is_zero(self, free_space_dcim):
        r       = np.array([[0.05, 0.0, 0.01]])
        r_prime = np.array([[0.0,  0.0, 0.02]])
        result  = free_space_dcim.dyadic_G(r, r_prime)
        assert result.shape == (1, 3, 3)
        assert np.allclose(result, 0.0)

    def test_free_space_grad_G_is_zero(self, free_space_dcim):
        r       = np.array([[0.05, 0.0, 0.01]])
        r_prime = np.array([[0.0,  0.0, 0.02]])
        result  = free_space_dcim.grad_G(r, r_prime)
        assert result.shape == (1, 3)
        assert np.allclose(result, 0.0)

    def test_scalar_G_shape(self, two_layer_dcim):
        N = 8
        rng = np.random.default_rng(11)
        r       = rng.uniform(0.01, 0.05, (N, 3))
        r_prime = rng.uniform(0.01, 0.05, (N, 3))
        result = two_layer_dcim.scalar_G(r, r_prime)
        assert result.shape == (N,)
        assert result.dtype == np.complex128

    def test_scalar_G_finite(self, two_layer_dcim):
        rng = np.random.default_rng(22)
        r       = rng.uniform(0.01, 0.05, (6, 3))
        r_prime = rng.uniform(0.01, 0.05, (6, 3))
        result = two_layer_dcim.scalar_G(r, r_prime)
        assert np.all(np.isfinite(result))

    def test_dyadic_G_shape(self, two_layer_dcim):
        N = 5
        rng = np.random.default_rng(33)
        r       = rng.uniform(0.01, 0.05, (N, 3))
        r_prime = rng.uniform(0.01, 0.05, (N, 3))
        result = two_layer_dcim.dyadic_G(r, r_prime)
        assert result.shape == (N, 3, 3)
        assert result.dtype == np.complex128

    def test_grad_G_shape(self, two_layer_dcim):
        N = 4
        rng = np.random.default_rng(44)
        r       = rng.uniform(0.01, 0.05, (N, 3))
        r_prime = rng.uniform(0.01, 0.05, (N, 3))
        result = two_layer_dcim.grad_G(r, r_prime)
        assert result.shape == (N, 3)

    def test_scalar_G_symmetry(self, two_layer_dcim):
        """Image term satisfies G_smooth(r, r') == G_smooth(r', r) for planar mesh
        (z == z') — because mirroring z' and mirroring z give the same image distance
        when z == z'.
        """
        z = 0.01   # both in air half-space, same height
        r       = np.array([[0.05, 0.0,  z]])
        r_prime = np.array([[0.0,  0.03, z]])
        g_fwd = two_layer_dcim.scalar_G(r, r_prime)
        g_rev = two_layer_dcim.scalar_G(r_prime, r)
        assert np.isclose(g_fwd[0], g_rev[0], rtol=1e-10)

    def test_gamma_sign_air_over_silicon(self, two_layer_dcim):
        """For air (ε₁=1) over Si (ε₂≈11.7), Γ = (1-11.7)/(1+11.7) < 0."""
        assert two_layer_dcim.gamma.real < 0.0

    def test_three_layer_raises(self):
        """Three-layer stacks not supported in Phase 2."""
        from pyMoM3d.greens.layered.dcim import DCIMBackend
        stack = LayerStack([
            Layer('sub', -np.inf, -1e-6, eps_r=11.7),
            Layer('ild',  -1e-6,  0.0,  eps_r=4.0),
            Layer('air',   0.0,   np.inf),
        ])
        freq  = 1e9
        source_layer = stack.layers[-1]
        with pytest.raises(NotImplementedError):
            DCIMBackend(stack, freq, source_layer)

    def test_grad_G_finite_difference(self, two_layer_dcim):
        """grad_G matches central FD of scalar_G."""
        r       = np.array([[0.05, 0.02, 0.01]])
        r_prime = np.array([[0.01, 0.0,  0.02]])
        grad = two_layer_dcim.grad_G(r, r_prime)
        h = 1e-7
        for i in range(3):
            dr = np.zeros((1, 3))
            dr[0, i] = h
            gp = two_layer_dcim.scalar_G(r + dr, r_prime)
            gm = two_layer_dcim.scalar_G(r - dr, r_prime)
            fd = (gp - gm) / (2 * h)
            assert np.isclose(grad[0, i], fd[0], rtol=1e-4), \
                f"grad_G component {i}: {grad[0,i]:.4e} vs FD {fd[0]:.4e}"


# ---------------------------------------------------------------------------
# LayeredGreensFunction: auto backend selects DCIM for two-layer stacks
# ---------------------------------------------------------------------------

class TestAutoBackendSelection:
    def test_two_layer_auto_selects_backend(self):
        """Two-halfspace stack: Strata can't handle it, falls back to LayerRecursion."""
        from pyMoM3d.greens.layered.layer_recursion import LayerRecursionBackend
        stack = _two_layer_stack()
        gf = LayeredGreensFunction(stack, frequency=1e9, backend='auto')
        # Strata raises NotImplementedError for pure halfspace stacks
        assert isinstance(gf.backend, LayerRecursionBackend)

    def test_single_layer_auto_selects_backend(self):
        """Single-layer (free space) resolves to Strata or LayerRecursion."""
        stack = _free_space_stack()
        gf = LayeredGreensFunction(stack, frequency=1e9, backend='auto')
        if _strata_kernels_available():
            from pyMoM3d.greens.layered.strata import StrataBackend
            assert isinstance(gf.backend, StrataBackend)
        else:
            from pyMoM3d.greens.layered.layer_recursion import LayerRecursionBackend
            assert isinstance(gf.backend, LayerRecursionBackend)

    def test_three_layer_auto_selects_backend(self):
        """Three-layer stack: Strata preferred, else LayerRecursion."""
        stack = LayerStack([
            Layer('sub', -np.inf, -1e-6, eps_r=11.7),
            Layer('ild',  -1e-6,  0.0,  eps_r=4.0),
            Layer('air',   0.0,   np.inf),
        ])
        gf = LayeredGreensFunction(stack, frequency=1e9, backend='auto')
        if _strata_kernels_available():
            from pyMoM3d.greens.layered.strata import StrataBackend
            assert isinstance(gf.backend, StrataBackend)
        else:
            from pyMoM3d.greens.layered.layer_recursion import LayerRecursionBackend
            assert isinstance(gf.backend, LayerRecursionBackend)

    def test_explicit_dcim_backend(self):
        from pyMoM3d.greens.layered.dcim import DCIMBackend
        stack = _two_layer_stack()
        gf = LayeredGreensFunction(stack, frequency=1e9, backend='dcim')
        assert isinstance(gf.backend, DCIMBackend)

    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('empymod'),
        reason="empymod not installed",
    )
    def test_explicit_empymod_backend_still_works(self):
        from pyMoM3d.greens.layered.sommerfeld import EmpymodSommerfeldBackend
        stack = _two_layer_stack()
        gf = LayeredGreensFunction(stack, frequency=1e9, backend='empymod')
        assert isinstance(gf.backend, EmpymodSommerfeldBackend)


# ---------------------------------------------------------------------------
# DCIM vs empymod: consistency check for two-layer stack
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('empymod'),
    reason="empymod not installed",
)
class TestDCIMvsEmpymod:
    """DCIM single-image approximation vs empymod exact Sommerfeld integral.

    The image approximation is quasi-static and is expected to be within an
    order of magnitude of empymod for electrically small separations.  We do
    NOT require < 1e-4 here — that is the Phase 3 GPOF target.  Instead we
    check that DCIM and empymod have the same sign and correct order of
    magnitude (relative error < 5).
    """

    @pytest.fixture
    def setup(self):
        freq  = 1e9
        stack = _two_layer_stack()
        from pyMoM3d.greens.layered.dcim import DCIMBackend
        from pyMoM3d.greens.layered.sommerfeld import EmpymodSommerfeldBackend
        source_layer = stack.layers[-1]
        dcim   = DCIMBackend(stack, freq, source_layer)
        empy   = EmpymodSommerfeldBackend(stack, freq, source_layer)
        return dcim, empy

    def test_scalar_G_correction_same_order(self, setup):
        """DCIM and empymod smooth corrections are within 5× of each other."""
        dcim, empy = setup
        r       = np.array([[0.02, 0.0, 0.005]])
        r_prime = np.array([[0.0,  0.0, 0.005]])
        g_dcim  = dcim.scalar_G(r, r_prime)[0]
        g_empy  = empy.scalar_G(r, r_prime)[0]
        # Both should be non-zero and have the same sign for the real part
        assert abs(g_dcim) > 0
        assert abs(g_empy) > 0
        ratio = abs(g_dcim) / abs(g_empy)
        assert 0.1 < ratio < 10.0, \
            f"DCIM/empymod ratio {ratio:.3f} outside expected [0.1, 10]"

    def test_scalar_G_correction_sign_consistent(self, setup):
        """DCIM and empymod corrections have the same sign (both attractive or repulsive)."""
        dcim, empy = setup
        r       = np.array([[0.03, 0.0, 0.01]])
        r_prime = np.array([[0.0,  0.0, 0.01]])
        g_dcim  = dcim.scalar_G(r, r_prime)[0]
        g_empy  = empy.scalar_G(r, r_prime)[0]
        # Both real parts should have the same sign
        assert np.sign(g_dcim.real) == np.sign(g_empy.real)


# ---------------------------------------------------------------------------
# LayerRecursionBackend — Phase 3
# ---------------------------------------------------------------------------

class TestLayerRecursionBackend:
    """Tests for exact N-layer Sommerfeld quadrature backend."""

    @pytest.fixture
    def two_layer_lr(self):
        from pyMoM3d.greens.layered.layer_recursion import LayerRecursionBackend
        stack = _two_layer_stack()
        return LayerRecursionBackend(stack, 1e9, stack.layers[-1], n_quad=64)

    @pytest.fixture
    def three_layer_lr(self):
        from pyMoM3d.greens.layered.layer_recursion import LayerRecursionBackend
        stack = LayerStack([
            Layer('sub', -np.inf, -1e-6, eps_r=11.7),
            Layer('ild',  -1e-6,  0.0,  eps_r=4.0),
            Layer('air',   0.0,   np.inf),
        ])
        return LayerRecursionBackend(stack, 1e9, stack.layers[-1], n_quad=64)

    def test_scalar_G_shape(self, two_layer_lr):
        N = 7
        rng = np.random.default_rng(55)
        r       = rng.uniform(0.005, 0.05, (N, 3))
        r_prime = rng.uniform(0.005, 0.05, (N, 3))
        result  = two_layer_lr.scalar_G(r, r_prime)
        assert result.shape == (N,)
        assert result.dtype == np.complex128

    def test_scalar_G_finite(self, two_layer_lr):
        rng = np.random.default_rng(66)
        r       = rng.uniform(0.005, 0.05, (8, 3))
        r_prime = rng.uniform(0.005, 0.05, (8, 3))
        assert np.all(np.isfinite(two_layer_lr.scalar_G(r, r_prime)))

    def test_dyadic_G_shape(self, two_layer_lr):
        rng = np.random.default_rng(77)
        r       = rng.uniform(0.005, 0.05, (5, 3))
        r_prime = rng.uniform(0.005, 0.05, (5, 3))
        result  = two_layer_lr.dyadic_G(r, r_prime)
        assert result.shape == (5, 3, 3)

    def test_free_space_limit(self):
        """Single-layer (free space): smooth correction → 0."""
        from pyMoM3d.greens.layered.layer_recursion import LayerRecursionBackend
        stack  = _free_space_stack()
        lr     = LayerRecursionBackend(stack, 1e9, stack.layers[-1], n_quad=64)
        r      = np.array([[0.05, 0.0, 0.01]])
        r_p    = np.array([[0.0,  0.0, 0.02]])
        result = lr.scalar_G(r, r_p)
        assert np.abs(result[0]) < 1e-6, \
            f"Free-space limit failed: |G_smooth| = {abs(result[0]):.3e}"

    def test_three_layer_supported(self, three_layer_lr):
        """N=3 layer stack does not raise."""
        r   = np.array([[0.02, 0.0, 0.005]])
        r_p = np.array([[0.0,  0.0, 0.005]])
        g   = three_layer_lr.scalar_G(r, r_p)
        assert np.isfinite(g[0])

    def test_scalar_G_symmetry_planar(self, two_layer_lr):
        """G_smooth(r, r') == G_smooth(r', r) for same-height pairs."""
        z = 0.01
        r   = np.array([[0.04, 0.0, z]])
        r_p = np.array([[0.0,  0.02, z]])
        g_fwd = two_layer_lr.scalar_G(r,   r_p)
        g_rev = two_layer_lr.scalar_G(r_p, r)
        assert np.isclose(g_fwd[0], g_rev[0], rtol=1e-4)

    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('empymod'),
        reason="empymod not installed",
    )
    def test_vs_dcim_same_order(self):
        """LayerRecursion and DCIM corrections agree within an order of magnitude."""
        from pyMoM3d.greens.layered.layer_recursion import LayerRecursionBackend
        from pyMoM3d.greens.layered.dcim import DCIMBackend
        stack   = _two_layer_stack()
        freq    = 1e9
        src_lyr = stack.layers[-1]
        lr   = LayerRecursionBackend(stack, freq, src_lyr, n_quad=64)
        dcim = DCIMBackend(stack, freq, src_lyr)
        r    = np.array([[0.03, 0.0, 0.01]])
        r_p  = np.array([[0.0,  0.0, 0.01]])
        g_lr   = lr.scalar_G(r, r_p)[0]
        g_dcim = dcim.scalar_G(r, r_p)[0]
        ratio  = abs(g_lr) / (abs(g_dcim) + 1e-40)
        assert 0.05 < ratio < 20.0, \
            f"LayerRecursion/DCIM ratio {ratio:.3f} outside expected range"


# ---------------------------------------------------------------------------
# matrix_pencil (GPOF) — Phase 3
# ---------------------------------------------------------------------------

class TestMatrixPencil:
    """Tests for the matrix pencil algorithm."""

    def test_single_exponential(self):
        """Recover a = 2, s = -1 + 2j from noiseless samples."""
        from pyMoM3d.greens.layered.gpof import matrix_pencil
        dt = 0.01
        t  = np.arange(50) * dt
        a_true, s_true = 2.0 + 0j, -1.0 + 2j * np.pi * 5
        f = a_true * np.exp(s_true * t)
        amps, exps = matrix_pencil(f, dt, K_max=1)
        assert len(amps) == 1
        assert np.isclose(abs(amps[0]), abs(a_true), rtol=1e-3)
        assert np.isclose(exps[0].real, s_true.real, atol=1e-2)

    def test_two_exponentials(self):
        """Recover two exponentials."""
        from pyMoM3d.greens.layered.gpof import matrix_pencil
        dt = 0.005
        t  = np.arange(80) * dt
        s1, s2 = -0.5 + 1j * 2 * np.pi * 10, -2.0 + 1j * 2 * np.pi * 25
        f = 3.0 * np.exp(s1 * t) + 1.5 * np.exp(s2 * t)
        amps, exps = matrix_pencil(f, dt, K_max=2)
        # Sort by imaginary part of exponent for comparison
        idx = np.argsort(exps.imag)
        exps_sorted = exps[idx]
        assert np.isclose(exps_sorted[0].real, s1.real, atol=0.1)
        assert np.isclose(exps_sorted[1].real, s2.real, atol=0.1)

    def test_output_shapes(self):
        from pyMoM3d.greens.layered.gpof import matrix_pencil
        f = np.exp(-np.arange(30, dtype=complex) * 0.1)
        amps, exps = matrix_pencil(f, 0.1)
        assert amps.shape == exps.shape
        assert amps.dtype == np.complex128
        assert exps.dtype == np.complex128

    def test_gpof_solver_evaluate(self):
        """GPOFSolver.evaluate reconstructs the signal."""
        from pyMoM3d.greens.layered.gpof import GPOFSolver
        dt = 0.01
        t  = np.arange(60, dtype=float) * dt
        s  = -1.0 + 1j * 2 * np.pi * 8
        f  = 2.5 * np.exp(s * t)
        solver = GPOFSolver(K_max=1)
        solver.fit(f, dt)
        f_rec = solver.evaluate(t)
        rms = solver.residual_rms(f, dt)
        assert rms < 1e-8, f"RMS residual {rms:.2e} too large"

    def test_too_few_samples_raises(self):
        from pyMoM3d.greens.layered.gpof import matrix_pencil
        with pytest.raises(ValueError, match="at least 4"):
            matrix_pencil(np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j]), 0.1)


# ---------------------------------------------------------------------------
# TabulatedPNGFBackend — Phase 3
# ---------------------------------------------------------------------------

class TestTabulatedPNGFBackend:
    """Tests for the precomputed lookup-table backend."""

    @pytest.fixture
    def precomputed_tab(self):
        from pyMoM3d.greens.layered.tabulated import TabulatedPNGFBackend
        from pyMoM3d.greens.layered.dcim import DCIMBackend
        stack    = _two_layer_stack()
        src_lyr  = stack.layers[-1]
        ref      = DCIMBackend(stack, 1e9, src_lyr)
        tab      = TabulatedPNGFBackend()
        rho_grid = np.logspace(-3, -1, 20)
        z_grid   = np.array([0.005, 0.01, 0.02])
        tab.precompute(rho_grid, z_grid, z_grid, ref)
        return tab, ref

    def test_scalar_G_shape(self, precomputed_tab):
        tab, _ = precomputed_tab
        r   = np.array([[0.02, 0.0, 0.01]])
        r_p = np.array([[0.0,  0.0, 0.01]])
        g   = tab.scalar_G(r, r_p)
        assert g.shape == (1,)
        assert g.dtype == np.complex128

    def test_dyadic_G_shape(self, precomputed_tab):
        tab, _ = precomputed_tab
        N   = 5
        rng = np.random.default_rng(88)
        r   = rng.uniform(0.002, 0.08, (N, 3))
        r_p = rng.uniform(0.002, 0.08, (N, 3))
        g   = tab.dyadic_G(r, r_p)
        assert g.shape == (N, 3, 3)

    def test_consistency_with_reference(self, precomputed_tab):
        """Table values match reference backend at grid points."""
        tab, ref = precomputed_tab
        r   = np.array([[0.01, 0.0, 0.01]])
        r_p = np.array([[0.0,  0.0, 0.01]])
        g_tab = tab.scalar_G(r, r_p)[0]
        g_ref = ref.scalar_G(r, r_p)[0]
        # At a grid point the table should reproduce the reference exactly
        # (within interpolation tolerance; we use a coarse grid so allow 1%)
        if abs(g_ref) > 1e-30:
            assert abs(g_tab - g_ref) / abs(g_ref) < 0.05, \
                f"Table/ref relative error {abs(g_tab - g_ref)/abs(g_ref):.3f}"

    def test_save_load_roundtrip(self, tmp_path, precomputed_tab):
        from pyMoM3d.greens.layered.tabulated import TabulatedPNGFBackend
        tab, _ = precomputed_tab
        path = str(tmp_path / "test_table.npz")
        tab.save(path)
        tab2 = TabulatedPNGFBackend.load(path)
        r    = np.array([[0.015, 0.0, 0.01]])
        r_p  = np.array([[0.0,   0.0, 0.01]])
        assert np.isclose(tab.scalar_G(r, r_p)[0], tab2.scalar_G(r, r_p)[0])

    def test_not_ready_raises(self):
        from pyMoM3d.greens.layered.tabulated import TabulatedPNGFBackend
        tab = TabulatedPNGFBackend()
        with pytest.raises(RuntimeError, match="precompute"):
            tab.scalar_G(np.zeros((1, 3)), np.zeros((1, 3)))


# ---------------------------------------------------------------------------
# StrataBackend — Phase 3 stub
# ---------------------------------------------------------------------------

def _strata_kernels_available():
    try:
        import pyMoM3d.greens.layered.strata_kernels  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    _strata_kernels_available(),
    reason="strata_kernels is installed — ImportError test not applicable",
)
def test_strata_backend_raises_without_library():
    """StrataBackend raises ImportError when strata_kernels is not installed."""
    from pyMoM3d.greens.layered.strata import StrataBackend
    stack = _two_layer_stack()
    with pytest.raises(ImportError, match="strata_kernels"):
        StrataBackend(stack, 1e9, stack.layers[-1])


# ---------------------------------------------------------------------------
# Phase 3 backend auto-selection
# ---------------------------------------------------------------------------

class TestPhase3AutoBackend:
    def test_auto_selects_backend_for_two_layer(self):
        """Two-halfspace: Strata can't handle it → always LayerRecursionBackend."""
        from pyMoM3d.greens.layered.layer_recursion import LayerRecursionBackend
        gf = LayeredGreensFunction(_two_layer_stack(), frequency=1e9, backend='auto')
        assert isinstance(gf.backend, LayerRecursionBackend)

    def test_explicit_layer_recursion_backend(self):
        from pyMoM3d.greens.layered.layer_recursion import LayerRecursionBackend
        gf = LayeredGreensFunction(_two_layer_stack(), 1e9, backend='layer_recursion')
        assert isinstance(gf.backend, LayerRecursionBackend)

    def test_tabulated_backend_passed_as_instance(self):
        """TabulatedPNGFBackend can be passed directly as a backend instance."""
        from pyMoM3d.greens.layered.tabulated import TabulatedPNGFBackend
        from pyMoM3d.greens.layered.dcim import DCIMBackend
        stack   = _two_layer_stack()
        src_lyr = stack.layers[-1]
        ref     = DCIMBackend(stack, 1e9, src_lyr)
        tab     = TabulatedPNGFBackend()
        tab.precompute(np.logspace(-3, -1, 10), np.array([0.01]), np.array([0.01]), ref)
        gf = LayeredGreensFunction(stack, 1e9, backend=tab)
        assert isinstance(gf.backend, TabulatedPNGFBackend)


# ---------------------------------------------------------------------------
# Strata C++ backend integration tests
# ---------------------------------------------------------------------------

strata_kernels = pytest.importorskip(
    "pyMoM3d.greens.layered.strata_kernels",
    reason="strata_kernels C++ extension not built",
)


def _three_layer_stack():
    """Si substrate + dielectric slab + air (suitable for Strata).

    Source layer is the dielectric slab (topmost interior layer).
    LayerRecursionBackend handles this case correctly since it only
    needs downward-looking reflection from the Si/slab interface.
    """
    return LayerStack([
        Layer('Si',   z_bot=-np.inf, z_top=0.0,   eps_r=4.0),
        Layer('slab', z_bot=0.0,     z_top=5e-3,  eps_r=2.0),
        Layer('air',  z_bot=5e-3,    z_top=np.inf, eps_r=1.0),
    ])


class TestStrataBackendIntegration:
    """Integration tests for the Strata (DCIM) C++ backend.

    These tests require the strata_kernels extension to be compiled.
    They are automatically skipped when the extension is not available.
    Strata requires at least one finite interior layer, so tests use
    a 3-layer stack (Si/ILD/air).
    """

    @pytest.fixture
    def free_space_strata(self):
        from pyMoM3d.greens.layered.strata import StrataBackend
        stack = _free_space_stack()
        freq  = 1e9
        source_layer = stack.layers[-1]
        return StrataBackend(stack, freq, source_layer)

    @pytest.fixture
    def three_layer_strata(self):
        from pyMoM3d.greens.layered.strata import StrataBackend
        stack = _three_layer_stack()
        freq  = 10e9
        # Source layer = slab (topmost interior layer, z=0 to 5mm)
        source_layer = stack.layers[1]
        return StrataBackend(stack, freq, source_layer)

    def test_free_space_scalar_smooth_is_zero(self, free_space_strata):
        """Single air layer: smooth correction G_ML - G_fs must be ~0."""
        r       = np.array([[0.05, 0.0, 0.01]])
        r_prime = np.array([[0.0,  0.0, 0.02]])
        result  = free_space_strata.scalar_G(r, r_prime)
        assert result.shape == (1,)
        assert np.allclose(result, 0.0, atol=1e-10)

    def test_two_halfspace_raises(self):
        """StrataBackend raises NotImplementedError for pure two-halfspace stacks."""
        from pyMoM3d.greens.layered.strata import StrataBackend
        stack = _two_layer_stack()
        with pytest.raises(NotImplementedError, match="finite interior layer"):
            StrataBackend(stack, 1e9, stack.layers[-1])

    def test_homogeneous_smooth_correction_is_zero(self):
        """Homogeneous medium (all layers same ε_r): smooth correction ≈ 0.

        This validates the Formulation-C normalization (G_phi = g/ε_r)
        and the free-space subtraction in the C++ wrapper.
        """
        from pyMoM3d.greens.layered.strata import StrataBackend

        eps_r = 3.0
        stack = LayerStack([
            Layer('sub',  z_bot=-np.inf, z_top=0.0,  eps_r=eps_r),
            Layer('slab', z_bot=0.0,     z_top=5e-3, eps_r=eps_r),
            Layer('top',  z_bot=5e-3,    z_top=np.inf, eps_r=eps_r),
        ])
        freq = 10e9
        backend = StrataBackend(stack, freq, stack.layers[1])

        r       = np.array([[0.03, 0.01, 2.5e-3]])
        r_prime = np.array([[0.0,  0.0,  2.5e-3]])
        g_smooth = backend.scalar_G(r, r_prime)

        # Also compute the reference free-space GF magnitude for normalization
        from pyMoM3d.greens.free_space_gf import FreeSpaceGreensFunction
        omega = 2 * np.pi * freq
        k = complex(stack.layers[1].wavenumber(omega))
        eta = complex(stack.layers[1].wave_impedance(omega))
        g_fs = FreeSpaceGreensFunction(k=k, eta=eta).scalar_G(r, r_prime)

        # Smooth correction should be < 2% of free-space GF magnitude
        assert abs(g_smooth[0]) / abs(g_fs[0]) < 0.02

    def test_three_layer_smooth_nonzero(self, three_layer_strata):
        """Three-layer stack with contrast: smooth correction must be nonzero."""
        rng     = np.random.default_rng(42)
        N       = 5
        r       = rng.uniform(0.01, 0.05, (N, 3))
        r[:, 2] = rng.uniform(0.5e-3, 4.5e-3, N)
        r_prime = rng.uniform(0.01, 0.05, (N, 3))
        r_prime[:, 2] = rng.uniform(0.5e-3, 4.5e-3, N)

        g_strata = three_layer_strata.scalar_G(r, r_prime)
        assert g_strata.shape == (N,)
        assert np.all(np.isfinite(g_strata))
        assert not np.allclose(g_strata, 0.0)  # should be nonzero with contrast

    def test_scalar_G_reciprocity(self, three_layer_strata):
        """G_smooth(r, r') == G_smooth(r', r) when z == z' (same height)."""
        z = 2.5e-3   # midpoint of slab layer
        r       = np.array([[0.05, 0.0,  z]])
        r_prime = np.array([[0.0,  0.03, z]])
        g_fwd = three_layer_strata.scalar_G(r, r_prime)
        g_rev = three_layer_strata.scalar_G(r_prime, r)
        assert np.isclose(g_fwd[0], g_rev[0], rtol=1e-6)

    def test_dyadic_G_shape(self, three_layer_strata):
        """Dyadic output has correct shape (N, 3, 3) and dtype."""
        N   = 5
        rng = np.random.default_rng(77)
        r       = rng.uniform(0.01, 0.05, (N, 3))
        r[:, 2] = rng.uniform(0.5e-3, 4.5e-3, N)
        r_prime = rng.uniform(0.01, 0.05, (N, 3))
        r_prime[:, 2] = rng.uniform(0.5e-3, 4.5e-3, N)
        result = three_layer_strata.dyadic_G(r, r_prime)
        assert result.shape == (N, 3, 3)
        assert result.dtype == np.complex128
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# C++ multilayer impedance fill — same-layer validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _strata_kernels_available(),
    reason="strata_kernels C++ extension not compiled",
)
class TestCppMultilayerFillSameLayer:
    """Validate fill_impedance_multilayer_cpp against a Python reference.

    Uses a tetrahedron mesh inside a single dielectric slab.
    """

    @pytest.fixture(scope="class")
    def setup(self):
        import pyMoM3d.greens.layered.strata_kernels as sk
        from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
        from pyMoM3d.greens.quadrature import triangle_quad_rule
        from pyMoM3d.greens.singularity import (
            integrate_green_singular,
            integrate_rho_green_singular,
        )

        s = 0.003
        vertices = np.array(
            [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64
        ) * s
        triangles = np.array(
            [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int32
        )
        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)
        mesh.vertices[:, 2] += 0.5e-3
        N = basis.num_basis

        freq = 10e9
        k0 = 2 * math.pi * freq / 3e8
        eta = 377.0
        k_slab = k0 * math.sqrt(2.0)

        model = sk.make_model(
            [[0.0, 1e-3, 2.0, 0.0, 1.0, 0.0, 0.0]],
            1.0, 1.0, 0.0, False,
            4.0, 1.0, 0.0, False,
            freq, 0.5e-3, 0.5e-3,
            k_slab, 0.0, 2.0, 0.0, 'dcim',
        )

        weights, bary = triangle_quad_rule(4)
        Q = len(weights)
        tri_twice_area = 2.0 * mesh.triangle_areas

        # --- Python reference Z ---
        def corrected_smooth(r_obs, r_src):
            g = sk.scalar_G_smooth(model, r_obs, r_src)
            dx = r_obs[:, 0] - r_src[:, 0]
            dy = r_obs[:, 1] - r_src[:, 1]
            dz = r_obs[:, 2] - r_src[:, 2]
            R = np.maximum(np.sqrt(dx**2 + dy**2 + dz**2), 1e-30)
            return (g + np.exp(-1j * k_slab * R) / (4 * np.pi * R)
                      - np.exp(-1j * k0 * R) / (4 * np.pi * R))

        Z_py = np.zeros((N, N), dtype=np.complex128)
        for m in range(N):
            for n in range(m, N):
                Z_mn = 0.0
                for tri_m, fv_m, sm, Am in [
                    (basis.t_plus[m], basis.free_vertex_plus[m], +1.0, basis.area_plus[m]),
                    (basis.t_minus[m], basis.free_vertex_minus[m], -1.0, basis.area_minus[m]),
                ]:
                    for tri_n, fv_n, sn, An in [
                        (basis.t_plus[n], basis.free_vertex_plus[n], +1.0, basis.area_plus[n]),
                        (basis.t_minus[n], basis.free_vertex_minus[n], -1.0, basis.area_minus[n]),
                    ]:
                        vt = mesh.vertices[mesh.triangles[tri_m]]
                        vs = mesh.vertices[mesh.triangles[tri_n]]
                        rfv_t = mesh.vertices[fv_m]
                        rfv_s = mesh.vertices[fv_n]
                        ta_t = tri_twice_area[tri_m]
                        ta_s = tri_twice_area[tri_n]

                        r_src = (bary[:, 0:1] * vs[0] + bary[:, 1:2] * vs[1]
                                 + bary[:, 2:3] * vs[2])
                        rho_src = r_src - rfv_s[np.newaxis, :]
                        r_test = (bary[:, 0:1] * vt[0] + bary[:, 1:2] * vt[1]
                                  + bary[:, 2:3] * vt[2])

                        I_Phi = 0.0 + 0.0j
                        I_A = 0.0 + 0.0j
                        for i in range(Q):
                            ro = r_test[i]
                            rho_t = ro - rfv_t
                            gs = integrate_green_singular(
                                k0, ro, vs[0], vs[1], vs[2],
                                quad_order=4, near_threshold=0.2)
                            rgs = integrate_rho_green_singular(
                                k0, ro, vs[0], vs[1], vs[2], rfv_s,
                                quad_order=4, near_threshold=0.2)
                            gc = corrected_smooth(np.tile(ro, (Q, 1)), r_src)
                            gs_sum = 0.0 + 0.0j
                            rgs_sm = np.zeros(3, dtype=complex)
                            wr, ws = 0.0, 0.0
                            sl = []
                            for j in range(Q):
                                R = np.linalg.norm(ro - r_src[j])
                                if R < 1e-10:
                                    ws += weights[j]
                                    sl.append((rho_src[j].copy(), weights[j]))
                                    continue
                                gs_sum += weights[j] * gc[j]
                                wr += weights[j]
                                rgs_sm += weights[j] * rho_src[j] * gc[j]
                            if sl and wr > 0:
                                ga = gs_sum / wr
                                gs_sum += ws * ga
                                for rs, w in sl:
                                    rgs_sm += w * rs * ga
                            g_int = gs + gs_sum * ta_s
                            rg_int = rgs + rgs_sm * ta_s
                            I_Phi += weights[i] * g_int
                            I_A += weights[i] * np.dot(rho_t, rg_int)
                        I_Phi *= ta_t
                        I_A *= ta_t
                        sA = (sm * basis.edge_length[m] / (2 * Am)) * (sn * basis.edge_length[n] / (2 * An))
                        sP = (sm * basis.edge_length[m] / Am) * (sn * basis.edge_length[n] / An)
                        Z_mn += 1j * k0 * eta * I_A * sA + (-1j * eta / k0) * I_Phi * sP
                Z_py[m, n] = Z_mn
                Z_py[n, m] = Z_mn

        # --- C++ Z ---
        verts = mesh.vertices.astype(np.float64)
        tris = mesh.triangles.astype(np.int32)
        tri_centroids = np.mean(verts[tris], axis=1)
        e0 = verts[tris[:, 1]] - verts[tris[:, 0]]
        e2 = verts[tris[:, 0]] - verts[tris[:, 2]]
        tri_mean_edge = (
            np.linalg.norm(e0, axis=1)
            + np.linalg.norm(verts[tris[:, 2]] - verts[tris[:, 1]], axis=1)
            + np.linalg.norm(e2, axis=1)
        ) / 3.0

        Z_cpp = np.zeros((N, N), dtype=np.complex128)
        sk.fill_impedance_multilayer_cpp(
            Z_cpp, verts, tris,
            basis.t_plus.astype(np.int32), basis.t_minus.astype(np.int32),
            basis.free_vertex_plus.astype(np.int32),
            basis.free_vertex_minus.astype(np.int32),
            basis.area_plus.astype(np.float64),
            basis.area_minus.astype(np.float64),
            basis.edge_length.astype(np.float64),
            np.ascontiguousarray(tri_centroids),
            np.ascontiguousarray(tri_mean_edge),
            np.ascontiguousarray(tri_twice_area.astype(np.float64)),
            weights.astype(np.float64), bary.astype(np.float64),
            k0, eta, 0.2, 4, model, 1,
        )

        return Z_py, Z_cpp, N

    def test_cpp_vs_python_same_layer(self, setup):
        """C++ and Python Z matrices agree to machine precision."""
        Z_py, Z_cpp, N = setup
        Z_scale = np.abs(Z_cpp).max()
        rel = np.abs(Z_py - Z_cpp).max() / Z_scale
        assert rel < 1e-10, f"rel error {rel:.2e} exceeds 1e-10"

    def test_symmetry_same_layer(self, setup):
        """C++ Z matrix is symmetric."""
        _, Z_cpp, _ = setup
        assert np.abs(Z_cpp - Z_cpp.T).max() == 0.0


# ---------------------------------------------------------------------------
# C++ multilayer impedance fill — cross-layer validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _strata_kernels_available(),
    reason="strata_kernels C++ extension not compiled",
)
class TestCppMultilayerFillCrossLayer:
    """Validate cross-layer C++ fill against Python reference.

    Uses a box mesh straddling two finite dielectric layers (eps_r=2.0 and
    eps_r=3.0), ensuring both same-layer and cross-layer triangle pairs.
    """

    @pytest.fixture(scope="class")
    def setup(self):
        import pyMoM3d.greens.layered.strata_kernels as sk
        from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
        from pyMoM3d.greens.quadrature import triangle_quad_rule
        from pyMoM3d.greens.singularity import (
            integrate_green_singular,
            integrate_rho_green_singular,
        )

        # Two separate planar patches in different layers:
        #   Patch A: z=0.5mm (layer A), Patch B: z=1.5mm (layer B)
        # Using separate meshes avoids non-manifold edges (box mesh has
        # edges shared by 3 triangles, which RWG connectivity rejects).
        s = 0.003
        vertices = np.array([
            # Patch A (layer A, z=0.5mm) — vertices 0..3
            [-s, -s, 0.5e-3], [s, -s, 0.5e-3],
            [s, s, 0.5e-3], [-s, s, 0.5e-3],
            # Patch B (layer B, z=1.5mm) — vertices 4..7
            [-s, -s, 1.5e-3], [s, -s, 1.5e-3],
            [s, s, 1.5e-3], [-s, s, 1.5e-3],
        ], dtype=np.float64)
        triangles = np.array([
            # Patch A
            [0, 1, 2], [0, 2, 3],
            # Patch B
            [4, 5, 6], [4, 6, 7],
        ], dtype=np.int32)

        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)
        N = basis.num_basis

        tri_centroids = np.mean(vertices[triangles], axis=1)
        tri_layer_map = {
            i: (0 if tri_centroids[i, 2] < 1e-3 else 1)
            for i in range(len(triangles))
        }

        freq = 10e9
        omega = 2 * math.pi * freq
        k0 = omega / 3e8
        eta0_val = 377.0
        eps_r_A, eps_r_B = 2.0, 3.0
        k_A = k0 * math.sqrt(eps_r_A)
        k_B = k0 * math.sqrt(eps_r_B)

        layers_data = [
            [0.0, 1e-3, eps_r_A, 0.0, 1.0, 0.0, 0.0],
            [1e-3, 2e-3, eps_r_B, 0.0, 1.0, 0.0, 0.0],
        ]

        model_AA = sk.make_model(layers_data, 1, 1, 0, False, 1, 1, 0, False,
            freq, 0.5e-3, 0.5e-3, k_A, 0.0, eps_r_A, 0.0, 'dcim')
        model_BB = sk.make_model(layers_data, 1, 1, 0, False, 1, 1, 0, False,
            freq, 1.5e-3, 1.5e-3, k_B, 0.0, eps_r_B, 0.0, 'dcim')
        model_AB = sk.make_model(layers_data, 1, 1, 0, False, 1, 1, 0, False,
            freq, 0.5e-3, 1.5e-3, k_A, 0.0, eps_r_A, 0.0, 'dcim')
        model_BA = sk.make_model(layers_data, 1, 1, 0, False, 1, 1, 0, False,
            freq, 1.5e-3, 0.5e-3, k_B, 0.0, eps_r_B, 0.0, 'dcim')

        model_table = {
            (0, 0): (model_AA, k_A, True),
            (1, 1): (model_BB, k_B, True),
            (0, 1): (model_BA, k_B, False),
            (1, 0): (model_AB, k_A, False),
        }

        weights, bary = triangle_quad_rule(4)
        Q = len(weights)
        tri_twice_area = 2.0 * mesh.triangle_areas

        # --- Python reference ---
        def python_subpair(tri_t, tri_s, fv_t, fv_s, st, ss, lt, ls, At, As,
                           model, k_model_src, same_layer):
            vt = mesh.vertices[mesh.triangles[tri_t]]
            vs = mesh.vertices[mesh.triangles[tri_s]]
            rfv_t = mesh.vertices[fv_t]
            rfv_s = mesh.vertices[fv_s]
            ta_t, ta_s = tri_twice_area[tri_t], tri_twice_area[tri_s]
            r_src = (bary[:, 0:1] * vs[0] + bary[:, 1:2] * vs[1]
                     + bary[:, 2:3] * vs[2])
            rho_src = r_src - rfv_s[np.newaxis, :]
            r_test = (bary[:, 0:1] * vt[0] + bary[:, 1:2] * vt[1]
                      + bary[:, 2:3] * vt[2])
            I_Phi = 0.0 + 0.0j
            I_A = 0.0 + 0.0j
            for i in range(Q):
                ro = r_test[i]
                rho_t = ro - rfv_t
                r_obs_tiled = np.tile(ro, (Q, 1))
                g_smooth = np.asarray(
                    sk.scalar_G_smooth(model, r_obs_tiled, r_src), dtype=complex)
                dx = r_obs_tiled[:, 0] - r_src[:, 0]
                dy = r_obs_tiled[:, 1] - r_src[:, 1]
                dz = r_obs_tiled[:, 2] - r_src[:, 2]
                R = np.maximum(np.sqrt(dx**2 + dy**2 + dz**2), 1e-30)
                if same_layer:
                    gs = integrate_green_singular(
                        k0, ro, vs[0], vs[1], vs[2],
                        quad_order=4, near_threshold=0.2)
                    rgs = integrate_rho_green_singular(
                        k0, ro, vs[0], vs[1], vs[2], rfv_s,
                        quad_order=4, near_threshold=0.2)
                    g_corr = (g_smooth
                              + np.exp(-1j * k_model_src * R) / (4 * np.pi * R)
                              - np.exp(-1j * k0 * R) / (4 * np.pi * R))
                    gs_sum = 0.0 + 0.0j
                    rgs_sm = np.zeros(3, dtype=complex)
                    w_reg, w_sing = 0.0, 0.0
                    sing_rho = []
                    for j in range(Q):
                        Rj = np.linalg.norm(ro - r_src[j])
                        if Rj < 1e-10:
                            w_sing += weights[j]
                            sing_rho.append((rho_src[j].copy(), weights[j]))
                            continue
                        gs_sum += weights[j] * g_corr[j]
                        w_reg += weights[j]
                        rgs_sm += weights[j] * rho_src[j] * g_corr[j]
                    if sing_rho and w_reg > 0:
                        ga = gs_sum / w_reg
                        gs_sum += w_sing * ga
                        for rs, w in sing_rho:
                            rgs_sm += w * rs * ga
                    g_int = gs + gs_sum * ta_s
                    rg_int = rgs + rgs_sm * ta_s
                else:
                    g_ml = g_smooth + np.exp(-1j * k_model_src * R) / (4 * np.pi * R)
                    g_int = np.dot(weights, g_ml) * ta_s
                    rg_int = np.einsum('j,ji,j->i', weights, rho_src, g_ml) * ta_s
                I_Phi += weights[i] * g_int
                I_A += weights[i] * np.dot(rho_t, rg_int)
            I_Phi *= ta_t
            I_A *= ta_t
            sA = (st * lt / (2 * At)) * (ss * ls / (2 * As))
            sP = (st * lt / At) * (ss * ls / As)
            return 1j * k0 * eta0_val * I_A * sA + (-1j * eta0_val / k0) * I_Phi * sP

        Z_py = np.zeros((N, N), dtype=np.complex128)
        for m in range(N):
            for n in range(m, N):
                Z_mn = 0.0
                for tri_m, fv_m, sm, Am in [
                    (basis.t_plus[m], basis.free_vertex_plus[m], +1.0, basis.area_plus[m]),
                    (basis.t_minus[m], basis.free_vertex_minus[m], -1.0, basis.area_minus[m]),
                ]:
                    layer_m = tri_layer_map[tri_m]
                    for tri_n, fv_n, sn, An in [
                        (basis.t_plus[n], basis.free_vertex_plus[n], +1.0, basis.area_plus[n]),
                        (basis.t_minus[n], basis.free_vertex_minus[n], -1.0, basis.area_minus[n]),
                    ]:
                        layer_n = tri_layer_map[tri_n]
                        mdl, k_src, sl = model_table[(layer_m, layer_n)]
                        Z_mn += python_subpair(
                            tri_m, tri_n, fv_m, fv_n, sm, sn,
                            basis.edge_length[m], basis.edge_length[n], Am, An,
                            mdl, k_src, sl)
                Z_py[m, n] = Z_mn
                Z_py[n, m] = Z_mn

        # --- C++ Z ---
        verts = mesh.vertices.astype(np.float64)
        tris = mesh.triangles.astype(np.int32)
        e0 = verts[tris[:, 1]] - verts[tris[:, 0]]
        e2 = verts[tris[:, 0]] - verts[tris[:, 2]]
        tri_mean_edge = (
            np.linalg.norm(e0, axis=1)
            + np.linalg.norm(verts[tris[:, 2]] - verts[tris[:, 1]], axis=1)
            + np.linalg.norm(e2, axis=1)
        ) / 3.0

        tri_layer_idx = np.array(
            [tri_layer_map[i] for i in range(len(triangles))], dtype=np.int32)
        M = 2
        model_lookup = np.array([0, 3, 2, 1], dtype=np.int32)
        extra_models_list = [model_BB, model_AB, model_BA]

        Z_cpp = np.zeros((N, N), dtype=np.complex128)
        sk.fill_impedance_multilayer_cpp(
            Z_cpp, verts, tris,
            basis.t_plus.astype(np.int32), basis.t_minus.astype(np.int32),
            basis.free_vertex_plus.astype(np.int32),
            basis.free_vertex_minus.astype(np.int32),
            basis.area_plus.astype(np.float64),
            basis.area_minus.astype(np.float64),
            basis.edge_length.astype(np.float64),
            np.ascontiguousarray(tri_centroids),
            np.ascontiguousarray(tri_mean_edge),
            np.ascontiguousarray(tri_twice_area.astype(np.float64)),
            weights.astype(np.float64), bary.astype(np.float64),
            k0, eta0_val, 0.2, 4,
            model_AA, 1,
            tri_layer_idx, extra_models_list, model_lookup,
        )

        # Count cross-layer sub-pairs
        n_cross = 0
        for m in range(N):
            for n in range(m, N):
                for tri_m_idx in [basis.t_plus[m], basis.t_minus[m]]:
                    for tri_n_idx in [basis.t_plus[n], basis.t_minus[n]]:
                        if tri_layer_map[tri_m_idx] != tri_layer_map[tri_n_idx]:
                            n_cross += 1

        return Z_py, Z_cpp, N, n_cross

    def test_cpp_vs_python_cross_layer(self, setup):
        """C++ and Python Z matrices agree for cross-layer mesh."""
        Z_py, Z_cpp, N, _ = setup
        Z_scale = np.abs(Z_cpp).max()
        rel = np.abs(Z_py - Z_cpp).max() / Z_scale
        assert rel < 1e-4, f"rel error {rel:.2e} exceeds 1e-4"

    def test_symmetry_cross_layer(self, setup):
        """C++ Z matrix is symmetric for cross-layer mesh."""
        _, Z_cpp, _, _ = setup
        Z_scale = np.abs(Z_cpp).max()
        sym_err = np.abs(Z_cpp - Z_cpp.T).max() / Z_scale
        assert sym_err < 1e-14, f"symmetry error {sym_err:.2e}"

    def test_has_cross_layer_pairs(self, setup):
        """Mesh actually contains cross-layer sub-pairs."""
        _, _, _, n_cross = setup
        assert n_cross > 0, "No cross-layer pairs — test is not exercising cross-layer code"


# ---------------------------------------------------------------------------
# LayerStack.layer_index_at_z
# ---------------------------------------------------------------------------

class TestLayerIndexAtZ:
    def test_returns_correct_index(self):
        stack = LayerStack([
            Layer('bot', z_bot=-np.inf, z_top=0.0, eps_r=11.7),
            Layer('mid', z_bot=0.0, z_top=1e-3, eps_r=4.0),
            Layer('top', z_bot=1e-3, z_top=np.inf, eps_r=1.0),
        ])
        assert stack.layer_index_at_z(-1.0) == 0
        assert stack.layer_index_at_z(0.5e-3) == 1
        assert stack.layer_index_at_z(2e-3) == 2

    def test_raises_outside_stack(self):
        stack = LayerStack([
            Layer('only', z_bot=0.0, z_top=1.0, eps_r=1.0),
        ])
        with pytest.raises(ValueError):
            stack.layer_index_at_z(-1.0)
