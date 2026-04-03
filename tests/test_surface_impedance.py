"""Tests for surface impedance boundary condition."""

import numpy as np
import pytest

from pyMoM3d import (
    create_rectangular_mesh, compute_rwg_connectivity, mu0,
)
from pyMoM3d.mom.surface_impedance import (
    ConductorProperties, build_gram_matrix, apply_surface_impedance,
)


# ---------------------------------------------------------------------------
# ConductorProperties tests
# ---------------------------------------------------------------------------

class TestConductorProperties:
    """Test surface impedance Z_s(f) = (gamma/sigma) * coth(gamma*t)."""

    def test_dc_limit(self):
        """At DC, Z_s → 1/(sigma*t) (sheet resistance)."""
        cu = ConductorProperties(sigma=5.8e7, thickness=1e-6)
        Z_s = cu.surface_impedance(0.0)
        R_sh = 1.0 / (5.8e7 * 1e-6)
        assert abs(Z_s - R_sh) < 1e-10 * R_sh

    def test_dc_sheet_resistance(self):
        """Verify dc_sheet_resistance() method."""
        cu = ConductorProperties(sigma=5.8e7, thickness=2e-6)
        R_sh = cu.dc_sheet_resistance()
        assert abs(R_sh - 1.0 / (5.8e7 * 2e-6)) < 1e-10

    def test_thick_conductor_limit(self):
        """For t >> delta, Z_s → (1+j)/(sigma*delta) = sqrt(j*omega*mu/sigma)."""
        # Use a very thick conductor
        cu = ConductorProperties(sigma=5.8e7, thickness=1.0)  # 1 meter thick
        freq = 1e9
        omega = 2 * np.pi * freq
        Z_s = cu.surface_impedance(freq)

        # Analytical thick limit
        Z_s_thick = np.sqrt(1j * omega * mu0 / 5.8e7)
        assert abs(Z_s - Z_s_thick) / abs(Z_s_thick) < 1e-6

    def test_skin_depth(self):
        """Verify skin depth calculation."""
        cu = ConductorProperties(sigma=5.8e7, thickness=1e-6)
        freq = 1e9
        delta = cu.skin_depth(freq)
        expected = np.sqrt(2.0 / (2 * np.pi * freq * mu0 * 5.8e7))
        assert abs(delta - expected) / expected < 1e-10

    def test_impedance_has_positive_real_part(self):
        """Z_s must have positive real part (dissipative)."""
        cu = ConductorProperties(sigma=5.8e7, thickness=1e-6)
        for freq in [1e6, 1e8, 1e9, 10e9, 100e9]:
            Z_s = cu.surface_impedance(freq)
            assert Z_s.real > 0, f"Re(Z_s) = {Z_s.real} at {freq/1e9} GHz"

    def test_impedance_increases_with_frequency(self):
        """|Z_s| should increase with frequency (skin effect)."""
        cu = ConductorProperties(sigma=5.8e7, thickness=10e-6)
        freqs = [1e8, 1e9, 10e9, 100e9]
        zs = [abs(cu.surface_impedance(f)) for f in freqs]
        for i in range(len(zs) - 1):
            assert zs[i + 1] > zs[i], (
                f"|Z_s| at {freqs[i+1]/1e9} GHz = {zs[i+1]:.4e} "
                f"<= {zs[i]:.4e} at {freqs[i]/1e9} GHz"
            )

    def test_thin_conductor_higher_resistance(self):
        """Thinner conductor should have higher Z_s (at least Re part)."""
        thin = ConductorProperties(sigma=5.8e7, thickness=0.5e-6)
        thick = ConductorProperties(sigma=5.8e7, thickness=5e-6)
        freq = 5e9
        assert thin.surface_impedance(freq).real > thick.surface_impedance(freq).real


# ---------------------------------------------------------------------------
# Gram matrix tests
# ---------------------------------------------------------------------------

class TestGramMatrix:
    """Test RWG Gram (mass) matrix assembly."""

    @pytest.fixture
    def simple_mesh(self):
        """Small rectangular mesh for testing."""
        mesh = create_rectangular_mesh(width=0.05, height=0.01, nx=4, ny=2)
        basis = compute_rwg_connectivity(mesh)
        return mesh, basis

    def test_gram_shape(self, simple_mesh):
        """G should be N x N."""
        mesh, basis = simple_mesh
        G = build_gram_matrix(basis, mesh)
        N = basis.num_basis
        assert G.shape == (N, N)

    def test_gram_symmetric(self, simple_mesh):
        """G must be symmetric: G_mn = G_nm."""
        mesh, basis = simple_mesh
        G = build_gram_matrix(basis, mesh)
        assert np.allclose(G, G.T, atol=1e-15)

    def test_gram_positive_diagonal(self, simple_mesh):
        """Diagonal entries G_nn = ∫|f_n|^2 dS > 0."""
        mesh, basis = simple_mesh
        G = build_gram_matrix(basis, mesh)
        for n in range(basis.num_basis):
            assert G[n, n] > 0, f"G[{n},{n}] = {G[n,n]} <= 0"

    def test_gram_positive_semidefinite(self, simple_mesh):
        """G should be positive semi-definite."""
        mesh, basis = simple_mesh
        G = build_gram_matrix(basis, mesh)
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals >= -1e-14), (
            f"Negative eigenvalue: min = {eigvals.min()}"
        )

    def test_gram_sparsity_pattern(self, simple_mesh):
        """G_mn != 0 only when basis m and n share a triangle."""
        mesh, basis = simple_mesh
        G = build_gram_matrix(basis, mesh)
        N = basis.num_basis

        # Build set of triangle-sharing basis pairs
        T = len(mesh.triangles)
        tri_basis = [set() for _ in range(T)]
        for n in range(N):
            tri_basis[int(basis.t_plus[n])].add(n)
            tri_basis[int(basis.t_minus[n])].add(n)

        sharing = set()
        for t in range(T):
            for m in tri_basis[t]:
                for n in tri_basis[t]:
                    sharing.add((m, n))

        for m in range(N):
            for n in range(N):
                if (m, n) not in sharing:
                    assert abs(G[m, n]) < 1e-15, (
                        f"G[{m},{n}] = {G[m,n]} but bases don't share a triangle"
                    )


# ---------------------------------------------------------------------------
# apply_surface_impedance tests
# ---------------------------------------------------------------------------

class TestApplySurfaceImpedance:
    """Test that surface impedance is correctly added to Z matrix."""

    def test_adds_zs_times_gram(self):
        """Z_new = Z + Z_s * G."""
        mesh = create_rectangular_mesh(width=0.05, height=0.01, nx=4, ny=2)
        basis = compute_rwg_connectivity(mesh)
        N = basis.num_basis

        Z = np.zeros((N, N), dtype=np.complex128)
        cu = ConductorProperties(sigma=5.8e7, thickness=1e-6)
        freq = 5e9
        G = build_gram_matrix(basis, mesh)
        Z_s = cu.surface_impedance(freq)

        apply_surface_impedance(Z, basis, mesh, cu, freq, gram_matrix=G)

        expected = Z_s * G
        assert np.allclose(Z, expected, atol=1e-20)

    def test_preserves_existing_z(self):
        """Surface impedance should add to existing Z, not replace it."""
        mesh = create_rectangular_mesh(width=0.05, height=0.01, nx=4, ny=2)
        basis = compute_rwg_connectivity(mesh)
        N = basis.num_basis

        Z_orig = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        Z = Z_orig.copy()
        cu = ConductorProperties(sigma=5.8e7, thickness=1e-6)
        freq = 5e9
        G = build_gram_matrix(basis, mesh)
        Z_s = cu.surface_impedance(freq)

        apply_surface_impedance(Z, basis, mesh, cu, freq, gram_matrix=G)

        expected = Z_orig + Z_s * G
        assert np.allclose(Z, expected, atol=1e-14)
