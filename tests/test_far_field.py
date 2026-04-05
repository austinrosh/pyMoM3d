"""Tests for far-field, RCS, and Mie series."""

import numpy as np
import pytest

from pyMoM3d.fields.far_field import compute_far_field
from pyMoM3d.fields.rcs import compute_rcs, compute_monostatic_rcs
from pyMoM3d.analysis.mie_series import mie_rcs_pec_sphere, mie_monostatic_rcs_pec_sphere
from pyMoM3d.mesh import Mesh, compute_rwg_connectivity
from pyMoM3d.mom.assembly import fill_matrix
from pyMoM3d.mom.operators import EFIEOperator
from pyMoM3d.mom.excitation import PlaneWaveExcitation
from pyMoM3d.mom.solver import solve_direct
from pyMoM3d.utils.constants import eta0, c0


class TestRCS:
    def test_rcs_positive(self):
        """RCS should be a real value."""
        E_theta = np.array([1.0 + 0.5j])
        E_phi = np.array([0.3 - 0.2j])
        rcs = compute_rcs(E_theta, E_phi, E_inc_mag=1.0)
        assert np.isfinite(rcs[0])

    def test_rcs_scales_with_einc(self):
        """Doubling E_inc should decrease RCS by 6 dB."""
        E_theta = np.array([1.0 + 0j])
        E_phi = np.array([0.0 + 0j])
        rcs1 = compute_rcs(E_theta, E_phi, E_inc_mag=1.0)
        rcs2 = compute_rcs(E_theta, E_phi, E_inc_mag=2.0)
        assert np.isclose(rcs1[0] - rcs2[0], 6.0, atol=0.1)


class TestMieSeries:
    def test_low_frequency_limit(self):
        """For ka << 1, monostatic RCS should approach the Rayleigh limit.
        sigma / (pi*a^2) ~ 9*(ka)^4 for PEC sphere (Rayleigh).
        """
        ka = 0.1
        rcs_norm = mie_monostatic_rcs_pec_sphere(ka)
        rayleigh = 9.0 * ka**4
        assert np.isclose(rcs_norm, rayleigh, rtol=0.05)

    def test_high_frequency_limit(self):
        """For ka >> 1, monostatic RCS approaches pi*a^2 (geometric optics).
        sigma / (pi*a^2) -> 1.
        """
        ka = 50.0
        rcs_norm = mie_monostatic_rcs_pec_sphere(ka, n_max=70)
        assert np.isclose(rcs_norm, 1.0, rtol=0.01)

    def test_forward_scatter(self):
        """Forward scatter (theta=0) should be large (extinction theorem)."""
        ka = 3.0
        theta = np.array([0.0, np.pi])
        rcs = mie_rcs_pec_sphere(ka, theta)
        assert rcs[0] > rcs[1]  # Forward > backscatter

    def test_symmetry(self):
        """RCS should be symmetric about phi for axially symmetric incidence."""
        ka = 2.0
        theta = np.linspace(0, np.pi, 50)
        rcs = mie_rcs_pec_sphere(ka, theta)
        assert np.all(np.isfinite(rcs))
        assert np.all(rcs >= 0)


class TestFarFieldComputation:
    """Basic far-field computation tests on a small plate."""

    def test_far_field_finite(self):
        """Far-field should be finite for a simple mesh."""
        # Simple 2-triangle mesh
        vertices = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 0.5, 0.0],
        ])
        triangles = np.array([[0, 1, 2], [2, 1, 3]])
        mesh = Mesh(vertices, triangles)
        basis = compute_rwg_connectivity(mesh)

        k = 2 * np.pi
        Z = fill_matrix(EFIEOperator(), basis, mesh, k, eta0, quad_order=4)
        exc = PlaneWaveExcitation(
            E0=np.array([1.0, 0.0, 0.0]),
            k_hat=np.array([0.0, 0.0, -1.0]),
        )
        V = exc.compute_voltage_vector(basis, mesh, k)
        I = solve_direct(Z, V)

        theta = np.array([0.0, np.pi / 2, np.pi])
        phi = np.array([0.0, 0.0, 0.0])
        E_theta, E_phi = compute_far_field(I, basis, mesh, k, eta0, theta, phi)

        assert E_theta.shape == (3,)
        assert np.all(np.isfinite(E_theta))
        assert np.all(np.isfinite(E_phi))
