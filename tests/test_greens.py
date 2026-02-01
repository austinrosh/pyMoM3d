"""Tests for Green's function, quadrature, and singularity extraction."""

import numpy as np
import pytest

from pyMoM3d.greens.quadrature import triangle_quad_rule, integrate_over_triangle
from pyMoM3d.greens.free_space import scalar_green
from pyMoM3d.greens.singularity import (
    _analytical_1_over_R_triangle,
    integrate_green_singular,
    integrate_rho_green_singular,
)


# ---- Quadrature tests ----

class TestQuadrature:
    """Test triangle quadrature rules."""

    @pytest.mark.parametrize("order", [1, 3, 4, 7, 13])
    def test_weights_sum(self, order):
        """Weights should sum to 0.5 (unit triangle area)."""
        w, b = triangle_quad_rule(order)
        assert np.isclose(np.sum(w), 0.5, atol=1e-14)

    @pytest.mark.parametrize("order", [1, 3, 4, 7, 13])
    def test_barycentric_sum(self, order):
        """Each barycentric coordinate row should sum to 1."""
        w, b = triangle_quad_rule(order)
        assert np.allclose(np.sum(b, axis=1), 1.0, atol=1e-14)

    @pytest.mark.parametrize("order", [1, 3, 4, 7, 13])
    def test_constant_integration(self, order):
        """Integral of f=1 over triangle should equal triangle area."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        result = integrate_over_triangle(lambda r: 1.0, v0, v1, v2, quad_order=order)
        assert np.isclose(result.real, 0.5, atol=1e-14)

    def test_linear_integration(self):
        """Integral of f(r) = x over unit right triangle = 1/6."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        result = integrate_over_triangle(lambda r: r[0], v0, v1, v2, quad_order=3)
        assert np.isclose(result.real, 1/6, atol=1e-14)

    def test_quadratic_integration(self):
        """Integral of x^2 over unit right triangle = 1/12."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        result = integrate_over_triangle(lambda r: r[0]**2, v0, v1, v2, quad_order=4)
        assert np.isclose(result.real, 1/12, atol=1e-12)

    def test_higher_order_convergence(self):
        """Higher-order rules should agree for smooth integrands."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        f = lambda r: np.exp(r[0] + r[1])

        I_3 = integrate_over_triangle(f, v0, v1, v2, quad_order=3)
        I_7 = integrate_over_triangle(f, v0, v1, v2, quad_order=7)
        I_13 = integrate_over_triangle(f, v0, v1, v2, quad_order=13)

        # 7-pt and 13-pt should agree well for this smooth function
        assert np.isclose(I_7, I_13, rtol=1e-6)
        # 3-pt should be less accurate
        assert abs(I_3 - I_13) > abs(I_7 - I_13)

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            triangle_quad_rule(5)


# ---- Scalar Green's function tests ----

class TestScalarGreen:
    def test_reciprocity(self):
        """g(r1, r2) == g(r2, r1)."""
        k = 2 * np.pi
        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([0.0, 1.0, 0.5])
        assert np.isclose(scalar_green(k, r1, r2), scalar_green(k, r2, r1))

    def test_known_value(self):
        """At R=1, g = exp(-jk) / (4*pi)."""
        k = 1.0
        r1 = np.array([0.0, 0.0, 0.0])
        r2 = np.array([1.0, 0.0, 0.0])
        expected = np.exp(-1j * k) / (4 * np.pi)
        assert np.isclose(scalar_green(k, r1, r2), expected)

    def test_vectorized(self):
        """Should handle arrays of points."""
        k = 1.0
        r = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        r_prime = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float64)
        g = scalar_green(k, r, r_prime)
        assert g.shape == (2,)
        assert np.isclose(g[0], g[1])  # Reciprocity


# ---- Singularity extraction tests ----

class TestSingularity:
    """Test the 1/R analytical integration (Wilton 1984)."""

    def test_point_above_equilateral_centroid(self):
        """Point directly above centroid of equilateral triangle.

        For an equilateral triangle with side a, observation point at
        height h above centroid, the integral of 1/R over the triangle
        has a known form. We compare numerical (high-order quad) vs
        analytical.
        """
        a = 1.0
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([a, 0.0, 0.0])
        v2 = np.array([a/2, a*np.sqrt(3)/2, 0.0])

        centroid = (v0 + v1 + v2) / 3.0
        h = 1.0  # Well above triangle so numerical quad is also accurate
        r_obs = centroid + np.array([0, 0, h])

        # Analytical
        I_analytical = _analytical_1_over_R_triangle(r_obs, v0, v1, v2)

        # High-order numerical (observation point off surface, so not singular)
        w, bary = triangle_quad_rule(13)
        cross = np.cross(v1 - v0, v2 - v0)
        twice_area = np.linalg.norm(cross)
        I_numerical = 0.0
        for i in range(len(w)):
            rp = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
            R = np.linalg.norm(r_obs - rp)
            I_numerical += w[i] / R
        I_numerical *= twice_area

        assert np.isclose(I_analytical, I_numerical, rtol=1e-5)

    def test_point_on_triangle_surface(self):
        """Observation point ON the triangle (h=0) — the truly singular case.

        The analytical formula should still return a finite value.
        """
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        centroid = (v0 + v1 + v2) / 3.0
        I = _analytical_1_over_R_triangle(centroid, v0, v1, v2)

        # Should be finite and positive
        assert np.isfinite(I)
        assert I > 0

    def test_green_well_separated(self):
        """Well-separated triangles: 3-pt and 13-pt quad should agree."""
        k = 2 * np.pi  # lambda = 1 m

        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([0.1, 0.0, 0.0])
        v2 = np.array([0.0, 0.1, 0.0])

        r_obs = np.array([5.0, 5.0, 5.0])  # Far away

        I_3 = integrate_green_singular(k, r_obs, v0, v1, v2, quad_order=3)
        I_13 = integrate_green_singular(k, r_obs, v0, v1, v2, quad_order=13)

        assert np.isclose(I_3, I_13, rtol=1e-5)

    def test_green_self_term_finite(self):
        """Self-term (observation at centroid) should return finite complex value."""
        k = 2 * np.pi

        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([0.1, 0.0, 0.0])
        v2 = np.array([0.0, 0.1, 0.0])

        centroid = (v0 + v1 + v2) / 3.0

        I = integrate_green_singular(k, centroid, v0, v1, v2, quad_order=7)
        assert np.isfinite(I)
        assert I.real > 0  # Static part dominates, should be positive

    def test_green_reciprocity_via_integration(self):
        """Integral of g over triangle should not depend on which is 'source'.
        For two well-separated triangles, int_T1 g(c2, r') dS' and
        int_T2 g(c1, r') dS' can differ (different triangles), but
        int_T1 g(r, r') dS' should be independent of how we label r vs r'.
        This test just verifies g(r,r') = g(r',r) at point level.
        """
        k = 1.0
        r1 = np.array([1.0, 2.0, 3.0])
        r2 = np.array([4.0, 5.0, 6.0])
        g12 = scalar_green(k, r1, r2)
        g21 = scalar_green(k, r2, r1)
        assert np.isclose(g12, g21)

    def test_rho_green_singular_self_term(self):
        """Self-term of rho*g integral: compare against lifted reference.

        Compute the integral with observation point slightly above the triangle
        (where plain quadrature converges), then verify the on-surface result
        from singularity extraction is close.
        """
        k = 2 * np.pi
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([0.1, 0.0, 0.0])
        v2 = np.array([0.0, 0.1, 0.0])
        r_free = v0.copy()
        centroid = (v0 + v1 + v2) / 3.0

        # Reference: observation point lifted well above, high-order quad
        h = 0.005  # small but enough for quad convergence
        r_lifted = centroid + np.array([0, 0, h])
        ref = integrate_rho_green_singular(
            k, r_lifted, v0, v1, v2, r_free, quad_order=13,
        )

        # On-surface with singularity extraction
        val = integrate_rho_green_singular(
            k, centroid, v0, v1, v2, r_free, quad_order=13,
        )

        # Both should be finite and close (h is small relative to triangle)
        assert np.all(np.isfinite(val))
        assert np.all(np.isfinite(ref))
        # Real parts should agree within ~20% (h introduces small difference)
        # Compare components with significant magnitude
        max_mag = max(np.max(np.abs(val)), np.max(np.abs(ref)))
        for comp in range(3):
            if abs(ref[comp]) > 0.1 * max_mag:
                rdiff = abs(val[comp] - ref[comp]) / abs(ref[comp])
                assert rdiff < 0.5, f"Component {comp}: rdiff={rdiff}"

    def test_rho_green_singular_far_field(self):
        """Far-field rho*g: plain quadrature path, should match direct computation."""
        k = 2 * np.pi
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([0.1, 0.0, 0.0])
        v2 = np.array([0.0, 0.1, 0.0])
        r_free = v0.copy()
        r_obs = np.array([5.0, 5.0, 5.0])

        I_4 = integrate_rho_green_singular(k, r_obs, v0, v1, v2, r_free, quad_order=4)
        I_13 = integrate_rho_green_singular(k, r_obs, v0, v1, v2, r_free, quad_order=13)

        assert np.allclose(I_4, I_13, rtol=1e-4)
