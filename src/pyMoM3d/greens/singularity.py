"""Singularity extraction for Green's function integration over triangles.

When the observation point is on or near the source triangle, the 1/R
kernel is singular or nearly singular.  We decompose:

    g(R) = 1/(4*pi*R) + [g(R) - 1/(4*pi*R)]

The 1/(4*pi*R) part is integrated analytically using the Wilton et al. (1984)
formulae.  The smooth remainder [exp(-jkR)/(4*pi*R) - 1/(4*pi*R)] is
integrated by standard quadrature.

References
----------
- Wilton, Rao, Glisson, Schaubert, Al-Bundak, Butler (1984).
  "Potential integrals for uniform and linear source distributions on
  polygonal and polyhedral domains."  IEEE Trans. AP-32(3), pp. 276-281.
- Graglia (1993). On the numerical integration of the linear shape
  functions times the 3-D Green's function or its gradient on a plane
  triangle. IEEE Trans. AP-41(10).
"""

import numpy as np
from .quadrature import triangle_quad_rule


def _analytical_1_over_R_triangle(
    r_obs: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    """Analytically integrate 1/R over a triangle.

    Computes: integral_T  1/|r_obs - r'| dS'

    Uses the formulation from Graglia (1993), IEEE Trans. AP-41(10).
    The integral is decomposed into contributions from each edge of the
    triangle.

    Parameters
    ----------
    r_obs : ndarray, shape (3,)
        Observation point.
    v0, v1, v2 : ndarray, shape (3,)
        Triangle vertices (CCW winding assumed).

    Returns
    -------
    result : float
        The integral value (always non-negative).
    """
    vertices = [v0, v1, v2]

    # Triangle normal
    n_vec = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(n_vec)
    if area2 < 1e-30:
        return 0.0
    n_hat = n_vec / area2

    # Signed height of r_obs above triangle plane
    d = np.dot(r_obs - v0, n_hat)

    # Project observation point onto triangle plane
    r_proj = r_obs - d * n_hat

    # Vectors from r_obs to each vertex
    w = [vi - r_obs for vi in vertices]
    R = [np.linalg.norm(wi) for wi in w]

    result = 0.0

    for i in range(3):
        j = (i + 1) % 3

        # Edge endpoints
        va = vertices[i]
        vb = vertices[j]

        # Edge vector and length
        edge_vec = vb - va
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        t_hat = edge_vec / edge_len

        # Outward edge normal in the triangle plane
        # n_hat x t_hat gives a vector perpendicular to the edge in the
        # triangle plane. For a CCW-wound triangle, this points inward.
        # We want outward, so negate.
        m_hat = -np.cross(n_hat, t_hat)

        # Verify it points outward (away from opposite vertex)
        opp = vertices[(i + 2) % 3]
        if np.dot(m_hat, (va + vb) / 2.0 - opp) < 0:
            m_hat = -m_hat

        # 2D vectors from r_proj to edge endpoints
        rho_a = va - r_proj
        rho_b = vb - r_proj

        # Signed perpendicular distance from r_proj to edge line
        rho_0 = np.dot(rho_a, m_hat)

        # Tangential coordinates along edge
        t_a = np.dot(rho_a, t_hat)
        t_b = np.dot(rho_b, t_hat)

        # 3D distances from r_obs to edge endpoints
        R_a = R[i]
        R_b = R[j]

        # Logarithmic term: ln((t_b + R_b) / (t_a + R_a))
        arg_num = t_b + R_b
        arg_den = t_a + R_a
        if abs(arg_den) > 1e-30 and arg_num / arg_den > 0:
            ln_term = np.log(arg_num / arg_den)
        else:
            ln_term = 0.0

        # Arctan term (solid angle)
        abs_d = abs(d)
        if abs_d > 1e-14:
            P0_sq = rho_0**2 + d**2
            atan_b = np.arctan2(rho_0 * t_b, P0_sq + abs_d * R_b)
            atan_a = np.arctan2(rho_0 * t_a, P0_sq + abs_d * R_a)
            atan_term = atan_b - atan_a
        else:
            atan_term = 0.0

        result += rho_0 * ln_term - abs_d * atan_term

    return result


def integrate_green_singular(
    k: float,
    r_obs: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    quad_order: int = 4,
    near_threshold: float = 0.2,
) -> complex:
    """Integrate scalar Green's function over a source triangle, handling
    the singularity when the observation point is on or near the triangle.

    Uses singularity extraction:
        integral g(R) dS' = integral 1/(4*pi*R) dS'
                           + integral [g(R) - 1/(4*pi*R)] dS'

    The first term is evaluated analytically (Wilton 1984).
    The second term is smooth and evaluated by Gauss quadrature.

    Parameters
    ----------
    k : float
        Wavenumber (rad/m).
    r_obs : ndarray, shape (3,)
        Observation point.
    v0, v1, v2 : ndarray, shape (3,)
        Source triangle vertices.
    quad_order : int
        Number of quadrature points for smooth remainder.
    near_threshold : float
        If min distance to triangle < near_threshold * mean_edge_length,
        use singularity extraction.  Otherwise, use plain quadrature.

    Returns
    -------
    result : complex128
        Integral of g(r_obs, r') over the source triangle.
    """
    r_obs = np.asarray(r_obs, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    # Check if we need singularity extraction
    # Use minimum distance to any vertex for robust near-detection
    dist = min(np.linalg.norm(r_obs - v0),
               np.linalg.norm(r_obs - v1),
               np.linalg.norm(r_obs - v2))
    mean_edge = (np.linalg.norm(v1 - v0) + np.linalg.norm(v2 - v1)
                 + np.linalg.norm(v0 - v2)) / 3.0

    if dist > near_threshold * mean_edge * 3.0 and mean_edge > 1e-30:
        # Far enough: plain quadrature
        weights, bary = triangle_quad_rule(quad_order)
        cross = np.cross(v1 - v0, v2 - v0)
        twice_area = np.linalg.norm(cross)

        result = 0.0 + 0.0j
        for i in range(len(weights)):
            r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
            R = np.linalg.norm(r_obs - r_prime)
            if R < 1e-30:
                R = 1e-30
            result += weights[i] * np.exp(-1j * k * R) / (4.0 * np.pi * R)

        return result * twice_area

    # Near/self term: singularity extraction
    # Analytical part: integral of 1/(4*pi*R)
    I_static = _analytical_1_over_R_triangle(r_obs, v0, v1, v2) / (4.0 * np.pi)

    # Smooth remainder: integral of [exp(-jkR)/(4*pi*R) - 1/(4*pi*R)]
    weights, bary = triangle_quad_rule(quad_order)
    cross = np.cross(v1 - v0, v2 - v0)
    twice_area = np.linalg.norm(cross)

    I_smooth = 0.0 + 0.0j
    for i in range(len(weights)):
        r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
        R = np.linalg.norm(r_obs - r_prime)
        if R < 1e-30:
            # At R=0: exp(-jkR)/(4*pi*R) - 1/(4*pi*R) -> -jk/(4*pi)
            remainder = -1j * k / (4.0 * np.pi)
        else:
            g_full = np.exp(-1j * k * R) / (4.0 * np.pi * R)
            g_static = 1.0 / (4.0 * np.pi * R)
            remainder = g_full - g_static
        I_smooth += weights[i] * remainder

    I_smooth *= twice_area

    return I_static + I_smooth


def integrate_rho_green_singular(
    k: float,
    r_obs: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    r_free_vertex: np.ndarray,
    quad_order: int = 4,
    near_threshold: float = 0.2,
) -> np.ndarray:
    """Integrate rho(r') * g(r_obs, r') over a source triangle with singularity extraction.

    Computes: integral_T (r' - r_free_vertex) * exp(-jkR)/(4*pi*R) dS'

    where R = |r_obs - r'|, using singularity extraction to handle R -> 0.

    Decomposition:
        integral rho * g dS' = integral rho * [g - 1/(4*pi*R)] dS'   (smooth)
                              + integral rho / (4*pi*R) dS'           (singular)

    The singular part is further split:
        integral rho/(4*pi*R) dS' = (1/4pi) integral (r'-r_obs)/R dS'
                                   + (r_obs - r_free) / (4pi) * integral 1/R dS'

    Parameters
    ----------
    k : float
        Wavenumber (rad/m).
    r_obs : ndarray, shape (3,)
        Observation point.
    v0, v1, v2 : ndarray, shape (3,)
        Source triangle vertices.
    r_free_vertex : ndarray, shape (3,)
        Free vertex of the RWG basis function on the source triangle.
    quad_order : int
        Number of quadrature points.
    near_threshold : float
        Near-field threshold for singularity extraction.

    Returns
    -------
    result : ndarray, shape (3,), complex128
        Vector-valued integral.
    """
    r_obs = np.asarray(r_obs, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    r_free_vertex = np.asarray(r_free_vertex, dtype=np.float64)

    # Use minimum distance to any vertex for robust near-detection
    # (centroid distance can miss cases where r_obs is on the triangle)
    dist = min(np.linalg.norm(r_obs - v0),
               np.linalg.norm(r_obs - v1),
               np.linalg.norm(r_obs - v2))
    mean_edge = (np.linalg.norm(v1 - v0) + np.linalg.norm(v2 - v1)
                 + np.linalg.norm(v0 - v2)) / 3.0

    weights, bary = triangle_quad_rule(quad_order)
    cross = np.cross(v1 - v0, v2 - v0)
    twice_area = np.linalg.norm(cross)

    if dist > near_threshold * mean_edge * 3.0 and mean_edge > 1e-30:
        # Far: plain quadrature (no singularity issue)
        result = np.zeros(3, dtype=np.complex128)
        for i in range(len(weights)):
            r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
            rho = r_prime - r_free_vertex
            R = np.linalg.norm(r_obs - r_prime)
            if R < 1e-30:
                R = 1e-30
            g_val = np.exp(-1j * k * R) / (4.0 * np.pi * R)
            result += weights[i] * rho * g_val
        return result * twice_area

    # Near/self: singularity extraction
    # 1) Analytical: integral 1/R dS'
    I_1_over_R = _analytical_1_over_R_triangle(r_obs, v0, v1, v2)

    # 2) Quadrature for (r'-r_obs)/R — bounded integrand (magnitude <= 1)
    I_Rhat = np.zeros(3, dtype=np.float64)
    for i in range(len(weights)):
        r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
        diff = r_prime - r_obs
        R = np.linalg.norm(diff)
        if R < 1e-30:
            # At R=0, (r'-r)/R is undefined but measure-zero; skip
            continue
        I_Rhat += weights[i] * diff / R
    I_Rhat *= twice_area

    # Singular part: (1/4pi) * I_Rhat + (r_obs - r_free) / (4pi) * I_1_over_R
    I_singular = (I_Rhat + (r_obs - r_free_vertex) * I_1_over_R) / (4.0 * np.pi)

    # 3) Smooth remainder: integral rho * [g - 1/(4*pi*R)] dS'
    I_smooth = np.zeros(3, dtype=np.complex128)
    for i in range(len(weights)):
        r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
        rho = r_prime - r_free_vertex
        R = np.linalg.norm(r_obs - r_prime)
        if R < 1e-30:
            # limit of [exp(-jkR)/(4piR) - 1/(4piR)] * rho as R->0 is -jk/(4pi) * rho
            remainder = -1j * k / (4.0 * np.pi)
        else:
            g_full = np.exp(-1j * k * R) / (4.0 * np.pi * R)
            g_static = 1.0 / (4.0 * np.pi * R)
            remainder = g_full - g_static
        I_smooth += weights[i] * rho * remainder
    I_smooth *= twice_area

    return I_singular + I_smooth
