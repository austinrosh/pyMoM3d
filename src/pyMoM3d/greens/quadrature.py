"""Symmetric Gauss quadrature rules on triangles.

Barycentric coordinates (L1, L2, L3) with L3 = 1 - L1 - L2.
Cartesian point: r = L1*v0 + L2*v1 + L3*v2.
Integral approximation: sum(w_i * f(r_i)) * 2*Area  (rules normalised so
sum(w_i) = 0.5, matching the unit-triangle area convention).

References
----------
- Dunavant, D. A. (1985). High degree efficient symmetrical Gaussian
  quadrature rules for the triangle. IJNME 21, 1129-1148.
"""

import numpy as np
from typing import Tuple

_QUAD_CACHE: dict = {}


def triangle_quad_rule(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return symmetric Gauss quadrature rule on a triangle.

    Parameters
    ----------
    order : int
        Number of quadrature points. Supported: 1, 3, 4, 7, 13.

    Returns
    -------
    weights : ndarray, shape (N,)
        Quadrature weights (sum = 0.5, unit-triangle convention).
    bary : ndarray, shape (N, 3)
        Barycentric coordinates (L1, L2, L3) per point.
    """
    if order in _QUAD_CACHE:
        return _QUAD_CACHE[order]

    if order == 1:
        # Centroid rule — exact for degree 1
        w = np.array([0.5])
        b = np.array([[1/3, 1/3, 1/3]])
    elif order == 3:
        # Degree 2
        w = np.array([1/6, 1/6, 1/6])
        b = np.array([
            [2/3, 1/6, 1/6],
            [1/6, 2/3, 1/6],
            [1/6, 1/6, 2/3],
        ])
    elif order == 4:
        # Degree 3
        w = np.array([-27/96, 25/96, 25/96, 25/96])
        b = np.array([
            [1/3, 1/3, 1/3],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ])
    elif order == 7:
        # Degree 5 (Dunavant)
        a1 = 0.059715871789770
        b1 = 0.470142064105115
        a2 = 0.797426985353087
        b2 = 0.101286507323456

        w1 = 0.1125
        w2 = 0.066197076394253
        w3 = 0.062969590272414

        w = np.array([w1, w2, w2, w2, w3, w3, w3])
        b = np.array([
            [1/3, 1/3, 1/3],
            [a1, b1, b1],
            [b1, a1, b1],
            [b1, b1, a1],
            [a2, b2, b2],
            [b2, a2, b2],
            [b2, b2, a2],
        ])
    elif order == 13:
        # Degree 7 (Dunavant)
        w_center = 0.0
        # Actually use the correct 13-point rule
        # From Dunavant (1985) Table III, degree 7
        a1 = 0.260345966079038
        b1 = 0.065130102902216  # (1-2*a1) is wrong, use Dunavant directly
        # Re-derive: use the well-known 13-point rule
        # Group 1: centroid (1 point)
        L1_1 = 1/3
        w1 = -0.149570044467670 / 2.0

        # Group 2: 3 points
        L1_2a = 0.260345966079038
        L1_2b = 0.479308067841923  # = (1 - 2*L1_2a)
        w2 = 0.175615257433204 / 2.0

        # Group 3: 3 points
        L1_3a = 0.065130102902216
        L1_3b = 0.869739794195568  # = (1 - 2*L1_3a)
        w3 = 0.053347235608839 / 2.0

        # Group 4: 6 points
        L1_4a = 0.048690315425316
        L1_4b = 0.312865496004875
        L1_4c = 0.638444188569809  # = 1 - L1_4a - L1_4b
        w4 = 0.077113760890257 / 2.0

        w = np.array([
            w1,
            w2, w2, w2,
            w3, w3, w3,
            w4, w4, w4, w4, w4, w4,
        ])
        b = np.array([
            [L1_1, L1_1, L1_1],
            [L1_2a, L1_2a, L1_2b],
            [L1_2a, L1_2b, L1_2a],
            [L1_2b, L1_2a, L1_2a],
            [L1_3a, L1_3a, L1_3b],
            [L1_3a, L1_3b, L1_3a],
            [L1_3b, L1_3a, L1_3a],
            [L1_4a, L1_4b, L1_4c],
            [L1_4a, L1_4c, L1_4b],
            [L1_4b, L1_4a, L1_4c],
            [L1_4b, L1_4c, L1_4a],
            [L1_4c, L1_4a, L1_4b],
            [L1_4c, L1_4b, L1_4a],
        ])
    else:
        raise ValueError(f"Unsupported quadrature order {order}. Use 1, 3, 4, 7, or 13.")

    result = w.astype(np.float64), b.astype(np.float64)
    _QUAD_CACHE[order] = result
    return result


def integrate_over_triangle(
    func,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    quad_order: int = 4,
) -> complex:
    """Integrate a scalar or complex function over a triangle.

    Parameters
    ----------
    func : callable
        Function f(r) -> scalar/complex, where r is shape (3,).
    v0, v1, v2 : ndarray, shape (3,)
        Triangle vertices.
    quad_order : int
        Number of quadrature points.

    Returns
    -------
    result : complex
        Approximate integral of f over the triangle.
    """
    weights, bary = triangle_quad_rule(quad_order)

    # Jacobian = 2 * triangle area
    cross = np.cross(v1 - v0, v2 - v0)
    twice_area = np.linalg.norm(cross)

    result = 0.0 + 0.0j
    for i in range(len(weights)):
        r = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
        result += weights[i] * func(r)

    return result * twice_area
