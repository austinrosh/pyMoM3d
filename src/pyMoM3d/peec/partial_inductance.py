"""Partial inductance computation via Neumann formula.

The partial mutual inductance between two filamentary current segments is:

    L_ij = (mu0 / 4pi) * integral integral (dl_i . dl_j) / R

For PEEC, the current in each segment flows along the segment direction,
so dl_i = hat{u}_i ds_i and dl_j = hat{u}_j ds_j, giving:

    L_ij = (mu0 / 4pi) * (hat{u}_i . hat{u}_j)
           * integral_0^{l_i} integral_0^{l_j} ds_i ds_j / R(s_i, s_j)

The double line integral is evaluated with Gauss-Legendre quadrature.

For self-inductance (i == j), the filamentary integral diverges (1/R
singularity when source and field points coincide).  The standard
regularization uses the Geometric Mean Distance (GMD) of the conductor
cross-section:

    L_self = (mu0 * l) / (2*pi) * [ln(2*l / GMD) - 1 + GMD/(2*l)]

For a rectangular cross-section w x t:

    ln(GMD) ~ -3/2 + ln(w + t)   (approximate, Grover)

or more accurately:

    GMD = 0.2235 * (w + t)  for w >> t or t >> w

References
----------
* Grover, F. W. (1946). "Inductance Calculations." Dover.
* Ruehli, A. E. (1974). "Equivalent Circuit Models for Three-Dimensional
  Multiconductor Systems." IEEE Trans. MTT, vol. 22, no. 3.
* Rosa, E. B. (1908). "The Self and Mutual Inductances of Linear
  Conductors." Bulletin of the Bureau of Standards, vol. 4, no. 2.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..utils.constants import mu0
from .trace import TraceSegment


# 8-point Gauss-Legendre nodes and weights on [-1, 1]
_GL8_NODES, _GL8_WEIGHTS = np.polynomial.legendre.leggauss(8)


def _gmd_rectangular(w: float, t: float) -> float:
    """Geometric Mean Distance for a rectangular cross-section.

    Uses the exact Grover formula for a rectangle:

        ln(GMD) = ln(sqrt(w^2 + t^2)) - 1/2
                  - (2/3)*[(t/w)*arctan(w/t) + (w/t)*arctan(t/w)]
                  + (1/12)*[(t/w)^2 * ln(1 + (w/t)^2)
                           + (w/t)^2 * ln(1 + (t/w)^2)]

    This is the GMD of a rectangle from itself (AMGD).

    Parameters
    ----------
    w : float
        Width (m).
    t : float
        Thickness (m).

    Returns
    -------
    gmd : float
        Geometric mean distance (m).
    """
    if w < 1e-30 or t < 1e-30:
        # Degenerate: use the non-zero dimension
        return max(w, t) * np.exp(-0.5)

    r = t / w
    r_inv = w / t

    ln_gmd = (np.log(np.sqrt(w**2 + t**2))
              - 0.5
              - (2.0 / 3.0) * (r * np.arctan(r_inv) + r_inv * np.arctan(r))
              + (1.0 / 12.0) * (r**2 * np.log(1.0 + r_inv**2)
                                + r_inv**2 * np.log(1.0 + r**2)))

    return np.exp(ln_gmd)


def self_inductance(seg: TraceSegment) -> float:
    """Partial self-inductance of a straight segment.

    Uses the exact formula for a filament with rectangular cross-section
    regularized by GMD:

        L_self = (mu0 * l) / (2*pi) * [ln(2*l / GMD) - 1 + GMD/(2*l) + mu_r/4]

    The mu_r/4 term accounts for internal inductance of the conductor.
    For non-magnetic conductors (mu_r = 1), this contributes mu0/(8*pi)
    per unit length.

    Parameters
    ----------
    seg : TraceSegment

    Returns
    -------
    L : float
        Self-inductance (H).
    """
    l = seg.length
    if l < 1e-15:
        return 0.0

    gmd = _gmd_rectangular(seg.width, seg.thickness)
    mu_r = seg.conductor.mu_r

    L = (mu0 * l / (2.0 * np.pi)) * (
        np.log(2.0 * l / gmd) - 1.0 + gmd / (2.0 * l) + mu_r / 4.0
    )
    return L


def mutual_inductance_filaments(
    seg_i: TraceSegment,
    seg_j: TraceSegment,
) -> float:
    """Partial mutual inductance between two filamentary segments.

    Evaluates the Neumann integral along segment centerlines using
    8-point Gauss-Legendre quadrature on each segment:

        L_ij = (mu0 / 4pi) * (u_i . u_j)
               * sum_p sum_q w_p w_q * (l_i/2)(l_j/2) / R(p,q)

    Parameters
    ----------
    seg_i, seg_j : TraceSegment
        The two segments.

    Returns
    -------
    L_ij : float
        Mutual inductance (H).  Can be negative for anti-parallel segments.
    """
    l_i = seg_i.length
    l_j = seg_j.length
    if l_i < 1e-15 or l_j < 1e-15:
        return 0.0

    u_i = seg_i.direction
    u_j = seg_j.direction
    dot_uij = np.dot(u_i, u_j)

    if abs(dot_uij) < 1e-15:
        # Perpendicular segments: mutual inductance is zero
        return 0.0

    mid_i = seg_i.midpoint
    mid_j = seg_j.midpoint

    # Map GL nodes from [-1,1] to segment parametrization
    # r_i(s) = mid_i + u_i * (l_i/2) * s,  s in [-1, 1]
    half_li = l_i / 2.0
    half_lj = l_j / 2.0

    # Quadrature points on segment i: shape (8, 3)
    pts_i = mid_i[np.newaxis, :] + np.outer(_GL8_NODES * half_li, u_i)
    # Quadrature points on segment j: shape (8, 3)
    pts_j = mid_j[np.newaxis, :] + np.outer(_GL8_NODES * half_lj, u_j)

    # Distance matrix: R[p, q] = |pts_i[p] - pts_j[q]|
    diff = pts_i[:, np.newaxis, :] - pts_j[np.newaxis, :, :]  # (8, 8, 3)
    R = np.sqrt(np.sum(diff**2, axis=2))  # (8, 8)

    # Prevent division by zero (shouldn't happen for distinct segments)
    R = np.maximum(R, 1e-30)

    # Weighted sum: sum_p sum_q w_p w_q / R[p,q]
    integrand = 1.0 / R  # (8, 8)
    integral = np.dot(_GL8_WEIGHTS, integrand @ _GL8_WEIGHTS)

    # Include Jacobians: ds_i = (l_i/2) ds, ds_j = (l_j/2) ds
    L_ij = (mu0 / (4.0 * np.pi)) * dot_uij * half_li * half_lj * integral

    return L_ij


def mutual_inductance_ribbon(
    seg_i: TraceSegment,
    seg_j: TraceSegment,
    n_width_points: int = 3,
) -> float:
    """Mutual inductance with finite-width ribbon correction.

    Averages the filamentary Neumann integral over the conductor
    cross-section width using Gauss-Legendre quadrature.  This
    improves accuracy for nearby segments where the filamentary
    approximation breaks down.

    Parameters
    ----------
    seg_i, seg_j : TraceSegment
    n_width_points : int
        Number of quadrature points across the width.

    Returns
    -------
    L_ij : float
        Mutual inductance (H).
    """
    u_i = seg_i.direction
    u_j = seg_j.direction
    dot_uij = np.dot(u_i, u_j)

    if abs(dot_uij) < 1e-15:
        return 0.0

    l_i = seg_i.length
    l_j = seg_j.length
    if l_i < 1e-15 or l_j < 1e-15:
        return 0.0

    # Transverse direction for each segment (perpendicular in the
    # horizontal plane, or arbitrary if segment is vertical)
    def _transverse(u):
        if abs(u[2]) < 0.9:
            n = np.cross(u, np.array([0.0, 0.0, 1.0]))
        else:
            n = np.cross(u, np.array([1.0, 0.0, 0.0]))
        return n / np.linalg.norm(n)

    n_i = _transverse(u_i)
    n_j = _transverse(u_j)

    # Gauss-Legendre points across width
    gl_w_nodes, gl_w_weights = np.polynomial.legendre.leggauss(n_width_points)

    half_wi = seg_i.width / 2.0
    half_wj = seg_j.width / 2.0

    # Average over width positions
    total = 0.0
    weight_sum = 0.0

    half_li = l_i / 2.0
    half_lj = l_j / 2.0

    for pi, wwi in enumerate(gl_w_weights):
        offset_i = n_i * (gl_w_nodes[pi] * half_wi)
        mid_i_shifted = seg_i.midpoint + offset_i

        pts_i = mid_i_shifted[np.newaxis, :] + np.outer(
            _GL8_NODES * half_li, u_i)

        for pj, wwj in enumerate(gl_w_weights):
            offset_j = n_j * (gl_w_nodes[pj] * half_wj)
            mid_j_shifted = seg_j.midpoint + offset_j

            pts_j = mid_j_shifted[np.newaxis, :] + np.outer(
                _GL8_NODES * half_lj, u_j)

            diff = pts_i[:, np.newaxis, :] - pts_j[np.newaxis, :, :]
            R = np.sqrt(np.sum(diff**2, axis=2))
            R = np.maximum(R, 1e-30)

            integral = np.dot(_GL8_WEIGHTS, (1.0 / R) @ _GL8_WEIGHTS)
            total += wwi * wwj * integral
            weight_sum += wwi * wwj

    total /= weight_sum
    L_ij = (mu0 / (4.0 * np.pi)) * dot_uij * half_li * half_lj * total
    return L_ij


def partial_inductance_matrix(
    segments: List[TraceSegment],
    use_ribbon: bool = False,
    n_width_points: int = 3,
) -> np.ndarray:
    """Assemble the M x M partial inductance matrix.

    Lp[i,j] = mutual inductance between segments i and j.
    Lp[i,i] = self-inductance of segment i.

    The matrix is symmetric: Lp[i,j] = Lp[j,i] (reciprocity).

    Parameters
    ----------
    segments : list of TraceSegment, length M
    use_ribbon : bool
        If True, use finite-width ribbon formula for off-diagonal terms.
    n_width_points : int
        Gauss points across width for ribbon formula.

    Returns
    -------
    Lp : ndarray, shape (M, M), float64
        Partial inductance matrix (H).  Symmetric.
    """
    M = len(segments)
    Lp = np.zeros((M, M), dtype=np.float64)

    # Diagonal: self-inductance
    for i in range(M):
        Lp[i, i] = self_inductance(segments[i])

    # Off-diagonal: mutual inductance
    mutual_fn = mutual_inductance_ribbon if use_ribbon else mutual_inductance_filaments

    for i in range(M):
        for j in range(i + 1, M):
            if use_ribbon:
                L_ij = mutual_fn(segments[i], segments[j],
                                 n_width_points=n_width_points)
            else:
                L_ij = mutual_fn(segments[i], segments[j])
            Lp[i, j] = L_ij
            Lp[j, i] = L_ij

    return Lp
