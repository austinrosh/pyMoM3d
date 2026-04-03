"""Surface impedance boundary condition for finite-thickness conductors.

For PEC surfaces, the tangential electric field is zero: n̂ × E = 0.
For imperfect conductors, the boundary condition becomes:

    n̂ × E = Z_s (n̂ × H × n̂)

where Z_s is the surface impedance.  In the MoM formulation, this adds
a term Z_s · G to the impedance matrix, where G is the Gram (mass)
matrix of the RWG testing and basis functions:

    G_mn = ∫∫ f_m(r) · f_n(r) dS

For a planar conductor of thickness t, conductivity σ, and permeability μ:

    Z_s(f) = (γ / σ) · coth(γ t)

where γ = (1+j) / δ,  δ = √(2 / (ωμσ))  is the skin depth.

At high frequencies (t >> δ): Z_s → (1+j) / (σδ) = √(jωμ/σ)
At DC (ω → 0): Z_s → 1 / (σ t)  (DC sheet resistance)

References
----------
* Senior, T. B. A. (1960). "Impedance boundary conditions for
  imperfectly conducting surfaces." Applied Scientific Research B,
  8(1), 418-436.
* Rao, S. M. (1999). "Time Domain Electromagnetics." Academic Press.
  Ch. 6: Impedance boundary conditions in MoM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils.constants import mu0


@dataclass
class ConductorProperties:
    """Material properties for a finite-thickness conductor.

    Parameters
    ----------
    sigma : float
        Conductivity (S/m).  Copper: 5.8e7, Aluminum: 3.77e7.
    thickness : float
        Conductor thickness (m).
    mu_r : float
        Relative permeability (dimensionless).  Default 1.0.
    name : str
        Optional label for the conductor material.
    """
    sigma: float
    thickness: float
    mu_r: float = 1.0
    name: str = ''

    def surface_impedance(self, freq: float) -> complex:
        """Compute frequency-dependent surface impedance Z_s(f).

        Z_s = (γ/σ) · coth(γt)

        where γ = (1+j)/δ, δ = √(2/(ωμσ)).

        Parameters
        ----------
        freq : float
            Frequency (Hz).

        Returns
        -------
        Z_s : complex
            Surface impedance (Ω/□).
        """
        omega = 2.0 * np.pi * freq
        mu = self.mu_r * mu0

        if omega < 1e-30 or self.sigma < 1e-30:
            # DC limit: sheet resistance
            return 1.0 / (self.sigma * self.thickness) if self.sigma > 0 else 0.0

        # Skin depth
        delta = np.sqrt(2.0 / (omega * mu * self.sigma))

        # Complex propagation constant
        gamma = (1.0 + 1.0j) / delta

        # Z_s = (gamma / sigma) * coth(gamma * t)
        gt = gamma * self.thickness

        # Use stable coth computation to avoid overflow for large |gt|
        if abs(gt) > 50:
            # coth(x) → 1 for large |x| (thick conductor limit)
            return gamma / self.sigma
        else:
            return (gamma / self.sigma) / np.tanh(gt)

    def skin_depth(self, freq: float) -> float:
        """Skin depth δ = √(2/(ωμσ)).

        Parameters
        ----------
        freq : float
            Frequency (Hz).

        Returns
        -------
        delta : float
            Skin depth (m).
        """
        omega = 2.0 * np.pi * freq
        mu = self.mu_r * mu0
        if omega < 1e-30 or self.sigma < 1e-30:
            return float('inf')
        return np.sqrt(2.0 / (omega * mu * self.sigma))

    def dc_sheet_resistance(self) -> float:
        """DC sheet resistance R_sh = 1/(σt).

        Returns
        -------
        R_sh : float
            Sheet resistance (Ω/□).
        """
        if self.sigma < 1e-30 or self.thickness < 1e-30:
            return float('inf')
        return 1.0 / (self.sigma * self.thickness)


def build_gram_matrix(
    rwg_basis,
    mesh,
    quad_order: int = 4,
) -> np.ndarray:
    """Assemble the RWG Gram (mass) matrix G.

    G_mn = ∫ f_m(r) · f_n(r) dS

    where f_m and f_n are RWG basis functions.  The integral is nonzero
    only when the basis functions share a supporting triangle.  Each RWG
    basis function is supported on exactly 2 triangles (T+ and T-), so
    G is very sparse — at most a 4-band structure.

    For efficiency, we exploit the analytical result for the inner product
    of two RWG functions on a common triangle:

        ∫_T f_m · f_n dS = (l_m l_n) / (4A) · [ρ_m · ρ_n]_avg

    where ρ_m, ρ_n are the position vectors relative to the free vertices.

    For same-basis (m == n, same triangle):
        ∫_T |f_n|^2 dS = (l_n^2) / (4A^2) · ∫_T |ρ|^2 dS

    Parameters
    ----------
    rwg_basis : RWGBasis
    mesh : Mesh
    quad_order : int
        Quadrature order for numerical integration.

    Returns
    -------
    G : ndarray, shape (N, N), float64
        Symmetric Gram matrix.
    """
    from ..greens.quadrature import triangle_quad_rule

    N = rwg_basis.num_basis
    G = np.zeros((N, N), dtype=np.float64)

    weights, bary = triangle_quad_rule(quad_order)
    verts = mesh.vertices
    tris = mesh.triangles

    t_plus = rwg_basis.t_plus
    t_minus = rwg_basis.t_minus
    edge_length = rwg_basis.edge_length

    # Get free vertex indices
    fv_plus = rwg_basis.free_vertex_plus
    fv_minus = rwg_basis.free_vertex_minus

    # Build triangle-to-basis mapping for efficient lookup
    # tri_basis[t] = list of (basis_idx, sign, free_vertex_global_idx)
    T = len(tris)
    tri_basis = [[] for _ in range(T)]
    for n in range(N):
        tp = int(t_plus[n])
        tm = int(t_minus[n])
        tri_basis[tp].append((n, +1.0, int(fv_plus[n])))
        tri_basis[tm].append((n, -1.0, int(fv_minus[n])))

    # For each triangle, compute the Gram contribution from all
    # basis function pairs supported on that triangle
    Q = len(weights)
    for t in range(T):
        basis_list = tri_basis[t]
        if not basis_list:
            continue

        v0, v1, v2 = verts[tris[t]]
        area = mesh.triangle_areas[t]

        # Quadrature points on this triangle
        r_quad = np.empty((Q, 3), dtype=np.float64)
        for p in range(Q):
            r_quad[p] = bary[p, 0] * v0 + bary[p, 1] * v1 + bary[p, 2] * v2

        twice_area = 2.0 * area

        for i, (m, sign_m, fv_m) in enumerate(basis_list):
            l_m = edge_length[m]
            r_free_m = verts[fv_m]  # free vertex of basis m

            # ρ_m = sign_m * (r - r_free) for T+, sign_m * (r_free - r) for T-
            # But with sign convention: on T+, f = (l/2A)(r - r_free)
            # on T-, f = (l/2A)(r_free - r)
            # So f_m = (l_m / (2A)) * sign_m * (r_quad - r_free_m) when sign=+1
            #        = (l_m / (2A)) * (r_free_m - r_quad) when sign=-1
            if sign_m > 0:
                rho_m = r_quad - r_free_m  # (Q, 3)
            else:
                rho_m = r_free_m - r_quad  # (Q, 3)

            pf_m = l_m / twice_area

            for j in range(i, len(basis_list)):
                n_idx, sign_n, fv_n = basis_list[j]
                l_n = edge_length[n_idx]

                if sign_n > 0:
                    rho_n = r_quad - verts[fv_n]
                else:
                    rho_n = verts[fv_n] - r_quad

                pf_n = l_n / twice_area

                # G contribution = pf_m * pf_n * ∫_T rho_m · rho_n dS
                # dS on triangle = twice_area * sum(w_p * ...)
                dot_products = np.sum(rho_m * rho_n, axis=1)  # (Q,)
                integral = twice_area * np.dot(weights, dot_products)

                val = pf_m * pf_n * integral

                G[m, n_idx] += val
                if m != n_idx:
                    G[n_idx, m] += val

    return G


def apply_surface_impedance(
    Z: np.ndarray,
    rwg_basis,
    mesh,
    conductor: ConductorProperties,
    freq: float,
    quad_order: int = 4,
    gram_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add surface impedance contribution to the MoM impedance matrix.

    Modifies Z in-place: Z_new = Z + Z_s(f) · G

    Parameters
    ----------
    Z : ndarray, shape (N, N), complex128
        MoM impedance matrix (modified in-place).
    rwg_basis : RWGBasis
    mesh : Mesh
    conductor : ConductorProperties
        Conductor material properties.
    freq : float
        Frequency (Hz).
    quad_order : int
        Quadrature order for Gram matrix assembly.
    gram_matrix : ndarray, optional
        Pre-computed Gram matrix.  If None, it is assembled internally.
        Pass this to avoid recomputing G at every frequency point.

    Returns
    -------
    Z : ndarray, shape (N, N), complex128
        Modified impedance matrix (same object as input).
    """
    Z_s = conductor.surface_impedance(freq)

    if gram_matrix is None:
        gram_matrix = build_gram_matrix(rwg_basis, mesh, quad_order)

    Z += Z_s * gram_matrix

    return Z
