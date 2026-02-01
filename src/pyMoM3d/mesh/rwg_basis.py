"""RWG basis function data structure.

Stores per-basis-function geometric data needed for MoM impedance fill,
excitation computation, and far-field integration.

Convention (see CONVENTIONS.md):
- t_plus: triangle where current flows AWAY from free vertex (div > 0)
- t_minus: triangle where current flows TOWARD free vertex (div < 0)
- f_n(r) = (l_n / 2A+) * rho+ on T+,  (l_n / 2A-) * rho- on T-
- rho+ = r - r_free+,  rho- = r_free- - r
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RWGBasis:
    """Per-basis-function arrays for RWG basis functions.

    Only interior edges (shared by exactly two triangles) become basis
    functions.  Boundary edges are excluded.

    Parameters
    ----------
    num_basis : int
        Number of basis functions.
    edge_index : ndarray, shape (N,)
        Index into mesh.edges for each basis function.
    edge_length : ndarray, shape (N,)
        Length of the shared edge l_n.
    t_plus : ndarray, shape (N,)
        Triangle index for T+ (current flows away from free vertex).
    t_minus : ndarray, shape (N,)
        Triangle index for T- (current flows toward free vertex).
    free_vertex_plus : ndarray, shape (N,)
        Vertex index of the free vertex in T+ (opposite the shared edge).
    free_vertex_minus : ndarray, shape (N,)
        Vertex index of the free vertex in T- (opposite the shared edge).
    area_plus : ndarray, shape (N,)
        Area of T+.
    area_minus : ndarray, shape (N,)
        Area of T-.

    Attributes
    ----------
    num_boundary_edges : int
        Number of boundary edges found (for diagnostics).
    """

    num_basis: int
    edge_index: np.ndarray
    edge_length: np.ndarray
    t_plus: np.ndarray
    t_minus: np.ndarray
    free_vertex_plus: np.ndarray
    free_vertex_minus: np.ndarray
    area_plus: np.ndarray
    area_minus: np.ndarray
    num_boundary_edges: int = 0

    def get_free_vertex_plus_coords(self, mesh) -> np.ndarray:
        """Return (N, 3) coordinates of the free vertices on T+."""
        return mesh.vertices[self.free_vertex_plus]

    def get_free_vertex_minus_coords(self, mesh) -> np.ndarray:
        """Return (N, 3) coordinates of the free vertices on T-."""
        return mesh.vertices[self.free_vertex_minus]

    def validate(self, mesh) -> None:
        """Sanity-check RWG basis data against the mesh.

        Raises
        ------
        ValueError
            If any consistency check fails.
        """
        if self.num_basis != len(self.edge_index):
            raise ValueError("num_basis does not match edge_index length")

        # Free vertex must not lie on the shared edge
        for n in range(self.num_basis):
            edge = mesh.edges[self.edge_index[n]]
            ev = set(edge)
            if self.free_vertex_plus[n] in ev:
                raise ValueError(
                    f"Basis {n}: free_vertex_plus {self.free_vertex_plus[n]} "
                    f"lies on the shared edge {edge}"
                )
            if self.free_vertex_minus[n] in ev:
                raise ValueError(
                    f"Basis {n}: free_vertex_minus {self.free_vertex_minus[n]} "
                    f"lies on the shared edge {edge}"
                )
