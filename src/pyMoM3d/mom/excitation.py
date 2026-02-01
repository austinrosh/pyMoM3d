"""Excitation (voltage) vectors for MoM.

Provides abstract base class and concrete implementations for
plane wave and delta-gap feed excitations.
"""

import numpy as np
from abc import ABC, abstractmethod

from ..mesh.mesh_data import Mesh
from ..mesh.rwg_basis import RWGBasis
from ..greens.quadrature import triangle_quad_rule


class Excitation(ABC):
    """Abstract base class for excitation sources."""

    @abstractmethod
    def compute_voltage_vector(
        self, rwg_basis: RWGBasis, mesh: Mesh, k: float
    ) -> np.ndarray:
        """Compute the voltage (RHS) vector V.

        Parameters
        ----------
        rwg_basis : RWGBasis
        mesh : Mesh
        k : float
            Wavenumber (rad/m).

        Returns
        -------
        V : ndarray, shape (N,), complex128
        """
        ...


class PlaneWaveExcitation(Excitation):
    """Uniform plane wave excitation.

    E_inc(r) = E0 * exp(-j * k * k_hat . r)

    Parameters
    ----------
    E0 : ndarray, shape (3,)
        Electric field polarization vector (V/m).
    k_hat : ndarray, shape (3,)
        Unit propagation direction.
    """

    def __init__(self, E0: np.ndarray, k_hat: np.ndarray):
        self.E0 = np.asarray(E0, dtype=np.complex128)
        self.k_hat = np.asarray(k_hat, dtype=np.float64)
        self.k_hat = self.k_hat / np.linalg.norm(self.k_hat)

        # E0 must be perpendicular to k_hat
        if abs(np.dot(self.E0.real, self.k_hat)) > 1e-10:
            raise ValueError("E0 must be perpendicular to k_hat")

    def compute_voltage_vector(
        self, rwg_basis: RWGBasis, mesh: Mesh, k: float
    ) -> np.ndarray:
        N = rwg_basis.num_basis
        V = np.zeros(N, dtype=np.complex128)

        for n in range(N):
            for (tri, fv, sign, area) in [
                (rwg_basis.t_plus[n], rwg_basis.free_vertex_plus[n], +1.0, rwg_basis.area_plus[n]),
                (rwg_basis.t_minus[n], rwg_basis.free_vertex_minus[n], -1.0, rwg_basis.area_minus[n]),
            ]:
                verts = mesh.vertices[mesh.triangles[tri]]
                r_fv = mesh.vertices[fv]

                weights, bary = triangle_quad_rule(4)
                cross = np.cross(verts[1] - verts[0], verts[2] - verts[0])
                twice_area = np.linalg.norm(cross)

                scale = sign * rwg_basis.edge_length[n] / (2.0 * area)

                for i in range(len(weights)):
                    r = bary[i, 0] * verts[0] + bary[i, 1] * verts[1] + bary[i, 2] * verts[2]
                    rho = r - r_fv
                    E_inc = self.E0 * np.exp(-1j * k * np.dot(self.k_hat, r))
                    V[n] += weights[i] * np.dot(rho, E_inc) * scale * twice_area

        return V


class DeltaGapExcitation(Excitation):
    """Delta-gap voltage source at a specified edge.

    V_m = voltage for the feed basis function, 0 for all others.

    Parameters
    ----------
    edge_index : int
        Basis function index (NOT mesh edge index) for the feed.
    voltage : complex
        Applied voltage (V).
    """

    def __init__(self, basis_index: int, voltage: complex = 1.0):
        self.basis_index = basis_index
        self.voltage = complex(voltage)

    def compute_voltage_vector(
        self, rwg_basis: RWGBasis, mesh: Mesh, k: float
    ) -> np.ndarray:
        N = rwg_basis.num_basis
        V = np.zeros(N, dtype=np.complex128)
        if 0 <= self.basis_index < N:
            V[self.basis_index] = self.voltage
        return V


def find_nearest_edge(mesh: Mesh, rwg_basis: RWGBasis, point: np.ndarray) -> int:
    """Find the basis function whose shared edge midpoint is nearest to a point.

    Parameters
    ----------
    mesh : Mesh
    rwg_basis : RWGBasis
    point : ndarray, shape (3,)

    Returns
    -------
    basis_index : int
    """
    point = np.asarray(point, dtype=np.float64)
    best_dist = np.inf
    best_idx = 0

    for n in range(rwg_basis.num_basis):
        edge = mesh.edges[rwg_basis.edge_index[n]]
        midpoint = 0.5 * (mesh.vertices[edge[0]] + mesh.vertices[edge[1]])
        dist = np.linalg.norm(midpoint - point)
        if dist < best_dist:
            best_dist = dist
            best_idx = n

    return best_idx
