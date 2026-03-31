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


    def compute_mfie_voltage_vector(
        self, rwg_basis: RWGBasis, mesh: Mesh, k: float, eta: float
    ) -> np.ndarray:
        """Compute the MFIE RHS vector for a plane wave excitation.

        V_m^MFIE = ∑_{T ∈ {T_m+, T_m-}}
                   ∫_T f_m(r) · [n̂_T × H_inc(r)] dS

        where H_inc = (k̂ × E_0) * exp(−jk · r) / η.

        Parameters
        ----------
        rwg_basis : RWGBasis
        mesh : Mesh
        k : float
            Wavenumber (rad/m).
        eta : float
            Intrinsic impedance of the medium (Ω).

        Returns
        -------
        V : ndarray, shape (N,), complex128
        """
        N = rwg_basis.num_basis
        V = np.zeros(N, dtype=np.complex128)

        # H_inc(r) = (k̂ × E_0) * exp(−jk*k̂·r) / η
        H_amp = np.cross(self.k_hat, self.E0) / eta  # (3,) complex

        weights, bary = triangle_quad_rule(4)

        for n in range(N):
            for (tri, fv, sign, area) in [
                (rwg_basis.t_plus[n],  rwg_basis.free_vertex_plus[n],
                 +1.0, rwg_basis.area_plus[n]),
                (rwg_basis.t_minus[n], rwg_basis.free_vertex_minus[n],
                 -1.0, rwg_basis.area_minus[n]),
            ]:
                verts = mesh.vertices[mesh.triangles[tri]]
                r_fv  = mesh.vertices[fv]

                cross = np.cross(verts[1] - verts[0], verts[2] - verts[0])
                twice_area = np.linalg.norm(cross)

                scale = sign * rwg_basis.edge_length[n] / (2.0 * area)

                for i in range(len(weights)):
                    r = (bary[i, 0] * verts[0]
                         + bary[i, 1] * verts[1]
                         + bary[i, 2] * verts[2])
                    rho   = r - r_fv
                    H_inc = H_amp * np.exp(-1j * k * np.dot(self.k_hat, r))
                    # n̂×RWG testing: V_m = ∫ f_m · H_inc dS
                    # (n̂·f_m = 0, so only the tangential part of H_inc contributes)
                    V[n] += weights[i] * np.dot(rho, H_inc) * scale * twice_area

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
            V[self.basis_index] = self.voltage * rwg_basis.edge_length[self.basis_index]
        return V


def find_feed_edges(
    mesh: Mesh,
    rwg_basis: RWGBasis,
    feed_x: float,
    tol: float = None,
) -> list:
    """Find all interior basis functions whose shared edge crosses x=feed_x transversely.

    For strip dipole feed models, identifies edges that:
    1. Straddle the feed line (one vertex on each side, or both on it)
    2. Are approximately perpendicular to x (i.e., y-directed)

    Parameters
    ----------
    mesh : Mesh
    rwg_basis : RWGBasis
    feed_x : float
        x-coordinate of the feed line.
    tol : float, optional
        Positional tolerance. Defaults to half the minimum edge length.

    Returns
    -------
    indices : list of int
        Basis function indices for edges crossing the feed line.
    """
    if tol is None:
        # Estimate from mesh
        lengths = []
        for n in range(min(rwg_basis.num_basis, 50)):
            e = mesh.edges[rwg_basis.edge_index[n]]
            lengths.append(np.linalg.norm(
                mesh.vertices[e[1]] - mesh.vertices[e[0]]))
        tol = 0.5 * min(lengths) if lengths else 1e-6

    indices = []
    for n in range(rwg_basis.num_basis):
        e = mesh.edges[rwg_basis.edge_index[n]]
        va = mesh.vertices[e[0]]
        vb = mesh.vertices[e[1]]
        mid_x = 0.5 * (va[0] + vb[0])

        # Edge midpoint must be near the feed line
        if abs(mid_x - feed_x) > tol:
            continue

        # Edge must be approximately transverse (y-directed)
        edge_dir = vb - va
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-30:
            continue
        edge_dir /= edge_len

        # Accept if |y-component| > |x-component| (more transverse than longitudinal)
        if abs(edge_dir[1]) > abs(edge_dir[0]):
            indices.append(n)

    return indices


class StripDeltaGapExcitation(Excitation):
    """Delta-gap excitation distributed across multiple transverse edges.

    For strip dipole feed models where the gap spans multiple mesh edges.
    Applies V_m = voltage to each transverse feed edge.

    The input impedance is computed as Z_in = voltage / I_terminal,
    where I_terminal is the total current crossing the feed gap.

    Parameters
    ----------
    feed_basis_indices : list of int
        Basis function indices for edges crossing the feed gap.
    voltage : complex
        Applied voltage (V).
    """

    def __init__(self, feed_basis_indices: list, voltage: complex = 1.0):
        self.feed_basis_indices = list(feed_basis_indices)
        self.voltage = complex(voltage)

    def compute_voltage_vector(
        self, rwg_basis: RWGBasis, mesh: Mesh, k: float
    ) -> np.ndarray:
        """Compute voltage vector for the strip delta gap.

        For RWG basis functions with f_n = l_n/(2A) * rho, the Galerkin
        testing of a delta-gap field E = V_0 * delta(x) * x_hat gives:

            V_m = V_0 * integral_edge f_m . n_hat dl = V_0 * l_m

        because f . n_hat = 1 at the shared edge (an RWG identity).
        """
        N = rwg_basis.num_basis
        V = np.zeros(N, dtype=np.complex128)
        for idx in self.feed_basis_indices:
            if 0 <= idx < N:
                V[idx] = self.voltage * rwg_basis.edge_length[idx]
        return V

    def compute_input_impedance(
        self,
        I_coeffs: np.ndarray,
        rwg_basis: RWGBasis,
        mesh: Mesh,
    ) -> complex:
        """Compute input impedance from the solved current coefficients.

        For RWG with f_n = l_n/(2A)*rho, the physical terminal current
        (line integral of J across the feed gap) is:

            I_terminal = sum_n I_n * l_n

        and the input impedance is Z_in = V_0 / I_terminal.

        Parameters
        ----------
        I_coeffs : ndarray, shape (N,), complex128
            Current expansion coefficients.
        rwg_basis : RWGBasis
        mesh : Mesh

        Returns
        -------
        Z_in : complex
        """
        I_terminal = 0.0 + 0.0j
        for idx in self.feed_basis_indices:
            I_terminal += I_coeffs[idx] * rwg_basis.edge_length[idx]

        if abs(I_terminal) < 1e-30:
            return np.inf + 0j
        return self.voltage / I_terminal


def find_feed_edges_near_center(
    mesh: Mesh,
    rwg_basis: RWGBasis,
    element_center: np.ndarray,
    dipole_axis: np.ndarray,
    tol: float = None,
    perp_tol: float = None,
) -> list:
    """Find feed edges near an element center for array configurations.

    Generalized feed edge finder that works for any dipole orientation.
    Identifies basis functions whose shared edge:
    1. Has midpoint near element_center along the dipole axis (at the feed).
    2. Has midpoint near element_center in the perpendicular directions
       (belongs to this element, not a neighbor).
    3. Is more transverse than longitudinal relative to dipole_axis.

    Parameters
    ----------
    mesh : Mesh
    rwg_basis : RWGBasis
    element_center : ndarray, shape (3,)
        3D center position of the antenna element.
    dipole_axis : ndarray, shape (3,)
        Unit vector along the dipole arm direction.
    tol : float, optional
        Tolerance along dipole axis for feed location.
        Defaults to half the minimum edge length.
    perp_tol : float, optional
        Tolerance perpendicular to dipole axis for element membership.
        Defaults to 5 * tol.

    Returns
    -------
    indices : list of int
        Basis function indices for feed edges at this element.
    """
    element_center = np.asarray(element_center, dtype=np.float64)
    dipole_axis = np.asarray(dipole_axis, dtype=np.float64)
    dipole_axis = dipole_axis / np.linalg.norm(dipole_axis)

    if tol is None:
        lengths = []
        for n in range(min(rwg_basis.num_basis, 50)):
            e = mesh.edges[rwg_basis.edge_index[n]]
            lengths.append(np.linalg.norm(
                mesh.vertices[e[1]] - mesh.vertices[e[0]]))
        tol = 0.5 * min(lengths) if lengths else 1e-6

    if perp_tol is None:
        perp_tol = 5.0 * tol

    indices = []
    for n in range(rwg_basis.num_basis):
        e = mesh.edges[rwg_basis.edge_index[n]]
        va = mesh.vertices[e[0]]
        vb = mesh.vertices[e[1]]
        mid = 0.5 * (va + vb)

        # Vector from element center to edge midpoint
        r = mid - element_center

        # Check projection along dipole axis (edge at the feed location)
        proj_along = abs(np.dot(r, dipole_axis))
        if proj_along > tol:
            continue

        # Check perpendicular distance (edge belongs to this element)
        r_perp = r - np.dot(r, dipole_axis) * dipole_axis
        dist_perp = np.linalg.norm(r_perp)
        if dist_perp > perp_tol:
            continue

        # Check edge direction: more transverse than longitudinal
        edge_dir = vb - va
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-30:
            continue
        edge_dir /= edge_len

        if abs(np.dot(edge_dir, dipole_axis)) < np.linalg.norm(
            np.cross(edge_dir, dipole_axis)
        ):
            indices.append(n)

    return indices


class MultiPortExcitation(Excitation):
    """Multi-port excitation for antenna arrays.

    Each port is a list of feed basis indices driven with an independent
    complex voltage. This enables simultaneous excitation of multiple
    antenna elements with different amplitudes and phases.

    Parameters
    ----------
    port_feed_indices : list of list of int
        Feed basis indices for each port.
    voltages : ndarray, shape (N_ports,), complex128
        Complex voltage for each port.
    """

    def __init__(self, port_feed_indices: list, voltages: np.ndarray):
        self.port_feed_indices = [list(p) for p in port_feed_indices]
        self.voltages = np.asarray(voltages, dtype=np.complex128)
        if len(self.port_feed_indices) != len(self.voltages):
            raise ValueError(
                f"Number of ports ({len(self.port_feed_indices)}) must match "
                f"number of voltages ({len(self.voltages)})"
            )

    def compute_voltage_vector(
        self, rwg_basis: RWGBasis, mesh: Mesh, k: float
    ) -> np.ndarray:
        """Compute voltage vector with multi-port excitation.

        For each port p, sets V[idx] = voltages[p] * edge_length[idx]
        for all feed indices in port p.
        """
        N = rwg_basis.num_basis
        V = np.zeros(N, dtype=np.complex128)
        for p, indices in enumerate(self.port_feed_indices):
            for idx in indices:
                if 0 <= idx < N:
                    V[idx] = self.voltages[p] * rwg_basis.edge_length[idx]
        return V

    def compute_port_impedance(
        self,
        I_coeffs: np.ndarray,
        rwg_basis: RWGBasis,
        mesh: Mesh,
        port: int,
    ) -> complex:
        """Compute active (scan) impedance at a specific port.

        Z_in = voltages[port] / I_terminal, where
        I_terminal = sum(I_n * l_n) over the port's feed edges.

        Parameters
        ----------
        I_coeffs : ndarray, shape (N,), complex128
        rwg_basis : RWGBasis
        mesh : Mesh
        port : int
            Port index.

        Returns
        -------
        Z_in : complex
        """
        I_terminal = 0.0 + 0.0j
        for idx in self.port_feed_indices[port]:
            I_terminal += I_coeffs[idx] * rwg_basis.edge_length[idx]

        if abs(I_terminal) < 1e-30:
            return np.inf + 0j
        return self.voltages[port] / I_terminal


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
