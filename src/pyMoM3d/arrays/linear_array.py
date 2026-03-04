"""Linear dipole antenna array support.

Provides LinearDipoleArray class for structured linear array simulations,
plus helper functions for mesh combining, array factor computation, and
excitation weight generation.
"""

import numpy as np
import scipy.linalg

from ..mesh.mesh_data import Mesh
from ..mesh.rwg_basis import RWGBasis
from ..mesh.gmsh_mesher import GmshMesher
from ..mesh.rwg_connectivity import compute_rwg_connectivity
from ..mom.impedance import fill_impedance_matrix
from ..mom.excitation import find_feed_edges_near_center, MultiPortExcitation
from ..mom.solver import solve_direct
from ..fields.far_field import compute_far_field
from ..analysis.pattern_analysis import compute_directivity
from ..utils.constants import c0, eta0


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def combine_meshes(meshes: list) -> tuple:
    """Combine multiple disjoint meshes into one.

    Generalizes build_two_antenna_mesh from friis_validation.py to N meshes.
    Vstacks vertices, offsets and vstacks triangles.

    Parameters
    ----------
    meshes : list of Mesh
        Individual antenna meshes.

    Returns
    -------
    combined : Mesh
        Single mesh containing all antennas.
    vertex_offsets : list of int
        Cumulative vertex offset for each input mesh.
    """
    vertex_offsets = []
    all_verts = []
    all_tris = []
    offset = 0

    for m in meshes:
        vertex_offsets.append(offset)
        all_verts.append(m.vertices)
        all_tris.append(m.triangles + offset)
        offset += len(m.vertices)

    combined_verts = np.vstack(all_verts)
    combined_tris = np.vstack(all_tris)
    return Mesh(combined_verts, combined_tris), vertex_offsets


def compute_array_factor(theta, phi, element_positions, amplitudes, k):
    """Compute analytical array factor.

    AF = sum_n a_n * exp(j*k*r_hat . d_n)

    Parameters
    ----------
    theta : ndarray, shape (M,)
        Polar angles (radians).
    phi : ndarray, shape (M,)
        Azimuthal angles (radians).
    element_positions : ndarray, shape (N, 3)
        3D positions of array elements.
    amplitudes : ndarray, shape (N,), complex
        Complex excitation weights.
    k : float
        Wavenumber (rad/m).

    Returns
    -------
    AF : ndarray, shape (M,), complex
        Array factor at each observation angle.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    element_positions = np.asarray(element_positions)
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)

    # Observation direction unit vectors
    r_hat = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])  # (M, 3)

    AF = np.zeros(len(theta), dtype=np.complex128)
    for n in range(len(amplitudes)):
        phase = k * r_hat @ element_positions[n]
        AF += amplitudes[n] * np.exp(1j * phase)

    return AF


def uniform_excitation(n_elements, voltage=1.0):
    """Generate uniform excitation weights.

    Parameters
    ----------
    n_elements : int
    voltage : float or complex

    Returns
    -------
    weights : ndarray, shape (n_elements,), complex128
    """
    return np.full(n_elements, complex(voltage), dtype=np.complex128)


def progressive_phase_excitation(n_elements, phase_shift, voltage=1.0):
    """Generate progressive phase excitation weights.

    weights[n] = voltage * exp(j * n * phase_shift)

    Parameters
    ----------
    n_elements : int
    phase_shift : float
        Inter-element phase shift (radians).
    voltage : float or complex

    Returns
    -------
    weights : ndarray, shape (n_elements,), complex128
    """
    n = np.arange(n_elements)
    return complex(voltage) * np.exp(1j * n * phase_shift)


def arbitrary_excitation(amplitudes, phases):
    """Generate excitation weights from amplitude and phase arrays.

    Parameters
    ----------
    amplitudes : array-like, shape (N,)
        Amplitude per element.
    phases : array-like, shape (N,)
        Phase per element (radians).

    Returns
    -------
    weights : ndarray, shape (N,), complex128
    """
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    return amplitudes * np.exp(1j * phases)


def scan_angle_to_phase_shift(theta_scan, phi_scan, k, d, array_axis='x'):
    """Convert desired beam direction to progressive phase shift.

    Parameters
    ----------
    theta_scan : float
        Scan angle theta (radians).
    phi_scan : float
        Scan angle phi (radians).
    k : float
        Wavenumber (rad/m).
    d : float
        Element spacing (meters).
    array_axis : str
        Array axis: 'x', 'y', or 'z'.

    Returns
    -------
    beta : float
        Progressive phase shift (radians).
    """
    if array_axis == 'x':
        return -k * d * np.sin(theta_scan) * np.cos(phi_scan)
    elif array_axis == 'y':
        return -k * d * np.sin(theta_scan) * np.sin(phi_scan)
    elif array_axis == 'z':
        return -k * d * np.cos(theta_scan)
    else:
        raise ValueError(f"array_axis must be 'x', 'y', or 'z', got '{array_axis}'")


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

_AXIS_VECTORS = {
    'x': np.array([1.0, 0.0, 0.0]),
    'y': np.array([0.0, 1.0, 0.0]),
    'z': np.array([0.0, 0.0, 1.0]),
}


def _rotation_matrix_from_x_to(target_axis):
    """Compute rotation matrix that maps the x-axis to target_axis.

    The native dipole is along x in the mesh. This function returns
    the rotation matrix R such that R @ [1,0,0] = target_axis.

    Parameters
    ----------
    target_axis : str or ndarray
        'x', 'y', or 'z', or a unit vector.

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    if isinstance(target_axis, str):
        target = _AXIS_VECTORS[target_axis].copy()
    else:
        target = np.asarray(target_axis, dtype=np.float64)
        target = target / np.linalg.norm(target)

    source = np.array([1.0, 0.0, 0.0])

    # Check if source and target are the same
    if np.allclose(source, target):
        return np.eye(3)

    # Check if source and target are opposite
    if np.allclose(source, -target):
        # 180-degree rotation about z-axis
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

    # Rodrigues' rotation formula
    v = np.cross(source, target)
    s = np.linalg.norm(v)
    c = np.dot(source, target)

    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return R


# ---------------------------------------------------------------------------
# LinearDipoleArray class
# ---------------------------------------------------------------------------

class LinearDipoleArray:
    """Linear array of strip dipole antennas.

    Parameters
    ----------
    n_elements : int
        Number of array elements.
    spacing : float
        Element spacing in meters.
    frequency : float
        Operating frequency in Hz.
    dipole_length : float, optional
        Dipole arm length. Default: lambda/2.
    strip_width : float, optional
        Strip width. Default: lambda/100.
    dipole_axis : str
        Dipole orientation axis: 'x', 'y', or 'z'. Default: 'z'.
    array_axis : str
        Array layout axis: 'x', 'y', or 'z'. Default: 'x'.
    mesh_edge_length : float, optional
        Target mesh edge length. Default: lambda/15.
    """

    def __init__(
        self,
        n_elements: int,
        spacing: float,
        frequency: float,
        dipole_length: float = None,
        strip_width: float = None,
        dipole_axis: str = 'z',
        array_axis: str = 'x',
        mesh_edge_length: float = None,
    ):
        self.n_elements = n_elements
        self.spacing = spacing
        self.frequency = frequency
        self.lam = c0 / frequency
        self.k = 2.0 * np.pi * frequency / c0

        self.dipole_length = dipole_length if dipole_length is not None else self.lam / 2
        self.strip_width = strip_width if strip_width is not None else self.lam / 100
        self.dipole_axis_name = dipole_axis
        self.array_axis_name = array_axis
        self.mesh_edge_length = mesh_edge_length if mesh_edge_length is not None else self.lam / 15

        self.dipole_axis_vec = _AXIS_VECTORS[dipole_axis].copy()
        self.array_axis_vec = _AXIS_VECTORS[array_axis].copy()

        if dipole_axis == array_axis:
            raise ValueError("dipole_axis and array_axis must be different")

        # Built by _build_array
        self.mesh = None
        self.basis = None
        self.element_feed_indices = None
        self.element_positions = None

        # Cached Z-matrix and factorization
        self._Z = None
        self._lu = None
        self._piv = None

        self._build_array()

    def _build_array(self):
        """Build the combined array mesh and find feed edges."""
        # Compute element center positions centered at origin
        positions = np.zeros((self.n_elements, 3))
        for n in range(self.n_elements):
            offset = (n - (self.n_elements - 1) / 2.0) * self.spacing
            positions[n] = offset * self.array_axis_vec
        self.element_positions = positions

        # Rotation matrix from native x-axis to dipole_axis
        R = _rotation_matrix_from_x_to(self.dipole_axis_name)

        mesher = GmshMesher(target_edge_length=self.mesh_edge_length)

        element_meshes = []
        for n in range(self.n_elements):
            # Mesh in native orientation (dipole along x, strip along y)
            m = mesher.mesh_plate_with_feed(
                width=self.dipole_length,
                height=self.strip_width,
                feed_x=0.0,
                center=(0, 0, 0),
            )

            # Rotate vertices to desired orientation
            rotated_verts = (R @ m.vertices.T).T

            # Translate to element position
            rotated_verts += positions[n]

            element_meshes.append(Mesh(rotated_verts, m.triangles.copy()))

        # Combine all element meshes
        self.mesh, _ = combine_meshes(element_meshes)

        # Compute RWG basis on combined mesh
        self.basis = compute_rwg_connectivity(self.mesh)

        # Find feed edges for each element
        self.element_feed_indices = []
        for n in range(self.n_elements):
            feed_idx = find_feed_edges_near_center(
                self.mesh, self.basis,
                element_center=positions[n],
                dipole_axis=self.dipole_axis_vec,
            )
            self.element_feed_indices.append(feed_idx)

    def fill_impedance_matrix(self, **kwargs):
        """Fill the impedance matrix for the combined array mesh.

        Caches the result and LU factorization for repeated solves.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to fill_impedance_matrix().

        Returns
        -------
        Z : ndarray, shape (N, N), complex128
        """
        self._Z = fill_impedance_matrix(
            self.basis, self.mesh, self.k, eta0, **kwargs
        )
        # Cache LU factorization for fast repeated solves
        self._lu, self._piv = scipy.linalg.lu_factor(self._Z)
        return self._Z

    def compute_voltage_vector(self, weights):
        """Build voltage vector from per-element complex weights.

        Parameters
        ----------
        weights : ndarray, shape (n_elements,), complex128
            Complex excitation weight per element.

        Returns
        -------
        V : ndarray, shape (N_basis,), complex128
        """
        weights = np.asarray(weights, dtype=np.complex128)
        exc = MultiPortExcitation(self.element_feed_indices, weights)
        return exc.compute_voltage_vector(self.basis, self.mesh, self.k)

    def solve(self, weights, Z=None, method='direct'):
        """Solve for current coefficients given excitation weights.

        Parameters
        ----------
        weights : ndarray, shape (n_elements,), complex128
        Z : ndarray, optional
            Pre-computed impedance matrix. If None, uses cached or computes.
        method : str
            'direct' (default) or 'gmres'.

        Returns
        -------
        I : ndarray, shape (N_basis,), complex128
        """
        V = self.compute_voltage_vector(weights)

        if Z is not None:
            return solve_direct(Z, V)

        if self._Z is None:
            self.fill_impedance_matrix()

        # Use cached LU factorization for speed
        if self._lu is not None:
            return scipy.linalg.lu_solve((self._lu, self._piv), V)

        return solve_direct(self._Z, V)

    def compute_element_currents(self, I):
        """Compute terminal current per element.

        I_term_n = sum(I_m * l_m) for m in element n's feed edges.

        Parameters
        ----------
        I : ndarray, shape (N_basis,), complex128

        Returns
        -------
        currents : list of complex
        """
        currents = []
        for feed_indices in self.element_feed_indices:
            I_term = 0.0 + 0.0j
            for idx in feed_indices:
                I_term += I[idx] * self.basis.edge_length[idx]
            currents.append(I_term)
        return currents

    def compute_element_impedances(self, I, weights):
        """Compute active (scan) impedance per element.

        Z_in_n = weights[n] / I_term_n

        Parameters
        ----------
        I : ndarray, shape (N_basis,), complex128
        weights : ndarray, shape (n_elements,), complex128

        Returns
        -------
        impedances : list of complex
        """
        weights = np.asarray(weights, dtype=np.complex128)
        currents = self.compute_element_currents(I)
        impedances = []
        for n in range(self.n_elements):
            if abs(currents[n]) < 1e-30:
                impedances.append(np.inf + 0j)
            else:
                impedances.append(weights[n] / currents[n])
        return impedances

    def compute_far_field(self, I, theta, phi):
        """Compute far-field pattern.

        Parameters
        ----------
        I : ndarray, shape (N_basis,), complex128
        theta : ndarray, shape (M,)
        phi : ndarray, shape (M,)

        Returns
        -------
        E_theta : ndarray, shape (M,), complex128
        E_phi : ndarray, shape (M,), complex128
        """
        return compute_far_field(I, self.basis, self.mesh, self.k, eta0,
                                 theta, phi)

    def compute_directivity(self, I, n_theta=91, n_phi=72):
        """Compute full directivity pattern.

        Parameters
        ----------
        I : ndarray, shape (N_basis,), complex128
        n_theta : int
        n_phi : int

        Returns
        -------
        D : ndarray, shape (n_theta, n_phi)
        D_max : float
        D_max_dBi : float
        """
        theta_grid = np.linspace(0.001, np.pi - 0.001, n_theta)
        phi_grid = np.linspace(0.0, 2.0 * np.pi - 2.0 * np.pi / n_phi, n_phi)

        E_th_2d = np.zeros((n_theta, n_phi), dtype=np.complex128)
        E_ph_2d = np.zeros((n_theta, n_phi), dtype=np.complex128)
        for j in range(n_phi):
            E_th_2d[:, j], E_ph_2d[:, j] = compute_far_field(
                I, self.basis, self.mesh, self.k, eta0,
                theta_grid, np.full_like(theta_grid, phi_grid[j]))

        D, D_max, D_max_dBi = compute_directivity(
            E_th_2d, E_ph_2d, theta_grid, phi_grid, eta0)
        return D, D_max, D_max_dBi

    def compute_array_factor(self, theta, phi, weights):
        """Compute analytical array factor with stored element positions.

        Parameters
        ----------
        theta : ndarray
        phi : ndarray
        weights : ndarray, shape (n_elements,), complex128

        Returns
        -------
        AF : ndarray, complex128
        """
        return compute_array_factor(
            theta, phi, self.element_positions,
            np.asarray(weights, dtype=np.complex128), self.k)

    def get_scan_phase_shift(self, theta_scan, phi_scan):
        """Compute progressive phase shift for a desired scan angle.

        Parameters
        ----------
        theta_scan : float
        phi_scan : float

        Returns
        -------
        beta : float
        """
        return scan_angle_to_phase_shift(
            theta_scan, phi_scan, self.k, self.spacing,
            self.array_axis_name)
