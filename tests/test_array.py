"""Tests for antenna array support."""

import numpy as np
import pytest

from pyMoM3d.arrays.linear_array import (
    combine_meshes,
    compute_array_factor,
    uniform_excitation,
    progressive_phase_excitation,
    arbitrary_excitation,
    scan_angle_to_phase_shift,
    LinearDipoleArray,
)
from pyMoM3d.mom.excitation import (
    find_feed_edges_near_center,
    MultiPortExcitation,
)
from pyMoM3d.mesh.mesh_data import Mesh
from pyMoM3d.mesh.rwg_connectivity import compute_rwg_connectivity
from pyMoM3d.mesh.gmsh_mesher import GmshMesher
from pyMoM3d.utils.constants import c0, eta0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_plates():
    """Create two small plate meshes for testing combine_meshes."""
    mesher = GmshMesher(target_edge_length=0.01)
    m1 = mesher.mesh_plate_with_feed(width=0.03, height=0.002,
                                      feed_x=0.0, center=(0, 0, 0))
    m2 = mesher.mesh_plate_with_feed(width=0.03, height=0.002,
                                      feed_x=0.0, center=(0.04, 0, 0))
    return m1, m2


# ---------------------------------------------------------------------------
# combine_meshes tests
# ---------------------------------------------------------------------------

class TestCombineMeshes:
    def test_preserves_counts(self):
        """Combined mesh has correct vertex and triangle counts."""
        m1, m2 = _make_two_plates()
        combined, offsets = combine_meshes([m1, m2])

        assert len(combined.vertices) == len(m1.vertices) + len(m2.vertices)
        assert len(combined.triangles) == len(m1.triangles) + len(m2.triangles)
        assert offsets[0] == 0
        assert offsets[1] == len(m1.vertices)

    def test_three_meshes(self):
        """Works with 3 meshes."""
        mesher = GmshMesher(target_edge_length=0.01)
        meshes = [
            mesher.mesh_plate_with_feed(width=0.03, height=0.002,
                                         feed_x=0.0, center=(i * 0.04, 0, 0))
            for i in range(3)
        ]
        combined, offsets = combine_meshes(meshes)
        total_v = sum(len(m.vertices) for m in meshes)
        total_t = sum(len(m.triangles) for m in meshes)
        assert len(combined.vertices) == total_v
        assert len(combined.triangles) == total_t

    def test_rwg_on_combined(self):
        """RWG connectivity works on combined mesh without inter-body edges."""
        m1, m2 = _make_two_plates()
        combined, _ = combine_meshes([m1, m2])
        basis = compute_rwg_connectivity(combined)

        # Should have basis functions from both meshes
        basis1 = compute_rwg_connectivity(m1)
        basis2 = compute_rwg_connectivity(m2)

        # Combined basis count should equal sum of individual counts
        # (no shared edges between disjoint bodies)
        assert basis.num_basis == basis1.num_basis + basis2.num_basis


# ---------------------------------------------------------------------------
# find_feed_edges_near_center tests
# ---------------------------------------------------------------------------

class TestFindFeedEdgesNearCenter:
    def test_finds_edges_x_dipole(self):
        """Finds feed edges for a standard x-directed dipole at origin."""
        mesher = GmshMesher(target_edge_length=0.004)
        mesh = mesher.mesh_plate_with_feed(
            width=0.03, height=0.002, feed_x=0.0, center=(0, 0, 0))
        basis = compute_rwg_connectivity(mesh)

        indices = find_feed_edges_near_center(
            mesh, basis,
            element_center=np.array([0.0, 0.0, 0.0]),
            dipole_axis=np.array([1.0, 0.0, 0.0]),
        )
        assert len(indices) > 0

    def test_finds_edges_z_dipole(self):
        """Finds feed edges for a z-directed dipole (rotated from x)."""
        mesher = GmshMesher(target_edge_length=0.004)
        mesh = mesher.mesh_plate_with_feed(
            width=0.03, height=0.002, feed_x=0.0, center=(0, 0, 0))

        # Rotate: x->z, y->y, z->-x
        from pyMoM3d.arrays.linear_array import _rotation_matrix_from_x_to
        R = _rotation_matrix_from_x_to('z')
        rotated_verts = (R @ mesh.vertices.T).T
        mesh_rot = Mesh(rotated_verts, mesh.triangles.copy())
        basis = compute_rwg_connectivity(mesh_rot)

        indices = find_feed_edges_near_center(
            mesh_rot, basis,
            element_center=np.array([0.0, 0.0, 0.0]),
            dipole_axis=np.array([0.0, 0.0, 1.0]),
        )
        assert len(indices) > 0

    def test_correct_count_per_element(self):
        """Each element in a 2-element array gets feed edges."""
        mesher = GmshMesher(target_edge_length=0.004)
        from pyMoM3d.arrays.linear_array import _rotation_matrix_from_x_to
        R = _rotation_matrix_from_x_to('z')

        meshes = []
        positions = [np.array([-0.02, 0, 0]), np.array([0.02, 0, 0])]
        for pos in positions:
            m = mesher.mesh_plate_with_feed(
                width=0.03, height=0.002, feed_x=0.0, center=(0, 0, 0))
            rotated_verts = (R @ m.vertices.T).T + pos
            meshes.append(Mesh(rotated_verts, m.triangles.copy()))

        combined, _ = combine_meshes(meshes)
        basis = compute_rwg_connectivity(combined)

        for pos in positions:
            indices = find_feed_edges_near_center(
                combined, basis,
                element_center=pos,
                dipole_axis=np.array([0, 0, 1]),
            )
            assert len(indices) > 0, f"No feed edges found at position {pos}"


# ---------------------------------------------------------------------------
# Excitation helper tests
# ---------------------------------------------------------------------------

class TestExcitationHelpers:
    def test_uniform_excitation(self):
        w = uniform_excitation(4, voltage=2.0)
        assert w.shape == (4,)
        assert np.allclose(w, 2.0)

    def test_progressive_phase(self):
        w = progressive_phase_excitation(4, np.pi / 4, voltage=1.0)
        assert w.shape == (4,)
        assert np.isclose(np.abs(w[0]), 1.0)
        assert np.isclose(np.abs(w[3]), 1.0)
        phase_diff = np.angle(w[1]) - np.angle(w[0])
        assert np.isclose(phase_diff, np.pi / 4, atol=1e-10)

    def test_arbitrary_excitation(self):
        w = arbitrary_excitation([1, 2, 3], [0, np.pi / 2, np.pi])
        assert w.shape == (3,)
        assert np.isclose(np.abs(w[0]), 1.0)
        assert np.isclose(np.abs(w[1]), 2.0)
        assert np.isclose(np.angle(w[2]), np.pi, atol=1e-10)

    def test_scan_angle_to_phase_x(self):
        k = 2 * np.pi
        d = 0.5
        # Broadside: theta=pi/2, phi=0 -> beta = -k*d*sin(pi/2)*cos(0) = -k*d
        beta = scan_angle_to_phase_shift(np.pi / 2, 0.0, k, d, 'x')
        assert np.isclose(beta, -k * d)

    def test_scan_angle_to_phase_invalid_axis(self):
        with pytest.raises(ValueError):
            scan_angle_to_phase_shift(0.0, 0.0, 1.0, 1.0, 'w')


# ---------------------------------------------------------------------------
# compute_array_factor tests
# ---------------------------------------------------------------------------

class TestArrayFactor:
    def test_broadside_peak(self):
        """Uniform excitation has maximum AF at broadside."""
        N = 4
        k = 2 * np.pi  # lambda = 1
        d = 0.5
        positions = np.zeros((N, 3))
        for n in range(N):
            positions[n, 0] = (n - (N - 1) / 2.0) * d

        weights = uniform_excitation(N)

        # At broadside (perpendicular to array axis), all elements
        # are in phase so AF = N. Test at theta=90, phi=90 (y-axis).
        AF_bs = compute_array_factor(
            np.array([np.pi / 2]), np.array([np.pi / 2]),
            positions, weights, k)
        assert np.isclose(np.abs(AF_bs[0]), N, atol=0.01), \
            f"AF at broadside = {np.abs(AF_bs[0]):.3f}, expected {N}"

        # At phi=0, theta=90 (along array axis, endfire), AF should be < N
        # for d=0.5*lambda with centered positions
        AF_ef = compute_array_factor(
            np.array([np.pi / 2]), np.array([0.0]),
            positions, weights, k)
        assert np.abs(AF_ef[0]) < N - 0.01, \
            f"AF at endfire = {np.abs(AF_ef[0]):.3f}, should be < {N}"

    def test_steered_peak(self):
        """Progressive phase shifts the peak."""
        N = 8
        k = 2 * np.pi
        d = 0.5
        positions = np.zeros((N, 3))
        for n in range(N):
            positions[n, 0] = (n - (N - 1) / 2.0) * d

        # Steer to theta=60 deg (30 deg from broadside)
        beta = scan_angle_to_phase_shift(np.radians(60), 0.0, k, d, 'x')
        weights = progressive_phase_excitation(N, beta)

        theta = np.linspace(0.001, np.pi - 0.001, 361)
        phi = np.zeros_like(theta)

        AF = compute_array_factor(theta, phi, positions, weights, k)
        AF_mag = np.abs(AF)

        peak_idx = np.argmax(AF_mag)
        peak_theta = np.degrees(theta[peak_idx])
        assert abs(peak_theta - 60.0) < 3.0, \
            f"AF peak at {peak_theta} deg, expected ~60 deg"


# ---------------------------------------------------------------------------
# MultiPortExcitation tests
# ---------------------------------------------------------------------------

class TestMultiPortExcitation:
    def test_voltage_vector_shape(self):
        """Voltage vector has correct shape and non-zero entries."""
        mesher = GmshMesher(target_edge_length=0.004)
        mesh = mesher.mesh_plate_with_feed(
            width=0.03, height=0.002, feed_x=0.0, center=(0, 0, 0))
        basis = compute_rwg_connectivity(mesh)

        from pyMoM3d.mom.excitation import find_feed_edges
        feed = find_feed_edges(mesh, basis, feed_x=0.0)

        exc = MultiPortExcitation(
            port_feed_indices=[feed],
            voltages=np.array([1.0 + 0j]),
        )
        V = exc.compute_voltage_vector(basis, mesh, 1.0)

        assert V.shape == (basis.num_basis,)
        assert np.any(V != 0)
        # Only feed edges should be non-zero
        non_zero = np.nonzero(V)[0]
        assert set(non_zero).issubset(set(feed))

    def test_mismatched_ports_voltages(self):
        """Raises ValueError if port count != voltage count."""
        with pytest.raises(ValueError):
            MultiPortExcitation(
                port_feed_indices=[[0, 1], [2, 3]],
                voltages=np.array([1.0]),
            )


# ---------------------------------------------------------------------------
# LinearDipoleArray tests
# ---------------------------------------------------------------------------

class TestLinearDipoleArray:
    @pytest.fixture(scope='class')
    def small_array(self):
        """2-element array for fast testing."""
        freq = 5e9
        lam = c0 / freq
        return LinearDipoleArray(
            n_elements=2,
            spacing=0.5 * lam,
            frequency=freq,
            dipole_axis='z',
            array_axis='x',
            mesh_edge_length=lam / 10,
        )

    def test_mesh_created(self, small_array):
        assert small_array.mesh is not None
        assert small_array.basis is not None
        assert small_array.basis.num_basis > 0

    def test_element_positions(self, small_array):
        assert small_array.element_positions.shape == (2, 3)
        # Centered at origin, spaced along x
        assert np.isclose(small_array.element_positions[0, 0],
                          -small_array.spacing / 2, atol=1e-10)
        assert np.isclose(small_array.element_positions[1, 0],
                          small_array.spacing / 2, atol=1e-10)

    def test_feed_edges_found(self, small_array):
        assert len(small_array.element_feed_indices) == 2
        for feed in small_array.element_feed_indices:
            assert len(feed) > 0

    def test_solve_returns_current(self, small_array):
        Z = small_array.fill_impedance_matrix()
        weights = uniform_excitation(2)
        I = small_array.solve(weights)
        assert I.shape == (small_array.basis.num_basis,)
        assert np.any(I != 0)

    def test_element_currents(self, small_array):
        weights = uniform_excitation(2)
        I = small_array.solve(weights)
        currents = small_array.compute_element_currents(I)
        assert len(currents) == 2
        # Both elements should have non-zero current
        for c in currents:
            assert abs(c) > 0

    def test_element_impedances(self, small_array):
        weights = uniform_excitation(2)
        I = small_array.solve(weights)
        Z_active = small_array.compute_element_impedances(I, weights)
        assert len(Z_active) == 2
        # Impedances should be finite and have positive real part
        for z in Z_active:
            assert np.isfinite(z)
            assert z.real > 0

    def test_symmetry_uniform_excitation(self, small_array):
        """Two-element array with uniform excitation should have symmetric currents."""
        weights = uniform_excitation(2)
        I = small_array.solve(weights)
        currents = small_array.compute_element_currents(I)
        # Magnitudes should be equal (symmetric array + symmetric excitation)
        assert np.isclose(abs(currents[0]), abs(currents[1]), rtol=0.05)

    def test_dipole_axis_validation(self):
        """Should raise if dipole_axis == array_axis."""
        with pytest.raises(ValueError):
            LinearDipoleArray(
                n_elements=2, spacing=0.03, frequency=5e9,
                dipole_axis='x', array_axis='x',
            )

    def test_voltage_vector(self, small_array):
        weights = np.array([1.0, 2.0], dtype=complex)
        V = small_array.compute_voltage_vector(weights)
        assert V.shape == (small_array.basis.num_basis,)
        assert np.any(V != 0)

    def test_array_factor_method(self, small_array):
        theta = np.linspace(0.1, np.pi - 0.1, 50)
        phi = np.zeros_like(theta)
        weights = uniform_excitation(2)
        AF = small_array.compute_array_factor(theta, phi, weights)
        assert AF.shape == (50,)
        assert np.any(np.abs(AF) > 0)

    def test_get_scan_phase_shift(self, small_array):
        beta = small_array.get_scan_phase_shift(np.pi / 3, 0.0)
        assert np.isfinite(beta)
