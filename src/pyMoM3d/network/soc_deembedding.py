"""Short-Open Calibration (SOC) de-embedding for planar MoM.

Implements the VSOC method of Okhmatovski et al. (2003) to remove port
discontinuity artifacts from MoM S-parameter extraction.  The method
mirrors the feed network geometry about the reference plane, applies
symmetric (short) and antisymmetric (open) excitations, and extracts the
ABCD error matrix of each port's feed network.

Algorithm (scalar / single-mode per port)
-----------------------------------------
1. Mirror the feed submesh about the reference plane.
2. Combine original feed + mirror into one mesh; compute RWG basis.
3. Solve the 2-port (original port + mirror port) → Y-matrix + currents.
4. Superpose currents for symmetric (V = [+1, +1]) and antisymmetric
   (V = [+1, −1]) excitations.
5. Extract terminal currents I_in^s, I_in^o at the input port and
   I_ref^s at the reference-plane seam edges.
6. Compute the ABCD error matrix:

       A =  I_ref^s / (I_in^o − I_in^s)
       B = −V_in / I_ref^s
       C =  I_in^o · I_ref^s / (I_in^o − I_in^s)
       D = −I_in^s / I_ref^s

   where V_in = V_ref = 1 V.

7. De-embed: T_DUT = T_err1⁻¹ · T_total · T_err2⁻¹

References
----------
[1] Okhmatovski, Morsey, Cangellaris, "On Deembedding of Port
    Discontinuities in Full-Wave CAD Models of Multiport Circuits,"
    IEEE Trans. MTT, vol. 51, no. 12, Dec. 2003.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..mesh.mesh_data import Mesh
from ..mesh.mirror import mirror_mesh_x, combine_meshes, extract_submesh
from ..mesh.rwg_connectivity import compute_rwg_connectivity
from .port import Port
from .network_result import NetworkResult

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# ABCD / S-parameter conversion utilities
# -----------------------------------------------------------------------

def abcd_to_s(T: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Convert 2×2 ABCD matrix to S-parameter matrix.

    Uses the standard conversion (Pozar, Microwave Engineering):

        S11 = (A + B/Z0 − C·Z0 − D) / Δ
        S12 = 2·(AD − BC) / Δ
        S21 = 2 / Δ
        S22 = (−A + B/Z0 − C·Z0 + D) / Δ

    where Δ = A + B/Z0 + C·Z0 + D.

    Parameters
    ----------
    T : ndarray, shape (2, 2), complex
        ABCD transmission matrix.
    Z0 : float
        Reference impedance (Ω).

    Returns
    -------
    S : ndarray, shape (2, 2), complex
    """
    A, B, C, D = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
    denom = A + B / Z0 + C * Z0 + D
    det_T = A * D - B * C
    S = np.array([
        [A + B / Z0 - C * Z0 - D, 2.0 * det_T],
        [2.0,                      -A + B / Z0 - C * Z0 + D],
    ], dtype=np.complex128) / denom
    return S


def s_to_abcd(S: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Convert 2×2 S-parameter matrix to ABCD matrix.

    Parameters
    ----------
    S : ndarray, shape (2, 2), complex
    Z0 : float
        Reference impedance (Ω).

    Returns
    -------
    T : ndarray, shape (2, 2), complex
    """
    S11, S12, S21, S22 = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
    denom = 2.0 * S21
    A = ((1 + S11) * (1 - S22) + S12 * S21) / denom
    B = Z0 * ((1 + S11) * (1 + S22) - S12 * S21) / denom
    C = ((1 - S11) * (1 - S22) - S12 * S21) / (Z0 * denom)
    D = ((1 - S11) * (1 + S22) + S12 * S21) / denom
    return np.array([[A, B], [C, D]], dtype=np.complex128)


def y_to_abcd(Y: np.ndarray) -> np.ndarray:
    """Convert 2×2 Y-parameter matrix to ABCD matrix.

    Parameters
    ----------
    Y : ndarray, shape (2, 2), complex

    Returns
    -------
    T : ndarray, shape (2, 2), complex
    """
    Y11, Y12, Y21, Y22 = Y[0, 0], Y[0, 1], Y[1, 0], Y[1, 1]
    A = -Y22 / Y21
    B = -1.0 / Y21
    C = -(Y11 * Y22 - Y12 * Y21) / Y21
    D = -Y11 / Y21
    return np.array([[A, B], [C, D]], dtype=np.complex128)


def abcd_to_y(T: np.ndarray) -> np.ndarray:
    """Convert 2×2 ABCD matrix to Y-parameter matrix.

    Parameters
    ----------
    T : ndarray, shape (2, 2), complex

    Returns
    -------
    Y : ndarray, shape (2, 2), complex
    """
    A, B, C, D = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
    Y11 = D / B
    Y12 = -(A * D - B * C) / B
    Y21 = -1.0 / B
    Y22 = A / B
    return np.array([[Y11, Y12], [Y21, Y22]], dtype=np.complex128)


def invert_abcd(T: np.ndarray) -> np.ndarray:
    """Invert a 2×2 ABCD matrix (reciprocal network: det T = 1).

    For a reciprocal network, T⁻¹ = [[D, −B], [−C, A]].

    Parameters
    ----------
    T : ndarray, shape (2, 2), complex

    Returns
    -------
    T_inv : ndarray, shape (2, 2), complex
    """
    A, B, C, D = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
    det = A * D - B * C
    return np.array([[D, -B], [-C, A]], dtype=np.complex128) / det


# -----------------------------------------------------------------------
# Seam-edge identification
# -----------------------------------------------------------------------

def _find_seam_edges(
    mesh: Mesh,
    rwg_basis,
    x_ref: float,
    x_orig_side: float,
    tol: float = 1e-8,
) -> tuple[list[int], list[int]]:
    """Find RWG basis functions whose shared edge lies on the reference plane.

    Returns basis indices and signs such that positive current corresponds
    to flow from the *original* side toward the *mirror* side.

    Parameters
    ----------
    mesh : Mesh
        Combined (original + mirror) mesh.
    rwg_basis : RWGBasis
        RWG basis of the combined mesh.
    x_ref : float
        x-coordinate of the reference plane (seam).
    x_orig_side : float
        An x-coordinate on the original (non-mirrored) side, used to
        determine sign orientation.  Typically the port x-coordinate.
    tol : float
        Tolerance for matching edge midpoints to x_ref.

    Returns
    -------
    indices : list of int
        RWG basis indices at the seam.
    signs : list of int
        +1 if current flows from original → mirror, −1 otherwise.
    """
    indices = []
    signs = []

    verts = mesh.vertices
    edges = mesh.edges  # shape (E, 2)
    for n in range(rwg_basis.num_basis):
        # Shared-edge midpoint from mesh.edges
        edge = edges[rwg_basis.edge_index[n]]
        v0, v1 = edge[0], edge[1]
        mid_x = 0.5 * (verts[v0, 0] + verts[v1, 0])

        if abs(mid_x - x_ref) < tol:
            indices.append(n)
            # Determine sign: T+ centroid should be on original side
            # for positive flow from original → mirror
            tri_plus = rwg_basis.t_plus[n]
            centroid_plus_x = verts[mesh.triangles[tri_plus]].mean(axis=0)[0]
            # If T+ is on the original side (same side as port),
            # current flows from original → mirror → sign = +1
            orig_side = (x_orig_side < x_ref)
            plus_on_orig = (centroid_plus_x < x_ref) if orig_side else (centroid_plus_x > x_ref)
            signs.append(+1 if plus_on_orig else -1)

    return indices, signs


def _seam_current(
    I_coeffs: np.ndarray,
    rwg_basis,
    seam_indices: list[int],
    seam_signs: list[int],
) -> complex:
    """Compute net current crossing the seam from stored RWG coefficients.

    Parameters
    ----------
    I_coeffs : ndarray, shape (N,), complex
    rwg_basis : RWGBasis
    seam_indices, seam_signs : list
        From :func:`_find_seam_edges`.

    Returns
    -------
    I_ref : complex
        Net current crossing the reference plane.
    """
    I_ref = 0.0 + 0.0j
    for idx, sgn in zip(seam_indices, seam_signs):
        I_ref += sgn * I_coeffs[idx] * rwg_basis.edge_length[idx]
    return I_ref


# -----------------------------------------------------------------------
# SOCDeembedding class
# -----------------------------------------------------------------------

class SOCDeembedding:
    """Short-Open Calibration de-embedding for planar MoM.

    Removes port discontinuity artifacts from MoM S-parameter extraction
    using the VSOC method of Okhmatovski et al. (2003).

    Parameters
    ----------
    simulation : Simulation
        Fully initialised simulation (mesh, RWG basis, config with
        layer_stack if applicable).  Used to derive operator settings
        for the mirrored-structure solves.
    ports : list of Port
        Port definitions on the full mesh.
    reference_plane_x : list of float
        x-coordinate of the reference plane for each port.
    port_x : list of float, optional
        x-coordinate of each port's delta-gap.  If None, inferred from
        the first feed edge midpoint of each port.
    Z0 : float, optional
        Reference impedance for S-parameter normalisation (Ω).
    symmetric : bool, optional
        If True (default), assume all error boxes are identical and
        compute only one.  Set False for asymmetric feed networks.

    Examples
    --------
    >>> soc = SOCDeembedding(sim, [p1, p2],
    ...                       reference_plane_x=[x_ref1, x_ref2])
    >>> for result in raw_results:
    ...     cal_result = soc.deembed(result)
    """

    def __init__(
        self,
        simulation,
        ports: List[Port],
        reference_plane_x: List[float],
        port_x: Optional[List[float]] = None,
        Z0: float = 50.0,
        symmetric: bool = True,
        strip_z: Optional[float] = None,
        use_qs: bool = False,
        probe_feeds: bool = False,
        n_dielectric_images: int = 0,
    ):
        self.sim = simulation
        self.ports = list(ports)
        self.reference_plane_x = list(reference_plane_x)
        self.Z0 = float(Z0)
        self.symmetric = symmetric
        self.strip_z = strip_z
        self.use_qs = use_qs
        self.probe_feeds = probe_feeds
        self.n_dielectric_images = n_dielectric_images

        if len(self.reference_plane_x) != len(self.ports):
            raise ValueError(
                f"reference_plane_x length ({len(self.reference_plane_x)}) "
                f"must match number of ports ({len(self.ports)})"
            )

        # Infer port x-coordinates from feed edge midpoints
        if port_x is not None:
            self.port_x = list(port_x)
        else:
            self.port_x = []
            verts = self.sim.mesh.vertices
            edges = self.sim.mesh.edges
            for port in self.ports:
                idx = port.feed_basis_indices[0]
                edge = edges[self.sim.basis.edge_index[idx]]
                px = 0.5 * (verts[edge[0], 0] + verts[edge[1], 0])
                self.port_x.append(float(px))

        # Cache for error-box ABCDs: {(port_idx, freq): ndarray}
        self._abcd_cache: dict = {}

    # ------------------------------------------------------------------
    # Error-box computation
    # ------------------------------------------------------------------

    def compute_error_abcd(self, port_index: int, freq: float) -> np.ndarray:
        """Compute the 2×2 ABCD error matrix for one port at one frequency.

        Builds the mirrored feed structure, solves the 2-port system, and
        extracts the ABCD from the Y-matrix of the mirrored 2-port.

        For a symmetric, reciprocal error box (A = D, AD − BC = 1), the
        mirrored 2-port Y-matrix yields:

            A² = (Y21 − Y11) / (2·Y21)
            B  = −1 / (2·A·Y21)
            C  = (A² − 1) / B
            D  = A

        This avoids the seam-current extraction that fails with RWG basis
        functions (net current through the seam is zero by symmetry under
        symmetric excitation).

        Parameters
        ----------
        port_index : int
            Index into ``self.ports``.
        freq : float
            Frequency (Hz).

        Returns
        -------
        T_err : ndarray, shape (2, 2), complex128
            ABCD error matrix of the feed network.
        """
        key = (port_index, freq)
        if key in self._abcd_cache:
            return self._abcd_cache[key]

        x_port = self.port_x[port_index]
        x_ref = self.reference_plane_x[port_index]

        # 1. Extract feed submesh with margin beyond port so port edges
        #    remain interior (not boundary) after extraction.
        stats = self.sim.mesh.get_statistics()
        margin = 2.0 * stats['mean_edge_length']
        if x_port < x_ref:
            x_lo = x_port - margin
            x_hi = x_ref
        else:
            x_lo = x_ref
            x_hi = x_port + margin
        feed_sub, _ = extract_submesh(self.sim.mesh, x_min=x_lo, x_max=x_hi)

        # 2. Mirror the feed about the reference plane
        mirrored = mirror_mesh_x(feed_sub, x_plane=x_ref)

        # 3. Combine feed + mirror
        combined = combine_meshes(feed_sub, mirrored)
        basis_comb = compute_rwg_connectivity(combined)

        # 4. Create ports on the combined mesh with correct feed signs
        from ..mom.excitation import (
            find_feed_edges, find_edge_port_feed_edges, compute_feed_signs,
        )
        x_mirror_port = 2.0 * x_ref - x_port

        if self.strip_z is not None:
            feed_idx_orig = find_edge_port_feed_edges(
                combined, basis_comb, port_x=x_port, strip_z=self.strip_z)
            feed_idx_mirr = find_edge_port_feed_edges(
                combined, basis_comb, port_x=x_mirror_port, strip_z=self.strip_z)
        else:
            feed_idx_orig = find_feed_edges(combined, basis_comb, feed_x=x_port)
            feed_idx_mirr = find_feed_edges(combined, basis_comb, feed_x=x_mirror_port)
        signs_orig = compute_feed_signs(combined, basis_comb, feed_idx_orig)
        port_orig = Port(
            name='orig', feed_basis_indices=feed_idx_orig,
            feed_signs=signs_orig,
        )

        signs_mirr = compute_feed_signs(combined, basis_comb, feed_idx_mirr)
        port_mirr = Port(
            name='mirror', feed_basis_indices=feed_idx_mirr,
            feed_signs=signs_mirr,
        )

        # 5. Solve the 2-port mirrored structure → Y-matrix
        from ..simulation import Simulation, SimulationConfig

        cfg = SimulationConfig(
            frequency=freq,
            excitation=None,
            formulation=self.sim.config.formulation,
            quad_order=self.sim.config.quad_order,
            near_threshold=self.sim.config.near_threshold,
            backend=self.sim.config.backend,
            layer_stack=self.sim.config.layer_stack,
            source_layer_name=self.sim.config.source_layer_name,
            gf_backend=getattr(self.sim.config, 'gf_backend', 'auto'),
        )
        sim_mirror = Simulation(cfg, mesh=combined)

        if self.use_qs:
            from ..mom.quasi_static import QuasiStaticSolver
            qs = QuasiStaticSolver(
                sim_mirror, [port_orig, port_mirr],
                Z0=self.Z0, store_currents=False,
                probe_feeds=self.probe_feeds,
                n_dielectric_images=self.n_dielectric_images,
            )
            [result] = qs.extract([freq])
        else:
            from .extractor import NetworkExtractor
            extractor = NetworkExtractor(
                sim_mirror, [port_orig, port_mirr],
                Z0=self.Z0, store_currents=False,
            )
            [result] = extractor.extract([freq])

        # 6. Extract error-box ABCD from mirrored 2-port Y-matrix.
        #
        #    The mirrored structure = error_box @ reversed(error_box).
        #    For a symmetric reciprocal error box (A=D, AD-BC=1):
        #
        #        T_full = [[2AD-1, 2AB], [2CD, 2AD-1]]
        #
        #    Converting to Y:
        #        Y11 = Y22 = (2AD-1)/(2AB)
        #        Y21 = Y12 = -1/(2AB)
        #
        #    Inverting:
        #        A² = (Y21 - Y11) / (2·Y21)
        #        B  = -1 / (2·A·Y21)
        #        C  = (A² - 1) / B
        #        D  = A
        Y = result.Y_matrix
        Y11 = Y[0, 0]
        Y21 = Y[1, 0]

        if abs(Y21) < 1e-30:
            raise RuntimeError(
                f"SOC: Y21 ≈ 0 at {freq/1e9:.3f} GHz — "
                f"no coupling between original and mirror ports."
            )

        A_sq = (Y21 - Y11) / (2.0 * Y21)

        # Choose the branch of sqrt closest to +1 (short feed section:
        # A = cos(βd) ≈ 1 for βd ≪ 1).
        A = np.sqrt(A_sq)
        if A.real < 0:
            A = -A

        B = -1.0 / (2.0 * A * Y21)
        C = (A_sq - 1.0) / B
        D = A

        T_err = np.array([[A, B], [C, D]], dtype=np.complex128)

        # Verify reciprocity: det(T) should be ≈ 1
        det_T = A * D - B * C
        logger.info(
            f"SOC port {port_index} @ {freq/1e9:.3f} GHz: "
            f"A={A:.4e}, B={B:.4e}, C={C:.4e}, D={D:.4e}, det={det_T:.4e}"
        )
        if abs(det_T - 1.0) > 0.1:
            logger.warning(
                f"SOC port {port_index}: det(T_err) = {det_T:.4f} "
                f"(expected 1.0 for reciprocal network)"
            )

        self._abcd_cache[key] = T_err
        return T_err

    # ------------------------------------------------------------------
    # De-embedding
    # ------------------------------------------------------------------

    def deembed(self, network_result: NetworkResult) -> NetworkResult:
        """De-embed error boxes from a raw NetworkResult.

        For a 2-port DUT:
            T_DUT = T_err1⁻¹ · T_total · T_err2⁻¹

        For a 1-port DUT:
            Γ_DUT = (Γ_meas − S11_err) / (S22_err · (Γ_meas − S11_err) + S21_err · S12_err)
            (simplified 1-port de-embedding via error-box S-parameters)

        Parameters
        ----------
        network_result : NetworkResult
            Raw (uncalibrated) network result from NetworkExtractor.

        Returns
        -------
        NetworkResult
            De-embedded result with corrected Z_matrix.
        """
        freq = network_result.frequency
        P = len(self.ports)

        if P == 1:
            return self._deembed_1port(network_result, freq)
        elif P == 2:
            return self._deembed_2port(network_result, freq)
        else:
            raise NotImplementedError(
                f"SOC de-embedding for {P}-port networks is not yet implemented. "
                f"Use 1-port or 2-port."
            )

    def _deembed_2port(
        self, result: NetworkResult, freq: float
    ) -> NetworkResult:
        """De-embed a 2-port network via ABCD cascade."""
        # Compute error boxes
        T_err1 = self.compute_error_abcd(0, freq)
        if self.symmetric:
            T_err2 = T_err1.copy()
        else:
            T_err2 = self.compute_error_abcd(1, freq)

        # Convert raw Y → ABCD
        T_total = y_to_abcd(result.Y_matrix)

        # De-embed: T_DUT = T_err1^{-1} @ T_total @ T_err2^{-1}
        T_err1_inv = invert_abcd(T_err1)
        T_err2_inv = invert_abcd(T_err2)
        T_dut = T_err1_inv @ T_total @ T_err2_inv

        # Convert DUT ABCD → S → Z
        S_dut = abcd_to_s(T_dut, Z0=result.Z0)
        I_id = np.eye(2, dtype=np.complex128)
        Z_dut = result.Z0 * np.linalg.solve(I_id - S_dut, I_id + S_dut)

        return NetworkResult(
            frequency=freq,
            Z_matrix=Z_dut,
            port_names=list(result.port_names),
            Z0=result.Z0,
            condition_number=result.condition_number,
        )

    def _deembed_1port(
        self, result: NetworkResult, freq: float
    ) -> NetworkResult:
        """De-embed a 1-port network via error-box S-parameters."""
        T_err = self.compute_error_abcd(0, freq)
        S_err = abcd_to_s(T_err, Z0=result.Z0)

        # Measured reflection coefficient
        Gamma_meas = result.S_matrix[0, 0]

        # 1-port de-embedding:
        # Γ_DUT = (Γ_meas − S11_err) / (S22_err · Γ_meas − det(S_err))
        det_S = S_err[0, 0] * S_err[1, 1] - S_err[0, 1] * S_err[1, 0]
        Gamma_dut = (Gamma_meas - S_err[0, 0]) / (
            S_err[1, 1] * Gamma_meas - det_S
        )

        # Convert back to Z
        if abs(1.0 - Gamma_dut) > 1e-30:
            Z_dut = result.Z0 * (1.0 + Gamma_dut) / (1.0 - Gamma_dut)
        else:
            Z_dut = np.inf + 0j
        Z_mat = np.array([[Z_dut]], dtype=np.complex128)

        return NetworkResult(
            frequency=freq,
            Z_matrix=Z_mat,
            port_names=list(result.port_names),
            Z0=result.Z0,
            condition_number=result.condition_number,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def deembed_sweep(
        self, results: List[NetworkResult]
    ) -> List[NetworkResult]:
        """De-embed a list of NetworkResults (frequency sweep).

        Parameters
        ----------
        results : list of NetworkResult

        Returns
        -------
        list of NetworkResult
        """
        return [self.deembed(r) for r in results]
