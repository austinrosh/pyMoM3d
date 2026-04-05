"""Quasi-static MoM solver for planar circuits on layered substrates.

Separates the EFIE impedance matrix into frequency-independent inductance
(L) and elastance (P = C⁻¹) matrices, enabling algebraic frequency sweeps
without re-filling the matrix at each frequency.  This is equivalent to
Keysight Momentum's "RF mode."

Algorithm
---------
1. Compute static Green's functions G_A (vector potential) and G_φ (scalar
   potential) with PEC ground-plane images.  These are frequency-independent.

2. Assemble:
   - L_raw (N×N): vector-potential integral ∫∫ f_m · G_A · f_n dS dS'
   - G_s   (T×T): scalar Green's matrix ∫∫ G_φ dS dS'
   - D     (T×N): RWG divergence matrix (sparse)
   - P = D^T @ G_s @ D  (N×N): elastance matrix

3. At each frequency ω = 2πf:
       Z(ω) = jωμ₀ · L_raw + 1/(jωε₀ε_r) · P
   Solve Z @ I = V and extract Y/Z/S parameters.

Probe feed mode
---------------
When ``probe_feeds=True``, vertical probe current sources are added at
each port location.  Each probe adds one degree of freedom to the system.
The probe's tip charge is distributed among the strip triangles at the
junction vertex (attachment mode), ensuring current continuity and avoiding
the point-charge singularity.

The probe provides a conductive return path through the PEC ground at all
frequencies, fixing the low-frequency failure of strip delta-gap ports.

References: Okhmatovski et al., IEEE TMTT 2003 (via-port model);
Momentum documentation (grounded source); Makarov (probe-fed patch).

Advantages over full-wave MPIE
------------------------------
- One matrix fill for all frequencies (10–100× speedup for sweeps)
- With probe feeds: well-conditioned port model at all frequencies
- Correct Z-matrix in quasi-static limit (validated within ~4% of full-wave)

Limitations
-----------
- Valid only when the structure is electrically small (max dimension < λ/2)
- No radiation effects (no wave propagation, no coupling to surface waves)
- PEC-only model omits dielectric enhancement of G_φ; ε_r correction
  is applied via the SP prefactor (Formulation C convention)
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ..utils.constants import c0, mu0, eps0, eta0
from ..network.network_result import NetworkResult


class QuasiStaticSolver:
    """Quasi-static MoM solver with algebraic frequency sweep.

    Parameters
    ----------
    simulation : Simulation
        Fully initialised simulation (mesh and RWG basis computed).
    ports : list of Port
        Port definitions for network extraction.
    Z0 : float
        Reference impedance for S-parameter normalisation (Ω).
    store_currents : bool
        If True, store current coefficients in NetworkResult.
    n_dielectric_images : int
        Number of dielectric image terms for capacitance.
    probe_feeds : bool
        If True, add vertical probe current sources from PEC ground to
        the strip at each port location.  Fixes low-frequency port failure.
    probe_radius : float
        Wire radius for the probe feeds (m).  Default 0.2 mm.
    """

    def __init__(
        self,
        simulation,
        ports,
        Z0: float = 50.0,
        store_currents: bool = False,
        n_dielectric_images: int = 0,
        probe_feeds: bool = False,
        probe_radius: float = 0.2e-3,
    ):
        self.sim = simulation
        self.ports = list(ports)
        self.Z0 = float(Z0)
        self.store_currents = store_currents
        self.n_diel_images = n_dielectric_images
        self.probe_feeds = probe_feeds
        self.probe_radius = probe_radius

        self._precompute()

    def _precompute(self):
        """Fill L_raw, G_s, D, P once using static Green's functions."""
        from ..greens.layered.static import StaticLayeredGF
        from ..mom.operators.efie_layered import MultilayerEFIEOperator
        from ..mom.assembly import fill_matrix
        from ..mom.aefie import build_divergence_matrix, fill_scalar_green_matrix

        mesh = self.sim.mesh
        basis = self.sim.basis

        if self.sim.config.layer_stack is None:
            raise ValueError(
                "QuasiStaticSolver requires a layer_stack in SimulationConfig"
            )
        static_gf = StaticLayeredGF(
            self.sim.config.layer_stack,
            source_layer_name=self.sim.config.source_layer_name,
            n_dielectric_images=self.n_diel_images,
        )

        k_ref = complex(static_gf.wavenumber)
        eta_ref = complex(static_gf.wave_impedance)

        # --- L_raw: vector-potential integral (N×N) ---
        op_vp = MultilayerEFIEOperator(static_gf, a_only=True)
        Z_A = fill_matrix(
            op_vp, basis, mesh,
            k_ref, eta_ref,
            quad_order=self.sim.config.quad_order,
            near_threshold=self.sim.config.near_threshold,
            backend='numpy',
        )
        self.L_raw = Z_A / (1j * k_ref * eta_ref)
        self._Z_A_ref = Z_A
        self._k_ref = k_ref
        self._eps_r = static_gf.backend.eps_r

        # --- G_s: scalar Green's matrix (T×T) ---
        self.G_s = fill_scalar_green_matrix(
            mesh, float(np.real(k_ref)),
            quad_order=self.sim.config.quad_order,
            near_threshold=self.sim.config.near_threshold,
            backend='auto',
            greens_fn=static_gf,
        )

        # --- D: divergence matrix (T×N, sparse) ---
        self.D = build_divergence_matrix(basis, mesh)

        if self.probe_feeds:
            self._setup_probe_feeds()
        else:
            D_dense = self.D.toarray()
            self.P = D_dense.T @ self.G_s @ D_dense
            self.V_all = np.column_stack([
                p.build_excitation_vector(self.sim.basis) for p in self.ports
            ])

    def _setup_probe_feeds(self):
        """Set up probe feeds using the attachment mode at junction vertices.

        Each probe is a single DOF representing uniform current from the
        PEC ground to the strip.  The probe's tip charge is distributed
        among the triangles adjacent to the junction vertex on the strip
        (attachment mode).  This avoids the point-charge self-potential
        singularity and ensures the probe charge interacts correctly with
        the strip through the existing G_s matrix.

        The divergence matrix D is extended with one column per probe:
            D_ext[t, N_s + p] = 1 / A_total  for triangles t at vertex v_p
        where A_total is the sum of attachment triangle areas.  This gives
        uniform charge density and satisfies charge conservation:
        sum_t D_ext[t, p] * A_t = 1  (unit charge per unit probe current).

        The probe base charge (-1) goes to the PEC ground (absorbed,
        zero potential), so it contributes nothing to D.
        """
        mesh = self.sim.mesh
        basis = self.sim.basis
        N_s = basis.num_basis
        T = len(mesh.triangles)
        N_probes = len(self.ports)

        # Layer geometry — find nearest PEC ground plane to strip
        stack = self.sim.config.layer_stack
        pec_z_candidates = []
        for layer in stack.layers:
            if layer.is_pec:
                if np.isfinite(layer.z_top):
                    pec_z_candidates.append(layer.z_top)
                if np.isfinite(layer.z_bot):
                    pec_z_candidates.append(layer.z_bot)
        if not pec_z_candidates:
            raise ValueError("probe_feeds requires a PEC ground layer")

        # Strip z-coordinate: use max z (strip level), not mean, so that
        # edge port meshes with vertical plates don't shift z_strip down.
        z_strip = float(np.max(mesh.vertices[:, 2]))
        # Pick nearest PEC plane as ground for probe feeds
        dists = [abs(z_strip - zp) for zp in pec_z_candidates]
        z_ground = pec_z_candidates[int(np.argmin(dists))]
        h_via = abs(z_strip - z_ground)
        self._h_via = h_via
        self._z_strip = z_strip

        # Via inductance: L = μ₀h/(2π) × [ln(2h/a) - 1]
        L_via = mu0 * h_via / (2.0 * np.pi) * (
            np.log(2.0 * h_via / self.probe_radius) - 1.0
        )

        # Find probe vertex positions from port feed edges.
        # The probe connects at the short end of the strip (x-edge),
        # centered in y.  We find the x-coordinate of the port's feed
        # edges, then pick the strip-level vertex at that x that is
        # closest to the strip's x-boundary (the short end).
        probe_vertices = []
        strip_mask = np.abs(mesh.vertices[:, 2] - z_strip) < 1e-8
        x_min_strip = float(mesh.vertices[strip_mask, 0].min())
        x_max_strip = float(mesh.vertices[strip_mask, 0].max())
        y_center = float(np.mean(mesh.vertices[strip_mask, 1]))

        for port in self.ports:
            feed_idx = port.feed_basis_indices
            # Determine the feed x-coordinate from the feed edge midpoints
            feed_x_vals = []
            for idx in feed_idx:
                ei = basis.edge_index[idx]
                v0, v1 = mesh.edges[ei]
                mid_x = 0.5 * (mesh.vertices[v0, 0] + mesh.vertices[v1, 0])
                feed_x_vals.append(mid_x)
            feed_x = float(np.mean(feed_x_vals))

            # Determine which strip end this port is near
            if feed_x < 0.5 * (x_min_strip + x_max_strip):
                target_x = x_min_strip  # left end
            else:
                target_x = x_max_strip  # right end

            # Pick the strip-level vertex at the strip end, closest to
            # y-center (middle of the short edge)
            candidates = np.where(strip_mask)[0]
            x_match = np.abs(mesh.vertices[candidates, 0] - target_x) < 1e-8
            if not np.any(x_match):
                # Fallback: closest to target_x
                dists = np.abs(mesh.vertices[candidates, 0] - target_x)
                x_match = dists < (dists.min() + 1e-8)
            end_verts = candidates[x_match]
            y_dists = np.abs(mesh.vertices[end_verts, 1] - y_center)
            vi = int(end_verts[np.argmin(y_dists)])
            probe_vertices.append(vi)

        # Build extended divergence matrix D_ext (T × (N_s + N_probes))
        D_dense = self.D.toarray()  # (T, N_s)
        D_ext = np.zeros((T, N_s + N_probes), dtype=np.float64)
        D_ext[:, :N_s] = D_dense

        for p, vi in enumerate(probe_vertices):
            # Find strip-level triangles at this vertex.
            # Only include triangles whose centroid is at z ≈ z_strip
            # (not plate triangles), so the probe charge stays on the
            # strip conductor.
            tri_at_vertex = []
            for t in range(T):
                if vi in mesh.triangles[t]:
                    z_centroid = mesh.vertices[mesh.triangles[t], 2].mean()
                    if abs(z_centroid - z_strip) < 0.5 * h_via:
                        tri_at_vertex.append(t)

            if not tri_at_vertex:
                # Fallback: use all triangles at vertex
                tri_at_vertex = [
                    t for t in range(T) if vi in mesh.triangles[t]
                ]

            if not tri_at_vertex:
                raise ValueError(
                    f"Port '{self.ports[p].name}': no triangles at vertex {vi}"
                )

            # Distribute the probe tip charge (+1 per unit I_probe) uniformly
            # over the attachment triangles.  Charge conservation requires
            # sum_t D_ext[t, probe] * A_t = 1, so D_ext = 1 / A_total.
            areas = np.array([mesh.triangle_areas[t] for t in tri_at_vertex])
            A_total = areas.sum()

            for t in tri_at_vertex:
                D_ext[t, N_s + p] = 1.0 / A_total

        # Build extended P matrix: P_ext = D_ext^T @ G_s @ D_ext
        self.P_hybrid = D_ext.T @ self.G_s @ D_ext  # (N_s+N_p, N_s+N_p)

        # Build extended L_raw matrix
        N_total = N_s + N_probes
        self.L_raw_hybrid = np.zeros((N_total, N_total), dtype=np.float64)
        self.L_raw_hybrid[:N_s, :N_s] = np.real(self.L_raw)
        # VP coupling wire-surface = 0 (orthogonal: vertical × horizontal)
        for p in range(N_probes):
            self.L_raw_hybrid[N_s + p, N_s + p] = L_via

        # Build excitation vectors: V_ref applied at each probe DOF
        self.V_all = np.zeros((N_total, N_probes), dtype=np.complex128)
        for p in range(N_probes):
            self.V_all[N_s + p, p] = self.ports[p].V_ref

        self._N_surface = N_s
        self._N_probes = N_probes
        self._probe_vertices = probe_vertices

    def extract(
        self,
        frequencies: Union[float, List[float]],
    ) -> List[NetworkResult]:
        """Extract network parameters at one or more frequencies.

        Parameters
        ----------
        frequencies : float or list of float
            Frequencies (Hz).

        Returns
        -------
        list of NetworkResult
        """
        if self.probe_feeds:
            return self._extract_probe(frequencies)
        else:
            return self._extract_standard(frequencies)

    def _extract_standard(
        self,
        frequencies: Union[float, List[float]],
    ) -> List[NetworkResult]:
        """Standard extraction using AEFIE (no probe feeds)."""
        from .aefie import solve_aefie

        if np.isscalar(frequencies):
            frequencies = [float(frequencies)]
        else:
            frequencies = list(frequencies)

        P_ports = len(self.ports)
        results = []

        for freq in frequencies:
            omega = 2.0 * np.pi * freq
            k = omega / c0
            eta = eta0 / self._eps_r

            Z_A = self._Z_A_ref * (k / self._k_ref)
            I_all = solve_aefie(Z_A, self.G_s, self.D, self.V_all, k, eta)

            Y_mat = np.zeros((P_ports, P_ports), dtype=np.complex128)
            for q, port_q in enumerate(self.ports):
                for p in range(P_ports):
                    I_term = port_q.terminal_current(
                        I_all[:, p], self.sim.basis
                    )
                    V_ref = self.ports[p].V_ref
                    Y_mat[q, p] = I_term / V_ref

            if P_ports == 1:
                if abs(Y_mat[0, 0]) > 1e-30:
                    Z_mat = np.array([[1.0 / Y_mat[0, 0]]])
                else:
                    Z_mat = np.array([[np.inf + 0j]])
            else:
                try:
                    Z_mat = np.linalg.inv(Y_mat)
                except np.linalg.LinAlgError:
                    Z_mat = np.full((P_ports, P_ports), np.inf + 0j)

            results.append(NetworkResult(
                frequency=freq,
                Z_matrix=Z_mat,
                port_names=[p.name for p in self.ports],
                Z0=self.Z0,
                condition_number=0.0,
                I_solutions=I_all.copy() if self.store_currents else None,
            ))

        return results

    def _extract_probe(
        self,
        frequencies: Union[float, List[float]],
    ) -> List[NetworkResult]:
        """Extraction with probe feeds — direct solve of hybrid system.

        Z(ω) = jωμ₀ · L_raw_hybrid + 1/(jωε₀ε_r) · P_hybrid

        The probe DOFs provide a grounded current source at each port.
        Terminal current = probe current; terminal voltage = V_ref.
        """
        if np.isscalar(frequencies):
            frequencies = [float(frequencies)]
        else:
            frequencies = list(frequencies)

        P_ports = self._N_probes
        N_total = self._N_surface + self._N_probes
        results = []

        for freq in frequencies:
            omega = 2.0 * np.pi * freq

            vp_pf = 1j * omega * mu0
            sp_pf = 1.0 / (1j * omega * eps0 * self._eps_r)

            Z_hybrid = vp_pf * self.L_raw_hybrid + sp_pf * self.P_hybrid

            try:
                I_all = np.linalg.solve(Z_hybrid, self.V_all)
            except np.linalg.LinAlgError:
                I_all = np.full_like(self.V_all, np.inf + 0j)

            # Y-matrix: Y[q, p] = I_probe_q / V_ref_p
            Y_mat = np.zeros((P_ports, P_ports), dtype=np.complex128)
            for q in range(P_ports):
                for p in range(P_ports):
                    I_probe_q = I_all[self._N_surface + q, p]
                    V_ref_p = self.ports[p].V_ref
                    Y_mat[q, p] = I_probe_q / V_ref_p

            if P_ports == 1:
                if abs(Y_mat[0, 0]) > 1e-30:
                    Z_mat = np.array([[1.0 / Y_mat[0, 0]]])
                else:
                    Z_mat = np.array([[np.inf + 0j]])
            else:
                try:
                    Z_mat = np.linalg.inv(Y_mat)
                except np.linalg.LinAlgError:
                    Z_mat = np.full((P_ports, P_ports), np.inf + 0j)

            I_surf = I_all[:self._N_surface, :] if self.store_currents else None

            results.append(NetworkResult(
                frequency=freq,
                Z_matrix=Z_mat,
                port_names=[p.name for p in self.ports],
                Z0=self.Z0,
                condition_number=float(np.linalg.cond(Z_hybrid)),
                I_solutions=I_surf,
            ))

        return results
