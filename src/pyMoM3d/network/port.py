"""Port abstraction for lumped-port network extraction.

A Port maps a set of RWG basis functions to a circuit terminal.  Port
definitions depend only on RWG basis indexing and are independent of the
underlying Green's function, ensuring compatibility with future multilayer
and advanced kernel (PNGF, surrogate) implementations.

Physical interpretation
-----------------------
In the delta-gap model, V_ref is applied as a **uniform voltage distribution
across the gap**: each basis function m in the port receives

    V[m] = sign_m * V_ref * l_m

where l_m is the edge length and sign_m ∈ {+1, −1} corrects for RWG
orientation (see ``feed_signs``).  This enforces a uniform tangential
E-field across the gap and is consistent with the existing DeltaGapExcitation
and StripDeltaGapExcitation conventions.

The terminal current is

    I_term = Σ_{m ∈ port} sign_m * I[m] * l_m

and the port impedance is Z = V_ref / I_term.

All extracted Z/Y/S parameters are conditional on the chosen port geometry,
reference plane, and return path.

Return path
-----------
The surface integral equation (EFIE/CFIE) distributes return current over
the whole meshed PEC surface.  For single-conductor structures (dipole,
patch, sphere) this is physically correct.  For multi-conductor structures
(microstrip, coupled lines) **both conductors must be explicitly meshed**.
If the ground plane is absent, return current has no path and the simulation
produces non-physical results (non-passive S-matrix, lack of reciprocity).

For structures with a well-defined local return conductor, use a differential
port (``return_basis_indices``), which applies ±V_ref antisymmetrically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Port:
    """Lumped port defined by a set of RWG basis functions.

    Parameters
    ----------
    name : str
        Port label (used in NetworkResult.port_names).
    feed_basis_indices : list of int
        RWG basis indices on the signal side (V_ref applied here).
    return_basis_indices : list of int, optional
        RWG basis indices on the return/ground side (−V_ref applied here).
        Empty list (default) → single-ended port; return current is
        distributed over the rest of the mesh.
    feed_signs : list of int, optional
        Per-edge sign correction (+1 or −1) for ``feed_basis_indices``.
        Must have the same length as ``feed_basis_indices`` if non-empty.
        Use this when ``find_feed_edges`` returns edges whose RWG T+/T−
        winding is opposite to the intended port current direction.
        Default: all +1.
    V_ref : complex, optional
        Reference excitation voltage (V).  Default 1.0 V.
    gap_width : float, optional
        Physical width of the feed gap (m).  Default 0.0 selects the
        standard delta-gap model.  When > 0, ``build_excitation_vector``
        uses the finite-width distributed excitation of Lo, Jiang & Chew
        (2013): a uniform E-field pulse of amplitude ``V_ref / gap_width``
        over the port region, producing a bounded (non-singular) excitation
        and mesh-independent port impedance.  Requires passing ``mesh``
        to ``build_excitation_vector``.
    """

    name: str
    feed_basis_indices: List[int]
    return_basis_indices: List[int] = field(default_factory=list)
    feed_signs: List[int] = field(default_factory=list)
    V_ref: complex = 1.0
    gap_width: float = 0.0

    def __post_init__(self):
        if self.feed_signs and len(self.feed_signs) != len(self.feed_basis_indices):
            raise ValueError(
                f"Port '{self.name}': feed_signs length ({len(self.feed_signs)}) "
                f"must match feed_basis_indices length ({len(self.feed_basis_indices)})"
            )

    @property
    def is_differential(self) -> bool:
        """True if this port has an explicit return-conductor definition."""
        return len(self.return_basis_indices) > 0

    # ------------------------------------------------------------------
    # Factory helpers (thin wrappers around existing find_feed_edges*)
    # ------------------------------------------------------------------

    @staticmethod
    def from_x_plane(
        mesh,
        rwg_basis,
        x_coord: float,
        name: str = 'P1',
        tol: float = None,
    ) -> 'Port':
        """Create a single-ended port at a transverse x-plane.

        Wraps :func:`~pyMoM3d.mom.excitation.find_feed_edges`.

        Parameters
        ----------
        mesh : Mesh
        rwg_basis : RWGBasis
        x_coord : float
            x-coordinate of the feed plane.
        name : str
        tol : float, optional
            Tolerance for edge midpoint proximity.  Defaults to half the
            minimum edge length (same as find_feed_edges default).

        Returns
        -------
        Port
        """
        from ..mom.excitation import find_feed_edges
        indices = find_feed_edges(mesh, rwg_basis, feed_x=x_coord, tol=tol)
        if not indices:
            raise ValueError(
                f"Port.from_x_plane: no feed edges found near x={x_coord}"
            )
        return Port(name=name, feed_basis_indices=indices)

    @staticmethod
    def from_center(
        mesh,
        rwg_basis,
        center: np.ndarray,
        axis: np.ndarray,
        name: str = 'P1',
        tol: float = None,
        perp_tol: float = None,
    ) -> 'Port':
        """Create a single-ended port near a 3-D point, for any dipole axis.

        Wraps :func:`~pyMoM3d.mom.excitation.find_feed_edges_near_center`.

        Parameters
        ----------
        mesh : Mesh
        rwg_basis : RWGBasis
        center : array-like, shape (3,)
            3-D centre of the feed region.
        axis : array-like, shape (3,)
            Dipole / current axis direction.
        name : str
        tol, perp_tol : float, optional

        Returns
        -------
        Port
        """
        from ..mom.excitation import find_feed_edges_near_center
        indices = find_feed_edges_near_center(
            mesh, rwg_basis,
            np.asarray(center, dtype=np.float64),
            np.asarray(axis, dtype=np.float64),
            tol=tol, perp_tol=perp_tol,
        )
        if not indices:
            raise ValueError(
                f"Port.from_center: no feed edges found near center={center}"
            )
        return Port(name=name, feed_basis_indices=indices)

    @staticmethod
    def from_vertex(
        mesh,
        rwg_basis,
        vertex_pos: np.ndarray,
        name: str = 'probe',
        tol: float = None,
    ) -> 'Port':
        """Create a probe feed port at a surface mesh vertex.

        Models a coaxial probe by finding the mesh vertex closest to
        ``vertex_pos`` and exciting all RWG basis functions sharing that
        vertex.  Signs are set so that current flows radially outward
        from the probe point, consistent with a vertical current source
        injecting current at this location.

        The PEC ground plane (if present) is handled by the layered
        Green's function — no explicit wire is needed.

        Parameters
        ----------
        mesh : Mesh
        rwg_basis : RWGBasis
        vertex_pos : array-like, shape (2,) or (3,)
            Position of the probe.  If 2D (x, y), the z-coordinate is
            taken from the mesh.  If 3D, the nearest vertex is found.
        name : str
        tol : float, optional
            Maximum distance to the nearest vertex.  Defaults to the
            mean edge length.

        Returns
        -------
        Port
        """
        vertex_pos = np.asarray(vertex_pos, dtype=np.float64).ravel()
        verts = mesh.vertices

        if len(vertex_pos) == 2:
            # 2D: find nearest vertex in x-y plane
            dists = np.sqrt(
                (verts[:, 0] - vertex_pos[0])**2
                + (verts[:, 1] - vertex_pos[1])**2
            )
        else:
            dists = np.linalg.norm(verts - vertex_pos[np.newaxis, :], axis=1)

        vi = int(np.argmin(dists))

        if tol is None:
            tol = float(np.mean(rwg_basis.edge_length))
        if dists[vi] > tol:
            raise ValueError(
                f"Port.from_vertex: nearest vertex is {dists[vi]*1e3:.3f} mm "
                f"away (tol={tol*1e3:.3f} mm)"
            )

        # Find all basis functions connected to this vertex
        indices = []
        signs = []
        for n in range(rwg_basis.num_basis):
            if rwg_basis.free_vertex_plus[n] == vi:
                # Current flows AWAY from vertex on T+ → sign +1
                indices.append(n)
                signs.append(+1)
            elif rwg_basis.free_vertex_minus[n] == vi:
                # Current flows TOWARD vertex on T- → sign -1 for outward
                indices.append(n)
                signs.append(-1)

        if not indices:
            raise ValueError(
                f"Port.from_vertex: no RWG edges found at vertex {vi} "
                f"({verts[vi]})"
            )

        return Port(
            name=name,
            feed_basis_indices=indices,
            feed_signs=signs,
        )

    @staticmethod
    def from_nonradiating_gap(
        mesh,
        rwg_basis,
        port_x: float,
        name: str = 'P1',
        tol: float = None,
    ) -> 'Port':
        """Create a non-radiating port from half-RWG basis functions at a gap.

        For meshes prepared with :func:`~pyMoM3d.mesh.port_mesh.split_mesh_at_x`
        and extended with :func:`~pyMoM3d.mesh.port_mesh.add_half_rwg_basis`,
        selects the half-RWG basis functions at the specified port gap and
        computes signs for +x current flow.

        This port model eliminates the radiating current artifact of the
        standard delta-gap model (Liu et al. 2018).

        Parameters
        ----------
        mesh : Mesh
        rwg_basis : RWGBasis
            Extended RWG basis containing half-RWG entries (from
            ``add_half_rwg_basis``).
        port_x : float
            x-coordinate of the port gap.
        name : str
        tol : float, optional
            Tolerance for identifying port edges.

        Returns
        -------
        Port
        """
        if tol is None:
            tol = 0.01 * float(np.mean(rwg_basis.edge_length))

        # Find half-RWG basis functions at this port location.
        # These are identified by having their T+ and T- centroids on
        # opposite sides of port_x, with the "shared edge" at port_x.
        indices = []
        signs = []

        for n in range(rwg_basis.num_basis):
            edge = mesh.edges[rwg_basis.edge_index[n]]
            va = mesh.vertices[edge[0]]
            vb = mesh.vertices[edge[1]]
            mid_x = 0.5 * (va[0] + vb[0])

            if abs(mid_x - port_x) > tol:
                continue

            # Check that T+ and T- are on opposite sides of the gap
            tp_centroid = mesh.vertices[
                mesh.triangles[rwg_basis.t_plus[n]]
            ].mean(axis=0)
            tm_centroid = mesh.vertices[
                mesh.triangles[rwg_basis.t_minus[n]]
            ].mean(axis=0)

            if (tp_centroid[0] - port_x) * (tm_centroid[0] - port_x) < 0:
                # T+ and T- are on opposite sides → this is a half-RWG
                indices.append(n)

                # Sign: +1 if current flows in +x direction
                fvp = mesh.vertices[rwg_basis.free_vertex_plus[n]]
                if fvp[0] < port_x:
                    signs.append(+1)  # T+ is on left, current flows right
                else:
                    signs.append(-1)  # T+ is on right, current flows left

        if not indices:
            raise ValueError(
                f"Port.from_nonradiating_gap: no half-RWG basis functions "
                f"found at x = {port_x}.  Ensure the mesh was split and "
                f"add_half_rwg_basis() was called for this location."
            )

        return Port(
            name=name,
            feed_basis_indices=indices,
            feed_signs=signs,
        )

    @staticmethod
    def from_edge_port(
        mesh,
        rwg_basis,
        port_x: float,
        strip_z: float,
        name: str = 'P1',
        direction: np.ndarray = None,
        tol: float = None,
    ) -> 'Port':
        """Create a port at an edge-fed vertical plate junction.

        For microstrip geometries meshed with
        :meth:`~pyMoM3d.mesh.GmshMesher.mesh_microstrip_with_edge_ports`,
        selects RWG basis functions at the strip-plate junction (where the
        horizontal strip meets the vertical plate) and computes signs for
        current flowing from the strip downward through the plate toward
        the ground plane.

        Parameters
        ----------
        mesh : Mesh
        rwg_basis : RWGBasis
        port_x : float
            x-coordinate of the port edge (strip end / plate location).
        strip_z : float
            z-coordinate of the strip surface (top of the vertical plate).
        name : str
        direction : ndarray, shape (3,), optional
            Reference current direction at the junction.  Default
            ``[0, 0, -1]`` (current flows from strip downward into plate).
        tol : float, optional
            Positional tolerance for edge detection.

        Returns
        -------
        Port
        """
        from ..mom.excitation import (
            find_edge_port_feed_edges,
            compute_feed_signs_along_direction,
        )
        if direction is None:
            direction = np.array([0.0, 0.0, -1.0])
        indices = find_edge_port_feed_edges(
            mesh, rwg_basis, port_x=port_x, strip_z=strip_z, tol=tol,
        )
        if not indices:
            raise ValueError(
                f"Port.from_edge_port: no junction edges found at "
                f"x={port_x}, z={strip_z}"
            )
        signs = compute_feed_signs_along_direction(
            mesh, rwg_basis, indices, direction=direction,
        )
        return Port(
            name=name,
            feed_basis_indices=indices,
            feed_signs=signs,
        )

    @staticmethod
    def differential(
        mesh,
        rwg_basis,
        signal_x: float,
        return_x: float,
        name: str = 'P1',
        tol: float = None,
    ) -> 'Port':
        """Create a differential (two-conductor) port.

        Applies +V_ref to edges at ``signal_x`` and −V_ref to edges at
        ``return_x``.  Use for microstrip, coplanar-waveguide, or any
        geometry with an explicit local return conductor.

        Parameters
        ----------
        mesh : Mesh
        rwg_basis : RWGBasis
        signal_x : float
            x-coordinate of the signal feed plane.
        return_x : float
            x-coordinate of the return conductor plane.
        name : str
        tol : float, optional

        Returns
        -------
        Port
        """
        from ..mom.excitation import find_feed_edges
        sig = find_feed_edges(mesh, rwg_basis, feed_x=signal_x, tol=tol)
        ret = find_feed_edges(mesh, rwg_basis, feed_x=return_x, tol=tol)
        if not sig:
            raise ValueError(
                f"Port.differential: no signal edges found near x={signal_x}"
            )
        if not ret:
            raise ValueError(
                f"Port.differential: no return edges found near x={return_x}"
            )
        return Port(name=name, feed_basis_indices=sig, return_basis_indices=ret)

    # ------------------------------------------------------------------
    # Excitation and measurement
    # ------------------------------------------------------------------

    def build_excitation_vector(self, rwg_basis, mesh=None) -> np.ndarray:
        """Build the RHS voltage vector for this port excited alone.

        When ``gap_width == 0`` (default, delta-gap model):

            V[m] = sign_m * V_ref * l_m

        When ``gap_width > 0`` (finite-width model, Lo/Jiang/Chew 2013):

            V[m] = sign_m * (V_ref / d) * overlap_m

        where ``d`` is the gap width and ``overlap_m`` is the Galerkin
        inner product of the uniform pulse field ``E_inc = (V/d) * n_hat``
        with basis function m over the port region.  This requires the
        ``mesh`` argument.

        For each return edge m (differential port):
            V[m] = −V_ref * l_m  (always delta-gap style)

        Parameters
        ----------
        rwg_basis : RWGBasis
        mesh : Mesh, optional
            Required when ``gap_width > 0``.

        Returns
        -------
        V : ndarray, shape (N,), complex128
        """
        N = rwg_basis.num_basis
        V = np.zeros(N, dtype=np.complex128)
        signs = self.feed_signs if self.feed_signs else [+1] * len(self.feed_basis_indices)

        if self.gap_width > 0 and mesh is not None:
            V = self._build_finite_width_rhs(rwg_basis, mesh)
        else:
            for s, idx in zip(signs, self.feed_basis_indices):
                if 0 <= idx < N:
                    V[idx] = s * self.V_ref * rwg_basis.edge_length[idx]

        for idx in self.return_basis_indices:
            if 0 <= idx < N:
                V[idx] = -self.V_ref * rwg_basis.edge_length[idx]
        return V

    def _build_finite_width_rhs(self, rwg_basis, mesh) -> np.ndarray:
        """Build RHS using finite-width distributed excitation.

        Applies a uniform incident field E_inc = (V_ref / d) * x_hat
        over the port region [port_x - d/2, port_x + d/2], where port_x
        is determined from the feed edge locations.

        The Galerkin inner product <f_m, E_inc> is computed by integrating
        the x-component of each RWG basis function over the intersection
        of its support with the port region.

        For basis functions at the port edges (standard feed edges), this
        gives approximately V[m] = sign_m * V_ref * l_m (same as delta-gap
        when d equals one mesh cell).

        For adjacent basis functions whose triangles partially overlap the
        port region, this gives a fractional contribution that smooths the
        excitation.
        """
        N = rwg_basis.num_basis
        V = np.zeros(N, dtype=np.complex128)
        d = self.gap_width

        # Determine port center from feed edge midpoints
        feed_x_coords = []
        for idx in self.feed_basis_indices:
            e = mesh.edges[rwg_basis.edge_index[idx]]
            mid_x = 0.5 * (mesh.vertices[e[0]][0] + mesh.vertices[e[1]][0])
            feed_x_coords.append(mid_x)
        port_x = np.mean(feed_x_coords)

        x_lo = port_x - d / 2.0
        x_hi = port_x + d / 2.0

        signs = (
            self.feed_signs if self.feed_signs
            else [+1] * len(self.feed_basis_indices)
        )
        sign_map = dict(zip(self.feed_basis_indices, signs))

        # Compute overlap for ALL basis functions (not just feed edges)
        for n in range(N):
            overlap = self._basis_overlap_x(
                n, rwg_basis, mesh, x_lo, x_hi,
            )
            if abs(overlap) < 1e-30:
                continue
            # Determine sign: use port sign if this is a feed edge,
            # otherwise infer from RWG orientation
            if n in sign_map:
                s = sign_map[n]
            else:
                s = self._infer_sign(n, rwg_basis, mesh, port_x)
            V[n] = s * (self.V_ref / d) * overlap
        return V

    @staticmethod
    def _basis_overlap_x(
        n: int, rwg_basis, mesh, x_lo: float, x_hi: float,
    ) -> float:
        """Compute the overlap integral of RWG basis function n with a
        uniform x-directed field over the region [x_lo, x_hi].

        Returns the approximate integral:
            integral_{support(f_n) ∩ [x_lo,x_hi]} |f_n · x_hat| dS

        For an RWG basis function, the x-component contribution from each
        triangle is approximated by evaluating the basis function at the
        triangle centroid (1-point quadrature) times the clipped triangle
        area.
        """
        verts = mesh.vertices
        tris = mesh.triangles

        total = 0.0
        for tri_idx, area, fv_idx, rho_sign in [
            (rwg_basis.t_plus[n], rwg_basis.area_plus[n],
             rwg_basis.free_vertex_plus[n], +1.0),
            (rwg_basis.t_minus[n], rwg_basis.area_minus[n],
             rwg_basis.free_vertex_minus[n], -1.0),
        ]:
            tri_verts = verts[tris[tri_idx]]
            centroid = tri_verts.mean(axis=0)

            # Clip: fraction of triangle overlapping [x_lo, x_hi]
            tri_x = tri_verts[:, 0]
            tri_x_min = tri_x.min()
            tri_x_max = tri_x.max()
            if tri_x_max <= x_lo or tri_x_min >= x_hi:
                continue  # No overlap
            # Linear approximation of clipped area fraction
            tri_span = tri_x_max - tri_x_min
            if tri_span < 1e-15:
                frac = 1.0
            else:
                clipped_lo = max(tri_x_min, x_lo)
                clipped_hi = min(tri_x_max, x_hi)
                frac = (clipped_hi - clipped_lo) / tri_span
            frac = min(frac, 1.0)

            # RWG basis value at centroid: (l_n / 2A) * rho
            l_n = rwg_basis.edge_length[n]
            fv = verts[fv_idx]
            if rho_sign > 0:
                rho = centroid - fv        # rho+ = r - r_free+
            else:
                rho = fv - centroid        # rho- = r_free- - r

            # x-component of f_n at centroid
            f_x = (l_n / (2.0 * area)) * rho[0]

            # Contribution: |f_x| * clipped_area (unsigned — sign handled by caller)
            total += abs(f_x) * area * frac

        return total

    @staticmethod
    def _infer_sign(n: int, rwg_basis, mesh, port_x: float) -> int:
        """Infer the excitation sign for a non-feed basis function.

        Uses the same convention as compute_feed_signs: +1 if the RWG
        current flows in the +x direction through the port region.
        """
        e = mesh.edges[rwg_basis.edge_index[n]]
        mid_x = 0.5 * (mesh.vertices[e[0]][0] + mesh.vertices[e[1]][0])
        fvp_x = mesh.vertices[rwg_basis.free_vertex_plus[n]][0]
        # On T+, current flows from free_vertex_plus toward the shared edge.
        # If mid_x > fvp_x, current flows in +x direction.
        return +1 if mid_x > fvp_x else -1

    def terminal_current(self, I_coeffs: np.ndarray, rwg_basis) -> complex:
        """Compute the terminal current flowing into this port.

            I_term = Σ_{m ∈ signal} sign_m * I[m] * l_m

        Only the signal-side edges contribute; return-side edges are not
        included (they carry the same current with opposite sign).

        Parameters
        ----------
        I_coeffs : ndarray, shape (N,), complex128
            RWG current coefficients.
        rwg_basis : RWGBasis

        Returns
        -------
        I_term : complex
        """
        signs = self.feed_signs if self.feed_signs else [+1] * len(self.feed_basis_indices)
        return sum(
            s * I_coeffs[m] * rwg_basis.edge_length[m]
            for s, m in zip(signs, self.feed_basis_indices)
        )


@dataclass
class GroundVia:
    """Lumped via connecting conductor basis functions to the GF ground plane.

    Models a vertical connection from the conductor mesh (at z = z_mesh)
    down to the PEC ground plane (implicit in the layered Green's function).
    Implemented by adding a small impedance Z_via to the Z_sys diagonal at
    the via location, forcing near-zero potential there.

    This is the MoM equivalent of a "via-to-ground port" in EMX / Palace.
    No vertical mesh is needed — the ground plane is already in the GF.

    Parameters
    ----------
    name : str
        Via label (for diagnostics).
    basis_indices : list of int
        RWG basis indices at the via location.
    impedance : complex, optional
        Total via impedance (Ohm).  Default 0 (ideal short to ground).
        For a physical via, can include R_via + jωL_via.
    """

    name: str
    basis_indices: List[int]
    impedance: complex = 0.0

    def apply_to_matrix(self, Z_sys: np.ndarray, rwg_basis) -> None:
        """Add via ground-return path to the impedance matrix.

        For an ideal via (impedance ≈ 0), adds a very small impedance
        to the diagonal, effectively shorting those basis functions to
        ground.  For a finite impedance, distributes it across the
        parallel via edges.
        """
        n_via = len(self.basis_indices)
        if n_via == 0:
            return
        for m in self.basis_indices:
            if abs(self.impedance) < 1e-30:
                # Ideal short: add small impedance (large admittance)
                Z_sys[m, m] += 1e-8
            else:
                Z_sys[m, m] += self.impedance / n_via
