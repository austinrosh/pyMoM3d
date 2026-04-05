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
    """

    name: str
    feed_basis_indices: List[int]
    return_basis_indices: List[int] = field(default_factory=list)
    feed_signs: List[int] = field(default_factory=list)
    V_ref: complex = 1.0

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

    def build_excitation_vector(self, rwg_basis) -> np.ndarray:
        """Build the RHS voltage vector for this port excited alone.

        For each signal edge m:
            V[m] = sign_m * V_ref * l_m

        For each return edge m (differential port):
            V[m] = −V_ref * l_m

        This enforces a uniform tangential E-field across the gap, consistent
        with the DeltaGapExcitation / StripDeltaGapExcitation convention.

        Parameters
        ----------
        rwg_basis : RWGBasis

        Returns
        -------
        V : ndarray, shape (N,), complex128
        """
        N = rwg_basis.num_basis
        V = np.zeros(N, dtype=np.complex128)
        signs = self.feed_signs if self.feed_signs else [+1] * len(self.feed_basis_indices)
        for s, idx in zip(signs, self.feed_basis_indices):
            if 0 <= idx < N:
                V[idx] = s * self.V_ref * rwg_basis.edge_length[idx]
        for idx in self.return_basis_indices:
            if 0 <= idx < N:
                V[idx] = -self.V_ref * rwg_basis.edge_length[idx]
        return V

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
