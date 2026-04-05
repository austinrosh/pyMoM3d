"""Wire mesh and rooftop basis functions for thin-wire MoM.

Provides data structures and connectivity computation for 1D piecewise-linear
(rooftop) basis functions on wire segments.  Each basis function spans two
adjacent segments, peaked at the shared node — analogous to RWG basis
functions on triangular meshes but in 1D.

Conventions
-----------
- The rooftop basis function f_n is centred at interior node n.
- On seg_plus (the segment "below" the peak in the node ordering):
    f_n(s) = (s - s_start) / length_plus  ·  d̂_plus
  where s is the arc-length parameter along the segment, and d̂_plus is the
  unit direction vector of the segment.
- On seg_minus (the segment "above" the peak):
    f_n(s) = (s_end - s) / length_minus  ·  d̂_minus
- Divergence: div(f_n) = +1/length_plus on seg_plus,
                         -1/length_minus on seg_minus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class WireSegment:
    """A single wire segment between two nodes.

    Parameters
    ----------
    node_start : int
        Index into WireMesh.nodes for the start node.
    node_end : int
        Index into WireMesh.nodes for the end node.
    length : float
        Segment length (m).
    direction : ndarray, shape (3,)
        Unit direction vector from start to end.
    radius : float
        Wire radius (m) for thin-wire approximation.
    """

    node_start: int
    node_end: int
    length: float
    direction: np.ndarray
    radius: float


@dataclass
class WireMesh:
    """Wire geometry: nodes and segments.

    Parameters
    ----------
    nodes : ndarray, shape (N_nodes, 3)
        Node coordinates.
    segments : list of WireSegment
        Ordered list of wire segments.
    """

    nodes: np.ndarray
    segments: List[WireSegment]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @staticmethod
    def vertical_probe(
        x: float,
        y: float,
        z_bot: float,
        z_top: float,
        radius: float,
        n_segments: int = 3,
    ) -> 'WireMesh':
        """Create a vertical probe wire with uniform segmentation.

        Parameters
        ----------
        x, y : float
            Horizontal position of the probe (m).
        z_bot : float
            Bottom z-coordinate (ground plane).
        z_top : float
            Top z-coordinate (strip surface).
        radius : float
            Wire radius (m).
        n_segments : int
            Number of segments along the wire.

        Returns
        -------
        WireMesh
        """
        if n_segments < 1:
            raise ValueError("n_segments must be >= 1")
        if z_top <= z_bot:
            raise ValueError("z_top must be > z_bot")

        n_nodes = n_segments + 1
        z_vals = np.linspace(z_bot, z_top, n_nodes)
        nodes = np.zeros((n_nodes, 3), dtype=np.float64)
        nodes[:, 0] = x
        nodes[:, 1] = y
        nodes[:, 2] = z_vals

        dl = (z_top - z_bot) / n_segments
        z_hat = np.array([0.0, 0.0, 1.0])

        segments = []
        for i in range(n_segments):
            segments.append(WireSegment(
                node_start=i,
                node_end=i + 1,
                length=dl,
                direction=z_hat.copy(),
                radius=radius,
            ))

        return WireMesh(nodes=nodes, segments=segments)

    @staticmethod
    def from_nodes(
        nodes: np.ndarray,
        radius: float,
    ) -> 'WireMesh':
        """Create a WireMesh from an ordered sequence of node coordinates.

        Parameters
        ----------
        nodes : ndarray, shape (N, 3)
            Ordered node coordinates.  Segments connect consecutive nodes.
        radius : float
            Wire radius (m).

        Returns
        -------
        WireMesh
        """
        nodes = np.asarray(nodes, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (N, 3)")
        if len(nodes) < 2:
            raise ValueError("Need at least 2 nodes")

        segments = []
        for i in range(len(nodes) - 1):
            diff = nodes[i + 1] - nodes[i]
            length = float(np.linalg.norm(diff))
            if length < 1e-30:
                raise ValueError(f"Zero-length segment between nodes {i} and {i+1}")
            direction = diff / length
            segments.append(WireSegment(
                node_start=i,
                node_end=i + 1,
                length=length,
                direction=direction,
                radius=radius,
            ))

        return WireMesh(nodes=nodes.copy(), segments=segments)


@dataclass
class WireBasis:
    """Rooftop (piecewise-linear) basis functions on wire segments.

    Each basis function spans two adjacent segments, peaked at the shared
    interior node.  Analogous to RWG basis functions but in 1D.

    For N_seg segments there are N_seg - 1 interior nodes, hence
    N_seg - 1 basis functions.

    Parameters
    ----------
    num_basis : int
        Number of basis functions.
    wire_mesh : WireMesh
        Associated wire mesh.
    seg_plus : ndarray, shape (N,)
        Segment index for the "plus" (rising) half of each basis function.
    seg_minus : ndarray, shape (N,)
        Segment index for the "minus" (falling) half.
    node_index : ndarray, shape (N,)
        Interior node index (peak of rooftop) for each basis function.
    length_plus : ndarray, shape (N,)
        Length of seg_plus for each basis function.
    length_minus : ndarray, shape (N,)
        Length of seg_minus for each basis function.
    """

    num_basis: int
    wire_mesh: WireMesh
    seg_plus: np.ndarray
    seg_minus: np.ndarray
    node_index: np.ndarray
    length_plus: np.ndarray
    length_minus: np.ndarray


def compute_wire_connectivity(wire_mesh: WireMesh) -> WireBasis:
    """Build WireBasis from WireMesh.

    Analogous to ``compute_rwg_connectivity`` for surface meshes.
    Each interior node (shared by two adjacent connected segments)
    defines one rooftop basis function.

    Supports disjoint wire chains (e.g. multiple probe wires in a
    single WireMesh).  Only consecutive segments that share a node
    produce basis functions; gaps between disjoint chains are skipped.

    Parameters
    ----------
    wire_mesh : WireMesh

    Returns
    -------
    WireBasis
    """
    n_seg = wire_mesh.num_segments
    if n_seg < 2:
        raise ValueError("Need at least 2 segments for wire basis functions")

    seg_plus_list = []
    seg_minus_list = []
    node_index_list = []
    length_plus_list = []
    length_minus_list = []

    for i in range(n_seg - 1):
        seg_p = wire_mesh.segments[i]
        seg_m = wire_mesh.segments[i + 1]

        # Only create a basis function if segments share a node
        if seg_p.node_end == seg_m.node_start:
            seg_plus_list.append(i)
            seg_minus_list.append(i + 1)
            node_index_list.append(seg_p.node_end)
            length_plus_list.append(seg_p.length)
            length_minus_list.append(seg_m.length)

    n_basis = len(seg_plus_list)
    if n_basis == 0:
        raise ValueError(
            "No connected segment pairs found — need at least 2 "
            "connected segments for a wire basis function"
        )

    return WireBasis(
        num_basis=n_basis,
        wire_mesh=wire_mesh,
        seg_plus=np.array(seg_plus_list, dtype=np.int32),
        seg_minus=np.array(seg_minus_list, dtype=np.int32),
        node_index=np.array(node_index_list, dtype=np.int32),
        length_plus=np.array(length_plus_list, dtype=np.float64),
        length_minus=np.array(length_minus_list, dtype=np.float64),
    )
