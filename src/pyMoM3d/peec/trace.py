"""PEEC trace geometry abstractions.

Defines conductor traces as ordered sequences of straight filamentary
segments with constrained current flow direction.  This is the critical
difference from surface-MoM (RWG): current is physically constrained
to flow along the segment direction, eliminating the 2D surface current
freedom that makes inductance extraction impossible on surface meshes.

Each segment carries width, thickness, and material properties for
computing partial inductance, capacitance, and conductor loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from ..mom.surface_impedance import ConductorProperties


@dataclass
class TraceSegment:
    """A single straight conductor segment for PEEC.

    Parameters
    ----------
    start : ndarray, shape (3,)
        Start point of segment centerline (m).
    end : ndarray, shape (3,)
        End point of segment centerline (m).
    width : float
        Trace width (m).
    thickness : float
        Conductor thickness (m).
    conductor : ConductorProperties
        Material properties.
    """

    start: np.ndarray
    end: np.ndarray
    width: float
    thickness: float
    conductor: ConductorProperties

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=np.float64)
        self.end = np.asarray(self.end, dtype=np.float64)

    @property
    def length(self) -> float:
        """Segment length (m)."""
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        """Unit vector along segment, shape (3,)."""
        d = self.end - self.start
        return d / np.linalg.norm(d)

    @property
    def midpoint(self) -> np.ndarray:
        """Segment midpoint, shape (3,)."""
        return 0.5 * (self.start + self.end)

    @property
    def cross_section_area(self) -> float:
        """Cross-section area w * t (m^2)."""
        return self.width * self.thickness


@dataclass
class Trace:
    """An ordered sequence of connected segments forming a conductor path.

    Segments are connected end-to-end: segment[i].end == segment[i+1].start.
    Current continuity is enforced at junctions via KCL in the circuit model.

    Parameters
    ----------
    name : str
        Label for this trace.
    segments : list of TraceSegment
        Ordered list of segments.
    """

    name: str
    segments: List[TraceSegment] = field(default_factory=list)

    @property
    def total_length(self) -> float:
        """Total trace length (m)."""
        return sum(s.length for s in self.segments)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @staticmethod
    def from_centerline(
        points: np.ndarray,
        width: float,
        thickness: float,
        conductor: ConductorProperties,
        segments_per_section: Optional[int] = None,
        min_segment_length: Optional[float] = None,
        name: str = 'trace',
    ) -> 'Trace':
        """Build a trace from centerline waypoints.

        Each pair of consecutive waypoints defines a straight section.
        Each section is subdivided into approximately equal-length segments.

        Parameters
        ----------
        points : ndarray, shape (N, 3)
            Ordered centerline waypoints (m).
        width : float
            Trace width (m).
        thickness : float
            Conductor thickness (m).
        conductor : ConductorProperties
            Material properties.
        segments_per_section : int, optional
            Fixed number of segments per straight section.
            If None, adaptive subdivision is used (segment length ~ width).
        min_segment_length : float, optional
            Minimum segment length for adaptive subdivision.
            Default: width / 2.
        name : str
            Trace label.

        Returns
        -------
        Trace
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {points.shape}")
        if len(points) < 2:
            raise ValueError("Need at least 2 waypoints")

        if min_segment_length is None:
            min_segment_length = width / 2.0

        segments = []
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            section_length = float(np.linalg.norm(p1 - p0))

            if section_length < 1e-15:
                continue

            if segments_per_section is not None:
                n_sub = max(1, segments_per_section)
            else:
                # Adaptive: segment length ~ width, but at least 1
                n_sub = max(1, int(np.round(section_length / width)))
                # Ensure segments aren't too short
                while n_sub > 1 and section_length / n_sub < min_segment_length:
                    n_sub -= 1

            direction = (p1 - p0) / section_length
            sub_length = section_length / n_sub

            for j in range(n_sub):
                s0 = p0 + direction * (j * sub_length)
                s1 = p0 + direction * ((j + 1) * sub_length)
                segments.append(TraceSegment(
                    start=s0, end=s1,
                    width=width, thickness=thickness,
                    conductor=conductor,
                ))

        return Trace(name=name, segments=segments)

    @staticmethod
    def rectangular_spiral(
        n_turns: float,
        d_out: float,
        w_trace: float,
        s_space: float,
        thickness: float,
        conductor: ConductorProperties,
        z: float = 0.0,
        segments_per_section: Optional[int] = None,
        name: str = 'spiral',
    ) -> 'Trace':
        """Build a square planar spiral trace.

        Mirrors the geometry from aefie_spiral_validation.py.
        The spiral starts at the outer edge and winds inward.

        Parameters
        ----------
        n_turns : float
            Number of turns (e.g., 2.5).
        d_out : float
            Outer dimension (m).
        w_trace : float
            Trace width (m).
        s_space : float
            Spacing between adjacent turns (m).
        thickness : float
            Conductor thickness (m).
        conductor : ConductorProperties
            Material properties.
        z : float
            Z-coordinate of the spiral plane (m).
        segments_per_section : int, optional
            Fixed segments per straight section.  If None, adaptive.
        name : str
            Trace label.

        Returns
        -------
        Trace
        """
        pitch = w_trace + s_space
        n_full = int(n_turns)
        has_half = (n_turns - n_full) >= 0.4

        # Generate centerline waypoints
        # Spiral starts at bottom-left outer corner, goes right
        # Starting point: center the spiral at origin
        x = -d_out / 2.0
        y = -d_out / 2.0
        waypoints = [(x, y, z)]

        # Direction order: right, up, left, down
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        for turn in range(n_full):
            inset = turn * pitch
            for d_idx, (dx, dy) in enumerate(directions):
                side = d_out - 2 * inset
                if d_idx >= 2:
                    side -= pitch
                if side <= 0:
                    break
                x += dx * side
                y += dy * side
                waypoints.append((x, y, z))

        # Half turn: right then up
        if has_half:
            inset = n_full * pitch
            for d_idx, (dx, dy) in enumerate(directions[:2]):
                side = d_out - 2 * inset
                if d_idx >= 2:
                    side -= pitch
                if side <= 0:
                    break
                x += dx * side
                y += dy * side
                waypoints.append((x, y, z))

        points = np.array(waypoints, dtype=np.float64)

        return Trace.from_centerline(
            points=points,
            width=w_trace,
            thickness=thickness,
            conductor=conductor,
            segments_per_section=segments_per_section,
            name=name,
        )


@dataclass
class PEECPort:
    """Port definition for PEEC extraction.

    A port is defined at a segment where voltage is applied and
    current is measured.

    Parameters
    ----------
    name : str
        Port label.
    positive_segment_idx : int
        Index into TraceNetwork.all_segments for the positive terminal.
    negative_segment_idx : int
        Index for negative terminal.  -1 means ground return.
    V_ref : complex
        Reference voltage for excitation.
    """

    name: str
    positive_segment_idx: int
    negative_segment_idx: int = -1
    V_ref: complex = 1.0


@dataclass
class TraceNetwork:
    """Collection of traces with port definitions.

    Parameters
    ----------
    traces : list of Trace
        All conductor traces in the network.
    ports : list of PEECPort
        Port definitions referencing segments in all_segments.
    """

    traces: List[Trace]
    ports: List[PEECPort]

    @property
    def all_segments(self) -> List[TraceSegment]:
        """Flat list of all segments across all traces."""
        segs = []
        for trace in self.traces:
            segs.extend(trace.segments)
        return segs

    @property
    def num_segments(self) -> int:
        return sum(t.num_segments for t in self.traces)

    def build_connectivity(self) -> tuple[np.ndarray, int]:
        """Build the node connectivity for the circuit model.

        Each segment has two nodes (start, end).  Segments within a trace
        share nodes at junctions (segment[i].end == segment[i+1].start).

        Returns
        -------
        connectivity : ndarray, shape (M, 2), int
            Node indices [start_node, end_node] for each segment.
        num_nodes : int
            Total number of unique nodes.
        """
        connectivity = []
        node_idx = 0

        for trace in self.traces:
            if not trace.segments:
                continue
            # First segment gets two new nodes
            start_node = node_idx
            node_idx += 1
            end_node = node_idx
            node_idx += 1
            connectivity.append([start_node, end_node])

            # Subsequent segments share the end node of the previous
            for _ in range(1, len(trace.segments)):
                start_node = end_node
                end_node = node_idx
                node_idx += 1
                connectivity.append([start_node, end_node])

        return np.array(connectivity, dtype=np.int32), node_idx
