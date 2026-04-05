"""Non-uniform 2D rectangular grid for FDM Laplace solver.

Provides geometric refinement near conductor edges where field singularities
(1/sqrt(r) charge density at strip corners) require fine resolution.

The grid is a tensor product of two 1D coordinate arrays (x, y).  Each 1D
array is built by specifying refinement points and a geometric growth rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class NonUniformGrid:
    """Non-uniform 2D rectangular grid.

    Parameters
    ----------
    x : ndarray, shape (Nx,)
        Sorted x-coordinates of grid lines.
    y : ndarray, shape (Ny,)
        Sorted y-coordinates of grid lines.
    """

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        if len(self.x) < 3 or len(self.y) < 3:
            raise ValueError("Grid must have at least 3 points in each direction")

    @property
    def Nx(self) -> int:
        return len(self.x)

    @property
    def Ny(self) -> int:
        return len(self.y)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.Nx, self.Ny)

    @property
    def num_points(self) -> int:
        return self.Nx * self.Ny

    def dx(self) -> np.ndarray:
        """Cell widths in x, shape (Nx-1,)."""
        return np.diff(self.x)

    def dy(self) -> np.ndarray:
        """Cell widths in y, shape (Ny-1,)."""
        return np.diff(self.y)

    def ij_to_flat(self, i: int, j: int) -> int:
        """Convert (i, j) grid index to flat index (row-major)."""
        return i * self.Ny + j

    def flat_to_ij(self, idx: int) -> tuple[int, int]:
        """Convert flat index to (i, j) grid index."""
        return divmod(idx, self.Ny)


def _build_1d_grid(
    x_min: float,
    x_max: float,
    base_spacing: float,
    refine_points: Sequence[float],
    refine_spacing: float,
    growth_rate: float,
) -> np.ndarray:
    """Build a non-uniform 1D coordinate array with geometric grading.

    Places fine spacing (``refine_spacing``) near each refinement point
    and grows geometrically to ``base_spacing`` away from them.

    Parameters
    ----------
    x_min, x_max : float
        Domain extent.
    base_spacing : float
        Maximum cell size far from refinement points.
    refine_points : sequence of float
        Coordinates requiring fine resolution.
    refine_spacing : float
        Minimum cell size at refinement points.
    growth_rate : float
        Geometric growth ratio (>1).  Spacing increases by this factor
        per cell moving away from a refinement point.

    Returns
    -------
    x : ndarray
        Sorted, unique 1D coordinates.
    """
    if growth_rate <= 1.0:
        raise ValueError(f"growth_rate must be > 1, got {growth_rate}")

    # Start with domain boundaries and refinement points
    key_points = sorted(set([x_min, x_max] + list(refine_points)))

    all_coords = set()
    all_coords.add(x_min)
    all_coords.add(x_max)

    # Fill between consecutive key points
    for k in range(len(key_points) - 1):
        a, b = key_points[k], key_points[k + 1]
        span = b - a
        if span <= 0:
            continue

        a_is_refine = any(abs(a - rp) < 1e-15 for rp in refine_points)
        b_is_refine = any(abs(b - rp) < 1e-15 for rp in refine_points)

        if a_is_refine and b_is_refine:
            # Both ends refined — grade from both sides, meet in middle
            coords_left = _grade_from_point(a, span / 2, refine_spacing, base_spacing, growth_rate, direction=+1)
            coords_right = _grade_from_point(b, span / 2, refine_spacing, base_spacing, growth_rate, direction=-1)
            all_coords.update(coords_left)
            all_coords.update(coords_right)
        elif a_is_refine:
            # Fine at left, grow toward right
            coords = _grade_from_point(a, span, refine_spacing, base_spacing, growth_rate, direction=+1)
            all_coords.update(coords)
        elif b_is_refine:
            # Fine at right, grow toward left
            coords = _grade_from_point(b, span, refine_spacing, base_spacing, growth_rate, direction=-1)
            all_coords.update(coords)
        else:
            # No refinement — uniform at base spacing
            n = max(2, int(np.ceil(span / base_spacing)))
            all_coords.update(np.linspace(a, b, n + 1).tolist())

    result = np.array(sorted(all_coords))
    return result


def _grade_from_point(
    x0: float,
    span: float,
    h_min: float,
    h_max: float,
    rate: float,
    direction: int,
) -> list[float]:
    """Generate graded coordinates from x0 over a given span.

    Parameters
    ----------
    x0 : float
        Starting coordinate.
    span : float
        Total distance to cover.
    h_min : float
        Starting cell size.
    h_max : float
        Maximum cell size.
    rate : float
        Growth rate.
    direction : int
        +1 for increasing x, -1 for decreasing x.

    Returns
    -------
    coords : list of float
    """
    coords = [x0]
    pos = 0.0
    h = h_min
    while pos < span - 1e-15:
        step = min(h, span - pos)
        pos += step
        coords.append(x0 + direction * pos)
        h = min(h * rate, h_max)
    return coords


def build_grid(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    base_spacing: float,
    refine_x: Sequence[float] = (),
    refine_y: Sequence[float] = (),
    refine_spacing: Optional[float] = None,
    growth_rate: float = 1.3,
) -> NonUniformGrid:
    """Build a non-uniform 2D grid with geometric refinement.

    Parameters
    ----------
    x_range : (x_min, x_max)
        Domain extent in x.
    y_range : (y_min, y_max)
        Domain extent in y.
    base_spacing : float
        Maximum cell size far from features.
    refine_x : sequence of float
        x-coordinates needing fine resolution (e.g., conductor edges).
    refine_y : sequence of float
        y-coordinates needing fine resolution (e.g., dielectric interfaces).
    refine_spacing : float, optional
        Minimum cell size at refinement points.  Default: base_spacing / 10.
    growth_rate : float
        Geometric growth ratio.  Default 1.3.

    Returns
    -------
    NonUniformGrid
    """
    if refine_spacing is None:
        refine_spacing = base_spacing / 10.0

    x = _build_1d_grid(x_range[0], x_range[1], base_spacing,
                        list(refine_x), refine_spacing, growth_rate)
    y = _build_1d_grid(y_range[0], y_range[1], base_spacing,
                        list(refine_y), refine_spacing, growth_rate)
    return NonUniformGrid(x=x, y=y)


def build_grid_for_cross_section(
    cross_section,
    margin_factor: float = 5.0,
    base_cells: int = 200,
    refine_factor: float = 0.1,
    growth_rate: float = 1.3,
) -> NonUniformGrid:
    """Automatically build a grid from a CrossSection geometry.

    Places refinement at all conductor edges and dielectric interfaces.
    Domain extends ``margin_factor`` times the substrate height beyond
    the outermost conductor.

    Parameters
    ----------
    cross_section : CrossSection
        Geometry definition.
    margin_factor : float
        Domain extends this many substrate heights beyond conductors.
    base_cells : int
        Approximate number of cells in the larger dimension.
    refine_factor : float
        Ratio of finest to coarsest cell size.
    growth_rate : float
        Geometric growth ratio for grading.

    Returns
    -------
    NonUniformGrid
    """
    # Collect feature coordinates.
    # Signal conductors always contribute x-features.
    # Coplanar grounds (finite-width, same y as signal) also contribute
    # x-features because the gap edges are critical for CPW-type structures.
    # Wide "infinite" ground planes do not define x-features.
    x_features = []
    y_features = []
    x_bounds = []
    y_bounds = []

    signal_y_set = set()
    for c in cross_section.conductors:
        if abs(c.voltage) > 1e-30:
            signal_y_set.add(round(c.y_min, 12))
            signal_y_set.add(round(c.y_max, 12))

    for c in cross_section.conductors:
        y_features.extend([c.y_min, c.y_max])
        y_bounds.extend([c.y_min, c.y_max])
        if abs(c.voltage) > 1e-30:
            # Signal conductors: edges are physical features
            x_features.extend([c.x_min, c.x_max])
            x_bounds.extend([c.x_min, c.x_max])
        else:
            # Ground conductor: check if coplanar (same y as a signal)
            # and finite width (not a wide "infinite" ground plane)
            is_coplanar = (
                round(c.y_min, 12) in signal_y_set
                or round(c.y_max, 12) in signal_y_set
            )
            x_span = c.x_max - c.x_min
            is_finite = x_span < 0.1  # < 100mm — not an "infinite" plane
            if is_coplanar and is_finite:
                x_features.extend([c.x_min, c.x_max])
                x_bounds.extend([c.x_min, c.x_max])

    for d in cross_section.dielectric_regions:
        y_features.extend([d.y_min, d.y_max])
        y_bounds.extend([d.y_min, d.y_max])

    if not y_bounds:
        raise ValueError("CrossSection has no conductors or dielectrics")
    if not x_bounds:
        # Fallback: use all conductors for x bounds
        for c in cross_section.conductors:
            x_bounds.extend([c.x_min, c.x_max])
        if not x_bounds:
            raise ValueError("CrossSection has no conductors")

    # Find signal conductor bounding box for margin calculation
    signals = cross_section.signal_conductors()
    if signals:
        sig_y = [c.y_min for c in signals] + [c.y_max for c in signals]
        sig_x = [c.x_min for c in signals] + [c.x_max for c in signals]
    else:
        sig_y = y_bounds
        sig_x = x_bounds

    # Detect ground planes that bound the domain in y
    ground_y_values = sorted(set(
        c.y_min for c in cross_section.ground_conductors()
    ))

    # Substrate height: distance between nearest ground planes surrounding signal.
    # Coplanar grounds (at the same y as the signal) are not enclosing planes.
    sig_y_center = 0.5 * (min(sig_y) + max(sig_y))
    sig_y_min = min(sig_y)
    sig_y_max = max(sig_y)
    tol = 1e-15
    grounds_below = [y for y in ground_y_values if y < sig_y_min - tol]
    grounds_above = [y for y in ground_y_values if y > sig_y_max + tol]

    has_ground_below = len(grounds_below) > 0
    has_ground_above = len(grounds_above) > 0

    # Characteristic dimension for margin
    if has_ground_below and has_ground_above:
        # Enclosed (stripline): domain is bounded by ground planes
        y_char = grounds_above[0] - grounds_below[-1]
    elif has_ground_below:
        # Microstrip/CPW: open above
        y_char = min(sig_y) - grounds_below[-1]
    else:
        y_char = max(y_bounds) - min(y_bounds)
    y_char = max(y_char, 1e-6)

    margin = margin_factor * y_char

    # x-range: always extend with margin beyond signal conductors
    x_min = min(sig_x) - margin
    x_max = max(sig_x) + margin

    # y-range: clamp to ground planes where they exist
    if has_ground_below:
        y_min = grounds_below[-1]  # Clamp to bottom ground
    else:
        y_min = min(y_bounds) - margin

    if has_ground_above:
        y_max = grounds_above[0]   # Clamp to top ground
    else:
        y_max = max(y_bounds) + margin

    span = max(x_max - x_min, y_max - y_min)
    base_spacing = span / base_cells
    refine_spacing = base_spacing * refine_factor

    # Deduplicate feature coordinates
    refine_x = sorted(set(x_features))
    refine_y = sorted(set(y_features))

    return build_grid(
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        base_spacing=base_spacing,
        refine_x=refine_x,
        refine_y=refine_y,
        refine_spacing=refine_spacing,
        growth_rate=growth_rate,
    )
