"""Cross-section geometry definitions for 2D field solver.

Defines conductors and dielectric regions on a 2D cross-section plane.
The cross-section x-axis is the transverse direction; the y-axis is
height (corresponding to z in the 3D solver).

Zero-thickness conductors (y_min == y_max) are modelled as internal
Dirichlet boundaries within the dielectric.  Finite-thickness conductors
fill a rectangular region with Dirichlet boundary conditions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .grid import NonUniformGrid


@dataclass
class Conductor:
    """A conductor in the 2D cross-section.

    Parameters
    ----------
    name : str
        Identifier.
    x_min, x_max : float
        Horizontal extent (m).
    y_min, y_max : float
        Vertical extent (m).  For zero-thickness: y_min == y_max.
    voltage : float
        Dirichlet potential (V).  Signal conductor = 1, ground = 0.
    """

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    voltage: float = 0.0

    @property
    def is_zero_thickness(self) -> bool:
        return abs(self.y_max - self.y_min) < 1e-15

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x_min + self.x_max)


@dataclass
class DielectricRegion:
    """A rectangular dielectric region.

    Parameters
    ----------
    name : str
        Identifier.
    x_min, x_max : float
        Horizontal extent (m).
    y_min, y_max : float
        Vertical extent (m).
    eps_r : float
        Relative permittivity.
    """

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    eps_r: float = 1.0


@dataclass
class CrossSection:
    """Complete 2D cross-section geometry.

    Collects conductors and dielectric regions.  Provides methods
    to build permittivity maps and conductor masks on a grid.

    Parameters
    ----------
    conductors : list of Conductor
    dielectric_regions : list of DielectricRegion
    """

    conductors: List[Conductor] = field(default_factory=list)
    dielectric_regions: List[DielectricRegion] = field(default_factory=list)

    def signal_conductors(self) -> List[Conductor]:
        """Return conductors with non-zero voltage."""
        return [c for c in self.conductors if abs(c.voltage) > 1e-30]

    def ground_conductors(self) -> List[Conductor]:
        """Return conductors with zero voltage."""
        return [c for c in self.conductors if abs(c.voltage) <= 1e-30]

    def conductor_edges_x(self) -> List[float]:
        """All x-coordinates of conductor edges (for grid refinement)."""
        edges = []
        for c in self.conductors:
            edges.extend([c.x_min, c.x_max])
        return sorted(set(edges))

    def conductor_edges_y(self) -> List[float]:
        """All y-coordinates of conductor edges (for grid refinement)."""
        edges = []
        for c in self.conductors:
            edges.extend([c.y_min, c.y_max])
        return sorted(set(edges))

    def dielectric_interfaces_y(self) -> List[float]:
        """All y-coordinates of dielectric interfaces."""
        interfaces = []
        for d in self.dielectric_regions:
            interfaces.extend([d.y_min, d.y_max])
        return sorted(set(interfaces))

    def eps_r_at_point(self, x: float, y: float) -> float:
        """Return eps_r at a point by checking dielectric regions.

        Regions are checked in reverse order (last added wins).
        Default is 1.0 (vacuum) if no region contains the point.
        """
        for d in reversed(self.dielectric_regions):
            if d.x_min <= x <= d.x_max and d.y_min <= y <= d.y_max:
                return d.eps_r
        return 1.0

    def eps_r_map(self, grid: NonUniformGrid) -> np.ndarray:
        """Build 2D permittivity array on grid.

        Parameters
        ----------
        grid : NonUniformGrid

        Returns
        -------
        eps : ndarray, shape (Nx, Ny), float64
            Relative permittivity at each grid point.
        """
        Nx, Ny = grid.shape
        eps = np.ones((Nx, Ny), dtype=np.float64)

        for d in self.dielectric_regions:
            # Find grid points inside this region
            ix = (grid.x >= d.x_min - 1e-15) & (grid.x <= d.x_max + 1e-15)
            iy = (grid.y >= d.y_min - 1e-15) & (grid.y <= d.y_max + 1e-15)
            eps[np.ix_(ix, iy)] = d.eps_r

        return eps

    def conductor_mask(self, grid: NonUniformGrid) -> np.ndarray:
        """Boolean mask: True where grid point is inside/on a conductor.

        For zero-thickness conductors, matches grid points at y = y_min
        within the x-range.

        Parameters
        ----------
        grid : NonUniformGrid

        Returns
        -------
        mask : ndarray, shape (Nx, Ny), bool
        """
        Nx, Ny = grid.shape
        mask = np.zeros((Nx, Ny), dtype=bool)

        for c in self.conductors:
            ix = (grid.x >= c.x_min - 1e-15) & (grid.x <= c.x_max + 1e-15)
            if c.is_zero_thickness:
                iy = np.abs(grid.y - c.y_min) < 1e-15
            else:
                iy = (grid.y >= c.y_min - 1e-15) & (grid.y <= c.y_max + 1e-15)
            mask[np.ix_(ix, iy)] = True

        return mask

    def voltage_map(self, grid: NonUniformGrid) -> np.ndarray:
        """Voltage values at conductor grid points.

        Parameters
        ----------
        grid : NonUniformGrid

        Returns
        -------
        V : ndarray, shape (Nx, Ny), float64
            Voltage at conductor points, 0 elsewhere.
        """
        Nx, Ny = grid.shape
        V = np.zeros((Nx, Ny), dtype=np.float64)

        for c in self.conductors:
            ix = (grid.x >= c.x_min - 1e-15) & (grid.x <= c.x_max + 1e-15)
            if c.is_zero_thickness:
                iy = np.abs(grid.y - c.y_min) < 1e-15
            else:
                iy = (grid.y >= c.y_min - 1e-15) & (grid.y <= c.y_max + 1e-15)
            V[np.ix_(ix, iy)] = c.voltage

        return V

    def vacuum_copy(self) -> CrossSection:
        """Return a copy with all dielectric regions set to eps_r = 1.0."""
        cs = copy.deepcopy(self)
        for d in cs.dielectric_regions:
            d.eps_r = 1.0
        return cs
