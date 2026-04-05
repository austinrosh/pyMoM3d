"""2D Laplace FDM solver for transmission line cross-sections.

Solves ∇·(ε_r ∇V) = 0 on a non-uniform rectangular grid using a
5-point finite-difference stencil with harmonic averaging of permittivity
at cell faces.

The solver builds a sparse CSR system matrix and uses
``scipy.sparse.linalg.spsolve`` for the direct solve.

Charge per unit length is computed via Gauss's law contour integration
around each conductor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import NonUniformGrid, build_grid_for_cross_section
from .geometry import CrossSection, Conductor


@dataclass
class LaplaceSolution:
    """Solution of a 2D Laplace problem.

    Parameters
    ----------
    V : ndarray, shape (Nx, Ny)
        Electric potential on the grid.
    grid : NonUniformGrid
        The computational grid.
    cross_section : CrossSection
        Geometry used.
    eps_r_map : ndarray, shape (Nx, Ny)
        Permittivity distribution used in the solve.
    """

    V: np.ndarray
    grid: NonUniformGrid
    cross_section: CrossSection
    eps_r_map: np.ndarray


class LaplaceSolver2D:
    """2D Laplace equation solver for electrostatic cross-sections.

    Parameters
    ----------
    cross_section : CrossSection
        Geometry definition with conductors and dielectric regions.
    grid : NonUniformGrid, optional
        Computational grid.  If None, built automatically from geometry.
    margin_factor : float
        Domain margin as multiple of substrate height (if grid is auto-built).
    base_cells : int
        Approximate grid cells in larger dimension (if grid is auto-built).
    """

    def __init__(
        self,
        cross_section: CrossSection,
        grid: Optional[NonUniformGrid] = None,
        margin_factor: float = 5.0,
        base_cells: int = 200,
    ):
        self.cross_section = cross_section
        if grid is None:
            grid = build_grid_for_cross_section(
                cross_section,
                margin_factor=margin_factor,
                base_cells=base_cells,
            )
        self.grid = grid

    def solve(
        self,
        eps_r_map: Optional[np.ndarray] = None,
    ) -> LaplaceSolution:
        """Solve the Laplace equation.

        Parameters
        ----------
        eps_r_map : ndarray, shape (Nx, Ny), optional
            Permittivity distribution.  If None, built from cross-section.

        Returns
        -------
        LaplaceSolution
        """
        grid = self.grid
        cs = self.cross_section

        if eps_r_map is None:
            eps_r_map = cs.eps_r_map(grid)

        cond_mask = cs.conductor_mask(grid)
        volt_map = cs.voltage_map(grid)

        A = self._build_system_matrix(eps_r_map, cond_mask)
        b = self._build_rhs(cond_mask, volt_map)

        V_flat = spla.spsolve(A, b)
        V = V_flat.reshape(grid.shape)

        return LaplaceSolution(
            V=V,
            grid=grid,
            cross_section=cs,
            eps_r_map=eps_r_map,
        )

    def _build_system_matrix(
        self,
        eps: np.ndarray,
        cond_mask: np.ndarray,
    ) -> sp.csr_matrix:
        """Build the sparse FDM system matrix.

        Uses a 5-point stencil on the non-uniform grid with harmonic
        averaging of permittivity at cell faces.

        Interior point (i,j):
            eps_e * (V[i+1,j] - V[i,j]) / dx_e  (east flux per unit dy)
            + similar for west, north, south
            = 0

        Conductor and boundary points: identity row (Dirichlet).
        """
        grid = self.grid
        Nx, Ny = grid.shape
        N = Nx * Ny

        dx = grid.dx()  # (Nx-1,)
        dy = grid.dy()  # (Ny-1,)

        # Preallocate COO arrays — at most 5 entries per point
        max_nnz = 5 * N
        row = np.empty(max_nnz, dtype=np.int32)
        col = np.empty(max_nnz, dtype=np.int32)
        data = np.empty(max_nnz, dtype=np.float64)
        nnz = 0

        for i in range(Nx):
            for j in range(Ny):
                idx = i * Ny + j
                is_boundary = (i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1)

                if cond_mask[i, j] or is_boundary:
                    # Dirichlet: V[i,j] = prescribed value
                    row[nnz] = idx
                    col[nnz] = idx
                    data[nnz] = 1.0
                    nnz += 1
                    continue

                # Non-uniform spacing
                dx_w = dx[i - 1]      # x[i] - x[i-1]
                dx_e = dx[i]          # x[i+1] - x[i]
                dy_s = dy[j - 1]      # y[j] - y[j-1]
                dy_n = dy[j]          # y[j+1] - y[j]
                dx_c = 0.5 * (dx_w + dx_e)
                dy_c = 0.5 * (dy_s + dy_n)

                # Harmonic average of permittivity at each face
                eps_e = _harmonic_avg(eps[i, j], eps[i + 1, j])
                eps_w = _harmonic_avg(eps[i, j], eps[i - 1, j])
                eps_n = _harmonic_avg(eps[i, j], eps[i, j + 1])
                eps_s = _harmonic_avg(eps[i, j], eps[i, j - 1])

                # Coefficients
                c_e = eps_e / (dx_e * dx_c)
                c_w = eps_w / (dx_w * dx_c)
                c_n = eps_n / (dy_n * dy_c)
                c_s = eps_s / (dy_s * dy_c)
                c_c = -(c_e + c_w + c_n + c_s)

                # Center
                row[nnz] = idx; col[nnz] = idx; data[nnz] = c_c; nnz += 1
                # East
                row[nnz] = idx; col[nnz] = (i + 1) * Ny + j; data[nnz] = c_e; nnz += 1
                # West
                row[nnz] = idx; col[nnz] = (i - 1) * Ny + j; data[nnz] = c_w; nnz += 1
                # North
                row[nnz] = idx; col[nnz] = i * Ny + (j + 1); data[nnz] = c_n; nnz += 1
                # South
                row[nnz] = idx; col[nnz] = i * Ny + (j - 1); data[nnz] = c_s; nnz += 1

        row = row[:nnz]
        col = col[:nnz]
        data = data[:nnz]

        return sp.csr_matrix((data, (row, col)), shape=(N, N))

    def _build_rhs(
        self,
        cond_mask: np.ndarray,
        volt_map: np.ndarray,
    ) -> np.ndarray:
        """Build the RHS vector.

        Non-zero only at Dirichlet points (conductors and boundaries).
        """
        grid = self.grid
        Nx, Ny = grid.shape
        b = np.zeros(Nx * Ny, dtype=np.float64)

        for i in range(Nx):
            for j in range(Ny):
                if cond_mask[i, j]:
                    b[i * Ny + j] = volt_map[i, j]
                # Boundary points: V = 0 (already zero in b)

        return b

    def integrate_charge(
        self,
        solution: LaplaceSolution,
        conductor: Conductor,
    ) -> float:
        """Compute charge per unit length on a conductor via Gauss's law.

        Integrates ε₀·ε_r·(∂V/∂n) around a rectangular contour enclosing
        the conductor.  The contour passes through grid faces (between
        adjacent grid points), ensuring correct flux calculation.

        The contour is placed between the conductor boundary and the first
        non-conductor grid row/column outside.

        Parameters
        ----------
        solution : LaplaceSolution
            Solved potential.
        conductor : Conductor
            The conductor to integrate around.

        Returns
        -------
        Q_pul : float
            Charge per unit length (C/m).
        """
        from ..utils.constants import eps0

        grid = solution.grid
        V = solution.V
        eps = solution.eps_r_map
        x, y = grid.x, grid.y
        Nx, Ny = grid.shape

        # Find the grid index ranges bounding the conductor
        i_min = np.searchsorted(x, conductor.x_min - 1e-15)
        i_max = np.searchsorted(x, conductor.x_max + 1e-15) - 1

        if conductor.is_zero_thickness:
            j_cond = np.argmin(np.abs(y - conductor.y_min))
            j_min = j_cond
            j_max = j_cond
        else:
            j_min = np.searchsorted(y, conductor.y_min - 1e-15)
            j_max = np.searchsorted(y, conductor.y_max + 1e-15) - 1

        # Contour faces: one half-step outside the conductor boundary.
        # Bottom face: between j_min-1 and j_min  (flux through face at y between these)
        # Top face:    between j_max and j_max+1
        # Left face:   between i_min-1 and i_min
        # Right face:  between i_max and i_max+1

        Q = 0.0

        # --- Bottom face: horizontal face between rows j_min-1 and j_min ---
        # Outward normal points down (-y).
        # Flux = -ε·(V[i,j_min] - V[i,j_min-1]) / (y[j_min] - y[j_min-1]) * dx
        if j_min > 0:
            j_face = j_min
            for i in range(i_min, i_max + 1):
                dVdy = (V[i, j_face] - V[i, j_face - 1]) / (y[j_face] - y[j_face - 1])
                eps_face = _harmonic_avg(eps[i, j_face - 1], eps[i, j_face])
                dx = _cell_width(x, i)
                Q += -eps_face * dVdy * dx

        # --- Top face: horizontal face between rows j_max and j_max+1 ---
        # Outward normal points up (+y).
        # Flux = +ε·(V[i,j_max+1] - V[i,j_max]) / (y[j_max+1] - y[j_max]) * dx
        if j_max < Ny - 1:
            j_face = j_max
            for i in range(i_min, i_max + 1):
                dVdy = (V[i, j_face + 1] - V[i, j_face]) / (y[j_face + 1] - y[j_face])
                eps_face = _harmonic_avg(eps[i, j_face], eps[i, j_face + 1])
                dx = _cell_width(x, i)
                Q += eps_face * dVdy * dx

        # --- Left face: vertical face between columns i_min-1 and i_min ---
        # Outward normal points left (-x).
        # Flux spans from j_min to j_max, but we must exclude corners (already in top/bottom).
        if i_min > 0:
            i_face = i_min
            for j in range(j_min, j_max + 1):
                dVdx = (V[i_face, j] - V[i_face - 1, j]) / (x[i_face] - x[i_face - 1])
                eps_face = _harmonic_avg(eps[i_face - 1, j], eps[i_face, j])
                dy = _cell_width(y, j)
                Q += -eps_face * dVdx * dy

        # --- Right face: vertical face between columns i_max and i_max+1 ---
        # Outward normal points right (+x).
        if i_max < Nx - 1:
            i_face = i_max
            for j in range(j_min, j_max + 1):
                dVdx = (V[i_face + 1, j] - V[i_face, j]) / (x[i_face + 1] - x[i_face])
                eps_face = _harmonic_avg(eps[i_face, j], eps[i_face + 1, j])
                dy = _cell_width(y, j)
                Q += eps_face * dVdx * dy

        return Q * eps0


def _harmonic_avg(a: float, b: float) -> float:
    """Harmonic average of two values.  Handles a==b without division issues."""
    if abs(a - b) < 1e-30 * (abs(a) + abs(b) + 1e-30):
        return a
    return 2.0 * a * b / (a + b)


def _cell_width(coords: np.ndarray, idx: int) -> float:
    """Width of the dual cell centered at coords[idx].

    For interior points: half-cell on each side.
    For boundary points: half-cell on the interior side.
    """
    N = len(coords)
    if idx == 0:
        return 0.5 * (coords[1] - coords[0])
    elif idx == N - 1:
        return 0.5 * (coords[-1] - coords[-2])
    else:
        return 0.5 * (coords[idx + 1] - coords[idx - 1])
