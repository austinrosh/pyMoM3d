"""TabulatedPNGFBackend: precomputed lookup-table Green's function.

Precomputes the smooth correction G_ML - G_fs on a 3D grid
(ρ, z_src, z_obs) using any GreensBackend, then serves evaluations via
scipy RegularGridInterpolator.  After precomputation, each evaluation is
O(1) with negligible overhead — critical for large frequency sweeps where
the same mesh is reused at many frequencies.

Usage
-----
    from pyMoM3d.greens.layered.tabulated import TabulatedPNGFBackend
    from pyMoM3d.greens.layered.layer_recursion import LayerRecursionBackend

    # Build reference backend
    ref = LayerRecursionBackend(stack, freq, source_layer)

    # Precompute table over mesh geometry
    tab = TabulatedPNGFBackend()
    tab.precompute(
        rho_grid    = np.logspace(-4, -1, 80),   # 0.1 mm to 100 mm
        z_src_grid  = np.array([0.005]),          # single z-plane (flat mesh)
        z_obs_grid  = np.array([0.005]),
        reference_backend = ref,
    )

    # Drop-in replacement for the reference backend
    g_smooth = tab.scalar_G(r_obs, r_src)

    # Persist to disk
    tab.save('gf_table_1GHz.npz')
    tab2 = TabulatedPNGFBackend.load('gf_table_1GHz.npz')

Grid design notes
-----------------
- ρ: use log-spaced grid to capture both near-field and far-field behaviour.
  Minimum ρ should be slightly above zero (e.g., λ/1000) to avoid evaluating
  the correction at the source point.
- z_src, z_obs: for flat meshes (all elements at the same z), a single-point
  grid suffices.  For volumetric or curved meshes, use a fine linear grid
  spanning the mesh extent.
- All 9 dyadic components are precomputed and stored.
"""

from __future__ import annotations

import numpy as np

from ..base import GreensBackend


class TabulatedPNGFBackend(GreensBackend):
    """Lookup-table smooth correction backend.

    Must call ``precompute()`` (or ``load()``) before use.

    Parameters
    ----------
    interpolation_method : str
        Interpolation method passed to RegularGridInterpolator.
        'linear' (default) or 'cubic'.
    """

    def __init__(self, interpolation_method: str = 'linear'):
        self._method   = interpolation_method
        self._ready    = False
        # Grid axes
        self._rho_grid   = None
        self._zsrc_grid  = None
        self._zobs_grid  = None
        # Raw table arrays (stored for save/load roundtrip)
        self._scalar_re  = None
        self._scalar_im  = None
        self._dyadic_re  = None
        self._dyadic_im  = None
        # Interpolators: one per component (scalar, 9 dyadic)
        self._interp_scalar     = None
        self._interp_dyadic     = None   # list of 9 interpolators (row, col)

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def precompute(
        self,
        rho_grid: np.ndarray,
        z_src_grid: np.ndarray,
        z_obs_grid: np.ndarray,
        reference_backend: GreensBackend,
    ) -> None:
        """Evaluate the reference backend on the grid and build interpolators.

        Parameters
        ----------
        rho_grid : (N_rho,) float
            Radial separations ρ = √((x-x')² + (y-y')²).  Must be strictly
            positive.  Log-spacing recommended.
        z_src_grid : (N_zs,) float
            Source z-coordinates (z' values).
        z_obs_grid : (N_zo,) float
            Observation z-coordinates (z values).
        reference_backend : GreensBackend
            Any backend implementing scalar_G and dyadic_G.  Typically
            LayerRecursionBackend for accuracy.
        """
        from scipy.interpolate import RegularGridInterpolator

        rho_grid   = np.asarray(rho_grid,   dtype=np.float64)
        z_src_grid = np.asarray(z_src_grid, dtype=np.float64)
        z_obs_grid = np.asarray(z_obs_grid, dtype=np.float64)

        N_rho = len(rho_grid)
        N_zs  = len(z_src_grid)
        N_zo  = len(z_obs_grid)

        # Allocate tables: (N_rho, N_zs, N_zo)
        scalar_re = np.empty((N_rho, N_zs, N_zo), dtype=np.float64)
        scalar_im = np.empty_like(scalar_re)
        dyadic_re = np.empty((N_rho, N_zs, N_zo, 3, 3), dtype=np.float64)
        dyadic_im = np.empty_like(dyadic_re)

        # Evaluate reference backend at all grid points
        for iz_s, z_s in enumerate(z_src_grid):
            for iz_o, z_o in enumerate(z_obs_grid):
                # Build batched (r, r') for all ρ values at this (z_s, z_o)
                # Place all points along x-axis: r = (ρ, 0, z_o), r' = (0, 0, z_s)
                r_obs   = np.column_stack([
                    rho_grid, np.zeros(N_rho), np.full(N_rho, z_o)
                ])
                r_src   = np.column_stack([
                    np.zeros(N_rho), np.zeros(N_rho), np.full(N_rho, z_s)
                ])

                g_s = reference_backend.scalar_G(r_obs, r_src)   # (N_rho,)
                g_d = reference_backend.dyadic_G(r_obs, r_src)   # (N_rho, 3, 3)

                scalar_re[:, iz_s, iz_o] = g_s.real
                scalar_im[:, iz_s, iz_o] = g_s.imag
                dyadic_re[:, iz_s, iz_o] = g_d.real
                dyadic_im[:, iz_s, iz_o] = g_d.imag

        # Store raw arrays for save/load roundtrip
        self._scalar_re = scalar_re
        self._scalar_im = scalar_im
        self._dyadic_re = dyadic_re
        self._dyadic_im = dyadic_im

        # Build interpolators
        axes = (rho_grid, z_src_grid, z_obs_grid)
        kw   = dict(method=self._method, bounds_error=False, fill_value=None)

        self._interp_scalar = (
            RegularGridInterpolator(axes, scalar_re, **kw),
            RegularGridInterpolator(axes, scalar_im, **kw),
        )

        self._interp_dyadic = []
        for row in range(3):
            row_list = []
            for col in range(3):
                row_list.append((
                    RegularGridInterpolator(axes, dyadic_re[..., row, col], **kw),
                    RegularGridInterpolator(axes, dyadic_im[..., row, col], **kw),
                ))
            self._interp_dyadic.append(row_list)

        self._rho_grid   = rho_grid
        self._zsrc_grid  = z_src_grid
        self._zobs_grid  = z_obs_grid
        self._ready      = True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the precomputed table to a .npz file.

        Parameters
        ----------
        path : str
            File path (extension .npz recommended).
        """
        if not self._ready:
            raise RuntimeError("No table to save.  Call precompute() first.")

        np.savez(
            path,
            rho_grid=self._rho_grid,
            z_src_grid=self._zsrc_grid,
            z_obs_grid=self._zobs_grid,
            scalar_re=self._scalar_re, scalar_im=self._scalar_im,
            dyadic_re=self._dyadic_re, dyadic_im=self._dyadic_im,
        )

    @classmethod
    def load(cls, path: str, interpolation_method: str = 'linear') -> 'TabulatedPNGFBackend':
        """Load a previously saved table.

        Parameters
        ----------
        path : str
            Path to .npz file created by save().
        interpolation_method : str
            Interpolation method.

        Returns
        -------
        TabulatedPNGFBackend (ready to use immediately)
        """
        from scipy.interpolate import RegularGridInterpolator

        data = np.load(path)
        obj  = cls(interpolation_method=interpolation_method)

        rho_grid   = data['rho_grid']
        z_src_grid = data['z_src_grid']
        z_obs_grid = data['z_obs_grid']
        scalar_re  = data['scalar_re']
        scalar_im  = data['scalar_im']
        dyadic_re  = data['dyadic_re']
        dyadic_im  = data['dyadic_im']

        axes = (rho_grid, z_src_grid, z_obs_grid)
        kw   = dict(method=interpolation_method, bounds_error=False, fill_value=None)

        obj._interp_scalar = (
            RegularGridInterpolator(axes, scalar_re, **kw),
            RegularGridInterpolator(axes, scalar_im, **kw),
        )
        obj._interp_dyadic = []
        for row in range(3):
            row_list = []
            for col in range(3):
                row_list.append((
                    RegularGridInterpolator(axes, dyadic_re[..., row, col], **kw),
                    RegularGridInterpolator(axes, dyadic_im[..., row, col], **kw),
                ))
            obj._interp_dyadic.append(row_list)

        obj._rho_grid  = rho_grid
        obj._zsrc_grid = z_src_grid
        obj._zobs_grid = z_obs_grid
        obj._scalar_re = scalar_re
        obj._scalar_im = scalar_im
        obj._dyadic_re = dyadic_re
        obj._dyadic_im = dyadic_im
        obj._ready     = True
        return obj

    # ------------------------------------------------------------------
    # GreensBackend interface
    # ------------------------------------------------------------------

    def scalar_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Smooth correction via table lookup.

        Parameters
        ----------
        r, r_prime : (..., 3) float64

        Returns
        -------
        (...,) complex128
        """
        self._check_ready()
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]

        pts = self._query_points(r.reshape(-1, 3), r_prime.reshape(-1, 3))
        g_re = self._interp_scalar[0](pts)
        g_im = self._interp_scalar[1](pts)
        return (g_re + 1j * g_im).reshape(shape)

    def dyadic_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """All 9 dyadic components via table lookup.

        Parameters
        ----------
        r, r_prime : (..., 3) float64

        Returns
        -------
        (..., 3, 3) complex128
        """
        self._check_ready()
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]
        N       = int(np.prod(shape)) if shape else 1

        pts = self._query_points(r.reshape(-1, 3), r_prime.reshape(-1, 3))
        G   = np.zeros((N, 3, 3), dtype=np.complex128)
        for row in range(3):
            for col in range(3):
                re = self._interp_dyadic[row][col][0](pts)
                im = self._interp_dyadic[row][col][1](pts)
                G[:, row, col] = re + 1j * im

        return G.reshape(shape + (3, 3))

    def grad_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Gradient via central finite differences of scalar_G (table lookup)."""
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)

        # Use finite FD step adapted to the coarsest grid spacing
        h    = max(np.diff(self._rho_grid).min(), 1e-9) * 0.1
        grad = np.zeros(r.shape, dtype=np.complex128)
        for i in range(3):
            dr        = np.zeros_like(r)
            dr[..., i] = h
            gp        = self.scalar_G(r + dr, r_prime)
            gm        = self.scalar_G(r - dr, r_prime)
            grad[..., i] = (gp - gm) / (2.0 * h)
        return grad

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "TabulatedPNGFBackend: call precompute() or load() before use."
            )

    def _query_points(self, r_flat: np.ndarray, rp_flat: np.ndarray) -> np.ndarray:
        """Build (N, 3) query array [ρ, z_src, z_obs] for interpolator."""
        dx  = r_flat[:, 0] - rp_flat[:, 0]
        dy  = r_flat[:, 1] - rp_flat[:, 1]
        rho = np.sqrt(dx**2 + dy**2)
        z_obs = r_flat[:, 2]
        z_src = rp_flat[:, 2]
        return np.column_stack([rho, z_src, z_obs])

