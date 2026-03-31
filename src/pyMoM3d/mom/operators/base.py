"""Abstract base class for MoM impedance matrix operators."""

from abc import ABC, abstractmethod

import numpy as np


class AbstractOperator(ABC):
    """Base class for EFIE, MFIE, and CFIE operators.

    Each operator encapsulates:
      - Per-pair kernel computation (numpy path, called by the shared assembly loop)
      - Fast-backend dispatch (cpp / numba, operator-specific)
      - Any post-assembly corrections (e.g., MFIE identity term)

    The shared assembly loop in ``assembly.fill_matrix`` calls:
      1. ``supports_backend(backend)`` to decide whether to use the fast path.
      2. ``fill_fast(...)`` if the fast path is available.
      3. ``compute_pair_numpy(...)`` per triangle pair otherwise.
      4. ``post_assembly(...)`` after the loop completes.

    Class attributes
    ----------------
    is_symmetric : bool
        If True (default), the assembled matrix is symmetric and the numpy loop
        only fills the upper triangle, mirroring Z[n,m] = Z[m,n].  Set to False
        for operators whose matrices are not symmetric (MFIE, CFIE).
    """

    is_symmetric: bool = True

    def supports_backend(self, backend: str) -> bool:
        """Return True if this operator has a fast path for *backend*.

        The base implementation returns True only for 'numpy'.  Subclasses
        override to advertise cpp / numba support.
        """
        return backend == 'numpy'

    def fill_fast(
        self,
        backend: str,
        Z: np.ndarray,
        rwg_basis,
        mesh,
        k: float,
        eta: float,
        tri_centroids: np.ndarray,
        tri_mean_edge: np.ndarray,
        tri_twice_area: np.ndarray,
        tri_normals: np.ndarray,
        weights: np.ndarray,
        bary: np.ndarray,
        quad_order: int,
        near_threshold: float,
    ) -> None:
        """Fill *Z* in-place using a fast (non-Python-loop) backend.

        Only called when ``supports_backend(backend)`` returns True and
        backend != 'numpy'.  Raises ``NotImplementedError`` by default.

        Parameters
        ----------
        tri_normals : ndarray, shape (N_t, 3)
            Outward unit normals per triangle.  Ignored by EFIE; required by
            MFIE and CFIE.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a fast path for backend='{backend}'"
        )

    @abstractmethod
    def compute_pair_numpy(
        self,
        k: float,
        eta: float,
        mesh,
        tri_test: int,
        tri_src: int,
        fv_test: int,
        fv_src: int,
        sign_test: float,
        sign_src: float,
        l_test: float,
        l_src: float,
        A_test: float,
        A_src: float,
        quad_order: int,
        near_threshold: float,
        weights: np.ndarray,
        bary: np.ndarray,
        twice_area_test: float,
        twice_area_src: float,
        is_near: bool,
        n_hat_test: np.ndarray,
    ) -> complex:
        """Return the Z[m, n] contribution from one triangle-pair interaction.

        Called once per (tri_test, tri_src) combination in the assembly loop.
        The returned value is accumulated into Z[m, n] over all four triangle
        pairs that make up a single (m, n) basis-function interaction.

        Parameters
        ----------
        k, eta : float
            Wavenumber and intrinsic impedance.
        mesh : Mesh
            Surface mesh (provides vertices and triangles).
        tri_test, tri_src : int
            Triangle indices for test (m) and source (n) interactions.
        fv_test, fv_src : int
            Free-vertex indices for the RWG basis functions.
        sign_test, sign_src : float
            +1 for T+ triangles, -1 for T- triangles.
        l_test, l_src : float
            Edge lengths of the basis functions.
        A_test, A_src : float
            Triangle areas.
        quad_order : int
            Quadrature order.
        near_threshold : float
            Near-field singularity extraction threshold.
        weights, bary : ndarray
            Precomputed quadrature rule.
        twice_area_test, twice_area_src : float
            Precomputed 2 * triangle area.
        is_near : bool
            True if singularity extraction should be used.
        n_hat_test : ndarray, shape (3,)
            Outward unit normal of the test triangle.  Ignored by EFIE;
            required by MFIE.
        """
        ...

    def post_assembly(
        self,
        Z: np.ndarray,
        rwg_basis,
        mesh,
        k: float,
        eta: float,
    ) -> None:
        """Optional post-assembly correction applied after the main loop.

        The base implementation is a no-op.  MFIEOperator uses this to add
        the (1/2) identity term.
        """
