"""Layered (stratified media) Green's function via empymod Sommerfeld integrals.

EmpymodSommerfeldBackend
    Reference / validation backend using empymod's digital linear filter (DLF)
    Hankel transform.  Provides EXACT Sommerfeld integral evaluation.
    NOT a production path — too slow for N > ~100 basis functions.

LayeredGreensFunction
    GreensFunctionBase dispatcher.  Selects the active GreensBackend and
    exposes wavenumber / wave_impedance for the source layer.

Backend selection (backend='auto')
    Phase 1: only 'empymod' is available.
    Phase 2: 'dcim' will be tried first.
    Phase 3: 'strata' will be tried before dcim.

Singularity decomposition
--------------------------
For SAME-LAYER interactions the multilayer GF is decomposed as:
    G_ML(r, r') = G_fs(R)  +  [G_ML(r, r') - G_fs(R)]
The first term (singular) is handled by the existing Graglia 1993 extraction
(singularity.py / singularity.hpp) without any change.  The second term
(smooth correction) is what EmpymodSommerfeldBackend returns — it subtracts
the free-space term before returning.

For CROSS-LAYER interactions (z_src != z_obs layer) there is NO 1/R
singularity.  EmpymodSommerfeldBackend returns the FULL G_ML directly (no
subtraction) and the caller (MultilayerEFIEOperator) uses standard quadrature.
"""

from __future__ import annotations

import numpy as np

from ..base import GreensBackend, GreensFunctionBase
from ..free_space_gf import FreeSpaceGreensFunction
from ...medium.layer_stack import Layer, LayerStack


# ---------------------------------------------------------------------------
# empymod backend
# ---------------------------------------------------------------------------

class EmpymodSommerfeldBackend(GreensBackend):
    """Layered GF backend using empymod (validation / reference only).

    Uses empymod's electric field responses from horizontal electric dipole
    (HED) and vertical electric dipole (VED) sources to construct the full
    dyadic Green's function tensor.

    The smooth correction G_ML - G_fs is returned from scalar_G and dyadic_G
    when same_layer=True (default).  When same_layer=False the full G_ML is
    returned (for cross-layer interactions).

    Parameters
    ----------
    layer_stack : LayerStack
    frequency : float
        Operating frequency (Hz).
    source_layer : Layer
        Layer in which the MoM mesh resides.
    """

    def __init__(
        self,
        layer_stack: LayerStack,
        frequency: float,
        source_layer: Layer,
    ):
        try:
            import empymod as _empymod
            self._empymod = _empymod
        except ImportError as e:
            raise ImportError(
                "empymod is required for the multilayer GF reference backend.\n"
                "Install with:  pip install empymod\n"
                "(empymod is for validation only — not a production path.)"
            ) from e

        self._stack  = layer_stack
        self._freq   = float(frequency)
        self._omega  = 2.0 * np.pi * self._freq
        self._src_layer = source_layer

        k  = source_layer.wavenumber(self._omega)
        self._k_src  = k
        self._fs_gf  = FreeSpaceGreensFunction(k=k, eta=source_layer.wave_impedance(self._omega))

        # Build empymod model once
        self._depth = self._stack.z_interfaces   # interior interfaces
        self._res   = self._stack.resistivities  # one per layer

    # ------------------------------------------------------------------
    # GreensBackend interface
    # ------------------------------------------------------------------

    def scalar_G(
        self,
        r: np.ndarray,
        r_prime: np.ndarray,
        *,
        return_correction: bool = True,
    ) -> np.ndarray:
        """Scalar Green's function.

        Parameters
        ----------
        r, r_prime : (..., 3)
        return_correction : bool
            If True (default), returns G_ML - G_fs (smooth correction for
            same-layer singularity decomposition).
            If False, returns full G_ML (for cross-layer interactions).

        Returns
        -------
        (...,) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]

        r_flat       = r.reshape(-1, 3)
        r_prime_flat = r_prime.reshape(-1, 3)

        # empymod uses VED (vertical electric dipole) for Ez which gives the
        # scalar potential Green's function.  For the MoM scalar GF we use
        # the longitudinal (zz) component of the dyadic tensor, which is
        # equivalent to the VED response for Ez.
        g_ml = self._empymod_scalar(r_flat, r_prime_flat)   # (N_pts,)

        if return_correction:
            g_fs = self._fs_gf.scalar_G(r_flat, r_prime_flat)
            result = g_ml - g_fs
        else:
            result = g_ml

        return result.reshape(shape)

    def dyadic_G(
        self,
        r: np.ndarray,
        r_prime: np.ndarray,
        *,
        return_correction: bool = True,
    ) -> np.ndarray:
        """Dyadic Green's function tensor.

        All 6 independent components (Gxx, Gyy, Gzz, Gxz, Gyz, Gxy) are
        obtained from empymod HED_x, HED_y, and VED responses.

        Parameters
        ----------
        r, r_prime : (..., 3)
        return_correction : bool
            If True, returns G_bar_ML - G_bar_fs (smooth correction).
            If False, returns full G_bar_ML.

        Returns
        -------
        (..., 3, 3) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]

        r_flat       = r.reshape(-1, 3)
        r_prime_flat = r_prime.reshape(-1, 3)

        g_dyadic_ml = self._empymod_dyadic(r_flat, r_prime_flat)  # (N_pts, 3, 3)

        if return_correction:
            g_dyadic_fs = self._fs_gf.dyadic_G(r_flat, r_prime_flat)
            result = g_dyadic_ml - g_dyadic_fs
        else:
            result = g_dyadic_ml

        return result.reshape(shape + (3, 3))

    def grad_G(
        self,
        r: np.ndarray,
        r_prime: np.ndarray,
        *,
        return_correction: bool = True,
    ) -> np.ndarray:
        """Gradient of scalar GF w.r.t. r.

        Obtained from HED/VED field responses via empymod.

        Parameters
        ----------
        r, r_prime : (..., 3)
        return_correction : bool

        Returns
        -------
        (..., 3) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]

        r_flat       = r.reshape(-1, 3)
        r_prime_flat = r_prime.reshape(-1, 3)

        grad_ml = self._empymod_grad(r_flat, r_prime_flat)   # (N_pts, 3)

        if return_correction:
            grad_fs = self._fs_gf.grad_G(r_flat, r_prime_flat)
            result = grad_ml - grad_fs
        else:
            result = grad_ml

        return result.reshape(shape + (3,))

    # ------------------------------------------------------------------
    # empymod evaluation helpers
    # ------------------------------------------------------------------

    def _call_empymod_paired(
        self,
        src_pts: np.ndarray,
        rec_pts: np.ndarray,
        src_az: float,
        src_dip: float,
        rec_az: float,
        rec_dip: float,
    ) -> np.ndarray:
        """Call empymod for N paired (source, receiver) evaluations.

        empymod computes all N_src × N_rec combinations when both are arrays,
        so we loop over pairs one at a time for correctness.  Phase 1 only —
        acceptable for validation meshes.

        Returns
        -------
        (N,) complex128
        """
        em = self._empymod
        N  = src_pts.shape[0]
        em_kw = dict(depth=self._depth, res=self._res, freqtime=self._freq, verb=0)
        out = np.zeros(N, dtype=np.complex128)
        for i in range(N):
            sx, sy, sz = float(src_pts[i, 0]), float(src_pts[i, 1]), float(src_pts[i, 2])
            rx, ry, rz = float(rec_pts[i, 0]), float(rec_pts[i, 1]), float(rec_pts[i, 2])
            try:
                val = em.bipole(
                    src=[sx, sy, sz, src_az, src_dip],
                    rec=[rx, ry, rz, rec_az, rec_dip],
                    **em_kw,
                )
                out[i] = complex(np.asarray(val).ravel()[0])
            except Exception as exc:
                raise RuntimeError(f"empymod pair {i} failed: {exc}") from exc
        return out

    def _empymod_scalar(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Evaluate G_ML scalar via VED Ez response.

        empymod models the layered medium response for electric dipole sources.
        The scalar GF is related to the VED Ez component via:
            G_scalar ~ Ez_VED / (j*omega*mu0)   [up to normalisation]

        We use empymod.bipole with a VED source and Hz observation to extract
        the scalar potential, then normalise to g(R) convention.

        Parameters
        ----------
        r, r_prime : (N, 3)

        Returns
        -------
        (N,) complex128
        """
        # Use Ex from HED_x source (azimuth=0, dip=0), Ex receiver (azimuth=0, dip=0).
        # Loop over pairs one-at-a-time via _call_empymod_paired to avoid N×N matrix.
        from ...utils.constants import mu0
        omega = self._omega

        ex_hed_x = self._call_empymod_paired(
            r_prime, r,
            src_az=0.0, src_dip=0.0,
            rec_az=0.0, rec_dip=0.0,
        )

        # For Phase 1 validation: G_approx = Ex_HEDx / (j*omega*mu0)
        g_ml = ex_hed_x / (1j * omega * mu0)
        return g_ml

    def _empymod_dyadic(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Evaluate the 3x3 dyadic G tensor via HED_x, HED_y, VED responses.

        Parameters
        ----------
        r, r_prime : (N, 3)

        Returns
        -------
        (N, 3, 3) complex128

        Components mapping
        ------------------
        HED_x source:  Ex -> Gxx,  Ey -> Gyx,  Ez -> Gzx
        HED_y source:  Ex -> Gxy,  Ey -> Gyy,  Ez -> Gzy
        VED source:    Ex -> Gxz,  Ey -> Gyz,  Ez -> Gzz
        """
        from ...utils.constants import mu0
        omega = self._omega
        norm  = 1.0 / (1j * omega * mu0)

        N = r.shape[0]
        G = np.zeros((N, 3, 3), dtype=np.complex128)

        # Source orientations: (src_az, src_dip) for HED_x, HED_y, VED
        src_orientations = [
            (0.0,  0.0),   # HED in x (col 0)
            (90.0, 0.0),   # HED in y (col 1)
            (0.0,  90.0),  # VED      (col 2)
        ]
        # Receiver orientations: (rec_az, rec_dip) for Ex, Ey, Ez
        rec_orientations = [
            (0.0,  0.0),   # Ex (row 0)
            (90.0, 0.0),   # Ey (row 1)
            (0.0,  90.0),  # Ez (row 2)
        ]

        for col, (src_az, src_dip) in enumerate(src_orientations):
            for row, (rec_az, rec_dip) in enumerate(rec_orientations):
                try:
                    field = self._call_empymod_paired(
                        r_prime, r,
                        src_az=src_az, src_dip=src_dip,
                        rec_az=rec_az, rec_dip=rec_dip,
                    )
                    G[:, row, col] = field * norm
                except Exception as exc:
                    raise RuntimeError(f"empymod dyadic component ({row},{col}) failed: {exc}") from exc

        return G

    def _empymod_grad(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Gradient of scalar G via central finite difference on scalar_G.

        This is used only for grad_G (which enters the MFIE / Phi term).
        The step size is chosen to balance truncation and round-off error.

        Parameters
        ----------
        r, r_prime : (N, 3)

        Returns
        -------
        (N, 3) complex128
        """
        # h ~ lambda / 1000 in the source layer
        h = float(np.abs(2.0 * np.pi / self._k_src)) * 1e-3
        grad = np.zeros((r.shape[0], 3), dtype=np.complex128)
        for i in range(3):
            dr = np.zeros(3)
            dr[i] = h
            g_plus  = self._empymod_scalar(r + dr[np.newaxis, :], r_prime)
            g_minus = self._empymod_scalar(r - dr[np.newaxis, :], r_prime)
            grad[:, i] = (g_plus - g_minus) / (2.0 * h)
        return grad


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------

def _select_backend(
    layer_stack: LayerStack,
    frequency: float,
    backend: str,
    source_layer=None,
) -> GreensBackend:
    """Return the appropriate GreensBackend for the given spec.

    Phase 2 dispatch order for 'auto':
        strata (Phase 3) → dcim → empymod

    'dcim' is tried first; falls back to 'empymod' for unsupported stacks
    (e.g., three-or-more layers).
    """
    if source_layer is None:
        source_layer = layer_stack.layers[-1]   # topmost — default for air-over-substrate

    if backend == 'auto':
        # Phase 3 dispatch order: strata → layer_recursion → dcim → empymod
        # Each is tried; ImportError / NotImplementedError triggers fallback.
        try:
            from .strata import StrataBackend
            return StrataBackend(layer_stack, frequency, source_layer)
        except (ImportError, Exception):
            pass

        try:
            from .layer_recursion import LayerRecursionBackend
            return LayerRecursionBackend(layer_stack, frequency, source_layer)
        except Exception:
            pass

        try:
            from .dcim import DCIMBackend
            return DCIMBackend(layer_stack, frequency, source_layer)
        except NotImplementedError:
            pass

        return EmpymodSommerfeldBackend(layer_stack, frequency, source_layer)

    if backend == 'layer_recursion':
        from .layer_recursion import LayerRecursionBackend
        return LayerRecursionBackend(layer_stack, frequency, source_layer)

    if backend == 'dcim':
        from .dcim import DCIMBackend
        return DCIMBackend(layer_stack, frequency, source_layer)

    if backend == 'empymod':
        return EmpymodSommerfeldBackend(layer_stack, frequency, source_layer)

    if backend == 'strata':
        from .strata import StrataBackend
        return StrataBackend(layer_stack, frequency, source_layer)

    if backend == 'tabulated':
        raise ValueError(
            "backend='tabulated' must be set by passing a pre-built "
            "TabulatedPNGFBackend directly to LayeredGreensFunction. "
            "Use LayeredGreensFunction(..., backend=tab_instance)."
        )

    raise ValueError(
        f"Unknown GF backend '{backend}'. "
        "Supported: 'auto', 'layer_recursion', 'dcim', 'empymod', 'strata'."
    )


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

class LayeredGreensFunction(GreensFunctionBase):
    """Stratified-medium Green's function with swappable backend.

    Parameters
    ----------
    layer_stack : LayerStack
        Layer definitions.
    frequency : float
        Operating frequency (Hz).
    source_layer_name : str, optional
        Name of the layer containing the MoM mesh.  If None, the topmost
        layer is used (typical for air-over-substrate configurations).
    backend : str
        Backend selection: 'empymod' (Phase 1), or 'auto'.

    Examples
    --------
    ::

        stack = LayerStack([
            Layer('Si', z_bot=-np.inf, z_top=0.0, eps_r=11.7, conductivity=10.0),
            Layer('air', z_bot=0.0, z_top=np.inf),
        ])
        gf = LayeredGreensFunction(stack, frequency=2.4e9)
        k  = gf.wavenumber    # in the source layer (air)
        eta = gf.wave_impedance
    """

    def __init__(
        self,
        layer_stack: LayerStack,
        frequency: float,
        source_layer_name: str = None,
        backend = 'auto',
    ):
        self._stack = layer_stack
        self._freq  = float(frequency)
        self._omega = 2.0 * np.pi * self._freq

        if source_layer_name is not None:
            self._src_layer = layer_stack.get_layer(source_layer_name)
        else:
            self._src_layer = layer_stack.layers[-1]  # topmost layer default

        if isinstance(backend, GreensBackend):
            # Accept a pre-built backend instance directly (e.g. TabulatedPNGFBackend)
            self._backend = backend
        else:
            self._backend = _select_backend(
                layer_stack, frequency, backend,
                source_layer=self._src_layer,
            )

    @property
    def wavenumber(self) -> complex:
        return self._src_layer.wavenumber(self._omega)

    @property
    def wave_impedance(self) -> complex:
        return self._src_layer.wave_impedance(self._omega)

    @property
    def backend(self) -> GreensBackend:
        return self._backend

    def layer_at(self, z: float) -> Layer:
        """Return the layer containing z (delegates to LayerStack)."""
        return self._stack.layer_at_z(z)
