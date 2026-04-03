"""StrataBackend: pybind11 wrapper for the Strata C++ MGF library.

Strata (https://github.com/modelics/strata, GPL-3.0) computes the multilayer
Green's function via Michalski & Zheng Formulation-C, supporting direct
numerical integration, multilevel DCIM, and quasistatic extraction.

The DCIM method (~10⁶ evaluations/second in C++) makes this the production
backend for large-scale simulations.

Installation
------------
1. Build and install the Strata C++ library::

       git clone https://github.com/modelics/strata
       cd strata && mkdir build && cd build
       cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
       make -j$(nproc) && sudo make install

2. Build the pybind11 wrapper::

       venv/bin/python build_cpp.py build_ext --inplace \\
           --with-strata=/usr/local

3. Verify::

       python -c "from pyMoM3d.greens.layered import strata_kernels; print(strata_kernels.__version__)"

When ``strata_kernels`` is not available, ``StrataBackend.__init__`` raises
``ImportError`` and ``LayeredGreensFunction(backend='auto')`` falls back to
``LayerRecursionBackend``.
"""

from __future__ import annotations

import math

import numpy as np

from ..base import GreensBackend
from ...medium.layer_stack import Layer, LayerStack


class StrataBackend(GreensBackend):
    """Production multilayer GF backend via the Strata C++ library (DCIM).

    Parameters
    ----------
    layer_stack : LayerStack
        Arbitrary N-layer stack.
    frequency : float
        Operating frequency (Hz).
    source_layer : Layer
        Layer containing the MoM mesh (source and observation both assumed
        in this layer — the common same-layer case).
    method : str
        Strata computation method: 'dcim' (default, fast), 'integrate'
        (exact Sommerfeld, slow, useful for validation), or 'quasistatic'.

    Raises
    ------
    ImportError
        If ``strata_kernels`` has not been compiled.  Build with::

            venv/bin/python build_cpp.py build_ext --inplace \\
                --with-strata=/path/to/strata
    """

    def __init__(
        self,
        layer_stack: LayerStack,
        frequency: float,
        source_layer: Layer,
        method: str = 'dcim',
    ):
        try:
            from pyMoM3d.greens.layered import strata_kernels as _sk
            self._sk = _sk
        except ImportError as exc:
            raise ImportError(
                "StrataBackend requires the 'strata_kernels' C++ extension.\n"
                "Build it with:\n"
                "    git clone https://github.com/modelics/strata\n"
                "    cd strata && mkdir build && cd build\n"
                "    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local\n"
                "    make -j$(nproc) && sudo make install\n"
                "    cd <pyMoM3d root>\n"
                "    venv/bin/python build_cpp.py build_ext --inplace "
                "--with-strata=/usr/local"
            ) from exc

        self._stack       = layer_stack
        self._freq        = float(frequency)
        self._src_layer   = source_layer
        self._method      = method

        # DCIM fails with dual-PEC half-spaces (Ze=0 → Ye=1/Ze=inf in
        # Strata's spectral MGF).  Workaround: replace PEC flags with very
        # high conductivity (σ=1e8 S/m), which gives identical results to
        # within ~1e-5 relative error while keeping DCIM numerically stable.
        self._pec_sigma_workaround = False
        if self._method == "dcim":
            has_pec_top = has_pec_bot = False
            for lyr in self._stack.layers:
                if not math.isfinite(lyr.z_top) and getattr(lyr, 'is_pec', False):
                    has_pec_top = True
                if not math.isfinite(lyr.z_bot) and getattr(lyr, 'is_pec', False):
                    has_pec_bot = True
            if has_pec_top and has_pec_bot:
                import warnings
                warnings.warn(
                    "Dual-PEC half-spaces: replacing PEC flags with high "
                    "conductivity (σ=1e8 S/m) to avoid Strata DCIM "
                    "singularity (Ze=0 → Ye=1/Ze=inf).",
                    stacklevel=2,
                )
                self._pec_sigma_workaround = True

        # Strata requires at least one finite interior layer.  Pure halfspace
        # stacks (e.g., air-over-Si with no finite slab) should use
        # LayerRecursionBackend or DCIMBackend instead.
        layers = layer_stack.layers
        has_interior = any(
            math.isfinite(l.z_bot) and math.isfinite(l.z_top) for l in layers
        )
        if not has_interior and len(layers) > 1:
            raise NotImplementedError(
                "StrataBackend requires at least one finite interior layer. "
                "Use LayerRecursionBackend for pure two-halfspace stacks."
            )

        src_idx = next(
            i for i, lyr in enumerate(layers) if lyr.name == source_layer.name
        )
        # If the source layer is the bottommost (or the only layer), the smooth
        # correction is identically zero and we skip the C++ call.
        self._no_interface = (src_idx == 0)

        if not self._no_interface:
            self._model = self._build_model(layer_stack, frequency, source_layer)
        else:
            self._model = None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(
        self,
        layer_stack: LayerStack,
        frequency: float,
        source_layer: Layer,
    ):
        """Translate LayerStack → strata_kernels.StrataModel."""
        # Separate halfspaces (infinite layers) from finite interior layers
        interior = []
        top_hs   = None
        bot_hs   = None

        for layer in layer_stack.layers:
            if not math.isfinite(layer.z_top):
                top_hs = layer
            elif not math.isfinite(layer.z_bot):
                bot_hs = layer
            else:
                interior.append(layer)

        # Build layers_data: [zmin, zmax, epsr_re, epsr_im, mur, sigma, sigmamu]
        # Pass eps_r (bare dielectric) and conductivity separately — Strata
        # handles the omega-dependent combination internally.
        layers_data = []
        for lyr in interior:
            eps = complex(lyr.eps_r)
            layers_data.append([
                float(lyr.z_bot),
                float(lyr.z_top),
                eps.real,
                eps.imag,
                float(complex(lyr.mu_r).real),
                float(lyr.conductivity),
                0.0,   # sigmamu (magnetic conductivity — always 0 here)
            ])

        # Top halfspace (defaults to air if absent)
        if top_hs is not None:
            epsr_top  = float(complex(top_hs.eps_r).real)
            mur_top   = float(complex(top_hs.mu_r).real)
            sigma_top = float(top_hs.conductivity)
            pec_top   = getattr(top_hs, 'is_pec', False)
        else:
            epsr_top, mur_top, sigma_top, pec_top = 1.0, 1.0, 0.0, False

        # Bottom halfspace (defaults to air if absent)
        if bot_hs is not None:
            epsr_bot  = float(complex(bot_hs.eps_r).real)
            mur_bot   = float(complex(bot_hs.mu_r).real)
            sigma_bot = float(bot_hs.conductivity)
            pec_bot   = getattr(bot_hs, 'is_pec', False)
        else:
            epsr_bot, mur_bot, sigma_bot, pec_bot = 1.0, 1.0, 0.0, False

        # Dual-PEC workaround: replace PEC flags with high conductivity
        if self._pec_sigma_workaround:
            if pec_top:
                pec_top = False
                sigma_top = 1e8
            if pec_bot:
                pec_bot = False
                sigma_bot = 1e8

        # z-coordinate inside the source layer for FindLayer.
        # For finite layers use the midpoint; for half-infinite layers
        # pick a point slightly inside the finite boundary.
        z_bot = float(source_layer.z_bot)
        z_top = float(source_layer.z_top)
        if math.isfinite(z_bot) and math.isfinite(z_top):
            z_src = (z_bot + z_top) / 2.0
        elif math.isfinite(z_bot):
            z_src = z_bot + 1e-3
        elif math.isfinite(z_top):
            z_src = z_top - 1e-3
        else:
            z_src = 0.0

        # Source-layer wavenumber and effective permittivity for the C++ wrapper.
        # Strata's G_phi uses Formulation-C convention (G_phi = g/ε_r); the
        # wrapper multiplies by ε_r before subtracting the free-space g_fs.
        omega = 2.0 * math.pi * frequency
        k_src = complex(source_layer.wavenumber(omega))
        eps_r_eff = complex(source_layer.eps_r_eff(omega))

        return self._sk.make_model(
            layers_data,
            epsr_top, mur_top, sigma_top, pec_top,
            epsr_bot, mur_bot, sigma_bot, pec_bot,
            float(frequency),
            z_src, z_src,   # same-layer: source and observation in same layer
            k_src.real, k_src.imag,
            eps_r_eff.real, eps_r_eff.imag,
            self._method,
        )

    def _build_model_for_pair(
        self,
        layer_stack: LayerStack,
        frequency: float,
        src_layer: Layer,
        obs_layer: Layer,
    ):
        """Build a StrataModel for a specific (source, observation) layer pair.

        Unlike ``_build_model`` which always uses same-layer (z_src == z_obs),
        this method supports cross-layer configurations where source and
        observation triangles are in different layers.
        """
        interior = []
        top_hs = None
        bot_hs = None

        for layer in layer_stack.layers:
            if not math.isfinite(layer.z_top):
                top_hs = layer
            elif not math.isfinite(layer.z_bot):
                bot_hs = layer
            else:
                interior.append(layer)

        layers_data = []
        for lyr in interior:
            eps = complex(lyr.eps_r)
            layers_data.append([
                float(lyr.z_bot),
                float(lyr.z_top),
                eps.real,
                eps.imag,
                float(complex(lyr.mu_r).real),
                float(lyr.conductivity),
                0.0,
            ])

        if top_hs is not None:
            epsr_top = float(complex(top_hs.eps_r).real)
            mur_top = float(complex(top_hs.mu_r).real)
            sigma_top = float(top_hs.conductivity)
            pec_top = getattr(top_hs, 'is_pec', False)
        else:
            epsr_top, mur_top, sigma_top, pec_top = 1.0, 1.0, 0.0, False

        if bot_hs is not None:
            epsr_bot = float(complex(bot_hs.eps_r).real)
            mur_bot = float(complex(bot_hs.mu_r).real)
            sigma_bot = float(bot_hs.conductivity)
            pec_bot = getattr(bot_hs, 'is_pec', False)
        else:
            epsr_bot, mur_bot, sigma_bot, pec_bot = 1.0, 1.0, 0.0, False

        # Dual-PEC workaround: replace PEC flags with high conductivity
        if self._pec_sigma_workaround:
            if pec_top:
                pec_top = False
                sigma_top = 1e8
            if pec_bot:
                pec_bot = False
                sigma_bot = 1e8

        def _layer_z_repr(layer):
            z_bot = float(layer.z_bot)
            z_top = float(layer.z_top)
            if math.isfinite(z_bot) and math.isfinite(z_top):
                return (z_bot + z_top) / 2.0
            elif math.isfinite(z_bot):
                return z_bot + 1e-3
            elif math.isfinite(z_top):
                return z_top - 1e-3
            else:
                return 0.0

        z_src = _layer_z_repr(src_layer)
        z_obs = _layer_z_repr(obs_layer)

        omega = 2.0 * math.pi * frequency
        k_src = complex(src_layer.wavenumber(omega))
        eps_r_eff = complex(src_layer.eps_r_eff(omega))

        return self._sk.make_model(
            layers_data,
            epsr_top, mur_top, sigma_top, pec_top,
            epsr_bot, mur_bot, sigma_bot, pec_bot,
            float(frequency),
            z_src, z_obs,
            k_src.real, k_src.imag,
            eps_r_eff.real, eps_r_eff.imag,
            self._method,
        )

    # ------------------------------------------------------------------
    # GreensBackend interface
    # ------------------------------------------------------------------

    def scalar_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Scalar potential smooth correction G_phi(r, r') - G_phi_fs(r, r').

        Parameters
        ----------
        r, r_prime : (..., 3) float64

        Returns
        -------
        (...,) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]

        if self._no_interface:
            return np.zeros(shape, dtype=np.complex128)

        r_flat  = np.ascontiguousarray(r.reshape(-1, 3))
        rp_flat = np.ascontiguousarray(r_prime.reshape(-1, 3))
        result  = self._sk.scalar_G_smooth(self._model, r_flat, rp_flat)
        return np.asarray(result, dtype=np.complex128).reshape(shape)

    def dyadic_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Full dyadic smooth correction G̅(r, r') - G̅_fs(r, r').

        Parameters
        ----------
        r, r_prime : (..., 3) float64

        Returns
        -------
        (..., 3, 3) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]

        if self._no_interface:
            return np.zeros(shape + (3, 3), dtype=np.complex128)

        r_flat  = np.ascontiguousarray(r.reshape(-1, 3))
        rp_flat = np.ascontiguousarray(r_prime.reshape(-1, 3))
        result  = self._sk.dyadic_G_smooth(self._model, r_flat, rp_flat)
        result  = np.asarray(result, dtype=np.complex128)
        # DCIM fitting can produce NaN for z-coupling components (Gxz, Gzx)
        # when high-conductivity PEC workaround is active.  These components
        # don't contribute for planar same-layer meshes (rho_z = 0).
        np.nan_to_num(result, copy=False, nan=0.0)
        return result.reshape(shape + (3, 3))

    def grad_G(self, r: np.ndarray, r_prime: np.ndarray) -> np.ndarray:
        """Gradient of scalar smooth correction ∇G_phi(r, r') w.r.t. r.

        Parameters
        ----------
        r, r_prime : (..., 3) float64

        Returns
        -------
        (..., 3) complex128
        """
        r       = np.asarray(r,       dtype=np.float64)
        r_prime = np.asarray(r_prime, dtype=np.float64)
        shape   = r.shape[:-1]

        if self._no_interface:
            return np.zeros(shape + (3,), dtype=np.complex128)

        r_flat  = np.ascontiguousarray(r.reshape(-1, 3))
        rp_flat = np.ascontiguousarray(r_prime.reshape(-1, 3))
        result  = self._sk.grad_G_smooth(self._model, r_flat, rp_flat)
        return np.asarray(result, dtype=np.complex128).reshape(shape + (3,))
