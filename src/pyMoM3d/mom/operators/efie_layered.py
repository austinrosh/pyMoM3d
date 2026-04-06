"""MultilayerEFIEOperator: EFIE with a stratified-media Green's function.

Extends EFIEOperator to replace the free-space smooth-remainder quadrature
with the layered-medium smooth correction G_ML - G_fs.  The singular term
(Graglia 1993 extraction) is UNCHANGED — it always uses the free-space 1/R
formula because the singularity structure is identical in both cases.

Same-layer vs. cross-layer interactions
----------------------------------------
Same-layer  (source and test triangle in the same homogeneous layer):
    G_ML = G_fs + (G_ML - G_fs)
    Singular term: existing Graglia extraction (unchanged).
    Smooth remainder: quadrature using (G_ML - G_fs) correction.

Cross-layer (source and test triangle in different layers):
    No 1/R singularity exists.  Use full G_ML via standard quadrature.

Backend support
---------------
'cpp'   — C++ kernel calling Strata ComputeMGF directly (production path).
          Available when strata_kernels is compiled with --with-strata.
'numpy' — Python fallback using any GreensFunctionBase backend.
"""

from __future__ import annotations

import numpy as np

from ...greens.singularity import integrate_green_singular, integrate_rho_green_singular
from ...greens.base import GreensFunctionBase
from .efie import EFIEOperator

# ---------------------------------------------------------------------------
# Optional C++ multilayer backend
# ---------------------------------------------------------------------------
try:
    from ...greens.layered.strata_kernels import (
        fill_impedance_multilayer_cpp as _fill_impedance_multilayer_cpp,
    )
    _CPP_ML_AVAILABLE = True
except ImportError:
    _CPP_ML_AVAILABLE = False


class MultilayerEFIEOperator(EFIEOperator):
    """EFIE operator using a stratified-media (layered) Green's function.

    Parameters
    ----------
    greens_fn : GreensFunctionBase
        A LayeredGreensFunction (or FreeSpaceGreensFunction for testing).
        Must expose scalar_G(r, r_prime) returning the SMOOTH CORRECTION
        G_ML - G_fs for same-layer interactions and the FULL G_ML for
        cross-layer interactions.
    a_only : bool, optional
        If True, assemble only the vector-potential (A) term, dropping
        the scalar-potential (Phi) term.  Used for A-EFIE.  Default False.
    """

    def __init__(
        self,
        greens_fn: GreensFunctionBase,
        a_only: bool = False,
        sp_exclude_indices=None,
    ):
        super().__init__()
        self._greens = greens_fn
        self._a_only = a_only
        self._sp_exclude = (
            np.asarray(sorted(sp_exclude_indices), dtype=np.intp)
            if sp_exclude_indices else np.empty(0, dtype=np.intp)
        )
        # Detect StrataBackend with a valid model for C++ fast path
        self._strata_model = None
        self._strata_backend = None
        self._layer_stack = None
        if hasattr(greens_fn, 'backend'):
            backend = greens_fn.backend
            if hasattr(backend, '_model') and backend._model is not None:
                self._strata_model = backend._model
                self._strata_backend = backend
            if hasattr(backend, '_stack'):
                self._layer_stack = backend._stack
        if self._layer_stack is None and hasattr(greens_fn, '_stack'):
            self._layer_stack = greens_fn._stack

    def supports_backend(self, backend: str) -> bool:
        if backend == 'cpp':
            return _CPP_ML_AVAILABLE and self._strata_model is not None
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
        if backend == 'cpp':
            verts    = mesh.vertices.astype(np.float64, copy=False)
            tris     = mesh.triangles.astype(np.int32, copy=False)
            t_plus   = rwg_basis.t_plus.astype(np.int32, copy=False)
            t_minus  = rwg_basis.t_minus.astype(np.int32, copy=False)
            fv_plus  = rwg_basis.free_vertex_plus.astype(np.int32, copy=False)
            fv_minus = rwg_basis.free_vertex_minus.astype(np.int32, copy=False)
            a_plus   = rwg_basis.area_plus.astype(np.float64, copy=False)
            a_minus  = rwg_basis.area_minus.astype(np.float64, copy=False)
            elen     = rwg_basis.edge_length.astype(np.float64, copy=False)
            cents    = np.ascontiguousarray(tri_centroids, dtype=np.float64)
            medge    = np.ascontiguousarray(tri_mean_edge, dtype=np.float64)
            tarea    = np.ascontiguousarray(tri_twice_area, dtype=np.float64)

            # --- Cross-layer support ---
            # Determine per-triangle layer indices and build models for
            # each unique (src_layer, obs_layer) pair.
            tri_layer_idx = np.empty(0, dtype=np.int32)
            extra_models = []
            model_lookup = np.empty(0, dtype=np.int32)

            if self._layer_stack is not None and self._strata_backend is not None:
                n_tri = len(mesh.triangles)
                tri_layer_idx_raw = np.zeros(n_tri, dtype=np.int32)
                for i in range(n_tri):
                    z_c = float(cents[i, 2])
                    try:
                        tri_layer_idx_raw[i] = self._layer_stack.layer_index_at_z(z_c)
                    except ValueError:
                        tri_layer_idx_raw[i] = -1

                unique_layers = np.unique(tri_layer_idx_raw[tri_layer_idx_raw >= 0])

                if len(unique_layers) > 1:
                    # Remap layer indices to 0..M-1
                    max_layer = int(unique_layers.max())
                    remap = np.full(max_layer + 1, -1, dtype=np.int32)
                    for new_idx, old_idx in enumerate(unique_layers):
                        remap[old_idx] = new_idx
                    M = len(unique_layers)

                    tri_layer_idx = np.zeros(n_tri, dtype=np.int32)
                    for i in range(n_tri):
                        raw = tri_layer_idx_raw[i]
                        tri_layer_idx[i] = remap[raw] if raw >= 0 else 0

                    # Build model lookup: model_lookup[obs_remap * M + src_remap]
                    # Model 0 = the existing self._strata_model (same-layer for
                    # the primary source layer)
                    layers = self._layer_stack.layers
                    freq = self._strata_backend._freq

                    # Identify which remapped index corresponds to the primary
                    # same-layer model
                    primary_layer = self._strata_backend._src_layer
                    primary_idx_raw = next(
                        i for i, l in enumerate(layers) if l.name == primary_layer.name
                    )
                    primary_remap = int(remap[primary_idx_raw])

                    model_lookup = np.full(M * M, -1, dtype=np.int32)
                    extra_models = []

                    for obs_ri in range(M):
                        for src_ri in range(M):
                            obs_layer_idx = int(unique_layers[obs_ri])
                            src_layer_idx = int(unique_layers[src_ri])

                            # Check if this is the primary same-layer pair
                            if (obs_ri == primary_remap and src_ri == primary_remap):
                                model_lookup[obs_ri * M + src_ri] = 0
                            else:
                                src_layer = layers[src_layer_idx]
                                obs_layer = layers[obs_layer_idx]
                                m = self._strata_backend._build_model_for_pair(
                                    self._layer_stack, freq, src_layer, obs_layer,
                                )
                                extra_models.append(m)
                                model_lookup[obs_ri * M + src_ri] = len(extra_models)
                                # extra_models are 1-indexed (0 = default model)

            # C++ kernel uses real k for free-space singular extraction
            # (Graglia 1993) and real eta for EFIE prefactors.  For the
            # phantom source layer k and eta are nearly real; any small
            # imaginary part from conductivity is handled by the Strata
            # smooth correction.
            k_real   = float(k.real)   if hasattr(k, 'real')   else float(k)
            eta_real = float(eta.real) if hasattr(eta, 'real') else float(eta)

            _fill_impedance_multilayer_cpp(
                Z, verts, tris, t_plus, t_minus, fv_plus, fv_minus,
                a_plus, a_minus, elen, cents, medge, tarea,
                weights, bary,
                k_real, eta_real, float(near_threshold), int(quad_order),
                self._strata_model,
                0,  # num_threads: 0 = OMP default
                tri_layer_idx,
                extra_models,
                model_lookup,
                self._a_only,
            )

            # SP charge exclusion for half-RWG port basis functions
            # (Liu et al. 2018): half-RWG divergence creates spurious
            # line charges at the gap edges.  Remove SP only for
            # port-to-port interactions (both m AND n are port basis)
            # to eliminate gap capacitance while preserving physical
            # charge coupling to the rest of the structure.
            if len(self._sp_exclude) > 0 and not self._a_only:
                N = Z.shape[0]
                Z_vp = np.zeros((N, N), dtype=np.complex128)
                _fill_impedance_multilayer_cpp(
                    Z_vp, verts, tris, t_plus, t_minus, fv_plus, fv_minus,
                    a_plus, a_minus, elen, cents, medge, tarea,
                    weights, bary,
                    k_real, eta_real, float(near_threshold), int(quad_order),
                    self._strata_model,
                    0,
                    tri_layer_idx,
                    extra_models,
                    model_lookup,
                    True,  # a_only=True → VP-only
                )
                idx = self._sp_exclude
                Z[np.ix_(idx, idx)] = Z_vp[np.ix_(idx, idx)]
        else:
            raise ValueError(
                f"MultilayerEFIEOperator.fill_fast: unknown backend '{backend}'"
            )

    def compute_pair_numpy(
        self,
        k, eta, mesh,
        tri_test, tri_src,
        fv_test, fv_src,
        sign_test, sign_src,
        l_test, l_src,
        A_test, A_src,
        quad_order, near_threshold,
        weights, bary,
        twice_area_test, twice_area_src,
        is_near,
        n_hat_test,
    ) -> complex:
        """Compute EFIE contribution from one triangle pair using multilayer GF.

        For same-layer pairs the singular term is computed exactly as in the
        free-space EFIEOperator, and the smooth remainder uses G_ML - G_fs.
        For cross-layer pairs the full G_ML is used throughout.
        """
        verts_test = mesh.vertices[mesh.triangles[tri_test]]
        verts_src  = mesh.vertices[mesh.triangles[tri_src]]
        r_fv_test  = mesh.vertices[fv_test]
        r_fv_src   = mesh.vertices[fv_src]

        # Determine same-layer or cross-layer
        z_test = float(np.mean(verts_test[:, 2]))
        z_src  = float(np.mean(verts_src[:, 2]))
        gf = self._greens

        same_layer = True
        if hasattr(gf, 'layer_at'):
            try:
                same_layer = (gf.layer_at(z_test).name == gf.layer_at(z_src).name)
            except Exception:
                same_layer = True  # conservative: use same-layer path

        # ------------------------------------------------------------------
        # Build quadrature point arrays (all Q points at once for batched GF)
        # ------------------------------------------------------------------
        Q = len(weights)

        # Test quadrature points and rho_test vectors: (Q, 3)
        r_obs_all  = (bary[:, 0:1] * verts_test[0]
                    + bary[:, 1:2] * verts_test[1]
                    + bary[:, 2:3] * verts_test[2])   # (Q, 3)
        rho_test_all = r_obs_all - r_fv_test[np.newaxis, :]   # (Q, 3)

        # Source quadrature points: (Q, 3)
        r_src_all = (bary[:, 0:1] * verts_src[0]
                   + bary[:, 1:2] * verts_src[1]
                   + bary[:, 2:3] * verts_src[2])     # (Q, 3)
        rho_src_all = r_src_all - r_fv_src[np.newaxis, :]   # (Q, 3)

        # ------------------------------------------------------------------
        # Scalar potential integral I_Phi
        # ------------------------------------------------------------------
        I_Phi_raw = 0.0 + 0.0j

        for i in range(Q):
            r_obs_i = r_obs_all[i]

            if same_layer and is_near:
                # Singular term (Graglia) — free-space, unchanged
                g_singular = integrate_green_singular(
                    k, r_obs_i,
                    verts_src[0], verts_src[1], verts_src[2],
                    quad_order=quad_order, near_threshold=near_threshold,
                )
                # Smooth correction: G_ML - G_fs integrated over source triangle
                # Guard against catastrophic cancellation at R ≈ 0:
                # both G_ML and G_fs diverge as 1/R, so their pointwise
                # difference is inaccurate for R < threshold.  Interpolate
                # the correction at singular points from regular ones
                # (matches C++ kernel's R_SMOOTH_THRESH logic).
                R_SMOOTH_THRESH = 1e-10
                r_obs_tiled = np.tile(r_obs_i, (Q, 1))  # (Q, 3)
                R_src = np.linalg.norm(r_obs_tiled - r_src_all, axis=-1)
                regular = R_src >= R_SMOOTH_THRESH
                if np.all(regular):
                    g_correction = gf.scalar_G(r_obs_tiled, r_src_all)
                    g_smooth = np.dot(weights, g_correction) * twice_area_src
                elif np.any(regular):
                    g_correction = gf.scalar_G(
                        r_obs_tiled[regular], r_src_all[regular])
                    w_reg = weights[regular]
                    w_sing = weights[~regular]
                    g_corr_avg = np.dot(w_reg, g_correction) / w_reg.sum()
                    g_smooth = (np.dot(w_reg, g_correction)
                                + w_sing.sum() * g_corr_avg) * twice_area_src
                else:
                    g_smooth = 0.0 + 0.0j
                g_int = g_singular + g_smooth

            elif same_layer:
                # Far field: free-space quadrature + smooth correction
                # Free-space term:
                diff = r_obs_i[np.newaxis, :] - r_src_all   # (Q, 3)
                R = np.linalg.norm(diff, axis=-1)            # (Q,)
                R = np.maximum(R, 1e-30)
                g_fs_vals = np.exp(-1j * k * R) / (4.0 * np.pi * R)   # (Q,)
                g_fs_int = np.dot(weights, g_fs_vals) * twice_area_src

                # Smooth correction:
                r_obs_tiled = np.tile(r_obs_i, (Q, 1))
                g_corr_vals = gf.scalar_G(r_obs_tiled, r_src_all)   # (Q,)
                g_smooth = np.dot(weights, g_corr_vals) * twice_area_src
                g_int = g_fs_int + g_smooth

            else:
                # Cross-layer: full G_ML, no singularity extraction.
                # Backend returns smooth correction (G_ML - G_fs), so we
                # must add G_fs back to get the full Green's function.
                r_obs_tiled = np.tile(r_obs_i, (Q, 1))
                diff = r_obs_i[np.newaxis, :] - r_src_all
                R = np.linalg.norm(diff, axis=-1)
                R = np.maximum(R, 1e-30)
                g_fs_vals = np.exp(-1j * k * R) / (4.0 * np.pi * R)
                g_corr_vals = gf.scalar_G(r_obs_tiled, r_src_all)
                g_ml_vals = g_fs_vals + g_corr_vals
                g_int = np.dot(weights, g_ml_vals) * twice_area_src

            I_Phi_raw += weights[i] * g_int

        I_Phi_raw *= twice_area_test

        # ------------------------------------------------------------------
        # Vector potential integral I_A  (uses dyadic G_A, NOT scalar g)
        #
        # In layered media the vector potential GF G_A is a dyadic (3×3).
        # In free space G_A = g_fs · I, so the singular extraction
        # (Graglia) still uses scalar g_fs × rho_src.  The smooth
        # correction uses (G_A - g_fs·I) · rho_src (dyadic product).
        # ------------------------------------------------------------------
        I_A_raw = 0.0 + 0.0j

        # Check if dyadic_G is available on the backend
        _has_dyadic = hasattr(gf, 'dyadic_G')

        for i in range(Q):
            r_obs_i    = r_obs_all[i]
            rho_test_i = rho_test_all[i]

            if same_layer and is_near:
                # Singular rho-weighted term (Graglia) — free-space g_fs × I
                rho_src_g = integrate_rho_green_singular(
                    k, r_obs_i,
                    verts_src[0], verts_src[1], verts_src[2],
                    r_fv_src, quad_order=quad_order, near_threshold=near_threshold,
                )
                # Smooth correction: (G_A - g_fs·I) · rho_src
                # Same R ≈ 0 guard as scalar potential path.
                R_SMOOTH_THRESH = 1e-10
                r_obs_tiled = np.tile(r_obs_i, (Q, 1))
                R_src = np.linalg.norm(r_obs_tiled - r_src_all, axis=-1)
                regular = R_src >= R_SMOOTH_THRESH
                if _has_dyadic:
                    if np.all(regular):
                        ga_corr = gf.dyadic_G(r_obs_tiled, r_src_all)
                        ga_dot_rho = np.einsum('jik,jk->ji', ga_corr, rho_src_all)
                        rho_src_g_smooth = np.einsum('j,ji->i', weights, ga_dot_rho) * twice_area_src
                    elif np.any(regular):
                        ga_corr = gf.dyadic_G(
                            r_obs_tiled[regular], r_src_all[regular])
                        ga_dot_rho = np.einsum('jik,jk->ji', ga_corr, rho_src_all[regular])
                        w_reg = weights[regular]
                        w_sing = weights[~regular]
                        # Weighted average for interpolation at singular points
                        ga_dot_rho_avg = np.einsum('j,ji->i', w_reg, ga_dot_rho) / w_reg.sum()
                        rho_src_g_smooth = (np.einsum('j,ji->i', w_reg, ga_dot_rho)
                                            + w_sing.sum() * ga_dot_rho_avg) * twice_area_src
                    else:
                        rho_src_g_smooth = np.zeros(3, dtype=np.complex128)
                else:
                    if np.all(regular):
                        g_corr_vals = gf.scalar_G(r_obs_tiled, r_src_all)
                        rho_src_g_smooth = np.einsum('j,ji,j->i', weights, rho_src_all, g_corr_vals) * twice_area_src
                    elif np.any(regular):
                        g_corr_vals = gf.scalar_G(
                            r_obs_tiled[regular], r_src_all[regular])
                        w_reg = weights[regular]
                        w_sing = weights[~regular]
                        g_corr_avg = np.dot(w_reg, g_corr_vals) / w_reg.sum()
                        rho_src_g_smooth = (np.einsum('j,ji,j->i', w_reg, rho_src_all[regular], g_corr_vals)
                                            + w_sing.sum() * g_corr_avg * rho_src_all[~regular].sum(axis=0)) * twice_area_src
                    else:
                        rho_src_g_smooth = np.zeros(3, dtype=np.complex128)
                rho_src_g_int = rho_src_g + rho_src_g_smooth

            elif same_layer:
                # Free-space rho-g term: g_fs · I · rho_src = g_fs * rho_src
                diff = r_obs_i[np.newaxis, :] - r_src_all   # (Q, 3)
                R = np.linalg.norm(diff, axis=-1)
                R = np.maximum(R, 1e-30)
                g_fs_vals = np.exp(-1j * k * R) / (4.0 * np.pi * R)   # (Q,)
                rho_src_g_fs = np.einsum('j,ji,j->i', weights, rho_src_all, g_fs_vals) * twice_area_src

                # Smooth correction: (G_A - g_fs·I) · rho_src
                r_obs_tiled = np.tile(r_obs_i, (Q, 1))
                if _has_dyadic:
                    ga_corr = gf.dyadic_G(r_obs_tiled, r_src_all)
                    ga_dot_rho = np.einsum('jik,jk->ji', ga_corr, rho_src_all)
                    rho_src_g_corr = np.einsum('j,ji->i', weights, ga_dot_rho) * twice_area_src
                else:
                    g_corr_vals = gf.scalar_G(r_obs_tiled, r_src_all)
                    rho_src_g_corr = np.einsum('j,ji,j->i', weights, rho_src_all, g_corr_vals) * twice_area_src
                rho_src_g_int = rho_src_g_fs + rho_src_g_corr

            else:
                # Cross-layer: full G_A (no singularity extraction).
                # Backend returns smooth correction, so add G_fs back.
                r_obs_tiled = np.tile(r_obs_i, (Q, 1))
                diff = r_obs_i[np.newaxis, :] - r_src_all
                R = np.linalg.norm(diff, axis=-1)
                R = np.maximum(R, 1e-30)
                g_fs_vals = np.exp(-1j * k * R) / (4.0 * np.pi * R)
                if _has_dyadic:
                    # Dyadic correction + free-space isotropic part
                    ga_corr = gf.dyadic_G(r_obs_tiled, r_src_all)
                    # Full dyadic = correction + g_fs * I
                    eye3 = np.eye(3)[np.newaxis, :, :]  # (1, 3, 3)
                    ga_full = ga_corr + g_fs_vals[:, np.newaxis, np.newaxis] * eye3
                    ga_dot_rho = np.einsum('jik,jk->ji', ga_full, rho_src_all)
                    rho_src_g_int = np.einsum('j,ji->i', weights, ga_dot_rho) * twice_area_src
                else:
                    g_corr_vals = gf.scalar_G(r_obs_tiled, r_src_all)
                    g_ml_vals = g_fs_vals + g_corr_vals
                    rho_src_g_int = np.einsum('j,ji,j->i', weights, rho_src_all, g_ml_vals) * twice_area_src

            I_A_raw += weights[i] * np.dot(rho_test_i, rho_src_g_int)

        I_A_raw *= twice_area_test

        # ------------------------------------------------------------------
        # Assemble Z contribution
        # ------------------------------------------------------------------
        scale_A   = (sign_test * l_test / (2.0 * A_test)) * (sign_src * l_src / (2.0 * A_src))
        prefactor_A   = 1j * k * eta

        if self._a_only:
            return prefactor_A * I_A_raw * scale_A

        scale_Phi = (sign_test * l_test / A_test)          * (sign_src * l_src / A_src)
        prefactor_Phi = -1j * eta / k

        return prefactor_A * I_A_raw * scale_A + prefactor_Phi * I_Phi_raw * scale_Phi
