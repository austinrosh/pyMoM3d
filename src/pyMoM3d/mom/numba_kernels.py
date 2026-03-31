"""Numba JIT-compiled kernels for EFIE impedance matrix fill.

All functions are compiled with nopython=True (no Python fallback, maximum
performance).  The top-level ``fill_Z_numba`` function uses ``prange`` for
parallel execution across rows of Z via OpenMP.

Graceful fallback: if numba is not installed, this module sets
``NUMBA_AVAILABLE = False`` and provides no JIT functions.  The caller in
``impedance.py`` checks this flag before dispatching.

Design constraints (GPU-readiness)
-----------------------------------
- All inputs are flat contiguous NumPy arrays (SOA layout).
- No Python objects inside any JIT function.
- The quadrature rule (weights, bary) is passed as pre-allocated arrays.
- The output Z matrix is written in-place via a pre-allocated buffer.
- 2-argument min() is used (Numba nopython-safe) instead of Python min().
"""

import numpy as np

try:
    import numba
    from numba import prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    # ------------------------------------------------------------------ #
    #  Analytical 1/R integral  (Graglia 1993 / Wilton 1984)              #
    # ------------------------------------------------------------------ #

    @numba.jit(nopython=True, cache=True)
    def _nb_analytical_1_over_R(r_obs, v0, v1, v2):
        """Analytically integrate 1/|r_obs - r'| over a triangle.

        Port of ``_analytical_1_over_R_triangle`` in greens/singularity.py.
        Fully Numba nopython-compatible: no Python lists, no np.asarray.
        """
        # Stack vertices into indexable array
        verts = np.empty((3, 3), dtype=np.float64)
        verts[0, 0] = v0[0]; verts[0, 1] = v0[1]; verts[0, 2] = v0[2]
        verts[1, 0] = v1[0]; verts[1, 1] = v1[1]; verts[1, 2] = v1[2]
        verts[2, 0] = v2[0]; verts[2, 1] = v2[1]; verts[2, 2] = v2[2]

        # Triangle normal via cross product (v1-v0) x (v2-v0)
        n_vec = np.cross(v1 - v0, v2 - v0)
        area2 = np.linalg.norm(n_vec)
        if area2 < 1e-30:
            return 0.0
        n_hat = n_vec / area2

        # Signed height of r_obs above the triangle plane
        d = np.dot(r_obs - v0, n_hat)

        # Projection of r_obs onto triangle plane
        r_proj = r_obs - d * n_hat

        # Distance from r_obs to each vertex
        R_arr = np.empty(3, dtype=np.float64)
        for ii in range(3):
            R_arr[ii] = np.linalg.norm(verts[ii] - r_obs)

        result = 0.0
        abs_d = abs(d)

        for i in range(3):
            j = (i + 1) % 3
            va = verts[i]
            vb = verts[j]

            edge_vec = vb - va
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-30:
                continue
            t_hat = edge_vec / edge_len

            # Outward edge normal in the triangle plane: -(n_hat x t_hat)
            m_hat = -np.cross(n_hat, t_hat)

            # Verify it points outward (away from opposite vertex)
            opp = verts[(i + 2) % 3]
            mid = (va + vb) * 0.5
            if np.dot(m_hat, mid - opp) < 0.0:
                m_hat = -m_hat

            rho_a = va - r_proj
            rho_b = vb - r_proj

            rho_0 = np.dot(rho_a, m_hat)
            t_a = np.dot(rho_a, t_hat)
            t_b = np.dot(rho_b, t_hat)

            R_a = R_arr[i]
            R_b = R_arr[j]

            arg_num = t_b + R_b
            arg_den = t_a + R_a
            if abs(arg_den) > 1e-30 and arg_num / arg_den > 0.0:
                ln_term = np.log(arg_num / arg_den)
            else:
                ln_term = 0.0

            if abs_d > 1e-14:
                P0_sq = rho_0 * rho_0 + d * d
                atan_b = np.arctan2(rho_0 * t_b, P0_sq + abs_d * R_b)
                atan_a = np.arctan2(rho_0 * t_a, P0_sq + abs_d * R_a)
                atan_term = atan_b - atan_a
            else:
                atan_term = 0.0

            result += rho_0 * ln_term - abs_d * atan_term

        return result

    # ------------------------------------------------------------------ #
    #  Scalar Green's function integral with singularity extraction        #
    # ------------------------------------------------------------------ #

    @numba.jit(nopython=True, cache=True)
    def _nb_green_singular(k, r_obs, v0, v1, v2, weights, bary, twice_area,
                           near_threshold):
        """Integrate scalar g(r_obs, r') over a source triangle.

        Port of ``integrate_green_singular`` in greens/singularity.py.
        Uses internal near/far check based on vertex distances (finer than the
        centroid-based outer check in fill_Z_numba).
        """
        # Minimum distance from r_obs to any source vertex
        d0 = np.linalg.norm(r_obs - v0)
        d1 = np.linalg.norm(r_obs - v1)
        d2 = np.linalg.norm(r_obs - v2)
        dist = d0 if d0 < d1 else d1
        if d2 < dist:
            dist = d2

        e01 = np.linalg.norm(v1 - v0)
        e12 = np.linalg.norm(v2 - v1)
        e20 = np.linalg.norm(v0 - v2)
        mean_edge = (e01 + e12 + e20) / 3.0

        nq = weights.shape[0]
        four_pi = 4.0 * np.pi

        if dist > near_threshold * mean_edge * 3.0 and mean_edge > 1e-30:
            # Far: plain Gauss quadrature
            result = 0.0 + 0.0j
            for i in range(nq):
                r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
                R = np.linalg.norm(r_obs - r_prime)
                if R < 1e-30:
                    R = 1e-30
                result += weights[i] * np.exp(-1j * k * R) / (four_pi * R)
            return result * twice_area

        # Near/self: singularity extraction  g = 1/(4piR) + [g - 1/(4piR)]
        I_static = _nb_analytical_1_over_R(r_obs, v0, v1, v2) / four_pi

        I_smooth = 0.0 + 0.0j
        for i in range(nq):
            r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
            R = np.linalg.norm(r_obs - r_prime)
            if R < 1e-30:
                # limit as R->0: exp(-jkR)/(4piR) - 1/(4piR) -> -jk/(4pi)
                remainder = -1j * k / four_pi
            else:
                remainder = (np.exp(-1j * k * R) - 1.0) / (four_pi * R)
            I_smooth += weights[i] * remainder

        return I_static + I_smooth * twice_area

    # ------------------------------------------------------------------ #
    #  Vector rho*g integral with singularity extraction                   #
    # ------------------------------------------------------------------ #

    @numba.jit(nopython=True, cache=True)
    def _nb_rho_green_singular(k, r_obs, v0, v1, v2, r_fv_src, weights, bary,
                               twice_area, near_threshold):
        """Integrate (r' - r_fv_src) * g(r_obs, r') over source triangle.

        Port of ``integrate_rho_green_singular`` in greens/singularity.py.
        """
        d0 = np.linalg.norm(r_obs - v0)
        d1 = np.linalg.norm(r_obs - v1)
        d2 = np.linalg.norm(r_obs - v2)
        dist = d0 if d0 < d1 else d1
        if d2 < dist:
            dist = d2

        e01 = np.linalg.norm(v1 - v0)
        e12 = np.linalg.norm(v2 - v1)
        e20 = np.linalg.norm(v0 - v2)
        mean_edge = (e01 + e12 + e20) / 3.0

        nq = weights.shape[0]
        four_pi = 4.0 * np.pi

        if dist > near_threshold * mean_edge * 3.0 and mean_edge > 1e-30:
            # Far: plain Gauss quadrature
            result = np.zeros(3, dtype=np.complex128)
            for i in range(nq):
                r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
                rho = r_prime - r_fv_src
                R = np.linalg.norm(r_obs - r_prime)
                if R < 1e-30:
                    R = 1e-30
                g_val = np.exp(-1j * k * R) / (four_pi * R)
                result += weights[i] * rho * g_val
            return result * twice_area

        # Near/self: singularity extraction
        # (1) Analytical: integral 1/R dS'
        I_1_over_R = _nb_analytical_1_over_R(r_obs, v0, v1, v2)

        # (2) Quadrature for (r'-r_obs)/R — bounded integrand
        I_Rhat = np.zeros(3, dtype=np.float64)
        for i in range(nq):
            r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
            diff = r_prime - r_obs
            R = np.linalg.norm(diff)
            if R > 1e-30:
                I_Rhat += weights[i] * diff / R
        I_Rhat = I_Rhat * twice_area

        # Singular part: (1/4pi)[I_Rhat + (r_obs - r_fv) * I_1_over_R]
        I_singular = np.empty(3, dtype=np.complex128)
        for ix in range(3):
            I_singular[ix] = (I_Rhat[ix] + (r_obs[ix] - r_fv_src[ix]) * I_1_over_R) / four_pi

        # (3) Smooth remainder: rho * [g - 1/(4piR)]
        I_smooth = np.zeros(3, dtype=np.complex128)
        for i in range(nq):
            r_prime = bary[i, 0] * v0 + bary[i, 1] * v1 + bary[i, 2] * v2
            rho = r_prime - r_fv_src
            R = np.linalg.norm(r_obs - r_prime)
            if R < 1e-30:
                remainder = -1j * k / four_pi
            else:
                remainder = (np.exp(-1j * k * R) - 1.0) / (four_pi * R)
            I_smooth += weights[i] * rho * remainder

        return I_singular + I_smooth * twice_area

    # ------------------------------------------------------------------ #
    #  Single triangle-pair kernel                                         #
    # ------------------------------------------------------------------ #

    @numba.jit(nopython=True, cache=True)
    def _nb_triangle_pair(k, verts_test, verts_src, r_fv_test, r_fv_src,
                          sign_test, sign_src, l_test, l_src, A_test, A_src,
                          weights, bary, twice_area_test, twice_area_src,
                          is_near, quad_order, near_threshold):
        """Compute (I_A, I_Phi) for one test/source triangle pair.

        Port of ``_compute_triangle_pair`` in mom/impedance.py.
        Far-field branch combines the two inner j-loops into one for better
        cache utilisation (same weights/bary, compute R and g_val once).
        """
        nq = weights.shape[0]
        four_pi = 4.0 * np.pi
        I_A_raw = 0.0 + 0.0j
        I_Phi_raw = 0.0 + 0.0j

        for i in range(nq):
            r_obs = (bary[i, 0] * verts_test[0] + bary[i, 1] * verts_test[1]
                     + bary[i, 2] * verts_test[2])
            rho_test = r_obs - r_fv_test

            if is_near:
                g_int = _nb_green_singular(
                    k, r_obs, verts_src[0], verts_src[1], verts_src[2],
                    weights, bary, twice_area_src, near_threshold,
                )
                rho_src_g_int = _nb_rho_green_singular(
                    k, r_obs, verts_src[0], verts_src[1], verts_src[2],
                    r_fv_src, weights, bary, twice_area_src, near_threshold,
                )
            else:
                # Far field: single fused j-loop (computes g and rho*g together)
                g_int = 0.0 + 0.0j
                rho_src_g_int = np.zeros(3, dtype=np.complex128)
                for j in range(nq):
                    r_prime = (bary[j, 0] * verts_src[0] + bary[j, 1] * verts_src[1]
                               + bary[j, 2] * verts_src[2])
                    R = np.linalg.norm(r_obs - r_prime)
                    if R < 1e-30:
                        R = 1e-30
                    g_val = np.exp(-1j * k * R) / (four_pi * R)
                    g_int += weights[j] * g_val
                    rho_src_g_int += weights[j] * (r_prime - r_fv_src) * g_val
                g_int *= twice_area_src
                rho_src_g_int = rho_src_g_int * twice_area_src

            I_Phi_raw += weights[i] * g_int

            # Dot product: rho_test (float64) · rho_src_g_int (complex128)
            dot_val = (rho_test[0] * rho_src_g_int[0]
                       + rho_test[1] * rho_src_g_int[1]
                       + rho_test[2] * rho_src_g_int[2])
            I_A_raw += weights[i] * dot_val

        I_A_raw *= twice_area_test
        I_Phi_raw *= twice_area_test

        scale_A = (sign_test * l_test / (2.0 * A_test)) * (sign_src * l_src / (2.0 * A_src))
        scale_Phi = (sign_test * l_test / A_test) * (sign_src * l_src / A_src)

        return I_A_raw * scale_A, I_Phi_raw * scale_Phi

    # ------------------------------------------------------------------ #
    #  Parallel impedance matrix fill                                      #
    # ------------------------------------------------------------------ #

    @numba.jit(nopython=True, parallel=True, cache=True)
    def fill_Z_numba(Z, vertices, triangles,
                     t_plus, t_minus, fv_plus, fv_minus,
                     area_plus, area_minus, edge_length,
                     tri_centroids, tri_mean_edge, tri_twice_area,
                     weights, bary, k, eta, near_threshold, quad_order):
        """Fill impedance matrix Z in-place using Numba JIT + OpenMP parallelism.

        Outer loop over rows m uses ``prange`` (parallel).  Inner loop over
        columns n >= m is serial within each thread.  Write pattern:
          Thread m writes Z[m, n] for n >= m (row m from diagonal onward)
          Thread m writes Z[n, m] for n > m  (column m below diagonal)
        No write-write races between threads (proved in plan §2).

        Parameters
        ----------
        Z : complex128 (N, N)
            Pre-allocated output matrix.  Written in-place.
        vertices : float64 (N_v, 3)
        triangles : int32 (N_t, 3)
        t_plus, t_minus : int32 (N,)  — T+/T- triangle indices per basis
        fv_plus, fv_minus : int32 (N,) — free vertex indices per basis
        area_plus, area_minus : float64 (N,) — triangle areas per basis
        edge_length : float64 (N,) — shared edge lengths
        tri_centroids : float64 (N_t, 3)
        tri_mean_edge : float64 (N_t,)
        tri_twice_area : float64 (N_t,)
        weights, bary : float64 — precomputed quadrature rule
        k, eta : float64 — wavenumber, intrinsic impedance
        near_threshold : float64
        quad_order : int64
        """
        N = t_plus.shape[0]
        prefactor_A = 1j * k * eta
        prefactor_Phi = -1j * eta / k
        near_thresh_scaled = near_threshold * 3.0

        for m in prange(N):
            # --- Cache test-triangle geometry for basis m (both +/-) ---
            tri_mp = t_plus[m]
            verts_mp = np.empty((3, 3), dtype=np.float64)
            verts_mp[0] = vertices[triangles[tri_mp, 0]]
            verts_mp[1] = vertices[triangles[tri_mp, 1]]
            verts_mp[2] = vertices[triangles[tri_mp, 2]]
            r_fv_mp = vertices[fv_plus[m]]
            centroid_mp = tri_centroids[tri_mp]
            twice_area_mp = tri_twice_area[tri_mp]
            A_mp = area_plus[m]

            tri_mm = t_minus[m]
            verts_mm = np.empty((3, 3), dtype=np.float64)
            verts_mm[0] = vertices[triangles[tri_mm, 0]]
            verts_mm[1] = vertices[triangles[tri_mm, 1]]
            verts_mm[2] = vertices[triangles[tri_mm, 2]]
            r_fv_mm = vertices[fv_minus[m]]
            centroid_mm = tri_centroids[tri_mm]
            twice_area_mm = tri_twice_area[tri_mm]
            A_mm = area_minus[m]

            l_m = edge_length[m]

            for n in range(m, N):
                # --- Source triangle geometry for basis n ---
                tri_np = t_plus[n]
                verts_np = np.empty((3, 3), dtype=np.float64)
                verts_np[0] = vertices[triangles[tri_np, 0]]
                verts_np[1] = vertices[triangles[tri_np, 1]]
                verts_np[2] = vertices[triangles[tri_np, 2]]
                r_fv_np = vertices[fv_plus[n]]
                centroid_np = tri_centroids[tri_np]
                twice_area_np = tri_twice_area[tri_np]
                mean_edge_np = tri_mean_edge[tri_np]
                A_np = area_plus[n]

                tri_nm = t_minus[n]
                verts_nm = np.empty((3, 3), dtype=np.float64)
                verts_nm[0] = vertices[triangles[tri_nm, 0]]
                verts_nm[1] = vertices[triangles[tri_nm, 1]]
                verts_nm[2] = vertices[triangles[tri_nm, 2]]
                r_fv_nm = vertices[fv_minus[n]]
                centroid_nm = tri_centroids[tri_nm]
                twice_area_nm = tri_twice_area[tri_nm]
                mean_edge_nm = tri_mean_edge[tri_nm]
                A_nm = area_minus[n]

                l_n = edge_length[n]

                I_A_total = 0.0 + 0.0j
                I_Phi_total = 0.0 + 0.0j

                # (m+, n+)
                dist = np.linalg.norm(centroid_mp - centroid_np)
                is_near = dist < near_thresh_scaled * mean_edge_np if mean_edge_np > 1e-30 else True
                I_A, I_Phi = _nb_triangle_pair(
                    k, verts_mp, verts_np, r_fv_mp, r_fv_np,
                    1.0, 1.0, l_m, l_n, A_mp, A_np,
                    weights, bary, twice_area_mp, twice_area_np,
                    is_near, quad_order, near_threshold,
                )
                I_A_total += I_A
                I_Phi_total += I_Phi

                # (m+, n-)
                dist = np.linalg.norm(centroid_mp - centroid_nm)
                is_near = dist < near_thresh_scaled * mean_edge_nm if mean_edge_nm > 1e-30 else True
                I_A, I_Phi = _nb_triangle_pair(
                    k, verts_mp, verts_nm, r_fv_mp, r_fv_nm,
                    1.0, -1.0, l_m, l_n, A_mp, A_nm,
                    weights, bary, twice_area_mp, twice_area_nm,
                    is_near, quad_order, near_threshold,
                )
                I_A_total += I_A
                I_Phi_total += I_Phi

                # (m-, n+)
                dist = np.linalg.norm(centroid_mm - centroid_np)
                is_near = dist < near_thresh_scaled * mean_edge_np if mean_edge_np > 1e-30 else True
                I_A, I_Phi = _nb_triangle_pair(
                    k, verts_mm, verts_np, r_fv_mm, r_fv_np,
                    -1.0, 1.0, l_m, l_n, A_mm, A_np,
                    weights, bary, twice_area_mm, twice_area_np,
                    is_near, quad_order, near_threshold,
                )
                I_A_total += I_A
                I_Phi_total += I_Phi

                # (m-, n-)
                dist = np.linalg.norm(centroid_mm - centroid_nm)
                is_near = dist < near_thresh_scaled * mean_edge_nm if mean_edge_nm > 1e-30 else True
                I_A, I_Phi = _nb_triangle_pair(
                    k, verts_mm, verts_nm, r_fv_mm, r_fv_nm,
                    -1.0, -1.0, l_m, l_n, A_mm, A_nm,
                    weights, bary, twice_area_mm, twice_area_nm,
                    is_near, quad_order, near_threshold,
                )
                I_A_total += I_A
                I_Phi_total += I_Phi

                Z[m, n] = prefactor_A * I_A_total + prefactor_Phi * I_Phi_total
                if m != n:
                    Z[n, m] = Z[m, n]

    # ------------------------------------------------------------------ #
    #  MFIE per-pair kernel                                                #
    # ------------------------------------------------------------------ #

    @numba.jit(nopython=True, cache=True)
    def _nb_triangle_pair_mfie(k, n_hat_test,
                               verts_test, verts_src,
                               r_fv_test, r_fv_src,
                               sign_test, sign_src,
                               l_test, l_src, A_test, A_src,
                               weights, bary,
                               weights_near, bary_near,
                               twice_area_test, twice_area_src,
                               is_near):
        """Compute scaled K-term I_K for one test/source triangle pair.

        The MFIE K-kernel is:
            C(r,r') = n̂_m·(r−r') * (1+jkR)*exp(−jkR) / (4πR³)

        Near pairs use higher-order quadrature (weights_near / bary_near) to
        mitigate the O(1/R) near-field singularity.

        Returns scaled I_K = sign_test*sign_src * (l_m*l_n)/(4*A_m*A_n) * raw_integral
        """
        four_pi = 4.0 * np.pi

        if is_near:
            w  = weights_near
            bw = bary_near
        else:
            w  = weights
            bw = bary

        nq = w.shape[0]
        I_K_raw = 0.0 + 0.0j

        for i in range(nq):
            r_obs = (bw[i, 0] * verts_test[0]
                     + bw[i, 1] * verts_test[1]
                     + bw[i, 2] * verts_test[2])
            rho_m = r_obs - r_fv_test

            I_K_inner = 0.0 + 0.0j
            for j in range(nq):
                r_src = (bw[j, 0] * verts_src[0]
                         + bw[j, 1] * verts_src[1]
                         + bw[j, 2] * verts_src[2])
                rho_n = r_src - r_fv_src
                R_vec = r_obs - r_src
                R = np.linalg.norm(R_vec)
                if R < 1e-30:
                    continue
                # n̂×RWG kernel: −G_kernel · R_vec · (ρ_m × ρ_n)
                jkR = 1j * k * R
                G_kernel = (1.0 + jkR) * np.exp(-jkR) / (four_pi * R ** 3)
                cross_x = rho_m[1] * rho_n[2] - rho_m[2] * rho_n[1]
                cross_y = rho_m[2] * rho_n[0] - rho_m[0] * rho_n[2]
                cross_z = rho_m[0] * rho_n[1] - rho_m[1] * rho_n[0]
                R_dot_cross = (R_vec[0] * cross_x
                               + R_vec[1] * cross_y
                               + R_vec[2] * cross_z)
                I_K_inner += w[j] * (G_kernel * R_dot_cross)

            I_K_inner *= twice_area_src
            I_K_raw   += w[i] * I_K_inner

        I_K_raw *= twice_area_test

        scale = ((sign_test * l_test / (2.0 * A_test))
                 * (sign_src  * l_src  / (2.0 * A_src)))
        return I_K_raw * scale

    # ------------------------------------------------------------------ #
    #  MFIE parallel fill  (full N×N — not symmetric)                     #
    # ------------------------------------------------------------------ #

    @numba.jit(nopython=True, parallel=True, cache=True)
    def fill_Z_mfie_numba(Z, vertices, triangles, tri_normals,
                          t_plus, t_minus, fv_plus, fv_minus,
                          area_plus, area_minus, edge_length,
                          tri_centroids, tri_mean_edge, tri_twice_area,
                          weights, bary, weights_near, bary_near,
                          k, near_threshold):
        """Fill MFIE K-term matrix Z in-place (K-term only; Gram added by post_assembly).

        Write pattern: thread m writes Z[m, 0..N-1] exclusively.
        No cross-thread races — simpler than EFIE.
        """
        N = t_plus.shape[0]
        near_thresh_scaled = near_threshold * 3.0

        for m in prange(N):
            tri_mp = t_plus[m]
            verts_mp = np.empty((3, 3), dtype=np.float64)
            verts_mp[0] = vertices[triangles[tri_mp, 0]]
            verts_mp[1] = vertices[triangles[tri_mp, 1]]
            verts_mp[2] = vertices[triangles[tri_mp, 2]]
            r_fv_mp     = vertices[fv_plus[m]]
            centroid_mp = tri_centroids[tri_mp]
            twice_area_mp = tri_twice_area[tri_mp]
            A_mp = area_plus[m]
            n_hat_mp = tri_normals[tri_mp]

            tri_mm = t_minus[m]
            verts_mm = np.empty((3, 3), dtype=np.float64)
            verts_mm[0] = vertices[triangles[tri_mm, 0]]
            verts_mm[1] = vertices[triangles[tri_mm, 1]]
            verts_mm[2] = vertices[triangles[tri_mm, 2]]
            r_fv_mm     = vertices[fv_minus[m]]
            centroid_mm = tri_centroids[tri_mm]
            twice_area_mm = tri_twice_area[tri_mm]
            A_mm = area_minus[m]
            n_hat_mm = tri_normals[tri_mm]

            l_m = edge_length[m]

            for n in range(N):
                tri_np = t_plus[n]
                verts_np = np.empty((3, 3), dtype=np.float64)
                verts_np[0] = vertices[triangles[tri_np, 0]]
                verts_np[1] = vertices[triangles[tri_np, 1]]
                verts_np[2] = vertices[triangles[tri_np, 2]]
                r_fv_np     = vertices[fv_plus[n]]
                centroid_np = tri_centroids[tri_np]
                twice_area_np = tri_twice_area[tri_np]
                mean_edge_np  = tri_mean_edge[tri_np]
                A_np = area_plus[n]

                tri_nm = t_minus[n]
                verts_nm = np.empty((3, 3), dtype=np.float64)
                verts_nm[0] = vertices[triangles[tri_nm, 0]]
                verts_nm[1] = vertices[triangles[tri_nm, 1]]
                verts_nm[2] = vertices[triangles[tri_nm, 2]]
                r_fv_nm     = vertices[fv_minus[n]]
                centroid_nm = tri_centroids[tri_nm]
                twice_area_nm = tri_twice_area[tri_nm]
                mean_edge_nm  = tri_mean_edge[tri_nm]
                A_nm = area_minus[n]

                l_n = edge_length[n]
                I_K_total = 0.0 + 0.0j

                # (m+, n+)
                dist = np.linalg.norm(centroid_mp - centroid_np)
                is_near = dist < near_thresh_scaled * mean_edge_np if mean_edge_np > 1e-30 else True
                I_K_total += _nb_triangle_pair_mfie(
                    k, n_hat_mp, verts_mp, verts_np, r_fv_mp, r_fv_np,
                    1.0, 1.0, l_m, l_n, A_mp, A_np,
                    weights, bary, weights_near, bary_near,
                    twice_area_mp, twice_area_np, is_near,
                )

                # (m+, n-)
                dist = np.linalg.norm(centroid_mp - centroid_nm)
                is_near = dist < near_thresh_scaled * mean_edge_nm if mean_edge_nm > 1e-30 else True
                I_K_total += _nb_triangle_pair_mfie(
                    k, n_hat_mp, verts_mp, verts_nm, r_fv_mp, r_fv_nm,
                    1.0, -1.0, l_m, l_n, A_mp, A_nm,
                    weights, bary, weights_near, bary_near,
                    twice_area_mp, twice_area_nm, is_near,
                )

                # (m-, n+)
                dist = np.linalg.norm(centroid_mm - centroid_np)
                is_near = dist < near_thresh_scaled * mean_edge_np if mean_edge_np > 1e-30 else True
                I_K_total += _nb_triangle_pair_mfie(
                    k, n_hat_mm, verts_mm, verts_np, r_fv_mm, r_fv_np,
                    -1.0, 1.0, l_m, l_n, A_mm, A_np,
                    weights, bary, weights_near, bary_near,
                    twice_area_mm, twice_area_np, is_near,
                )

                # (m-, n-)
                dist = np.linalg.norm(centroid_mm - centroid_nm)
                is_near = dist < near_thresh_scaled * mean_edge_nm if mean_edge_nm > 1e-30 else True
                I_K_total += _nb_triangle_pair_mfie(
                    k, n_hat_mm, verts_mm, verts_nm, r_fv_mm, r_fv_nm,
                    -1.0, -1.0, l_m, l_n, A_mm, A_nm,
                    weights, bary, weights_near, bary_near,
                    twice_area_mm, twice_area_nm, is_near,
                )

                Z[m, n] = I_K_total

    # ------------------------------------------------------------------ #
    #  CFIE per-pair fused kernel  (EFIE + MFIE in one double loop)       #
    # ------------------------------------------------------------------ #

    @numba.jit(nopython=True, cache=True)
    def _nb_triangle_pair_cfie(k, n_hat_test,
                               verts_test, verts_src,
                               r_fv_test, r_fv_src,
                               sign_test, sign_src,
                               l_test, l_src, A_test, A_src,
                               weights, bary,
                               weights_near, bary_near,
                               twice_area_test, twice_area_src,
                               is_near, near_threshold):
        """Fused EFIE + MFIE per-pair kernel returning (I_A, I_Phi, I_K) scaled.

        Computes all three integrals in a single double loop over quadrature
        points.  The EFIE near-field branch uses singularity extraction;
        the MFIE K-term near-field branch uses higher-order quadrature.
        """
        four_pi = 4.0 * np.pi

        # --- EFIE integrals (I_A, I_Phi) — same logic as _nb_triangle_pair ---
        nq_efie = weights.shape[0]
        I_A_raw   = 0.0 + 0.0j
        I_Phi_raw = 0.0 + 0.0j

        for i in range(nq_efie):
            r_obs = (bary[i, 0] * verts_test[0]
                     + bary[i, 1] * verts_test[1]
                     + bary[i, 2] * verts_test[2])
            rho_test = r_obs - r_fv_test

            if is_near:
                g_int = _nb_green_singular(
                    k, r_obs, verts_src[0], verts_src[1], verts_src[2],
                    weights, bary, twice_area_src, near_threshold,
                )
                rho_src_g_int = _nb_rho_green_singular(
                    k, r_obs, verts_src[0], verts_src[1], verts_src[2],
                    r_fv_src, weights, bary, twice_area_src, near_threshold,
                )
            else:
                g_int = 0.0 + 0.0j
                rho_src_g_int = np.zeros(3, dtype=np.complex128)
                for j in range(nq_efie):
                    r_prime = (bary[j, 0] * verts_src[0]
                               + bary[j, 1] * verts_src[1]
                               + bary[j, 2] * verts_src[2])
                    R = np.linalg.norm(r_obs - r_prime)
                    if R < 1e-30:
                        R = 1e-30
                    g_val = np.exp(-1j * k * R) / (four_pi * R)
                    g_int += weights[j] * g_val
                    rho_src_g_int += weights[j] * (r_prime - r_fv_src) * g_val
                g_int          *= twice_area_src
                rho_src_g_int   = rho_src_g_int * twice_area_src

            I_Phi_raw += weights[i] * g_int
            dot_val = (rho_test[0] * rho_src_g_int[0]
                       + rho_test[1] * rho_src_g_int[1]
                       + rho_test[2] * rho_src_g_int[2])
            I_A_raw += weights[i] * dot_val

        I_A_raw   *= twice_area_test
        I_Phi_raw *= twice_area_test

        scale_A   = (sign_test * l_test / (2.0 * A_test)) * (sign_src * l_src / (2.0 * A_src))
        scale_Phi = (sign_test * l_test / A_test) * (sign_src * l_src / A_src)
        I_A_scaled   = I_A_raw   * scale_A
        I_Phi_scaled = I_Phi_raw * scale_Phi

        # --- MFIE K-term (I_K) — higher-order quad for near pairs ---
        if is_near:
            w  = weights_near
            bw = bary_near
        else:
            w  = weights
            bw = bary

        nq_mfie = w.shape[0]
        I_K_raw = 0.0 + 0.0j

        for i in range(nq_mfie):
            r_obs = (bw[i, 0] * verts_test[0]
                     + bw[i, 1] * verts_test[1]
                     + bw[i, 2] * verts_test[2])
            rho_m = r_obs - r_fv_test

            I_K_inner = 0.0 + 0.0j
            for j in range(nq_mfie):
                r_src = (bw[j, 0] * verts_src[0]
                         + bw[j, 1] * verts_src[1]
                         + bw[j, 2] * verts_src[2])
                rho_n = r_src - r_fv_src
                R_vec = r_obs - r_src
                R = np.linalg.norm(R_vec)
                if R < 1e-30:
                    continue
                # n̂×RWG kernel: −G_kernel · R_vec · (ρ_m × ρ_n)
                jkR = 1j * k * R
                G_kernel = (1.0 + jkR) * np.exp(-jkR) / (four_pi * R ** 3)
                cross_x = rho_m[1] * rho_n[2] - rho_m[2] * rho_n[1]
                cross_y = rho_m[2] * rho_n[0] - rho_m[0] * rho_n[2]
                cross_z = rho_m[0] * rho_n[1] - rho_m[1] * rho_n[0]
                R_dot_cross = (R_vec[0] * cross_x
                               + R_vec[1] * cross_y
                               + R_vec[2] * cross_z)
                I_K_inner += w[j] * (G_kernel * R_dot_cross)

            I_K_inner *= twice_area_src
            I_K_raw   += w[i] * I_K_inner

        I_K_raw *= twice_area_test
        scale_K = scale_A  # same RWG amplitude factor as I_A
        I_K_scaled = I_K_raw * scale_K

        return I_A_scaled, I_Phi_scaled, I_K_scaled

    # ------------------------------------------------------------------ #
    #  CFIE parallel fill  (full N×N, fused EFIE+MFIE)                    #
    # ------------------------------------------------------------------ #

    @numba.jit(nopython=True, parallel=True, cache=True)
    def fill_Z_cfie_numba(Z, vertices, triangles, tri_normals,
                          t_plus, t_minus, fv_plus, fv_minus,
                          area_plus, area_minus, edge_length,
                          tri_centroids, tri_mean_edge, tri_twice_area,
                          weights, bary, weights_near, bary_near,
                          k, eta, alpha, near_threshold):
        """Fill CFIE impedance matrix Z in-place.

        Z_mn = alpha*(jkη*I_A − jη/k*I_Phi) + (1−alpha)*η*I_K

        Full N×N loop (not symmetric). Each thread writes its own row m.
        """
        N = t_plus.shape[0]
        prefactor_A   = 1j * k * eta
        prefactor_Phi = -1j * eta / k
        near_thresh_scaled = near_threshold * 3.0

        for m in prange(N):
            tri_mp = t_plus[m]
            verts_mp = np.empty((3, 3), dtype=np.float64)
            verts_mp[0] = vertices[triangles[tri_mp, 0]]
            verts_mp[1] = vertices[triangles[tri_mp, 1]]
            verts_mp[2] = vertices[triangles[tri_mp, 2]]
            r_fv_mp     = vertices[fv_plus[m]]
            centroid_mp = tri_centroids[tri_mp]
            twice_area_mp = tri_twice_area[tri_mp]
            A_mp = area_plus[m]
            n_hat_mp = tri_normals[tri_mp]

            tri_mm = t_minus[m]
            verts_mm = np.empty((3, 3), dtype=np.float64)
            verts_mm[0] = vertices[triangles[tri_mm, 0]]
            verts_mm[1] = vertices[triangles[tri_mm, 1]]
            verts_mm[2] = vertices[triangles[tri_mm, 2]]
            r_fv_mm     = vertices[fv_minus[m]]
            centroid_mm = tri_centroids[tri_mm]
            twice_area_mm = tri_twice_area[tri_mm]
            A_mm = area_minus[m]
            n_hat_mm = tri_normals[tri_mm]

            l_m = edge_length[m]

            for n in range(N):
                tri_np = t_plus[n]
                verts_np = np.empty((3, 3), dtype=np.float64)
                verts_np[0] = vertices[triangles[tri_np, 0]]
                verts_np[1] = vertices[triangles[tri_np, 1]]
                verts_np[2] = vertices[triangles[tri_np, 2]]
                r_fv_np     = vertices[fv_plus[n]]
                centroid_np = tri_centroids[tri_np]
                twice_area_np = tri_twice_area[tri_np]
                mean_edge_np  = tri_mean_edge[tri_np]
                A_np = area_plus[n]

                tri_nm = t_minus[n]
                verts_nm = np.empty((3, 3), dtype=np.float64)
                verts_nm[0] = vertices[triangles[tri_nm, 0]]
                verts_nm[1] = vertices[triangles[tri_nm, 1]]
                verts_nm[2] = vertices[triangles[tri_nm, 2]]
                r_fv_nm     = vertices[fv_minus[n]]
                centroid_nm = tri_centroids[tri_nm]
                twice_area_nm = tri_twice_area[tri_nm]
                mean_edge_nm  = tri_mean_edge[tri_nm]
                A_nm = area_minus[n]

                l_n = edge_length[n]

                I_A_total   = 0.0 + 0.0j
                I_Phi_total = 0.0 + 0.0j
                I_K_total   = 0.0 + 0.0j

                # (m+, n+)
                dist = np.linalg.norm(centroid_mp - centroid_np)
                is_near = dist < near_thresh_scaled * mean_edge_np if mean_edge_np > 1e-30 else True
                I_A, I_Phi, I_K = _nb_triangle_pair_cfie(
                    k, n_hat_mp, verts_mp, verts_np, r_fv_mp, r_fv_np,
                    1.0, 1.0, l_m, l_n, A_mp, A_np,
                    weights, bary, weights_near, bary_near,
                    twice_area_mp, twice_area_np, is_near, near_threshold,
                )
                I_A_total += I_A; I_Phi_total += I_Phi; I_K_total += I_K

                # (m+, n-)
                dist = np.linalg.norm(centroid_mp - centroid_nm)
                is_near = dist < near_thresh_scaled * mean_edge_nm if mean_edge_nm > 1e-30 else True
                I_A, I_Phi, I_K = _nb_triangle_pair_cfie(
                    k, n_hat_mp, verts_mp, verts_nm, r_fv_mp, r_fv_nm,
                    1.0, -1.0, l_m, l_n, A_mp, A_nm,
                    weights, bary, weights_near, bary_near,
                    twice_area_mp, twice_area_nm, is_near, near_threshold,
                )
                I_A_total += I_A; I_Phi_total += I_Phi; I_K_total += I_K

                # (m-, n+)
                dist = np.linalg.norm(centroid_mm - centroid_np)
                is_near = dist < near_thresh_scaled * mean_edge_np if mean_edge_np > 1e-30 else True
                I_A, I_Phi, I_K = _nb_triangle_pair_cfie(
                    k, n_hat_mm, verts_mm, verts_np, r_fv_mm, r_fv_np,
                    -1.0, 1.0, l_m, l_n, A_mm, A_np,
                    weights, bary, weights_near, bary_near,
                    twice_area_mm, twice_area_np, is_near, near_threshold,
                )
                I_A_total += I_A; I_Phi_total += I_Phi; I_K_total += I_K

                # (m-, n-)
                dist = np.linalg.norm(centroid_mm - centroid_nm)
                is_near = dist < near_thresh_scaled * mean_edge_nm if mean_edge_nm > 1e-30 else True
                I_A, I_Phi, I_K = _nb_triangle_pair_cfie(
                    k, n_hat_mm, verts_mm, verts_nm, r_fv_mm, r_fv_nm,
                    -1.0, -1.0, l_m, l_n, A_mm, A_nm,
                    weights, bary, weights_near, bary_near,
                    twice_area_mm, twice_area_nm, is_near, near_threshold,
                )
                I_A_total += I_A; I_Phi_total += I_Phi; I_K_total += I_K

                efie_contrib = prefactor_A * I_A_total + prefactor_Phi * I_Phi_total
                Z[m, n] = alpha * efie_contrib + (1.0 - alpha) * eta * I_K_total
