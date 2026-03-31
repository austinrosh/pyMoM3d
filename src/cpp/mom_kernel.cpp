/**
 * mom_kernel.cpp — C++ EFIE impedance matrix fill kernel.
 *
 * Implements fill_impedance_cpp(), the production C++ backend for
 * fill_impedance_matrix() in mom/impedance.py.
 *
 * Architecture
 * ------------
 * - All parallelism lives inside this file (OpenMP over rows m).
 * - GIL is released by pybind11 before this function is called.
 * - Uses 2D block tiling (BLOCK_SIZE × BLOCK_SIZE sub-matrices) for
 *   cache efficiency and GPU-readiness: the same block structure maps
 *   directly to CUDA thread block tiles in a future GPU port.
 * - Loop structure: outer OpenMP prange over block-rows bm,
 *   inner serial loop over block-columns bn ≥ bm.
 *
 * Race-condition proof (write pattern)
 * -------------------------------------
 * Thread A (block bm_A, bn_A) writes:
 *   Z[bm_A:, bn_A:]  (upper triangle elements)
 *   Z[bn_A:, bm_A:]  (symmetric copies, lower triangle)
 * Thread B (block bm_B, bn_B) never touches the same cells because:
 *   Upper: only if bm_A == bm_B AND bn_A == bn_B (same thread)
 *   Lower cross: would require bn_A == bm_B AND bm_A == bn_B,
 *                i.e. bm_A == bn_B >= bm_B == bn_A >= bm_A — contradiction.
 *
 * OpenMP on macOS
 * ---------------
 * Apple clang requires  -Xpreprocessor -fopenmp  plus  -lomp.
 * build_cpp.py handles this automatically using the libomp Homebrew prefix.
 * If OpenMP is unavailable, _OPENMP is undefined and the code compiles as
 * a single-threaded C++ kernel (still much faster than Python).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cstdint>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "singularity.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
static constexpr int BLOCK_SIZE = 32;   // 2-D tile side length

// ---------------------------------------------------------------------------
// Helper: load a triangle's 3 vertices from flat arrays
// ---------------------------------------------------------------------------
static inline void load_triangle(
    const double* vertices,
    const int32_t* triangles,
    int tri,
    v3 out[3]) noexcept
{
    for (int v = 0; v < 3; ++v) {
        int idx = triangles[tri * 3 + v];
        out[v] = {vertices[idx*3], vertices[idx*3+1], vertices[idx*3+2]};
    }
}

static inline v3 load_vertex(const double* vertices, int idx) noexcept {
    return {vertices[idx*3], vertices[idx*3+1], vertices[idx*3+2]};
}

static inline v3 load_centroid(const double* tri_centroids, int tri) noexcept {
    return {tri_centroids[tri*3], tri_centroids[tri*3+1], tri_centroids[tri*3+2]};
}

// ---------------------------------------------------------------------------
// Single triangle-pair kernel  →  (I_A, I_Phi)
//
// Fuses the two inner j-loops from the Python version (g and rho*g) into
// one loop when in far-field mode: R and g_val are computed once per (i,j).
// ---------------------------------------------------------------------------
static inline void triangle_pair(
    double k,
    const v3 verts_test[3], const v3 verts_src[3],
    const v3& r_fv_test, const v3& r_fv_src,
    double sign_test, double sign_src,
    double l_test, double l_src,
    double A_test, double A_src,
    int nq, const double* weights, const double* bary,
    double twice_area_test, double twice_area_src,
    bool is_near, double near_threshold,
    cd& I_A_out, cd& I_Phi_out) noexcept
{
    constexpr double four_pi = 4.0 * M_PI;
    cd I_A_raw{0.0, 0.0};
    cd I_Phi_raw{0.0, 0.0};

    for (int i = 0; i < nq; ++i) {
        v3 r_obs = bary_interp(bary + i*3, verts_test[0], verts_test[1], verts_test[2]);
        v3 rho_test = sub3(r_obs, r_fv_test);

        cd g_int;
        cv3 rho_src_g_int{};

        if (is_near) {
            g_int = green_singular(k, r_obs,
                                   verts_src[0], verts_src[1], verts_src[2],
                                   nq, weights, bary, twice_area_src, near_threshold);
            rho_src_g_int = rho_green_singular(k, r_obs,
                                               verts_src[0], verts_src[1], verts_src[2],
                                               r_fv_src,
                                               nq, weights, bary, twice_area_src, near_threshold);
        } else {
            // Far field: fused j-loop (compute R and g_val once per j)
            g_int = cd{0.0, 0.0};
            for (int j = 0; j < nq; ++j) {
                v3 r_prime = bary_interp(bary + j*3, verts_src[0], verts_src[1], verts_src[2]);
                double R = norm3(sub3(r_obs, r_prime));
                if (R < 1e-30) R = 1e-30;
                cd g_val = std::exp(cd{0.0, -k * R}) / (four_pi * R);
                g_int += weights[j] * g_val;
                v3 rho = sub3(r_prime, r_fv_src);
                rho_src_g_int[0] += weights[j] * rho[0] * g_val;
                rho_src_g_int[1] += weights[j] * rho[1] * g_val;
                rho_src_g_int[2] += weights[j] * rho[2] * g_val;
            }
            g_int          *= twice_area_src;
            rho_src_g_int[0] *= twice_area_src;
            rho_src_g_int[1] *= twice_area_src;
            rho_src_g_int[2] *= twice_area_src;
        }

        I_Phi_raw += weights[i] * g_int;

        // Dot product: rho_test (real) · rho_src_g_int (complex)
        cd dot_val = (rho_test[0] * rho_src_g_int[0]
                    + rho_test[1] * rho_src_g_int[1]
                    + rho_test[2] * rho_src_g_int[2]);
        I_A_raw += weights[i] * dot_val;
    }

    I_A_raw   *= twice_area_test;
    I_Phi_raw *= twice_area_test;

    double scale_A   = (sign_test * l_test / (2.0 * A_test)) * (sign_src * l_src / (2.0 * A_src));
    double scale_Phi = (sign_test * l_test / A_test) * (sign_src * l_src / A_src);

    I_A_out   = I_A_raw   * scale_A;
    I_Phi_out = I_Phi_raw * scale_Phi;
}

// ---------------------------------------------------------------------------
// Main impedance matrix fill — called from Python via pybind11
// ---------------------------------------------------------------------------
void fill_impedance_cpp(
    py::array_t<cd, py::array::c_style>      Z_arr,
    py::array_t<double,  py::array::c_style> vertices_arr,
    py::array_t<int32_t, py::array::c_style> triangles_arr,
    py::array_t<int32_t, py::array::c_style> t_plus_arr,
    py::array_t<int32_t, py::array::c_style> t_minus_arr,
    py::array_t<int32_t, py::array::c_style> fv_plus_arr,
    py::array_t<int32_t, py::array::c_style> fv_minus_arr,
    py::array_t<double,  py::array::c_style> area_plus_arr,
    py::array_t<double,  py::array::c_style> area_minus_arr,
    py::array_t<double,  py::array::c_style> edge_length_arr,
    py::array_t<double,  py::array::c_style> tri_centroids_arr,
    py::array_t<double,  py::array::c_style> tri_mean_edge_arr,
    py::array_t<double,  py::array::c_style> tri_twice_area_arr,
    py::array_t<double,  py::array::c_style> weights_arr,
    py::array_t<double,  py::array::c_style> bary_arr,
    double k, double eta,
    double near_threshold,
    int    quad_order,
    int    num_threads)
{
    // --- Obtain raw pointers (no copies, zero-overhead) ---
    cd*       Z               = Z_arr.mutable_data();
    const double*  vertices       = vertices_arr.data();
    const int32_t* triangles      = triangles_arr.data();
    const int32_t* t_plus         = t_plus_arr.data();
    const int32_t* t_minus        = t_minus_arr.data();
    const int32_t* fv_plus        = fv_plus_arr.data();
    const int32_t* fv_minus       = fv_minus_arr.data();
    const double*  area_plus      = area_plus_arr.data();
    const double*  area_minus     = area_minus_arr.data();
    const double*  edge_length    = edge_length_arr.data();
    const double*  tri_centroids  = tri_centroids_arr.data();
    const double*  tri_mean_edge  = tri_mean_edge_arr.data();
    const double*  tri_twice_area = tri_twice_area_arr.data();
    const double*  weights        = weights_arr.data();
    const double*  bary           = bary_arr.data();  // (nq, 3) row-major

    const int N  = static_cast<int>(t_plus_arr.shape(0));
    const int nq = static_cast<int>(weights_arr.shape(0));

    const cd   prefactor_A   = cd{0.0, k * eta};
    const cd   prefactor_Phi = cd{0.0, -eta / k};
    const double near_thresh_scaled = near_threshold * 3.0;

#ifdef _OPENMP
    if (num_threads > 0)
        omp_set_num_threads(num_threads);

    // Release the GIL — pybind11 already did this before calling us.
    // Outer parallel loop over block-rows bm.
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int bm = 0; bm < N; bm += BLOCK_SIZE) {
        const int m_end = std::min(bm + BLOCK_SIZE, N);

        // --- Cache geometry for all test basis in [bm, m_end) ---
        // Two triangles per basis (t_plus, t_minus)
        v3 verts_mp[BLOCK_SIZE][3], verts_mm[BLOCK_SIZE][3];
        v3 r_fv_mp[BLOCK_SIZE],    r_fv_mm[BLOCK_SIZE];
        v3 centroid_mp[BLOCK_SIZE], centroid_mm[BLOCK_SIZE];
        double ta_mp[BLOCK_SIZE], ta_mm[BLOCK_SIZE];  // twice_area
        double Ap[BLOCK_SIZE],   Am[BLOCK_SIZE];       // area_plus/minus
        double lm[BLOCK_SIZE];                         // edge_length

        for (int m = bm; m < m_end; ++m) {
            int loc = m - bm;
            int tri_mp = t_plus[m];
            load_triangle(vertices, triangles, tri_mp, verts_mp[loc]);
            r_fv_mp[loc]    = load_vertex(vertices, fv_plus[m]);
            centroid_mp[loc]= load_centroid(tri_centroids, tri_mp);
            ta_mp[loc]      = tri_twice_area[tri_mp];
            Ap[loc]         = area_plus[m];

            int tri_mm = t_minus[m];
            load_triangle(vertices, triangles, tri_mm, verts_mm[loc]);
            r_fv_mm[loc]    = load_vertex(vertices, fv_minus[m]);
            centroid_mm[loc]= load_centroid(tri_centroids, tri_mm);
            ta_mm[loc]      = tri_twice_area[tri_mm];
            Am[loc]         = area_minus[m];

            lm[loc] = edge_length[m];
        }

        // --- Iterate over block-columns bn ≥ bm ---
        for (int bn = bm; bn < N; bn += BLOCK_SIZE) {
            const int n_end = std::min(bn + BLOCK_SIZE, N);

            // Cache geometry for source basis in [bn, n_end)
            v3 verts_np[BLOCK_SIZE][3], verts_nm[BLOCK_SIZE][3];
            v3 r_fv_np[BLOCK_SIZE],    r_fv_nm[BLOCK_SIZE];
            v3 centroid_np[BLOCK_SIZE], centroid_nm[BLOCK_SIZE];
            double ta_np[BLOCK_SIZE], ta_nm[BLOCK_SIZE];
            double me_np[BLOCK_SIZE], me_nm[BLOCK_SIZE]; // mean_edge
            double An_p[BLOCK_SIZE],  An_m[BLOCK_SIZE];
            double ln[BLOCK_SIZE];

            for (int n = bn; n < n_end; ++n) {
                int loc = n - bn;
                int tri_np = t_plus[n];
                load_triangle(vertices, triangles, tri_np, verts_np[loc]);
                r_fv_np[loc]    = load_vertex(vertices, fv_plus[n]);
                centroid_np[loc]= load_centroid(tri_centroids, tri_np);
                ta_np[loc]      = tri_twice_area[tri_np];
                me_np[loc]      = tri_mean_edge[tri_np];
                An_p[loc]       = area_plus[n];

                int tri_nm = t_minus[n];
                load_triangle(vertices, triangles, tri_nm, verts_nm[loc]);
                r_fv_nm[loc]    = load_vertex(vertices, fv_minus[n]);
                centroid_nm[loc]= load_centroid(tri_centroids, tri_nm);
                ta_nm[loc]      = tri_twice_area[tri_nm];
                me_nm[loc]      = tri_mean_edge[tri_nm];
                An_m[loc]       = area_minus[n];

                ln[loc] = edge_length[n];
            }

            // --- Element loop within the (bm, bn) block ---
            for (int m = bm; m < m_end; ++m) {
                int lm_idx = m - bm;
                int n_start = std::max(bn, m);  // upper triangle only

                for (int n = n_start; n < n_end; ++n) {
                    int ln_idx = n - bn;

                    cd I_A_total{0.0, 0.0};
                    cd I_Phi_total{0.0, 0.0};

                    // 4 triangle-pair interactions: (m+,n+), (m+,n-), (m-,n+), (m-,n-)
                    double dist, me;
                    bool is_near;
                    cd I_A, I_Phi;

                    // (m+, n+)
                    me = me_np[ln_idx];
                    dist = norm3(sub3(centroid_mp[lm_idx], centroid_np[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    triangle_pair(k,
                        verts_mp[lm_idx], verts_np[ln_idx],
                        r_fv_mp[lm_idx],  r_fv_np[ln_idx],
                        +1.0, +1.0, lm[lm_idx], ln[ln_idx], Ap[lm_idx], An_p[ln_idx],
                        nq, weights, bary, ta_mp[lm_idx], ta_np[ln_idx],
                        is_near, near_threshold, I_A, I_Phi);
                    I_A_total += I_A; I_Phi_total += I_Phi;

                    // (m+, n-)
                    me = me_nm[ln_idx];
                    dist = norm3(sub3(centroid_mp[lm_idx], centroid_nm[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    triangle_pair(k,
                        verts_mp[lm_idx], verts_nm[ln_idx],
                        r_fv_mp[lm_idx],  r_fv_nm[ln_idx],
                        +1.0, -1.0, lm[lm_idx], ln[ln_idx], Ap[lm_idx], An_m[ln_idx],
                        nq, weights, bary, ta_mp[lm_idx], ta_nm[ln_idx],
                        is_near, near_threshold, I_A, I_Phi);
                    I_A_total += I_A; I_Phi_total += I_Phi;

                    // (m-, n+)
                    me = me_np[ln_idx];
                    dist = norm3(sub3(centroid_mm[lm_idx], centroid_np[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    triangle_pair(k,
                        verts_mm[lm_idx], verts_np[ln_idx],
                        r_fv_mm[lm_idx],  r_fv_np[ln_idx],
                        -1.0, +1.0, lm[lm_idx], ln[ln_idx], Am[lm_idx], An_p[ln_idx],
                        nq, weights, bary, ta_mm[lm_idx], ta_np[ln_idx],
                        is_near, near_threshold, I_A, I_Phi);
                    I_A_total += I_A; I_Phi_total += I_Phi;

                    // (m-, n-)
                    me = me_nm[ln_idx];
                    dist = norm3(sub3(centroid_mm[lm_idx], centroid_nm[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    triangle_pair(k,
                        verts_mm[lm_idx], verts_nm[ln_idx],
                        r_fv_mm[lm_idx],  r_fv_nm[ln_idx],
                        -1.0, -1.0, lm[lm_idx], ln[ln_idx], Am[lm_idx], An_m[ln_idx],
                        nq, weights, bary, ta_mm[lm_idx], ta_nm[ln_idx],
                        is_near, near_threshold, I_A, I_Phi);
                    I_A_total += I_A; I_Phi_total += I_Phi;

                    cd val = prefactor_A * I_A_total + prefactor_Phi * I_Phi_total;
                    Z[m * N + n] = val;
                    if (m != n)
                        Z[n * N + m] = val;
                }
            }
        } // bn block loop
    } // bm block loop (OpenMP parallel)
}

// ---------------------------------------------------------------------------
// Helper: load outward unit normal for triangle tri
// ---------------------------------------------------------------------------
static inline v3 load_normal(const double* tri_normals, int tri) noexcept {
    return {tri_normals[tri*3], tri_normals[tri*3+1], tri_normals[tri*3+2]};
}

// ---------------------------------------------------------------------------
// MFIE single triangle-pair kernel  →  I_K
//
// Computes the scaled K-term integral:
//   I_K = scale * ∫∫ (ρ_m·ρ_n) * n̂_m·(r−r')*(1+jkR)*exp(−jkR)/(4πR³) dS' dS
//
// Near pairs use higher-order quadrature (nq_near / weights_near / bary_near).
// ---------------------------------------------------------------------------
static inline cd triangle_pair_mfie(
    double k,
    const v3& n_hat_test,
    const v3 verts_test[3], const v3 verts_src[3],
    const v3& r_fv_test, const v3& r_fv_src,
    double sign_test, double sign_src,
    double l_test, double l_src,
    double A_test, double A_src,
    int nq,       const double* weights,       const double* bary,
    int nq_near,  const double* weights_near,  const double* bary_near,
    double twice_area_test, double twice_area_src,
    bool is_near) noexcept
{
    constexpr double four_pi = 4.0 * M_PI;

    // Select quadrature rule
    int    nq_use  = is_near ? nq_near  : nq;
    const double* w_use  = is_near ? weights_near  : weights;
    const double* bw_use = is_near ? bary_near      : bary;

    cd I_K_raw{0.0, 0.0};

    for (int i = 0; i < nq_use; ++i) {
        v3 r_obs = bary_interp(bw_use + i*3, verts_test[0], verts_test[1], verts_test[2]);
        v3 rho_m = sub3(r_obs, r_fv_test);

        cd I_K_inner{0.0, 0.0};
        for (int j = 0; j < nq_use; ++j) {
            v3 r_src  = bary_interp(bw_use + j*3, verts_src[0], verts_src[1], verts_src[2]);
            v3 rho_n  = sub3(r_src, r_fv_src);
            v3 R_vec  = sub3(r_obs, r_src);
            double R  = norm3(R_vec);
            if (R < 1e-30) continue;

            // MFIE kernel: C = n̂·(r−r') * (1+jkR)*exp(−jkR) / (4πR³)
            double ndot = dot3(n_hat_test, R_vec);
            double rdot = dot3(rho_m, rho_n);
            cd jkR{0.0, k * R};
            cd C = ndot * (cd{1.0, 0.0} + jkR) * std::exp(-jkR) / (four_pi * R * R * R);
            I_K_inner += w_use[j] * rdot * C;
        }
        I_K_inner *= twice_area_src;
        I_K_raw   += w_use[i] * I_K_inner;
    }
    I_K_raw *= twice_area_test;

    double scale = (sign_test * l_test / (2.0 * A_test))
                 * (sign_src  * l_src  / (2.0 * A_src));
    return I_K_raw * scale;
}

// ---------------------------------------------------------------------------
// MFIE impedance matrix fill — full N×N, OpenMP parallel over block-rows
//
// Write pattern: thread bm writes Z[bm:m_end, 0:N] exclusively.
// No cross-thread races (each thread owns its own rows).
// ---------------------------------------------------------------------------
void fill_impedance_mfie_cpp(
    py::array_t<cd,      py::array::c_style> Z_arr,
    py::array_t<double,  py::array::c_style> vertices_arr,
    py::array_t<int32_t, py::array::c_style> triangles_arr,
    py::array_t<double,  py::array::c_style> tri_normals_arr,
    py::array_t<int32_t, py::array::c_style> t_plus_arr,
    py::array_t<int32_t, py::array::c_style> t_minus_arr,
    py::array_t<int32_t, py::array::c_style> fv_plus_arr,
    py::array_t<int32_t, py::array::c_style> fv_minus_arr,
    py::array_t<double,  py::array::c_style> area_plus_arr,
    py::array_t<double,  py::array::c_style> area_minus_arr,
    py::array_t<double,  py::array::c_style> edge_length_arr,
    py::array_t<double,  py::array::c_style> tri_centroids_arr,
    py::array_t<double,  py::array::c_style> tri_mean_edge_arr,
    py::array_t<double,  py::array::c_style> tri_twice_area_arr,
    py::array_t<double,  py::array::c_style> weights_arr,
    py::array_t<double,  py::array::c_style> bary_arr,
    py::array_t<double,  py::array::c_style> weights_near_arr,
    py::array_t<double,  py::array::c_style> bary_near_arr,
    double k,
    double near_threshold,
    int    quad_order,
    int    num_threads)
{
    cd*            Z               = Z_arr.mutable_data();
    const double*  vertices        = vertices_arr.data();
    const int32_t* triangles       = triangles_arr.data();
    const double*  tri_normals     = tri_normals_arr.data();
    const int32_t* t_plus          = t_plus_arr.data();
    const int32_t* t_minus         = t_minus_arr.data();
    const int32_t* fv_plus         = fv_plus_arr.data();
    const int32_t* fv_minus        = fv_minus_arr.data();
    const double*  area_plus       = area_plus_arr.data();
    const double*  area_minus      = area_minus_arr.data();
    const double*  edge_length     = edge_length_arr.data();
    const double*  tri_centroids   = tri_centroids_arr.data();
    const double*  tri_mean_edge   = tri_mean_edge_arr.data();
    const double*  tri_twice_area  = tri_twice_area_arr.data();
    const double*  weights         = weights_arr.data();
    const double*  bary            = bary_arr.data();
    const double*  weights_near    = weights_near_arr.data();
    const double*  bary_near       = bary_near_arr.data();

    const int N       = static_cast<int>(t_plus_arr.shape(0));
    const int nq      = static_cast<int>(weights_arr.shape(0));
    const int nq_near = static_cast<int>(weights_near_arr.shape(0));
    const double near_thresh_scaled = near_threshold * 3.0;

#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int bm = 0; bm < N; bm += BLOCK_SIZE) {
        const int m_end = std::min(bm + BLOCK_SIZE, N);

        // Cache test-basis geometry for block [bm, m_end)
        v3     verts_mp[BLOCK_SIZE][3], verts_mm[BLOCK_SIZE][3];
        v3     r_fv_mp[BLOCK_SIZE],     r_fv_mm[BLOCK_SIZE];
        v3     centroid_mp[BLOCK_SIZE], centroid_mm[BLOCK_SIZE];
        v3     n_hat_mp[BLOCK_SIZE],    n_hat_mm[BLOCK_SIZE];
        double ta_mp[BLOCK_SIZE],       ta_mm[BLOCK_SIZE];
        double Ap[BLOCK_SIZE],          Am[BLOCK_SIZE];
        double lm[BLOCK_SIZE];

        for (int m = bm; m < m_end; ++m) {
            int loc = m - bm;
            int tri_mp = t_plus[m];
            load_triangle(vertices, triangles, tri_mp, verts_mp[loc]);
            r_fv_mp[loc]     = load_vertex(vertices, fv_plus[m]);
            centroid_mp[loc] = load_centroid(tri_centroids, tri_mp);
            n_hat_mp[loc]    = load_normal(tri_normals, tri_mp);
            ta_mp[loc]       = tri_twice_area[tri_mp];
            Ap[loc]          = area_plus[m];

            int tri_mm = t_minus[m];
            load_triangle(vertices, triangles, tri_mm, verts_mm[loc]);
            r_fv_mm[loc]     = load_vertex(vertices, fv_minus[m]);
            centroid_mm[loc] = load_centroid(tri_centroids, tri_mm);
            n_hat_mm[loc]    = load_normal(tri_normals, tri_mm);
            ta_mm[loc]       = tri_twice_area[tri_mm];
            Am[loc]          = area_minus[m];
            lm[loc]          = edge_length[m];
        }

        // Iterate over ALL block-columns (full N×N — no symmetry)
        for (int bn = 0; bn < N; bn += BLOCK_SIZE) {
            const int n_end = std::min(bn + BLOCK_SIZE, N);

            v3     verts_np[BLOCK_SIZE][3], verts_nm[BLOCK_SIZE][3];
            v3     r_fv_np[BLOCK_SIZE],     r_fv_nm[BLOCK_SIZE];
            v3     centroid_np[BLOCK_SIZE], centroid_nm[BLOCK_SIZE];
            double ta_np[BLOCK_SIZE],       ta_nm[BLOCK_SIZE];
            double me_np[BLOCK_SIZE],       me_nm[BLOCK_SIZE];
            double An_p[BLOCK_SIZE],        An_m[BLOCK_SIZE];
            double ln[BLOCK_SIZE];

            for (int n = bn; n < n_end; ++n) {
                int loc = n - bn;
                int tri_np = t_plus[n];
                load_triangle(vertices, triangles, tri_np, verts_np[loc]);
                r_fv_np[loc]     = load_vertex(vertices, fv_plus[n]);
                centroid_np[loc] = load_centroid(tri_centroids, tri_np);
                ta_np[loc]       = tri_twice_area[tri_np];
                me_np[loc]       = tri_mean_edge[tri_np];
                An_p[loc]        = area_plus[n];

                int tri_nm = t_minus[n];
                load_triangle(vertices, triangles, tri_nm, verts_nm[loc]);
                r_fv_nm[loc]     = load_vertex(vertices, fv_minus[n]);
                centroid_nm[loc] = load_centroid(tri_centroids, tri_nm);
                ta_nm[loc]       = tri_twice_area[tri_nm];
                me_nm[loc]       = tri_mean_edge[tri_nm];
                An_m[loc]        = area_minus[n];
                ln[loc]          = edge_length[n];
            }

            for (int m = bm; m < m_end; ++m) {
                int lm_idx = m - bm;

                for (int n = bn; n < n_end; ++n) {
                    int ln_idx = n - bn;

                    cd I_K_total{0.0, 0.0};
                    double dist, me;
                    bool is_near;

                    // (m+, n+)
                    me = me_np[ln_idx];
                    dist = norm3(sub3(centroid_mp[lm_idx], centroid_np[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    I_K_total += triangle_pair_mfie(k, n_hat_mp[lm_idx],
                        verts_mp[lm_idx], verts_np[ln_idx],
                        r_fv_mp[lm_idx],  r_fv_np[ln_idx],
                        +1.0, +1.0, lm[lm_idx], ln[ln_idx], Ap[lm_idx], An_p[ln_idx],
                        nq, weights, bary, nq_near, weights_near, bary_near,
                        ta_mp[lm_idx], ta_np[ln_idx], is_near);

                    // (m+, n-)
                    me = me_nm[ln_idx];
                    dist = norm3(sub3(centroid_mp[lm_idx], centroid_nm[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    I_K_total += triangle_pair_mfie(k, n_hat_mp[lm_idx],
                        verts_mp[lm_idx], verts_nm[ln_idx],
                        r_fv_mp[lm_idx],  r_fv_nm[ln_idx],
                        +1.0, -1.0, lm[lm_idx], ln[ln_idx], Ap[lm_idx], An_m[ln_idx],
                        nq, weights, bary, nq_near, weights_near, bary_near,
                        ta_mp[lm_idx], ta_nm[ln_idx], is_near);

                    // (m-, n+)
                    me = me_np[ln_idx];
                    dist = norm3(sub3(centroid_mm[lm_idx], centroid_np[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    I_K_total += triangle_pair_mfie(k, n_hat_mm[lm_idx],
                        verts_mm[lm_idx], verts_np[ln_idx],
                        r_fv_mm[lm_idx],  r_fv_np[ln_idx],
                        -1.0, +1.0, lm[lm_idx], ln[ln_idx], Am[lm_idx], An_p[ln_idx],
                        nq, weights, bary, nq_near, weights_near, bary_near,
                        ta_mm[lm_idx], ta_np[ln_idx], is_near);

                    // (m-, n-)
                    me = me_nm[ln_idx];
                    dist = norm3(sub3(centroid_mm[lm_idx], centroid_nm[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    I_K_total += triangle_pair_mfie(k, n_hat_mm[lm_idx],
                        verts_mm[lm_idx], verts_nm[ln_idx],
                        r_fv_mm[lm_idx],  r_fv_nm[ln_idx],
                        -1.0, -1.0, lm[lm_idx], ln[ln_idx], Am[lm_idx], An_m[ln_idx],
                        nq, weights, bary, nq_near, weights_near, bary_near,
                        ta_mm[lm_idx], ta_nm[ln_idx], is_near);

                    Z[m * N + n] = I_K_total;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CFIE single triangle-pair fused kernel  →  (I_A, I_Phi, I_K)
//
// Computes EFIE integrals (I_A, I_Phi) with singularity extraction for near
// pairs, and MFIE K-term (I_K) with higher-order quadrature for near pairs.
// ---------------------------------------------------------------------------
static inline void triangle_pair_cfie(
    double k,
    const v3& n_hat_test,
    const v3 verts_test[3], const v3 verts_src[3],
    const v3& r_fv_test, const v3& r_fv_src,
    double sign_test, double sign_src,
    double l_test, double l_src,
    double A_test, double A_src,
    int nq,       const double* weights,       const double* bary,
    int nq_near,  const double* weights_near,  const double* bary_near,
    double twice_area_test, double twice_area_src,
    bool is_near, double near_threshold,
    cd& I_A_out, cd& I_Phi_out, cd& I_K_out) noexcept
{
    // --- EFIE integrals (same as triangle_pair) ---
    triangle_pair(k, verts_test, verts_src, r_fv_test, r_fv_src,
                  sign_test, sign_src, l_test, l_src, A_test, A_src,
                  nq, weights, bary, twice_area_test, twice_area_src,
                  is_near, near_threshold, I_A_out, I_Phi_out);

    // --- MFIE K-term ---
    I_K_out = triangle_pair_mfie(k, n_hat_test, verts_test, verts_src,
                                  r_fv_test, r_fv_src,
                                  sign_test, sign_src, l_test, l_src, A_test, A_src,
                                  nq, weights, bary, nq_near, weights_near, bary_near,
                                  twice_area_test, twice_area_src, is_near);
}

// ---------------------------------------------------------------------------
// CFIE impedance matrix fill — full N×N, OpenMP parallel, fused EFIE+MFIE
// ---------------------------------------------------------------------------
void fill_impedance_cfie_cpp(
    py::array_t<cd,      py::array::c_style> Z_arr,
    py::array_t<double,  py::array::c_style> vertices_arr,
    py::array_t<int32_t, py::array::c_style> triangles_arr,
    py::array_t<double,  py::array::c_style> tri_normals_arr,
    py::array_t<int32_t, py::array::c_style> t_plus_arr,
    py::array_t<int32_t, py::array::c_style> t_minus_arr,
    py::array_t<int32_t, py::array::c_style> fv_plus_arr,
    py::array_t<int32_t, py::array::c_style> fv_minus_arr,
    py::array_t<double,  py::array::c_style> area_plus_arr,
    py::array_t<double,  py::array::c_style> area_minus_arr,
    py::array_t<double,  py::array::c_style> edge_length_arr,
    py::array_t<double,  py::array::c_style> tri_centroids_arr,
    py::array_t<double,  py::array::c_style> tri_mean_edge_arr,
    py::array_t<double,  py::array::c_style> tri_twice_area_arr,
    py::array_t<double,  py::array::c_style> weights_arr,
    py::array_t<double,  py::array::c_style> bary_arr,
    py::array_t<double,  py::array::c_style> weights_near_arr,
    py::array_t<double,  py::array::c_style> bary_near_arr,
    double k, double eta, double alpha,
    double near_threshold,
    int    quad_order,
    int    num_threads)
{
    cd*            Z               = Z_arr.mutable_data();
    const double*  vertices        = vertices_arr.data();
    const int32_t* triangles       = triangles_arr.data();
    const double*  tri_normals     = tri_normals_arr.data();
    const int32_t* t_plus          = t_plus_arr.data();
    const int32_t* t_minus         = t_minus_arr.data();
    const int32_t* fv_plus         = fv_plus_arr.data();
    const int32_t* fv_minus        = fv_minus_arr.data();
    const double*  area_plus       = area_plus_arr.data();
    const double*  area_minus      = area_minus_arr.data();
    const double*  edge_length     = edge_length_arr.data();
    const double*  tri_centroids   = tri_centroids_arr.data();
    const double*  tri_mean_edge   = tri_mean_edge_arr.data();
    const double*  tri_twice_area  = tri_twice_area_arr.data();
    const double*  weights         = weights_arr.data();
    const double*  bary            = bary_arr.data();
    const double*  weights_near    = weights_near_arr.data();
    const double*  bary_near       = bary_near_arr.data();

    const int N       = static_cast<int>(t_plus_arr.shape(0));
    const int nq      = static_cast<int>(weights_arr.shape(0));
    const int nq_near = static_cast<int>(weights_near_arr.shape(0));

    const cd   prefactor_A   = cd{0.0, k * eta};
    const cd   prefactor_Phi = cd{0.0, -eta / k};
    const double near_thresh_scaled = near_threshold * 3.0;

#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int bm = 0; bm < N; bm += BLOCK_SIZE) {
        const int m_end = std::min(bm + BLOCK_SIZE, N);

        v3     verts_mp[BLOCK_SIZE][3], verts_mm[BLOCK_SIZE][3];
        v3     r_fv_mp[BLOCK_SIZE],     r_fv_mm[BLOCK_SIZE];
        v3     centroid_mp[BLOCK_SIZE], centroid_mm[BLOCK_SIZE];
        v3     n_hat_mp[BLOCK_SIZE],    n_hat_mm[BLOCK_SIZE];
        double ta_mp[BLOCK_SIZE],       ta_mm[BLOCK_SIZE];
        double Ap[BLOCK_SIZE],          Am[BLOCK_SIZE];
        double lm[BLOCK_SIZE];

        for (int m = bm; m < m_end; ++m) {
            int loc = m - bm;
            int tri_mp = t_plus[m];
            load_triangle(vertices, triangles, tri_mp, verts_mp[loc]);
            r_fv_mp[loc]     = load_vertex(vertices, fv_plus[m]);
            centroid_mp[loc] = load_centroid(tri_centroids, tri_mp);
            n_hat_mp[loc]    = load_normal(tri_normals, tri_mp);
            ta_mp[loc]       = tri_twice_area[tri_mp];
            Ap[loc]          = area_plus[m];

            int tri_mm = t_minus[m];
            load_triangle(vertices, triangles, tri_mm, verts_mm[loc]);
            r_fv_mm[loc]     = load_vertex(vertices, fv_minus[m]);
            centroid_mm[loc] = load_centroid(tri_centroids, tri_mm);
            n_hat_mm[loc]    = load_normal(tri_normals, tri_mm);
            ta_mm[loc]       = tri_twice_area[tri_mm];
            Am[loc]          = area_minus[m];
            lm[loc]          = edge_length[m];
        }

        for (int bn = 0; bn < N; bn += BLOCK_SIZE) {
            const int n_end = std::min(bn + BLOCK_SIZE, N);

            v3     verts_np[BLOCK_SIZE][3], verts_nm[BLOCK_SIZE][3];
            v3     r_fv_np[BLOCK_SIZE],     r_fv_nm[BLOCK_SIZE];
            v3     centroid_np[BLOCK_SIZE], centroid_nm[BLOCK_SIZE];
            double ta_np[BLOCK_SIZE],       ta_nm[BLOCK_SIZE];
            double me_np[BLOCK_SIZE],       me_nm[BLOCK_SIZE];
            double An_p[BLOCK_SIZE],        An_m[BLOCK_SIZE];
            double ln[BLOCK_SIZE];

            for (int n = bn; n < n_end; ++n) {
                int loc = n - bn;
                int tri_np = t_plus[n];
                load_triangle(vertices, triangles, tri_np, verts_np[loc]);
                r_fv_np[loc]     = load_vertex(vertices, fv_plus[n]);
                centroid_np[loc] = load_centroid(tri_centroids, tri_np);
                ta_np[loc]       = tri_twice_area[tri_np];
                me_np[loc]       = tri_mean_edge[tri_np];
                An_p[loc]        = area_plus[n];

                int tri_nm = t_minus[n];
                load_triangle(vertices, triangles, tri_nm, verts_nm[loc]);
                r_fv_nm[loc]     = load_vertex(vertices, fv_minus[n]);
                centroid_nm[loc] = load_centroid(tri_centroids, tri_nm);
                ta_nm[loc]       = tri_twice_area[tri_nm];
                me_nm[loc]       = tri_mean_edge[tri_nm];
                An_m[loc]        = area_minus[n];
                ln[loc]          = edge_length[n];
            }

            for (int m = bm; m < m_end; ++m) {
                int lm_idx = m - bm;

                for (int n = bn; n < n_end; ++n) {
                    int ln_idx = n - bn;

                    cd I_A_total{0.0, 0.0}, I_Phi_total{0.0, 0.0}, I_K_total{0.0, 0.0};
                    double dist, me;
                    bool is_near;
                    cd I_A, I_Phi, I_K;

                    // (m+, n+)
                    me = me_np[ln_idx];
                    dist = norm3(sub3(centroid_mp[lm_idx], centroid_np[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    triangle_pair_cfie(k, n_hat_mp[lm_idx],
                        verts_mp[lm_idx], verts_np[ln_idx],
                        r_fv_mp[lm_idx],  r_fv_np[ln_idx],
                        +1.0, +1.0, lm[lm_idx], ln[ln_idx], Ap[lm_idx], An_p[ln_idx],
                        nq, weights, bary, nq_near, weights_near, bary_near,
                        ta_mp[lm_idx], ta_np[ln_idx], is_near, near_threshold,
                        I_A, I_Phi, I_K);
                    I_A_total += I_A; I_Phi_total += I_Phi; I_K_total += I_K;

                    // (m+, n-)
                    me = me_nm[ln_idx];
                    dist = norm3(sub3(centroid_mp[lm_idx], centroid_nm[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    triangle_pair_cfie(k, n_hat_mp[lm_idx],
                        verts_mp[lm_idx], verts_nm[ln_idx],
                        r_fv_mp[lm_idx],  r_fv_nm[ln_idx],
                        +1.0, -1.0, lm[lm_idx], ln[ln_idx], Ap[lm_idx], An_m[ln_idx],
                        nq, weights, bary, nq_near, weights_near, bary_near,
                        ta_mp[lm_idx], ta_nm[ln_idx], is_near, near_threshold,
                        I_A, I_Phi, I_K);
                    I_A_total += I_A; I_Phi_total += I_Phi; I_K_total += I_K;

                    // (m-, n+)
                    me = me_np[ln_idx];
                    dist = norm3(sub3(centroid_mm[lm_idx], centroid_np[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    triangle_pair_cfie(k, n_hat_mm[lm_idx],
                        verts_mm[lm_idx], verts_np[ln_idx],
                        r_fv_mm[lm_idx],  r_fv_np[ln_idx],
                        -1.0, +1.0, lm[lm_idx], ln[ln_idx], Am[lm_idx], An_p[ln_idx],
                        nq, weights, bary, nq_near, weights_near, bary_near,
                        ta_mm[lm_idx], ta_np[ln_idx], is_near, near_threshold,
                        I_A, I_Phi, I_K);
                    I_A_total += I_A; I_Phi_total += I_Phi; I_K_total += I_K;

                    // (m-, n-)
                    me = me_nm[ln_idx];
                    dist = norm3(sub3(centroid_mm[lm_idx], centroid_nm[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    triangle_pair_cfie(k, n_hat_mm[lm_idx],
                        verts_mm[lm_idx], verts_nm[ln_idx],
                        r_fv_mm[lm_idx],  r_fv_nm[ln_idx],
                        -1.0, -1.0, lm[lm_idx], ln[ln_idx], Am[lm_idx], An_m[ln_idx],
                        nq, weights, bary, nq_near, weights_near, bary_near,
                        ta_mm[lm_idx], ta_nm[ln_idx], is_near, near_threshold,
                        I_A, I_Phi, I_K);
                    I_A_total += I_A; I_Phi_total += I_Phi; I_K_total += I_K;

                    cd efie = prefactor_A * I_A_total + prefactor_Phi * I_Phi_total;
                    Z[m * N + n] = alpha * efie + (1.0 - alpha) * eta * I_K_total;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// pybind11 module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_cpp_kernels, m) {
    m.doc() = "C++ EFIE impedance matrix fill kernel (OpenMP + 2D block tiling)";

    m.attr("OMP_ENABLED") = []() -> bool {
#ifdef _OPENMP
        return true;
#else
        return false;
#endif
    }();

    m.def("fill_impedance_cpp", &fill_impedance_cpp,
          py::arg("Z"),
          py::arg("vertices"),
          py::arg("triangles"),
          py::arg("t_plus"),
          py::arg("t_minus"),
          py::arg("fv_plus"),
          py::arg("fv_minus"),
          py::arg("area_plus"),
          py::arg("area_minus"),
          py::arg("edge_length"),
          py::arg("tri_centroids"),
          py::arg("tri_mean_edge"),
          py::arg("tri_twice_area"),
          py::arg("weights"),
          py::arg("bary"),
          py::arg("k"),
          py::arg("eta"),
          py::arg("near_threshold"),
          py::arg("quad_order"),
          py::arg("num_threads") = 0,
          R"pbdoc(
Fill the EFIE impedance matrix Z in-place.

All array arguments must be C-contiguous NumPy arrays with the dtypes
documented in fill_impedance_matrix().  Z is written in-place; the
function returns None.

Parameters
----------
num_threads : int
    Number of OpenMP threads (0 = use OMP_NUM_THREADS or hardware default).
          )pbdoc",
          py::call_guard<py::gil_scoped_release>());

    m.def("fill_impedance_mfie_cpp", &fill_impedance_mfie_cpp,
          py::arg("Z"),
          py::arg("vertices"),
          py::arg("triangles"),
          py::arg("tri_normals"),
          py::arg("t_plus"),
          py::arg("t_minus"),
          py::arg("fv_plus"),
          py::arg("fv_minus"),
          py::arg("area_plus"),
          py::arg("area_minus"),
          py::arg("edge_length"),
          py::arg("tri_centroids"),
          py::arg("tri_mean_edge"),
          py::arg("tri_twice_area"),
          py::arg("weights"),
          py::arg("bary"),
          py::arg("weights_near"),
          py::arg("bary_near"),
          py::arg("k"),
          py::arg("near_threshold"),
          py::arg("quad_order"),
          py::arg("num_threads") = 0,
          R"pbdoc(
Fill the MFIE K-term matrix Z in-place (Gram term added by post_assembly).

tri_normals : float64 (N_t, 3) — outward unit normals per triangle.
weights_near / bary_near : higher-order quadrature for near-field pairs.
          )pbdoc",
          py::call_guard<py::gil_scoped_release>());

    m.def("fill_impedance_cfie_cpp", &fill_impedance_cfie_cpp,
          py::arg("Z"),
          py::arg("vertices"),
          py::arg("triangles"),
          py::arg("tri_normals"),
          py::arg("t_plus"),
          py::arg("t_minus"),
          py::arg("fv_plus"),
          py::arg("fv_minus"),
          py::arg("area_plus"),
          py::arg("area_minus"),
          py::arg("edge_length"),
          py::arg("tri_centroids"),
          py::arg("tri_mean_edge"),
          py::arg("tri_twice_area"),
          py::arg("weights"),
          py::arg("bary"),
          py::arg("weights_near"),
          py::arg("bary_near"),
          py::arg("k"),
          py::arg("eta"),
          py::arg("alpha"),
          py::arg("near_threshold"),
          py::arg("quad_order"),
          py::arg("num_threads") = 0,
          R"pbdoc(
Fill the CFIE impedance matrix Z in-place (fused EFIE + MFIE single pass).

Z_mn = alpha*(jkη*I_A − jη/k*I_Phi) + (1−alpha)*η*I_K
Gram term for MFIE contribution added by post_assembly.
          )pbdoc",
          py::call_guard<py::gil_scoped_release>());
}
