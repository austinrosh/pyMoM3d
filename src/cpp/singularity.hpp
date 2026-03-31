/**
 * singularity.hpp — Inline C++ kernels for EFIE Green's function integrals.
 *
 * Implements singularity extraction following:
 *   - Wilton et al. (1984) IEEE Trans. AP-32(3)
 *   - Graglia (1993) IEEE Trans. AP-41(10)
 *
 * All functions are inline, header-only, and depend only on the C++17 standard
 * library.  No Eigen, no external math libraries.
 *
 * Coordinate convention
 * ---------------------
 * 3-D vectors are represented as std::array<double, 3> (type alias v3).
 * Complex vectors are std::array<std::complex<double>, 3> (type alias cv3).
 * Triangle barycentric coordinates bary[i*3 + {0,1,2}] for quad point i.
 *
 * GPU-readiness note
 * ------------------
 * Every function receives pre-computed arrays (weights, bary, twice_area)
 * rather than calling a Python lookup.  The bary pointer is row-major
 * (nq × 3), matching NumPy's default layout and what a CUDA kernel would
 * receive via cudaMemcpy.
 */

#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <algorithm>   // std::min

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
using v3  = std::array<double, 3>;
using cv3 = std::array<std::complex<double>, 3>;
using cd  = std::complex<double>;

// ---------------------------------------------------------------------------
// Inline 3-D real vector operations
// ---------------------------------------------------------------------------
[[nodiscard]] inline v3 sub3(const v3& a, const v3& b) noexcept {
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}
[[nodiscard]] inline v3 add3(const v3& a, const v3& b) noexcept {
    return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}
[[nodiscard]] inline v3 scale3(double s, const v3& a) noexcept {
    return {s*a[0], s*a[1], s*a[2]};
}
[[nodiscard]] inline double dot3(const v3& a, const v3& b) noexcept {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
[[nodiscard]] inline double norm3sq(const v3& a) noexcept {
    return a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
}
[[nodiscard]] inline double norm3(const v3& a) noexcept {
    return std::sqrt(norm3sq(a));
}
[[nodiscard]] inline v3 cross3(const v3& a, const v3& b) noexcept {
    return { a[1]*b[2]-a[2]*b[1],
             a[2]*b[0]-a[0]*b[2],
             a[0]*b[1]-a[1]*b[0] };
}

// ---------------------------------------------------------------------------
// Barycentric interpolation (inline, called in inner quad loop)
// ---------------------------------------------------------------------------
[[nodiscard]] inline v3 bary_interp(const double* bary_row,
                                     const v3& v0, const v3& v1, const v3& v2) noexcept {
    double L0 = bary_row[0], L1 = bary_row[1], L2 = bary_row[2];
    return { L0*v0[0]+L1*v1[0]+L2*v2[0],
             L0*v0[1]+L1*v1[1]+L2*v2[1],
             L0*v0[2]+L1*v1[2]+L2*v2[2] };
}

// ---------------------------------------------------------------------------
// Analytical integral of 1/|r_obs - r'| over a triangle  (Graglia 1993)
//
// Returns: ∫_T  1/|r_obs - r'|  dS'
// ---------------------------------------------------------------------------
[[nodiscard]] inline double analytical_1_over_R(
    const v3& r_obs,
    const v3& v0, const v3& v1, const v3& v2) noexcept
{
    const v3 verts[3] = {v0, v1, v2};

    // Triangle normal
    v3 n_vec = cross3(sub3(v1, v0), sub3(v2, v0));
    double area2 = norm3(n_vec);
    if (area2 < 1e-30) return 0.0;
    v3 n_hat = scale3(1.0 / area2, n_vec);

    // Signed height of r_obs above triangle plane
    double d = dot3(sub3(r_obs, v0), n_hat);

    // Projection of r_obs onto the triangle plane
    v3 r_proj = sub3(r_obs, scale3(d, n_hat));

    // Distance from r_obs to each vertex
    double R_arr[3];
    for (int i = 0; i < 3; ++i)
        R_arr[i] = norm3(sub3(verts[i], r_obs));

    double result = 0.0;
    double abs_d = std::abs(d);

    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3;
        const v3& va = verts[i];
        const v3& vb = verts[j];

        v3 edge_vec = sub3(vb, va);
        double edge_len = norm3(edge_vec);
        if (edge_len < 1e-30) continue;
        v3 t_hat = scale3(1.0 / edge_len, edge_vec);

        // Outward edge normal in the triangle plane: -(n_hat × t_hat)
        v3 m_hat = scale3(-1.0, cross3(n_hat, t_hat));

        // Verify it points outward (away from opposite vertex)
        const v3& opp = verts[(i + 2) % 3];
        v3 mid = scale3(0.5, add3(va, vb));
        if (dot3(m_hat, sub3(mid, opp)) < 0.0)
            m_hat = scale3(-1.0, m_hat);

        v3 rho_a = sub3(va, r_proj);
        v3 rho_b = sub3(vb, r_proj);

        double rho_0 = dot3(rho_a, m_hat);
        double t_a   = dot3(rho_a, t_hat);
        double t_b   = dot3(rho_b, t_hat);

        double R_a = R_arr[i];
        double R_b = R_arr[j];

        // Logarithmic term
        double arg_num = t_b + R_b;
        double arg_den = t_a + R_a;
        double ln_term = 0.0;
        if (std::abs(arg_den) > 1e-30 && (arg_num / arg_den) > 0.0)
            ln_term = std::log(arg_num / arg_den);

        // Arctan term (solid angle)
        double atan_term = 0.0;
        if (abs_d > 1e-14) {
            double P0_sq = rho_0*rho_0 + d*d;
            double atan_b = std::atan2(rho_0 * t_b, P0_sq + abs_d * R_b);
            double atan_a = std::atan2(rho_0 * t_a, P0_sq + abs_d * R_a);
            atan_term = atan_b - atan_a;
        }

        result += rho_0 * ln_term - abs_d * atan_term;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Scalar Green's function integral with singularity extraction
//
// Returns: ∫_T  exp(-jkR)/(4πR)  dS'   where R = |r_obs - r'|
//
// Parameters
// ----------
// nq        : number of quadrature points
// weights   : pointer to nq weights  (sum = 0.5)
// bary      : pointer to (nq × 3) barycentric coordinates, row-major
// twice_area: 2 × area of source triangle  (= ||(v1-v0)×(v2-v0)||)
// ---------------------------------------------------------------------------
[[nodiscard]] inline cd green_singular(
    double k, const v3& r_obs,
    const v3& v0, const v3& v1, const v3& v2,
    int nq, const double* weights, const double* bary,
    double twice_area, double near_threshold) noexcept
{
    constexpr double four_pi = 4.0 * M_PI;

    // Internal near/far decision: min distance to any source vertex
    double d0 = norm3(sub3(r_obs, v0));
    double d1 = norm3(sub3(r_obs, v1));
    double d2 = norm3(sub3(r_obs, v2));
    double dist = std::min(d0, std::min(d1, d2));

    double e01 = norm3(sub3(v1, v0));
    double e12 = norm3(sub3(v2, v1));
    double e20 = norm3(sub3(v0, v2));
    double mean_edge = (e01 + e12 + e20) / 3.0;

    if (dist > near_threshold * mean_edge * 3.0 && mean_edge > 1e-30) {
        // Far: plain Gauss quadrature
        cd result{0.0, 0.0};
        for (int i = 0; i < nq; ++i) {
            v3 r_prime = bary_interp(bary + i*3, v0, v1, v2);
            double R = norm3(sub3(r_obs, r_prime));
            if (R < 1e-30) R = 1e-30;
            result += weights[i] * std::exp(cd{0.0, -k * R}) / (four_pi * R);
        }
        return result * twice_area;
    }

    // Near/self: singularity extraction  g = 1/(4πR) + [g − 1/(4πR)]
    double I_static = analytical_1_over_R(r_obs, v0, v1, v2) / four_pi;

    cd I_smooth{0.0, 0.0};
    for (int i = 0; i < nq; ++i) {
        v3 r_prime = bary_interp(bary + i*3, v0, v1, v2);
        double R = norm3(sub3(r_obs, r_prime));
        cd remainder;
        if (R < 1e-30) {
            // limit as R→0: [exp(-jkR) − 1]/(4πR) → −jk/(4π)
            remainder = cd{0.0, -k / four_pi};
        } else {
            // [exp(-jkR) − 1]/(4πR) is smooth
            remainder = (std::exp(cd{0.0, -k * R}) - 1.0) / (four_pi * R);
        }
        I_smooth += weights[i] * remainder;
    }
    return cd{I_static, 0.0} + I_smooth * twice_area;
}

// ---------------------------------------------------------------------------
// Vector rho·g integral with singularity extraction
//
// Returns: ∫_T  (r' − r_fv_src) · exp(-jkR)/(4πR)  dS'
// ---------------------------------------------------------------------------
[[nodiscard]] inline cv3 rho_green_singular(
    double k, const v3& r_obs,
    const v3& v0, const v3& v1, const v3& v2,
    const v3& r_fv_src,
    int nq, const double* weights, const double* bary,
    double twice_area, double near_threshold) noexcept
{
    constexpr double four_pi = 4.0 * M_PI;

    double d0 = norm3(sub3(r_obs, v0));
    double d1 = norm3(sub3(r_obs, v1));
    double d2 = norm3(sub3(r_obs, v2));
    double dist = std::min(d0, std::min(d1, d2));

    double e01 = norm3(sub3(v1, v0));
    double e12 = norm3(sub3(v2, v1));
    double e20 = norm3(sub3(v0, v2));
    double mean_edge = (e01 + e12 + e20) / 3.0;

    if (dist > near_threshold * mean_edge * 3.0 && mean_edge > 1e-30) {
        // Far: plain Gauss quadrature
        cv3 result{};
        for (int i = 0; i < nq; ++i) {
            v3 r_prime = bary_interp(bary + i*3, v0, v1, v2);
            v3 rho = sub3(r_prime, r_fv_src);
            double R = norm3(sub3(r_obs, r_prime));
            if (R < 1e-30) R = 1e-30;
            cd g_val = std::exp(cd{0.0, -k * R}) / (four_pi * R);
            result[0] += weights[i] * rho[0] * g_val;
            result[1] += weights[i] * rho[1] * g_val;
            result[2] += weights[i] * rho[2] * g_val;
        }
        result[0] *= twice_area;
        result[1] *= twice_area;
        result[2] *= twice_area;
        return result;
    }

    // Near/self: singularity extraction
    // (1) Analytical: ∫ 1/R dS'
    double I_1_over_R = analytical_1_over_R(r_obs, v0, v1, v2);

    // (2) Quadrature for (r'−r_obs)/R  — bounded (magnitude ≤ 1 everywhere)
    v3 I_Rhat{};
    for (int i = 0; i < nq; ++i) {
        v3 r_prime = bary_interp(bary + i*3, v0, v1, v2);
        v3 diff = sub3(r_prime, r_obs);
        double R = norm3(diff);
        if (R > 1e-30) {
            I_Rhat[0] += weights[i] * diff[0] / R;
            I_Rhat[1] += weights[i] * diff[1] / R;
            I_Rhat[2] += weights[i] * diff[2] / R;
        }
    }
    I_Rhat = scale3(twice_area, I_Rhat);

    // Singular part: (1/4π)[I_Rhat + (r_obs − r_fv) · I_1_over_R]
    v3 r_obs_fv = sub3(r_obs, r_fv_src);
    cv3 I_singular;
    for (int c = 0; c < 3; ++c)
        I_singular[c] = (I_Rhat[c] + r_obs_fv[c] * I_1_over_R) / four_pi;

    // (3) Smooth remainder: ∫ rho · [g − 1/(4πR)] dS'
    cv3 I_smooth{};
    for (int i = 0; i < nq; ++i) {
        v3 r_prime = bary_interp(bary + i*3, v0, v1, v2);
        v3 rho = sub3(r_prime, r_fv_src);
        double R = norm3(sub3(r_obs, r_prime));
        cd remainder;
        if (R < 1e-30) {
            remainder = cd{0.0, -k / four_pi};
        } else {
            remainder = (std::exp(cd{0.0, -k * R}) - 1.0) / (four_pi * R);
        }
        I_smooth[0] += weights[i] * rho[0] * remainder;
        I_smooth[1] += weights[i] * rho[1] * remainder;
        I_smooth[2] += weights[i] * rho[2] * remainder;
    }

    return { I_singular[0] + I_smooth[0] * twice_area,
             I_singular[1] + I_smooth[1] * twice_area,
             I_singular[2] + I_smooth[2] * twice_area };
}
