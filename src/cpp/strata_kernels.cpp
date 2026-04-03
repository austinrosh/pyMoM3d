/**
 * strata_kernels.cpp
 *
 * pybind11 wrapper for the Strata multilayer Green's function library.
 * https://github.com/modelics/strata
 *
 * Exposes batched evaluation of the smooth correction G_ML - G_fs for both
 * the scalar potential (G_phi) and dyadic (G̅) components, matching the
 * GreensBackend interface consumed by MultilayerEFIEOperator.
 *
 * Strata's ComputeMGF always returns the FULL multilayer GF (G_ML).
 * This wrapper analytically subtracts the free-space term G_fs so that
 * the returned values are the smooth correction G_ML - G_fs, which is
 * what the MoM operator expects (the singular G_fs is handled separately
 * via Graglia singularity extraction).
 *
 * Formulation: Michalski & Zheng Formulation-C (MGF.hpp).
 *
 * Build (from repo root, after installing Strata):
 *     venv/bin/python build_cpp.py build_ext --inplace --with-strata=/path/to/strata
 *
 * Usage from Python:
 *     from pyMoM3d.greens.layered import strata_kernels as sk
 *     model = sk.make_model(layers, ..., frequency, z_src, z_obs, k_src_re, k_src_im)
 *     G_phi = sk.scalar_G_smooth(model, r_obs, r_src)   # (N,) complex128
 *     G_dyd = sk.dyadic_G_smooth(model, r_obs, r_src)   # (N,3,3) complex128
 */

#include <array>
#include <complex>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "MGF.hpp"
#include "layers.hpp"
#include "singularity.hpp"  // Graglia singularity extraction (header-only)

namespace py = pybind11;
using cdouble = std::complex<double>;

static constexpr double PI = 3.14159265358979323846;
static constexpr double R_MIN = 1.0e-30;  // clamp to avoid division by zero


// ── Free-space Green's function helpers ──────────────────────────────────────
//
// These compute the free-space GF in the source-layer medium (wavenumber k_src)
// so we can subtract it from the full multilayer GF returned by Strata.

/**
 * Scalar free-space GF: g(R) = exp(-jkR) / (4πR)
 */
inline cdouble free_space_scalar(cdouble k, double R) {
    if (R < R_MIN) R = R_MIN;
    return std::exp(-cdouble(0, 1) * k * R) / (4.0 * PI * R);
}

/**
 * Dyadic free-space GF (Chew, "Waves and Fields"):
 *   G̅_fs = g(R) · { [1 + 1/(jkR) - 1/(kR)²]·I
 *                   + [-1 - 3/(jkR) + 3/(kR)²]·r̂r̂ }
 *
 * Writes 9 components into G_fs[row*3 + col], row-major.
 */
inline void free_space_dyadic(cdouble k, double dx, double dy, double dz,
                               std::array<cdouble, 9>& G_fs) {
    double R = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (R < R_MIN) R = R_MIN;

    cdouble g = std::exp(-cdouble(0, 1) * k * R) / (4.0 * PI * R);
    cdouble kR = k * R;
    cdouble inv_jkR = 1.0 / (cdouble(0, 1) * kR);
    cdouble inv_kR2 = 1.0 / (kR * kR);

    cdouble A = g * (1.0 + inv_jkR - inv_kR2);      // coefficient for I
    cdouble B = g * (-1.0 - 3.0 * inv_jkR + 3.0 * inv_kR2);  // coeff for r̂r̂

    // Unit vector components
    double rhat[3] = {dx / R, dy / R, dz / R};

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            cdouble Irc = (r == c) ? 1.0 : 0.0;
            G_fs[r * 3 + c] = A * Irc + B * rhat[r] * rhat[c];
        }
    }
}

/**
 * Gradient of scalar free-space GF:
 *   ∇g = g(R) · (-jk - 1/R) · (r - r') / R
 *
 * Returns 3-component gradient w.r.t. observation point r.
 */
inline void free_space_grad_scalar(cdouble k, double dx, double dy, double dz,
                                    std::array<cdouble, 3>& grad_fs) {
    double R = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (R < R_MIN) R = R_MIN;

    cdouble g = std::exp(-cdouble(0, 1) * k * R) / (4.0 * PI * R);
    cdouble factor = g * (-cdouble(0, 1) * k - 1.0 / R) / R;

    grad_fs[0] = factor * dx;
    grad_fs[1] = factor * dy;
    grad_fs[2] = factor * dz;
}


// ── StrataModel ───────────────────────────────────────────────────────────────
//
// Holds a fully configured MGF object (which internally owns its LayerManager)
// plus the source-layer wavenumber needed for free-space subtraction.

class StrataModel {
public:
    MGF mgf;
    cdouble k_src;     // source-layer wavenumber for G_fs subtraction
    cdouble eps_r_src; // source-layer relative permittivity (for G_phi → g conversion)
    StrataModel() : k_src(0.0), eps_r_src(1.0) {}
};


// ── make_model ────────────────────────────────────────────────────────────────

/**
 * Build a StrataModel from Python-side layer parameters.
 *
 * layers    : list of [zmin_m, zmax_m, epsr_re, epsr_im, mur, sigma_Sm, sigmamu]
 *             Interior finite layers only (halfspaces passed separately).
 *             All lengths in metres, conductivity in S/m.
 * epsr_top, mur_top, sigma_top, pec_top  : top halfspace
 * epsr_bot, mur_bot, sigma_bot, pec_bot  : bottom halfspace
 * frequency : Hz
 * z_src     : z-coordinate of any point in the source layer (used to locate
 *             the layer index via LayerManager::FindLayer).
 * z_obs     : z-coordinate of any point in the observation layer.
 * k_src_re, k_src_im : real and imaginary parts of the source-layer wavenumber
 *                       (rad/m), used for free-space GF subtraction.
 * eps_r_src_re, eps_r_src_im : real and imaginary parts of the source-layer
 *                               effective relative permittivity.  Strata's G_phi
 *                               returns g/ε_r (Formulation-C convention); we
 *                               multiply by ε_r before subtracting g_fs.
 * method    : "dcim"       → MGF_DCIM (default, fast precomputed images)
 *             "integrate"  → MGF_INTEGRATE (exact, slow, useful for validation)
 *             "quasistatic"→ MGF_QUASISTATIC
 */
StrataModel make_model(
    const std::vector<std::array<double, 7>>& layers,
    double epsr_top, double mur_top, double sigma_top, bool pec_top,
    double epsr_bot, double mur_bot, double sigma_bot, bool pec_bot,
    double frequency,
    double z_src, double z_obs,
    double k_src_re, double k_src_im,
    double eps_r_src_re, double eps_r_src_im,
    const std::string& method = "dcim"
) {
    // ── Build LayerManager ──────────────────────────────────────────────────
    LayerManager lm;

    for (const auto& l : layers) {
        double   zmin    = l[0];
        double   zmax    = l[1];
        cdouble  epsr    = cdouble(l[2], l[3]);
        double   mur     = l[4];
        double   sigma   = l[5];
        double   sigmamu = l[6];
        lm.AddLayer(zmin, zmax, epsr, mur, sigma, sigmamu);
    }

    lm.SetHalfspaces(epsr_top, mur_top, sigma_top,
                     epsr_bot, mur_bot, sigma_bot,
                     pec_top, pec_bot);
    lm.ProcessLayers(frequency);

    // Register source/observation z-coordinates so Strata can set up the
    // spectral sampling grid for DCIM image generation.
    std::vector<double> z_nodes = {z_src};
    if (std::abs(z_obs - z_src) > 1e-15)
        z_nodes.push_back(z_obs);
    lm.InsertNodes_z(z_nodes);

    // ── MGF settings ───────────────────────────────────────────────────────
    MGF_settings s;

    if (method == "integrate")
        s.method = MGF_INTEGRATE;
    else if (method == "quasistatic")
        s.method = MGF_QUASISTATIC;
    else
        s.method = MGF_DCIM;          // default: DCIM — production speed

    s.extract_singularities = false;  // we subtract G_fs analytically below
    s.verbose               = false;  // suppress Strata stdout logging

    // ── Initialise and configure MGF ────────────────────────────────────────
    StrataModel model;
    model.k_src     = cdouble(k_src_re, k_src_im);
    model.eps_r_src = cdouble(eps_r_src_re, eps_r_src_im);
    model.mgf.Initialize(frequency, lm, s);

    int src_idx = lm.FindLayer(z_src);
    int obs_idx = lm.FindLayer(z_obs);
    model.mgf.SetLayers(src_idx, obs_idx);

    return model;
}


// ── Batched evaluation ────────────────────────────────────────────────────────

/**
 * Scalar potential smooth correction G_phi(r, r') − G_phi_fs(r, r').
 *
 * Strata returns the full multilayer G_phi; we subtract the free-space
 * scalar GF analytically.
 *
 * r_obs, r_src : (N, 3) float64 arrays (x, y, z in metres).
 * Returns (N,) complex128.
 */
py::array_t<cdouble> scalar_G_smooth(
    StrataModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_src
) {
    auto obs = r_obs.unchecked<2>();   // shape (N, 3)
    auto src = r_src.unchecked<2>();
    const ssize_t N = obs.shape(0);

    auto result = py::array_t<cdouble>(N);
    auto res    = result.mutable_unchecked<1>();

    std::array<cdouble, 9> G;
    cdouble G_phi;
    const cdouble k       = model.k_src;
    const cdouble eps_r   = model.eps_r_src;

    for (ssize_t i = 0; i < N; ++i) {
        const double dx = obs(i, 0) - src(i, 0);
        const double dy = obs(i, 1) - src(i, 1);
        const double dz = obs(i, 2) - src(i, 2);
        const double z  = obs(i, 2);
        const double zp = src(i, 2);

        // Strata returns G_phi = g/ε_r (Formulation-C convention).
        // Convert to our convention: g = G_phi * ε_r, then subtract g_fs.
        model.mgf.ComputeMGF(dx, dy, z, zp, G, G_phi);
        if (std::isnan(G_phi.real()) || std::isnan(G_phi.imag())) G_phi = {0.0, 0.0};

        double R = std::sqrt(dx*dx + dy*dy + dz*dz);
        res(i) = G_phi * eps_r - free_space_scalar(k, R);
    }
    return result;
}

/**
 * Vector potential smooth correction G_A(r, r') − g_fs(R) · I.
 *
 * Strata's G_dyadic is the Formulation-C vector potential G_A.
 * In free space G_A = g_fs · I (scalar GF times identity), so the smooth
 * correction G_A - g_fs · I vanishes.  In layered media the off-diagonal
 * and modified diagonal components capture the substrate / ground effects.
 *
 * r_obs, r_src : (N, 3) float64.
 * Returns (N, 3, 3) complex128.  Row-major: result[i, row, col].
 */
py::array_t<cdouble> dyadic_G_smooth(
    StrataModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_src
) {
    auto obs = r_obs.unchecked<2>();
    auto src = r_src.unchecked<2>();
    const ssize_t N = obs.shape(0);

    auto result = py::array_t<cdouble>({N, (ssize_t)3, (ssize_t)3});
    auto res    = result.mutable_unchecked<3>();

    std::array<cdouble, 9> G;
    cdouble G_phi;
    const cdouble k = model.k_src;

    for (ssize_t i = 0; i < N; ++i) {
        const double dx = obs(i, 0) - src(i, 0);
        const double dy = obs(i, 1) - src(i, 1);
        const double dz = obs(i, 2) - src(i, 2);
        const double z  = obs(i, 2);
        const double zp = src(i, 2);

        // Full multilayer vector potential from Strata
        model.mgf.ComputeMGF(dx, dy, z, zp, G, G_phi);
        for (auto& v : G) if (std::isnan(v.real()) || std::isnan(v.imag())) v = {0.0, 0.0};

        // Free-space vector potential: g_fs(R) * I
        double R = std::sqrt(dx*dx + dy*dy + dz*dz);
        cdouble g_fs = free_space_scalar(k, R);

        // Subtract: G_A_smooth = G_A_ML - g_fs * I
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                res(i, r, c) = G[r * 3 + c] - ((r == c) ? g_fs : cdouble{0.0, 0.0});
    }
    return result;
}

/**
 * Gradient of scalar smooth correction ∇[G_phi - G_phi_fs] w.r.t. r.
 *
 * Uses central finite differences on Strata's full G_phi, then subtracts
 * the analytically computed ∇G_fs.
 *
 * r_obs, r_src : (N, 3) float64.
 * Returns (N, 3) complex128.
 */
py::array_t<cdouble> grad_G_smooth(
    StrataModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_src,
    double h = 1.0e-9   // FD step (m); 1 nm suits chip-scale geometries
) {
    auto obs_arr = r_obs.unchecked<2>();
    auto src_arr = r_src.unchecked<2>();
    const ssize_t N = obs_arr.shape(0);

    auto result = py::array_t<cdouble>({N, (ssize_t)3});
    auto res    = result.mutable_unchecked<2>();

    std::array<cdouble, 9> G_dummy;
    cdouble Gp, Gm;
    std::array<cdouble, 3> grad_fs;
    const cdouble k = model.k_src;

    for (ssize_t i = 0; i < N; ++i) {
        const double x  = obs_arr(i, 0), xp = src_arr(i, 0);
        const double y  = obs_arr(i, 1), yp = src_arr(i, 1);
        const double z  = obs_arr(i, 2), zp = src_arr(i, 2);

        // FD gradient of full multilayer G_phi (Strata returns g/ε_r,
        // so multiply by ε_r to get ∇g_ML before subtracting ∇g_fs)
        // dG/dx
        model.mgf.ComputeMGF(x + h - xp, y - yp, z,     zp, G_dummy, Gp);
        model.mgf.ComputeMGF(x - h - xp, y - yp, z,     zp, G_dummy, Gm);
        res(i, 0) = (Gp - Gm) / (2.0 * h) * model.eps_r_src;

        // dG/dy
        model.mgf.ComputeMGF(x - xp, y + h - yp, z,     zp, G_dummy, Gp);
        model.mgf.ComputeMGF(x - xp, y - h - yp, z,     zp, G_dummy, Gm);
        res(i, 1) = (Gp - Gm) / (2.0 * h) * model.eps_r_src;

        // dG/dz
        model.mgf.ComputeMGF(x - xp, y - yp,     z + h, zp, G_dummy, Gp);
        model.mgf.ComputeMGF(x - xp, y - yp,     z - h, zp, G_dummy, Gm);
        res(i, 2) = (Gp - Gm) / (2.0 * h) * model.eps_r_src;

        // Subtract analytical ∇G_fs
        free_space_grad_scalar(k, x - xp, y - yp, z - zp, grad_fs);
        res(i, 0) -= grad_fs[0];
        res(i, 1) -= grad_fs[1];
        res(i, 2) -= grad_fs[2];
    }
    return result;
}


// ── Multilayer EFIE impedance matrix fill ────────────────────────────────────
//
// C++-accelerated version of MultilayerEFIEOperator.compute_pair_numpy().
// Calls Strata's ComputeMGF directly from C++ — no Python roundtrip.
// Reuses singularity extraction from singularity.hpp (identical to free-space).

using cd = std::complex<double>;

static constexpr int ML_BLOCK_SIZE = 32;   // 2-D tile side length

// ---------------------------------------------------------------------------
// Helper: load a triangle's 3 vertices from flat arrays
// ---------------------------------------------------------------------------
static inline void ml_load_triangle(
    const double* vertices, const int32_t* triangles,
    int tri, v3 out[3]) noexcept
{
    for (int v = 0; v < 3; ++v) {
        int idx = triangles[tri * 3 + v];
        out[v] = {vertices[idx*3], vertices[idx*3+1], vertices[idx*3+2]};
    }
}

static inline v3 ml_load_vertex(const double* vertices, int idx) noexcept {
    return {vertices[idx*3], vertices[idx*3+1], vertices[idx*3+2]};
}

static inline v3 ml_load_centroid(const double* tri_centroids, int tri) noexcept {
    return {tri_centroids[tri*3], tri_centroids[tri*3+1], tri_centroids[tri*3+2]};
}

// ---------------------------------------------------------------------------
// Multilayer triangle-pair kernel
//
// Same mathematical structure as the free-space triangle_pair() in
// mom_kernel.cpp, but:
//   Near-field: Graglia singularity extraction (free-space 1/R)
//               + smooth correction inner loop via Strata ComputeMGF
//   Far-field:  Full G_ML from Strata (no free-space computation needed)
// ---------------------------------------------------------------------------
static inline void triangle_pair_multilayer(
    double k,
    const v3 verts_test[3], const v3 verts_src[3],
    const v3& r_fv_test, const v3& r_fv_src,
    double sign_test, double sign_src,
    double l_test, double l_src,
    double A_test, double A_src,
    int nq, const double* weights, const double* bary,
    double twice_area_test, double twice_area_src,
    bool is_near, double near_threshold,
    MGF& mgf,
    cdouble k_src,
    cdouble eps_r_src,
    bool same_layer,
    cd& I_A_out, cd& I_Phi_out) noexcept
{
    cd I_A_raw{0.0, 0.0};
    cd I_Phi_raw{0.0, 0.0};

    for (int i = 0; i < nq; ++i) {
        v3 r_obs = bary_interp(bary + i*3,
                               verts_test[0], verts_test[1], verts_test[2]);
        v3 rho_test = sub3(r_obs, r_fv_test);

        cd g_int;
        cv3 rho_src_g_int{};

        if (same_layer && is_near) {
            // ── Singular term (Graglia) — free-space, unchanged ──
            g_int = green_singular(k, r_obs,
                                   verts_src[0], verts_src[1], verts_src[2],
                                   nq, weights, bary, twice_area_src,
                                   near_threshold);
            rho_src_g_int = rho_green_singular(k, r_obs,
                                               verts_src[0], verts_src[1],
                                               verts_src[2], r_fv_src,
                                               nq, weights, bary,
                                               twice_area_src, near_threshold);

            // ── Smooth correction: (G_ML - G_fs) via Strata ──
            //
            // G_ML - G_fs is smooth (singularities cancel), but both
            // terms individually diverge as 1/R.  For R ≈ 0 (self-
            // triangle, coincident quad points) we cannot evaluate the
            // subtraction numerically.  Instead we:
            //   1. Accumulate the smooth correction from non-singular points.
            //   2. For singular points (R < threshold), use the weighted
            //      average of the non-singular corrections as an estimate.
            //      This works because G_ML - G_fs varies slowly.
            cd g_smooth{0.0, 0.0};
            cv3 rho_g_smooth{};
            std::array<cdouble, 9> G_dyadic;
            cdouble G_phi;

            // Accumulate weighted G_A correction for singular-point interpolation
            std::array<cdouble, 9> ga_corr_sum{};

            // Sanitize NaN from DCIM z-coupling components (dual-PEC workaround)
            auto sanitize_nan = [](std::array<cdouble, 9>& G, cdouble& Gp) {
                for (auto& v : G) if (std::isnan(v.real()) || std::isnan(v.imag())) v = {0.0, 0.0};
                if (std::isnan(Gp.real()) || std::isnan(Gp.imag())) Gp = {0.0, 0.0};
            };
            constexpr double R_SMOOTH_THRESH = 1e-10;
            double w_regular = 0.0;   // sum of weights for non-singular points
            double w_singular = 0.0;  // sum of weights for singular points
            // Singular point rho vectors (need for rho*g correction)
            v3 rho_singular[16];  // max nq we'll see
            double w_singular_arr[16];
            int n_singular = 0;

            for (int j = 0; j < nq; ++j) {
                v3 r_src = bary_interp(bary + j*3,
                                       verts_src[0], verts_src[1], verts_src[2]);

                double dx = r_obs[0] - r_src[0];
                double dy = r_obs[1] - r_src[1];
                double dz = r_obs[2] - r_src[2];
                double R  = std::sqrt(dx*dx + dy*dy + dz*dz);

                if (R < R_SMOOTH_THRESH) {
                    // Record singular point for later interpolation
                    w_singular += weights[j];
                    rho_singular[n_singular] = sub3(r_src, r_fv_src);
                    w_singular_arr[n_singular] = weights[j];
                    n_singular++;
                    continue;
                }

                mgf.ComputeMGF(dx, dy, r_obs[2], r_src[2], G_dyadic, G_phi);
                sanitize_nan(G_dyadic, G_phi);

                // Scalar potential: Formulation-C G_phi = g/ε_r → g_ML = G_phi * ε_r
                cd g_fs_val = free_space_scalar(cdouble(k, 0.0), R);
                cd g_corr = G_phi * eps_r_src - g_fs_val;

                g_smooth += weights[j] * g_corr;
                w_regular += weights[j];

                // Vector potential: use dyadic G_A correction
                // G_A_corr[i][j] = G_dyadic[i*3+j] - δ_ij * g_fs
                // In free space G_A = g_fs * I, so the correction vanishes.
                v3 rho = sub3(r_src, r_fv_src);
                for (int ii = 0; ii < 3; ++ii) {
                    cd ga_rho_ii{0.0, 0.0};
                    for (int jj = 0; jj < 3; ++jj) {
                        cd ga_corr_ij = G_dyadic[ii*3+jj]
                                      - ((ii == jj) ? g_fs_val : cd{0.0, 0.0});
                        ga_corr_sum[ii*3+jj] += weights[j] * ga_corr_ij;
                        ga_rho_ii += ga_corr_ij * rho[jj];
                    }
                    rho_g_smooth[ii] += weights[j] * ga_rho_ii;
                }
            }

            // Interpolate: estimate corrections at singular points using the
            // weighted average from regular points.
            if (n_singular > 0 && w_regular > 0.0) {
                cd g_corr_avg = g_smooth / w_regular;
                g_smooth += w_singular * g_corr_avg;

                // Average dyadic G_A correction for singular-point interpolation
                std::array<cdouble, 9> ga_corr_avg{};
                for (int p = 0; p < 9; ++p)
                    ga_corr_avg[p] = ga_corr_sum[p] / w_regular;

                for (int s = 0; s < n_singular; ++s) {
                    for (int ii = 0; ii < 3; ++ii) {
                        cd ga_rho_ii{0.0, 0.0};
                        for (int jj = 0; jj < 3; ++jj)
                            ga_rho_ii += ga_corr_avg[ii*3+jj] * rho_singular[s][jj];
                        rho_g_smooth[ii] += w_singular_arr[s] * ga_rho_ii;
                    }
                }
            }

            // Add smooth correction (scaled by twice_area_src)
            g_int += g_smooth * twice_area_src;
            rho_src_g_int[0] += rho_g_smooth[0] * twice_area_src;
            rho_src_g_int[1] += rho_g_smooth[1] * twice_area_src;
            rho_src_g_int[2] += rho_g_smooth[2] * twice_area_src;

        } else {
            // ── Far field: use full G_ML directly (no singularity) ──
            g_int = cd{0.0, 0.0};
            std::array<cdouble, 9> G_dyadic;
            cdouble G_phi;

            for (int j = 0; j < nq; ++j) {
                v3 r_src = bary_interp(bary + j*3,
                                       verts_src[0], verts_src[1], verts_src[2]);

                double dx = r_obs[0] - r_src[0];
                double dy = r_obs[1] - r_src[1];

                mgf.ComputeMGF(dx, dy, r_obs[2], r_src[2], G_dyadic, G_phi);
                // Sanitize NaN from DCIM (dual-PEC workaround)
                for (auto& v : G_dyadic) if (std::isnan(v.real()) || std::isnan(v.imag())) v = {0.0, 0.0};
                if (std::isnan(G_phi.real()) || std::isnan(G_phi.imag())) G_phi = {0.0, 0.0};

                // Scalar potential: full G_ML = G_phi * ε_r (Formulation-C)
                cd g_ml = G_phi * eps_r_src;
                g_int += weights[j] * g_ml;

                // Vector potential: use full dyadic G_A from Strata
                v3 rho = sub3(r_src, r_fv_src);
                for (int ii = 0; ii < 3; ++ii) {
                    cd ga_rho_ii{0.0, 0.0};
                    for (int jj = 0; jj < 3; ++jj)
                        ga_rho_ii += G_dyadic[ii*3+jj] * rho[jj];
                    rho_src_g_int[ii] += weights[j] * ga_rho_ii;
                }
            }

            g_int              *= twice_area_src;
            rho_src_g_int[0]   *= twice_area_src;
            rho_src_g_int[1]   *= twice_area_src;
            rho_src_g_int[2]   *= twice_area_src;
        }

        I_Phi_raw += weights[i] * g_int;

        cd dot_val = (rho_test[0] * rho_src_g_int[0]
                    + rho_test[1] * rho_src_g_int[1]
                    + rho_test[2] * rho_src_g_int[2]);
        I_A_raw += weights[i] * dot_val;
    }

    I_A_raw   *= twice_area_test;
    I_Phi_raw *= twice_area_test;

    double scale_A   = (sign_test * l_test / (2.0 * A_test))
                     * (sign_src  * l_src  / (2.0 * A_src));
    double scale_Phi = (sign_test * l_test / A_test)
                     * (sign_src  * l_src  / A_src);

    I_A_out   = I_A_raw   * scale_A;
    I_Phi_out = I_Phi_raw * scale_Phi;
}

// ---------------------------------------------------------------------------
// Thread safety test: verify MGF copies produce identical results
// ---------------------------------------------------------------------------
bool test_mgf_thread_safety(StrataModel& model, int num_threads) {
    if (num_threads < 2) num_threads = 2;

    // Reference: single-threaded evaluation at a set of test points
    const int N_test = 20;
    std::vector<cdouble> ref_results(N_test);

    for (int i = 0; i < N_test; ++i) {
        double dx = 0.001 * (i + 1);  // 1mm to 20mm separation
        double dy = 0.0005 * i;
        double z  = 0.5e-6;           // 0.5 μm above interface
        double zp = 0.5e-6;
        std::array<cdouble, 9> G;
        cdouble G_phi;
        model.mgf.ComputeMGF(dx, dy, z, zp, G, G_phi);
        ref_results[i] = G_phi;
    }

    // Create per-thread copies
    std::vector<MGF> copies(num_threads, model.mgf);

    // Evaluate from multiple threads
    std::vector<cdouble> par_results(num_threads * N_test);
    bool pass = true;

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < N_test; ++i) {
            double dx = 0.001 * (i + 1);
            double dy = 0.0005 * i;
            double z  = 0.5e-6;
            double zp = 0.5e-6;
            std::array<cdouble, 9> G;
            cdouble G_phi;
            copies[t].ComputeMGF(dx, dy, z, zp, G, G_phi);
            par_results[t * N_test + i] = G_phi;
        }
    }

    // Compare
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < N_test; ++i) {
            cdouble diff = par_results[t * N_test + i] - ref_results[i];
            if (std::abs(diff) > 1e-14 * std::abs(ref_results[i])) {
                pass = false;
            }
        }
    }
    return pass;
}

// ---------------------------------------------------------------------------
// Main multilayer impedance matrix fill
// ---------------------------------------------------------------------------
void fill_impedance_multilayer_cpp(
    py::array_t<cd, py::array::c_style>      Z_arr,
    py::array_t<double,  py::array::c_style>  vertices_arr,
    py::array_t<int32_t, py::array::c_style>  triangles_arr,
    py::array_t<int32_t, py::array::c_style>  t_plus_arr,
    py::array_t<int32_t, py::array::c_style>  t_minus_arr,
    py::array_t<int32_t, py::array::c_style>  fv_plus_arr,
    py::array_t<int32_t, py::array::c_style>  fv_minus_arr,
    py::array_t<double,  py::array::c_style>  area_plus_arr,
    py::array_t<double,  py::array::c_style>  area_minus_arr,
    py::array_t<double,  py::array::c_style>  edge_length_arr,
    py::array_t<double,  py::array::c_style>  tri_centroids_arr,
    py::array_t<double,  py::array::c_style>  tri_mean_edge_arr,
    py::array_t<double,  py::array::c_style>  tri_twice_area_arr,
    py::array_t<double,  py::array::c_style>  weights_arr,
    py::array_t<double,  py::array::c_style>  bary_arr,
    double k, double eta,
    double near_threshold,
    int    quad_order,
    StrataModel& strata_model,
    int    num_threads,
    // Cross-layer support (optional)
    py::array_t<int32_t, py::array::c_style>  tri_layer_idx_arr,
    py::list                                   extra_models,
    py::array_t<int32_t, py::array::c_style>  model_lookup_arr,
    bool a_only = false)
{
    // --- Obtain raw pointers ---
    cd*            Z              = Z_arr.mutable_data();
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
    const double*  bary           = bary_arr.data();

    const int N  = static_cast<int>(t_plus_arr.shape(0));
    const int nq = static_cast<int>(weights_arr.shape(0));

    const cd   prefactor_A   = cd{0.0, k * eta};
    const cd   prefactor_Phi = cd{0.0, -eta / k};
    const double near_thresh_scaled = near_threshold * 3.0;

    // --- Cross-layer setup ---
    const bool has_cross_layer = (tri_layer_idx_arr.size() > 0);
    const int32_t* tri_layer = has_cross_layer ? tri_layer_idx_arr.data() : nullptr;
    const int32_t* model_lut = has_cross_layer ? model_lookup_arr.data() : nullptr;
    const int n_extra = static_cast<int>(extra_models.size());
    const int n_models = 1 + n_extra;  // model 0 = strata_model

    // Determine M (number of unique layers) from model_lookup dimensions
    int M = 0;
    if (has_cross_layer && model_lookup_arr.size() > 0) {
        // model_lookup is flat (M*M,); M = sqrt(size)
        M = static_cast<int>(std::round(std::sqrt(
            static_cast<double>(model_lookup_arr.size()))));
    }

    // Extract per-model parameters
    std::vector<cdouble> model_eps_r(n_models);
    std::vector<cdouble> model_k_src(n_models);
    model_eps_r[0] = strata_model.eps_r_src;
    model_k_src[0] = strata_model.k_src;
    for (int m = 0; m < n_extra; ++m) {
        auto& em = extra_models[m].cast<StrataModel&>();
        model_eps_r[m + 1] = em.eps_r_src;
        model_k_src[m + 1] = em.k_src;
    }

    // --- Per-thread MGF copies for thread safety ---
    int actual_threads = 1;
#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    actual_threads = omp_get_max_threads();
#endif

    // thread_mgfs[tid * n_models + model_idx]
    std::vector<MGF> thread_mgfs(actual_threads * n_models);
    for (int t = 0; t < actual_threads; ++t)
        thread_mgfs[t * n_models + 0] = strata_model.mgf;
    for (int m = 0; m < n_extra; ++m) {
        auto& em = extra_models[m].cast<StrataModel&>();
        for (int t = 0; t < actual_threads; ++t)
            thread_mgfs[t * n_models + (m + 1)] = em.mgf;
    }

    // --- Block-tiled loop (identical structure to mom_kernel.cpp) ---
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int bm = 0; bm < N; bm += ML_BLOCK_SIZE) {
        const int m_end = std::min(bm + ML_BLOCK_SIZE, N);

        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        // Cache test-basis geometry
        v3 verts_mp[ML_BLOCK_SIZE][3], verts_mm[ML_BLOCK_SIZE][3];
        v3 r_fv_mp[ML_BLOCK_SIZE],    r_fv_mm[ML_BLOCK_SIZE];
        v3 centroid_mp[ML_BLOCK_SIZE], centroid_mm[ML_BLOCK_SIZE];
        double ta_mp[ML_BLOCK_SIZE], ta_mm[ML_BLOCK_SIZE];
        double Ap[ML_BLOCK_SIZE],    Am[ML_BLOCK_SIZE];
        double lm_arr[ML_BLOCK_SIZE];
        int layer_mp[ML_BLOCK_SIZE], layer_mm[ML_BLOCK_SIZE];

        for (int m = bm; m < m_end; ++m) {
            int loc = m - bm;
            int tri_mp = t_plus[m];
            ml_load_triangle(vertices, triangles, tri_mp, verts_mp[loc]);
            r_fv_mp[loc]     = ml_load_vertex(vertices, fv_plus[m]);
            centroid_mp[loc] = ml_load_centroid(tri_centroids, tri_mp);
            ta_mp[loc]       = tri_twice_area[tri_mp];
            Ap[loc]          = area_plus[m];
            layer_mp[loc]    = has_cross_layer ? tri_layer[tri_mp] : 0;

            int tri_mm = t_minus[m];
            ml_load_triangle(vertices, triangles, tri_mm, verts_mm[loc]);
            r_fv_mm[loc]     = ml_load_vertex(vertices, fv_minus[m]);
            centroid_mm[loc] = ml_load_centroid(tri_centroids, tri_mm);
            ta_mm[loc]       = tri_twice_area[tri_mm];
            Am[loc]          = area_minus[m];
            layer_mm[loc]    = has_cross_layer ? tri_layer[tri_mm] : 0;

            lm_arr[loc] = edge_length[m];
        }

        // Iterate block-columns
        for (int bn = bm; bn < N; bn += ML_BLOCK_SIZE) {
            const int n_end = std::min(bn + ML_BLOCK_SIZE, N);

            // Cache source-basis geometry
            v3 verts_np[ML_BLOCK_SIZE][3], verts_nm[ML_BLOCK_SIZE][3];
            v3 r_fv_np[ML_BLOCK_SIZE],    r_fv_nm[ML_BLOCK_SIZE];
            v3 centroid_np[ML_BLOCK_SIZE], centroid_nm[ML_BLOCK_SIZE];
            double ta_np[ML_BLOCK_SIZE], ta_nm[ML_BLOCK_SIZE];
            double me_np[ML_BLOCK_SIZE], me_nm[ML_BLOCK_SIZE];
            double An_p[ML_BLOCK_SIZE],  An_m[ML_BLOCK_SIZE];
            double ln_arr[ML_BLOCK_SIZE];
            int layer_np[ML_BLOCK_SIZE], layer_nm[ML_BLOCK_SIZE];

            for (int n = bn; n < n_end; ++n) {
                int loc = n - bn;
                int tri_np = t_plus[n];
                ml_load_triangle(vertices, triangles, tri_np, verts_np[loc]);
                r_fv_np[loc]     = ml_load_vertex(vertices, fv_plus[n]);
                centroid_np[loc] = ml_load_centroid(tri_centroids, tri_np);
                ta_np[loc]       = tri_twice_area[tri_np];
                me_np[loc]       = tri_mean_edge[tri_np];
                An_p[loc]        = area_plus[n];
                layer_np[loc]    = has_cross_layer ? tri_layer[tri_np] : 0;

                int tri_nm = t_minus[n];
                ml_load_triangle(vertices, triangles, tri_nm, verts_nm[loc]);
                r_fv_nm[loc]     = ml_load_vertex(vertices, fv_minus[n]);
                centroid_nm[loc] = ml_load_centroid(tri_centroids, tri_nm);
                ta_nm[loc]       = tri_twice_area[tri_nm];
                me_nm[loc]       = tri_mean_edge[tri_nm];
                An_m[loc]        = area_minus[n];
                layer_nm[loc]    = has_cross_layer ? tri_layer[tri_nm] : 0;

                ln_arr[loc] = edge_length[n];
            }

            // Element loop within the (bm, bn) block
            for (int m = bm; m < m_end; ++m) {
                int lm_idx = m - bm;
                int n_start = std::max(bn, m);  // upper triangle only

                for (int n = n_start; n < n_end; ++n) {
                    int ln_idx = n - bn;

                    cd I_A_total{0.0, 0.0};
                    cd I_Phi_total{0.0, 0.0};

                    double dist, me;
                    bool is_near;
                    cd I_A, I_Phi;

                    // Helper: select model for a (test_layer, src_layer) pair
                    // Returns model index into thread_mgfs / model_eps_r arrays.
                    auto select_model = [&](int lt, int ls) -> int {
                        if (!has_cross_layer) return 0;
                        return model_lut[lt * M + ls];
                    };

                    int midx; bool sl;

                    // (m+, n+)
                    me = me_np[ln_idx];
                    dist = norm3(sub3(centroid_mp[lm_idx], centroid_np[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    sl = (layer_mp[lm_idx] == layer_np[ln_idx]);
                    midx = select_model(layer_mp[lm_idx], layer_np[ln_idx]);
                    triangle_pair_multilayer(k,
                        verts_mp[lm_idx], verts_np[ln_idx],
                        r_fv_mp[lm_idx],  r_fv_np[ln_idx],
                        +1.0, +1.0, lm_arr[lm_idx], ln_arr[ln_idx],
                        Ap[lm_idx], An_p[ln_idx],
                        nq, weights, bary, ta_mp[lm_idx], ta_np[ln_idx],
                        is_near, near_threshold,
                        thread_mgfs[tid * n_models + midx],
                        model_k_src[midx], model_eps_r[midx], sl,
                        I_A, I_Phi);
                    I_A_total += I_A; I_Phi_total += I_Phi;

                    // (m+, n-)
                    me = me_nm[ln_idx];
                    dist = norm3(sub3(centroid_mp[lm_idx], centroid_nm[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    sl = (layer_mp[lm_idx] == layer_nm[ln_idx]);
                    midx = select_model(layer_mp[lm_idx], layer_nm[ln_idx]);
                    triangle_pair_multilayer(k,
                        verts_mp[lm_idx], verts_nm[ln_idx],
                        r_fv_mp[lm_idx],  r_fv_nm[ln_idx],
                        +1.0, -1.0, lm_arr[lm_idx], ln_arr[ln_idx],
                        Ap[lm_idx], An_m[ln_idx],
                        nq, weights, bary, ta_mp[lm_idx], ta_nm[ln_idx],
                        is_near, near_threshold,
                        thread_mgfs[tid * n_models + midx],
                        model_k_src[midx], model_eps_r[midx], sl,
                        I_A, I_Phi);
                    I_A_total += I_A; I_Phi_total += I_Phi;

                    // (m-, n+)
                    me = me_np[ln_idx];
                    dist = norm3(sub3(centroid_mm[lm_idx], centroid_np[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    sl = (layer_mm[lm_idx] == layer_np[ln_idx]);
                    midx = select_model(layer_mm[lm_idx], layer_np[ln_idx]);
                    triangle_pair_multilayer(k,
                        verts_mm[lm_idx], verts_np[ln_idx],
                        r_fv_mm[lm_idx],  r_fv_np[ln_idx],
                        -1.0, +1.0, lm_arr[lm_idx], ln_arr[ln_idx],
                        Am[lm_idx], An_p[ln_idx],
                        nq, weights, bary, ta_mm[lm_idx], ta_np[ln_idx],
                        is_near, near_threshold,
                        thread_mgfs[tid * n_models + midx],
                        model_k_src[midx], model_eps_r[midx], sl,
                        I_A, I_Phi);
                    I_A_total += I_A; I_Phi_total += I_Phi;

                    // (m-, n-)
                    me = me_nm[ln_idx];
                    dist = norm3(sub3(centroid_mm[lm_idx], centroid_nm[ln_idx]));
                    is_near = (me > 1e-30) ? (dist < near_thresh_scaled * me) : true;
                    sl = (layer_mm[lm_idx] == layer_nm[ln_idx]);
                    midx = select_model(layer_mm[lm_idx], layer_nm[ln_idx]);
                    triangle_pair_multilayer(k,
                        verts_mm[lm_idx], verts_nm[ln_idx],
                        r_fv_mm[lm_idx],  r_fv_nm[ln_idx],
                        -1.0, -1.0, lm_arr[lm_idx], ln_arr[ln_idx],
                        Am[lm_idx], An_m[ln_idx],
                        nq, weights, bary, ta_mm[lm_idx], ta_nm[ln_idx],
                        is_near, near_threshold,
                        thread_mgfs[tid * n_models + midx],
                        model_k_src[midx], model_eps_r[midx], sl,
                        I_A, I_Phi);
                    I_A_total += I_A; I_Phi_total += I_Phi;

                    cd val = a_only ? (prefactor_A * I_A_total)
                                   : (prefactor_A * I_A_total + prefactor_Phi * I_Phi_total);
                    Z[m * N + n] = val;
                    if (m != n)
                        Z[n * N + m] = val;
                }
            }
        } // bn block loop
    } // bm block loop (OpenMP parallel)
}


// ── Module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(strata_kernels, m) {
    m.doc() =
        "pybind11 wrapper for the Strata multilayer Green's function library "
        "(https://github.com/modelics/strata, GPL-3.0). "
        "Exposes batched smooth correction G_ML - G_fs for scalar and dyadic GF.";
    m.attr("__version__") = "0.2.0";

    py::class_<StrataModel>(m, "StrataModel",
        "Pre-configured MGF handle for one (frequency, src_layer, obs_layer).")
        .def(py::init<>());

    m.def("make_model", &make_model,
        py::arg("layers"),
        py::arg("epsr_top"), py::arg("mur_top"), py::arg("sigma_top"), py::arg("pec_top"),
        py::arg("epsr_bot"), py::arg("mur_bot"), py::arg("sigma_bot"), py::arg("pec_bot"),
        py::arg("frequency"),
        py::arg("z_src"), py::arg("z_obs"),
        py::arg("k_src_re"), py::arg("k_src_im"),
        py::arg("eps_r_src_re"), py::arg("eps_r_src_im"),
        py::arg("method") = "dcim",
        R"doc(
Build a StrataModel from layer parameters.

Parameters
----------
layers : list of [zmin, zmax, epsr_re, epsr_im, mur, sigma, sigmamu]
    Interior finite layers. All SI units (m, S/m).
epsr_top, mur_top, sigma_top, pec_top : float, float, float, bool
    Top halfspace parameters.
epsr_bot, mur_bot, sigma_bot, pec_bot : float, float, float, bool
    Bottom halfspace parameters.
frequency : float
    Operating frequency (Hz).
z_src, z_obs : float
    z-coordinates of any point in the source / observation layer (m).
    Used to locate layer indices via LayerManager::FindLayer.
k_src_re, k_src_im : float
    Real and imaginary parts of the source-layer wavenumber (rad/m).
    Used for free-space GF subtraction.
eps_r_src_re, eps_r_src_im : float
    Real and imaginary parts of the source-layer effective relative
    permittivity.  Strata returns G_phi = g/ε_r (Formulation-C); we
    multiply by ε_r before subtracting g_fs.
method : str
    'dcim' (default), 'integrate', or 'quasistatic'.
)doc");

    m.def("scalar_G_smooth", &scalar_G_smooth,
        py::arg("model"), py::arg("r_obs"), py::arg("r_src"),
        "Batched scalar potential smooth correction (G_ML - G_fs). "
        "r_obs, r_src: (N,3) float64 → (N,) complex128.");

    m.def("dyadic_G_smooth", &dyadic_G_smooth,
        py::arg("model"), py::arg("r_obs"), py::arg("r_src"),
        "Batched dyadic smooth correction (G_ML - G_fs). "
        "r_obs, r_src: (N,3) float64 → (N,3,3) complex128.");

    m.def("grad_G_smooth", &grad_G_smooth,
        py::arg("model"), py::arg("r_obs"), py::arg("r_src"),
        py::arg("h") = 1.0e-9,
        "Gradient of scalar smooth correction via central FD on G_ML minus analytical ∇G_fs. "
        "r_obs, r_src: (N,3) float64 → (N,3) complex128.");

    m.def("test_mgf_thread_safety", &test_mgf_thread_safety,
        py::arg("model"), py::arg("num_threads") = 4,
        "Verify that MGF copies produce identical results across threads. "
        "Returns True if all results match.");

    m.def("fill_impedance_multilayer_cpp", &fill_impedance_multilayer_cpp,
        py::arg("Z"),
        py::arg("vertices"), py::arg("triangles"),
        py::arg("t_plus"), py::arg("t_minus"),
        py::arg("fv_plus"), py::arg("fv_minus"),
        py::arg("area_plus"), py::arg("area_minus"),
        py::arg("edge_length"),
        py::arg("tri_centroids"), py::arg("tri_mean_edge"), py::arg("tri_twice_area"),
        py::arg("weights"), py::arg("bary"),
        py::arg("k"), py::arg("eta"),
        py::arg("near_threshold"), py::arg("quad_order"),
        py::arg("strata_model"),
        py::arg("num_threads") = 0,
        py::arg("tri_layer_idx") = py::array_t<int32_t>(),
        py::arg("extra_models") = py::list(),
        py::arg("model_lookup") = py::array_t<int32_t>(),
        py::arg("a_only") = false,
        "C++-accelerated multilayer EFIE impedance matrix fill. "
        "Calls Strata ComputeMGF directly from C++ with OpenMP parallelism. "
        "Optional cross-layer support via tri_layer_idx, extra_models, model_lookup. "
        "Set a_only=True for vector-potential-only assembly (A-EFIE).");
}
