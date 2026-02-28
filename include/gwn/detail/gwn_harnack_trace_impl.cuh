#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_distance_impl.cuh"
#include "gwn_query_gradient_impl.cuh"
#include "gwn_query_vec3_impl.cuh"
#include "gwn_query_winding_gradient_impl.cuh"
#include "gwn_query_winding_impl.cuh"

namespace gwn {
namespace detail {

// ---------------------------------------------------------------------------
// Harnack trace result — returned per ray.
// ---------------------------------------------------------------------------

template <gwn_real_type Real> struct gwn_harnack_trace_result {
    Real t{Real(-1)};       ///< Ray parameter at hit (negative ⟹ no hit).
    Real winding{Real(0)};  ///< Winding number at the hit point.
    Real normal_x{Real(0)}; ///< Surface normal x (unit gradient direction).
    Real normal_y{Real(0)};
    Real normal_z{Real(0)};
    int iterations{0}; ///< Number of marching iterations.

    __host__ __device__ constexpr bool hit() const noexcept { return t >= Real(0); }
};

// ---------------------------------------------------------------------------
// Harnack step size (pure math).
//
// Given current function value f_t, target f_star, lower bound c (such that
// f − c > 0 on B_R), and ball radius R, returns the maximum safe step ρ
// guaranteed by the Harnack inequality in 3D.
//
// Formula (harmonic.tex §3.1.2):
//   a = (f_t − c) / (f* − c)
//   ρ = (R/2) |a + 2 − √(a² + 8a)|
// ---------------------------------------------------------------------------

template <gwn_real_type Real>
__host__ __device__ inline Real
gwn_harnack_step_size(Real const f_t, Real const f_star, Real const c, Real const R) noexcept {
    if (R <= Real(0))
        return Real(0);

    Real const denom = f_star - c;
    if (denom <= Real(0))
        return R; // target ≤ lower bound ⟹ safe full step

    Real const a = (f_t - c) / denom;
    if (a <= Real(0))
        return R * Real(1e-4); // Taylor error can push f_t < c; nudge forward

    Real const disc = a * a + Real(8) * a;
    if (disc <= Real(0))
        return R;

    Real rho = (R / Real(2)) * abs(a + Real(2) - sqrt(disc));

    // Clamp to [0, R].
    if (rho < Real(0))
        rho = Real(0);
    if (rho > R)
        rho = R;

    return rho;
}

// Apply runtime stepping guards used by the tracer:
// 1) Harnack safe step
// 2) over-step acceleration
// 3) minimum-step anti-stall
// 4) final clamp to [0, R]
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_harnack_constrained_step(
    Real const f_t, Real const f_star, Real const c, Real const R,
    Real const overstep_factor = Real(1), Real const min_abs_step = Real(0),
    Real const min_relative_step = Real(0)
) noexcept {
    if (R <= Real(0))
        return Real(0);

    Real rho = gwn_harnack_step_size(f_t, f_star, c, R);
    rho *= overstep_factor;
    if (rho > R)
        rho = R;

    Real const min_step = (R > min_abs_step) ? (R * min_relative_step) : min_abs_step;
    if (rho < min_step)
        rho = min_step;

    // Final safety clamp is required: min-step can otherwise push rho > R.
    if (rho > R)
        rho = R;
    if (rho < Real(0))
        rho = Real(0);
    return rho;
}

template <gwn_real_type Real>
__host__ __device__ inline Real gwn_glsl_mod(Real const x, Real const y) noexcept {
    if (!(y > Real(0)))
        return x;
    return x - y * floor(x / y);
}

// Two-sided step bound for angle-valued tracing (Algorithm 2):
// step to the closer of the adjacent periodic level sets, then apply the
// same runtime guards used by the face-distance tracer.
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_harnack_constrained_two_sided_step(
    Real const f_t, Real const lower_levelset, Real const upper_levelset, Real const c,
    Real const R, Real const overstep_factor = Real(1), Real const min_abs_step = Real(0),
    Real const min_relative_step = Real(0)
) noexcept {
    if (R <= Real(0))
        return Real(0);
    if (!(upper_levelset > lower_levelset))
        return Real(0);

    Real const rho_lo = gwn_harnack_step_size(f_t, lower_levelset, c, R);
    Real const rho_hi = gwn_harnack_step_size(f_t, upper_levelset, c, R);
    Real rho = (rho_lo < rho_hi) ? rho_lo : rho_hi;

    rho *= overstep_factor;
    if (rho > R)
        rho = R;

    Real const min_step = (R > min_abs_step) ? (R * min_relative_step) : min_abs_step;
    if (rho < min_step)
        rho = min_step;

    if (rho > R)
        rho = R;
    if (rho < Real(0))
        rho = Real(0);
    return rho;
}

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline gwn_harnack_trace_result<Real> gwn_harnack_trace_angle_ray_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree, Real ray_ox,
    Real ray_oy, Real ray_oz, Real ray_dx, Real ray_dy, Real ray_dz, Real const target_winding,
    Real const epsilon, int const max_iterations, Real const t_max, Real const accuracy_scale
) noexcept;

// ---------------------------------------------------------------------------
// Unified per-ray Harnack trace entry point.
//
// Single execution path:
//   • angle-valued + edge-distance mode (Algorithm 2)
// ---------------------------------------------------------------------------

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline gwn_harnack_trace_result<Real> gwn_harnack_trace_ray_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree, Real ray_ox,
    Real ray_oy, Real ray_oz, Real ray_dx, Real ray_dy, Real ray_dz, Real const target_winding,
    Real const epsilon, int const max_iterations, Real const t_max, Real const accuracy_scale
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_harnack_trace_ray currently supports Order 0, 1, and 2."
    );
    return gwn_harnack_trace_angle_ray_impl<Order, Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, moment_tree, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
        target_winding, epsilon, max_iterations, t_max, accuracy_scale
    );
}

// ---------------------------------------------------------------------------
// Per-ray Harnack trace (angle-valued, Algorithm 2).
//
// Uses:
//   • wrapped angle representative in [0, 4π) around target phase
//   • safe-ball radius R = distance to nearest triangle
//   • two-sided Harnack step bound against fixed wrapped bounds {0, 4π}
//   • overstepping with backoff (paper §3.1.4 / reference implementation)
// ---------------------------------------------------------------------------

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline gwn_harnack_trace_result<Real> gwn_harnack_trace_angle_ray_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree, Real ray_ox,
    Real ray_oy, Real ray_oz, Real ray_dx, Real ray_dy, Real ray_dz, Real const target_winding,
    Real const epsilon, int const max_iterations, Real const t_max, Real const accuracy_scale
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_harnack_trace_angle_ray currently supports Order 0, 1, and 2."
    );

    gwn_harnack_trace_result<Real> result;

    Real const dir_len2 = ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz;
    if (!(dir_len2 > Real(0)))
        return result;
    Real dir_len;
    if constexpr (std::is_same_v<Real, float>)
        dir_len = sqrtf(dir_len2);
    else
        dir_len = sqrt(dir_len2);

    constexpr Real k_pi = Real(3.14159265358979323846264338327950288L);
    constexpr Real k_four_pi = Real(4) * k_pi;
    Real const target_omega = k_four_pi * target_winding;

    auto eval_w_and_grad = [&](Real const px, Real const py, Real const pz) {
        return gwn_winding_and_gradient_point_bvh_taylor_impl<
            Order, Width, Real, Index, StackCapacity>(
            geometry, bvh, moment_tree, px, py, pz, accuracy_scale
        );
    };

    auto eval_grad_shading = [&](Real const, Real const, Real const,
                                 gwn_query_vec3<Real> const &fallback) -> gwn_query_vec3<Real> {
        return fallback;
    };

    auto eval_closest_triangle_normal = [&](Real const px, Real const py,
                                            Real const pz) -> gwn_query_vec3<Real> {
        auto const r =
            gwn_closest_triangle_normal_point_bvh_impl<Width, Real, Index, StackCapacity>(
                geometry, bvh, aabb_tree, px, py, pz, std::numeric_limits<Real>::infinity()
            );
        return gwn_query_vec3<Real>(r.normal_x, r.normal_y, r.normal_z);
    };

    auto fill_result = [&](Real t_, Real w_, gwn_query_vec3<Real> const &g, int iters) {
        result.t = t_;
        result.winding = w_;
        result.iterations = iters;
        constexpr Real k_grad_floor = Real(1e-8);

        gwn_query_vec3<Real> n = g;
        Real gm = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        if (!(gm > k_grad_floor) || !isfinite(gm)) {
            Real const px = ray_ox + t_ * ray_dx;
            Real const py = ray_oy + t_ * ray_dy;
            Real const pz = ray_oz + t_ * ray_dz;
            n = eval_closest_triangle_normal(px, py, pz);
            gm = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        }
        if (gm > Real(0)) {
            Real const inv_gm = Real(1) / gm;
            result.normal_x = n.x * inv_gm;
            result.normal_y = n.y * inv_gm;
            result.normal_z = n.z * inv_gm;
        }
    };

    Real t = Real(0);
    Real t_overstep = Real(0);
    Real R_prev = std::numeric_limits<Real>::infinity();
    int iter = 0;
    while (t < t_max) {
        // Reference semantics: max-iteration guard is checked in-loop rather
        // than encoded as a strict for-loop bound.
        if (iter > max_iterations)
            break;

        Real const t_eval = t + t_overstep;

        Real const px = ray_ox + t_eval * ray_dx;
        Real const py = ray_oy + t_eval * ray_dy;
        Real const pz = ray_oz + t_eval * ray_dz;

        auto const wg = eval_w_and_grad(px, py, pz);
        Real const w = wg.winding;
        gwn_query_vec3<Real> const grad_w = wg.gradient;
        Real const grad_w_mag =
            sqrt(grad_w.x * grad_w.x + grad_w.y * grad_w.y + grad_w.z * grad_w.z);
        Real const omega = k_four_pi * w;
        Real const val = gwn_glsl_mod<Real>(omega - target_omega, k_four_pi);
        Real const lower_levelset = Real(0);
        Real const upper_levelset = k_four_pi;
        Real const dist_lo = val - lower_levelset;
        Real const dist_hi = upper_levelset - val;
        Real const dist = (dist_lo < dist_hi) ? dist_lo : dist_hi;
        Real const grad_omega_mag = k_four_pi * grad_w_mag;

        // R(x) = distance to the nearest triangle (face + edges + vertices).
        // This subsumes singular-edge distance because every singular edge is
        // geometrically a subset of at least one triangle.
        Real const R = gwn_unsigned_distance_point_bvh_impl<Width, Real, Index, StackCapacity>(
            geometry, bvh, aabb_tree, px, py, pz, R_prev
        );
        if (!(R >= Real(0)))
            break;
        R_prev = R;

        Real const c = -k_four_pi;
        Real rho =
            gwn_harnack_constrained_two_sided_step(val, lower_levelset, upper_levelset, c, R);
        rho /= dir_len;
        if (!(rho >= Real(0)))
            break;

        if (rho >= t_overstep) {
            // Reference-style terminal checks:
            // 1) gradient-scaled level-set proximity (paper §3.1.2)
            // 2) very small safe-ball radius near singular boundary
            if (dist < epsilon * grad_omega_mag || R < epsilon) {
                gwn_query_vec3<Real> grad_shading = eval_grad_shading(px, py, pz, grad_w);
                if (R < epsilon)
                    grad_shading = eval_closest_triangle_normal(px, py, pz);
                fill_result(t_eval, w, grad_shading, iter + 1);
                return result;
            }

            t += t_overstep + rho;
            t_overstep = rho * Real(0.75); // next attempted step is 1.75 * rho
            result.iterations = iter + 1;
        } else {
            // Overstep was too aggressive; retry from committed t without overstep.
            t_overstep = Real(0);
            R_prev = std::numeric_limits<Real>::infinity();
        }
        ++iter;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Batch functor for Harnack tracing.
// ---------------------------------------------------------------------------

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
struct gwn_harnack_trace_batch_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> bvh{};
    gwn_bvh_aabb_accessor<Width, Real, Index> aabb_tree{};
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> moment_tree{};

    cuda::std::span<Real const> ray_origin_x{};
    cuda::std::span<Real const> ray_origin_y{};
    cuda::std::span<Real const> ray_origin_z{};
    cuda::std::span<Real const> ray_dir_x{};
    cuda::std::span<Real const> ray_dir_y{};
    cuda::std::span<Real const> ray_dir_z{};

    cuda::std::span<Real> output_t{};
    cuda::std::span<Real> output_normal_x{};
    cuda::std::span<Real> output_normal_y{};
    cuda::std::span<Real> output_normal_z{};

    Real target_winding{};
    Real epsilon{};
    int max_iterations{};
    Real t_max{};
    Real accuracy_scale{};

    __device__ void operator()(std::size_t const ray_id) const {
        auto const res = gwn_harnack_trace_ray_impl<Order, Width, Real, Index, StackCapacity>(
            geometry, bvh, aabb_tree, moment_tree, ray_origin_x[ray_id], ray_origin_y[ray_id],
            ray_origin_z[ray_id], ray_dir_x[ray_id], ray_dir_y[ray_id], ray_dir_z[ray_id],
            target_winding, epsilon, max_iterations, t_max, accuracy_scale
        );
        output_t[ray_id] = res.t;
        output_normal_x[ray_id] = res.normal_x;
        output_normal_y[ray_id] = res.normal_y;
        output_normal_z[ray_id] = res.normal_z;
    }
};

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
using gwn_harnack_trace_angle_batch_functor =
    gwn_harnack_trace_batch_functor<Order, Width, Real, Index, StackCapacity>;

} // namespace detail
} // namespace gwn
