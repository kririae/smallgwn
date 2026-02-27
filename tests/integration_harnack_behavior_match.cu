#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_utils.hpp"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

namespace {

struct HalfOctahedronMesh {
    std::array<Real, 5> vx{1, -1, 0, 0, 0};
    std::array<Real, 5> vy{0, 0, 1, -1, 0};
    std::array<Real, 5> vz{0, 0, 0, 0, 1};
    std::array<Index, 4> i0{0, 2, 1, 3};
    std::array<Index, 4> i1{2, 1, 3, 0};
    std::array<Index, 4> i2{4, 4, 4, 4};
};

struct RaysSoA {
    std::vector<Real> ox;
    std::vector<Real> oy;
    std::vector<Real> oz;
    std::vector<Real> dx;
    std::vector<Real> dy;
    std::vector<Real> dz;

    std::size_t size() const { return ox.size(); }
};

enum class EvalMode : int {
    taylor = 0,
    exact = 1
};

enum class TraceTerminalReason : int {
    unknown = 0,
    invalid_ray_dir = 1,
    hit_grad_stop = 2,
    hit_closed_shell = 3,
    invalid_radius = 4,
    invalid_rho = 5,
    nonpositive_rho = 6,
    tmax_exceeded = 7,
    max_iterations_reached = 8
};

struct TraceDiagnostic {
    Real t{-1};
    int hit{0};
    int iterations{0};
    int commit_count{0};
    int backoff_count{0};
    int terminal_reason{static_cast<int>(TraceTerminalReason::unknown)};
    Real last_t_eval{0};
    Real last_dist{0};
    Real last_grad_omega_mag{0};
    Real last_R{0};
    Real last_rho{0};
};

constexpr int k_width = 4;
constexpr int k_stack = gwn::k_gwn_default_traversal_stack_capacity;
constexpr int k_block_size = 128;

template <class T>
__device__ inline T diag_sqrt(T const value) noexcept {
    if constexpr (std::is_same_v<T, float>)
        return sqrtf(value);
    return sqrt(value);
}

template <int Order>
__device__ inline Real eval_w_exact(
    gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_accessor<Real, Index> const &bvh, Real const px, Real const py,
    Real const pz
) noexcept {
    return gwn::detail::gwn_winding_number_point_bvh_exact_impl<k_width, Real, Index, k_stack>(
        geometry, bvh, px, py, pz
    );
}

template <int Order, EvalMode Mode>
__device__ inline Real eval_w(
    gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn::gwn_bvh4_moment_accessor<Order, Real, Index> const &moment_tree, Real const px,
    Real const py, Real const pz, Real const accuracy_scale
) noexcept {
    if constexpr (Mode == EvalMode::taylor) {
        return gwn::detail::gwn_winding_number_point_bvh_taylor_impl<
            Order, k_width, Real, Index, k_stack>(geometry, bvh, moment_tree, px, py, pz, accuracy_scale);
    }
    return eval_w_exact<Order>(geometry, bvh, px, py, pz);
}

template <int Order, EvalMode Mode>
__device__ inline gwn::detail::gwn_query_vec3<Real> eval_grad_w(
    gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn::gwn_bvh4_moment_accessor<Order, Real, Index> const &moment_tree, Real const px,
    Real const py, Real const pz, Real const accuracy_scale
) noexcept {
    if constexpr (Mode == EvalMode::taylor) {
        return gwn::detail::gwn_winding_gradient_point_bvh_taylor_impl<
            Order, k_width, Real, Index, k_stack>(geometry, bvh, moment_tree, px, py, pz, accuracy_scale);
    }

    // Test-only exact fallback: finite-difference gradient of exact winding.
    Real const h = Real(1e-4);
    Real const inv_2h = Real(0.5) / h;
    Real const wxp = eval_w_exact<Order>(geometry, bvh, px + h, py, pz);
    Real const wxm = eval_w_exact<Order>(geometry, bvh, px - h, py, pz);
    Real const wyp = eval_w_exact<Order>(geometry, bvh, px, py + h, pz);
    Real const wym = eval_w_exact<Order>(geometry, bvh, px, py - h, pz);
    Real const wzp = eval_w_exact<Order>(geometry, bvh, px, py, pz + h);
    Real const wzm = eval_w_exact<Order>(geometry, bvh, px, py, pz - h);
    return gwn::detail::gwn_query_vec3<Real>(
        (wxp - wxm) * inv_2h, (wyp - wym) * inv_2h, (wzp - wzm) * inv_2h
    );
}

template <class T>
__device__ inline T ref_glsl_mod(T const x, T const y) noexcept {
    return x - y * floor(x / y);
}

template <class T>
__device__ inline T ref_two_sided_step(
    T const fx, T const lo_bound, T const hi_bound, T const shift, T const R, T const ray_dir_len
) noexcept {
    if (!(R > T(0)) || !(ray_dir_len > T(0)))
        return T(0);

    T const v_denom = lo_bound + shift;
    T const w_denom = hi_bound + shift;
    if (!(v_denom > T(0)) || !(w_denom > T(0)))
        return T(0);

    T const v = (fx + shift) / v_denom;
    T const w = (fx + shift) / w_denom;
    T const v_disc = v * v + T(8) * v;
    T const w_disc = w * w + T(8) * w;
    if (!(v_disc > T(0)) || !(w_disc > T(0)))
        return T(0);

    T const lo_r = -R / T(2) * (v + T(2) - diag_sqrt(v_disc));
    T const hi_r = R / T(2) * (w + T(2) - diag_sqrt(w_disc));
    T r = ((lo_r < hi_r) ? lo_r : hi_r) / ray_dir_len;
    if (r < T(0))
        r = T(0);
    return r;
}

template <int Order, EvalMode Mode>
__device__ inline TraceDiagnostic trace_reference_semantics(
    gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn::gwn_bvh4_aabb_accessor<Real, Index> const &aabb_tree,
    gwn::gwn_bvh4_moment_accessor<Order, Real, Index> const &moment_tree, Real ray_ox, Real ray_oy,
    Real ray_oz, Real ray_dx, Real ray_dy, Real ray_dz, Real const target_winding,
    Real const epsilon, int const max_iterations, Real const t_max, Real const accuracy_scale
) noexcept {
    TraceDiagnostic diag{};

    Real const dir_len2 = ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz;
    if (!(dir_len2 > Real(0))) {
        diag.terminal_reason = static_cast<int>(TraceTerminalReason::invalid_ray_dir);
        return diag;
    }
    Real dir_len;
    if constexpr (std::is_same_v<Real, float>)
        dir_len = sqrtf(dir_len2);
    else
        dir_len = sqrt(dir_len2);

    constexpr Real k_pi = Real(3.14159265358979323846264338327950288L);
    constexpr Real k_four_pi = Real(4) * k_pi;
    Real const target_omega = k_four_pi * target_winding;

    Real t = Real(0);
    Real t_overstep = Real(0);
    int iter = 0;
    while (t < t_max) {
        if (iter > max_iterations) {
            diag.terminal_reason = static_cast<int>(TraceTerminalReason::max_iterations_reached);
            return diag;
        }

        Real const t_eval = t + t_overstep;
        diag.last_t_eval = t_eval;
        diag.iterations = iter + 1;

        Real const px = ray_ox + t_eval * ray_dx;
        Real const py = ray_oy + t_eval * ray_dy;
        Real const pz = ray_oz + t_eval * ray_dz;

        Real const w =
            eval_w<Order, Mode>(geometry, bvh, moment_tree, px, py, pz, accuracy_scale);
        Real const omega = k_four_pi * w;
        Real const wrapped = ref_glsl_mod<Real>(omega - target_omega, k_four_pi);

        Real const lower_levelset = Real(0);
        Real const upper_levelset = k_four_pi;
        Real const dist_lo = wrapped - lower_levelset;
        Real const dist_hi = upper_levelset - wrapped;
        Real const dist = (dist_lo < dist_hi) ? dist_lo : dist_hi;
        diag.last_dist = dist;

        gwn::detail::gwn_query_vec3<Real> const grad_w =
            eval_grad_w<Order, Mode>(geometry, bvh, moment_tree, px, py, pz, accuracy_scale);
        Real const grad_w_mag =
            diag_sqrt(grad_w.x * grad_w.x + grad_w.y * grad_w.y + grad_w.z * grad_w.z);
        Real const grad_omega_mag = k_four_pi * grad_w_mag;
        diag.last_grad_omega_mag = grad_omega_mag;

        Real const R = gwn::detail::gwn_unsigned_edge_distance_point_bvh_impl<
            k_width, Real, Index, k_stack>(
            geometry, bvh, aabb_tree, px, py, pz, std::numeric_limits<Real>::infinity()
        );
        diag.last_R = R;
        if (!(R >= Real(0))) {
            diag.terminal_reason = static_cast<int>(TraceTerminalReason::invalid_radius);
            return diag;
        }

        Real const rho =
            ref_two_sided_step(wrapped, lower_levelset, upper_levelset, k_four_pi, R, dir_len);
        diag.last_rho = rho;
        if (!(rho >= Real(0))) {
            diag.terminal_reason = static_cast<int>(TraceTerminalReason::invalid_rho);
            return diag;
        }

        if (rho >= t_overstep) {
            if (dist < epsilon * grad_omega_mag) {
                diag.hit = 1;
                diag.t = t_eval;
                diag.terminal_reason = static_cast<int>(TraceTerminalReason::hit_grad_stop);
                return diag;
            }
            if (R < epsilon) {
                diag.hit = 1;
                diag.t = t_eval;
                diag.terminal_reason = static_cast<int>(TraceTerminalReason::hit_closed_shell);
                return diag;
            }

            t += t_overstep + rho;
            t_overstep = rho * Real(0.75);
            ++diag.commit_count;
        } else {
            t_overstep = Real(0);
            ++diag.backoff_count;
        }
        ++iter;
    }

    diag.terminal_reason = static_cast<int>(TraceTerminalReason::tmax_exceeded);
    return diag;
}

template <int Order>
__global__ void harnack_trace_smallgwn_kernel(
    gwn::gwn_geometry_accessor<Real, Index> geometry,
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh,
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb_tree,
    gwn::gwn_bvh4_moment_accessor<Order, Real, Index> moment_tree,
    Real const *ray_ox, Real const *ray_oy, Real const *ray_oz,
    Real const *ray_dx, Real const *ray_dy, Real const *ray_dz, std::size_t const ray_count,
    Real const target_winding, Real const epsilon, int const max_iterations, Real const t_max,
    Real const accuracy_scale, TraceDiagnostic *out_diag
) {
    std::size_t const ray_id = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray_id >= ray_count)
        return;

    auto const res = gwn::detail::gwn_harnack_trace_ray_impl<Order, k_width, Real, Index, k_stack>(
        geometry, bvh, aabb_tree, moment_tree, ray_ox[ray_id], ray_oy[ray_id], ray_oz[ray_id],
        ray_dx[ray_id], ray_dy[ray_id], ray_dz[ray_id], target_winding, epsilon, max_iterations,
        t_max, accuracy_scale
    );

    TraceDiagnostic diag{};
    diag.t = res.t;
    diag.hit = res.hit() ? 1 : 0;
    diag.iterations = res.iterations;
    diag.terminal_reason = static_cast<int>(
        res.hit() ? TraceTerminalReason::hit_grad_stop : TraceTerminalReason::unknown
    );
    out_diag[ray_id] = diag;
}

template <int Order, EvalMode Mode>
__global__ void harnack_trace_reference_kernel(
    gwn::gwn_geometry_accessor<Real, Index> geometry,
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh,
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb_tree,
    gwn::gwn_bvh4_moment_accessor<Order, Real, Index> moment_tree,
    Real const *ray_ox, Real const *ray_oy, Real const *ray_oz,
    Real const *ray_dx, Real const *ray_dy, Real const *ray_dz, std::size_t const ray_count,
    Real const target_winding, Real const epsilon, int const max_iterations, Real const t_max,
    Real const accuracy_scale, TraceDiagnostic *out_diag
) {
    std::size_t const ray_id = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray_id >= ray_count)
        return;

    out_diag[ray_id] = trace_reference_semantics<Order, Mode>(
        geometry, bvh, aabb_tree, moment_tree, ray_ox[ray_id], ray_oy[ray_id], ray_oz[ray_id],
        ray_dx[ray_id], ray_dy[ray_id], ray_dz[ray_id], target_winding, epsilon, max_iterations,
        t_max, accuracy_scale
    );
}

template <int Order>
struct TraceHarness {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Order, Real, Index> moment;

    bool build(HalfOctahedronMesh const &mesh, std::string &error_message) {
        gwn::gwn_status s = geometry.upload(
            cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
            cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
            cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
            cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
            cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
            cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
        );
        if (s.error() == gwn::gwn_error::cuda_runtime_error)
            return false;
        if (!s.is_ok()) {
            error_message = "geometry upload failed: " + gwn::tests::status_to_debug_string(s);
            return false;
        }

        s = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, k_width, Real, Index>(
            geometry, bvh, aabb, moment
        );
        if (!s.is_ok()) {
            error_message = "BVH build failed: " + gwn::tests::status_to_debug_string(s);
            return false;
        }

        return true;
    }

    bool run_smallgwn(
        RaysSoA const &rays, Real const target_winding, Real const epsilon, int const max_iterations,
        Real const t_max, Real const accuracy_scale, std::vector<TraceDiagnostic> &out,
        std::string &error_message
    ) {
        gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
        gwn::gwn_device_array<TraceDiagnostic> d_out;
        if (!resize_and_upload_rays(rays, d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, error_message))
            return false;
        if (!d_out.resize(rays.size()).is_ok()) {
            error_message = "failed to allocate output diagnostics";
            return false;
        }

        dim3 const block(k_block_size);
        dim3 const grid(static_cast<unsigned int>((rays.size() + k_block_size - 1) / k_block_size));
        harnack_trace_smallgwn_kernel<Order><<<grid, block>>>(
            geometry.accessor(), bvh.accessor(), aabb.accessor(), moment.accessor(),
            d_ox.data(), d_oy.data(), d_oz.data(), d_dx.data(), d_dy.data(), d_dz.data(),
            rays.size(), target_winding, epsilon, max_iterations, t_max, accuracy_scale, d_out.data()
        );
        if (cudaError_t const launch_err = cudaGetLastError(); launch_err != cudaSuccess) {
            error_message =
                std::string("smallgwn kernel launch failed: ") + cudaGetErrorString(launch_err);
            return false;
        }
        if (cudaError_t const sync_err = cudaDeviceSynchronize(); sync_err != cudaSuccess) {
            error_message =
                std::string("smallgwn kernel sync failed: ") + cudaGetErrorString(sync_err);
            return false;
        }

        out.resize(rays.size());
        if (!d_out.copy_to_host(cuda::std::span<TraceDiagnostic>(out.data(), out.size())).is_ok()) {
            error_message = "failed to copy smallgwn diagnostics to host";
            return false;
        }
        if (cudaError_t const sync_err = cudaDeviceSynchronize(); sync_err != cudaSuccess) {
            error_message =
                std::string("smallgwn copy sync failed: ") + cudaGetErrorString(sync_err);
            return false;
        }
        return true;
    }

    template <EvalMode Mode>
    bool run_reference(
        RaysSoA const &rays, Real const target_winding, Real const epsilon, int const max_iterations,
        Real const t_max, Real const accuracy_scale, std::vector<TraceDiagnostic> &out,
        std::string &error_message
    ) {
        gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
        gwn::gwn_device_array<TraceDiagnostic> d_out;
        if (!resize_and_upload_rays(rays, d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, error_message))
            return false;
        if (!d_out.resize(rays.size()).is_ok()) {
            error_message = "failed to allocate output diagnostics";
            return false;
        }

        dim3 const block(k_block_size);
        dim3 const grid(static_cast<unsigned int>((rays.size() + k_block_size - 1) / k_block_size));
        harnack_trace_reference_kernel<Order, Mode><<<grid, block>>>(
            geometry.accessor(), bvh.accessor(), aabb.accessor(), moment.accessor(),
            d_ox.data(), d_oy.data(), d_oz.data(), d_dx.data(), d_dy.data(), d_dz.data(),
            rays.size(), target_winding, epsilon, max_iterations, t_max, accuracy_scale, d_out.data()
        );
        if (cudaError_t const launch_err = cudaGetLastError(); launch_err != cudaSuccess) {
            error_message =
                std::string("reference kernel launch failed: ") + cudaGetErrorString(launch_err);
            return false;
        }
        if (cudaError_t const sync_err = cudaDeviceSynchronize(); sync_err != cudaSuccess) {
            error_message =
                std::string("reference kernel sync failed: ") + cudaGetErrorString(sync_err);
            return false;
        }

        out.resize(rays.size());
        if (!d_out.copy_to_host(cuda::std::span<TraceDiagnostic>(out.data(), out.size())).is_ok()) {
            error_message = "failed to copy reference diagnostics to host";
            return false;
        }
        if (cudaError_t const sync_err = cudaDeviceSynchronize(); sync_err != cudaSuccess) {
            error_message =
                std::string("reference copy sync failed: ") + cudaGetErrorString(sync_err);
            return false;
        }
        return true;
    }

private:
    static bool resize_and_upload_rays(
        RaysSoA const &rays, gwn::gwn_device_array<Real> &d_ox, gwn::gwn_device_array<Real> &d_oy,
        gwn::gwn_device_array<Real> &d_oz, gwn::gwn_device_array<Real> &d_dx,
        gwn::gwn_device_array<Real> &d_dy, gwn::gwn_device_array<Real> &d_dz,
        std::string &error_message
    ) {
        bool ok = d_ox.resize(rays.size()).is_ok() && d_oy.resize(rays.size()).is_ok() &&
                  d_oz.resize(rays.size()).is_ok() && d_dx.resize(rays.size()).is_ok() &&
                  d_dy.resize(rays.size()).is_ok() && d_dz.resize(rays.size()).is_ok();
        ok = ok &&
             d_ox.copy_from_host(cuda::std::span<Real const>(rays.ox.data(), rays.size())).is_ok() &&
             d_oy.copy_from_host(cuda::std::span<Real const>(rays.oy.data(), rays.size())).is_ok() &&
             d_oz.copy_from_host(cuda::std::span<Real const>(rays.oz.data(), rays.size())).is_ok() &&
             d_dx.copy_from_host(cuda::std::span<Real const>(rays.dx.data(), rays.size())).is_ok() &&
             d_dy.copy_from_host(cuda::std::span<Real const>(rays.dy.data(), rays.size())).is_ok() &&
             d_dz.copy_from_host(cuda::std::span<Real const>(rays.dz.data(), rays.size())).is_ok();
        if (!ok) {
            error_message = "ray allocation/upload failed";
            return false;
        }
        return true;
    }
};

void append_ray(
    RaysSoA &rays, Real const ox, Real const oy, Real const oz, Real const dx, Real const dy,
    Real const dz
) {
    rays.ox.push_back(ox);
    rays.oy.push_back(oy);
    rays.oz.push_back(oz);
    rays.dx.push_back(dx);
    rays.dy.push_back(dy);
    rays.dz.push_back(dz);
}

RaysSoA make_basic_forward_rays() {
    RaysSoA rays;
    for (int yi = -1; yi <= 1; ++yi) {
        for (int xi = -1; xi <= 1; ++xi)
            append_ray(
                rays, Real(0.2) * Real(xi), Real(0.2) * Real(yi), Real(2), Real(0), Real(0),
                Real(-1)
            );
    }
    return rays;
}

RaysSoA make_branchcut_rays() {
    RaysSoA rays;
    std::array<Real, 4> const xs{Real(-0.45), Real(-0.30), Real(0.30), Real(0.45)};
    std::array<Real, 3> const ys{Real(-0.15), Real(0), Real(0.15)};
    for (Real const y : ys) {
        for (Real const x : xs)
            append_ray(rays, x, y, Real(2), Real(0), Real(0), Real(-1));
    }
    return rays;
}

RaysSoA make_edge_stress_rays() {
    RaysSoA rays;
    append_ray(rays, Real(0.49), Real(0.00), Real(2), Real(0), Real(0), Real(-1));
    append_ray(rays, Real(-0.49), Real(0.00), Real(2), Real(0), Real(0), Real(-1));
    append_ray(rays, Real(0.00), Real(0.49), Real(2), Real(0), Real(0), Real(-1));
    append_ray(rays, Real(0.00), Real(-0.49), Real(2), Real(0), Real(0), Real(-1));
    append_ray(rays, Real(0.35), Real(0.35), Real(2), Real(0), Real(0), Real(-1));
    append_ray(rays, Real(-0.35), Real(0.35), Real(2), Real(0), Real(0), Real(-1));
    append_ray(rays, Real(0.35), Real(-0.35), Real(2), Real(0), Real(0), Real(-1));
    append_ray(rays, Real(-0.35), Real(-0.35), Real(2), Real(0), Real(0), Real(-1));
    return rays;
}

char const *reason_to_cstr(int const reason) {
    switch (static_cast<TraceTerminalReason>(reason)) {
    case TraceTerminalReason::unknown:
        return "unknown";
    case TraceTerminalReason::invalid_ray_dir:
        return "invalid_ray_dir";
    case TraceTerminalReason::hit_grad_stop:
        return "hit_grad_stop";
    case TraceTerminalReason::hit_closed_shell:
        return "hit_closed_shell";
    case TraceTerminalReason::invalid_radius:
        return "invalid_radius";
    case TraceTerminalReason::invalid_rho:
        return "invalid_rho";
    case TraceTerminalReason::nonpositive_rho:
        return "nonpositive_rho";
    case TraceTerminalReason::tmax_exceeded:
        return "tmax_exceeded";
    case TraceTerminalReason::max_iterations_reached:
        return "max_iterations_reached";
    }
    return "invalid_reason";
}

struct ComparisonSummary {
    int algorithmic_mismatch{0};
    int taylor_exact_drift{0};
    int exact_fallback_samples{0};
    std::string detail;
};

ComparisonSummary compare_behavior(
    std::vector<TraceDiagnostic> const &smallgwn_diag,
    std::vector<TraceDiagnostic> const &ref_taylor_diag,
    std::vector<TraceDiagnostic> const &ref_exact_diag, Real const t_tolerance,
    int const max_detail_rows
) {
    ComparisonSummary summary;
    std::ostringstream detail;

    std::size_t const n = smallgwn_diag.size();
    for (std::size_t i = 0; i < n; ++i) {
        TraceDiagnostic const &s = smallgwn_diag[i];
        TraceDiagnostic const &t = ref_taylor_diag[i];
        TraceDiagnostic const &e = ref_exact_diag[i];

        bool const s_hit = s.hit != 0;
        bool const t_hit = t.hit != 0;
        bool const e_hit = e.hit != 0;

        bool const taylor_exact_drift =
            (t_hit != e_hit) || (t_hit && e_hit && std::abs(t.t - e.t) > t_tolerance);
        if (taylor_exact_drift)
            ++summary.taylor_exact_drift;

        if (e.iterations > 0)
            ++summary.exact_fallback_samples;

        bool const algorithmic_mismatch =
            (s_hit != t_hit) ||
            (s_hit && t_hit && std::abs(s.t - t.t) > t_tolerance) ||
            (std::abs(s.iterations - t.iterations) > 1);
        if (!algorithmic_mismatch)
            continue;

        ++summary.algorithmic_mismatch;
        if (summary.algorithmic_mismatch > max_detail_rows)
            continue;

        detail << "ray[" << i << "] mismatch";
        detail << " s(hit=" << s_hit << ", t=" << s.t << ", it=" << s.iterations << ")";
        detail << " t(hit=" << t_hit << ", t=" << t.t << ", it=" << t.iterations
               << ", reason=" << reason_to_cstr(t.terminal_reason) << ")";
        detail << " e(hit=" << e_hit << ", t=" << e.t << ", it=" << e.iterations
               << ", reason=" << reason_to_cstr(e.terminal_reason) << ")";
        detail << " drift_taylor_exact=" << (taylor_exact_drift ? "yes" : "no");
        detail << " [t_eval=" << t.last_t_eval << ", dist=" << t.last_dist
               << ", grad=" << t.last_grad_omega_mag << ", R=" << t.last_R
               << ", rho=" << t.last_rho << ", commits=" << t.commit_count
               << ", backoffs=" << t.backoff_count << "]\n";
    }

    detail << "summary: algorithmic_mismatch=" << summary.algorithmic_mismatch
           << ", taylor_exact_drift=" << summary.taylor_exact_drift
           << ", exact_fallback_samples=" << summary.exact_fallback_samples << "\n";
    summary.detail = detail.str();
    return summary;
}

template <int Order>
ComparisonSummary run_behavior_case(
    HalfOctahedronMesh const &mesh, RaysSoA const &rays, Real const target_winding,
    Real const epsilon, int const max_iterations, Real const t_max, Real const accuracy_scale
) {
    TraceHarness<Order> harness;
    std::string error_message;
    if (!harness.build(mesh, error_message))
        return ComparisonSummary{1, 0, 0, error_message};

    std::vector<TraceDiagnostic> smallgwn_diag;
    std::vector<TraceDiagnostic> ref_taylor_diag;
    std::vector<TraceDiagnostic> ref_exact_diag;
    if (!harness.run_smallgwn(
            rays, target_winding, epsilon, max_iterations, t_max, accuracy_scale, smallgwn_diag,
            error_message
        )) {
        return ComparisonSummary{1, 0, 0, error_message};
    }
    if (!harness.template run_reference<EvalMode::taylor>(
            rays, target_winding, epsilon, max_iterations, t_max, accuracy_scale, ref_taylor_diag,
            error_message
        )) {
        return ComparisonSummary{1, 0, 0, error_message};
    }
    if (!harness.template run_reference<EvalMode::exact>(
            rays, target_winding, epsilon, max_iterations, t_max, accuracy_scale, ref_exact_diag,
            error_message
        )) {
        return ComparisonSummary{1, 0, 0, error_message};
    }

    return compare_behavior(
        smallgwn_diag, ref_taylor_diag, ref_exact_diag,
        /*t_tolerance=*/Real(2e-3), /*max_detail_rows=*/6
    );
}

} // namespace

TEST_F(CudaFixture, harnack_behavior_match_reference_semantics_basic_and_branchcut) {
    HalfOctahedronMesh mesh;
    RaysSoA rays = make_basic_forward_rays();
    RaysSoA branch = make_branchcut_rays();
    rays.ox.insert(rays.ox.end(), branch.ox.begin(), branch.ox.end());
    rays.oy.insert(rays.oy.end(), branch.oy.begin(), branch.oy.end());
    rays.oz.insert(rays.oz.end(), branch.oz.begin(), branch.oz.end());
    rays.dx.insert(rays.dx.end(), branch.dx.begin(), branch.dx.end());
    rays.dy.insert(rays.dy.end(), branch.dy.begin(), branch.dy.end());
    rays.dz.insert(rays.dz.end(), branch.dz.begin(), branch.dz.end());

    ComparisonSummary const summary = run_behavior_case<1>(
        mesh, rays,
        /*target_winding=*/Real(0.5), /*epsilon=*/Real(1e-3),
        /*max_iterations=*/512, /*t_max=*/Real(100), /*accuracy_scale=*/Real(2)
    );

    EXPECT_EQ(summary.algorithmic_mismatch, 0) << summary.detail;
    EXPECT_GT(summary.exact_fallback_samples, 0) << "exact fallback evaluator did not run";
}

TEST_F(CudaFixture, harnack_behavior_match_reference_semantics_edge_stress) {
    HalfOctahedronMesh mesh;
    RaysSoA const rays = make_edge_stress_rays();

    ComparisonSummary const summary = run_behavior_case<1>(
        mesh, rays,
        /*target_winding=*/Real(0.5), /*epsilon=*/Real(1e-3),
        /*max_iterations=*/512, /*t_max=*/Real(100), /*accuracy_scale=*/Real(2)
    );

    EXPECT_EQ(summary.algorithmic_mismatch, 0) << summary.detail;
    EXPECT_GT(summary.exact_fallback_samples, 0) << "exact fallback evaluator did not run";
}
