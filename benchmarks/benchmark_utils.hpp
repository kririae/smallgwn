#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gwn/gwn_utils.cuh>

#include "test_utils.hpp"

namespace gwn::bench {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using HostMesh = gwn::tests::HostMesh;

struct gwn_benchmark_stage_stats {
    double mean_ms{0.0};
    double p50_ms{0.0};
    double p95_ms{0.0};
};

struct gwn_benchmark_stage_result {
    std::string stage{};
    std::string unit{};

    int warmup_iters{0};
    int measure_iters{0};

    std::size_t vertex_count{0};
    std::size_t triangle_count{0};
    std::size_t query_count{0};

    double latency_mean_ms{0.0};
    double latency_p50_ms{0.0};
    double latency_p95_ms{0.0};
    double throughput_per_s{0.0};

    bool success{false};
    std::string error_message{};
};

struct gwn_ray_soa {
    std::array<std::vector<Real>, 3> origin{};
    std::array<std::vector<Real>, 3> direction{};
};

[[nodiscard]] inline std::string gwn_status_to_string(gwn_status const &status) {
    if (status.is_ok())
        return "ok";
    return gwn::tests::status_to_debug_string(status);
}

[[nodiscard]] inline double percentile_ms(std::vector<float> samples, double const q) {
    if (samples.empty())
        return 0.0;
    std::sort(samples.begin(), samples.end());

    double const clamped_q = std::clamp(q, 0.0, 1.0);
    double const pos = clamped_q * static_cast<double>(samples.size() - 1);
    auto const lo = static_cast<std::size_t>(pos);
    std::size_t const hi = std::min<std::size_t>(samples.size() - 1, lo + 1);
    double const t = pos - static_cast<double>(lo);

    return static_cast<double>(samples[lo]) * (1.0 - t) + static_cast<double>(samples[hi]) * t;
}

[[nodiscard]] inline gwn_benchmark_stage_stats
summarize_stage_ms(std::vector<float> const &samples) {
    gwn_benchmark_stage_stats stats{};
    if (samples.empty())
        return stats;

    double sum = 0.0;
    for (float const sample : samples)
        sum += static_cast<double>(sample);
    stats.mean_ms = sum / static_cast<double>(samples.size());
    stats.p50_ms = percentile_ms(samples, 0.50);
    stats.p95_ms = percentile_ms(samples, 0.95);
    return stats;
}

template <class Fn>
gwn_status gwn_measure_stage_latency_ms(
    int const warmup_iters, int const measure_iters, cudaStream_t const stream,
    bool const stream_sync_per_iter, Fn &&fn, std::vector<float> &out_latency_ms
) noexcept {
    if (warmup_iters < 0 || measure_iters <= 0)
        return gwn_status::invalid_argument("Benchmark warmup/iteration counts are invalid.");

    out_latency_ms.clear();
    out_latency_ms.reserve(static_cast<std::size_t>(measure_iters));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaEventCreate(&start)));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaEventCreate(&stop)));
    auto cleanup_events = gwn_make_scope_exit([&]() noexcept {
        (void)cudaEventDestroy(stop);
        (void)cudaEventDestroy(start);
    });

    for (int i = 0; i < warmup_iters; ++i) {
        GWN_RETURN_ON_ERROR(fn());
        if (stream_sync_per_iter)
            GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    }

    for (int i = 0; i < measure_iters; ++i) {
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaEventRecord(start, stream)));
        GWN_RETURN_ON_ERROR(fn());
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaEventRecord(stop, stream)));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaEventSynchronize(stop)));

        float elapsed_ms = 0.0f;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaEventElapsedTime(&elapsed_ms, start, stop)));
        out_latency_ms.push_back(elapsed_ms);

        if (stream_sync_per_iter)
            GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    }

    return gwn_status::ok();
}

[[nodiscard]] inline std::array<std::vector<Real>, 3> gwn_make_mixed_query_soa(
    HostMesh const &mesh, std::size_t const query_count, std::uint64_t const seed
) {
    std::array<std::vector<Real>, 3> queries{};
    queries[0].reserve(query_count);
    queries[1].reserve(query_count);
    queries[2].reserve(query_count);

    if (mesh.vertex_x.empty())
        return queries;

    Real min_x = std::numeric_limits<Real>::max();
    Real min_y = std::numeric_limits<Real>::max();
    Real min_z = std::numeric_limits<Real>::max();
    Real max_x = std::numeric_limits<Real>::lowest();
    Real max_y = std::numeric_limits<Real>::lowest();
    Real max_z = std::numeric_limits<Real>::lowest();

    for (std::size_t i = 0; i < mesh.vertex_x.size(); ++i) {
        min_x = std::min(min_x, mesh.vertex_x[i]);
        min_y = std::min(min_y, mesh.vertex_y[i]);
        min_z = std::min(min_z, mesh.vertex_z[i]);
        max_x = std::max(max_x, mesh.vertex_x[i]);
        max_y = std::max(max_y, mesh.vertex_y[i]);
        max_z = std::max(max_z, mesh.vertex_z[i]);
    }

    Real const center_x = (min_x + max_x) * Real(0.5);
    Real const center_y = (min_y + max_y) * Real(0.5);
    Real const center_z = (min_z + max_z) * Real(0.5);
    Real const extent_x = std::max(max_x - min_x, Real(1e-3));
    Real const extent_y = std::max(max_y - min_y, Real(1e-3));
    Real const extent_z = std::max(max_z - min_z, Real(1e-3));

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Real> near_dist(Real(-0.85), Real(0.85));
    std::uniform_real_distribution<Real> far_scale_dist(Real(1.5), Real(3.0));
    std::uniform_real_distribution<Real> far_jitter_dist(Real(-0.25), Real(0.25));
    std::uniform_int_distribution<int> sign_dist(0, 1);

    auto const signed_scale = [&](Real const scale) {
        return (sign_dist(rng) == 0) ? -scale : scale;
    };
    auto const append_query = [&](Real const x, Real const y, Real const z) {
        queries[0].push_back(x);
        queries[1].push_back(y);
        queries[2].push_back(z);
    };

    append_query(center_x, center_y, center_z);

    for (std::size_t query_id = queries[0].size(); query_id < query_count; ++query_id) {
        bool const near = (query_id % 10) < 7; // 70% near-field / 30% far-field.
        if (near) {
            append_query(
                center_x + near_dist(rng) * extent_x, center_y + near_dist(rng) * extent_y,
                center_z + near_dist(rng) * extent_z
            );
            continue;
        }

        Real const sx = signed_scale(far_scale_dist(rng));
        Real const sy = signed_scale(far_scale_dist(rng));
        Real const sz = signed_scale(far_scale_dist(rng));
        append_query(
            center_x + (sx + far_jitter_dist(rng)) * extent_x,
            center_y + (sy + far_jitter_dist(rng)) * extent_y,
            center_z + (sz + far_jitter_dist(rng)) * extent_z
        );
    }

    return queries;
}

[[nodiscard]] inline gwn_ray_soa gwn_make_mixed_ray_soa(
    HostMesh const &mesh, std::size_t const ray_count, std::uint64_t const seed
) {
    gwn_ray_soa rays{};
    for (int axis = 0; axis < 3; ++axis) {
        rays.origin[axis].reserve(ray_count);
        rays.direction[axis].reserve(ray_count);
    }

    if (mesh.vertex_x.empty())
        return rays;

    Real min_x = std::numeric_limits<Real>::max();
    Real min_y = std::numeric_limits<Real>::max();
    Real min_z = std::numeric_limits<Real>::max();
    Real max_x = std::numeric_limits<Real>::lowest();
    Real max_y = std::numeric_limits<Real>::lowest();
    Real max_z = std::numeric_limits<Real>::lowest();

    for (std::size_t i = 0; i < mesh.vertex_x.size(); ++i) {
        min_x = std::min(min_x, mesh.vertex_x[i]);
        min_y = std::min(min_y, mesh.vertex_y[i]);
        min_z = std::min(min_z, mesh.vertex_z[i]);
        max_x = std::max(max_x, mesh.vertex_x[i]);
        max_y = std::max(max_y, mesh.vertex_y[i]);
        max_z = std::max(max_z, mesh.vertex_z[i]);
    }

    Real const center_x = (min_x + max_x) * Real(0.5);
    Real const center_y = (min_y + max_y) * Real(0.5);
    Real const center_z = (min_z + max_z) * Real(0.5);
    Real const extent_x = std::max(max_x - min_x, Real(1e-3));
    Real const extent_y = std::max(max_y - min_y, Real(1e-3));
    Real const extent_z = std::max(max_z - min_z, Real(1e-3));
    Real const extent_max = std::max(extent_x, std::max(extent_y, extent_z));

    std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ULL);
    std::uniform_real_distribution<Real> unit_dist(Real(-1), Real(1));
    std::uniform_real_distribution<Real> shell_dist(Real(1.6), Real(3.2));
    std::uniform_real_distribution<Real> jitter_dist(Real(-0.08), Real(0.08));
    std::uniform_real_distribution<Real> normal_offset_dist(Real(0.2), Real(0.45));

    auto normalize_or = [](Real &x, Real &y, Real &z, Real const fallback_x, Real const fallback_y,
                           Real const fallback_z) {
        Real const n2 = x * x + y * y + z * z;
        if (!(n2 > Real(0))) {
            x = fallback_x;
            y = fallback_y;
            z = fallback_z;
            return;
        }

        Real const inv_n = Real(1) / std::sqrt(n2);
        x *= inv_n;
        y *= inv_n;
        z *= inv_n;
    };

    auto sample_unit = [&]() {
        Real x = unit_dist(rng);
        Real y = unit_dist(rng);
        Real z = unit_dist(rng);
        normalize_or(x, y, z, Real(1), Real(0), Real(0));
        return std::array<Real, 3>{x, y, z};
    };

    auto fetch_vertex = [&](Index const idx) {
        std::size_t const i = static_cast<std::size_t>(idx);
        return std::array<Real, 3>{mesh.vertex_x[i], mesh.vertex_y[i], mesh.vertex_z[i]};
    };

    auto push_ray = [&](Real const ox, Real const oy, Real const oz, Real const dx, Real const dy,
                        Real const dz) {
        rays.origin[0].push_back(ox);
        rays.origin[1].push_back(oy);
        rays.origin[2].push_back(oz);
        rays.direction[0].push_back(dx);
        rays.direction[1].push_back(dy);
        rays.direction[2].push_back(dz);
    };

    for (std::size_t ray_id = 0; ray_id < ray_count; ++ray_id) {
        if ((ray_id % 4) < 2 && !mesh.tri_i0.empty()) {
            // Triangle-centric rays keep a stable, non-degenerate hit/miss mix on most meshes.
            std::size_t const tri_id = ray_id % mesh.tri_i0.size();
            auto const a = fetch_vertex(mesh.tri_i0[tri_id]);
            auto const b = fetch_vertex(mesh.tri_i1[tri_id]);
            auto const c = fetch_vertex(mesh.tri_i2[tri_id]);

            Real const cx = (a[0] + b[0] + c[0]) / Real(3);
            Real const cy = (a[1] + b[1] + c[1]) / Real(3);
            Real const cz = (a[2] + b[2] + c[2]) / Real(3);

            Real nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
            Real ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
            Real nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            auto const random_unit = sample_unit();
            normalize_or(nx, ny, nz, random_unit[0], random_unit[1], random_unit[2]);

            Real side = (ray_id & 1u) == 0u ? Real(1) : Real(-1);
            Real const offset = normal_offset_dist(rng) * extent_max;
            Real ox = cx + side * offset * nx;
            Real oy = cy + side * offset * ny;
            Real oz = cz + side * offset * nz;

            Real dx = cx - ox;
            Real dy = cy - oy;
            Real dz = cz - oz;

            if ((ray_id % 4) == 1u) {
                // Miss-biased counterpart shot away from the sampled triangle.
                dx = -dx;
                dy = -dy;
                dz = -dz;
            }

            dx += jitter_dist(rng) * random_unit[0];
            dy += jitter_dist(rng) * random_unit[1];
            dz += jitter_dist(rng) * random_unit[2];
            normalize_or(dx, dy, dz, -side * nx, -side * ny, -side * nz);
            push_ray(ox, oy, oz, dx, dy, dz);
            continue;
        }

        // Global rays from outside the bounding volume cover traversal-heavy cases.
        auto const outward = sample_unit();
        Real const shell = shell_dist(rng);

        Real const ox = center_x + outward[0] * shell * extent_x;
        Real const oy = center_y + outward[1] * shell * extent_y;
        Real const oz = center_z + outward[2] * shell * extent_z;

        Real inward_x = center_x - ox;
        Real inward_y = center_y - oy;
        Real inward_z = center_z - oz;
        normalize_or(inward_x, inward_y, inward_z, Real(0), Real(0), Real(1));

        auto const j = sample_unit();
        Real dx = inward_x;
        Real dy = inward_y;
        Real dz = inward_z;

        if ((ray_id % 4) == 2u) {
            // Hit-biased long rays.
            dx = inward_x + jitter_dist(rng) * j[0];
            dy = inward_y + jitter_dist(rng) * j[1];
            dz = inward_z + jitter_dist(rng) * j[2];
        } else {
            // Grazing / miss-biased rays.
            Real tx = inward_y * j[2] - inward_z * j[1];
            Real ty = inward_z * j[0] - inward_x * j[2];
            Real tz = inward_x * j[1] - inward_y * j[0];
            normalize_or(tx, ty, tz, Real(1), Real(0), Real(0));
            dx = Real(0.2) * inward_x + Real(0.8) * tx;
            dy = Real(0.2) * inward_y + Real(0.8) * ty;
            dz = Real(0.2) * inward_z + Real(0.8) * tz;
        }

        normalize_or(dx, dy, dz, inward_x, inward_y, inward_z);
        push_ray(ox, oy, oz, dx, dy, dz);
    }

    return rays;
}

[[nodiscard]] inline std::string gwn_now_timestamp_string() {
    using std::chrono::system_clock;
    auto const now = system_clock::now();
    std::time_t const now_time = system_clock::to_time_t(now);
    std::tm utc_tm{};
#if defined(_WIN32)
    gmtime_s(&utc_tm, &now_time);
#else
    gmtime_r(&now_time, &utc_tm);
#endif
    char buffer[32]{};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_tm);
    return std::string(buffer);
}

} // namespace gwn::bench
