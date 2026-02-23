#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
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
    std::size_t const lo = static_cast<std::size_t>(pos);
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
