#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_utils.hpp"

namespace {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using HostMesh = gwn::tests::HostMesh;

struct TimingSummary {
    double mean_ms{0.0};
    double p50_ms{0.0};
    double p95_ms{0.0};
};

[[nodiscard]] float get_env_positive_float(char const *name, float const default_value) {
    char const *value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return default_value;
    char *end = nullptr;
    float const parsed = std::strtof(value, &end);
    if (end == value || *end != '\0' || !(parsed > 0.0f))
        return default_value;
    return parsed;
}

[[nodiscard]] TimingSummary summarize_timings(std::vector<float> timings_ms) {
    TimingSummary summary{};
    if (timings_ms.empty())
        return summary;

    double sum = 0.0;
    for (float const timing : timings_ms)
        sum += static_cast<double>(timing);
    summary.mean_ms = sum / static_cast<double>(timings_ms.size());

    std::sort(timings_ms.begin(), timings_ms.end());
    auto const percentile = [&](double const q) {
        double const pos = q * static_cast<double>(timings_ms.size() - 1);
        std::size_t const lo = static_cast<std::size_t>(pos);
        std::size_t const hi = std::min<std::size_t>(timings_ms.size() - 1, lo + 1);
        double const t = pos - static_cast<double>(lo);
        return static_cast<double>(timings_ms[lo]) * (1.0 - t) +
               static_cast<double>(timings_ms[hi]) * t;
    };
    summary.p50_ms = percentile(0.50);
    summary.p95_ms = percentile(0.95);
    return summary;
}

template <class LbvhBuildFn, class HplocBuildFn>
gwn::gwn_status measure_topology_pair_latency_ms(
    int const warmup_iters, int const measure_iters, cudaStream_t const stream,
    LbvhBuildFn &&lbvh_build_fn, HplocBuildFn &&hploc_build_fn, std::vector<float> &lbvh_latency_ms,
    std::vector<float> &hploc_latency_ms
) noexcept {
    if (warmup_iters < 0 || measure_iters <= 0) {
        return gwn::gwn_status::invalid_argument(
            "Invalid topology performance warmup/iteration counts."
        );
    }

    lbvh_latency_ms.clear();
    hploc_latency_ms.clear();
    lbvh_latency_ms.reserve(static_cast<std::size_t>(measure_iters));
    hploc_latency_ms.reserve(static_cast<std::size_t>(measure_iters));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaEventCreate(&start)));
    GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaEventCreate(&stop)));
    auto cleanup_events = gwn::gwn_make_scope_exit([&]() noexcept {
        (void)cudaEventDestroy(stop);
        (void)cudaEventDestroy(start);
    });

    for (int i = 0; i < warmup_iters; ++i) {
        GWN_RETURN_ON_ERROR(lbvh_build_fn());
        GWN_RETURN_ON_ERROR(hploc_build_fn());
        GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    }

    for (int i = 0; i < measure_iters; ++i) {
        GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaEventRecord(start, stream)));
        GWN_RETURN_ON_ERROR(lbvh_build_fn());
        GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaEventRecord(stop, stream)));
        GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaEventSynchronize(stop)));
        float lbvh_elapsed_ms = 0.0f;
        GWN_RETURN_ON_ERROR(
            gwn::gwn_cuda_to_status(cudaEventElapsedTime(&lbvh_elapsed_ms, start, stop))
        );
        lbvh_latency_ms.push_back(lbvh_elapsed_ms);

        GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaEventRecord(start, stream)));
        GWN_RETURN_ON_ERROR(hploc_build_fn());
        GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaEventRecord(stop, stream)));
        GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaEventSynchronize(stop)));
        float hploc_elapsed_ms = 0.0f;
        GWN_RETURN_ON_ERROR(
            gwn::gwn_cuda_to_status(cudaEventElapsedTime(&hploc_elapsed_ms, start, stop))
        );
        hploc_latency_ms.push_back(hploc_elapsed_ms);
    }

    return gwn::gwn_status::ok();
}

} // namespace

TEST(smallgwn_integration_performance, hploc_topology_build_ratio_gate) {
    SMALLGWN_SKIP_IF_NO_CUDA();

    std::vector<std::filesystem::path> const model_paths = gwn::tests::collect_model_paths();
    ASSERT_FALSE(model_paths.empty())
        << "No model input found. Set SMALLGWN_MODEL_DATA_DIR or SMALLGWN_MODEL_PATH.";

    std::size_t const model_limit =
        gwn::tests::get_env_positive_size_t("SMALLGWN_HPLOC_PERF_MODEL_LIMIT", 3);
    int const warmup_iters = gwn::tests::get_env_positive_int("SMALLGWN_HPLOC_PERF_WARMUP", 2);
    int const measure_iters = gwn::tests::get_env_positive_int("SMALLGWN_HPLOC_PERF_ITERS", 8);
    float const ratio_limit = get_env_positive_float("SMALLGWN_HPLOC_TOPOLOGY_RATIO_LIMIT", 2.0f);

    std::size_t tested_model_count = 0;
    double worst_ratio = 0.0;

    for (std::filesystem::path const &model_path : model_paths) {
        if (tested_model_count >= model_limit)
            break;

        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value())
            continue;
        HostMesh const &mesh = *maybe_mesh;
        if (mesh.tri_i0.empty())
            continue;

        cudaStream_t stream{};
        gwn::gwn_status const create_stream_status =
            gwn::gwn_cuda_to_status(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(create_stream_status);
        ASSERT_TRUE(create_stream_status.is_ok())
            << gwn::tests::status_to_debug_string(create_stream_status);
        auto destroy_stream =
            gwn::gwn_make_scope_exit([&]() noexcept { (void)cudaStreamDestroy(stream); });

        gwn::gwn_geometry_object<Real, Index> geometry;
        gwn::gwn_status const upload_status = geometry.upload(
            cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()), stream
        );
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
        ASSERT_TRUE(upload_status.is_ok()) << gwn::tests::status_to_debug_string(upload_status);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        gwn::gwn_bvh_object<Real, Index> topology_lbvh;
        gwn::gwn_bvh_object<Real, Index> topology_hploc;
        auto lbvh_build = [&]() noexcept {
            return gwn::gwn_bvh_topology_build_lbvh<4, Real, Index>(
                geometry, topology_lbvh, stream
            );
        };
        auto hploc_build = [&]() noexcept {
            return gwn::gwn_bvh_topology_build_hploc<4, Real, Index>(
                geometry, topology_hploc, stream
            );
        };

        std::vector<float> lbvh_latency_ms{};
        std::vector<float> hploc_latency_ms{};
        gwn::gwn_status const measure_status = measure_topology_pair_latency_ms(
            warmup_iters, measure_iters, stream, lbvh_build, hploc_build, lbvh_latency_ms,
            hploc_latency_ms
        );
        ASSERT_TRUE(measure_status.is_ok()) << gwn::tests::status_to_debug_string(measure_status);

        TimingSummary const lbvh_summary = summarize_timings(lbvh_latency_ms);
        TimingSummary const hploc_summary = summarize_timings(hploc_latency_ms);
        ASSERT_GT(lbvh_summary.p50_ms, 0.0);
        ASSERT_GT(hploc_summary.p50_ms, 0.0);

        double const p50_ratio = hploc_summary.p50_ms / lbvh_summary.p50_ms;
        worst_ratio = std::max(worst_ratio, p50_ratio);
        ++tested_model_count;

        std::cout << "[gwn-perf] model=" << model_path.filename().string()
                  << " stage=topology_build ratio(hploc/lbvh)=" << p50_ratio
                  << " limit=" << ratio_limit << " warmup=" << warmup_iters
                  << " iters=" << measure_iters << " lbvh(mean/p50/p95_ms)=" << lbvh_summary.mean_ms
                  << "/" << lbvh_summary.p50_ms << "/" << lbvh_summary.p95_ms
                  << " hploc(mean/p50/p95_ms)=" << hploc_summary.mean_ms << "/"
                  << hploc_summary.p50_ms << "/" << hploc_summary.p95_ms << std::endl;

        EXPECT_LE(p50_ratio, static_cast<double>(ratio_limit))
            << "H-PLOC topology build regressed above configured ratio limit on "
            << model_path.string();
    }

    ASSERT_GT(tested_model_count, 0u) << "No valid OBJ models were exercised.";
    std::cout << "[gwn-perf] tested_models=" << tested_model_count
              << " worst_ratio(hploc/lbvh)=" << worst_ratio << std::endl;
}
