#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "reference_cpu.cuh"
#include "test_utils.hpp"

namespace {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::HostMesh;

enum class topology_builder {
    k_lbvh,
    k_hploc,
};

[[nodiscard]] char const *to_builder_name(topology_builder const builder) noexcept {
    switch (builder) {
    case topology_builder::k_lbvh: return "lbvh";
    case topology_builder::k_hploc: return "hploc";
    }
    return "unknown";
}

template <int Order>
gwn::gwn_status build_facade_for_builder(
    topology_builder const builder, gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_object<Real, Index> &topology,
    gwn::gwn_bvh4_aabb_object<Real, Index> &aabb,
    gwn::gwn_bvh4_moment_object<Order, Real, Index> &moment
) {
    if (builder == topology_builder::k_hploc) {
        return gwn::gwn_bvh_facade_build_topology_aabb_moment_hploc<Order, 4, Real, Index>(
            geometry, topology, aabb, moment
        );
    }
    return gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
        geometry, topology, aabb, moment
    );
}

// Voxel grid helpers (unique to correctness tests).

struct MeshBounds {
    Real min_x{Real(0)};
    Real min_y{Real(0)};
    Real min_z{Real(0)};
    Real max_x{Real(0)};
    Real max_y{Real(0)};
    Real max_z{Real(0)};
};

struct VoxelGridSpec {
    std::size_t nx{1};
    std::size_t ny{1};
    std::size_t nz{1};
    Real origin_x{Real(0)};
    Real origin_y{Real(0)};
    Real origin_z{Real(0)};
    Real dx{Real(1)};
    Real dy{Real(1)};
    Real dz{Real(1)};

    [[nodiscard]] std::size_t count() const noexcept { return nx * ny * nz; }
};

struct ErrorSummary {
    double max_abs{0.0};
    double mean_abs{0.0};
    double p95_abs{0.0};
    double p99_abs{0.0};
};

struct RunningErrorStats {
    double max_abs{0.0};
    double mean_abs{0.0};
    std::size_t count{0};
    std::size_t over_threshold{0};
};

// Helpers.

[[nodiscard]] MeshBounds compute_mesh_bounds(HostMesh const &mesh) {
    MeshBounds bounds{};
    bounds.min_x = std::numeric_limits<Real>::max();
    bounds.min_y = std::numeric_limits<Real>::max();
    bounds.min_z = std::numeric_limits<Real>::max();
    bounds.max_x = std::numeric_limits<Real>::lowest();
    bounds.max_y = std::numeric_limits<Real>::lowest();
    bounds.max_z = std::numeric_limits<Real>::lowest();

    for (std::size_t i = 0; i < mesh.vertex_x.size(); ++i) {
        bounds.min_x = std::min(bounds.min_x, mesh.vertex_x[i]);
        bounds.min_y = std::min(bounds.min_y, mesh.vertex_y[i]);
        bounds.min_z = std::min(bounds.min_z, mesh.vertex_z[i]);
        bounds.max_x = std::max(bounds.max_x, mesh.vertex_x[i]);
        bounds.max_y = std::max(bounds.max_y, mesh.vertex_y[i]);
        bounds.max_z = std::max(bounds.max_z, mesh.vertex_z[i]);
    }

    return bounds;
}

[[nodiscard]] VoxelGridSpec make_voxel_grid(MeshBounds const &bounds, std::size_t target_points) {
    constexpr Real k_min_extent = Real(1e-3);
    constexpr Real k_padding_ratio = Real(0.05);

    target_points = std::max<std::size_t>(target_points, 1);

    Real extent_x = std::max(bounds.max_x - bounds.min_x, k_min_extent);
    Real extent_y = std::max(bounds.max_y - bounds.min_y, k_min_extent);
    Real extent_z = std::max(bounds.max_z - bounds.min_z, k_min_extent);

    Real const pad_x = std::max(extent_x * k_padding_ratio, k_min_extent);
    Real const pad_y = std::max(extent_y * k_padding_ratio, k_min_extent);
    Real const pad_z = std::max(extent_z * k_padding_ratio, k_min_extent);

    Real const min_x = bounds.min_x - pad_x;
    Real const min_y = bounds.min_y - pad_y;
    Real const min_z = bounds.min_z - pad_z;
    Real const max_x = bounds.max_x + pad_x;
    Real const max_y = bounds.max_y + pad_y;
    Real const max_z = bounds.max_z + pad_z;

    extent_x = std::max(max_x - min_x, k_min_extent);
    extent_y = std::max(max_y - min_y, k_min_extent);
    extent_z = std::max(max_z - min_z, k_min_extent);

    Real const max_extent = std::max({extent_x, extent_y, extent_z, k_min_extent});
    auto const ratio_x = static_cast<double>(extent_x / max_extent);
    auto const ratio_y = static_cast<double>(extent_y / max_extent);
    auto const ratio_z = static_cast<double>(extent_z / max_extent);
    double const ratio_volume = std::max(ratio_x * ratio_y * ratio_z, 1e-12);

    auto const to_dim = [](double const value) {
        return std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(value)));
    };

    double scale = std::cbrt(static_cast<double>(target_points) / ratio_volume);
    std::size_t nx = to_dim(scale * ratio_x);
    std::size_t ny = to_dim(scale * ratio_y);
    std::size_t nz = to_dim(scale * ratio_z);

    for (int iter = 0; iter < 5; ++iter) {
        std::size_t const current = nx * ny * nz;
        if (current == target_points)
            break;
        double const adjust =
            std::cbrt(static_cast<double>(target_points) / static_cast<double>(current));
        nx = to_dim(static_cast<double>(nx) * adjust);
        ny = to_dim(static_cast<double>(ny) * adjust);
        nz = to_dim(static_cast<double>(nz) * adjust);
    }

    auto const choose_decrement_axis = [&]() {
        Real const cell_x = extent_x / static_cast<Real>(nx);
        Real const cell_y = extent_y / static_cast<Real>(ny);
        Real const cell_z = extent_z / static_cast<Real>(nz);
        if (cell_x >= cell_y && cell_x >= cell_z)
            return 0;
        if (cell_y >= cell_x && cell_y >= cell_z)
            return 1;
        return 2;
    };

    auto const choose_increment_axis = [&]() {
        Real const cell_x = extent_x / static_cast<Real>(nx);
        Real const cell_y = extent_y / static_cast<Real>(ny);
        Real const cell_z = extent_z / static_cast<Real>(nz);
        if (cell_x >= cell_y && cell_x >= cell_z)
            return 0;
        if (cell_y >= cell_x && cell_y >= cell_z)
            return 1;
        return 2;
    };

    while (nx * ny * nz > target_points) {
        int const axis = choose_decrement_axis();
        if (axis == 0 && nx > 1)
            --nx;
        else if (axis == 1 && ny > 1)
            --ny;
        else if (axis == 2 && nz > 1)
            --nz;
        else
            break;
    }

    while (nx * ny * nz < target_points) {
        int const axis = choose_increment_axis();
        if (axis == 0)
            ++nx;
        else if (axis == 1)
            ++ny;
        else
            ++nz;
    }

    VoxelGridSpec grid{};
    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.origin_x = min_x;
    grid.origin_y = min_y;
    grid.origin_z = min_z;
    grid.dx = extent_x / static_cast<Real>(nx);
    grid.dy = extent_y / static_cast<Real>(ny);
    grid.dz = extent_z / static_cast<Real>(nz);
    return grid;
}

[[nodiscard]] std::size_t
compute_center_linear_index(VoxelGridSpec const &grid, MeshBounds const &bounds) {
    Real const center_x = (bounds.min_x + bounds.max_x) * Real(0.5);
    Real const center_y = (bounds.min_y + bounds.max_y) * Real(0.5);
    Real const center_z = (bounds.min_z + bounds.max_z) * Real(0.5);

    auto const to_index = [](Real const center, Real const origin, Real const step, std::size_t n) {
        auto const raw_index = static_cast<double>((center - origin) / step);
        long long index = static_cast<long long>(std::floor(raw_index));
        if (index < 0)
            index = 0;
        if (index >= static_cast<long long>(n))
            index = static_cast<long long>(n) - 1;
        return static_cast<std::size_t>(index);
    };

    std::size_t const ix = to_index(center_x, grid.origin_x, grid.dx, grid.nx);
    std::size_t const iy = to_index(center_y, grid.origin_y, grid.dy, grid.ny);
    std::size_t const iz = to_index(center_z, grid.origin_z, grid.dz, grid.nz);
    return (iz * grid.ny + iy) * grid.nx + ix;
}

void fill_query_chunk(
    VoxelGridSpec const &grid, std::size_t const begin, std::size_t const count,
    std::vector<Real> &query_x, std::vector<Real> &query_y, std::vector<Real> &query_z
) {
    query_x.resize(count);
    query_y.resize(count);
    query_z.resize(count);

    std::size_t const xy = grid.nx * grid.ny;
    for (std::size_t offset = 0; offset < count; ++offset) {
        std::size_t const linear_index = begin + offset;
        std::size_t const iz = linear_index / xy;
        std::size_t const rem = linear_index - iz * xy;
        std::size_t const iy = rem / grid.nx;
        std::size_t const ix = rem - iy * grid.nx;

        query_x[offset] = grid.origin_x + (static_cast<Real>(ix) + Real(0.5)) * grid.dx;
        query_y[offset] = grid.origin_y + (static_cast<Real>(iy) + Real(0.5)) * grid.dy;
        query_z[offset] = grid.origin_z + (static_cast<Real>(iz) + Real(0.5)) * grid.dz;
    }
}

[[nodiscard]] std::vector<std::size_t>
select_sample_indices(std::size_t const total_count, std::size_t const sample_count) {
    std::vector<std::size_t> indices(sample_count, 0);
    if (total_count == 0 || sample_count == 0)
        return indices;

    for (std::size_t i = 0; i < sample_count; ++i)
        indices[i] = (i * total_count) / sample_count;

    return indices;
}

void fill_sampled_queries(
    VoxelGridSpec const &grid, std::span<std::size_t const> const indices,
    std::vector<Real> &query_x, std::vector<Real> &query_y, std::vector<Real> &query_z
) {
    query_x.resize(indices.size());
    query_y.resize(indices.size());
    query_z.resize(indices.size());

    std::size_t const xy = grid.nx * grid.ny;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        std::size_t const linear_index = indices[i];
        std::size_t const iz = linear_index / xy;
        std::size_t const rem = linear_index - iz * xy;
        std::size_t const iy = rem / grid.nx;
        std::size_t const ix = rem - iy * grid.nx;

        query_x[i] = grid.origin_x + (static_cast<Real>(ix) + Real(0.5)) * grid.dx;
        query_y[i] = grid.origin_y + (static_cast<Real>(iy) + Real(0.5)) * grid.dy;
        query_z[i] = grid.origin_z + (static_cast<Real>(iz) + Real(0.5)) * grid.dz;
    }
}

void accumulate_running_error(
    std::span<Real const> const lhs, std::span<Real const> const rhs, Real const threshold,
    RunningErrorStats &stats
) {
    double sum_abs = stats.mean_abs * static_cast<double>(stats.count);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        double const error = std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]));
        stats.max_abs = std::max(stats.max_abs, error);
        if (error > static_cast<double>(threshold))
            ++stats.over_threshold;
        sum_abs += error;
    }
    stats.count += lhs.size();
    stats.mean_abs = (stats.count == 0) ? 0.0 : (sum_abs / static_cast<double>(stats.count));
}

[[nodiscard]] ErrorSummary
summarize_error(std::span<Real const> const lhs, std::span<Real const> const rhs) {
    ErrorSummary summary{};
    if (lhs.empty())
        return summary;

    std::vector<double> absolute_errors(lhs.size(), 0.0);
    double sum_abs = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        double const error = std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]));
        absolute_errors[i] = error;
        summary.max_abs = std::max(summary.max_abs, error);
        sum_abs += error;
    }
    summary.mean_abs = sum_abs / static_cast<double>(lhs.size());

    std::sort(absolute_errors.begin(), absolute_errors.end());
    auto const percentile = [&](double const q) {
        auto const idx = static_cast<std::size_t>(
            std::floor(q * static_cast<double>(absolute_errors.size() - 1))
        );
        return absolute_errors[idx];
    };
    summary.p95_abs = percentile(0.95);
    summary.p99_abs = percentile(0.99);
    return summary;
}

// Integration Tests.

template <int Order> void run_voxel_rebuild_consistency_test() {
    static_assert(Order == 1 || Order == 2);

    std::vector<std::filesystem::path> const model_paths = gwn::tests::collect_model_paths();
    ASSERT_FALSE(model_paths.empty())
        << "No model input found. Set SMALLGWN_MODEL_DATA_DIR or SMALLGWN_MODEL_PATH.";

    std::size_t const target_total_points =
        gwn::tests::get_env_positive_size_t("SMALLGWN_VOXEL_TOTAL_POINTS", 10'000'000);
    std::size_t const chunk_size =
        gwn::tests::get_env_positive_size_t("SMALLGWN_VOXEL_QUERY_CHUNK_SIZE", 1'000'000);
    std::size_t const target_points_per_model = std::max<std::size_t>(
        65'536, target_total_points / std::max<std::size_t>(1, model_paths.size())
    );

    constexpr Real k_accuracy_scale = Real(2);
    constexpr Real k_consistency_epsilon = Real(3e-4);

    std::size_t tested_model_count = 0;
    bool any_nonzero_winding = false;
    for (std::filesystem::path const &model_path : model_paths) {
        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value())
            continue;
        HostMesh const &mesh = *maybe_mesh;

        MeshBounds const bounds = compute_mesh_bounds(mesh);
        VoxelGridSpec const grid = make_voxel_grid(bounds, target_points_per_model);
        std::size_t const center_linear_index = compute_center_linear_index(grid, bounds);
        std::size_t const query_count = grid.count();
        if (query_count == 0)
            continue;

        gwn::gwn_geometry_object<Real, Index> geometry;
        gwn::gwn_status const upload_status = geometry.upload(
            cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size())
        );
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
        ASSERT_TRUE(upload_status.is_ok()) << gwn::tests::status_to_debug_string(upload_status);

        gwn::gwn_bvh4_topology_object<Real, Index> bvh_iterative;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb_iterative;
        gwn::gwn_bvh4_moment_object<Order, Real, Index> data_iterative;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
                         geometry, bvh_iterative, aabb_iterative, data_iterative
        )
                         .is_ok()));

        gwn::gwn_bvh4_topology_object<Real, Index> bvh_levelwise;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb_levelwise;
        gwn::gwn_bvh4_moment_object<Order, Real, Index> data_levelwise;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
                         geometry, bvh_levelwise, aabb_levelwise, data_levelwise
        )
                         .is_ok()));

        std::size_t const alloc_count = std::min(chunk_size, query_count);
        gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out_iter, d_out_lw;
        ASSERT_TRUE(d_qx.resize(alloc_count).is_ok());
        ASSERT_TRUE(d_qy.resize(alloc_count).is_ok());
        ASSERT_TRUE(d_qz.resize(alloc_count).is_ok());
        ASSERT_TRUE(d_out_iter.resize(alloc_count).is_ok());
        ASSERT_TRUE(d_out_lw.resize(alloc_count).is_ok());

        std::vector<Real> query_x{};
        std::vector<Real> query_y{};
        std::vector<Real> query_z{};
        std::vector<Real> host_iterative{};
        std::vector<Real> host_levelwise{};
        RunningErrorStats stats{};
        double max_abs_winding = 0.0;
        bool center_seen = false;
        Real center_iterative = Real(0);
        Real center_levelwise = Real(0);

        for (std::size_t begin = 0; begin < query_count; begin += alloc_count) {
            std::size_t const count = std::min(alloc_count, query_count - begin);
            fill_query_chunk(grid, begin, count, query_x, query_y, query_z);
            host_iterative.resize(count);
            host_levelwise.resize(count);

            ASSERT_TRUE(
                d_qx.copy_from_host(cuda::std::span<Real const>(query_x.data(), count)).is_ok()
            );
            ASSERT_TRUE(
                d_qy.copy_from_host(cuda::std::span<Real const>(query_y.data(), count)).is_ok()
            );
            ASSERT_TRUE(
                d_qz.copy_from_host(cuda::std::span<Real const>(query_z.data(), count)).is_ok()
            );

            ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, Real, Index>(
                             geometry.accessor(), bvh_iterative.accessor(),
                             data_iterative.accessor(),
                             cuda::std::span<Real const>(d_qx.data(), count),
                             cuda::std::span<Real const>(d_qy.data(), count),
                             cuda::std::span<Real const>(d_qz.data(), count),
                             cuda::std::span<Real>(d_out_iter.data(), count), k_accuracy_scale
            )
                             .is_ok()));

            ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, Real, Index>(
                             geometry.accessor(), bvh_levelwise.accessor(),
                             data_levelwise.accessor(),
                             cuda::std::span<Real const>(d_qx.data(), count),
                             cuda::std::span<Real const>(d_qy.data(), count),
                             cuda::std::span<Real const>(d_qz.data(), count),
                             cuda::std::span<Real>(d_out_lw.data(), count), k_accuracy_scale
            )
                             .is_ok()));

            ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
            ASSERT_TRUE(
                d_out_iter.copy_to_host(cuda::std::span<Real>(host_iterative.data(), count)).is_ok()
            );
            ASSERT_TRUE(
                d_out_lw.copy_to_host(cuda::std::span<Real>(host_levelwise.data(), count)).is_ok()
            );
            ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

            for (std::size_t i = 0; i < count; ++i) {
                max_abs_winding =
                    std::max(max_abs_winding, std::abs(static_cast<double>(host_iterative[i])));
            }
            if (center_linear_index >= begin && center_linear_index < begin + count) {
                std::size_t const local_index = center_linear_index - begin;
                center_iterative = host_iterative[local_index];
                center_levelwise = host_levelwise[local_index];
                center_seen = true;
            }

            accumulate_running_error(
                std::span<Real const>(host_iterative.data(), host_iterative.size()),
                std::span<Real const>(host_levelwise.data(), host_levelwise.size()),
                k_consistency_epsilon, stats
            );
        }

        ++tested_model_count;
        any_nonzero_winding = any_nonzero_winding || (max_abs_winding > 1e-8);
        std::cout << "[gwn-correctness] model=" << model_path.filename().string()
                  << " triangles=" << mesh.tri_i0.size() << " voxel_grid=" << grid.nx << "x"
                  << grid.ny << "x" << grid.nz << " queries=" << query_count
                  << " center_iterative=" << center_iterative
                  << " center_levelwise=" << center_levelwise
                  << " max_abs_winding=" << max_abs_winding << " max_abs_diff=" << stats.max_abs
                  << " mean_abs_diff=" << stats.mean_abs << " over_eps=" << stats.over_threshold
                  << std::endl;

        EXPECT_TRUE(center_seen) << "Center voxel was not covered for " << model_path.string();
        EXPECT_EQ(stats.over_threshold, 0u)
            << "Order-" << Order << " rebuild mismatch on model " << model_path.string();
    }

    ASSERT_GT(tested_model_count, 0u) << "No valid OBJ models were exercised.";
    EXPECT_TRUE(any_nonzero_winding)
        << "All tested models produced near-zero winding values across voxel queries.";
}

TEST(smallgwn_integration_correctness, voxel_order1_rebuild_consistency) {
    run_voxel_rebuild_consistency_test<1>();
}

TEST(smallgwn_integration_correctness, voxel_order2_rebuild_consistency) {
    run_voxel_rebuild_consistency_test<2>();
}

template <int Order> void run_voxel_hploc_vs_lbvh_consistency_test() {
    static_assert(Order == 1 || Order == 2);

    std::vector<std::filesystem::path> const model_paths = gwn::tests::collect_model_paths();
    ASSERT_FALSE(model_paths.empty())
        << "No model input found. Set SMALLGWN_MODEL_DATA_DIR or SMALLGWN_MODEL_PATH.";

    std::size_t const model_limit =
        gwn::tests::get_env_positive_size_t("SMALLGWN_HPLOC_COMPARE_MODEL_LIMIT", 4);
    std::size_t const target_total_points =
        gwn::tests::get_env_positive_size_t("SMALLGWN_HPLOC_COMPARE_TOTAL_POINTS", 2'000'000);
    std::size_t const max_sample_count =
        gwn::tests::get_env_positive_size_t("SMALLGWN_HPLOC_COMPARE_MAX_SAMPLES", 80'000);
    std::size_t const min_sample_count =
        gwn::tests::get_env_positive_size_t("SMALLGWN_HPLOC_COMPARE_MIN_SAMPLES", 16'384);
    std::size_t const target_points_per_model =
        std::max<std::size_t>(65'536, target_total_points / std::max<std::size_t>(1, model_limit));

    constexpr Real k_accuracy_scale = Real(2);
    constexpr Real k_max_abs_epsilon = Real(3e-1);
    constexpr Real k_p99_abs_epsilon = Real(8e-2);

    std::size_t tested_model_count = 0;
    for (std::filesystem::path const &model_path : model_paths) {
        if (tested_model_count >= model_limit)
            break;
        SCOPED_TRACE(model_path.string());

        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value())
            continue;
        HostMesh const &mesh = *maybe_mesh;
        if (mesh.tri_i0.empty())
            continue;

        MeshBounds const bounds = compute_mesh_bounds(mesh);
        VoxelGridSpec const grid = make_voxel_grid(bounds, target_points_per_model);
        std::size_t const query_count = grid.count();
        if (query_count == 0)
            continue;
        std::size_t const sample_floor = std::min(min_sample_count, query_count);
        std::size_t const sample_count = std::min(query_count, max_sample_count);
        if (sample_count < sample_floor)
            continue;
        std::vector<std::size_t> const sampled_indices =
            select_sample_indices(query_count, sample_count);

        std::vector<Real> query_x{};
        std::vector<Real> query_y{};
        std::vector<Real> query_z{};
        fill_sampled_queries(
            grid, std::span<std::size_t const>(sampled_indices.data(), sampled_indices.size()),
            query_x, query_y, query_z
        );

        gwn::gwn_geometry_object<Real, Index> geometry;
        gwn::gwn_status const upload_status = geometry.upload(
            cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size())
        );
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
        ASSERT_TRUE(upload_status.is_ok()) << gwn::tests::status_to_debug_string(upload_status);

        gwn::gwn_bvh4_topology_object<Real, Index> bvh_lbvh;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb_lbvh;
        gwn::gwn_bvh4_moment_object<Order, Real, Index> moment_lbvh;
        gwn::gwn_bvh4_topology_object<Real, Index> bvh_hploc;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb_hploc;
        gwn::gwn_bvh4_moment_object<Order, Real, Index> moment_hploc;

        ASSERT_TRUE((build_facade_for_builder<Order>(
                         topology_builder::k_lbvh, geometry, bvh_lbvh, aabb_lbvh, moment_lbvh
        )
                         .is_ok()));
        ASSERT_TRUE((build_facade_for_builder<Order>(
                         topology_builder::k_hploc, geometry, bvh_hploc, aabb_hploc, moment_hploc
        )
                         .is_ok()));

        gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
        ASSERT_TRUE(d_qx.resize(sample_count).is_ok());
        ASSERT_TRUE(d_qy.resize(sample_count).is_ok());
        ASSERT_TRUE(d_qz.resize(sample_count).is_ok());
        ASSERT_TRUE(d_out.resize(sample_count).is_ok());
        ASSERT_TRUE(
            d_qx.copy_from_host(cuda::std::span<Real const>(query_x.data(), sample_count)).is_ok()
        );
        ASSERT_TRUE(
            d_qy.copy_from_host(cuda::std::span<Real const>(query_y.data(), sample_count)).is_ok()
        );
        ASSERT_TRUE(
            d_qz.copy_from_host(cuda::std::span<Real const>(query_z.data(), sample_count)).is_ok()
        );

        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, Real, Index>(
                         geometry.accessor(), bvh_lbvh.accessor(), moment_lbvh.accessor(),
                         d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> output_lbvh(sample_count, Real(0));
        ASSERT_TRUE(d_out
                        .copy_to_host(cuda::std::span<Real>(output_lbvh.data(), output_lbvh.size()))
                        .is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, Real, Index>(
                         geometry.accessor(), bvh_hploc.accessor(), moment_hploc.accessor(),
                         d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> output_hploc(sample_count, Real(0));
        ASSERT_TRUE(
            d_out.copy_to_host(cuda::std::span<Real>(output_hploc.data(), output_hploc.size()))
                .is_ok()
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ErrorSummary const builder_diff = summarize_error(
            std::span<Real const>(output_hploc.data(), output_hploc.size()),
            std::span<Real const>(output_lbvh.data(), output_lbvh.size())
        );

        ++tested_model_count;
        std::cout << "[gwn-correctness] builders=(" << to_builder_name(topology_builder::k_lbvh)
                  << "," << to_builder_name(topology_builder::k_hploc)
                  << ") model=" << model_path.filename().string() << " order=" << Order
                  << " triangles=" << mesh.tri_i0.size() << " samples=" << sample_count
                  << " diff(max/p99/p95/mean)=" << builder_diff.max_abs << "/"
                  << builder_diff.p99_abs << "/" << builder_diff.p95_abs << "/"
                  << builder_diff.mean_abs << std::endl;

        EXPECT_LE(builder_diff.max_abs, k_max_abs_epsilon)
            << "Order-" << Order << " Taylor mismatch (max) between LBVH and H-PLOC on "
            << model_path.string();
        EXPECT_LE(builder_diff.p99_abs, k_p99_abs_epsilon)
            << "Order-" << Order << " Taylor mismatch (p99) between LBVH and H-PLOC on "
            << model_path.string();
    }

    ASSERT_GT(tested_model_count, 0u) << "No valid OBJ models were exercised.";
}

TEST(smallgwn_integration_correctness, voxel_order1_hploc_vs_lbvh_consistency_on_sampled_models) {
    run_voxel_hploc_vs_lbvh_consistency_test<1>();
}

TEST(smallgwn_integration_correctness, voxel_order2_hploc_vs_lbvh_consistency_on_sampled_models) {
    run_voxel_hploc_vs_lbvh_consistency_test<2>();
}

#if 0
TEST(smallgwn_integration_correctness, voxel_exact_and_taylor_match_cpu_on_small_models) {
    std::vector<std::filesystem::path> const model_paths = gwn::tests::collect_model_paths();
    if (model_paths.empty())
        GTEST_SKIP() << "No model input found. Set SMALLGWN_MODEL_DATA_DIR or SMALLGWN_MODEL_PATH.";

    std::size_t const target_total_points =
        gwn::tests::get_env_positive_size_t("SMALLGWN_VOXEL_TOTAL_POINTS", 10'000'000);
    std::size_t const target_points_per_model = std::max<std::size_t>(
        65'536, target_total_points / std::max<std::size_t>(1, model_paths.size()));
    std::size_t const cpu_work_budget =
        gwn::tests::get_env_positive_size_t("SMALLGWN_CPU_WORK_BUDGET", 120'000'000);
    std::size_t const max_cpu_samples =
        gwn::tests::get_env_positive_size_t("SMALLGWN_CPU_MAX_SAMPLES", 50'000);
    std::size_t const min_cpu_samples =
        gwn::tests::get_env_positive_size_t("SMALLGWN_CPU_MIN_SAMPLES", 4'096);

    constexpr Real k_accuracy_scale = Real(2);
    std::size_t tested_model_count = 0;
    bool any_nonzero_exact = false;

    for (std::filesystem::path const& model_path : model_paths) {
        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value())
            continue;
        HostMesh const& mesh = *maybe_mesh;

        std::size_t const tri_count = mesh.tri_i0.size();
        if (tri_count == 0)
            continue;

        MeshBounds const bounds = compute_mesh_bounds(mesh);
        VoxelGridSpec const grid = make_voxel_grid(bounds, target_points_per_model);
        std::size_t const center_linear_index = compute_center_linear_index(grid, bounds);
        std::size_t const voxel_count = grid.count();
        if (voxel_count == 0)
            continue;

        std::size_t const samples_by_work = std::max<std::size_t>(1, cpu_work_budget / tri_count);
        std::size_t const sample_count =
            std::min({voxel_count, max_cpu_samples, samples_by_work});
        if (sample_count < min_cpu_samples) {
            std::cout << "[gwn-correctness] skip model=" << model_path.filename().string()
                      << " triangles=" << tri_count
                      << " reason=cpu_reference_budget sample_count=" << sample_count << std::endl;
            continue;
        }

        std::vector<std::size_t> sample_indices = select_sample_indices(voxel_count, sample_count);
        if (!sample_indices.empty())
            sample_indices[0] = center_linear_index;
        std::vector<Real> query_x{};
        std::vector<Real> query_y{};
        std::vector<Real> query_z{};
        fill_sampled_queries(
            grid, std::span<std::size_t const>(sample_indices.data(), sample_indices.size()),
            query_x, query_y, query_z);

        std::vector<Real> reference_output(sample_count, Real(0));
        gwn::gwn_status const reference_status =
            gwn::tests::reference_winding_number_batch<Real, Index>(
                std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
                std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
                std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
                std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
                std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
                std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()),
                std::span<Real const>(query_x.data(), query_x.size()),
                std::span<Real const>(query_y.data(), query_y.size()),
                std::span<Real const>(query_z.data(), query_z.size()),
                std::span<Real>(reference_output.data(), reference_output.size()));
        ASSERT_TRUE(reference_status.is_ok())
            << gwn::tests::status_to_debug_string(reference_status);

        gwn::gwn_geometry_object<Real, Index> geometry;
        gwn::gwn_status const upload_status = geometry.upload(
            cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()));
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
        ASSERT_TRUE(upload_status.is_ok()) << gwn::tests::status_to_debug_string(upload_status);

        gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
        ASSERT_TRUE(d_qx.resize(sample_count).is_ok());
        ASSERT_TRUE(d_qy.resize(sample_count).is_ok());
        ASSERT_TRUE(d_qz.resize(sample_count).is_ok());
        ASSERT_TRUE(d_out.resize(sample_count).is_ok());
        ASSERT_TRUE(d_qx.copy_from_host(
            cuda::std::span<Real const>(query_x.data(), sample_count)).is_ok());
        ASSERT_TRUE(d_qy.copy_from_host(
            cuda::std::span<Real const>(query_y.data(), sample_count)).is_ok());
        ASSERT_TRUE(d_qz.copy_from_host(
            cuda::std::span<Real const>(query_z.data(), sample_count)).is_ok());

        std::vector<Real> exact_output(sample_count, Real(0));
        std::vector<Real> order0_output(sample_count, Real(0));
        std::vector<Real> order1_iterative_output(sample_count, Real(0));
        std::vector<Real> order1_levelwise_output(sample_count, Real(0));

        // Exact.
        gwn::gwn_bvh4_topology_object<Real, Index> bvh_exact;
        ASSERT_TRUE(
            (gwn::gwn_bvh_topology_build_lbvh<4, Real, Index>(geometry, bvh_exact)
                 .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
                        geometry.accessor(), bvh_exact.accessor(),
                        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span())
                        .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(exact_output.data(), exact_output.size())).is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        // Order 0.
        gwn::gwn_bvh4_topology_object<Real, Index> bvh_order0;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb_order0;
        gwn::gwn_bvh4_moment_object<0, Real, Index> data_order0;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<0, 4, Real, Index>(geometry, bvh_order0, aabb_order0, data_order0)
                        .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
                        geometry.accessor(), bvh_order0.accessor(), data_order0.accessor(),
                        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale)
                        .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(order0_output.data(), order0_output.size())).is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        // Order 1 iterative.
        gwn::gwn_bvh4_topology_object<Real, Index> bvh_order1_iter;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb_order1_iter;
        gwn::gwn_bvh4_moment_object<1, Real, Index> data_order1_iter;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(geometry, bvh_order1_iter, aabb_order1_iter, data_order1_iter)
                        .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                        geometry.accessor(), bvh_order1_iter.accessor(),
                        data_order1_iter.accessor(),
                        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale)
                        .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(order1_iterative_output.data(), order1_iterative_output.size()))
                        .is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        // Order 1 levelwise.
        gwn::gwn_bvh4_topology_object<Real, Index> bvh_order1_lw;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb_order1_lw;
        gwn::gwn_bvh4_moment_object<1, Real, Index> data_order1_lw;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(geometry, bvh_order1_lw, aabb_order1_lw, data_order1_lw)
                        .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                        geometry.accessor(), bvh_order1_lw.accessor(),
                        data_order1_lw.accessor(),
                        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale)
                        .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(order1_levelwise_output.data(), order1_levelwise_output.size()))
                        .is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ErrorSummary const exact_vs_cpu = summarize_error(
            std::span<Real const>(exact_output.data(), exact_output.size()),
            std::span<Real const>(reference_output.data(), reference_output.size()));
        ErrorSummary const order0_vs_exact = summarize_error(
            std::span<Real const>(order0_output.data(), order0_output.size()),
            std::span<Real const>(exact_output.data(), exact_output.size()));
        ErrorSummary const order1_vs_exact = summarize_error(
            std::span<Real const>(order1_iterative_output.data(), order1_iterative_output.size()),
            std::span<Real const>(exact_output.data(), exact_output.size()));
        ErrorSummary const levelwise_vs_iterative = summarize_error(
            std::span<Real const>(order1_levelwise_output.data(), order1_levelwise_output.size()),
            std::span<Real const>(order1_iterative_output.data(),
                                  order1_iterative_output.size()));
        double const exact_value_max_abs = [&]() {
            double value = 0.0;
            for (Real const winding : exact_output)
                value = std::max(value, std::abs(static_cast<double>(winding)));
            return value;
        }();

        ++tested_model_count;
        any_nonzero_exact = any_nonzero_exact || (exact_value_max_abs > 1e-8);
        std::cout << "[gwn-correctness] model=" << model_path.filename().string()
                  << " triangles=" << tri_count << " cpu_samples=" << sample_count
                  << " exact_vs_cpu(max/p99/mean)=" << exact_vs_cpu.max_abs << "/"
                  << exact_vs_cpu.p99_abs << "/" << exact_vs_cpu.mean_abs
                  << " order0_vs_exact(p95/p99)=" << order0_vs_exact.p95_abs << "/"
                  << order0_vs_exact.p99_abs
                  << " order1_vs_exact(p95/p99)=" << order1_vs_exact.p95_abs << "/"
                  << order1_vs_exact.p99_abs
                  << " levelwise_vs_iter(max/p99)=" << levelwise_vs_iterative.max_abs << "/"
                  << levelwise_vs_iterative.p99_abs << std::endl;

        EXPECT_LE(exact_vs_cpu.p99_abs, 2e-3);
        EXPECT_LE(exact_vs_cpu.max_abs, 2e-2);
        EXPECT_LE(order0_vs_exact.p95_abs, 8e-2);
        EXPECT_LE(order0_vs_exact.p99_abs, 2e-1);
        EXPECT_LE(order1_vs_exact.p95_abs, 5e-2);
        EXPECT_LE(order1_vs_exact.p99_abs, 1.2e-1);
        EXPECT_LE(levelwise_vs_iterative.max_abs, 1e-3);
        EXPECT_LE(levelwise_vs_iterative.p99_abs, 3e-4);
    }

    ASSERT_GT(tested_model_count, 0u)
        << "No models satisfied CPU-reference sampling budget for correctness checks.";
    EXPECT_TRUE(any_nonzero_exact)
        << "All sampled models produced near-zero exact winding values.";
}
#endif

} // namespace
