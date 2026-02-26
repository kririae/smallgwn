#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_utils.hpp"

namespace {

using Real = gwn::tests::Real;
using gwn::tests::HostMesh;

static_assert(
    true, "Taylor-only integration matrix: exact API intentionally excluded."
);

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

[[nodiscard]] std::size_t builder_slot(topology_builder const builder) noexcept {
    return (builder == topology_builder::k_lbvh) ? 0u : 1u;
}

[[nodiscard]] std::size_t width_slot(int const width) noexcept {
    switch (width) {
    case 2: return 0u;
    case 3: return 1u;
    case 4: return 2u;
    case 8: return 3u;
    default: return 0u;
    }
}

[[nodiscard]] std::optional<std::size_t>
get_env_size_t_allow_zero(char const *name) {
    char const *value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return std::nullopt;
    std::size_t parsed = 0;
    char const *end = value + std::char_traits<char>::length(value);
    auto const [ptr, ec] = std::from_chars(value, end, parsed);
    if (ec != std::errc() || ptr != end)
        return std::nullopt;
    return parsed;
}

[[nodiscard]] std::size_t
read_env_size_t_allow_zero(char const *name, std::size_t const default_value) {
    std::optional<std::size_t> const parsed = get_env_size_t_allow_zero(name);
    return parsed.has_value() ? *parsed : default_value;
}

struct MeshBounds {
    Real min_x{Real(0)};
    Real min_y{Real(0)};
    Real min_z{Real(0)};
    Real max_x{Real(0)};
    Real max_y{Real(0)};
    Real max_z{Real(0)};
};

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

struct QuerySoA {
    std::vector<Real> x{};
    std::vector<Real> y{};
    std::vector<Real> z{};
};

[[nodiscard]] QuerySoA make_lattice_queries(HostMesh const &mesh, std::size_t target_points) {
    target_points = std::max<std::size_t>(target_points, 512u);
    MeshBounds const bounds = compute_mesh_bounds(mesh);

    constexpr Real k_pad_ratio = Real(0.05);
    constexpr Real k_min_extent = Real(1e-3);

    Real const extent_x = std::max(bounds.max_x - bounds.min_x, k_min_extent);
    Real const extent_y = std::max(bounds.max_y - bounds.min_y, k_min_extent);
    Real const extent_z = std::max(bounds.max_z - bounds.min_z, k_min_extent);

    Real const min_x = bounds.min_x - extent_x * k_pad_ratio;
    Real const min_y = bounds.min_y - extent_y * k_pad_ratio;
    Real const min_z = bounds.min_z - extent_z * k_pad_ratio;
    Real const max_x = bounds.max_x + extent_x * k_pad_ratio;
    Real const max_y = bounds.max_y + extent_y * k_pad_ratio;
    Real const max_z = bounds.max_z + extent_z * k_pad_ratio;

    std::size_t side = static_cast<std::size_t>(std::llround(std::cbrt(
        static_cast<double>(target_points)
    )));
    side = std::max<std::size_t>(side, 2u);
    std::size_t const count = side * side * side;

    QuerySoA query{};
    query.x.reserve(count);
    query.y.reserve(count);
    query.z.reserve(count);

    for (std::size_t iz = 0; iz < side; ++iz) {
        Real const tz = (static_cast<Real>(iz) + Real(0.5)) / static_cast<Real>(side);
        Real const qz = min_z + tz * (max_z - min_z);
        for (std::size_t iy = 0; iy < side; ++iy) {
            Real const ty = (static_cast<Real>(iy) + Real(0.5)) / static_cast<Real>(side);
            Real const qy = min_y + ty * (max_y - min_y);
            for (std::size_t ix = 0; ix < side; ++ix) {
                Real const tx = (static_cast<Real>(ix) + Real(0.5)) / static_cast<Real>(side);
                Real const qx = min_x + tx * (max_x - min_x);
                query.x.push_back(qx);
                query.y.push_back(qy);
                query.z.push_back(qz);
            }
        }
    }
    return query;
}

template <typename IndexT> struct MeshIndices {
    std::vector<IndexT> tri_i0{};
    std::vector<IndexT> tri_i1{};
    std::vector<IndexT> tri_i2{};
};

template <typename IndexT>
[[nodiscard]] MeshIndices<IndexT> cast_mesh_indices(HostMesh const &mesh) {
    MeshIndices<IndexT> cast{};
    cast.tri_i0.reserve(mesh.tri_i0.size());
    cast.tri_i1.reserve(mesh.tri_i1.size());
    cast.tri_i2.reserve(mesh.tri_i2.size());
    for (std::size_t i = 0; i < mesh.tri_i0.size(); ++i) {
        cast.tri_i0.push_back(static_cast<IndexT>(mesh.tri_i0[i]));
        cast.tri_i1.push_back(static_cast<IndexT>(mesh.tri_i1[i]));
        cast.tri_i2.push_back(static_cast<IndexT>(mesh.tri_i2[i]));
    }
    return cast;
}

struct ComboRunResult {
    bool ok{false};
    std::vector<Real> output{};
    std::string error{};
};

template <
    int Order, int Width, typename IndexT,
    std::enable_if_t<std::is_unsigned_v<IndexT>, int> = 0>
[[nodiscard]] ComboRunResult run_combo(
    topology_builder const builder, HostMesh const &mesh, MeshIndices<IndexT> const &index_mesh,
    QuerySoA const &query, Real const accuracy_scale
) {
    ComboRunResult result{};

    gwn::gwn_geometry_object<Real, IndexT> geometry{};
    gwn::gwn_status const upload_status = geometry.upload(
        cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
        cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
        cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
        cuda::std::span<IndexT const>(index_mesh.tri_i0.data(), index_mesh.tri_i0.size()),
        cuda::std::span<IndexT const>(index_mesh.tri_i1.data(), index_mesh.tri_i1.size()),
        cuda::std::span<IndexT const>(index_mesh.tri_i2.data(), index_mesh.tri_i2.size())
    );
    if (!upload_status.is_ok()) {
        result.error = gwn::tests::status_to_debug_string(upload_status);
        return result;
    }

    gwn::gwn_bvh_topology_object<Width, Real, IndexT> topology{};
    gwn::gwn_bvh_aabb_tree_object<Width, Real, IndexT> aabb{};
    gwn::gwn_bvh_moment_tree_object<Width, Order, Real, IndexT> moment{};

    gwn::gwn_status build_status = gwn::gwn_status::internal_error("builder dispatch failed");
    if (builder == topology_builder::k_hploc) {
        build_status = gwn::gwn_bvh_facade_build_topology_aabb_moment_hploc<Order, Width, Real, IndexT>(
            geometry, topology, aabb, moment
        );
    } else {
        build_status = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, Width, Real, IndexT>(
            geometry, topology, aabb, moment
        );
    }
    if (!build_status.is_ok()) {
        result.error = gwn::tests::status_to_debug_string(build_status);
        return result;
    }

    std::size_t const count = query.x.size();
    gwn::gwn_device_array<Real> d_qx{};
    gwn::gwn_device_array<Real> d_qy{};
    gwn::gwn_device_array<Real> d_qz{};
    gwn::gwn_device_array<Real> d_out{};

    if (!d_qx.resize(count).is_ok() || !d_qy.resize(count).is_ok() || !d_qz.resize(count).is_ok() ||
        !d_out.resize(count).is_ok()) {
        result.error = "Failed to allocate device buffers for query batch.";
        return result;
    }

    if (!d_qx.copy_from_host(cuda::std::span<Real const>(query.x.data(), count)).is_ok() ||
        !d_qy.copy_from_host(cuda::std::span<Real const>(query.y.data(), count)).is_ok() ||
        !d_qz.copy_from_host(cuda::std::span<Real const>(query.z.data(), count)).is_ok()) {
        result.error = "Failed to upload query buffers to device.";
        return result;
    }

    gwn::gwn_status const query_status =
        gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, Width, Real, IndexT>(
            geometry.accessor(), topology.accessor(), moment.accessor(), d_qx.span(), d_qy.span(),
            d_qz.span(), d_out.span(), accuracy_scale
        );
    if (!query_status.is_ok()) {
        result.error = gwn::tests::status_to_debug_string(query_status);
        return result;
    }

    cudaError_t const sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        result.error = std::string("CUDA synchronize failed: ") + cudaGetErrorString(sync_status);
        return result;
    }

    result.output.resize(count, Real(0));
    if (!d_out.copy_to_host(cuda::std::span<Real>(result.output.data(), result.output.size())).is_ok()) {
        result.error = "Failed to copy winding output to host.";
        return result;
    }
    cudaError_t const sync_after_copy = cudaDeviceSynchronize();
    if (sync_after_copy != cudaSuccess) {
        result.error = std::string("CUDA synchronize after copy failed: ") +
                       cudaGetErrorString(sync_after_copy);
        return result;
    }

    result.ok = true;
    return result;
}

template <typename IndexT>
[[nodiscard]] ComboRunResult run_combo_for_width(
    int const width, int const order, topology_builder const builder, HostMesh const &mesh,
    MeshIndices<IndexT> const &index_mesh, QuerySoA const &query, Real const accuracy_scale
) {
    switch (width) {
    case 2:
        if (order == 1)
            return run_combo<1, 2>(builder, mesh, index_mesh, query, accuracy_scale);
        return run_combo<2, 2>(builder, mesh, index_mesh, query, accuracy_scale);
    case 3:
        if (order == 1)
            return run_combo<1, 3>(builder, mesh, index_mesh, query, accuracy_scale);
        return run_combo<2, 3>(builder, mesh, index_mesh, query, accuracy_scale);
    case 4:
        if (order == 1)
            return run_combo<1, 4>(builder, mesh, index_mesh, query, accuracy_scale);
        return run_combo<2, 4>(builder, mesh, index_mesh, query, accuracy_scale);
    case 8:
        if (order == 1)
            return run_combo<1, 8>(builder, mesh, index_mesh, query, accuracy_scale);
        return run_combo<2, 8>(builder, mesh, index_mesh, query, accuracy_scale);
    default: {
        ComboRunResult invalid{};
        invalid.error = "Unsupported width.";
        return invalid;
    }
    }
}

struct ErrorSummary {
    double max_abs{0.0};
    double p99_abs{0.0};
};

[[nodiscard]] ErrorSummary
summarize_error(std::span<Real const> const lhs, std::span<Real const> const rhs) {
    ErrorSummary summary{};
    if (lhs.size() != rhs.size() || lhs.empty())
        return summary;

    std::vector<double> abs_error(lhs.size(), 0.0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        double const diff = std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]));
        abs_error[i] = diff;
        summary.max_abs = std::max(summary.max_abs, diff);
    }
    std::sort(abs_error.begin(), abs_error.end());
    std::size_t const p99_index = static_cast<std::size_t>(
        std::floor(0.99 * static_cast<double>(abs_error.size() - 1))
    );
    summary.p99_abs = abs_error[p99_index];
    return summary;
}

[[nodiscard]] std::string join_failures(std::vector<std::string> const &failures) {
    std::string out;
    for (std::size_t i = 0; i < failures.size(); ++i) {
        if (i != 0)
            out += "\n";
        out += failures[i];
    }
    return out;
}

struct MatrixSummary {
    std::size_t width_count{0};
    std::size_t index_count{0};
    std::size_t combo_count{0};
    std::size_t combo_count_per_order{0};
    std::size_t tested_model_count{0};
    bool seen_width_2{false};
    bool seen_width_3{false};
    bool seen_width_4{false};
    bool seen_width_8{false};
    bool seen_index_u32{false};
    bool seen_index_u64{false};
    bool seen_order_1{false};
    bool seen_order_2{false};
    std::vector<std::string> failures{};
};

void record_combo_coverage(MatrixSummary &summary, int const width, bool const is_u64, int const order) {
    summary.seen_width_2 = summary.seen_width_2 || (width == 2);
    summary.seen_width_3 = summary.seen_width_3 || (width == 3);
    summary.seen_width_4 = summary.seen_width_4 || (width == 4);
    summary.seen_width_8 = summary.seen_width_8 || (width == 8);
    summary.seen_index_u32 = summary.seen_index_u32 || !is_u64;
    summary.seen_index_u64 = summary.seen_index_u64 || is_u64;
    summary.seen_order_1 = summary.seen_order_1 || (order == 1);
    summary.seen_order_2 = summary.seen_order_2 || (order == 2);
}

void check_threshold(
    MatrixSummary &summary, std::span<Real const> const lhs, std::span<Real const> const rhs,
    double const max_threshold, double const p99_threshold, std::string context
) {
    if (lhs.size() != rhs.size() || lhs.empty()) {
        summary.failures.push_back(context + " has mismatched/empty outputs.");
        return;
    }
    ErrorSummary const diff = summarize_error(lhs, rhs);
    if (diff.max_abs > max_threshold || diff.p99_abs > p99_threshold) {
        summary.failures.push_back(
            context + " exceeded thresholds (max=" + std::to_string(diff.max_abs) +
            ", p99=" + std::to_string(diff.p99_abs) + ")"
        );
    }
}

[[nodiscard]] MatrixSummary run_matrix_profile(
    std::size_t const default_model_limit, std::size_t const default_total_points,
    int const default_order_max
) {
    constexpr std::array<int, 4> k_widths = {2, 3, 4, 8};
    constexpr std::array<topology_builder, 2> k_builders = {
        topology_builder::k_lbvh,
        topology_builder::k_hploc,
    };
    constexpr Real k_accuracy_scale = Real(2);

    constexpr double k_builder_max_abs = 3e-1;
    constexpr double k_builder_p99_abs = 8e-2;
    constexpr double k_width_max_abs = 4e-1;
    constexpr double k_width_p99_abs = 1e-1;
    constexpr double k_index_max_abs = 1e-2;
    constexpr double k_index_p99_abs = 2e-3;

    MatrixSummary summary{};

    std::size_t const model_limit =
        read_env_size_t_allow_zero("SMALLGWN_MATRIX_MODEL_LIMIT", default_model_limit);
    std::size_t total_points =
        read_env_size_t_allow_zero("SMALLGWN_MATRIX_TOTAL_POINTS", default_total_points);
    if (total_points == 0)
        total_points = default_total_points;

    int order_max = gwn::tests::get_env_positive_int("SMALLGWN_MATRIX_ORDER_MAX", default_order_max);
    order_max = std::max(1, std::min(order_max, default_order_max));

    std::vector<std::filesystem::path> model_paths = gwn::tests::collect_model_paths();
    if (model_paths.empty()) {
        summary.failures.push_back(
            "No model input found. Set SMALLGWN_MODEL_DATA_DIR or SMALLGWN_MODEL_PATH."
        );
        return summary;
    }
    if (model_limit > 0 && model_paths.size() > model_limit)
        model_paths.resize(model_limit);

    std::size_t const points_per_model = std::max<std::size_t>(
        4096u, total_points / std::max<std::size_t>(1u, model_paths.size())
    );

    for (std::filesystem::path const &model_path : model_paths) {
        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value())
            continue;
        HostMesh const &mesh = *maybe_mesh;
        if (mesh.tri_i0.empty())
            continue;

        QuerySoA const query = make_lattice_queries(mesh, points_per_model);
        if (query.x.empty())
            continue;

        MeshIndices<std::uint32_t> const mesh_u32 = cast_mesh_indices<std::uint32_t>(mesh);
        MeshIndices<std::uint64_t> const mesh_u64 = cast_mesh_indices<std::uint64_t>(mesh);

        for (int order = 1; order <= order_max; ++order) {
            std::array<std::array<std::array<std::vector<Real>, 4>, 2>, 2> outputs{};
            bool order_ok = true;

            for (topology_builder const builder : k_builders) {
                for (int index_slot = 0; index_slot < 2; ++index_slot) {
                    bool const is_u64 = (index_slot == 1);
                    for (int const width : k_widths) {
                        ComboRunResult result = is_u64
                            ? run_combo_for_width<std::uint64_t>(
                                  width, order, builder, mesh, mesh_u64, query, k_accuracy_scale
                              )
                            : run_combo_for_width<std::uint32_t>(
                                  width, order, builder, mesh, mesh_u32, query, k_accuracy_scale
                              );
                        if (!result.ok) {
                            summary.failures.push_back(
                                model_path.filename().string() + " order=" + std::to_string(order) +
                                " builder=" + to_builder_name(builder) + " width=" +
                                std::to_string(width) + " index=" + (is_u64 ? "u64" : "u32") +
                                " failed: " + result.error
                            );
                            order_ok = false;
                            continue;
                        }
                        record_combo_coverage(summary, width, is_u64, order);
                        outputs[builder_slot(builder)][static_cast<std::size_t>(index_slot)]
                               [width_slot(width)] = std::move(result.output);
                    }
                }
            }

            if (!order_ok)
                continue;

            for (int width_index = 0; width_index < 4; ++width_index) {
                int const width = k_widths[static_cast<std::size_t>(width_index)];
                for (int index = 0; index < 2; ++index) {
                    std::span<Real const> const lbvh_span{
                        outputs[0u][static_cast<std::size_t>(index)][static_cast<std::size_t>(width_index)]};
                    std::span<Real const> const hploc_span{
                        outputs[1u][static_cast<std::size_t>(index)][static_cast<std::size_t>(width_index)]};
                    check_threshold(
                        summary, hploc_span, lbvh_span, k_builder_max_abs, k_builder_p99_abs,
                        model_path.filename().string() + " order=" + std::to_string(order) +
                            " builder_diff width=" + std::to_string(width) + " index=" +
                            (index == 0 ? "u32" : "u64")
                    );
                }
            }

            for (int builder = 0; builder < 2; ++builder) {
                for (int index = 0; index < 2; ++index) {
                    std::span<Real const> const width4{
                        outputs[static_cast<std::size_t>(builder)][static_cast<std::size_t>(index)][2u]};
                    for (int width_index = 0; width_index < 4; ++width_index) {
                        int const width = k_widths[static_cast<std::size_t>(width_index)];
                        if (width == 4)
                            continue;
                        std::span<Real const> const width_span{
                            outputs[static_cast<std::size_t>(builder)][static_cast<std::size_t>(index)]
                                   [static_cast<std::size_t>(width_index)]};
                        check_threshold(
                            summary, width_span, width4, k_width_max_abs, k_width_p99_abs,
                            model_path.filename().string() + " order=" + std::to_string(order) +
                                " width_diff builder=" +
                                to_builder_name(k_builders[static_cast<std::size_t>(builder)]) +
                                " width=" + std::to_string(width) + " vs width=4 index=" +
                                (index == 0 ? "u32" : "u64")
                        );
                    }
                }
            }

            for (int builder = 0; builder < 2; ++builder) {
                for (int width_index = 0; width_index < 4; ++width_index) {
                    int const width = k_widths[static_cast<std::size_t>(width_index)];
                    std::span<Real const> const u32_span{
                        outputs[static_cast<std::size_t>(builder)][0u][static_cast<std::size_t>(width_index)]};
                    std::span<Real const> const u64_span{
                        outputs[static_cast<std::size_t>(builder)][1u][static_cast<std::size_t>(width_index)]};
                    check_threshold(
                        summary, u64_span, u32_span, k_index_max_abs, k_index_p99_abs,
                        model_path.filename().string() + " order=" + std::to_string(order) +
                            " index_diff builder=" +
                            to_builder_name(k_builders[static_cast<std::size_t>(builder)]) +
                            " width=" + std::to_string(width)
                    );
                }
            }
        }

        ++summary.tested_model_count;
    }

    summary.width_count = static_cast<std::size_t>(summary.seen_width_2) +
                          static_cast<std::size_t>(summary.seen_width_3) +
                          static_cast<std::size_t>(summary.seen_width_4) +
                          static_cast<std::size_t>(summary.seen_width_8);
    summary.index_count = static_cast<std::size_t>(summary.seen_index_u32) +
                          static_cast<std::size_t>(summary.seen_index_u64);
    summary.combo_count = summary.width_count * summary.index_count * 2u;
    summary.combo_count_per_order = summary.combo_count;
    return summary;
}

[[nodiscard]] MatrixSummary run_light_profile() {
    return run_matrix_profile(2u, 200'000u, 1);
}

[[nodiscard]] MatrixSummary run_heavy_profile() {
    return run_matrix_profile(0u, 2'000'000u, 2);
}

TEST(smallgwn_integration_taylor_matrix, light_order1_width_index_builder_matrix) {
    SMALLGWN_SKIP_IF_NO_CUDA();
    MatrixSummary const summary = run_light_profile();

    EXPECT_TRUE(summary.failures.empty()) << join_failures(summary.failures);
    EXPECT_GT(summary.tested_model_count, 0u) << "No valid OBJ models were exercised.";
    EXPECT_EQ(summary.width_count, 4u);
    EXPECT_EQ(summary.index_count, 2u);
    EXPECT_EQ(summary.combo_count, 16u);
    EXPECT_TRUE(summary.seen_width_2);
    EXPECT_TRUE(summary.seen_width_3);
    EXPECT_TRUE(summary.seen_width_4);
    EXPECT_TRUE(summary.seen_width_8);
    EXPECT_TRUE(summary.seen_index_u32);
    EXPECT_TRUE(summary.seen_index_u64);
}

TEST(smallgwn_integration_taylor_matrix, heavy_order1_order2_width_index_builder_matrix) {
    SMALLGWN_SKIP_IF_NO_CUDA();
    MatrixSummary const summary = run_heavy_profile();

    EXPECT_TRUE(summary.failures.empty()) << join_failures(summary.failures);
    EXPECT_GT(summary.tested_model_count, 0u) << "No valid OBJ models were exercised.";
    EXPECT_TRUE(summary.seen_order_1);
    EXPECT_TRUE(summary.seen_order_2);
    EXPECT_EQ(summary.width_count, 4u);
    EXPECT_EQ(summary.index_count, 2u);
    EXPECT_EQ(summary.combo_count_per_order, 16u);
}

} // namespace
