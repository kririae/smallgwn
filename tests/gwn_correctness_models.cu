#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "reference_cpu.hpp"

namespace {

using Real = float;
using Index = std::int64_t;

struct HostMesh {
    std::vector<Real> vertex_x;
    std::vector<Real> vertex_y;
    std::vector<Real> vertex_z;
    std::vector<Index> tri_i0;
    std::vector<Index> tri_i1;
    std::vector<Index> tri_i2;
};

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

struct RunningErrorStats {
    double max_abs{0.0};
    double mean_abs{0.0};
    std::size_t count{0};
    std::size_t over_threshold{0};
};

struct ErrorSummary {
    double max_abs{0.0};
    double mean_abs{0.0};
    double p95_abs{0.0};
    double p99_abs{0.0};
};

[[nodiscard]] bool is_cuda_runtime_unavailable_message(std::string_view const message) noexcept {
    return message.find("cudaErrorNoDevice") != std::string_view::npos ||
           message.find("cudaErrorInsufficientDriver") != std::string_view::npos ||
           message.find("cudaErrorInitializationError") != std::string_view::npos ||
           message.find("cudaErrorSystemDriverMismatch") != std::string_view::npos ||
           message.find("cudaErrorOperatingSystem") != std::string_view::npos ||
           message.find("cudaErrorSystemNotReady") != std::string_view::npos ||
           message.find("cudaErrorNotSupported") != std::string_view::npos;
}

[[nodiscard]] std::string status_to_debug_string(gwn::gwn_status const &status) {
    std::ostringstream out;
    out << status.message();
    if (status.has_location()) {
        std::source_location const loc = status.location();
        out << " at " << loc.file_name() << ":" << loc.line();
    }
    return out.str();
}

[[nodiscard]] std::size_t get_env_positive_size_t(char const *name, std::size_t default_value) {
    char const *value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return default_value;

    std::size_t parsed = 0;
    char const *end = value + std::char_traits<char>::length(value);
    auto const [ptr, ec] = std::from_chars(value, end, parsed);
    if (ec != std::errc() || ptr != end || parsed == 0)
        return default_value;
    return parsed;
}

[[nodiscard]] std::string_view trim_left(std::string_view const line) noexcept {
    std::size_t start = 0;
    while (start < line.size() &&
           (line[start] == ' ' || line[start] == '\t' || line[start] == '\r')) {
        ++start;
    }
    return line.substr(start);
}

[[nodiscard]] std::optional<Index>
parse_obj_index(std::string_view const token, std::size_t const vertex_count) {
    if (token.empty())
        return std::nullopt;

    std::size_t const slash = token.find('/');
    std::string_view const index_token =
        (slash == std::string_view::npos) ? token : token.substr(0, slash);
    if (index_token.empty())
        return std::nullopt;

    Index raw = 0;
    char const *begin = index_token.data();
    char const *end = index_token.data() + index_token.size();
    auto const [ptr, ec] = std::from_chars(begin, end, raw);
    if (ec != std::errc() || ptr != end || raw == 0)
        return std::nullopt;

    Index const resolved = (raw > 0) ? (raw - 1) : (static_cast<Index>(vertex_count) + raw);
    if (resolved < 0 || static_cast<std::size_t>(resolved) >= vertex_count)
        return std::nullopt;

    return resolved;
}

[[nodiscard]] std::optional<HostMesh> load_obj_mesh(std::filesystem::path const &path) {
    std::ifstream input(path);
    if (!input.is_open())
        return std::nullopt;

    HostMesh mesh;
    std::string line;
    while (std::getline(input, line)) {
        std::string_view const trimmed = trim_left(line);
        if (trimmed.size() < 2 || trimmed[0] == '#')
            continue;

        if (trimmed.starts_with("v ")) {
            std::istringstream in(std::string(trimmed.substr(2)));
            Real x = Real(0);
            Real y = Real(0);
            Real z = Real(0);
            if (!(in >> x >> y >> z))
                continue;
            mesh.vertex_x.push_back(x);
            mesh.vertex_y.push_back(y);
            mesh.vertex_z.push_back(z);
            continue;
        }

        if (!trimmed.starts_with("f "))
            continue;

        std::istringstream in(std::string(trimmed.substr(2)));
        std::vector<Index> polygon{};
        std::string token;
        while (in >> token) {
            std::optional<Index> const parsed = parse_obj_index(token, mesh.vertex_x.size());
            if (parsed.has_value())
                polygon.push_back(*parsed);
        }
        if (polygon.size() < 3)
            continue;

        Index const first = polygon[0];
        for (std::size_t corner = 1; corner + 1 < polygon.size(); ++corner) {
            mesh.tri_i0.push_back(first);
            mesh.tri_i1.push_back(polygon[corner]);
            mesh.tri_i2.push_back(polygon[corner + 1]);
        }
    }

    if (mesh.vertex_x.empty() || mesh.tri_i0.empty())
        return std::nullopt;
    return mesh;
}

[[nodiscard]] std::vector<std::filesystem::path>
collect_obj_model_paths(std::filesystem::path const &model_dir) {
    std::vector<std::filesystem::path> model_paths{};
    for (auto const &entry : std::filesystem::directory_iterator(model_dir)) {
        if (!entry.is_regular_file())
            continue;
        std::filesystem::path const path = entry.path();
        if (path.extension() == ".obj")
            model_paths.push_back(path);
    }
    std::sort(model_paths.begin(), model_paths.end());
    model_paths.erase(std::unique(model_paths.begin(), model_paths.end()), model_paths.end());
    return model_paths;
}

[[nodiscard]] std::vector<std::filesystem::path> collect_model_paths() {
    std::vector<std::filesystem::path> model_paths{};

    if (char const *path_env = std::getenv("SMALLGWN_MODEL_PATH");
        path_env != nullptr && *path_env != '\0') {
        std::filesystem::path const path(path_env);
        if (std::filesystem::is_regular_file(path) && path.extension() == ".obj")
            model_paths.push_back(path);
    }

    if (char const *dir_env = std::getenv("SMALLGWN_MODEL_DATA_DIR");
        dir_env != nullptr && *dir_env != '\0') {
        std::filesystem::path const path(dir_env);
        if (std::filesystem::is_directory(path)) {
            auto const dir_models = collect_obj_model_paths(path);
            model_paths.insert(model_paths.end(), dir_models.begin(), dir_models.end());
        }
    } else {
        std::filesystem::path const default_path("/tmp/common-3d-test-models/data");
        if (std::filesystem::is_directory(default_path)) {
            auto const dir_models = collect_obj_model_paths(default_path);
            model_paths.insert(model_paths.end(), dir_models.begin(), dir_models.end());
        }
    }

    std::sort(model_paths.begin(), model_paths.end());
    model_paths.erase(std::unique(model_paths.begin(), model_paths.end()), model_paths.end());
    return model_paths;
}

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
    double const ratio_x = static_cast<double>(extent_x / max_extent);
    double const ratio_y = static_cast<double>(extent_y / max_extent);
    double const ratio_z = static_cast<double>(extent_z / max_extent);
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
        double const raw_index = static_cast<double>((center - origin) / step);
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
        std::size_t const idx = static_cast<std::size_t>(
            std::floor(q * static_cast<double>(absolute_errors.size() - 1))
        );
        return absolute_errors[idx];
    };
    summary.p95_abs = percentile(0.95);
    summary.p99_abs = percentile(0.99);
    return summary;
}

TEST(gwn_correctness_models, voxel_order1_levelwise_matches_iterative) {
    std::vector<std::filesystem::path> const model_paths = collect_model_paths();
    if (model_paths.empty())
        GTEST_SKIP() << "No model input found. Set SMALLGWN_MODEL_DATA_DIR or SMALLGWN_MODEL_PATH.";

    std::size_t const target_total_points =
        get_env_positive_size_t("SMALLGWN_VOXEL_TOTAL_POINTS", 10'000'000);
    std::size_t const chunk_size =
        get_env_positive_size_t("SMALLGWN_VOXEL_QUERY_CHUNK_SIZE", 1'000'000);
    std::size_t const target_points_per_model = std::max<std::size_t>(
        65'536, target_total_points / std::max<std::size_t>(1, model_paths.size())
    );

    constexpr Real k_accuracy_scale = Real(2);
    constexpr Real k_consistency_epsilon = Real(3e-4);

    std::size_t tested_model_count = 0;
    bool any_nonzero_winding = false;
    for (std::filesystem::path const &model_path : model_paths) {
        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = load_obj_mesh(model_path);
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
        if (!upload_status.is_ok() && upload_status.error() == gwn::gwn_error::cuda_runtime_error &&
            is_cuda_runtime_unavailable_message(upload_status.message())) {
            GTEST_SKIP() << "CUDA runtime unavailable: " << upload_status.message();
        }
        ASSERT_TRUE(upload_status.is_ok()) << status_to_debug_string(upload_status);

        gwn::gwn_bvh_object<Real, Index> bvh_iterative;
        gwn::gwn_status const build_iterative_status =
            gwn::gwn_build_bvh4_lbvh_taylor<1, Real, Index>(
                geometry.accessor(), bvh_iterative.accessor()
            );
        ASSERT_TRUE(build_iterative_status.is_ok())
            << status_to_debug_string(build_iterative_status);

        gwn::gwn_bvh_object<Real, Index> bvh_levelwise;
        gwn::gwn_status const build_levelwise_status =
            gwn::gwn_build_bvh4_lbvh_taylor_levelwise<1, Real, Index>(
                geometry.accessor(), bvh_levelwise.accessor()
            );
        ASSERT_TRUE(build_levelwise_status.is_ok())
            << status_to_debug_string(build_levelwise_status);

        Real *d_query_x = nullptr;
        Real *d_query_y = nullptr;
        Real *d_query_z = nullptr;
        Real *d_output_iterative = nullptr;
        Real *d_output_levelwise = nullptr;
        auto cleanup = gwn::gwn_make_scope_exit([&]() noexcept {
            if (d_output_levelwise != nullptr)
                (void)gwn::gwn_cuda_free(d_output_levelwise);
            if (d_output_iterative != nullptr)
                (void)gwn::gwn_cuda_free(d_output_iterative);
            if (d_query_z != nullptr)
                (void)gwn::gwn_cuda_free(d_query_z);
            if (d_query_y != nullptr)
                (void)gwn::gwn_cuda_free(d_query_y);
            if (d_query_x != nullptr)
                (void)gwn::gwn_cuda_free(d_query_x);
        });

        std::size_t const alloc_count = std::min(chunk_size, query_count);
        std::size_t const alloc_bytes = alloc_count * sizeof(Real);
        ASSERT_TRUE(
            gwn::gwn_cuda_malloc(reinterpret_cast<void **>(&d_query_x), alloc_bytes).is_ok()
        );
        ASSERT_TRUE(
            gwn::gwn_cuda_malloc(reinterpret_cast<void **>(&d_query_y), alloc_bytes).is_ok()
        );
        ASSERT_TRUE(
            gwn::gwn_cuda_malloc(reinterpret_cast<void **>(&d_query_z), alloc_bytes).is_ok()
        );
        ASSERT_TRUE(
            gwn::gwn_cuda_malloc(reinterpret_cast<void **>(&d_output_iterative), alloc_bytes)
                .is_ok()
        );
        ASSERT_TRUE(
            gwn::gwn_cuda_malloc(reinterpret_cast<void **>(&d_output_levelwise), alloc_bytes)
                .is_ok()
        );

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

            std::size_t const bytes = count * sizeof(Real);
            ASSERT_EQ(
                cudaSuccess, cudaMemcpy(d_query_x, query_x.data(), bytes, cudaMemcpyHostToDevice)
            );
            ASSERT_EQ(
                cudaSuccess, cudaMemcpy(d_query_y, query_y.data(), bytes, cudaMemcpyHostToDevice)
            );
            ASSERT_EQ(
                cudaSuccess, cudaMemcpy(d_query_z, query_z.data(), bytes, cudaMemcpyHostToDevice)
            );

            gwn::gwn_status const iterative_query_status =
                gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                    geometry.accessor(), bvh_iterative.accessor(),
                    cuda::std::span<Real const>(d_query_x, count),
                    cuda::std::span<Real const>(d_query_y, count),
                    cuda::std::span<Real const>(d_query_z, count),
                    cuda::std::span<Real>(d_output_iterative, count), k_accuracy_scale
                );
            ASSERT_TRUE(iterative_query_status.is_ok())
                << status_to_debug_string(iterative_query_status);

            gwn::gwn_status const levelwise_query_status =
                gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                    geometry.accessor(), bvh_levelwise.accessor(),
                    cuda::std::span<Real const>(d_query_x, count),
                    cuda::std::span<Real const>(d_query_y, count),
                    cuda::std::span<Real const>(d_query_z, count),
                    cuda::std::span<Real>(d_output_levelwise, count), k_accuracy_scale
                );
            ASSERT_TRUE(levelwise_query_status.is_ok())
                << status_to_debug_string(levelwise_query_status);

            ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
            ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(host_iterative.data(), d_output_iterative, bytes, cudaMemcpyDeviceToHost)
            );
            ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(host_levelwise.data(), d_output_levelwise, bytes, cudaMemcpyDeviceToHost)
            );

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
            << "Levelwise and iterative order-1 Taylor mismatch on model " << model_path.string();
    }

    ASSERT_GT(tested_model_count, 0u) << "No valid OBJ models were exercised.";
    EXPECT_TRUE(any_nonzero_winding)
        << "All tested models produced near-zero winding values across voxel queries.";
}

TEST(gwn_correctness_models, voxel_exact_and_taylor_match_cpu_on_small_models) {
    std::vector<std::filesystem::path> const model_paths = collect_model_paths();
    if (model_paths.empty())
        GTEST_SKIP() << "No model input found. Set SMALLGWN_MODEL_DATA_DIR or SMALLGWN_MODEL_PATH.";

    std::size_t const target_total_points =
        get_env_positive_size_t("SMALLGWN_VOXEL_TOTAL_POINTS", 10'000'000);
    std::size_t const target_points_per_model = std::max<std::size_t>(
        65'536, target_total_points / std::max<std::size_t>(1, model_paths.size())
    );
    std::size_t const cpu_work_budget =
        get_env_positive_size_t("SMALLGWN_CPU_WORK_BUDGET", 120'000'000);
    std::size_t const max_cpu_samples = get_env_positive_size_t("SMALLGWN_CPU_MAX_SAMPLES", 50'000);
    std::size_t const min_cpu_samples = get_env_positive_size_t("SMALLGWN_CPU_MIN_SAMPLES", 4'096);

    constexpr Real k_accuracy_scale = Real(2);
    std::size_t tested_model_count = 0;
    bool any_nonzero_exact = false;

    for (std::filesystem::path const &model_path : model_paths) {
        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = load_obj_mesh(model_path);
        if (!maybe_mesh.has_value())
            continue;
        HostMesh const &mesh = *maybe_mesh;

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
        std::size_t const sample_count = std::min({voxel_count, max_cpu_samples, samples_by_work});
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
            query_x, query_y, query_z
        );

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
                std::span<Real>(reference_output.data(), reference_output.size())
            );
        ASSERT_TRUE(reference_status.is_ok()) << status_to_debug_string(reference_status);

        gwn::gwn_geometry_object<Real, Index> geometry;
        gwn::gwn_status const upload_status = geometry.upload(
            cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size())
        );
        if (!upload_status.is_ok() && upload_status.error() == gwn::gwn_error::cuda_runtime_error &&
            is_cuda_runtime_unavailable_message(upload_status.message())) {
            GTEST_SKIP() << "CUDA runtime unavailable: " << upload_status.message();
        }
        ASSERT_TRUE(upload_status.is_ok()) << status_to_debug_string(upload_status);

        Real *d_query_x = nullptr;
        Real *d_query_y = nullptr;
        Real *d_query_z = nullptr;
        Real *d_output = nullptr;
        auto cleanup = gwn::gwn_make_scope_exit([&]() noexcept {
            if (d_output != nullptr)
                (void)gwn::gwn_cuda_free(d_output);
            if (d_query_z != nullptr)
                (void)gwn::gwn_cuda_free(d_query_z);
            if (d_query_y != nullptr)
                (void)gwn::gwn_cuda_free(d_query_y);
            if (d_query_x != nullptr)
                (void)gwn::gwn_cuda_free(d_query_x);
        });

        std::size_t const query_bytes = sample_count * sizeof(Real);
        ASSERT_TRUE(
            gwn::gwn_cuda_malloc(reinterpret_cast<void **>(&d_query_x), query_bytes).is_ok()
        );
        ASSERT_TRUE(
            gwn::gwn_cuda_malloc(reinterpret_cast<void **>(&d_query_y), query_bytes).is_ok()
        );
        ASSERT_TRUE(
            gwn::gwn_cuda_malloc(reinterpret_cast<void **>(&d_query_z), query_bytes).is_ok()
        );
        ASSERT_TRUE(
            gwn::gwn_cuda_malloc(reinterpret_cast<void **>(&d_output), query_bytes).is_ok()
        );

        ASSERT_EQ(
            cudaSuccess, cudaMemcpy(d_query_x, query_x.data(), query_bytes, cudaMemcpyHostToDevice)
        );
        ASSERT_EQ(
            cudaSuccess, cudaMemcpy(d_query_y, query_y.data(), query_bytes, cudaMemcpyHostToDevice)
        );
        ASSERT_EQ(
            cudaSuccess, cudaMemcpy(d_query_z, query_z.data(), query_bytes, cudaMemcpyHostToDevice)
        );

        std::vector<Real> exact_output(sample_count, Real(0));
        std::vector<Real> order0_output(sample_count, Real(0));
        std::vector<Real> order1_iterative_output(sample_count, Real(0));
        std::vector<Real> order1_levelwise_output(sample_count, Real(0));

        gwn::gwn_bvh_object<Real, Index> bvh_exact;
        gwn::gwn_status const exact_build_status =
            gwn::gwn_build_bvh4_lbvh<Real, Index>(geometry.accessor(), bvh_exact.accessor());
        ASSERT_TRUE(exact_build_status.is_ok()) << status_to_debug_string(exact_build_status);

        gwn::gwn_status const exact_query_status =
            gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
                geometry.accessor(), bvh_exact.accessor(),
                cuda::std::span<Real const>(d_query_x, sample_count),
                cuda::std::span<Real const>(d_query_y, sample_count),
                cuda::std::span<Real const>(d_query_z, sample_count),
                cuda::std::span<Real>(d_output, sample_count)
            );
        ASSERT_TRUE(exact_query_status.is_ok()) << status_to_debug_string(exact_query_status);
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(exact_output.data(), d_output, query_bytes, cudaMemcpyDeviceToHost)
        );

        gwn::gwn_bvh_object<Real, Index> bvh_order0;
        gwn::gwn_status const order0_build_status = gwn::gwn_build_bvh4_lbvh_taylor<0, Real, Index>(
            geometry.accessor(), bvh_order0.accessor()
        );
        ASSERT_TRUE(order0_build_status.is_ok()) << status_to_debug_string(order0_build_status);
        gwn::gwn_status const order0_query_status =
            gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
                geometry.accessor(), bvh_order0.accessor(),
                cuda::std::span<Real const>(d_query_x, sample_count),
                cuda::std::span<Real const>(d_query_y, sample_count),
                cuda::std::span<Real const>(d_query_z, sample_count),
                cuda::std::span<Real>(d_output, sample_count), k_accuracy_scale
            );
        ASSERT_TRUE(order0_query_status.is_ok()) << status_to_debug_string(order0_query_status);
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(order0_output.data(), d_output, query_bytes, cudaMemcpyDeviceToHost)
        );

        gwn::gwn_bvh_object<Real, Index> bvh_order1_iterative;
        gwn::gwn_status const order1_iterative_build_status =
            gwn::gwn_build_bvh4_lbvh_taylor<1, Real, Index>(
                geometry.accessor(), bvh_order1_iterative.accessor()
            );
        ASSERT_TRUE(order1_iterative_build_status.is_ok())
            << status_to_debug_string(order1_iterative_build_status);
        gwn::gwn_status const order1_iterative_query_status =
            gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                geometry.accessor(), bvh_order1_iterative.accessor(),
                cuda::std::span<Real const>(d_query_x, sample_count),
                cuda::std::span<Real const>(d_query_y, sample_count),
                cuda::std::span<Real const>(d_query_z, sample_count),
                cuda::std::span<Real>(d_output, sample_count), k_accuracy_scale
            );
        ASSERT_TRUE(order1_iterative_query_status.is_ok())
            << status_to_debug_string(order1_iterative_query_status);
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(
                order1_iterative_output.data(), d_output, query_bytes, cudaMemcpyDeviceToHost
            )
        );

        gwn::gwn_bvh_object<Real, Index> bvh_order1_levelwise;
        gwn::gwn_status const order1_levelwise_build_status =
            gwn::gwn_build_bvh4_lbvh_taylor_levelwise<1, Real, Index>(
                geometry.accessor(), bvh_order1_levelwise.accessor()
            );
        ASSERT_TRUE(order1_levelwise_build_status.is_ok())
            << status_to_debug_string(order1_levelwise_build_status);
        gwn::gwn_status const order1_levelwise_query_status =
            gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                geometry.accessor(), bvh_order1_levelwise.accessor(),
                cuda::std::span<Real const>(d_query_x, sample_count),
                cuda::std::span<Real const>(d_query_y, sample_count),
                cuda::std::span<Real const>(d_query_z, sample_count),
                cuda::std::span<Real>(d_output, sample_count), k_accuracy_scale
            );
        ASSERT_TRUE(order1_levelwise_query_status.is_ok())
            << status_to_debug_string(order1_levelwise_query_status);
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(
                order1_levelwise_output.data(), d_output, query_bytes, cudaMemcpyDeviceToHost
            )
        );

        ErrorSummary const exact_vs_cpu = summarize_error(
            std::span<Real const>(exact_output.data(), exact_output.size()),
            std::span<Real const>(reference_output.data(), reference_output.size())
        );
        ErrorSummary const order0_vs_exact = summarize_error(
            std::span<Real const>(order0_output.data(), order0_output.size()),
            std::span<Real const>(exact_output.data(), exact_output.size())
        );
        ErrorSummary const order1_vs_exact = summarize_error(
            std::span<Real const>(order1_iterative_output.data(), order1_iterative_output.size()),
            std::span<Real const>(exact_output.data(), exact_output.size())
        );
        ErrorSummary const levelwise_vs_iterative = summarize_error(
            std::span<Real const>(order1_levelwise_output.data(), order1_levelwise_output.size()),
            std::span<Real const>(order1_iterative_output.data(), order1_iterative_output.size())
        );
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
    EXPECT_TRUE(any_nonzero_exact) << "All sampled models produced near-zero exact winding values.";
}

} // namespace
