#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gwn/gwn.cuh>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

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

[[nodiscard]] std::optional<std::filesystem::path> find_model_data_dir() {
    if (char const *env = std::getenv("SMALLGWN_MODEL_DATA_DIR"); env != nullptr && *env != '\0') {
        std::filesystem::path const path(env);
        if (std::filesystem::is_directory(path))
            return path;
    }

    std::filesystem::path const default_path("/tmp/common-3d-test-models/data");
    if (std::filesystem::is_directory(default_path))
        return default_path;

    return std::nullopt;
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

std::array<std::vector<Real>, 3> make_query_soa(HostMesh const &mesh) {
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
    Real const extent_x = std::max(max_x - min_x, Real(1e-2));
    Real const extent_y = std::max(max_y - min_y, Real(1e-2));
    Real const extent_z = std::max(max_z - min_z, Real(1e-2));

    std::array<std::array<Real, 3>, 9> const queries = {{
        {{center_x, center_y, center_z}},
        {{center_x + Real(1.3) * extent_x, center_y, center_z}},
        {{center_x - Real(1.3) * extent_x, center_y, center_z}},
        {{center_x, center_y + Real(1.3) * extent_y, center_z}},
        {{center_x, center_y - Real(1.3) * extent_y, center_z}},
        {{center_x, center_y, center_z + Real(1.3) * extent_z}},
        {{center_x, center_y, center_z - Real(1.3) * extent_z}},
        {{center_x + Real(0.31) * extent_x,
          center_y - Real(0.27) * extent_y,
          center_z + Real(0.23) * extent_z}},
        {{center_x - Real(0.19) * extent_x,
          center_y + Real(0.37) * extent_y,
          center_z - Real(0.29) * extent_z}},
    }};

    std::array<std::vector<Real>, 3> soa{};
    soa[0].reserve(queries.size());
    soa[1].reserve(queries.size());
    soa[2].reserve(queries.size());
    for (auto const &query : queries) {
        soa[0].push_back(query[0]);
        soa[1].push_back(query[1]);
        soa[2].push_back(query[2]);
    }
    return soa;
}

void assert_bvh_structure(
    gwn::gwn_bvh_accessor<Real, Index> const &accessor, std::size_t const primitive_count
) {
    ASSERT_EQ(accessor.primitive_indices.size(), primitive_count);

    std::vector<Index> primitive_indices(primitive_count, Index(0));
    if (primitive_count > 0) {
        ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(
                primitive_indices.data(),
                accessor.primitive_indices.data(),
                primitive_count * sizeof(Index),
                cudaMemcpyDeviceToHost
            )
        );
    }

    std::vector<int> primitive_seen(primitive_count, 0);
    for (Index const index : primitive_indices) {
        ASSERT_GE(index, Index(0));
        ASSERT_LT(static_cast<std::size_t>(index), primitive_count);
        ++primitive_seen[static_cast<std::size_t>(index)];
    }
    for (int const seen : primitive_seen)
        EXPECT_EQ(seen, 1);

    if (accessor.root_kind == gwn::gwn_bvh_child_kind::k_leaf) {
        ASSERT_TRUE(accessor.nodes.empty());
        ASSERT_GE(accessor.root_index, Index(0));
        ASSERT_GE(accessor.root_count, Index(0));
        std::size_t const begin = static_cast<std::size_t>(accessor.root_index);
        std::size_t const count = static_cast<std::size_t>(accessor.root_count);
        ASSERT_LE(begin, primitive_count);
        ASSERT_LE(count, primitive_count - begin);
        return;
    }

    ASSERT_EQ(accessor.root_kind, gwn::gwn_bvh_child_kind::k_internal);
    ASSERT_FALSE(accessor.nodes.empty());
    ASSERT_GE(accessor.root_index, Index(0));
    ASSERT_LT(static_cast<std::size_t>(accessor.root_index), accessor.nodes.size());

    std::vector<gwn::gwn_bvh4_node_soa<Real, Index>> nodes(accessor.nodes.size());
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            nodes.data(),
            accessor.nodes.data(),
            nodes.size() * sizeof(nodes[0]),
            cudaMemcpyDeviceToHost
        )
    );

    std::vector<int> sorted_slot_seen(primitive_count, 0);
    std::vector<Index> stack{accessor.root_index};
    std::vector<char> visited(nodes.size(), 0);
    while (!stack.empty()) {
        Index const node_index = stack.back();
        stack.pop_back();
        ASSERT_GE(node_index, Index(0));
        ASSERT_LT(static_cast<std::size_t>(node_index), nodes.size());
        if (visited[static_cast<std::size_t>(node_index)] != 0)
            continue;
        visited[static_cast<std::size_t>(node_index)] = 1;

        auto const &node = nodes[static_cast<std::size_t>(node_index)];
        for (int slot = 0; slot < 4; ++slot) {
            EXPECT_LE(node.child_min_x[slot], node.child_max_x[slot]);
            EXPECT_LE(node.child_min_y[slot], node.child_max_y[slot]);
            EXPECT_LE(node.child_min_z[slot], node.child_max_z[slot]);

            auto const kind = static_cast<gwn::gwn_bvh_child_kind>(node.child_kind[slot]);
            if (kind == gwn::gwn_bvh_child_kind::k_invalid)
                continue;
            if (kind == gwn::gwn_bvh_child_kind::k_internal) {
                ASSERT_GE(node.child_index[slot], Index(0));
                ASSERT_LT(static_cast<std::size_t>(node.child_index[slot]), nodes.size());
                stack.push_back(node.child_index[slot]);
                continue;
            }
            ASSERT_EQ(kind, gwn::gwn_bvh_child_kind::k_leaf);
            ASSERT_GE(node.child_index[slot], Index(0));
            ASSERT_GE(node.child_count[slot], Index(0));

            std::size_t const begin = static_cast<std::size_t>(node.child_index[slot]);
            std::size_t const count = static_cast<std::size_t>(node.child_count[slot]);
            ASSERT_LE(begin, primitive_count);
            ASSERT_LE(count, primitive_count - begin);
            for (std::size_t i = begin; i < begin + count; ++i)
                ++sorted_slot_seen[i];
        }
    }
    for (int const seen : sorted_slot_seen)
        EXPECT_EQ(seen, 1);
}

TEST(smallgwn_bvh_models, bvh_exact_batch_matches_cpu_on_common_models) {
    std::optional<std::filesystem::path> const model_dir = find_model_data_dir();
    if (!model_dir.has_value()) {
        GTEST_SKIP() << "Model directory not found. Set SMALLGWN_MODEL_DATA_DIR or "
                        "clone models to /tmp/common-3d-test-models/data.";
    }

    std::array<std::string_view, 5> const model_names = {
        "suzanne.obj", "teapot.obj", "cow.obj", "stanford-bunny.obj", "armadillo.obj"
    };

    for (std::string_view const model_name : model_names) {
        std::filesystem::path const model_path = *model_dir / model_name;
        if (!std::filesystem::exists(model_path))
            continue;

        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = load_obj_mesh(model_path);
        ASSERT_TRUE(maybe_mesh.has_value());
        HostMesh const &mesh = *maybe_mesh;

        auto const query_soa = make_query_soa(mesh);
        std::vector<Real> reference_output(query_soa[0].size(), Real(0));
        gwn::gwn_status const reference_status =
            gwn::tests::reference_winding_number_batch<Real, Index>(
                std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
                std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
                std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
                std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
                std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
                std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()),
                std::span<Real const>(query_soa[0].data(), query_soa[0].size()),
                std::span<Real const>(query_soa[1].data(), query_soa[1].size()),
                std::span<Real const>(query_soa[2].data(), query_soa[2].size()),
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

        gwn::gwn_bvh_object<Real, Index> bvh;
        gwn::gwn_status const build_status =
            gwn::gwn_build_bvh4_lbvh<Real, Index>(geometry.accessor(), bvh.accessor());
        ASSERT_TRUE(build_status.is_ok()) << status_to_debug_string(build_status);
        ASSERT_TRUE(bvh.has_bvh());

        assert_bvh_structure(bvh.accessor(), mesh.tri_i0.size());

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

        std::size_t const query_count = query_soa[0].size();
        std::size_t const query_bytes = query_count * sizeof(Real);

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
            cudaSuccess,
            cudaMemcpy(d_query_x, query_soa[0].data(), query_bytes, cudaMemcpyHostToDevice)
        );
        ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(d_query_y, query_soa[1].data(), query_bytes, cudaMemcpyHostToDevice)
        );
        ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(d_query_z, query_soa[2].data(), query_bytes, cudaMemcpyHostToDevice)
        );

        gwn::gwn_status const query_status =
            gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
                geometry.accessor(),
                bvh.accessor(),
                cuda::std::span<Real const>(d_query_x, query_count),
                cuda::std::span<Real const>(d_query_y, query_count),
                cuda::std::span<Real const>(d_query_z, query_count),
                cuda::std::span<Real>(d_output, query_count)
            );
        ASSERT_TRUE(query_status.is_ok()) << status_to_debug_string(query_status);
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        std::vector<Real> gpu_output(query_count, Real(0));
        ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(gpu_output.data(), d_output, query_bytes, cudaMemcpyDeviceToHost)
        );

        for (std::size_t i = 0; i < query_count; ++i)
            EXPECT_NEAR(gpu_output[i], reference_output[i], Real(5e-4)) << "query index: " << i;
    }
}

} // namespace
