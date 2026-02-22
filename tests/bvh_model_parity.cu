#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "reference_cpu.hpp"

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
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

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

[[nodiscard]] bool is_cuda_runtime_unavailable(
    const cudaError_t result) noexcept {
  return result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
         result == cudaErrorInitializationError ||
         result == cudaErrorSystemDriverMismatch ||
         result == cudaErrorOperatingSystem ||
         result == cudaErrorSystemNotReady || result == cudaErrorNotSupported;
}

[[nodiscard]] std::string status_to_debug_string(
    const gwn::gwn_status& status) {
  std::ostringstream out;
  out << status.message();
  if (status.has_detail_code()) {
    out << " [detail_code=" << status.detail_code() << "]";
  }
  if (status.is_cuda_runtime_error()) {
    const cudaError_t error = status.cuda_error();
    out << " [cuda_name=" << cudaGetErrorName(error) << ", cuda_message=\""
        << cudaGetErrorString(error) << "\"]";
  }
  if (status.has_location()) {
    const std::source_location loc = status.location();
    out << " at " << loc.file_name() << ":" << loc.line();
  }
  return out.str();
}

[[nodiscard]] std::optional<std::filesystem::path> find_model_data_dir() {
  if (const char* env = std::getenv("SMALLGWN_MODEL_DATA_DIR");
      env != nullptr && *env != '\0') {
    const std::filesystem::path path(env);
    if (std::filesystem::is_directory(path)) {
      return path;
    }
  }

  const std::filesystem::path default_path("/tmp/common-3d-test-models/data");
  if (std::filesystem::is_directory(default_path)) {
    return default_path;
  }

  return std::nullopt;
}

[[nodiscard]] std::string_view trim_left(const std::string_view line) noexcept {
  std::size_t start = 0;
  while (start < line.size() &&
         (line[start] == ' ' || line[start] == '\t' || line[start] == '\r')) {
    ++start;
  }
  return line.substr(start);
}

[[nodiscard]] std::optional<Index> parse_obj_index(
    const std::string_view token,
    const std::size_t vertex_count) {
  if (token.empty()) {
    return std::nullopt;
  }

  const std::size_t slash = token.find('/');
  const std::string_view index_token =
      (slash == std::string_view::npos) ? token : token.substr(0, slash);
  if (index_token.empty()) {
    return std::nullopt;
  }

  Index raw = 0;
  const char* begin = index_token.data();
  const char* end = index_token.data() + index_token.size();
  const auto [ptr, ec] = std::from_chars(begin, end, raw);
  if (ec != std::errc() || ptr != end || raw == 0) {
    return std::nullopt;
  }

  const Index resolved =
      (raw > 0) ? (raw - 1) : (static_cast<Index>(vertex_count) + raw);
  if (resolved < 0 || static_cast<std::size_t>(resolved) >= vertex_count) {
    return std::nullopt;
  }

  return resolved;
}

[[nodiscard]] std::optional<HostMesh> load_obj_mesh(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    return std::nullopt;
  }

  HostMesh mesh;
  std::string line;
  while (std::getline(input, line)) {
    const std::string_view trimmed = trim_left(line);
    if (trimmed.size() < 2 || trimmed[0] == '#') {
      continue;
    }

    if (trimmed.starts_with("v ")) {
      std::istringstream in(std::string(trimmed.substr(2)));
      Real x = Real(0);
      Real y = Real(0);
      Real z = Real(0);
      if (!(in >> x >> y >> z)) {
        continue;
      }
      mesh.vertex_x.push_back(x);
      mesh.vertex_y.push_back(y);
      mesh.vertex_z.push_back(z);
      continue;
    }

    if (!trimmed.starts_with("f ")) {
      continue;
    }

    std::istringstream in(std::string(trimmed.substr(2)));
    std::vector<Index> polygon{};
    std::string token;
    while (in >> token) {
      const std::optional<Index> parsed =
          parse_obj_index(token, mesh.vertex_x.size());
      if (parsed.has_value()) {
        polygon.push_back(*parsed);
      }
    }
    if (polygon.size() < 3) {
      continue;
    }

    const Index first = polygon[0];
    for (std::size_t corner = 1; corner + 1 < polygon.size(); ++corner) {
      mesh.tri_i0.push_back(first);
      mesh.tri_i1.push_back(polygon[corner]);
      mesh.tri_i2.push_back(polygon[corner + 1]);
    }
  }

  if (mesh.vertex_x.empty() || mesh.tri_i0.empty()) {
    return std::nullopt;
  }

  return mesh;
}

std::array<std::vector<Real>, 3> make_query_soa(const HostMesh& mesh) {
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

  const Real center_x = (min_x + max_x) * Real(0.5);
  const Real center_y = (min_y + max_y) * Real(0.5);
  const Real center_z = (min_z + max_z) * Real(0.5);
  const Real extent_x = std::max(max_x - min_x, Real(1e-2));
  const Real extent_y = std::max(max_y - min_y, Real(1e-2));
  const Real extent_z = std::max(max_z - min_z, Real(1e-2));

  const std::array<std::array<Real, 3>, 9> queries = {{
      {{center_x, center_y, center_z}},
      {{center_x + Real(1.3) * extent_x, center_y, center_z}},
      {{center_x - Real(1.3) * extent_x, center_y, center_z}},
      {{center_x, center_y + Real(1.3) * extent_y, center_z}},
      {{center_x, center_y - Real(1.3) * extent_y, center_z}},
      {{center_x, center_y, center_z + Real(1.3) * extent_z}},
      {{center_x, center_y, center_z - Real(1.3) * extent_z}},
      {{center_x + Real(0.31) * extent_x, center_y - Real(0.27) * extent_y,
        center_z + Real(0.23) * extent_z}},
      {{center_x - Real(0.19) * extent_x, center_y + Real(0.37) * extent_y,
        center_z - Real(0.29) * extent_z}},
  }};

  std::array<std::vector<Real>, 3> soa{};
  soa[0].reserve(queries.size());
  soa[1].reserve(queries.size());
  soa[2].reserve(queries.size());
  for (const auto& query : queries) {
    soa[0].push_back(query[0]);
    soa[1].push_back(query[1]);
    soa[2].push_back(query[2]);
  }
  return soa;
}

void assert_bvh_structure(const gwn::gwn_bvh_accessor<Real, Index>& accessor,
                          const std::size_t primitive_count) {
  ASSERT_EQ(accessor.primitive_indices.size(), primitive_count);

  std::vector<Index> primitive_indices(primitive_count, Index(0));
  if (primitive_count > 0) {
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(primitive_indices.data(), accessor.primitive_indices.data(),
                   primitive_count * sizeof(Index), cudaMemcpyDeviceToHost));
  }

  std::vector<int> primitive_seen(primitive_count, 0);
  for (const Index index : primitive_indices) {
    ASSERT_GE(index, Index(0));
    ASSERT_LT(static_cast<std::size_t>(index), primitive_count);
    ++primitive_seen[static_cast<std::size_t>(index)];
  }
  for (const int seen : primitive_seen) {
    EXPECT_EQ(seen, 1);
  }

  if (accessor.root_kind == gwn::gwn_bvh_child_kind::k_leaf) {
    ASSERT_TRUE(accessor.nodes.empty());
    ASSERT_GE(accessor.root_index, Index(0));
    ASSERT_GE(accessor.root_count, Index(0));
    const std::size_t begin = static_cast<std::size_t>(accessor.root_index);
    const std::size_t count = static_cast<std::size_t>(accessor.root_count);
    ASSERT_LE(begin, primitive_count);
    ASSERT_LE(count, primitive_count - begin);
    return;
  }

  ASSERT_EQ(accessor.root_kind, gwn::gwn_bvh_child_kind::k_internal);
  ASSERT_FALSE(accessor.nodes.empty());
  ASSERT_GE(accessor.root_index, Index(0));
  ASSERT_LT(static_cast<std::size_t>(accessor.root_index),
            accessor.nodes.size());

  std::vector<gwn::gwn_bvh4_node_soa<Real, Index>> nodes(accessor.nodes.size());
  ASSERT_EQ(cudaSuccess, cudaMemcpy(nodes.data(), accessor.nodes.data(),
                                    nodes.size() * sizeof(nodes[0]),
                                    cudaMemcpyDeviceToHost));

  std::vector<int> sorted_slot_seen(primitive_count, 0);
  std::vector<Index> stack{accessor.root_index};
  std::vector<char> visited(nodes.size(), 0);
  while (!stack.empty()) {
    const Index node_index = stack.back();
    stack.pop_back();
    ASSERT_GE(node_index, Index(0));
    ASSERT_LT(static_cast<std::size_t>(node_index), nodes.size());
    if (visited[static_cast<std::size_t>(node_index)] != 0) {
      continue;
    }
    visited[static_cast<std::size_t>(node_index)] = 1;

    const auto& node = nodes[static_cast<std::size_t>(node_index)];
    for (int slot = 0; slot < 4; ++slot) {
      EXPECT_LE(node.child_min_x[slot], node.child_max_x[slot]);
      EXPECT_LE(node.child_min_y[slot], node.child_max_y[slot]);
      EXPECT_LE(node.child_min_z[slot], node.child_max_z[slot]);

      const auto kind =
          static_cast<gwn::gwn_bvh_child_kind>(node.child_kind[slot]);
      if (kind == gwn::gwn_bvh_child_kind::k_invalid) {
        continue;
      }
      if (kind == gwn::gwn_bvh_child_kind::k_internal) {
        ASSERT_GE(node.child_index[slot], Index(0));
        ASSERT_LT(static_cast<std::size_t>(node.child_index[slot]),
                  nodes.size());
        stack.push_back(node.child_index[slot]);
        continue;
      }
      ASSERT_EQ(kind, gwn::gwn_bvh_child_kind::k_leaf);
      ASSERT_GE(node.child_index[slot], Index(0));
      ASSERT_GE(node.child_count[slot], Index(0));

      const std::size_t begin =
          static_cast<std::size_t>(node.child_index[slot]);
      const std::size_t count =
          static_cast<std::size_t>(node.child_count[slot]);
      ASSERT_LE(begin, primitive_count);
      ASSERT_LE(count, primitive_count - begin);
      for (std::size_t i = begin; i < begin + count; ++i) {
        ++sorted_slot_seen[i];
      }
    }
  }
  for (const int seen : sorted_slot_seen) {
    EXPECT_EQ(seen, 1);
  }
}

TEST(smallgwn_bvh_models, bvh_exact_batch_matches_cpu_on_common_models) {
  const std::optional<std::filesystem::path> model_dir = find_model_data_dir();
  if (!model_dir.has_value()) {
    GTEST_SKIP() << "Model directory not found. Set SMALLGWN_MODEL_DATA_DIR or "
                    "clone models to /tmp/common-3d-test-models/data.";
  }

  const std::array<std::string_view, 5> model_names = {
      "suzanne.obj", "teapot.obj", "cow.obj", "stanford-bunny.obj",
      "armadillo.obj"};

  for (const std::string_view model_name : model_names) {
    const std::filesystem::path model_path = *model_dir / model_name;
    if (!std::filesystem::exists(model_path)) {
      continue;
    }

    SCOPED_TRACE(model_path.string());
    const std::optional<HostMesh> maybe_mesh = load_obj_mesh(model_path);
    ASSERT_TRUE(maybe_mesh.has_value());
    const HostMesh& mesh = *maybe_mesh;

    const auto query_soa = make_query_soa(mesh);
    std::vector<Real> reference_output(query_soa[0].size(), Real(0));
    const gwn::gwn_status reference_status =
        gwn::tests::reference_winding_number_batch<Real, Index>(
            std::span<const Real>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            std::span<const Real>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            std::span<const Real>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            std::span<const Index>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            std::span<const Index>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            std::span<const Index>(mesh.tri_i2.data(), mesh.tri_i2.size()),
            std::span<const Real>(query_soa[0].data(), query_soa[0].size()),
            std::span<const Real>(query_soa[1].data(), query_soa[1].size()),
            std::span<const Real>(query_soa[2].data(), query_soa[2].size()),
            std::span<Real>(reference_output.data(), reference_output.size()));
    ASSERT_TRUE(reference_status.is_ok())
        << status_to_debug_string(reference_status);

    gwn::gwn_geometry_object<Real, Index> geometry;
    const gwn::gwn_status upload_status = geometry.upload(
        cuda::std::span<const Real>(mesh.vertex_x.data(), mesh.vertex_x.size()),
        cuda::std::span<const Real>(mesh.vertex_y.data(), mesh.vertex_y.size()),
        cuda::std::span<const Real>(mesh.vertex_z.data(), mesh.vertex_z.size()),
        cuda::std::span<const Index>(mesh.tri_i0.data(), mesh.tri_i0.size()),
        cuda::std::span<const Index>(mesh.tri_i1.data(), mesh.tri_i1.size()),
        cuda::std::span<const Index>(mesh.tri_i2.data(), mesh.tri_i2.size()));
    if (!upload_status.is_ok() && upload_status.is_cuda_runtime_error() &&
        is_cuda_runtime_unavailable(upload_status.cuda_error())) {
      GTEST_SKIP() << "CUDA runtime unavailable: " << upload_status.message();
    }
    ASSERT_TRUE(upload_status.is_ok()) << status_to_debug_string(upload_status);

    gwn::gwn_bvh_object<Real, Index> bvh;
    const gwn::gwn_status build_status = gwn::gwn_build_bvh4_lbvh<Real, Index>(
        geometry.accessor(), bvh.accessor());
    ASSERT_TRUE(build_status.is_ok()) << status_to_debug_string(build_status);
    ASSERT_TRUE(bvh.has_bvh());

    assert_bvh_structure(bvh.accessor(), mesh.tri_i0.size());

    Real* d_query_x = nullptr;
    Real* d_query_y = nullptr;
    Real* d_query_z = nullptr;
    Real* d_output = nullptr;
    auto cleanup = gwn::gwn_make_scope_exit([&]() noexcept {
      if (d_output != nullptr) {
        (void)gwn::gwn_cuda_free(d_output);
      }
      if (d_query_z != nullptr) {
        (void)gwn::gwn_cuda_free(d_query_z);
      }
      if (d_query_y != nullptr) {
        (void)gwn::gwn_cuda_free(d_query_y);
      }
      if (d_query_x != nullptr) {
        (void)gwn::gwn_cuda_free(d_query_x);
      }
    });

    const std::size_t query_count = query_soa[0].size();
    const std::size_t query_bytes = query_count * sizeof(Real);

    ASSERT_TRUE(
        gwn::gwn_cuda_malloc(reinterpret_cast<void**>(&d_query_x), query_bytes)
            .is_ok());
    ASSERT_TRUE(
        gwn::gwn_cuda_malloc(reinterpret_cast<void**>(&d_query_y), query_bytes)
            .is_ok());
    ASSERT_TRUE(
        gwn::gwn_cuda_malloc(reinterpret_cast<void**>(&d_query_z), query_bytes)
            .is_ok());
    ASSERT_TRUE(
        gwn::gwn_cuda_malloc(reinterpret_cast<void**>(&d_output), query_bytes)
            .is_ok());

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_query_x, query_soa[0].data(),
                                      query_bytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_query_y, query_soa[1].data(),
                                      query_bytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_query_z, query_soa[2].data(),
                                      query_bytes, cudaMemcpyHostToDevice));

    const gwn::gwn_status query_status =
        gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
            geometry.accessor(), bvh.accessor(),
            cuda::std::span<const Real>(d_query_x, query_count),
            cuda::std::span<const Real>(d_query_y, query_count),
            cuda::std::span<const Real>(d_query_z, query_count),
            cuda::std::span<Real>(d_output, query_count));
    ASSERT_TRUE(query_status.is_ok()) << status_to_debug_string(query_status);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> gpu_output(query_count, Real(0));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(gpu_output.data(), d_output, query_bytes,
                                      cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < query_count; ++i) {
      EXPECT_NEAR(gpu_output[i], reference_output[i], Real(5e-4))
          << "query index: " << i;
    }
  }
}

}  // namespace
