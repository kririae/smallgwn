#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
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

template <int Width>
gwn::gwn_status build_topology_for_builder(
    topology_builder const builder, gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh_topology_object<Width, Real, Index> &topology
) {
    if (builder == topology_builder::k_hploc)
        return gwn::gwn_bvh_topology_build_hploc<Width, Real, Index>(geometry, topology);
    return gwn::gwn_bvh_topology_build_lbvh<Width, Real, Index>(geometry, topology);
}

template <int Order>
gwn::gwn_status build_facade_for_builder(
    topology_builder const builder, gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_object<Real, Index> &topology, gwn::gwn_bvh4_aabb_object<Real, Index> &aabb,
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

// Query generation helpers.

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
        {{center_x + Real(0.31) * extent_x, center_y - Real(0.27) * extent_y,
          center_z + Real(0.23) * extent_z}},
        {{center_x - Real(0.19) * extent_x, center_y + Real(0.37) * extent_y,
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

std::array<std::vector<Real>, 3> make_integration_query_soa(HostMesh const &mesh) {
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

    std::array<std::vector<Real>, 3> soa{};
    auto append_query = [&](Real const x, Real const y, Real const z) {
        soa[0].push_back(x);
        soa[1].push_back(y);
        soa[2].push_back(z);
    };

    constexpr std::array<Real, 3> k_lattice_scale = {Real(-0.61), Real(0.13), Real(0.77)};
    for (Real const sx : k_lattice_scale) {
        for (Real const sy : k_lattice_scale) {
            for (Real const sz : k_lattice_scale) {
                append_query(
                    center_x + sx * extent_x, center_y + sy * extent_y, center_z + sz * extent_z
                );
            }
        }
    }

    constexpr std::array<Real, 2> k_far_scale = {Real(-2.5), Real(2.5)};
    for (Real const scale : k_far_scale) {
        append_query(center_x + scale * extent_x, center_y, center_z);
        append_query(center_x, center_y + scale * extent_y, center_z);
        append_query(center_x, center_y, center_z + scale * extent_z);
    }

    constexpr std::array<Real, 2> k_corner_scale = {Real(-1.8), Real(1.8)};
    for (Real const sx : k_corner_scale) {
        for (Real const sy : k_corner_scale) {
            for (Real const sz : k_corner_scale) {
                append_query(
                    center_x + sx * extent_x, center_y + sy * extent_y, center_z + sz * extent_z
                );
            }
        }
    }

    return soa;
}

// BVH structure verification (copied from original, kept local since
// integration tests are the primary user).

template <int Width>
void assert_bvh_structure_wide(
    gwn::gwn_bvh_topology_accessor<Width, Real, Index> const &accessor,
    std::size_t const primitive_count
) {
    static_assert(Width >= 2);
    ASSERT_EQ(accessor.primitive_indices.size(), primitive_count);

    std::vector<Index> primitive_indices(primitive_count, Index(0));
    if (primitive_count > 0) {
        ASSERT_EQ(
            cudaSuccess, cudaMemcpy(
                             primitive_indices.data(), accessor.primitive_indices.data(),
                             primitive_count * sizeof(Index), cudaMemcpyDeviceToHost
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
        auto const begin = static_cast<std::size_t>(accessor.root_index);
        auto const count = static_cast<std::size_t>(accessor.root_count);
        ASSERT_LE(begin, primitive_count);
        ASSERT_LE(count, primitive_count - begin);
        return;
    }

    ASSERT_EQ(accessor.root_kind, gwn::gwn_bvh_child_kind::k_internal);
    ASSERT_FALSE(accessor.nodes.empty());
    ASSERT_GE(accessor.root_index, Index(0));
    ASSERT_LT(static_cast<std::size_t>(accessor.root_index), accessor.nodes.size());

    std::vector<gwn::gwn_bvh_topology_node_soa<Width, Index>> nodes(accessor.nodes.size());
    ASSERT_EQ(
        cudaSuccess, cudaMemcpy(
                         nodes.data(), accessor.nodes.data(), nodes.size() * sizeof(nodes[0]),
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
        for (int slot = 0; slot < Width; ++slot) {
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

            auto const begin = static_cast<std::size_t>(node.child_index[slot]);
            auto const count = static_cast<std::size_t>(node.child_count[slot]);
            ASSERT_LE(begin, primitive_count);
            ASSERT_LE(count, primitive_count - begin);
            for (std::size_t i = begin; i < begin + count; ++i)
                ++sorted_slot_seen[i];
        }
    }
    for (int const seen : sorted_slot_seen)
        EXPECT_EQ(seen, 1);
}

void assert_bvh_structure(
    gwn::gwn_bvh4_topology_accessor<Real, Index> const &accessor, std::size_t const primitive_count
) {
    gwn::gwn_bvh_topology_accessor<4, Real, Index> topology4_accessor{};
    topology4_accessor.nodes = accessor.nodes;
    topology4_accessor.primitive_indices = accessor.primitive_indices;
    topology4_accessor.root_kind = accessor.root_kind;
    topology4_accessor.root_index = accessor.root_index;
    topology4_accessor.root_count = accessor.root_count;
    assert_bvh_structure_wide<4>(topology4_accessor, primitive_count);
}

// Integration Tests.

#if 0
TEST(smallgwn_integration_model, bvh_exact_batch_matches_cpu_on_common_models) {
    std::optional<std::filesystem::path> const model_dir = gwn::tests::find_model_data_dir();
    if (!model_dir.has_value()) {
        GTEST_SKIP() << "Model directory not found. Set SMALLGWN_MODEL_DATA_DIR or "
                        "clone models to /tmp/common-3d-test-models/data.";
    }

    std::array<std::string_view, 5> const model_names = {
        "suzanne.obj", "teapot.obj", "cow.obj", "stanford-bunny.obj", "armadillo.obj"};

    for (std::string_view const model_name : model_names) {
        std::filesystem::path const model_path = *model_dir / model_name;
        if (!std::filesystem::exists(model_path))
            continue;

        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        ASSERT_TRUE(maybe_mesh.has_value());
        HostMesh const& mesh = *maybe_mesh;

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
        ASSERT_TRUE(upload_status.is_ok())
            << gwn::tests::status_to_debug_string(upload_status);

        gwn::gwn_bvh4_topology_object<Real, Index> bvh;
        gwn::gwn_status const build_status =
            gwn::gwn_bvh_topology_build_lbvh<4, Real, Index>(geometry, bvh);
        ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);
        ASSERT_TRUE(bvh.has_data());
        assert_bvh_structure(bvh.accessor(), mesh.tri_i0.size());

        std::size_t const query_count = query_soa[0].size();
        gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
        ASSERT_TRUE(d_qx.resize(query_count).is_ok());
        ASSERT_TRUE(d_qy.resize(query_count).is_ok());
        ASSERT_TRUE(d_qz.resize(query_count).is_ok());
        ASSERT_TRUE(d_out.resize(query_count).is_ok());
        ASSERT_TRUE(d_qx.copy_from_host(
            cuda::std::span<Real const>(query_soa[0].data(), query_count)).is_ok());
        ASSERT_TRUE(d_qy.copy_from_host(
            cuda::std::span<Real const>(query_soa[1].data(), query_count)).is_ok());
        ASSERT_TRUE(d_qz.copy_from_host(
            cuda::std::span<Real const>(query_soa[2].data(), query_count)).is_ok());

        gwn::gwn_status const query_status =
            gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
                geometry.accessor(), bvh.accessor(),
                d_qx.span(), d_qy.span(), d_qz.span(), d_out.span());
        ASSERT_TRUE(query_status.is_ok()) << gwn::tests::status_to_debug_string(query_status);
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        std::vector<Real> gpu_output(query_count, Real(0));
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(gpu_output.data(), gpu_output.size())).is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        for (std::size_t i = 0; i < query_count; ++i)
            EXPECT_NEAR(gpu_output[i], reference_output[i], Real(5e-4)) << "query index: " << i;
    }
}

TEST(smallgwn_integration_model, bvh_binary_to_wide_matches_width4_exact_queries) {
    std::optional<std::filesystem::path> const model_dir = gwn::tests::find_model_data_dir();
    if (!model_dir.has_value()) {
        GTEST_SKIP() << "Model directory not found. Set SMALLGWN_MODEL_DATA_DIR or "
                        "clone models to /tmp/common-3d-test-models/data.";
    }

    std::array<std::string_view, 4> const model_names = {
        "suzanne.obj", "teapot.obj", "cow.obj", "stanford-bunny.obj"};

    std::size_t tested_model_count = 0;
    for (std::string_view const model_name : model_names) {
        std::filesystem::path const model_path = *model_dir / model_name;
        if (!std::filesystem::exists(model_path))
            continue;

        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        ASSERT_TRUE(maybe_mesh.has_value());
        HostMesh const& mesh = *maybe_mesh;
        auto const query_soa = make_query_soa(mesh);
        std::size_t const query_count = query_soa[0].size();

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

        gwn::gwn_bvh_topology_object<2, Real, Index> bvh2;
        gwn::gwn_bvh4_topology_object<Real, Index> bvh4;
        gwn::gwn_bvh_topology_object<8, Real, Index> bvh8;
        ASSERT_TRUE(
            (gwn::gwn_bvh_topology_build_lbvh<2, Real, Index>(geometry, bvh2)
                 .is_ok()));
        ASSERT_TRUE(
            (gwn::gwn_bvh_topology_build_lbvh<4, Real, Index>(geometry, bvh4)
                 .is_ok()));
        ASSERT_TRUE(
            (gwn::gwn_bvh_topology_build_lbvh<8, Real, Index>(geometry, bvh8)
                 .is_ok()));

        assert_bvh_structure_wide<2>(bvh2.accessor(), mesh.tri_i0.size());
        assert_bvh_structure(bvh4.accessor(), mesh.tri_i0.size());
        assert_bvh_structure_wide<8>(bvh8.accessor(), mesh.tri_i0.size());

        gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
        ASSERT_TRUE(d_qx.resize(query_count).is_ok());
        ASSERT_TRUE(d_qy.resize(query_count).is_ok());
        ASSERT_TRUE(d_qz.resize(query_count).is_ok());
        ASSERT_TRUE(d_out.resize(query_count).is_ok());
        ASSERT_TRUE(d_qx.copy_from_host(
            cuda::std::span<Real const>(query_soa[0].data(), query_count)).is_ok());
        ASSERT_TRUE(d_qy.copy_from_host(
            cuda::std::span<Real const>(query_soa[1].data(), query_count)).is_ok());
        ASSERT_TRUE(d_qz.copy_from_host(
            cuda::std::span<Real const>(query_soa[2].data(), query_count)).is_ok());

        // Width=4.
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
                        geometry.accessor(), bvh4.accessor(),
                        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span())
                        .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> output4(query_count, Real(0));
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(output4.data(), output4.size())).is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        // Width=2.
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_exact<2, Real, Index>(
                        geometry.accessor(), bvh2.accessor(),
                        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span())
                        .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> output2(query_count, Real(0));
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(output2.data(), output2.size())).is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        // Width=8.
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_exact<8, Real, Index>(
                        geometry.accessor(), bvh8.accessor(),
                        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span())
                        .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> output8(query_count, Real(0));
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(output8.data(), output8.size())).is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ++tested_model_count;
        for (std::size_t qi = 0; qi < query_count; ++qi) {
            EXPECT_NEAR(output2[qi], output4[qi], Real(5e-4)) << "query index: " << qi;
            EXPECT_NEAR(output8[qi], output4[qi], Real(5e-4)) << "query index: " << qi;
        }
    }

    ASSERT_GT(tested_model_count, 0u) << "No valid OBJ models were exercised.";
}

TEST(smallgwn_integration_model, integration_exact_and_taylor_consistency_on_common_models) {
    std::optional<std::filesystem::path> const model_dir = gwn::tests::find_model_data_dir();
    if (!model_dir.has_value()) {
        GTEST_SKIP() << "Model directory not found. Set SMALLGWN_MODEL_DATA_DIR or "
                        "clone models to /tmp/common-3d-test-models/data.";
    }

    std::vector<std::filesystem::path> const model_paths =
        gwn::tests::collect_obj_model_paths(*model_dir);
    if (model_paths.empty())
        GTEST_SKIP() << "No .obj models found in " << model_dir->string();

    constexpr Real k_order0_epsilon = Real(6.5e-2);
    constexpr Real k_order1_epsilon = Real(4.5e-2);
    constexpr Real k_accuracy_scale = Real(2);

    std::size_t tested_model_count = 0;
    for (std::filesystem::path const& model_path : model_paths) {
        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value())
            continue;
        HostMesh const& mesh = *maybe_mesh;
        auto const query_soa = make_integration_query_soa(mesh);
        std::size_t const query_count = query_soa[0].size();
        if (query_count == 0)
            continue;

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

        gwn::gwn_bvh4_topology_object<Real, Index> bvh;
        gwn::gwn_bvh4_aabb_object<Real, Index> bvh_aabb;
        ASSERT_TRUE(
            (gwn::gwn_bvh_topology_build_lbvh<4, Real, Index>(geometry, bvh)
                 .is_ok()));
        ASSERT_TRUE(bvh.has_data());
        assert_bvh_structure(bvh.accessor(), mesh.tri_i0.size());

        gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
        ASSERT_TRUE(d_qx.resize(query_count).is_ok());
        ASSERT_TRUE(d_qy.resize(query_count).is_ok());
        ASSERT_TRUE(d_qz.resize(query_count).is_ok());
        ASSERT_TRUE(d_out.resize(query_count).is_ok());
        ASSERT_TRUE(d_qx.copy_from_host(
            cuda::std::span<Real const>(query_soa[0].data(), query_count)).is_ok());
        ASSERT_TRUE(d_qy.copy_from_host(
            cuda::std::span<Real const>(query_soa[1].data(), query_count)).is_ok());
        ASSERT_TRUE(d_qz.copy_from_host(
            cuda::std::span<Real const>(query_soa[2].data(), query_count)).is_ok());

        // Exact.
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
                        geometry.accessor(), bvh.accessor(),
                        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span())
                        .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> exact_output(query_count, Real(0));
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(exact_output.data(), exact_output.size())).is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        // Taylor order 0.
        gwn::gwn_bvh4_moment_object<0, Real, Index> bvh_data_o0;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<0, 4, Real, Index>(geometry, bvh, bvh_aabb, bvh_data_o0)
                        .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
                        geometry.accessor(), bvh.accessor(), bvh_data_o0.accessor(),
                        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale)
                        .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> order0_output(query_count, Real(0));
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(order0_output.data(), order0_output.size())).is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        // Taylor order 1.
        gwn::gwn_bvh4_moment_object<1, Real, Index> bvh_data_o1;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(geometry, bvh, bvh_aabb, bvh_data_o1)
                        .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                        geometry.accessor(), bvh.accessor(), bvh_data_o1.accessor(),
                        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale)
                        .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> order1_output(query_count, Real(0));
        ASSERT_TRUE(d_out.copy_to_host(
            cuda::std::span<Real>(order1_output.data(), order1_output.size())).is_ok());
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ++tested_model_count;
        for (std::size_t qi = 0; qi < query_count; ++qi) {
            EXPECT_NEAR(order0_output[qi], exact_output[qi], k_order0_epsilon)
                << "query index: " << qi;
            EXPECT_NEAR(order1_output[qi], exact_output[qi], k_order1_epsilon)
                << "query index: " << qi;
        }
    }

    ASSERT_GT(tested_model_count, 0u) << "No valid OBJ models were exercised.";
}
#endif

template <int Order>
void run_integration_taylor_rebuild_consistency_on_common_models(Real const k_consistency_epsilon) {
    static_assert(Order == 1 || Order == 2);

    std::optional<std::filesystem::path> const model_dir = gwn::tests::find_model_data_dir();
    ASSERT_TRUE(model_dir.has_value())
        << "Model directory not found. Set SMALLGWN_MODEL_DATA_DIR or "
           "clone models to /tmp/common-3d-test-models/data.";

    std::vector<std::filesystem::path> const model_paths =
        gwn::tests::collect_obj_model_paths(*model_dir);
    ASSERT_FALSE(model_paths.empty()) << "No .obj models found in " << model_dir->string();

    constexpr Real k_accuracy_scale = Real(2);

    std::size_t tested_model_count = 0;
    for (std::filesystem::path const &model_path : model_paths) {
        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value())
            continue;
        HostMesh const &mesh = *maybe_mesh;
        auto const query_soa = make_integration_query_soa(mesh);
        std::size_t const query_count = query_soa[0].size();
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

        gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
        ASSERT_TRUE(d_qx.resize(query_count).is_ok());
        ASSERT_TRUE(d_qy.resize(query_count).is_ok());
        ASSERT_TRUE(d_qz.resize(query_count).is_ok());
        ASSERT_TRUE(d_out.resize(query_count).is_ok());
        ASSERT_TRUE(
            d_qx.copy_from_host(cuda::std::span<Real const>(query_soa[0].data(), query_count))
                .is_ok()
        );
        ASSERT_TRUE(
            d_qy.copy_from_host(cuda::std::span<Real const>(query_soa[1].data(), query_count))
                .is_ok()
        );
        ASSERT_TRUE(
            d_qz.copy_from_host(cuda::std::span<Real const>(query_soa[2].data(), query_count))
                .is_ok()
        );

        // Build/query A.
        gwn::gwn_bvh4_topology_object<Real, Index> bvh_a;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb_a;
        gwn::gwn_bvh4_moment_object<Order, Real, Index> data_a;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
                         geometry, bvh_a, aabb_a, data_a
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, Real, Index>(
                         geometry.accessor(), bvh_a.accessor(), data_a.accessor(), d_qx.span(),
                         d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> output_a(query_count, Real(0));
        ASSERT_TRUE(
            d_out.copy_to_host(cuda::std::span<Real>(output_a.data(), output_a.size())).is_ok()
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        // Build/query B.
        gwn::gwn_bvh4_topology_object<Real, Index> bvh_b;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb_b;
        gwn::gwn_bvh4_moment_object<Order, Real, Index> data_b;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
                         geometry, bvh_b, aabb_b, data_b
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, Real, Index>(
                         geometry.accessor(), bvh_b.accessor(), data_b.accessor(), d_qx.span(),
                         d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> output_b(query_count, Real(0));
        ASSERT_TRUE(
            d_out.copy_to_host(cuda::std::span<Real>(output_b.data(), output_b.size())).is_ok()
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ++tested_model_count;
        for (std::size_t qi = 0; qi < query_count; ++qi)
            EXPECT_NEAR(output_b[qi], output_a[qi], k_consistency_epsilon) << "query index: " << qi;
    }

    ASSERT_GT(tested_model_count, 0u) << "No valid OBJ models were exercised.";
}

TEST(smallgwn_integration_model, integration_taylor_rebuild_consistency_on_common_models_order1) {
    run_integration_taylor_rebuild_consistency_on_common_models<1>(Real(3e-4));
}

TEST(smallgwn_integration_model, integration_taylor_rebuild_consistency_on_common_models_order2) {
    run_integration_taylor_rebuild_consistency_on_common_models<2>(Real(3e-4));
}

TEST(smallgwn_integration_model, integration_taylor_matches_hdk_cpu_order0_order1_order2) {
    std::optional<std::filesystem::path> const model_dir = gwn::tests::find_model_data_dir();
    ASSERT_TRUE(model_dir.has_value())
        << "Model directory not found. Set SMALLGWN_MODEL_DATA_DIR or "
           "clone models to /tmp/common-3d-test-models/data.";

    std::vector<std::filesystem::path> const model_paths =
        gwn::tests::collect_obj_model_paths(*model_dir);
    ASSERT_FALSE(model_paths.empty()) << "No .obj models found in " << model_dir->string();

    constexpr Real k_accuracy_scale = Real(2);
    constexpr Real k_order0_epsilon = Real(5e-2);
    constexpr Real k_order1_epsilon = Real(3e-2);
    constexpr Real k_order2_epsilon = Real(2e-2);

    std::size_t tested_model_count = 0;
    for (std::filesystem::path const &model_path : model_paths) {
        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value())
            continue;
        HostMesh const &mesh = *maybe_mesh;
        auto const query_soa = make_integration_query_soa(mesh);
        std::size_t const query_count = query_soa[0].size();
        if (query_count == 0)
            continue;

        std::vector<Real> cpu_order0(query_count, Real(0));
        std::vector<Real> cpu_order1(query_count, Real(0));
        std::vector<Real> cpu_order2(query_count, Real(0));
        ASSERT_TRUE((gwn::tests::reference_winding_number_batch_hdk_taylor<Real, Index>(
                         std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
                         std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
                         std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
                         std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
                         std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
                         std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()),
                         std::span<Real const>(query_soa[0].data(), query_soa[0].size()),
                         std::span<Real const>(query_soa[1].data(), query_soa[1].size()),
                         std::span<Real const>(query_soa[2].data(), query_soa[2].size()),
                         std::span<Real>(cpu_order0.data(), cpu_order0.size()), 0, k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::tests::reference_winding_number_batch_hdk_taylor<Real, Index>(
                         std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
                         std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
                         std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
                         std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
                         std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
                         std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()),
                         std::span<Real const>(query_soa[0].data(), query_soa[0].size()),
                         std::span<Real const>(query_soa[1].data(), query_soa[1].size()),
                         std::span<Real const>(query_soa[2].data(), query_soa[2].size()),
                         std::span<Real>(cpu_order1.data(), cpu_order1.size()), 1, k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::tests::reference_winding_number_batch_hdk_taylor<Real, Index>(
                         std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
                         std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
                         std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
                         std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
                         std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
                         std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()),
                         std::span<Real const>(query_soa[0].data(), query_soa[0].size()),
                         std::span<Real const>(query_soa[1].data(), query_soa[1].size()),
                         std::span<Real const>(query_soa[2].data(), query_soa[2].size()),
                         std::span<Real>(cpu_order2.data(), cpu_order2.size()), 2, k_accuracy_scale
        )
                         .is_ok()));

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

        gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
        ASSERT_TRUE(d_qx.resize(query_count).is_ok());
        ASSERT_TRUE(d_qy.resize(query_count).is_ok());
        ASSERT_TRUE(d_qz.resize(query_count).is_ok());
        ASSERT_TRUE(d_out.resize(query_count).is_ok());
        ASSERT_TRUE(
            d_qx.copy_from_host(cuda::std::span<Real const>(query_soa[0].data(), query_count))
                .is_ok()
        );
        ASSERT_TRUE(
            d_qy.copy_from_host(cuda::std::span<Real const>(query_soa[1].data(), query_count))
                .is_ok()
        );
        ASSERT_TRUE(
            d_qz.copy_from_host(cuda::std::span<Real const>(query_soa[2].data(), query_count))
                .is_ok()
        );

        gwn::gwn_bvh4_topology_object<Real, Index> bvh;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb;

        gwn::gwn_bvh4_moment_object<0, Real, Index> data_o0;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<0, 4, Real, Index>(
                         geometry, bvh, aabb, data_o0
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
                         geometry.accessor(), bvh.accessor(), data_o0.accessor(), d_qx.span(),
                         d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> gpu_order0(query_count, Real(0));
        ASSERT_TRUE(
            d_out.copy_to_host(cuda::std::span<Real>(gpu_order0.data(), gpu_order0.size())).is_ok()
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        gwn::gwn_bvh4_moment_object<1, Real, Index> data_o1;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
                         geometry, bvh, aabb, data_o1
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                         geometry.accessor(), bvh.accessor(), data_o1.accessor(), d_qx.span(),
                         d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> gpu_order1(query_count, Real(0));
        ASSERT_TRUE(
            d_out.copy_to_host(cuda::std::span<Real>(gpu_order1.data(), gpu_order1.size())).is_ok()
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        gwn::gwn_bvh4_moment_object<2, Real, Index> data_o2;
        ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<2, 4, Real, Index>(
                         geometry, bvh, aabb, data_o2
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<2, Real, Index>(
                         geometry.accessor(), bvh.accessor(), data_o2.accessor(), d_qx.span(),
                         d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> gpu_order2(query_count, Real(0));
        ASSERT_TRUE(
            d_out.copy_to_host(cuda::std::span<Real>(gpu_order2.data(), gpu_order2.size())).is_ok()
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ++tested_model_count;
        for (std::size_t qi = 0; qi < query_count; ++qi) {
            EXPECT_NEAR(gpu_order0[qi], cpu_order0[qi], k_order0_epsilon) << "query index: " << qi;
            EXPECT_NEAR(gpu_order1[qi], cpu_order1[qi], k_order1_epsilon) << "query index: " << qi;
            EXPECT_NEAR(gpu_order2[qi], cpu_order2[qi], k_order2_epsilon) << "query index: " << qi;
        }
    }

    ASSERT_GT(tested_model_count, 0u) << "No valid OBJ models were exercised.";
}

#if 0  // Exact batch API removed, Taylor-only public surface.
TEST(smallgwn_integration_model, integration_hploc_exact_and_taylor_consistency_sampled_models) {
    std::vector<std::filesystem::path> const model_paths = gwn::tests::collect_model_paths();
    ASSERT_FALSE(model_paths.empty())
        << "No model input found. Set SMALLGWN_MODEL_DATA_DIR or SMALLGWN_MODEL_PATH.";

    std::size_t const model_limit =
        gwn::tests::get_env_positive_size_t("SMALLGWN_HPLOC_INTEGRATION_MODEL_LIMIT", 6);

    constexpr Real k_order0_epsilon = Real(8e-2);
    constexpr Real k_order1_epsilon = Real(6e-2);
    constexpr Real k_accuracy_scale = Real(2);

    std::size_t tested_model_count = 0;
    for (std::filesystem::path const &model_path : model_paths) {
        if (tested_model_count >= model_limit)
            break;
        SCOPED_TRACE(model_path.string());

        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value())
            continue;
        HostMesh const &mesh = *maybe_mesh;
        auto const query_soa = make_integration_query_soa(mesh);
        std::size_t const query_count = query_soa[0].size();
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

        gwn::gwn_bvh4_topology_object<Real, Index> bvh;
        gwn::gwn_bvh4_aabb_object<Real, Index> bvh_aabb;
        ASSERT_TRUE(
            (build_topology_for_builder<4>(topology_builder::k_hploc, geometry, bvh).is_ok())
        );
        ASSERT_TRUE(bvh.has_data());
        assert_bvh_structure(bvh.accessor(), mesh.tri_i0.size());

        gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
        ASSERT_TRUE(d_qx.resize(query_count).is_ok());
        ASSERT_TRUE(d_qy.resize(query_count).is_ok());
        ASSERT_TRUE(d_qz.resize(query_count).is_ok());
        ASSERT_TRUE(d_out.resize(query_count).is_ok());
        ASSERT_TRUE(
            d_qx.copy_from_host(cuda::std::span<Real const>(query_soa[0].data(), query_count))
                .is_ok()
        );
        ASSERT_TRUE(
            d_qy.copy_from_host(cuda::std::span<Real const>(query_soa[1].data(), query_count))
                .is_ok()
        );
        ASSERT_TRUE(
            d_qz.copy_from_host(cuda::std::span<Real const>(query_soa[2].data(), query_count))
                .is_ok()
        );

        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
                         geometry.accessor(), bvh.accessor(), d_qx.span(), d_qy.span(), d_qz.span(),
                         d_out.span()
        )
                         .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> exact_output(query_count, Real(0));
        ASSERT_TRUE(
            d_out.copy_to_host(cuda::std::span<Real>(exact_output.data(), exact_output.size()))
                .is_ok()
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        gwn::gwn_bvh4_moment_object<0, Real, Index> bvh_data_o0;
        ASSERT_TRUE((build_facade_for_builder<0>(
                         topology_builder::k_hploc, geometry, bvh, bvh_aabb, bvh_data_o0
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
                         geometry.accessor(), bvh.accessor(), bvh_data_o0.accessor(), d_qx.span(),
                         d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> order0_output(query_count, Real(0));
        ASSERT_TRUE(
            d_out.copy_to_host(cuda::std::span<Real>(order0_output.data(), order0_output.size()))
                .is_ok()
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        gwn::gwn_bvh4_moment_object<1, Real, Index> bvh_data_o1;
        ASSERT_TRUE((build_facade_for_builder<1>(
                         topology_builder::k_hploc, geometry, bvh, bvh_aabb, bvh_data_o1
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                         geometry.accessor(), bvh.accessor(), bvh_data_o1.accessor(), d_qx.span(),
                         d_qy.span(), d_qz.span(), d_out.span(), k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> order1_output(query_count, Real(0));
        ASSERT_TRUE(
            d_out.copy_to_host(cuda::std::span<Real>(order1_output.data(), order1_output.size()))
                .is_ok()
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ++tested_model_count;
        std::cout << "[gwn-integration] builder=" << to_builder_name(topology_builder::k_hploc)
                  << " model=" << model_path.filename().string() << " queries=" << query_count
                  << std::endl;
        for (std::size_t qi = 0; qi < query_count; ++qi) {
            EXPECT_NEAR(order0_output[qi], exact_output[qi], k_order0_epsilon)
                << "query index: " << qi;
            EXPECT_NEAR(order1_output[qi], exact_output[qi], k_order1_epsilon)
                << "query index: " << qi;
        }
    }

    ASSERT_GT(tested_model_count, 0u) << "No valid OBJ models were exercised.";
}
#endif // Exact batch API removed

} // namespace
