#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "reference_cpu.cuh"
#include "test_utils.cuh"

namespace {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using HostMesh = gwn::tests::HostMesh;
constexpr int k_stack_capacity = gwn::tests::k_test_stack_capacity;
constexpr gwn::gwn_query_batch_config k_query_batch_config{
    .block_size = gwn::k_gwn_default_query_batch_block_size,
    .stack_capacity = k_stack_capacity,
};

template <int Width, gwn::gwn_real_type TestReal, gwn::gwn_index_type TestIndex>
void expect_internal_nodes_are_complete(
    gwn::gwn_bvh_object<Width, TestReal, TestIndex> const &bvh
) {
    auto const accessor = bvh.accessor();
    if (accessor.has_leaf_root()) {
        EXPECT_TRUE(accessor.nodes.empty());
        EXPECT_EQ(accessor.internal_stack_bound, 0u);
        EXPECT_EQ(accessor.packed_stack_bound, 0u);
        return;
    }

    ASSERT_TRUE(accessor.has_internal_root());
    std::vector<gwn::gwn_bvh_node<Width, TestReal>> nodes(accessor.nodes.size());
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            nodes.data(), accessor.nodes.data(), accessor.nodes.size_bytes(), cudaMemcpyDeviceToHost
        )
    );

    struct stack_bounds {
        std::uint64_t internal = 0;
        std::uint64_t packed = 0;
    };

    std::vector<std::uint8_t> visited(nodes.size(), std::uint8_t(0));
    std::size_t reachable_count = 0;
    auto compute_stack_bounds = [&](auto const &self,
                                    std::size_t const node_index) -> stack_bounds {
        if (node_index >= nodes.size()) {
            ADD_FAILURE() << "Internal child offset is out of bounds.";
            return {};
        }
        if (visited[node_index]) {
            ADD_FAILURE() << "Internal child is reachable more than once.";
            return {};
        }
        visited[node_index] = std::uint8_t(1);
        ++reachable_count;

        std::uint64_t valid_child_count = 0;
        std::uint64_t internal_child_count = 0;
        stack_bounds max_child{};
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const &child = nodes[node_index].child(child_slot);
            if (!child.is_valid())
                continue;
            ++valid_child_count;
            if (child.is_internal()) {
                ++internal_child_count;
                auto const child_bounds = self(self, static_cast<std::size_t>(child.offset()));
                max_child.internal = std::max(max_child.internal, child_bounds.internal);
                max_child.packed = std::max(max_child.packed, child_bounds.packed);
                continue;
            }

            if (!child.is_leaf()) {
                ADD_FAILURE() << "Valid child is neither internal nor leaf.";
                continue;
            }
            if (child.offset() > accessor.primitive_indices.size() ||
                child.primitive_count() > accessor.primitive_indices.size() - child.offset()) {
                ADD_FAILURE() << "Leaf primitive range is out of bounds.";
            }
        }
        if (valid_child_count == 0u) {
            ADD_FAILURE() << "Internal node has no valid children.";
            return {};
        }

        stack_bounds result{};
        if (internal_child_count != 0) {
            result.internal =
                std::max(internal_child_count, internal_child_count - 1u + max_child.internal);
        }
        result.packed = valid_child_count - 1u + max_child.packed;
        return result;
    };

    auto const actual = compute_stack_bounds(
        compute_stack_bounds, static_cast<std::size_t>(accessor.root.offset())
    );
    EXPECT_EQ(reachable_count, nodes.size());
    EXPECT_EQ(accessor.internal_stack_bound, actual.internal);
    EXPECT_EQ(accessor.packed_stack_bound, actual.packed);
}

[[nodiscard]] std::array<std::vector<Real>, 3> make_builder_queries(HostMesh const &mesh) {
    std::array<Real, 3> lower{
        std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max(),
        std::numeric_limits<Real>::max()
    };
    std::array<Real, 3> upper{
        std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest(),
        std::numeric_limits<Real>::lowest()
    };
    for (std::size_t vertex_id = 0; vertex_id < mesh.vertex_x.size(); ++vertex_id) {
        std::array<Real, 3> const vertex{
            mesh.vertex_x[vertex_id], mesh.vertex_y[vertex_id], mesh.vertex_z[vertex_id]
        };
        for (int axis = 0; axis < 3; ++axis) {
            lower[axis] = std::min(lower[axis], vertex[axis]);
            upper[axis] = std::max(upper[axis], vertex[axis]);
        }
    }

    std::array<Real, 3> center{};
    std::array<Real, 3> extent{};
    for (int axis = 0; axis < 3; ++axis) {
        center[axis] = (lower[axis] + upper[axis]) * Real(0.5);
        extent[axis] = std::max(upper[axis] - lower[axis], Real(1e-2));
    }
    std::array<std::array<Real, 3>, 9> const points{{
        center,
        {center[0] + Real(1.3) * extent[0], center[1], center[2]},
        {center[0] - Real(1.3) * extent[0], center[1], center[2]},
        {center[0], center[1] + Real(1.3) * extent[1], center[2]},
        {center[0], center[1] - Real(1.3) * extent[1], center[2]},
        {center[0], center[1], center[2] + Real(1.3) * extent[2]},
        {center[0], center[1], center[2] - Real(1.3) * extent[2]},
        {center[0] + Real(0.31) * extent[0], center[1] - Real(0.27) * extent[1],
         center[2] + Real(0.23) * extent[2]},
        {center[0] - Real(0.19) * extent[0], center[1] + Real(0.37) * extent[1],
         center[2] - Real(0.29) * extent[2]},
    }};

    std::array<std::vector<Real>, 3> query{};
    for (int axis = 0; axis < 3; ++axis) {
        query[axis].reserve(points.size());
        for (auto const &point : points)
            query[axis].push_back(point[axis]);
    }
    return query;
}

template <int Width>
void expect_builder_queries_match_reference(
    gwn::gwn_bvh_build_method const method, HostMesh const &mesh,
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    std::array<std::vector<Real>, 3> const &query, std::uint32_t const hploc_search_radius = 8
) {
    std::size_t const count = query[0].size();
    gwn::gwn_bvh_object<Width, Real, Index> bvh{};
    ASSERT_TRUE(
        gwn::gwn_build_bvh(
            geometry, bvh,
            gwn::gwn_bvh_build_options{.method = method, .hploc_search_radius = hploc_search_radius}
        )
            .is_ok()
    );
    ASSERT_NO_FATAL_FAILURE(expect_internal_nodes_are_complete(bvh));
    gwn::gwn_bvh_moment_object<Width, 2, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<2>(bvh, moment).is_ok());

    std::array<gwn::detail::gwn_device_array<Real>, 3> device_query{};
    gwn::detail::gwn_device_array<Real> exact{};
    gwn::detail::gwn_device_array<Real> descended{};
    for (int axis = 0; axis < 3; ++axis) {
        device_query[axis].resize(count);
        device_query[axis].copy_from_host(cuda::std::span<Real const>(query[axis]));
    }
    exact.resize(count);
    descended.resize(count);
    ASSERT_TRUE(
        gwn::gwn_compute_winding_number_exact_batch(
            bvh, gwn::tests::device_input_span(device_query[0].span()),
            gwn::tests::device_input_span(device_query[1].span()),
            gwn::tests::device_input_span(device_query[2].span()),
            gwn::tests::device_span(exact.span())
        )
            .is_ok()
    );
    // A large accuracy scale forces every finite child to descend. Both GPU paths must visit the
    // same leaf-ordered records, and the independent CPU result below guards their shared input.
    ASSERT_TRUE(
        (gwn::gwn_compute_winding_number_taylor_batch<2, k_query_batch_config, Width, Real, Index>(
             bvh, moment, gwn::tests::device_input_span(device_query[0].span()),
             gwn::tests::device_input_span(device_query[1].span()),
             gwn::tests::device_input_span(device_query[2].span()),
             gwn::tests::device_span(descended.span()), Real(1e6)
        )
             .is_ok())
    );

    std::vector<Real> host_exact(count);
    std::vector<Real> host_descended(count);
    std::vector<Real> reference(count);
    exact.copy_to_host(cuda::std::span<Real>(host_exact));
    descended.copy_to_host(cuda::std::span<Real>(host_descended));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_TRUE(
        (gwn::tests::reference_winding_number_batch<Real, Index>(
             cuda::std::span<Real const>(mesh.vertex_x), cuda::std::span<Real const>(mesh.vertex_y),
             cuda::std::span<Real const>(mesh.vertex_z), cuda::std::span<Index const>(mesh.tri_i0),
             cuda::std::span<Index const>(mesh.tri_i1), cuda::std::span<Index const>(mesh.tri_i2),
             cuda::std::span<Real const>(query[0]), cuda::std::span<Real const>(query[1]),
             cuda::std::span<Real const>(query[2]), cuda::std::span<Real>(reference)
        )
             .is_ok())
    );
    auto const winding_tolerance = [](Real const lhs, Real const rhs) {
        if constexpr (std::is_same_v<Real, float>)
            return Real(2e-5) + Real(2e-5) * std::max(std::abs(lhs), std::abs(rhs));
        else
            return Real(2e-6);
    };
    for (std::size_t query_id = 0; query_id < count; ++query_id) {
        EXPECT_NEAR(
            host_exact[query_id], reference[query_id],
            winding_tolerance(host_exact[query_id], reference[query_id])
        );
        EXPECT_NEAR(
            host_descended[query_id], reference[query_id],
            winding_tolerance(host_descended[query_id], reference[query_id])
        );
        EXPECT_NEAR(
            host_descended[query_id], host_exact[query_id],
            winding_tolerance(host_descended[query_id], host_exact[query_id])
        );
    }
}

TEST(gwn_builder_workflow, every_builder_reaches_the_complete_triangle_sequence) {
    SMALLGWN_SKIP_IF_NO_CUDA();

    std::size_t tested_model_count = 0;
    for (std::filesystem::path const &mesh_path : gwn::tests::collect_mesh_paths()) {
        SCOPED_TRACE(mesh_path.string());
        std::optional<HostMesh> const mesh = gwn::tests::load_obj_mesh(mesh_path);
        if (!mesh.has_value()) {
            std::cerr << "smallgwn input skip: failed to parse " << mesh_path << '\n';
            continue;
        }
        ++tested_model_count;
        gwn::gwn_geometry_object<Real, Index> geometry{};
        gwn::gwn_status const upload_status = gwn::gwn_upload_geometry(
            geometry, gwn::tests::host_span(cuda::std::span<Real const>(mesh->vertex_x)),
            gwn::tests::host_span(cuda::std::span<Real const>(mesh->vertex_y)),
            gwn::tests::host_span(cuda::std::span<Real const>(mesh->vertex_z)),
            gwn::tests::host_span(cuda::std::span<Index const>(mesh->tri_i0)),
            gwn::tests::host_span(cuda::std::span<Index const>(mesh->tri_i1)),
            gwn::tests::host_span(cuda::std::span<Index const>(mesh->tri_i2))
        );
        ASSERT_TRUE(upload_status.is_ok()) << gwn::tests::status_to_debug_string(upload_status);
        auto const query = make_builder_queries(*mesh);
        expect_builder_queries_match_reference<2>(
            gwn::gwn_bvh_build_method::k_lbvh, *mesh, geometry, query
        );
        expect_builder_queries_match_reference<3>(
            gwn::gwn_bvh_build_method::k_lbvh, *mesh, geometry, query
        );
        expect_builder_queries_match_reference<4>(
            gwn::gwn_bvh_build_method::k_lbvh, *mesh, geometry, query
        );
        expect_builder_queries_match_reference<2>(
            gwn::gwn_bvh_build_method::k_hploc, *mesh, geometry, query
        );
        expect_builder_queries_match_reference<3>(
            gwn::gwn_bvh_build_method::k_hploc, *mesh, geometry, query
        );
        expect_builder_queries_match_reference<4>(
            gwn::gwn_bvh_build_method::k_hploc, *mesh, geometry, query
        );
    }
    EXPECT_GT(tested_model_count, 0u);
}

TEST(gwn_builder_workflow, hploc_builds_repeated_equal_bounds) {
    SMALLGWN_SKIP_IF_NO_CUDA();

    std::array<std::vector<Real>, 3> const query{{
        {Real(0.2), Real(0.7), Real(-0.3)},
        {Real(0.2), Real(0.1), Real(0.3)},
        {Real(1), Real(-2), Real(0.5)},
    }};

    // Counts straddle the half-warp list threshold, warp width, and multi-block construction.
    // Equal bounds make every nearest-neighbor decision depend on the packed tie convention.
    for (std::size_t const triangle_count :
         {15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 129u, 513u}) {
        SCOPED_TRACE(triangle_count);
        HostMesh mesh{};
        mesh.vertex_x = {Real(0), Real(1), Real(0)};
        mesh.vertex_y = {Real(0), Real(0), Real(1)};
        mesh.vertex_z = {Real(0), Real(0), Real(0)};
        mesh.tri_i0.assign(triangle_count, Index(0));
        mesh.tri_i1.assign(triangle_count, Index(1));
        mesh.tri_i2.assign(triangle_count, Index(2));

        gwn::gwn_geometry_object<Real, Index> geometry{};
        ASSERT_TRUE(
            gwn::gwn_upload_geometry(
                geometry, gwn::tests::host_span(cuda::std::span<Real const>(mesh.vertex_x)),
                gwn::tests::host_span(cuda::std::span<Real const>(mesh.vertex_y)),
                gwn::tests::host_span(cuda::std::span<Real const>(mesh.vertex_z)),
                gwn::tests::host_span(cuda::std::span<Index const>(mesh.tri_i0)),
                gwn::tests::host_span(cuda::std::span<Index const>(mesh.tri_i1)),
                gwn::tests::host_span(cuda::std::span<Index const>(mesh.tri_i2))
            )
                .is_ok()
        );

        for (std::uint32_t const search_radius : {1u, 4u, 8u}) {
            SCOPED_TRACE(search_radius);
            expect_builder_queries_match_reference<4>(
                gwn::gwn_bvh_build_method::k_hploc, mesh, geometry, query, search_radius
            );
        }
    }
}

TEST(gwn_builder_workflow, hploc_double_uint64_matches_exact_reference) {
    SMALLGWN_SKIP_IF_NO_CUDA();

    using TestReal = double;
    using TestIndex = std::uint64_t;
    std::vector<TestReal> const vertex_x{TestReal(0), TestReal(1), TestReal(0)};
    std::vector<TestReal> const vertex_y{TestReal(0), TestReal(0), TestReal(1)};
    std::vector<TestReal> const vertex_z{TestReal(0), TestReal(0), TestReal(0)};
    constexpr std::size_t k_triangle_count = 33;
    std::vector<TestIndex> const tri_i0(k_triangle_count, TestIndex(0));
    std::vector<TestIndex> const tri_i1(k_triangle_count, TestIndex(1));
    std::vector<TestIndex> const tri_i2(k_triangle_count, TestIndex(2));

    gwn::gwn_geometry_object<TestReal, TestIndex> geometry{};
    ASSERT_TRUE(
        gwn::gwn_upload_geometry(
            geometry, gwn::tests::host_span(cuda::std::span<TestReal const>(vertex_x)),
            gwn::tests::host_span(cuda::std::span<TestReal const>(vertex_y)),
            gwn::tests::host_span(cuda::std::span<TestReal const>(vertex_z)),
            gwn::tests::host_span(cuda::std::span<TestIndex const>(tri_i0)),
            gwn::tests::host_span(cuda::std::span<TestIndex const>(tri_i1)),
            gwn::tests::host_span(cuda::std::span<TestIndex const>(tri_i2))
        )
            .is_ok()
    );
    gwn::gwn_bvh4_object<TestReal, TestIndex> bvh{};
    ASSERT_TRUE(
        gwn::gwn_build_bvh(
            geometry, bvh,
            gwn::gwn_bvh_build_options{
                .method = gwn::gwn_bvh_build_method::k_hploc,
                .hploc_search_radius = 3,
            }
        )
            .is_ok()
    );
    ASSERT_NO_FATAL_FAILURE(expect_internal_nodes_are_complete(bvh));
    gwn::gwn_bvh4_moment_object<2, TestReal, TestIndex> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<2>(bvh, moment).is_ok());

    std::array<std::vector<TestReal>, 3> const query{{
        {TestReal(0.2), TestReal(0.7), TestReal(-0.3)},
        {TestReal(0.2), TestReal(0.1), TestReal(0.3)},
        {TestReal(1), TestReal(-2), TestReal(0.5)},
    }};
    std::array<gwn::detail::gwn_device_array<TestReal>, 3> device_query{};
    gwn::detail::gwn_device_array<TestReal> exact{};
    gwn::detail::gwn_device_array<TestReal> descended{};
    for (int axis = 0; axis < 3; ++axis) {
        device_query[axis].resize(query[axis].size());
        device_query[axis].copy_from_host(cuda::std::span<TestReal const>(query[axis]));
    }
    exact.resize(query[0].size());
    descended.resize(query[0].size());
    ASSERT_TRUE(
        gwn::gwn_compute_winding_number_exact_batch(
            bvh, gwn::tests::device_input_span(device_query[0].span()),
            gwn::tests::device_input_span(device_query[1].span()),
            gwn::tests::device_input_span(device_query[2].span()),
            gwn::tests::device_span(exact.span())
        )
            .is_ok()
    );
    ASSERT_TRUE((gwn::gwn_compute_winding_number_taylor_batch<2>(
                     bvh, moment, gwn::tests::device_input_span(device_query[0].span()),
                     gwn::tests::device_input_span(device_query[1].span()),
                     gwn::tests::device_input_span(device_query[2].span()),
                     gwn::tests::device_span(descended.span()), TestReal(1e6)
    )
                     .is_ok()));

    std::vector<TestReal> actual(query[0].size());
    std::vector<TestReal> actual_descended(query[0].size());
    std::vector<TestReal> reference(query[0].size());
    exact.copy_to_host(cuda::std::span<TestReal>(actual));
    descended.copy_to_host(cuda::std::span<TestReal>(actual_descended));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_TRUE(
        (gwn::tests::reference_winding_number_batch<TestReal, TestIndex>(
             cuda::std::span<TestReal const>(vertex_x), cuda::std::span<TestReal const>(vertex_y),
             cuda::std::span<TestReal const>(vertex_z), cuda::std::span<TestIndex const>(tri_i0),
             cuda::std::span<TestIndex const>(tri_i1), cuda::std::span<TestIndex const>(tri_i2),
             cuda::std::span<TestReal const>(query[0]), cuda::std::span<TestReal const>(query[1]),
             cuda::std::span<TestReal const>(query[2]), cuda::std::span<TestReal>(reference)
        )
             .is_ok())
    );
    for (std::size_t query_id = 0; query_id < query[0].size(); ++query_id) {
        EXPECT_NEAR(actual[query_id], reference[query_id], TestReal(1e-12));
        EXPECT_NEAR(actual_descended[query_id], reference[query_id], TestReal(1e-12));
    }
}

} // namespace
