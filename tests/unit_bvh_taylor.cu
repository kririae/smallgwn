#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_utils.hpp"

// ---------------------------------------------------------------------------
// BVH Taylor data build unit tests — verifies Taylor moment data existence
// and basic validity for Order=0 and Order=1.
// ---------------------------------------------------------------------------

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

namespace {

enum class taylor_topology_builder {
    k_lbvh,
    k_hploc,
};

// Helper to upload octahedron geometry.
std::optional<gwn::gwn_geometry_object<Real, Index>> upload_octahedron() {
    std::vector<Real> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::vector<Index> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::vector<Index> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::vector<Index> i2{4, 4, 4, 4, 5, 5, 5, 5};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = geometry.upload(
        cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    if (!status.is_ok())
        return std::nullopt;
    return geometry;
}

template <int Order>
gwn::gwn_status build_taylor_tree(
    taylor_topology_builder const builder, gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_object<Real, Index> &bvh, gwn::gwn_bvh4_aabb_object<Real, Index> &aabb,
    gwn::gwn_bvh4_moment_object<Real, Index> &data
) {
    if (builder == taylor_topology_builder::k_hploc) {
        return gwn::gwn_bvh_facade_build_topology_aabb_moment_hploc<Order, 4, Real, Index>(
            geometry, bvh, aabb, data
        );
    }
    return gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
        geometry, bvh, aabb, data
    );
}

} // namespace

// ---------------------------------------------------------------------------
// Order 0 — build succeeds and has_taylor_order<0>.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, taylor_order0_build_marks_accessor) {
    auto maybe_geo = upload_octahedron();
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Real, Index> data;
    gwn::gwn_status const status =
        gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<0, 4, Real, Index>(
            geometry, bvh, aabb, data
        );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    EXPECT_TRUE(data.accessor().template has_taylor_order<0>());
    EXPECT_TRUE(data.has_data());
}

// ---------------------------------------------------------------------------
// Order 1 — build succeeds and has_taylor_order<1>.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, taylor_order1_build_marks_accessor) {
    auto maybe_geo = upload_octahedron();
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Real, Index> data;
    gwn::gwn_status const status =
        gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
            geometry, bvh, aabb, data
        );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    EXPECT_TRUE(data.accessor().template has_taylor_order<1>());
}

TEST_F(CudaFixture, taylor_order0_build_marks_accessor_hploc) {
    auto maybe_geo = upload_octahedron();
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Real, Index> data;
    gwn::gwn_status const status =
        build_taylor_tree<0>(taylor_topology_builder::k_hploc, geometry, bvh, aabb, data);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    EXPECT_TRUE(data.accessor().template has_taylor_order<0>());
    EXPECT_TRUE(data.has_data());
}

TEST_F(CudaFixture, taylor_order1_build_marks_accessor_hploc) {
    auto maybe_geo = upload_octahedron();
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Real, Index> data;
    gwn::gwn_status const status =
        build_taylor_tree<1>(taylor_topology_builder::k_hploc, geometry, bvh, aabb, data);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    EXPECT_TRUE(data.accessor().template has_taylor_order<1>());
}

// ---------------------------------------------------------------------------
// Rebuild path keeps requested order data valid.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, taylor_rebuild_order1_marks_accessor) {
    auto maybe_geo = upload_octahedron();
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Real, Index> data;
    gwn::gwn_status const status =
        gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
            geometry, bvh, aabb, data
        );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    EXPECT_TRUE(data.accessor().template has_taylor_order<1>());
}

// ---------------------------------------------------------------------------
// Default data accessor is empty.
// ---------------------------------------------------------------------------

TEST(smallgwn_unit_bvh_taylor, default_data_is_empty) {
    gwn::gwn_bvh_moment_tree_accessor<4, Real, Index> data_acc{};
    EXPECT_TRUE(data_acc.empty());
    EXPECT_FALSE(data_acc.has_taylor_order<0>());
    EXPECT_FALSE(data_acc.has_taylor_order<1>());
}

// ---------------------------------------------------------------------------
// Taylor data nodes have finite values (not NaN).
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, taylor_order0_nodes_are_finite) {
    auto maybe_geo = upload_octahedron();
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Real, Index> data;
    gwn::gwn_status const status =
        gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<0, 4, Real, Index>(
            geometry, bvh, aabb, data
        );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    auto const &data_acc = data.accessor();
    ASSERT_TRUE(data_acc.template has_taylor_order<0>());

    std::size_t const node_count = data_acc.taylor_order0_nodes.size();
    ASSERT_GT(node_count, 0u);

    using TaylorNode0 = gwn::gwn_bvh_taylor_node_soa<4, 0, Real>;
    std::vector<TaylorNode0> host_nodes(node_count);
    ASSERT_EQ(
        cudaSuccess, cudaMemcpy(
                         host_nodes.data(), data_acc.taylor_order0_nodes.data(),
                         node_count * sizeof(TaylorNode0), cudaMemcpyDeviceToHost
                     )
    );

    for (std::size_t n = 0; n < node_count; ++n) {
        auto const &node = host_nodes[n];
        for (int slot = 0; slot < 4; ++slot) {
            EXPECT_TRUE(std::isfinite(node.child_average_x[slot]))
                << "NaN at node=" << n << " slot=" << slot;
            EXPECT_TRUE(std::isfinite(node.child_average_y[slot]));
            EXPECT_TRUE(std::isfinite(node.child_average_z[slot]));
            EXPECT_TRUE(std::isfinite(node.child_n_x[slot]));
            EXPECT_TRUE(std::isfinite(node.child_n_y[slot]));
            EXPECT_TRUE(std::isfinite(node.child_n_z[slot]));
            EXPECT_TRUE(std::isfinite(node.child_max_p_dist2[slot]));
            // max_p_dist2 should be non-negative.
            EXPECT_GE(node.child_max_p_dist2[slot], Real(0));
        }
    }
}

// ---------------------------------------------------------------------------
// Taylor data clear resets state.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, taylor_data_clear_resets) {
    auto maybe_geo = upload_octahedron();
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Real, Index> data;
    ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<0, 4, Real, Index>(
                     geometry, bvh, aabb, data
    )
                     .is_ok()));
    ASSERT_TRUE(data.has_data());

    data.clear();
    EXPECT_FALSE(data.has_data());
    EXPECT_TRUE(data.accessor().empty());
}

// ---------------------------------------------------------------------------
// Single triangle — Taylor build with trivial geometry.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, taylor_single_triangle) {
    std::vector<Real> vx{1.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 1.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 1.0f};
    std::vector<Index> i0{0}, i1{1}, i2{2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const upload_status = geometry.upload(
        cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
    ASSERT_TRUE(upload_status.is_ok());

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Real, Index> data;
    gwn::gwn_status const status =
        gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<0, 4, Real, Index>(
            geometry, bvh, aabb, data
        );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_TRUE(bvh.has_data());
    EXPECT_FALSE(data.accessor().template has_taylor_order<0>());
    EXPECT_FALSE(data.has_data());
}
