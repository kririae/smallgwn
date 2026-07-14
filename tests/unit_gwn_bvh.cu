#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.cuh"
#include "test_utils.cuh"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using GwnBvhTest = gwn::tests::CudaFixture;
using GwnBvhStreamTest = gwn::tests::CudaStreamFixture;

using BvhAccessor = gwn::gwn_bvh4_accessor<Real, Index>;
using BvhObject = gwn::gwn_bvh4_object<Real, Index>;
using MomentAccessor = gwn::gwn_bvh4_moment_accessor<1, Real, Index>;
using MomentObject = gwn::gwn_bvh4_moment_object<1, Real, Index>;

static_assert(std::is_trivially_copyable_v<BvhAccessor>);
static_assert(std::is_trivially_copyable_v<MomentAccessor>);
static_assert(std::is_same_v<BvhAccessor, gwn::gwn_bvh_accessor<4, Real, Index>>);

// Owning BVH types keep one canonical mutable accessor. The const overload exposes that same
// device view through the object's const boundary.
static_assert(std::is_same_v<decltype(std::declval<BvhObject &>().accessor()), BvhAccessor &>);
static_assert(
    std::is_same_v<decltype(std::declval<BvhObject const &>().accessor()), BvhAccessor const &>
);
static_assert(
    std::is_same_v<decltype(std::declval<MomentObject &>().accessor()), MomentAccessor &>
);
static_assert(std::is_same_v<
              decltype(std::declval<MomentObject const &>().accessor()), MomentAccessor const &>);

namespace {

template <class T>
[[nodiscard]] std::vector<T>
copy_to_host(cuda::std::span<T> const source, cudaStream_t const stream = cudaStreamLegacy) {
    std::vector<T> result(source.size());
    EXPECT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            result.data(), source.data(), source.size_bytes(), cudaMemcpyDeviceToHost, stream
        )
    );
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    return result;
}

} // namespace

TEST(gwn_bvh_child, packed_reference_has_stable_boundaries_and_meaning) {
    using Child = gwn::gwn_bvh_child<Real>;
    constexpr std::uint64_t k_valid = 0x0000800000000000ULL;
    constexpr std::uint64_t k_offset_mask = 0x00007fffffffffffULL;

    Child invalid{};
    EXPECT_FALSE(invalid.is_valid());
    EXPECT_FALSE(invalid.is_internal());
    EXPECT_FALSE(invalid.is_leaf());

    Child internal{};
    internal.reference = k_valid | 42u;
    EXPECT_TRUE(internal.is_valid());
    EXPECT_TRUE(internal.is_internal());
    EXPECT_FALSE(internal.is_leaf());
    EXPECT_EQ(internal.offset(), 42u);
    EXPECT_EQ(internal.primitive_count(), 0u);

    Child leaf{};
    leaf.reference = k_valid | 17u | (std::uint64_t(3) << 48);
    EXPECT_TRUE(leaf.is_valid());
    EXPECT_FALSE(leaf.is_internal());
    EXPECT_TRUE(leaf.is_leaf());
    EXPECT_EQ(leaf.offset(), 17u);
    EXPECT_EQ(leaf.primitive_count(), 3u);

    EXPECT_TRUE(Child::can_encode_offset(k_offset_mask));
    EXPECT_FALSE(Child::can_encode_offset(k_offset_mask + 1u));
    EXPECT_TRUE(Child::can_encode_primitive_count(0xffffu));
    EXPECT_FALSE(Child::can_encode_primitive_count(0x10000u));
}

TEST_F(GwnBvhTest, default_build_produces_a_complete_leaf_root) {
    gwn::tests::SingleTriangleMesh const mesh{};
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(mesh.vx), cuda::std::span<Real const>(mesh.vy),
        cuda::std::span<Real const>(mesh.vz), cuda::std::span<Index const>(mesh.i0),
        cuda::std::span<Index const>(mesh.i1), cuda::std::span<Index const>(mesh.i2)
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    gwn::gwn_bvh4_object<Real, Index> bvh;
    status = gwn::gwn_build_bvh(geometry, bvh);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    auto const accessor = bvh.accessor();
    ASSERT_TRUE(accessor.is_valid());
    EXPECT_TRUE(accessor.has_leaf_root());
    EXPECT_FALSE(accessor.has_internal_root());
    EXPECT_TRUE(accessor.nodes.empty());
    EXPECT_EQ(accessor.root.offset(), 0u);
    EXPECT_EQ(accessor.root.primitive_count(), 1u);
    EXPECT_EQ(accessor.internal_stack_bound, 0u);
    EXPECT_EQ(accessor.packed_stack_bound, 0u);
    EXPECT_EQ(accessor.root.bounds.min_x, Real(0));
    EXPECT_EQ(accessor.root.bounds.min_y, Real(0));
    EXPECT_EQ(accessor.root.bounds.min_z, Real(0));
    EXPECT_EQ(accessor.root.bounds.max_x, Real(1));
    EXPECT_EQ(accessor.root.bounds.max_y, Real(1));
    EXPECT_EQ(accessor.root.bounds.max_z, Real(0));

    auto const primitive_indices = copy_to_host(accessor.primitive_indices);
    auto const triangles = copy_to_host(accessor.triangles);
    ASSERT_EQ(primitive_indices, std::vector<Index>{0});
    ASSERT_EQ(triangles.size(), 1u);
    EXPECT_EQ(triangles[0].v0_x, Real(0));
    EXPECT_EQ(triangles[0].v0_y, Real(0));
    EXPECT_EQ(triangles[0].v0_z, Real(0));
    EXPECT_EQ(triangles[0].e1_x, Real(1));
    EXPECT_EQ(triangles[0].e1_y, Real(0));
    EXPECT_EQ(triangles[0].e1_z, Real(0));
    EXPECT_EQ(triangles[0].e2_x, Real(0));
    EXPECT_EQ(triangles[0].e2_y, Real(1));
    EXPECT_EQ(triangles[0].e2_z, Real(0));
}

TEST_F(GwnBvhTest, build_produces_complete_query_records) {
    std::array<Real, 6> const x{Real(0), Real(1), Real(0), Real(4), Real(6), Real(4)};
    std::array<Real, 6> const y{Real(0), Real(0), Real(2), Real(0), Real(0), Real(3)};
    std::array<Real, 6> const z{Real(1), Real(1), Real(1), Real(-2), Real(-2), Real(-2)};
    std::array<Index, 2> const i0{0, 3};
    std::array<Index, 2> const i1{1, 4};
    std::array<Index, 2> const i2{2, 5};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(x), cuda::std::span<Real const>(y),
        cuda::std::span<Real const>(z), cuda::std::span<Index const>(i0),
        cuda::std::span<Index const>(i1), cuda::std::span<Index const>(i2)
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    gwn::gwn_bvh4_object<Real, Index> bvh;
    status = gwn::gwn_build_bvh(
        geometry, bvh, gwn::gwn_bvh_build_options{.method = gwn::gwn_bvh_build_method::k_lbvh}
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_TRUE(bvh.has_data());

    auto const accessor = bvh.accessor();
    ASSERT_TRUE(accessor.is_valid());
    ASSERT_TRUE(accessor.root.is_internal());
    ASSERT_EQ(accessor.nodes.size(), 1u);
    ASSERT_EQ(accessor.primitive_indices.size(), 2u);
    ASSERT_EQ(accessor.triangles.size(), 2u);

    auto const nodes = copy_to_host(accessor.nodes);
    auto const primitive_indices = copy_to_host(accessor.primitive_indices);
    auto const triangles = copy_to_host(accessor.triangles);
    auto const &root_node = nodes.at(static_cast<std::size_t>(accessor.root.offset()));

    std::array<bool, 2> seen{false, false};
    std::size_t leaf_count = 0;
    for (int slot = 0; slot < 4; ++slot) {
        auto const &child = root_node.child(slot);
        if (!child.is_valid())
            continue;

        ASSERT_TRUE(child.is_leaf());
        ASSERT_EQ(child.primitive_count(), 1u);
        auto const sorted_index = static_cast<std::size_t>(child.offset());
        ASSERT_LT(sorted_index, primitive_indices.size());
        Index const primitive_id = primitive_indices[sorted_index];
        ASSERT_LT(primitive_id, Index(2));
        EXPECT_FALSE(seen[primitive_id]);
        seen[primitive_id] = true;

        auto const &triangle = triangles[sorted_index];
        auto const vertex = [&](Index const vertex_id) {
            return std::array<Real, 3>{x[vertex_id], y[vertex_id], z[vertex_id]};
        };
        auto const v0 = vertex(i0[primitive_id]);
        auto const v1 = vertex(i1[primitive_id]);
        auto const v2 = vertex(i2[primitive_id]);
        EXPECT_EQ(triangle.v0_x, v0[0]);
        EXPECT_EQ(triangle.v0_y, v0[1]);
        EXPECT_EQ(triangle.v0_z, v0[2]);
        EXPECT_EQ(triangle.e1_x, v1[0] - v0[0]);
        EXPECT_EQ(triangle.e1_y, v1[1] - v0[1]);
        EXPECT_EQ(triangle.e1_z, v1[2] - v0[2]);
        EXPECT_EQ(triangle.e2_x, v2[0] - v0[0]);
        EXPECT_EQ(triangle.e2_y, v2[1] - v0[1]);
        EXPECT_EQ(triangle.e2_z, v2[2] - v0[2]);
        ++leaf_count;
    }

    EXPECT_EQ(leaf_count, 2u);
    EXPECT_TRUE(seen[0]);
    EXPECT_TRUE(seen[1]);
}

TEST_F(GwnBvhTest, refit_replaces_geometry_data_and_preserves_primitive_order) {
    std::array<Real, 6> x{Real(0), Real(1), Real(0), Real(4), Real(6), Real(4)};
    std::array<Real, 6> y{Real(0), Real(0), Real(2), Real(0), Real(0), Real(3)};
    std::array<Real, 6> z{Real(1), Real(1), Real(1), Real(-2), Real(-2), Real(-2)};
    std::array<Index, 2> const i0{0, 3};
    std::array<Index, 2> const i1{1, 4};
    std::array<Index, 2> const i2{2, 5};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(x), cuda::std::span<Real const>(y),
        cuda::std::span<Real const>(z), cuda::std::span<Index const>(i0),
        cuda::std::span<Index const>(i1), cuda::std::span<Index const>(i2)
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    gwn::gwn_bvh4_object<Real, Index> bvh;
    status = gwn::gwn_build_bvh(
        geometry, bvh, gwn::gwn_bvh_build_options{.method = gwn::gwn_bvh_build_method::k_lbvh}
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    auto const primitive_order_before = copy_to_host(bvh.accessor().primitive_indices);

    x = {Real(-7), Real(-2), Real(-7), Real(8), Real(11), Real(8)};
    y = {Real(5), Real(5), Real(9), Real(-4), Real(-4), Real(2)};
    z = {Real(3), Real(3), Real(3), Real(-6), Real(-6), Real(-6)};
    status = gwn::gwn_update_geometry(
        geometry, cuda::std::span<Real const>(x), cuda::std::span<Real const>(y),
        cuda::std::span<Real const>(z)
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    status = gwn::gwn_refit_bvh(geometry, bvh);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    auto const accessor = bvh.accessor();
    ASSERT_TRUE(accessor.is_valid());
    EXPECT_EQ(copy_to_host(accessor.primitive_indices), primitive_order_before);
    EXPECT_EQ(accessor.root.bounds.min_x, Real(-7));
    EXPECT_EQ(accessor.root.bounds.min_y, Real(-4));
    EXPECT_EQ(accessor.root.bounds.min_z, Real(-6));
    EXPECT_EQ(accessor.root.bounds.max_x, Real(11));
    EXPECT_EQ(accessor.root.bounds.max_y, Real(9));
    EXPECT_EQ(accessor.root.bounds.max_z, Real(3));

    auto const triangles = copy_to_host(accessor.triangles);
    auto const primitive_indices = copy_to_host(accessor.primitive_indices);
    for (std::size_t sorted_index = 0; sorted_index < triangles.size(); ++sorted_index) {
        Index const primitive_id = primitive_indices[sorted_index];
        auto const &triangle = triangles[sorted_index];
        Index const v0 = i0[primitive_id];
        Index const v1 = i1[primitive_id];
        Index const v2 = i2[primitive_id];
        EXPECT_EQ(triangle.v0_x, x[v0]);
        EXPECT_EQ(triangle.v0_y, y[v0]);
        EXPECT_EQ(triangle.v0_z, z[v0]);
        EXPECT_EQ(triangle.e1_x, x[v1] - x[v0]);
        EXPECT_EQ(triangle.e1_y, y[v1] - y[v0]);
        EXPECT_EQ(triangle.e1_z, z[v1] - z[v0]);
        EXPECT_EQ(triangle.e2_x, x[v2] - x[v0]);
        EXPECT_EQ(triangle.e2_y, y[v2] - y[v0]);
        EXPECT_EQ(triangle.e2_z, z[v2] - z[v0]);
    }
}

TEST_F(GwnBvhStreamTest, object_replacement_preserves_failure_and_stream_contracts) {
    gwn::tests::SingleTriangleMesh const mesh{};
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(mesh.vx), cuda::std::span<Real const>(mesh.vy),
        cuda::std::span<Real const>(mesh.vz), cuda::std::span<Index const>(mesh.i0),
        cuda::std::span<Index const>(mesh.i1), cuda::std::span<Index const>(mesh.i2), stream_a_
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    gwn::gwn_bvh4_object<Real, Index> bvh;
    status = gwn::gwn_build_bvh(geometry, bvh, {}, stream_a_);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_TRUE(bvh.has_data());
    ASSERT_EQ(bvh.stream(), stream_a_);
    auto *const nodes_before = bvh.accessor().nodes.data();
    auto *const primitives_before = bvh.accessor().primitive_indices.data();
    auto *const triangles_before = bvh.accessor().triangles.data();
    auto const root_before = bvh.accessor().root;

    status = gwn::gwn_build_bvh(
        geometry, bvh,
        gwn::gwn_bvh_build_options{.method = static_cast<gwn::gwn_bvh_build_method>(0xff)},
        stream_b_
    );
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
    EXPECT_EQ(bvh.stream(), stream_a_);
    EXPECT_EQ(bvh.accessor().nodes.data(), nodes_before);
    EXPECT_EQ(bvh.accessor().primitive_indices.data(), primitives_before);
    EXPECT_EQ(bvh.accessor().triangles.data(), triangles_before);
    EXPECT_EQ(bvh.accessor().root.reference, root_before.reference);

    for (std::uint32_t const search_radius : {0u, 9u}) {
        status = gwn::gwn_build_bvh(
            geometry, bvh,
            gwn::gwn_bvh_build_options{
                .method = gwn::gwn_bvh_build_method::k_hploc,
                .hploc_search_radius = search_radius,
            },
            stream_b_
        );
        EXPECT_FALSE(status.is_ok());
        EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
        EXPECT_EQ(bvh.stream(), stream_a_);
        EXPECT_EQ(bvh.accessor().nodes.data(), nodes_before);
        EXPECT_EQ(bvh.accessor().primitive_indices.data(), primitives_before);
        EXPECT_EQ(bvh.accessor().triangles.data(), triangles_before);
        EXPECT_EQ(bvh.accessor().root.reference, root_before.reference);
    }

    gwn::gwn_bvh4_object<Real, Index> moved(std::move(bvh));
    EXPECT_FALSE(bvh.has_data());
    EXPECT_TRUE(moved.has_data());
    EXPECT_EQ(moved.stream(), stream_a_);

    gwn::gwn_geometry_object<Real, Index> empty_geometry;
    status = gwn::gwn_build_bvh(empty_geometry, moved, {}, stream_b_);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_FALSE(moved.has_data());
    EXPECT_EQ(moved.stream(), stream_b_);

    moved.clear();
    moved.set_stream(stream_a_);
    EXPECT_FALSE(moved.has_data());
    EXPECT_EQ(moved.stream(), stream_a_);
}

TEST_F(GwnBvhStreamTest, hploc_multi_triangle_replacement_rebinds_stream) {
    gwn::tests::SingleTriangleMesh const single_mesh{};
    gwn::gwn_geometry_object<Real, Index> single_geometry;
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        single_geometry, cuda::std::span<Real const>(single_mesh.vx),
        cuda::std::span<Real const>(single_mesh.vy), cuda::std::span<Real const>(single_mesh.vz),
        cuda::std::span<Index const>(single_mesh.i0), cuda::std::span<Index const>(single_mesh.i1),
        cuda::std::span<Index const>(single_mesh.i2), stream_a_
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    gwn::gwn_bvh4_object<Real, Index> bvh;
    status = gwn::gwn_build_bvh(single_geometry, bvh, {}, stream_a_);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    std::array<Real, 3> const x{Real(0), Real(1), Real(0)};
    std::array<Real, 3> const y{Real(0), Real(0), Real(1)};
    std::array<Real, 3> const z{Real(0), Real(0), Real(0)};
    constexpr std::size_t k_triangle_count = 33;
    std::array<Index, k_triangle_count> i0{};
    std::array<Index, k_triangle_count> i1{};
    std::array<Index, k_triangle_count> i2{};
    i1.fill(Index(1));
    i2.fill(Index(2));

    gwn::gwn_geometry_object<Real, Index> multi_geometry;
    status = gwn::gwn_upload_geometry(
        multi_geometry, cuda::std::span<Real const>(x), cuda::std::span<Real const>(y),
        cuda::std::span<Real const>(z), cuda::std::span<Index const>(i0),
        cuda::std::span<Index const>(i1), cuda::std::span<Index const>(i2), stream_b_
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    status = gwn::gwn_build_bvh(
        multi_geometry, bvh,
        gwn::gwn_bvh_build_options{.method = gwn::gwn_bvh_build_method::k_hploc}, stream_b_
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_EQ(bvh.stream(), stream_b_);
    ASSERT_TRUE(bvh.accessor().has_internal_root());

    auto primitive_indices = copy_to_host(bvh.accessor().primitive_indices, stream_b_);
    ASSERT_EQ(primitive_indices.size(), k_triangle_count);
    std::sort(primitive_indices.begin(), primitive_indices.end());
    for (std::size_t primitive_id = 0; primitive_id < primitive_indices.size(); ++primitive_id)
        EXPECT_EQ(primitive_indices[primitive_id], static_cast<Index>(primitive_id));
}

TEST_F(GwnBvhStreamTest, refit_rejects_topology_change_without_mutating_bvh) {
    gwn::tests::SingleTriangleMesh const mesh{};
    gwn::gwn_geometry_object<Real, Index> one_triangle;
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        one_triangle, cuda::std::span<Real const>(mesh.vx), cuda::std::span<Real const>(mesh.vy),
        cuda::std::span<Real const>(mesh.vz), cuda::std::span<Index const>(mesh.i0),
        cuda::std::span<Index const>(mesh.i1), cuda::std::span<Index const>(mesh.i2), stream_a_
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    gwn::gwn_bvh4_object<Real, Index> bvh;
    status = gwn::gwn_build_bvh(one_triangle, bvh, {}, stream_a_);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    auto const root_before = bvh.accessor().root;
    auto const triangle_before = copy_to_host(bvh.accessor().triangles, stream_a_).at(0);

    std::array<Real, 4> const x{Real(0), Real(1), Real(0), Real(1)};
    std::array<Real, 4> const y{Real(0), Real(0), Real(1), Real(1)};
    std::array<Real, 4> const z{Real(0), Real(0), Real(0), Real(0)};
    std::array<Index, 2> const i0{0, 1};
    std::array<Index, 2> const i1{1, 3};
    std::array<Index, 2> const i2{2, 2};
    gwn::gwn_geometry_object<Real, Index> two_triangles;
    status = gwn::gwn_upload_geometry(
        two_triangles, cuda::std::span<Real const>(x), cuda::std::span<Real const>(y),
        cuda::std::span<Real const>(z), cuda::std::span<Index const>(i0),
        cuda::std::span<Index const>(i1), cuda::std::span<Index const>(i2), stream_b_
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    status = gwn::gwn_refit_bvh(two_triangles, bvh, stream_b_);
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
    EXPECT_EQ(bvh.stream(), stream_a_);
    EXPECT_EQ(bvh.accessor().root.reference, root_before.reference);
    EXPECT_EQ(bvh.accessor().root.bounds.min_x, root_before.bounds.min_x);
    EXPECT_EQ(bvh.accessor().root.bounds.max_y, root_before.bounds.max_y);
    auto const triangle_after = copy_to_host(bvh.accessor().triangles, stream_a_).at(0);
    EXPECT_EQ(triangle_after.v0_x, triangle_before.v0_x);
    EXPECT_EQ(triangle_after.v0_y, triangle_before.v0_y);
    EXPECT_EQ(triangle_after.v0_z, triangle_before.v0_z);
    EXPECT_EQ(triangle_after.e1_x, triangle_before.e1_x);
    EXPECT_EQ(triangle_after.e1_y, triangle_before.e1_y);
    EXPECT_EQ(triangle_after.e1_z, triangle_before.e1_z);
    EXPECT_EQ(triangle_after.e2_x, triangle_before.e2_x);
    EXPECT_EQ(triangle_after.e2_y, triangle_before.e2_y);
    EXPECT_EQ(triangle_after.e2_z, triangle_before.e2_z);
}

TEST_F(GwnBvhStreamTest, failed_in_place_refit_retains_clear_and_rebuild_paths) {
    auto const expect_contract =
        [&](cuda::std::span<Real const> const x, cuda::std::span<Real const> const y,
            cuda::std::span<Real const> const z, cuda::std::span<Index const> const i0,
            cuda::std::span<Index const> const i1, cuda::std::span<Index const> const i2,
            bool const clear_before_rebuild) {
        gwn::gwn_geometry_object<Real, Index> geometry;
        gwn::gwn_status status = gwn::gwn_upload_geometry(geometry, x, y, z, i0, i1, i2, stream_a_);
        ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

        gwn::gwn_bvh4_object<Real, Index> bvh;
        ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh, {}, stream_a_).is_ok());
        gwn::gwn_bvh4_moment_object<1, Real, Index> moment;
        ASSERT_TRUE(gwn::gwn_refit_bvh_moment<1>(bvh, moment, stream_a_).is_ok());
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));

        std::size_t const damaged_triangle = i0.size() - 1;
        Index const original_index = i0[damaged_triangle];
        Index const invalid_index = static_cast<Index>(geometry.vertex_count());
        ASSERT_EQ(
            cudaSuccess, cudaMemcpyAsync(
                             geometry.accessor().tri_i0.data() + damaged_triangle, &invalid_index,
                             sizeof(invalid_index), cudaMemcpyHostToDevice, stream_b_
                         )
        );

        status = gwn::gwn_refit_bvh(geometry, bvh, stream_b_);
        EXPECT_EQ(status.error(), gwn::gwn_error::internal_error);
        EXPECT_FALSE(bvh.has_data());
        EXPECT_FALSE(moment.accessor().is_valid_for(bvh.accessor()));
        EXPECT_EQ(bvh.stream(), stream_b_);

        if (clear_before_rebuild) {
            bvh.clear();
            EXPECT_FALSE(bvh.has_data());
            EXPECT_TRUE(bvh.accessor().nodes.empty());
            EXPECT_TRUE(bvh.accessor().primitive_indices.empty());
            EXPECT_TRUE(bvh.accessor().triangles.empty());
        }

        ASSERT_EQ(
            cudaSuccess, cudaMemcpyAsync(
                             geometry.accessor().tri_i0.data() + damaged_triangle, &original_index,
                             sizeof(original_index), cudaMemcpyHostToDevice, stream_b_
                         )
        );
        status = gwn::gwn_build_bvh(geometry, bvh, {}, stream_b_);
        EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
        EXPECT_TRUE(bvh.has_data());
        EXPECT_EQ(bvh.stream(), stream_b_);
    };

    gwn::tests::SingleTriangleMesh const leaf_mesh{};
    expect_contract(
        cuda::std::span<Real const>(leaf_mesh.vx), cuda::std::span<Real const>(leaf_mesh.vy),
        cuda::std::span<Real const>(leaf_mesh.vz), cuda::std::span<Index const>(leaf_mesh.i0),
        cuda::std::span<Index const>(leaf_mesh.i1), cuda::std::span<Index const>(leaf_mesh.i2), true
    );

    std::array<Real, 6> const x{Real(0), Real(1), Real(0), Real(4), Real(5), Real(4)};
    std::array<Real, 6> const y{Real(0), Real(0), Real(1), Real(0), Real(0), Real(1)};
    std::array<Real, 6> const z{};
    std::array<Index, 2> const i0{0, 3};
    std::array<Index, 2> const i1{1, 4};
    std::array<Index, 2> const i2{2, 5};
    expect_contract(
        cuda::std::span<Real const>(x), cuda::std::span<Real const>(y),
        cuda::std::span<Real const>(z), cuda::std::span<Index const>(i0),
        cuda::std::span<Index const>(i1), cuda::std::span<Index const>(i2), false
    );
}

TEST_F(GwnBvhTest, public_build_rejects_primitive_count_above_native_counter_range) {
    using BigIndex = std::uint64_t;
    gwn::tests::SingleTriangleMesh const mesh{};
    std::array<BigIndex, 1> const i0{0};
    std::array<BigIndex, 1> const i1{1};
    std::array<BigIndex, 1> const i2{2};
    gwn::gwn_geometry_object<Real, BigIndex> geometry;
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(mesh.vx), cuda::std::span<Real const>(mesh.vy),
        cuda::std::span<Real const>(mesh.vz), cuda::std::span<BigIndex const>(i0),
        cuda::std::span<BigIndex const>(i1), cuda::std::span<BigIndex const>(i2)
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    auto &accessor = geometry.accessor();
    auto const original_i0 = accessor.tri_i0;
    auto const original_i1 = accessor.tri_i1;
    auto const original_i2 = accessor.tri_i2;
    constexpr std::size_t k_too_many =
        static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) + 1u;
    accessor.tri_i0 = cuda::std::span<BigIndex>(original_i0.data(), k_too_many);
    accessor.tri_i1 = cuda::std::span<BigIndex>(original_i1.data(), k_too_many);
    accessor.tri_i2 = cuda::std::span<BigIndex>(original_i2.data(), k_too_many);

    gwn::gwn_bvh4_object<Real, BigIndex> bvh;
    status = gwn::gwn_build_bvh(geometry, bvh);
    accessor.tri_i0 = original_i0;
    accessor.tri_i1 = original_i1;
    accessor.tri_i2 = original_i2;

    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
    EXPECT_FALSE(bvh.has_data());
}
