#include <array>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_utils.hpp"

// gwn_geometry_object / gwn_geometry_accessor unit tests.
// Covers upload validation, error paths, empty geometry, SoA mismatch.

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

// Accessor, host-side validation.

TEST(smallgwn_unit_geometry, default_accessor_is_empty_and_valid) {
    gwn::gwn_geometry_accessor<Real, Index> accessor{};
    EXPECT_TRUE(accessor.is_valid());
    EXPECT_FALSE(accessor.has_singular_edges());
    EXPECT_EQ(accessor.singular_edge_count, 0u);
    EXPECT_EQ(accessor.vertex_count(), 0u);
    EXPECT_EQ(accessor.triangle_count(), 0u);
    EXPECT_EQ(accessor.tri_boundary_edge_mask.size(), 0u);
}

TEST(smallgwn_unit_geometry, boundary_edge_mask_single_triangle_all_bits_set) {
    std::array<Index, 1> const i0{0};
    std::array<Index, 1> const i1{1};
    std::array<Index, 1> const i2{2};
    std::array<std::uint8_t, 1> mask{0};

    gwn::gwn_status const s = gwn::gwn_compute_triangle_boundary_edge_mask<Index>(
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size()),
        cuda::std::span<std::uint8_t>(mask.data(), mask.size())
    );
    ASSERT_TRUE(s.is_ok()) << gwn::tests::status_to_debug_string(s);
    EXPECT_EQ(mask[0], std::uint8_t(0x7u));
}

TEST(smallgwn_unit_geometry, boundary_edge_mask_consistent_shared_edge_is_not_boundary) {
    // Tri 0: (0,1,2)
    // Tri 1: (2,1,3) -- shared edge is (1,2)/(2,1), so interior.
    std::array<Index, 2> const i0{0, 2};
    std::array<Index, 2> const i1{1, 1};
    std::array<Index, 2> const i2{2, 3};
    std::array<std::uint8_t, 2> mask{0, 0};

    gwn::gwn_status const s = gwn::gwn_compute_triangle_boundary_edge_mask<Index>(
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size()),
        cuda::std::span<std::uint8_t>(mask.data(), mask.size())
    );
    ASSERT_TRUE(s.is_ok()) << gwn::tests::status_to_debug_string(s);
    EXPECT_EQ(mask[0], std::uint8_t(0x5u));
    EXPECT_EQ(mask[1], std::uint8_t(0x6u));
}

TEST(smallgwn_unit_geometry, boundary_edge_mask_inconsistent_shared_edge_marks_boundary) {
    // Tri 0: (0,1,2)
    // Tri 1: (0,1,3) -- shared edge (0,1) has same orientation in both triangles.
    std::array<Index, 2> const i0{0, 0};
    std::array<Index, 2> const i1{1, 1};
    std::array<Index, 2> const i2{2, 3};
    std::array<std::uint8_t, 2> mask{0, 0};

    gwn::gwn_status const s = gwn::gwn_compute_triangle_boundary_edge_mask<Index>(
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size()),
        cuda::std::span<std::uint8_t>(mask.data(), mask.size())
    );
    ASSERT_TRUE(s.is_ok()) << gwn::tests::status_to_debug_string(s);
    EXPECT_EQ(mask[0], std::uint8_t(0x7u));
    EXPECT_EQ(mask[1], std::uint8_t(0x7u));
}

TEST(smallgwn_unit_geometry, boundary_edge_mask_non_manifold_shared_edge_marks_boundary) {
    // Three triangles share edge (0,1) -> non-manifold, must be boundary.
    std::array<Index, 3> const i0{0, 1, 0};
    std::array<Index, 3> const i1{1, 0, 1};
    std::array<Index, 3> const i2{2, 3, 4};
    std::array<std::uint8_t, 3> mask{0, 0, 0};

    gwn::gwn_status const s = gwn::gwn_compute_triangle_boundary_edge_mask<Index>(
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size()),
        cuda::std::span<std::uint8_t>(mask.data(), mask.size())
    );
    ASSERT_TRUE(s.is_ok()) << gwn::tests::status_to_debug_string(s);
    EXPECT_EQ(mask[0], std::uint8_t(0x7u));
    EXPECT_EQ(mask[1], std::uint8_t(0x7u));
    EXPECT_EQ(mask[2], std::uint8_t(0x7u));
}

TEST(smallgwn_unit_geometry, boundary_edge_mask_rejects_mismatched_output_size) {
    std::array<Index, 2> const i0{0, 0};
    std::array<Index, 2> const i1{1, 1};
    std::array<Index, 2> const i2{2, 3};
    std::array<std::uint8_t, 1> mask{0};

    gwn::gwn_status const s = gwn::gwn_compute_triangle_boundary_edge_mask<Index>(
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size()),
        cuda::std::span<std::uint8_t>(mask.data(), mask.size())
    );
    EXPECT_FALSE(s.is_ok());
    EXPECT_EQ(s.error(), gwn::gwn_error::invalid_argument);
}

TEST(smallgwn_unit_geometry, accessor_spans_are_mutable_by_default) {
    using accessor_t = gwn::gwn_geometry_accessor<Real, Index>;
    static_assert(std::is_same_v<decltype(accessor_t{}.vertex_x), cuda::std::span<Real>>);
    static_assert(std::is_same_v<decltype(accessor_t{}.vertex_y), cuda::std::span<Real>>);
    static_assert(std::is_same_v<decltype(accessor_t{}.vertex_z), cuda::std::span<Real>>);
    static_assert(std::is_same_v<decltype(accessor_t{}.tri_i0), cuda::std::span<Index>>);
    static_assert(std::is_same_v<decltype(accessor_t{}.tri_i1), cuda::std::span<Index>>);
    static_assert(std::is_same_v<decltype(accessor_t{}.tri_i2), cuda::std::span<Index>>);
    SUCCEED();
}

TEST(smallgwn_unit_geometry, upload_rejects_non_empty_null_storage_spans) {
    std::array<Real, 1> const vx{1.0f};
    std::array<Real, 1> const vy{0.0f};
    std::array<Real, 1> const vz{0.0f};
    auto const *null_index = static_cast<Index const *>(nullptr);

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(null_index, 1), cuda::std::span<Index const>(null_index, 1),
        cuda::std::span<Index const>(null_index, 1)
    );

    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

// Upload, success path.

TEST_F(CudaFixture, upload_valid_single_triangle) {
    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vy{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0}, i1{1}, i2{2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    EXPECT_EQ(geometry.vertex_count(), 3u);
    EXPECT_EQ(geometry.triangle_count(), 1u);
    EXPECT_TRUE(geometry.accessor().is_valid());
    EXPECT_TRUE(geometry.accessor().has_singular_edges());
    EXPECT_EQ(geometry.accessor().singular_edge_count, 3u);

    std::vector<std::uint8_t> host_mask(1, 0);
    gwn::gwn_status const copy_status = gwn::detail::gwn_copy_d2h<std::uint8_t>(
        cuda::std::span<std::uint8_t>(host_mask.data(), host_mask.size()),
        geometry.accessor().tri_boundary_edge_mask, gwn::gwn_default_stream()
    );
    ASSERT_TRUE(copy_status.is_ok()) << gwn::tests::status_to_debug_string(copy_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(host_mask[0], std::uint8_t(0x7u));
}

TEST_F(CudaFixture, upload_closed_tetra_has_zero_singular_edges) {
    std::array<Real, 4> const vx{Real(0), Real(1), Real(0), Real(0)};
    std::array<Real, 4> const vy{Real(0), Real(0), Real(1), Real(0)};
    std::array<Real, 4> const vz{Real(0), Real(0), Real(0), Real(1)};
    std::array<Index, 4> const i0{0, 0, 0, 1};
    std::array<Index, 4> const i1{2, 1, 3, 2};
    std::array<Index, 4> const i2{1, 3, 2, 3};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_FALSE(geometry.accessor().has_singular_edges());
    EXPECT_EQ(geometry.accessor().singular_edge_count, 0u);
}

TEST_F(CudaFixture, upload_open_tetra_has_nonzero_singular_edges) {
    std::array<Real, 4> const vx{Real(0), Real(1), Real(0), Real(0)};
    std::array<Real, 4> const vy{Real(0), Real(0), Real(1), Real(0)};
    std::array<Real, 4> const vz{Real(0), Real(0), Real(0), Real(1)};
    std::array<Index, 3> const i0{0, 0, 0};
    std::array<Index, 3> const i1{2, 1, 3};
    std::array<Index, 3> const i2{1, 3, 2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_TRUE(geometry.accessor().has_singular_edges());
    EXPECT_GT(geometry.accessor().singular_edge_count, 0u);
}

// Upload, error paths: SoA length mismatch.

TEST_F(CudaFixture, upload_vertex_soa_length_mismatch) {
    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 2> const vy{0.0f, 1.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0}, i1{1}, i2{2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, upload_triangle_soa_length_mismatch) {
    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vy{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0};
    std::array<Index, 2> const i1{1, 0};
    std::array<Index, 1> const i2{2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, upload_triangle_index_out_of_range_rejected) {
    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vy{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0};
    std::array<Index, 1> const i1{1};
    std::array<Index, 1> const i2{3}; // vertex_count == 3, so this is out of range.

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, upload_triangles_without_vertices_rejected) {
    std::array<Index, 1> const i0{0};
    std::array<Index, 1> const i1{0};
    std::array<Index, 1> const i2{0};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>{}, cuda::std::span<Real const>{},
        cuda::std::span<Real const>{}, cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

// Upload, empty geometry is valid (zero triangles).

TEST_F(CudaFixture, upload_empty_geometry) {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>{}, cuda::std::span<Real const>{},
        cuda::std::span<Real const>{}, cuda::std::span<Index const>{},
        cuda::std::span<Index const>{}, cuda::std::span<Index const>{}
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    EXPECT_EQ(geometry.vertex_count(), 0u);
    EXPECT_EQ(geometry.triangle_count(), 0u);
}

// Clear releases geometry.

TEST_F(CudaFixture, clear_releases_geometry) {
    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vy{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0}, i1{1}, i2{2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const upload_status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
    ASSERT_TRUE(upload_status.is_ok());

    geometry.clear();
    EXPECT_EQ(geometry.vertex_count(), 0u);
    EXPECT_EQ(geometry.triangle_count(), 0u);
}

// Re-upload overwrites previous data.

TEST_F(CudaFixture, re_upload_overwrites_previous) {
    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vy{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0}, i1{1}, i2{2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok());
    EXPECT_EQ(geometry.triangle_count(), 1u);

    // Upload with 2 triangles.
    std::array<Real, 4> const vx2{1.0f, 0.0f, 0.0f, 0.5f};
    std::array<Real, 4> const vy2{0.0f, 1.0f, 0.0f, 0.5f};
    std::array<Real, 4> const vz2{0.0f, 0.0f, 1.0f, 0.0f};
    std::array<Index, 2> const i0b{0, 0}, i1b{1, 2}, i2b{2, 3};

    status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx2.data(), vx2.size()),
        cuda::std::span<Real const>(vy2.data(), vy2.size()),
        cuda::std::span<Real const>(vz2.data(), vz2.size()),
        cuda::std::span<Index const>(i0b.data(), i0b.size()),
        cuda::std::span<Index const>(i1b.data(), i1b.size()),
        cuda::std::span<Index const>(i2b.data(), i2b.size())
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(geometry.triangle_count(), 2u);
    EXPECT_EQ(geometry.vertex_count(), 4u);
}

// Move semantics preserve accessor validity.

TEST_F(CudaFixture, move_preserves_accessor) {
    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vy{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0}, i1{1}, i2{2};

    gwn::gwn_geometry_object<Real, Index> src;
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        src, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok());

    gwn::gwn_geometry_object<Real, Index> dst(std::move(src));
    EXPECT_EQ(dst.triangle_count(), 1u);
    EXPECT_EQ(dst.vertex_count(), 3u);
    EXPECT_TRUE(dst.accessor().is_valid());
    EXPECT_EQ(src.triangle_count(), 0u);
}
