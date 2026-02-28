#include <array>

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
    EXPECT_EQ(accessor.vertex_count(), 0u);
    EXPECT_EQ(accessor.triangle_count(), 0u);
}

// Upload, success path.

TEST_F(CudaFixture, upload_valid_single_triangle) {
    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vy{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0}, i1{1}, i2{2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = geometry.upload(
        cuda::std::span<Real const>(vx.data(), vx.size()),
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
}

// Upload, error paths: SoA length mismatch.

TEST_F(CudaFixture, upload_vertex_soa_length_mismatch) {
    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 2> const vy{0.0f, 1.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0}, i1{1}, i2{2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = geometry.upload(
        cuda::std::span<Real const>(vx.data(), vx.size()),
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
    gwn::gwn_status const status = geometry.upload(
        cuda::std::span<Real const>(vx.data(), vx.size()),
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

// Upload, empty geometry is valid (zero triangles).

TEST_F(CudaFixture, upload_empty_geometry) {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status = geometry.upload(
        cuda::std::span<Real const>{}, cuda::std::span<Real const>{}, cuda::std::span<Real const>{},
        cuda::std::span<Index const>{}, cuda::std::span<Index const>{},
        cuda::std::span<Index const>{}
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
    gwn::gwn_status status = geometry.upload(
        cuda::std::span<Real const>(vx.data(), vx.size()),
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

    status = geometry.upload(
        cuda::std::span<Real const>(vx2.data(), vx2.size()),
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
    gwn::gwn_status const status = src.upload(
        cuda::std::span<Real const>(vx.data(), vx.size()),
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

