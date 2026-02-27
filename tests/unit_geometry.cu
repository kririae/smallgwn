#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_utils.hpp"

// ---------------------------------------------------------------------------
// gwn_geometry_object / gwn_geometry_accessor unit tests.
// Covers upload validation, error paths, empty geometry, SoA mismatch.
// ---------------------------------------------------------------------------

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;
using gwn::tests::CudaStreamFixture;

// ---------------------------------------------------------------------------
// Accessor — host-side validation.
// ---------------------------------------------------------------------------

TEST(smallgwn_unit_geometry, default_accessor_is_empty_and_valid) {
    gwn::gwn_geometry_accessor<Real, Index> accessor{};
    EXPECT_TRUE(accessor.is_valid());
    EXPECT_EQ(accessor.vertex_count(), 0u);
    EXPECT_EQ(accessor.triangle_count(), 0u);
}

TEST(smallgwn_unit_geometry, singular_edge_extraction_octahedron_is_empty) {
    std::array<Index, 8> const i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> const i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> const i2{4, 4, 4, 4, 5, 5, 5, 5};
    auto const edges = gwn::detail::gwn_extract_singular_edges<Index>(
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    EXPECT_EQ(edges.i0.size(), 0u);
    EXPECT_EQ(edges.i1.size(), 0u);
}

TEST(smallgwn_unit_geometry, singular_edge_extraction_half_octahedron_has_boundary_loop) {
    std::array<Index, 4> const i0{0, 2, 1, 3};
    std::array<Index, 4> const i1{2, 1, 3, 0};
    std::array<Index, 4> const i2{4, 4, 4, 4};
    auto const edges = gwn::detail::gwn_extract_singular_edges<Index>(
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    EXPECT_EQ(edges.i0.size(), 4u);
    EXPECT_EQ(edges.i1.size(), 4u);
}

// ---------------------------------------------------------------------------
// Upload — success path.
// ---------------------------------------------------------------------------

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
    EXPECT_EQ(geometry.singular_edge_count(), 3u);
    EXPECT_TRUE(geometry.accessor().is_valid());
}

TEST_F(CudaFixture, singular_edges_ignore_interior_triangulation_edges) {
    // Planar square split into two triangles: the interior diagonal is not a
    // singular edge of the solid-angle field and must be excluded.
    std::array<Real, 4> const vx{-1.0f, 1.0f, 1.0f, -1.0f};
    std::array<Real, 4> const vy{-1.0f, -1.0f, 1.0f, 1.0f};
    std::array<Real, 4> const vz{0.0f, 0.0f, 0.0f, 0.0f};
    std::array<Index, 2> const i0{0, 0};
    std::array<Index, 2> const i1{1, 2};
    std::array<Index, 2> const i2{2, 3};

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

    EXPECT_EQ(geometry.singular_edge_count(), 4u);
}

// ---------------------------------------------------------------------------
// Upload — error paths: SoA length mismatch.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Upload — empty geometry is valid (zero triangles).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Clear releases geometry.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Re-upload overwrites previous data.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Move semantics preserve accessor validity.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// GPU singular edge build parity with CPU reference.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, gpu_singular_edges_match_cpu_reference_single_triangle) {
    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vy{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0}, i1{1}, i2{2};

    // CPU reference.
    auto const cpu_edges = gwn::detail::gwn_extract_singular_edges<Index>(
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );

    // GPU path (via upload).
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

    EXPECT_EQ(geometry.singular_edge_count(), cpu_edges.i0.size());

    // Download GPU results and compare.
    std::vector<Index> gpu_i0(geometry.singular_edge_count());
    std::vector<Index> gpu_i1(geometry.singular_edge_count());
    if (!gpu_i0.empty()) {
        cudaMemcpy(gpu_i0.data(), geometry.accessor().singular_edge_i0.data(),
                   gpu_i0.size() * sizeof(Index), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_i1.data(), geometry.accessor().singular_edge_i1.data(),
                   gpu_i1.size() * sizeof(Index), cudaMemcpyDeviceToHost);
    }

    // Both should be sorted by (i0, i1), so direct comparison works.
    ASSERT_EQ(gpu_i0.size(), cpu_edges.i0.size());
    for (std::size_t e = 0; e < gpu_i0.size(); ++e) {
        EXPECT_EQ(gpu_i0[e], cpu_edges.i0[e]) << "edge " << e << " i0 mismatch";
        EXPECT_EQ(gpu_i1[e], cpu_edges.i1[e]) << "edge " << e << " i1 mismatch";
    }
}

TEST_F(CudaFixture, gpu_singular_edges_match_cpu_reference_single_triangle_uint64) {
    using Index64 = std::uint64_t;

    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vy{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index64, 1> const i0{0}, i1{1}, i2{2};

    auto const cpu_edges = gwn::detail::gwn_extract_singular_edges<Index64>(
        cuda::std::span<Index64 const>(i0.data(), i0.size()),
        cuda::std::span<Index64 const>(i1.data(), i1.size()),
        cuda::std::span<Index64 const>(i2.data(), i2.size())
    );

    gwn::gwn_geometry_object<Real, Index64> geometry;
    gwn::gwn_status const status = geometry.upload(
        cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index64 const>(i0.data(), i0.size()),
        cuda::std::span<Index64 const>(i1.data(), i1.size()),
        cuda::std::span<Index64 const>(i2.data(), i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    EXPECT_EQ(geometry.singular_edge_count(), cpu_edges.i0.size());

    std::vector<Index64> gpu_i0(geometry.singular_edge_count());
    std::vector<Index64> gpu_i1(geometry.singular_edge_count());
    if (!gpu_i0.empty()) {
        cudaMemcpy(
            gpu_i0.data(), geometry.accessor().singular_edge_i0.data(),
            gpu_i0.size() * sizeof(Index64), cudaMemcpyDeviceToHost
        );
        cudaMemcpy(
            gpu_i1.data(), geometry.accessor().singular_edge_i1.data(),
            gpu_i1.size() * sizeof(Index64), cudaMemcpyDeviceToHost
        );
    }

    ASSERT_EQ(gpu_i0.size(), cpu_edges.i0.size());
    for (std::size_t e = 0; e < gpu_i0.size(); ++e) {
        EXPECT_EQ(gpu_i0[e], cpu_edges.i0[e]) << "edge " << e << " i0 mismatch";
        EXPECT_EQ(gpu_i1[e], cpu_edges.i1[e]) << "edge " << e << " i1 mismatch";
    }
}
