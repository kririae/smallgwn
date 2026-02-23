#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_utils.hpp"

// ---------------------------------------------------------------------------
// BVH topology build unit tests — structure invariants, single/small meshes.
// ---------------------------------------------------------------------------

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

namespace {

// Uploads a small mesh and returns geometry object. Returns nullopt on
// CUDA unavailability.
std::optional<gwn::gwn_geometry_object<Real, Index>> upload_mesh(
    std::vector<Real> const &vx, std::vector<Real> const &vy, std::vector<Real> const &vz,
    std::vector<Index> const &i0, std::vector<Index> const &i1, std::vector<Index> const &i2
) {
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

// Verify BVH structure: all primitives referenced exactly once, bounds valid.
template <int Width>
void verify_bvh_structure(
    gwn::gwn_bvh_topology_accessor<Width, Real, Index> const &accessor,
    std::size_t const primitive_count
) {
    ASSERT_EQ(accessor.primitive_indices.size(), primitive_count);

    if (primitive_count == 0)
        return;

    std::vector<Index> prim(primitive_count);
    ASSERT_EQ(
        cudaSuccess, cudaMemcpy(
                         prim.data(), accessor.primitive_indices.data(),
                         primitive_count * sizeof(Index), cudaMemcpyDeviceToHost
                     )
    );

    std::vector<int> seen(primitive_count, 0);
    for (Index const idx : prim) {
        ASSERT_GE(idx, Index(0));
        ASSERT_LT(static_cast<std::size_t>(idx), primitive_count);
        ++seen[static_cast<std::size_t>(idx)];
    }
    for (int const s : seen)
        EXPECT_EQ(s, 1);
}

} // namespace

// ---------------------------------------------------------------------------
// Single triangle produces a leaf BVH.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, single_triangle_bvh) {
    std::vector<Real> vx{1.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 1.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 1.0f};
    std::vector<Index> i0{0}, i1{1}, i2{2};

    auto maybe_geo = upload_mesh(vx, vy, vz, i0, i1, i2);
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh_object<Real, Index> bvh;
    gwn::gwn_status const build_status =
        gwn::gwn_build_bvh_topology_lbvh<4, Real, Index>(geometry, bvh);
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);
    ASSERT_TRUE(bvh.has_bvh());

    auto const &acc = bvh.accessor();
    EXPECT_TRUE(acc.is_valid());
    EXPECT_EQ(acc.primitive_indices.size(), 1u);
    verify_bvh_structure<4>(acc, 1);
}

// ---------------------------------------------------------------------------
// Two-triangle mesh.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, two_triangle_bvh) {
    std::vector<Real> vx{1.0f, 0.0f, 0.0f, -1.0f};
    std::vector<Real> vy{0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 1.0f, 0.0f};
    std::vector<Index> i0{0, 0}, i1{1, 2}, i2{2, 3};

    auto maybe_geo = upload_mesh(vx, vy, vz, i0, i1, i2);
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh_object<Real, Index> bvh;
    gwn::gwn_status const build_status =
        gwn::gwn_build_bvh_topology_lbvh<4, Real, Index>(geometry, bvh);
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);
    ASSERT_TRUE(bvh.has_bvh());

    verify_bvh_structure<4>(bvh.accessor(), 2);
}

// ---------------------------------------------------------------------------
// Octahedron (8 triangles) — ensures internal nodes.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, octahedron_8_triangles) {
    std::vector<Real> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::vector<Index> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::vector<Index> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::vector<Index> i2{4, 4, 4, 4, 5, 5, 5, 5};

    auto maybe_geo = upload_mesh(vx, vy, vz, i0, i1, i2);
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh_object<Real, Index> bvh;
    gwn::gwn_status const build_status =
        gwn::gwn_build_bvh_topology_lbvh<4, Real, Index>(geometry, bvh);
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);
    ASSERT_TRUE(bvh.has_bvh());

    auto const &acc = bvh.accessor();
    EXPECT_TRUE(acc.is_valid());
    EXPECT_EQ(acc.root_kind, gwn::gwn_bvh_child_kind::k_internal);
    verify_bvh_structure<4>(acc, 8);
}

// ---------------------------------------------------------------------------
// Width=2 and Width=8 builds.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, binary_bvh_build) {
    std::vector<Real> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::vector<Index> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::vector<Index> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::vector<Index> i2{4, 4, 4, 4, 5, 5, 5, 5};

    auto maybe_geo = upload_mesh(vx, vy, vz, i0, i1, i2);
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh_topology_object<2, Real, Index> bvh2;
    gwn::gwn_status const status = gwn::gwn_build_bvh_topology_lbvh<2, Real, Index>(geometry, bvh2);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_TRUE(bvh2.has_bvh());
    verify_bvh_structure<2>(bvh2.accessor(), 8);
}

TEST_F(CudaFixture, wide8_bvh_build) {
    std::vector<Real> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::vector<Index> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::vector<Index> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::vector<Index> i2{4, 4, 4, 4, 5, 5, 5, 5};

    auto maybe_geo = upload_mesh(vx, vy, vz, i0, i1, i2);
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh_topology_object<8, Real, Index> bvh8;
    gwn::gwn_status const status = gwn::gwn_build_bvh_topology_lbvh<8, Real, Index>(geometry, bvh8);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_TRUE(bvh8.has_bvh());
    verify_bvh_structure<8>(bvh8.accessor(), 8);
}

// ---------------------------------------------------------------------------
// Degenerate: all-coplanar triangles (zero-thickness BVH).
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, coplanar_triangles_build_succeeds) {
    // Two triangles in the xy-plane (z=0).
    std::vector<Real> vx{0.0f, 1.0f, 0.0f, 1.0f};
    std::vector<Real> vy{0.0f, 0.0f, 1.0f, 1.0f};
    std::vector<Real> vz{0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Index> i0{0, 1}, i1{1, 2}, i2{2, 3};

    auto maybe_geo = upload_mesh(vx, vy, vz, i0, i1, i2);
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh_object<Real, Index> bvh;
    gwn::gwn_status const status = gwn::gwn_build_bvh_topology_lbvh<4, Real, Index>(geometry, bvh);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_TRUE(bvh.has_bvh());
    verify_bvh_structure<4>(bvh.accessor(), 2);
}

// ---------------------------------------------------------------------------
// Degenerate: zero-area triangle (two identical vertices).
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, zero_area_triangle_build_succeeds) {
    std::vector<Real> vx{0.0f, 0.0f, 1.0f};
    std::vector<Real> vy{0.0f, 0.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 0.0f};
    std::vector<Index> i0{0}, i1{1}, i2{2};

    auto maybe_geo = upload_mesh(vx, vy, vz, i0, i1, i2);
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh_object<Real, Index> bvh;
    gwn::gwn_status const status = gwn::gwn_build_bvh_topology_lbvh<4, Real, Index>(geometry, bvh);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_TRUE(bvh.has_bvh());
}

// ---------------------------------------------------------------------------
// Rebuild replaces previous BVH.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, rebuild_replaces_previous_bvh) {
    std::vector<Real> vx{1.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 1.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 1.0f};
    std::vector<Index> i0{0}, i1{1}, i2{2};

    auto maybe_geo = upload_mesh(vx, vy, vz, i0, i1, i2);
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh_object<Real, Index> bvh;
    ASSERT_TRUE((gwn::gwn_build_bvh_topology_lbvh<4, Real, Index>(geometry, bvh).is_ok()));
    ASSERT_TRUE(bvh.has_bvh());

    // Rebuild (should succeed overwriting previous).
    ASSERT_TRUE((gwn::gwn_build_bvh_topology_lbvh<4, Real, Index>(geometry, bvh).is_ok()));
    ASSERT_TRUE(bvh.has_bvh());
    verify_bvh_structure<4>(bvh.accessor(), 1);
}

// ---------------------------------------------------------------------------
// Clear resets BVH.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, clear_resets_bvh) {
    std::vector<Real> vx{1.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 1.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 1.0f};
    std::vector<Index> i0{0}, i1{1}, i2{2};

    auto maybe_geo = upload_mesh(vx, vy, vz, i0, i1, i2);
    if (!maybe_geo.has_value())
        GTEST_SKIP() << "CUDA unavailable";
    auto &geometry = *maybe_geo;

    gwn::gwn_bvh_object<Real, Index> bvh;
    ASSERT_TRUE((gwn::gwn_build_bvh_topology_lbvh<4, Real, Index>(geometry, bvh).is_ok()));
    ASSERT_TRUE(bvh.has_bvh());

    bvh.clear();
    EXPECT_FALSE(bvh.has_bvh());
}
