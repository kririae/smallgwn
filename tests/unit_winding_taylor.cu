#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "reference_cpu.cuh"
#include "test_fixtures.hpp"
#include "test_utils.hpp"

// ---------------------------------------------------------------------------
// Taylor winding number unit tests — far-field accuracy, order comparison,
// accuracy_scale behavior.
// ---------------------------------------------------------------------------

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

namespace {

// Helper: upload octahedron, build BVH + Taylor, query Taylor WN.
struct TaylorTestContext {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh_object<Real, Index> bvh;
    gwn::gwn_bvh_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh_moment_object<Real, Index> data;
    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;

    bool ready = false;
};

void setup_octahedron_taylor(TaylorTestContext &ctx, int order) {
    std::vector<Real> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::vector<Index> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::vector<Index> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::vector<Index> i2{4, 4, 4, 4, 5, 5, 5, 5};

    gwn::gwn_status const upload_status = ctx.geometry.upload(
        cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    if (!upload_status.is_ok())
        return;

    gwn::gwn_status build_status;
    if (order == 0)
        build_status = gwn::gwn_build_bvh4_topology_aabb_moment_lbvh<0, Real, Index>(
            ctx.geometry, ctx.bvh, ctx.aabb, ctx.data
        );
    else
        build_status = gwn::gwn_build_bvh4_topology_aabb_moment_lbvh<1, Real, Index>(
            ctx.geometry, ctx.bvh, ctx.aabb, ctx.data
        );

    if (!build_status.is_ok())
        return;

    ctx.ready = true;
}

} // namespace

// ---------------------------------------------------------------------------
// Far-field queries: Taylor order 0 matches exact within loose tolerance.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, taylor_order0_far_field_matches_exact) {
    constexpr Real k_eps = 3e-2f;

    TaylorTestContext ctx;
    setup_octahedron_taylor(ctx, 0);
    if (!ctx.ready)
        GTEST_SKIP() << "CUDA unavailable";

    // Far queries — away from the mesh.
    std::array<Real, 4> const qx{3.5f, -3.0f, 0.0f, 0.0f};
    std::array<Real, 4> const qy{0.0f, 0.0f, 3.5f, -3.0f};
    std::array<Real, 4> const qz{0.0f, 0.0f, 0.0f, 0.0f};

    // CPU exact reference.
    std::array<Real, 6> const vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::array<Index, 8> const i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> const i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> const i2{4, 4, 4, 4, 5, 5, 5, 5};

    std::vector<Real> ref(qx.size(), 0.0f);
    ASSERT_TRUE((gwn::tests::reference_winding_number_batch<Real, Index>(
                     std::span<Real const>(vx), std::span<Real const>(vy),
                     std::span<Real const>(vz), std::span<Index const>(i0),
                     std::span<Index const>(i1), std::span<Index const>(i2),
                     std::span<Real const>(qx), std::span<Real const>(qy),
                     std::span<Real const>(qz), std::span<Real>(ref)
    )
                     .is_ok()));

    ASSERT_TRUE(ctx.d_qx.resize(qx.size()).is_ok());
    ASSERT_TRUE(ctx.d_qy.resize(qy.size()).is_ok());
    ASSERT_TRUE(ctx.d_qz.resize(qz.size()).is_ok());
    ASSERT_TRUE(ctx.d_out.resize(qx.size()).is_ok());
    ASSERT_TRUE(ctx.d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(ctx.d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(ctx.d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());

    gwn::gwn_status const query_status =
        gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
            ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.data.accessor(), ctx.d_qx.span(),
            ctx.d_qy.span(), ctx.d_qz.span(), ctx.d_out.span(), Real(2)
        );
    ASSERT_TRUE(query_status.is_ok()) << gwn::tests::status_to_debug_string(query_status);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> gpu(qx.size(), 0.0f);
    ASSERT_TRUE(ctx.d_out.copy_to_host(cuda::std::span<Real>(gpu.data(), gpu.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    for (std::size_t i = 0; i < qx.size(); ++i)
        EXPECT_NEAR(gpu[i], ref[i], k_eps) << "query " << i;
}

// ---------------------------------------------------------------------------
// Order 1 should be more accurate than Order 0 for far-field queries.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, order1_more_accurate_than_order0) {
    std::array<Real, 6> const vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::array<Index, 8> const i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> const i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> const i2{4, 4, 4, 4, 5, 5, 5, 5};

    std::array<Real, 4> const qx{3.5f, -3.0f, 0.0f, 0.0f};
    std::array<Real, 4> const qy{0.0f, 0.0f, 3.5f, -3.0f};
    std::array<Real, 4> const qz{0.0f, 0.0f, 0.0f, 0.0f};

    // CPU reference.
    std::vector<Real> ref(qx.size(), 0.0f);
    ASSERT_TRUE((gwn::tests::reference_winding_number_batch<Real, Index>(
                     std::span<Real const>(vx), std::span<Real const>(vy),
                     std::span<Real const>(vz), std::span<Index const>(i0),
                     std::span<Index const>(i1), std::span<Index const>(i2),
                     std::span<Real const>(qx), std::span<Real const>(qy),
                     std::span<Real const>(qz), std::span<Real>(ref)
    )
                     .is_ok()));

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

    // Build Order=0.
    gwn::gwn_bvh_object<Real, Index> bvh0;
    gwn::gwn_bvh_aabb_object<Real, Index> aabb0;
    gwn::gwn_bvh_moment_object<Real, Index> data0;
    ASSERT_TRUE(
        (gwn::gwn_build_bvh4_topology_aabb_moment_lbvh<0, Real, Index>(geometry, bvh0, aabb0, data0)
             .is_ok())
    );

    // Build Order=1.
    gwn::gwn_bvh_object<Real, Index> bvh1;
    gwn::gwn_bvh_aabb_object<Real, Index> aabb1;
    gwn::gwn_bvh_moment_object<Real, Index> data1;
    ASSERT_TRUE(
        (gwn::gwn_build_bvh4_topology_aabb_moment_lbvh<1, Real, Index>(geometry, bvh1, aabb1, data1)
             .is_ok())
    );

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    ASSERT_TRUE(d_qx.resize(qx.size()).is_ok());
    ASSERT_TRUE(d_qy.resize(qy.size()).is_ok());
    ASSERT_TRUE(d_qz.resize(qz.size()).is_ok());
    ASSERT_TRUE(d_out.resize(qx.size()).is_ok());
    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());

    // Query Order 0.
    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
                     geometry.accessor(), bvh0.accessor(), data0.accessor(), d_qx.span(),
                     d_qy.span(), d_qz.span(), d_out.span(), Real(2)
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<Real> out0(qx.size(), 0.0f);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(out0.data(), out0.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    // Query Order 1.
    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                     geometry.accessor(), bvh1.accessor(), data1.accessor(), d_qx.span(),
                     d_qy.span(), d_qz.span(), d_out.span(), Real(2)
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<Real> out1(qx.size(), 0.0f);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(out1.data(), out1.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    // Order 1 max error should be ≤ Order 0 max error.
    Real max_err0 = 0.0f, max_err1 = 0.0f;
    for (std::size_t i = 0; i < qx.size(); ++i) {
        max_err0 = std::max(max_err0, std::abs(out0[i] - ref[i]));
        max_err1 = std::max(max_err1, std::abs(out1[i] - ref[i]));
    }
    EXPECT_LE(max_err1, max_err0 + 1e-6f) << "Order 1 should be at least as accurate as Order 0";

    // Both should have bounded error.
    EXPECT_LE(max_err0, 3e-2f);
    EXPECT_LE(max_err1, 1e-2f);
}

// ---------------------------------------------------------------------------
// Two independent builds produce consistent results.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, repeated_build_matches_order1) {
    constexpr Real k_eps = 3e-4f;

    std::array<Real, 6> const vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::array<Index, 8> const i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> const i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> const i2{4, 4, 4, 4, 5, 5, 5, 5};

    std::array<Real, 4> const qx{3.5f, -3.0f, 0.0f, 0.0f};
    std::array<Real, 4> const qy{0.0f, 0.0f, 3.5f, -3.0f};
    std::array<Real, 4> const qz{0.0f, 0.0f, 0.0f, 0.0f};

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

    // Build A.
    gwn::gwn_bvh_object<Real, Index> bvh_a;
    gwn::gwn_bvh_aabb_object<Real, Index> aabb_a;
    gwn::gwn_bvh_moment_object<Real, Index> data_a;
    ASSERT_TRUE((gwn::gwn_build_bvh4_topology_aabb_moment_lbvh<1, Real, Index>(
                     geometry, bvh_a, aabb_a, data_a
    )
                     .is_ok()));

    // Build B.
    gwn::gwn_bvh_object<Real, Index> bvh_b;
    gwn::gwn_bvh_aabb_object<Real, Index> aabb_b;
    gwn::gwn_bvh_moment_object<Real, Index> data_b;
    ASSERT_TRUE((gwn::gwn_build_bvh4_topology_aabb_moment_lbvh<1, Real, Index>(
                     geometry, bvh_b, aabb_b, data_b
    )
                     .is_ok()));

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    ASSERT_TRUE(d_qx.resize(qx.size()).is_ok());
    ASSERT_TRUE(d_qy.resize(qy.size()).is_ok());
    ASSERT_TRUE(d_qz.resize(qz.size()).is_ok());
    ASSERT_TRUE(d_out.resize(qx.size()).is_ok());
    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());

    // Query A.
    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                     geometry.accessor(), bvh_a.accessor(), data_a.accessor(), d_qx.span(),
                     d_qy.span(), d_qz.span(), d_out.span(), Real(2)
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<Real> out_a(qx.size(), 0.0f);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(out_a.data(), out_a.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    // Query B.
    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                     geometry.accessor(), bvh_b.accessor(), data_b.accessor(), d_qx.span(),
                     d_qy.span(), d_qz.span(), d_out.span(), Real(2)
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<Real> out_b(qx.size(), 0.0f);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(out_b.data(), out_b.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    for (std::size_t i = 0; i < qx.size(); ++i)
        EXPECT_NEAR(out_a[i], out_b[i], k_eps) << "query " << i;
}

// ---------------------------------------------------------------------------
// Taylor query with missing data returns error.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, taylor_query_with_no_data_returns_error) {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh_accessor<Real, Index> bvh{};
    gwn::gwn_bvh_moment4_accessor<Real, Index> data{};
    cuda::std::span<Real const> empty{};
    cuda::std::span<Real> output{};

    gwn::gwn_status const status = gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
        geometry, bvh, data, empty, empty, empty, output
    );
    EXPECT_FALSE(status.is_ok());
}
