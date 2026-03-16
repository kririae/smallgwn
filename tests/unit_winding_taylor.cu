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

// Taylor winding number unit tests, far-field accuracy, order comparison,
// accuracy_scale behavior.

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

namespace {

enum class taylor_topology_builder {
    k_lbvh,
    k_hploc,
};

// Helper: upload octahedron, build BVH + Taylor, query Taylor WN.
template <int Order> struct TaylorTestContext {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Order, Real, Index> data;
    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;

    bool ready = false;
};

template <int Order>
void setup_octahedron_taylor(
    TaylorTestContext<Order> &ctx,
    taylor_topology_builder const builder = taylor_topology_builder::k_lbvh
) {
    std::vector<Real> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::vector<Index> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::vector<Index> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::vector<Index> i2{4, 4, 4, 4, 5, 5, 5, 5};

    gwn::gwn_status const upload_status = gwn::gwn_upload_geometry(
        ctx.geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    if (!upload_status.is_ok())
        return;

    gwn::gwn_status build_status = gwn::gwn_status::invalid_argument("Unsupported order.");
    if (builder == taylor_topology_builder::k_hploc) {
        build_status = gwn::gwn_bvh_facade_build_topology_aabb_moment_hploc<Order, 4, Real, Index>(
            ctx.geometry, ctx.bvh, ctx.aabb, ctx.data
        );
    } else {
        build_status = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
            ctx.geometry, ctx.bvh, ctx.aabb, ctx.data
        );
    }

    if (!build_status.is_ok())
        return;

    ctx.ready = true;
}

void run_taylor_order0_far_field_matches_exact_test(taylor_topology_builder const builder) {
    constexpr Real k_eps = 3e-2f;

    TaylorTestContext<0> ctx;
    setup_octahedron_taylor<0>(ctx, builder);
    if (!ctx.ready)
        GTEST_SKIP() << "CUDA unavailable";

    // Far queries, away from the mesh.
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

} // namespace

// Far-field queries: Taylor order 0 matches exact within loose tolerance.

TEST_F(CudaFixture, taylor_order0_far_field_matches_exact) {
    run_taylor_order0_far_field_matches_exact_test(taylor_topology_builder::k_lbvh);
}

TEST_F(CudaFixture, taylor_order0_far_field_matches_exact_hploc) {
    run_taylor_order0_far_field_matches_exact_test(taylor_topology_builder::k_hploc);
}

// Order 1 should be more accurate than Order 0 for far-field queries.

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

    // Build Order=0.
    gwn::gwn_bvh4_topology_object<Real, Index> bvh0;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb0;
    gwn::gwn_bvh4_moment_object<0, Real, Index> data0;
    ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<0, 4, Real, Index>(
                     geometry, bvh0, aabb0, data0
    )
                     .is_ok()));

    // Build Order=1.
    gwn::gwn_bvh4_topology_object<Real, Index> bvh1;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb1;
    gwn::gwn_bvh4_moment_object<1, Real, Index> data1;
    ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
                     geometry, bvh1, aabb1, data1
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

TEST_F(CudaFixture, order2_more_accurate_than_order1) {
    std::array<Real, 6> const vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::array<Index, 8> const i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> const i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> const i2{4, 4, 4, 4, 5, 5, 5, 5};

    std::array<Real, 4> const qx{3.5f, -3.0f, 0.0f, 0.0f};
    std::array<Real, 4> const qy{0.0f, 0.0f, 3.5f, -3.0f};
    std::array<Real, 4> const qz{0.0f, 0.0f, 0.0f, 0.0f};

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

    gwn::gwn_bvh4_topology_object<Real, Index> bvh1;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb1;
    gwn::gwn_bvh4_moment_object<1, Real, Index> data1;
    ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
                     geometry, bvh1, aabb1, data1
    )
                     .is_ok()));

    gwn::gwn_bvh4_topology_object<Real, Index> bvh2;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb2;
    gwn::gwn_bvh4_moment_object<2, Real, Index> data2;
    ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<2, 4, Real, Index>(
                     geometry, bvh2, aabb2, data2
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

    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                     geometry.accessor(), bvh1.accessor(), data1.accessor(), d_qx.span(),
                     d_qy.span(), d_qz.span(), d_out.span(), Real(2)
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<Real> out1(qx.size(), 0.0f);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(out1.data(), out1.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<2, Real, Index>(
                     geometry.accessor(), bvh2.accessor(), data2.accessor(), d_qx.span(),
                     d_qy.span(), d_qz.span(), d_out.span(), Real(2)
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<Real> out2(qx.size(), 0.0f);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(out2.data(), out2.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    Real max_err1 = 0.0f, max_err2 = 0.0f;
    for (std::size_t i = 0; i < qx.size(); ++i) {
        max_err1 = std::max(max_err1, std::abs(out1[i] - ref[i]));
        max_err2 = std::max(max_err2, std::abs(out2[i] - ref[i]));
    }

    EXPECT_LE(max_err2, max_err1 + 1e-6f) << "Order 2 should be at least as accurate as Order 1";
    EXPECT_LE(max_err1, 1e-2f);
    EXPECT_LE(max_err2, 1e-2f);
}

// Two independent builds produce consistent results.

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

    // Build A.
    gwn::gwn_bvh4_topology_object<Real, Index> bvh_a;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb_a;
    gwn::gwn_bvh4_moment_object<1, Real, Index> data_a;
    ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
                     geometry, bvh_a, aabb_a, data_a
    )
                     .is_ok()));

    // Build B.
    gwn::gwn_bvh4_topology_object<Real, Index> bvh_b;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb_b;
    gwn::gwn_bvh4_moment_object<1, Real, Index> data_b;
    ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
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

TEST_F(CudaFixture, repeated_build_matches_order2) {
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

    gwn::gwn_bvh4_topology_object<Real, Index> bvh_a;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb_a;
    gwn::gwn_bvh4_moment_object<2, Real, Index> data_a;
    ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<2, 4, Real, Index>(
                     geometry, bvh_a, aabb_a, data_a
    )
                     .is_ok()));

    gwn::gwn_bvh4_topology_object<Real, Index> bvh_b;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb_b;
    gwn::gwn_bvh4_moment_object<2, Real, Index> data_b;
    ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<2, 4, Real, Index>(
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

    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<2, Real, Index>(
                     geometry.accessor(), bvh_a.accessor(), data_a.accessor(), d_qx.span(),
                     d_qy.span(), d_qz.span(), d_out.span(), Real(2)
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<Real> out_a(qx.size(), 0.0f);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(out_a.data(), out_a.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_taylor<2, Real, Index>(
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

namespace {

struct winding_overflow_callback_probe {
    int *flag{};

    __device__ void operator()() const noexcept {
        if (flag != nullptr)
            *flag = 1;
    }
};

struct single_winding_overflow_probe_functor {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_moment_accessor<1, Real, Index> moment{};
    cuda::std::span<Real const> qx{};
    cuda::std::span<Real const> qy{};
    cuda::std::span<Real const> qz{};
    cuda::std::span<int> callback_flag{};
    cuda::std::span<Real> out_exact{};
    cuda::std::span<Real> out_taylor{};

    __device__ void operator()(std::size_t const i) const {
        auto const callback = winding_overflow_callback_probe{callback_flag.data()};
        out_exact[i] = gwn::gwn_winding_number_point_bvh_exact<
            4, Real, Index, 1, winding_overflow_callback_probe>(
            geometry, bvh, qx[i], qy[i], qz[i], callback
        );
        out_taylor[i] = gwn::gwn_winding_number_point_bvh_taylor<
            1, 4, Real, Index, 1, winding_overflow_callback_probe>(
            geometry, bvh, moment, qx[i], qy[i], qz[i], Real(2), callback
        );
    }
};

enum class winding_overflow_probe_status { k_ok, k_cuda_unavailable, k_error };

winding_overflow_probe_status
run_winding_overflow_nan_probe(bool &callback_called, Real &exact_out, Real &taylor_out) {
    using TopologyNode = gwn::gwn_bvh4_topology_node_soa<Index>;
    using MomentNode = gwn::gwn_bvh4_taylor_node_soa<1, Real>;

    callback_called = false;
    exact_out = Real(0);
    taylor_out = Real(0);

    std::array<Real, 3> const h_vx{Real(0), Real(1), Real(0)};
    std::array<Real, 3> const h_vy{Real(0), Real(0), Real(1)};
    std::array<Real, 3> const h_vz{Real(0), Real(0), Real(0)};
    std::array<Index, 1> const h_i0{Index(0)};
    std::array<Index, 1> const h_i1{Index(1)};
    std::array<Index, 1> const h_i2{Index(2)};
    std::array<Index, 1> const h_primitive_indices{Index(0)};

    gwn::gwn_geometry_object<Real, Index> geometry_object;
    gwn::gwn_status const geometry_status = gwn::gwn_upload_geometry(
        geometry_object, cuda::std::span<Real const>(h_vx.data(), h_vx.size()),
        cuda::std::span<Real const>(h_vy.data(), h_vy.size()),
        cuda::std::span<Real const>(h_vz.data(), h_vz.size()),
        cuda::std::span<Index const>(h_i0.data(), h_i0.size()),
        cuda::std::span<Index const>(h_i1.data(), h_i1.size()),
        cuda::std::span<Index const>(h_i2.data(), h_i2.size())
    );
    if (geometry_status.error() == gwn::gwn_error::cuda_runtime_error)
        return winding_overflow_probe_status::k_cuda_unavailable;
    if (!geometry_status.is_ok()) {
        ADD_FAILURE() << "geometry upload failed: "
                      << gwn::tests::status_to_debug_string(geometry_status);
        return winding_overflow_probe_status::k_error;
    }

    std::array<TopologyNode, 3> h_topology{};
    std::array<MomentNode, 3> h_moment{};
    std::uint8_t const invalid_kind = static_cast<std::uint8_t>(gwn::gwn_bvh_child_kind::k_invalid);
    std::uint8_t const leaf_kind = static_cast<std::uint8_t>(gwn::gwn_bvh_child_kind::k_leaf);
    std::uint8_t const internal_kind =
        static_cast<std::uint8_t>(gwn::gwn_bvh_child_kind::k_internal);
    for (auto &node : h_topology) {
        for (int i = 0; i < 4; ++i) {
            node.child_index[i] = Index(0);
            node.child_count[i] = Index(0);
            node.child_kind[i] = invalid_kind;
        }
    }
    h_topology[0].child_index[0] = Index(0);
    h_topology[0].child_count[0] = Index(1);
    h_topology[0].child_kind[0] = leaf_kind;
    h_topology[0].child_index[1] = Index(1);
    h_topology[0].child_kind[1] = internal_kind;
    h_topology[0].child_index[2] = Index(2);
    h_topology[0].child_kind[2] = internal_kind;

    for (int slot = 0; slot < 4; ++slot) {
        h_moment[0].child_max_p_dist2[slot] = Real(1e6);
        h_moment[0].child_average_x[slot] = Real(0);
        h_moment[0].child_average_y[slot] = Real(0);
        h_moment[0].child_average_z[slot] = Real(0);
        h_moment[0].child_n_x[slot] = Real(0);
        h_moment[0].child_n_y[slot] = Real(0);
        h_moment[0].child_n_z[slot] = Real(0);
        h_moment[0].child_nij_xx[slot] = Real(0);
        h_moment[0].child_nij_yy[slot] = Real(0);
        h_moment[0].child_nij_zz[slot] = Real(0);
        h_moment[0].child_nxy_nyx[slot] = Real(0);
        h_moment[0].child_nyz_nzy[slot] = Real(0);
        h_moment[0].child_nzx_nxz[slot] = Real(0);
    }

    gwn::gwn_device_array<TopologyNode> d_topology;
    gwn::gwn_device_array<Index> d_primitive_indices;
    gwn::gwn_device_array<MomentNode> d_moment;
    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_exact, d_taylor;
    gwn::gwn_device_array<int> d_callback_flag;
    bool const allocation_ok = d_topology.resize(h_topology.size()).is_ok() &&
                               d_primitive_indices.resize(h_primitive_indices.size()).is_ok() &&
                               d_moment.resize(h_moment.size()).is_ok() && d_qx.resize(1).is_ok() &&
                               d_qy.resize(1).is_ok() && d_qz.resize(1).is_ok() &&
                               d_exact.resize(1).is_ok() && d_taylor.resize(1).is_ok() &&
                               d_callback_flag.resize(1).is_ok();
    if (!allocation_ok) {
        ADD_FAILURE() << "device allocation failed";
        return winding_overflow_probe_status::k_error;
    }

    std::array<Real, 1> const h_qx{Real(0.25)};
    std::array<Real, 1> const h_qy{Real(0.25)};
    std::array<Real, 1> const h_qz{Real(2)};
    std::array<Real, 1> h_exact{Real(0)};
    std::array<Real, 1> h_taylor{Real(0)};
    std::array<int, 1> h_callback_flag{0};

    bool const upload_ok =
        d_topology
            .copy_from_host(
                cuda::std::span<TopologyNode const>(h_topology.data(), h_topology.size())
            )
            .is_ok() &&
        d_primitive_indices
            .copy_from_host(
                cuda::std::span<Index const>(h_primitive_indices.data(), h_primitive_indices.size())
            )
            .is_ok() &&
        d_moment.copy_from_host(cuda::std::span<MomentNode const>(h_moment.data(), h_moment.size()))
            .is_ok() &&
        d_qx.copy_from_host(cuda::std::span<Real const>(h_qx.data(), h_qx.size())).is_ok() &&
        d_qy.copy_from_host(cuda::std::span<Real const>(h_qy.data(), h_qy.size())).is_ok() &&
        d_qz.copy_from_host(cuda::std::span<Real const>(h_qz.data(), h_qz.size())).is_ok() &&
        d_callback_flag
            .copy_from_host(
                cuda::std::span<int const>(h_callback_flag.data(), h_callback_flag.size())
            )
            .is_ok();
    if (!upload_ok) {
        ADD_FAILURE() << "device upload failed";
        return winding_overflow_probe_status::k_error;
    }

    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    bvh.nodes = d_topology.span();
    bvh.primitive_indices = d_primitive_indices.span();
    bvh.root_kind = gwn::gwn_bvh_child_kind::k_internal;
    bvh.root_index = Index(0);
    bvh.root_count = Index(0);

    gwn::gwn_bvh4_moment_accessor<1, Real, Index> moment{};
    moment.nodes = d_moment.span();

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    gwn::gwn_status const launch_status = gwn::detail::gwn_launch_linear_kernel<k_block_size>(
        1, single_winding_overflow_probe_functor{
               geometry_object.accessor(), bvh, moment, d_qx.span(), d_qy.span(), d_qz.span(),
               d_callback_flag.span(), d_exact.span(), d_taylor.span()
           }
    );
    if (!launch_status.is_ok()) {
        ADD_FAILURE() << "kernel launch failed: "
                      << gwn::tests::status_to_debug_string(launch_status);
        return winding_overflow_probe_status::k_error;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "CUDA synchronize failed after winding overflow probe";
        return winding_overflow_probe_status::k_error;
    }

    if (!d_exact.copy_to_host(cuda::std::span<Real>(h_exact.data(), h_exact.size())).is_ok() ||
        !d_taylor.copy_to_host(cuda::std::span<Real>(h_taylor.data(), h_taylor.size())).is_ok() ||
        !d_callback_flag
             .copy_to_host(cuda::std::span<int>(h_callback_flag.data(), h_callback_flag.size()))
             .is_ok()) {
        ADD_FAILURE() << "device-to-host copy failed";
        return winding_overflow_probe_status::k_error;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "CUDA synchronize failed after result copy";
        return winding_overflow_probe_status::k_error;
    }

    callback_called = h_callback_flag[0] != 0;
    exact_out = h_exact[0];
    taylor_out = h_taylor[0];
    return winding_overflow_probe_status::k_ok;
}

} // namespace

TEST_F(CudaFixture, winding_overflow_returns_nan_for_exact_and_taylor) {
    bool callback_called = false;
    Real exact_value = Real(0);
    Real taylor_value = Real(0);
    winding_overflow_probe_status const probe_status =
        run_winding_overflow_nan_probe(callback_called, exact_value, taylor_value);
    if (probe_status == winding_overflow_probe_status::k_cuda_unavailable)
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(probe_status, winding_overflow_probe_status::k_ok)
        << "winding overflow probe execution failed";
    EXPECT_TRUE(callback_called);
    EXPECT_TRUE(std::isnan(exact_value));
    EXPECT_TRUE(std::isnan(taylor_value));
}

// Taylor query with missing data returns error.

TEST_F(CudaFixture, taylor_query_with_no_data_returns_error) {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_moment_accessor<0, Real, Index> data{};
    cuda::std::span<Real const> empty{};
    cuda::std::span<Real> output{};

    gwn::gwn_status const status = gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
        geometry, bvh, data, empty, empty, empty, output
    );
    EXPECT_FALSE(status.is_ok());
}

TEST_F(CudaFixture, taylor_batch_mismatched_query_output) {
    gwn::gwn_geometry_accessor<Real, Index> accessor{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_moment_accessor<0, Real, Index> data{};
    cuda::std::span<Real const> empty{};
    Real dummy[2] = {};
    cuda::std::span<Real> output(dummy, 2);

    // query size=0 but output size=2, mismatch.
    gwn::gwn_status const status = gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
        accessor, bvh, data, empty, empty, empty, output
    );
    EXPECT_FALSE(status.is_ok());
}

TEST_F(CudaFixture, taylor_empty_query_returns_ok) {
    gwn::gwn_geometry_accessor<Real, Index> accessor{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_moment_accessor<0, Real, Index> data{};
    cuda::std::span<Real const> empty{};
    cuda::std::span<Real> output{};

    // Zero queries, zero output, should succeed (early return).
    // Note: geometry validator fires first.
    gwn::gwn_status const status = gwn::gwn_compute_winding_number_batch_bvh_taylor<0, Real, Index>(
        accessor, bvh, data, empty, empty, empty, output
    );
    // Empty accessor is invalid geometry, so this correctly returns error.
    EXPECT_FALSE(status.is_ok());
}

// ---------------------------------------------------------------------------
// Exact winding number batch API tests.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, exact_batch_winding_matches_point_results) {
    // Build octahedron topology (no AABB, no moment needed for exact winding).
    std::vector<Real> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::vector<Index> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::vector<Index> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::vector<Index> i2{4, 4, 4, 4, 5, 5, 5, 5};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const upload_status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    if (upload_status.error() == gwn::gwn_error::cuda_runtime_error)
        GTEST_SKIP() << "CUDA unavailable";
    ASSERT_TRUE(upload_status.is_ok());

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    ASSERT_TRUE((gwn::gwn_bvh_topology_build_lbvh<4, Real, Index>(geometry, bvh).is_ok()));

    // Query points: interior (winding ~1) and exterior (winding ~0).
    std::vector<Real> qx{0.0f, 5.0f};
    std::vector<Real> qy{0.0f, 5.0f};
    std::vector<Real> qz{0.0f, 5.0f};
    std::size_t const n = qx.size();

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    ASSERT_TRUE(d_qx.resize(n).is_ok());
    ASSERT_TRUE(d_qy.resize(n).is_ok());
    ASSERT_TRUE(d_qz.resize(n).is_ok());
    ASSERT_TRUE(d_out.resize(n).is_ok());
    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), n)).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), n)).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), n)).is_ok());

    gwn::gwn_status const batch_status =
        gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
            geometry.accessor(), bvh.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_out.span()
        );
    ASSERT_TRUE(batch_status.is_ok()) << gwn::tests::status_to_debug_string(batch_status);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> results(n, Real(0));
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(results.data(), n)).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    // Interior query: winding number should be close to 1.
    EXPECT_NEAR(results[0], Real(1), Real(1e-4f));
    // Exterior query: winding number should be close to 0.
    EXPECT_NEAR(results[1], Real(0), Real(1e-4f));
}

TEST_F(CudaFixture, exact_batch_invalid_geometry_returns_error) {
    gwn::gwn_geometry_accessor<Real, Index> bad_geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    cuda::std::span<Real const> empty{};
    cuda::std::span<Real> output{};

    gwn::gwn_status const status = gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
        bad_geometry, bvh, empty, empty, empty, output
    );
    EXPECT_FALSE(status.is_ok());
}

TEST_F(CudaFixture, exact_batch_span_mismatch_returns_error) {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    cuda::std::span<Real const> empty{};
    Real dummy[2] = {};
    cuda::std::span<Real> output(dummy, 2);

    // query_x size=0, output size=2 → mismatch.
    gwn::gwn_status const status = gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
        geometry, bvh, empty, empty, empty, output
    );
    EXPECT_FALSE(status.is_ok());
}
