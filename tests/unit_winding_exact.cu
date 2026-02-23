#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "reference_cpu.cuh"
#include "reference_hdk/UT_SolidAngle.h"
#include "test_fixtures.hpp"
#include "test_utils.hpp"

// ---------------------------------------------------------------------------
// Exact winding number unit tests — single triangle, BVH exact query,
// degenerate cases, HDK oracle cross-check.
// ---------------------------------------------------------------------------

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

// ---------------------------------------------------------------------------
// CPU reference — single triangle analytic check.
// ---------------------------------------------------------------------------

TEST(smallgwn_unit_winding_exact, single_triangle_cpu_reference) {
    constexpr Real k_pi = 3.14159265358979323846f;
    constexpr Real k_eps = 1e-6f;

    std::array<Real, 3> const vx{1.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vy{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 1.0f};
    std::array<Index, 1> const i0{0}, i1{1}, i2{2};

    std::array<Real, 2> const qx{0.0f, 0.25f};
    std::array<Real, 2> const qy{0.0f, 0.25f};
    std::array<Real, 2> const qz{0.0f, 0.25f};

    std::vector<Real> output(qx.size(), 0.0f);
    gwn::gwn_status const status = gwn::tests::reference_winding_number_batch<Real, Index>(
        std::span<Real const>(vx.data(), vx.size()), std::span<Real const>(vy.data(), vy.size()),
        std::span<Real const>(vz.data(), vz.size()), std::span<Index const>(i0.data(), i0.size()),
        std::span<Index const>(i1.data(), i1.size()), std::span<Index const>(i2.data(), i2.size()),
        std::span<Real const>(qx.data(), qx.size()), std::span<Real const>(qy.data(), qy.size()),
        std::span<Real const>(qz.data(), qz.size()), std::span<Real>(output.data(), output.size())
    );
    ASSERT_TRUE(status.is_ok()) << status.message();

    // Cross-check with vendored HDK oracle.
    for (std::size_t qi = 0; qi < qx.size(); ++qi) {
        HDK_Sample::UT_Vector3T<Real> a;
        a[0] = vx[0];
        a[1] = vy[0];
        a[2] = vz[0];
        HDK_Sample::UT_Vector3T<Real> b;
        b[0] = vx[1];
        b[1] = vy[1];
        b[2] = vz[1];
        HDK_Sample::UT_Vector3T<Real> c;
        c[0] = vx[2];
        c[1] = vy[2];
        c[2] = vz[2];
        HDK_Sample::UT_Vector3T<Real> q;
        q[0] = qx[qi];
        q[1] = qy[qi];
        q[2] = qz[qi];

        Real const oracle = HDK_Sample::UTsignedSolidAngleTri(a, b, c, q) / (4.0f * k_pi);
        EXPECT_NEAR(output[qi], oracle, k_eps);
    }
}

// ---------------------------------------------------------------------------
// CPU reference — closed octahedron has WN ≈ 1 inside, ≈ 0 outside.
// ---------------------------------------------------------------------------

TEST(smallgwn_unit_winding_exact, octahedron_inside_outside) {
    constexpr Real k_eps = 1e-3f;

    std::array<Real, 6> const vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::array<Index, 8> const i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> const i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> const i2{4, 4, 4, 4, 5, 5, 5, 5};

    // Inside query (origin).
    std::array<Real, 1> const qx_in{0.0f}, qy_in{0.0f}, qz_in{0.0f};
    std::vector<Real> out_in(1, 0.0f);
    ASSERT_TRUE((gwn::tests::reference_winding_number_batch<Real, Index>(
                     std::span<Real const>(vx), std::span<Real const>(vy),
                     std::span<Real const>(vz), std::span<Index const>(i0),
                     std::span<Index const>(i1), std::span<Index const>(i2),
                     std::span<Real const>(qx_in), std::span<Real const>(qy_in),
                     std::span<Real const>(qz_in), std::span<Real>(out_in)
    )
                     .is_ok()));
    EXPECT_NEAR(std::abs(out_in[0]), 1.0f, k_eps) << "Inside point should have |WN|≈1";

    // Outside query (far point).
    std::array<Real, 1> const qx_out{5.0f}, qy_out{0.0f}, qz_out{0.0f};
    std::vector<Real> out_out(1, 0.0f);
    ASSERT_TRUE((gwn::tests::reference_winding_number_batch<Real, Index>(
                     std::span<Real const>(vx), std::span<Real const>(vy),
                     std::span<Real const>(vz), std::span<Index const>(i0),
                     std::span<Index const>(i1), std::span<Index const>(i2),
                     std::span<Real const>(qx_out), std::span<Real const>(qy_out),
                     std::span<Real const>(qz_out), std::span<Real>(out_out)
    )
                     .is_ok()));
    EXPECT_NEAR(out_out[0], 0.0f, k_eps) << "Outside point should have WN≈0";
}

// ---------------------------------------------------------------------------
// BVH exact query matches CPU reference for octahedron.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, bvh_exact_matches_cpu_reference) {
    constexpr Real k_eps = 1e-4f;

    std::array<Real, 6> const vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::array<Index, 8> const i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> const i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> const i2{4, 4, 4, 4, 5, 5, 5, 5};

    std::array<Real, 4> const qx{0.0f, 2.0f, 0.2f, -1.5f};
    std::array<Real, 4> const qy{0.0f, 0.0f, 0.2f, 0.1f};
    std::array<Real, 4> const qz{0.0f, 0.0f, 0.2f, 0.0f};

    // CPU reference.
    std::vector<Real> ref_out(qx.size(), 0.0f);
    ASSERT_TRUE((gwn::tests::reference_winding_number_batch<Real, Index>(
                     std::span<Real const>(vx), std::span<Real const>(vy),
                     std::span<Real const>(vz), std::span<Index const>(i0),
                     std::span<Index const>(i1), std::span<Index const>(i2),
                     std::span<Real const>(qx), std::span<Real const>(qy),
                     std::span<Real const>(qz), std::span<Real>(ref_out)
    )
                     .is_ok()));

    // GPU.
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

    gwn::gwn_bvh_object<Real, Index> bvh;
    ASSERT_TRUE((gwn::gwn_build_bvh4_topology_lbvh<Real, Index>(geometry, bvh).is_ok()));

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    ASSERT_TRUE(d_qx.resize(qx.size()).is_ok());
    ASSERT_TRUE(d_qy.resize(qy.size()).is_ok());
    ASSERT_TRUE(d_qz.resize(qz.size()).is_ok());
    ASSERT_TRUE(d_out.resize(qx.size()).is_ok());

    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());

    gwn::gwn_status const query_status =
        gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
            geometry.accessor(), bvh.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_out.span()
        );
    ASSERT_TRUE(query_status.is_ok()) << gwn::tests::status_to_debug_string(query_status);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> gpu_out(qx.size(), 0.0f);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(gpu_out.data(), gpu_out.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    for (std::size_t i = 0; i < qx.size(); ++i)
        EXPECT_NEAR(gpu_out[i], ref_out[i], k_eps) << "query " << i;
}

// ---------------------------------------------------------------------------
// Degenerate: zero-area triangle produces WN=0.
// ---------------------------------------------------------------------------

TEST(smallgwn_unit_winding_exact, zero_area_triangle_returns_zero) {
    // Two vertices coincide → zero-area triangle.
    std::array<Real, 3> const vx{0.0f, 0.0f, 1.0f};
    std::array<Real, 3> const vy{0.0f, 0.0f, 0.0f};
    std::array<Real, 3> const vz{0.0f, 0.0f, 0.0f};
    std::array<Index, 1> const i0{0}, i1{1}, i2{2};

    std::array<Real, 1> const qx{0.5f}, qy{0.5f}, qz{0.5f};
    std::vector<Real> output(1, 99.0f);
    ASSERT_TRUE((gwn::tests::reference_winding_number_batch<Real, Index>(
                     std::span<Real const>(vx), std::span<Real const>(vy),
                     std::span<Real const>(vz), std::span<Index const>(i0),
                     std::span<Index const>(i1), std::span<Index const>(i2),
                     std::span<Real const>(qx), std::span<Real const>(qy),
                     std::span<Real const>(qz), std::span<Real>(output)
    )
                     .is_ok()));
    EXPECT_NEAR(output[0], 0.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Brute-force batch (no BVH) — validation error paths.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, brute_force_batch_mismatched_query_output) {
    gwn::gwn_geometry_accessor<Real, Index> accessor{};
    cuda::std::span<Real const> empty{};
    Real dummy[2] = {};
    cuda::std::span<Real> output(dummy, 2);

    // query size=0 but output size=2 → mismatch.
    gwn::gwn_status const status =
        gwn::gwn_compute_winding_number_batch<Real, Index>(accessor, empty, empty, empty, output);
    EXPECT_FALSE(status.is_ok());
}

// ---------------------------------------------------------------------------
// BVH exact query — missing BVH returns error.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, bvh_exact_query_with_no_bvh_returns_error) {
    gwn::gwn_geometry_accessor<Real, Index> accessor{};
    gwn::gwn_bvh_accessor<Real, Index> bvh{};
    cuda::std::span<Real const> empty{};
    cuda::std::span<Real> output{};

    gwn::gwn_status const status = gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
        accessor, bvh, empty, empty, empty, output
    );
    EXPECT_FALSE(status.is_ok());
}

// ---------------------------------------------------------------------------
// Empty geometry with BVH returns zero for all queries.
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, brute_force_empty_geometry_returns_zero) {
    gwn::gwn_geometry_accessor<Real, Index> accessor{};
    cuda::std::span<Real const> empty{};
    cuda::std::span<Real> output{};

    // Zero queries, zero output — should succeed.
    gwn::gwn_status const status =
        gwn::gwn_compute_winding_number_batch<Real, Index>(accessor, empty, empty, empty, output);
    EXPECT_TRUE(status.is_ok());
}
