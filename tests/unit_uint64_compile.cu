#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn_boundary.cuh>
#include <gwn/gwn_bvh.cuh>
#include <gwn/gwn_bvh_facade.cuh>
#include <gwn/gwn_bvh_refit.cuh>
#include <gwn/gwn_bvh_topology_build.cuh>
#include <gwn/gwn_geometry.cuh>
#include <gwn/gwn_query.cuh>

#include "test_fixtures.hpp"
#include "test_utils.hpp"

using Real = float;
using Index = std::uint64_t;
using Real64 = double;
using Index64 = std::uint64_t;
using gwn::tests::CudaFixture;

template gwn::gwn_status gwn::gwn_bvh_topology_build_lbvh<4, Real, Index, std::uint64_t>(
    gwn_geometry_object<Real, Index> const &, gwn_bvh_topology_object<4, Real, Index> &,
    cudaStream_t
) noexcept;

template gwn::gwn_status gwn::gwn_bvh_topology_build_hploc<4, Real, Index, std::uint64_t>(
    gwn_geometry_object<Real, Index> const &, gwn_bvh_topology_object<4, Real, Index> &,
    cudaStream_t
) noexcept;

template gwn::gwn_status gwn::gwn_bvh_topology_build_lbvh<4, Real64, Index64, std::uint64_t>(
    gwn_geometry_object<Real64, Index64> const &, gwn_bvh_topology_object<4, Real64, Index64> &,
    cudaStream_t
) noexcept;

template gwn::gwn_status gwn::gwn_bvh_topology_build_hploc<4, Real64, Index64, std::uint64_t>(
    gwn_geometry_object<Real64, Index64> const &, gwn_bvh_topology_object<4, Real64, Index64> &,
    cudaStream_t
) noexcept;

TEST(smallgwn_unit_uint64_compile, topology_templates_instantiate) { SUCCEED(); }

TEST(smallgwn_unit_uint64_compile, upload_and_harnack_templates_instantiate) {
    gwn::gwn_geometry_object<Real, Index> geometry{};

    cuda::std::span<Real const> vx{};
    cuda::std::span<Real const> vy{};
    cuda::std::span<Real const> vz{};
    cuda::std::span<Index const> i0{};
    cuda::std::span<Index const> i1{};
    cuda::std::span<Index const> i2{};
    EXPECT_TRUE(gwn::gwn_upload_geometry(geometry, vx, vy, vz, i0, i1, i2).is_ok());

    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    bvh.root_kind = gwn::gwn_bvh_child_kind::k_leaf;
    bvh.root_index = Index(0);
    bvh.root_count = Index(0);

    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};
    gwn::gwn_bvh4_moment_accessor<0, Real, Index> m0{};
    gwn::gwn_bvh4_moment_accessor<1, Real, Index> m1{};
    gwn::gwn_bvh4_moment_accessor<2, Real, Index> m2{};

    cuda::std::span<Real const> ray_ox{};
    cuda::std::span<Real const> ray_oy{};
    cuda::std::span<Real const> ray_oz{};
    cuda::std::span<Real const> ray_dx{};
    cuda::std::span<Real const> ray_dy{};
    cuda::std::span<Real const> ray_dz{};
    cuda::std::span<Real> out_t{};
    cuda::std::span<Real> out_nx{};
    cuda::std::span<Real> out_ny{};
    cuda::std::span<Real> out_nz{};

    EXPECT_TRUE((gwn::gwn_compute_harnack_trace_batch_bvh_taylor<0, Real, Index>(
                     geometry.accessor(), bvh, aabb, m0, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy,
                     ray_dz, out_t, out_nx, out_ny, out_nz
                 ))
                    .is_ok());
    EXPECT_TRUE((gwn::gwn_compute_harnack_trace_batch_bvh_taylor<1, Real, Index>(
                     geometry.accessor(), bvh, aabb, m1, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy,
                     ray_dz, out_t, out_nx, out_ny, out_nz
                 ))
                    .is_ok());
    EXPECT_TRUE((gwn::gwn_compute_harnack_trace_batch_bvh_taylor<2, Real, Index>(
                     geometry.accessor(), bvh, aabb, m2, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy,
                     ray_dz, out_t, out_nx, out_ny, out_nz
                 ))
                    .is_ok());
}

TEST(smallgwn_unit_uint64_compile, boundary_chain_and_antipodal_templates_instantiate) {
    gwn::gwn_geometry_object<Real, Index> geometry{};

    cuda::std::span<Real const> vx{};
    cuda::std::span<Real const> vy{};
    cuda::std::span<Real const> vz{};
    cuda::std::span<Index const> i0{};
    cuda::std::span<Index const> i1{};
    cuda::std::span<Index const> i2{};
    EXPECT_TRUE(gwn::gwn_upload_geometry(geometry, vx, vy, vz, i0, i1, i2).is_ok());

    gwn::gwn_boundary_chain_object<Index> boundary{};
    EXPECT_TRUE(gwn::gwn_build_boundary_chain(geometry.accessor(), boundary).is_ok());

    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    bvh.root_kind = gwn::gwn_bvh_child_kind::k_leaf;
    bvh.root_index = Index(0);
    bvh.root_count = Index(0);

    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> output{};
    cuda::std::span<Real> output_x{};
    cuda::std::span<Real> output_y{};
    cuda::std::span<Real> output_z{};

    EXPECT_TRUE((gwn::gwn_compute_winding_number_batch_bvh_antipodal<Real, Index>(
                     geometry.accessor(), bvh, aabb, boundary.accessor(), query_x, query_y, query_z,
                     output
                 ))
                    .is_ok());
    EXPECT_TRUE((gwn::gwn_compute_winding_gradient_batch_antipodal<Real, Index>(
                     geometry.accessor(), boundary.accessor(), query_x, query_y, query_z, output_x,
                     output_y, output_z
                 ))
                    .is_ok());
}

TEST(smallgwn_unit_uint64_compile, double_uint64_upload_build_and_query_templates_instantiate) {
    gwn::gwn_geometry_object<Real64, Index64> geometry{};

    cuda::std::span<Real64 const> vx{};
    cuda::std::span<Real64 const> vy{};
    cuda::std::span<Real64 const> vz{};
    cuda::std::span<Index64 const> i0{};
    cuda::std::span<Index64 const> i1{};
    cuda::std::span<Index64 const> i2{};
    EXPECT_TRUE(gwn::gwn_upload_geometry(geometry, vx, vy, vz, i0, i1, i2).is_ok());

    gwn::gwn_bvh4_topology_object<Real64, Index64> topology{};
    EXPECT_TRUE((gwn::gwn_bvh_topology_build_lbvh<4, Real64, Index64>(geometry, topology)).is_ok());
    EXPECT_TRUE(
        (gwn::gwn_bvh_topology_build_hploc<4, Real64, Index64>(geometry, topology)).is_ok()
    );

    gwn::gwn_bvh4_topology_accessor<Real64, Index64> bvh{};
    bvh.root_kind = gwn::gwn_bvh_child_kind::k_leaf;
    bvh.root_index = Index64(0);
    bvh.root_count = Index64(0);

    gwn::gwn_bvh4_aabb_accessor<Real64, Index64> aabb{};
    gwn::gwn_bvh4_moment_accessor<1, Real64, Index64> moment{};

    cuda::std::span<Real64 const> ray_ox{};
    cuda::std::span<Real64 const> ray_oy{};
    cuda::std::span<Real64 const> ray_oz{};
    cuda::std::span<Real64 const> ray_dx{};
    cuda::std::span<Real64 const> ray_dy{};
    cuda::std::span<Real64 const> ray_dz{};
    cuda::std::span<Real64> out_t{};
    cuda::std::span<Real64> out_nx{};
    cuda::std::span<Real64> out_ny{};
    cuda::std::span<Real64> out_nz{};

    EXPECT_TRUE((gwn::gwn_compute_harnack_trace_batch_bvh_taylor<1, Real64, Index64>(
                     geometry.accessor(), bvh, aabb, moment, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy,
                     ray_dz, out_t, out_nx, out_ny, out_nz
                 ))
                    .is_ok());
}

TEST_F(CudaFixture, uint64_non_empty_hploc_refit_query_matches_lbvh) {
    std::vector<Real> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Real> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::vector<Real> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::vector<Index> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::vector<Index> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::vector<Index> i2{4, 4, 4, 4, 5, 5, 5, 5};

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
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    gwn::gwn_bvh4_topology_object<Real, Index> lbvh_topology;
    gwn::gwn_bvh4_topology_object<Real, Index> hploc_topology;
    status = gwn::gwn_bvh_topology_build_lbvh<4, Real, Index>(geometry, lbvh_topology);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    status = gwn::gwn_bvh_topology_build_hploc<4, Real, Index>(geometry, hploc_topology);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    std::vector<Real> moved_z{0.0f, 0.0f, 0.0f, 0.0f, 1.25f, -1.25f};
    status = gwn::gwn_update_geometry(
        geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(moved_z.data(), moved_z.size())
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    gwn::gwn_bvh4_aabb_object<Real, Index> lbvh_aabb;
    gwn::gwn_bvh4_aabb_object<Real, Index> hploc_aabb;
    gwn::gwn_bvh4_moment_object<2, Real, Index> lbvh_moment;
    gwn::gwn_bvh4_moment_object<2, Real, Index> hploc_moment;
    status = gwn::gwn_bvh_refit_aabb_moment<2, 4, Real, Index>(
        geometry, lbvh_topology, lbvh_aabb, lbvh_moment
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    status = gwn::gwn_bvh_refit_aabb_moment<2, 4, Real, Index>(
        geometry, hploc_topology, hploc_aabb, hploc_moment
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    std::vector<Real> qx{0.0f, 2.5f, -2.5f, 0.25f};
    std::vector<Real> qy{0.0f, 0.1f, 0.2f, -0.35f};
    std::vector<Real> qz{0.0f, 0.2f, -0.3f, 0.45f};
    gwn::gwn_device_array<Real> d_qx;
    gwn::gwn_device_array<Real> d_qy;
    gwn::gwn_device_array<Real> d_qz;
    gwn::gwn_device_array<Real> d_lbvh_out;
    gwn::gwn_device_array<Real> d_hploc_out;
    ASSERT_TRUE(d_qx.resize(qx.size()).is_ok());
    ASSERT_TRUE(d_qy.resize(qy.size()).is_ok());
    ASSERT_TRUE(d_qz.resize(qz.size()).is_ok());
    ASSERT_TRUE(d_lbvh_out.resize(qx.size()).is_ok());
    ASSERT_TRUE(d_hploc_out.resize(qx.size()).is_ok());
    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());

    status = gwn::gwn_compute_winding_number_batch_bvh_taylor<2, Real, Index>(
        geometry.accessor(), lbvh_topology.accessor(), lbvh_moment.accessor(),
        cuda::std::span<Real const>(d_qx.data(), qx.size()),
        cuda::std::span<Real const>(d_qy.data(), qy.size()),
        cuda::std::span<Real const>(d_qz.data(), qz.size()),
        cuda::std::span<Real>(d_lbvh_out.data(), qx.size()), Real(2)
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    status = gwn::gwn_compute_winding_number_batch_bvh_taylor<2, Real, Index>(
        geometry.accessor(), hploc_topology.accessor(), hploc_moment.accessor(),
        cuda::std::span<Real const>(d_qx.data(), qx.size()),
        cuda::std::span<Real const>(d_qy.data(), qy.size()),
        cuda::std::span<Real const>(d_qz.data(), qz.size()),
        cuda::std::span<Real>(d_hploc_out.data(), qx.size()), Real(2)
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    std::vector<Real> lbvh_out(qx.size(), Real(0));
    std::vector<Real> hploc_out(qx.size(), Real(0));
    ASSERT_TRUE(
        d_lbvh_out.copy_to_host(cuda::std::span<Real>(lbvh_out.data(), lbvh_out.size())).is_ok()
    );
    ASSERT_TRUE(
        d_hploc_out.copy_to_host(cuda::std::span<Real>(hploc_out.data(), hploc_out.size())).is_ok()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    for (std::size_t i = 0; i < qx.size(); ++i) {
        ASSERT_TRUE(std::isfinite(lbvh_out[i]));
        ASSERT_TRUE(std::isfinite(hploc_out[i]));
        EXPECT_NEAR(lbvh_out[i], hploc_out[i], Real(1e-5));
    }
}
