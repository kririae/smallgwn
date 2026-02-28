#include <cstdint>

#include <gtest/gtest.h>

#include <gwn/gwn_bvh.cuh>
#include <gwn/gwn_bvh_topology_build.cuh>
#include <gwn/gwn_geometry.cuh>
#include <gwn/gwn_query.cuh>

using Real = float;
using Index = std::uint64_t;

template gwn::gwn_status gwn::gwn_bvh_topology_build_lbvh<4, Real, Index, std::uint64_t>(
    gwn_geometry_object<Real, Index> const &, gwn_bvh_topology_object<4, Real, Index> &,
    cudaStream_t
) noexcept;

template gwn::gwn_status gwn::gwn_bvh_topology_build_hploc<4, Real, Index, std::uint64_t>(
    gwn_geometry_object<Real, Index> const &, gwn_bvh_topology_object<4, Real, Index> &,
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
    EXPECT_TRUE(geometry.upload(vx, vy, vz, i0, i1, i2).is_ok());

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
        geometry.accessor(), bvh, aabb, m0, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
        out_t, out_nx, out_ny, out_nz
    ))
                    .is_ok());
    EXPECT_TRUE((gwn::gwn_compute_harnack_trace_batch_bvh_taylor<1, Real, Index>(
        geometry.accessor(), bvh, aabb, m1, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
        out_t, out_nx, out_ny, out_nz
    ))
                    .is_ok());
    EXPECT_TRUE((gwn::gwn_compute_harnack_trace_batch_bvh_taylor<2, Real, Index>(
        geometry.accessor(), bvh, aabb, m2, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
        out_t, out_nx, out_ny, out_nz
    ))
                    .is_ok());

}
