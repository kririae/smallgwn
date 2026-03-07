#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_utils.hpp"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;
using gwn::tests::SingleTriangleMesh;

namespace {

template <class Mesh>
bool build_edge_distance_scene(
    Mesh const &mesh, gwn::gwn_geometry_object<Real, Index> &geometry,
    gwn::gwn_bvh4_topology_object<Real, Index> &bvh, gwn::gwn_bvh4_aabb_object<Real, Index> &aabb
) {
    gwn::gwn_status const geometry_status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
        cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
        cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
        cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
        cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
        cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
    );
    if (geometry_status.error() == gwn::gwn_error::cuda_runtime_error)
        return false;
    if (!geometry_status.is_ok()) {
        ADD_FAILURE() << "geometry upload: " << gwn::tests::status_to_debug_string(geometry_status);
        return false;
    }

    gwn::gwn_status const build_status =
        gwn::gwn_bvh_facade_build_topology_aabb_lbvh<4, Real, Index>(geometry, bvh, aabb);
    if (!build_status.is_ok()) {
        ADD_FAILURE() << "BVH build: " << gwn::tests::status_to_debug_string(build_status);
        return false;
    }

    return true;
}

} // namespace

TEST_F(CudaFixture, edge_distance_rejects_invalid_accessors) {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    if (!gwn::tests::resize_device_arrays(1u, d_qx, d_qy, d_qz, d_out))
        GTEST_SKIP() << "CUDA unavailable";

    gwn::gwn_status const status =
        gwn::gwn_compute_unsigned_boundary_edge_distance_batch_bvh<Real, Index>(
            geometry, bvh, aabb, d_qx.span(), d_qy.span(), d_qz.span(), d_out.span()
        );
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, edge_distance_rejects_mismatched_output) {
    SingleTriangleMesh mesh;
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    if (!build_edge_distance_scene(mesh, geometry, bvh, aabb))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    bool const allocate_ok =
        gwn::tests::resize_device_arrays(1u, d_qx, d_qy, d_qz) && d_out.resize(2u).is_ok();
    if (!allocate_ok)
        GTEST_SKIP() << "CUDA unavailable";

    gwn::gwn_status const status =
        gwn::gwn_compute_unsigned_boundary_edge_distance_batch_bvh<Real, Index>(
            geometry.accessor(), bvh.accessor(), aabb.accessor(), d_qx.span(), d_qy.span(),
            d_qz.span(), d_out.span()
        );
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, edge_distance_matches_expected_value_on_single_triangle) {
    SingleTriangleMesh mesh;
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    if (!build_edge_distance_scene(mesh, geometry, bvh, aabb))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    std::array<Real, 1> const qx{Real(0.5)};
    std::array<Real, 1> const qy{Real(0)};
    std::array<Real, 1> const qz{Real(0.25)};

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    if (!gwn::tests::resize_device_arrays(1u, d_qx, d_qy, d_qz, d_out))
        GTEST_SKIP() << "CUDA unavailable";

    bool const upload_ok =
        d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok() &&
        d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok() &&
        d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok();
    if (!upload_ok)
        GTEST_SKIP() << "CUDA unavailable";

    gwn::gwn_status const status =
        gwn::gwn_compute_unsigned_boundary_edge_distance_batch_bvh<Real, Index>(
            geometry.accessor(), bvh.accessor(), aabb.accessor(), d_qx.span(), d_qy.span(),
            d_qz.span(), d_out.span()
        );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> output{Real(-1)};
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(output.data(), output.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_NEAR(output[0], Real(0.25), Real(1e-5));
}
