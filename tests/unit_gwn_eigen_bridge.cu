#include <cstdint>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <gwn/gwn_eigen_bridge.cuh>

#include "test_fixtures.cuh"
#include "test_utils.cuh"

using Real = float;
using Index = std::uint32_t;
using GwnEigenBridgeTest = gwn::tests::CudaFixture;

TEST_F(GwnEigenBridgeTest, upload_from_eigen_accepts_valid_mesh) {
    Eigen::Matrix<Real, 3, 3, Eigen::RowMajor> vertices;
    vertices << Real(0), Real(0), Real(0), Real(1), Real(0), Real(0), Real(0), Real(1), Real(0);

    Eigen::Matrix<int, 1, 3, Eigen::RowMajor> triangles;
    triangles << 0, 1, 2;

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const status =
        gwn::gwn_upload_geometry_from_eigen(geometry, vertices, triangles);
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(geometry.vertex_count(), 3u);
    EXPECT_EQ(geometry.triangle_count(), 1u);
    EXPECT_TRUE(geometry.accessor().is_valid());
}

TEST_F(GwnEigenBridgeTest, failed_upload_from_eigen_clears_previous_geometry) {
    Eigen::Matrix<Real, 3, 3, Eigen::RowMajor> vertices;
    vertices << Real(0), Real(0), Real(0), Real(1), Real(0), Real(0), Real(0), Real(1), Real(0);

    gwn::gwn_geometry_object<Real, Index> geometry;
    Eigen::Matrix<int, 1, 3, Eigen::RowMajor> valid_triangles;
    valid_triangles << 0, 1, 2;
    ASSERT_TRUE(gwn::gwn_upload_geometry_from_eigen(geometry, vertices, valid_triangles).is_ok());

    Eigen::Matrix<int, 1, 3, Eigen::RowMajor> invalid_triangles;
    invalid_triangles << 0, 1, 3;
    gwn::gwn_status const status =
        gwn::gwn_upload_geometry_from_eigen(geometry, vertices, invalid_triangles);
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
    EXPECT_FALSE(geometry.has_data());
    EXPECT_EQ(geometry.vertex_count(), 0u);
    EXPECT_EQ(geometry.triangle_count(), 0u);
}
