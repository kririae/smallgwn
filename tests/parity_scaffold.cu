#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "reference_cpu.hpp"
#include "reference_hdk/UT_SolidAngle.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace {

bool is_cuda_runtime_unavailable(const cudaError_t result) {
  return result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
         result == cudaErrorOperatingSystem ||
         result == cudaErrorSystemNotReady;
}

}  // namespace

TEST(smallgwn_parity_scaffold,
     cpu_reference_matches_repository_triangle_oracle) {
  constexpr float k_pi = 3.14159265358979323846f;
  constexpr float k_epsilon = 1e-6f;

  const std::array<float, 3> vertex_x{1.0f, 0.0f, 0.0f};
  const std::array<float, 3> vertex_y{0.0f, 1.0f, 0.0f};
  const std::array<float, 3> vertex_z{0.0f, 0.0f, 1.0f};
  const std::array<std::int64_t, 1> tri_i0{0};
  const std::array<std::int64_t, 1> tri_i1{1};
  const std::array<std::int64_t, 1> tri_i2{2};

  const std::array<float, 2> query_x{0.0f, 0.25f};
  const std::array<float, 2> query_y{0.0f, 0.25f};
  const std::array<float, 2> query_z{0.0f, 0.25f};

  std::vector<float> reference_output(query_x.size(), 0.0f);
  const gwn::gwn_status status =
      gwn::tests::reference_winding_number_batch<float, std::int64_t>(
          std::span<const float>(vertex_x.data(), vertex_x.size()),
          std::span<const float>(vertex_y.data(), vertex_y.size()),
          std::span<const float>(vertex_z.data(), vertex_z.size()),
          std::span<const std::int64_t>(tri_i0.data(), tri_i0.size()),
          std::span<const std::int64_t>(tri_i1.data(), tri_i1.size()),
          std::span<const std::int64_t>(tri_i2.data(), tri_i2.size()),
          std::span<const float>(query_x.data(), query_x.size()),
          std::span<const float>(query_y.data(), query_y.size()),
          std::span<const float>(query_z.data(), query_z.size()),
          std::span<float>(reference_output.data(), reference_output.size()));
  ASSERT_TRUE(status.is_ok()) << status.message();

  for (std::size_t query_id = 0; query_id < query_x.size(); ++query_id) {
    HDK_Sample::UT_Vector3T<float> a;
    HDK_Sample::UT_Vector3T<float> b;
    HDK_Sample::UT_Vector3T<float> c;
    HDK_Sample::UT_Vector3T<float> q;
    a[0] = vertex_x[0];
    a[1] = vertex_y[0];
    a[2] = vertex_z[0];
    b[0] = vertex_x[1];
    b[1] = vertex_y[1];
    b[2] = vertex_z[1];
    c[0] = vertex_x[2];
    c[1] = vertex_y[2];
    c[2] = vertex_z[2];
    q[0] = query_x[query_id];
    q[1] = query_y[query_id];
    q[2] = query_z[query_id];

    const float oracle_solid_angle =
        HDK_Sample::UTsignedSolidAngleTri(a, b, c, q);
    const float oracle_winding = oracle_solid_angle / (4.0f * k_pi);
    EXPECT_NEAR(reference_output[query_id], oracle_winding, k_epsilon);
  }
}

TEST(smallgwn_parity_scaffold, bvh_exact_batch_matches_cpu_reference) {
  constexpr float k_epsilon = 1e-4f;

  const std::array<float, 6> vertex_x{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const std::array<float, 6> vertex_y{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
  const std::array<float, 6> vertex_z{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};

  const std::array<std::int64_t, 8> tri_i0{0, 2, 1, 3, 2, 1, 3, 0};
  const std::array<std::int64_t, 8> tri_i1{2, 1, 3, 0, 0, 2, 1, 3};
  const std::array<std::int64_t, 8> tri_i2{4, 4, 4, 4, 5, 5, 5, 5};

  const std::array<float, 4> query_x{0.0f, 2.0f, 0.2f, -1.5f};
  const std::array<float, 4> query_y{0.0f, 0.0f, 0.2f, 0.1f};
  const std::array<float, 4> query_z{0.0f, 0.0f, 0.2f, 0.0f};

  std::vector<float> reference_output(query_x.size(), 0.0f);
  const gwn::gwn_status reference_status =
      gwn::tests::reference_winding_number_batch<float, std::int64_t>(
          std::span<const float>(vertex_x.data(), vertex_x.size()),
          std::span<const float>(vertex_y.data(), vertex_y.size()),
          std::span<const float>(vertex_z.data(), vertex_z.size()),
          std::span<const std::int64_t>(tri_i0.data(), tri_i0.size()),
          std::span<const std::int64_t>(tri_i1.data(), tri_i1.size()),
          std::span<const std::int64_t>(tri_i2.data(), tri_i2.size()),
          std::span<const float>(query_x.data(), query_x.size()),
          std::span<const float>(query_y.data(), query_y.size()),
          std::span<const float>(query_z.data(), query_z.size()),
          std::span<float>(reference_output.data(), reference_output.size()));
  ASSERT_TRUE(reference_status.is_ok()) << reference_status.message();

  gwn::gwn_geometry_object<float, std::int64_t> geometry;
  const gwn::gwn_status upload_status = geometry.upload(
      cuda::std::span<const float>(vertex_x.data(), vertex_x.size()),
      cuda::std::span<const float>(vertex_y.data(), vertex_y.size()),
      cuda::std::span<const float>(vertex_z.data(), vertex_z.size()),
      cuda::std::span<const std::int64_t>(tri_i0.data(), tri_i0.size()),
      cuda::std::span<const std::int64_t>(tri_i1.data(), tri_i1.size()),
      cuda::std::span<const std::int64_t>(tri_i2.data(), tri_i2.size()));
  if (!upload_status.is_ok() &&
      upload_status.error() == gwn::gwn_error::cuda_runtime_error &&
      is_cuda_runtime_unavailable(upload_status.cuda_result())) {
    GTEST_SKIP() << "CUDA runtime unavailable: " << upload_status.message();
  }
  ASSERT_TRUE(upload_status.is_ok()) << upload_status.message();

  const gwn::gwn_status build_status = geometry.build_bvh();
  ASSERT_TRUE(build_status.is_ok()) << build_status.message();
  ASSERT_TRUE(geometry.has_bvh());
  const auto& bvh = geometry.bvh_accessor();
  EXPECT_TRUE(bvh.is_valid());
  EXPECT_EQ(bvh.root_kind, gwn::gwn_bvh_child_kind::k_internal);
  EXPECT_EQ(bvh.primitive_indices.size(), tri_i0.size());

  float* d_query_x = nullptr;
  float* d_query_y = nullptr;
  float* d_query_z = nullptr;
  float* d_output = nullptr;
  const std::size_t query_bytes = query_x.size() * sizeof(float);
  cudaError_t result = cudaMalloc(&d_query_x, query_bytes);
  if (is_cuda_runtime_unavailable(result)) {
    GTEST_SKIP() << "CUDA runtime unavailable: " << cudaGetErrorString(result);
  }
  ASSERT_EQ(cudaSuccess, result);
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_query_y, query_bytes));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_query_z, query_bytes));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_output, query_bytes));
  const auto cleanup = [&]() {
    if (d_output != nullptr) {
      (void)cudaFree(d_output);
      d_output = nullptr;
    }
    if (d_query_z != nullptr) {
      (void)cudaFree(d_query_z);
      d_query_z = nullptr;
    }
    if (d_query_y != nullptr) {
      (void)cudaFree(d_query_y);
      d_query_y = nullptr;
    }
    if (d_query_x != nullptr) {
      (void)cudaFree(d_query_x);
      d_query_x = nullptr;
    }
  };

  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_query_x, query_x.data(), query_bytes,
                                    cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_query_y, query_y.data(), query_bytes,
                                    cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_query_z, query_z.data(), query_bytes,
                                    cudaMemcpyHostToDevice));

  const gwn::gwn_status bvh_query_status =
      gwn::gwn_compute_winding_number_batch_bvh_exact<float, std::int64_t>(
          geometry.accessor(),
          cuda::std::span<const float>(d_query_x, query_x.size()),
          cuda::std::span<const float>(d_query_y, query_y.size()),
          cuda::std::span<const float>(d_query_z, query_z.size()),
          cuda::std::span<float>(d_output, query_x.size()));
  ASSERT_TRUE(bvh_query_status.is_ok()) << bvh_query_status.message();
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<float> gpu_output(query_x.size(), 0.0f);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(gpu_output.data(), d_output, query_bytes,
                                    cudaMemcpyDeviceToHost));
  cleanup();

  for (std::size_t query_id = 0; query_id < query_x.size(); ++query_id) {
    EXPECT_NEAR(gpu_output[query_id], reference_output[query_id], k_epsilon);
  }
}
