#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "reference_cpu.hpp"
#include "reference_hdk/UT_SolidAngle.h"

#include <array>
#include <cmath>
#include <span>
#include <vector>

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
