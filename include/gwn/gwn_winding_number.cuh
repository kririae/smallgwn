#pragma once

#include "gwn_geometry.cuh"
#include "gwn_utils.hpp"

#if !__has_include(<Eigen/Core>) || !__has_include(<Eigen/Geometry>)
#error \
    "gwn_winding_number.cuh requires Eigen/Core and Eigen/Geometry in the include path."
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gwn {

template <class Real>
using gwn_vec3 = Eigen::Matrix<Real, 3, 1>;

template <class Real>
__host__ __device__ inline Real gwn_signed_solid_angle_triangle(
    const gwn_vec3<Real>& a,
    const gwn_vec3<Real>& b,
    const gwn_vec3<Real>& c,
    const gwn_vec3<Real>& q) noexcept {
  gwn_vec3<Real> qa = a - q;
  gwn_vec3<Real> qb = b - q;
  gwn_vec3<Real> qc = c - q;

  const Real a_length = qa.norm();
  const Real b_length = qb.norm();
  const Real c_length = qc.norm();
  if (a_length == Real(0) || b_length == Real(0) || c_length == Real(0)) {
    return Real(0);
  }

  qa /= a_length;
  qb /= b_length;
  qc /= c_length;

  const Real numerator = qa.dot((qb - qa).cross(qc - qa));
  if (numerator == Real(0)) {
    return Real(0);
  }

  const Real denominator = Real(1) + qa.dot(qb) + qa.dot(qc) + qb.dot(qc);
  return Real(2) * atan2(numerator, denominator);
}

template <class Real, class Index>
__device__ inline Real gwn_winding_number_point(
    const gwn_geometry_accessor<Real, Index>& geometry,
    const Real qx,
    const Real qy,
    const Real qz) noexcept {
  if (!geometry.is_valid()) {
    return Real(0);
  }

  constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
  const std::size_t vertex_count = geometry.vertex_count();
  const std::size_t triangle_count = geometry.triangle_count();
  const gwn_vec3<Real> query(qx, qy, qz);

  Real omega_sum = Real(0);
  for (std::size_t tri = 0; tri < triangle_count; ++tri) {
    const Index ia = geometry.tri_i0[tri];
    const Index ib = geometry.tri_i1[tri];
    const Index ic = geometry.tri_i2[tri];
    if (ia < Index(0) || ib < Index(0) || ic < Index(0)) {
      continue;
    }

    const std::size_t a_index = static_cast<std::size_t>(ia);
    const std::size_t b_index = static_cast<std::size_t>(ib);
    const std::size_t c_index = static_cast<std::size_t>(ic);
    if (a_index >= vertex_count || b_index >= vertex_count ||
        c_index >= vertex_count) {
      continue;
    }

    const gwn_vec3<Real> a(geometry.vertex_x[a_index],
                           geometry.vertex_y[a_index],
                           geometry.vertex_z[a_index]);
    const gwn_vec3<Real> b(geometry.vertex_x[b_index],
                           geometry.vertex_y[b_index],
                           geometry.vertex_z[b_index]);
    const gwn_vec3<Real> c(geometry.vertex_x[c_index],
                           geometry.vertex_y[c_index],
                           geometry.vertex_z[c_index]);

    omega_sum += gwn_signed_solid_angle_triangle(a, b, c, query);
  }

  return omega_sum / (Real(4) * k_pi);
}

template <class Real, class Index>
__global__ inline void gwn_winding_number_batch_kernel(
    const gwn_geometry_accessor<Real, Index> geometry,
    const cuda::std::span<const Real> query_x,
    const cuda::std::span<const Real> query_y,
    const cuda::std::span<const Real> query_z,
    const cuda::std::span<Real> output) {
  const std::size_t query_id = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
  if (query_id >= output.size()) {
    return;
  }

  output[query_id] = gwn_winding_number_point(
      geometry, query_x[query_id], query_y[query_id], query_z[query_id]);
}

template <class Real, class Index = std::int64_t>
gwn_status gwn_compute_winding_number_batch(
    const gwn_geometry_accessor<Real, Index>& geometry,
    const cuda::std::span<const Real> query_x,
    const cuda::std::span<const Real> query_y,
    const cuda::std::span<const Real> query_z,
    const cuda::std::span<Real> output,
    const cudaStream_t stream = 0) noexcept {
  if (!geometry.is_valid()) {
    return gwn_status::invalid_argument(
        "Geometry accessor contains mismatched span lengths.");
  }

  if (query_x.size() != query_y.size() || query_x.size() != query_z.size()) {
    return gwn_status::invalid_argument(
        "Query SoA spans must have identical lengths.");
  }

  if (query_x.size() != output.size()) {
    return gwn_status::invalid_argument(
        "Output span size must match query count.");
  }

  if (output.empty()) {
    return gwn_status::ok();
  }

#if defined(__CUDACC__)
  constexpr int k_block_size = 128;
  const int block_count = static_cast<int>(
      (output.size() + static_cast<std::size_t>(k_block_size) - 1) /
      static_cast<std::size_t>(k_block_size));
  gwn_winding_number_batch_kernel<Real, Index>
      <<<block_count, k_block_size, 0, stream>>>(geometry, query_x, query_y,
                                                 query_z, output);
  return gwn_cuda_to_status(cudaGetLastError());
#else
  (void)stream;
  return gwn_status::invalid_argument(
      "Batch kernel launch requires CUDA compilation.");
#endif
}

}  // namespace gwn
