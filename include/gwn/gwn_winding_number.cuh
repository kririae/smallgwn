#pragma once

#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"

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
    const gwn_vec3<Real>& q) noexcept;

template <class Real, class Index>
__device__ inline Real gwn_triangle_solid_angle_from_primitive(
    const gwn_geometry_accessor<Real, Index>& geometry,
    const Index primitive_id,
    const gwn_vec3<Real>& query) noexcept {
  if (primitive_id < Index(0) ||
      static_cast<std::size_t>(primitive_id) >= geometry.triangle_count()) {
    return Real(0);
  }

  const std::size_t triangle_id = static_cast<std::size_t>(primitive_id);
  const Index ia = geometry.tri_i0[triangle_id];
  const Index ib = geometry.tri_i1[triangle_id];
  const Index ic = geometry.tri_i2[triangle_id];
  if (ia < Index(0) || ib < Index(0) || ic < Index(0)) {
    return Real(0);
  }

  const std::size_t a_index = static_cast<std::size_t>(ia);
  const std::size_t b_index = static_cast<std::size_t>(ib);
  const std::size_t c_index = static_cast<std::size_t>(ic);
  if (a_index >= geometry.vertex_count() ||
      b_index >= geometry.vertex_count() ||
      c_index >= geometry.vertex_count()) {
    return Real(0);
  }

  const gwn_vec3<Real> a(geometry.vertex_x[a_index], geometry.vertex_y[a_index],
                         geometry.vertex_z[a_index]);
  const gwn_vec3<Real> b(geometry.vertex_x[b_index], geometry.vertex_y[b_index],
                         geometry.vertex_z[b_index]);
  const gwn_vec3<Real> c(geometry.vertex_x[c_index], geometry.vertex_y[c_index],
                         geometry.vertex_z[c_index]);
  return gwn_signed_solid_angle_triangle(a, b, c, query);
}

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
  const std::size_t triangle_count = geometry.triangle_count();
  const gwn_vec3<Real> query(qx, qy, qz);

  Real omega_sum = Real(0);
  for (std::size_t tri = 0; tri < triangle_count; ++tri) {
    omega_sum += gwn_triangle_solid_angle_from_primitive<Real, Index>(
        geometry, static_cast<Index>(tri), query);
  }

  return omega_sum / (Real(4) * k_pi);
}

template <class Real, class Index>
__device__ inline Real gwn_winding_number_point_bvh_exact(
    const gwn_geometry_accessor<Real, Index>& geometry,
    const gwn_bvh_accessor<Real, Index>& bvh,
    const Real qx,
    const Real qy,
    const Real qz) noexcept {
  if (!geometry.is_valid() || !bvh.is_valid()) {
    return Real(0);
  }

  constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
  constexpr int k_stack_capacity = 128;
  Index stack[k_stack_capacity];
  int stack_size = 0;

  const gwn_vec3<Real> query(qx, qy, qz);
  Real omega_sum = Real(0);
  if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
    for (Index primitive_offset = 0; primitive_offset < bvh.root_count;
         ++primitive_offset) {
      const Index primitive_index =
          bvh.primitive_indices[static_cast<std::size_t>(bvh.root_index +
                                                         primitive_offset)];
      omega_sum += gwn_triangle_solid_angle_from_primitive<Real, Index>(
          geometry, primitive_index, query);
    }
    return omega_sum / (Real(4) * k_pi);
  }

  if (bvh.root_kind != gwn_bvh_child_kind::k_internal) {
    return Real(0);
  }

  stack[stack_size++] = bvh.root_index;
  while (stack_size > 0) {
    const Index node_index = stack[--stack_size];
    if (node_index < Index(0) ||
        static_cast<std::size_t>(node_index) >= bvh.nodes.size()) {
      continue;
    }

    const gwn_bvh4_node_soa<Real, Index>& node =
        bvh.nodes[static_cast<std::size_t>(node_index)];
    GWN_PRAGMA_UNROLL
    for (int child_slot = 0; child_slot < 4; ++child_slot) {
      const auto child_kind =
          static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
      if (child_kind == gwn_bvh_child_kind::k_invalid) {
        continue;
      }

      if (child_kind == gwn_bvh_child_kind::k_internal) {
        if (stack_size < k_stack_capacity) {
          stack[stack_size++] = node.child_index[child_slot];
        }
        continue;
      }

      if (child_kind != gwn_bvh_child_kind::k_leaf) {
        continue;
      }

      const Index begin = node.child_index[child_slot];
      const Index count = node.child_count[child_slot];
      for (Index primitive_offset = 0; primitive_offset < count;
           ++primitive_offset) {
        const Index sorted_primitive_index = begin + primitive_offset;
        if (sorted_primitive_index < Index(0) ||
            static_cast<std::size_t>(sorted_primitive_index) >=
                bvh.primitive_indices.size()) {
          continue;
        }
        const Index primitive_index =
            bvh.primitive_indices[static_cast<std::size_t>(
                sorted_primitive_index)];
        omega_sum += gwn_triangle_solid_angle_from_primitive<Real, Index>(
            geometry, primitive_index, query);
      }
    }
  }

  return omega_sum / (Real(4) * k_pi);
}

namespace detail {

template <class Real, class Index>
struct gwn_winding_number_batch_functor {
  gwn_geometry_accessor<Real, Index> geometry{};
  cuda::std::span<const Real> query_x{};
  cuda::std::span<const Real> query_y{};
  cuda::std::span<const Real> query_z{};
  cuda::std::span<Real> output{};

  __device__ void operator()(const std::size_t query_id) const {
    output[query_id] = gwn_winding_number_point(
        geometry, query_x[query_id], query_y[query_id], query_z[query_id]);
  }
};

template <class Real, class Index>
struct gwn_winding_number_batch_bvh_exact_functor {
  gwn_geometry_accessor<Real, Index> geometry{};
  gwn_bvh_accessor<Real, Index> bvh{};
  cuda::std::span<const Real> query_x{};
  cuda::std::span<const Real> query_y{};
  cuda::std::span<const Real> query_z{};
  cuda::std::span<Real> output{};

  __device__ void operator()(const std::size_t query_id) const {
    output[query_id] = gwn_winding_number_point_bvh_exact(
        geometry, bvh, query_x[query_id], query_y[query_id], query_z[query_id]);
  }
};

template <class Real, class Index>
[[nodiscard]] inline gwn_winding_number_batch_functor<Real, Index>
gwn_make_winding_number_batch_functor(
    const gwn_geometry_accessor<Real, Index>& geometry,
    const cuda::std::span<const Real> query_x,
    const cuda::std::span<const Real> query_y,
    const cuda::std::span<const Real> query_z,
    const cuda::std::span<Real> output) {
  return gwn_winding_number_batch_functor<Real, Index>{
      geometry, query_x, query_y, query_z, output};
}

template <class Real, class Index>
[[nodiscard]] inline gwn_winding_number_batch_bvh_exact_functor<Real, Index>
gwn_make_winding_number_batch_bvh_exact_functor(
    const gwn_geometry_accessor<Real, Index>& geometry,
    const gwn_bvh_accessor<Real, Index>& bvh,
    const cuda::std::span<const Real> query_x,
    const cuda::std::span<const Real> query_y,
    const cuda::std::span<const Real> query_z,
    const cuda::std::span<Real> output) {
  return gwn_winding_number_batch_bvh_exact_functor<Real, Index>{
      geometry, bvh, query_x, query_y, query_z, output};
}

}  // namespace detail

template <class Real, class Index = std::int64_t>
gwn_status gwn_compute_winding_number_batch(
    const gwn_geometry_accessor<Real, Index>& geometry,
    const cuda::std::span<const Real> query_x,
    const cuda::std::span<const Real> query_y,
    const cuda::std::span<const Real> query_z,
    const cuda::std::span<Real> output,
    const cudaStream_t stream = gwn_default_stream()) noexcept {
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
  if (!gwn_span_has_storage(query_x) || !gwn_span_has_storage(query_y) ||
      !gwn_span_has_storage(query_z) || !gwn_span_has_storage(output)) {
    return gwn_status::invalid_argument(
        "Query/output spans must use non-null storage when non-empty.");
  }

  if (output.empty()) {
    return gwn_status::ok();
  }

  constexpr int k_block_size = detail::k_gwn_default_block_size;
  return detail::gwn_launch_linear_kernel<k_block_size>(
      output.size(),
      detail::gwn_make_winding_number_batch_functor<Real, Index>(
          geometry, query_x, query_y, query_z, output),
      stream);
}

template <class Real, class Index = std::int64_t>
gwn_status gwn_compute_winding_number_batch_bvh_exact(
    const gwn_geometry_accessor<Real, Index>& geometry,
    const gwn_bvh_accessor<Real, Index>& bvh,
    const cuda::std::span<const Real> query_x,
    const cuda::std::span<const Real> query_y,
    const cuda::std::span<const Real> query_z,
    const cuda::std::span<Real> output,
    const cudaStream_t stream = gwn_default_stream()) noexcept {
  if (!geometry.is_valid()) {
    return gwn_status::invalid_argument(
        "Geometry accessor contains mismatched span lengths.");
  }
  if (!bvh.is_valid()) {
    return gwn_status::invalid_argument("BVH accessor is invalid.");
  }
  if (query_x.size() != query_y.size() || query_x.size() != query_z.size()) {
    return gwn_status::invalid_argument(
        "Query SoA spans must have identical lengths.");
  }
  if (query_x.size() != output.size()) {
    return gwn_status::invalid_argument(
        "Output span size must match query count.");
  }
  if (!gwn_span_has_storage(query_x) || !gwn_span_has_storage(query_y) ||
      !gwn_span_has_storage(query_z) || !gwn_span_has_storage(output)) {
    return gwn_status::invalid_argument(
        "Query/output spans must use non-null storage when non-empty.");
  }
  if (output.empty()) {
    return gwn_status::ok();
  }

  constexpr int k_block_size = detail::k_gwn_default_block_size;
  return detail::gwn_launch_linear_kernel<k_block_size>(
      output.size(),
      detail::gwn_make_winding_number_batch_bvh_exact_functor<Real, Index>(
          geometry, bvh, query_x, query_y, query_z, output),
      stream);
}

}  // namespace gwn
