#pragma once

#include "gwn_geometry.cuh"

#if !__has_include(<Eigen/Core>)
#error "gwn_eigen_bridge.hpp requires Eigen/Core in the include path."
#endif

#include <Eigen/Core>

#include <exception>
#include <vector>

namespace gwn {

template <class Real = float,
          class Index = std::int64_t,
          class DerivedV,
          class DerivedF>
gwn_status gwn_upload_from_eigen(
    gwn_geometry_object<Real, Index>& object,
    const Eigen::MatrixBase<DerivedV>& vertices,
    const Eigen::MatrixBase<DerivedF>& triangles,
    const cudaStream_t stream = gwn_default_stream()) noexcept try {
  if (vertices.cols() != 3 || triangles.cols() != 3) {
    return gwn_status::invalid_argument(
        "Eigen inputs must be Nx3 vertices and Mx3 triangles.");
  }

  const Eigen::Index vertex_count = vertices.rows();
  const Eigen::Index triangle_count = triangles.rows();
  if (vertex_count < 0 || triangle_count < 0) {
    return gwn_status::invalid_argument(
        "Eigen inputs cannot have negative sizes.");
  }

  std::vector<Real> x(static_cast<std::size_t>(vertex_count));
  std::vector<Real> y(static_cast<std::size_t>(vertex_count));
  std::vector<Real> z(static_cast<std::size_t>(vertex_count));
  for (Eigen::Index i = 0; i < vertex_count; ++i) {
    x[static_cast<std::size_t>(i)] = static_cast<Real>(vertices(i, 0));
    y[static_cast<std::size_t>(i)] = static_cast<Real>(vertices(i, 1));
    z[static_cast<std::size_t>(i)] = static_cast<Real>(vertices(i, 2));
  }

  std::vector<Index> i0(static_cast<std::size_t>(triangle_count));
  std::vector<Index> i1(static_cast<std::size_t>(triangle_count));
  std::vector<Index> i2(static_cast<std::size_t>(triangle_count));
  for (Eigen::Index i = 0; i < triangle_count; ++i) {
    i0[static_cast<std::size_t>(i)] = static_cast<Index>(triangles(i, 0));
    i1[static_cast<std::size_t>(i)] = static_cast<Index>(triangles(i, 1));
    i2[static_cast<std::size_t>(i)] = static_cast<Index>(triangles(i, 2));
  }

  return object.upload(cuda::std::span<const Real>(x.data(), x.size()),
                       cuda::std::span<const Real>(y.data(), y.size()),
                       cuda::std::span<const Real>(z.data(), z.size()),
                       cuda::std::span<const Index>(i0.data(), i0.size()),
                       cuda::std::span<const Index>(i1.data(), i1.size()),
                       cuda::std::span<const Index>(i2.data(), i2.size()),
                       stream);
} catch (const std::exception&) {
  return gwn_status::internal_error(
      "Unhandled std::exception in gwn_upload_from_eigen.");
} catch (...) {
  return gwn_status::internal_error(
      "Unhandled unknown exception in gwn_upload_from_eigen.");
}

}  // namespace gwn
