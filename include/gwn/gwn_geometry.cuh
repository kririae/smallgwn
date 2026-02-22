#pragma once

#include "gwn_utils.hpp"

#include <cuda_runtime_api.h>
#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace gwn {

template <class Real, class Index = std::int64_t>
struct gwn_geometry_accessor {
  using real_type = Real;
  using index_type = Index;

  cuda::std::span<const Real> vertex_x{};
  cuda::std::span<const Real> vertex_y{};
  cuda::std::span<const Real> vertex_z{};

  cuda::std::span<const Index> tri_i0{};
  cuda::std::span<const Index> tri_i1{};
  cuda::std::span<const Index> tri_i2{};

  __host__ __device__ constexpr std::size_t vertex_count() const noexcept {
    return vertex_x.size();
  }
  __host__ __device__ constexpr std::size_t triangle_count() const noexcept {
    return tri_i0.size();
  }

  __host__ __device__ constexpr bool is_valid() const noexcept {
    return vertex_x.size() == vertex_y.size() &&
           vertex_x.size() == vertex_z.size() &&
           tri_i0.size() == tri_i1.size() && tri_i0.size() == tri_i2.size();
  }
};

namespace detail {

template <class T>
gwn_status gwn_upload_span(cuda::std::span<const T>& dst,
                           const cuda::std::span<const T> src,
                           const cudaStream_t stream) {
  dst = {};
  if (src.empty()) {
    return gwn_status::ok();
  }

  void* device_ptr = nullptr;
  gwn_status status = gwn_cuda_to_status(
      cudaMallocAsync(&device_ptr, src.size_bytes(), stream));
  if (!status.is_ok()) {
    return status;
  }

  status = gwn_cuda_to_status(cudaMemcpyAsync(device_ptr, src.data(),
                                              src.size_bytes(),
                                              cudaMemcpyHostToDevice, stream));
  if (!status.is_ok()) {
    (void)cudaFreeAsync(device_ptr, stream);
    return status;
  }

  status = gwn_cuda_to_status(cudaStreamSynchronize(stream));
  if (!status.is_ok()) {
    (void)cudaFreeAsync(device_ptr, stream);
    return status;
  }

  dst = cuda::std::span<const T>(static_cast<const T*>(device_ptr), src.size());
  return gwn_status::ok();
}

template <class T>
void gwn_release_span(cuda::std::span<const T>& span_view,
                      const cudaStream_t stream) noexcept {
  if (span_view.data() != nullptr) {
    (void)cudaFreeAsync(const_cast<T*>(span_view.data()), stream);
    span_view = {};
  }
}

template <class Real, class Index>
void gwn_release_accessor(gwn_geometry_accessor<Real, Index>& accessor,
                          const cudaStream_t stream) noexcept {
  gwn_release_span(accessor.tri_i2, stream);
  gwn_release_span(accessor.tri_i1, stream);
  gwn_release_span(accessor.tri_i0, stream);
  gwn_release_span(accessor.vertex_z, stream);
  gwn_release_span(accessor.vertex_y, stream);
  gwn_release_span(accessor.vertex_x, stream);
}

template <class Real, class Index>
gwn_status gwn_upload_accessor(gwn_geometry_accessor<Real, Index>& accessor,
                               const cuda::std::span<const Real> x,
                               const cuda::std::span<const Real> y,
                               const cuda::std::span<const Real> z,
                               const cuda::std::span<const Index> i0,
                               const cuda::std::span<const Index> i1,
                               const cuda::std::span<const Index> i2,
                               const cudaStream_t stream) {
  if (x.size() != y.size() || x.size() != z.size()) {
    return gwn_status::invalid_argument(
        "Vertex SoA spans must have identical lengths.");
  }
  if (i0.size() != i1.size() || i0.size() != i2.size()) {
    return gwn_status::invalid_argument(
        "Triangle SoA spans must have identical lengths.");
  }

  gwn_status status = gwn_upload_span(accessor.vertex_x, x, stream);
  if (!status.is_ok()) {
    return status;
  }

  status = gwn_upload_span(accessor.vertex_y, y, stream);
  if (!status.is_ok()) {
    return status;
  }

  status = gwn_upload_span(accessor.vertex_z, z, stream);
  if (!status.is_ok()) {
    return status;
  }

  status = gwn_upload_span(accessor.tri_i0, i0, stream);
  if (!status.is_ok()) {
    return status;
  }

  status = gwn_upload_span(accessor.tri_i1, i1, stream);
  if (!status.is_ok()) {
    return status;
  }

  return gwn_upload_span(accessor.tri_i2, i2, stream);
}

}  // namespace detail

template <class Real = float, class Index = std::int64_t>
class gwn_geometry_object final : public gwn_noncopyable {
  static_assert(std::is_floating_point_v<Real>,
                "Real must be a floating-point type.");
  static_assert(std::is_integral_v<Index>, "Index must be an integral type.");

 public:
  using real_type = Real;
  using index_type = Index;
  using accessor_type = gwn_geometry_accessor<Real, Index>;

  gwn_geometry_object() = default;

  gwn_geometry_object(const cuda::std::span<const Real> x,
                      const cuda::std::span<const Real> y,
                      const cuda::std::span<const Real> z,
                      const cuda::std::span<const Index> i0,
                      const cuda::std::span<const Index> i1,
                      const cuda::std::span<const Index> i2,
                      const cudaStream_t stream = 0) {
    gwn_throw_if_error(upload(x, y, z, i0, i1, i2, stream));
  }

  gwn_geometry_object(gwn_geometry_object&& other) noexcept {
    swap(*this, other);
  }

  gwn_geometry_object& operator=(gwn_geometry_object other) noexcept {
    swap(*this, other);
    return *this;
  }

  ~gwn_geometry_object() { clear(); }

  gwn_status upload(const cuda::std::span<const Real> x,
                    const cuda::std::span<const Real> y,
                    const cuda::std::span<const Real> z,
                    const cuda::std::span<const Index> i0,
                    const cuda::std::span<const Index> i1,
                    const cuda::std::span<const Index> i2,
                    const cudaStream_t stream = 0) noexcept {
    gwn_geometry_object staging;
    gwn_status status = detail::gwn_upload_accessor(staging.accessor_, x, y, z,
                                                    i0, i1, i2, stream);
    if (!status.is_ok()) {
      staging.clear(stream);
      return status;
    }

    swap(*this, staging);
    return gwn_status::ok();
  }

  void clear(const cudaStream_t stream = 0) noexcept {
    detail::gwn_release_accessor(accessor_, stream);
  }

  [[nodiscard]] const accessor_type& accessor() const noexcept {
    return accessor_;
  }
  [[nodiscard]] std::size_t vertex_count() const noexcept {
    return accessor_.vertex_count();
  }
  [[nodiscard]] std::size_t triangle_count() const noexcept {
    return accessor_.triangle_count();
  }

  friend void swap(gwn_geometry_object& lhs,
                   gwn_geometry_object& rhs) noexcept {
    using std::swap;
    swap(lhs.accessor_, rhs.accessor_);
  }

 private:
  accessor_type accessor_{};
};

}  // namespace gwn
