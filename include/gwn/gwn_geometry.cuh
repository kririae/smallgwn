#pragma once

#include "gwn_bvh.cuh"
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
  gwn_bvh_accessor<Real, Index> bvh{};

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

  __host__ __device__ constexpr bool has_bvh() const noexcept {
    return bvh.is_valid();
  }
};

template <class Real, class Index>
gwn_status gwn_build_bvh4_lbvh(
    gwn_geometry_accessor<Real, Index>& accessor,
    cudaStream_t stream = gwn_default_stream()) noexcept;

namespace detail {

template <class T>
[[nodiscard]] constexpr T* gwn_mutable_data(
    const cuda::std::span<const T> span) noexcept {
  return const_cast<T*>(span.data());
}

template <class T>
gwn_status gwn_upload_span(cuda::std::span<const T>& dst,
                           cuda::std::span<const T> src,
                           const cudaStream_t stream) {
  dst = {};
  if (src.empty()) {
    return gwn_status::ok();
  }

  void* device_ptr = nullptr;
  gwn_status status = gwn_cuda_malloc(&device_ptr, src.size_bytes(), stream);
  if (!status.is_ok()) {
    return status;
  }

  status = gwn_cuda_to_status(cudaMemcpyAsync(device_ptr, src.data(),
                                              src.size_bytes(),
                                              cudaMemcpyHostToDevice, stream));
  if (!status.is_ok()) {
    (void)gwn_cuda_free(device_ptr, stream);
    return status;
  }

  status = gwn_cuda_to_status(cudaStreamSynchronize(stream));
  if (!status.is_ok()) {
    (void)gwn_cuda_free(device_ptr, stream);
    return status;
  }

  dst = cuda::std::span<const T>(static_cast<const T*>(device_ptr), src.size());
  return gwn_status::ok();
}

template <class T>
gwn_status gwn_allocate_span(cuda::std::span<const T>& dst,
                             const std::size_t count,
                             const cudaStream_t stream) {
  if (dst.data() != nullptr) {
    (void)gwn_cuda_free(gwn_mutable_data(dst), stream);
    dst = {};
  }

  if (count == 0) {
    return gwn_status::ok();
  }

  void* raw_ptr = nullptr;
  gwn_status status = gwn_cuda_malloc(&raw_ptr, count * sizeof(T), stream);
  if (!status.is_ok()) {
    return status;
  }

  dst = cuda::std::span<const T>(static_cast<const T*>(raw_ptr), count);
  return gwn_status::ok();
}

template <class T>
gwn_status gwn_copy_device_to_span(cuda::std::span<const T>& dst,
                                   const T* src,
                                   const std::size_t count,
                                   const cudaStream_t stream) {
  gwn_status status = gwn_allocate_span(dst, count, stream);
  if (!status.is_ok() || count == 0) {
    return status;
  }

  status = gwn_cuda_to_status(
      cudaMemcpyAsync(gwn_mutable_data(dst), src, count * sizeof(T),
                      cudaMemcpyDeviceToDevice, stream));
  if (!status.is_ok()) {
    (void)gwn_cuda_free(gwn_mutable_data(dst), stream);
    dst = {};
    return status;
  }

  return gwn_cuda_to_status(cudaStreamSynchronize(stream));
}

template <class T>
void gwn_release_span(cuda::std::span<const T>& span_view,
                      const cudaStream_t stream) noexcept {
  if (span_view.data() != nullptr) {
    (void)gwn_cuda_free(gwn_mutable_data(span_view), stream);
    span_view = {};
  }
}

template <class... Spans>
void gwn_release_spans(const cudaStream_t stream, Spans&... spans) noexcept {
  (gwn_release_span(spans, stream), ...);
}

template <class Real, class Index>
void gwn_release_bvh(gwn_bvh_accessor<Real, Index>& bvh,
                     const cudaStream_t stream) noexcept {
  gwn_release_spans(stream, bvh.primitive_indices, bvh.nodes);
  bvh.root_kind = gwn_bvh_child_kind::k_invalid;
  bvh.root_index = 0;
  bvh.root_count = 0;
}

template <class Real, class Index>
void gwn_release_accessor(gwn_geometry_accessor<Real, Index>& accessor,
                          const cudaStream_t stream) noexcept {
  gwn_release_bvh(accessor.bvh, stream);
  gwn_release_spans(stream, accessor.tri_i2, accessor.tri_i1, accessor.tri_i0,
                    accessor.vertex_z, accessor.vertex_y, accessor.vertex_x);
}

template <class Real, class Index>
gwn_status gwn_upload_accessor(gwn_geometry_accessor<Real, Index>& accessor,
                               cuda::std::span<const Real> x,
                               cuda::std::span<const Real> y,
                               cuda::std::span<const Real> z,
                               cuda::std::span<const Index> i0,
                               cuda::std::span<const Index> i1,
                               cuda::std::span<const Index> i2,
                               const cudaStream_t stream) {
  if (x.size() != y.size() || x.size() != z.size()) {
    return gwn_status::invalid_argument(
        "Vertex SoA spans must have identical lengths.");
  }
  if (i0.size() != i1.size() || i0.size() != i2.size()) {
    return gwn_status::invalid_argument(
        "Triangle SoA spans must have identical lengths.");
  }

  const auto upload_one = [&](auto& dst, auto src) {
    return gwn_upload_span(dst, src, stream);
  };

  if (auto status = upload_one(accessor.vertex_x, x); !status.is_ok()) {
    return status;
  }
  if (auto status = upload_one(accessor.vertex_y, y); !status.is_ok()) {
    return status;
  }
  if (auto status = upload_one(accessor.vertex_z, z); !status.is_ok()) {
    return status;
  }
  if (auto status = upload_one(accessor.tri_i0, i0); !status.is_ok()) {
    return status;
  }
  if (auto status = upload_one(accessor.tri_i1, i1); !status.is_ok()) {
    return status;
  }

  return upload_one(accessor.tri_i2, i2);
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
                      const cudaStream_t stream = gwn_default_stream()) {
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
                    const cudaStream_t stream = gwn_default_stream()) noexcept {
    gwn_geometry_object staging;
    if (auto status = detail::gwn_upload_accessor(staging.accessor_, x, y, z,
                                                  i0, i1, i2, stream);
        !status.is_ok()) {
      staging.clear(stream);
      return status;
    }

    swap(*this, staging);
    return gwn_status::ok();
  }

  void clear(const cudaStream_t stream = gwn_default_stream()) noexcept {
    detail::gwn_release_accessor(accessor_, stream);
  }

  void clear_bvh(const cudaStream_t stream = gwn_default_stream()) noexcept {
    detail::gwn_release_bvh(accessor_.bvh, stream);
  }

  gwn_status build_bvh(
      const cudaStream_t stream = gwn_default_stream()) noexcept;

  [[nodiscard]] const accessor_type& accessor() const noexcept {
    return accessor_;
  }
  [[nodiscard]] std::size_t vertex_count() const noexcept {
    return accessor_.vertex_count();
  }
  [[nodiscard]] std::size_t triangle_count() const noexcept {
    return accessor_.triangle_count();
  }
  [[nodiscard]] const gwn_bvh_accessor<Real, Index>& bvh_accessor()
      const noexcept {
    return accessor_.bvh;
  }
  [[nodiscard]] bool has_bvh() const noexcept { return accessor_.has_bvh(); }

  friend void swap(gwn_geometry_object& lhs,
                   gwn_geometry_object& rhs) noexcept {
    using std::swap;
    swap(lhs.accessor_, rhs.accessor_);
  }

 private:
  accessor_type accessor_{};
};

template <class Real, class Index>
gwn_status gwn_geometry_object<Real, Index>::build_bvh(
    const cudaStream_t stream) noexcept {
  return gwn_build_bvh4_lbvh<Real, Index>(accessor_, stream);
}

}  // namespace gwn
