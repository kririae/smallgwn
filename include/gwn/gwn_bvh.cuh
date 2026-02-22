#pragma once

#include "gwn_utils.hpp"

#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace gwn {

enum class gwn_bvh_child_kind : std::uint8_t {
  k_invalid = 0,
  k_internal = 1,
  k_leaf = 2,
};

template <class Real>
struct gwn_aabb {
  Real min_x;
  Real min_y;
  Real min_z;
  Real max_x;
  Real max_y;
  Real max_z;
};

template <class Real, class Index = std::int64_t>
struct gwn_bvh4_node_soa {
  Real child_min_x[4];
  Real child_min_y[4];
  Real child_min_z[4];
  Real child_max_x[4];
  Real child_max_y[4];
  Real child_max_z[4];
  Index child_index[4];
  Index child_count[4];
  std::uint8_t child_kind[4];
};

template <class Real, class Index = std::int64_t>
struct gwn_bvh_accessor {
  using real_type = Real;
  using index_type = Index;

  cuda::std::span<const gwn_bvh4_node_soa<Real, Index>> nodes{};
  cuda::std::span<const Index> primitive_indices{};
  gwn_bvh_child_kind root_kind = gwn_bvh_child_kind::k_invalid;
  Index root_index = 0;
  Index root_count = 0;

  __host__ __device__ constexpr bool has_internal_root() const noexcept {
    return root_kind == gwn_bvh_child_kind::k_internal;
  }

  __host__ __device__ constexpr bool has_leaf_root() const noexcept {
    return root_kind == gwn_bvh_child_kind::k_leaf;
  }

  __host__ __device__ constexpr bool is_valid() const noexcept {
    if (root_kind == gwn_bvh_child_kind::k_invalid) {
      return false;
    }

    if (root_kind == gwn_bvh_child_kind::k_internal) {
      return !nodes.empty() && root_index >= 0 &&
             static_cast<std::size_t>(root_index) < nodes.size();
    }

    if (root_kind == gwn_bvh_child_kind::k_leaf) {
      return root_index >= 0 && root_count >= 0 &&
             (static_cast<std::size_t>(root_index + root_count) <=
              primitive_indices.size());
    }

    return false;
  }
};

namespace detail {

template <class T>
void gwn_release_bvh_span(cuda::std::span<const T>& span_view,
                          const cudaStream_t stream) noexcept {
  if (span_view.data() != nullptr) {
    (void)gwn_cuda_free(const_cast<T*>(span_view.data()), stream);
    span_view = {};
  }
}

template <class Real, class Index>
void gwn_release_bvh_accessor(gwn_bvh_accessor<Real, Index>& bvh,
                              const cudaStream_t stream) noexcept {
  gwn_release_bvh_span(bvh.primitive_indices, stream);
  gwn_release_bvh_span(bvh.nodes, stream);
  bvh.root_kind = gwn_bvh_child_kind::k_invalid;
  bvh.root_index = 0;
  bvh.root_count = 0;
}

}  // namespace detail

template <class Real = float, class Index = std::int64_t>
class gwn_bvh_object final : public gwn_noncopyable {
 public:
  using real_type = Real;
  using index_type = Index;
  using accessor_type = gwn_bvh_accessor<Real, Index>;

  gwn_bvh_object() = default;

  gwn_bvh_object(gwn_bvh_object&& other) noexcept { swap(*this, other); }

  gwn_bvh_object& operator=(gwn_bvh_object other) noexcept {
    swap(*this, other);
    return *this;
  }

  ~gwn_bvh_object() { clear(); }

  void clear(const cudaStream_t stream = gwn_default_stream()) noexcept {
    detail::gwn_release_bvh_accessor(accessor_, stream);
  }

  [[nodiscard]] accessor_type& accessor() noexcept { return accessor_; }
  [[nodiscard]] const accessor_type& accessor() const noexcept {
    return accessor_;
  }
  [[nodiscard]] bool has_bvh() const noexcept { return accessor_.is_valid(); }

  friend void swap(gwn_bvh_object& lhs, gwn_bvh_object& rhs) noexcept {
    using std::swap;
    swap(lhs.accessor_, rhs.accessor_);
  }

 private:
  accessor_type accessor_{};
};

}  // namespace gwn
