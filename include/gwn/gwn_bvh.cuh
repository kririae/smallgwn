#pragma once

#include <cuda/std/span>

#include <cstddef>
#include <cstdint>

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

}  // namespace gwn
