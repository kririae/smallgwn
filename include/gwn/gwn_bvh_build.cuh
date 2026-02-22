#pragma once

#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"

#include <cuda_runtime_api.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/memory.h>
#include <thrust/sort.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gwn {
namespace detail {

template <class Real, class Index>
struct gwn_build_entry {
  gwn_aabb<Real> bounds;
  std::uint8_t kind;
  Index index;
  Index count;
};

template <class Real>
__host__ __device__ inline Real gwn_clamp01(const Real value) noexcept {
  if (value < Real(0)) {
    return Real(0);
  }
  if (value > Real(1)) {
    return Real(1);
  }
  return value;
}

__host__ __device__ inline std::uint32_t gwn_expand_bits_10_to_30(
    const std::uint32_t value) noexcept {
  std::uint32_t x = value & 0x000003ffu;
  x = (x | (x << 16)) & 0x030000ffu;
  x = (x | (x << 8)) & 0x0300f00fu;
  x = (x | (x << 4)) & 0x030c30c3u;
  x = (x | (x << 2)) & 0x09249249u;
  return x;
}

template <class Real>
__host__ __device__ inline std::uint32_t
gwn_encode_morton_30(const Real nx, const Real ny, const Real nz) noexcept {
  const auto x = static_cast<std::uint32_t>(gwn_clamp01(nx) * Real(1023));
  const auto y = static_cast<std::uint32_t>(gwn_clamp01(ny) * Real(1023));
  const auto z = static_cast<std::uint32_t>(gwn_clamp01(nz) * Real(1023));
  return (gwn_expand_bits_10_to_30(x) << 2) |
         (gwn_expand_bits_10_to_30(y) << 1) | gwn_expand_bits_10_to_30(z);
}

template <class Real>
__host__ __device__ inline gwn_aabb<Real> gwn_aabb_union(
    const gwn_aabb<Real>& left,
    const gwn_aabb<Real>& right) noexcept {
  return gwn_aabb<Real>{
      std::min(left.min_x, right.min_x), std::min(left.min_y, right.min_y),
      std::min(left.min_z, right.min_z), std::max(left.max_x, right.max_x),
      std::max(left.max_y, right.max_y), std::max(left.max_z, right.max_z)};
}

template <class Real, class Index>
struct gwn_compute_triangle_aabbs_and_morton_functor {
  gwn_geometry_accessor<Real, Index> geometry{};
  Real scene_min_x{};
  Real scene_min_y{};
  Real scene_min_z{};
  Real scene_inv_x{};
  Real scene_inv_y{};
  Real scene_inv_z{};
  cuda::std::span<gwn_aabb<Real>> primitive_aabbs{};
  cuda::std::span<std::uint32_t> morton_codes{};
  cuda::std::span<Index> primitive_indices{};

  __device__ void operator()(const std::size_t triangle_id) const {
    primitive_indices[triangle_id] = static_cast<Index>(triangle_id);

    const Index ia = geometry.tri_i0[triangle_id];
    const Index ib = geometry.tri_i1[triangle_id];
    const Index ic = geometry.tri_i2[triangle_id];
    if (ia < Index(0) || ib < Index(0) || ic < Index(0) ||
        static_cast<std::size_t>(ia) >= geometry.vertex_count() ||
        static_cast<std::size_t>(ib) >= geometry.vertex_count() ||
        static_cast<std::size_t>(ic) >= geometry.vertex_count()) {
      primitive_aabbs[triangle_id] =
          gwn_aabb<Real>{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
      morton_codes[triangle_id] = 0;
      return;
    }

    const std::size_t a = static_cast<std::size_t>(ia);
    const std::size_t b = static_cast<std::size_t>(ib);
    const std::size_t c = static_cast<std::size_t>(ic);

    const Real ax = geometry.vertex_x[a];
    const Real ay = geometry.vertex_y[a];
    const Real az = geometry.vertex_z[a];
    const Real bx = geometry.vertex_x[b];
    const Real by = geometry.vertex_y[b];
    const Real bz = geometry.vertex_z[b];
    const Real cx = geometry.vertex_x[c];
    const Real cy = geometry.vertex_y[c];
    const Real cz = geometry.vertex_z[c];

    const gwn_aabb<Real> bounds{
        std::min(ax, std::min(bx, cx)), std::min(ay, std::min(by, cy)),
        std::min(az, std::min(bz, cz)), std::max(ax, std::max(bx, cx)),
        std::max(ay, std::max(by, cy)), std::max(az, std::max(bz, cz)),
    };
    primitive_aabbs[triangle_id] = bounds;

    const Real center_x = (bounds.min_x + bounds.max_x) * Real(0.5);
    const Real center_y = (bounds.min_y + bounds.max_y) * Real(0.5);
    const Real center_z = (bounds.min_z + bounds.max_z) * Real(0.5);
    morton_codes[triangle_id] =
        gwn_encode_morton_30((center_x - scene_min_x) * scene_inv_x,
                             (center_y - scene_min_y) * scene_inv_y,
                             (center_z - scene_min_z) * scene_inv_z);
  }
};

template <class Real, class Index>
struct gwn_gather_sorted_aabbs_functor {
  cuda::std::span<const gwn_aabb<Real>> unsorted_aabbs{};
  cuda::std::span<const Index> sorted_primitive_indices{};
  cuda::std::span<gwn_aabb<Real>> sorted_aabbs{};

  __device__ void operator()(const std::size_t primitive_id) const {
    const std::size_t source_id =
        static_cast<std::size_t>(sorted_primitive_indices[primitive_id]);
    sorted_aabbs[primitive_id] = unsorted_aabbs[source_id];
  }
};

template <class Real, class Index>
struct gwn_build_leaf_entries_functor {
  cuda::std::span<const gwn_aabb<Real>> sorted_aabbs{};
  std::size_t primitive_count{};
  std::size_t leaf_primitive_capacity{};
  cuda::std::span<gwn_build_entry<Real, Index>> leaf_entries{};

  __device__ void operator()(const std::size_t leaf_id) const {
    const std::size_t begin = leaf_id * leaf_primitive_capacity;
    const std::size_t end =
        std::min(begin + leaf_primitive_capacity, primitive_count);

    gwn_aabb<Real> bounds = sorted_aabbs[begin];
    for (std::size_t primitive_id = begin + 1; primitive_id < end;
         ++primitive_id) {
      bounds = gwn_aabb_union(bounds, sorted_aabbs[primitive_id]);
    }

    gwn_build_entry<Real, Index> entry{};
    entry.bounds = bounds;
    entry.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
    entry.index = static_cast<Index>(begin);
    entry.count = static_cast<Index>(end - begin);
    leaf_entries[leaf_id] = entry;
  }
};

template <class Real, class Index>
struct gwn_build_bvh4_parent_level_functor {
  cuda::std::span<const gwn_build_entry<Real, Index>> child_entries{};
  std::size_t child_count{};
  cuda::std::span<gwn_build_entry<Real, Index>> parent_entries{};
  cuda::std::span<gwn_bvh4_node_soa<Real, Index>> parent_nodes{};

  __device__ void operator()(const std::size_t parent_id) const {
    gwn_bvh4_node_soa<Real, Index> node{};
    gwn_aabb<Real> parent_bounds{};
    bool has_child = false;

    for (int child_slot = 0; child_slot < 4; ++child_slot) {
      const std::size_t child_id =
          parent_id * 4 + static_cast<std::size_t>(child_slot);
      if (child_id >= child_count) {
        node.child_min_x[child_slot] = Real(0);
        node.child_min_y[child_slot] = Real(0);
        node.child_min_z[child_slot] = Real(0);
        node.child_max_x[child_slot] = Real(0);
        node.child_max_y[child_slot] = Real(0);
        node.child_max_z[child_slot] = Real(0);
        node.child_index[child_slot] = 0;
        node.child_count[child_slot] = 0;
        node.child_kind[child_slot] =
            static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
        continue;
      }

      const gwn_build_entry<Real, Index> child = child_entries[child_id];
      node.child_min_x[child_slot] = child.bounds.min_x;
      node.child_min_y[child_slot] = child.bounds.min_y;
      node.child_min_z[child_slot] = child.bounds.min_z;
      node.child_max_x[child_slot] = child.bounds.max_x;
      node.child_max_y[child_slot] = child.bounds.max_y;
      node.child_max_z[child_slot] = child.bounds.max_z;
      node.child_index[child_slot] = child.index;
      node.child_count[child_slot] = child.count;
      node.child_kind[child_slot] = child.kind;

      if (!has_child) {
        parent_bounds = child.bounds;
        has_child = true;
      } else {
        parent_bounds = gwn_aabb_union(parent_bounds, child.bounds);
      }
    }

    parent_nodes[parent_id] = node;

    gwn_build_entry<Real, Index> parent{};
    parent.bounds = parent_bounds;
    parent.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
    parent.index = static_cast<Index>(parent_id);
    parent.count = 0;
    parent_entries[parent_id] = parent;
  }
};

template <class Real, class Index>
struct gwn_patch_child_indices_functor {
  cuda::std::span<gwn_bvh4_node_soa<Real, Index>> nodes{};
  Index child_level_offset{};

  __device__ void operator()(const std::size_t node_id) const {
    gwn_bvh4_node_soa<Real, Index>& node = nodes[node_id];
    for (int child_slot = 0; child_slot < 4; ++child_slot) {
      if (node.child_kind[child_slot] ==
          static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
        node.child_index[child_slot] += child_level_offset;
      }
    }
  }
};

}  // namespace detail

template <class Real, class Index>
gwn_status gwn_build_bvh4_lbvh(gwn_geometry_accessor<Real, Index>& accessor,
                               cudaStream_t stream) noexcept {
  if (!accessor.is_valid()) {
    return gwn_status::invalid_argument(
        "Geometry accessor is invalid for BVH construction.");
  }

  detail::gwn_release_bvh(accessor.bvh, stream);
  if (accessor.triangle_count() == 0) {
    return gwn_status::ok();
  }
  if (accessor.vertex_count() == 0) {
    return gwn_status::invalid_argument(
        "Cannot build BVH with triangles but zero vertices.");
  }

  const std::size_t primitive_count = accessor.triangle_count();
  constexpr std::size_t k_leaf_primitive_capacity = 4;
  constexpr int k_block_size = detail::k_gwn_default_block_size;

  auto exec = thrust::cuda::par.on(stream);
  thrust::device_ptr<const Real> vx_ptr(accessor.vertex_x.data());
  thrust::device_ptr<const Real> vy_ptr(accessor.vertex_y.data());
  thrust::device_ptr<const Real> vz_ptr(accessor.vertex_z.data());
  auto x_pair =
      thrust::minmax_element(exec, vx_ptr, vx_ptr + accessor.vertex_count());
  auto y_pair =
      thrust::minmax_element(exec, vy_ptr, vy_ptr + accessor.vertex_count());
  auto z_pair =
      thrust::minmax_element(exec, vz_ptr, vz_ptr + accessor.vertex_count());

  Real scene_min_x = Real(0);
  Real scene_max_x = Real(0);
  Real scene_min_y = Real(0);
  Real scene_max_y = Real(0);
  Real scene_min_z = Real(0);
  Real scene_max_z = Real(0);
  gwn_status status = gwn_cuda_to_status(
      cudaMemcpyAsync(&scene_min_x, thrust::raw_pointer_cast(x_pair.first),
                      sizeof(Real), cudaMemcpyDeviceToHost, stream));
  if (!status.is_ok()) {
    return status;
  }
  status = gwn_cuda_to_status(
      cudaMemcpyAsync(&scene_max_x, thrust::raw_pointer_cast(x_pair.second),
                      sizeof(Real), cudaMemcpyDeviceToHost, stream));
  if (!status.is_ok()) {
    return status;
  }
  status = gwn_cuda_to_status(
      cudaMemcpyAsync(&scene_min_y, thrust::raw_pointer_cast(y_pair.first),
                      sizeof(Real), cudaMemcpyDeviceToHost, stream));
  if (!status.is_ok()) {
    return status;
  }
  status = gwn_cuda_to_status(
      cudaMemcpyAsync(&scene_max_y, thrust::raw_pointer_cast(y_pair.second),
                      sizeof(Real), cudaMemcpyDeviceToHost, stream));
  if (!status.is_ok()) {
    return status;
  }
  status = gwn_cuda_to_status(
      cudaMemcpyAsync(&scene_min_z, thrust::raw_pointer_cast(z_pair.first),
                      sizeof(Real), cudaMemcpyDeviceToHost, stream));
  if (!status.is_ok()) {
    return status;
  }
  status = gwn_cuda_to_status(
      cudaMemcpyAsync(&scene_max_z, thrust::raw_pointer_cast(z_pair.second),
                      sizeof(Real), cudaMemcpyDeviceToHost, stream));
  if (!status.is_ok()) {
    return status;
  }
  status = gwn_cuda_to_status(cudaStreamSynchronize(stream));
  if (!status.is_ok()) {
    return status;
  }

  const Real scene_inv_x = (scene_max_x > scene_min_x)
                               ? Real(1) / (scene_max_x - scene_min_x)
                               : Real(1);
  const Real scene_inv_y = (scene_max_y > scene_min_y)
                               ? Real(1) / (scene_max_y - scene_min_y)
                               : Real(1);
  const Real scene_inv_z = (scene_max_z > scene_min_z)
                               ? Real(1) / (scene_max_z - scene_min_z)
                               : Real(1);

  thrust::device_vector<gwn_aabb<Real>> primitive_aabbs(primitive_count);
  thrust::device_vector<std::uint32_t> morton_codes(primitive_count);
  thrust::device_vector<Index> sorted_primitive_indices(primitive_count);
  const auto primitive_aabbs_span = cuda::std::span<gwn_aabb<Real>>(
      thrust::raw_pointer_cast(primitive_aabbs.data()), primitive_count);
  const auto morton_codes_span = cuda::std::span<std::uint32_t>(
      thrust::raw_pointer_cast(morton_codes.data()), primitive_count);
  const auto sorted_primitive_indices_span = cuda::std::span<Index>(
      thrust::raw_pointer_cast(sorted_primitive_indices.data()),
      primitive_count);
  status = detail::gwn_launch_linear_kernel<k_block_size>(
      primitive_count,
      detail::gwn_compute_triangle_aabbs_and_morton_functor<Real, Index>{
          accessor, scene_min_x, scene_min_y, scene_min_z, scene_inv_x,
          scene_inv_y, scene_inv_z, primitive_aabbs_span, morton_codes_span,
          sorted_primitive_indices_span},
      stream);
  if (!status.is_ok()) {
    return status;
  }

  thrust::stable_sort_by_key(exec, morton_codes.begin(), morton_codes.end(),
                             sorted_primitive_indices.begin());

  thrust::device_vector<gwn_aabb<Real>> sorted_aabbs(primitive_count);
  const auto sorted_aabbs_span = cuda::std::span<gwn_aabb<Real>>(
      thrust::raw_pointer_cast(sorted_aabbs.data()), primitive_count);
  status = detail::gwn_launch_linear_kernel<k_block_size>(
      primitive_count,
      detail::gwn_gather_sorted_aabbs_functor<Real, Index>{
          primitive_aabbs_span, sorted_primitive_indices_span,
          sorted_aabbs_span},
      stream);
  if (!status.is_ok()) {
    return status;
  }

  status = detail::gwn_copy_device_to_span(accessor.bvh.primitive_indices,
                                           sorted_primitive_indices_span.data(),
                                           primitive_count, stream);
  if (!status.is_ok()) {
    detail::gwn_release_bvh(accessor.bvh, stream);
    return status;
  }

  const std::size_t leaf_count =
      (primitive_count + k_leaf_primitive_capacity - 1) /
      k_leaf_primitive_capacity;
  thrust::device_vector<detail::gwn_build_entry<Real, Index>> current_entries(
      leaf_count);
  auto current_entries_span =
      cuda::std::span<detail::gwn_build_entry<Real, Index>>(
          thrust::raw_pointer_cast(current_entries.data()), leaf_count);
  status = detail::gwn_launch_linear_kernel<k_block_size>(
      leaf_count,
      detail::gwn_build_leaf_entries_functor<Real, Index>{
          sorted_aabbs_span, primitive_count, k_leaf_primitive_capacity,
          current_entries_span},
      stream);
  if (!status.is_ok()) {
    detail::gwn_release_bvh(accessor.bvh, stream);
    return status;
  }

  std::vector<thrust::device_vector<gwn_bvh4_node_soa<Real, Index>>>
      levels_bottom;
  std::size_t current_count = leaf_count;
  while (current_count > 1) {
    const std::size_t parent_count = (current_count + 3) / 4;
    thrust::device_vector<detail::gwn_build_entry<Real, Index>> parent_entries(
        parent_count);
    thrust::device_vector<gwn_bvh4_node_soa<Real, Index>> parent_nodes(
        parent_count);

    const auto current_entries_const_span =
        cuda::std::span<const detail::gwn_build_entry<Real, Index>>(
            thrust::raw_pointer_cast(current_entries.data()), current_count);
    const auto parent_entries_span =
        cuda::std::span<detail::gwn_build_entry<Real, Index>>(
            thrust::raw_pointer_cast(parent_entries.data()), parent_count);
    const auto parent_nodes_span =
        cuda::std::span<gwn_bvh4_node_soa<Real, Index>>(
            thrust::raw_pointer_cast(parent_nodes.data()), parent_count);

    status = detail::gwn_launch_linear_kernel<k_block_size>(
        parent_count,
        detail::gwn_build_bvh4_parent_level_functor<Real, Index>{
            current_entries_const_span, current_count, parent_entries_span,
            parent_nodes_span},
        stream);
    if (!status.is_ok()) {
      detail::gwn_release_bvh(accessor.bvh, stream);
      return status;
    }

    levels_bottom.push_back(std::move(parent_nodes));
    current_entries = std::move(parent_entries);
    current_count = parent_count;
  }

  if (levels_bottom.empty()) {
    accessor.bvh.root_kind = gwn_bvh_child_kind::k_leaf;
    accessor.bvh.root_index = 0;
    accessor.bvh.root_count = static_cast<Index>(primitive_count);
    return gwn_status::ok();
  }

  const std::size_t level_count = levels_bottom.size();
  std::vector<std::size_t> level_node_counts(level_count, 0);
  std::vector<std::size_t> level_offsets(level_count, 0);
  std::size_t total_node_count = 0;
  for (std::size_t level = 0; level < level_count; ++level) {
    const std::size_t bottom_index = level_count - 1 - level;
    level_node_counts[level] = levels_bottom[bottom_index].size();
    level_offsets[level] = total_node_count;
    total_node_count += level_node_counts[level];
  }

  status =
      detail::gwn_allocate_span(accessor.bvh.nodes, total_node_count, stream);
  if (!status.is_ok()) {
    detail::gwn_release_bvh(accessor.bvh, stream);
    return status;
  }

  gwn_bvh4_node_soa<Real, Index>* final_nodes =
      const_cast<gwn_bvh4_node_soa<Real, Index>*>(accessor.bvh.nodes.data());
  auto final_nodes_span = cuda::std::span<gwn_bvh4_node_soa<Real, Index>>(
      final_nodes, total_node_count);
  for (std::size_t level = 0; level < level_count; ++level) {
    const std::size_t bottom_index = level_count - 1 - level;
    status = gwn_cuda_to_status(cudaMemcpyAsync(
        final_nodes + level_offsets[level],
        thrust::raw_pointer_cast(levels_bottom[bottom_index].data()),
        level_node_counts[level] * sizeof(gwn_bvh4_node_soa<Real, Index>),
        cudaMemcpyDeviceToDevice, stream));
    if (!status.is_ok()) {
      detail::gwn_release_bvh(accessor.bvh, stream);
      return status;
    }
  }

  for (std::size_t level = 0; level + 1 < level_count; ++level) {
    status = detail::gwn_launch_linear_kernel<k_block_size>(
        level_node_counts[level],
        detail::gwn_patch_child_indices_functor<Real, Index>{
            final_nodes_span.subspan(level_offsets[level],
                                     level_node_counts[level]),
            static_cast<Index>(level_offsets[level + 1])},
        stream);
    if (!status.is_ok()) {
      detail::gwn_release_bvh(accessor.bvh, stream);
      return status;
    }
  }

  status = gwn_cuda_to_status(cudaStreamSynchronize(stream));
  if (!status.is_ok()) {
    detail::gwn_release_bvh(accessor.bvh, stream);
    return status;
  }

  accessor.bvh.root_kind = gwn_bvh_child_kind::k_internal;
  accessor.bvh.root_index = 0;
  accessor.bvh.root_count = 0;
  return gwn_status::ok();
}

}  // namespace gwn
