#pragma once

#include <cstddef>
#include <cstdint>

#include "gwn/detail/gwn_bvh_build_common.cuh"

namespace gwn {
namespace detail {
template <class Index> struct gwn_binary_child_ref {
    std::uint8_t kind{};
    Index index{};
};

template <class Index> struct gwn_binary_node {
    gwn_binary_child_ref<Index> left{};
    gwn_binary_child_ref<Index> right{};
};

template <class Index> struct gwn_build_binary_topology_functor {
    cuda::std::span<std::uint32_t const> morton_codes{};
    cuda::std::span<gwn_binary_node<Index>> binary_nodes{};
    cuda::std::span<Index> internal_parent{};
    cuda::std::span<std::uint8_t> internal_parent_slot{};
    cuda::std::span<Index> leaf_parent{};
    cuda::std::span<std::uint8_t> leaf_parent_slot{};

    [[nodiscard]] __device__ inline int delta(Index const i, Index const j) const noexcept {
        Index const leaf_count = static_cast<Index>(morton_codes.size());
        if (j < Index(0) || j >= leaf_count)
            return -1;

        std::uint32_t const code_i = morton_codes[static_cast<std::size_t>(i)];
        std::uint32_t const code_j = morton_codes[static_cast<std::size_t>(j)];
        if (code_i == code_j) {
            std::uint32_t const diff = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(i) ^ static_cast<std::uint64_t>(j)
            );
            if (diff == 0)
                return 64;
            return 32 + __clz(diff);
        }
        return __clz(code_i ^ code_j);
    }

    __device__ void operator()(std::size_t const internal_id_u) const {
        Index const internal_id = static_cast<Index>(internal_id_u);
        Index const leaf_count = static_cast<Index>(morton_codes.size());
        if (leaf_count <= Index(1))
            return;

        int const direction = (delta(internal_id, internal_id + Index(1)) -
                                   delta(internal_id, internal_id - Index(1)) >=
                               0)
                                  ? 1
                                  : -1;
        int const delta_min = delta(internal_id, internal_id - static_cast<Index>(direction));

        Index range_length_max = Index(2);
        while (delta(internal_id, internal_id + range_length_max * static_cast<Index>(direction)) >
               delta_min) {
            range_length_max <<= 1;
        }

        Index range_length = Index(0);
        for (Index step = range_length_max >> 1; step > 0; step >>= 1) {
            if (delta(
                    internal_id, internal_id + (range_length + step) * static_cast<Index>(direction)
                ) > delta_min) {
                range_length += step;
            }
        }

        Index const j = internal_id + range_length * static_cast<Index>(direction);
        Index const first = std::min(internal_id, j);
        Index const last = std::max(internal_id, j);

        int const delta_node = delta(first, last);
        Index split = first;
        Index step = last - first;
        do {
            step = (step + Index(1)) >> 1;
            Index const candidate = split + step;
            if (candidate < last && delta(first, candidate) > delta_node)
                split = candidate;
        } while (step > Index(1));

        gwn_binary_node<Index> node{};
        if (split == first) {
            node.left.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
            node.left.index = split;
        } else {
            node.left.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
            node.left.index = split;
        }

        if (split + Index(1) == last) {
            node.right.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
            node.right.index = split + Index(1);
        } else {
            node.right.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
            node.right.index = split + Index(1);
        }

        binary_nodes[internal_id_u] = node;
        if (node.left.kind == static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
            internal_parent[static_cast<std::size_t>(node.left.index)] = internal_id;
            internal_parent_slot[static_cast<std::size_t>(node.left.index)] = 0;
        } else {
            leaf_parent[static_cast<std::size_t>(node.left.index)] = internal_id;
            leaf_parent_slot[static_cast<std::size_t>(node.left.index)] = 0;
        }
        if (node.right.kind == static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
            internal_parent[static_cast<std::size_t>(node.right.index)] = internal_id;
            internal_parent_slot[static_cast<std::size_t>(node.right.index)] = 1;
        } else {
            leaf_parent[static_cast<std::size_t>(node.right.index)] = internal_id;
            leaf_parent_slot[static_cast<std::size_t>(node.right.index)] = 1;
        }
    }
};

template <class Real, class Index> struct gwn_accumulate_binary_bounds_pass_functor {
    cuda::std::span<gwn_aabb<Real> const> sorted_leaf_aabbs{};
    cuda::std::span<Index const> leaf_parent{};
    cuda::std::span<std::uint8_t const> leaf_parent_slot{};
    cuda::std::span<Index const> internal_parent{};
    cuda::std::span<std::uint8_t const> internal_parent_slot{};
    cuda::std::span<gwn_aabb<Real>> internal_bounds{};
    cuda::std::span<gwn_aabb<Real>> pending_left_bounds{};
    cuda::std::span<gwn_aabb<Real>> pending_right_bounds{};
    cuda::std::span<unsigned int> child_arrival_count{};

    __device__ void operator()(std::size_t const leaf_id) const {
        if (leaf_id >= sorted_leaf_aabbs.size() || leaf_id >= leaf_parent.size() ||
            leaf_id >= leaf_parent_slot.size()) {
            return;
        }

        Index parent = leaf_parent[leaf_id];
        if (parent < Index(0))
            return;

        std::uint8_t child_slot = leaf_parent_slot[leaf_id];
        if (child_slot > 1)
            return;

        gwn_aabb<Real> merged_bounds = sorted_leaf_aabbs[leaf_id];
        while (parent >= Index(0)) {
            std::size_t const parent_id = static_cast<std::size_t>(parent);
            if (parent_id >= internal_bounds.size() || parent_id >= pending_left_bounds.size() ||
                parent_id >= pending_right_bounds.size() ||
                parent_id >= child_arrival_count.size() || parent_id >= internal_parent.size() ||
                parent_id >= internal_parent_slot.size()) {
                return;
            }

            if (child_slot == 0)
                pending_left_bounds[parent_id] = merged_bounds;
            else
                pending_right_bounds[parent_id] = merged_bounds;

            __threadfence();
            unsigned int const prev_count = atomicAdd(child_arrival_count.data() + parent_id, 1u);
            if (prev_count == 0)
                return;

            gwn_aabb<Real> const left_bounds = pending_left_bounds[parent_id];
            gwn_aabb<Real> const right_bounds = pending_right_bounds[parent_id];
            merged_bounds = gwn_aabb_union(left_bounds, right_bounds);
            internal_bounds[parent_id] = merged_bounds;

            Index const next_parent = internal_parent[parent_id];
            if (next_parent < Index(0))
                return;

            child_slot = internal_parent_slot[parent_id];
            if (child_slot > 1)
                return;
            parent = next_parent;
        }
    }
};

} // namespace detail
} // namespace gwn
