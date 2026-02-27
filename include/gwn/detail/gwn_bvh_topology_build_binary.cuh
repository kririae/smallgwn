#pragma once

#include <cstddef>
#include <cstdint>

#include "gwn_bvh_topology_build_common.cuh"

namespace gwn {
namespace detail {
template <gwn_index_type Index> struct gwn_binary_child_ref {
    Index index{};
    std::uint8_t kind{};
};

template <gwn_index_type Index> struct gwn_binary_node {
    gwn_binary_child_ref<Index> left{};
    gwn_binary_child_ref<Index> right{};
};

template <gwn_index_type Index, class MortonCode> struct gwn_build_binary_topology_functor {
    cuda::std::span<MortonCode const> morton_codes{};
    cuda::std::span<gwn_binary_node<Index>> binary_nodes{};
    cuda::std::span<Index> internal_parent{};
    cuda::std::span<std::uint8_t> internal_parent_slot{};
    cuda::std::span<Index> leaf_parent{};
    cuda::std::span<std::uint8_t> leaf_parent_slot{};

    [[nodiscard]] __device__ inline int
    delta(std::int64_t const i, std::int64_t const j) const noexcept {
        auto const leaf_count = static_cast<std::int64_t>(morton_codes.size());
        if (j < 0 || j >= leaf_count)
            return -1;

        MortonCode const code_i = morton_codes[static_cast<std::size_t>(i)];
        MortonCode const code_j = morton_codes[static_cast<std::size_t>(j)];
        if (code_i == code_j) {
            std::uint64_t const diff =
                static_cast<std::uint64_t>(i) ^ static_cast<std::uint64_t>(j);
            if (diff == 0)
                return static_cast<int>(sizeof(MortonCode) * 8 + 64);
            return static_cast<int>(sizeof(MortonCode) * 8) + __clzll(diff);
        }

        if constexpr (sizeof(MortonCode) == sizeof(std::uint32_t))
            return __clz(static_cast<std::uint32_t>(code_i ^ code_j));
        else
            return __clzll(static_cast<std::uint64_t>(code_i ^ code_j));
    }

    __device__ void operator()(std::size_t const internal_id_u) const {
        auto const internal_id = static_cast<std::int64_t>(internal_id_u);
        auto const leaf_count = static_cast<std::int64_t>(morton_codes.size());
        if (leaf_count <= 1)
            return;
        auto const internal_id_index = static_cast<Index>(internal_id);

        int const direction =
            (delta(internal_id, internal_id + 1) - delta(internal_id, internal_id - 1) >= 0) ? 1
                                                                                             : -1;
        int const delta_min = delta(internal_id, internal_id - direction);

        std::int64_t range_length_max = 2;
        while (delta(internal_id, internal_id + range_length_max * direction) > delta_min)
            range_length_max <<= 1;

        std::int64_t range_length = 0;
        for (std::int64_t step = range_length_max >> 1; step > 0; step >>= 1)
            if (delta(internal_id, internal_id + (range_length + step) * direction) > delta_min)
                range_length += step;

        std::int64_t const j = internal_id + range_length * direction;
        std::int64_t const first = std::min(internal_id, j);
        std::int64_t const last = std::max(internal_id, j);

        int const delta_node = delta(first, last);
        std::int64_t split = first;
        std::int64_t step = last - first;
        do {
            step = (step + 1) >> 1;
            std::int64_t const candidate = split + step;
            if (candidate < last && delta(first, candidate) > delta_node)
                split = candidate;
        } while (step > 1);

        gwn_binary_node<Index> node{};
        auto const split_index = static_cast<Index>(split);
        auto const split_plus_one_index = static_cast<Index>(split + 1);
        if (split == first) {
            node.left.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
            node.left.index = split_index;
        } else {
            node.left.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
            node.left.index = split_index;
        }

        if ((split + 1) == last) {
            node.right.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
            node.right.index = split_plus_one_index;
        } else {
            node.right.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
            node.right.index = split_plus_one_index;
        }

        binary_nodes[internal_id_u] = node;
        if (node.left.kind == static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
            internal_parent[static_cast<std::size_t>(node.left.index)] = internal_id_index;
            internal_parent_slot[static_cast<std::size_t>(node.left.index)] = 0;
        } else {
            leaf_parent[static_cast<std::size_t>(node.left.index)] = internal_id_index;
            leaf_parent_slot[static_cast<std::size_t>(node.left.index)] = 0;
        }
        if (node.right.kind == static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
            internal_parent[static_cast<std::size_t>(node.right.index)] = internal_id_index;
            internal_parent_slot[static_cast<std::size_t>(node.right.index)] = 1;
        } else {
            leaf_parent[static_cast<std::size_t>(node.right.index)] = internal_id_index;
            leaf_parent_slot[static_cast<std::size_t>(node.right.index)] = 1;
        }
    }
};

template <gwn_real_type Real, gwn_index_type Index>
struct gwn_accumulate_binary_bounds_pass_functor {
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
        if (gwn_is_invalid_index(parent))
            return;

        std::uint8_t child_slot = leaf_parent_slot[leaf_id];
        if (child_slot > 1)
            return;

        gwn_aabb<Real> merged_bounds = sorted_leaf_aabbs[leaf_id];
        while (gwn_is_valid_index(parent)) {
            auto const parent_id = static_cast<std::size_t>(parent);
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
            if (gwn_is_invalid_index(next_parent))
                return;

            child_slot = internal_parent_slot[parent_id];
            if (child_slot > 1)
                return;
            parent = next_parent;
        }
    }
};

template <gwn_index_type Index> struct gwn_binary_parent_temporaries {
    gwn_device_array<std::uint8_t> internal_parent_slot;
    gwn_device_array<Index> leaf_parent;
    gwn_device_array<std::uint8_t> leaf_parent_slot;
};

template <gwn_index_type Index>
gwn_status gwn_prepare_binary_topology_buffers(
    std::size_t const primitive_count, gwn_device_array<gwn_binary_node<Index>> &binary_nodes,
    gwn_device_array<Index> &binary_internal_parent, gwn_binary_parent_temporaries<Index> &temps,
    cudaStream_t const stream
) noexcept {
    std::size_t const internal_count = primitive_count - 1;
    GWN_RETURN_ON_ERROR(binary_nodes.resize(internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_internal_parent.resize(internal_count, stream));
    GWN_RETURN_ON_ERROR(temps.internal_parent_slot.resize(internal_count, stream));
    GWN_RETURN_ON_ERROR(temps.leaf_parent.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(temps.leaf_parent_slot.resize(primitive_count, stream));

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(binary_internal_parent.data(), 0xff, internal_count * sizeof(Index), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        temps.internal_parent_slot.data(), 0xff, internal_count * sizeof(std::uint8_t), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(temps.leaf_parent.data(), 0xff, primitive_count * sizeof(Index), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        temps.leaf_parent_slot.data(), 0xff, primitive_count * sizeof(std::uint8_t), stream
    )));

    return gwn_status::ok();
}

} // namespace detail
} // namespace gwn
