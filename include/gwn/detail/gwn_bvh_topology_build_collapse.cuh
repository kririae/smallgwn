#pragma once

#include <cstddef>

#include "gwn_bvh_topology_build_binary.cuh"

namespace gwn {
namespace detail {
template <int Value> struct gwn_log2_pow2;

template <> struct gwn_log2_pow2<1> {
    static constexpr int value = 0;
};

template <int Value> struct gwn_log2_pow2 {
    static_assert(Value > 1 && (Value & (Value - 1)) == 0, "Value must be a power of two.");
    static constexpr int value = 1 + gwn_log2_pow2<(Value >> 1)>::value;
};

template <int Width, gwn_index_type Index> struct gwn_collapse_summarize_pass_functor {
    static constexpr int k_collapse_depth = gwn_log2_pow2<Width>::value;

    cuda::std::span<Index const> internal_parent{};
    cuda::std::span<std::uint8_t> internal_is_wide_root{};
    cuda::std::span<Index> internal_wide_node_id{};
    cuda::std::span<Index> wide_node_binary_root{};
    unsigned int *wide_node_count{};
    Index root_internal_index{};

    __device__ void operator()(std::size_t const internal_id_u) const {
        if (internal_id_u >= internal_parent.size() ||
            internal_id_u >= internal_is_wide_root.size() ||
            internal_id_u >= internal_wide_node_id.size() || wide_node_count == nullptr) {
            return;
        }

        auto const internal_id = static_cast<Index>(internal_id_u);
        if (internal_id == root_internal_index)
            return;

        int depth = 0;
        std::size_t hop_count = 0;
        Index cursor = internal_id;
        while (gwn_is_valid_index(cursor) && cursor != root_internal_index) {
            auto const cursor_u = static_cast<std::size_t>(cursor);
            if (cursor_u >= internal_parent.size())
                return;
            if (hop_count++ > internal_parent.size())
                return;
            ++depth;
            cursor = internal_parent[cursor_u];
        }
        if (cursor != root_internal_index)
            return;
        if ((depth % k_collapse_depth) != 0)
            return;

        unsigned int const wide_node_id = atomicAdd(wide_node_count, 1u);
        if (wide_node_id >= wide_node_binary_root.size())
            return;

        internal_is_wide_root[internal_id_u] = std::uint8_t(1);
        internal_wide_node_id[internal_id_u] = static_cast<Index>(wide_node_id);
        wide_node_binary_root[wide_node_id] = internal_id;
    }
};

template <int Width, gwn_index_type Index> struct gwn_collapse_emit_nodes_pass_functor {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static constexpr int k_stack_capacity = Width * 2 + 8;

    cuda::std::span<gwn_binary_node<Index> const> binary_nodes{};
    cuda::std::span<std::uint8_t const> internal_is_wide_root{};
    cuda::std::span<Index const> internal_wide_node_id{};
    cuda::std::span<Index const> wide_node_binary_root{};
    cuda::std::span<gwn_bvh_topology_node_soa<Width, Index>> output_nodes{};
    unsigned int *overflow_flag{};

    __device__ static void
    gwn_set_invalid_child(gwn_bvh_topology_node_soa<Width, Index> &node, int const slot) {
        node.child_index[slot] = Index(0);
        node.child_count[slot] = Index(0);
        node.child_kind[slot] = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
    }

    __device__ void operator()(std::size_t const wide_node_id_u) const {
        if (wide_node_id_u >= output_nodes.size() || wide_node_id_u >= wide_node_binary_root.size())
            return;

        gwn_bvh_topology_node_soa<Width, Index> output_node{};
        Index const binary_root = wide_node_binary_root[wide_node_id_u];
        if (!gwn_index_in_bounds(binary_root, binary_nodes.size())) {
            GWN_PRAGMA_UNROLL
            for (int slot = 0; slot < Width; ++slot)
                gwn_set_invalid_child(output_node, slot);
            output_nodes[wide_node_id_u] = output_node;
            if (overflow_flag != nullptr)
                atomicExch(overflow_flag, 1u);
            return;
        }

        gwn_binary_child_ref<Index> stack[k_stack_capacity]{};
        int stack_size = 0;
        stack[stack_size++] = gwn_binary_child_ref<Index>{
            static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal), binary_root
        };

        int written_children = 0;
        while (stack_size > 0) {
            gwn_binary_child_ref<Index> const ref = stack[--stack_size];
            auto const kind = static_cast<gwn_bvh_child_kind>(ref.kind);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;

            if (written_children >= Width) {
                if (overflow_flag != nullptr)
                    atomicExch(overflow_flag, 1u);
                break;
            }

            if (kind == gwn_bvh_child_kind::k_leaf) {
                int const slot = written_children++;
                output_node.child_index[slot] = ref.index;
                output_node.child_count[slot] = Index(1);
                output_node.child_kind[slot] =
                    static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
                continue;
            }

            if (kind != gwn_bvh_child_kind::k_internal ||
                !gwn_index_in_bounds(ref.index, binary_nodes.size())) {
                gwn_set_invalid_child(output_node, written_children++);
                if (overflow_flag != nullptr)
                    atomicExch(overflow_flag, 1u);
                continue;
            }

            auto const internal_index_u = static_cast<std::size_t>(ref.index);
            bool const is_child_wide_root = (ref.index != binary_root) &&
                                            (internal_index_u < internal_is_wide_root.size()) &&
                                            (internal_is_wide_root[internal_index_u] != 0);

            if (is_child_wide_root) {
                if (internal_index_u >= internal_wide_node_id.size()) {
                    gwn_set_invalid_child(output_node, written_children++);
                    if (overflow_flag != nullptr)
                        atomicExch(overflow_flag, 1u);
                    continue;
                }

                Index const child_wide_id = internal_wide_node_id[internal_index_u];
                if (!gwn_index_in_bounds(child_wide_id, output_nodes.size())) {
                    gwn_set_invalid_child(output_node, written_children++);
                    if (overflow_flag != nullptr)
                        atomicExch(overflow_flag, 1u);
                    continue;
                }

                int const slot = written_children++;
                output_node.child_index[slot] = child_wide_id;
                output_node.child_count[slot] = Index(0);
                output_node.child_kind[slot] =
                    static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
                continue;
            }

            gwn_binary_node<Index> const binary_node = binary_nodes[internal_index_u];
            if (stack_size + 2 > k_stack_capacity) {
                if (overflow_flag != nullptr)
                    atomicExch(overflow_flag, 1u);
                break;
            }
            stack[stack_size++] = binary_node.right;
            stack[stack_size++] = binary_node.left;
        }

        for (int slot = written_children; slot < Width; ++slot)
            gwn_set_invalid_child(output_node, slot);

        output_nodes[wide_node_id_u] = output_node;
    }
};

} // namespace detail
} // namespace gwn
