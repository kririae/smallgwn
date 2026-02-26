#pragma once

#include <cuda/atomic>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "gwn_bvh_topology_build_binary.cuh"

namespace gwn {
namespace detail {

inline constexpr unsigned int k_gwn_collapse_slot_invalid = 0u;
inline constexpr unsigned int k_gwn_collapse_slot_ready = 1u;

template <gwn_index_type Index> struct gwn_collapse_work_item {
    std::uint8_t binary_kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
    Index binary_index = Index(0);
    Index wide_node_id = gwn_invalid_index<Index>();
};

template <gwn_index_type Index> struct gwn_collapse_slot_entry {
    gwn_collapse_work_item<Index> work{};
    unsigned int state = k_gwn_collapse_slot_invalid;
};

template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_collapse_convert_kernel_params {
    cuda::std::span<gwn_binary_node<Index> const> binary_nodes{};
    cuda::std::span<gwn_aabb<Real> const> binary_internal_bounds{};
    cuda::std::span<gwn_collapse_slot_entry<Index>> slots{};
    cuda::std::span<gwn_bvh_topology_node_soa<Width, Index>> output_nodes{};
    unsigned int *tail = nullptr;
    unsigned int *wide_node_counter = nullptr;
    unsigned int *block_counter = nullptr;
    unsigned int *error_flag = nullptr;
};

template <class T>
[[nodiscard]] __device__ inline cuda::atomic_ref<T, cuda::thread_scope_device>
gwn_collapse_atomic_ref(T &value) {
    return cuda::atomic_ref<T, cuda::thread_scope_device>(value);
}

[[nodiscard]] __device__ inline unsigned int
gwn_collapse_slot_state_load(unsigned int const &state) noexcept {
    return gwn_collapse_atomic_ref(const_cast<unsigned int &>(state))
        .load(cuda::memory_order_acquire);
}

__device__ inline void
gwn_collapse_slot_state_store_release(unsigned int &state, unsigned int const value) noexcept {
    gwn_collapse_atomic_ref(state).store(value, cuda::memory_order_release);
}

__device__ inline void
gwn_collapse_slot_state_store_relaxed(unsigned int &state, unsigned int const value) noexcept {
    gwn_collapse_atomic_ref(state).store(value, cuda::memory_order_relaxed);
}

[[nodiscard]] __device__ inline bool
gwn_collapse_has_error(unsigned int const *error_flag) noexcept {
    if (error_flag == nullptr)
        return false;
    return gwn_collapse_atomic_ref(*const_cast<unsigned int *>(error_flag))
               .load(cuda::memory_order_acquire) != 0u;
}

__device__ inline void gwn_collapse_signal_error(unsigned int *error_flag) noexcept {
    if (error_flag != nullptr)
        atomicExch(error_flag, 1u);
}

template <int Width, gwn_index_type Index>
__device__ inline void
gwn_collapse_set_invalid_child(gwn_bvh_topology_node_soa<Width, Index> &node, int const slot) {
    node.child_index[slot] = Index(0);
    node.child_count[slot] = Index(0);
    node.child_kind[slot] = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
}

template <int BlockSize, int Width, gwn_real_type Real, gwn_index_type Index>
__global__ __launch_bounds__(BlockSize) void gwn_collapse_convert_kernel(
    gwn_collapse_convert_kernel_params<Width, Real, Index> const params
) {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    __shared__ unsigned int logical_block_id;
    if (threadIdx.x == 0)
        logical_block_id = atomicAdd(params.block_counter, 1u);
    __syncthreads();

    std::size_t const slot_id =
        static_cast<std::size_t>(logical_block_id) * static_cast<std::size_t>(BlockSize) +
        static_cast<std::size_t>(threadIdx.x);
    if (slot_id >= params.slots.size())
        return;

    auto *slot = params.slots.data() + slot_id;
    while (true) {
        while (gwn_collapse_slot_state_load(slot->state) != k_gwn_collapse_slot_ready)
            if (gwn_collapse_has_error(params.error_flag))
                return;

        gwn_collapse_work_item<Index> const work = slot->work;
        gwn_collapse_slot_state_store_relaxed(slot->state, k_gwn_collapse_slot_invalid);
        auto const work_kind = static_cast<gwn_bvh_child_kind>(work.binary_kind);
        if (work_kind == gwn_bvh_child_kind::k_leaf)
            return;
        if (work_kind != gwn_bvh_child_kind::k_internal ||
            !gwn_index_in_bounds(work.binary_index, params.binary_nodes.size()) ||
            !gwn_index_in_bounds(work.wide_node_id, params.output_nodes.size())) {
            gwn_collapse_signal_error(params.error_flag);
            return;
        }

        gwn_binary_child_ref<Index> candidates[Width]{};
        gwn_collapse_work_item<Index> tasks[Width]{};
        auto const binary_root_node =
            params.binary_nodes[static_cast<std::size_t>(work.binary_index)];
        candidates[0] = binary_root_node.left;
        candidates[1] = binary_root_node.right;
        int candidate_count = 2;
        GWN_ASSERT(
            static_cast<gwn_bvh_child_kind>(candidates[0].kind) != gwn_bvh_child_kind::k_invalid,
            "collapse: left child of binary root is invalid"
        );
        GWN_ASSERT(
            static_cast<gwn_bvh_child_kind>(candidates[1].kind) != gwn_bvh_child_kind::k_invalid,
            "collapse: right child of binary root is invalid"
        );

        while (candidate_count < Width) {
            int best_slot = -1;
            Real best_area = -std::numeric_limits<Real>::infinity();
            Index best_internal = gwn_invalid_index<Index>();

            for (int i = 0; i < candidate_count; ++i) {
                auto const candidate_kind = static_cast<gwn_bvh_child_kind>(candidates[i].kind);
                if (candidate_kind != gwn_bvh_child_kind::k_internal)
                    continue;
                if (!gwn_index_in_bounds(
                        candidates[i].index, params.binary_internal_bounds.size()
                    ) ||
                    !gwn_index_in_bounds(candidates[i].index, params.binary_nodes.size()))
                    continue;

                Real const area = gwn_aabb_half_area(
                    params.binary_internal_bounds[static_cast<std::size_t>(candidates[i].index)]
                );
                if (best_slot < 0 || area > best_area ||
                    (area == best_area && candidates[i].index < best_internal)) {
                    best_slot = i;
                    best_area = area;
                    best_internal = candidates[i].index;
                }
            }

            if (best_slot < 0)
                break;
            GWN_ASSERT(candidate_count >= 2, "collapse: candidate_count shrank below 2");

            auto const expanded =
                params.binary_nodes[static_cast<std::size_t>(candidates[best_slot].index)];
            candidates[best_slot] = expanded.left;
            candidates[candidate_count] = expanded.right;
            ++candidate_count;
        }

        gwn_bvh_topology_node_soa<Width, Index> output_node{};
        GWN_PRAGMA_UNROLL
        for (int slot_index = 0; slot_index < Width; ++slot_index)
            gwn_collapse_set_invalid_child(output_node, slot_index);

        for (int i = 0; i < candidate_count; ++i) {
            auto const candidate_kind = static_cast<gwn_bvh_child_kind>(candidates[i].kind);
            if (candidate_kind == gwn_bvh_child_kind::k_leaf) {
                output_node.child_kind[i] = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
                output_node.child_index[i] = candidates[i].index;
                output_node.child_count[i] = Index(1);
                tasks[i].binary_kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
                tasks[i].binary_index = candidates[i].index;
                tasks[i].wide_node_id = gwn_invalid_index<Index>();
                continue;
            }

            if (candidate_kind != gwn_bvh_child_kind::k_internal ||
                !gwn_index_in_bounds(candidates[i].index, params.binary_nodes.size())) {
                gwn_collapse_signal_error(params.error_flag);
                return;
            }

            unsigned int const wide_child = atomicAdd(params.wide_node_counter, 1u);
            if (wide_child >= params.output_nodes.size()) {
                gwn_collapse_signal_error(params.error_flag);
                return;
            }

            output_node.child_kind[i] = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
            output_node.child_index[i] = static_cast<Index>(wide_child);
            output_node.child_count[i] = Index(0);
            tasks[i].binary_kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
            tasks[i].binary_index = candidates[i].index;
            tasks[i].wide_node_id = static_cast<Index>(wide_child);
        }

        GWN_ASSERT(
            gwn_index_in_bounds(work.wide_node_id, params.output_nodes.size()),
            "collapse: wide_node_id out of bounds before write"
        );
        params.output_nodes[static_cast<std::size_t>(work.wide_node_id)] = output_node;

        slot->work = tasks[0];
        gwn_collapse_slot_state_store_release(slot->state, k_gwn_collapse_slot_ready);
        for (int i = 1; i < candidate_count; ++i) {
            unsigned int const append_slot = atomicAdd(params.tail, 1u);
            if (append_slot >= params.slots.size()) {
                gwn_collapse_signal_error(params.error_flag);
                return;
            }
            auto *append_entry = params.slots.data() + append_slot;
            append_entry->work = tasks[i];
            gwn_collapse_slot_state_store_release(append_entry->state, k_gwn_collapse_slot_ready);
        }
    }
}

} // namespace detail
} // namespace gwn
