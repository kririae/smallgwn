#pragma once

#include <cuda/atomic>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cub/device/device_reduce.cuh>

#include "gwn_bvh_build_binary.cuh"
#include "gwn_device_array.cuh"

namespace gwn {
namespace detail {

inline constexpr unsigned int k_gwn_bvh_collapse_slot_invalid = 0u;
inline constexpr unsigned int k_gwn_bvh_collapse_slot_ready = 1u;

/// \brief One pending binary child and its canonical wide-node position.
///
/// An invalid \c wide_node_index marks a leaf task, which terminates its persistent consumer.
template <gwn_index_type Index> struct gwn_bvh_collapse_work_item {
    Index binary_index = Index(0);
    Index wide_node_index = gwn_invalid_index<Index>();
};

/// \brief Publication slot used by the persistent wide-collapse work queue.
///
/// Producers write \c work before publishing \c state with release ordering. Consumers acquire
/// \c state before reading the item, so a child task never observes a partially written record.
template <gwn_index_type Index> struct gwn_bvh_collapse_slot {
    gwn_bvh_collapse_work_item<Index> work{};
    unsigned int state = k_gwn_bvh_collapse_slot_invalid;
};

/// \brief Topology stack bounds carried from a wide node to its internal children.
struct gwn_bvh_collapse_stack_bound {
    unsigned int internal_bound = 0;
    unsigned int packed_bound = 0;
};

/// \brief Reduce independent topology stack bounds component-wise.
struct gwn_bvh_collapse_stack_bound_max {
    [[nodiscard]] __host__ __device__ gwn_bvh_collapse_stack_bound operator()(
        gwn_bvh_collapse_stack_bound const lhs, gwn_bvh_collapse_stack_bound const rhs
    ) const noexcept {
        return {
            std::max(lhs.internal_bound, rhs.internal_bound),
            std::max(lhs.packed_bound, rhs.packed_bound),
        };
    }
};

/// \brief Device state shared by the native child-AoS wide-collapse kernel.
template <int Width, gwn_real_type Real, gwn_index_type Index> struct gwn_bvh_collapse_params {
    cuda::std::span<gwn_binary_node<Index> const> binary_nodes{};
    cuda::std::span<gwn_aabb<Real> const> binary_internal_bounds{};
    cuda::std::span<gwn_aabb<Real> const> sorted_primitive_bounds{};
    cuda::std::span<gwn_bvh_collapse_slot<Index>> slots{};
    cuda::std::span<gwn_bvh_node<Width, Real>> output_nodes{};
    cuda::std::span<std::uint64_t> reorder_key{};
    cuda::std::span<gwn_bvh_collapse_stack_bound> stack_bound{};
    unsigned int *tail = nullptr;
    unsigned int *wide_node_count = nullptr;
    unsigned int *block_count = nullptr;
    unsigned int *error_flag = nullptr;
};

template <class T>
[[nodiscard]] __device__ inline cuda::atomic_ref<T, cuda::thread_scope_device>
gwn_bvh_collapse_atomic_ref(T &value) noexcept {
    return cuda::atomic_ref<T, cuda::thread_scope_device>(value);
}

[[nodiscard]] __device__ inline unsigned int
gwn_bvh_collapse_load_state(unsigned int const &state) noexcept {
    return gwn_bvh_collapse_atomic_ref(const_cast<unsigned int &>(state))
        .load(cuda::memory_order_acquire);
}

__device__ inline void
gwn_bvh_collapse_store_release(unsigned int &state, unsigned int const value) noexcept {
    gwn_bvh_collapse_atomic_ref(state).store(value, cuda::memory_order_release);
}

__device__ inline void
gwn_bvh_collapse_store_relaxed(unsigned int &state, unsigned int const value) noexcept {
    gwn_bvh_collapse_atomic_ref(state).store(value, cuda::memory_order_relaxed);
}

/// \brief Collapse a binary BVH directly into canonical child-AoS wide nodes.
///
/// Each persistent thread consumes one publication slot. Expanding the largest-area internal
/// candidate first preserves the existing wide-tree quality rule, while assigning bounds and the
/// packed reference together guarantees that traversal never needs a second hierarchy lookup.
template <int BlockSize, int Width, gwn_real_type Real, gwn_index_type Index>
__global__ __launch_bounds__(BlockSize) void gwn_bvh_collapse_binary_wide_kernel(
    gwn_bvh_collapse_params<Width, Real, Index> const params
) {
    static_assert(Width >= 2, "BVH width must be at least 2.");

    auto signal_error = [&]() noexcept {
        if (params.error_flag != nullptr)
            atomicExch(params.error_flag, 1u);
    };
    auto has_error = [&]() noexcept {
        return params.error_flag != nullptr &&
               gwn_bvh_collapse_atomic_ref(*params.error_flag).load(cuda::memory_order_acquire) !=
                   0u;
    };
    auto encode_reference = [&](std::uint64_t const offset,
                                std::uint64_t const primitive_count) noexcept {
        // Check before shifting. A truncated offset can still point at allocated memory, which
        // makes this failure especially hard to distinguish from a traversal bug.
        if (!gwn_bvh_child<Real>::can_encode_offset(offset) ||
            !gwn_bvh_child<Real>::can_encode_primitive_count(primitive_count)) {
            signal_error();
            return std::uint64_t(0);
        }
        return gwn_bvh_child<Real>::k_valid_mask | offset |
               (primitive_count << gwn_bvh_child<Real>::k_primitive_count_shift);
    };
    // Logical block IDs make every publication slot have exactly one persistent consumer even
    // when CUDA schedules physical blocks in a different order.
    __shared__ unsigned int logical_block_index;
    if (threadIdx.x == 0)
        logical_block_index = atomicAdd(params.block_count, 1u);
    __syncthreads();

    std::size_t const slot_index =
        static_cast<std::size_t>(logical_block_index) * static_cast<std::size_t>(BlockSize) +
        static_cast<std::size_t>(threadIdx.x);
    if (slot_index >= params.slots.size())
        return;

    auto *slot = params.slots.data() + slot_index;
    while (true) {
        // A thread keeps its slot for the whole kernel. Leaf tasks end that thread; internal tasks
        // publish one child back to the same slot and append the remaining children to the tail.
        while (gwn_bvh_collapse_load_state(slot->state) != k_gwn_bvh_collapse_slot_ready)
            if (has_error())
                return;

        gwn_bvh_collapse_work_item<Index> const work = slot->work;
        gwn_bvh_collapse_store_relaxed(slot->state, k_gwn_bvh_collapse_slot_invalid);
        // Leaf tasks own no wide node. Consuming one retires the slot's persistent thread.
        if (gwn_is_invalid_index(work.wide_node_index))
            return;
        if (!gwn_index_in_bounds(work.binary_index, params.binary_nodes.size()) ||
            !gwn_index_in_bounds(work.wide_node_index, params.output_nodes.size())) {
            signal_error();
            return;
        }
        std::uint32_t const work_depth =
            work.wide_node_index == Index(0)
                ? 0u
                : static_cast<std::uint32_t>(
                      params.reorder_key[static_cast<std::size_t>(work.wide_node_index)] >> 32
                  );

        gwn_binary_child_ref<Index> candidates[Width]{};
        gwn_bvh_collapse_work_item<Index> tasks[Width]{};
        auto const binary_root = params.binary_nodes[static_cast<std::size_t>(work.binary_index)];
        candidates[0] = binary_root.left;
        candidates[1] = binary_root.right;
        int candidate_count = 2;

        while (candidate_count < Width) {
            // Expanding the largest internal bound gives the wide node more useful separation than
            // expanding the first internal candidate in binary-tree order.
            int best_slot = -1;
            Real best_area = -std::numeric_limits<Real>::infinity();
            Index best_internal = gwn_invalid_index<Index>();
            for (int candidate_slot = 0; candidate_slot < candidate_count; ++candidate_slot) {
                auto const kind =
                    static_cast<gwn_binary_child_kind>(candidates[candidate_slot].kind);
                if (kind != gwn_binary_child_kind::k_internal ||
                    !gwn_index_in_bounds(
                        candidates[candidate_slot].index, params.binary_internal_bounds.size()
                    ) ||
                    !gwn_index_in_bounds(
                        candidates[candidate_slot].index, params.binary_nodes.size()
                    )) {
                    continue;
                }

                Real const area =
                    gwn_aabb_half_area(params.binary_internal_bounds[static_cast<std::size_t>(
                        candidates[candidate_slot].index
                    )]);
                if (best_slot >= 0 && area < best_area)
                    continue;
                if (best_slot >= 0 && area == best_area &&
                    candidates[candidate_slot].index >= best_internal) {
                    continue;
                }
                best_slot = candidate_slot;
                best_area = area;
                best_internal = candidates[candidate_slot].index;
            }

            if (best_slot < 0)
                break;
            auto const expanded =
                params.binary_nodes[static_cast<std::size_t>(candidates[best_slot].index)];
            candidates[best_slot] = expanded.left;
            candidates[candidate_count] = expanded.right;
            ++candidate_count;
        }

        gwn_bvh_node<Width, Real> output_node{};
        for (int candidate_slot = 0; candidate_slot < candidate_count; ++candidate_slot) {
            auto const candidate = candidates[candidate_slot];
            auto const candidate_kind = static_cast<gwn_binary_child_kind>(candidate.kind);
            auto &output_child = output_node.child(candidate_slot);

            if (candidate_kind == gwn_binary_child_kind::k_leaf) {
                if (!gwn_index_in_bounds(candidate.index, params.sorted_primitive_bounds.size())) {
                    signal_error();
                    return;
                }
                output_child.bounds =
                    params.sorted_primitive_bounds[static_cast<std::size_t>(candidate.index)];
                // Bounds and reference are written through the same child object. Traversal can
                // load both without consulting a separate hierarchy allocation.
                output_child.reference =
                    encode_reference(static_cast<std::uint64_t>(candidate.index), 1u);
                tasks[candidate_slot].binary_index = candidate.index;
                continue;
            }

            if (candidate_kind != gwn_binary_child_kind::k_internal ||
                !gwn_index_in_bounds(candidate.index, params.binary_nodes.size()) ||
                !gwn_index_in_bounds(candidate.index, params.binary_internal_bounds.size())) {
                signal_error();
                return;
            }

            unsigned int const wide_child = atomicAdd(params.wide_node_count, 1u);
            if (wide_child >= params.output_nodes.size()) {
                signal_error();
                return;
            }
            output_child.bounds =
                params.binary_internal_bounds[static_cast<std::size_t>(candidate.index)];
            output_child.reference = encode_reference(wide_child, 0u);
            tasks[candidate_slot].binary_index = candidate.index;
            tasks[candidate_slot].wide_node_index = static_cast<Index>(wide_child);
            std::uint32_t const child_depth = work_depth + 1u;

            // The collapse already knows both values needed by the later breadth-first sort.
            // The key is visible when the task's release publication becomes visible, so its
            // consumer can recover depth without enlarging every queued work item.
            // The build's UINT32 primitive limit guarantees that every wide parent fits low bits.
            params.reorder_key[wide_child] = (std::uint64_t(child_depth) << 32) |
                                             static_cast<std::uint32_t>(work.wide_node_index);
        }

        if (work.wide_node_index == Index(0))
            params.reorder_key[0] = 0;
        auto const output_node_index = static_cast<std::size_t>(work.wide_node_index);
        params.output_nodes[output_node_index] = output_node;

        unsigned int internal_child_count = 0;
        for (int candidate_slot = 0; candidate_slot < candidate_count; ++candidate_slot)
            internal_child_count += output_node.child(candidate_slot).is_internal() ? 1u : 0u;

        // With P pending references on entry, canonical traversal pushes k - 1 of its k internal
        // children and enters the remaining child directly. Packed ray traversal does the same
        // with m valid children. Carrying those child prefixes top-down turns both exact
        // recurrences into one maximum reduction.
        // Every pending item owns a disjoint non-empty primitive range, so both values remain
        // within the UINT32 primitive-count limit checked before launch.
        auto const stack_prefix = params.stack_bound[output_node_index];
        unsigned int const internal_push_count =
            internal_child_count == 0u ? 0u : internal_child_count - 1u;
        gwn_bvh_collapse_stack_bound const node_stack_bound{
            stack_prefix.internal_bound + internal_push_count,
            stack_prefix.packed_bound + static_cast<unsigned int>(candidate_count - 1),
        };
        params.stack_bound[output_node_index] = node_stack_bound;
        for (int candidate_slot = 0; candidate_slot < candidate_count; ++candidate_slot) {
            auto const child_index = tasks[candidate_slot].wide_node_index;
            if (gwn_is_invalid_index(child_index))
                continue;
            if (!gwn_index_in_bounds(child_index, params.stack_bound.size())) {
                signal_error();
                return;
            }
            params.stack_bound[static_cast<std::size_t>(child_index)] = {
                node_stack_bound.internal_bound,
                node_stack_bound.packed_bound,
            };
        }

        // Reusing the consumer's slot for the first task saves one tail reservation per internal
        // node. The release also publishes each child's stack prefix before another SM acquires
        // the corresponding task.
        slot->work = tasks[0];
        gwn_bvh_collapse_store_release(slot->state, k_gwn_bvh_collapse_slot_ready);
        for (int candidate_slot = 1; candidate_slot < candidate_count; ++candidate_slot) {
            unsigned int const append_slot = atomicAdd(params.tail, 1u);
            if (append_slot >= params.slots.size()) {
                signal_error();
                return;
            }
            auto *append_entry = params.slots.data() + append_slot;
            append_entry->work = tasks[candidate_slot];
            gwn_bvh_collapse_store_release(append_entry->state, k_gwn_bvh_collapse_slot_ready);
        }
    }
}

/// \brief Run native wide collapse into exact-capacity \p output_nodes.
///
/// \p output_nodes must contain one slot per binary internal node. On success,
/// \p output_node_count identifies the initialized prefix retained by the canonical BVH.
template <int Width, gwn_real_type Real, gwn_index_type Index>
void gwn_bvh_collapse_binary_wide_impl(
    cuda::std::span<gwn_binary_node<Index> const> const binary_nodes,
    cuda::std::span<gwn_aabb<Real> const> const binary_internal_bounds,
    cuda::std::span<gwn_aabb<Real> const> const sorted_primitive_bounds,
    cuda::std::span<gwn_bvh_node<Width, Real>> const output_nodes,
    cuda::std::span<std::uint64_t> const reorder_key, std::size_t &output_node_count,
    std::uint32_t &internal_stack_bound, std::uint32_t &packed_stack_bound,
    Index const root_internal_index, cudaStream_t const stream
) {
    std::size_t const binary_internal_count = binary_nodes.size();
    std::size_t const primitive_count = sorted_primitive_bounds.size();
    if (binary_internal_count == 0)
        return;
    if (binary_internal_bounds.size() != binary_internal_count ||
        primitive_count != binary_internal_count + 1 ||
        output_nodes.size() != binary_internal_count ||
        reorder_key.size() != binary_internal_count ||
        !gwn_index_in_bounds(root_internal_index, binary_internal_count)) {
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_build_collapse,
            "Canonical wide collapse received inconsistent binary BVH data."
        );
    }
    if (primitive_count > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_build_collapse,
            "Canonical wide collapse primitive count exceeds its counter range."
        );
    }
    if (!gwn_bvh_child<Real>::can_encode_offset(binary_internal_count - 1) ||
        !gwn_bvh_child<Real>::can_encode_offset(primitive_count - 1)) {
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_build_collapse,
            "Canonical wide collapse offset exceeds the packed child range."
        );
    }

    constexpr int k_block_size = k_gwn_default_block_size;
    if (primitive_count > gwn_max_linear_kernel_elements_1d<k_block_size>()) {
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_build_collapse,
            "Canonical wide collapse primitive count exceeds the launch range."
        );
    }

    gwn_device_array<gwn_bvh_collapse_slot<Index>> slots{};
    gwn_device_array<gwn_bvh_collapse_stack_bound> stack_bound{};
    gwn_device_array<gwn_bvh_collapse_stack_bound> reduced_stack_bound{};
    gwn_device_array<unsigned int> counters{};
    gwn_device_array<std::uint8_t> reduce_storage{};
    slots.resize(primitive_count, stream);
    stack_bound.resize(binary_internal_count, stream);
    reduced_stack_bound.resize(1, stream);
    counters.resize(4, stream);
    gwn_throw_status_error(gwn_cuda_to_status(cudaMemsetAsync(
        slots.data(), 0, primitive_count * sizeof(gwn_bvh_collapse_slot<Index>), stream
    )));
    gwn_throw_status_error(gwn_cuda_to_status(cudaMemsetAsync(
        stack_bound.data(), 0, binary_internal_count * sizeof(gwn_bvh_collapse_stack_bound), stream
    )));

    gwn_bvh_collapse_slot<Index> root_slot{};
    root_slot.work.binary_index = root_internal_index;
    root_slot.work.wide_node_index = Index(0);
    root_slot.state = k_gwn_bvh_collapse_slot_ready;
    gwn_throw_status_error(gwn_cuda_to_status(
        cudaMemcpyAsync(slots.data(), &root_slot, sizeof(root_slot), cudaMemcpyHostToDevice, stream)
    ));

    // tail and node count start after the root. Physical block IDs and the error flag start at 0.
    unsigned int const initial_counters[4]{1u, 1u, 0u, 0u};
    gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
        counters.data(), initial_counters, sizeof(initial_counters), cudaMemcpyHostToDevice, stream
    )));

    gwn_bvh_collapse_params<Width, Real, Index> const params{
        binary_nodes,        binary_internal_bounds, sorted_primitive_bounds,
        slots.span(),        output_nodes,           reorder_key,
        stack_bound.span(),  counters.data(),        counters.data() + 1,
        counters.data() + 2, counters.data() + 3,
    };
    gwn_bvh_collapse_binary_wide_kernel<k_block_size, Width, Real, Index>
        <<<gwn_grid_dim_1d<k_block_size>(primitive_count), gwn_block_dim_1d<k_block_size>(), 0,
           stream>>>(params);
    gwn_throw_status_error(gwn_check_last_kernel());

    std::size_t reduce_storage_bytes = 0;
    std::uint64_t const cub_node_count = binary_internal_count;
    gwn_throw_status_error(gwn_cuda_to_status(
        cub::DeviceReduce::Reduce(
            nullptr, reduce_storage_bytes, stack_bound.data(), reduced_stack_bound.data(),
            cub_node_count, gwn_bvh_collapse_stack_bound_max{}, gwn_bvh_collapse_stack_bound{},
            stream
        )
    ));
    reduce_storage.resize(reduce_storage_bytes, stream);
    gwn_throw_status_error(gwn_cuda_to_status(
        cub::DeviceReduce::Reduce(
            reduce_storage.data(), reduce_storage_bytes, stack_bound.data(),
            reduced_stack_bound.data(), cub_node_count, gwn_bvh_collapse_stack_bound_max{},
            gwn_bvh_collapse_stack_bound{}, stream
        )
    ));

    // Collapse allocates the maximum possible node count up front. The device counter tells us how
    // much of that scratch array belongs in the final BVH, so this is a required synchronization.
    unsigned int host_counters[4]{};
    gwn_bvh_collapse_stack_bound host_stack_bound{};
    gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
        host_counters, counters.data(), sizeof(host_counters), cudaMemcpyDeviceToHost, stream
    )));
    gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_stack_bound, reduced_stack_bound.data(), sizeof(host_stack_bound),
        cudaMemcpyDeviceToHost, stream
    )));
    gwn_throw_status_error(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_counters[3] != 0u || host_counters[1] == 0u ||
        host_counters[1] > binary_internal_count) {
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_build_collapse, "Canonical wide collapse reported an invalid output."
        );
    }

    output_node_count = host_counters[1];
    internal_stack_bound = host_stack_bound.internal_bound;
    packed_stack_bound = host_stack_bound.packed_bound;
}

} // namespace detail
} // namespace gwn
