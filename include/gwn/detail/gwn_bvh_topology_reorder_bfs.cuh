#pragma once

/// \file gwn_bvh_topology_reorder_bfs.cuh
/// \brief GPU-native BFS reorder pass for wide BVH topology nodes.
///
/// Reorders the internal-node array of a wide BVH so that nodes appear in
/// breadth-first order.  This improves cache locality during top-down
/// traversal because parent nodes are stored before their children, and
/// siblings are contiguous in memory.
///
/// The implementation uses PRAM primitives mapped to CUDA kernels:
/// (1) build parent array, (2) pointer-jumping depth computation,
/// (3) CUB stable radix sort by depth, (4) inverse permutation + scatter
/// with child remap.  Everything runs on-device with zero host sync.

#include <cub/device/device_radix_sort.cuh>

#include <cstddef>
#include <cstdint>

#include "../gwn_bvh.cuh"
#include "../gwn_kernel_utils.cuh"
#include "../gwn_utils.cuh"

namespace gwn {
namespace detail {

// --------------- Kernel 1: build parent array ---------------

template <int Width, gwn_index_type Index>
struct gwn_reorder_build_parent_functor {
    cuda::std::span<gwn_bvh_topology_node_soa<Width, Index> const> nodes;
    Index *parent; // output: parent[child] = node_index

    __device__ void operator()(std::size_t const node_index) const {
        auto const &node = nodes[node_index];
        auto const idx = static_cast<Index>(node_index);
        GWN_PRAGMA_UNROLL
        for (int s = 0; s < Width; ++s) {
            if (static_cast<gwn_bvh_child_kind>(node.child_kind[s]) ==
                gwn_bvh_child_kind::k_internal) {
                auto const child = static_cast<std::size_t>(node.child_index[s]);
                if (child < nodes.size())
                    parent[child] = idx;
            }
        }
    }
};

// --------------- Kernel 2: one round of pointer jumping ---------------

template <gwn_index_type Index>
struct gwn_reorder_pointer_jump_functor {
    Index const *jump_in;
    Index *jump_out;
    std::uint16_t const *depth_in;
    std::uint16_t *depth_out;
    Index root_index;

    __device__ void operator()(std::size_t const i) const {
        auto const idx = static_cast<Index>(i);
        if (idx == root_index) {
            jump_out[i] = root_index;
            depth_out[i] = 0;
            return;
        }
        Index const j = jump_in[i];
        auto const j_sz = static_cast<std::size_t>(j);
        depth_out[i] = depth_in[i] + depth_in[j_sz];
        jump_out[i] = jump_in[j_sz];
    }
};

// --------------- Kernel 3: build inverse permutation ---------------

template <gwn_index_type Index>
struct gwn_reorder_inverse_perm_functor {
    Index const *permutation; // permutation[new_id] = old_id
    Index *inverse;           // inverse[old_id] = new_id

    __device__ void operator()(std::size_t const new_id) const {
        auto const old_id = static_cast<std::size_t>(permutation[new_id]);
        inverse[old_id] = static_cast<Index>(new_id);
    }
};

// --------------- Kernel 4: scatter nodes + remap child indices ---------------

template <int Width, gwn_index_type Index>
struct gwn_reorder_scatter_remap_functor {
    cuda::std::span<gwn_bvh_topology_node_soa<Width, Index> const> input;
    gwn_bvh_topology_node_soa<Width, Index> *output;
    Index const *permutation;  // permutation[new_id] = old_id
    Index const *inverse_perm; // inverse_perm[old_id] = new_id

    __device__ void operator()(std::size_t const new_id) const {
        auto const old_id = static_cast<std::size_t>(permutation[new_id]);
        auto node = input[old_id];
        GWN_PRAGMA_UNROLL
        for (int s = 0; s < Width; ++s) {
            if (static_cast<gwn_bvh_child_kind>(node.child_kind[s]) ==
                gwn_bvh_child_kind::k_internal) {
                auto const child_old = static_cast<std::size_t>(node.child_index[s]);
                node.child_index[s] = inverse_perm[child_old];
            }
        }
        output[new_id] = node;
    }
};

// --------------- Helper functors for initialisation ---------------

template <gwn_index_type Index>
struct gwn_reorder_init_depth_functor {
    std::uint16_t *depth;
    Index root;
    __device__ void operator()(std::size_t const i) const {
        depth[i] = (static_cast<Index>(i) == root) ? std::uint16_t(0) : std::uint16_t(1);
    }
};

template <gwn_index_type Index>
struct gwn_reorder_iota_functor {
    Index *data;
    __device__ void operator()(std::size_t const i) const {
        data[i] = static_cast<Index>(i);
    }
};

// --------------- Orchestrator ---------------

/// \brief Reorder topology nodes into breadth-first order (GPU-native).
///
/// \tparam Width  BVH node fan-out.
/// \tparam Index  Integer index type.
///
/// \param nodes       Device span of topology nodes (size is unchanged).
/// \param root_index  On entry, the current root node index; on exit, set to 0.
/// \param stream      CUDA stream for async operations.
///
/// \return \c gwn_status::ok() on success.
template <int Width, gwn_index_type Index>
gwn_status gwn_bvh_topology_reorder_bfs(
    cuda::std::span<gwn_bvh_topology_node_soa<Width, Index>> nodes, Index &root_index,
    cudaStream_t const stream
) noexcept {
    static_assert(Width >= 2, "BVH node width must be at least 2.");

    // Nothing to reorder for empty or single-node trees.
    if (nodes.size() <= 1) {
        if (!nodes.empty())
            root_index = Index(0);
        return gwn_status::ok();
    }

    using node_type = gwn_bvh_topology_node_soa<Width, Index>;
    constexpr int k_block = k_gwn_default_block_size;
    std::size_t const N = nodes.size();
    auto const N_u64 = static_cast<std::uint64_t>(N);

    // --- Allocate temporaries ---
    gwn_device_array<Index> parent{};
    gwn_device_array<Index> jump_a{};
    gwn_device_array<Index> jump_b{};
    gwn_device_array<std::uint16_t> depth_a{};
    gwn_device_array<std::uint16_t> depth_b{};
    gwn_device_array<Index> permutation{};
    gwn_device_array<std::uint16_t> depth_sorted{};
    gwn_device_array<Index> inverse_perm{};
    gwn_device_array<node_type> output{};
    gwn_device_array<std::uint8_t> cub_temp{};

    GWN_RETURN_ON_ERROR(parent.resize(N, stream));
    GWN_RETURN_ON_ERROR(jump_a.resize(N, stream));
    GWN_RETURN_ON_ERROR(jump_b.resize(N, stream));
    GWN_RETURN_ON_ERROR(depth_a.resize(N, stream));
    GWN_RETURN_ON_ERROR(depth_b.resize(N, stream));
    GWN_RETURN_ON_ERROR(permutation.resize(N, stream));
    GWN_RETURN_ON_ERROR(depth_sorted.resize(N, stream));
    GWN_RETURN_ON_ERROR(inverse_perm.resize(N, stream));
    GWN_RETURN_ON_ERROR(output.resize(N, stream));

    // --- Step 1: Build parent array ---
    // Initialise parent to zero (root's parent is itself, i.e. parent[0] = 0).
    GWN_RETURN_ON_ERROR(
        gwn_cuda_to_status(cudaMemsetAsync(parent.data(), 0, N * sizeof(Index), stream))
    );

    GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
        N, gwn_reorder_build_parent_functor<Width, Index>{nodes, parent.data()}, stream
    ));

    // --- Step 2: Pointer jumping to compute depth ---
    // Initialise: depth = 1 for all, 0 for root.  jump = parent.
    {
        // Set depth: root = 0, others = 1.  (memset can't set uint16 to 1.)
        GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
            N, gwn_reorder_init_depth_functor<Index>{depth_a.data(), root_index}, stream
        ));

        // Copy parent into jump_a, then free parent (no longer needed).
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
            jump_a.data(), parent.data(), N * sizeof(Index), cudaMemcpyDeviceToDevice, stream
        )));
        GWN_RETURN_ON_ERROR(parent.clear(stream));
    }

    // Pointer jumping: ceil(log2(D_max)) rounds.
    // D_max for width-4 BVH of 5M primitives ~ 10, so 4 rounds suffice.
    // We use 8 rounds to be safe for any reasonable tree depth (< 256).
    constexpr int k_max_rounds = 8;
    Index *jump_src = jump_a.data();
    Index *jump_dst = jump_b.data();
    std::uint16_t *depth_src = depth_a.data();
    std::uint16_t *depth_dst = depth_b.data();

    for (int round = 0; round < k_max_rounds; ++round) {
        GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
            N,
            gwn_reorder_pointer_jump_functor<Index>{
                jump_src, jump_dst, depth_src, depth_dst, root_index},
            stream
        ));
        // Swap src/dst for next round.
        using std::swap;
        swap(jump_src, jump_dst);
        swap(depth_src, depth_dst);
    }
    // After the loop, depth_src holds final depths.

    // --- Step 3: Stable radix sort by depth ---
    // Initialise permutation to identity [0, 1, 2, ...].
    {
        GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
            N, gwn_reorder_iota_functor<Index>{permutation.data()}, stream
        ));
    }

    // CUB radix sort: key = depth (uint16), value = node index.
    // Depth range [0, 256) -> only 8 bits needed.
    {
        std::size_t cub_temp_bytes = 0;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceRadixSort::SortPairs(
            nullptr, cub_temp_bytes, depth_src, depth_sorted.data(), permutation.data(),
            inverse_perm.data(), // reuse as temp value output
            N_u64, 0, 8, stream
        )));
        GWN_RETURN_ON_ERROR(cub_temp.resize(cub_temp_bytes, stream));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceRadixSort::SortPairs(
            cub_temp.data(), cub_temp_bytes, depth_src, depth_sorted.data(), permutation.data(),
            inverse_perm.data(), N_u64, 0, 8, stream
        )));
    }
    // After sort: inverse_perm now holds permutation[new_id] = old_id.
    // (We reused inverse_perm as the value output buffer.)
    // Swap so permutation holds the result.
    {
        using std::swap;
        swap(permutation, inverse_perm);
    }
    // Now: permutation[new_id] = old_id.

    // --- Step 4a: Build inverse permutation ---
    GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
        N,
        gwn_reorder_inverse_perm_functor<Index>{permutation.data(), inverse_perm.data()}, stream
    ));

    // --- Step 4b: Scatter + remap ---
    GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
        N,
        gwn_reorder_scatter_remap_functor<Width, Index>{
            nodes, output.data(), permutation.data(), inverse_perm.data()},
        stream
    ));

    // --- Copy output back into nodes ---
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        nodes.data(), output.data(), N * sizeof(node_type), cudaMemcpyDeviceToDevice, stream
    )));

    root_index = Index(0);
    return gwn_status::ok();
}

} // namespace detail
} // namespace gwn
