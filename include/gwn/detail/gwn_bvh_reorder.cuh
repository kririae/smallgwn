#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cub/device/device_radix_sort.cuh>

#include "../gwn_bvh.cuh"
#include "gwn_bvh_status_helpers.cuh"
#include "gwn_device_array.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {
namespace detail {

/// \brief Initialize the identity permutation for breadth-first reorder.
template <gwn_index_type Index> struct gwn_bvh_prepare_permutation_functor {
    cuda::std::span<Index> permutation{};

    void __device__ operator()(std::size_t const node_index) const noexcept {
        permutation[node_index] = static_cast<Index>(node_index);
    }
};

/// \brief Invert the sorted new-to-old node permutation.
template <gwn_index_type Index> struct gwn_bvh_inverse_permutation_functor {
    cuda::std::span<Index const> permutation{};
    cuda::std::span<Index> inverse{};

    void __device__ operator()(std::size_t const new_index) const noexcept {
        inverse[static_cast<std::size_t>(permutation[new_index])] = static_cast<Index>(new_index);
    }
};

/// \brief Scatter nodes into breadth-first order and remap packed internal offsets.
template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_bvh_scatter_reordered_nodes_functor {
    cuda::std::span<gwn_bvh_node<Width, Real> const> input{};
    cuda::std::span<gwn_bvh_node<Width, Real>> output{};
    cuda::std::span<Index const> permutation{};
    cuda::std::span<Index const> inverse{};

    void __device__ operator()(std::size_t const new_index) const noexcept {
        auto node = input[static_cast<std::size_t>(permutation[new_index])];
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto &child = node.child(child_slot);
            if (!child.is_internal())
                continue;

            auto const old_child = static_cast<std::size_t>(child.offset());
            GWN_ASSERT(old_child < inverse.size(), "BVH reorder child offset is out of bounds.");
            std::uint64_t const new_child = inverse[old_child];
            child.reference = (child.reference & ~gwn_bvh_child<Real>::k_offset_mask) | new_child;
        }
        output[new_index] = node;
    }
};

/// \brief Reorder canonical BVH nodes breadth-first.
template <int Width, gwn_real_type Real, gwn_index_type Index>
void gwn_bvh_reorder_impl(
    cuda::std::span<gwn_bvh_node<Width, Real>> const nodes,
    cuda::std::span<std::uint64_t const> const reorder_key, cudaStream_t const stream
) {
    static_assert(Width >= 2, "BVH width must be at least 2.");

    if (reorder_key.size() != nodes.size()) {
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_build_reorder, "Canonical BVH reorder key size does not match nodes."
        );
    }
    if (nodes.size() <= 1)
        return;
    if (nodes.size() > std::numeric_limits<std::uint32_t>::max()) {
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_build_reorder,
            "Canonical BVH BFS reorder supports at most UINT32_MAX internal nodes."
        );
    }

    constexpr int k_block_size = k_gwn_default_block_size;
    std::size_t const node_count = nodes.size();
    std::uint64_t const cub_node_count = node_count;
    gwn_device_array<Index> permutation{};
    gwn_device_array<Index> inverse{};
    gwn_device_array<std::uint64_t> sorted_key{};
    gwn_device_array<gwn_bvh_node<Width, Real>> output{};
    gwn_device_array<std::uint8_t> sort_storage{};

    permutation.resize(node_count, stream);
    inverse.resize(node_count, stream);
    sorted_key.resize(node_count, stream);
    output.resize(node_count, stream);
    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_block_size>(
            node_count, gwn_bvh_prepare_permutation_functor<Index>{permutation.span()}, stream
        )
    );

    // Collapse supplies one (depth, parent) key per node. Sorting those keys keeps parents before
    // children and siblings contiguous without reconstructing the same tree metadata here.
    std::size_t sort_storage_bytes = 0;
    gwn_throw_status_error(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_storage_bytes, reorder_key.data(), sorted_key.data(), permutation.data(),
            inverse.data(), cub_node_count, 0, 64, stream
        )
    ));
    sort_storage.resize(sort_storage_bytes, stream);
    gwn_throw_status_error(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            sort_storage.data(), sort_storage_bytes, reorder_key.data(), sorted_key.data(),
            permutation.data(), inverse.data(), cub_node_count, 0, 64, stream
        )
    ));

    using std::swap;
    swap(permutation, inverse);
    // CUB wrote new_id -> old_id. The scatter also needs old_id -> new_id to patch every packed
    // internal offset before the temporary node array replaces the original order.
    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_block_size>(
            node_count,
            gwn_bvh_inverse_permutation_functor<Index>{permutation.span(), inverse.span()}, stream
        )
    );
    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_block_size>(
            node_count,
            gwn_bvh_scatter_reordered_nodes_functor<Width, Real, Index>{
                cuda::std::span<gwn_bvh_node<Width, Real> const>(nodes.data(), nodes.size()),
                output.span(),
                permutation.span(),
                inverse.span(),
            },
            stream
        )
    );
    gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
        nodes.data(), output.data(), nodes.size_bytes(), cudaMemcpyDeviceToDevice, stream
    )));
}

} // namespace detail
} // namespace gwn
