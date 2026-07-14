#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_bvh_build_binary.cuh"
#include "gwn_bvh_build_common.cuh"
#include "gwn_device_array.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {
namespace detail {

/// \brief Build binary hierarchy and internal bounds with the LBVH algorithm.
template <gwn_real_type Real, gwn_index_type Index, class MortonCode = std::uint64_t>
void gwn_bvh_build_binary_lbvh(
    cuda::std::span<MortonCode const> const sorted_morton_codes,
    cuda::std::span<gwn_aabb<Real> const> const sorted_leaf_aabbs,
    gwn_device_array<gwn_binary_node<Index>> &binary_nodes,
    gwn_device_array<Index> &binary_internal_parent,
    gwn_device_array<gwn_aabb<Real>> &binary_internal_bounds,
    cudaStream_t const stream = cudaStreamLegacy
) {
    static_assert(
        std::is_same_v<MortonCode, std::uint32_t> || std::is_same_v<MortonCode, std::uint64_t>,
        "MortonCode must be std::uint32_t or std::uint64_t."
    );
    std::size_t const primitive_count = sorted_morton_codes.size();
    if (primitive_count == 0) {
        binary_nodes.clear(stream);
        binary_internal_parent.clear(stream);
        binary_internal_bounds.clear(stream);
        return;
    }

    if (sorted_leaf_aabbs.size() != primitive_count)
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_build_binary_lbvh, "LBVH binary builder sorted leaf AABB size mismatch."
        );

    if (primitive_count == 1) {
        binary_nodes.clear(stream);
        binary_internal_parent.clear(stream);
        binary_internal_bounds.clear(stream);
        return;
    }

    std::size_t const binary_internal_count = primitive_count - 1;
    GWN_ASSERT(
        binary_internal_count == primitive_count - 1,
        "LBVH: binary internal count invariant violated"
    );
    constexpr int k_block_size = k_gwn_default_block_size;
    gwn_binary_parent_temporaries<Index> temps{};
    gwn_prepare_binary_hierarchy_buffers(
        primitive_count, binary_nodes, binary_internal_parent, temps, stream
    );

    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count - 1,
            gwn_build_binary_hierarchy_functor<Index, MortonCode>{
                sorted_morton_codes, binary_nodes.span(), binary_internal_parent.span(),
                temps.internal_parent_slot.span(), temps.leaf_parent.span(),
                temps.leaf_parent_slot.span()
            },
            stream
        )
    );

    binary_internal_bounds.resize(binary_internal_count, stream);
    gwn_device_array<gwn_aabb<Real>> pending_left_bounds{};
    gwn_device_array<gwn_aabb<Real>> pending_right_bounds{};
    gwn_device_array<unsigned int> child_arrival_count{};
    pending_left_bounds.resize(binary_internal_count, stream);
    pending_right_bounds.resize(binary_internal_count, stream);
    child_arrival_count.resize(binary_internal_count, stream);
    gwn_throw_status_error(gwn_cuda_to_status(cudaMemsetAsync(
        child_arrival_count.data(), 0, binary_internal_count * sizeof(unsigned int), stream
    )));

    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            gwn_accumulate_binary_bounds_pass_functor<Real, Index>{
                sorted_leaf_aabbs, temps.leaf_parent.span(), temps.leaf_parent_slot.span(),
                binary_internal_parent.span(), temps.internal_parent_slot.span(),
                binary_internal_bounds.span(), pending_left_bounds.span(),
                pending_right_bounds.span(), child_arrival_count.span()
            },
            stream
        )
    );
}

} // namespace detail
} // namespace gwn
