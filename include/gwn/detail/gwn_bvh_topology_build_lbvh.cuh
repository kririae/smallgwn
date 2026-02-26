#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "../gwn_kernel_utils.cuh"
#include "gwn_bvh_topology_build_binary.cuh"
#include "gwn_bvh_topology_build_collapse.cuh"
#include "gwn_bvh_topology_build_common.cuh"

namespace gwn {
namespace detail {

template <gwn_real_type Real, gwn_index_type Index, class MortonCode = std::uint64_t>
gwn_status gwn_bvh_topology_build_binary_lbvh(
    cuda::std::span<MortonCode const> const sorted_morton_codes,
    cuda::std::span<gwn_aabb<Real> const> const sorted_leaf_aabbs,
    gwn_device_array<gwn_binary_node<Index>> &binary_nodes,
    gwn_device_array<Index> &binary_internal_parent,
    gwn_device_array<gwn_aabb<Real>> &binary_internal_bounds,
    cudaStream_t const stream = gwn_default_stream()
) {
    static_assert(
        std::is_same_v<MortonCode, std::uint32_t> || std::is_same_v<MortonCode, std::uint64_t>,
        "MortonCode must be std::uint32_t or std::uint64_t."
    );
    std::size_t const primitive_count = sorted_morton_codes.size();
    if (primitive_count == 0) {
        GWN_RETURN_ON_ERROR(binary_nodes.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_parent.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_bounds.clear(stream));
        return gwn_status::ok();
    }

    if (sorted_leaf_aabbs.size() != primitive_count)
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_binary_lbvh,
            "LBVH binary builder sorted leaf AABB size mismatch."
        );

    if (primitive_count == 1) {
        GWN_RETURN_ON_ERROR(binary_nodes.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_parent.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_bounds.clear(stream));
        return gwn_status::ok();
    }

    std::size_t const binary_internal_count = primitive_count - 1;
    GWN_ASSERT(
        binary_internal_count == primitive_count - 1,
        "LBVH: binary internal count invariant violated"
    );
    constexpr int k_block_size = k_gwn_default_block_size;
    gwn_binary_parent_temporaries<Index> temps{};
    GWN_RETURN_ON_ERROR(gwn_prepare_binary_topology_buffers(
        primitive_count, binary_nodes, binary_internal_parent, temps, stream
    ));

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count - 1,
            gwn_build_binary_topology_functor<Index, MortonCode>{
                sorted_morton_codes, binary_nodes.span(), binary_internal_parent.span(),
                temps.internal_parent_slot.span(), temps.leaf_parent.span(),
                temps.leaf_parent_slot.span()
            },
            stream
        )
    );

    GWN_RETURN_ON_ERROR(binary_internal_bounds.resize(binary_internal_count, stream));
    gwn_device_array<gwn_aabb<Real>> pending_left_bounds{};
    gwn_device_array<gwn_aabb<Real>> pending_right_bounds{};
    gwn_device_array<unsigned int> child_arrival_count{};
    GWN_RETURN_ON_ERROR(pending_left_bounds.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(pending_right_bounds.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(child_arrival_count.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        child_arrival_count.data(), 0, binary_internal_count * sizeof(unsigned int), stream
    )));

    GWN_RETURN_ON_ERROR(
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

    return gwn_status::ok();
}

template <int Width, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_bvh_topology_build_collapse_binary_wide(
    cuda::std::span<gwn_binary_node<Index> const> const binary_nodes,
    cuda::std::span<gwn_aabb<Real> const> const binary_internal_bounds,
    std::size_t const primitive_count, gwn_bvh_topology_accessor<Width, Real, Index> &staging_topology,
    Index const root_internal_index, cudaStream_t const stream = gwn_default_stream()
) {
    static_assert(Width >= 2, "BVH width must be at least 2.");

    std::size_t const binary_internal_count = binary_nodes.size();
    if (binary_internal_count == 0)
        return gwn_status::ok();
    if (binary_internal_bounds.size() != binary_internal_count)
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_collapse,
            "Binary topology collapse bounds size does not match internal node count."
        );
    if (!gwn_index_in_bounds(root_internal_index, binary_internal_count))
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_collapse, "Binary topology collapse root index is invalid."
        );
    if (primitive_count != (binary_internal_count + 1))
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_collapse,
            "Binary topology collapse requires one primitive per binary leaf."
        );
    if (primitive_count > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_topology_collapse,
            "Binary topology collapse primitive count exceeds fixed-slot counter range."
        );

    constexpr int k_block_size = k_gwn_default_block_size;
    gwn_device_array<gwn_bvh_topology_node_soa<Width, Index>> wide_nodes_scratch{};
    gwn_device_array<gwn_collapse_slot_entry<Index>> slots{};
    gwn_device_array<unsigned int> tail{};
    gwn_device_array<unsigned int> wide_node_counter{};
    gwn_device_array<unsigned int> block_counter{};
    gwn_device_array<unsigned int> error_flag{};
    GWN_RETURN_ON_ERROR(wide_nodes_scratch.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(slots.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(tail.resize(1, stream));
    GWN_RETURN_ON_ERROR(wide_node_counter.resize(1, stream));
    GWN_RETURN_ON_ERROR(block_counter.resize(1, stream));
    GWN_RETURN_ON_ERROR(error_flag.resize(1, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(slots.data(), 0, primitive_count * sizeof(gwn_collapse_slot_entry<Index>), stream)
    ));

    gwn_collapse_slot_entry<Index> root_slot{};
    root_slot.work.binary_kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
    root_slot.work.binary_index = root_internal_index;
    root_slot.work.wide_node_id = Index(0);
    root_slot.state = k_gwn_collapse_slot_ready;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        slots.data(), &root_slot, sizeof(gwn_collapse_slot_entry<Index>), cudaMemcpyHostToDevice,
        stream
    )));

    unsigned int const tail_init = 1u;
    unsigned int const wide_node_counter_init = 1u;
    unsigned int const block_counter_init = 0u;
    unsigned int const error_flag_init = 0u;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        tail.data(), &tail_init, sizeof(unsigned int), cudaMemcpyHostToDevice, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        wide_node_counter.data(), &wide_node_counter_init, sizeof(unsigned int),
        cudaMemcpyHostToDevice, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        block_counter.data(), &block_counter_init, sizeof(unsigned int), cudaMemcpyHostToDevice,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        error_flag.data(), &error_flag_init, sizeof(unsigned int), cudaMemcpyHostToDevice, stream
    )));

    gwn_collapse_convert_kernel_params<Width, Real, Index> const params{
        binary_nodes, binary_internal_bounds, slots.span(), wide_nodes_scratch.span(), tail.data(),
        wide_node_counter.data(), block_counter.data(), error_flag.data()
    };
    int const block_count = gwn_block_count_1d<k_block_size>(primitive_count);
    gwn_collapse_convert_kernel<k_block_size, Width, Real, Index>
        <<<block_count, k_block_size, 0, stream>>>(params);
    GWN_RETURN_ON_ERROR(gwn_check_last_kernel());

    unsigned int host_error_flag = 0u;
    unsigned int host_wide_node_count = 0u;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_error_flag, error_flag.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_wide_node_count, wide_node_counter.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_error_flag != 0u) {
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_collapse,
            "Single-kernel wide collapse reported a fixed-slot conversion error."
        );
    }
    GWN_ASSERT(host_wide_node_count >= 1u, "collapse: at least the root wide node must exist");
    if (host_wide_node_count == 0u || host_wide_node_count > binary_internal_count)
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_collapse,
            "Single-kernel wide collapse produced an invalid output node count."
        );

    std::size_t const actual_wide_count = static_cast<std::size_t>(host_wide_node_count);
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging_topology.nodes, actual_wide_count, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_d2d(
        staging_topology.nodes,
        cuda::std::span<gwn_bvh_topology_node_soa<Width, Index> const>(
            wide_nodes_scratch.data(), actual_wide_count
        ),
        stream
    ));
    return gwn_status::ok();
}

} // namespace detail
} // namespace gwn
