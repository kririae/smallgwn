#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "../gwn_kernel_utils.cuh"
#include "gwn_bvh_topology_build_binary.cuh"
#include "gwn_bvh_topology_build_collapse.cuh"
#include "gwn_bvh_topology_build_common.cuh"

namespace gwn {
namespace detail {

template <class Real, class Index, class MortonCode = std::uint64_t>
gwn_status gwn_bvh_topology_build_binary_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_device_array<Index> &sorted_primitive_indices,
    gwn_device_array<gwn_binary_node<Index>> &binary_nodes,
    gwn_device_array<Index> &binary_internal_parent,
    cudaStream_t const stream = gwn_default_stream()
) {
    std::size_t const primitive_count = geometry.triangle_count();
    if (primitive_count == 0) {
        GWN_RETURN_ON_ERROR(sorted_primitive_indices.clear(stream));
        GWN_RETURN_ON_ERROR(binary_nodes.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_parent.clear(stream));
        return gwn_status::ok();
    }

    gwn_scene_aabb<Real> scene{};
    GWN_RETURN_ON_ERROR(gwn_compute_scene_aabb(geometry, scene, stream));

    gwn_device_array<MortonCode> sorted_morton_codes{};
    gwn_device_array<gwn_aabb<Real>> primitive_aabbs{};
    GWN_RETURN_ON_ERROR((gwn_compute_and_sort_morton<MortonCode>(
        geometry, scene, sorted_primitive_indices, sorted_morton_codes, primitive_aabbs, stream
    )));

    if (primitive_count == 1) {
        GWN_RETURN_ON_ERROR(binary_nodes.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_parent.clear(stream));
        return gwn_status::ok();
    }

    constexpr int k_block_size = k_gwn_default_block_size;
    gwn_binary_parent_temporaries<Index> temps{};
    GWN_RETURN_ON_ERROR(gwn_prepare_binary_topology_buffers(
        primitive_count, binary_nodes, binary_internal_parent, temps, stream
    ));

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count - 1,
            gwn_build_binary_topology_functor<Index, MortonCode>{
                sorted_morton_codes.span(), binary_nodes.span(), binary_internal_parent.span(),
                temps.internal_parent_slot.span(), temps.leaf_parent.span(),
                temps.leaf_parent_slot.span()
            },
            stream
        )
    );

    return gwn_status::ok();
}

template <int Width, class Real, class Index>
gwn_status gwn_bvh_topology_build_collapse_binary_lbvh(
    cuda::std::span<gwn_binary_node<Index> const> const binary_nodes,
    cuda::std::span<Index const> const binary_internal_parent,
    gwn_bvh_topology_accessor<Width, Real, Index> &staging_topology,
    Index const root_internal_index, cudaStream_t const stream = gwn_default_stream()
) {
    static_assert(
        (Width & (Width - 1)) == 0, "BVH collapse currently requires power-of-two Width."
    );

    std::size_t const binary_internal_count = binary_nodes.size();
    if (binary_internal_count == 0)
        return gwn_status::ok();
    if (binary_internal_parent.size() != binary_internal_count)
        return gwn_status::internal_error("Binary topology collapse input size mismatch.");
    if (!gwn_index_in_bounds(root_internal_index, binary_internal_count))
        return gwn_status::internal_error("Binary topology collapse root index is invalid.");

    constexpr int k_block_size = k_gwn_default_block_size;

    gwn_device_array<std::uint8_t> collapse_internal_is_wide_root{};
    gwn_device_array<Index> collapse_internal_wide_node_id{};
    gwn_device_array<Index> collapse_wide_node_binary_root{};
    GWN_RETURN_ON_ERROR(collapse_internal_is_wide_root.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(collapse_internal_wide_node_id.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(collapse_wide_node_binary_root.resize(binary_internal_count, stream));

    auto const collapse_internal_is_wide_root_span = collapse_internal_is_wide_root.span();
    auto const collapse_internal_wide_node_id_span = collapse_internal_wide_node_id.span();
    auto const collapse_wide_node_binary_root_span = collapse_wide_node_binary_root.span();
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        collapse_internal_is_wide_root_span.data(), 0, binary_internal_count * sizeof(std::uint8_t),
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        collapse_internal_wide_node_id_span.data(), 0xff, binary_internal_count * sizeof(Index),
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        collapse_wide_node_binary_root_span.data(), 0xff, binary_internal_count * sizeof(Index),
        stream
    )));

    void *collapse_wide_count_raw = nullptr;
    GWN_RETURN_ON_ERROR(gwn_cuda_malloc(&collapse_wide_count_raw, sizeof(unsigned int), stream));
    auto cleanup_collapse_wide_count = gwn_make_scope_exit([&]() noexcept {
        (void)gwn_cuda_free(collapse_wide_count_raw, stream);
    });
    unsigned int *collapse_wide_count = static_cast<unsigned int *>(collapse_wide_count_raw);
    unsigned int const collapse_wide_count_init = 1u;
    std::size_t const root_internal_index_u = static_cast<std::size_t>(root_internal_index);
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_wide_count, &collapse_wide_count_init, sizeof(unsigned int),
        cudaMemcpyHostToDevice, stream
    )));
    std::uint8_t const collapse_root_is_wide = std::uint8_t(1);
    Index const collapse_root_wide_node_id = Index(0);
    Index const collapse_root_binary_root = root_internal_index;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_internal_is_wide_root_span.data() + root_internal_index_u, &collapse_root_is_wide,
        sizeof(std::uint8_t), cudaMemcpyHostToDevice, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_internal_wide_node_id_span.data() + root_internal_index_u,
        &collapse_root_wide_node_id, sizeof(Index), cudaMemcpyHostToDevice, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_wide_node_binary_root_span.data(), &collapse_root_binary_root, sizeof(Index),
        cudaMemcpyHostToDevice, stream
    )));

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            binary_internal_count,
            gwn_collapse_summarize_pass_functor<Width, Index>{
                binary_internal_parent, collapse_internal_is_wide_root_span,
                collapse_internal_wide_node_id_span, collapse_wide_node_binary_root_span,
                collapse_wide_count, root_internal_index
            },
            stream
        )
    );

    unsigned int host_collapse_wide_count = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_collapse_wide_count, collapse_wide_count, sizeof(unsigned int),
        cudaMemcpyDeviceToHost, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

    if (host_collapse_wide_count == 0 || host_collapse_wide_count > binary_internal_count) {
        return gwn_status::internal_error(
            "BVH collapse summarize produced invalid wide-node count."
        );
    }

    GWN_RETURN_ON_ERROR(gwn_allocate_span(
        staging_topology.nodes, static_cast<std::size_t>(host_collapse_wide_count), stream
    ));
    auto const staging_nodes_span = cuda::std::span<gwn_bvh_topology_node_soa<Width, Index>>(
        const_cast<gwn_bvh_topology_node_soa<Width, Index> *>(staging_topology.nodes.data()),
        static_cast<std::size_t>(host_collapse_wide_count)
    );

    void *collapse_overflow_raw = nullptr;
    GWN_RETURN_ON_ERROR(gwn_cuda_malloc(&collapse_overflow_raw, sizeof(unsigned int), stream));
    auto cleanup_collapse_overflow =
        gwn_make_scope_exit([&]() noexcept { (void)gwn_cuda_free(collapse_overflow_raw, stream); });
    unsigned int *collapse_overflow = static_cast<unsigned int *>(collapse_overflow_raw);
    GWN_RETURN_ON_ERROR(
        gwn_cuda_to_status(cudaMemsetAsync(collapse_overflow, 0, sizeof(unsigned int), stream))
    );

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            static_cast<std::size_t>(host_collapse_wide_count),
            gwn_collapse_emit_nodes_pass_functor<Width, Index>{
                binary_nodes, collapse_internal_is_wide_root_span,
                collapse_internal_wide_node_id_span,
                cuda::std::span<Index const>(
                    collapse_wide_node_binary_root_span.data(),
                    static_cast<std::size_t>(host_collapse_wide_count)
                ),
                staging_nodes_span, collapse_overflow
            },
            stream
        )
    );

    unsigned int host_collapse_overflow = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_collapse_overflow, collapse_overflow, sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_collapse_overflow != 0) {
        return gwn_status::internal_error(
            "BVH collapse execute pass overflowed fixed-width node capacity."
        );
    }

    return gwn_status::ok();
}

} // namespace detail
} // namespace gwn
