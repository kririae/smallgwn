#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

#include "detail/gwn_bvh_build_collapse.cuh"
#include "detail/gwn_bvh_build_common.cuh"
#include "detail/gwn_bvh_build_taylor_async.cuh"
#include "detail/gwn_bvh_build_taylor_levelwise.cuh"
#include "detail/gwn_bvh_build_topology.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {
/// \brief Build a fixed-width LBVH topology from triangle geometry.
///
/// \remark This function only writes topology data (`nodes`, `primitive_indices`, root metadata).
/// \remark Existing topology memory in `bvh` is released before commit.
template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &bvh,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    static_assert(Width >= 2, "BVH width must be at least 2.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor is invalid for BVH construction.");

    if (geometry.triangle_count() == 0) {
        detail::gwn_release_bvh_topology_accessor(bvh, stream);
        return gwn_status::ok();
    }
    if (geometry.vertex_count() == 0)
        return gwn_status::invalid_argument("Cannot build BVH with triangles but zero vertices.");

    std::size_t const primitive_count = geometry.triangle_count();
    constexpr int k_block_size = detail::k_gwn_default_block_size;
    gwn_bvh_topology_accessor<Width, Real, Index> staging_bvh{};
    auto cleanup_staging_bvh = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_topology_accessor(staging_bvh, stream);
    });
    auto commit_staging_bvh = [&]() -> gwn_status {
        detail::gwn_release_bvh_topology_accessor(bvh, stream);
        bvh = staging_bvh;
        cleanup_staging_bvh.release();
        return gwn_status::ok();
    };

    Real scene_min_x = Real(0);
    Real scene_max_x = Real(0);
    Real scene_min_y = Real(0);
    Real scene_max_y = Real(0);
    Real scene_min_z = Real(0);
    Real scene_max_z = Real(0);
    gwn_device_array<Real> scene_axis_min{};
    gwn_device_array<Real> scene_axis_max{};
    gwn_device_array<std::uint8_t> scene_reduce_temp{};
    GWN_RETURN_ON_ERROR(scene_axis_min.resize(1, stream));
    GWN_RETURN_ON_ERROR(scene_axis_max.resize(1, stream));
    GWN_RETURN_ON_ERROR(
        detail::gwn_reduce_minmax_cub(
            geometry.vertex_x, scene_axis_min, scene_axis_max, scene_reduce_temp, scene_min_x,
            scene_max_x, stream
        )
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_reduce_minmax_cub(
            geometry.vertex_y, scene_axis_min, scene_axis_max, scene_reduce_temp, scene_min_y,
            scene_max_y, stream
        )
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_reduce_minmax_cub(
            geometry.vertex_z, scene_axis_min, scene_axis_max, scene_reduce_temp, scene_min_z,
            scene_max_z, stream
        )
    );
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

    Real const scene_inv_x =
        (scene_max_x > scene_min_x) ? Real(1) / (scene_max_x - scene_min_x) : Real(1);
    Real const scene_inv_y =
        (scene_max_y > scene_min_y) ? Real(1) / (scene_max_y - scene_min_y) : Real(1);
    Real const scene_inv_z =
        (scene_max_z > scene_min_z) ? Real(1) / (scene_max_z - scene_min_z) : Real(1);

    gwn_device_array<gwn_aabb<Real>> primitive_aabbs{};
    gwn_device_array<std::uint32_t> morton_codes{};
    gwn_device_array<Index> primitive_indices{};
    GWN_RETURN_ON_ERROR(primitive_aabbs.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(morton_codes.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(primitive_indices.resize(primitive_count, stream));
    auto const primitive_aabbs_span = primitive_aabbs.span();
    auto const morton_codes_span = morton_codes.span();
    auto const primitive_indices_span = primitive_indices.span();
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            detail::gwn_compute_triangle_aabbs_and_morton_functor<Real, Index>{
                geometry, scene_min_x, scene_min_y, scene_min_z, scene_inv_x, scene_inv_y,
                scene_inv_z, primitive_aabbs_span, morton_codes_span, primitive_indices_span
            },
            stream
        )
    );

    if (primitive_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return gwn_status::invalid_argument("CUB radix sort input exceeds int32 item count.");
    int const radix_item_count = static_cast<int>(primitive_count);
    gwn_device_array<std::uint32_t> sorted_morton_codes{};
    gwn_device_array<Index> sorted_primitive_indices{};
    gwn_device_array<std::uint8_t> radix_sort_temp{};
    GWN_RETURN_ON_ERROR(sorted_morton_codes.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(sorted_primitive_indices.resize(primitive_count, stream));

    std::size_t radix_sort_temp_bytes = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            nullptr, radix_sort_temp_bytes, morton_codes_span.data(), sorted_morton_codes.data(),
            primitive_indices_span.data(), sorted_primitive_indices.data(), radix_item_count, 0,
            static_cast<int>(sizeof(std::uint32_t) * 8), stream
        )
    ));
    GWN_RETURN_ON_ERROR(radix_sort_temp.resize(radix_sort_temp_bytes, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            radix_sort_temp.data(), radix_sort_temp_bytes, morton_codes_span.data(),
            sorted_morton_codes.data(), primitive_indices_span.data(),
            sorted_primitive_indices.data(), radix_item_count, 0,
            static_cast<int>(sizeof(std::uint32_t) * 8), stream
        )
    ));
    auto const sorted_morton_codes_span = sorted_morton_codes.span();
    auto const sorted_primitive_indices_span = sorted_primitive_indices.span();

    gwn_device_array<gwn_aabb<Real>> sorted_aabbs{};
    GWN_RETURN_ON_ERROR(sorted_aabbs.resize(primitive_count, stream));
    auto const sorted_aabbs_span = sorted_aabbs.span();
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            detail::gwn_gather_sorted_aabbs_functor<Real, Index>{
                primitive_aabbs_span, sorted_primitive_indices_span, sorted_aabbs_span
            },
            stream
        )
    );

    GWN_RETURN_ON_ERROR(
        detail::gwn_copy_device_to_span(
            staging_bvh.primitive_indices, sorted_primitive_indices_span.data(), primitive_count,
            stream
        )
    );

    if (primitive_count == 1) {
        staging_bvh.root_kind = gwn_bvh_child_kind::k_leaf;
        staging_bvh.root_index = Index(0);
        staging_bvh.root_count = Index(1);
        return commit_staging_bvh();
    }

    std::size_t const binary_internal_count = primitive_count - 1;
    gwn_device_array<detail::gwn_binary_node<Index>> binary_nodes{};
    gwn_device_array<Index> binary_internal_parent{};
    gwn_device_array<std::uint8_t> binary_internal_parent_slot{};
    gwn_device_array<Index> binary_leaf_parent{};
    gwn_device_array<std::uint8_t> binary_leaf_parent_slot{};
    GWN_RETURN_ON_ERROR(binary_nodes.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_internal_parent.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_internal_parent_slot.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_leaf_parent.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(binary_leaf_parent_slot.resize(primitive_count, stream));
    auto const binary_nodes_span = binary_nodes.span();
    auto const binary_internal_parent_span = binary_internal_parent.span();
    auto const binary_internal_parent_slot_span = binary_internal_parent_slot.span();
    auto const binary_leaf_parent_span = binary_leaf_parent.span();
    auto const binary_leaf_parent_slot_span = binary_leaf_parent_slot.span();
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        binary_internal_parent_span.data(), 0xff, binary_internal_count * sizeof(Index), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        binary_internal_parent_slot_span.data(), 0xff, binary_internal_count * sizeof(std::uint8_t),
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        binary_leaf_parent_span.data(), 0xff, primitive_count * sizeof(Index), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        binary_leaf_parent_slot_span.data(), 0xff, primitive_count * sizeof(std::uint8_t), stream
    )));
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            binary_internal_count,
            detail::gwn_build_binary_topology_functor<Index>{
                sorted_morton_codes_span, binary_nodes_span, binary_internal_parent_span,
                binary_internal_parent_slot_span, binary_leaf_parent_span,
                binary_leaf_parent_slot_span
            },
            stream
        )
    );

    Index const root_internal_index = Index(0);

    gwn_device_array<gwn_aabb<Real>> binary_internal_bounds{};
    gwn_device_array<gwn_aabb<Real>> binary_pending_left{};
    gwn_device_array<gwn_aabb<Real>> binary_pending_right{};
    gwn_device_array<unsigned int> binary_child_arrivals{};
    GWN_RETURN_ON_ERROR(binary_internal_bounds.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_pending_left.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_pending_right.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_child_arrivals.resize(binary_internal_count, stream));
    auto const binary_internal_bounds_span = binary_internal_bounds.span();
    auto const binary_pending_left_span = binary_pending_left.span();
    auto const binary_pending_right_span = binary_pending_right.span();
    auto const binary_child_arrivals_span = binary_child_arrivals.span();
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        binary_child_arrivals_span.data(), 0, binary_internal_count * sizeof(unsigned int), stream
    )));

    // Binary bounds pass: launch one thread per leaf and propagate bounds upward.
    // Each internal node finalizes after both child arrivals.
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            detail::gwn_accumulate_binary_bounds_pass_functor<Real, Index>{
                cuda::std::span<gwn_aabb<Real> const>(
                    sorted_aabbs_span.data(), sorted_aabbs_span.size()
                ),
                cuda::std::span<Index const>(
                    binary_leaf_parent_span.data(), binary_leaf_parent_span.size()
                ),
                cuda::std::span<std::uint8_t const>(
                    binary_leaf_parent_slot_span.data(), binary_leaf_parent_slot_span.size()
                ),
                cuda::std::span<Index const>(
                    binary_internal_parent_span.data(), binary_internal_parent_span.size()
                ),
                cuda::std::span<std::uint8_t const>(
                    binary_internal_parent_slot_span.data(), binary_internal_parent_slot_span.size()
                ),
                binary_internal_bounds_span, binary_pending_left_span, binary_pending_right_span,
                binary_child_arrivals_span
            },
            stream
        )
    );

    static_assert(
        (Width & (Width - 1)) == 0, "BVH collapse currently requires power-of-two Width."
    );
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
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_wide_count, &collapse_wide_count_init, sizeof(unsigned int),
        cudaMemcpyHostToDevice, stream
    )));
    std::uint8_t const collapse_root_is_wide = std::uint8_t(1);
    Index const collapse_root_wide_node_id = Index(0);
    Index const collapse_root_binary_root = root_internal_index;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_internal_is_wide_root_span.data(), &collapse_root_is_wide, sizeof(std::uint8_t),
        cudaMemcpyHostToDevice, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_internal_wide_node_id_span.data(), &collapse_root_wide_node_id, sizeof(Index),
        cudaMemcpyHostToDevice, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_wide_node_binary_root_span.data(), &collapse_root_binary_root, sizeof(Index),
        cudaMemcpyHostToDevice, stream
    )));

    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            binary_internal_count,
            detail::gwn_collapse_summarize_pass_functor<Width, Index>{
                cuda::std::span<Index const>(
                    binary_internal_parent_span.data(), binary_internal_parent_span.size()
                ),
                collapse_internal_is_wide_root_span, collapse_internal_wide_node_id_span,
                collapse_wide_node_binary_root_span, collapse_wide_count, root_internal_index
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

    GWN_RETURN_ON_ERROR(
        detail::gwn_allocate_span(
            staging_bvh.nodes, static_cast<std::size_t>(host_collapse_wide_count), stream
        )
    );
    auto const staging_nodes_span = cuda::std::span<gwn_bvh_node_soa<Width, Real, Index>>(
        const_cast<gwn_bvh_node_soa<Width, Real, Index> *>(staging_bvh.nodes.data()),
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
        detail::gwn_launch_linear_kernel<k_block_size>(
            static_cast<std::size_t>(host_collapse_wide_count),
            detail::gwn_collapse_emit_nodes_pass_functor<Width, Real, Index>{
                cuda::std::span<detail::gwn_binary_node<Index> const>(
                    binary_nodes_span.data(), binary_nodes_span.size()
                ),
                cuda::std::span<gwn_aabb<Real> const>(
                    sorted_aabbs_span.data(), sorted_aabbs_span.size()
                ),
                cuda::std::span<gwn_aabb<Real> const>(
                    binary_internal_bounds_span.data(), binary_internal_bounds_span.size()
                ),
                cuda::std::span<std::uint8_t const>(
                    collapse_internal_is_wide_root_span.data(),
                    collapse_internal_is_wide_root_span.size()
                ),
                cuda::std::span<Index const>(
                    collapse_internal_wide_node_id_span.data(),
                    collapse_internal_wide_node_id_span.size()
                ),
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

    staging_bvh.root_kind = gwn_bvh_child_kind::k_internal;
    staging_bvh.root_index = Index(0);
    staging_bvh.root_count = Index(0);
    return commit_staging_bvh();
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_build_bvh_lbvh.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_build_bvh_lbvh.");
}

/// \brief Convenience wrapper for `Width == 4`.
template <class Real, class Index>
gwn_status gwn_build_bvh4_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &bvh,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_build_bvh_lbvh<4, Real, Index>(geometry, bvh, stream);
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_build_bvh4_lbvh.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_build_bvh4_lbvh.");
}

/// \brief Build 4-wide LBVH topology and per-node Taylor data (fully GPU async propagation).
///
/// \remark This rebuilds topology first, then computes the requested Taylor order.
/// \remark Previously stored data orders are released before writing the requested one.
template <int Order, class Real, class Index>
gwn_status gwn_build_bvh4_lbvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &topology,
    gwn_bvh_data4_accessor<Real, Index> &data_tree, cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    static_assert(
        Order == 0 || Order == 1,
        "gwn_build_bvh4_lbvh_taylor currently supports Order 0 and Order 1."
    );

    GWN_RETURN_ON_ERROR(gwn_build_bvh4_lbvh(geometry, topology, stream));

    detail::gwn_release_bvh_data_accessor(data_tree, stream);

    if (!topology.has_internal_root())
        return gwn_status::ok();

    using moment_type = detail::gwn_device_taylor_moment<Order, Real>;
    using taylor_node_type = gwn_bvh4_taylor_node_soa<Order, Real>;

    std::size_t const node_count = topology.nodes.size();
    if (node_count == 0)
        return gwn_status::ok();

    if (topology.root_index < Index(0) ||
        static_cast<std::size_t>(topology.root_index) >= node_count)
        return gwn_status::internal_error("BVH root index out of range for Taylor construction.");
    if (node_count > (std::numeric_limits<std::size_t>::max() / std::size_t(4)))
        return gwn_status::internal_error("Taylor async construction node count overflow.");

    std::size_t const pending_count = node_count * std::size_t(4);
    constexpr int k_block_size = detail::k_gwn_default_block_size;

    cuda::std::span<taylor_node_type const> taylor_nodes_device{};
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(taylor_nodes_device, node_count, stream));
    auto cleanup_taylor_nodes = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_span(taylor_nodes_device, stream);
    });

    gwn_device_array<Index> internal_parent_device{};
    gwn_device_array<std::uint8_t> internal_parent_slot_device{};
    gwn_device_array<std::uint8_t> internal_arity_device{};
    gwn_device_array<unsigned int> internal_arrivals_device{};
    gwn_device_array<moment_type> pending_child_moments_device{};
    GWN_RETURN_ON_ERROR(internal_parent_device.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(internal_parent_slot_device.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(internal_arity_device.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(internal_arrivals_device.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(pending_child_moments_device.resize(pending_count, stream));

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        const_cast<taylor_node_type *>(taylor_nodes_device.data()), 0,
        node_count * sizeof(taylor_node_type), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(internal_parent_device.data(), 0xff, node_count * sizeof(Index), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        internal_parent_slot_device.data(), 0xff, node_count * sizeof(std::uint8_t), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(internal_arity_device.data(), 0, node_count * sizeof(std::uint8_t), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        internal_arrivals_device.data(), 0, node_count * sizeof(unsigned int), stream
    )));

    gwn_device_array<unsigned int> error_flag_device{};
    GWN_RETURN_ON_ERROR(error_flag_device.resize(1, stream));
    GWN_RETURN_ON_ERROR(error_flag_device.zero(stream));
    unsigned int *error_flag = error_flag_device.data();

    auto const internal_parent = internal_parent_device.span();
    auto const internal_parent_slot = internal_parent_slot_device.span();
    auto const internal_arity = internal_arity_device.span();
    auto const internal_arrivals = internal_arrivals_device.span();
    auto const pending_child_moments = pending_child_moments_device.span();
    auto const taylor_nodes = cuda::std::span<taylor_node_type>(
        const_cast<taylor_node_type *>(taylor_nodes_device.data()), taylor_nodes_device.size()
    );

    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            node_count,
            detail::gwn_prepare_taylor_async_topology_functor<Real, Index>{
                topology, internal_parent, internal_parent_slot, internal_arity, error_flag
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            pending_count,
            detail::gwn_build_taylor_async_from_leaves_functor<Order, Real, Index>{
                geometry, topology,
                cuda::std::span<Index const>(internal_parent.data(), internal_parent.size()),
                cuda::std::span<std::uint8_t const>(
                    internal_parent_slot.data(), internal_parent_slot.size()
                ),
                cuda::std::span<std::uint8_t const>(internal_arity.data(), internal_arity.size()),
                internal_arrivals, pending_child_moments, taylor_nodes, error_flag
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            node_count,
            detail::gwn_validate_taylor_async_convergence_functor<Index>{
                cuda::std::span<std::uint8_t const>(internal_arity.data(), internal_arity.size()),
                cuda::std::span<unsigned int const>(
                    internal_arrivals.data(), internal_arrivals.size()
                ),
                error_flag
            },
            stream
        )
    );

    unsigned int host_error_flag = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_error_flag, error_flag_device.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_error_flag != 0) {
        return gwn_status::internal_error(
            "Taylor async construction failed topology/propagation validation."
        );
    }

    if constexpr (Order == 0)
        data_tree.taylor_order0_nodes = taylor_nodes_device;
    else
        data_tree.taylor_order1_nodes = taylor_nodes_device;
    cleanup_taylor_nodes.release();
    return gwn_status::ok();
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_build_bvh4_lbvh_taylor.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_build_bvh4_lbvh_taylor.");
}

/// \brief Build 4-wide LBVH topology and Taylor data with host-reconstructed level order.
///
/// \remark This variant computes Taylor nodes from deepest level to root.
template <int Order, class Real, class Index>
gwn_status gwn_build_bvh4_lbvh_taylor_levelwise(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &topology,
    gwn_bvh_data4_accessor<Real, Index> &data_tree, cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    static_assert(
        Order == 0 || Order == 1,
        "gwn_build_bvh4_lbvh_taylor_levelwise currently supports Order 0 and Order 1."
    );

    GWN_RETURN_ON_ERROR(gwn_build_bvh4_lbvh(geometry, topology, stream));

    detail::gwn_release_bvh_data_accessor(data_tree, stream);

    if (!topology.has_internal_root())
        return gwn_status::ok();

    using moment_type = detail::gwn_device_taylor_moment<Order, Real>;
    using taylor_node_type = gwn_bvh4_taylor_node_soa<Order, Real>;

    std::size_t const node_count = topology.nodes.size();
    if (node_count == 0)
        return gwn_status::ok();
    if (topology.root_index != Index(0))
        return gwn_status::internal_error(
            "Taylor levelwise construction expects root node index to be zero."
        );

    std::vector<std::vector<Index>> level_node_ids{};
    {
        std::vector<gwn_bvh4_node_soa<Real, Index>> host_nodes(node_count);
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
            host_nodes.data(), topology.nodes.data(),
            node_count * sizeof(gwn_bvh4_node_soa<Real, Index>), cudaMemcpyDeviceToHost, stream
        )));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

        std::vector<int> node_depth(node_count, -1);
        std::vector<Index> queue{};
        queue.reserve(node_count);
        queue.push_back(topology.root_index);
        node_depth[static_cast<std::size_t>(topology.root_index)] = 0;

        std::size_t queue_head = 0;
        while (queue_head < queue.size()) {
            Index const node_index = queue[queue_head++];
            if (node_index < Index(0) || static_cast<std::size_t>(node_index) >= node_count) {
                return gwn_status::internal_error(
                    "Levelwise Taylor traversal encountered out-of-range node index."
                );
            }

            int const depth = node_depth[static_cast<std::size_t>(node_index)];
            if (depth < 0)
                continue;

            if (level_node_ids.size() <= static_cast<std::size_t>(depth))
                level_node_ids.resize(static_cast<std::size_t>(depth) + 1);
            level_node_ids[static_cast<std::size_t>(depth)].push_back(node_index);

            gwn_bvh4_node_soa<Real, Index> const &node =
                host_nodes[static_cast<std::size_t>(node_index)];
            for (int child_slot = 0; child_slot < 4; ++child_slot) {
                if (static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]) !=
                    gwn_bvh_child_kind::k_internal) {
                    continue;
                }

                Index const child_index = node.child_index[child_slot];
                if (child_index < Index(0) || static_cast<std::size_t>(child_index) >= node_count) {
                    return gwn_status::internal_error(
                        "Levelwise Taylor traversal encountered out-of-range child node index."
                    );
                }

                if (node_depth[static_cast<std::size_t>(child_index)] >= 0)
                    continue;

                node_depth[static_cast<std::size_t>(child_index)] = depth + 1;
                queue.push_back(child_index);
            }
        }

        if (level_node_ids.empty())
            return gwn_status::ok();
    }

    std::size_t counted_nodes = 0;
    for (auto const &level_nodes : level_node_ids)
        counted_nodes += level_nodes.size();
    if (counted_nodes != node_count)
        return gwn_status::internal_error("Levelwise Taylor node-count reconstruction mismatch.");

    for (std::size_t level = 0; level < level_node_ids.size(); ++level) {
        std::vector<Index> &level_nodes = level_node_ids[level];
        if (level_nodes.empty())
            continue;
        std::sort(level_nodes.begin(), level_nodes.end());
    }

    cuda::std::span<taylor_node_type const> taylor_nodes_device{};
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(taylor_nodes_device, node_count, stream));
    auto cleanup_taylor_nodes = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_span(taylor_nodes_device, stream);
    });

    gwn_device_array<moment_type> node_moments_device{};
    GWN_RETURN_ON_ERROR(node_moments_device.resize(node_count, stream));

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        const_cast<taylor_node_type *>(taylor_nodes_device.data()), 0,
        node_count * sizeof(taylor_node_type), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(node_moments_device.data(), 0, node_count * sizeof(moment_type), stream)
    ));

    auto const node_moments = node_moments_device.span();
    auto const taylor_nodes = cuda::std::span<taylor_node_type>(
        const_cast<taylor_node_type *>(taylor_nodes_device.data()), node_count
    );

    constexpr int k_block_size = detail::k_gwn_default_block_size;
    for (std::size_t level = level_node_ids.size(); level-- > 0;) {
        std::vector<Index> const &level_nodes = level_node_ids[level];
        if (level_nodes.empty())
            continue;

        gwn_device_array<Index> level_node_ids_device{};
        GWN_RETURN_ON_ERROR(level_node_ids_device.copy_from_host(
            cuda::std::span<Index const>(level_nodes.data(), level_nodes.size()), stream
        ));
        auto const level_node_ids_span = cuda::std::span<Index const>(
            level_node_ids_device.data(), level_node_ids_device.size()
        );

        GWN_RETURN_ON_ERROR(
            detail::gwn_launch_linear_kernel<k_block_size>(
                level_node_ids_span.size(),
                detail::gwn_build_taylor_levelwise_functor<Order, Real, Index>{
                    geometry, topology, node_moments, taylor_nodes, level_node_ids_span
                },
                stream
            )
        );
    }

    if constexpr (Order == 0)
        data_tree.taylor_order0_nodes = taylor_nodes_device;
    else
        data_tree.taylor_order1_nodes = taylor_nodes_device;
    cleanup_taylor_nodes.release();
    return gwn_status::ok();
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in gwn_build_bvh4_lbvh_taylor_levelwise."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in gwn_build_bvh4_lbvh_taylor_levelwise."
    );
}
} // namespace gwn
