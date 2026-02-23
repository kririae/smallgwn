#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>

#include <cub/device/device_radix_sort.cuh>

#include "detail/gwn_bvh_build_collapse.cuh"
#include "detail/gwn_bvh_build_common.cuh"
#include "detail/gwn_bvh_build_topology.cuh"
#include "detail/gwn_bvh_refit_async.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {

namespace detail {

template <class Real, class Index>
gwn_status gwn_build_binary_lbvh_topology(
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

    constexpr int k_block_size = k_gwn_default_block_size;

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
    GWN_RETURN_ON_ERROR(gwn_reduce_minmax_cub(
        geometry.vertex_x, scene_axis_min, scene_axis_max, scene_reduce_temp, scene_min_x,
        scene_max_x, stream
    ));
    GWN_RETURN_ON_ERROR(gwn_reduce_minmax_cub(
        geometry.vertex_y, scene_axis_min, scene_axis_max, scene_reduce_temp, scene_min_y,
        scene_max_y, stream
    ));
    GWN_RETURN_ON_ERROR(gwn_reduce_minmax_cub(
        geometry.vertex_z, scene_axis_min, scene_axis_max, scene_reduce_temp, scene_min_z,
        scene_max_z, stream
    ));
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
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            gwn_compute_triangle_aabbs_and_morton_functor<Real, Index>{
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

    if (primitive_count == 1) {
        GWN_RETURN_ON_ERROR(binary_nodes.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_parent.clear(stream));
        return gwn_status::ok();
    }

    std::size_t const binary_internal_count = primitive_count - 1;
    gwn_device_array<std::uint8_t> binary_internal_parent_slot{};
    gwn_device_array<Index> binary_leaf_parent{};
    gwn_device_array<std::uint8_t> binary_leaf_parent_slot{};
    GWN_RETURN_ON_ERROR(binary_nodes.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_internal_parent.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_internal_parent_slot.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_leaf_parent.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(binary_leaf_parent_slot.resize(primitive_count, stream));

    auto const sorted_morton_codes_span = sorted_morton_codes.span();
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
        gwn_launch_linear_kernel<k_block_size>(
            binary_internal_count,
            gwn_build_binary_topology_functor<Index>{
                sorted_morton_codes_span, binary_nodes_span, binary_internal_parent_span,
                binary_internal_parent_slot_span, binary_leaf_parent_span,
                binary_leaf_parent_slot_span
            },
            stream
        )
    );

    return gwn_status::ok();
}

template <int Width, class Real, class Index>
gwn_status gwn_collapse_binary_lbvh_topology(
    cuda::std::span<gwn_binary_node<Index> const> const binary_nodes,
    cuda::std::span<Index const> const binary_internal_parent,
    gwn_bvh_topology_accessor<Width, Real, Index> &staging_topology,
    cudaStream_t const stream = gwn_default_stream()
) {
    static_assert(
        (Width & (Width - 1)) == 0, "BVH collapse currently requires power-of-two Width."
    );

    std::size_t const binary_internal_count = binary_nodes.size();
    if (binary_internal_count == 0)
        return gwn_status::ok();
    if (binary_internal_parent.size() != binary_internal_count)
        return gwn_status::internal_error("Binary topology collapse input size mismatch.");

    constexpr int k_block_size = k_gwn_default_block_size;
    Index const root_internal_index = Index(0);

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

namespace detail {

/// \brief Build a fixed-width LBVH topology tree from triangle geometry.
template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    static_assert(Width >= 2, "BVH width must be at least 2.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor is invalid for BVH construction.");

    if (geometry.triangle_count() == 0) {
        detail::gwn_release_bvh_topology_tree_accessor(topology, stream);
        return gwn_status::ok();
    }
    if (geometry.vertex_count() == 0)
        return gwn_status::invalid_argument("Cannot build BVH with triangles but zero vertices.");

    std::size_t const primitive_count = geometry.triangle_count();

    gwn_bvh_topology_accessor<Width, Real, Index> staging_topology{};
    auto cleanup_staging_topology = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_topology_tree_accessor(staging_topology, stream);
    });
    auto commit_staging_topology = [&]() -> gwn_status {
        detail::gwn_release_bvh_topology_tree_accessor(topology, stream);
        topology = staging_topology;
        cleanup_staging_topology.release();
        return gwn_status::ok();
    };

    gwn_device_array<Index> sorted_primitive_indices{};
    gwn_device_array<detail::gwn_binary_node<Index>> binary_nodes{};
    gwn_device_array<Index> binary_internal_parent{};
    GWN_RETURN_ON_ERROR(
        detail::gwn_build_binary_lbvh_topology(
            geometry, sorted_primitive_indices, binary_nodes, binary_internal_parent, stream
        )
    );

    GWN_RETURN_ON_ERROR(
        detail::gwn_allocate_span(staging_topology.primitive_indices, primitive_count, stream)
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_copy_d2d(
            staging_topology.primitive_indices,
            cuda::std::span<Index const>(
                sorted_primitive_indices.data(), sorted_primitive_indices.size()
            ),
            stream
        )
    );

    if (primitive_count == 1) {
        staging_topology.root_kind = gwn_bvh_child_kind::k_leaf;
        staging_topology.root_index = Index(0);
        staging_topology.root_count = Index(1);
        return commit_staging_topology();
    }

    GWN_RETURN_ON_ERROR((detail::gwn_collapse_binary_lbvh_topology<Width, Real, Index>(
        cuda::std::span<detail::gwn_binary_node<Index> const>(
            binary_nodes.data(), binary_nodes.size()
        ),
        cuda::std::span<Index const>(binary_internal_parent.data(), binary_internal_parent.size()),
        staging_topology, stream
    )));

    staging_topology.root_kind = gwn_bvh_child_kind::k_internal;
    staging_topology.root_index = Index(0);
    staging_topology.root_count = Index(0);
    return commit_staging_topology();
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_build_bvh_topology_lbvh.");
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in gwn_build_bvh_topology_lbvh."
    );
}

template <class Real, class Index>
gwn_status gwn_build_bvh4_topology_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_build_bvh_topology_lbvh<4, Real, Index>(geometry, topology, stream);
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_build_bvh4_topology_lbvh.");
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in gwn_build_bvh4_topology_lbvh."
    );
}

/// \brief Refit AABB payload tree for a prebuilt topology tree.
template <int Width, class Real, class Index>
gwn_status gwn_refit_bvh_aabb(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor is invalid for AABB refit.");
    if (!topology.is_valid())
        return gwn_status::invalid_argument("Topology accessor is invalid for AABB refit.");

    gwn_bvh_aabb_accessor<Width, Real, Index> staging_aabb{};
    auto cleanup_staging_aabb = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_aabb_tree_accessor(staging_aabb, stream);
    });
    auto commit_staging_aabb = [&]() -> gwn_status {
        detail::gwn_release_bvh_aabb_tree_accessor(aabb_tree, stream);
        aabb_tree = staging_aabb;
        cleanup_staging_aabb.release();
        return gwn_status::ok();
    };

    if (!topology.has_internal_root())
        return commit_staging_aabb();

    std::size_t const node_count = topology.nodes.size();
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(staging_aabb.nodes, node_count, stream));
    auto const staging_nodes = cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real>>(
        const_cast<gwn_bvh_aabb_node_soa<Width, Real> *>(staging_aabb.nodes.data()), node_count
    );
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(staging_nodes.data(), 0, node_count * sizeof(staging_nodes[0]), stream)
    ));

    using traits = detail::gwn_aabb_refit_traits<Width, Real, Index>;
    typename traits::output_context const output_context{staging_nodes};
    GWN_RETURN_ON_ERROR((detail::gwn_run_refit_pass<traits, Width, Real, Index>(
        geometry, topology, output_context, stream
    )));

    return commit_staging_aabb();
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_refit_bvh_aabb.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_refit_bvh_aabb.");
}

/// \brief Refit Taylor/moment payload tree for a prebuilt topology tree.
template <int Order, int Width, class Real, class Index>
gwn_status gwn_refit_bvh_moment(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_accessor<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    static_assert(
        Order == 0 || Order == 1, "gwn_refit_bvh_moment currently supports Order 0 and Order 1."
    );

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor is invalid for moment refit.");
    if (!topology.is_valid())
        return gwn_status::invalid_argument("Topology accessor is invalid for moment refit.");
    if (!aabb_tree.is_valid_for(topology))
        return gwn_status::invalid_argument("AABB tree accessor is invalid for moment refit.");

    gwn_bvh_moment_accessor<Width, Real, Index> staging_moment{};
    auto cleanup_staging_moment = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_moment_tree_accessor(staging_moment, stream);
    });
    auto commit_staging_moment = [&]() -> gwn_status {
        detail::gwn_release_bvh_moment_tree_accessor(moment_tree, stream);
        moment_tree = staging_moment;
        cleanup_staging_moment.release();
        return gwn_status::ok();
    };

    if (!topology.has_internal_root())
        return commit_staging_moment();

    using node_type = gwn_bvh_taylor_node_soa<Width, Order, Real>;
    cuda::std::span<node_type const> staging_nodes{};
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(staging_nodes, topology.nodes.size(), stream));
    auto cleanup_staging_nodes =
        gwn_make_scope_exit([&]() noexcept { detail::gwn_free_span(staging_nodes, stream); });
    auto const staging_nodes_mutable = cuda::std::span<node_type>(
        const_cast<node_type *>(staging_nodes.data()), staging_nodes.size()
    );
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        staging_nodes_mutable.data(), 0, staging_nodes_mutable.size() * sizeof(node_type), stream
    )));

    using traits = detail::gwn_moment_refit_traits<Order, Width, Real, Index>;
    typename traits::output_context const output_context{
        staging_nodes_mutable, cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real> const>(
                                   aabb_tree.nodes.data(), aabb_tree.nodes.size()
                               )
    };
    GWN_RETURN_ON_ERROR((detail::gwn_run_refit_pass<traits, Width, Real, Index>(
        geometry, topology, output_context, stream
    )));

    if constexpr (Order == 0)
        staging_moment.taylor_order0_nodes = staging_nodes;
    else
        staging_moment.taylor_order1_nodes = staging_nodes;
    cleanup_staging_nodes.release();

    return commit_staging_moment();
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_refit_bvh_moment.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_refit_bvh_moment.");
}

template <class Real, class Index>
gwn_status gwn_refit_bvh4_aabb(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_accessor<Real, Index> const &topology, gwn_bvh_aabb4_accessor<Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_refit_bvh_aabb<4, Real, Index>(geometry, topology, aabb_tree, stream);
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_refit_bvh4_aabb.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_refit_bvh4_aabb.");
}

template <int Order, class Real, class Index>
gwn_status gwn_refit_bvh4_moment(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_accessor<Real, Index> const &topology,
    gwn_bvh_aabb4_accessor<Real, Index> const &aabb_tree,
    gwn_bvh_moment4_accessor<Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_refit_bvh_moment<Order, 4, Real, Index>(
        geometry, topology, aabb_tree, moment_tree, stream
    );
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_refit_bvh4_moment.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_refit_bvh4_moment.");
}

/// \brief Build topology and AABB trees from geometry.
template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_aabb_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    GWN_RETURN_ON_ERROR(
        (gwn_build_bvh_topology_lbvh<Width, Real, Index>(geometry, topology, stream))
    );
    return gwn_refit_bvh_aabb<Width, Real, Index>(geometry, topology, aabb_tree, stream);
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in gwn_build_bvh_topology_aabb_lbvh."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in gwn_build_bvh_topology_aabb_lbvh."
    );
}

template <class Real, class Index>
gwn_status gwn_build_bvh4_topology_aabb_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &topology,
    gwn_bvh_aabb4_accessor<Real, Index> &aabb_tree, cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_build_bvh_topology_aabb_lbvh<4, Real, Index>(geometry, topology, aabb_tree, stream);
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in gwn_build_bvh4_topology_aabb_lbvh."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in gwn_build_bvh4_topology_aabb_lbvh."
    );
}

/// \brief Build topology, AABB, and moment trees from geometry.
template <int Order, int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_aabb_moment_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree,
    gwn_bvh_moment_accessor<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    GWN_RETURN_ON_ERROR((
        gwn_build_bvh_topology_aabb_lbvh<Width, Real, Index>(geometry, topology, aabb_tree, stream)
    ));
    return gwn_refit_bvh_moment<Order, Width, Real, Index>(
        geometry, topology, aabb_tree, moment_tree, stream
    );
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in gwn_build_bvh_topology_aabb_moment_lbvh."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in gwn_build_bvh_topology_aabb_moment_lbvh."
    );
}

template <int Order, class Real, class Index>
gwn_status gwn_build_bvh4_topology_aabb_moment_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &topology,
    gwn_bvh_aabb4_accessor<Real, Index> &aabb_tree,
    gwn_bvh_moment4_accessor<Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_build_bvh_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
        geometry, topology, aabb_tree, moment_tree, stream
    );
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in gwn_build_bvh4_topology_aabb_moment_lbvh."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in gwn_build_bvh4_topology_aabb_moment_lbvh."
    );
}

} // namespace detail

template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    GWN_RETURN_ON_ERROR((detail::gwn_build_bvh_topology_lbvh<Width, Real, Index>(
        geometry.accessor(), topology.accessor(), stream
    )));
    topology.set_stream(stream);
    return gwn_status::ok();
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in object overload of gwn_build_bvh_topology_lbvh."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in object overload of gwn_build_bvh_topology_lbvh."
    );
}

template <class Real, class Index>
gwn_status gwn_build_bvh4_topology_lbvh(
    gwn_geometry_object<Real, Index> const &geometry, gwn_bvh_object<Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_build_bvh_topology_lbvh<4, Real, Index>(geometry, topology, stream);
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in object overload of gwn_build_bvh4_topology_lbvh."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in object overload of gwn_build_bvh4_topology_lbvh."
    );
}

template <int Width, class Real, class Index>
gwn_status gwn_refit_bvh_aabb(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> const &topology,
    gwn_bvh_aabb_tree_object_t<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    GWN_RETURN_ON_ERROR((detail::gwn_refit_bvh_aabb<Width, Real, Index>(
        geometry.accessor(), topology.accessor(), aabb_tree.accessor(), stream
    )));
    aabb_tree.set_stream(stream);
    return gwn_status::ok();
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in object overload of gwn_refit_bvh_aabb."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in object overload of gwn_refit_bvh_aabb."
    );
}

template <class Real, class Index>
gwn_status gwn_refit_bvh4_aabb(
    gwn_geometry_object<Real, Index> const &geometry, gwn_bvh_object<Real, Index> const &topology,
    gwn_bvh_aabb_object<Real, Index> &aabb_tree, cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_refit_bvh_aabb<4, Real, Index>(geometry, topology, aabb_tree, stream);
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in object overload of gwn_refit_bvh4_aabb."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in object overload of gwn_refit_bvh4_aabb."
    );
}

template <int Order, int Width, class Real, class Index>
gwn_status gwn_refit_bvh_moment(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> const &topology,
    gwn_bvh_aabb_tree_object_t<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_object_t<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    GWN_RETURN_ON_ERROR((detail::gwn_refit_bvh_moment<Order, Width, Real, Index>(
        geometry.accessor(), topology.accessor(), aabb_tree.accessor(), moment_tree.accessor(),
        stream
    )));
    moment_tree.set_stream(stream);
    return gwn_status::ok();
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in object overload of gwn_refit_bvh_moment."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in object overload of gwn_refit_bvh_moment."
    );
}

template <int Order, class Real, class Index>
gwn_status gwn_refit_bvh4_moment(
    gwn_geometry_object<Real, Index> const &geometry, gwn_bvh_object<Real, Index> const &topology,
    gwn_bvh_aabb_object<Real, Index> const &aabb_tree,
    gwn_bvh_moment_object<Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_refit_bvh_moment<Order, 4, Real, Index>(
        geometry, topology, aabb_tree, moment_tree, stream
    );
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in object overload of gwn_refit_bvh4_moment."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in object overload of gwn_refit_bvh4_moment."
    );
}

template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_aabb_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_object_t<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    GWN_RETURN_ON_ERROR((detail::gwn_build_bvh_topology_aabb_lbvh<Width, Real, Index>(
        geometry.accessor(), topology.accessor(), aabb_tree.accessor(), stream
    )));
    topology.set_stream(stream);
    aabb_tree.set_stream(stream);
    return gwn_status::ok();
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in object overload of gwn_build_bvh_topology_aabb_lbvh."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in object overload of gwn_build_bvh_topology_aabb_lbvh."
    );
}

template <class Real, class Index>
gwn_status gwn_build_bvh4_topology_aabb_lbvh(
    gwn_geometry_object<Real, Index> const &geometry, gwn_bvh_object<Real, Index> &topology,
    gwn_bvh_aabb_object<Real, Index> &aabb_tree, cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_build_bvh_topology_aabb_lbvh<4, Real, Index>(geometry, topology, aabb_tree, stream);
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in object overload of gwn_build_bvh4_topology_aabb_lbvh."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in object overload of gwn_build_bvh4_topology_aabb_lbvh."
    );
}

template <int Order, int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_aabb_moment_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_object_t<Width, Real, Index> &aabb_tree,
    gwn_bvh_moment_tree_object_t<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    GWN_RETURN_ON_ERROR((detail::gwn_build_bvh_topology_aabb_moment_lbvh<Order, Width, Real, Index>(
        geometry.accessor(), topology.accessor(), aabb_tree.accessor(), moment_tree.accessor(),
        stream
    )));
    topology.set_stream(stream);
    aabb_tree.set_stream(stream);
    moment_tree.set_stream(stream);
    return gwn_status::ok();
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in object overload of gwn_build_bvh_topology_aabb_moment_lbvh."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in object overload of gwn_build_bvh_topology_aabb_moment_lbvh."
    );
}

template <int Order, class Real, class Index>
gwn_status gwn_build_bvh4_topology_aabb_moment_lbvh(
    gwn_geometry_object<Real, Index> const &geometry, gwn_bvh_object<Real, Index> &topology,
    gwn_bvh_aabb_object<Real, Index> &aabb_tree, gwn_bvh_moment_object<Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_build_bvh_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
        geometry, topology, aabb_tree, moment_tree, stream
    );
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in object overload of gwn_build_bvh4_topology_aabb_moment_lbvh."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in object overload of "
        "gwn_build_bvh4_topology_aabb_moment_lbvh."
    );
}

} // namespace gwn
