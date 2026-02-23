#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <exception>

#include "detail/gwn_bvh_build_lbvh.cuh"
#include "detail/gwn_bvh_refit_async.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {

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
///
/// \note Each call replaces the entire moment accessor. If you need multiple
///       Taylor orders to coexist, you must maintain separate moment objects
///       and call this function once per order.
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

} // namespace detail

// ---------------------------------------------------------------------------
// Public object-based build/refit APIs.
// ---------------------------------------------------------------------------
// Exception translation is handled at the detail layer; these thin wrappers
// only forward and update stream binding on success.
// ---------------------------------------------------------------------------

template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_build_bvh_topology_lbvh<Width, Real, Index>(
        geometry.accessor(), topology.accessor(), stream
    )));
    topology.set_stream(stream);
    return gwn_status::ok();
}

template <int Width, class Real, class Index>
gwn_status gwn_refit_bvh_aabb(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> const &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_refit_bvh_aabb<Width, Real, Index>(
        geometry.accessor(), topology.accessor(), aabb_tree.accessor(), stream
    )));
    aabb_tree.set_stream(stream);
    return gwn_status::ok();
}

template <int Order, int Width, class Real, class Index>
gwn_status gwn_refit_bvh_moment(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> const &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_object<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_refit_bvh_moment<Order, Width, Real, Index>(
        geometry.accessor(), topology.accessor(), aabb_tree.accessor(), moment_tree.accessor(),
        stream
    )));
    moment_tree.set_stream(stream);
    return gwn_status::ok();
}

template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_aabb_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_build_bvh_topology_aabb_lbvh<Width, Real, Index>(
        geometry.accessor(), topology.accessor(), aabb_tree.accessor(), stream
    )));
    topology.set_stream(stream);
    aabb_tree.set_stream(stream);
    return gwn_status::ok();
}

template <int Order, int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_aabb_moment_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    gwn_bvh_moment_tree_object<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_build_bvh_topology_aabb_moment_lbvh<Order, Width, Real, Index>(
        geometry.accessor(), topology.accessor(), aabb_tree.accessor(), moment_tree.accessor(),
        stream
    )));
    topology.set_stream(stream);
    aabb_tree.set_stream(stream);
    moment_tree.set_stream(stream);
    return gwn_status::ok();
}

} // namespace gwn
