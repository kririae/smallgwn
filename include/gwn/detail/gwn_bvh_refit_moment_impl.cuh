#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "gwn_bvh_refit_async.cuh"
#include "gwn_bvh_status_helpers.cuh"

namespace gwn {
namespace detail {

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_bvh_refit_moment_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_accessor<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_try_translate_status("gwn_bvh_refit_moment_impl", [&]() -> gwn_status {
        static_assert(
            Order == 0 || Order == 1 || Order == 2,
            "gwn_bvh_refit_moment currently supports Order 0, 1, and 2."
        );

        if (!geometry.is_valid())
            return gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_refit_moment, "Geometry accessor is invalid for moment refit."
            );
        if (!topology.is_valid())
            return gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_refit_moment, "Topology accessor is invalid for moment refit."
            );
        if (!aabb_tree.is_valid_for(topology))
            return gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_refit_moment, "AABB tree accessor is invalid for moment refit."
            );

        auto const release_moment = [](gwn_bvh_moment_accessor<Width, Real, Index> &tree,
                                       cudaStream_t const stream_to_release) noexcept {
            gwn_release_bvh_moment_tree_accessor(tree, stream_to_release);
        };

        auto const build_moment =
            [&](gwn_bvh_moment_accessor<Width, Real, Index> &staging_moment) -> gwn_status {
            if (!topology.has_internal_root())
                return gwn_status::ok();

            using node_type = gwn_bvh_taylor_node_soa<Width, Order, Real>;
            cuda::std::span<node_type const> staging_nodes{};
            GWN_RETURN_ON_ERROR(gwn_allocate_span(staging_nodes, topology.nodes.size(), stream));
            auto cleanup_staging_nodes =
                gwn_make_scope_exit([&]() noexcept { gwn_free_span(staging_nodes, stream); });
            auto const staging_nodes_mutable = cuda::std::span<node_type>(
                const_cast<node_type *>(staging_nodes.data()), staging_nodes.size()
            );
            GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
                staging_nodes_mutable.data(), 0, staging_nodes_mutable.size() * sizeof(node_type),
                stream
            )));

            using traits = gwn_moment_refit_traits<Order, Width, Real, Index>;
            typename traits::output_context const output_context{
                staging_nodes_mutable, cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real> const>(
                                           aabb_tree.nodes.data(), aabb_tree.nodes.size()
                                       )
            };
            GWN_RETURN_ON_ERROR((gwn_run_refit_pass<traits, Width, Real, Index>(
                geometry, topology, output_context, stream
            )));

            if constexpr (Order == 0)
                staging_moment.taylor_order0_nodes = staging_nodes;
            else if constexpr (Order == 1)
                staging_moment.taylor_order1_nodes = staging_nodes;
            else
                staging_moment.taylor_order2_nodes = staging_nodes;
            cleanup_staging_nodes.release();
            return gwn_status::ok();
        };

        return gwn_replace_accessor_with_staging(moment_tree, release_moment, build_moment, stream);
    });
}

} // namespace detail
} // namespace gwn
