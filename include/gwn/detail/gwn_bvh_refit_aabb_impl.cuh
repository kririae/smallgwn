#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "gwn/detail/gwn_bvh_refit_async.cuh"
#include "gwn/detail/gwn_bvh_status_helpers.cuh"

namespace gwn {
namespace detail {

template <int Width, class Real, class Index>
gwn_status gwn_bvh_refit_aabb_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_try_translate_status("gwn_bvh_refit_aabb_impl", [&]() -> gwn_status {
        if (!geometry.is_valid())
            return gwn_status::invalid_argument("Geometry accessor is invalid for AABB refit.");
        if (!topology.is_valid())
            return gwn_status::invalid_argument("Topology accessor is invalid for AABB refit.");

        auto const release_aabb = [](gwn_bvh_aabb_accessor<Width, Real, Index> &tree,
                                     cudaStream_t const stream_to_release) noexcept {
            gwn_release_bvh_aabb_tree_accessor(tree, stream_to_release);
        };

        auto const build_aabb =
            [&](gwn_bvh_aabb_accessor<Width, Real, Index> &staging_aabb) -> gwn_status {
            if (!topology.has_internal_root())
                return gwn_status::ok();

            std::size_t const node_count = topology.nodes.size();
            GWN_RETURN_ON_ERROR(gwn_allocate_span(staging_aabb.nodes, node_count, stream));
            auto const staging_nodes = cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real>>(
                const_cast<gwn_bvh_aabb_node_soa<Width, Real> *>(staging_aabb.nodes.data()),
                node_count
            );
            GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
                staging_nodes.data(), 0, node_count * sizeof(staging_nodes[0]), stream
            )));

            using traits = gwn_aabb_refit_traits<Width, Real, Index>;
            typename traits::output_context const output_context{staging_nodes};
            return gwn_run_refit_pass<traits, Width, Real, Index>(
                geometry, topology, output_context, stream
            );
        };

        return gwn_replace_accessor_with_staging(aabb_tree, release_aabb, build_aabb, stream);
    });
}

} // namespace detail
} // namespace gwn
