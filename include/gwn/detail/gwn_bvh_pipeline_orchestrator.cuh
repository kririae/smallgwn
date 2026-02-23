#pragma once

#include <cuda_runtime_api.h>

#include "gwn/detail/gwn_bvh_pipeline_aabb.cuh"
#include "gwn/detail/gwn_bvh_pipeline_moment.cuh"
#include "gwn/detail/gwn_bvh_pipeline_topology.cuh"

namespace gwn {
namespace detail {

template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_aabb_lbvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR(
        (gwn_build_bvh_topology_lbvh_impl<Width, Real, Index>(geometry, topology, stream))
    );
    return gwn_refit_bvh_aabb_impl<Width, Real, Index>(geometry, topology, aabb_tree, stream);
}

template <int Order, int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_aabb_moment_lbvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree,
    gwn_bvh_moment_accessor<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((gwn_build_bvh_topology_aabb_lbvh_impl<Width, Real, Index>(
        geometry, topology, aabb_tree, stream
    )));
    return gwn_refit_bvh_moment_impl<Order, Width, Real, Index>(
        geometry, topology, aabb_tree, moment_tree, stream
    );
}

} // namespace detail
} // namespace gwn
