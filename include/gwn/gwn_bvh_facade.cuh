#pragma once

#include <cuda_runtime_api.h>

#include "gwn_bvh_refit.cuh"
#include "gwn_bvh_topology_build.cuh"

namespace gwn {

template <int Width, class Real, class Index>
gwn_status gwn_bvh_facade_build_topology_aabb_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR(
        (gwn_bvh_topology_build_lbvh<Width, Real, Index>(geometry, topology, stream))
    );
    return gwn_bvh_refit_aabb<Width, Real, Index>(geometry, topology, aabb_tree, stream);
}

template <int Order, int Width, class Real, class Index>
gwn_status gwn_bvh_facade_build_topology_aabb_moment_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    gwn_bvh_moment_tree_object<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((gwn_bvh_facade_build_topology_aabb_lbvh<Width, Real, Index>(
        geometry, topology, aabb_tree, stream
    )));
    return gwn_bvh_refit_moment<Order, Width, Real, Index>(
        geometry, topology, aabb_tree, moment_tree, stream
    );
}

} // namespace gwn
