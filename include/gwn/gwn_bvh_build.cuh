#pragma once

#include <cuda_runtime_api.h>

#include "detail/gwn_bvh_pipeline_orchestrator.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"

namespace gwn {

template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_build_bvh_topology_lbvh_impl<Width, Real, Index>(
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
    GWN_RETURN_ON_ERROR((detail::gwn_refit_bvh_aabb_impl<Width, Real, Index>(
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
    GWN_RETURN_ON_ERROR((detail::gwn_refit_bvh_moment_impl<Order, Width, Real, Index>(
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
    GWN_RETURN_ON_ERROR((detail::gwn_build_bvh_topology_aabb_lbvh_impl<Width, Real, Index>(
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
    GWN_RETURN_ON_ERROR(
        (detail::gwn_build_bvh_topology_aabb_moment_lbvh_impl<Order, Width, Real, Index>(
            geometry.accessor(), topology.accessor(), aabb_tree.accessor(), moment_tree.accessor(),
            stream
        ))
    );
    topology.set_stream(stream);
    aabb_tree.set_stream(stream);
    moment_tree.set_stream(stream);
    return gwn_status::ok();
}

} // namespace gwn
