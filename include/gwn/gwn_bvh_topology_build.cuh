#pragma once

#include <cuda_runtime_api.h>

#include "detail/gwn_bvh_topology_build_impl.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"

namespace gwn {

template <int Width, class Real, class Index>
gwn_status gwn_bvh_topology_build_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_bvh_topology_build_lbvh_impl<Width, Real, Index>(
        geometry.accessor(), topology.accessor(), stream
    )));
    topology.set_stream(stream);
    return gwn_status::ok();
}

} // namespace gwn
