#pragma once

/// \file gwn_bvh_topology_build.cuh
/// \brief Public API for BVH topology construction.

#include <cuda_runtime_api.h>

#include "detail/gwn_bvh_topology_build_impl.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"

namespace gwn {

/// \brief Build a wide BVH topology from triangle geometry using an LBVH
///        (Linear BVH) algorithm.
///
/// Computes Morton codes for triangle centroids, radix-sorts them, builds a
/// binary radix tree, and then collapses it into a \p Width -ary tree stored
/// in \p topology.
///
/// On success the bound stream of \p topology is updated to \p stream.
///
/// \tparam Width  BVH node fan-out (e.g. 4 for a 4-wide BVH).
/// \tparam Real   Floating-point type (\c float or \c double).
/// \tparam Index  Signed integer index type.
///
/// \param[in]     geometry  Uploaded triangle mesh (vertices + indices).
/// \param[in,out] topology  Destination topology object; any previous data is
///                          released before the new tree is written.
/// \param[in]     stream    CUDA stream for all asynchronous work.
///
/// \return \c gwn_status::ok() on success, or an error status on failure.
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
