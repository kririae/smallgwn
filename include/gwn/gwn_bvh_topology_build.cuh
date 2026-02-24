#pragma once

/// \file gwn_bvh_topology_build.cuh
/// \brief Public API for BVH topology construction.
///
/// \remark Public entrypoints are object-based; accessor-level build routines
///         remain internal under \c gwn::detail.
/// \remark Morton precision is configurable via the \p MortonCode template
///         parameter:
///         - \c std::uint32_t : 30-bit Morton (10 bits/axis), lower setup cost.
///         - \c std::uint64_t : 63-bit Morton (21 bits/axis), higher precision.
/// \remark LBVH and H-PLOC share the same preprocess stage
///         (scene bounds + Morton generation + radix sort), and only differ
///         in the binary-builder strategy.

#include <cuda_runtime_api.h>

#include <cstdint>

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
/// \tparam MortonCode Morton key type:
///                    - \c std::uint32_t : 30-bit Morton.
///                    - \c std::uint64_t : 63-bit Morton (default).
///
/// \param[in]     geometry  Uploaded triangle mesh (vertices + indices).
/// \param[in,out] topology  Destination topology object; any previous data is
///                          released before the new tree is written.
/// \param[in]     stream    CUDA stream for all asynchronous work.
///
/// \return \c gwn_status::ok() on success, or an error status on failure.
///
/// \remark \p MortonCode only changes topology build (Morton generation/sort
///         and binary radix structure), not query kernel math.
template <int Width, gwn_real_type Real, gwn_index_type Index, class MortonCode = std::uint64_t>
gwn_status gwn_bvh_topology_build_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_bvh_topology_build_lbvh_impl<Width, Real, Index, MortonCode>(
        geometry.accessor(), topology.accessor(), stream
    )));
    topology.set_stream(stream);
    return gwn_status::ok();
}

/// \brief Build a wide BVH topology from triangle geometry using an H-PLOC
///        agglomerative builder.
///
/// Computes Morton order for primitives, then performs a GPU-parallel
/// hierarchical PLOC merge to produce a binary tree that is collapsed into
/// a \p Width -ary topology stored in \p topology.
///
/// On success the bound stream of \p topology is updated to \p stream.
///
/// \tparam Width  BVH node fan-out (e.g. 4 for a 4-wide BVH).
/// \tparam Real   Floating-point type (\c float or \c double).
/// \tparam Index  Integer index type.
/// \tparam MortonCode Morton key type:
///                    - \c std::uint32_t : 30-bit Morton.
///                    - \c std::uint64_t : 63-bit Morton (default).
///
/// \param[in]     geometry  Uploaded triangle mesh (vertices + indices).
/// \param[in,out] topology  Destination topology object.
/// \param[in]     stream    CUDA stream for all asynchronous work.
///
/// \return \c gwn_status::ok() on success, or an error status on failure.
///
/// \remark \p MortonCode only changes topology build (Morton generation/sort
///         and H-PLOC bottom-up hierarchy guidance), not query kernel math.
template <int Width, gwn_real_type Real, gwn_index_type Index, class MortonCode = std::uint64_t>
gwn_status gwn_bvh_topology_build_hploc(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_bvh_topology_build_hploc_impl<Width, Real, Index, MortonCode>(
        geometry.accessor(), topology.accessor(), stream
    )));
    topology.set_stream(stream);
    return gwn_status::ok();
}

} // namespace gwn
