#pragma once

/// \file gwn_bvh_facade.cuh
/// \brief Convenience one-call BVH construction workflows.
///
/// These functions compose the lower-level topology-build and payload-refit
/// steps into single calls for common use-cases.
///
/// \remark Public entrypoints are object-based and forward into detail-layer
///         accessor composition internally.
/// \remark Morton precision is configurable by \p MortonCode and is forwarded
///         to topology builders. Payload refit/query math is unchanged.

#include <cuda_runtime_api.h>

#include <cstdint>

#include "gwn_bvh_refit.cuh"
#include "gwn_bvh_topology_build.cuh"

namespace gwn {

/// \brief Build BVH topology via LBVH and immediately refit AABB bounds.
///
/// Equivalent to calling \c gwn_bvh_topology_build_lbvh followed by
/// \c gwn_bvh_refit_aabb on the same stream.
///
/// \tparam Width  BVH node fan-out.
/// \tparam Real   Floating-point type.
/// \tparam Index  Signed integer index type.
/// \tparam MortonCode Morton key type:
///                    - \c std::uint32_t : 30-bit Morton.
///                    - \c std::uint64_t : 63-bit Morton (default).
///
/// \param[in]     geometry   Uploaded triangle mesh.
/// \param[in,out] topology   Destination topology object.
/// \param[in,out] aabb_tree  Destination AABB tree object.
/// \param[in]     stream     CUDA stream for all asynchronous work.
///
/// \return \c gwn_status::ok() on success.
template <int Width, gwn_real_type Real, gwn_index_type Index, class MortonCode = std::uint64_t>
gwn_status gwn_bvh_facade_build_topology_aabb_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR(
        (gwn_bvh_topology_build_lbvh<Width, Real, Index, MortonCode>(geometry, topology, stream))
    );
    return gwn_bvh_refit_aabb<Width, Real, Index>(geometry, topology, aabb_tree, stream);
}

/// \brief Build BVH topology via H-PLOC and immediately refit AABB bounds.
///
/// Equivalent to calling \c gwn_bvh_topology_build_hploc followed by
/// \c gwn_bvh_refit_aabb on the same stream.
///
/// \tparam Width  BVH node fan-out.
/// \tparam Real   Floating-point type.
/// \tparam Index  Integer index type.
/// \tparam MortonCode Morton key type:
///                    - \c std::uint32_t : 30-bit Morton.
///                    - \c std::uint64_t : 63-bit Morton (default).
///
/// \param[in]     geometry   Uploaded triangle mesh.
/// \param[in,out] topology   Destination topology object.
/// \param[in,out] aabb_tree  Destination AABB tree object.
/// \param[in]     stream     CUDA stream for all asynchronous work.
///
/// \return \c gwn_status::ok() on success.
template <int Width, gwn_real_type Real, gwn_index_type Index, class MortonCode = std::uint64_t>
gwn_status gwn_bvh_facade_build_topology_aabb_hploc(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR(
        (gwn_bvh_topology_build_hploc<Width, Real, Index, MortonCode>(geometry, topology, stream))
    );
    return gwn_bvh_refit_aabb<Width, Real, Index>(geometry, topology, aabb_tree, stream);
}

/// \brief Build BVH topology via LBVH, refit AABB bounds, and compute Taylor
///        moments — all in a single call.
///
/// Equivalent to calling \c gwn_bvh_facade_build_topology_aabb_lbvh followed
/// by \c gwn_bvh_refit_moment on the same stream.
///
/// \tparam Order  Taylor expansion order (0, 1, or 2).
/// \tparam Width  BVH node fan-out.
/// \tparam Real   Floating-point type.
/// \tparam Index  Signed integer index type.
/// \tparam MortonCode Morton key type:
///                    - \c std::uint32_t : 30-bit Morton.
///                    - \c std::uint64_t : 63-bit Morton (default).
///
/// \param[in]     geometry     Uploaded triangle mesh.
/// \param[in,out] topology     Destination topology object.
/// \param[in,out] aabb_tree    Destination AABB tree object.
/// \param[in,out] moment_tree  Destination moment tree object.
/// \param[in]     stream       CUDA stream for all asynchronous work.
///
/// \return \c gwn_status::ok() on success.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    class MortonCode = std::uint64_t>
gwn_status gwn_bvh_facade_build_topology_aabb_moment_lbvh(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    gwn_bvh_moment_tree_object<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((gwn_bvh_facade_build_topology_aabb_lbvh<Width, Real, Index, MortonCode>(
        geometry, topology, aabb_tree, stream
    )));
    return gwn_bvh_refit_moment<Order, Width, Real, Index>(
        geometry, topology, aabb_tree, moment_tree, stream
    );
}

/// \brief Build BVH topology via H-PLOC, refit AABB bounds, and compute Taylor
///        moments — all in a single call.
///
/// Equivalent to calling \c gwn_bvh_facade_build_topology_aabb_hploc followed
/// by \c gwn_bvh_refit_moment on the same stream.
///
/// \tparam Order  Taylor expansion order (0, 1, or 2).
/// \tparam Width  BVH node fan-out.
/// \tparam Real   Floating-point type.
/// \tparam Index  Integer index type.
/// \tparam MortonCode Morton key type:
///                    - \c std::uint32_t : 30-bit Morton.
///                    - \c std::uint64_t : 63-bit Morton (default).
///
/// \param[in]     geometry     Uploaded triangle mesh.
/// \param[in,out] topology     Destination topology object.
/// \param[in,out] aabb_tree    Destination AABB tree object.
/// \param[in,out] moment_tree  Destination moment tree object.
/// \param[in]     stream       CUDA stream for all asynchronous work.
///
/// \return \c gwn_status::ok() on success.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    class MortonCode = std::uint64_t>
gwn_status gwn_bvh_facade_build_topology_aabb_moment_hploc(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    gwn_bvh_moment_tree_object<Width, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((gwn_bvh_facade_build_topology_aabb_hploc<Width, Real, Index, MortonCode>(
        geometry, topology, aabb_tree, stream
    )));
    return gwn_bvh_refit_moment<Order, Width, Real, Index>(
        geometry, topology, aabb_tree, moment_tree, stream
    );
}

} // namespace gwn
