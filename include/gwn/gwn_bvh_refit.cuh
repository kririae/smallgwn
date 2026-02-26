#pragma once

/// \file gwn_bvh_refit.cuh
/// \brief Public API for BVH payload refit (AABB bounds and Taylor moments).
///
/// \remark Public entrypoints are object-based; accessor-level refit routines
///         are internal under \c gwn::detail.

#include <cuda_runtime_api.h>

#include "detail/gwn_bvh_refit_aabb_impl.cuh"
#include "detail/gwn_bvh_refit_moment_impl.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"

namespace gwn {

/// \brief Refit axis-aligned bounding boxes for every node of an existing BVH
///        topology.
///
/// Leaf AABBs are computed from triangle bounds; internal-node AABBs are
/// propagated bottom-up through an asynchronous GPU refit pass.
///
/// On success the bound stream of \p aabb_tree is updated to \p stream.
///
/// \tparam Width  BVH node fan-out.
/// \tparam Real   Floating-point type.
/// \tparam Index  Signed integer index type.
///
/// \param[in]     geometry   Uploaded triangle mesh.
/// \param[in]     topology   Previously built BVH topology.
/// \param[in,out] aabb_tree  Destination AABB tree object; previous data is
///                           released before writing.
/// \param[in]     stream     CUDA stream for all asynchronous work.
///
/// \return \c gwn_status::ok() on success.
template <int Width, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_bvh_refit_aabb(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> const &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_bvh_refit_aabb_impl<Width, Real, Index>(
        geometry.accessor(), topology.accessor(), aabb_tree.accessor(), stream
    )));
    aabb_tree.set_stream(stream);
    return gwn_status::ok();
}

/// \brief Refit Taylor multipole moments for every node of an existing BVH.
///
/// Computes order-\p Order Taylor expansion data used by the approximate
/// winding-number query path.  The AABB tree is required to derive the
/// \c max_p_dist2 far-field radius for each node.
///
/// Each call performs a **full replace** of the moment data — only the
/// requested \p Order slot is populated.  To maintain multiple orders
/// simultaneously, use separate \c gwn_bvh_moment_tree_object instances.
///
/// On success the bound stream of \p moment_tree is updated to \p stream.
///
/// \tparam Order  Taylor expansion order (0, 1, or 2).
/// \tparam Width  BVH node fan-out.
/// \tparam Real   Floating-point type.
/// \tparam Index  Signed integer index type.
///
/// \param[in]     geometry     Uploaded triangle mesh.
/// \param[in]     topology     Previously built BVH topology.
/// \param[in]     aabb_tree    Previously refit AABB tree (used for
///                             \c max_p_dist2 computation).
/// \param[in,out] moment_tree  Destination moment tree object; previous data
///                             is released before writing.
/// \param[in]     stream       CUDA stream for all asynchronous work.
///
/// \return \c gwn_status::ok() on success.
template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_bvh_refit_moment(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> const &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_object<Width, Order, Real, Index> &moment_tree,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    GWN_RETURN_ON_ERROR((detail::gwn_bvh_refit_moment_impl<Order, Width, Real, Index>(
        geometry.accessor(), topology.accessor(), aabb_tree.accessor(), moment_tree.accessor(),
        stream
    )));
    moment_tree.set_stream(stream);
    return gwn_status::ok();
}

} // namespace gwn
