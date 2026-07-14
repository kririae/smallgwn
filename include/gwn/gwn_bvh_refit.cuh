#pragma once

/// \file gwn_bvh_refit.cuh
/// \brief Public API for canonical BVH and Taylor moment refit.
///
/// \remark Public entrypoints are object-based; accessor-level refit routines
///         are internal under \c gwn::detail.

#include <cuda_runtime_api.h>

#include "detail/gwn_bvh_refit_moment_impl.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"

namespace gwn {

/// \brief Refit all geometry-derived data in a canonical BVH.
///
/// Bounds and leaf-ordered triangle records are replaced from \p geometry while hierarchy
/// references, primitive order, root kind, and maximum depth are preserved. The geometry must
/// retain the triangle indexing used to build \p bvh. On success \p bvh is rebound to \p stream.
/// A failure before mutation begins preserves the previous BVH and stream binding. A later failure
/// makes the BVH unqueryable and binds it to \p stream; callers may then clear or rebuild it.
template <int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_refit_bvh(
    gwn_geometry_object<Real, Index> const &geometry, gwn_bvh_object<Width, Real, Index> &bvh,
    cudaStream_t const stream = gwn_default_stream()
) noexcept;

/// \brief Refit one Taylor moment order from a canonical BVH.
///
/// Triangle records provide the additive leaf moments and child-local bounds provide the
/// far-field radius. The selected \p moment is fully replaced; other moment objects are not
/// modified. On success \p moment is rebound to \p stream.
template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_refit_bvh_moment(
    gwn_bvh_object<Width, Real, Index> const &bvh,
    gwn_bvh_moment_object<Width, Order, Real, Index> &moment,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    gwn_bvh_moment_object<Width, Order, Real, Index> staging;
    staging.set_stream(stream);
    GWN_RETURN_ON_ERROR((detail::gwn_refit_bvh_moment_impl<Order, Width, Real, Index>(
        bvh.accessor(), staging.accessor(), stream
    )));
    swap(moment, staging);
    // staging owns the replaced moment and retains its original stream. Its destructor therefore
    // releases old coefficients only after work already ordered on that stream.
    return gwn_status::ok();
}

} // namespace gwn

#include "detail/gwn_bvh_refit_impl.cuh"
