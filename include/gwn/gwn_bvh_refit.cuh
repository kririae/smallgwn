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
/// references, primitive order, root kind, \c internal_stack_bound, and \c packed_stack_bound are
/// preserved. The geometry must retain the triangle indexing used to build \p bvh. On success
/// \p bvh is rebound to \p stream.
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
/// modified. The operation clears \p moment on \p stream before validation or allocation. On
/// failure, \p moment is empty and remains bound to \p stream.
/// \remark The caller orders every prior use of \p moment before \p stream.
template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_refit_bvh_moment(
    gwn_bvh_object<Width, Real, Index> const &bvh,
    gwn_bvh_moment_object<Width, Order, Real, Index> &moment,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    moment.set_stream(stream);
    moment.clear();
    return detail::gwn_try_translate_status("gwn_refit_bvh_moment", [&]() {
        gwn_bvh_moment_object<Width, Order, Real, Index> candidate;
        candidate.set_stream(stream);
        detail::gwn_refit_bvh_moment_impl<Order, Width, Real, Index>(
            bvh.accessor(), candidate.accessor(), stream
        );
        swap(moment, candidate);
    });
}

} // namespace gwn

#include "detail/gwn_bvh_refit_impl.cuh"
