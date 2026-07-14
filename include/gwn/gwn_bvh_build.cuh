#pragma once

/// \file gwn_bvh_build.cuh
/// \brief Public construction API for complete canonical BVHs.

#include <cuda_runtime_api.h>

#include <cstdint>

#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"

namespace gwn {

/// \brief Binary hierarchy algorithm used by \c gwn_build_bvh.
enum class gwn_bvh_build_method : std::uint8_t {
    k_hploc, ///< Hierarchical parallel locally ordered clustering.
    k_lbvh,  ///< Linear BVH built from sorted Morton codes.
};

/// \brief Options controlling canonical BVH construction.
struct gwn_bvh_build_options {
    /// \brief Binary build method.
    gwn_bvh_build_method method = gwn_bvh_build_method::k_hploc;
    /// \brief H-PLOC nearest-neighbor search radius in the inclusive range [1, 8].
    std::uint32_t hploc_search_radius = 8;
};

/// \brief Build a complete queryable BVH from uploaded triangle geometry.
///
/// The selected binary builder feeds a native child-AoS wide collapse. On success \p bvh owns
/// bounds, packed hierarchy references, primitive order, and leaf-ordered triangle records, and
/// is rebound to \p stream. Previous data is preserved when construction fails.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, gwn_index_type MortonCode = std::uint64_t>
[[nodiscard]] gwn_status gwn_build_bvh(
    gwn_geometry_object<Real, Index> const &geometry, gwn_bvh_object<Width, Real, Index> &bvh,
    gwn_bvh_build_options const options = {}, cudaStream_t const stream = gwn_default_stream()
) noexcept;

} // namespace gwn

#include "detail/gwn_bvh_build_impl.cuh"
