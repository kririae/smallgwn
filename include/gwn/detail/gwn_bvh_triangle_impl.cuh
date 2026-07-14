#pragma once

#include <algorithm>
#include <cstddef>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"

namespace gwn {
namespace detail {

/// \brief Geometry-derived BVH data computed from one original triangle.
template <gwn_real_type Real> struct gwn_bvh_triangle_payload {
    gwn_bvh_triangle<Real> triangle{};
    gwn_aabb<Real> bounds{};
    bool is_valid = false;
};

/// \brief Compute the canonical query record and bounds for one triangle primitive.
///
/// Build and refit share this choke point so bounds and the leaf-ordered record cannot observe
/// different vertex loads or drift to different index-validation policies.
template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline gwn_bvh_triangle_payload<Real> gwn_compute_bvh_triangle(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_index
) noexcept {
    gwn_bvh_triangle_payload<Real> result{};
    if (!gwn_index_in_bounds(primitive_index, geometry.triangle_count()))
        return result;

    auto const triangle_index = static_cast<std::size_t>(primitive_index);
    Index const i0 = geometry.tri_i0[triangle_index];
    Index const i1 = geometry.tri_i1[triangle_index];
    Index const i2 = geometry.tri_i2[triangle_index];
    if (!gwn_index_in_bounds(i0, geometry.vertex_count()) ||
        !gwn_index_in_bounds(i1, geometry.vertex_count()) ||
        !gwn_index_in_bounds(i2, geometry.vertex_count())) {
        return result;
    }

    auto const v0 = static_cast<std::size_t>(i0);
    auto const v1 = static_cast<std::size_t>(i1);
    auto const v2 = static_cast<std::size_t>(i2);
    Real const v0_x = geometry.vertex_x[v0];
    Real const v0_y = geometry.vertex_y[v0];
    Real const v0_z = geometry.vertex_z[v0];
    Real const v1_x = geometry.vertex_x[v1];
    Real const v1_y = geometry.vertex_y[v1];
    Real const v1_z = geometry.vertex_z[v1];
    Real const v2_x = geometry.vertex_x[v2];
    Real const v2_y = geometry.vertex_y[v2];
    Real const v2_z = geometry.vertex_z[v2];

    // Derive the query record and its build/refit bounds from the same vertex loads. This keeps the
    // two stored representations consistent without re-reading geometry between computations.
    result.triangle = gwn_bvh_triangle<Real>{
        v0_x,        v0_y,        v0_z,        v1_x - v0_x, v1_y - v0_y,
        v1_z - v0_z, v2_x - v0_x, v2_y - v0_y, v2_z - v0_z,
    };
    result.bounds = gwn_aabb<Real>{
        std::min(v0_x, std::min(v1_x, v2_x)), std::min(v0_y, std::min(v1_y, v2_y)),
        std::min(v0_z, std::min(v1_z, v2_z)), std::max(v0_x, std::max(v1_x, v2_x)),
        std::max(v0_y, std::max(v1_y, v2_y)), std::max(v0_z, std::max(v1_z, v2_z)),
    };
    result.is_valid = true;
    return result;
}

} // namespace detail
} // namespace gwn
