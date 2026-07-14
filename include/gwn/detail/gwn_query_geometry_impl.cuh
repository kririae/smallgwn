#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../gwn_geometry.cuh"
#include "gwn_query_vec3_impl.cuh"

namespace gwn {
namespace detail {

/// \brief Evaluate the oriented solid angle of one triangle at a query point.
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_signed_solid_angle_triangle_impl(
    gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b, gwn_query_vec3<Real> const &c,
    gwn_query_vec3<Real> const &q
) noexcept {
    gwn_query_vec3<Real> qa = a - q;
    gwn_query_vec3<Real> qb = b - q;
    gwn_query_vec3<Real> qc = c - q;

    Real const a_length = gwn_query_norm(qa);
    Real const b_length = gwn_query_norm(qb);
    Real const c_length = gwn_query_norm(qc);
    if (a_length == Real(0) || b_length == Real(0) || c_length == Real(0))
        return Real(0);

    qa /= a_length;
    qb /= b_length;
    qc /= c_length;

    // Unit directions keep the atan2 arguments within a comparable scale. This kernel defines a
    // query on a triangle vertex and a zero oriented numerator to contribute zero.
    Real const numerator = gwn_query_dot(qa, gwn_query_cross(qb - qa, qc - qa));
    if (numerator == Real(0))
        return Real(0);

    Real const denominator =
        Real(1) + gwn_query_dot(qa, qb) + gwn_query_dot(qa, qc) + gwn_query_dot(qb, qc);
    using std::atan2;
    return Real(2) * atan2(numerator, denominator);
}

/// \brief Compute squared distance from a point to the closed triangle.
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_point_triangle_distance_squared_impl(
    gwn_query_vec3<Real> const &p, gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b,
    gwn_query_vec3<Real> const &c
) noexcept {
    gwn_query_vec3<Real> const ab = b - a;
    gwn_query_vec3<Real> const ac = c - a;
    gwn_query_vec3<Real> const ap = p - a;

    // The sign tests partition the triangle's Voronoi regions. Each early return projects onto
    // the corresponding vertex or edge; the remaining region projects onto the triangle face.
    Real const d1 = gwn_query_dot(ab, ap);
    Real const d2 = gwn_query_dot(ac, ap);
    if (d1 <= Real(0) && d2 <= Real(0))
        return gwn_query_squared_norm(ap);

    gwn_query_vec3<Real> const bp = p - b;
    Real const d3 = gwn_query_dot(ab, bp);
    Real const d4 = gwn_query_dot(ac, bp);
    if (d3 >= Real(0) && d4 <= d3)
        return gwn_query_squared_norm(bp);

    Real const vc = d1 * d4 - d3 * d2;
    if (vc <= Real(0) && d1 >= Real(0) && d3 <= Real(0)) {
        Real const v = d1 / (d1 - d3);
        return gwn_query_squared_norm(p - (a + v * ab));
    }

    gwn_query_vec3<Real> const cp = p - c;
    Real const d5 = gwn_query_dot(ab, cp);
    Real const d6 = gwn_query_dot(ac, cp);
    if (d6 >= Real(0) && d5 <= d6)
        return gwn_query_squared_norm(cp);

    Real const vb = d5 * d2 - d1 * d6;
    if (vb <= Real(0) && d2 >= Real(0) && d6 <= Real(0)) {
        Real const w = d2 / (d2 - d6);
        return gwn_query_squared_norm(p - (a + w * ac));
    }

    Real const va = d3 * d6 - d5 * d4;
    if (va <= Real(0) && (d4 - d3) >= Real(0) && (d5 - d6) >= Real(0)) {
        Real const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return gwn_query_squared_norm(p - (b + w * (c - b)));
    }

    Real const denom = Real(1) / (va + vb + vc);
    Real const v = vb * denom;
    Real const w = vc * denom;
    Real const result = gwn_query_squared_norm(p - (a + v * ab + w * ac));
    GWN_ASSERT(!(result < Real(0)), "point-triangle distance squared is negative");
    return result;
}

/// \brief Compute squared distance from a point to a closed axis-aligned box.
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_aabb_min_distance_squared_impl(
    Real const qx, Real const qy, Real const qz, Real const min_x, Real const min_y,
    Real const min_z, Real const max_x, Real const max_y, Real const max_z
) noexcept {
    Real const dx = std::max(min_x - qx, Real(0)) + std::max(qx - max_x, Real(0));
    Real const dy = std::max(min_y - qy, Real(0)) + std::max(qy - max_y, Real(0));
    Real const dz = std::max(min_z - qz, Real(0)) + std::max(qz - max_z, Real(0));
    Real const result = dx * dx + dy * dy + dz * dz;
    GWN_ASSERT(!(result < Real(0)), "AABB min distance squared is negative");
    return result;
}

} // namespace detail
} // namespace gwn
