#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "../gwn_geometry.cuh"
#include "gwn_query_vec3_impl.cuh"

namespace gwn {
namespace detail {

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

    Real const numerator = gwn_query_dot(qa, gwn_query_cross(qb - qa, qc - qa));
    if (numerator == Real(0))
        return Real(0);

    Real const denominator =
        Real(1) + gwn_query_dot(qa, qb) + gwn_query_dot(qa, qc) + gwn_query_dot(qb, qc);
    using std::atan2;
    return Real(2) * atan2(numerator, denominator);
}

/// \brief Squared distance from point \p p to line segment [a, b].
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_point_segment_distance_squared_impl(
    gwn_query_vec3<Real> const &p, gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b
) noexcept {
    gwn_query_vec3<Real> const ab = b - a;
    gwn_query_vec3<Real> const ap = p - a;
    Real const ab_dot_ab = gwn_query_dot(ab, ab);
    if (!(ab_dot_ab > Real(0)))
        return gwn_query_squared_norm(ap); // degenerate segment
    Real t = gwn_query_dot(ap, ab) / ab_dot_ab;
    if (t < Real(0)) t = Real(0);
    if (t > Real(1)) t = Real(1);
    return gwn_query_squared_norm(p - (a + t * ab));
}

/// \brief Min squared distance from point \p p to the 3 edges of triangle (a, b, c).
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_point_triangle_edge_distance_squared_impl(
    gwn_query_vec3<Real> const &p, gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b,
    gwn_query_vec3<Real> const &c
) noexcept {
    Real d2_ab = gwn_point_segment_distance_squared_impl(p, a, b);
    Real d2_bc = gwn_point_segment_distance_squared_impl(p, b, c);
    Real d2_ca = gwn_point_segment_distance_squared_impl(p, c, a);
    Real result = d2_ab;
    if (d2_bc < result) result = d2_bc;
    if (d2_ca < result) result = d2_ca;
    return result;
}

template <gwn_real_type Real>
__host__ __device__ inline Real gwn_point_triangle_distance_squared_impl(
    gwn_query_vec3<Real> const &p, gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b,
    gwn_query_vec3<Real> const &c
) noexcept {
    gwn_query_vec3<Real> const ab = b - a;
    gwn_query_vec3<Real> const ac = c - a;
    gwn_query_vec3<Real> const ap = p - a;

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

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline Real gwn_triangle_solid_angle_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_query_vec3<Real> const &query
) noexcept {
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return Real(0);

    auto const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ic, geometry.vertex_count())) {
        return Real(0);
    }

    auto const a_index = static_cast<std::size_t>(ia);
    auto const b_index = static_cast<std::size_t>(ib);
    auto const c_index = static_cast<std::size_t>(ic);

    gwn_query_vec3<Real> const a(
        geometry.vertex_x[a_index], geometry.vertex_y[a_index], geometry.vertex_z[a_index]
    );
    gwn_query_vec3<Real> const b(
        geometry.vertex_x[b_index], geometry.vertex_y[b_index], geometry.vertex_z[b_index]
    );
    gwn_query_vec3<Real> const c(
        geometry.vertex_x[c_index], geometry.vertex_y[c_index], geometry.vertex_z[c_index]
    );
    return gwn_signed_solid_angle_triangle_impl(a, b, c, query);
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline Real gwn_triangle_distance_squared_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_query_vec3<Real> const &query
) noexcept {
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return std::numeric_limits<Real>::infinity();

    auto const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ic, geometry.vertex_count())) {
        return std::numeric_limits<Real>::infinity();
    }

    auto const a_index = static_cast<std::size_t>(ia);
    auto const b_index = static_cast<std::size_t>(ib);
    auto const c_index = static_cast<std::size_t>(ic);

    gwn_query_vec3<Real> const a(
        geometry.vertex_x[a_index], geometry.vertex_y[a_index], geometry.vertex_z[a_index]
    );
    gwn_query_vec3<Real> const b(
        geometry.vertex_x[b_index], geometry.vertex_y[b_index], geometry.vertex_z[b_index]
    );
    gwn_query_vec3<Real> const c(
        geometry.vertex_x[c_index], geometry.vertex_y[c_index], geometry.vertex_z[c_index]
    );
    return gwn_point_triangle_distance_squared_impl(query, a, b, c);
}

/// \brief Min squared distance from query to the 3 edges of a triangle primitive.
template <gwn_real_type Real, gwn_index_type Index>
__device__ inline Real gwn_triangle_edge_distance_squared_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_query_vec3<Real> const &query
) noexcept {
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return std::numeric_limits<Real>::infinity();

    auto const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ic, geometry.vertex_count())) {
        return std::numeric_limits<Real>::infinity();
    }

    auto const a_index = static_cast<std::size_t>(ia);
    auto const b_index = static_cast<std::size_t>(ib);
    auto const c_index = static_cast<std::size_t>(ic);

    gwn_query_vec3<Real> const a(
        geometry.vertex_x[a_index], geometry.vertex_y[a_index], geometry.vertex_z[a_index]
    );
    gwn_query_vec3<Real> const b(
        geometry.vertex_x[b_index], geometry.vertex_y[b_index], geometry.vertex_z[b_index]
    );
    gwn_query_vec3<Real> const c(
        geometry.vertex_x[c_index], geometry.vertex_y[c_index], geometry.vertex_z[c_index]
    );
    return gwn_point_triangle_edge_distance_squared_impl(query, a, b, c);
}

template <gwn_real_type Real>
__host__ __device__ inline Real gwn_aabb_min_distance_squared_impl(
    Real const qx, Real const qy, Real const qz, Real const min_x, Real const min_y,
    Real const min_z, Real const max_x, Real const max_y, Real const max_z
) noexcept {
    auto const clamp_delta = [](Real q, Real lo, Real hi) -> Real {
        if (q < lo)
            return lo - q;
        if (q > hi)
            return q - hi;
        return Real(0);
    };
    Real const dx = clamp_delta(qx, min_x, max_x);
    Real const dy = clamp_delta(qy, min_y, max_y);
    Real const dz = clamp_delta(qz, min_z, max_z);
    Real const result = dx * dx + dy * dy + dz * dz;
    GWN_ASSERT(!(result < Real(0)), "AABB min distance squared is negative");
    return result;
}

} // namespace detail
} // namespace gwn
