#pragma once

#include "../gwn_geometry.cuh"

#if !__has_include(<Eigen/Core>) || !__has_include(<Eigen/Geometry>)
#error "gwn_query geometry detail requires Eigen/Core and Eigen/Geometry in the include path."
#endif

#include <cmath>
#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gwn {
namespace detail {

template <gwn_real_type Real> using gwn_query_vec3 = Eigen::Matrix<Real, 3, 1>;

template <gwn_real_type Real>
__host__ __device__ inline Real gwn_signed_solid_angle_triangle_impl(
    gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b, gwn_query_vec3<Real> const &c,
    gwn_query_vec3<Real> const &q
) noexcept {
    gwn_query_vec3<Real> qa = a - q;
    gwn_query_vec3<Real> qb = b - q;
    gwn_query_vec3<Real> qc = c - q;

    Real const a_length = qa.norm();
    Real const b_length = qb.norm();
    Real const c_length = qc.norm();
    if (a_length == Real(0) || b_length == Real(0) || c_length == Real(0))
        return Real(0);

    qa /= a_length;
    qb /= b_length;
    qc /= c_length;

    Real const numerator = qa.dot((qb - qa).cross(qc - qa));
    if (numerator == Real(0))
        return Real(0);

    Real const denominator = Real(1) + qa.dot(qb) + qa.dot(qc) + qb.dot(qc);
    return Real(2) * atan2(numerator, denominator);
}

template <gwn_real_type Real>
__host__ __device__ inline Real gwn_point_triangle_distance_squared_impl(
    gwn_query_vec3<Real> const &p, gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b,
    gwn_query_vec3<Real> const &c
) noexcept {
    gwn_query_vec3<Real> const ab = b - a;
    gwn_query_vec3<Real> const ac = c - a;
    gwn_query_vec3<Real> const ap = p - a;

    Real const d1 = ab.dot(ap);
    Real const d2 = ac.dot(ap);
    if (d1 <= Real(0) && d2 <= Real(0))
        return ap.squaredNorm();

    gwn_query_vec3<Real> const bp = p - b;
    Real const d3 = ab.dot(bp);
    Real const d4 = ac.dot(bp);
    if (d3 >= Real(0) && d4 <= d3)
        return bp.squaredNorm();

    Real const vc = d1 * d4 - d3 * d2;
    if (vc <= Real(0) && d1 >= Real(0) && d3 <= Real(0)) {
        Real const v = d1 / (d1 - d3);
        return (p - (a + v * ab)).squaredNorm();
    }

    gwn_query_vec3<Real> const cp = p - c;
    Real const d5 = ab.dot(cp);
    Real const d6 = ac.dot(cp);
    if (d6 >= Real(0) && d5 <= d6)
        return cp.squaredNorm();

    Real const vb = d5 * d2 - d1 * d6;
    if (vb <= Real(0) && d2 >= Real(0) && d6 <= Real(0)) {
        Real const w = d2 / (d2 - d6);
        return (p - (a + w * ac)).squaredNorm();
    }

    Real const va = d3 * d6 - d5 * d4;
    if (va <= Real(0) && (d4 - d3) >= Real(0) && (d5 - d6) >= Real(0)) {
        Real const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + w * (c - b))).squaredNorm();
    }

    Real const denom = Real(1) / (va + vb + vc);
    Real const v = vb * denom;
    Real const w = vc * denom;
    return (p - (a + v * ab + w * ac)).squaredNorm();
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
    return dx * dx + dy * dy + dz * dz;
}

} // namespace detail
} // namespace gwn
