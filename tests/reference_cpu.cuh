#pragma once

#include <gwn/gwn_utils.cuh>

#if !__has_include(<Eigen/Core>) || !__has_include(<Eigen/Geometry>)
#error "reference_cpu.hpp requires Eigen/Core and Eigen/Geometry in the include path."
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "reference_hdk/UT_SolidAngle.h"

namespace gwn::tests {

template <class Real> using reference_vec3 = Eigen::Matrix<Real, 3, 1>;

template <class Real>
inline Real reference_signed_solid_angle_triangle(
    reference_vec3<Real> const &a, reference_vec3<Real> const &b, reference_vec3<Real> const &c,
    reference_vec3<Real> const &q
) noexcept {
    reference_vec3<Real> qa = a - q;
    reference_vec3<Real> qb = b - q;
    reference_vec3<Real> qc = c - q;

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
    return Real(2) * std::atan2(numerator, denominator);
}

template <class Real, class Index = std::int64_t>
inline Real reference_winding_number_point(
    std::span<Real const> vertex_x, std::span<Real const> vertex_y, std::span<Real const> vertex_z,
    std::span<Index const> tri_i0, std::span<Index const> tri_i1, std::span<Index const> tri_i2,
    Real const qx, Real const qy, Real const qz
) noexcept {
    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    reference_vec3<Real> const query(qx, qy, qz);
    Real omega_sum = Real(0);

    for (std::size_t tri = 0; tri < tri_i0.size(); ++tri) {
        Index const ia = tri_i0[tri];
        Index const ib = tri_i1[tri];
        Index const ic = tri_i2[tri];
        if (ia < Index(0) || ib < Index(0) || ic < Index(0))
            continue;

        std::size_t const a_index = static_cast<std::size_t>(ia);
        std::size_t const b_index = static_cast<std::size_t>(ib);
        std::size_t const c_index = static_cast<std::size_t>(ic);
        if (a_index >= vertex_x.size() || b_index >= vertex_x.size() || c_index >= vertex_x.size())
            continue;

        reference_vec3<Real> const a(vertex_x[a_index], vertex_y[a_index], vertex_z[a_index]);
        reference_vec3<Real> const b(vertex_x[b_index], vertex_y[b_index], vertex_z[b_index]);
        reference_vec3<Real> const c(vertex_x[c_index], vertex_y[c_index], vertex_z[c_index]);

        omega_sum += reference_signed_solid_angle_triangle(a, b, c, query);
    }

    return omega_sum / (Real(4) * k_pi);
}

template <class Real, class Index = std::int64_t>
inline gwn_status reference_winding_number_batch(
    std::span<Real const> vertex_x, std::span<Real const> vertex_y, std::span<Real const> vertex_z,
    std::span<Index const> tri_i0, std::span<Index const> tri_i1, std::span<Index const> tri_i2,
    std::span<Real const> query_x, std::span<Real const> query_y, std::span<Real const> query_z,
    std::span<Real> output
) {
    if (vertex_x.size() != vertex_y.size() || vertex_x.size() != vertex_z.size())
        return gwn_status::invalid_argument("Vertex SoA spans must have identical lengths.");
    if (tri_i0.size() != tri_i1.size() || tri_i0.size() != tri_i2.size())
        return gwn_status::invalid_argument("Triangle SoA spans must have identical lengths.");
    if (query_x.size() != query_y.size() || query_x.size() != query_z.size())
        return gwn_status::invalid_argument("Query SoA spans must have identical lengths.");
    if (query_x.size() != output.size())
        return gwn_status::invalid_argument("Output span size must match query count.");

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, output.size()),
        [&](tbb::blocked_range<std::size_t> const &range) {
        for (std::size_t query_id = range.begin(); query_id < range.end(); ++query_id) {
            output[query_id] = reference_winding_number_point<Real, Index>(
                vertex_x, vertex_y, vertex_z, tri_i0, tri_i1, tri_i2, query_x[query_id],
                query_y[query_id], query_z[query_id]
            );
        }
    }
    );

    return gwn_status::ok();
}

template <class Real, class Index = std::int64_t>
inline gwn_status reference_winding_number_batch_hdk_taylor(
    std::span<Real const> vertex_x, std::span<Real const> vertex_y, std::span<Real const> vertex_z,
    std::span<Index const> tri_i0, std::span<Index const> tri_i1, std::span<Index const> tri_i2,
    std::span<Real const> query_x, std::span<Real const> query_y, std::span<Real const> query_z,
    std::span<Real> output, int const order, Real const accuracy_scale
) {
    if (vertex_x.size() != vertex_y.size() || vertex_x.size() != vertex_z.size())
        return gwn_status::invalid_argument("Vertex SoA spans must have identical lengths.");
    if (tri_i0.size() != tri_i1.size() || tri_i0.size() != tri_i2.size())
        return gwn_status::invalid_argument("Triangle SoA spans must have identical lengths.");
    if (query_x.size() != query_y.size() || query_x.size() != query_z.size())
        return gwn_status::invalid_argument("Query SoA spans must have identical lengths.");
    if (query_x.size() != output.size())
        return gwn_status::invalid_argument("Output span size must match query count.");
    if (order < 0 || order > 2)
        return gwn_status::invalid_argument("HDK Taylor order must be in [0, 2].");

    std::size_t const vertex_count = vertex_x.size();
    std::size_t const triangle_count = tri_i0.size();
    if (vertex_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        triangle_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return gwn_status::invalid_argument("Mesh is too large for HDK int32 indexing.");
    }

    std::vector<HDK_Sample::UT_Vector3T<Real>> positions(vertex_count);
    for (std::size_t vertex_id = 0; vertex_id < vertex_count; ++vertex_id) {
        auto &p = positions[vertex_id];
        p[0] = vertex_x[vertex_id];
        p[1] = vertex_y[vertex_id];
        p[2] = vertex_z[vertex_id];
    }

    std::vector<int> triangle_points(3 * triangle_count, 0);
    for (std::size_t triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
        Index const ia = tri_i0[triangle_id];
        Index const ib = tri_i1[triangle_id];
        Index const ic = tri_i2[triangle_id];
        if (ia < Index(0) || ib < Index(0) || ic < Index(0))
            return gwn_status::invalid_argument("Triangle index must be non-negative.");
        std::size_t const a = static_cast<std::size_t>(ia);
        std::size_t const b = static_cast<std::size_t>(ib);
        std::size_t const c = static_cast<std::size_t>(ic);
        if (a >= vertex_count || b >= vertex_count || c >= vertex_count)
            return gwn_status::invalid_argument("Triangle index exceeds vertex count.");
        triangle_points[3 * triangle_id + 0] = static_cast<int>(a);
        triangle_points[3 * triangle_id + 1] = static_cast<int>(b);
        triangle_points[3 * triangle_id + 2] = static_cast<int>(c);
    }

    HDK_Sample::UT_SolidAngle<Real, Real> solid_angle(
        static_cast<int>(triangle_count), triangle_points.data(), static_cast<int>(vertex_count),
        positions.data(), order
    );

    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    Real const inv_4pi = Real(1) / (Real(4) * k_pi);
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, output.size()),
        [&](tbb::blocked_range<std::size_t> const &range) {
        for (std::size_t query_id = range.begin(); query_id < range.end(); ++query_id) {
            HDK_Sample::UT_Vector3T<Real> query{};
            query[0] = query_x[query_id];
            query[1] = query_y[query_id];
            query[2] = query_z[query_id];
            output[query_id] = solid_angle.computeSolidAngle(query, accuracy_scale) * inv_4pi;
        }
    }
    );

    return gwn_status::ok();
}

} // namespace gwn::tests
