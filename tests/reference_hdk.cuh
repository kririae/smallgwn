#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <gwn/gwn_utils.cuh>

#include "reference_hdk/UT_SolidAngle.h"

namespace gwn::tests {

/// Compute Taylor winding numbers with the vendored HDK implementation.
template <class Real, class Index = std::uint32_t>
inline gwn_status reference_winding_number_batch_hdk_taylor(
    cuda::std::span<Real const> const vertex_x, cuda::std::span<Real const> const vertex_y,
    cuda::std::span<Real const> const vertex_z, cuda::std::span<Index const> const tri_i0,
    cuda::std::span<Index const> const tri_i1, cuda::std::span<Index const> const tri_i2,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output, int const order,
    Real const accuracy_scale
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
        positions[vertex_id][0] = vertex_x[vertex_id];
        positions[vertex_id][1] = vertex_y[vertex_id];
        positions[vertex_id][2] = vertex_z[vertex_id];
    }

    std::vector<int> triangle_points(3 * triangle_count, 0);
    for (std::size_t triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
        Index const ia = tri_i0[triangle_id];
        Index const ib = tri_i1[triangle_id];
        Index const ic = tri_i2[triangle_id];
        if (gwn_is_invalid_index(ia) || gwn_is_invalid_index(ib) || gwn_is_invalid_index(ic))
            return gwn_status::invalid_argument("Triangle index must be non-negative.");
        auto const a = static_cast<std::size_t>(ia);
        auto const b = static_cast<std::size_t>(ib);
        auto const c = static_cast<std::size_t>(ic);
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
