#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../gwn_boundary.cuh"
#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_common_impl.cuh"
#include "gwn_query_vec3_impl.cuh"

namespace gwn {
namespace detail {

template <gwn_real_type Real> struct gwn_query_vec2 {
    Real x{};
    Real y{};
};

enum class gwn_antipodal_ray_axis : std::uint8_t {
    k_x = 0,
    k_y = 1,
    k_z = 2,
};

enum class gwn_antipodal_projected_edge_sign : std::int8_t {
    k_negative = -1,
    k_zero = 0,
    k_positive = 1,
};

enum class gwn_antipodal_axis_result : std::uint8_t {
    k_done,
    k_singular,
};

template <gwn_real_type Real> struct gwn_antipodal_value_result {
    Real value{std::numeric_limits<Real>::quiet_NaN()};
    gwn_antipodal_axis_result status{gwn_antipodal_axis_result::k_singular};
};

template <gwn_real_type Real> struct gwn_antipodal_gradient_result {
    gwn_query_vec3<Real> value{};
    gwn_antipodal_axis_result status{gwn_antipodal_axis_result::k_singular};
};

template <gwn_real_type Real> struct gwn_antipodal_projected_edge_classification {
    gwn_antipodal_projected_edge_sign sign{gwn_antipodal_projected_edge_sign::k_zero};
    gwn_antipodal_axis_result status{gwn_antipodal_axis_result::k_done};
};

/// \brief Project a vector onto the coordinate plane orthogonal to the ray axis.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_query_vec2<Real> gwn_antipodal_project_impl(
    gwn_query_vec3<Real> const &value, gwn_antipodal_ray_axis const ray_axis
) noexcept {
    if (ray_axis == gwn_antipodal_ray_axis::k_x)
        return gwn_query_vec2<Real>{value.y, value.z};
    if (ray_axis == gwn_antipodal_ray_axis::k_y)
        return gwn_query_vec2<Real>{value.z, value.x};
    return gwn_query_vec2<Real>{value.x, value.y};
}

/// \brief Classify a projected edge for Antipodal ray crossing and area branch
///        selection.
///
/// The Antipodal method uses the same projected boundary-edge sign in the
/// integer ray-crossing term and in the spherical-area term. Sharing this
/// predicate keeps both branch changes aligned for the current ray axis.
///
/// \param edge_start First endpoint relative to the query point.
/// \param edge_end Second endpoint relative to the query point.
/// \param ray_axis Coordinate ray axis used by the Antipodal query.
/// \return The perturbed projected-edge sign plus the singular retry status
///         for the current axis.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_antipodal_projected_edge_classification<Real>
gwn_antipodal_projected_edge_classify_impl(
    gwn_query_vec3<Real> const &edge_start, gwn_query_vec3<Real> const &edge_end,
    gwn_antipodal_ray_axis const ray_axis
) noexcept {
    gwn_query_vec2<Real> const a = gwn_antipodal_project_impl(edge_start, ray_axis);
    gwn_query_vec2<Real> const b = gwn_antipodal_project_impl(edge_end, ray_axis);
    auto const sign_from_scalar = [](Real const value) {
        if (value > Real(0))
            return gwn_antipodal_projected_edge_sign::k_positive;
        if (value < Real(0))
            return gwn_antipodal_projected_edge_sign::k_negative;
        return gwn_antipodal_projected_edge_sign::k_zero;
    };

    Real const real_part = a.x * b.y - a.y * b.x;
    gwn_antipodal_projected_edge_sign sign = sign_from_scalar(real_part);
    if (sign == gwn_antipodal_projected_edge_sign::k_zero) {
        // Lexicographic epsilon coefficients reproduce the paper's symbolic perturbation without
        // choosing a scale-dependent floating-point epsilon.
        Real const epsilon_1_coefficient = a.y - b.y;
        sign = sign_from_scalar(epsilon_1_coefficient);
        if (sign == gwn_antipodal_projected_edge_sign::k_zero) {
            Real const epsilon_2_coefficient = b.x - a.x;
            sign = sign_from_scalar(epsilon_2_coefficient);
        }
    }

    // A collinear segment whose endpoints face opposite directions contains the projected query.
    // Its crossing assignment depends on the ray axis, so the complete query retries another axis.
    bool const contains_origin = real_part == Real(0) && (a.x * b.x + a.y * b.y <= Real(0));
    gwn_antipodal_axis_result const status =
        contains_origin ? gwn_antipodal_axis_result::k_singular : gwn_antipodal_axis_result::k_done;
    return {sign, status};
}

[[nodiscard]] __device__ inline gwn_antipodal_ray_axis
gwn_antipodal_ray_axis_for_retry_impl(int const retry_id) noexcept {
    if (retry_id == 1)
        return gwn_antipodal_ray_axis::k_x;
    if (retry_id == 2)
        return gwn_antipodal_ray_axis::k_y;
    return gwn_antipodal_ray_axis::k_z;
}

template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_query_vec3<Real>
gwn_antipodal_ray_direction_impl(gwn_antipodal_ray_axis const ray_axis) noexcept {
    if (ray_axis == gwn_antipodal_ray_axis::k_x)
        return gwn_query_vec3<Real>(Real(1), Real(0), Real(0));
    if (ray_axis == gwn_antipodal_ray_axis::k_y)
        return gwn_query_vec3<Real>(Real(0), Real(1), Real(0));
    return gwn_query_vec3<Real>(Real(0), Real(0), Real(1));
}

/// \brief Classify one oriented triangle's positive-axis crossing contribution.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_antipodal_axis_result gwn_ray_triangle_crossing_antipodal_impl(
    gwn_query_vec3<Real> const &origin, gwn_antipodal_ray_axis const ray_axis,
    gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b, gwn_query_vec3<Real> const &c,
    Real &crossing_sign
) noexcept {
    gwn_query_vec3<Real> const pa = a - origin;
    gwn_query_vec3<Real> const pb = b - origin;
    gwn_query_vec3<Real> const pc = c - origin;

    gwn_query_vec3<Real> const normal = gwn_query_cross(pb - pa, pc - pa);
    Real const normal_axis = ray_axis == gwn_antipodal_ray_axis::k_x   ? normal.x
                             : ray_axis == gwn_antipodal_ray_axis::k_y ? normal.y
                                                                       : normal.z;
    if (normal_axis == Real(0))
        return gwn_antipodal_axis_result::k_done;

    // The axis component of the oriented normal determines both the intersection parameter and
    // the signed crossing contribution. Keeping this computation orientation-aware avoids a
    // separate ray-triangle hit followed by a second orientation test.
    Real const numer_t = gwn_query_dot(normal, pa);
    Real const t = numer_t / normal_axis;
    auto const edge_ab = gwn_antipodal_projected_edge_classify_impl(pa, pb, ray_axis);
    auto const edge_bc = gwn_antipodal_projected_edge_classify_impl(pb, pc, ray_axis);
    auto const edge_ca = gwn_antipodal_projected_edge_classify_impl(pc, pa, ray_axis);
    if (edge_ab.status == gwn_antipodal_axis_result::k_singular ||
        edge_bc.status == gwn_antipodal_axis_result::k_singular ||
        edge_ca.status == gwn_antipodal_axis_result::k_singular) {
        return gwn_antipodal_axis_result::k_singular;
    }
    if (edge_ab.sign == gwn_antipodal_projected_edge_sign::k_zero ||
        edge_bc.sign == gwn_antipodal_projected_edge_sign::k_zero ||
        edge_ca.sign == gwn_antipodal_projected_edge_sign::k_zero) {
        return gwn_antipodal_axis_result::k_singular;
    }

    // Matching perturbed edge signs mean the projected query lies inside the oriented triangle.
    // Mixed signs reject it without evaluating a crossing contribution.
    int const i_ab = static_cast<int>(edge_ab.sign);
    int const i_bc = static_cast<int>(edge_bc.sign);
    int const i_ca = static_cast<int>(edge_ca.sign);
    if (!((i_ab >= 0 && i_bc >= 0 && i_ca >= 0) || (i_ab <= 0 && i_bc <= 0 && i_ca <= 0)))
        return gwn_antipodal_axis_result::k_done;
    // A triangle through the query cannot be assigned consistently to either ray half-line.
    if (t == Real(0))
        return gwn_antipodal_axis_result::k_singular;
    if (!(t > Real(0)))
        return gwn_antipodal_axis_result::k_done;

    crossing_sign = normal_axis > Real(0) ? Real(1) : Real(-1);
    return gwn_antipodal_axis_result::k_done;
}

/// \brief Compute signed spherical area for one oriented boundary edge and ray axis.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline Real gwn_antipodal_signed_spherical_area_impl(
    gwn_query_vec3<Real> const &minus_ray_dir, gwn_query_vec3<Real> const &edge_start,
    gwn_query_vec3<Real> const &edge_end
) noexcept {
    using Calc = Real;

    Calc const start_norm = gwn_query_norm(edge_start);
    Calc const end_norm = gwn_query_norm(edge_end);
    if (!(start_norm > Calc(0)) || !(end_norm > Calc(0)))
        return std::numeric_limits<Calc>::quiet_NaN();

    // Antipodal mesh term from Martens et al. 2026: signed spherical area of
    // triangle (-ray, edge_start, edge_end), evaluated with atan2.
    Calc const numerator = gwn_query_dot(minus_ray_dir, gwn_query_cross(edge_start, edge_end));
    Calc const denominator =
        start_norm * end_norm + gwn_query_dot(minus_ray_dir, edge_start) * end_norm +
        gwn_query_dot(edge_start, edge_end) + gwn_query_dot(edge_end, minus_ray_dir) * start_norm;

    return Calc(2) * atan2(numerator, denominator);
}

/// \brief Sum the boundary spherical-area contribution for one ray axis.
template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline gwn_antipodal_value_result<Real>
gwn_antipodal_boundary_contribution_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, gwn_query_vec3<Real> const &query,
    gwn_antipodal_ray_axis const ray_axis
) noexcept {
    Real const pi = Real(3.141592653589793238462643383279502884);
    gwn_query_vec3<Real> const ray_dir = gwn_antipodal_ray_direction_impl<Real>(ray_axis);
    gwn_query_vec3<Real> const minus_ray_dir = -ray_dir;

    Real area_sum = Real(0);
    Real area_compensation = Real(0);
    for (std::size_t edge_id = 0; edge_id < boundary_chain.edge_count(); ++edge_id) {
        // Boundary rows come from gwn_build_boundary_chain.
        Index const start_index = boundary_chain.start_vertex[edge_id];
        Index const end_index = boundary_chain.end_vertex[edge_id];
        auto const start = static_cast<std::size_t>(start_index);
        auto const end = static_cast<std::size_t>(end_index);
        gwn_query_vec3<Real> const edge_start(
            geometry.vertex_x[start] - query.x, geometry.vertex_y[start] - query.y,
            geometry.vertex_z[start] - query.z
        );
        gwn_query_vec3<Real> const edge_end(
            geometry.vertex_x[end] - query.x, geometry.vertex_y[end] - query.y,
            geometry.vertex_z[end] - query.z
        );
        auto const edge_class =
            gwn_antipodal_projected_edge_classify_impl(edge_start, edge_end, ray_axis);
        if (edge_class.status == gwn_antipodal_axis_result::k_singular)
            return {};
        Real area = gwn_antipodal_signed_spherical_area_impl(minus_ray_dir, edge_start, edge_end);
        if (ray_axis != gwn_antipodal_ray_axis::k_z &&
            edge_class.sign == gwn_antipodal_projected_edge_sign::k_positive && area < Real(0)) {
            // atan2 returns the principal spherical area. For the retried X and Y projections,
            // the perturbed edge sign selects the equivalent positive branch used by the crossing
            // convention. Adding 2*pi changes the value but not its gradient.
            area += Real(2) * pi;
        }
        // Kahan summation keeps boundary-heavy meshes stable in Real precision.
        Real const term = static_cast<Real>(boundary_chain.multiplicity[edge_id]) * area;
        Real const compensated_term = term - area_compensation;
        Real const next_sum = area_sum + compensated_term;
        area_compensation = (next_sum - area_sum) - compensated_term;
        area_sum = next_sum;
    }

    return {area_sum / (Real(4) * pi), gwn_antipodal_axis_result::k_done};
}

template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_query_vec3<Real> gwn_antipodal_gradient_nan_impl() noexcept {
    Real const nan = std::numeric_limits<Real>::quiet_NaN();
    return gwn_query_vec3<Real>(nan, nan, nan);
}

/// \brief Differentiate one signed spherical-area term with respect to the query position.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_query_vec3<Real>
gwn_antipodal_signed_spherical_area_gradient_impl(
    gwn_query_vec3<Real> const &minus_ray_dir, gwn_query_vec3<Real> const &edge_start,
    gwn_query_vec3<Real> const &edge_end
) noexcept {
    Real const start_norm = gwn_query_norm(edge_start);
    Real const end_norm = gwn_query_norm(edge_end);
    if (start_norm == Real(0) || end_norm == Real(0))
        return gwn_antipodal_gradient_nan_impl<Real>();

    Real const inv_start_norm = Real(1) / start_norm;
    Real const inv_end_norm = Real(1) / end_norm;
    gwn_query_vec3<Real> const start_unit = edge_start * inv_start_norm;
    gwn_query_vec3<Real> const end_unit = edge_end * inv_end_norm;

    Real const numerator = gwn_query_dot(minus_ray_dir, gwn_query_cross(edge_start, edge_end));
    Real const denominator =
        start_norm * end_norm + gwn_query_dot(minus_ray_dir, edge_start) * end_norm +
        gwn_query_dot(minus_ray_dir, edge_end) * start_norm + gwn_query_dot(edge_start, edge_end);
    Real const denom2 = numerator * numerator + denominator * denominator;
    if (denom2 == Real(0))
        return gwn_antipodal_gradient_nan_impl<Real>();

    // Differentiate 2*atan2(numerator, denominator) with respect to the query position. The query
    // shifts both edge vectors by the same amount, which produces the paired numerator and
    // denominator gradients below.
    Real const c_n = Real(2) * denominator / denom2;
    Real const c_d = Real(-2) * numerator / denom2;
    gwn_query_vec3<Real> const g_n =
        -(gwn_query_cross(edge_end, minus_ray_dir) + gwn_query_cross(minus_ray_dir, edge_start));
    gwn_query_vec3<Real> const t1 =
        end_norm * (minus_ray_dir + end_unit +
                    (Real(1) + gwn_query_dot(minus_ray_dir, end_unit)) * start_unit);
    gwn_query_vec3<Real> const t2 =
        start_norm * (minus_ray_dir + start_unit +
                      (Real(1) + gwn_query_dot(minus_ray_dir, start_unit)) * end_unit);
    gwn_query_vec3<Real> const g_d = -(t1 + t2);
    return c_n * g_n + c_d * g_d;
}

/// \brief Sum the boundary spherical-area gradient for one ray axis.
template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline gwn_antipodal_gradient_result<Real>
gwn_antipodal_boundary_gradient_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, gwn_query_vec3<Real> const &query,
    gwn_antipodal_ray_axis const ray_axis
) noexcept {
    Real const pi = Real(3.141592653589793238462643383279502884);
    gwn_query_vec3<Real> const ray_dir = gwn_antipodal_ray_direction_impl<Real>(ray_axis);
    gwn_query_vec3<Real> const minus_ray_dir = -ray_dir;

    gwn_query_vec3<Real> gradient_sum(Real(0), Real(0), Real(0));
    gwn_query_vec3<Real> gradient_compensation(Real(0), Real(0), Real(0));
    for (std::size_t edge_id = 0; edge_id < boundary_chain.edge_count(); ++edge_id) {
        Index const start_index = boundary_chain.start_vertex[edge_id];
        Index const end_index = boundary_chain.end_vertex[edge_id];
        auto const start = static_cast<std::size_t>(start_index);
        auto const end = static_cast<std::size_t>(end_index);
        gwn_query_vec3<Real> const edge_start(
            geometry.vertex_x[start] - query.x, geometry.vertex_y[start] - query.y,
            geometry.vertex_z[start] - query.z
        );
        gwn_query_vec3<Real> const edge_end(
            geometry.vertex_x[end] - query.x, geometry.vertex_y[end] - query.y,
            geometry.vertex_z[end] - query.z
        );
        auto const edge_class =
            gwn_antipodal_projected_edge_classify_impl(edge_start, edge_end, ray_axis);
        if (edge_class.status == gwn_antipodal_axis_result::k_singular)
            return {};

        gwn_query_vec3<Real> const edge_gradient =
            gwn_antipodal_signed_spherical_area_gradient_impl(minus_ray_dir, edge_start, edge_end);

        gwn_query_vec3<Real> const term =
            static_cast<Real>(boundary_chain.multiplicity[edge_id]) * edge_gradient;
        // Vector Kahan summation limits cancellation error on long oriented boundary chains.
        gwn_query_vec3<Real> const compensated_term = term - gradient_compensation;
        gwn_query_vec3<Real> const next_sum = gradient_sum + compensated_term;
        gradient_compensation = (next_sum - gradient_sum) - compensated_term;
        gradient_sum = next_sum;
    }

    return {gradient_sum / (Real(4) * pi), gwn_antipodal_axis_result::k_done};
}

/// \brief Evaluate Antipodal boundary gradient with coordinate-axis retries.
template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline gwn_query_vec3<Real> gwn_winding_gradient_point_antipodal_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, gwn_query_vec3<Real> const &query
) noexcept {
    // Retry the full boundary derivative because its branch choice and singularity test share one
    // projected coordinate plane.
    for (int retry_id = 0; retry_id < 3; ++retry_id) {
        gwn_antipodal_ray_axis const ray_axis = gwn_antipodal_ray_axis_for_retry_impl(retry_id);
        auto const gradient =
            gwn_antipodal_boundary_gradient_impl(geometry, boundary_chain, query, ray_axis);
        if (gradient.status == gwn_antipodal_axis_result::k_done)
            return gradient.value;
    }
    return gwn_antipodal_gradient_nan_impl<Real>();
}

/// \brief Sum signed positive-axis triangle crossings through the canonical BVH.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback, bool ValidateAccessor = true>
[[nodiscard]] __device__ inline gwn_antipodal_value_result<Real> gwn_signed_ray_crossing_count_impl(
    gwn_bvh_accessor<Width, Real, Index> const &bvh, gwn_query_vec3<Real> const &origin,
    gwn_antipodal_ray_axis const ray_axis, OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(Width >= 2, "BVH width requires at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity requires a positive value.");

    if constexpr (ValidateAccessor)
        if (!bvh.is_valid())
            return {std::numeric_limits<Real>::quiet_NaN(), gwn_antipodal_axis_result::k_done};

    Real crossing_sum = Real(0);
    auto const intersects_positive_axis_ray = [&](gwn_aabb<Real> const &bounds) noexcept {
        // Antipodal crossings use a positive coordinate ray. Its two zero-direction slabs reduce
        // to containment tests, while the ray-axis slab only needs to extend beyond the origin.
        if (ray_axis == gwn_antipodal_ray_axis::k_x)
            return origin.y >= bounds.min_y && origin.y <= bounds.max_y &&
                   origin.z >= bounds.min_z && origin.z <= bounds.max_z && bounds.max_x >= origin.x;
        if (ray_axis == gwn_antipodal_ray_axis::k_y)
            return origin.z >= bounds.min_z && origin.z <= bounds.max_z &&
                   origin.x >= bounds.min_x && origin.x <= bounds.max_x && bounds.max_y >= origin.y;
        return origin.x >= bounds.min_x && origin.x <= bounds.max_x && origin.y >= bounds.min_y &&
               origin.y <= bounds.max_y && bounds.max_z >= origin.z;
    };
    auto visit_leaf = [&](gwn_bvh_child<Real> const &leaf) noexcept {
        for (std::uint32_t primitive_offset = 0; primitive_offset < leaf.primitive_count();
             ++primitive_offset) {
            std::uint64_t const sorted_index = leaf.offset() + primitive_offset;
            if (sorted_index >= bvh.triangles.size())
                continue;

            auto const &triangle = bvh.triangles[static_cast<std::size_t>(sorted_index)];
            gwn_query_vec3<Real> const a(triangle.v0_x, triangle.v0_y, triangle.v0_z);
            gwn_query_vec3<Real> const b(
                triangle.v0_x + triangle.e1_x, triangle.v0_y + triangle.e1_y,
                triangle.v0_z + triangle.e1_z
            );
            gwn_query_vec3<Real> const c(
                triangle.v0_x + triangle.e2_x, triangle.v0_y + triangle.e2_y,
                triangle.v0_z + triangle.e2_z
            );

            Real crossing_sign = Real(0);
            auto const crossing =
                gwn_ray_triangle_crossing_antipodal_impl(origin, ray_axis, a, b, c, crossing_sign);
            // One projected edge through the query invalidates this axis. The caller must retry the
            // entire crossing and boundary calculation on the next coordinate axis.
            if (crossing == gwn_antipodal_axis_result::k_singular)
                return false;
            crossing_sum += crossing_sign;
        }
        return true;
    };

    if (bvh.has_leaf_root()) {
        if (!visit_leaf(bvh.root))
            return {};
        return {crossing_sum, gwn_antipodal_axis_result::k_done};
    }

    // Only internal node offsets survive across iterations, so the topology's Index type is the
    // complete pending state and keeps the default uint32_t stack compact.
    Index stack[StackCapacity];
    int stack_size = 0;
    auto node_index = static_cast<Index>(bvh.root.offset());
    while (true) {
        if (gwn_index_in_bounds(node_index, bvh.nodes.size())) {
            auto const &node = bvh.nodes[static_cast<std::size_t>(node_index)];
            GWN_DETAIL_PRAGMA_UNROLL
            for (int child_slot = 0; child_slot < Width; ++child_slot) {
                auto const &child = node.child(child_slot);
                if (!child.is_valid())
                    continue;
                if (!intersects_positive_axis_ray(child.bounds))
                    continue;

                // Leaf work stays in the current node so only internal node offsets consume the
                // topology-bounded stack shared by every canonical traversal.
                if (child.is_leaf()) {
                    if (!visit_leaf(child))
                        return {};
                    continue;
                }
                if (!child.is_internal())
                    continue;
                if (stack_size >= StackCapacity) {
                    overflow_callback();
                    return {
                        std::numeric_limits<Real>::quiet_NaN(), gwn_antipodal_axis_result::k_done
                    };
                }
                stack[stack_size++] = static_cast<Index>(child.offset());
            }
        }

        if (stack_size == 0)
            break;
        node_index = stack[--stack_size];
    }
    return {crossing_sum, gwn_antipodal_axis_result::k_done};
}

/// \brief Combine Antipodal crossing and boundary terms with shared axis retries.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline Real gwn_winding_number_antipodal_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, Real const qx, Real const qy,
    Real const qz, OverflowCallback const &overflow_callback = {}
) noexcept {
    gwn_query_vec3<Real> const query(qx, qy, qz);
    // Crossing and boundary terms must use the same axis. If either term is singular, discard the
    // partial result and restart both terms on the next coordinate axis.
    for (int retry_id = 0; retry_id < 3; ++retry_id) {
        gwn_antipodal_ray_axis const ray_axis = gwn_antipodal_ray_axis_for_retry_impl(retry_id);
        auto const crossings = gwn_signed_ray_crossing_count_impl<
            Width, Real, Index, StackCapacity, OverflowCallback, false>(
            bvh, query, ray_axis, overflow_callback
        );
        if (crossings.status == gwn_antipodal_axis_result::k_singular)
            continue;

        auto const boundary =
            gwn_antipodal_boundary_contribution_impl(geometry, boundary_chain, query, ray_axis);
        if (boundary.status == gwn_antipodal_axis_result::k_singular)
            continue;
        return crossings.value + boundary.value;
    }

    return std::numeric_limits<Real>::quiet_NaN();
}

/// \brief Invoke canonical Antipodal winding traversal for one batch element.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
struct gwn_winding_number_antipodal_batch_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    gwn_boundary_chain_accessor<Index> boundary_chain{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_winding{};
    OverflowCallback overflow_callback{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        // Host validation covers all shared objects once. The point implementation keeps only the
        // three axis retries and their numerical singularity checks on the per-query path.
        out_winding[query_id] =
            gwn_winding_number_antipodal_impl<Width, Real, Index, StackCapacity, OverflowCallback>(
                geometry, bvh, boundary_chain, query_x[query_id], query_y[query_id],
                query_z[query_id], overflow_callback
            );
    }
};

/// \brief Invoke Antipodal boundary-gradient evaluation for one batch element.
template <gwn_real_type Real, gwn_index_type Index>
struct gwn_winding_gradient_antipodal_batch_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_boundary_chain_accessor<Index> boundary_chain{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_gradient_x{};
    cuda::std::span<Real> out_gradient_y{};
    cuda::std::span<Real> out_gradient_z{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        gwn_query_vec3<Real> const query(query_x[query_id], query_y[query_id], query_z[query_id]);
        gwn_query_vec3<Real> const gradient =
            gwn_winding_gradient_point_antipodal_impl(geometry, boundary_chain, query);
        out_gradient_x[query_id] = gradient.x;
        out_gradient_y[query_id] = gradient.y;
        out_gradient_z[query_id] = gradient.z;
    }
};

} // namespace detail
} // namespace gwn
