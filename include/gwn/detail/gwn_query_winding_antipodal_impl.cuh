#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../gwn_boundary.cuh"
#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_common_impl.cuh"
#include "gwn_query_ray_impl.cuh"
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
        Real const epsilon_1_coefficient = a.y - b.y;
        sign = sign_from_scalar(epsilon_1_coefficient);
        if (sign == gwn_antipodal_projected_edge_sign::k_zero) {
            Real const epsilon_2_coefficient = b.x - a.x;
            sign = sign_from_scalar(epsilon_2_coefficient);
        }
    }

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

    int const i_ab = static_cast<int>(edge_ab.sign);
    int const i_bc = static_cast<int>(edge_bc.sign);
    int const i_ca = static_cast<int>(edge_ca.sign);
    if (!((i_ab >= 0 && i_bc >= 0 && i_ca >= 0) || (i_ab <= 0 && i_bc <= 0 && i_ca <= 0)))
        return gwn_antipodal_axis_result::k_done;
    if (t == Real(0))
        return gwn_antipodal_axis_result::k_singular;
    if (!(t > Real(0)))
        return gwn_antipodal_axis_result::k_done;

    crossing_sign = normal_axis > Real(0) ? Real(1) : Real(-1);
    return gwn_antipodal_axis_result::k_done;
}

template <gwn_real_type Real>
__device__ inline Real gwn_antipodal_signed_spherical_area_impl(
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
        if (!isfinite(area))
            return {std::numeric_limits<Real>::quiet_NaN(), gwn_antipodal_axis_result::k_done};
        if (ray_axis != gwn_antipodal_ray_axis::k_z &&
            edge_class.sign == gwn_antipodal_projected_edge_sign::k_positive && area < Real(0)) {
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
__device__ inline gwn_query_vec3<Real> gwn_antipodal_gradient_nan_impl() noexcept {
    Real const nan = std::numeric_limits<Real>::quiet_NaN();
    return gwn_query_vec3<Real>(nan, nan, nan);
}

template <gwn_real_type Real>
__device__ inline gwn_query_vec3<Real> gwn_antipodal_signed_spherical_area_gradient_impl(
    gwn_query_vec3<Real> const &minus_ray_dir, gwn_query_vec3<Real> const &edge_start,
    gwn_query_vec3<Real> const &edge_end
) noexcept {
    Real const start_norm = gwn_query_norm(edge_start);
    Real const end_norm = gwn_query_norm(edge_end);
    if (!(start_norm > Real(0)) || !(end_norm > Real(0)))
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
    if (!(denom2 > Real(0)) || !isfinite(denom2))
        return gwn_antipodal_gradient_nan_impl<Real>();

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
    gwn_query_vec3<Real> const gradient = c_n * g_n + c_d * g_d;
    if (!isfinite(gradient.x) || !isfinite(gradient.y) || !isfinite(gradient.z))
        return gwn_antipodal_gradient_nan_impl<Real>();
    return gradient;
}

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
        if (!isfinite(edge_gradient.x) || !isfinite(edge_gradient.y) || !isfinite(edge_gradient.z))
            return {edge_gradient, gwn_antipodal_axis_result::k_done};

        gwn_query_vec3<Real> const term =
            static_cast<Real>(boundary_chain.multiplicity[edge_id]) * edge_gradient;
        gwn_query_vec3<Real> const compensated_term = term - gradient_compensation;
        gwn_query_vec3<Real> const next_sum = gradient_sum + compensated_term;
        gradient_compensation = (next_sum - gradient_sum) - compensated_term;
        gradient_sum = next_sum;
    }

    return {gradient_sum / (Real(4) * pi), gwn_antipodal_axis_result::k_done};
}

template <int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline gwn_antipodal_value_result<Real>
gwn_signed_ray_visit_leaf_range_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh, gwn_query_vec3<Real> const &origin,
    gwn_antipodal_ray_axis const ray_axis, Index const begin, Index const count
) noexcept {
    Real crossing_sum = Real(0);
    for (Index offset = 0; offset < count; ++offset) {
        Index const sorted_index = begin + offset;
        if (!gwn_index_in_bounds(sorted_index, bvh.primitive_indices.size()))
            continue;

        Index const primitive_id = bvh.primitive_indices[static_cast<std::size_t>(sorted_index)];
        gwn_query_vec3<Real> a{};
        gwn_query_vec3<Real> b{};
        gwn_query_vec3<Real> c{};
        if (!gwn_ray_load_triangle_vertices_from_primitive_impl(geometry, primitive_id, a, b, c))
            continue;

        Real crossing_sign = Real(0);
        auto const crossing =
            gwn_ray_triangle_crossing_antipodal_impl(origin, ray_axis, a, b, c, crossing_sign);
        if (crossing == gwn_antipodal_axis_result::k_singular)
            return {};
        if (crossing_sign == Real(0))
            continue;
        crossing_sum += crossing_sign;
    }
    return {crossing_sum, gwn_antipodal_axis_result::k_done};
}

template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline gwn_query_vec3<Real> gwn_winding_gradient_point_antipodal_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, gwn_query_vec3<Real> const &query
) noexcept {
    for (int retry_id = 0; retry_id < 3; ++retry_id) {
        gwn_antipodal_ray_axis const ray_axis = gwn_antipodal_ray_axis_for_retry_impl(retry_id);
        auto const gradient =
            gwn_antipodal_boundary_gradient_impl(geometry, boundary_chain, query, ray_axis);
        if (gradient.status == gwn_antipodal_axis_result::k_done)
            return gradient.value;
    }
    return gwn_antipodal_gradient_nan_impl<Real>();
}

template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline gwn_antipodal_value_result<Real>
gwn_signed_ray_crossing_count_bvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, gwn_query_vec3<Real> const &origin,
    gwn_antipodal_ray_axis const ray_axis, OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(Width >= 2, "BVH width requires at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity requires a positive value.");

    if (!geometry.is_valid() || !bvh.is_valid() || !aabb_tree.is_valid_for(bvh))
        return {std::numeric_limits<Real>::quiet_NaN(), gwn_antipodal_axis_result::k_done};

    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        return gwn_signed_ray_visit_leaf_range_impl<Width, Real, Index>(
            geometry, bvh, origin, ray_axis, bvh.root_index, bvh.root_count
        );
    }
    if (bvh.root_kind != gwn_bvh_child_kind::k_internal)
        return {std::numeric_limits<Real>::quiet_NaN(), gwn_antipodal_axis_result::k_done};

    gwn_query_vec3<Real> const direction = gwn_antipodal_ray_direction_impl<Real>(ray_axis);
    auto const ray_dir_precomp =
        gwn_ray_make_dir_precompute_impl(direction.x, direction.y, direction.z);
    Real const t_min = Real(0);
    Real const t_max = std::numeric_limits<Real>::infinity();

    // BVH traversal contributes the integer ray-crossing term.
    Real crossing_sum = Real(0);
    Index stack[StackCapacity];
    int stack_size = 0;
    stack[stack_size++] = bvh.root_index;
    while (stack_size > 0) {
        Index const node_index = stack[--stack_size];
        GWN_ASSERT(stack_size >= 0, "antipodal ray count: stack underflow");
        if (!gwn_index_in_bounds(node_index, bvh.nodes.size()) ||
            !gwn_index_in_bounds(node_index, aabb_tree.nodes.size())) {
            continue;
        }

        auto const &topology_node = bvh.nodes[static_cast<std::size_t>(node_index)];
        auto const &aabb_node = aabb_tree.nodes[static_cast<std::size_t>(node_index)];
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const child_kind =
                static_cast<gwn_bvh_child_kind>(topology_node.child_kind[child_slot]);
            if (child_kind == gwn_bvh_child_kind::k_invalid)
                continue;
            if (child_kind != gwn_bvh_child_kind::k_internal &&
                child_kind != gwn_bvh_child_kind::k_leaf)
                continue;

            auto const interval = gwn_ray_aabb_intersect_interval_impl<Real>(
                origin.x, origin.y, origin.z, ray_dir_precomp, aabb_node.child_min_x[child_slot],
                aabb_node.child_min_y[child_slot], aabb_node.child_min_z[child_slot],
                aabb_node.child_max_x[child_slot], aabb_node.child_max_y[child_slot],
                aabb_node.child_max_z[child_slot], t_min, t_max
            );
            if (!interval.hit)
                continue;

            if (child_kind == gwn_bvh_child_kind::k_leaf) {
                auto const leaf_result = gwn_signed_ray_visit_leaf_range_impl<Width, Real, Index>(
                    geometry, bvh, origin, ray_axis, topology_node.child_index[child_slot],
                    topology_node.child_count[child_slot]
                );
                if (leaf_result.status == gwn_antipodal_axis_result::k_singular)
                    return {};
                crossing_sum += leaf_result.value;
                continue;
            }

            if (stack_size >= StackCapacity) {
                overflow_callback();
                return {std::numeric_limits<Real>::quiet_NaN(), gwn_antipodal_axis_result::k_done};
            }
            stack[stack_size++] = topology_node.child_index[child_slot];
        }
    }
    return {crossing_sum, gwn_antipodal_axis_result::k_done};
}

template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline Real gwn_winding_number_point_bvh_antipodal_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, Real const qx, Real const qy,
    Real const qz, OverflowCallback const &overflow_callback = {}
) noexcept {
    if (!geometry.is_valid() || !bvh.is_valid() || !aabb_tree.is_valid_for(bvh) ||
        !boundary_chain.is_valid())
        return std::numeric_limits<Real>::quiet_NaN();
    if (boundary_chain.mesh_vertex_count != geometry.vertex_count() ||
        boundary_chain.mesh_triangle_count != geometry.triangle_count())
        return std::numeric_limits<Real>::quiet_NaN();

    gwn_query_vec3<Real> const query(qx, qy, qz);
    for (int retry_id = 0; retry_id < 3; ++retry_id) {
        gwn_antipodal_ray_axis const ray_axis = gwn_antipodal_ray_axis_for_retry_impl(retry_id);
        auto const crossings = gwn_signed_ray_crossing_count_bvh_impl<
            Width, Real, Index, StackCapacity, OverflowCallback>(
            geometry, bvh, aabb_tree, query, ray_axis, overflow_callback
        );
        if (crossings.status == gwn_antipodal_axis_result::k_singular)
            continue;
        if (!isfinite(crossings.value))
            return crossings.value;

        auto const boundary =
            gwn_antipodal_boundary_contribution_impl(geometry, boundary_chain, query, ray_axis);
        if (boundary.status == gwn_antipodal_axis_result::k_singular)
            continue;
        return crossings.value + boundary.value;
    }

    return std::numeric_limits<Real>::quiet_NaN();
}

template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
struct gwn_winding_number_batch_bvh_antipodal_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> bvh{};
    gwn_bvh_aabb_accessor<Width, Real, Index> aabb_tree{};
    gwn_boundary_chain_accessor<Index> boundary_chain{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_winding{};
    OverflowCallback overflow_callback{};

    __device__ void operator()(std::size_t const query_id) const {
        out_winding[query_id] = gwn_winding_number_point_bvh_antipodal_impl<
            Width, Real, Index, StackCapacity, OverflowCallback>(
            geometry, bvh, aabb_tree, boundary_chain, query_x[query_id], query_y[query_id],
            query_z[query_id], overflow_callback
        );
    }
};

template <gwn_real_type Real, gwn_index_type Index>
struct gwn_winding_gradient_batch_antipodal_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_boundary_chain_accessor<Index> boundary_chain{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_gradient_x{};
    cuda::std::span<Real> out_gradient_y{};
    cuda::std::span<Real> out_gradient_z{};

    __device__ void operator()(std::size_t const query_id) const {
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
