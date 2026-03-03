#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_vec3_impl.cuh"

namespace gwn {
namespace detail {

template <gwn_real_type Real, gwn_index_type Index> struct gwn_ray_first_hit_result {
    Real t{Real(-1)};
    Index primitive_id{gwn_invalid_index<Index>()};

    __host__ __device__ constexpr bool hit() const noexcept { return t >= Real(0); }
};

template <gwn_real_type Real> struct gwn_ray_aabb_interval {
    bool hit{false};
    Real t_near{Real(0)};
    Real t_far{Real(0)};
};

template <gwn_real_type Real>
__device__ inline gwn_ray_aabb_interval<Real> gwn_ray_aabb_intersect_interval_impl(
    Real const ray_ox, Real const ray_oy, Real const ray_oz, Real const ray_dx, Real const ray_dy,
    Real const ray_dz, Real const min_x, Real const min_y, Real const min_z, Real const max_x,
    Real const max_y, Real const max_z, Real const t_min, Real const t_max
) noexcept {
    gwn_ray_aabb_interval<Real> result{};
    if (!(t_max >= t_min))
        return result;

    Real t_near = t_min;
    Real t_far = t_max;

    auto update_axis = [&](Real const origin, Real const direction, Real const lo,
                           Real const hi) -> bool {
        if (direction == Real(0))
            return origin >= lo && origin <= hi;

        Real const inv_dir = Real(1) / direction;
        Real t0 = (lo - origin) * inv_dir;
        Real t1 = (hi - origin) * inv_dir;
        if (t0 > t1) {
            Real const tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        t_near = std::max(t_near, t0);
        t_far = std::min(t_far, t1);
        return t_near <= t_far;
    };

    if (!update_axis(ray_ox, ray_dx, min_x, max_x))
        return result;
    if (!update_axis(ray_oy, ray_dy, min_y, max_y))
        return result;
    if (!update_axis(ray_oz, ray_dz, min_z, max_z))
        return result;

    if (!(t_far >= t_near))
        return result;

    result.hit = true;
    result.t_near = t_near;
    result.t_far = t_far;
    return result;
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ inline gwn_query_vec3<Real> gwn_stable_triangle_normal_impl(
    gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b, gwn_query_vec3<Real> const &c
) noexcept {
    // Embree reference: common/math/vec3.h::stable_triangle_normal
    using std::abs;

    Real const ab_x = a.z * b.y;
    Real const ab_y = a.x * b.z;
    Real const ab_z = a.y * b.x;
    Real const bc_x = b.z * c.y;
    Real const bc_y = b.x * c.z;
    Real const bc_z = b.y * c.x;

    gwn_query_vec3<Real> const cross_ab(a.y * b.z - ab_x, a.z * b.x - ab_y, a.x * b.y - ab_z);
    gwn_query_vec3<Real> const cross_bc(b.y * c.z - bc_x, b.z * c.x - bc_y, b.x * c.y - bc_z);

    gwn_query_vec3<Real> normal = cross_bc;
    if (abs(ab_x) < abs(bc_x))
        normal.x = cross_ab.x;
    if (abs(ab_y) < abs(bc_y))
        normal.y = cross_ab.y;
    if (abs(ab_z) < abs(bc_z))
        normal.z = cross_ab.z;
    return normal;
}

template <gwn_real_type Real>
__device__ inline bool gwn_ray_triangle_intersect_robust_impl(
    gwn_query_vec3<Real> const &origin, gwn_query_vec3<Real> const &direction,
    gwn_query_vec3<Real> const &v0, gwn_query_vec3<Real> const &v1, gwn_query_vec3<Real> const &v2,
    Real const t_min, Real const t_max, Real &t_out
) noexcept {
    // Embree reference:
    // kernels/geometry/triangle_intersector_pluecker.h (Apache-2.0)
    gwn_query_vec3<Real> const p0 = v0 - origin;
    gwn_query_vec3<Real> const p1 = v1 - origin;
    gwn_query_vec3<Real> const p2 = v2 - origin;

    gwn_query_vec3<Real> const e0 = p2 - p0;
    gwn_query_vec3<Real> const e1 = p0 - p1;
    gwn_query_vec3<Real> const e2 = p1 - p2;

    Real const u = gwn_query_dot(gwn_query_cross(e0, p2 + p0), direction);
    Real const v = gwn_query_dot(gwn_query_cross(e1, p0 + p1), direction);
    Real const w = gwn_query_dot(gwn_query_cross(e2, p1 + p2), direction);
    Real const uvw = u + v + w;
    using std::abs;
    Real const edge_eps = std::numeric_limits<Real>::epsilon() * abs(uvw);
    Real const min_edge = std::min(u, std::min(v, w));
    Real const max_edge = std::max(u, std::max(v, w));

    if (!(min_edge >= -edge_eps || max_edge <= edge_eps))
        return false;

    gwn_query_vec3<Real> const ng = gwn_stable_triangle_normal_impl(e0, e1, e2);
    Real const den = Real(2) * gwn_query_dot(ng, direction);
    if (den == Real(0))
        return false;

    Real const t_num = Real(2) * gwn_query_dot(p0, ng);
    Real const t = t_num / den;
    if (t < t_min || t > t_max)
        return false;

    t_out = t;
    return true;
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_ray_triangle_intersect_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_query_vec3<Real> const &origin, gwn_query_vec3<Real> const &direction, Real const t_min,
    Real const t_max, Real &t_out
) noexcept {
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return false;

    auto const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ic, geometry.vertex_count())) {
        return false;
    }

    auto const a_idx = static_cast<std::size_t>(ia);
    auto const b_idx = static_cast<std::size_t>(ib);
    auto const c_idx = static_cast<std::size_t>(ic);
    gwn_query_vec3<Real> const a(
        geometry.vertex_x[a_idx], geometry.vertex_y[a_idx], geometry.vertex_z[a_idx]
    );
    gwn_query_vec3<Real> const b(
        geometry.vertex_x[b_idx], geometry.vertex_y[b_idx], geometry.vertex_z[b_idx]
    );
    gwn_query_vec3<Real> const c(
        geometry.vertex_x[c_idx], geometry.vertex_y[c_idx], geometry.vertex_z[c_idx]
    );

    return gwn_ray_triangle_intersect_robust_impl(origin, direction, a, b, c, t_min, t_max, t_out);
}

template <gwn_real_type Real>
__device__ inline void gwn_swap_children_if_greater_entry_t_impl(
    Real &lhs_t_near, Real &rhs_t_near, int &lhs_slot, int &rhs_slot, std::uint8_t &lhs_kind,
    std::uint8_t &rhs_kind
) noexcept {
    if (!(lhs_t_near > rhs_t_near))
        return;

    using std::swap;
    swap(lhs_t_near, rhs_t_near);
    swap(lhs_slot, rhs_slot);
    swap(lhs_kind, rhs_kind);
}

template <int Width, gwn_real_type Real>
__device__ inline void gwn_sort_children_by_entry_t_impl(
    Real (&child_entry_t)[Width], int (&child_slot_order)[Width], std::uint8_t (&child_kind)[Width]
) noexcept {
    if constexpr (Width == 2) {
        gwn_swap_children_if_greater_entry_t_impl(
            child_entry_t[0], child_entry_t[1], child_slot_order[0], child_slot_order[1],
            child_kind[0], child_kind[1]
        );
    } else if constexpr (Width == 4) {
        gwn_swap_children_if_greater_entry_t_impl(
            child_entry_t[0], child_entry_t[1], child_slot_order[0], child_slot_order[1],
            child_kind[0], child_kind[1]
        );
        gwn_swap_children_if_greater_entry_t_impl(
            child_entry_t[2], child_entry_t[3], child_slot_order[2], child_slot_order[3],
            child_kind[2], child_kind[3]
        );
        gwn_swap_children_if_greater_entry_t_impl(
            child_entry_t[0], child_entry_t[2], child_slot_order[0], child_slot_order[2],
            child_kind[0], child_kind[2]
        );
        gwn_swap_children_if_greater_entry_t_impl(
            child_entry_t[1], child_entry_t[3], child_slot_order[1], child_slot_order[3],
            child_kind[1], child_kind[3]
        );
        gwn_swap_children_if_greater_entry_t_impl(
            child_entry_t[1], child_entry_t[2], child_slot_order[1], child_slot_order[2],
            child_kind[1], child_kind[2]
        );
    } else {
        GWN_PRAGMA_UNROLL
        for (int pass = 0; pass < Width; ++pass) {
            int const start = pass & 1;
            GWN_PRAGMA_UNROLL
            for (int i = start; (i + 1) < Width; i += 2) {
                gwn_swap_children_if_greater_entry_t_impl(
                    child_entry_t[i], child_entry_t[i + 1], child_slot_order[i],
                    child_slot_order[i + 1], child_kind[i], child_kind[i + 1]
                );
            }
        }
    }
}

template <int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline gwn_ray_first_hit_result<Real, Index> gwn_ray_first_hit_bvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, Real const ray_ox,
    Real const ray_oy, Real const ray_oz, Real const ray_dx, Real const ray_dy, Real const ray_dz,
    Real const t_min, Real const t_max
) noexcept {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    gwn_ray_first_hit_result<Real, Index> result{};
    if (!geometry.is_valid() || !bvh.is_valid() || !aabb_tree.is_valid_for(bvh))
        return result;
    if (!(t_max >= t_min))
        return result;

    Real const dir_len2 = ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz;
    if (!(dir_len2 > Real(0)))
        return result;

    gwn_query_vec3<Real> const origin(ray_ox, ray_oy, ray_oz);
    gwn_query_vec3<Real> const direction(ray_dx, ray_dy, ray_dz);

    Real best_t = t_max;
    Index best_pi = gwn_invalid_index<Index>();
    bool found = false;

    auto test_leaf = [&](Index const begin, Index const count) {
        for (Index off = 0; off < count; ++off) {
            Index const si = begin + off;
            if (!gwn_index_in_bounds(si, bvh.primitive_indices.size()))
                continue;
            Index const pi = bvh.primitive_indices[static_cast<std::size_t>(si)];
            Real t_hit = Real(0);
            if (!gwn_ray_triangle_intersect_from_primitive_impl<Real, Index>(
                    geometry, pi, origin, direction, t_min, best_t, t_hit
                )) {
                continue;
            }

            if (!found || t_hit < best_t) {
                best_t = t_hit;
                best_pi = pi;
                found = true;
            }
        }
    };

    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        test_leaf(bvh.root_index, bvh.root_count);
        if (found) {
            result.t = best_t;
            result.primitive_id = best_pi;
        }
        return result;
    }

    if (bvh.root_kind != gwn_bvh_child_kind::k_internal)
        return result;

    Index stack[StackCapacity];
    int stack_size = 0;
    stack[stack_size++] = bvh.root_index;

    while (stack_size > 0) {
        Index const node_index = stack[--stack_size];
        GWN_ASSERT(stack_size >= 0, "ray query: stack underflow");
        if (!gwn_index_in_bounds(node_index, bvh.nodes.size()))
            continue;

        auto const &topo_node = bvh.nodes[static_cast<std::size_t>(node_index)];
        GWN_ASSERT(
            gwn_index_in_bounds(node_index, aabb_tree.nodes.size()),
            "ray query: node_index out of bounds for aabb_tree"
        );
        auto const &aabb_node = aabb_tree.nodes[static_cast<std::size_t>(node_index)];

        int child_slot_order[Width];
        Real child_entry_t[Width];
        std::uint8_t child_kind[Width];
        std::uint8_t constexpr k_invalid_kind =
            static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
        Real constexpr k_infinite_t = std::numeric_limits<Real>::infinity();
        GWN_PRAGMA_UNROLL
        for (int i = 0; i < Width; ++i) {
            child_slot_order[i] = 0;
            child_entry_t[i] = k_infinite_t;
            child_kind[i] = k_invalid_kind;
        }
        int child_count = 0;

        GWN_PRAGMA_UNROLL
        for (int s = 0; s < Width; ++s) {
            auto const kind = static_cast<gwn_bvh_child_kind>(topo_node.child_kind[s]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf)
                continue;

            auto const interval = gwn_ray_aabb_intersect_interval_impl<Real>(
                ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, aabb_node.child_min_x[s],
                aabb_node.child_min_y[s], aabb_node.child_min_z[s], aabb_node.child_max_x[s],
                aabb_node.child_max_y[s], aabb_node.child_max_z[s], t_min, best_t
            );
            if (!interval.hit)
                continue;

            if (child_count >= Width)
                continue;
            child_slot_order[child_count] = s;
            child_entry_t[child_count] = interval.t_near;
            child_kind[child_count] = topo_node.child_kind[s];
            ++child_count;
        }

        if (child_count > 1)
            gwn_sort_children_by_entry_t_impl<Width>(child_entry_t, child_slot_order, child_kind);

        GWN_PRAGMA_UNROLL
        for (int i = 0; i < Width; ++i) {
            if (static_cast<gwn_bvh_child_kind>(child_kind[i]) != gwn_bvh_child_kind::k_leaf)
                continue;
            if (child_entry_t[i] > best_t)
                continue;

            int const s = child_slot_order[i];
            test_leaf(topo_node.child_index[s], topo_node.child_count[s]);
        }

        GWN_PRAGMA_UNROLL
        for (int i = Width - 1; i >= 0; --i) {
            if (static_cast<gwn_bvh_child_kind>(child_kind[i]) != gwn_bvh_child_kind::k_internal)
                continue;
            if (child_entry_t[i] > best_t)
                continue;

            if (stack_size >= StackCapacity)
                gwn_trap();
            stack[stack_size++] = topo_node.child_index[child_slot_order[i]];
        }
    }

    if (found) {
        result.t = best_t;
        result.primitive_id = best_pi;
    }
    return result;
}

template <int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
struct gwn_ray_first_hit_batch_bvh_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> bvh{};
    gwn_bvh_aabb_accessor<Width, Real, Index> aabb_tree{};
    cuda::std::span<Real const> ray_origin_x{};
    cuda::std::span<Real const> ray_origin_y{};
    cuda::std::span<Real const> ray_origin_z{};
    cuda::std::span<Real const> ray_dir_x{};
    cuda::std::span<Real const> ray_dir_y{};
    cuda::std::span<Real const> ray_dir_z{};
    cuda::std::span<Real> output_t{};
    cuda::std::span<Index> output_primitive_id{};
    Real t_min{};
    Real t_max{};

    __device__ void operator()(std::size_t const ray_id) const {
        auto const hit = gwn_ray_first_hit_bvh_impl<Width, Real, Index, StackCapacity>(
            geometry, bvh, aabb_tree, ray_origin_x[ray_id], ray_origin_y[ray_id],
            ray_origin_z[ray_id], ray_dir_x[ray_id], ray_dir_y[ray_id], ray_dir_z[ray_id], t_min,
            t_max
        );
        output_t[ray_id] = hit.t;
        output_primitive_id[ray_id] = hit.primitive_id;
    }
};

template <int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
[[nodiscard]] inline gwn_ray_first_hit_batch_bvh_functor<Width, Real, Index, StackCapacity>
gwn_make_ray_first_hit_batch_bvh_functor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    cuda::std::span<Real const> const ray_origin_x, cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z, cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y, cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t, cuda::std::span<Index> const output_primitive_id,
    Real const t_min, Real const t_max
) {
    return gwn_ray_first_hit_batch_bvh_functor<Width, Real, Index, StackCapacity>{
        geometry,  bvh,       aabb_tree, ray_origin_x,        ray_origin_y, ray_origin_z, ray_dir_x,
        ray_dir_y, ray_dir_z, output_t,  output_primitive_id, t_min,        t_max
    };
}

} // namespace detail
} // namespace gwn
