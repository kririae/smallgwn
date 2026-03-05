#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_common_impl.cuh"
#include "gwn_query_vec3_impl.cuh"

namespace gwn {
namespace detail {

template <gwn_real_type Real, gwn_index_type Index> struct gwn_ray_first_hit_result {
    Real t{Real(-1)};
    Index primitive_id{gwn_invalid_index<Index>()};
    Real u{Real(0)};
    Real v{Real(0)};
    gwn_ray_first_hit_status status{gwn_ray_first_hit_status::k_miss};

    __host__ __device__ constexpr bool hit() const noexcept {
        return status == gwn_ray_first_hit_status::k_hit;
    }
};

template <gwn_real_type Real> struct gwn_ray_aabb_interval {
    bool hit{false};
    Real t_near{Real(0)};
    Real t_far{Real(0)};
};

template <gwn_real_type Real, gwn_index_type Index> struct gwn_ray_best_hit {
    Real t{Real(-1)};
    Index primitive_id{gwn_invalid_index<Index>()};
    Real u{Real(0)};
    Real v{Real(0)};
    bool found{false};
};

template <gwn_real_type Real> struct gwn_ray_dir_precompute {
    Real inv[3]{Real(0), Real(0), Real(0)};
    std::int8_t sign[3]{0, 0, 0}; // -1: negative, 0: zero, +1: positive
};

template <gwn_real_type Real>
[[nodiscard]] __device__ inline std::int8_t gwn_ray_direction_sign_impl(Real const dir) noexcept {
    if (dir > Real(0))
        return 1;
    if (dir < Real(0))
        return -1;
    return 0;
}

template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_ray_dir_precompute<Real>
gwn_ray_make_dir_precompute_impl(Real const dir_x, Real const dir_y, Real const dir_z) noexcept {
    gwn_ray_dir_precompute<Real> dir{};

    dir.sign[0] = gwn_ray_direction_sign_impl(dir_x);
    dir.sign[1] = gwn_ray_direction_sign_impl(dir_y);
    dir.sign[2] = gwn_ray_direction_sign_impl(dir_z);

    if (dir.sign[0] != 0)
        dir.inv[0] = Real(1) / dir_x;
    if (dir.sign[1] != 0)
        dir.inv[1] = Real(1) / dir_y;
    if (dir.sign[2] != 0)
        dir.inv[2] = Real(1) / dir_z;

    return dir;
}

template <gwn_real_type Real>
__device__ inline bool gwn_ray_aabb_update_axis_interval_impl(
    Real const origin, Real const lo, Real const hi, Real const inv_dir, std::int8_t const dir_sign,
    Real &t_near, Real &t_far
) noexcept {
    if (dir_sign == 0)
        return origin >= lo && origin <= hi;

    Real const near_plane = (dir_sign < 0) ? hi : lo;
    Real const far_plane = (dir_sign < 0) ? lo : hi;
    Real const t0 = (near_plane - origin) * inv_dir;
    Real const t1 = (far_plane - origin) * inv_dir;

    t_near = std::max(t_near, t0);
    t_far = std::min(t_far, t1);
    return t_near <= t_far;
}

template <gwn_real_type Real>
__device__ inline gwn_ray_aabb_interval<Real> gwn_ray_aabb_intersect_interval_impl(
    Real const ray_ox, Real const ray_oy, Real const ray_oz,
    gwn_ray_dir_precompute<Real> const &ray_dir, Real const min_x, Real const min_y,
    Real const min_z, Real const max_x, Real const max_y, Real const max_z, Real const t_min,
    Real const t_max
) noexcept {
    gwn_ray_aabb_interval<Real> result{};
    if (!(t_max >= t_min))
        return result;

    Real t_near = t_min;
    Real t_far = t_max;

    if (!gwn_ray_aabb_update_axis_interval_impl(
            ray_ox, min_x, max_x, ray_dir.inv[0], ray_dir.sign[0], t_near, t_far
        )) {
        return result;
    }
    if (!gwn_ray_aabb_update_axis_interval_impl(
            ray_oy, min_y, max_y, ray_dir.inv[1], ray_dir.sign[1], t_near, t_far
        )) {
        return result;
    }
    if (!gwn_ray_aabb_update_axis_interval_impl(
            ray_oz, min_z, max_z, ray_dir.inv[2], ray_dir.sign[2], t_near, t_far
        )) {
        return result;
    }

    if (t_far < t_near)
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

    Real const ab_mul_x = a.z * b.y;
    Real const ab_mul_y = a.x * b.z;
    Real const ab_mul_z = a.y * b.x;
    Real const bc_mul_x = b.z * c.y;
    Real const bc_mul_y = b.x * c.z;
    Real const bc_mul_z = b.y * c.x;

    gwn_query_vec3<Real> const cross_ab(
        a.y * b.z - ab_mul_x, a.z * b.x - ab_mul_y, a.x * b.y - ab_mul_z
    );
    gwn_query_vec3<Real> const cross_bc(
        b.y * c.z - bc_mul_x, b.z * c.x - bc_mul_y, b.x * c.y - bc_mul_z
    );

    // Select component-wise stable candidates as in Embree's stable normal routine.
    gwn_query_vec3<Real> normal = cross_bc;
    if (abs(ab_mul_x) < abs(bc_mul_x))
        normal.x = cross_ab.x;
    if (abs(ab_mul_y) < abs(bc_mul_y))
        normal.y = cross_ab.y;
    if (abs(ab_mul_z) < abs(bc_mul_z))
        normal.z = cross_ab.z;
    return normal;
}

template <gwn_real_type Real>
__device__ inline bool gwn_ray_triangle_intersect_robust_impl(
    gwn_query_vec3<Real> const &origin, gwn_query_vec3<Real> const &direction,
    gwn_query_vec3<Real> const &v0, gwn_query_vec3<Real> const &v1, gwn_query_vec3<Real> const &v2,
    Real const t_min, Real const t_max, Real &t_out, Real &u_out, Real &v_out
) noexcept {
    // Embree reference:
    // kernels/geometry/triangle_intersector_pluecker.h (Apache-2.0)
    // 1) Rebase vertices at ray origin.
    gwn_query_vec3<Real> const p0 = v0 - origin;
    gwn_query_vec3<Real> const p1 = v1 - origin;
    gwn_query_vec3<Real> const p2 = v2 - origin;

    // 2) Robust edge-function test in Pluecker space.
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

    // 3) Solve hit distance against stable geometric normal.
    gwn_query_vec3<Real> const ng = gwn_stable_triangle_normal_impl(e0, e1, e2);
    Real const den = Real(2) * gwn_query_dot(ng, direction);
    if (den == Real(0))
        return false;

    Real const t_num = Real(2) * gwn_query_dot(p0, ng);
    Real const t = t_num / den;
    if (t < t_min || t > t_max)
        return false;

    // Embree-style UV mapping for triangle (v0,v1,v2): u->v1 weight, v->v2 weight.
    // Match Embree's guarded reciprocal path for near-zero UVW.
    Real inv_uvw = Real(0);
    if (abs(uvw) >= Real(1e-18))
        inv_uvw = Real(1) / uvw;
    u_out = std::min(u * inv_uvw, Real(1));
    v_out = std::min(v * inv_uvw, Real(1));
    t_out = t;
    return true;
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_ray_load_triangle_vertices_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_query_vec3<Real> &a, gwn_query_vec3<Real> &b, gwn_query_vec3<Real> &c
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
    a = gwn_query_vec3<Real>(
        geometry.vertex_x[a_idx], geometry.vertex_y[a_idx], geometry.vertex_z[a_idx]
    );
    b = gwn_query_vec3<Real>(
        geometry.vertex_x[b_idx], geometry.vertex_y[b_idx], geometry.vertex_z[b_idx]
    );
    c = gwn_query_vec3<Real>(
        geometry.vertex_x[c_idx], geometry.vertex_y[c_idx], geometry.vertex_z[c_idx]
    );

    return true;
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_ray_triangle_intersect_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_query_vec3<Real> const &origin, gwn_query_vec3<Real> const &direction, Real const t_min,
    Real const t_max, Real &t_out, Real &u_out, Real &v_out
) noexcept {
    gwn_query_vec3<Real> a{};
    gwn_query_vec3<Real> b{};
    gwn_query_vec3<Real> c{};
    if (!gwn_ray_load_triangle_vertices_from_primitive_impl(geometry, primitive_id, a, b, c))
        return false;
    return gwn_ray_triangle_intersect_robust_impl(
        origin, direction, a, b, c, t_min, t_max, t_out, u_out, v_out
    );
}

template <int Width, gwn_real_type Real>
__device__ inline void gwn_sort_children_by_entry_t_impl(
    Real (&child_entry_t)[Width], int (&child_slot_order)[Width], std::uint8_t (&child_kind)[Width]
) noexcept {
    auto cmp_swap = [&](int const lhs, int const rhs) {
        if (!(child_entry_t[lhs] > child_entry_t[rhs]))
            return;
        using std::swap;
        swap(child_entry_t[lhs], child_entry_t[rhs]);
        swap(child_slot_order[lhs], child_slot_order[rhs]);
        swap(child_kind[lhs], child_kind[rhs]);
    };

    if constexpr (Width == 2) {
        cmp_swap(0, 1);
    } else if constexpr (Width == 4) {
        cmp_swap(0, 1);
        cmp_swap(2, 3);
        cmp_swap(0, 2);
        cmp_swap(1, 3);
        cmp_swap(1, 2);
    } else {
        GWN_PRAGMA_UNROLL
        for (int pass = 0; pass < Width; ++pass) {
            int const start = pass & 1;
            GWN_PRAGMA_UNROLL
            for (int i = start; (i + 1) < Width; i += 2)
                cmp_swap(i, i + 1);
        }
    }
}

template <int Width, gwn_real_type Real, gwn_index_type Index>
__device__ inline void gwn_ray_visit_leaf_primitive_range_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh, gwn_query_vec3<Real> const &origin,
    gwn_query_vec3<Real> const &direction, Real const t_min, Index const begin, Index const count,
    gwn_ray_best_hit<Real, Index> &best
) noexcept {
    for (Index off = 0; off < count; ++off) {
        Index const sorted_index = begin + off;
        if (!gwn_index_in_bounds(sorted_index, bvh.primitive_indices.size()))
            continue;

        Index const primitive_id = bvh.primitive_indices[static_cast<std::size_t>(sorted_index)];
        Real t_hit = Real(0);
        Real u_hit = Real(0);
        Real v_hit = Real(0);
        if (!gwn_ray_triangle_intersect_from_primitive_impl<Real, Index>(
                geometry, primitive_id, origin, direction, t_min, best.t, t_hit, u_hit, v_hit
            )) {
            continue;
        }

        if (!best.found || t_hit < best.t) {
            best.t = t_hit;
            best.primitive_id = primitive_id;
            best.u = u_hit;
            best.v = v_hit;
            best.found = true;
        }
    }
}

template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
__device__ inline gwn_ray_first_hit_result<Real, Index> gwn_ray_first_hit_bvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, Real const ray_ox,
    Real const ray_oy, Real const ray_oz, Real const ray_dx, Real const ray_dy, Real const ray_dz,
    Real const t_min, Real const t_max, OverflowCallback const &overflow_callback = {}
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
    auto const ray_dir_precomp = gwn_ray_make_dir_precompute_impl(ray_dx, ray_dy, ray_dz);

    gwn_ray_best_hit<Real, Index> best{};
    best.t = t_max;
    best.primitive_id = gwn_invalid_index<Index>();
    best.found = false;
    auto const set_result_from_best = [&]() noexcept {
        if (!best.found)
            return;
        result.t = best.t;
        result.primitive_id = best.primitive_id;
        result.u = best.u;
        result.v = best.v;
        result.status = gwn_ray_first_hit_status::k_hit;
    };

    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        gwn_ray_visit_leaf_primitive_range_impl<Width, Real, Index>(
            geometry, bvh, origin, direction, t_min, bvh.root_index, bvh.root_count, best
        );
        set_result_from_best();
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
                ray_ox, ray_oy, ray_oz, ray_dir_precomp, aabb_node.child_min_x[s],
                aabb_node.child_min_y[s], aabb_node.child_min_z[s], aabb_node.child_max_x[s],
                aabb_node.child_max_y[s], aabb_node.child_max_z[s], t_min, best.t
            );
            if (!interval.hit)
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
            if (child_entry_t[i] > best.t)
                continue;

            int const s = child_slot_order[i];
            gwn_ray_visit_leaf_primitive_range_impl<Width, Real, Index>(
                geometry, bvh, origin, direction, t_min, topo_node.child_index[s],
                topo_node.child_count[s], best
            );
        }

        // Push internal children in reverse near-to-far order so the nearest
        // node is popped first by the LIFO stack.
        GWN_PRAGMA_UNROLL
        for (int i = Width - 1; i >= 0; --i) {
            if (static_cast<gwn_bvh_child_kind>(child_kind[i]) != gwn_bvh_child_kind::k_internal)
                continue;
            if (child_entry_t[i] > best.t)
                continue;

            if (stack_size >= StackCapacity) {
                overflow_callback();
                set_result_from_best();
                result.status = gwn_ray_first_hit_status::k_overflow;
                return result;
            }
            stack[stack_size++] = topo_node.child_index[child_slot_order[i]];
        }
    }

    set_result_from_best();
    return result;
}

template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
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
    cuda::std::span<Real> out_t{};
    cuda::std::span<Index> out_primitive_id{};
    Real t_min{};
    Real t_max{};
    OverflowCallback overflow_callback{};

    __device__ void operator()(std::size_t const ray_id) const {
        auto const hit =
            gwn_ray_first_hit_bvh_impl<Width, Real, Index, StackCapacity, OverflowCallback>(
                geometry, bvh, aabb_tree, ray_origin_x[ray_id], ray_origin_y[ray_id],
                ray_origin_z[ray_id], ray_dir_x[ray_id], ray_dir_y[ray_id], ray_dir_z[ray_id],
                t_min, t_max, overflow_callback
            );
        out_t[ray_id] = hit.t;
        out_primitive_id[ray_id] = hit.primitive_id;
    }
};

} // namespace detail
} // namespace gwn
