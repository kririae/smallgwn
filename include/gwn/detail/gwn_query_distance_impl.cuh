#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_geometry_impl.cuh"
#include "gwn_query_winding_impl.cuh"

namespace gwn {
namespace detail {

template <gwn_real_type Real>
__device__ inline void gwn_swap_children_if_greater_impl(
    Real &lhs_dist2, Real &rhs_dist2, int &lhs_slot, int &rhs_slot, std::uint8_t &lhs_kind,
    std::uint8_t &rhs_kind
) noexcept {
    if (!(lhs_dist2 > rhs_dist2))
        return;

    Real const dist2_tmp = lhs_dist2;
    lhs_dist2 = rhs_dist2;
    rhs_dist2 = dist2_tmp;

    int const slot_tmp = lhs_slot;
    lhs_slot = rhs_slot;
    rhs_slot = slot_tmp;

    std::uint8_t const kind_tmp = lhs_kind;
    lhs_kind = rhs_kind;
    rhs_kind = kind_tmp;
}

template <int Width, gwn_real_type Real>
__device__ inline void gwn_sort_children_by_dist2_impl(
    Real (&child_box_dist2)[Width], int (&child_slot_order)[Width],
    std::uint8_t (&child_kind)[Width]
) noexcept {
    if constexpr (Width == 2) {
        gwn_swap_children_if_greater_impl(
            child_box_dist2[0], child_box_dist2[1], child_slot_order[0], child_slot_order[1],
            child_kind[0], child_kind[1]
        );
    } else if constexpr (Width == 4) {
        // Optimal 4-input sorting network (5 compare-swaps).
        gwn_swap_children_if_greater_impl(
            child_box_dist2[0], child_box_dist2[1], child_slot_order[0], child_slot_order[1],
            child_kind[0], child_kind[1]
        );
        gwn_swap_children_if_greater_impl(
            child_box_dist2[2], child_box_dist2[3], child_slot_order[2], child_slot_order[3],
            child_kind[2], child_kind[3]
        );
        gwn_swap_children_if_greater_impl(
            child_box_dist2[0], child_box_dist2[2], child_slot_order[0], child_slot_order[2],
            child_kind[0], child_kind[2]
        );
        gwn_swap_children_if_greater_impl(
            child_box_dist2[1], child_box_dist2[3], child_slot_order[1], child_slot_order[3],
            child_kind[1], child_kind[3]
        );
        gwn_swap_children_if_greater_impl(
            child_box_dist2[1], child_box_dist2[2], child_slot_order[1], child_slot_order[2],
            child_kind[1], child_kind[2]
        );
    } else {
        // Fixed pass-count odd-even network; no data-dependent inner while loop.
        GWN_PRAGMA_UNROLL
        for (int pass = 0; pass < Width; ++pass) {
            int const start = pass & 1;
            GWN_PRAGMA_UNROLL
            for (int i = start; (i + 1) < Width; i += 2) {
                gwn_swap_children_if_greater_impl(
                    child_box_dist2[i], child_box_dist2[i + 1], child_slot_order[i],
                    child_slot_order[i + 1], child_kind[i], child_kind[i + 1]
                );
            }
        }
    }
}

template <int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline Real gwn_unsigned_distance_point_bvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, Real const qx, Real const qy,
    Real const qz, Real const culling_band
) noexcept {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid() || !bvh.is_valid() || !aabb_tree.is_valid_for(bvh))
        return Real(0);

    gwn_query_vec3<Real> const query(qx, qy, qz);
    Real band = culling_band;
    if (!(band >= Real(0)))
        band = Real(0);

    Real best_dist2 = band * band;
    if (!isfinite(best_dist2))
        best_dist2 = std::numeric_limits<Real>::infinity();

    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        for (Index off = 0; off < bvh.root_count; ++off) {
            Index const si = bvh.root_index + off;
            if (!gwn_index_in_bounds(si, bvh.primitive_indices.size()))
                continue;
            Index const pi = bvh.primitive_indices[static_cast<std::size_t>(si)];
            Real const d2 =
                gwn_triangle_distance_squared_from_primitive_impl<Real, Index>(geometry, pi, query);
            if (d2 < best_dist2)
                best_dist2 = d2;
        }
        return sqrt(best_dist2);
    }

    if (bvh.root_kind != gwn_bvh_child_kind::k_internal)
        return Real(0);

    Index stack[StackCapacity];
    int stack_size = 0;
    stack[stack_size++] = bvh.root_index;

    while (stack_size > 0) {
        Index const node_index = stack[--stack_size];
        GWN_ASSERT(stack_size >= 0, "distance: stack underflow");
        if (!gwn_index_in_bounds(node_index, bvh.nodes.size()))
            continue;

        auto const &topo_node = bvh.nodes[static_cast<std::size_t>(node_index)];
        GWN_ASSERT(
            gwn_index_in_bounds(node_index, aabb_tree.nodes.size()),
            "distance: node_index out of bounds for aabb_tree"
        );
        auto const &aabb_node = aabb_tree.nodes[static_cast<std::size_t>(node_index)];
        int child_slot_order[Width];
        Real child_box_dist2[Width];
        std::uint8_t child_kind[Width];
        std::uint8_t constexpr k_invalid_child_kind =
            static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
        Real constexpr k_infinite_dist2 = std::numeric_limits<Real>::infinity();
        GWN_PRAGMA_UNROLL
        for (int i = 0; i < Width; ++i) {
            child_slot_order[i] = 0;
            child_box_dist2[i] = k_infinite_dist2;
            child_kind[i] = k_invalid_child_kind;
        }
        int child_count = 0;

        GWN_PRAGMA_UNROLL
        for (int s = 0; s < Width; ++s) {
            auto const kind = static_cast<gwn_bvh_child_kind>(topo_node.child_kind[s]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf)
                continue;

            Real const box_dist2 = gwn_aabb_min_distance_squared_impl(
                qx, qy, qz, aabb_node.child_min_x[s], aabb_node.child_min_y[s],
                aabb_node.child_min_z[s], aabb_node.child_max_x[s], aabb_node.child_max_y[s],
                aabb_node.child_max_z[s]
            );
            if (box_dist2 >= best_dist2)
                continue;

            if (child_count >= Width)
                continue;
            child_slot_order[child_count] = s;
            child_box_dist2[child_count] = box_dist2;
            child_kind[child_count] = topo_node.child_kind[s];
            ++child_count;
        }

        if (child_count > 1)
            gwn_sort_children_by_dist2_impl<Width>(child_box_dist2, child_slot_order, child_kind);

        // Visit leaves first to tighten best_dist2 before pushing internal nodes.
        GWN_PRAGMA_UNROLL
        for (int i = 0; i < Width; ++i) {
            if (static_cast<gwn_bvh_child_kind>(child_kind[i]) != gwn_bvh_child_kind::k_leaf)
                continue;
            if (child_box_dist2[i] >= best_dist2)
                continue;

            int const s = child_slot_order[i];

            Index const begin = topo_node.child_index[s];
            Index const count = topo_node.child_count[s];
            for (Index off = 0; off < count; ++off) {
                Index const si = begin + off;
                if (!gwn_index_in_bounds(si, bvh.primitive_indices.size()))
                    continue;
                Index const pi = bvh.primitive_indices[static_cast<std::size_t>(si)];
                Real const d2 = gwn_triangle_distance_squared_from_primitive_impl<Real, Index>(
                    geometry, pi, query
                );
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    if (!(best_dist2 > Real(0)))
                        return Real(0);
                }
            }
        }

        // Push internals in reverse order so the nearest child is processed first (LIFO stack).
        GWN_PRAGMA_UNROLL
        for (int i = Width - 1; i >= 0; --i) {
            if (static_cast<gwn_bvh_child_kind>(child_kind[i]) != gwn_bvh_child_kind::k_internal)
                continue;
            if (child_box_dist2[i] >= best_dist2)
                continue;

            if (stack_size >= StackCapacity)
                gwn_trap();
            stack[stack_size++] = topo_node.child_index[child_slot_order[i]];
        }
    }

    return sqrt(best_dist2);
}

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline Real gwn_signed_distance_point_bvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree, Real const qx,
    Real const qy, Real const qz, Real const winding_number_threshold, Real const culling_band,
    Real const accuracy_scale
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_signed_distance_point_bvh_impl currently supports Order 0, 1, and 2."
    );

    Real const dist = gwn_unsigned_distance_point_bvh_impl<Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, qx, qy, qz, culling_band
    );
    Real wn = Real(0);
    if (data_tree.is_valid_for(bvh)) {
        wn = gwn_winding_number_point_bvh_taylor_impl<Order, Width, Real, Index, StackCapacity>(
            geometry, bvh, data_tree, qx, qy, qz, accuracy_scale
        );
    }
    return wn >= winding_number_threshold ? -dist : dist;
}

} // namespace detail
} // namespace gwn
