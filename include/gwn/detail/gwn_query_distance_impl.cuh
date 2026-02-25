#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_geometry_impl.cuh"
#include "gwn_query_winding_impl.cuh"

namespace gwn {
namespace detail {

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
        if (!gwn_index_in_bounds(node_index, bvh.nodes.size()))
            continue;

        auto const &topo_node = bvh.nodes[static_cast<std::size_t>(node_index)];
        auto const &aabb_node = aabb_tree.nodes[static_cast<std::size_t>(node_index)];

        GWN_PRAGMA_UNROLL
        for (int s = 0; s < Width; ++s) {
            auto const kind = static_cast<gwn_bvh_child_kind>(topo_node.child_kind[s]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;

            Real const box_dist2 = gwn_aabb_min_distance_squared_impl(
                qx, qy, qz, aabb_node.child_min_x[s], aabb_node.child_min_y[s],
                aabb_node.child_min_z[s], aabb_node.child_max_x[s], aabb_node.child_max_y[s],
                aabb_node.child_max_z[s]
            );
            if (box_dist2 >= best_dist2)
                continue;

            if (kind == gwn_bvh_child_kind::k_internal) {
                if (stack_size >= StackCapacity)
                    __trap();
                stack[stack_size++] = topo_node.child_index[s];
                continue;
            }

            if (kind != gwn_bvh_child_kind::k_leaf)
                continue;

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
                if (d2 < best_dist2)
                    best_dist2 = d2;
            }
        }
    }

    return sqrt(best_dist2);
}

template <int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline Real gwn_signed_distance_point_bvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, Real const qx, Real const qy,
    Real const qz, Real const winding_number_threshold, Real const culling_band
) noexcept {
    Real const dist = gwn_unsigned_distance_point_bvh_impl<Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, qx, qy, qz, culling_band
    );
    Real const wn = gwn_winding_number_point_bvh_exact_impl<Width, Real, Index, StackCapacity>(
        geometry, bvh, qx, qy, qz
    );
    return wn >= winding_number_threshold ? -dist : dist;
}

} // namespace detail
} // namespace gwn
