#pragma once

#include <cstddef>

#include "gwn/detail/gwn_bvh_build_taylor_async.cuh"

namespace gwn {
namespace detail {
template <int Order, class Real, class Index> struct gwn_build_taylor_levelwise_functor {
    using moment_type = gwn_device_taylor_moment<Order, Real>;

    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_accessor<Real, Index> bvh{};
    cuda::std::span<moment_type> node_moments{};
    cuda::std::span<gwn_bvh4_taylor_node_soa<Order, Real>> taylor_nodes{};
    cuda::std::span<Index const> node_ids{};

    __device__ void operator()(std::size_t const local_node_id) const {
        if (local_node_id >= node_ids.size())
            return;
        Index const node_index = node_ids[local_node_id];
        if (node_index < Index(0))
            return;
        std::size_t const node_id = static_cast<std::size_t>(node_index);
        if (node_id >= bvh.nodes.size())
            return;

        gwn_bvh4_node_soa<Real, Index> const &node = bvh.nodes[node_id];
        moment_type child_moments[4]{};
        bool child_valid[4] = {false, false, false, false};
        bool has_child = false;
        gwn_aabb<Real> merged_bounds{};

        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;

            moment_type child{};
            if (kind == gwn_bvh_child_kind::k_internal) {
                Index const child_index = node.child_index[child_slot];
                if (child_index < Index(0) ||
                    static_cast<std::size_t>(child_index) >= bvh.nodes.size()) {
                    continue;
                }
                child = node_moments[static_cast<std::size_t>(child_index)];
            } else if (kind == gwn_bvh_child_kind::k_leaf) {
                child = gwn_compute_leaf_taylor_moment_cached<Order>(
                    geometry, bvh, node.child_index[child_slot], node.child_count[child_slot]
                );
            } else {
                continue;
            }

            child_moments[child_slot] = child;
            child_valid[child_slot] = true;

            gwn_aabb<Real> const bounds{
                node.child_min_x[child_slot], node.child_min_y[child_slot],
                node.child_min_z[child_slot], node.child_max_x[child_slot],
                node.child_max_y[child_slot], node.child_max_z[child_slot],
            };
            if (!has_child) {
                merged_bounds = bounds;
                has_child = true;
            } else {
                merged_bounds = gwn_aabb_union(merged_bounds, bounds);
            }
        }

        moment_type parent{};
        if (has_child) {
            GWN_PRAGMA_UNROLL
            for (int child_slot = 0; child_slot < 4; ++child_slot) {
                if (!child_valid[child_slot])
                    continue;
                moment_type const child = child_moments[child_slot];
                parent.area += child.area;
                parent.area_p_x += child.area_p_x;
                parent.area_p_y += child.area_p_y;
                parent.area_p_z += child.area_p_z;
                parent.n_x += child.n_x;
                parent.n_y += child.n_y;
                parent.n_z += child.n_z;
            }

            if (parent.area > Real(0)) {
                parent.average_x = parent.area_p_x / parent.area;
                parent.average_y = parent.area_p_y / parent.area;
                parent.average_z = parent.area_p_z / parent.area;
            } else {
                parent.average_x = (merged_bounds.min_x + merged_bounds.max_x) * Real(0.5);
                parent.average_y = (merged_bounds.min_y + merged_bounds.max_y) * Real(0.5);
                parent.average_z = (merged_bounds.min_z + merged_bounds.max_z) * Real(0.5);
            }
            parent.max_p_dist2 = gwn_bounds_max_p_dist2(
                merged_bounds, parent.average_x, parent.average_y, parent.average_z
            );

            if constexpr (Order >= 1) {
                GWN_PRAGMA_UNROLL
                for (int child_slot = 0; child_slot < 4; ++child_slot) {
                    if (!child_valid[child_slot])
                        continue;
                    moment_type const child = child_moments[child_slot];
                    Real const dx = child.average_x - parent.average_x;
                    Real const dy = child.average_y - parent.average_y;
                    Real const dz = child.average_z - parent.average_z;

                    parent.nij_xx += child.nij_xx + child.n_x * dx;
                    parent.nij_yy += child.nij_yy + child.n_y * dy;
                    parent.nij_zz += child.nij_zz + child.n_z * dz;
                    parent.nxy += child.nxy + child.n_x * dy;
                    parent.nyx += child.nyx + child.n_y * dx;
                    parent.nyz += child.nyz + child.n_y * dz;
                    parent.nzy += child.nzy + child.n_z * dy;
                    parent.nzx += child.nzx + child.n_z * dx;
                    parent.nxz += child.nxz + child.n_x * dz;
                }
            }
        }

        gwn_bvh4_taylor_node_soa<Order, Real> taylor{};
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot)
            if (child_valid[child_slot])
                gwn_write_taylor_child<Order>(taylor, child_slot, child_moments[child_slot]);
            else
                gwn_zero_taylor_child<Order>(taylor, child_slot);

        node_moments[node_id] = parent;
        taylor_nodes[node_id] = taylor;
    }
};

} // namespace detail
} // namespace gwn
