#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "gwn/detail/gwn_bvh_build_common.cuh"

namespace gwn {
namespace detail {
template <int Order, class Real> struct gwn_device_taylor_moment;

template <class Real> struct gwn_device_taylor_moment<0, Real> {
    Real area = Real(0);
    Real area_p_x = Real(0);
    Real area_p_y = Real(0);
    Real area_p_z = Real(0);
    Real average_x = Real(0);
    Real average_y = Real(0);
    Real average_z = Real(0);
    Real n_x = Real(0);
    Real n_y = Real(0);
    Real n_z = Real(0);
    Real max_p_dist2 = Real(0);
};

template <class Real> struct gwn_device_taylor_moment<1, Real> : gwn_device_taylor_moment<0, Real> {
    Real nij_xx = Real(0);
    Real nij_yy = Real(0);
    Real nij_zz = Real(0);
    Real nxy = Real(0);
    Real nyx = Real(0);
    Real nyz = Real(0);
    Real nzy = Real(0);
    Real nzx = Real(0);
    Real nxz = Real(0);
};

template <class Real>
[[nodiscard]] __host__ __device__ inline Real gwn_bounds_max_p_dist2(
    gwn_aabb<Real> const &bounds, Real const average_x, Real const average_y, Real const average_z
) noexcept {
    Real const dx = std::max(average_x - bounds.min_x, bounds.max_x - average_x);
    Real const dy = std::max(average_y - bounds.min_y, bounds.max_y - average_y);
    Real const dz = std::max(average_z - bounds.min_z, bounds.max_z - average_z);
    return dx * dx + dy * dy + dz * dz;
}

template <int Order, class Real>
__device__ inline void
gwn_zero_taylor_child(gwn_bvh4_taylor_node_soa<Order, Real> &node, int const child_slot) {
    node.child_max_p_dist2[child_slot] = Real(0);
    node.child_average_x[child_slot] = Real(0);
    node.child_average_y[child_slot] = Real(0);
    node.child_average_z[child_slot] = Real(0);
    node.child_n_x[child_slot] = Real(0);
    node.child_n_y[child_slot] = Real(0);
    node.child_n_z[child_slot] = Real(0);
    if constexpr (Order >= 1) {
        node.child_nij_xx[child_slot] = Real(0);
        node.child_nij_yy[child_slot] = Real(0);
        node.child_nij_zz[child_slot] = Real(0);
        node.child_nxy_nyx[child_slot] = Real(0);
        node.child_nyz_nzy[child_slot] = Real(0);
        node.child_nzx_nxz[child_slot] = Real(0);
    }
}

template <int Order, class Real>
__device__ inline void gwn_write_taylor_child(
    gwn_bvh4_taylor_node_soa<Order, Real> &node, int const child_slot,
    gwn_device_taylor_moment<Order, Real> const &moment
) {
    node.child_max_p_dist2[child_slot] = moment.max_p_dist2;
    node.child_average_x[child_slot] = moment.average_x;
    node.child_average_y[child_slot] = moment.average_y;
    node.child_average_z[child_slot] = moment.average_z;
    node.child_n_x[child_slot] = moment.n_x;
    node.child_n_y[child_slot] = moment.n_y;
    node.child_n_z[child_slot] = moment.n_z;
    if constexpr (Order >= 1) {
        node.child_nij_xx[child_slot] = moment.nij_xx;
        node.child_nij_yy[child_slot] = moment.nij_yy;
        node.child_nij_zz[child_slot] = moment.nij_zz;
        node.child_nxy_nyx[child_slot] = moment.nxy + moment.nyx;
        node.child_nyz_nzy[child_slot] = moment.nyz + moment.nzy;
        node.child_nzx_nxz[child_slot] = moment.nzx + moment.nxz;
    }
}

template <int Order, class Real, class Index>
__device__ inline bool gwn_compute_triangle_taylor_moment(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_device_taylor_moment<Order, Real> &moment, gwn_aabb<Real> &bounds
) noexcept {
    if (primitive_id < Index(0) ||
        static_cast<std::size_t>(primitive_id) >= geometry.triangle_count()) {
        return false;
    }

    std::size_t const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (ia < Index(0) || ib < Index(0) || ic < Index(0))
        return false;

    std::size_t const a_index = static_cast<std::size_t>(ia);
    std::size_t const b_index = static_cast<std::size_t>(ib);
    std::size_t const c_index = static_cast<std::size_t>(ic);
    if (a_index >= geometry.vertex_count() || b_index >= geometry.vertex_count() ||
        c_index >= geometry.vertex_count()) {
        return false;
    }

    Real const ax = geometry.vertex_x[a_index];
    Real const ay = geometry.vertex_y[a_index];
    Real const az = geometry.vertex_z[a_index];
    Real const bx = geometry.vertex_x[b_index];
    Real const by = geometry.vertex_y[b_index];
    Real const bz = geometry.vertex_z[b_index];
    Real const cx = geometry.vertex_x[c_index];
    Real const cy = geometry.vertex_y[c_index];
    Real const cz = geometry.vertex_z[c_index];

    bounds = gwn_aabb<Real>{
        std::min(ax, std::min(bx, cx)), std::min(ay, std::min(by, cy)),
        std::min(az, std::min(bz, cz)), std::max(ax, std::max(bx, cx)),
        std::max(ay, std::max(by, cy)), std::max(az, std::max(bz, cz)),
    };

    Real const abx = bx - ax;
    Real const aby = by - ay;
    Real const abz = bz - az;
    Real const acx = cx - ax;
    Real const acy = cy - ay;
    Real const acz = cz - az;

    moment.n_x = Real(0.5) * (aby * acz - abz * acy);
    moment.n_y = Real(0.5) * (abz * acx - abx * acz);
    moment.n_z = Real(0.5) * (abx * acy - aby * acx);

    Real const area2 = moment.n_x * moment.n_x + moment.n_y * moment.n_y + moment.n_z * moment.n_z;
    moment.area = sqrt(std::max(area2, Real(0)));
    moment.average_x = (ax + bx + cx) / Real(3);
    moment.average_y = (ay + by + cy) / Real(3);
    moment.average_z = (az + bz + cz) / Real(3);
    moment.area_p_x = moment.average_x * moment.area;
    moment.area_p_y = moment.average_y * moment.area;
    moment.area_p_z = moment.average_z * moment.area;
    moment.max_p_dist2 =
        gwn_bounds_max_p_dist2(bounds, moment.average_x, moment.average_y, moment.average_z);
    return true;
}

template <int Order, class Real, class Index>
__device__ inline gwn_device_taylor_moment<Order, Real> gwn_compute_leaf_taylor_moment(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> const &bvh,
    Index const begin, Index const count
) noexcept {
    using moment_type = gwn_device_taylor_moment<Order, Real>;

    moment_type leaf{};
    if (count <= Index(0))
        return leaf;

    bool has_primitive = false;
    gwn_aabb<Real> leaf_bounds{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
    for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
        Index const sorted_slot = begin + primitive_offset;
        if (sorted_slot < Index(0))
            continue;
        std::size_t const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
        if (sorted_slot_u >= bvh.primitive_indices.size())
            continue;

        Index const primitive_id = bvh.primitive_indices[sorted_slot_u];
        moment_type primitive{};
        gwn_aabb<Real> primitive_bounds{};
        if (!gwn_compute_triangle_taylor_moment<Order, Real, Index>(
                geometry, primitive_id, primitive, primitive_bounds
            )) {
            continue;
        }

        if (!has_primitive) {
            leaf_bounds = primitive_bounds;
            has_primitive = true;
        } else {
            leaf_bounds = gwn_aabb_union(leaf_bounds, primitive_bounds);
        }

        leaf.area += primitive.area;
        leaf.area_p_x += primitive.area_p_x;
        leaf.area_p_y += primitive.area_p_y;
        leaf.area_p_z += primitive.area_p_z;
        leaf.n_x += primitive.n_x;
        leaf.n_y += primitive.n_y;
        leaf.n_z += primitive.n_z;
    }

    if (!has_primitive)
        return leaf;

    if (leaf.area > Real(0)) {
        leaf.average_x = leaf.area_p_x / leaf.area;
        leaf.average_y = leaf.area_p_y / leaf.area;
        leaf.average_z = leaf.area_p_z / leaf.area;
    } else {
        leaf.average_x = (leaf_bounds.min_x + leaf_bounds.max_x) * Real(0.5);
        leaf.average_y = (leaf_bounds.min_y + leaf_bounds.max_y) * Real(0.5);
        leaf.average_z = (leaf_bounds.min_z + leaf_bounds.max_z) * Real(0.5);
    }
    leaf.max_p_dist2 =
        gwn_bounds_max_p_dist2(leaf_bounds, leaf.average_x, leaf.average_y, leaf.average_z);

    if constexpr (Order >= 1) {
        for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
            Index const sorted_slot = begin + primitive_offset;
            if (sorted_slot < Index(0))
                continue;
            std::size_t const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
            if (sorted_slot_u >= bvh.primitive_indices.size())
                continue;

            Index const primitive_id = bvh.primitive_indices[sorted_slot_u];
            moment_type primitive{};
            gwn_aabb<Real> primitive_bounds{};
            if (!gwn_compute_triangle_taylor_moment<Order, Real, Index>(
                    geometry, primitive_id, primitive, primitive_bounds
                )) {
                continue;
            }

            Real const dx = primitive.average_x - leaf.average_x;
            Real const dy = primitive.average_y - leaf.average_y;
            Real const dz = primitive.average_z - leaf.average_z;

            leaf.nij_xx += primitive.nij_xx + primitive.n_x * dx;
            leaf.nij_yy += primitive.nij_yy + primitive.n_y * dy;
            leaf.nij_zz += primitive.nij_zz + primitive.n_z * dz;
            leaf.nxy += primitive.nxy + primitive.n_x * dy;
            leaf.nyx += primitive.nyx + primitive.n_y * dx;
            leaf.nyz += primitive.nyz + primitive.n_y * dz;
            leaf.nzy += primitive.nzy + primitive.n_z * dy;
            leaf.nzx += primitive.nzx + primitive.n_z * dx;
            leaf.nxz += primitive.nxz + primitive.n_x * dz;
        }
    }

    return leaf;
}

template <int Order, class Real, class Index>
__device__ inline gwn_device_taylor_moment<Order, Real> gwn_compute_leaf_taylor_moment_cached(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> const &bvh,
    Index const begin, Index const count
) noexcept {
    using moment_type = gwn_device_taylor_moment<Order, Real>;

    constexpr int k_leaf_cache_capacity = 8;
    if (count <= Index(0))
        return moment_type{};
    if (count > Index(k_leaf_cache_capacity))
        return gwn_compute_leaf_taylor_moment<Order>(geometry, bvh, begin, count);

    moment_type primitive_cache[k_leaf_cache_capacity]{};
    gwn_aabb<Real> bounds_cache[k_leaf_cache_capacity]{};
    int cache_count = 0;

    moment_type leaf{};
    bool has_primitive = false;
    gwn_aabb<Real> leaf_bounds{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
    for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
        Index const sorted_slot = begin + primitive_offset;
        if (sorted_slot < Index(0))
            continue;
        std::size_t const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
        if (sorted_slot_u >= bvh.primitive_indices.size())
            continue;

        Index const primitive_id = bvh.primitive_indices[sorted_slot_u];
        moment_type primitive{};
        gwn_aabb<Real> primitive_bounds{};
        if (!gwn_compute_triangle_taylor_moment<Order, Real, Index>(
                geometry, primitive_id, primitive, primitive_bounds
            )) {
            continue;
        }

        primitive_cache[cache_count] = primitive;
        bounds_cache[cache_count] = primitive_bounds;
        ++cache_count;

        if (!has_primitive) {
            leaf_bounds = primitive_bounds;
            has_primitive = true;
        } else {
            leaf_bounds = gwn_aabb_union(leaf_bounds, primitive_bounds);
        }

        leaf.area += primitive.area;
        leaf.area_p_x += primitive.area_p_x;
        leaf.area_p_y += primitive.area_p_y;
        leaf.area_p_z += primitive.area_p_z;
        leaf.n_x += primitive.n_x;
        leaf.n_y += primitive.n_y;
        leaf.n_z += primitive.n_z;
    }

    if (!has_primitive)
        return leaf;

    if (leaf.area > Real(0)) {
        leaf.average_x = leaf.area_p_x / leaf.area;
        leaf.average_y = leaf.area_p_y / leaf.area;
        leaf.average_z = leaf.area_p_z / leaf.area;
    } else {
        leaf.average_x = (leaf_bounds.min_x + leaf_bounds.max_x) * Real(0.5);
        leaf.average_y = (leaf_bounds.min_y + leaf_bounds.max_y) * Real(0.5);
        leaf.average_z = (leaf_bounds.min_z + leaf_bounds.max_z) * Real(0.5);
    }
    leaf.max_p_dist2 =
        gwn_bounds_max_p_dist2(leaf_bounds, leaf.average_x, leaf.average_y, leaf.average_z);

    if constexpr (Order >= 1) {
        GWN_PRAGMA_UNROLL
        for (int cache_index = 0; cache_index < k_leaf_cache_capacity; ++cache_index) {
            if (cache_index >= cache_count)
                break;
            moment_type const primitive = primitive_cache[cache_index];
            Real const dx = primitive.average_x - leaf.average_x;
            Real const dy = primitive.average_y - leaf.average_y;
            Real const dz = primitive.average_z - leaf.average_z;

            leaf.nij_xx += primitive.nij_xx + primitive.n_x * dx;
            leaf.nij_yy += primitive.nij_yy + primitive.n_y * dy;
            leaf.nij_zz += primitive.nij_zz + primitive.n_z * dz;
            leaf.nxy += primitive.nxy + primitive.n_x * dy;
            leaf.nyx += primitive.nyx + primitive.n_y * dx;
            leaf.nyz += primitive.nyz + primitive.n_y * dz;
            leaf.nzy += primitive.nzy + primitive.n_z * dy;
            leaf.nzx += primitive.nzx + primitive.n_z * dx;
            leaf.nxz += primitive.nxz + primitive.n_x * dz;
        }
    }

    return leaf;
}

template <class Real, class Index> struct gwn_prepare_taylor_async_topology_functor {
    gwn_bvh_accessor<Real, Index> bvh{};
    cuda::std::span<Index> internal_parent{};
    cuda::std::span<std::uint8_t> internal_parent_slot{};
    cuda::std::span<std::uint8_t> internal_arity{};
    unsigned int *error_flag = nullptr;

    __device__ inline void gwn_mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ void operator()(std::size_t const node_id) const {
        if (node_id >= bvh.nodes.size() || node_id >= internal_parent.size() ||
            node_id >= internal_parent_slot.size() || node_id >= internal_arity.size()) {
            gwn_mark_error();
            return;
        }

        gwn_bvh4_node_soa<Real, Index> const &node = bvh.nodes[node_id];
        std::uint8_t node_arity = 0;
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;

            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf) {
                gwn_mark_error();
                continue;
            }

            ++node_arity;
            if (kind == gwn_bvh_child_kind::k_internal) {
                Index const child_index = node.child_index[child_slot];
                if (child_index < Index(0) ||
                    static_cast<std::size_t>(child_index) >= bvh.nodes.size()) {
                    gwn_mark_error();
                    continue;
                }
                std::size_t const child_index_u = static_cast<std::size_t>(child_index);
                internal_parent[child_index_u] = static_cast<Index>(node_id);
                internal_parent_slot[child_index_u] = static_cast<std::uint8_t>(child_slot);
                continue;
            }

            Index const leaf_begin = node.child_index[child_slot];
            Index const leaf_count = node.child_count[child_slot];
            if (leaf_begin < Index(0) || leaf_count < Index(0)) {
                gwn_mark_error();
                continue;
            }
            std::size_t const leaf_begin_u = static_cast<std::size_t>(leaf_begin);
            std::size_t const leaf_count_u = static_cast<std::size_t>(leaf_count);
            if (leaf_begin_u > bvh.primitive_indices.size() ||
                leaf_count_u > (bvh.primitive_indices.size() - leaf_begin_u)) {
                gwn_mark_error();
                continue;
            }
        }

        internal_arity[node_id] = node_arity;
        if (node_arity == 0)
            gwn_mark_error();
    }
};

template <int Order, class Real, class Index> struct gwn_build_taylor_async_from_leaves_functor {
    using moment_type = gwn_device_taylor_moment<Order, Real>;

    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_accessor<Real, Index> bvh{};
    cuda::std::span<Index const> internal_parent{};
    cuda::std::span<std::uint8_t const> internal_parent_slot{};
    cuda::std::span<std::uint8_t const> internal_arity{};
    cuda::std::span<unsigned int> internal_arrivals{};
    cuda::std::span<moment_type> pending_child_moments{};
    cuda::std::span<gwn_bvh4_taylor_node_soa<Order, Real>> taylor_nodes{};
    unsigned int *error_flag = nullptr;

    __device__ inline void gwn_mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ inline bool
    gwn_pending_index_is_valid(std::size_t const node_id, int const child_slot) const noexcept {
        if (child_slot < 0 || child_slot >= 4)
            return false;
        if (node_id > (std::numeric_limits<std::size_t>::max() / std::size_t(4)))
            return false;

        std::size_t const pending_index =
            node_id * std::size_t(4) + static_cast<std::size_t>(child_slot);
        return pending_index < pending_child_moments.size();
    }

    __device__ inline std::size_t
    gwn_pending_index(std::size_t const node_id, int const child_slot) const noexcept {
        return node_id * std::size_t(4) + static_cast<std::size_t>(child_slot);
    }

    __device__ bool
    gwn_finalize_node(std::size_t const node_id, moment_type &out_parent_moment) const noexcept {
        if (node_id >= bvh.nodes.size() || node_id >= taylor_nodes.size()) {
            gwn_mark_error();
            return false;
        }

        gwn_bvh4_node_soa<Real, Index> const &node = bvh.nodes[node_id];
        gwn_bvh4_taylor_node_soa<Order, Real> taylor{};
        moment_type parent{};
        bool has_child = false;
        gwn_aabb<Real> merged_bounds{};

        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid) {
                gwn_zero_taylor_child<Order>(taylor, child_slot);
                continue;
            }
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf) {
                gwn_zero_taylor_child<Order>(taylor, child_slot);
                gwn_mark_error();
                continue;
            }
            if (!gwn_pending_index_is_valid(node_id, child_slot)) {
                gwn_zero_taylor_child<Order>(taylor, child_slot);
                gwn_mark_error();
                continue;
            }

            moment_type const child_moment =
                pending_child_moments[gwn_pending_index(node_id, child_slot)];
            gwn_write_taylor_child<Order>(taylor, child_slot, child_moment);

            parent.area += child_moment.area;
            parent.area_p_x += child_moment.area_p_x;
            parent.area_p_y += child_moment.area_p_y;
            parent.area_p_z += child_moment.area_p_z;
            parent.n_x += child_moment.n_x;
            parent.n_y += child_moment.n_y;
            parent.n_z += child_moment.n_z;

            gwn_aabb<Real> const child_bounds{
                node.child_min_x[child_slot], node.child_min_y[child_slot],
                node.child_min_z[child_slot], node.child_max_x[child_slot],
                node.child_max_y[child_slot], node.child_max_z[child_slot],
            };
            if (!has_child) {
                merged_bounds = child_bounds;
                has_child = true;
            } else {
                merged_bounds = gwn_aabb_union(merged_bounds, child_bounds);
            }
        }

        if (has_child) {
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
                    auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
                    if (kind == gwn_bvh_child_kind::k_invalid)
                        continue;
                    if (!gwn_pending_index_is_valid(node_id, child_slot)) {
                        gwn_mark_error();
                        continue;
                    }
                    moment_type const child_moment =
                        pending_child_moments[gwn_pending_index(node_id, child_slot)];
                    Real const dx = child_moment.average_x - parent.average_x;
                    Real const dy = child_moment.average_y - parent.average_y;
                    Real const dz = child_moment.average_z - parent.average_z;

                    parent.nij_xx += child_moment.nij_xx + child_moment.n_x * dx;
                    parent.nij_yy += child_moment.nij_yy + child_moment.n_y * dy;
                    parent.nij_zz += child_moment.nij_zz + child_moment.n_z * dz;
                    parent.nxy += child_moment.nxy + child_moment.n_x * dy;
                    parent.nyx += child_moment.nyx + child_moment.n_y * dx;
                    parent.nyz += child_moment.nyz + child_moment.n_y * dz;
                    parent.nzy += child_moment.nzy + child_moment.n_z * dy;
                    parent.nzx += child_moment.nzx + child_moment.n_z * dx;
                    parent.nxz += child_moment.nxz + child_moment.n_x * dz;
                }
            }
        }

        taylor_nodes[node_id] = taylor;
        out_parent_moment = parent;
        return true;
    }

    __device__ void gwn_propagate_up(
        Index current_parent, std::uint8_t current_slot, moment_type current_moment
    ) const noexcept {
        while (current_parent >= Index(0)) {
            std::size_t const parent_id = static_cast<std::size_t>(current_parent);
            if (parent_id >= bvh.nodes.size() || parent_id >= internal_parent.size() ||
                parent_id >= internal_parent_slot.size() || parent_id >= internal_arity.size() ||
                parent_id >= internal_arrivals.size() || parent_id >= taylor_nodes.size()) {
                gwn_mark_error();
                return;
            }
            if (current_slot >= 4) {
                gwn_mark_error();
                return;
            }
            if (!gwn_pending_index_is_valid(parent_id, static_cast<int>(current_slot))) {
                gwn_mark_error();
                return;
            }

            pending_child_moments[gwn_pending_index(parent_id, static_cast<int>(current_slot))] =
                current_moment;
            __threadfence();

            unsigned int const previous_arrivals =
                atomicAdd(internal_arrivals.data() + parent_id, 1u);
            unsigned int const next_arrivals = previous_arrivals + 1u;
            unsigned int const expected_arrivals =
                static_cast<unsigned int>(internal_arity[parent_id]);
            if (expected_arrivals == 0u || expected_arrivals > 4u) {
                gwn_mark_error();
                return;
            }
            if (next_arrivals < expected_arrivals)
                return;
            if (next_arrivals > expected_arrivals) {
                gwn_mark_error();
                return;
            }

            __threadfence();

            moment_type parent_moment{};
            if (!gwn_finalize_node(parent_id, parent_moment))
                return;

            Index const parent_parent = internal_parent[parent_id];
            if (parent_parent < Index(0))
                return;

            std::uint8_t const parent_parent_slot = internal_parent_slot[parent_id];
            if (parent_parent_slot >= 4) {
                gwn_mark_error();
                return;
            }

            current_parent = parent_parent;
            current_slot = parent_parent_slot;
            current_moment = parent_moment;
        }
    }

    __device__ void operator()(std::size_t const edge_index) const {
        if (edge_index > (std::numeric_limits<std::size_t>::max() / std::size_t(4)))
            return;

        std::size_t const node_id = edge_index >> 2;
        int const child_slot = static_cast<int>(edge_index & std::size_t(3));
        if (node_id >= bvh.nodes.size())
            return;

        gwn_bvh4_node_soa<Real, Index> const &node = bvh.nodes[node_id];
        auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
        if (child_kind != gwn_bvh_child_kind::k_leaf)
            return;

        moment_type const leaf_moment = gwn_compute_leaf_taylor_moment_cached<Order>(
            geometry, bvh, node.child_index[child_slot], node.child_count[child_slot]
        );
        gwn_propagate_up(
            static_cast<Index>(node_id), static_cast<std::uint8_t>(child_slot), leaf_moment
        );
    }
};

template <class Index> struct gwn_validate_taylor_async_convergence_functor {
    cuda::std::span<std::uint8_t const> internal_arity{};
    cuda::std::span<unsigned int const> internal_arrivals{};
    unsigned int *error_flag = nullptr;

    __device__ void operator()(std::size_t const node_id) const {
        if (node_id >= internal_arity.size() || node_id >= internal_arrivals.size()) {
            if (error_flag != nullptr)
                atomicExch(error_flag, 1u);
            return;
        }

        unsigned int const expected_arrivals = static_cast<unsigned int>(internal_arity[node_id]);
        if (expected_arrivals == 0u || expected_arrivals > 4u ||
            internal_arrivals[node_id] != expected_arrivals) {
            if (error_flag != nullptr)
                atomicExch(error_flag, 1u);
        }
    }
};

} // namespace detail
} // namespace gwn
