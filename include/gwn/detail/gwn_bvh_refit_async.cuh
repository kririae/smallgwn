#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../gwn_kernel_utils.cuh"
#include "gwn_bvh_topology_build_common.cuh"

namespace gwn {
namespace detail {

template <int Order, gwn_real_type Real> struct gwn_device_taylor_moment;

template <gwn_real_type Real> struct gwn_device_taylor_moment<0, Real> {
    Real area = Real(0);
    Real area_p_x = Real(0);
    Real area_p_y = Real(0);
    Real area_p_z = Real(0);
    Real n_x = Real(0);
    Real n_y = Real(0);
    Real n_z = Real(0);
};

template <gwn_real_type Real>
struct gwn_device_taylor_moment<1, Real> : gwn_device_taylor_moment<0, Real> {
    Real nx_x = Real(0);
    Real nx_y = Real(0);
    Real nx_z = Real(0);
    Real ny_x = Real(0);
    Real ny_y = Real(0);
    Real ny_z = Real(0);
    Real nz_x = Real(0);
    Real nz_y = Real(0);
    Real nz_z = Real(0);
};

template <int Order, gwn_real_type Real>
__host__ __device__ inline void gwn_accumulate_taylor_moment(
    gwn_device_taylor_moment<Order, Real> &dst, gwn_device_taylor_moment<Order, Real> const &src
) noexcept {
    dst.area += src.area;
    dst.area_p_x += src.area_p_x;
    dst.area_p_y += src.area_p_y;
    dst.area_p_z += src.area_p_z;
    dst.n_x += src.n_x;
    dst.n_y += src.n_y;
    dst.n_z += src.n_z;
    if constexpr (Order >= 1) {
        dst.nx_x += src.nx_x;
        dst.nx_y += src.nx_y;
        dst.nx_z += src.nx_z;
        dst.ny_x += src.ny_x;
        dst.ny_y += src.ny_y;
        dst.ny_z += src.ny_z;
        dst.nz_x += src.nz_x;
        dst.nz_y += src.nz_y;
        dst.nz_z += src.nz_z;
    }
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ inline Real gwn_bounds_max_p_dist2(
    gwn_aabb<Real> const &bounds, Real const average_x, Real const average_y, Real const average_z
) noexcept {
    Real const dx = std::max(average_x - bounds.min_x, bounds.max_x - average_x);
    Real const dy = std::max(average_y - bounds.min_y, bounds.max_y - average_y);
    Real const dz = std::max(average_z - bounds.min_z, bounds.max_z - average_z);
    return dx * dx + dy * dy + dz * dz;
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_compute_triangle_aabb(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_aabb<Real> &bounds
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

    auto const a_index = static_cast<std::size_t>(ia);
    auto const b_index = static_cast<std::size_t>(ib);
    auto const c_index = static_cast<std::size_t>(ic);

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
    return true;
}

template <int Order, gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_compute_triangle_taylor_raw_moment(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_device_taylor_moment<Order, Real> &raw_moment
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

    auto const a_index = static_cast<std::size_t>(ia);
    auto const b_index = static_cast<std::size_t>(ib);
    auto const c_index = static_cast<std::size_t>(ic);

    Real const ax = geometry.vertex_x[a_index];
    Real const ay = geometry.vertex_y[a_index];
    Real const az = geometry.vertex_z[a_index];
    Real const bx = geometry.vertex_x[b_index];
    Real const by = geometry.vertex_y[b_index];
    Real const bz = geometry.vertex_z[b_index];
    Real const cx = geometry.vertex_x[c_index];
    Real const cy = geometry.vertex_y[c_index];
    Real const cz = geometry.vertex_z[c_index];

    Real const average_x = (ax + bx + cx) / Real(3);
    Real const average_y = (ay + by + cy) / Real(3);
    Real const average_z = (az + bz + cz) / Real(3);

    Real const abx = bx - ax;
    Real const aby = by - ay;
    Real const abz = bz - az;
    Real const acx = cx - ax;
    Real const acy = cy - ay;
    Real const acz = cz - az;

    Real const n_x = Real(0.5) * (aby * acz - abz * acy);
    Real const n_y = Real(0.5) * (abz * acx - abx * acz);
    Real const n_z = Real(0.5) * (abx * acy - aby * acx);

    Real const area2 = n_x * n_x + n_y * n_y + n_z * n_z;
    Real const area = sqrt(std::max(area2, Real(0)));

    raw_moment.area = area;
    raw_moment.area_p_x = average_x * area;
    raw_moment.area_p_y = average_y * area;
    raw_moment.area_p_z = average_z * area;
    raw_moment.n_x = n_x;
    raw_moment.n_y = n_y;
    raw_moment.n_z = n_z;
    if constexpr (Order >= 1) {
        raw_moment.nx_x = n_x * average_x;
        raw_moment.nx_y = n_x * average_y;
        raw_moment.nx_z = n_x * average_z;
        raw_moment.ny_x = n_y * average_x;
        raw_moment.ny_y = n_y * average_y;
        raw_moment.ny_z = n_y * average_z;
        raw_moment.nz_x = n_z * average_x;
        raw_moment.nz_y = n_z * average_y;
        raw_moment.nz_z = n_z * average_z;
    }
    return true;
}

template <int Width, gwn_index_type Index> struct gwn_refit_topology_arrays {
    cuda::std::span<Index> internal_parent{};
    cuda::std::span<std::uint8_t> internal_parent_slot{};
    cuda::std::span<std::uint8_t> internal_arity{};
    cuda::std::span<unsigned int> internal_arrivals{};
};

template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_prepare_refit_topology_functor {
    gwn_bvh_topology_accessor<Width, Real, Index> topology{};
    gwn_refit_topology_arrays<Width, Index> arrays{};
    unsigned int *error_flag = nullptr;

    __device__ inline void mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ void operator()(std::size_t const node_id) const {
        if (node_id >= topology.nodes.size() || node_id >= arrays.internal_parent.size() ||
            node_id >= arrays.internal_parent_slot.size() ||
            node_id >= arrays.internal_arity.size()) {
            mark_error();
            return;
        }

        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        std::uint8_t arity = 0;
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf) {
                mark_error();
                continue;
            }

            ++arity;
            if (kind == gwn_bvh_child_kind::k_internal) {
                Index const child_index = node.child_index[child_slot];
                if (!gwn_index_in_bounds(child_index, topology.nodes.size())) {
                    mark_error();
                    continue;
                }
                auto const child_id = static_cast<std::size_t>(child_index);
                arrays.internal_parent[child_id] = static_cast<Index>(node_id);
                arrays.internal_parent_slot[child_id] = static_cast<std::uint8_t>(child_slot);
                continue;
            }

            Index const leaf_begin = node.child_index[child_slot];
            Index const leaf_count = node.child_count[child_slot];
            if (gwn_is_invalid_index(leaf_begin) || gwn_is_invalid_index(leaf_count)) {
                mark_error();
                continue;
            }
            auto const begin = static_cast<std::size_t>(leaf_begin);
            auto const count = static_cast<std::size_t>(leaf_count);
            if (begin > topology.primitive_indices.size() ||
                count > (topology.primitive_indices.size() - begin)) {
                mark_error();
                continue;
            }
        }

        arrays.internal_arity[node_id] = arity;
        if (arity == 0)
            mark_error();
    }
};

template <int Width, gwn_real_type Real, gwn_index_type Index> struct gwn_aabb_refit_traits {
    using payload_type = gwn_aabb<Real>;

    struct output_context {
        cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real>> nodes{};
    };

    static constexpr char const *k_error_name =
        "AABB refit failed topology/propagation validation.";

    __device__ static bool make_leaf_payload(
        gwn_geometry_accessor<Real, Index> const &geometry,
        gwn_bvh_topology_accessor<Width, Real, Index> const &topology, std::size_t const node_id,
        int const child_slot, output_context const &, payload_type &payload
    ) noexcept {
        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        Index const begin = node.child_index[child_slot];
        Index const count = node.child_count[child_slot];

        bool has_bounds = false;
        payload = {};
        for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
            Index const sorted_slot = begin + primitive_offset;
            auto const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
            if (sorted_slot_u >= topology.primitive_indices.size())
                continue;

            Index const primitive_id = topology.primitive_indices[sorted_slot_u];
            gwn_aabb<Real> primitive_bounds{};
            if (!gwn_compute_triangle_aabb(geometry, primitive_id, primitive_bounds))
                continue;

            if (!has_bounds) {
                payload = primitive_bounds;
                has_bounds = true;
            } else {
                payload = gwn_aabb_union(payload, primitive_bounds);
            }
        }

        if (!has_bounds)
            payload = gwn_aabb<Real>{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
        return true;
    }

    __device__ static void combine(payload_type &dst, payload_type const &src) noexcept {
        dst = gwn_aabb_union(dst, src);
    }

    __device__ static void write_invalid(
        output_context const &context, std::size_t const node_id, int const child_slot
    ) noexcept {
        context.nodes[node_id].child_min_x[child_slot] = Real(0);
        context.nodes[node_id].child_min_y[child_slot] = Real(0);
        context.nodes[node_id].child_min_z[child_slot] = Real(0);
        context.nodes[node_id].child_max_x[child_slot] = Real(0);
        context.nodes[node_id].child_max_y[child_slot] = Real(0);
        context.nodes[node_id].child_max_z[child_slot] = Real(0);
    }

    __device__ static void write_valid(
        output_context const &context, std::size_t const node_id, int const child_slot,
        payload_type const &payload
    ) noexcept {
        context.nodes[node_id].child_min_x[child_slot] = payload.min_x;
        context.nodes[node_id].child_min_y[child_slot] = payload.min_y;
        context.nodes[node_id].child_min_z[child_slot] = payload.min_z;
        context.nodes[node_id].child_max_x[child_slot] = payload.max_x;
        context.nodes[node_id].child_max_y[child_slot] = payload.max_y;
        context.nodes[node_id].child_max_z[child_slot] = payload.max_z;
    }
};

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_moment_refit_traits {
    using payload_type = gwn_device_taylor_moment<Order, Real>;
    using output_node_type = gwn_bvh_taylor_node_soa<Width, Order, Real>;

    struct output_context {
        cuda::std::span<output_node_type> moment_nodes{};
        cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real> const> aabb_nodes{};
    };

    static constexpr char const *k_error_name =
        "Moment refit failed topology/propagation validation.";

    __device__ static bool make_leaf_payload(
        gwn_geometry_accessor<Real, Index> const &geometry,
        gwn_bvh_topology_accessor<Width, Real, Index> const &topology, std::size_t const node_id,
        int const child_slot, output_context const &, payload_type &payload
    ) noexcept {
        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        Index const begin = node.child_index[child_slot];
        Index const count = node.child_count[child_slot];

        payload = {};
        for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
            Index const sorted_slot = begin + primitive_offset;
            auto const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
            if (sorted_slot_u >= topology.primitive_indices.size())
                continue;

            Index const primitive_id = topology.primitive_indices[sorted_slot_u];
            payload_type primitive_payload{};
            if (!gwn_compute_triangle_taylor_raw_moment<Order>(
                    geometry, primitive_id, primitive_payload
                ))
                continue;
            gwn_accumulate_taylor_moment(payload, primitive_payload);
        }
        return true;
    }

    __device__ static void combine(payload_type &dst, payload_type const &src) noexcept {
        gwn_accumulate_taylor_moment(dst, src);
    }

    __device__ static void zero_child(output_node_type &node, int const child_slot) noexcept {
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

    __device__ static void write_invalid(
        output_context const &context, std::size_t const node_id, int const child_slot
    ) noexcept {
        zero_child(context.moment_nodes[node_id], child_slot);
    }

    __device__ static void write_valid(
        output_context const &context, std::size_t const node_id, int const child_slot,
        payload_type const &payload
    ) noexcept {
        output_node_type &node = context.moment_nodes[node_id];
        gwn_aabb<Real> const bounds{
            context.aabb_nodes[node_id].child_min_x[child_slot],
            context.aabb_nodes[node_id].child_min_y[child_slot],
            context.aabb_nodes[node_id].child_min_z[child_slot],
            context.aabb_nodes[node_id].child_max_x[child_slot],
            context.aabb_nodes[node_id].child_max_y[child_slot],
            context.aabb_nodes[node_id].child_max_z[child_slot],
        };

        Real average_x = Real(0);
        Real average_y = Real(0);
        Real average_z = Real(0);
        if (payload.area > Real(0)) {
            average_x = payload.area_p_x / payload.area;
            average_y = payload.area_p_y / payload.area;
            average_z = payload.area_p_z / payload.area;
        } else {
            average_x = (bounds.min_x + bounds.max_x) * Real(0.5);
            average_y = (bounds.min_y + bounds.max_y) * Real(0.5);
            average_z = (bounds.min_z + bounds.max_z) * Real(0.5);
        }

        node.child_max_p_dist2[child_slot] =
            gwn_bounds_max_p_dist2(bounds, average_x, average_y, average_z);
        node.child_average_x[child_slot] = average_x;
        node.child_average_y[child_slot] = average_y;
        node.child_average_z[child_slot] = average_z;
        node.child_n_x[child_slot] = payload.n_x;
        node.child_n_y[child_slot] = payload.n_y;
        node.child_n_z[child_slot] = payload.n_z;

        if constexpr (Order >= 1) {
            Real const nij_xx = payload.nx_x - average_x * payload.n_x;
            Real const nij_yy = payload.ny_y - average_y * payload.n_y;
            Real const nij_zz = payload.nz_z - average_z * payload.n_z;
            Real const nxy = payload.nx_y - average_y * payload.n_x;
            Real const nyx = payload.ny_x - average_x * payload.n_y;
            Real const nyz = payload.ny_z - average_z * payload.n_y;
            Real const nzy = payload.nz_y - average_y * payload.n_z;
            Real const nzx = payload.nz_x - average_x * payload.n_z;
            Real const nxz = payload.nx_z - average_z * payload.n_x;

            node.child_nij_xx[child_slot] = nij_xx;
            node.child_nij_yy[child_slot] = nij_yy;
            node.child_nij_zz[child_slot] = nij_zz;
            node.child_nxy_nyx[child_slot] = nxy + nyx;
            node.child_nyz_nzy[child_slot] = nyz + nzy;
            node.child_nzx_nxz[child_slot] = nzx + nxz;
        }
    }
};

template <int Width, gwn_index_type Index, class Payload> struct gwn_refit_pending_buffer {
    cuda::std::span<Payload> pending{};

    __device__ bool is_valid(std::size_t const node_id, int const child_slot) const noexcept {
        if (child_slot < 0 || child_slot >= Width)
            return false;
        if (node_id > (std::numeric_limits<std::size_t>::max() / std::size_t(Width)))
            return false;
        std::size_t const pending_index =
            node_id * std::size_t(Width) + static_cast<std::size_t>(child_slot);
        return pending_index < pending.size();
    }

    __device__ std::size_t index(std::size_t const node_id, int const child_slot) const noexcept {
        return node_id * std::size_t(Width) + static_cast<std::size_t>(child_slot);
    }
};

template <class Traits, int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_refit_from_leaves_functor {
    using payload_type = typename Traits::payload_type;
    using output_context = typename Traits::output_context;

    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> topology{};
    gwn_refit_topology_arrays<Width, Index> arrays{};
    gwn_refit_pending_buffer<Width, Index, payload_type> pending{};
    output_context context{};
    unsigned int *error_flag = nullptr;

    __device__ inline void mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ bool
    finalize_parent(std::size_t const node_id, payload_type &parent_payload) const noexcept {
        if (node_id >= topology.nodes.size())
            return false;

        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        bool has_child = false;
        bool initialized = false;
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf) {
                mark_error();
                continue;
            }
            if (!pending.is_valid(node_id, child_slot)) {
                mark_error();
                continue;
            }

            payload_type const child_payload = pending.pending[pending.index(node_id, child_slot)];
            if (!initialized) {
                parent_payload = child_payload;
                initialized = true;
            } else {
                Traits::combine(parent_payload, child_payload);
            }
            has_child = true;
        }
        return has_child;
    }

    __device__ void propagate_up(
        Index current_parent, std::uint8_t current_slot, payload_type current_payload
    ) const noexcept {
        while (gwn_is_valid_index(current_parent)) {
            auto const parent_id = static_cast<std::size_t>(current_parent);
            if (parent_id >= topology.nodes.size() || parent_id >= arrays.internal_parent.size() ||
                parent_id >= arrays.internal_parent_slot.size() ||
                parent_id >= arrays.internal_arity.size() ||
                parent_id >= arrays.internal_arrivals.size()) {
                mark_error();
                return;
            }
            if (current_slot >= Width ||
                !pending.is_valid(parent_id, static_cast<int>(current_slot))) {
                mark_error();
                return;
            }

            pending.pending[pending.index(parent_id, static_cast<int>(current_slot))] =
                current_payload;
            __threadfence();

            unsigned int const previous_arrivals =
                atomicAdd(arrays.internal_arrivals.data() + parent_id, 1u);
            unsigned int const next_arrivals = previous_arrivals + 1u;
            unsigned int const expected_arrivals =
                static_cast<unsigned int>(arrays.internal_arity[parent_id]);
            GWN_ASSERT(
                next_arrivals <= expected_arrivals,
                "refit: arrival count %u exceeds expected %u at node %zu",
                next_arrivals, expected_arrivals, parent_id
            );
            if (expected_arrivals == 0u || expected_arrivals > static_cast<unsigned int>(Width)) {
                mark_error();
                return;
            }
            if (next_arrivals < expected_arrivals)
                return;
            if (next_arrivals > expected_arrivals) {
                mark_error();
                return;
            }

            __threadfence();

            payload_type parent_payload{};
            if (!finalize_parent(parent_id, parent_payload)) {
                mark_error();
                return;
            }

            Index const parent_parent = arrays.internal_parent[parent_id];
            if (gwn_is_invalid_index(parent_parent))
                return;

            std::uint8_t const parent_parent_slot = arrays.internal_parent_slot[parent_id];
            if (parent_parent_slot >= Width) {
                mark_error();
                return;
            }

            current_parent = parent_parent;
            current_slot = parent_parent_slot;
            current_payload = parent_payload;
        }
    }

    __device__ void operator()(std::size_t const edge_index) const {
        if (edge_index > (std::numeric_limits<std::size_t>::max() / std::size_t(Width)))
            return;

        std::size_t const node_id = edge_index / std::size_t(Width);
        auto const child_slot = static_cast<int>(edge_index % std::size_t(Width));
        if (node_id >= topology.nodes.size())
            return;

        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
        if (child_kind != gwn_bvh_child_kind::k_leaf)
            return;

        payload_type leaf_payload{};
        if (!Traits::make_leaf_payload(
                geometry, topology, node_id, child_slot, context, leaf_payload
            )) {
            mark_error();
            return;
        }

        propagate_up(
            static_cast<Index>(node_id), static_cast<std::uint8_t>(child_slot), leaf_payload
        );
    }
};

template <int Width, gwn_index_type Index> struct gwn_validate_refit_convergence_functor {
    cuda::std::span<std::uint8_t const> internal_arity{};
    cuda::std::span<unsigned int const> internal_arrivals{};
    unsigned int *error_flag = nullptr;

    __device__ void operator()(std::size_t const node_id) const {
        if (node_id >= internal_arity.size() || node_id >= internal_arrivals.size()) {
            if (error_flag != nullptr)
                atomicExch(error_flag, 1u);
            return;
        }

        auto const expected_arrivals = static_cast<unsigned int>(internal_arity[node_id]);
        if (expected_arrivals == 0u || expected_arrivals > static_cast<unsigned int>(Width) ||
            internal_arrivals[node_id] != expected_arrivals) {
            if (error_flag != nullptr)
                atomicExch(error_flag, 1u);
        }
    }
};

template <class Traits, int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_finalize_refit_functor {
    using payload_type = typename Traits::payload_type;
    using output_context = typename Traits::output_context;

    gwn_bvh_topology_accessor<Width, Real, Index> topology{};
    gwn_refit_pending_buffer<Width, Index, payload_type> pending{};
    output_context context{};
    unsigned int *error_flag = nullptr;

    __device__ inline void mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ void operator()(std::size_t const node_id) const {
        if (node_id >= topology.nodes.size()) {
            mark_error();
            return;
        }

        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid) {
                Traits::write_invalid(context, node_id, child_slot);
                continue;
            }
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf) {
                Traits::write_invalid(context, node_id, child_slot);
                mark_error();
                continue;
            }
            if (!pending.is_valid(node_id, child_slot)) {
                Traits::write_invalid(context, node_id, child_slot);
                mark_error();
                continue;
            }
            payload_type const payload = pending.pending[pending.index(node_id, child_slot)];
            Traits::write_valid(context, node_id, child_slot, payload);
        }
    }
};

template <class Traits, int Width, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_run_refit_pass(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &topology,
    typename Traits::output_context const &output_context,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor is invalid for refit.");
    if (!topology.is_valid())
        return gwn_status::invalid_argument("Topology accessor is invalid for refit.");
    if (!topology.has_internal_root())
        return gwn_status::ok();

    constexpr int k_block_size = k_gwn_default_block_size;
    std::size_t const node_count = topology.nodes.size();
    if (node_count == 0)
        return gwn_status::ok();
    if (node_count > (std::numeric_limits<std::size_t>::max() / std::size_t(Width)))
        return gwn_status::internal_error("Refit node count overflow.");

    std::size_t const pending_count = node_count * std::size_t(Width);
    GWN_ASSERT(pending_count / std::size_t(Width) == node_count, "refit: pending_count overflow");
    using payload_type = typename Traits::payload_type;

    gwn_device_array<Index> internal_parent{};
    gwn_device_array<std::uint8_t> internal_parent_slot{};
    gwn_device_array<std::uint8_t> internal_arity{};
    gwn_device_array<unsigned int> internal_arrivals{};
    gwn_device_array<payload_type> pending_payloads{};
    gwn_device_array<unsigned int> error_flag_device{};

    GWN_RETURN_ON_ERROR(internal_parent.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(internal_parent_slot.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(internal_arity.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(internal_arrivals.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(pending_payloads.resize(pending_count, stream));
    GWN_RETURN_ON_ERROR(error_flag_device.resize(1, stream));

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(internal_parent.data(), 0xff, node_count * sizeof(Index), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        internal_parent_slot.data(), 0xff, node_count * sizeof(std::uint8_t), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(internal_arity.data(), 0, node_count * sizeof(std::uint8_t), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(internal_arrivals.data(), 0, node_count * sizeof(unsigned int), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(pending_payloads.data(), 0, pending_count * sizeof(payload_type), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(error_flag_device.data(), 0, sizeof(unsigned int), stream)
    ));

    gwn_refit_topology_arrays<Width, Index> const arrays{
        internal_parent.span(), internal_parent_slot.span(), internal_arity.span(),
        internal_arrivals.span()
    };
    gwn_refit_pending_buffer<Width, Index, payload_type> const pending{pending_payloads.span()};

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            node_count,
            gwn_prepare_refit_topology_functor<Width, Real, Index>{
                topology, arrays, error_flag_device.data()
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            pending_count,
            gwn_refit_from_leaves_functor<Traits, Width, Real, Index>{
                geometry, topology, arrays, pending, output_context, error_flag_device.data()
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            node_count,
            gwn_validate_refit_convergence_functor<Width, Index>{
                cuda::std::span<std::uint8_t const>(
                    arrays.internal_arity.data(), arrays.internal_arity.size()
                ),
                cuda::std::span<unsigned int const>(
                    arrays.internal_arrivals.data(), arrays.internal_arrivals.size()
                ),
                error_flag_device.data()
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            node_count,
            gwn_finalize_refit_functor<Traits, Width, Real, Index>{
                topology, pending, output_context, error_flag_device.data()
            },
            stream
        )
    );

    unsigned int host_error_flag = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_error_flag, error_flag_device.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_error_flag != 0)
        return gwn_status::internal_error(Traits::k_error_name);

    return gwn_status::ok();
}

} // namespace detail
} // namespace gwn
