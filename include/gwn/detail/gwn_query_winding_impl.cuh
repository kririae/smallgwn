#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_geometry_impl.cuh"
#include "gwn_query_winding_gradient_impl.cuh"

namespace gwn {
namespace detail {

template <int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline Real gwn_winding_number_point_bvh_exact_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh, Real const qx, Real const qy,
    Real const qz
) noexcept {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid() || !bvh.is_valid())
        return Real(0);

    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    Index stack[StackCapacity];
    int stack_size = 0;

    gwn_query_vec3<Real> const query(qx, qy, qz);
    Real omega_sum = Real(0);
    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        for (Index primitive_offset = 0; primitive_offset < bvh.root_count; ++primitive_offset) {
            Index const sorted_primitive_index = bvh.root_index + primitive_offset;
            if (!gwn_index_in_bounds(sorted_primitive_index, bvh.primitive_indices.size()))
                continue;
            Index const primitive_index =
                bvh.primitive_indices[static_cast<std::size_t>(sorted_primitive_index)];
            omega_sum += gwn_triangle_solid_angle_from_primitive_impl<Real, Index>(
                geometry, primitive_index, query
            );
        }
        return omega_sum / (Real(4) * k_pi);
    }

    if (bvh.root_kind != gwn_bvh_child_kind::k_internal)
        return Real(0);

    stack[stack_size++] = bvh.root_index;
    while (stack_size > 0) {
        Index const node_index = stack[--stack_size];
        GWN_ASSERT(stack_size >= 0, "winding exact: stack underflow");
        if (!gwn_index_in_bounds(node_index, bvh.nodes.size()))
            continue;

        gwn_bvh_topology_node_soa<Width, Index> const &node =
            bvh.nodes[static_cast<std::size_t>(node_index)];
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (child_kind == gwn_bvh_child_kind::k_invalid)
                continue;

            if (child_kind == gwn_bvh_child_kind::k_internal) {
                if (stack_size >= StackCapacity)
                    gwn_trap();
                stack[stack_size++] = node.child_index[child_slot];
                continue;
            }

            if (child_kind != gwn_bvh_child_kind::k_leaf)
                continue;

            Index const begin = node.child_index[child_slot];
            Index const count = node.child_count[child_slot];
            for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
                Index const sorted_primitive_index = begin + primitive_offset;
                if (!gwn_index_in_bounds(sorted_primitive_index, bvh.primitive_indices.size()))
                    continue;
                Index const primitive_index =
                    bvh.primitive_indices[static_cast<std::size_t>(sorted_primitive_index)];
                omega_sum += gwn_triangle_solid_angle_from_primitive_impl<Real, Index>(
                    geometry, primitive_index, query
                );
            }
        }
    }

    return omega_sum / (Real(4) * k_pi);
}

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline Real gwn_winding_number_point_bvh_taylor_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree, Real const qx,
    Real const qy, Real const qz, Real const accuracy_scale
) noexcept {
    return gwn_winding_and_gradient_point_bvh_taylor_impl<
        Order, Width, Real, Index, StackCapacity>(
        geometry, bvh, data_tree, qx, qy, qz, accuracy_scale
    ).winding;
}

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
struct gwn_winding_number_batch_bvh_taylor_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> bvh{};
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> data_tree{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> output{};
    Real accuracy_scale{};

    __device__ void operator()(std::size_t const query_id) const {
        output[query_id] =
            gwn_winding_number_point_bvh_taylor_impl<Order, Width, Real, Index, StackCapacity>(
                geometry, bvh, data_tree, query_x[query_id], query_y[query_id], query_z[query_id],
                accuracy_scale
            );
    }
};

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
[[nodiscard]] inline gwn_winding_number_batch_bvh_taylor_functor<
    Order, Width, Real, Index, StackCapacity>
gwn_make_winding_number_batch_bvh_taylor_functor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale
) {
    return gwn_winding_number_batch_bvh_taylor_functor<Order, Width, Real, Index, StackCapacity>{
        geometry, bvh, data_tree, query_x, query_y, query_z, output, accuracy_scale
    };
}

} // namespace detail
} // namespace gwn
