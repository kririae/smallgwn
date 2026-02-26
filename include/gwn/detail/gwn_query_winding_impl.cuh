#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_geometry_impl.cuh"

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
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_winding_number_point_bvh_taylor currently supports Order 0, 1, and 2."
    );
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid() || !bvh.is_valid())
        return Real(0);
    if (!data_tree.is_valid_for(bvh))
        return Real(0);

    auto const &taylor_nodes = data_tree.nodes;
    GWN_ASSERT(!taylor_nodes.empty(), "winding taylor: taylor_nodes empty for internal root");
    if (taylor_nodes.size() != bvh.nodes.size())
        return Real(0);

    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    Index stack[StackCapacity];
    int stack_size = 0;

    Real omega_sum = Real(0);
    Real const accuracy_scale2 = accuracy_scale * accuracy_scale;
    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        for (Index primitive_offset = 0; primitive_offset < bvh.root_count; ++primitive_offset) {
            Index const sorted_primitive_index = bvh.root_index + primitive_offset;
            if (!gwn_index_in_bounds(sorted_primitive_index, bvh.primitive_indices.size()))
                continue;
            Index const primitive_index =
                bvh.primitive_indices[static_cast<std::size_t>(sorted_primitive_index)];
            omega_sum += gwn_triangle_solid_angle_from_primitive_impl<Real, Index>(
                geometry, primitive_index, gwn_query_vec3<Real>(qx, qy, qz)
            );
        }
        return omega_sum / (Real(4) * k_pi);
    }

    if (bvh.root_kind != gwn_bvh_child_kind::k_internal)
        return Real(0);

    stack[stack_size++] = bvh.root_index;
    while (stack_size > 0) {
        Index const node_index = stack[--stack_size];
        GWN_ASSERT(stack_size >= 0, "winding taylor: stack underflow");
        if (!gwn_index_in_bounds(node_index, bvh.nodes.size()))
            continue;

        gwn_bvh_topology_node_soa<Width, Index> const &node =
            bvh.nodes[static_cast<std::size_t>(node_index)];
        auto const &taylor = taylor_nodes[static_cast<std::size_t>(node_index)];
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (child_kind == gwn_bvh_child_kind::k_invalid)
                continue;

            if (child_kind != gwn_bvh_child_kind::k_internal &&
                child_kind != gwn_bvh_child_kind::k_leaf) {
                continue;
            }

            Real const qrx = qx - taylor.child_average_x[child_slot];
            Real const qry = qy - taylor.child_average_y[child_slot];
            Real const qrz = qz - taylor.child_average_z[child_slot];
            Real const qlength2 = qrx * qrx + qry * qry + qrz * qrz;

            bool descend = !(qlength2 > Real(0));
            if (!descend)
                descend = qlength2 <= taylor.child_max_p_dist2[child_slot] * accuracy_scale2;

            if (!descend) {
                Real qlength_m1 = Real(1) / sqrt(qlength2);
                if constexpr (std::is_same_v<Real, float>)
                    qlength_m1 = rsqrtf(qlength2);
                Real const qlength_m2 = qlength_m1 * qlength_m1;

                Real const qnx = qrx * qlength_m1;
                Real const qny = qry * qlength_m1;
                Real const qnz = qrz * qlength_m1;
                Real omega_approx = -qlength_m2 * (qnx * taylor.child_n_x[child_slot] +
                                                   qny * taylor.child_n_y[child_slot] +
                                                   qnz * taylor.child_n_z[child_slot]);

                if constexpr (Order >= 1) {
                    Real const qxx = qnx * qnx;
                    Real const qyy = qny * qny;
                    Real const qzz = qnz * qnz;
                    Real const omega_1 =
                        qlength_m2 * qlength_m1 *
                        ((taylor.child_nij_xx[child_slot] + taylor.child_nij_yy[child_slot] +
                          taylor.child_nij_zz[child_slot]) -
                         Real(3) * (qxx * taylor.child_nij_xx[child_slot] +
                                    qyy * taylor.child_nij_yy[child_slot] +
                                    qzz * taylor.child_nij_zz[child_slot] +
                                    qnx * qny * taylor.child_nxy_nyx[child_slot] +
                                    qnx * qnz * taylor.child_nzx_nxz[child_slot] +
                                    qny * qnz * taylor.child_nyz_nzy[child_slot]));
                    omega_approx += omega_1;
                }

                if constexpr (Order >= 2) {
                    Real const qnx2 = qnx * qnx;
                    Real const qny2 = qny * qny;
                    Real const qnz2 = qnz * qnz;
                    Real const qnx3 = qnx2 * qnx;
                    Real const qny3 = qny2 * qny;
                    Real const qnz3 = qnz2 * qnz;
                    Real const qlength_m4 = qlength_m2 * qlength_m2;

                    Real const nijk_xxx = taylor.child_nijk_xxx[child_slot];
                    Real const nijk_yyy = taylor.child_nijk_yyy[child_slot];
                    Real const nijk_zzz = taylor.child_nijk_zzz[child_slot];

                    Real const temp0_x =
                        taylor.child_2nyyx_nxyy[child_slot] + taylor.child_2nzzx_nxzz[child_slot];
                    Real const temp0_y =
                        taylor.child_2nzzy_nyzz[child_slot] + taylor.child_2nxxy_nyxx[child_slot];
                    Real const temp0_z =
                        taylor.child_2nxxz_nzxx[child_slot] + taylor.child_2nyyz_nzyy[child_slot];

                    Real const temp1_x = qny * taylor.child_2nxxy_nyxx[child_slot] +
                                         qnz * taylor.child_2nxxz_nzxx[child_slot];
                    Real const temp1_y = qnz * taylor.child_2nyyz_nzyy[child_slot] +
                                         qnx * taylor.child_2nyyx_nxyy[child_slot];
                    Real const temp1_z = qnx * taylor.child_2nzzx_nxzz[child_slot] +
                                         qny * taylor.child_2nzzy_nyzz[child_slot];

                    Real const omega_2 =
                        qlength_m4 *
                        (Real(1.5) * (qnx * (Real(3) * nijk_xxx + temp0_x) +
                                      qny * (Real(3) * nijk_yyy + temp0_y) +
                                      qnz * (Real(3) * nijk_zzz + temp0_z)) -
                         Real(7.5) * (qnx3 * nijk_xxx + qny3 * nijk_yyy + qnz3 * nijk_zzz +
                                      qnx * qny * qnz * taylor.child_sum_permute_nxyz[child_slot] +
                                      qnx2 * temp1_x + qny2 * temp1_y + qnz2 * temp1_z));
                    omega_approx += omega_2;
                }

                if (isfinite(omega_approx)) {
                    omega_sum += omega_approx;
                    continue;
                }
                descend = true;
            }

            if (child_kind == gwn_bvh_child_kind::k_internal) {
                if (stack_size >= StackCapacity)
                    gwn_trap();
                stack[stack_size++] = node.child_index[child_slot];
                continue;
            }

            Index const begin = node.child_index[child_slot];
            Index const count = node.child_count[child_slot];
            gwn_query_vec3<Real> const query(qx, qy, qz);
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
