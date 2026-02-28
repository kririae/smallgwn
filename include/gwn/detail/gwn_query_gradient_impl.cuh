#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_geometry_impl.cuh"
#include "gwn_query_vec3_impl.cuh"
#include "gwn_query_winding_gradient_impl.cuh"

namespace gwn {
namespace detail {

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline gwn_query_vec3<Real> gwn_winding_gradient_point_exact_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Real const qx, Real const qy, Real const qz
) noexcept {
    gwn_query_vec3<Real> const zero(Real(0), Real(0), Real(0));
    if (!geometry.is_valid())
        return zero;

    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    constexpr Real k_inv_4pi = Real(1) / (Real(4) * k_pi);

    gwn_query_vec3<Real> const query(qx, qy, qz);
    gwn_query_vec3<Real> grad_sum(Real(0), Real(0), Real(0));
    for (std::size_t primitive_id = 0; primitive_id < geometry.triangle_count(); ++primitive_id) {
        grad_sum += gwn_triangle_gradient_from_primitive_impl<Real, Index>(
            geometry, static_cast<Index>(primitive_id), query
        );
    }

    return gwn_query_vec3<Real>(
        grad_sum.x * k_inv_4pi, grad_sum.y * k_inv_4pi, grad_sum.z * k_inv_4pi
    );
}

// BVH-accelerated Taylor gradient of the winding number.
//
// Mirrors gwn_winding_number_point_bvh_taylor_impl but accumulates a vec3.
// Uses the SAME precomputed moments, only the kernel derivatives increase
// by one order.

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline gwn_query_vec3<Real> gwn_winding_gradient_point_bvh_taylor_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree, Real const qx,
    Real const qy, Real const qz, Real const accuracy_scale
) noexcept {
    return gwn_winding_and_gradient_point_bvh_taylor_impl<Order, Width, Real, Index, StackCapacity>(
               geometry, bvh, data_tree, qx, qy, qz, accuracy_scale
    )
        .gradient;
}

// Batch functor

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
struct gwn_winding_gradient_batch_bvh_taylor_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> bvh{};
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> data_tree{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> output_x{};
    cuda::std::span<Real> output_y{};
    cuda::std::span<Real> output_z{};
    Real accuracy_scale{};

    __device__ void operator()(std::size_t const query_id) const {
        auto const grad =
            gwn_winding_gradient_point_bvh_taylor_impl<Order, Width, Real, Index, StackCapacity>(
                geometry, bvh, data_tree, query_x[query_id], query_y[query_id], query_z[query_id],
                accuracy_scale
            );
        output_x[query_id] = grad.x;
        output_y[query_id] = grad.y;
        output_z[query_id] = grad.z;
    }
};

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
[[nodiscard]] inline gwn_winding_gradient_batch_bvh_taylor_functor<
    Order, Width, Real, Index, StackCapacity>
gwn_make_winding_gradient_batch_bvh_taylor_functor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output_x,
    cuda::std::span<Real> const output_y, cuda::std::span<Real> const output_z,
    Real const accuracy_scale
) {
    return gwn_winding_gradient_batch_bvh_taylor_functor<Order, Width, Real, Index, StackCapacity>{
        geometry, bvh,      data_tree, query_x,  query_y,
        query_z,  output_x, output_y,  output_z, accuracy_scale
    };
}

} // namespace detail
} // namespace gwn
