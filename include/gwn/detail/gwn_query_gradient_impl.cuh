#pragma once

#include <cstddef>

#include "../gwn_bvh.cuh"
#include "gwn_query_winding_impl.cuh"

namespace gwn {
namespace detail {

/// \brief Compute Taylor winding gradient through the canonical BVH.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline gwn_query_vec3<Real> gwn_winding_gradient_taylor_impl(
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_accessor<Width, Order, Real, Index> const &moment, Real const qx, Real const qy,
    Real const qz, Real const accuracy_scale, OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_winding_taylor_impl<
               true, Order, Width, Real, Index, StackCapacity, OverflowCallback>(
               bvh, moment, qx, qy, qz, accuracy_scale, overflow_callback
    )
        .gradient;
}

/// \brief Invoke canonical Taylor gradient traversal for one batch element.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
struct gwn_winding_gradient_taylor_batch_functor {
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    gwn_bvh_moment_accessor<Width, Order, Real, Index> moment{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_grad_x{};
    cuda::std::span<Real> out_grad_y{};
    cuda::std::span<Real> out_grad_z{};
    Real accuracy_scale{};
    OverflowCallback overflow_callback{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        auto const grad = gwn_winding_gradient_taylor_impl<
            Order, Width, Real, Index, StackCapacity, OverflowCallback>(
            bvh, moment, query_x[query_id], query_y[query_id], query_z[query_id], accuracy_scale,
            overflow_callback
        );
        out_grad_x[query_id] = grad.x;
        out_grad_y[query_id] = grad.y;
        out_grad_z[query_id] = grad.z;
    }
};

} // namespace detail
} // namespace gwn
