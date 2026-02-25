#pragma once

/// \file gwn_query.cuh
/// \brief Public spatial query APIs for triangle meshes on the GPU.
///
/// Public query surface:
/// - Batch winding-number query:
///   `gwn_compute_winding_number_batch_bvh_taylor`.
/// - Single-point distance queries (`__device__`):
///   `gwn_unsigned_distance_point_bvh` and `gwn_signed_distance_point_bvh`.
///
/// Query implementation details live in `include/gwn/detail/gwn_query_*_impl.cuh`.

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "detail/gwn_query_distance_impl.cuh"
#include "detail/gwn_query_geometry_impl.cuh"
#include "detail/gwn_query_winding_impl.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {

/// \brief 3-component column vector parameterised on scalar type \p Real.
template <gwn_real_type Real> using gwn_vec3 = detail::gwn_query_vec3<Real>;

/// \brief Default capacity of the per-thread BVH traversal stack.
inline constexpr int k_gwn_default_traversal_stack_capacity = 64;

/// \brief Compute the unsigned distance from a query point to the closest
///        triangle using BVH + AABB-accelerated traversal.
///
/// \param culling_band World-unit narrow-band distance cap. Distances larger
///        than this value are clamped to this value. Use \c +infinity to
///        disable culling (default).
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
__device__ inline Real gwn_unsigned_distance_point_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, Real const qx, Real const qy,
    Real const qz, Real const culling_band = std::numeric_limits<Real>::infinity()
) noexcept {
    return detail::gwn_unsigned_distance_point_bvh_impl<Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, qx, qy, qz, culling_band
    );
}

/// \brief Compute the signed distance from a query point to the mesh using
///        BVH-accelerated traversal and Taylor winding-number sign inference.
///
/// \tparam Order Taylor winding-number order (currently 0 or 1).
/// \param data_tree Taylor moment payload tree aligned to \p bvh.
/// \param winding_number_threshold Inside/outside threshold applied to Taylor
///        winding number (default: \c 0.5).
/// \param culling_band World-unit narrow-band distance cap. Distances larger
///        than this value are clamped to this value before applying sign. Use
///        \c +infinity to disable culling (default).
/// \param accuracy_scale Taylor far-field acceptance scale (default: \c 2).
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
__device__ inline Real gwn_signed_distance_point_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree, Real const qx,
    Real const qy, Real const qz, Real const winding_number_threshold = Real(0.5),
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    Real const accuracy_scale = Real(2)
) noexcept {
    static_assert(
        Order == 0 || Order == 1,
        "gwn_signed_distance_point_bvh currently supports Order 0 and Order 1."
    );

    return detail::gwn_signed_distance_point_bvh_impl<Order, Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, data_tree, qx, qy, qz, winding_number_threshold, culling_band,
        accuracy_scale
    );
}

/// \brief Compute winding numbers for a batch using Taylor-accelerated BVH
///        traversal.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_winding_number_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream()
) noexcept {
    static_assert(
        Order == 0 || Order == 1,
        "gwn_compute_winding_number_batch_bvh_taylor currently supports Order 0 and Order 1."
    );
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
    if (!data_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH data tree is invalid for the given topology.");
    if (!data_tree.template has_taylor_order<Order>()) {
        return gwn_status::invalid_argument(
            "BVH data tree is missing requested Taylor-order data."
        );
    }
    if (query_x.size() != query_y.size() || query_x.size() != query_z.size())
        return gwn_status::invalid_argument("Query SoA spans must have identical lengths.");
    if (query_x.size() != output.size())
        return gwn_status::invalid_argument("Output span size must match query count.");
    if (!gwn_span_has_storage(query_x) || !gwn_span_has_storage(query_y) ||
        !gwn_span_has_storage(query_z) || !gwn_span_has_storage(output)) {
        return gwn_status::invalid_argument(
            "Query/output spans must use non-null storage when non-empty."
        );
    }
    if (output.empty())
        return gwn_status::ok();

    constexpr int k_block_size = detail::k_gwn_default_block_size;
    return detail::gwn_launch_linear_kernel<k_block_size>(
        output.size(),
        detail::gwn_make_winding_number_batch_bvh_taylor_functor<
            Order, Width, Real, Index, StackCapacity>(
            geometry, bvh, data_tree, query_x, query_y, query_z, output, accuracy_scale
        ),
        stream
    );
}

/// \brief Width-4 convenience wrapper for Taylor BVH batch winding-number
///        queries.
///
/// \copydetails gwn_compute_winding_number_batch_bvh_taylor
template <
    int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_winding_number_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_moment_accessor<Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, StackCapacity>(
        geometry, bvh, data_tree, query_x, query_y, query_z, output, accuracy_scale, stream
    );
}

} // namespace gwn
