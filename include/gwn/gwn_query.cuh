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

#include "detail/gwn_harnack_trace_impl.cuh"
#include "detail/gwn_query_distance_impl.cuh"
#include "detail/gwn_query_geometry_impl.cuh"
#include "detail/gwn_query_gradient_impl.cuh"
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
/// \tparam Order Taylor winding-number order (0, 1, or 2).
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
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree, Real const qx,
    Real const qy, Real const qz, Real const winding_number_threshold = Real(0.5),
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    Real const accuracy_scale = Real(2)
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_signed_distance_point_bvh currently supports Order 0, 1, and 2."
    );

    return detail::gwn_signed_distance_point_bvh_impl<Order, Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, data_tree, qx, qy, qz, winding_number_threshold, culling_band,
        accuracy_scale
    );
}

/// \brief Compute point-to-nearest-triangle-edge distances for a batch using
///        BVH + AABB-accelerated traversal.
template <
    int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_unsigned_edge_distance_batch_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
    if (!aabb_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH AABB tree is invalid for the given topology.");
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
        detail::gwn_make_unsigned_edge_distance_batch_bvh_functor<
            Width, Real, Index, StackCapacity>(
            geometry, bvh, aabb_tree, query_x, query_y, query_z, output, culling_band
        ),
        stream
    );
}

/// \brief Width-4 convenience wrapper for batch edge-distance queries.
template <
    gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_unsigned_edge_distance_batch_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_aabb_accessor<Real, Index> const &aabb_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_compute_unsigned_edge_distance_batch_bvh<4, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, query_x, query_y, query_z, output, culling_band, stream
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
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream()
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_compute_winding_number_batch_bvh_taylor currently supports Order 0, 1, and 2."
    );
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
    if (!data_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH data tree is invalid for the given topology.");
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
    gwn_bvh4_moment_accessor<Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, StackCapacity>(
        geometry, bvh, data_tree, query_x, query_y, query_z, output, accuracy_scale, stream
    );
}

/// \brief Compute winding-number gradient for a batch using
///        Taylor-accelerated BVH traversal.
///
/// Each query yields a 3-component gradient vector; the components are
/// written to three separate output spans (SoA layout).
///
/// \tparam Order Taylor winding-number order (0, 1, or 2).
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_winding_gradient_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output_x,
    cuda::std::span<Real> const output_y, cuda::std::span<Real> const output_z,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream()
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_compute_winding_gradient_batch_bvh_taylor currently supports Order 0, 1, and 2."
    );
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
    if (!data_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH data tree is invalid for the given topology.");
    if (query_x.size() != query_y.size() || query_x.size() != query_z.size())
        return gwn_status::invalid_argument("Query SoA spans must have identical lengths.");
    if (query_x.size() != output_x.size() || query_x.size() != output_y.size() ||
        query_x.size() != output_z.size())
        return gwn_status::invalid_argument("Output span sizes must match query count.");
    if (!gwn_span_has_storage(query_x) || !gwn_span_has_storage(query_y) ||
        !gwn_span_has_storage(query_z) || !gwn_span_has_storage(output_x) ||
        !gwn_span_has_storage(output_y) || !gwn_span_has_storage(output_z)) {
        return gwn_status::invalid_argument(
            "Query/output spans must use non-null storage when non-empty."
        );
    }
    if (output_x.empty())
        return gwn_status::ok();

    constexpr int k_block_size = detail::k_gwn_default_block_size;
    return detail::gwn_launch_linear_kernel<k_block_size>(
        output_x.size(),
        detail::gwn_make_winding_gradient_batch_bvh_taylor_functor<
            Order, Width, Real, Index, StackCapacity>(
            geometry, bvh, data_tree, query_x, query_y, query_z, output_x, output_y, output_z,
            accuracy_scale
        ),
        stream
    );
}

/// \brief Width-4 convenience wrapper for Taylor BVH batch winding-gradient
///        queries.
template <
    int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_winding_gradient_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_moment_accessor<Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output_x,
    cuda::std::span<Real> const output_y, cuda::std::span<Real> const output_z,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_compute_winding_gradient_batch_bvh_taylor<Order, 4, Real, Index, StackCapacity>(
        geometry, bvh, data_tree, query_x, query_y, query_z, output_x, output_y, output_z,
        accuracy_scale, stream
    );
}

// ---------------------------------------------------------------------------
// Harnack tracing API
// ---------------------------------------------------------------------------

/// \brief Harnack trace result type.
template <gwn_real_type Real>
using gwn_harnack_trace_result = detail::gwn_harnack_trace_result<Real>;

/// \brief Trace a single ray through the winding-number level set using the
///        Harnack inequality for guaranteed step safety (\c __device__ only).
///
/// \tparam Order Taylor winding-number order (0, 1, or 2).
/// \return \ref gwn_harnack_trace_result with hit parameter, normal, etc.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
__device__ inline gwn_harnack_trace_result<Real> gwn_harnack_trace_ray_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree,
    Real const ray_ox, Real const ray_oy, Real const ray_oz,
    Real const ray_dx, Real const ray_dy, Real const ray_dz,
    Real const target_winding = Real(0.5),
    Real const epsilon = Real(1e-4),
    int const max_iterations = 512,
    Real const t_max = Real(1e6),
    Real const accuracy_scale = Real(2)
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_harnack_trace_ray_bvh_taylor currently supports Order 0, 1, and 2."
    );
    return detail::gwn_harnack_trace_ray_impl<Order, Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, moment_tree,
        ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
        target_winding, epsilon, max_iterations, t_max, accuracy_scale
    );
}

/// \brief Batch Harnack trace: trace N rays in parallel through the
///        winding-number level set.
///
/// Each output span must have the same size as the ray-origin spans.
/// On success, \c output_t[i] is the hit parameter for ray \c i (negative
/// if no hit), and \c output_normal_{x,y,z}[i] are the surface normal.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_harnack_trace_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree,
    cuda::std::span<Real const> const ray_origin_x,
    cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z,
    cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y,
    cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t,
    cuda::std::span<Real> const output_normal_x,
    cuda::std::span<Real> const output_normal_y,
    cuda::std::span<Real> const output_normal_z,
    Real const target_winding = Real(0.5),
    Real const epsilon = Real(1e-4),
    int const max_iterations = 512,
    Real const t_max = Real(1e6),
    Real const accuracy_scale = Real(2),
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_compute_harnack_trace_batch_bvh_taylor currently supports Order 0, 1, and 2."
    );
    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
    if (!aabb_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH AABB tree is invalid for the given topology.");
    if (!moment_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH data tree is invalid for the given topology.");

    std::size_t const n = ray_origin_x.size();
    if (ray_origin_y.size() != n || ray_origin_z.size() != n ||
        ray_dir_x.size() != n || ray_dir_y.size() != n || ray_dir_z.size() != n ||
        output_t.size() != n || output_normal_x.size() != n ||
        output_normal_y.size() != n || output_normal_z.size() != n) {
        return gwn_status::invalid_argument("harnack trace: mismatched span sizes");
    }
    if (!gwn_span_has_storage(ray_origin_x) || !gwn_span_has_storage(ray_origin_y) ||
        !gwn_span_has_storage(ray_origin_z) || !gwn_span_has_storage(ray_dir_x) ||
        !gwn_span_has_storage(ray_dir_y) || !gwn_span_has_storage(ray_dir_z) ||
        !gwn_span_has_storage(output_t) || !gwn_span_has_storage(output_normal_x) ||
        !gwn_span_has_storage(output_normal_y) || !gwn_span_has_storage(output_normal_z)) {
        return gwn_status::invalid_argument(
            "Ray/output spans must use non-null storage when non-empty."
        );
    }
    if (n == 0)
        return gwn_status::ok();

    detail::gwn_harnack_trace_batch_functor<Order, Width, Real, Index, StackCapacity> functor{};
    functor.geometry = geometry;
    functor.bvh = bvh;
    functor.aabb_tree = aabb_tree;
    functor.moment_tree = moment_tree;
    functor.ray_origin_x = ray_origin_x;
    functor.ray_origin_y = ray_origin_y;
    functor.ray_origin_z = ray_origin_z;
    functor.ray_dir_x = ray_dir_x;
    functor.ray_dir_y = ray_dir_y;
    functor.ray_dir_z = ray_dir_z;
    functor.output_t = output_t;
    functor.output_normal_x = output_normal_x;
    functor.output_normal_y = output_normal_y;
    functor.output_normal_z = output_normal_z;
    functor.target_winding = target_winding;
    functor.epsilon = epsilon;
    functor.max_iterations = max_iterations;
    functor.t_max = t_max;
    functor.accuracy_scale = accuracy_scale;

    constexpr int k_block_size = detail::k_gwn_default_block_size;
    return detail::gwn_launch_linear_kernel<k_block_size>(n, functor, stream);
}

/// \brief Width-4 convenience wrapper for batch Harnack trace.
template <
    int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_harnack_trace_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_aabb_accessor<Real, Index> const &aabb_tree,
    gwn_bvh4_moment_accessor<Order, Real, Index> const &moment_tree,
    cuda::std::span<Real const> const ray_origin_x,
    cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z,
    cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y,
    cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t,
    cuda::std::span<Real> const output_normal_x,
    cuda::std::span<Real> const output_normal_y,
    cuda::std::span<Real> const output_normal_z,
    Real const target_winding = Real(0.5),
    Real const epsilon = Real(1e-4),
    int const max_iterations = 512,
    Real const t_max = Real(1e6),
    Real const accuracy_scale = Real(2),
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_compute_harnack_trace_batch_bvh_taylor<Order, 4, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, moment_tree,
        ray_origin_x, ray_origin_y, ray_origin_z,
        ray_dir_x, ray_dir_y, ray_dir_z,
        output_t, output_normal_x, output_normal_y, output_normal_z,
        target_winding, epsilon, max_iterations, t_max, accuracy_scale, stream
    );
}

/// \brief Trace a single ray using the angle-valued (mod 4π) Harnack tracer
///        with edge-distance safe balls (\c __device__ only).
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
__device__ inline gwn_harnack_trace_result<Real> gwn_harnack_trace_angle_ray_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree,
    Real const ray_ox, Real const ray_oy, Real const ray_oz,
    Real const ray_dx, Real const ray_dy, Real const ray_dz,
    Real const target_winding = Real(0.5),
    Real const epsilon = Real(1e-4),
    int const max_iterations = 512,
    Real const t_max = Real(1e6),
    Real const accuracy_scale = Real(2)
) noexcept {
    return gwn_harnack_trace_ray_bvh_taylor<Order, Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, moment_tree,
        ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
        target_winding, epsilon, max_iterations, t_max, accuracy_scale
    );
}

/// \brief Batch angle-valued Harnack trace:
///        traces rays against the mod-4π winding field using edge distance.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_harnack_trace_angle_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree,
    cuda::std::span<Real const> const ray_origin_x,
    cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z,
    cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y,
    cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t,
    cuda::std::span<Real> const output_normal_x,
    cuda::std::span<Real> const output_normal_y,
    cuda::std::span<Real> const output_normal_z,
    Real const target_winding = Real(0.5),
    Real const epsilon = Real(1e-4),
    int const max_iterations = 512,
    Real const t_max = Real(1e6),
    Real const accuracy_scale = Real(2),
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_compute_harnack_trace_batch_bvh_taylor<Order, Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, moment_tree,
        ray_origin_x, ray_origin_y, ray_origin_z,
        ray_dir_x, ray_dir_y, ray_dir_z,
        output_t, output_normal_x, output_normal_y, output_normal_z,
        target_winding, epsilon, max_iterations, t_max, accuracy_scale, stream
    );
}

/// \brief Width-4 convenience wrapper for batch angle-valued Harnack trace.
template <
    int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_harnack_trace_angle_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_aabb_accessor<Real, Index> const &aabb_tree,
    gwn_bvh4_moment_accessor<Order, Real, Index> const &moment_tree,
    cuda::std::span<Real const> const ray_origin_x,
    cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z,
    cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y,
    cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t,
    cuda::std::span<Real> const output_normal_x,
    cuda::std::span<Real> const output_normal_y,
    cuda::std::span<Real> const output_normal_z,
    Real const target_winding = Real(0.5),
    Real const epsilon = Real(1e-4),
    int const max_iterations = 512,
    Real const t_max = Real(1e6),
    Real const accuracy_scale = Real(2),
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_compute_harnack_trace_angle_batch_bvh_taylor<
        Order, 4, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, moment_tree,
        ray_origin_x, ray_origin_y, ray_origin_z,
        ray_dir_x, ray_dir_y, ray_dir_z,
        output_t, output_normal_x, output_normal_y, output_normal_z,
        target_winding, epsilon, max_iterations, t_max, accuracy_scale, stream
    );
}

} // namespace gwn
