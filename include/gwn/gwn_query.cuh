#pragma once

/// \file gwn_query.cuh
/// \brief Public query APIs for winding, gradients, distances, ray hits, Harnack tracing, and
///        hybrid tracing.
///
/// Query operations are available in two forms:
/// - Device point APIs (`__device__` functions callable from user kernels)
/// - Batch APIs (host-callable launchers that process arrays of queries)
///
/// Most query families provide both point and batch variants for symmetry and flexibility.
/// Implementation details live in `include/gwn/detail/gwn_query_*_impl.cuh`.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "detail/gwn_harnack_trace_impl.cuh"
#include "detail/gwn_query_common_impl.cuh"
#include "detail/gwn_query_distance_impl.cuh"
#include "detail/gwn_query_geometry_impl.cuh"
#include "detail/gwn_query_gradient_impl.cuh"
#include "detail/gwn_query_hybrid_impl.cuh"
#include "detail/gwn_query_ray_impl.cuh"
#include "detail/gwn_query_winding_impl.cuh"
#include "detail/gwn_scene_query_impl.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"
#include "gwn_scene.cuh"

namespace gwn {

/// \brief 3-component column vector parameterised on scalar type \p Real.
template <gwn_real_type Real> using gwn_vec3 = detail::gwn_query_vec3<Real>;

/// \brief Default capacity of the per-thread BVH traversal stack.
inline constexpr int k_gwn_default_traversal_stack_capacity = 64;

namespace detail {

template <gwn_real_type Real, gwn_index_type Index>
inline gwn_status gwn_validate_ray_first_hit_batch_spans(
    cuda::std::span<Real const> const ray_origin_x, cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z, cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y, cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t, cuda::std::span<Index> const output_primitive_id
) noexcept {
    std::size_t const n = ray_origin_x.size();
    if (ray_origin_y.size() != n || ray_origin_z.size() != n || ray_dir_x.size() != n ||
        ray_dir_y.size() != n || ray_dir_z.size() != n || output_t.size() != n ||
        output_primitive_id.size() != n) {
        return gwn_status::invalid_argument("ray first-hit: mismatched span sizes");
    }

    if (!gwn_span_has_storage(ray_origin_x) || !gwn_span_has_storage(ray_origin_y) ||
        !gwn_span_has_storage(ray_origin_z) || !gwn_span_has_storage(ray_dir_x) ||
        !gwn_span_has_storage(ray_dir_y) || !gwn_span_has_storage(ray_dir_z) ||
        !gwn_span_has_storage(output_t) || !gwn_span_has_storage(output_primitive_id)) {
        return gwn_status::invalid_argument(
            "ray first-hit: ray/output spans must use non-null storage when non-empty."
        );
    }

    return gwn_status::ok();
}

} // namespace detail

/// \brief Compute unsigned point-to-triangle distance (\c __device__ only).
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline Real gwn_unsigned_distance_point_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, Real const qx, Real const qy,
    Real const qz, Real const culling_band = std::numeric_limits<Real>::infinity(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    return detail::gwn_unsigned_distance_point_bvh_impl<
        Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, qx, qy, qz, culling_band, overflow_callback
    );
}

/// \brief Compute signed point-to-mesh distance using winding-number sign
///        (\c __device__ only).
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline Real gwn_signed_distance_point_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree, Real const qx,
    Real const qy, Real const qz, Real const winding_number_threshold = Real(0.5),
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    Real const accuracy_scale = Real(2), OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_signed_distance_point_bvh currently supports Order 0, 1, and 2."
    );

    return detail::gwn_signed_distance_point_bvh_impl<
        Order, Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, data_tree, qx, qy, qz, winding_number_threshold, culling_band,
        accuracy_scale, overflow_callback
    );
}

/// \brief Compute unsigned point-to-boundary-edge distance (\c __device__ only).
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline Real gwn_unsigned_boundary_edge_distance_point_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, Real const qx, Real const qy,
    Real const qz, Real const culling_band = std::numeric_limits<Real>::infinity(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    return detail::gwn_unsigned_boundary_edge_distance_point_bvh_impl<
        Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, qx, qy, qz, culling_band, overflow_callback
    );
}

/// \brief Compute point-to-boundary-edge distances for a batch.
template <
    int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_unsigned_boundary_edge_distance_batch_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
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
        detail::gwn_unsigned_boundary_edge_distance_batch_bvh_functor<
            Width, Real, Index, StackCapacity, OverflowCallback>{
            geometry, bvh, aabb_tree, query_x, query_y, query_z, output, culling_band,
            overflow_callback
        },
        stream
    );
}

/// \brief Width-4 convenience wrapper for batch boundary-edge distance queries.
template <
    gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_unsigned_boundary_edge_distance_batch_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_aabb_accessor<Real, Index> const &aabb_tree, cuda::std::span<Real const> const query_x,
    cuda::std::span<Real const> const query_y, cuda::std::span<Real const> const query_z,
    cuda::std::span<Real> const output,
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_compute_unsigned_boundary_edge_distance_batch_bvh<
        4, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, query_x, query_y, query_z, output, culling_band, stream,
        overflow_callback
    );
}

/// \brief First-hit result of a ray query against mesh triangles.
template <gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_ray_first_hit_result = detail::gwn_ray_first_hit_result<Real, Index>;

/// \brief Compute the nearest ray-triangle hit for a single ray
///        (\c __device__ only).
///
/// Returns hit distance \c t, primitive id, and Embree-style barycentrics
/// \c u/\c v (weights of triangle vertices v1/v2). Misses return
/// \c t = -1, \c primitive_id = gwn_invalid_index<Index>(), and
/// \c u = \c v = 0. The \c status field is one of
/// \c k_miss, \c k_hit, or \c k_overflow.
/// Ray direction vectors do not need to be normalized.
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline gwn_ray_first_hit_result<Real, Index> gwn_ray_first_hit_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, Real const ray_ox,
    Real const ray_oy, Real const ray_oz, Real const ray_dx, Real const ray_dy, Real const ray_dz,
    Real const t_min = Real(0), Real const t_max = std::numeric_limits<Real>::infinity(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");
    return detail::gwn_ray_first_hit_bvh_impl<Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, t_min, t_max,
        overflow_callback
    );
}

/// \brief Compute the nearest ray-triangle hit (`t` + primitive id) for a
///        batch of rays using BVH + AABB traversal.
///
/// Rays are evaluated over the interval [`t_min`, `t_max`]. Misses write
/// `t = -1` and `primitive_id = gwn_invalid_index<Index>()`.
template <
    int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_ray_first_hit_batch_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    cuda::std::span<Real const> const ray_origin_x, cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z, cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y, cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t, cuda::std::span<Index> const output_primitive_id,
    Real const t_min = Real(0), Real const t_max = std::numeric_limits<Real>::infinity(),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
    if (!aabb_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH AABB tree is invalid for the given topology.");

    gwn_status const span_status = detail::gwn_validate_ray_first_hit_batch_spans<Real, Index>(
        ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, output_t,
        output_primitive_id
    );
    if (!span_status.is_ok())
        return span_status;

    if (!(t_max >= t_min))
        return gwn_status::invalid_argument("ray first-hit: requires t_max >= t_min.");
    std::size_t const n = ray_origin_x.size();
    if (n == 0)
        return gwn_status::ok();

    constexpr int k_block_size = detail::k_gwn_default_block_size;
    return detail::gwn_launch_linear_kernel<k_block_size>(
        n,
        detail::gwn_ray_first_hit_batch_bvh_functor<
            Width, Real, Index, StackCapacity, OverflowCallback>{
            geometry, bvh, aabb_tree, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x,
            ray_dir_y, ray_dir_z, output_t, output_primitive_id, t_min, t_max, overflow_callback
        },
        stream
    );
}

/// \brief Width-4 convenience wrapper for batch ray first-hit queries.
template <
    gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_ray_first_hit_batch_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_aabb_accessor<Real, Index> const &aabb_tree,
    cuda::std::span<Real const> const ray_origin_x, cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z, cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y, cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t, cuda::std::span<Index> const output_primitive_id,
    Real const t_min = Real(0), Real const t_max = std::numeric_limits<Real>::infinity(),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_compute_ray_first_hit_batch_bvh<4, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y,
        ray_dir_z, output_t, output_primitive_id, t_min, t_max, stream, overflow_callback
    );
}

/// \brief Compute the nearest ray-triangle hit for a single BLAS or scene ray
///        query (\c __device__ only).
template <
    class Accel, int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
    requires(is_traversable_v<Accel>)
__device__ inline gwn_ray_hit_result<
    typename gwn_accel_traits<Accel>::real_type, typename gwn_accel_traits<Accel>::index_type>
gwn_ray_first_hit(
    Accel const &accel, typename gwn_accel_traits<Accel>::real_type const ray_ox,
    typename gwn_accel_traits<Accel>::real_type const ray_oy,
    typename gwn_accel_traits<Accel>::real_type const ray_oz,
    typename gwn_accel_traits<Accel>::real_type const ray_dx,
    typename gwn_accel_traits<Accel>::real_type const ray_dy,
    typename gwn_accel_traits<Accel>::real_type const ray_dz,
    typename gwn_accel_traits<Accel>::real_type const t_min =
        typename gwn_accel_traits<Accel>::real_type(0),
    typename gwn_accel_traits<Accel>::real_type const t_max =
        std::numeric_limits<typename gwn_accel_traits<Accel>::real_type>::infinity(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");
    return detail::gwn_ray_first_hit_accel_impl<Accel, StackCapacity, OverflowCallback>(
        accel, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, t_min, t_max, overflow_callback
    );
}

/// \brief Compute nearest ray-triangle hits for a batch against either a BLAS
///        accessor or a scene accessor.
template <
    class Accel, int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
    requires(is_traversable_v<Accel>)
gwn_status gwn_compute_ray_first_hit_batch(
    Accel const &accel,
    cuda::std::span<typename gwn_accel_traits<Accel>::real_type const> const ray_origin_x,
    cuda::std::span<typename gwn_accel_traits<Accel>::real_type const> const ray_origin_y,
    cuda::std::span<typename gwn_accel_traits<Accel>::real_type const> const ray_origin_z,
    cuda::std::span<typename gwn_accel_traits<Accel>::real_type const> const ray_dir_x,
    cuda::std::span<typename gwn_accel_traits<Accel>::real_type const> const ray_dir_y,
    cuda::std::span<typename gwn_accel_traits<Accel>::real_type const> const ray_dir_z,
    cuda::std::span<typename gwn_accel_traits<Accel>::real_type> const output_t,
    cuda::std::span<typename gwn_accel_traits<Accel>::index_type> const output_primitive_id,
    cuda::std::span<typename gwn_accel_traits<Accel>::index_type> const output_instance_id,
    typename gwn_accel_traits<Accel>::real_type const t_min =
        typename gwn_accel_traits<Accel>::real_type(0),
    typename gwn_accel_traits<Accel>::real_type const t_max =
        std::numeric_limits<typename gwn_accel_traits<Accel>::real_type>::infinity(),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    using Real = typename gwn_accel_traits<Accel>::real_type;
    using Index = typename gwn_accel_traits<Accel>::index_type;
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if constexpr (is_blas_accessor_v<Accel>) {
        if (!accel.geometry.is_valid())
            return gwn_status::invalid_argument(
                "Geometry accessor contains mismatched span lengths."
            );
        if (!accel.topology.is_valid())
            return gwn_status::invalid_argument("BVH accessor is invalid.");
        if (!accel.aabb.is_valid_for(accel.topology))
            return gwn_status::invalid_argument("BVH AABB tree is invalid for the given topology.");
    } else {
        if (!detail::gwn_scene_query_accel_has_basic_data_impl(accel))
            return gwn_status::invalid_argument("Scene accessor is invalid.");
    }

    gwn_status const span_status =
        detail::gwn_validate_unified_ray_first_hit_batch_spans<Real, Index>(
            ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, output_t,
            output_primitive_id, output_instance_id
        );
    if (!span_status.is_ok())
        return span_status;

    if (!(t_max >= t_min))
        return gwn_status::invalid_argument("ray first-hit: requires t_max >= t_min.");
    std::size_t const n = ray_origin_x.size();
    if (n == 0)
        return gwn_status::ok();

    constexpr int k_block_size = detail::k_gwn_default_block_size;
    return detail::gwn_launch_linear_kernel<k_block_size>(
        n,
        detail::gwn_ray_first_hit_batch_accel_functor<Accel, StackCapacity, OverflowCallback>{
            accel, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z,
            output_t, output_primitive_id, output_instance_id, t_min, t_max, overflow_callback
        },
        stream
    );
}

/// \brief Compute exact winding number for a single query point (\c __device__ only).
///
/// Uses BVH traversal and exact solid-angle evaluation at leaf nodes.
/// Returns the winding number as a scalar value.
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline Real gwn_winding_number_point_bvh_exact(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh, Real const qx, Real const qy,
    Real const qz, OverflowCallback const &overflow_callback = {}
) noexcept {
    return detail::gwn_winding_number_point_bvh_exact_impl<
        Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, qx, qy, qz, overflow_callback
    );
}

/// \brief Compute Taylor-approximated winding number for a single query point
///        (\c __device__ only).
///
/// Uses BVH traversal with Taylor moment approximation at far-field nodes
/// and exact solid-angle evaluation at near-field leaf nodes.
/// Returns the winding number as a scalar value.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline Real gwn_winding_number_point_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree, Real const qx,
    Real const qy, Real const qz, Real const accuracy_scale = Real(2),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_winding_number_point_bvh_taylor currently supports Order 0, 1, and 2."
    );
    return detail::gwn_winding_number_point_bvh_taylor_impl<
        Order, Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, data_tree, qx, qy, qz, accuracy_scale, overflow_callback
    );
}

/// \brief Compute winding numbers for a batch using Taylor-accelerated BVH
///        traversal.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_winding_number_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}
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
        detail::gwn_winding_number_batch_bvh_taylor_functor<
            Order, Width, Real, Index, StackCapacity, OverflowCallback>{
            geometry, bvh, data_tree, query_x, query_y, query_z, output, accuracy_scale,
            overflow_callback
        },
        stream
    );
}

/// \brief Width-4 convenience wrapper for Taylor BVH batch winding-number
///        queries.
///
/// \copydetails gwn_compute_winding_number_batch_bvh_taylor
template <
    int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_winding_number_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_moment_accessor<Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_compute_winding_number_batch_bvh_taylor<
        Order, 4, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, data_tree, query_x, query_y, query_z, output, accuracy_scale, stream,
        overflow_callback
    );
}

/// \brief Compute exact winding numbers for a batch using BVH traversal.
///
/// No moment tree or accuracy scale — every leaf is evaluated exactly.
///
/// \tparam Width BVH width.
template <
    int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_winding_number_batch_bvh_exact(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
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
        detail::gwn_winding_number_batch_bvh_exact_functor<
            Width, Real, Index, StackCapacity, OverflowCallback>{
            geometry, bvh, query_x, query_y, query_z, output, overflow_callback
        },
        stream
    );
}

/// \brief Width-4 convenience wrapper for exact BVH batch winding-number queries.
template <
    gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_winding_number_batch_bvh_exact(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh, cuda::std::span<Real const> const query_x,
    cuda::std::span<Real const> const query_y, cuda::std::span<Real const> const query_z,
    cuda::std::span<Real> const output, cudaStream_t const stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_compute_winding_number_batch_bvh_exact<
        4, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, query_x, query_y, query_z, output, stream, overflow_callback
    );
}

/// \brief Compute unsigned point-to-mesh distances for a batch.
///
/// \tparam Width BVH width.
template <
    int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_unsigned_distance_batch_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
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
        detail::gwn_unsigned_distance_batch_bvh_functor<
            Width, Real, Index, StackCapacity, OverflowCallback>{
            geometry, bvh, aabb_tree, query_x, query_y, query_z, output, culling_band,
            overflow_callback
        },
        stream
    );
}

/// \brief Width-4 convenience wrapper for batch unsigned distance queries.
template <
    gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_unsigned_distance_batch_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_aabb_accessor<Real, Index> const &aabb_tree, cuda::std::span<Real const> const query_x,
    cuda::std::span<Real const> const query_y, cuda::std::span<Real const> const query_z,
    cuda::std::span<Real> const output,
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_compute_unsigned_distance_batch_bvh<4, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, query_x, query_y, query_z, output, culling_band, stream,
        overflow_callback
    );
}

/// \brief Compute signed point-to-mesh distances for a batch using
///        winding-number sign.
///
/// \tparam Order Taylor winding-number order (0, 1, or 2).
/// \tparam Width BVH width.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_signed_distance_batch_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const winding_number_threshold = Real(0.5),
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_compute_signed_distance_batch_bvh currently supports Order 0, 1, and 2."
    );
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
    if (!aabb_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH AABB tree is invalid for the given topology.");
    if (!moment_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH moment tree is invalid for the given topology.");
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
        detail::gwn_signed_distance_batch_bvh_functor<
            Order, Width, Real, Index, StackCapacity, OverflowCallback>{
            geometry, bvh, aabb_tree, moment_tree, query_x, query_y, query_z, output,
            winding_number_threshold, culling_band, accuracy_scale, overflow_callback
        },
        stream
    );
}

/// \brief Width-4 convenience wrapper for batch signed distance queries.
template <
    int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_signed_distance_batch_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_aabb_accessor<Real, Index> const &aabb_tree,
    gwn_bvh4_moment_accessor<Order, Real, Index> const &moment_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const winding_number_threshold = Real(0.5),
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_compute_signed_distance_batch_bvh<
        Order, 4, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, moment_tree, query_x, query_y, query_z, output,
        winding_number_threshold, culling_band, accuracy_scale, stream, overflow_callback
    );
}

/// \brief Compute Taylor-approximated winding-number gradient for a single
///        query point (\c __device__ only).
///
/// Uses BVH traversal with Taylor moment approximation at far-field nodes
/// and exact solid-angle gradient evaluation at near-field leaf nodes.
/// Returns the gradient as a 3-component vector (gwn_vec3).
///
/// \tparam Order Taylor winding-number order (0, 1, or 2).
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline gwn_vec3<Real> gwn_winding_gradient_point_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree, Real const qx,
    Real const qy, Real const qz, Real const accuracy_scale = Real(2),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_winding_gradient_point_bvh_taylor currently supports Order 0, 1, and 2."
    );
    return detail::gwn_winding_gradient_point_bvh_taylor_impl<
        Order, Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, data_tree, qx, qy, qz, accuracy_scale, overflow_callback
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
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_winding_gradient_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output_x,
    cuda::std::span<Real> const output_y, cuda::std::span<Real> const output_z,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}
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
        detail::gwn_winding_gradient_batch_bvh_taylor_functor<
            Order, Width, Real, Index, StackCapacity, OverflowCallback>{
            geometry, bvh, data_tree, query_x, query_y, query_z, output_x, output_y, output_z,
            accuracy_scale, overflow_callback
        },
        stream
    );
}

/// \brief Width-4 convenience wrapper for Taylor BVH batch winding-gradient
///        queries.
template <
    int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_winding_gradient_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_moment_accessor<Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output_x,
    cuda::std::span<Real> const output_y, cuda::std::span<Real> const output_z,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_compute_winding_gradient_batch_bvh_taylor<
        Order, 4, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, data_tree, query_x, query_y, query_z, output_x, output_y, output_z,
        accuracy_scale, stream, overflow_callback
    );
}

/// \brief Harnack trace result type.
template <gwn_real_type Real>
using gwn_harnack_trace_result = detail::gwn_harnack_trace_result<Real>;

/// \brief Hybrid trace result type.
template <gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_hybrid_trace_result = detail::gwn_hybrid_trace_result<Real, Index>;

/// \brief Hybrid hit classification.
using gwn_hybrid_hit_kind = detail::gwn_hybrid_hit_kind;

/// \brief Hybrid triangle-hit normal policy.
using gwn_hybrid_triangle_normal_policy = detail::gwn_hybrid_triangle_normal_policy;

/// \brief Embree-style hybrid trace arguments.
template <gwn_real_type Real>
using gwn_hybrid_trace_arguments = detail::gwn_hybrid_trace_arguments<Real>;

/// \brief Initialize \ref gwn_hybrid_trace_arguments with default values.
template <gwn_real_type Real>
inline void gwn_init_hybrid_trace_arguments(gwn_hybrid_trace_arguments<Real> &arguments) noexcept {
    arguments = gwn_hybrid_trace_arguments<Real>{};
}

/// \brief Trace a single ray through the winding-number level set using the
///        Harnack inequality for guaranteed step safety (\c __device__ only).
///
/// \tparam Order Taylor winding-number order (0, 1, or 2).
/// \return \ref gwn_harnack_trace_result with hit parameter, normal, etc.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline gwn_harnack_trace_result<Real> gwn_harnack_trace_ray_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree, Real const ray_ox,
    Real const ray_oy, Real const ray_oz, Real const ray_dx, Real const ray_dy, Real const ray_dz,
    Real const target_winding = Real(0.5), Real const epsilon = Real(1e-4),
    int const max_iterations = 512, Real const t_max = Real(1e6),
    Real const accuracy_scale = Real(2), OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_harnack_trace_ray_bvh_taylor currently supports Order 0, 1, and 2."
    );
    return detail::gwn_harnack_trace_ray_impl<
        Order, Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, moment_tree, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
        target_winding, epsilon, max_iterations, t_max, accuracy_scale, overflow_callback
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
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_harnack_trace_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree,
    cuda::std::span<Real const> const ray_origin_x, cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z, cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y, cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t, cuda::std::span<Real> const output_normal_x,
    cuda::std::span<Real> const output_normal_y, cuda::std::span<Real> const output_normal_z,
    Real const target_winding = Real(0.5), Real const epsilon = Real(1e-4),
    int const max_iterations = 512, Real const t_max = Real(1e6),
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}
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
    if (ray_origin_y.size() != n || ray_origin_z.size() != n || ray_dir_x.size() != n ||
        ray_dir_y.size() != n || ray_dir_z.size() != n || output_t.size() != n ||
        output_normal_x.size() != n || output_normal_y.size() != n || output_normal_z.size() != n) {
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

    detail::gwn_harnack_trace_batch_functor<
        Order, Width, Real, Index, StackCapacity, OverflowCallback> const functor{
        geometry,        bvh,
        aabb_tree,       moment_tree,
        ray_origin_x,    ray_origin_y,
        ray_origin_z,    ray_dir_x,
        ray_dir_y,       ray_dir_z,
        output_t,        output_normal_x,
        output_normal_y, output_normal_z,
        target_winding,  epsilon,
        max_iterations,  t_max,
        accuracy_scale,  overflow_callback,
    };

    constexpr int k_block_size = detail::k_gwn_default_block_size;
    return detail::gwn_launch_linear_kernel<k_block_size>(n, functor, stream);
}

/// \brief Width-4 convenience wrapper for batch Harnack trace.
template <
    int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_harnack_trace_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_aabb_accessor<Real, Index> const &aabb_tree,
    gwn_bvh4_moment_accessor<Order, Real, Index> const &moment_tree,
    cuda::std::span<Real const> const ray_origin_x, cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z, cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y, cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t, cuda::std::span<Real> const output_normal_x,
    cuda::std::span<Real> const output_normal_y, cuda::std::span<Real> const output_normal_z,
    Real const target_winding = Real(0.5), Real const epsilon = Real(1e-4),
    int const max_iterations = 512, Real const t_max = Real(1e6),
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_compute_harnack_trace_batch_bvh_taylor<
        Order, 4, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, moment_tree, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x,
        ray_dir_y, ray_dir_z, output_t, output_normal_x, output_normal_y, output_normal_z,
        target_winding, epsilon, max_iterations, t_max, accuracy_scale, stream, overflow_callback
    );
}

/// \brief Trace a single ray using hybrid first-hit:
///        ray-triangle first-hit + conditioned Harnack fill.
///
/// The routine always invokes ray-triangle first-hit first. If the mesh is
/// globally closed (\c geometry.singular_edge_count == 0), Harnack fill is
/// skipped.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline gwn_hybrid_trace_result<Real, Index> gwn_hybrid_trace_ray_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree, Real const ray_ox,
    Real const ray_oy, Real const ray_oz, Real const ray_dx, Real const ray_dy, Real const ray_dz,
    gwn_hybrid_trace_arguments<Real> const &arguments = gwn_hybrid_trace_arguments<Real>{},
    OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_hybrid_trace_ray_bvh_taylor currently supports Order 0, 1, and 2."
    );
    return detail::gwn_hybrid_trace_ray_impl<
        Order, Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, moment_tree, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
        arguments, overflow_callback
    );
}

/// \brief Batch hybrid trace (ray-triangle first-hit + conditioned Harnack).
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_hybrid_trace_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree,
    cuda::std::span<Real const> const ray_origin_x, cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z, cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y, cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t, cuda::std::span<Real> const output_normal_x,
    cuda::std::span<Real> const output_normal_y, cuda::std::span<Real> const output_normal_z,
    cuda::std::span<Index> const output_primitive_id,
    gwn_hybrid_trace_arguments<Real> const &arguments = gwn_hybrid_trace_arguments<Real>{},
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_compute_hybrid_trace_batch_bvh_taylor currently supports Order 0, 1, and 2."
    );
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
    if (!aabb_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH AABB tree is invalid for the given topology.");
    if (!moment_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH data tree is invalid for the given topology.");
    if (!(arguments.t_max >= arguments.t_min))
        return gwn_status::invalid_argument("hybrid trace: requires arguments.t_max >= t_min.");

    std::size_t const n = ray_origin_x.size();
    if (ray_origin_y.size() != n || ray_origin_z.size() != n || ray_dir_x.size() != n ||
        ray_dir_y.size() != n || ray_dir_z.size() != n || output_t.size() != n ||
        output_normal_x.size() != n || output_normal_y.size() != n || output_normal_z.size() != n ||
        output_primitive_id.size() != n) {
        return gwn_status::invalid_argument("hybrid trace: mismatched span sizes");
    }

    if (!gwn_span_has_storage(ray_origin_x) || !gwn_span_has_storage(ray_origin_y) ||
        !gwn_span_has_storage(ray_origin_z) || !gwn_span_has_storage(ray_dir_x) ||
        !gwn_span_has_storage(ray_dir_y) || !gwn_span_has_storage(ray_dir_z) ||
        !gwn_span_has_storage(output_t) || !gwn_span_has_storage(output_normal_x) ||
        !gwn_span_has_storage(output_normal_y) || !gwn_span_has_storage(output_normal_z) ||
        !gwn_span_has_storage(output_primitive_id)) {
        return gwn_status::invalid_argument(
            "hybrid trace: ray/output spans must use non-null storage when non-empty."
        );
    }

    if (n == 0)
        return gwn_status::ok();

    detail::gwn_hybrid_trace_batch_functor<
        Order, Width, Real, Index, StackCapacity, OverflowCallback> const functor{
        geometry,
        bvh,
        aabb_tree,
        moment_tree,
        ray_origin_x,
        ray_origin_y,
        ray_origin_z,
        ray_dir_x,
        ray_dir_y,
        ray_dir_z,
        output_t,
        output_normal_x,
        output_normal_y,
        output_normal_z,
        output_primitive_id,
        arguments,
        overflow_callback,
    };

    constexpr int k_block_size = detail::k_gwn_default_block_size;
    return detail::gwn_launch_linear_kernel<k_block_size>(n, functor, stream);
}

/// \brief Width-4 convenience wrapper for batch hybrid trace.
template <
    int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_hybrid_trace_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn_bvh4_aabb_accessor<Real, Index> const &aabb_tree,
    gwn_bvh4_moment_accessor<Order, Real, Index> const &moment_tree,
    cuda::std::span<Real const> const ray_origin_x, cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z, cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y, cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t, cuda::std::span<Real> const output_normal_x,
    cuda::std::span<Real> const output_normal_y, cuda::std::span<Real> const output_normal_z,
    cuda::std::span<Index> const output_primitive_id,
    gwn_hybrid_trace_arguments<Real> const &arguments = gwn_hybrid_trace_arguments<Real>{},
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_compute_hybrid_trace_batch_bvh_taylor<
        Order, 4, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, aabb_tree, moment_tree, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x,
        ray_dir_y, ray_dir_z, output_t, output_normal_x, output_normal_y, output_normal_z,
        output_primitive_id, arguments, stream, overflow_callback
    );
}

} // namespace gwn
