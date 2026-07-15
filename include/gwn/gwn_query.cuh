#pragma once

/// \file gwn_query.cuh
/// \brief Public query APIs for winding, gradients, unsigned distance, and ray hits.
///
/// Query operations are available in two forms:
/// - Device point APIs (`__device__` functions callable from user kernels)
/// - Batch APIs (host-callable launchers that process arrays of queries)
///
/// Batch APIs validate their objects and spans, enqueue work on the supplied stream, and return
/// without synchronizing that stream. Device point APIs accept accessors and return values
/// directly. Traversal queries invoke their overflow callback before returning an overflow value;
/// the default callback traps.
/// Batch launchers use \c gwn_query_batch_config as a structural non-type template parameter. The
/// default block size is 256 threads; traversal batch launchers also use its stack capacity.
/// Their input and output spans must refer to device-accessible storage that remains valid until
/// the supplied stream completes.
///
/// Every query family provides both point and batch variants.
/// Implementation details live in `include/gwn/detail/gwn_query_*_impl.cuh`.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "detail/gwn_kernel_utils.cuh"
#include "detail/gwn_query_vec3_impl.cuh"
#include "gwn_boundary.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"

namespace gwn {

/// \brief 3-component column vector parameterised on scalar type \p Real.
template <gwn_real_type Real> using gwn_vec3 = detail::gwn_query_vec3<Real>;

/// \brief Default number of threads per query batch block.
inline constexpr int k_gwn_default_query_batch_block_size = 256;

/// \brief Default capacity of the per-thread BVH traversal stack.
inline constexpr int k_gwn_default_traversal_stack_capacity = 64;

/// \brief Compile-time launch configuration shared by query batch APIs.
struct gwn_query_batch_config {
    int block_size = k_gwn_default_query_batch_block_size; ///< Threads per CUDA block.
    int stack_capacity =
        k_gwn_default_traversal_stack_capacity; ///< Per-thread traversal stack, if applicable.
};

/// \brief Query batch configuration with a CUDA-supported compile-time block size.
template <gwn_query_batch_config Config>
concept gwn_query_batch_config_value = Config.block_size > 0 && Config.block_size <= 1024;

/// \brief Query batch configuration with a positive traversal stack capacity.
template <gwn_query_batch_config Config>
concept gwn_traversal_batch_config_value =
    gwn_query_batch_config_value<Config> && Config.stack_capacity > 0;

/// \brief Completion state of a ray first-hit query.
enum class gwn_ray_first_hit_status : std::uint8_t {
    k_miss = 0,    ///< Traversal completed without a triangle hit.
    k_hit = 1,     ///< Traversal completed with the nearest triangle hit.
    k_overflow = 2 ///< Traversal exhausted its per-thread stack capacity.
};

/// \brief First-hit result of a ray query against mesh triangles.
///
/// Barycentric accuracy follows the precision of \p Real and the conditioning of the intersected
/// triangle.
template <gwn_real_type Real, gwn_index_type Index = std::uint32_t>
struct gwn_ray_first_hit_result {
    Real t{Real(-1)};                 ///< Ray parameter at the hit, or -1 for a miss.
    Real u{Real(0)};                  ///< Barycentric weight of the second triangle vertex.
    Real v{Real(0)};                  ///< Barycentric weight of the third triangle vertex.
    Real geometric_normal_x{Real(0)}; ///< Unnormalized oriented geometric normal X component.
    Real geometric_normal_y{Real(0)}; ///< Unnormalized oriented geometric normal Y component.
    Real geometric_normal_z{Real(0)}; ///< Unnormalized oriented geometric normal Z component.
    Index primitive_id{gwn_invalid_index<Index>()}; ///< Original mesh primitive ID.
    gwn_ray_first_hit_status status{gwn_ray_first_hit_status::k_miss}; ///< Completion state.

    /// \brief Return whether traversal completed with a triangle hit.
    [[nodiscard]] __host__ __device__ constexpr bool hit() const noexcept {
        return status == gwn_ray_first_hit_status::k_hit;
    }

private:
    // Result arrays are copied as complete objects. Making the alignment tail an initialized member
    // prevents those bytes from carrying indeterminate device state into a host transfer.
    static constexpr std::size_t k_storage_padding_size = sizeof(Index) == 4 ? 3 : 7;
    std::uint8_t storage_padding_[k_storage_padding_size]{};
};

} // namespace gwn

#include "detail/gwn_query_common_impl.cuh"
#include "detail/gwn_query_distance_impl.cuh"
#include "detail/gwn_query_geometry_impl.cuh"
#include "detail/gwn_query_gradient_impl.cuh"
#include "detail/gwn_query_ray_impl.cuh"
#include "detail/gwn_query_winding_antipodal_impl.cuh"
#include "detail/gwn_query_winding_impl.cuh"

namespace gwn {

/// \brief Compute unsigned point-to-mesh distance from a canonical BVH in device code.
///
/// \p culling_band is the initial distance bound and the result when no triangle is closer.
/// Negative values are clamped to zero and infinity performs an unbounded search. Stack overflow
/// invokes \p overflow_callback and returns NaN when the callback returns. \p bvh must be valid.
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline Real gwn_unsigned_distance(
    gwn_bvh_accessor<Width, Real, Index> const &bvh, Real const qx, Real const qy, Real const qz,
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    GWN_ASSERT(bvh.is_valid(), "gwn_unsigned_distance requires a valid BVH accessor.");
    return detail::gwn_unsigned_distance_impl<Width, Real, Index, StackCapacity, OverflowCallback>(
        bvh, qx, qy, qz, culling_band, overflow_callback
    );
}

/// \brief Compute unsigned point-to-mesh distances for a batch.
///
/// Queries read bounds and leaf-ordered triangle records from \p bvh. \p culling_band limits the
/// returned distance as described by \ref gwn_unsigned_distance. Each overflowed query writes NaN
/// when its callback returns. \c Config controls the launch block size and traversal stack
/// capacity.
template <
    gwn_query_batch_config Config = gwn_query_batch_config{}, int Width, gwn_real_type Real,
    gwn_index_type Index = std::uint32_t,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
    requires gwn_traversal_batch_config_value<Config>
[[nodiscard]] gwn_status gwn_compute_unsigned_distance_batch(
    gwn_bvh_object<Width, Real, Index> const &bvh,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_x,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_y,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_z,
    gwn_device_span<Real> const device_output,
    Real const culling_band = std::numeric_limits<Real>::infinity(),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    auto const query_x = detail::gwn_cuda_span_view_impl(device_query_x);
    auto const query_y = detail::gwn_cuda_span_view_impl(device_query_y);
    auto const query_z = detail::gwn_cuda_span_view_impl(device_query_z);
    auto const output = detail::gwn_cuda_span_view_impl(device_output);
    auto const &accessor = bvh.accessor();
    if (!accessor.is_valid())
        return gwn_status::invalid_argument("BVH object contains no queryable data.");
    GWN_RETURN_ON_ERROR(
        (detail::gwn_validate_traversal_stack_capacity_impl<Config.stack_capacity>(accessor))
    );
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

    return detail::gwn_launch_linear_kernel<Config.block_size>(
        output.size(),
        detail::gwn_unsigned_distance_batch_functor<
            Width, Real, Index, Config.stack_capacity, OverflowCallback>{
            accessor, query_x, query_y, query_z, output, culling_band, overflow_callback
        },
        stream
    );
}

/// \brief Compute the nearest ray-triangle hit from a canonical BVH in device code.
///
/// The result contains hit distance, original primitive ID, barycentric weights for the second and
/// third triangle vertices, and three scalar components of the unnormalized oriented geometric
/// normal. A miss returns \c t=-1 and an invalid primitive ID. Stack overflow invokes
/// \p overflow_callback and returns the best hit found with \c k_overflow status when the callback
/// returns. \p bvh must be valid and the ray interval must satisfy `t_max >= t_min`.
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline gwn_ray_first_hit_result<Real, Index> gwn_ray_first_hit(
    gwn_bvh_accessor<Width, Real, Index> const &bvh, Real const ray_ox, Real const ray_oy,
    Real const ray_oz, Real const ray_dx, Real const ray_dy, Real const ray_dz,
    Real const t_min = Real(0), Real const t_max = std::numeric_limits<Real>::infinity(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    GWN_ASSERT(bvh.is_valid(), "gwn_ray_first_hit requires a valid BVH accessor.");
    GWN_ASSERT(t_max >= t_min, "gwn_ray_first_hit requires t_max >= t_min.");
    return detail::gwn_ray_first_hit_impl<Width, Real, Index, StackCapacity, OverflowCallback>(
        bvh, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, t_min, t_max, overflow_callback
    );
}

/// \brief Compute nearest ray-triangle hits for a batch.
///
/// Rays are evaluated on the closed interval [\p t_min, \p t_max]. Each output record has the
/// same fields and status semantics as \ref gwn_ray_first_hit. The launch is asynchronous with
/// respect to the host. For width-4 float BVHs with uint32 indices, the launcher selects packed
/// traversal when the root is internal, node references fit the packed encoding, and
/// \c packed_stack_bound does not exceed \c Config.stack_capacity. Otherwise it selects canonical
/// traversal with identical result semantics. \c Config also controls the launch block size.
template <
    gwn_query_batch_config Config = gwn_query_batch_config{}, int Width, gwn_real_type Real,
    gwn_index_type Index = std::uint32_t,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
    requires gwn_traversal_batch_config_value<Config>
[[nodiscard]] gwn_status gwn_compute_ray_first_hit_batch(
    gwn_bvh_object<Width, Real, Index> const &bvh,
    gwn_device_span<std::type_identity_t<Real> const> const device_ray_origin_x,
    gwn_device_span<std::type_identity_t<Real> const> const device_ray_origin_y,
    gwn_device_span<std::type_identity_t<Real> const> const device_ray_origin_z,
    gwn_device_span<std::type_identity_t<Real> const> const device_ray_dir_x,
    gwn_device_span<std::type_identity_t<Real> const> const device_ray_dir_y,
    gwn_device_span<std::type_identity_t<Real> const> const device_ray_dir_z,
    gwn_device_span<gwn_ray_first_hit_result<Real, Index>> const device_output,
    Real const t_min = Real(0), Real const t_max = std::numeric_limits<Real>::infinity(),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    auto const ray_origin_x = detail::gwn_cuda_span_view_impl(device_ray_origin_x);
    auto const ray_origin_y = detail::gwn_cuda_span_view_impl(device_ray_origin_y);
    auto const ray_origin_z = detail::gwn_cuda_span_view_impl(device_ray_origin_z);
    auto const ray_dir_x = detail::gwn_cuda_span_view_impl(device_ray_dir_x);
    auto const ray_dir_y = detail::gwn_cuda_span_view_impl(device_ray_dir_y);
    auto const ray_dir_z = detail::gwn_cuda_span_view_impl(device_ray_dir_z);
    auto const output = detail::gwn_cuda_span_view_impl(device_output);
    auto const &accessor = bvh.accessor();
    if (!accessor.is_valid())
        return gwn_status::invalid_argument("BVH object contains no queryable data.");
    GWN_RETURN_ON_ERROR(
        (detail::gwn_validate_traversal_stack_capacity_impl<Config.stack_capacity>(accessor))
    );

    gwn_status const span_status = detail::gwn_validate_ray_first_hit_batch_spans<Real, Index>(
        ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, output
    );
    if (!span_status.is_ok())
        return span_status;
    if (!(t_max >= t_min))
        return gwn_status::invalid_argument("ray first-hit: requires t_max >= t_min.");
    if (ray_origin_x.empty())
        return gwn_status::ok();

    auto const launch = [&]<bool UsePackedTraversal>() noexcept {
        return detail::gwn_launch_linear_kernel<Config.block_size>(
            ray_origin_x.size(),
            detail::gwn_ray_first_hit_batch_functor<
                Width, Real, Index, Config.stack_capacity, OverflowCallback, UsePackedTraversal>{
                accessor,
                ray_origin_x,
                ray_origin_y,
                ray_origin_z,
                ray_dir_x,
                ray_dir_y,
                ray_dir_z,
                output,
                t_min,
                t_max,
                overflow_callback,
            },
            stream
        );
    };

    if constexpr (
        Width == 4 && std::is_same_v<Real, float> && std::is_same_v<Index, std::uint32_t>
    ) {
        // A leaf payload reserves its high bit and two child-slot bits, leaving 29 bits for the
        // parent node offset. The same node-count limit also keeps internal payloads unambiguous.
        constexpr std::uint64_t k_leaf_parent_limit = std::uint64_t(1) << 29;
        // Dispatch before launch so the common kernel owns only its 32-bit pending-child stack.
        if (accessor.has_internal_root() && accessor.nodes.size() < k_leaf_parent_limit &&
            accessor.packed_stack_bound <= Config.stack_capacity) {
            return launch.template operator()<true>();
        }
    }
    // Canonical traversal stores only internal references, whose bound was validated above.
    return launch.template operator()<false>();
}

/// \brief Compute exact winding number from the canonical triangle sequence in device code.
///
/// Exact winding evaluates every triangle record directly. It does not read hierarchy nodes or
/// allocate a traversal stack. \p bvh must be valid.
template <int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline Real gwn_winding_number_exact(
    gwn_bvh_accessor<Width, Real, Index> const &bvh, Real const qx, Real const qy, Real const qz
) noexcept {
    GWN_ASSERT(bvh.is_valid(), "gwn_winding_number_exact requires a valid BVH accessor.");
    return detail::gwn_winding_number_exact_impl(bvh, qx, qy, qz);
}

/// \brief Compute exact winding numbers for a batch.
///
/// Each thread scans the complete canonical triangle sequence without hierarchy traversal.
/// \c Config controls the launch block size; its stack capacity is not applicable.
template <
    gwn_query_batch_config Config = gwn_query_batch_config{}, int Width, gwn_real_type Real,
    gwn_index_type Index = std::uint32_t>
    requires gwn_query_batch_config_value<Config>
[[nodiscard]] gwn_status gwn_compute_winding_number_exact_batch(
    gwn_bvh_object<Width, Real, Index> const &bvh,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_x,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_y,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_z,
    gwn_device_span<Real> const device_output, cudaStream_t const stream = gwn_default_stream()
) noexcept {
    auto const query_x = detail::gwn_cuda_span_view_impl(device_query_x);
    auto const query_y = detail::gwn_cuda_span_view_impl(device_query_y);
    auto const query_z = detail::gwn_cuda_span_view_impl(device_query_z);
    auto const output = detail::gwn_cuda_span_view_impl(device_output);
    auto const &accessor = bvh.accessor();
    if (!accessor.is_valid())
        return gwn_status::invalid_argument("BVH object contains no queryable data.");
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

    return detail::gwn_launch_linear_kernel<Config.block_size>(
        output.size(),
        detail::gwn_winding_number_exact_batch_functor<Width, Real, Index>{
            accessor, query_x, query_y, query_z, output
        },
        stream
    );
}

/// \brief Compute Taylor-approximated winding number in device code.
///
/// Far-field children use \p moment; near-field leaves use the canonical triangle records.
/// Stack overflow invokes \p overflow_callback and returns NaN when the callback returns.
/// \p bvh and \p moment must be valid and aligned. \p accuracy_scale must be nonnegative;
/// increasing it descends more children.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline Real gwn_winding_number_taylor(
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_accessor<Width, Order, Real, Index> const &moment, Real const qx, Real const qy,
    Real const qz, Real const accuracy_scale = Real(2),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_winding_number_taylor supports Order 0, 1, and 2."
    );
    GWN_ASSERT(
        moment.is_valid_for(bvh), "gwn_winding_number_taylor requires aligned BVH and moment data."
    );
    GWN_ASSERT(
        accuracy_scale >= Real(0), "gwn_winding_number_taylor requires nonnegative accuracy scale."
    );
    return detail::gwn_winding_number_taylor_impl<
        Order, Width, Real, Index, StackCapacity, OverflowCallback>(
        bvh, moment, qx, qy, qz, accuracy_scale, overflow_callback
    );
}

/// \brief Compute Taylor-approximated winding numbers for a batch.
///
/// \p accuracy_scale must be nonnegative. Each overflowed query writes NaN when its callback
/// returns. \c Config controls the launch block size and traversal stack capacity.
template <
    int Order, gwn_query_batch_config Config = gwn_query_batch_config{}, int Width,
    gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
    requires gwn_traversal_batch_config_value<Config>
[[nodiscard]] gwn_status gwn_compute_winding_number_taylor_batch(
    gwn_bvh_object<Width, Real, Index> const &bvh,
    gwn_bvh_moment_object<Width, Order, Real, Index> const &moment,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_x,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_y,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_z,
    gwn_device_span<Real> const device_output, Real const accuracy_scale = Real(2),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    auto const query_x = detail::gwn_cuda_span_view_impl(device_query_x);
    auto const query_y = detail::gwn_cuda_span_view_impl(device_query_y);
    auto const query_z = detail::gwn_cuda_span_view_impl(device_query_z);
    auto const output = detail::gwn_cuda_span_view_impl(device_output);
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_compute_winding_number_taylor_batch supports Order 0, 1, and 2."
    );
    auto const &bvh_accessor = bvh.accessor();
    auto const &moment_accessor = moment.accessor();
    if (!bvh_accessor.is_valid())
        return gwn_status::invalid_argument("BVH object contains no queryable data.");
    GWN_RETURN_ON_ERROR(
        (detail::gwn_validate_traversal_stack_capacity_impl<Config.stack_capacity>(bvh_accessor))
    );
    if (!moment_accessor.is_valid_for(bvh_accessor))
        return gwn_status::invalid_argument("Moment object is not aligned with the BVH.");
    if (!(accuracy_scale >= Real(0)))
        return gwn_status::invalid_argument("Taylor accuracy scale must be nonnegative.");
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

    return detail::gwn_launch_linear_kernel<Config.block_size>(
        output.size(),
        detail::gwn_winding_number_taylor_batch_functor<
            Order, Width, Real, Index, Config.stack_capacity, OverflowCallback>{
            bvh_accessor,
            moment_accessor,
            query_x,
            query_y,
            query_z,
            output,
            accuracy_scale,
            overflow_callback,
        },
        stream
    );
}

/// \brief Compute Antipodal winding number for one query point.
///
/// The query evaluates the signed ray-crossing term and the boundary spherical
/// area term from Martens et al. Retry order is `+Z`, `+X`, `+Y`. When all
/// three axes are singular, the result is NaN. Traversal overflow invokes
/// \p overflow_callback and returns NaN when the callback returns. All accessors must be valid;
/// \p boundary_chain must come from \p geometry's topology, and \p bvh must represent the same
/// geometry state.
///
/// Reference: Martens, Trettner, and Bessmeltsev, "The Antipodal Method:
/// Fast, Accurate, and Robust 3D Generalized Winding Numbers," ACM TOG 45(4),
/// 2026. https://doi.org/10.1145/3811323
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline Real gwn_winding_number_antipodal(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, Real const qx, Real const qy,
    Real const qz, OverflowCallback const &overflow_callback = {}
) noexcept {
    GWN_ASSERT(
        geometry.is_valid() && bvh.is_valid() && boundary_chain.is_valid(),
        "gwn_winding_number_antipodal requires valid accessors."
    );
    GWN_ASSERT(
        bvh.triangles.size() == geometry.triangle_count() &&
            boundary_chain.mesh_vertex_count == geometry.vertex_count() &&
            boundary_chain.mesh_triangle_count == geometry.triangle_count(),
        "gwn_winding_number_antipodal requires accessors for the same mesh shape."
    );
    return detail::gwn_winding_number_antipodal_impl<
        Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, boundary_chain, qx, qy, qz, overflow_callback
    );
}

/// \brief Compute Antipodal winding numbers for a batch.
///
/// The query evaluates the signed ray-crossing term and the boundary spherical
/// area term from Martens et al. Retry order is `+Z`, `+X`, `+Y`. Queries with
/// three singular axes write NaN. Overflowed queries also write NaN when their
/// callbacks return. The boundary chain must come from the geometry's topology, and the BVH must
/// represent the same geometry state; structural counts are validated before launch. \c Config
/// controls the launch block size and traversal stack capacity.
///
/// Reference: Martens, Trettner, and Bessmeltsev, "The Antipodal Method:
/// Fast, Accurate, and Robust 3D Generalized Winding Numbers," ACM TOG 45(4),
/// 2026. https://doi.org/10.1145/3811323
template <
    gwn_query_batch_config Config = gwn_query_batch_config{}, int Width, gwn_real_type Real,
    gwn_index_type Index = std::uint32_t,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
    requires gwn_traversal_batch_config_value<Config>
[[nodiscard]] gwn_status gwn_compute_winding_number_antipodal_batch(
    gwn_geometry_object<Real, Index> const &geometry, gwn_bvh_object<Width, Real, Index> const &bvh,
    gwn_boundary_chain_object<Index> const &boundary_chain,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_x,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_y,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_z,
    gwn_device_span<Real> const device_output, cudaStream_t const stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    auto const query_x = detail::gwn_cuda_span_view_impl(device_query_x);
    auto const query_y = detail::gwn_cuda_span_view_impl(device_query_y);
    auto const query_z = detail::gwn_cuda_span_view_impl(device_query_z);
    auto const output = detail::gwn_cuda_span_view_impl(device_output);
    auto const &geometry_accessor = geometry.accessor();
    auto const &bvh_accessor = bvh.accessor();
    auto const &boundary_accessor = boundary_chain.accessor();
    if (!geometry_accessor.is_valid())
        return gwn_status::invalid_argument("Geometry object contains no queryable data.");
    if (!bvh_accessor.is_valid())
        return gwn_status::invalid_argument("BVH object contains no queryable data.");
    if (bvh_accessor.triangles.size() != geometry_accessor.triangle_count())
        return gwn_status::invalid_argument("Geometry and BVH triangle counts differ.");
    GWN_RETURN_ON_ERROR(
        (detail::gwn_validate_traversal_stack_capacity_impl<Config.stack_capacity>(bvh_accessor))
    );
    if (!boundary_accessor.is_valid())
        return gwn_status::invalid_argument("Boundary chain object is unbuilt or invalid.");
    if (boundary_accessor.mesh_vertex_count != geometry_accessor.vertex_count() ||
        boundary_accessor.mesh_triangle_count != geometry_accessor.triangle_count()) {
        return gwn_status::invalid_argument("Boundary chain belongs to a different mesh shape.");
    }
    if (query_x.size() != query_y.size() || query_x.size() != query_z.size())
        return gwn_status::invalid_argument("Query SoA spans require identical lengths.");
    if (query_x.size() != output.size())
        return gwn_status::invalid_argument("Output span size differs from query count.");
    if (!gwn_span_has_storage(query_x) || !gwn_span_has_storage(query_y) ||
        !gwn_span_has_storage(query_z) || !gwn_span_has_storage(output)) {
        return gwn_status::invalid_argument(
            "Query/output spans require non-null storage when non-empty."
        );
    }
    if (output.empty())
        return gwn_status::ok();

    return detail::gwn_launch_linear_kernel<Config.block_size>(
        output.size(),
        detail::gwn_winding_number_antipodal_batch_functor<
            Width, Real, Index, Config.stack_capacity, OverflowCallback>{
            geometry_accessor, bvh_accessor, boundary_accessor, query_x, query_y, query_z, output,
            overflow_callback
        },
        stream
    );
}

/// \brief Compute Antipodal winding gradient for one query point.
///
/// The query differentiates the boundary spherical-area term from Martens et
/// al. Retry order is `+Z`, `+X`, `+Y`. When all three axes are singular, the
/// result is `(NaN, NaN, NaN)`. Both accessors must be valid, and \p boundary_chain must come from
/// \p geometry's topology.
///
/// Reference: The Antipodal Method: Fast, Accurate, and Robust 3D Generalized
/// Winding Numbers. https://doi.org/10.1145/3811323
template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline gwn_vec3<Real> gwn_winding_gradient_antipodal(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, Real const qx, Real const qy,
    Real const qz
) noexcept {
    GWN_ASSERT(
        geometry.is_valid() && boundary_chain.is_valid(),
        "gwn_winding_gradient_antipodal requires valid accessors."
    );
    GWN_ASSERT(
        boundary_chain.mesh_vertex_count == geometry.vertex_count() &&
            boundary_chain.mesh_triangle_count == geometry.triangle_count(),
        "gwn_winding_gradient_antipodal requires accessors for the same mesh shape."
    );

    detail::gwn_query_vec3<Real> const query(qx, qy, qz);
    return detail::gwn_winding_gradient_point_antipodal_impl(geometry, boundary_chain, query);
}

/// \brief Compute Antipodal winding gradients for a batch.
///
/// The query differentiates the boundary spherical-area term from Martens et
/// al. Retry order is `+Z`, `+X`, `+Y`. Queries with three singular axes write
/// `(NaN, NaN, NaN)`. The boundary chain must come from the geometry's topology; structural counts
/// are validated before launch. \c Config controls the launch block size; its stack capacity is not
/// applicable.
///
/// Reference: The Antipodal Method: Fast, Accurate, and Robust 3D Generalized
/// Winding Numbers. https://doi.org/10.1145/3811323
template <
    gwn_query_batch_config Config = gwn_query_batch_config{}, gwn_real_type Real,
    gwn_index_type Index = std::uint32_t>
    requires gwn_query_batch_config_value<Config>
[[nodiscard]] gwn_status gwn_compute_winding_gradient_antipodal_batch(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_boundary_chain_object<Index> const &boundary_chain,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_x,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_y,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_z,
    gwn_device_span<Real> const device_output_x, gwn_device_span<Real> const device_output_y,
    gwn_device_span<Real> const device_output_z, cudaStream_t const stream = gwn_default_stream()
) noexcept {
    auto const query_x = detail::gwn_cuda_span_view_impl(device_query_x);
    auto const query_y = detail::gwn_cuda_span_view_impl(device_query_y);
    auto const query_z = detail::gwn_cuda_span_view_impl(device_query_z);
    auto const output_x = detail::gwn_cuda_span_view_impl(device_output_x);
    auto const output_y = detail::gwn_cuda_span_view_impl(device_output_y);
    auto const output_z = detail::gwn_cuda_span_view_impl(device_output_z);
    auto const &geometry_accessor = geometry.accessor();
    auto const &boundary_accessor = boundary_chain.accessor();
    if (!geometry_accessor.is_valid())
        return gwn_status::invalid_argument("Geometry object contains no queryable data.");
    if (!boundary_accessor.is_valid())
        return gwn_status::invalid_argument("Boundary chain object is unbuilt or invalid.");
    if (boundary_accessor.mesh_vertex_count != geometry_accessor.vertex_count() ||
        boundary_accessor.mesh_triangle_count != geometry_accessor.triangle_count()) {
        return gwn_status::invalid_argument("Boundary chain belongs to a different mesh shape.");
    }
    if (query_x.size() != query_y.size() || query_x.size() != query_z.size())
        return gwn_status::invalid_argument("Query SoA spans require identical lengths.");
    if (query_x.size() != output_x.size() || query_x.size() != output_y.size() ||
        query_x.size() != output_z.size())
        return gwn_status::invalid_argument("Output span sizes differ from query count.");
    if (!gwn_span_has_storage(query_x) || !gwn_span_has_storage(query_y) ||
        !gwn_span_has_storage(query_z) || !gwn_span_has_storage(output_x) ||
        !gwn_span_has_storage(output_y) || !gwn_span_has_storage(output_z)) {
        return gwn_status::invalid_argument(
            "Query/output spans require non-null storage when non-empty."
        );
    }
    if (output_x.empty())
        return gwn_status::ok();

    return detail::gwn_launch_linear_kernel<Config.block_size>(
        output_x.size(),
        detail::gwn_winding_gradient_antipodal_batch_functor<Real, Index>{
            geometry_accessor, boundary_accessor, query_x, query_y, query_z, output_x, output_y,
            output_z
        },
        stream
    );
}

/// \brief Compute Taylor-approximated winding gradient in device code.
///
/// Far-field children use \p moment; near-field leaves use exact solid-angle derivatives from the
/// canonical triangle records. Stack overflow invokes \p overflow_callback and returns an all-NaN
/// vector when the callback returns. \p bvh and \p moment must be valid and aligned.
/// \p accuracy_scale must be nonnegative; increasing it descends more children.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline gwn_vec3<Real> gwn_winding_gradient_taylor(
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_accessor<Width, Order, Real, Index> const &moment, Real const qx, Real const qy,
    Real const qz, Real const accuracy_scale = Real(2),
    OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_winding_gradient_taylor supports Order 0, 1, and 2."
    );
    GWN_ASSERT(
        moment.is_valid_for(bvh),
        "gwn_winding_gradient_taylor requires aligned BVH and moment data."
    );
    GWN_ASSERT(
        accuracy_scale >= Real(0),
        "gwn_winding_gradient_taylor requires nonnegative accuracy scale."
    );
    return detail::gwn_winding_gradient_taylor_impl<
        Order, Width, Real, Index, StackCapacity, OverflowCallback>(
        bvh, moment, qx, qy, qz, accuracy_scale, overflow_callback
    );
}

/// \brief Compute Taylor-approximated winding gradients for a batch.
///
/// \p accuracy_scale must be nonnegative. Each overflowed query writes an all-NaN vector when its
/// callback returns. \c Config controls the launch block size and traversal stack capacity.
template <
    int Order, gwn_query_batch_config Config = gwn_query_batch_config{}, int Width,
    gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
    requires gwn_traversal_batch_config_value<Config>
[[nodiscard]] gwn_status gwn_compute_winding_gradient_taylor_batch(
    gwn_bvh_object<Width, Real, Index> const &bvh,
    gwn_bvh_moment_object<Width, Order, Real, Index> const &moment,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_x,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_y,
    gwn_device_span<std::type_identity_t<Real> const> const device_query_z,
    gwn_device_span<Real> const device_output_x, gwn_device_span<Real> const device_output_y,
    gwn_device_span<Real> const device_output_z, Real const accuracy_scale = Real(2),
    cudaStream_t const stream = gwn_default_stream(), OverflowCallback const &overflow_callback = {}
) noexcept {
    auto const query_x = detail::gwn_cuda_span_view_impl(device_query_x);
    auto const query_y = detail::gwn_cuda_span_view_impl(device_query_y);
    auto const query_z = detail::gwn_cuda_span_view_impl(device_query_z);
    auto const output_x = detail::gwn_cuda_span_view_impl(device_output_x);
    auto const output_y = detail::gwn_cuda_span_view_impl(device_output_y);
    auto const output_z = detail::gwn_cuda_span_view_impl(device_output_z);
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_compute_winding_gradient_taylor_batch supports Order 0, 1, and 2."
    );
    auto const &bvh_accessor = bvh.accessor();
    auto const &moment_accessor = moment.accessor();
    if (!bvh_accessor.is_valid())
        return gwn_status::invalid_argument("BVH object contains no queryable data.");
    GWN_RETURN_ON_ERROR(
        (detail::gwn_validate_traversal_stack_capacity_impl<Config.stack_capacity>(bvh_accessor))
    );
    if (!moment_accessor.is_valid_for(bvh_accessor))
        return gwn_status::invalid_argument("Moment object is not aligned with the BVH.");
    if (!(accuracy_scale >= Real(0)))
        return gwn_status::invalid_argument("Taylor accuracy scale must be nonnegative.");
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

    return detail::gwn_launch_linear_kernel<Config.block_size>(
        output_x.size(),
        detail::gwn_winding_gradient_taylor_batch_functor<
            Order, Width, Real, Index, Config.stack_capacity, OverflowCallback>{
            bvh_accessor,
            moment_accessor,
            query_x,
            query_y,
            query_z,
            output_x,
            output_y,
            output_z,
            accuracy_scale,
            overflow_callback,
        },
        stream
    );
}

} // namespace gwn
