#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "gwn_bvh_status_helpers.cuh"
#include "gwn_device_array.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {
namespace detail {

template <gwn_real_type Real, gwn_index_type Index>
void gwn_release_accessor(
    gwn_geometry_accessor<Real, Index> &accessor, cudaStream_t const stream
) noexcept {
    gwn_free_span(accessor.tri_i2, stream);
    gwn_free_span(accessor.tri_i1, stream);
    gwn_free_span(accessor.tri_i0, stream);
    gwn_free_span(accessor.vertex_z, stream);
    gwn_free_span(accessor.vertex_y, stream);
    gwn_free_span(accessor.vertex_x, stream);
}

/// \brief Mark whether any triangle index falls outside the uploaded vertex range.
template <gwn_index_type Index> struct gwn_validate_triangle_indices_functor {
    cuda::std::span<Index const> i0{};
    cuda::std::span<Index const> i1{};
    cuda::std::span<Index const> i2{};
    std::size_t vertex_count{0};
    cuda::std::span<std::uint32_t> invalid_flag{};

    __device__ void operator()(std::size_t const triangle_id) const {
        Index const a = i0[triangle_id];
        Index const b = i1[triangle_id];
        Index const c = i2[triangle_id];

        if (gwn_index_in_bounds(a, vertex_count) && gwn_index_in_bounds(b, vertex_count) &&
            gwn_index_in_bounds(c, vertex_count)) {
            return;
        }
        // Many invalid triangles may race to report the same condition. The flag carries no
        // triangle identity, so an idempotent atomic write is sufficient.
        atomicExch(&invalid_flag[0], std::uint32_t(1));
    }
};

/// \brief Validate uploaded triangle indices and publish one host-visible failure status.
template <gwn_index_type Index>
void gwn_validate_triangle_indices_device_impl(
    cuda::std::span<Index const> const i0, cuda::std::span<Index const> const i1,
    cuda::std::span<Index const> const i2, std::size_t const vertex_count,
    cudaStream_t const stream = cudaStreamLegacy
) {
    std::size_t const triangle_count = i0.size();
    if (i1.size() != triangle_count || i2.size() != triangle_count)
        throw std::invalid_argument("Triangle SoA spans must have identical lengths.");
    if (!gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2)) {
        throw std::invalid_argument(
            "Triangle index spans must use non-null storage when non-empty."
        );
    }
    if (triangle_count == 0)
        return;
    if (vertex_count == 0)
        throw std::invalid_argument("Triangle indices require non-empty vertex arrays.");

    gwn_device_array<std::uint32_t> invalid_flag(stream);
    invalid_flag.resize(1, stream);
    invalid_flag.zero(stream);

    constexpr int k_block_size = k_gwn_default_block_size;
    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_block_size>(
            triangle_count,
            gwn_validate_triangle_indices_functor<Index>{
                i0,
                i1,
                i2,
                vertex_count,
                invalid_flag.span(),
            },
            stream
        )
    );

    // Upload accepts host spans but validation runs after their asynchronous copies reach device
    // storage. Synchronizing the one-word result also guarantees that no invalid accessor is
    // published while validation is still pending.
    std::uint32_t host_invalid_flag = 0;
    invalid_flag.copy_to_host(cuda::std::span<std::uint32_t>(&host_invalid_flag, 1), stream);
    gwn_throw_status_error(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

    if (host_invalid_flag != 0)
        throw std::invalid_argument("Triangle indices must be in [0, vertex_count).");
}

/// \brief Upload into staging storage and replace an accessor only after index validation.
template <gwn_real_type Real, gwn_index_type Index>
void gwn_upload_accessor(
    gwn_geometry_accessor<Real, Index> &accessor, cuda::std::span<Real const> const x,
    cuda::std::span<Real const> const y, cuda::std::span<Real const> const z,
    cuda::std::span<Index const> const i0, cuda::std::span<Index const> const i1,
    cuda::std::span<Index const> const i2, cudaStream_t const stream
) {
    auto const release = [&](gwn_geometry_accessor<Real, Index> &target, cudaStream_t s) noexcept {
        gwn_release_accessor(target, s);
    };

    auto const build = [&](gwn_geometry_accessor<Real, Index> &staging) {
        if (x.size() != y.size() || x.size() != z.size())
            throw std::invalid_argument("Vertex SoA spans must have identical lengths.");
        if (i0.size() != i1.size() || i0.size() != i2.size())
            throw std::invalid_argument("Triangle SoA spans must have identical lengths.");
        if (x.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
            throw std::invalid_argument("Vertex count exceeds index type capacity.");
        if (i0.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
            throw std::invalid_argument("Triangle count exceeds index type capacity.");
        if (!gwn_span_has_storage(x) || !gwn_span_has_storage(y) || !gwn_span_has_storage(z) ||
            !gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2)) {
            throw std::invalid_argument("Geometry spans must use non-null storage when non-empty.");
        }

        gwn_allocate_span(staging.vertex_x, x.size(), stream);
        gwn_allocate_span(staging.vertex_y, y.size(), stream);
        gwn_allocate_span(staging.vertex_z, z.size(), stream);
        gwn_allocate_span(staging.tri_i0, i0.size(), stream);
        gwn_allocate_span(staging.tri_i1, i1.size(), stream);
        gwn_allocate_span(staging.tri_i2, i2.size(), stream);

        gwn_copy_h2d(staging.vertex_x, x, stream);
        gwn_copy_h2d(staging.vertex_y, y, stream);
        gwn_copy_h2d(staging.vertex_z, z, stream);
        gwn_copy_h2d(staging.tri_i0, i0, stream);
        gwn_copy_h2d(staging.tri_i1, i1, stream);
        gwn_copy_h2d(staging.tri_i2, i2, stream);

        // Validate the copied indices before staging replaces the old accessor. Failure therefore
        // preserves the previously queryable geometry and its stream binding.
        gwn_validate_triangle_indices_device_impl<Index>(
            staging.tri_i0, staging.tri_i1, staging.tri_i2, x.size(), stream
        );
    };

    gwn_replace_accessor_with_staging(accessor, release, build, stream);
}

/// \brief Enqueue same-size host position updates and bind their owning object for cleanup.
template <gwn_real_type Real, gwn_index_type Index>
void gwn_update_geometry_impl(
    gwn_geometry_object<Real, Index> &object, cuda::std::span<Real const> const x,
    cuda::std::span<Real const> const y, cuda::std::span<Real const> const z,
    cudaStream_t const stream
) {
    auto &accessor = object.accessor();
    if (!accessor.is_valid())
        throw std::invalid_argument("Geometry accessor is invalid for update.");
    if (x.size() != y.size() || x.size() != z.size())
        throw std::invalid_argument("Vertex SoA spans must have identical lengths.");
    if (x.size() != accessor.vertex_count())
        throw std::invalid_argument("Updated vertex count must match geometry.");
    if (!gwn_span_has_storage(x) || !gwn_span_has_storage(y) || !gwn_span_has_storage(z)) {
        throw std::invalid_argument(
            "Geometry update spans must use non-null storage when non-empty."
        );
    }

    // Every user-recoverable input error has been handled. From the first copy onward the caller
    // must be able to release all geometry storage after this stream's attempted mutation.
    object.set_stream(stream);
    gwn_copy_h2d(accessor.vertex_x, x, stream);
    gwn_copy_h2d(accessor.vertex_y, y, stream);
    // Geometry owns only positions and triangle indices. BVH bounds, triangle records, and moments
    // remain explicit refit steps, so the final copy completes the whole geometry update.
    gwn_copy_h2d(accessor.vertex_z, z, stream);
}

} // namespace detail

template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_upload_geometry(
    gwn_geometry_object<Real, Index> &object, gwn_host_span<Real const> const x,
    gwn_host_span<Real const> const y, gwn_host_span<Real const> const z,
    gwn_host_span<Index const> const i0, gwn_host_span<Index const> const i1,
    gwn_host_span<Index const> const i2, cudaStream_t const stream
) noexcept {
    auto const x_span = detail::gwn_cuda_span_view_impl(x);
    auto const y_span = detail::gwn_cuda_span_view_impl(y);
    auto const z_span = detail::gwn_cuda_span_view_impl(z);
    auto const i0_span = detail::gwn_cuda_span_view_impl(i0);
    auto const i1_span = detail::gwn_cuda_span_view_impl(i1);
    auto const i2_span = detail::gwn_cuda_span_view_impl(i2);
    return detail::gwn_try_translate_status("gwn_upload_geometry", [&]() {
        gwn_geometry_object<Real, Index> staging;
        detail::gwn_upload_accessor(
            staging.accessor(), x_span, y_span, z_span, i0_span, i1_span, i2_span, stream
        );
        staging.set_stream(stream);
        swap(object, staging);
        // staging now owns the replaced buffers and their original stream binding. Destruction
        // therefore cannot free them ahead of work already queued on the old stream.
    });
}

template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_update_geometry(
    gwn_geometry_object<Real, Index> &object, gwn_host_span<Real const> const x,
    gwn_host_span<Real const> const y, gwn_host_span<Real const> const z, cudaStream_t const stream
) noexcept {
    auto const x_span = detail::gwn_cuda_span_view_impl(x);
    auto const y_span = detail::gwn_cuda_span_view_impl(y);
    auto const z_span = detail::gwn_cuda_span_view_impl(z);
    return detail::gwn_try_translate_status("gwn_update_geometry", [&]() {
        detail::gwn_update_geometry_impl(object, x_span, y_span, z_span, stream);
    });
}

} // namespace gwn
