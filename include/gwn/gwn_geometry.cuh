#pragma once

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "gwn_utils.hpp"

namespace gwn {

template <class Real, class Index = std::int64_t> struct gwn_geometry_accessor {
    using real_type = Real;
    using index_type = Index;

    cuda::std::span<Real const> vertex_x{};
    cuda::std::span<Real const> vertex_y{};
    cuda::std::span<Real const> vertex_z{};

    cuda::std::span<Index const> tri_i0{};
    cuda::std::span<Index const> tri_i1{};
    cuda::std::span<Index const> tri_i2{};

    __host__ __device__ constexpr std::size_t vertex_count() const noexcept {
        return vertex_x.size();
    }
    __host__ __device__ constexpr std::size_t triangle_count() const noexcept {
        return tri_i0.size();
    }

    __host__ __device__ constexpr bool is_valid() const noexcept {
        return vertex_x.size() == vertex_y.size() && vertex_x.size() == vertex_z.size() &&
               tri_i0.size() == tri_i1.size() && tri_i0.size() == tri_i2.size() &&
               gwn_span_has_storage(vertex_x) && gwn_span_has_storage(vertex_y) &&
               gwn_span_has_storage(vertex_z) && gwn_span_has_storage(tri_i0) &&
               gwn_span_has_storage(tri_i1) && gwn_span_has_storage(tri_i2);
    }
};

namespace detail {

template <class T>
[[nodiscard]] constexpr T *gwn_mutable_data(cuda::std::span<T const> const span) noexcept {
    return const_cast<T *>(span.data());
}

template <class T>
gwn_status gwn_upload_span(
    cuda::std::span<T const> &dst, cuda::std::span<T const> src, cudaStream_t const stream
) {
    if (dst.data() != nullptr) {
        GWN_RETURN_ON_ERROR(gwn_cuda_free(gwn_mutable_data(dst), stream));
        dst = {};
    }

    dst = {};
    if (src.empty())
        return gwn_status::ok();

    void *device_ptr = nullptr;
    GWN_RETURN_ON_ERROR(gwn_cuda_malloc(&device_ptr, src.size_bytes(), stream));
    auto cleanup_device_ptr =
        gwn_make_scope_exit([&]() noexcept { (void)gwn_cuda_free(device_ptr, stream); });

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemcpyAsync(device_ptr, src.data(), src.size_bytes(), cudaMemcpyHostToDevice, stream)
    ));

    dst = cuda::std::span<T const>(static_cast<T const *>(device_ptr), src.size());
    cleanup_device_ptr.release();
    return gwn_status::ok();
}

template <class T>
gwn_status gwn_allocate_span(
    cuda::std::span<T const> &dst, std::size_t const count, cudaStream_t const stream
) {
    if (dst.data() != nullptr) {
        GWN_RETURN_ON_ERROR(gwn_cuda_free(gwn_mutable_data(dst), stream));
        dst = {};
    }

    if (count == 0)
        return gwn_status::ok();

    void *raw_ptr = nullptr;
    GWN_RETURN_ON_ERROR(gwn_cuda_malloc(&raw_ptr, count * sizeof(T), stream));

    dst = cuda::std::span<T const>(static_cast<T const *>(raw_ptr), count);
    return gwn_status::ok();
}

template <class T>
gwn_status gwn_copy_device_to_span(
    cuda::std::span<T const> &dst, T const *src, std::size_t const count, cudaStream_t const stream
) {
    GWN_RETURN_ON_ERROR(gwn_allocate_span(dst, count, stream));
    if (count == 0)
        return gwn_status::ok();

    auto cleanup_dst = gwn_make_scope_exit([&]() noexcept {
        (void)gwn_cuda_free(gwn_mutable_data(dst), stream);
        dst = {};
    });

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        gwn_mutable_data(dst), src, count * sizeof(T), cudaMemcpyDeviceToDevice, stream
    )));

    cleanup_dst.release();
    return gwn_status::ok();
}

template <class T>
void gwn_release_span(cuda::std::span<T const> &span_view, cudaStream_t const stream) noexcept {
    if (span_view.data() != nullptr) {
        gwn_status const status = gwn_cuda_free(gwn_mutable_data(span_view), stream);
        if (!status.is_ok())
            GWN_HANDLE_STATUS_FAIL(status);
        span_view = {};
    }
}

template <class... Spans>
void gwn_release_spans(cudaStream_t const stream, Spans &...spans) noexcept {
    (gwn_release_span(spans, stream), ...);
}

template <class Real, class Index>
void gwn_release_accessor(
    gwn_geometry_accessor<Real, Index> &accessor, cudaStream_t const stream
) noexcept {
    gwn_release_spans(
        stream, accessor.tri_i2, accessor.tri_i1, accessor.tri_i0, accessor.vertex_z,
        accessor.vertex_y, accessor.vertex_x
    );
}

template <class Real, class Index>
gwn_status gwn_upload_accessor(
    gwn_geometry_accessor<Real, Index> &accessor, cuda::std::span<Real const> x,
    cuda::std::span<Real const> y, cuda::std::span<Real const> z, cuda::std::span<Index const> i0,
    cuda::std::span<Index const> i1, cuda::std::span<Index const> i2, cudaStream_t const stream
) {
    if (x.size() != y.size() || x.size() != z.size())
        return gwn_status::invalid_argument("Vertex SoA spans must have identical lengths.");
    if (i0.size() != i1.size() || i0.size() != i2.size())
        return gwn_status::invalid_argument("Triangle SoA spans must have identical lengths.");

    auto cleanup_accessor =
        gwn_make_scope_exit([&]() noexcept { gwn_release_accessor(accessor, stream); });

    GWN_RETURN_ON_ERROR(gwn_upload_span(accessor.vertex_x, x, stream));
    GWN_RETURN_ON_ERROR(gwn_upload_span(accessor.vertex_y, y, stream));
    GWN_RETURN_ON_ERROR(gwn_upload_span(accessor.vertex_z, z, stream));
    GWN_RETURN_ON_ERROR(gwn_upload_span(accessor.tri_i0, i0, stream));
    GWN_RETURN_ON_ERROR(gwn_upload_span(accessor.tri_i1, i1, stream));
    GWN_RETURN_ON_ERROR(gwn_upload_span(accessor.tri_i2, i2, stream));

    cleanup_accessor.release();
    return gwn_status::ok();
}

} // namespace detail

template <class Real = float, class Index = std::int64_t>
class gwn_geometry_object final : public gwn_noncopyable {
    static_assert(std::is_floating_point_v<Real>, "Real must be a floating-point type.");
    static_assert(std::is_integral_v<Index>, "Index must be an integral type.");

public:
    using real_type = Real;
    using index_type = Index;
    using accessor_type = gwn_geometry_accessor<Real, Index>;

    gwn_geometry_object() = default;

    gwn_geometry_object(
        cuda::std::span<Real const> const x, cuda::std::span<Real const> const y,
        cuda::std::span<Real const> const z, cuda::std::span<Index const> const i0,
        cuda::std::span<Index const> const i1, cuda::std::span<Index const> const i2,
        cudaStream_t const stream = gwn_default_stream()
    ) {
        gwn_throw_if_error(upload(x, y, z, i0, i1, i2, stream));
    }

    gwn_geometry_object(gwn_geometry_object &&other) noexcept { swap(*this, other); }

    gwn_geometry_object &operator=(gwn_geometry_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    ~gwn_geometry_object() { clear(); }

    gwn_status upload(
        cuda::std::span<Real const> const x, cuda::std::span<Real const> const y,
        cuda::std::span<Real const> const z, cuda::std::span<Index const> const i0,
        cuda::std::span<Index const> const i1, cuda::std::span<Index const> const i2,
        cudaStream_t const stream = gwn_default_stream()
    ) noexcept {
        gwn_geometry_object staging;
        GWN_RETURN_ON_ERROR(
            detail::gwn_upload_accessor(staging.accessor_, x, y, z, i0, i1, i2, stream)
        );

        swap(*this, staging);
        return gwn_status::ok();
    }

    void clear(cudaStream_t const stream = gwn_default_stream()) noexcept {
        detail::gwn_release_accessor(accessor_, stream);
    }

    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }
    [[nodiscard]] std::size_t vertex_count() const noexcept { return accessor_.vertex_count(); }
    [[nodiscard]] std::size_t triangle_count() const noexcept { return accessor_.triangle_count(); }

    friend void swap(gwn_geometry_object &lhs, gwn_geometry_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
    }

private:
    accessor_type accessor_{};
};

} // namespace gwn
