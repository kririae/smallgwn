#pragma once

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "gwn_utils.cuh"

namespace gwn {

template <gwn_real_type Real, gwn_index_type Index = std::uint32_t> struct gwn_geometry_accessor {
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

template <gwn_real_type Real, gwn_index_type Index> class gwn_geometry_object;

template <gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_upload_geometry(
    gwn_geometry_object<Real, Index> &object, cuda::std::span<Real const> x,
    cuda::std::span<Real const> y, cuda::std::span<Real const> z, cuda::std::span<Index const> i0,
    cuda::std::span<Index const> i1, cuda::std::span<Index const> i2,
    cudaStream_t const stream = gwn_default_stream()
) noexcept;

namespace detail {

template <class... Spans>
void gwn_release_spans(cudaStream_t const stream, Spans &...spans) noexcept {
    (gwn_free_span(spans, stream), ...);
}

template <gwn_real_type Real, gwn_index_type Index>
void gwn_release_accessor(
    gwn_geometry_accessor<Real, Index> &accessor, cudaStream_t const stream
) noexcept {
    gwn_release_spans(
        stream, accessor.tri_i2, accessor.tri_i1, accessor.tri_i0, accessor.vertex_z,
        accessor.vertex_y, accessor.vertex_x
    );
}

template <gwn_index_type Index>
gwn_status gwn_validate_triangle_indices(
    cuda::std::span<Index const> const i0, cuda::std::span<Index const> const i1,
    cuda::std::span<Index const> const i2, std::size_t const vertex_count
) noexcept {
    std::size_t const triangle_count = i0.size();
    if (triangle_count == 0)
        return gwn_status::ok();

    if (vertex_count == 0)
        return gwn_status::invalid_argument("Triangle indices require non-empty vertex arrays.");

    for (std::size_t triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
        if (!gwn_index_in_bounds(i0[triangle_id], vertex_count) ||
            !gwn_index_in_bounds(i1[triangle_id], vertex_count) ||
            !gwn_index_in_bounds(i2[triangle_id], vertex_count)) {
            return gwn_status::invalid_argument("Triangle indices must be in [0, vertex_count).");
        }
    }

    return gwn_status::ok();
}

template <gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_upload_accessor(
    gwn_geometry_accessor<Real, Index> &accessor, cuda::std::span<Real const> x,
    cuda::std::span<Real const> y, cuda::std::span<Real const> z, cuda::std::span<Index const> i0,
    cuda::std::span<Index const> i1, cuda::std::span<Index const> i2, cudaStream_t const stream
) {
    if (x.size() != y.size() || x.size() != z.size())
        return gwn_status::invalid_argument("Vertex SoA spans must have identical lengths.");
    if (i0.size() != i1.size() || i0.size() != i2.size())
        return gwn_status::invalid_argument("Triangle SoA spans must have identical lengths.");
    if (!gwn_span_has_storage(x) || !gwn_span_has_storage(y) || !gwn_span_has_storage(z) ||
        !gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2)) {
        return gwn_status::invalid_argument(
            "Geometry spans must use non-null storage when non-empty."
        );
    }
    GWN_RETURN_ON_ERROR(gwn_validate_triangle_indices<Index>(i0, i1, i2, x.size()));

    gwn_geometry_accessor<Real, Index> staging{};
    auto cleanup_staging =
        gwn_make_scope_exit([&]() noexcept { gwn_release_accessor(staging, stream); });

    // Allocate and upload vertices + triangles.
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_x, x.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_y, y.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_z, z.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.tri_i0, i0.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.tri_i1, i1.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.tri_i2, i2.size(), stream));

    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.vertex_x, x, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.vertex_y, y, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.vertex_z, z, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.tri_i0, i0, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.tri_i1, i1, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.tri_i2, i2, stream));

    gwn_release_accessor(accessor, stream);
    accessor = staging;
    cleanup_staging.release();
    return gwn_status::ok();
}

} // namespace detail

/// \brief Owning host-side RAII wrapper for geometry accessor storage.
///
/// \remark `clear()` and destructor release memory on the currently bound stream.
/// \remark The bound stream is updated after successful `gwn_upload_geometry(..., stream)`.
template <gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
class gwn_geometry_object final : public gwn_noncopyable, public gwn_stream_mixin {
public:
    using real_type = Real;
    using index_type = Index;
    using accessor_type = gwn_geometry_accessor<Real, Index>;

    gwn_geometry_object() = default;

    gwn_geometry_object(gwn_geometry_object &&other) noexcept { swap(*this, other); }

    gwn_geometry_object &operator=(gwn_geometry_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    ~gwn_geometry_object() { clear(); }

    void clear() noexcept { detail::gwn_release_accessor(accessor_, stream()); }

    void clear(cudaStream_t const clear_stream) noexcept {
        cudaStream_t const release_stream = stream();
        detail::gwn_release_accessor(accessor_, release_stream);
        set_stream(clear_stream);
    }

    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }
    [[nodiscard]] std::size_t vertex_count() const noexcept { return accessor_.vertex_count(); }
    [[nodiscard]] std::size_t triangle_count() const noexcept { return accessor_.triangle_count(); }

    friend void swap(gwn_geometry_object &lhs, gwn_geometry_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    template <gwn_real_type R, gwn_index_type I>
    friend gwn_status gwn_upload_geometry(
        gwn_geometry_object<R, I> &object, cuda::std::span<R const> x, cuda::std::span<R const> y,
        cuda::std::span<R const> z, cuda::std::span<I const> i0, cuda::std::span<I const> i1,
        cuda::std::span<I const> i2, cudaStream_t const stream
    ) noexcept;

    accessor_type accessor_{};
};

template <gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_upload_geometry(
    gwn_geometry_object<Real, Index> &object, cuda::std::span<Real const> x,
    cuda::std::span<Real const> y, cuda::std::span<Real const> z, cuda::std::span<Index const> i0,
    cuda::std::span<Index const> i1, cuda::std::span<Index const> i2, cudaStream_t const stream
) noexcept {
    gwn_geometry_object<Real, Index> staging;
    staging.set_stream(stream);
    GWN_RETURN_ON_ERROR(detail::gwn_upload_accessor(staging.accessor_, x, y, z, i0, i1, i2, stream));

    swap(object, staging);
    return gwn_status::ok();
}
} // namespace gwn
