#pragma once

/// \file gwn_geometry.cuh
/// \brief Device geometry views, upload, position update, and owning storage.

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstddef>

#include "gwn_utils.cuh"

namespace gwn {

/// \brief Mutable device view of vertex positions and oriented triangle indices.
///
/// Position and index components use SoA storage. The accessor does not own its spans and remains
/// valid only while the originating \ref gwn_geometry_object keeps its storage alive.
template <gwn_real_type Real, gwn_index_type Index = std::uint32_t> struct gwn_geometry_accessor {
    using real_type = Real;
    using index_type = Index;

    cuda::std::span<Real> vertex_x{}; ///< Vertex X coordinates.
    cuda::std::span<Real> vertex_y{}; ///< Vertex Y coordinates.
    cuda::std::span<Real> vertex_z{}; ///< Vertex Z coordinates.

    cuda::std::span<Index> tri_i0{}; ///< First vertex index of each oriented triangle.
    cuda::std::span<Index> tri_i1{}; ///< Second vertex index of each oriented triangle.
    cuda::std::span<Index> tri_i2{}; ///< Third vertex index of each oriented triangle.

    /// \brief Return the number of vertices represented by the position spans.
    __host__ __device__ constexpr std::size_t vertex_count() const noexcept {
        return vertex_x.size();
    }

    /// \brief Return the number of triangles represented by the index spans.
    __host__ __device__ constexpr std::size_t triangle_count() const noexcept {
        return tri_i0.size();
    }

    /// \brief Return whether every SoA group has matching sizes and valid non-empty storage.
    __host__ __device__ constexpr bool is_valid() const noexcept {
        return vertex_x.size() == vertex_y.size() && vertex_x.size() == vertex_z.size() &&
               tri_i0.size() == tri_i1.size() && tri_i0.size() == tri_i2.size() &&
               gwn_span_has_storage(vertex_x) && gwn_span_has_storage(vertex_y) &&
               gwn_span_has_storage(vertex_z) && gwn_span_has_storage(tri_i0) &&
               gwn_span_has_storage(tri_i1) && gwn_span_has_storage(tri_i2);
    }
};

template <gwn_real_type Real, gwn_index_type Index> class gwn_geometry_object;

namespace detail {

template <gwn_real_type Real, gwn_index_type Index>
void gwn_release_accessor(
    gwn_geometry_accessor<Real, Index> &accessor, cudaStream_t stream
) noexcept;

} // namespace detail

/// \brief Upload host geometry to device buffers and validate triangle indices.
///
/// On success, \p object is atomically replaced with a fully initialized
/// geometry accessor bound to \p stream.
template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_upload_geometry(
    gwn_geometry_object<Real, Index> &object, cuda::std::span<Real const> const x,
    cuda::std::span<Real const> const y, cuda::std::span<Real const> const z,
    cuda::std::span<Index const> const i0, cuda::std::span<Index const> const i1,
    cuda::std::span<Index const> const i2, cudaStream_t stream = gwn_default_stream()
) noexcept;

/// \brief Update vertex positions of an existing geometry object.
///
/// Enqueues host-to-device position copies on \p stream. Triangle indices are preserved.
///
/// \remark \c gwn_status::ok() means the CUDA work was enqueued. It is not a
///         stream completion signal.
/// \remark Host span lifetime follows \c cudaMemcpyAsync. Keep host memory
///         alive until CUDA has read it.
/// \remark After input validation succeeds, \p object is bound to \p stream before copying starts.
///         A CUDA error may leave some position components updated; the object retains ownership
///         so the caller can clear it or replace it with \c gwn_upload_geometry.
template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_update_geometry(
    gwn_geometry_object<Real, Index> &object, cuda::std::span<Real const> const x,
    cuda::std::span<Real const> const y, cuda::std::span<Real const> const z,
    cudaStream_t stream = gwn_default_stream()
) noexcept;

/// \brief Owning host-side RAII wrapper for geometry accessor storage.
///
/// \remark `clear()` and destructor release memory on the currently bound stream.
/// \remark The bound stream is updated after successful `gwn_upload_geometry(..., stream)` and
///         `gwn_update_geometry(..., stream)`.
/// \remark `has_data()` reports whether the object owns any geometry span.
template <gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
class gwn_geometry_object final : public gwn_noncopyable, public gwn_stream_mixin {
public:
    using real_type = Real;
    using index_type = Index;
    using accessor_type = gwn_geometry_accessor<Real, Index>;

    /// \brief Construct an empty geometry object bound to the default stream.
    gwn_geometry_object() = default;

    /// \brief Move geometry storage and stream binding from \p other.
    gwn_geometry_object(gwn_geometry_object &&other) noexcept { swap(*this, other); }

    /// \brief Replace this object through move-and-swap assignment.
    gwn_geometry_object &operator=(gwn_geometry_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    /// \brief Release owned device storage on the currently bound stream.
    ~gwn_geometry_object() { clear(); }

    /// \brief Release owned device storage on the currently bound stream.
    void clear() noexcept { detail::gwn_release_accessor(accessor_, stream()); }

    /// \brief Return the mutable device accessor.
    [[nodiscard]] accessor_type &accessor() noexcept { return accessor_; }

    /// \brief Return the device accessor.
    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }

    /// \brief Return the uploaded vertex count.
    [[nodiscard]] std::size_t vertex_count() const noexcept { return accessor_.vertex_count(); }

    /// \brief Return the uploaded triangle count.
    [[nodiscard]] std::size_t triangle_count() const noexcept { return accessor_.triangle_count(); }

    /// \brief Return whether the object owns any geometry span.
    [[nodiscard]] bool has_data() const noexcept {
        return accessor_.vertex_count() != 0 || accessor_.triangle_count() != 0;
    }

    /// \brief Exchange geometry storage and stream bindings.
    friend void swap(gwn_geometry_object &lhs, gwn_geometry_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    accessor_type accessor_{};
};

} // namespace gwn

#include "detail/gwn_geometry_impl.cuh"
