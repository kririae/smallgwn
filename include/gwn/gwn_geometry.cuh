#pragma once

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "gwn_utils.cuh"

namespace gwn {

template <gwn_real_type Real, gwn_index_type Index = std::uint32_t> struct gwn_geometry_accessor {
    using real_type = Real;
    using index_type = Index;

    cuda::std::span<Real> vertex_x{};
    cuda::std::span<Real> vertex_y{};
    cuda::std::span<Real> vertex_z{};
    cuda::std::span<Real> vertex_nx{};
    cuda::std::span<Real> vertex_ny{};
    cuda::std::span<Real> vertex_nz{};

    cuda::std::span<Index> tri_i0{};
    cuda::std::span<Index> tri_i1{};
    cuda::std::span<Index> tri_i2{};
    cuda::std::span<std::uint8_t> tri_boundary_edge_mask{};
    Index singular_edge_count{0};

    __host__ __device__ constexpr std::size_t vertex_count() const noexcept {
        return vertex_x.size();
    }

    __host__ __device__ constexpr std::size_t triangle_count() const noexcept {
        return tri_i0.size();
    }

    __host__ __device__ constexpr bool has_singular_edges() const noexcept {
        return singular_edge_count != 0;
    }

    __host__ __device__ constexpr bool is_valid() const noexcept {
        bool const boundary_mask_size_ok =
            tri_boundary_edge_mask.size() == 0 || tri_boundary_edge_mask.size() == tri_i0.size();
        bool const vertex_normal_size_ok =
            (vertex_nx.size() == 0 && vertex_ny.size() == 0 && vertex_nz.size() == 0) ||
            (vertex_nx.size() == vertex_count() && vertex_ny.size() == vertex_count() &&
             vertex_nz.size() == vertex_count());

        return vertex_x.size() == vertex_y.size() && vertex_x.size() == vertex_z.size() &&
               tri_i0.size() == tri_i1.size() && tri_i0.size() == tri_i2.size() &&
               gwn_span_has_storage(vertex_x) && gwn_span_has_storage(vertex_y) &&
               gwn_span_has_storage(vertex_z) && gwn_span_has_storage(vertex_nx) &&
               gwn_span_has_storage(vertex_ny) && gwn_span_has_storage(vertex_nz) &&
               gwn_span_has_storage(tri_i0) && gwn_span_has_storage(tri_i1) &&
               gwn_span_has_storage(tri_i2) && vertex_normal_size_ok && boundary_mask_size_ok &&
               gwn_span_has_storage(tri_boundary_edge_mask);
    }
};

template <gwn_real_type Real, gwn_index_type Index> class gwn_geometry_object;

namespace detail {
template <gwn_real_type Real, gwn_index_type Index>
void gwn_release_accessor(
    gwn_geometry_accessor<Real, Index> &accessor, cudaStream_t stream
) noexcept;
} // namespace detail

template <gwn_index_type Index>
/// \brief Compute per-triangle boundary-edge masks from triangle index SoA.
///
/// Each output mask uses 3 low bits:
/// - bit 0: edge (i0, i1)
/// - bit 1: edge (i1, i2)
/// - bit 2: edge (i2, i0)
///
/// A bit is set when the corresponding edge is singular/boundary under the
/// manifold-orientation test used by robust tracing.
gwn_status gwn_compute_triangle_boundary_edge_mask(
    cuda::std::span<Index const> i0, cuda::std::span<Index const> i1,
    cuda::std::span<Index const> i2, cuda::std::span<std::uint8_t> out_mask
) noexcept;

template <gwn_real_type Real, gwn_index_type Index>
/// \brief Upload host geometry to device buffers and precompute geometry caches.
///
/// Upload performs:
/// - triangle-index validation,
/// - boundary-edge mask + singular-edge count preprocessing,
/// - vertex-normal accumulation/normalization.
///
/// On success, \p object is atomically replaced with a fully initialized
/// geometry accessor bound to \p stream.
gwn_status gwn_upload_geometry(
    gwn_geometry_object<Real, Index> &object, cuda::std::span<Real const> x,
    cuda::std::span<Real const> y, cuda::std::span<Real const> z, cuda::std::span<Index const> i0,
    cuda::std::span<Index const> i1, cuda::std::span<Index const> i2,
    cudaStream_t stream = gwn_default_stream()
) noexcept;

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

    void clear(cudaStream_t clear_stream) noexcept {
        cudaStream_t const release_stream = stream();
        detail::gwn_release_accessor(accessor_, release_stream);
        set_stream(clear_stream);
    }

    [[nodiscard]] accessor_type &accessor() noexcept { return accessor_; }
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
        cuda::std::span<I const> i2, cudaStream_t stream
    ) noexcept;

    accessor_type accessor_{};
};

} // namespace gwn

#include "detail/gwn_geometry_impl.cuh"
