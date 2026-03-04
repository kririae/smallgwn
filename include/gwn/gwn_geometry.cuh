#pragma once

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

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
    cuda::std::span<std::uint8_t const> tri_boundary_edge_mask{};
    std::uint32_t singular_edge_count{0};

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

template <gwn_index_type Index>
gwn_status gwn_compute_triangle_boundary_edge_mask(
    cuda::std::span<Index const> i0, cuda::std::span<Index const> i1,
    cuda::std::span<Index const> i2, cuda::std::span<std::uint8_t> out_mask
) noexcept;

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
        stream, accessor.tri_boundary_edge_mask, accessor.tri_i2, accessor.tri_i1, accessor.tri_i0,
        accessor.vertex_nz, accessor.vertex_ny, accessor.vertex_nx, accessor.vertex_z,
        accessor.vertex_y, accessor.vertex_x
    );
    accessor.singular_edge_count = 0;
}

template <gwn_index_type Index> struct gwn_boundary_edge_key {
    Index lo{};
    Index hi{};

    bool operator==(gwn_boundary_edge_key const &other) const noexcept {
        return lo == other.lo && hi == other.hi;
    }
};

template <gwn_index_type Index> struct gwn_boundary_edge_key_hash {
    std::size_t operator()(gwn_boundary_edge_key<Index> const &key) const noexcept {
        std::size_t const h0 = std::hash<Index>{}(key.lo);
        std::size_t const h1 = std::hash<Index>{}(key.hi);
        return h0 ^ (h1 + std::size_t(0x9e3779b97f4a7c15ull) + (h0 << 6) + (h0 >> 2));
    }
};

struct gwn_boundary_edge_info {
    std::uint32_t incident_count{0};
    int orientation_sum{0};
};

template <gwn_index_type Index>
[[nodiscard]] inline gwn_boundary_edge_key<Index>
gwn_make_boundary_edge_key(Index const a, Index const b) noexcept {
    if (a < b)
        return gwn_boundary_edge_key<Index>{a, b};
    return gwn_boundary_edge_key<Index>{b, a};
}

template <gwn_index_type Index>
[[nodiscard]] constexpr int
gwn_boundary_edge_orientation_sign(Index const a, Index const b) noexcept {
    return (a < b) ? 1 : -1;
}

template <gwn_index_type Index>
gwn_status gwn_compute_triangle_boundary_edge_mask_impl(
    cuda::std::span<Index const> const i0, cuda::std::span<Index const> const i1,
    cuda::std::span<Index const> const i2, cuda::std::span<std::uint8_t> const out_mask,
    std::uint32_t *const out_singular_edge_count = nullptr
) noexcept {
    std::size_t const triangle_count = i0.size();
    if (i1.size() != triangle_count || i2.size() != triangle_count)
        return gwn_status::invalid_argument("Triangle SoA spans must have identical lengths.");
    if (out_mask.size() != triangle_count)
        return gwn_status::invalid_argument(
            "Boundary-edge mask output size must match triangle count."
        );

    if (!gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2) ||
        !gwn_span_has_storage(out_mask)) {
        return gwn_status::invalid_argument(
            "Boundary-edge mask spans must use non-null storage when non-empty."
        );
    }
    if (triangle_count == 0) {
        if (out_singular_edge_count != nullptr)
            *out_singular_edge_count = 0;
        return gwn_status::ok();
    }

    std::unordered_map<
        gwn_boundary_edge_key<Index>, gwn_boundary_edge_info, gwn_boundary_edge_key_hash<Index>>
        edge_map;
    edge_map.reserve(triangle_count * 3);

    for (std::size_t tri = 0; tri < triangle_count; ++tri) {
        Index const verts[3] = {i0[tri], i1[tri], i2[tri]};
        for (int edge_id = 0; edge_id < 3; ++edge_id) {
            Index const a = verts[edge_id];
            Index const b = verts[(edge_id + 1) % 3];
            auto const key = gwn_make_boundary_edge_key<Index>(a, b);
            auto &info = edge_map[key];
            ++info.incident_count;
            info.orientation_sum += gwn_boundary_edge_orientation_sign(a, b);
        }
    }

    std::uint32_t singular_edge_count = 0;
    for (auto const &entry : edge_map) {
        gwn_boundary_edge_info const &info = entry.second;
        bool const is_boundary = info.incident_count != 2 || info.orientation_sum != 0;
        if (is_boundary)
            ++singular_edge_count;
    }

    std::fill(out_mask.begin(), out_mask.end(), std::uint8_t(0));
    for (std::size_t tri = 0; tri < triangle_count; ++tri) {
        Index const verts[3] = {i0[tri], i1[tri], i2[tri]};
        std::uint8_t mask = 0;
        for (int edge_id = 0; edge_id < 3; ++edge_id) {
            Index const a = verts[edge_id];
            Index const b = verts[(edge_id + 1) % 3];
            auto const key = gwn_make_boundary_edge_key<Index>(a, b);
            auto const it = edge_map.find(key);
            if (it == edge_map.end())
                return gwn_status::internal_error("Boundary-edge map lookup failed.");
            gwn_boundary_edge_info const &info = it->second;
            bool const is_boundary = info.incident_count != 2 || info.orientation_sum != 0;
            if (is_boundary)
                mask |= static_cast<std::uint8_t>(1u << edge_id);
        }
        out_mask[tri] = mask;
    }

    if (out_singular_edge_count != nullptr)
        *out_singular_edge_count = singular_edge_count;

    return gwn_status::ok();
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
gwn_status gwn_compute_vertex_normals_impl(
    cuda::std::span<Real const> const x, cuda::std::span<Real const> const y,
    cuda::std::span<Real const> const z, cuda::std::span<Index const> const i0,
    cuda::std::span<Index const> const i1, cuda::std::span<Index const> const i2,
    cuda::std::span<Real> const out_nx, cuda::std::span<Real> const out_ny,
    cuda::std::span<Real> const out_nz
) noexcept {
    std::size_t const vertex_count = x.size();
    std::size_t const triangle_count = i0.size();
    if (y.size() != vertex_count || z.size() != vertex_count)
        return gwn_status::invalid_argument("Vertex SoA spans must have identical lengths.");
    if (i1.size() != triangle_count || i2.size() != triangle_count)
        return gwn_status::invalid_argument("Triangle SoA spans must have identical lengths.");
    if (out_nx.size() != vertex_count || out_ny.size() != vertex_count ||
        out_nz.size() != vertex_count) {
        return gwn_status::invalid_argument("Vertex normal output spans must match vertex count.");
    }
    if (!gwn_span_has_storage(x) || !gwn_span_has_storage(y) || !gwn_span_has_storage(z) ||
        !gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2) ||
        !gwn_span_has_storage(out_nx) || !gwn_span_has_storage(out_ny) ||
        !gwn_span_has_storage(out_nz)) {
        return gwn_status::invalid_argument(
            "Vertex normal spans must use non-null storage when non-empty."
        );
    }

    std::fill(out_nx.begin(), out_nx.end(), Real(0));
    std::fill(out_ny.begin(), out_ny.end(), Real(0));
    std::fill(out_nz.begin(), out_nz.end(), Real(0));

    for (std::size_t tri = 0; tri < triangle_count; ++tri) {
        std::size_t const ia = static_cast<std::size_t>(i0[tri]);
        std::size_t const ib = static_cast<std::size_t>(i1[tri]);
        std::size_t const ic = static_cast<std::size_t>(i2[tri]);

        Real const ab_x = x[ib] - x[ia];
        Real const ab_y = y[ib] - y[ia];
        Real const ab_z = z[ib] - z[ia];
        Real const ac_x = x[ic] - x[ia];
        Real const ac_y = y[ic] - y[ia];
        Real const ac_z = z[ic] - z[ia];

        Real const n_x = ab_y * ac_z - ab_z * ac_y;
        Real const n_y = ab_z * ac_x - ab_x * ac_z;
        Real const n_z = ab_x * ac_y - ab_y * ac_x;

        out_nx[ia] += n_x;
        out_ny[ia] += n_y;
        out_nz[ia] += n_z;
        out_nx[ib] += n_x;
        out_ny[ib] += n_y;
        out_nz[ib] += n_z;
        out_nx[ic] += n_x;
        out_ny[ic] += n_y;
        out_nz[ic] += n_z;
    }

    for (std::size_t vi = 0; vi < vertex_count; ++vi) {
        Real const nx = out_nx[vi];
        Real const ny = out_ny[vi];
        Real const nz = out_nz[vi];
        Real const n2 = nx * nx + ny * ny + nz * nz;
        if (!(n2 > Real(0)))
            continue;
        Real const inv_norm = Real(1) / std::sqrt(n2);
        out_nx[vi] *= inv_norm;
        out_ny[vi] *= inv_norm;
        out_nz[vi] *= inv_norm;
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
    std::size_t const max_indexable = static_cast<std::size_t>(std::numeric_limits<Index>::max());
    if (x.size() > max_indexable)
        return gwn_status::invalid_argument("Vertex count exceeds index type capacity.");
    if (i0.size() > max_indexable)
        return gwn_status::invalid_argument("Triangle count exceeds index type capacity.");
    GWN_RETURN_ON_ERROR(gwn_validate_triangle_indices<Index>(i0, i1, i2, x.size()));

    std::vector<std::uint8_t> boundary_edge_mask(i0.size(), std::uint8_t(0));
    std::uint32_t singular_edge_count = 0;
    GWN_RETURN_ON_ERROR(
        gwn_compute_triangle_boundary_edge_mask_impl<Index>(
            i0, i1, i2,
            cuda::std::span<std::uint8_t>(boundary_edge_mask.data(), boundary_edge_mask.size()),
            &singular_edge_count
        )
    );
    std::vector<Real> vertex_nx(x.size(), Real(0));
    std::vector<Real> vertex_ny(x.size(), Real(0));
    std::vector<Real> vertex_nz(x.size(), Real(0));
    gwn_status const vertex_normal_status = gwn_compute_vertex_normals_impl<Real, Index>(
        x, y, z, i0, i1, i2, cuda::std::span<Real>(vertex_nx.data(), vertex_nx.size()),
        cuda::std::span<Real>(vertex_ny.data(), vertex_ny.size()),
        cuda::std::span<Real>(vertex_nz.data(), vertex_nz.size())
    );
    if (!vertex_normal_status.is_ok())
        return vertex_normal_status;

    gwn_geometry_accessor<Real, Index> staging{};
    auto cleanup_staging =
        gwn_make_scope_exit([&]() noexcept { gwn_release_accessor(staging, stream); });

    // Allocate and upload vertices + triangles.
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_x, x.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_y, y.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_z, z.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_nx, vertex_nx.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_ny, vertex_ny.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_nz, vertex_nz.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.tri_i0, i0.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.tri_i1, i1.size(), stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.tri_i2, i2.size(), stream));
    GWN_RETURN_ON_ERROR(
        gwn_allocate_span(staging.tri_boundary_edge_mask, boundary_edge_mask.size(), stream)
    );

    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.vertex_x, x, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.vertex_y, y, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.vertex_z, z, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(
        staging.vertex_nx, cuda::std::span<Real const>(vertex_nx.data(), vertex_nx.size()), stream
    ));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(
        staging.vertex_ny, cuda::std::span<Real const>(vertex_ny.data(), vertex_ny.size()), stream
    ));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(
        staging.vertex_nz, cuda::std::span<Real const>(vertex_nz.data(), vertex_nz.size()), stream
    ));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.tri_i0, i0, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.tri_i1, i1, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.tri_i2, i2, stream));
    GWN_RETURN_ON_ERROR(gwn_copy_h2d(
        staging.tri_boundary_edge_mask,
        cuda::std::span<std::uint8_t const>(boundary_edge_mask.data(), boundary_edge_mask.size()),
        stream
    ));

    gwn_release_accessor(accessor, stream);
    accessor = staging;
    accessor.singular_edge_count = singular_edge_count;
    cleanup_staging.release();
    return gwn_status::ok();
}

} // namespace detail

template <gwn_index_type Index>
gwn_status gwn_compute_triangle_boundary_edge_mask(
    cuda::std::span<Index const> i0, cuda::std::span<Index const> i1,
    cuda::std::span<Index const> i2, cuda::std::span<std::uint8_t> out_mask
) noexcept {
    return detail::gwn_compute_triangle_boundary_edge_mask_impl(i0, i1, i2, out_mask, nullptr);
}

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
    GWN_RETURN_ON_ERROR(
        detail::gwn_upload_accessor(staging.accessor_, x, y, z, i0, i1, i2, stream)
    );

    swap(object, staging);
    return gwn_status::ok();
}
} // namespace gwn
