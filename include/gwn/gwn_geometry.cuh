#pragma once

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <type_traits>
#include <utility>
#include <vector>

#include "gwn_utils.cuh"
#include "detail/gwn_singular_edge_build.cuh"

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

    // Oriented boundary/singular edges: undirected edge keys whose oriented
    // incidence does not cancel to zero across the input triangle soup.
    cuda::std::span<Index const> singular_edge_i0{};
    cuda::std::span<Index const> singular_edge_i1{};

    __host__ __device__ constexpr std::size_t vertex_count() const noexcept {
        return vertex_x.size();
    }
    __host__ __device__ constexpr std::size_t triangle_count() const noexcept {
        return tri_i0.size();
    }
    __host__ __device__ constexpr std::size_t singular_edge_count() const noexcept {
        return singular_edge_i0.size();
    }

    __host__ __device__ constexpr bool is_valid() const noexcept {
        return vertex_x.size() == vertex_y.size() && vertex_x.size() == vertex_z.size() &&
               tri_i0.size() == tri_i1.size() && tri_i0.size() == tri_i2.size() &&
               singular_edge_i0.size() == singular_edge_i1.size() &&
               gwn_span_has_storage(vertex_x) && gwn_span_has_storage(vertex_y) &&
               gwn_span_has_storage(vertex_z) && gwn_span_has_storage(tri_i0) &&
               gwn_span_has_storage(tri_i1) && gwn_span_has_storage(tri_i2) &&
               gwn_span_has_storage(singular_edge_i0) &&
               gwn_span_has_storage(singular_edge_i1);
    }
};

namespace detail {
template <gwn_index_type Index>
struct gwn_host_singular_edge_soa {
    std::vector<Index> i0{};
    std::vector<Index> i1{};
};

template <gwn_index_type Index>
[[nodiscard]] inline gwn_host_singular_edge_soa<Index> gwn_extract_singular_edges(
    cuda::std::span<Index const> const tri_i0, cuda::std::span<Index const> const tri_i1,
    cuda::std::span<Index const> const tri_i2
) {
    struct gwn_edge_key {
        std::uint64_t lo{};
        std::uint64_t hi{};

        bool operator==(gwn_edge_key const &other) const noexcept {
            return lo == other.lo && hi == other.hi;
        }
    };
    struct gwn_edge_key_hash {
        std::size_t operator()(gwn_edge_key const &key) const noexcept {
            std::size_t seed = std::hash<std::uint64_t>{}(key.lo);
            seed ^= std::hash<std::uint64_t>{}(key.hi) + 0x9e3779b97f4a7c15ULL + (seed << 6U) +
                    (seed >> 2U);
            return seed;
        }
    };

    std::unordered_map<gwn_edge_key, int, gwn_edge_key_hash> winding_by_edge{};
    winding_by_edge.reserve(tri_i0.size() * 3);

    auto const to_u64 = [](Index const idx) -> std::uint64_t {
        using unsigned_index = std::make_unsigned_t<Index>;
        return static_cast<std::uint64_t>(static_cast<unsigned_index>(idx));
    };
    auto const add_oriented_edge = [&](Index const a, Index const b) {
        if (a == b)
            return;
        std::uint64_t const ua = to_u64(a);
        std::uint64_t const ub = to_u64(b);
        gwn_edge_key key{};
        int orientation = 0;
        if (ua < ub) {
            key = gwn_edge_key{ua, ub};
            orientation = 1;
        } else {
            key = gwn_edge_key{ub, ua};
            orientation = -1;
        }
        winding_by_edge[key] += orientation;
    };

    for (std::size_t tri = 0; tri < tri_i0.size(); ++tri) {
        Index const a = tri_i0[tri];
        Index const b = tri_i1[tri];
        Index const c = tri_i2[tri];
        add_oriented_edge(a, b);
        add_oriented_edge(b, c);
        add_oriented_edge(c, a);
    }

    std::vector<std::array<std::uint64_t, 2>> singular_edges{};
    singular_edges.reserve(winding_by_edge.size());
    for (auto const &entry : winding_by_edge) {
        if (entry.second == 0)
            continue;
        singular_edges.push_back({entry.first.lo, entry.first.hi});
    }
    std::sort(
        singular_edges.begin(), singular_edges.end(),
        [](std::array<std::uint64_t, 2> const &lhs, std::array<std::uint64_t, 2> const &rhs) {
            if (lhs[0] != rhs[0])
                return lhs[0] < rhs[0];
            return lhs[1] < rhs[1];
        }
    );

    gwn_host_singular_edge_soa<Index> result{};
    result.i0.reserve(singular_edges.size());
    result.i1.reserve(singular_edges.size());
    for (auto const &edge : singular_edges) {
        result.i0.push_back(static_cast<Index>(edge[0]));
        result.i1.push_back(static_cast<Index>(edge[1]));
    }
    return result;
}

template <class... Spans>
void gwn_release_spans(cudaStream_t const stream, Spans &...spans) noexcept {
    (gwn_free_span(spans, stream), ...);
}

template <gwn_real_type Real, gwn_index_type Index>
void gwn_release_accessor(
    gwn_geometry_accessor<Real, Index> &accessor, cudaStream_t const stream
) noexcept {
    gwn_release_spans(
        stream, accessor.singular_edge_i1, accessor.singular_edge_i0, accessor.tri_i2,
        accessor.tri_i1, accessor.tri_i0, accessor.vertex_z, accessor.vertex_y, accessor.vertex_x
    );
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

    // Build singular edges from device-resident triangle indices.
    gwn_device_array<Index> se_i0{};
    gwn_device_array<Index> se_i1{};
    std::size_t se_count = 0;
    GWN_RETURN_ON_ERROR(gwn_build_singular_edges<Index>(
        staging.tri_i0, staging.tri_i1, staging.tri_i2,
        se_i0, se_i1, se_count, stream
    ));

    // Transfer ownership: device_array → accessor const spans.
    staging.singular_edge_i0 = cuda::std::span<Index const>(se_i0.release(), se_count);
    staging.singular_edge_i1 = cuda::std::span<Index const>(se_i1.release(), se_count);

    gwn_release_accessor(accessor, stream);
    accessor = staging;
    cleanup_staging.release();
    return gwn_status::ok();
}

} // namespace detail

/// \brief Owning host-side RAII wrapper for geometry accessor storage.
///
/// \remark `clear()` and destructor release memory on the currently bound stream.
/// \remark The bound stream is updated after successful `upload(..., stream)`.
template <gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
class gwn_geometry_object final : public gwn_noncopyable, public gwn_stream_mixin {
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
        staging.set_stream(stream);
        GWN_RETURN_ON_ERROR(
            detail::gwn_upload_accessor(staging.accessor_, x, y, z, i0, i1, i2, stream)
        );

        swap(*this, staging);
        return gwn_status::ok();
    }

    void clear() noexcept { detail::gwn_release_accessor(accessor_, stream()); }

    void clear(cudaStream_t const clear_stream) noexcept {
        cudaStream_t const release_stream = stream();
        detail::gwn_release_accessor(accessor_, release_stream);
        set_stream(clear_stream);
    }

    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }
    [[nodiscard]] std::size_t vertex_count() const noexcept { return accessor_.vertex_count(); }
    [[nodiscard]] std::size_t triangle_count() const noexcept { return accessor_.triangle_count(); }
    [[nodiscard]] std::size_t singular_edge_count() const noexcept {
        return accessor_.singular_edge_count();
    }

    friend void swap(gwn_geometry_object &lhs, gwn_geometry_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    accessor_type accessor_{};
};
} // namespace gwn
