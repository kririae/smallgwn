#pragma once

#include <cuda/std/utility>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cub/cub.cuh>

#include "../gwn_kernel_utils.cuh"
#include "gwn_bvh_status_helpers.cuh"

namespace gwn {
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
    accessor.singular_edge_count = Index(0);
}

template <gwn_index_type Index> struct gwn_boundary_edge_payload_hi {
    Index lo{};
    std::uint64_t tri_edge{};
    int orientation{0};
};

template <gwn_index_type Index> struct gwn_boundary_edge_payload_lo {
    Index hi{};
    std::uint64_t tri_edge{};
    int orientation{0};
};

template <gwn_index_type Index>
[[nodiscard]] __host__ __device__ inline std::uint64_t
encode_triangle_edge(std::size_t const triangle_id, int const local_edge) noexcept {
    return (static_cast<std::uint64_t>(triangle_id) << 2) |
           static_cast<std::uint64_t>(local_edge & 0x3);
}

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
        atomicExch(&invalid_flag[0], std::uint32_t(1));
    }
};

template <gwn_index_type Index>
gwn_status gwn_validate_triangle_indices_device_impl(
    cuda::std::span<Index const> const i0, cuda::std::span<Index const> const i1,
    cuda::std::span<Index const> const i2, std::size_t const vertex_count,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    std::size_t const triangle_count = i0.size();
    if (i1.size() != triangle_count || i2.size() != triangle_count)
        return gwn_status::invalid_argument("Triangle SoA spans must have identical lengths.");
    if (!gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2)) {
        return gwn_status::invalid_argument(
            "Triangle index spans must use non-null storage when non-empty."
        );
    }

    if (triangle_count == 0)
        return gwn_status::ok();
    if (vertex_count == 0)
        return gwn_status::invalid_argument("Triangle indices require non-empty vertex arrays.");

    gwn_device_array<std::uint32_t> invalid_flag(stream);
    GWN_RETURN_ON_ERROR(invalid_flag.resize(1, stream));
    GWN_RETURN_ON_ERROR(invalid_flag.zero(stream));

    constexpr int k_block_size = k_gwn_default_block_size;
    GWN_RETURN_ON_ERROR(
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

    std::uint32_t host_invalid_flag = 0;
    GWN_RETURN_ON_ERROR(
        invalid_flag.copy_to_host(cuda::std::span<std::uint32_t>(&host_invalid_flag, 1), stream)
    );
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

    if (host_invalid_flag != 0)
        return gwn_status::invalid_argument("Triangle indices must be in [0, vertex_count).");
    return gwn_status::ok();
}

template <gwn_index_type Index> struct gwn_build_edge_arrays_functor {
    cuda::std::span<Index const> i0{};
    cuda::std::span<Index const> i1{};
    cuda::std::span<Index const> i2{};
    cuda::std::span<Index> key_hi{};
    cuda::std::span<gwn_boundary_edge_payload_hi<Index>> payload_hi{};

    __device__ void operator()(std::size_t const triangle_id) const {
        Index const vertices[3] = {i0[triangle_id], i1[triangle_id], i2[triangle_id]};
        for (int edge_id = 0; edge_id < 3; ++edge_id) {
            Index const a = vertices[edge_id];
            Index const b = vertices[(edge_id + 1) % 3];

            Index lo = a;
            Index hi = b;
            if (hi < lo)
                cuda::std::swap(lo, hi);

            std::size_t const global_edge_id = triangle_id * 3 + static_cast<std::size_t>(edge_id);
            key_hi[global_edge_id] = hi;
            payload_hi[global_edge_id] = gwn_boundary_edge_payload_hi<Index>{
                lo,
                encode_triangle_edge<Index>(triangle_id, edge_id),
                (a < b) ? 1 : -1,
            };
        }
    }
};

template <gwn_index_type Index> struct gwn_prepare_second_sort_inputs_functor {
    cuda::std::span<Index const> sorted_key_hi{};
    cuda::std::span<gwn_boundary_edge_payload_hi<Index> const> sorted_payload_hi{};
    cuda::std::span<Index> key_lo{};
    cuda::std::span<gwn_boundary_edge_payload_lo<Index>> payload_lo{};

    __device__ void operator()(std::size_t const edge_id) const {
        key_lo[edge_id] = sorted_payload_hi[edge_id].lo;
        payload_lo[edge_id] = gwn_boundary_edge_payload_lo<Index>{
            sorted_key_hi[edge_id],
            sorted_payload_hi[edge_id].tri_edge,
            sorted_payload_hi[edge_id].orientation,
        };
    }
};

template <gwn_index_type Index> struct gwn_mark_boundary_edges_functor {
    cuda::std::span<Index const> sorted_key_lo{};
    cuda::std::span<gwn_boundary_edge_payload_lo<Index> const> sorted_payload_lo{};
    std::size_t edge_count{0};
    cuda::std::span<std::uint32_t> triangle_mask_u32{};
    cuda::std::span<std::uint32_t> boundary_head_flags{};

    __device__ bool same_key(std::size_t const a, std::size_t const b) const {
        return sorted_key_lo[a] == sorted_key_lo[b] &&
               sorted_payload_lo[a].hi == sorted_payload_lo[b].hi;
    }

    __device__ void operator()(std::size_t const edge_id) const {
        if (edge_id >= edge_count)
            return;
        if (edge_id > 0 && same_key(edge_id, edge_id - 1))
            return;

        std::size_t end = edge_id + 1;
        int orientation_sum = sorted_payload_lo[edge_id].orientation;
        while (end < edge_count && same_key(end, edge_id)) {
            orientation_sum += sorted_payload_lo[end].orientation;
            ++end;
        }

        std::size_t const incident_count = end - edge_id;
        bool const is_boundary = incident_count != 2 || orientation_sum != 0;
        boundary_head_flags[edge_id] = is_boundary ? std::uint32_t(1) : std::uint32_t(0);
        if (!is_boundary)
            return;

        for (std::size_t k = edge_id; k < end; ++k) {
            std::uint64_t const tri_edge = sorted_payload_lo[k].tri_edge;
            std::size_t const triangle_id = static_cast<std::size_t>(tri_edge >> 2);
            std::uint32_t const local_edge = static_cast<std::uint32_t>(tri_edge & 0x3ULL);
            atomicOr(&triangle_mask_u32[triangle_id], std::uint32_t(1u << local_edge));
        }
    }
};

struct gwn_pack_triangle_boundary_mask_functor {
    cuda::std::span<std::uint32_t const> in_u32{};
    cuda::std::span<std::uint8_t> out_u8{};

    __device__ void operator()(std::size_t const triangle_id) const {
        out_u8[triangle_id] = static_cast<std::uint8_t>(in_u32[triangle_id] & 0x7u);
    }
};

template <class Key, class Value>
gwn_status gwn_cub_sort_pairs_impl(
    Key *const keys_in, Key *const keys_out, Value *const values_in, Value *const values_out,
    std::size_t const count, cudaStream_t const stream
) noexcept {
    if (count == 0)
        return gwn_status::ok();
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return gwn_status::invalid_argument("CUDA radix sort supports at most INT_MAX items.");

    std::size_t temporary_bytes = 0;
    int const item_count = static_cast<int>(count);
    int const end_bit = static_cast<int>(sizeof(Key) * 8);

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            nullptr, temporary_bytes, keys_in, keys_out, values_in, values_out, item_count, 0,
            end_bit, stream
        )
    ));

    gwn_device_array<std::uint8_t> temporary_storage(stream);
    GWN_RETURN_ON_ERROR(temporary_storage.resize(temporary_bytes, stream));

    return gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            temporary_storage.data(), temporary_bytes, keys_in, keys_out, values_in, values_out,
            item_count, 0, end_bit, stream
        )
    );
}

template <gwn_index_type Index>
gwn_status gwn_compute_triangle_boundary_edge_mask_device_impl(
    cuda::std::span<Index const> const i0, cuda::std::span<Index const> const i1,
    cuda::std::span<Index const> const i2, cuda::std::span<std::uint8_t const> const out_mask,
    Index *const out_singular_edge_count = nullptr, cudaStream_t const stream = gwn_default_stream()
) noexcept {
    std::size_t const triangle_count = i0.size();
    if (i1.size() != triangle_count || i2.size() != triangle_count)
        return gwn_status::invalid_argument("Triangle SoA spans must have identical lengths.");
    if (out_mask.size() != triangle_count) {
        return gwn_status::invalid_argument(
            "Boundary-edge mask output size must match triangle count."
        );
    }
    if (!gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2) ||
        !gwn_span_has_storage(out_mask)) {
        return gwn_status::invalid_argument(
            "Boundary-edge mask spans must use non-null storage when non-empty."
        );
    }

    if (triangle_count == 0) {
        if (out_singular_edge_count != nullptr)
            *out_singular_edge_count = Index(0);
        return gwn_status::ok();
    }

    if (triangle_count > (std::numeric_limits<std::size_t>::max() / 3))
        return gwn_status::invalid_argument("Triangle count is too large for edge preprocessing.");
    if (triangle_count > (std::numeric_limits<std::uint64_t>::max() >> 2))
        return gwn_status::invalid_argument("Triangle count exceeds edge encoding capacity.");

    std::size_t const edge_count = triangle_count * 3;

    gwn_device_array<Index> key_hi_in(stream);
    gwn_device_array<Index> key_hi_out(stream);
    gwn_device_array<gwn_boundary_edge_payload_hi<Index>> payload_hi_in(stream);
    gwn_device_array<gwn_boundary_edge_payload_hi<Index>> payload_hi_out(stream);
    gwn_device_array<Index> key_lo_in(stream);
    gwn_device_array<Index> key_lo_out(stream);
    gwn_device_array<gwn_boundary_edge_payload_lo<Index>> payload_lo_in(stream);
    gwn_device_array<gwn_boundary_edge_payload_lo<Index>> payload_lo_out(stream);
    gwn_device_array<std::uint32_t> triangle_mask_u32(stream);
    gwn_device_array<std::uint32_t> boundary_head_flags(stream);
    gwn_device_array<std::uint32_t> singular_edge_count(stream);

    GWN_RETURN_ON_ERROR(key_hi_in.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(key_hi_out.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(payload_hi_in.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(payload_hi_out.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(key_lo_in.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(key_lo_out.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(payload_lo_in.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(payload_lo_out.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(triangle_mask_u32.resize(triangle_count, stream));
    GWN_RETURN_ON_ERROR(boundary_head_flags.resize(edge_count, stream));
    if (out_singular_edge_count != nullptr)
        GWN_RETURN_ON_ERROR(singular_edge_count.resize(1, stream));

    GWN_RETURN_ON_ERROR(triangle_mask_u32.zero(stream));
    GWN_RETURN_ON_ERROR(boundary_head_flags.zero(stream));

    constexpr int k_block_size = k_gwn_default_block_size;
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            triangle_count,
            gwn_build_edge_arrays_functor<Index>{
                i0,
                i1,
                i2,
                key_hi_in.span(),
                payload_hi_in.span(),
            },
            stream
        )
    );

    GWN_RETURN_ON_ERROR(gwn_cub_sort_pairs_impl(
        key_hi_in.data(), key_hi_out.data(), payload_hi_in.data(), payload_hi_out.data(),
        edge_count, stream
    ));

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            edge_count,
            gwn_prepare_second_sort_inputs_functor<Index>{
                key_hi_out.span(),
                payload_hi_out.span(),
                key_lo_in.span(),
                payload_lo_in.span(),
            },
            stream
        )
    );

    GWN_RETURN_ON_ERROR(gwn_cub_sort_pairs_impl(
        key_lo_in.data(), key_lo_out.data(), payload_lo_in.data(), payload_lo_out.data(),
        edge_count, stream
    ));

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            edge_count,
            gwn_mark_boundary_edges_functor<Index>{
                key_lo_out.span(),
                payload_lo_out.span(),
                edge_count,
                triangle_mask_u32.span(),
                boundary_head_flags.span(),
            },
            stream
        )
    );

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            triangle_count,
            gwn_pack_triangle_boundary_mask_functor{
                triangle_mask_u32.span(),
                cuda::std::span<std::uint8_t>(gwn_mutable_data(out_mask), out_mask.size()),
            },
            stream
        )
    );

    if (out_singular_edge_count != nullptr) {
        std::size_t reduction_temp_bytes = 0;
        std::uint64_t const item_count = static_cast<std::uint64_t>(edge_count);
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
            cub::DeviceReduce::Sum(
                nullptr, reduction_temp_bytes, boundary_head_flags.data(),
                singular_edge_count.data(), item_count, stream
            )
        ));

        gwn_device_array<std::uint8_t> reduction_temp_storage(stream);
        GWN_RETURN_ON_ERROR(reduction_temp_storage.resize(reduction_temp_bytes, stream));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
            cub::DeviceReduce::Sum(
                reduction_temp_storage.data(), reduction_temp_bytes, boundary_head_flags.data(),
                singular_edge_count.data(), item_count, stream
            )
        ));

        std::uint32_t host_singular_edge_count = 0;
        GWN_RETURN_ON_ERROR(singular_edge_count.copy_to_host(
            cuda::std::span<std::uint32_t>(&host_singular_edge_count, 1), stream
        ));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

        if (host_singular_edge_count > std::numeric_limits<Index>::max())
            return gwn_status::internal_error("Singular-edge count overflow.");
        *out_singular_edge_count = static_cast<Index>(host_singular_edge_count);
    }

    return gwn_status::ok();
}

template <gwn_real_type Real, gwn_index_type Index> struct gwn_accumulate_vertex_normals_functor {
    cuda::std::span<Real const> vertex_x{};
    cuda::std::span<Real const> vertex_y{};
    cuda::std::span<Real const> vertex_z{};
    cuda::std::span<Index const> tri_i0{};
    cuda::std::span<Index const> tri_i1{};
    cuda::std::span<Index const> tri_i2{};
    cuda::std::span<Real> vertex_nx{};
    cuda::std::span<Real> vertex_ny{};
    cuda::std::span<Real> vertex_nz{};

    __device__ void operator()(std::size_t const triangle_id) const {
        std::size_t const ia = static_cast<std::size_t>(tri_i0[triangle_id]);
        std::size_t const ib = static_cast<std::size_t>(tri_i1[triangle_id]);
        std::size_t const ic = static_cast<std::size_t>(tri_i2[triangle_id]);

        Real const ax = vertex_x[ia];
        Real const ay = vertex_y[ia];
        Real const az = vertex_z[ia];
        Real const bx = vertex_x[ib];
        Real const by = vertex_y[ib];
        Real const bz = vertex_z[ib];
        Real const cx = vertex_x[ic];
        Real const cy = vertex_y[ic];
        Real const cz = vertex_z[ic];

        Real const ab_x = bx - ax;
        Real const ab_y = by - ay;
        Real const ab_z = bz - az;
        Real const ac_x = cx - ax;
        Real const ac_y = cy - ay;
        Real const ac_z = cz - az;

        Real const nx = ab_y * ac_z - ab_z * ac_y;
        Real const ny = ab_z * ac_x - ab_x * ac_z;
        Real const nz = ab_x * ac_y - ab_y * ac_x;

        atomicAdd(&vertex_nx[ia], nx);
        atomicAdd(&vertex_ny[ia], ny);
        atomicAdd(&vertex_nz[ia], nz);
        atomicAdd(&vertex_nx[ib], nx);
        atomicAdd(&vertex_ny[ib], ny);
        atomicAdd(&vertex_nz[ib], nz);
        atomicAdd(&vertex_nx[ic], nx);
        atomicAdd(&vertex_ny[ic], ny);
        atomicAdd(&vertex_nz[ic], nz);
    }
};

template <gwn_real_type Real> struct gwn_normalize_vertex_normals_functor {
    cuda::std::span<Real> vertex_nx{};
    cuda::std::span<Real> vertex_ny{};
    cuda::std::span<Real> vertex_nz{};

    __device__ void operator()(std::size_t const vertex_id) const {
        using std::sqrt;

        Real const nx = vertex_nx[vertex_id];
        Real const ny = vertex_ny[vertex_id];
        Real const nz = vertex_nz[vertex_id];
        Real const norm2 = nx * nx + ny * ny + nz * nz;
        if (!(norm2 > Real(0)))
            return;

        Real const inv_norm = Real(1) / sqrt(norm2);
        vertex_nx[vertex_id] *= inv_norm;
        vertex_ny[vertex_id] *= inv_norm;
        vertex_nz[vertex_id] *= inv_norm;
    }
};

template <gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_compute_vertex_normals_device_impl(
    cuda::std::span<Real const> const vertex_x, cuda::std::span<Real const> const vertex_y,
    cuda::std::span<Real const> const vertex_z, cuda::std::span<Index const> const tri_i0,
    cuda::std::span<Index const> const tri_i1, cuda::std::span<Index const> const tri_i2,
    cuda::std::span<Real const> const out_nx, cuda::std::span<Real const> const out_ny,
    cuda::std::span<Real const> const out_nz, cudaStream_t const stream = gwn_default_stream()
) noexcept {
    std::size_t const vertex_count = vertex_x.size();
    std::size_t const triangle_count = tri_i0.size();

    if (vertex_y.size() != vertex_count || vertex_z.size() != vertex_count)
        return gwn_status::invalid_argument("Vertex SoA spans must have identical lengths.");
    if (tri_i1.size() != triangle_count || tri_i2.size() != triangle_count)
        return gwn_status::invalid_argument("Triangle SoA spans must have identical lengths.");
    if (out_nx.size() != vertex_count || out_ny.size() != vertex_count ||
        out_nz.size() != vertex_count) {
        return gwn_status::invalid_argument("Vertex normal output spans must match vertex count.");
    }

    if (!gwn_span_has_storage(vertex_x) || !gwn_span_has_storage(vertex_y) ||
        !gwn_span_has_storage(vertex_z) || !gwn_span_has_storage(tri_i0) ||
        !gwn_span_has_storage(tri_i1) || !gwn_span_has_storage(tri_i2) ||
        !gwn_span_has_storage(out_nx) || !gwn_span_has_storage(out_ny) ||
        !gwn_span_has_storage(out_nz)) {
        return gwn_status::invalid_argument(
            "Vertex normal spans must use non-null storage when non-empty."
        );
    }

    if (vertex_count == 0)
        return gwn_status::ok();

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(gwn_mutable_data(out_nx), 0, out_nx.size_bytes(), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(gwn_mutable_data(out_ny), 0, out_ny.size_bytes(), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(gwn_mutable_data(out_nz), 0, out_nz.size_bytes(), stream)
    ));

    constexpr int k_block_size = k_gwn_default_block_size;
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            triangle_count,
            gwn_accumulate_vertex_normals_functor<Real, Index>{
                vertex_x,
                vertex_y,
                vertex_z,
                tri_i0,
                tri_i1,
                tri_i2,
                cuda::std::span<Real>(gwn_mutable_data(out_nx), out_nx.size()),
                cuda::std::span<Real>(gwn_mutable_data(out_ny), out_ny.size()),
                cuda::std::span<Real>(gwn_mutable_data(out_nz), out_nz.size()),
            },
            stream
        )
    );

    return gwn_launch_linear_kernel<k_block_size>(
        vertex_count,
        gwn_normalize_vertex_normals_functor<Real>{
            cuda::std::span<Real>(gwn_mutable_data(out_nx), out_nx.size()),
            cuda::std::span<Real>(gwn_mutable_data(out_ny), out_ny.size()),
            cuda::std::span<Real>(gwn_mutable_data(out_nz), out_nz.size()),
        },
        stream
    );
}

template <gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_upload_accessor(
    gwn_geometry_accessor<Real, Index> &accessor, cuda::std::span<Real const> const x,
    cuda::std::span<Real const> const y, cuda::std::span<Real const> const z,
    cuda::std::span<Index const> const i0, cuda::std::span<Index const> const i1,
    cuda::std::span<Index const> const i2, cudaStream_t const stream
) {
    auto const release = [&](gwn_geometry_accessor<Real, Index> &target, cudaStream_t s) noexcept {
        gwn_release_accessor(target, s);
    };

    auto const build = [&](gwn_geometry_accessor<Real, Index> &staging) -> gwn_status {
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

        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_x, x.size(), stream));
        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_y, y.size(), stream));
        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_z, z.size(), stream));
        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_nx, x.size(), stream));
        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_ny, y.size(), stream));
        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.vertex_nz, z.size(), stream));
        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.tri_i0, i0.size(), stream));
        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.tri_i1, i1.size(), stream));
        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.tri_i2, i2.size(), stream));
        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.tri_boundary_edge_mask, i0.size(), stream));

        GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.vertex_x, x, stream));
        GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.vertex_y, y, stream));
        GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.vertex_z, z, stream));
        GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.tri_i0, i0, stream));
        GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.tri_i1, i1, stream));
        GWN_RETURN_ON_ERROR(gwn_copy_h2d(staging.tri_i2, i2, stream));

        GWN_RETURN_ON_ERROR(
            gwn_validate_triangle_indices_device_impl<Index>(
                staging.tri_i0, staging.tri_i1, staging.tri_i2, x.size(), stream
            )
        );

        Index singular_edge_count = Index(0);
        GWN_RETURN_ON_ERROR(
            gwn_compute_triangle_boundary_edge_mask_device_impl<Index>(
                staging.tri_i0, staging.tri_i1, staging.tri_i2, staging.tri_boundary_edge_mask,
                &singular_edge_count, stream
            )
        );

        GWN_RETURN_ON_ERROR((gwn_compute_vertex_normals_device_impl<Real, Index>(
            staging.vertex_x, staging.vertex_y, staging.vertex_z, staging.tri_i0, staging.tri_i1,
            staging.tri_i2, staging.vertex_nx, staging.vertex_ny, staging.vertex_nz, stream
        )));

        staging.singular_edge_count = singular_edge_count;
        return gwn_status::ok();
    };

    return gwn_replace_accessor_with_staging(accessor, release, build, stream);
}

} // namespace detail

template <gwn_index_type Index>
gwn_status gwn_compute_triangle_boundary_edge_mask(
    cuda::std::span<Index const> i0, cuda::std::span<Index const> i1,
    cuda::std::span<Index const> i2, cuda::std::span<std::uint8_t> out_mask
) noexcept {
    return detail::gwn_try_translate_status(
        "gwn_compute_triangle_boundary_edge_mask", [&]() -> gwn_status {
        std::size_t const triangle_count = i0.size();
        if (i1.size() != triangle_count || i2.size() != triangle_count)
            return gwn_status::invalid_argument("Triangle SoA spans must have identical lengths.");
        if (out_mask.size() != triangle_count) {
            return gwn_status::invalid_argument(
                "Boundary-edge mask output size must match triangle count."
            );
        }
        if (!gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2) ||
            !gwn_span_has_storage(out_mask)) {
            return gwn_status::invalid_argument(
                "Boundary-edge mask spans must use non-null storage when non-empty."
            );
        }

        gwn_device_array<Index> d_i0;
        gwn_device_array<Index> d_i1;
        gwn_device_array<Index> d_i2;
        gwn_device_array<std::uint8_t> d_mask;

        cudaStream_t const stream = gwn_default_stream();
        GWN_RETURN_ON_ERROR(d_i0.copy_from_host(i0, stream));
        GWN_RETURN_ON_ERROR(d_i1.copy_from_host(i1, stream));
        GWN_RETURN_ON_ERROR(d_i2.copy_from_host(i2, stream));
        GWN_RETURN_ON_ERROR(d_mask.resize(out_mask.size(), stream));

        GWN_RETURN_ON_ERROR(
            detail::gwn_compute_triangle_boundary_edge_mask_device_impl<Index>(
                d_i0.span(), d_i1.span(), d_i2.span(), d_mask.span(), nullptr, stream
            )
        );

        GWN_RETURN_ON_ERROR(d_mask.copy_to_host(out_mask, stream));
        return gwn_cuda_to_status(cudaStreamSynchronize(stream));
    }
    );
}

template <gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_upload_geometry(
    gwn_geometry_object<Real, Index> &object, cuda::std::span<Real const> x,
    cuda::std::span<Real const> y, cuda::std::span<Real const> z, cuda::std::span<Index const> i0,
    cuda::std::span<Index const> i1, cuda::std::span<Index const> i2, cudaStream_t const stream
) noexcept {
    return detail::gwn_try_translate_status("gwn_upload_geometry", [&]() -> gwn_status {
        gwn_geometry_object<Real, Index> staging;
        staging.set_stream(stream);
        GWN_RETURN_ON_ERROR(
            detail::gwn_upload_accessor(staging.accessor_, x, y, z, i0, i1, i2, stream)
        );

        swap(object, staging);
        return gwn_status::ok();
    });
}

} // namespace gwn
