#pragma once

#include <cuda/std/utility>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cub/cub.cuh>

#include "../gwn_boundary.cuh"
#include "gwn_bvh_status_helpers.cuh"
#include "gwn_device_array.cuh"
#include "gwn_geometry_impl.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {
namespace detail {

/// \brief Payload retained after sorting oriented edges by their high vertex index.
template <gwn_index_type Index> struct gwn_boundary_chain_payload_hi {
    Index lo{};
    int orientation{0};
};

/// \brief Payload retained after sorting oriented edges by their low vertex index.
template <gwn_index_type Index> struct gwn_boundary_chain_payload_lo {
    Index hi{};
    int orientation{0};
};

/// \brief Expand each oriented triangle into three canonical undirected edge keys.
template <gwn_index_type Index> struct gwn_boundary_chain_edge_functor {
    cuda::std::span<Index const> i0{};
    cuda::std::span<Index const> i1{};
    cuda::std::span<Index const> i2{};
    cuda::std::span<Index> key_hi{};
    cuda::std::span<gwn_boundary_chain_payload_hi<Index>> payload_hi{};

    __device__ void operator()(std::size_t const triangle_id) const {
        Index const vertices[3] = {i0[triangle_id], i1[triangle_id], i2[triangle_id]};
        for (int edge_id = 0; edge_id < 3; ++edge_id) {
            Index const start = vertices[edge_id];
            Index const end = vertices[(edge_id + 1) % 3];

            Index lo = start;
            Index hi = end;
            if (hi < lo)
                cuda::std::swap(lo, hi);

            std::size_t const output_id = triangle_id * 3 + static_cast<std::size_t>(edge_id);
            key_hi[output_id] = hi;
            // Orientation is measured against the canonical lo-to-hi direction. Equal endpoints
            // contribute zero to the algebraic boundary and disappear with zero-sum edge groups.
            payload_hi[output_id] = {lo, (start < end) ? 1 : (end < start ? -1 : 0)};
        }
    }
};

/// \brief Re-key high-sorted edges for the second radix pass over the low vertex index.
template <gwn_index_type Index> struct gwn_boundary_chain_rekey_functor {
    cuda::std::span<Index const> sorted_key_hi{};
    cuda::std::span<gwn_boundary_chain_payload_hi<Index> const> sorted_payload_hi{};
    cuda::std::span<Index> key_lo{};
    cuda::std::span<gwn_boundary_chain_payload_lo<Index>> payload_lo{};

    __device__ void operator()(std::size_t const edge_id) const {
        key_lo[edge_id] = sorted_payload_hi[edge_id].lo;
        payload_lo[edge_id] = {sorted_key_hi[edge_id], sorted_payload_hi[edge_id].orientation};
    }
};

/// \brief Sort key-payload pairs over every bit of an unsigned index key.
template <class Key, class Value>
void gwn_boundary_chain_sort_pairs_impl(
    Key *const keys_in, Key *const keys_out, Value *const values_in, Value *const values_out,
    std::size_t const count, cudaStream_t const stream
) {
    if (count == 0)
        return;
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("CUDA radix sort supports at most INT_MAX items.");

    std::size_t temporary_bytes = 0;
    int const item_count = static_cast<int>(count);
    int const end_bit = static_cast<int>(sizeof(Key) * 8);
    gwn_throw_status_error(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            nullptr, temporary_bytes, keys_in, keys_out, values_in, values_out, item_count, 0,
            end_bit, stream
        )
    ));

    gwn_device_array<std::uint8_t> temporary_storage(stream);
    temporary_storage.resize(temporary_bytes, stream);
    gwn_throw_status_error(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            temporary_storage.data(), temporary_bytes, keys_in, keys_out, values_in, values_out,
            item_count, 0, end_bit, stream
        )
    ));
}

/// \brief Release every span owned by a boundary-chain accessor.
template <gwn_index_type Index>
void gwn_release_boundary_chain_accessor(
    gwn_boundary_chain_accessor<Index> &accessor, cudaStream_t const stream
) noexcept {
    gwn_free_span(accessor.multiplicity, stream);
    gwn_free_span(accessor.end_vertex, stream);
    gwn_free_span(accessor.start_vertex, stream);
    accessor.mesh_vertex_count = 0;
    accessor.mesh_triangle_count = 0;
    accessor.is_built = false;
}

/// \brief Mark the first row of each edge group whose net orientation is nonzero.
template <gwn_index_type Index> struct gwn_boundary_chain_head_marker_functor {
    cuda::std::span<Index const> sorted_key_lo{};
    cuda::std::span<gwn_boundary_chain_payload_lo<Index> const> sorted_payload_lo{};
    std::size_t edge_count{0};
    cuda::std::span<std::uint64_t> boundary_head_markers{};

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

        // Boundary-chain rows are the algebraic boundary of the triangle
        // chain, so only nonzero net orientation emits an edge.
        boundary_head_markers[edge_id] =
            (orientation_sum != 0) ? std::uint64_t(1) : std::uint64_t(0);
    }
};

/// \brief Emit one canonical directed edge and multiplicity per marked edge group.
template <gwn_index_type Index> struct gwn_boundary_chain_emit_functor {
    cuda::std::span<Index const> sorted_key_lo{};
    cuda::std::span<gwn_boundary_chain_payload_lo<Index> const> sorted_payload_lo{};
    std::size_t edge_count{0};
    cuda::std::span<std::uint64_t const> boundary_head_markers{};
    cuda::std::span<std::uint64_t const> boundary_offsets{};
    cuda::std::span<Index> out_start_vertex{};
    cuda::std::span<Index> out_end_vertex{};
    cuda::std::span<std::uint64_t> out_multiplicity{};

    __device__ bool same_key(std::size_t const a, std::size_t const b) const {
        return sorted_key_lo[a] == sorted_key_lo[b] &&
               sorted_payload_lo[a].hi == sorted_payload_lo[b].hi;
    }

    __device__ void operator()(std::size_t const edge_id) const {
        if (edge_id >= edge_count || boundary_head_markers[edge_id] == 0)
            return;

        std::size_t end = edge_id + 1;
        int orientation_sum = sorted_payload_lo[edge_id].orientation;
        while (end < edge_count && same_key(end, edge_id)) {
            orientation_sum += sorted_payload_lo[end].orientation;
            ++end;
        }
        if (orientation_sum == 0)
            return;

        std::size_t const out_id = static_cast<std::size_t>(boundary_offsets[edge_id]);
        Index const lo = sorted_key_lo[edge_id];
        Index const hi = sorted_payload_lo[edge_id].hi;
        if (orientation_sum > 0) {
            out_start_vertex[out_id] = lo;
            out_end_vertex[out_id] = hi;
            out_multiplicity[out_id] = static_cast<std::uint64_t>(orientation_sum);
        } else {
            out_start_vertex[out_id] = hi;
            out_end_vertex[out_id] = lo;
            out_multiplicity[out_id] = static_cast<std::uint64_t>(-orientation_sum);
        }
    }
};

/// \brief Build the compact algebraic boundary from device triangle-index spans.
template <gwn_index_type Index>
void gwn_build_boundary_chain_device_impl(
    std::size_t const mesh_vertex_count, cuda::std::span<Index const> const i0,
    cuda::std::span<Index const> const i1, cuda::std::span<Index const> const i2,
    gwn_boundary_chain_accessor<Index> &staging, cudaStream_t const stream
) {
    std::size_t const triangle_count = i0.size();
    if (i1.size() != triangle_count || i2.size() != triangle_count)
        throw std::invalid_argument("Boundary chain triangle spans must match.");
    if (!gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2)) {
        throw std::invalid_argument(
            "Boundary chain triangle spans must use non-null storage when non-empty."
        );
    }

    gwn_validate_triangle_indices_device_impl<Index>(i0, i1, i2, mesh_vertex_count, stream);

    staging.mesh_vertex_count = mesh_vertex_count;
    staging.mesh_triangle_count = triangle_count;
    staging.is_built = true;
    if (triangle_count == 0)
        return;

    if (triangle_count > (std::numeric_limits<std::size_t>::max() / 3))
        throw std::invalid_argument("Boundary chain triangle count is too large.");
    if (triangle_count > (std::numeric_limits<std::uint64_t>::max() >> 2))
        throw std::invalid_argument("Boundary chain triangle count exceeds edge encoding.");

    std::size_t const edge_count = triangle_count * 3;
    if (edge_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Boundary chain preprocessing supports at most INT_MAX generated edges."
        );
    }
    int const edge_count_i32 = static_cast<int>(edge_count);

    gwn_device_array<Index> key_hi_in(stream);
    gwn_device_array<Index> key_hi_out(stream);
    gwn_device_array<gwn_boundary_chain_payload_hi<Index>> payload_hi_in(stream);
    gwn_device_array<gwn_boundary_chain_payload_hi<Index>> payload_hi_out(stream);
    gwn_device_array<Index> key_lo_in(stream);
    gwn_device_array<Index> key_lo_out(stream);
    gwn_device_array<gwn_boundary_chain_payload_lo<Index>> payload_lo_in(stream);
    gwn_device_array<gwn_boundary_chain_payload_lo<Index>> payload_lo_out(stream);
    gwn_device_array<std::uint64_t> boundary_head_markers(stream);
    gwn_device_array<std::uint64_t> boundary_offsets(stream);
    gwn_device_array<std::uint64_t> boundary_count_device(stream);

    key_hi_in.resize(edge_count, stream);
    key_hi_out.resize(edge_count, stream);
    payload_hi_in.resize(edge_count, stream);
    payload_hi_out.resize(edge_count, stream);
    key_lo_in.resize(edge_count, stream);
    key_lo_out.resize(edge_count, stream);
    payload_lo_in.resize(edge_count, stream);
    payload_lo_out.resize(edge_count, stream);
    boundary_head_markers.resize(edge_count, stream);
    boundary_offsets.resize(edge_count, stream);
    boundary_count_device.resize(1, stream);
    boundary_head_markers.zero(stream);

    // Build one oriented row per triangle edge.
    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_gwn_default_block_size>(
            triangle_count,
            gwn_boundary_chain_edge_functor<Index>{
                i0,
                i1,
                i2,
                key_hi_in.span(),
                payload_hi_in.span(),
            },
            stream
        )
    );

    // Canonicalize edge keys with two radix-sort passes.
    gwn_boundary_chain_sort_pairs_impl(
        key_hi_in.data(), key_hi_out.data(), payload_hi_in.data(), payload_hi_out.data(),
        edge_count, stream
    );

    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_gwn_default_block_size>(
            edge_count,
            gwn_boundary_chain_rekey_functor<Index>{
                key_hi_out.span(),
                payload_hi_out.span(),
                key_lo_in.span(),
                payload_lo_in.span(),
            },
            stream
        )
    );

    gwn_boundary_chain_sort_pairs_impl(
        key_lo_in.data(), key_lo_out.data(), payload_lo_in.data(), payload_lo_out.data(),
        edge_count, stream
    );

    // Mark one head row for each edge with nonzero algebraic orientation.
    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_gwn_default_block_size>(
            edge_count,
            gwn_boundary_chain_head_marker_functor<Index>{
                key_lo_out.span(),
                payload_lo_out.span(),
                edge_count,
                boundary_head_markers.span(),
            },
            stream
        )
    );

    std::size_t temp_bytes = 0;
    gwn_throw_status_error(gwn_cuda_to_status(
        cub::DeviceReduce::Sum(
            nullptr, temp_bytes, boundary_head_markers.data(), boundary_count_device.data(),
            edge_count_i32, stream
        )
    ));
    gwn_device_array<std::uint8_t> reduce_temp(stream);
    reduce_temp.resize(temp_bytes, stream);
    gwn_throw_status_error(gwn_cuda_to_status(
        cub::DeviceReduce::Sum(
            reduce_temp.data(), temp_bytes, boundary_head_markers.data(),
            boundary_count_device.data(), edge_count_i32, stream
        )
    ));

    std::uint64_t boundary_count = 0;
    boundary_count_device.copy_to_host(cuda::std::span<std::uint64_t>(&boundary_count, 1), stream);
    gwn_throw_status_error(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (boundary_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("Boundary chain edge count overflow.");

    // Allocate the compact chain before scan maps heads to output rows.
    std::size_t const output_count = static_cast<std::size_t>(boundary_count);
    gwn_allocate_span(staging.start_vertex, output_count, stream);
    gwn_allocate_span(staging.end_vertex, output_count, stream);
    gwn_allocate_span(staging.multiplicity, output_count, stream);
    if (output_count == 0)
        return;

    temp_bytes = 0;
    gwn_throw_status_error(gwn_cuda_to_status(
        cub::DeviceScan::ExclusiveSum(
            nullptr, temp_bytes, boundary_head_markers.data(), boundary_offsets.data(),
            edge_count_i32, stream
        )
    ));
    gwn_device_array<std::uint8_t> scan_temp(stream);
    scan_temp.resize(temp_bytes, stream);
    gwn_throw_status_error(gwn_cuda_to_status(
        cub::DeviceScan::ExclusiveSum(
            scan_temp.data(), temp_bytes, boundary_head_markers.data(), boundary_offsets.data(),
            edge_count_i32, stream
        )
    ));

    // Emit directed edges; negative orientation flips the stored direction.
    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_gwn_default_block_size>(
            edge_count,
            gwn_boundary_chain_emit_functor<Index>{
                key_lo_out.span(),
                payload_lo_out.span(),
                edge_count,
                boundary_head_markers.span(),
                boundary_offsets.span(),
                staging.start_vertex,
                staging.end_vertex,
                staging.multiplicity,
            },
            stream
        )
    );
}

/// \brief Replace a boundary-chain accessor only after staging has built successfully.
template <gwn_index_type Index>
void gwn_replace_boundary_chain_accessor(
    gwn_boundary_chain_accessor<Index> &target, std::size_t const mesh_vertex_count,
    cuda::std::span<Index const> const i0, cuda::std::span<Index const> const i1,
    cuda::std::span<Index const> const i2, cudaStream_t const stream
) {
    auto release = [](gwn_boundary_chain_accessor<Index> &accessor, cudaStream_t const s) noexcept {
        gwn_release_boundary_chain_accessor(accessor, s);
    };
    auto build = [&](gwn_boundary_chain_accessor<Index> &staging) {
        gwn_build_boundary_chain_device_impl(mesh_vertex_count, i0, i1, i2, staging, stream);
    };
    gwn_replace_accessor_with_staging(target, release, build, stream);
}

} // namespace detail

template <gwn_index_type Index>
[[nodiscard]] gwn_status gwn_build_boundary_chain(
    std::size_t const mesh_vertex_count, cuda::std::span<Index const> const i0,
    cuda::std::span<Index const> const i1, cuda::std::span<Index const> const i2,
    gwn_boundary_chain_object<Index> &out, cudaStream_t const stream
) noexcept {
    return detail::gwn_try_translate_status("gwn_build_boundary_chain", [&]() -> gwn_status {
        std::size_t const triangle_count = i0.size();
        if (i1.size() != triangle_count || i2.size() != triangle_count)
            return gwn_status::invalid_argument("Boundary chain triangle spans must match.");
        if (!gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2)) {
            return gwn_status::invalid_argument(
                "Boundary chain triangle spans must use non-null storage when non-empty."
            );
        }

        detail::gwn_device_array<Index> d_i0(stream);
        detail::gwn_device_array<Index> d_i1(stream);
        detail::gwn_device_array<Index> d_i2(stream);
        gwn_boundary_chain_object<Index> staging;
        d_i0.copy_from_host(i0, stream);
        d_i1.copy_from_host(i1, stream);
        d_i2.copy_from_host(i2, stream);
        detail::gwn_replace_boundary_chain_accessor(
            staging.accessor(), mesh_vertex_count,
            cuda::std::span<Index const>(d_i0.data(), d_i0.size()),
            cuda::std::span<Index const>(d_i1.data(), d_i1.size()),
            cuda::std::span<Index const>(d_i2.data(), d_i2.size()), stream
        );
        staging.set_stream(stream);
        swap(out, staging);
        // The replaced chain stays bound to its old stream until staging destroys it.
        return gwn_status::ok();
    });
}

template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_build_boundary_chain(
    gwn_geometry_object<Real, Index> const &geometry, gwn_boundary_chain_object<Index> &out,
    cudaStream_t const stream
) noexcept {
    return detail::gwn_try_translate_status("gwn_build_boundary_chain", [&]() -> gwn_status {
        auto const &accessor = geometry.accessor();
        if (!accessor.is_valid())
            return gwn_status::invalid_argument("Geometry object contains no queryable data.");
        gwn_boundary_chain_object<Index> staging;
        detail::gwn_replace_boundary_chain_accessor(
            staging.accessor(), accessor.vertex_count(),
            cuda::std::span<Index const>(accessor.tri_i0.data(), accessor.tri_i0.size()),
            cuda::std::span<Index const>(accessor.tri_i1.data(), accessor.tri_i1.size()),
            cuda::std::span<Index const>(accessor.tri_i2.data(), accessor.tri_i2.size()), stream
        );
        staging.set_stream(stream);
        swap(out, staging);
        // The replaced chain stays bound to its old stream until staging destroys it.
        return gwn_status::ok();
    });
}

} // namespace gwn
