#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cub/cub.cuh>

#include "../gwn_boundary.cuh"
#include "../gwn_kernel_utils.cuh"
#include "gwn_bvh_status_helpers.cuh"
#include "gwn_geometry_impl.cuh"

namespace gwn {
namespace detail {

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

template <gwn_index_type Index> struct gwn_boundary_chain_head_marker_functor {
    cuda::std::span<Index const> sorted_key_lo{};
    cuda::std::span<gwn_boundary_edge_payload_lo<Index> const> sorted_payload_lo{};
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

template <gwn_index_type Index> struct gwn_boundary_chain_emit_functor {
    cuda::std::span<Index const> sorted_key_lo{};
    cuda::std::span<gwn_boundary_edge_payload_lo<Index> const> sorted_payload_lo{};
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

template <gwn_index_type Index>
gwn_status gwn_build_boundary_chain_device_impl(
    std::size_t const mesh_vertex_count, cuda::std::span<Index const> const i0,
    cuda::std::span<Index const> const i1, cuda::std::span<Index const> const i2,
    gwn_boundary_chain_accessor<Index> &staging, cudaStream_t const stream
) noexcept {
    std::size_t const triangle_count = i0.size();
    if (i1.size() != triangle_count || i2.size() != triangle_count)
        return gwn_status::invalid_argument("Boundary chain triangle spans must match.");
    if (!gwn_span_has_storage(i0) || !gwn_span_has_storage(i1) || !gwn_span_has_storage(i2)) {
        return gwn_status::invalid_argument(
            "Boundary chain triangle spans must use non-null storage when non-empty."
        );
    }

    GWN_RETURN_ON_ERROR(
        gwn_validate_triangle_indices_device_impl<Index>(i0, i1, i2, mesh_vertex_count, stream)
    );

    staging.mesh_vertex_count = mesh_vertex_count;
    staging.mesh_triangle_count = triangle_count;
    staging.is_built = true;
    if (triangle_count == 0)
        return gwn_status::ok();

    if (triangle_count > (std::numeric_limits<std::size_t>::max() / 3))
        return gwn_status::invalid_argument("Boundary chain triangle count is too large.");
    if (triangle_count > (std::numeric_limits<std::uint64_t>::max() >> 2))
        return gwn_status::invalid_argument("Boundary chain triangle count exceeds edge encoding.");

    std::size_t const edge_count = triangle_count * 3;
    if (edge_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return gwn_status::invalid_argument(
            "Boundary chain preprocessing supports at most INT_MAX generated edges."
        );
    }
    int const edge_count_i32 = static_cast<int>(edge_count);

    gwn_device_array<Index> key_hi_in(stream);
    gwn_device_array<Index> key_hi_out(stream);
    gwn_device_array<gwn_boundary_edge_payload_hi<Index>> payload_hi_in(stream);
    gwn_device_array<gwn_boundary_edge_payload_hi<Index>> payload_hi_out(stream);
    gwn_device_array<Index> key_lo_in(stream);
    gwn_device_array<Index> key_lo_out(stream);
    gwn_device_array<gwn_boundary_edge_payload_lo<Index>> payload_lo_in(stream);
    gwn_device_array<gwn_boundary_edge_payload_lo<Index>> payload_lo_out(stream);
    gwn_device_array<std::uint64_t> boundary_head_markers(stream);
    gwn_device_array<std::uint64_t> boundary_offsets(stream);
    gwn_device_array<std::uint64_t> boundary_count_device(stream);

    GWN_RETURN_ON_ERROR(key_hi_in.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(key_hi_out.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(payload_hi_in.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(payload_hi_out.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(key_lo_in.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(key_lo_out.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(payload_lo_in.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(payload_lo_out.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(boundary_head_markers.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(boundary_offsets.resize(edge_count, stream));
    GWN_RETURN_ON_ERROR(boundary_count_device.resize(1, stream));
    GWN_RETURN_ON_ERROR(boundary_head_markers.zero(stream));

    // Build one oriented row per triangle edge.
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_gwn_default_block_size>(
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

    // Canonicalize edge keys with two radix-sort passes.
    GWN_RETURN_ON_ERROR(gwn_cub_sort_pairs_impl(
        key_hi_in.data(), key_hi_out.data(), payload_hi_in.data(), payload_hi_out.data(),
        edge_count, stream
    ));

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_gwn_default_block_size>(
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

    // Mark one head row for each edge with nonzero algebraic orientation.
    GWN_RETURN_ON_ERROR(
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
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceReduce::Sum(
            nullptr, temp_bytes, boundary_head_markers.data(), boundary_count_device.data(),
            edge_count_i32, stream
        )
    ));
    gwn_device_array<std::uint8_t> reduce_temp(stream);
    GWN_RETURN_ON_ERROR(reduce_temp.resize(temp_bytes, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceReduce::Sum(
            reduce_temp.data(), temp_bytes, boundary_head_markers.data(),
            boundary_count_device.data(), edge_count_i32, stream
        )
    ));

    std::uint64_t boundary_count = 0;
    GWN_RETURN_ON_ERROR(boundary_count_device.copy_to_host(
        cuda::std::span<std::uint64_t>(&boundary_count, 1), stream
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (boundary_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        return gwn_status::internal_error("Boundary chain edge count overflow.");

    // Allocate the compact chain before scan maps heads to output rows.
    std::size_t const output_count = static_cast<std::size_t>(boundary_count);
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.start_vertex, output_count, stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.end_vertex, output_count, stream));
    GWN_RETURN_ON_ERROR(gwn_allocate_span(staging.multiplicity, output_count, stream));
    if (output_count == 0)
        return gwn_status::ok();

    temp_bytes = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceScan::ExclusiveSum(
            nullptr, temp_bytes, boundary_head_markers.data(), boundary_offsets.data(),
            edge_count_i32, stream
        )
    ));
    gwn_device_array<std::uint8_t> scan_temp(stream);
    GWN_RETURN_ON_ERROR(scan_temp.resize(temp_bytes, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceScan::ExclusiveSum(
            scan_temp.data(), temp_bytes, boundary_head_markers.data(), boundary_offsets.data(),
            edge_count_i32, stream
        )
    ));

    // Emit directed edges; negative orientation flips the stored direction.
    return gwn_launch_linear_kernel<k_gwn_default_block_size>(
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
    );
}

template <gwn_index_type Index>
gwn_status gwn_replace_boundary_chain_accessor(
    gwn_boundary_chain_accessor<Index> &target, std::size_t const mesh_vertex_count,
    cuda::std::span<Index const> const i0, cuda::std::span<Index const> const i1,
    cuda::std::span<Index const> const i2, cudaStream_t const stream
) noexcept {
    auto release = [](gwn_boundary_chain_accessor<Index> &accessor, cudaStream_t const s) noexcept {
        gwn_release_boundary_chain_accessor(accessor, s);
    };
    auto build = [&](gwn_boundary_chain_accessor<Index> &staging) -> gwn_status {
        return gwn_build_boundary_chain_device_impl(mesh_vertex_count, i0, i1, i2, staging, stream);
    };
    return gwn_replace_accessor_with_staging(target, release, build, stream);
}

} // namespace detail

template <gwn_index_type Index>
gwn_status gwn_build_boundary_chain(
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

        gwn_device_array<Index> d_i0(stream);
        gwn_device_array<Index> d_i1(stream);
        gwn_device_array<Index> d_i2(stream);
        GWN_RETURN_ON_ERROR(d_i0.copy_from_host(i0, stream));
        GWN_RETURN_ON_ERROR(d_i1.copy_from_host(i1, stream));
        GWN_RETURN_ON_ERROR(d_i2.copy_from_host(i2, stream));
        GWN_RETURN_ON_ERROR(
            detail::gwn_replace_boundary_chain_accessor(
                out.accessor_, mesh_vertex_count,
                cuda::std::span<Index const>(d_i0.data(), d_i0.size()),
                cuda::std::span<Index const>(d_i1.data(), d_i1.size()),
                cuda::std::span<Index const>(d_i2.data(), d_i2.size()), stream
            )
        );
        out.set_stream(stream);
        return gwn_status::ok();
    });
}

template <gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_build_boundary_chain(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_boundary_chain_object<Index> &out,
    cudaStream_t const stream
) noexcept {
    return detail::gwn_try_translate_status("gwn_build_boundary_chain", [&]() -> gwn_status {
        if (!geometry.is_valid())
            return gwn_status::invalid_argument("Geometry accessor is invalid.");
        GWN_RETURN_ON_ERROR(
            detail::gwn_replace_boundary_chain_accessor(
                out.accessor_, geometry.vertex_count(),
                cuda::std::span<Index const>(geometry.tri_i0.data(), geometry.tri_i0.size()),
                cuda::std::span<Index const>(geometry.tri_i1.data(), geometry.tri_i1.size()),
                cuda::std::span<Index const>(geometry.tri_i2.data(), geometry.tri_i2.size()), stream
            )
        );
        out.set_stream(stream);
        return gwn_status::ok();
    });
}

} // namespace gwn
