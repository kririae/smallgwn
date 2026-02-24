#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "../gwn_kernel_utils.cuh"
#include "gwn_bvh_status_helpers.cuh"

namespace gwn {
namespace detail {
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_clamp01(Real const value) noexcept {
    if (value < Real(0))
        return Real(0);
    if (value > Real(1))
        return Real(1);
    return value;
}

/// \brief Morton encoding traits specialized by Morton code width.
///
/// \remark Supported types are \c std::uint32_t (10 bits/axis) and
///         \c std::uint64_t (21 bits/axis).
template <class MortonCode> struct gwn_morton_traits;

template <> struct gwn_morton_traits<std::uint32_t> {
    static constexpr std::uint32_t k_axis_quant_max = 1023u;

    __host__ __device__ static inline std::uint32_t
    expand_bits(std::uint32_t const value) noexcept {
        std::uint32_t x = value & 0x000003ffu;
        x = (x | (x << 16)) & 0x030000ffu;
        x = (x | (x << 8)) & 0x0300f00fu;
        x = (x | (x << 4)) & 0x030c30c3u;
        x = (x | (x << 2)) & 0x09249249u;
        return x;
    }
};

template <> struct gwn_morton_traits<std::uint64_t> {
    static constexpr std::uint64_t k_axis_quant_max = 2097151ull;

    __host__ __device__ static inline std::uint64_t
    expand_bits(std::uint64_t const value) noexcept {
        std::uint64_t x = value & 0x1fffffu;
        x = (x | (x << 32)) & 0x1f00000000ffffull;
        x = (x | (x << 16)) & 0x1f0000ff0000ffull;
        x = (x | (x << 8)) & 0x100f00f00f00f00full;
        x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
        x = (x | (x << 2)) & 0x1249249249249249ull;
        return x;
    }
};

/// \brief Encode normalized coordinates into an interleaved Morton key.
///
/// \tparam MortonCode Morton output type (\c std::uint32_t or \c std::uint64_t).
/// \tparam Real       Floating-point coordinate type.
///
/// \param nx Normalized x in [0,1].
/// \param ny Normalized y in [0,1].
/// \param nz Normalized z in [0,1].
///
/// \return Interleaved Morton key using \p MortonCode precision.
template <class MortonCode, gwn_real_type Real>
__host__ __device__ inline MortonCode
gwn_encode_morton(Real const nx, Real const ny, Real const nz) noexcept {
    static_assert(
        std::is_same_v<MortonCode, std::uint32_t> || std::is_same_v<MortonCode, std::uint64_t>,
        "MortonCode must be std::uint32_t or std::uint64_t."
    );
    using traits = gwn_morton_traits<MortonCode>;
    auto const x = static_cast<MortonCode>(gwn_clamp01(nx) * Real(traits::k_axis_quant_max));
    auto const y = static_cast<MortonCode>(gwn_clamp01(ny) * Real(traits::k_axis_quant_max));
    auto const z = static_cast<MortonCode>(gwn_clamp01(nz) * Real(traits::k_axis_quant_max));
    return (traits::expand_bits(x) << 2) | (traits::expand_bits(y) << 1) | traits::expand_bits(z);
}

template <gwn_real_type Real>
__host__ __device__ inline gwn_aabb<Real>
gwn_aabb_union(gwn_aabb<Real> const &left, gwn_aabb<Real> const &right) noexcept {
    return gwn_aabb<Real>{std::min(left.min_x, right.min_x), std::min(left.min_y, right.min_y),
                          std::min(left.min_z, right.min_z), std::max(left.max_x, right.max_x),
                          std::max(left.max_y, right.max_y), std::max(left.max_z, right.max_z)};
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ inline Real gwn_aabb_half_area(gwn_aabb<Real> const &bounds) {
    Real const dx = std::max(bounds.max_x - bounds.min_x, Real(0));
    Real const dy = std::max(bounds.max_y - bounds.min_y, Real(0));
    Real const dz = std::max(bounds.max_z - bounds.min_z, Real(0));
    return dx * dy + dy * dz + dz * dx;
}

template <gwn_real_type Real>
gwn_status gwn_reduce_minmax_cub(
    cuda::std::span<Real const> const values, gwn_device_array<Real> &min_result,
    gwn_device_array<Real> &max_result, gwn_device_array<std::uint8_t> &temp_storage,
    Real &host_min, Real &host_max, cudaStream_t const stream
) noexcept {
    if (values.empty())
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_topology_preprocess, "CUB reduction input span is empty."
        );
    auto const item_count = static_cast<std::uint64_t>(values.size());
    std::size_t min_temp_bytes = 0;
    std::size_t max_temp_bytes = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceReduce::Min(
            nullptr, min_temp_bytes, values.data(), min_result.data(), item_count, stream
        )
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceReduce::Max(
            nullptr, max_temp_bytes, values.data(), max_result.data(), item_count, stream
        )
    ));

    std::size_t const required_temp_bytes = std::max(min_temp_bytes, max_temp_bytes);
    if (temp_storage.size() < required_temp_bytes)
        GWN_RETURN_ON_ERROR(temp_storage.resize(required_temp_bytes, stream));

    std::size_t temp_storage_bytes = required_temp_bytes;

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceReduce::Min(
            temp_storage.data(), temp_storage_bytes, values.data(), min_result.data(), item_count,
            stream
        )
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceReduce::Max(
            temp_storage.data(), temp_storage_bytes, values.data(), max_result.data(), item_count,
            stream
        )
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemcpyAsync(&host_min, min_result.data(), sizeof(Real), cudaMemcpyDeviceToHost, stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemcpyAsync(&host_max, max_result.data(), sizeof(Real), cudaMemcpyDeviceToHost, stream)
    ));
    return gwn_status::ok();
}

template <gwn_real_type Real, gwn_index_type Index, class MortonCode>
struct gwn_compute_triangle_aabbs_and_morton_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    Real scene_min_x{};
    Real scene_min_y{};
    Real scene_min_z{};
    Real scene_inv_x{};
    Real scene_inv_y{};
    Real scene_inv_z{};
    cuda::std::span<gwn_aabb<Real>> primitive_aabbs{};
    cuda::std::span<MortonCode> morton_codes{};
    cuda::std::span<Index> primitive_indices{};

    __device__ void operator()(std::size_t const triangle_id) const {
        primitive_indices[triangle_id] = static_cast<Index>(triangle_id);

        Index const ia = geometry.tri_i0[triangle_id];
        Index const ib = geometry.tri_i1[triangle_id];
        Index const ic = geometry.tri_i2[triangle_id];
        if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
            !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
            !gwn_index_in_bounds(ic, geometry.vertex_count())) {
            primitive_aabbs[triangle_id] =
                gwn_aabb<Real>{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
            morton_codes[triangle_id] = 0;
            return;
        }

        auto const a = static_cast<std::size_t>(ia);
        auto const b = static_cast<std::size_t>(ib);
        auto const c = static_cast<std::size_t>(ic);

        Real const ax = geometry.vertex_x[a];
        Real const ay = geometry.vertex_y[a];
        Real const az = geometry.vertex_z[a];
        Real const bx = geometry.vertex_x[b];
        Real const by = geometry.vertex_y[b];
        Real const bz = geometry.vertex_z[b];
        Real const cx = geometry.vertex_x[c];
        Real const cy = geometry.vertex_y[c];
        Real const cz = geometry.vertex_z[c];

        gwn_aabb<Real> const bounds{
            std::min(ax, std::min(bx, cx)), std::min(ay, std::min(by, cy)),
            std::min(az, std::min(bz, cz)), std::max(ax, std::max(bx, cx)),
            std::max(ay, std::max(by, cy)), std::max(az, std::max(bz, cz)),
        };
        primitive_aabbs[triangle_id] = bounds;

        Real const center_x = (bounds.min_x + bounds.max_x) * Real(0.5);
        Real const center_y = (bounds.min_y + bounds.max_y) * Real(0.5);
        Real const center_z = (bounds.min_z + bounds.max_z) * Real(0.5);
        morton_codes[triangle_id] = gwn_encode_morton<MortonCode>(
            (center_x - scene_min_x) * scene_inv_x, (center_y - scene_min_y) * scene_inv_y,
            (center_z - scene_min_z) * scene_inv_z
        );
    }
};

template <gwn_real_type Real, gwn_index_type Index> struct gwn_gather_sorted_aabbs_functor {
    cuda::std::span<gwn_aabb<Real> const> unsorted_aabbs{};
    cuda::std::span<Index const> sorted_primitive_indices{};
    cuda::std::span<gwn_aabb<Real>> sorted_aabbs{};

    __device__ void operator()(std::size_t const primitive_id) const {
        std::size_t const source_id =
            static_cast<std::size_t>(sorted_primitive_indices[primitive_id]);
        sorted_aabbs[primitive_id] = unsorted_aabbs[source_id];
    }
};

template <gwn_real_type Real> struct gwn_scene_aabb {
    Real min_x{}, min_y{}, min_z{};
    Real max_x{}, max_y{}, max_z{};
    Real inv_x{}, inv_y{}, inv_z{};
};

template <gwn_real_type Real, gwn_index_type Index, class MortonCode>
struct gwn_topology_build_preprocess {
    gwn_device_array<Index> sorted_primitive_indices{};
    gwn_device_array<MortonCode> sorted_morton_codes{};
    gwn_device_array<gwn_aabb<Real>> primitive_aabbs{};
};

template <gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_compute_scene_aabb(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_scene_aabb<Real> &result,
    cudaStream_t const stream
) noexcept {
    gwn_device_array<Real> axis_min{};
    gwn_device_array<Real> axis_max{};
    gwn_device_array<std::uint8_t> reduce_temp{};
    GWN_RETURN_ON_ERROR(axis_min.resize(1, stream));
    GWN_RETURN_ON_ERROR(axis_max.resize(1, stream));
    GWN_RETURN_ON_ERROR(gwn_reduce_minmax_cub(
        geometry.vertex_x, axis_min, axis_max, reduce_temp, result.min_x, result.max_x, stream
    ));
    GWN_RETURN_ON_ERROR(gwn_reduce_minmax_cub(
        geometry.vertex_y, axis_min, axis_max, reduce_temp, result.min_y, result.max_y, stream
    ));
    GWN_RETURN_ON_ERROR(gwn_reduce_minmax_cub(
        geometry.vertex_z, axis_min, axis_max, reduce_temp, result.min_z, result.max_z, stream
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

    auto const safe_inv = [](Real const lo, Real const hi) noexcept {
        return (hi > lo) ? Real(1) / (hi - lo) : Real(1);
    };
    result.inv_x = safe_inv(result.min_x, result.max_x);
    result.inv_y = safe_inv(result.min_y, result.max_y);
    result.inv_z = safe_inv(result.min_z, result.max_z);
    return gwn_status::ok();
}

/// \brief Compute primitive AABBs + Morton keys and radix-sort primitive order.
///
/// \tparam MortonCode Morton key type (\c std::uint32_t or \c std::uint64_t).
/// \tparam Real       Floating-point coordinate type.
/// \tparam Index      Primitive index type.
///
/// \param geometry                 Source geometry accessor.
/// \param scene                    Precomputed scene bounds/inverses.
/// \param sorted_primitive_indices Output sorted primitive indices.
/// \param sorted_morton_codes      Output sorted Morton keys.
/// \param primitive_aabbs          Output unsorted primitive bounds.
/// \param stream                   CUDA stream.
///
/// \return \c gwn_status::ok() on success.
template <class MortonCode, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_compute_and_sort_morton(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_scene_aabb<Real> const &scene,
    gwn_device_array<Index> &sorted_primitive_indices,
    gwn_device_array<MortonCode> &sorted_morton_codes,
    gwn_device_array<gwn_aabb<Real>> &primitive_aabbs, cudaStream_t const stream
) noexcept {
    static_assert(
        std::is_same_v<MortonCode, std::uint32_t> || std::is_same_v<MortonCode, std::uint64_t>,
        "MortonCode must be std::uint32_t or std::uint64_t."
    );
    constexpr int k_block_size = k_gwn_default_block_size;
    std::size_t const primitive_count = geometry.triangle_count();

    gwn_device_array<MortonCode> morton_codes{};
    gwn_device_array<Index> primitive_indices{};
    GWN_RETURN_ON_ERROR(primitive_aabbs.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(morton_codes.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(primitive_indices.resize(primitive_count, stream));

    auto const primitive_aabbs_span = primitive_aabbs.span();
    auto const morton_codes_span = morton_codes.span();
    auto const primitive_indices_span = primitive_indices.span();
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            gwn_compute_triangle_aabbs_and_morton_functor<Real, Index, MortonCode>{
                geometry, scene.min_x, scene.min_y, scene.min_z, scene.inv_x, scene.inv_y,
                scene.inv_z, primitive_aabbs_span, morton_codes_span, primitive_indices_span
            },
            stream
        )
    );

    auto const radix_item_count = static_cast<std::uint64_t>(primitive_count);

    gwn_device_array<std::uint8_t> radix_sort_temp{};
    GWN_RETURN_ON_ERROR(sorted_morton_codes.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(sorted_primitive_indices.resize(primitive_count, stream));
    auto const radix_sort_end_bit = static_cast<int>(sizeof(MortonCode) * 8);

    std::size_t radix_sort_temp_bytes = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            nullptr, radix_sort_temp_bytes, morton_codes_span.data(), sorted_morton_codes.data(),
            primitive_indices_span.data(), sorted_primitive_indices.data(), radix_item_count, 0,
            radix_sort_end_bit, stream
        )
    ));
    GWN_RETURN_ON_ERROR(radix_sort_temp.resize(radix_sort_temp_bytes, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            radix_sort_temp.data(), radix_sort_temp_bytes, morton_codes_span.data(),
            sorted_morton_codes.data(), primitive_indices_span.data(),
            sorted_primitive_indices.data(), radix_item_count, 0, radix_sort_end_bit, stream
        )
    ));

    return gwn_status::ok();
}

template <class MortonCode, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_bvh_topology_build_preprocess(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_topology_build_preprocess<Real, Index, MortonCode> &preprocess, cudaStream_t const stream
) noexcept {
    gwn_scene_aabb<Real> scene{};
    GWN_RETURN_ON_ERROR(gwn_compute_scene_aabb(geometry, scene, stream));
    return gwn_compute_and_sort_morton<MortonCode>(
        geometry, scene, preprocess.sorted_primitive_indices, preprocess.sorted_morton_codes,
        preprocess.primitive_aabbs, stream
    );
}

} // namespace detail
} // namespace gwn
