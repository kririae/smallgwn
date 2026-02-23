#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cub/device/device_reduce.cuh>

#include "gwn/gwn_bvh.cuh"
#include "gwn/gwn_geometry.cuh"

namespace gwn {
namespace detail {
template <class Real, class Index> struct gwn_build_entry {
    gwn_aabb<Real> bounds;
    std::uint8_t kind;
    Index index;
    Index count;
};

template <class Real> __host__ __device__ inline Real gwn_clamp01(Real const value) noexcept {
    if (value < Real(0))
        return Real(0);
    if (value > Real(1))
        return Real(1);
    return value;
}

__host__ __device__ inline std::uint32_t
gwn_expand_bits_10_to_30(std::uint32_t const value) noexcept {
    std::uint32_t x = value & 0x000003ffu;
    x = (x | (x << 16)) & 0x030000ffu;
    x = (x | (x << 8)) & 0x0300f00fu;
    x = (x | (x << 4)) & 0x030c30c3u;
    x = (x | (x << 2)) & 0x09249249u;
    return x;
}

template <class Real>
__host__ __device__ inline std::uint32_t
gwn_encode_morton_30(Real const nx, Real const ny, Real const nz) noexcept {
    auto const x = static_cast<std::uint32_t>(gwn_clamp01(nx) * Real(1023));
    auto const y = static_cast<std::uint32_t>(gwn_clamp01(ny) * Real(1023));
    auto const z = static_cast<std::uint32_t>(gwn_clamp01(nz) * Real(1023));
    return (gwn_expand_bits_10_to_30(x) << 2) | (gwn_expand_bits_10_to_30(y) << 1) |
           gwn_expand_bits_10_to_30(z);
}

template <class Real>
__host__ __device__ inline gwn_aabb<Real>
gwn_aabb_union(gwn_aabb<Real> const &left, gwn_aabb<Real> const &right) noexcept {
    return gwn_aabb<Real>{std::min(left.min_x, right.min_x), std::min(left.min_y, right.min_y),
                          std::min(left.min_z, right.min_z), std::max(left.max_x, right.max_x),
                          std::max(left.max_y, right.max_y), std::max(left.max_z, right.max_z)};
}

template <class Real>
[[nodiscard]] __host__ __device__ inline Real gwn_aabb_half_area(gwn_aabb<Real> const &bounds) {
    Real const dx = std::max(bounds.max_x - bounds.min_x, Real(0));
    Real const dy = std::max(bounds.max_y - bounds.min_y, Real(0));
    Real const dz = std::max(bounds.max_z - bounds.min_z, Real(0));
    return dx * dy + dy * dz + dz * dx;
}

template <class Real>
gwn_status gwn_reduce_minmax_cub(
    cuda::std::span<Real const> const values, gwn_device_array<Real> &min_result,
    gwn_device_array<Real> &max_result, gwn_device_array<std::uint8_t> &temp_storage,
    Real &host_min, Real &host_max, cudaStream_t const stream
) noexcept {
    if (values.empty())
        return gwn_status::invalid_argument("CUB reduction input span is empty.");
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return gwn_status::invalid_argument("CUB reduction input exceeds int32 item count.");

    int const item_count = static_cast<int>(values.size());
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

template <class Real, class Index> struct gwn_compute_triangle_aabbs_and_morton_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    Real scene_min_x{};
    Real scene_min_y{};
    Real scene_min_z{};
    Real scene_inv_x{};
    Real scene_inv_y{};
    Real scene_inv_z{};
    cuda::std::span<gwn_aabb<Real>> primitive_aabbs{};
    cuda::std::span<std::uint32_t> morton_codes{};
    cuda::std::span<Index> primitive_indices{};

    __device__ void operator()(std::size_t const triangle_id) const {
        primitive_indices[triangle_id] = static_cast<Index>(triangle_id);

        Index const ia = geometry.tri_i0[triangle_id];
        Index const ib = geometry.tri_i1[triangle_id];
        Index const ic = geometry.tri_i2[triangle_id];
        if (ia < Index(0) || ib < Index(0) || ic < Index(0) ||
            static_cast<std::size_t>(ia) >= geometry.vertex_count() ||
            static_cast<std::size_t>(ib) >= geometry.vertex_count() ||
            static_cast<std::size_t>(ic) >= geometry.vertex_count()) {
            primitive_aabbs[triangle_id] =
                gwn_aabb<Real>{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
            morton_codes[triangle_id] = 0;
            return;
        }

        std::size_t const a = static_cast<std::size_t>(ia);
        std::size_t const b = static_cast<std::size_t>(ib);
        std::size_t const c = static_cast<std::size_t>(ic);

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
        morton_codes[triangle_id] = gwn_encode_morton_30(
            (center_x - scene_min_x) * scene_inv_x, (center_y - scene_min_y) * scene_inv_y,
            (center_z - scene_min_z) * scene_inv_z
        );
    }
};

template <class Real, class Index> struct gwn_gather_sorted_aabbs_functor {
    cuda::std::span<gwn_aabb<Real> const> unsorted_aabbs{};
    cuda::std::span<Index const> sorted_primitive_indices{};
    cuda::std::span<gwn_aabb<Real>> sorted_aabbs{};

    __device__ void operator()(std::size_t const primitive_id) const {
        std::size_t const source_id =
            static_cast<std::size_t>(sorted_primitive_indices[primitive_id]);
        sorted_aabbs[primitive_id] = unsorted_aabbs[source_id];
    }
};

} // namespace detail
} // namespace gwn
