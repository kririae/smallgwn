#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/memory.h>
#include <thrust/sort.h>

#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"

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

template <class Real, class Index> struct gwn_build_leaf_entries_functor {
    cuda::std::span<gwn_aabb<Real> const> sorted_aabbs{};
    std::size_t primitive_count{};
    std::size_t leaf_primitive_capacity{};
    cuda::std::span<gwn_build_entry<Real, Index>> leaf_entries{};

    __device__ void operator()(std::size_t const leaf_id) const {
        std::size_t const begin = leaf_id * leaf_primitive_capacity;
        std::size_t const end = std::min(begin + leaf_primitive_capacity, primitive_count);

        gwn_aabb<Real> bounds = sorted_aabbs[begin];
        for (std::size_t primitive_id = begin + 1; primitive_id < end; ++primitive_id)
            bounds = gwn_aabb_union(bounds, sorted_aabbs[primitive_id]);

        gwn_build_entry<Real, Index> entry{};
        entry.bounds = bounds;
        entry.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
        entry.index = static_cast<Index>(begin);
        entry.count = static_cast<Index>(end - begin);
        leaf_entries[leaf_id] = entry;
    }
};

template <class Real, class Index> struct gwn_build_bvh4_parent_level_functor {
    cuda::std::span<gwn_build_entry<Real, Index> const> child_entries{};
    std::size_t child_count{};
    cuda::std::span<gwn_build_entry<Real, Index>> parent_entries{};
    cuda::std::span<gwn_bvh4_node_soa<Real, Index>> parent_nodes{};

    __device__ void operator()(std::size_t const parent_id) const {
        gwn_bvh4_node_soa<Real, Index> node{};
        gwn_aabb<Real> parent_bounds{};
        bool has_child = false;

        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot) {
            std::size_t const child_id = parent_id * 4 + static_cast<std::size_t>(child_slot);
            if (child_id >= child_count) {
                node.child_min_x[child_slot] = Real(0);
                node.child_min_y[child_slot] = Real(0);
                node.child_min_z[child_slot] = Real(0);
                node.child_max_x[child_slot] = Real(0);
                node.child_max_y[child_slot] = Real(0);
                node.child_max_z[child_slot] = Real(0);
                node.child_index[child_slot] = 0;
                node.child_count[child_slot] = 0;
                node.child_kind[child_slot] =
                    static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
                continue;
            }

            gwn_build_entry<Real, Index> const child = child_entries[child_id];
            node.child_min_x[child_slot] = child.bounds.min_x;
            node.child_min_y[child_slot] = child.bounds.min_y;
            node.child_min_z[child_slot] = child.bounds.min_z;
            node.child_max_x[child_slot] = child.bounds.max_x;
            node.child_max_y[child_slot] = child.bounds.max_y;
            node.child_max_z[child_slot] = child.bounds.max_z;
            node.child_index[child_slot] = child.index;
            node.child_count[child_slot] = child.count;
            node.child_kind[child_slot] = child.kind;

            if (!has_child) {
                parent_bounds = child.bounds;
                has_child = true;
            } else {
                parent_bounds = gwn_aabb_union(parent_bounds, child.bounds);
            }
        }

        parent_nodes[parent_id] = node;

        gwn_build_entry<Real, Index> parent{};
        parent.bounds = parent_bounds;
        parent.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
        parent.index = static_cast<Index>(parent_id);
        parent.count = 0;
        parent_entries[parent_id] = parent;
    }
};

template <class Real, class Index> struct gwn_patch_child_indices_functor {
    cuda::std::span<gwn_bvh4_node_soa<Real, Index>> nodes{};
    Index child_level_offset{};

    __device__ void operator()(std::size_t const node_id) const {
        gwn_bvh4_node_soa<Real, Index> &node = nodes[node_id];
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot) {
            if (node.child_kind[child_slot] ==
                static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
                node.child_index[child_slot] += child_level_offset;
            }
        }
    }
};

} // namespace detail

template <class Real, class Index>
gwn_status gwn_build_bvh4_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &bvh,
    cudaStream_t stream = gwn_default_stream()
) noexcept try {
    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor is invalid for BVH construction.");

    if (geometry.triangle_count() == 0) {
        detail::gwn_release_bvh_accessor(bvh, stream);
        return gwn_status::ok();
    }
    if (geometry.vertex_count() == 0)
        return gwn_status::invalid_argument("Cannot build BVH with triangles but zero vertices.");

    std::size_t const primitive_count = geometry.triangle_count();
    constexpr std::size_t k_leaf_primitive_capacity = 4;
    constexpr int k_block_size = detail::k_gwn_default_block_size;
    gwn_bvh_accessor<Real, Index> staging_bvh{};
    auto cleanup_staging_bvh = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_accessor(staging_bvh, stream);
    });
    auto commit_staging_bvh = [&]() -> gwn_status {
        detail::gwn_release_bvh_accessor(bvh, stream);
        bvh = staging_bvh;
        cleanup_staging_bvh.release();
        return gwn_status::ok();
    };

    auto exec = thrust::cuda::par.on(stream);
    thrust::device_ptr<Real const> vx_ptr(geometry.vertex_x.data());
    thrust::device_ptr<Real const> vy_ptr(geometry.vertex_y.data());
    thrust::device_ptr<Real const> vz_ptr(geometry.vertex_z.data());
    auto x_pair = thrust::minmax_element(exec, vx_ptr, vx_ptr + geometry.vertex_count());
    auto y_pair = thrust::minmax_element(exec, vy_ptr, vy_ptr + geometry.vertex_count());
    auto z_pair = thrust::minmax_element(exec, vz_ptr, vz_ptr + geometry.vertex_count());

    Real scene_min_x = Real(0);
    Real scene_max_x = Real(0);
    Real scene_min_y = Real(0);
    Real scene_max_y = Real(0);
    Real scene_min_z = Real(0);
    Real scene_max_z = Real(0);
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &scene_min_x, thrust::raw_pointer_cast(x_pair.first), sizeof(Real), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &scene_max_x, thrust::raw_pointer_cast(x_pair.second), sizeof(Real), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &scene_min_y, thrust::raw_pointer_cast(y_pair.first), sizeof(Real), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &scene_max_y, thrust::raw_pointer_cast(y_pair.second), sizeof(Real), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &scene_min_z, thrust::raw_pointer_cast(z_pair.first), sizeof(Real), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &scene_max_z, thrust::raw_pointer_cast(z_pair.second), sizeof(Real), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

    Real const scene_inv_x =
        (scene_max_x > scene_min_x) ? Real(1) / (scene_max_x - scene_min_x) : Real(1);
    Real const scene_inv_y =
        (scene_max_y > scene_min_y) ? Real(1) / (scene_max_y - scene_min_y) : Real(1);
    Real const scene_inv_z =
        (scene_max_z > scene_min_z) ? Real(1) / (scene_max_z - scene_min_z) : Real(1);

    thrust::device_vector<gwn_aabb<Real>> primitive_aabbs(primitive_count);
    thrust::device_vector<std::uint32_t> morton_codes(primitive_count);
    thrust::device_vector<Index> sorted_primitive_indices(primitive_count);
    auto const primitive_aabbs_span = cuda::std::span<gwn_aabb<Real>>(
        thrust::raw_pointer_cast(primitive_aabbs.data()), primitive_count
    );
    auto const morton_codes_span = cuda::std::span<std::uint32_t>(
        thrust::raw_pointer_cast(morton_codes.data()), primitive_count
    );
    auto const sorted_primitive_indices_span = cuda::std::span<Index>(
        thrust::raw_pointer_cast(sorted_primitive_indices.data()), primitive_count
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            detail::gwn_compute_triangle_aabbs_and_morton_functor<Real, Index>{
                geometry, scene_min_x, scene_min_y, scene_min_z, scene_inv_x, scene_inv_y,
                scene_inv_z, primitive_aabbs_span, morton_codes_span, sorted_primitive_indices_span
            },
            stream
        )
    );

    thrust::stable_sort_by_key(
        exec, morton_codes.begin(), morton_codes.end(), sorted_primitive_indices.begin()
    );

    thrust::device_vector<gwn_aabb<Real>> sorted_aabbs(primitive_count);
    auto const sorted_aabbs_span = cuda::std::span<gwn_aabb<Real>>(
        thrust::raw_pointer_cast(sorted_aabbs.data()), primitive_count
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            detail::gwn_gather_sorted_aabbs_functor<Real, Index>{
                primitive_aabbs_span, sorted_primitive_indices_span, sorted_aabbs_span
            },
            stream
        )
    );

    GWN_RETURN_ON_ERROR(
        detail::gwn_copy_device_to_span(
            staging_bvh.primitive_indices, sorted_primitive_indices_span.data(), primitive_count,
            stream
        )
    );

    std::size_t const leaf_count =
        (primitive_count + k_leaf_primitive_capacity - 1) / k_leaf_primitive_capacity;
    thrust::device_vector<detail::gwn_build_entry<Real, Index>> current_entries(leaf_count);
    auto current_entries_span = cuda::std::span<detail::gwn_build_entry<Real, Index>>(
        thrust::raw_pointer_cast(current_entries.data()), leaf_count
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            leaf_count,
            detail::gwn_build_leaf_entries_functor<Real, Index>{
                sorted_aabbs_span, primitive_count, k_leaf_primitive_capacity, current_entries_span
            },
            stream
        )
    );

    std::vector<thrust::device_vector<gwn_bvh4_node_soa<Real, Index>>> levels_bottom;
    std::size_t current_count = leaf_count;
    while (current_count > 1) {
        std::size_t const parent_count = (current_count + 3) / 4;
        thrust::device_vector<detail::gwn_build_entry<Real, Index>> parent_entries(parent_count);
        thrust::device_vector<gwn_bvh4_node_soa<Real, Index>> parent_nodes(parent_count);

        auto const current_entries_const_span =
            cuda::std::span<detail::gwn_build_entry<Real, Index> const>(
                thrust::raw_pointer_cast(current_entries.data()), current_count
            );
        auto const parent_entries_span = cuda::std::span<detail::gwn_build_entry<Real, Index>>(
            thrust::raw_pointer_cast(parent_entries.data()), parent_count
        );
        auto const parent_nodes_span = cuda::std::span<gwn_bvh4_node_soa<Real, Index>>(
            thrust::raw_pointer_cast(parent_nodes.data()), parent_count
        );

        GWN_RETURN_ON_ERROR(
            detail::gwn_launch_linear_kernel<k_block_size>(
                parent_count,
                detail::gwn_build_bvh4_parent_level_functor<Real, Index>{
                    current_entries_const_span, current_count, parent_entries_span,
                    parent_nodes_span
                },
                stream
            )
        );

        levels_bottom.push_back(std::move(parent_nodes));
        current_entries = std::move(parent_entries);
        current_count = parent_count;
    }

    if (levels_bottom.empty()) {
        staging_bvh.root_kind = gwn_bvh_child_kind::k_leaf;
        staging_bvh.root_index = 0;
        staging_bvh.root_count = static_cast<Index>(primitive_count);
        return commit_staging_bvh();
    }

    std::size_t const level_count = levels_bottom.size();
    std::vector<std::size_t> level_node_counts(level_count, 0);
    std::vector<std::size_t> level_offsets(level_count, 0);
    std::size_t total_node_count = 0;
    for (std::size_t level = 0; level < level_count; ++level) {
        std::size_t const bottom_index = level_count - 1 - level;
        level_node_counts[level] = levels_bottom[bottom_index].size();
        level_offsets[level] = total_node_count;
        total_node_count += level_node_counts[level];
    }

    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(staging_bvh.nodes, total_node_count, stream));

    gwn_bvh4_node_soa<Real, Index> *final_nodes =
        const_cast<gwn_bvh4_node_soa<Real, Index> *>(staging_bvh.nodes.data());
    auto final_nodes_span =
        cuda::std::span<gwn_bvh4_node_soa<Real, Index>>(final_nodes, total_node_count);
    for (std::size_t level = 0; level < level_count; ++level) {
        std::size_t const bottom_index = level_count - 1 - level;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
            final_nodes + level_offsets[level],
            thrust::raw_pointer_cast(levels_bottom[bottom_index].data()),
            level_node_counts[level] * sizeof(gwn_bvh4_node_soa<Real, Index>),
            cudaMemcpyDeviceToDevice, stream
        )));
    }

    for (std::size_t level = 0; level + 1 < level_count; ++level) {
        GWN_RETURN_ON_ERROR(
            detail::gwn_launch_linear_kernel<k_block_size>(
                level_node_counts[level],
                detail::gwn_patch_child_indices_functor<Real, Index>{
                    final_nodes_span.subspan(level_offsets[level], level_node_counts[level]),
                    static_cast<Index>(level_offsets[level + 1])
                },
                stream
            )
        );
    }

    staging_bvh.root_kind = gwn_bvh_child_kind::k_internal;
    staging_bvh.root_index = 0;
    staging_bvh.root_count = 0;
    return commit_staging_bvh();
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_build_bvh4_lbvh.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_build_bvh4_lbvh.");
}

namespace detail {

template <int Order, class Real> struct gwn_host_taylor_moment;

template <class Real> struct gwn_host_taylor_moment<0, Real> {
    Real area = Real(0);
    Real area_p_x = Real(0);
    Real area_p_y = Real(0);
    Real area_p_z = Real(0);
    Real average_x = Real(0);
    Real average_y = Real(0);
    Real average_z = Real(0);
    Real n_x = Real(0);
    Real n_y = Real(0);
    Real n_z = Real(0);
    Real max_p_dist2 = Real(0);
};

template <class Real> struct gwn_host_taylor_moment<1, Real> : gwn_host_taylor_moment<0, Real> {
    Real nij_xx = Real(0);
    Real nij_yy = Real(0);
    Real nij_zz = Real(0);
    Real nxy = Real(0);
    Real nyx = Real(0);
    Real nyz = Real(0);
    Real nzy = Real(0);
    Real nzx = Real(0);
    Real nxz = Real(0);
};

template <class Real>
[[nodiscard]] inline Real gwn_bounds_max_p_dist2(
    gwn_aabb<Real> const &bounds, Real const average_x, Real const average_y, Real const average_z
) {
    Real const dx = std::max(average_x - bounds.min_x, bounds.max_x - average_x);
    Real const dy = std::max(average_y - bounds.min_y, bounds.max_y - average_y);
    Real const dz = std::max(average_z - bounds.min_z, bounds.max_z - average_z);
    return dx * dx + dy * dy + dz * dz;
}

template <int Order, class Real>
inline void
gwn_zero_taylor_child(gwn_bvh4_taylor_node_soa<Order, Real> &node, int const child_slot) {
    node.child_max_p_dist2[child_slot] = Real(0);
    node.child_average_x[child_slot] = Real(0);
    node.child_average_y[child_slot] = Real(0);
    node.child_average_z[child_slot] = Real(0);
    node.child_n_x[child_slot] = Real(0);
    node.child_n_y[child_slot] = Real(0);
    node.child_n_z[child_slot] = Real(0);
    if constexpr (Order >= 1) {
        node.child_nij_xx[child_slot] = Real(0);
        node.child_nij_yy[child_slot] = Real(0);
        node.child_nij_zz[child_slot] = Real(0);
        node.child_nxy_nyx[child_slot] = Real(0);
        node.child_nyz_nzy[child_slot] = Real(0);
        node.child_nzx_nxz[child_slot] = Real(0);
    }
}

template <int Order, class Real>
inline void gwn_write_taylor_child(
    gwn_bvh4_taylor_node_soa<Order, Real> &node, int const child_slot,
    gwn_host_taylor_moment<Order, Real> const &moment
) {
    node.child_max_p_dist2[child_slot] = moment.max_p_dist2;
    node.child_average_x[child_slot] = moment.average_x;
    node.child_average_y[child_slot] = moment.average_y;
    node.child_average_z[child_slot] = moment.average_z;
    node.child_n_x[child_slot] = moment.n_x;
    node.child_n_y[child_slot] = moment.n_y;
    node.child_n_z[child_slot] = moment.n_z;
    if constexpr (Order >= 1) {
        node.child_nij_xx[child_slot] = moment.nij_xx;
        node.child_nij_yy[child_slot] = moment.nij_yy;
        node.child_nij_zz[child_slot] = moment.nij_zz;
        node.child_nxy_nyx[child_slot] = moment.nxy + moment.nyx;
        node.child_nyz_nzy[child_slot] = moment.nyz + moment.nzy;
        node.child_nzx_nxz[child_slot] = moment.nzx + moment.nxz;
    }
}

template <class T>
gwn_status gwn_copy_device_span_to_host(
    std::vector<T> &host, cuda::std::span<T const> const device, cudaStream_t const stream
) noexcept {
    host.resize(device.size());
    if (host.empty())
        return gwn_status::ok();
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        host.data(), device.data(), host.size() * sizeof(T), cudaMemcpyDeviceToHost, stream
    )));
    return gwn_cuda_to_status(cudaStreamSynchronize(stream));
}

template <class T>
gwn_status gwn_upload_host_to_device_span(
    cuda::std::span<T const> &device, std::vector<T> const &host, cudaStream_t const stream
) noexcept {
    GWN_RETURN_ON_ERROR(gwn_allocate_span(device, host.size(), stream));
    if (host.empty())
        return gwn_status::ok();

    auto cleanup_device = gwn_make_scope_exit([&]() noexcept {
        (void)gwn_cuda_free(const_cast<T *>(device.data()), stream);
        device = {};
    });
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        const_cast<T *>(device.data()), host.data(), host.size() * sizeof(T),
        cudaMemcpyHostToDevice, stream
    )));
    cleanup_device.release();
    return gwn_status::ok();
}

} // namespace detail

template <int Order, class Real, class Index>
gwn_status gwn_build_bvh4_lbvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &bvh,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    static_assert(
        Order == 0 || Order == 1,
        "gwn_build_bvh4_lbvh_taylor currently supports Order 0 and Order 1."
    );

    GWN_RETURN_ON_ERROR(gwn_build_bvh4_lbvh(geometry, bvh, stream));

    detail::gwn_release_bvh_span(bvh.taylor_order2_nodes, stream);
    detail::gwn_release_bvh_span(bvh.taylor_order1_nodes, stream);
    detail::gwn_release_bvh_span(bvh.taylor_order0_nodes, stream);

    if (!bvh.has_internal_root())
        return gwn_status::ok();

    using moment_type = detail::gwn_host_taylor_moment<Order, Real>;
    using taylor_node_type = gwn_bvh4_taylor_node_soa<Order, Real>;

    std::vector<Real> vertex_x{};
    std::vector<Real> vertex_y{};
    std::vector<Real> vertex_z{};
    std::vector<Index> tri_i0{};
    std::vector<Index> tri_i1{};
    std::vector<Index> tri_i2{};
    std::vector<Index> sorted_primitive_indices{};
    std::vector<gwn_bvh4_node_soa<Real, Index>> host_nodes{};
    GWN_RETURN_ON_ERROR(detail::gwn_copy_device_span_to_host(vertex_x, geometry.vertex_x, stream));
    GWN_RETURN_ON_ERROR(detail::gwn_copy_device_span_to_host(vertex_y, geometry.vertex_y, stream));
    GWN_RETURN_ON_ERROR(detail::gwn_copy_device_span_to_host(vertex_z, geometry.vertex_z, stream));
    GWN_RETURN_ON_ERROR(detail::gwn_copy_device_span_to_host(tri_i0, geometry.tri_i0, stream));
    GWN_RETURN_ON_ERROR(detail::gwn_copy_device_span_to_host(tri_i1, geometry.tri_i1, stream));
    GWN_RETURN_ON_ERROR(detail::gwn_copy_device_span_to_host(tri_i2, geometry.tri_i2, stream));
    GWN_RETURN_ON_ERROR(
        detail::gwn_copy_device_span_to_host(
            sorted_primitive_indices, bvh.primitive_indices, stream
        )
    );
    GWN_RETURN_ON_ERROR(detail::gwn_copy_device_span_to_host(host_nodes, bvh.nodes, stream));

    std::size_t const primitive_count = tri_i0.size();
    std::vector<gwn_aabb<Real>> primitive_bounds(primitive_count);
    std::vector<moment_type> primitive_moments(primitive_count);

    auto const invalid_bounds =
        gwn_aabb<Real>{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
    for (std::size_t primitive_id = 0; primitive_id < primitive_count; ++primitive_id) {
        moment_type moment{};
        primitive_bounds[primitive_id] = invalid_bounds;

        Index const ia = tri_i0[primitive_id];
        Index const ib = tri_i1[primitive_id];
        Index const ic = tri_i2[primitive_id];
        if (ia < Index(0) || ib < Index(0) || ic < Index(0))
            continue;

        std::size_t const a_index = static_cast<std::size_t>(ia);
        std::size_t const b_index = static_cast<std::size_t>(ib);
        std::size_t const c_index = static_cast<std::size_t>(ic);
        if (a_index >= vertex_x.size() || b_index >= vertex_x.size() || c_index >= vertex_x.size())
            continue;

        Real const ax = vertex_x[a_index];
        Real const ay = vertex_y[a_index];
        Real const az = vertex_z[a_index];
        Real const bx = vertex_x[b_index];
        Real const by = vertex_y[b_index];
        Real const bz = vertex_z[b_index];
        Real const cx = vertex_x[c_index];
        Real const cy = vertex_y[c_index];
        Real const cz = vertex_z[c_index];

        gwn_aabb<Real> const bounds{
            std::min(ax, std::min(bx, cx)), std::min(ay, std::min(by, cy)),
            std::min(az, std::min(bz, cz)), std::max(ax, std::max(bx, cx)),
            std::max(ay, std::max(by, cy)), std::max(az, std::max(bz, cz)),
        };
        primitive_bounds[primitive_id] = bounds;

        Real const abx = bx - ax;
        Real const aby = by - ay;
        Real const abz = bz - az;
        Real const acx = cx - ax;
        Real const acy = cy - ay;
        Real const acz = cz - az;

        moment.n_x = Real(0.5) * (aby * acz - abz * acy);
        moment.n_y = Real(0.5) * (abz * acx - abx * acz);
        moment.n_z = Real(0.5) * (abx * acy - aby * acx);

        Real const area2 =
            moment.n_x * moment.n_x + moment.n_y * moment.n_y + moment.n_z * moment.n_z;
        moment.area = std::sqrt(std::max(area2, Real(0)));
        moment.average_x = (ax + bx + cx) / Real(3);
        moment.average_y = (ay + by + cy) / Real(3);
        moment.average_z = (az + bz + cz) / Real(3);
        moment.area_p_x = moment.average_x * moment.area;
        moment.area_p_y = moment.average_y * moment.area;
        moment.area_p_z = moment.average_z * moment.area;
        moment.max_p_dist2 = detail::gwn_bounds_max_p_dist2(
            bounds, moment.average_x, moment.average_y, moment.average_z
        );

        primitive_moments[primitive_id] = moment;
    }

    auto compute_leaf_moment = [&](Index const begin_index, Index const count) {
        moment_type leaf{};
        bool has_primitive = false;
        gwn_aabb<Real> leaf_bounds = invalid_bounds;

        for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
            Index const sorted_slot = begin_index + primitive_offset;
            if (sorted_slot < Index(0))
                continue;
            std::size_t const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
            if (sorted_slot_u >= sorted_primitive_indices.size())
                continue;

            Index const primitive_index = sorted_primitive_indices[sorted_slot_u];
            if (primitive_index < Index(0))
                continue;
            std::size_t const primitive_index_u = static_cast<std::size_t>(primitive_index);
            if (primitive_index_u >= primitive_moments.size())
                continue;

            moment_type const &primitive = primitive_moments[primitive_index_u];
            gwn_aabb<Real> const &primitive_bounds_current = primitive_bounds[primitive_index_u];
            if (!has_primitive) {
                leaf_bounds = primitive_bounds_current;
                has_primitive = true;
            } else {
                leaf_bounds = detail::gwn_aabb_union(leaf_bounds, primitive_bounds_current);
            }

            leaf.area += primitive.area;
            leaf.area_p_x += primitive.area_p_x;
            leaf.area_p_y += primitive.area_p_y;
            leaf.area_p_z += primitive.area_p_z;
            leaf.n_x += primitive.n_x;
            leaf.n_y += primitive.n_y;
            leaf.n_z += primitive.n_z;
        }

        if (!has_primitive)
            return leaf;

        if (leaf.area > Real(0)) {
            leaf.average_x = leaf.area_p_x / leaf.area;
            leaf.average_y = leaf.area_p_y / leaf.area;
            leaf.average_z = leaf.area_p_z / leaf.area;
        } else {
            leaf.average_x = (leaf_bounds.min_x + leaf_bounds.max_x) * Real(0.5);
            leaf.average_y = (leaf_bounds.min_y + leaf_bounds.max_y) * Real(0.5);
            leaf.average_z = (leaf_bounds.min_z + leaf_bounds.max_z) * Real(0.5);
        }

        leaf.max_p_dist2 = detail::gwn_bounds_max_p_dist2(
            leaf_bounds, leaf.average_x, leaf.average_y, leaf.average_z
        );

        if constexpr (Order >= 1) {
            for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
                Index const sorted_slot = begin_index + primitive_offset;
                if (sorted_slot < Index(0))
                    continue;
                std::size_t const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
                if (sorted_slot_u >= sorted_primitive_indices.size())
                    continue;

                Index const primitive_index = sorted_primitive_indices[sorted_slot_u];
                if (primitive_index < Index(0))
                    continue;
                std::size_t const primitive_index_u = static_cast<std::size_t>(primitive_index);
                if (primitive_index_u >= primitive_moments.size())
                    continue;

                moment_type const &primitive = primitive_moments[primitive_index_u];
                Real const dx = primitive.average_x - leaf.average_x;
                Real const dy = primitive.average_y - leaf.average_y;
                Real const dz = primitive.average_z - leaf.average_z;

                leaf.nij_xx += primitive.nij_xx + primitive.n_x * dx;
                leaf.nij_yy += primitive.nij_yy + primitive.n_y * dy;
                leaf.nij_zz += primitive.nij_zz + primitive.n_z * dz;
                leaf.nxy += primitive.nxy + primitive.n_x * dy;
                leaf.nyx += primitive.nyx + primitive.n_y * dx;
                leaf.nyz += primitive.nyz + primitive.n_y * dz;
                leaf.nzy += primitive.nzy + primitive.n_z * dy;
                leaf.nzx += primitive.nzx + primitive.n_z * dx;
                leaf.nxz += primitive.nxz + primitive.n_x * dz;
            }
        }

        return leaf;
    };

    std::vector<taylor_node_type> host_taylor_nodes(host_nodes.size());
    std::vector<moment_type> node_moments(host_nodes.size());
    std::vector<std::uint8_t> node_ready(host_nodes.size(), std::uint8_t(0));

    std::function<moment_type(Index)> compute_internal_moment;
    compute_internal_moment = [&](Index const node_index) -> moment_type {
        if (node_index < Index(0))
            return moment_type{};

        std::size_t const node_index_u = static_cast<std::size_t>(node_index);
        if (node_index_u >= host_nodes.size())
            return moment_type{};
        if (node_ready[node_index_u] != 0)
            return node_moments[node_index_u];

        gwn_bvh4_node_soa<Real, Index> const &node = host_nodes[node_index_u];
        taylor_node_type &taylor_node = host_taylor_nodes[node_index_u];

        std::array<moment_type, 4> child_moments{};
        std::array<gwn_aabb<Real>, 4> child_bounds{};
        std::array<bool, 4> child_valid{};
        child_valid.fill(false);

        bool has_child = false;
        gwn_aabb<Real> merged_bounds = invalid_bounds;
        moment_type parent{};

        for (int child_slot = 0; child_slot < 4; ++child_slot) {
            auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (child_kind == gwn_bvh_child_kind::k_invalid) {
                detail::gwn_zero_taylor_child<Order>(taylor_node, child_slot);
                continue;
            }

            gwn_aabb<Real> const bounds{
                node.child_min_x[child_slot], node.child_min_y[child_slot],
                node.child_min_z[child_slot], node.child_max_x[child_slot],
                node.child_max_y[child_slot], node.child_max_z[child_slot],
            };
            child_bounds[child_slot] = bounds;

            moment_type child_moment{};
            if (child_kind == gwn_bvh_child_kind::k_internal) {
                child_moment = compute_internal_moment(node.child_index[child_slot]);
            } else if (child_kind == gwn_bvh_child_kind::k_leaf) {
                child_moment =
                    compute_leaf_moment(node.child_index[child_slot], node.child_count[child_slot]);
            } else {
                detail::gwn_zero_taylor_child<Order>(taylor_node, child_slot);
                continue;
            }

            child_moments[child_slot] = child_moment;
            child_valid[child_slot] = true;
            detail::gwn_write_taylor_child<Order>(taylor_node, child_slot, child_moment);

            if (!has_child) {
                merged_bounds = bounds;
                has_child = true;
            } else {
                merged_bounds = detail::gwn_aabb_union(merged_bounds, bounds);
            }

            parent.area += child_moment.area;
            parent.area_p_x += child_moment.area_p_x;
            parent.area_p_y += child_moment.area_p_y;
            parent.area_p_z += child_moment.area_p_z;
            parent.n_x += child_moment.n_x;
            parent.n_y += child_moment.n_y;
            parent.n_z += child_moment.n_z;
        }

        if (!has_child) {
            node_moments[node_index_u] = parent;
            node_ready[node_index_u] = std::uint8_t(1);
            return parent;
        }

        if (parent.area > Real(0)) {
            parent.average_x = parent.area_p_x / parent.area;
            parent.average_y = parent.area_p_y / parent.area;
            parent.average_z = parent.area_p_z / parent.area;
        } else {
            parent.average_x = (merged_bounds.min_x + merged_bounds.max_x) * Real(0.5);
            parent.average_y = (merged_bounds.min_y + merged_bounds.max_y) * Real(0.5);
            parent.average_z = (merged_bounds.min_z + merged_bounds.max_z) * Real(0.5);
        }

        parent.max_p_dist2 = detail::gwn_bounds_max_p_dist2(
            merged_bounds, parent.average_x, parent.average_y, parent.average_z
        );

        if constexpr (Order >= 1) {
            for (int child_slot = 0; child_slot < 4; ++child_slot) {
                if (!child_valid[child_slot])
                    continue;

                moment_type const &child = child_moments[child_slot];
                Real const dx = child.average_x - parent.average_x;
                Real const dy = child.average_y - parent.average_y;
                Real const dz = child.average_z - parent.average_z;

                parent.nij_xx += child.nij_xx + child.n_x * dx;
                parent.nij_yy += child.nij_yy + child.n_y * dy;
                parent.nij_zz += child.nij_zz + child.n_z * dz;
                parent.nxy += child.nxy + child.n_x * dy;
                parent.nyx += child.nyx + child.n_y * dx;
                parent.nyz += child.nyz + child.n_y * dz;
                parent.nzy += child.nzy + child.n_z * dy;
                parent.nzx += child.nzx + child.n_z * dx;
                parent.nxz += child.nxz + child.n_x * dz;
            }
        }

        node_moments[node_index_u] = parent;
        node_ready[node_index_u] = std::uint8_t(1);
        return parent;
    };

    (void)compute_internal_moment(bvh.root_index);
    if constexpr (Order == 0) {
        return detail::gwn_upload_host_to_device_span(
            bvh.taylor_order0_nodes, host_taylor_nodes, stream
        );
    } else {
        return detail::gwn_upload_host_to_device_span(
            bvh.taylor_order1_nodes, host_taylor_nodes, stream
        );
    }
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_build_bvh4_lbvh_taylor.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_build_bvh4_lbvh_taylor.");
}

} // namespace gwn
