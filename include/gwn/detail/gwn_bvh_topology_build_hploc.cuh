#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "../gwn_kernel_utils.cuh"
#include "gwn_bvh_topology_build_binary.cuh"
#include "gwn_bvh_topology_build_common.cuh"

namespace gwn {
namespace detail {

inline constexpr std::uint32_t k_gwn_hploc_search_radius = 8u;
inline constexpr std::uint32_t k_gwn_hploc_merging_threshold = 16u;
inline constexpr std::uint32_t k_gwn_hploc_invalid_u32 = std::numeric_limits<std::uint32_t>::max();

template <class Real> struct gwn_hploc_node {
    gwn_aabb<Real> bounds{};
    std::uint32_t left_child = k_gwn_hploc_invalid_u32;
    std::uint32_t right_child = k_gwn_hploc_invalid_u32;
};

template <class Real> struct gwn_hploc_init_full_nodes_functor {
    cuda::std::span<gwn_aabb<Real> const> sorted_leaf_aabbs{};
    cuda::std::span<gwn_hploc_node<Real>> full_nodes{};
    cuda::std::span<std::uint32_t> cluster_indices{};

    __device__ void operator()(std::size_t const leaf_id) const {
        if (leaf_id >= sorted_leaf_aabbs.size() || leaf_id >= cluster_indices.size() ||
            leaf_id >= full_nodes.size()) {
            return;
        }

        gwn_hploc_node<Real> node{};
        node.bounds = sorted_leaf_aabbs[leaf_id];
        node.left_child = k_gwn_hploc_invalid_u32;
        node.right_child = static_cast<std::uint32_t>(leaf_id);
        full_nodes[leaf_id] = node;
        cluster_indices[leaf_id] = static_cast<std::uint32_t>(leaf_id);
    }
};

template <class Real>
[[nodiscard]] __device__ inline gwn_aabb<Real>
gwn_hploc_shfl_aabb(unsigned int const mask, gwn_aabb<Real> const &value, int const src_lane) {
    gwn_aabb<Real> result{};
    result.min_x = __shfl_sync(mask, value.min_x, src_lane);
    result.min_y = __shfl_sync(mask, value.min_y, src_lane);
    result.min_z = __shfl_sync(mask, value.min_z, src_lane);
    result.max_x = __shfl_sync(mask, value.max_x, src_lane);
    result.max_y = __shfl_sync(mask, value.max_y, src_lane);
    result.max_z = __shfl_sync(mask, value.max_z, src_lane);
    return result;
}

[[nodiscard]] __device__ inline uint2
gwn_hploc_shfl_uint2(unsigned int const mask, uint2 const value, int const src_lane) {
    uint2 result{};
    result.x = __shfl_sync(mask, value.x, src_lane);
    result.y = __shfl_sync(mask, value.y, src_lane);
    return result;
}

template <class Real>
[[nodiscard]] __device__ inline Real
gwn_hploc_union_half_area(gwn_aabb<Real> const &lhs, gwn_aabb<Real> const &rhs) {
    return gwn_aabb_half_area(gwn_aabb_union(lhs, rhs));
}

template <class Real, class Index>
[[nodiscard]] __device__ inline bool gwn_hploc_decode_child(
    std::uint32_t const full_child, std::uint32_t const primitive_count, std::uint8_t &kind_out,
    Index &index_out
) {
    if (full_child == k_gwn_hploc_invalid_u32)
        return false;
    if (full_child < primitive_count) {
        kind_out = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
        index_out = static_cast<Index>(full_child);
        return true;
    }

    std::uint32_t const internal_offset = full_child - primitive_count;
    kind_out = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
    index_out = static_cast<Index>(internal_offset);
    return true;
}

template <class Real, class Index> struct gwn_hploc_emit_binary_nodes_functor {
    cuda::std::span<gwn_hploc_node<Real> const> full_nodes{};
    cuda::std::span<gwn_binary_node<Index>> binary_nodes{};
    cuda::std::span<Index> internal_parent{};
    cuda::std::span<std::uint8_t> internal_parent_slot{};
    cuda::std::span<Index> leaf_parent{};
    cuda::std::span<std::uint8_t> leaf_parent_slot{};
    std::uint32_t primitive_count = 0;

    __device__ void operator()(std::size_t const internal_id_u) const {
        if (internal_id_u >= binary_nodes.size() || internal_id_u >= internal_parent.size() ||
            internal_id_u >= internal_parent_slot.size()) {
            return;
        }

        std::uint32_t const full_id = primitive_count + static_cast<std::uint32_t>(internal_id_u);
        if (full_id >= full_nodes.size())
            return;

        gwn_hploc_node<Real> const full_node = full_nodes[full_id];
        gwn_binary_node<Index> node{};

        std::uint8_t left_kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
        Index left_index = Index(0);
        std::uint8_t right_kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
        Index right_index = Index(0);

        if (!gwn_hploc_decode_child<Real, Index>(
                full_node.left_child, primitive_count, left_kind, left_index
            ) ||
            !gwn_hploc_decode_child<Real, Index>(
                full_node.right_child, primitive_count, right_kind, right_index
            )) {
            return;
        }

        node.left.kind = left_kind;
        node.left.index = left_index;
        node.right.kind = right_kind;
        node.right.index = right_index;
        binary_nodes[internal_id_u] = node;

        Index const parent_index = static_cast<Index>(internal_id_u);
        if (left_kind == static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
            std::size_t const child_u = static_cast<std::size_t>(left_index);
            if (child_u < internal_parent.size() && child_u < internal_parent_slot.size()) {
                internal_parent[child_u] = parent_index;
                internal_parent_slot[child_u] = 0;
            }
        } else {
            std::size_t const child_u = static_cast<std::size_t>(left_index);
            if (child_u < leaf_parent.size() && child_u < leaf_parent_slot.size()) {
                leaf_parent[child_u] = parent_index;
                leaf_parent_slot[child_u] = 0;
            }
        }

        if (right_kind == static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
            std::size_t const child_u = static_cast<std::size_t>(right_index);
            if (child_u < internal_parent.size() && child_u < internal_parent_slot.size()) {
                internal_parent[child_u] = parent_index;
                internal_parent_slot[child_u] = 1;
            }
        } else {
            std::size_t const child_u = static_cast<std::size_t>(right_index);
            if (child_u < leaf_parent.size() && child_u < leaf_parent_slot.size()) {
                leaf_parent[child_u] = parent_index;
                leaf_parent_slot[child_u] = 1;
            }
        }
    }
};

template <
    int BlockSize, std::uint32_t SearchRadius, std::uint32_t MergingThreshold, class Real,
    class MortonCode>
__global__ __launch_bounds__(BlockSize) void gwn_build_binary_hploc_kernel(
    std::uint32_t const primitive_count, gwn_hploc_node<Real> *full_nodes,
    std::uint32_t *cluster_indices, std::uint32_t *boundary_parent, unsigned int *cluster_count,
    MortonCode const *sorted_morton_codes, unsigned int *failure_flag
) {
    constexpr int k_warp_size = 32;
    static_assert((BlockSize % k_warp_size) == 0, "BlockSize must be a multiple of 32.");
    std::uint32_t const idx = static_cast<std::uint32_t>(gwn_global_thread_index_1d());
    std::uint32_t const lane = static_cast<std::uint32_t>(threadIdx.x) & (k_warp_size - 1u);

    auto const delta = [&](std::uint32_t const a, std::uint32_t const b) {
        MortonCode const key_a = sorted_morton_codes[a];
        MortonCode const key_b = sorted_morton_codes[b];
        if constexpr (std::is_same_v<MortonCode, std::uint32_t>) {
            // Tie-break equal Morton keys with primitive indices to keep
            // parent selection stable for 32-bit Morton paths.
            std::uint64_t const combined_a =
                (static_cast<std::uint64_t>(key_a) << 32u) | static_cast<std::uint64_t>(a);
            std::uint64_t const combined_b =
                (static_cast<std::uint64_t>(key_b) << 32u) | static_cast<std::uint64_t>(b);
            return combined_a ^ combined_b;
        }
        std::uint64_t const key_delta = static_cast<std::uint64_t>(key_a ^ key_b);
        if (key_delta == 0)
            return std::uint64_t(a ^ b);
        return key_delta;
    };

    auto const find_parent_id = [&](std::uint32_t const left, std::uint32_t const right) {
        if (left == 0 ||
            (right != (primitive_count - 1) && delta(right, right + 1) < delta(left - 1, left))) {
            return right;
        }
        return left - 1;
    };

    auto const load_indices = [&](std::uint32_t const start, std::uint32_t const end,
                                  std::uint32_t &cluster_idx_out,
                                  std::uint32_t const offset) -> std::uint32_t {
        std::uint32_t const index = lane - offset;
        std::uint32_t const load_count = min(end - start, MergingThreshold);
        bool const valid = index < load_count;
        if (valid)
            cluster_idx_out = cluster_indices[start + index];

        unsigned int const valid_mask =
            __ballot_sync(0xffffffffu, valid && cluster_idx_out != k_gwn_hploc_invalid_u32);
        return static_cast<std::uint32_t>(__popc(valid_mask));
    };

    auto const store_indices = [&](std::uint32_t const previous_num_prim,
                                   std::uint32_t const cluster_idx,
                                   std::uint32_t const left_start) {
        if (lane < previous_num_prim)
            cluster_indices[left_start + lane] = cluster_idx;
        __threadfence();
    };

    auto const merge_clusters_create_binary_node =
        [&](std::uint32_t const num_prim, std::uint32_t nearest_neighbor,
            std::uint32_t &cluster_idx, gwn_aabb<Real> &cluster_bounds) -> std::uint32_t {
        bool const lane_active = lane < num_prim;
        std::uint32_t safe_neighbor = nearest_neighbor;
        if (safe_neighbor >= num_prim)
            safe_neighbor = lane;
        std::uint32_t const nearest_neighbor_nn =
            __shfl_sync(0xffffffffu, safe_neighbor, static_cast<int>(safe_neighbor));
        bool const mutual_neighbor = lane_active && lane == nearest_neighbor_nn;
        bool const merge = mutual_neighbor && lane < safe_neighbor;

        unsigned int const merge_mask = __ballot_sync(0xffffffffu, merge);
        std::uint32_t const merge_count = static_cast<std::uint32_t>(__popc(merge_mask));

        std::uint32_t base_idx = 0;
        if (lane == 0)
            base_idx = atomicAdd(cluster_count, merge_count);
        base_idx = __shfl_sync(0xffffffffu, base_idx, 0);

        std::uint32_t const relative_idx =
            static_cast<std::uint32_t>(__popc(merge_mask << (k_warp_size - lane)));

        std::uint32_t const neighbor_cluster_idx =
            __shfl_sync(0xffffffffu, cluster_idx, static_cast<int>(safe_neighbor));
        gwn_aabb<Real> const neighbor_bounds =
            gwn_hploc_shfl_aabb(0xffffffffu, cluster_bounds, static_cast<int>(safe_neighbor));

        if (merge) {
            gwn_aabb<Real> merged_bounds = gwn_aabb_union(cluster_bounds, neighbor_bounds);
            std::uint32_t const merged_idx = base_idx + relative_idx;

            gwn_hploc_node<Real> merged_node{};
            merged_node.bounds = merged_bounds;
            merged_node.left_child = cluster_idx;
            merged_node.right_child = neighbor_cluster_idx;
            full_nodes[merged_idx] = merged_node;

            cluster_idx = merged_idx;
            cluster_bounds = merged_bounds;
        }

        unsigned int const valid_mask = __ballot_sync(0xffffffffu, merge || !mutual_neighbor);
        int const shift = __fns(valid_mask, 0u, lane + 1u);
        cluster_idx = __shfl_sync(0xffffffffu, cluster_idx, shift);
        if (shift == -1)
            cluster_idx = k_gwn_hploc_invalid_u32;
        cluster_bounds = gwn_hploc_shfl_aabb(0xffffffffu, cluster_bounds, shift);

        return num_prim - merge_count;
    };

    auto const find_nearest_neighbor = [&](std::uint32_t const num_prim,
                                           std::uint32_t const cluster_idx,
                                           gwn_aabb<Real> cluster_bounds) {
        (void)cluster_idx;
        uint2 min_area_idx = make_uint2(k_gwn_hploc_invalid_u32, k_gwn_hploc_invalid_u32);

        for (std::uint32_t radius = 1; radius <= SearchRadius; ++radius) {
            std::uint32_t const neighbor_lane = lane + radius;
            std::uint32_t area = k_gwn_hploc_invalid_u32;
            gwn_aabb<Real> const neighbor_bounds =
                gwn_hploc_shfl_aabb(0xffffffffu, cluster_bounds, static_cast<int>(neighbor_lane));
            if (neighbor_lane < num_prim) {
                gwn_aabb<Real> const merged_bounds =
                    gwn_aabb_union(cluster_bounds, neighbor_bounds);
                area = __float_as_uint(static_cast<float>(gwn_aabb_half_area(merged_bounds)));
                if (area < min_area_idx.x)
                    min_area_idx = make_uint2(area, neighbor_lane);
            }

            uint2 neighbor_nn =
                gwn_hploc_shfl_uint2(0xffffffffu, min_area_idx, static_cast<int>(neighbor_lane));
            if (area < neighbor_nn.x)
                neighbor_nn = make_uint2(area, lane);
            min_area_idx =
                gwn_hploc_shfl_uint2(0xffffffffu, neighbor_nn, static_cast<int>(lane - radius));
        }
        return min_area_idx.y;
    };

    auto const ploc_merge = [&](int const selected_lane, std::uint32_t const left_boundary,
                                std::uint32_t const right_boundary,
                                std::uint32_t const split_boundary, bool const is_final) {
        std::uint32_t const local_left = __shfl_sync(0xffffffffu, left_boundary, selected_lane);
        std::uint32_t const local_right_end =
            __shfl_sync(0xffffffffu, right_boundary, selected_lane) + 1u;
        std::uint32_t const local_split = __shfl_sync(0xffffffffu, split_boundary, selected_lane);
        std::uint32_t const local_right_start = local_split;

        std::uint32_t cluster_idx = k_gwn_hploc_invalid_u32;
        std::uint32_t const num_left = load_indices(local_left, local_split, cluster_idx, 0u);
        std::uint32_t const num_right =
            load_indices(local_right_start, local_right_end, cluster_idx, num_left);
        std::uint32_t num_prim = num_left + num_right;

        gwn_aabb<Real> cluster_bounds{};
        if (lane < num_prim)
            cluster_bounds = full_nodes[cluster_idx].bounds;

        std::uint32_t const threshold =
            __shfl_sync(0xffffffffu, is_final ? 1u : MergingThreshold, selected_lane);
        std::uint32_t inner_iter = 0;
        while (num_prim > threshold) {
            if (++inner_iter > 1024u) {
                if (failure_flag != nullptr)
                    atomicExch(failure_flag, 1u);
                break;
            }
            std::uint32_t const previous_num_prim = num_prim;
            std::uint32_t const nearest =
                find_nearest_neighbor(num_prim, cluster_idx, cluster_bounds);
            num_prim =
                merge_clusters_create_binary_node(num_prim, nearest, cluster_idx, cluster_bounds);
            if (num_prim == previous_num_prim) {
                std::uint32_t fallback_neighbor = lane;
                if ((lane & 1u) == 0u) {
                    if ((lane + 1u) < num_prim)
                        fallback_neighbor = lane + 1u;
                    else if (lane > 0u)
                        fallback_neighbor = lane - 1u;
                } else {
                    fallback_neighbor = lane - 1u;
                }

                num_prim = merge_clusters_create_binary_node(
                    num_prim, fallback_neighbor, cluster_idx, cluster_bounds
                );
                if (num_prim == previous_num_prim) {
                    if (failure_flag != nullptr)
                        atomicExch(failure_flag, 1u);
                    break;
                }
            }
        }

        store_indices(num_left + num_right, cluster_idx, local_left);
    };

    std::uint32_t left = idx;
    std::uint32_t right = idx;
    std::uint32_t split = 0;
    bool lane_active = idx < primitive_count;
    std::uint32_t outer_iter = 0;

    while (__ballot_sync(0xffffffffu, lane_active) != 0u) {
        if (++outer_iter > primitive_count * 8u + 1024u) {
            if (failure_flag != nullptr)
                atomicExch(failure_flag, 1u);
            break;
        }
        if (lane_active) {
            std::uint32_t previous_id = k_gwn_hploc_invalid_u32;
            if (find_parent_id(left, right) == right) {
                previous_id = atomicExch(boundary_parent + right, left);
                if (previous_id != k_gwn_hploc_invalid_u32) {
                    split = right + 1u;
                    right = previous_id;
                }
            } else {
                previous_id = atomicExch(boundary_parent + (left - 1u), right);
                if (previous_id != k_gwn_hploc_invalid_u32) {
                    split = left;
                    left = previous_id;
                }
            }

            if (previous_id == k_gwn_hploc_invalid_u32)
                lane_active = false;
        }

        std::uint32_t const size = right - left + 1u;
        bool const final = lane_active && (size == primitive_count);
        unsigned int active_mask =
            __ballot_sync(0xffffffffu, (lane_active && (size > MergingThreshold)) || final);
        while (active_mask != 0u) {
            int const selected_lane = __ffs(static_cast<int>(active_mask)) - 1;
            ploc_merge(selected_lane, left, right, split, final);
            active_mask &= (active_mask - 1u);
        }
    }
}

template <class Real, class Index, class MortonCode = std::uint64_t>
gwn_status gwn_bvh_topology_build_binary_hploc(
    cuda::std::span<Index const> const sorted_primitive_indices,
    cuda::std::span<MortonCode const> const sorted_morton_codes,
    cuda::std::span<gwn_aabb<Real> const> const primitive_aabbs,
    gwn_device_array<gwn_binary_node<Index>> &binary_nodes,
    gwn_device_array<Index> &binary_internal_parent, Index &root_internal_index,
    cudaStream_t const stream = gwn_default_stream()
) {
    static_assert(
        std::is_same_v<MortonCode, std::uint32_t> || std::is_same_v<MortonCode, std::uint64_t>,
        "MortonCode must be std::uint32_t or std::uint64_t."
    );
    std::size_t const primitive_count = sorted_primitive_indices.size();
    root_internal_index = gwn_invalid_index<Index>();
    if (primitive_count == 0) {
        GWN_RETURN_ON_ERROR(binary_nodes.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_parent.clear(stream));
        return gwn_status::ok();
    }
    if (sorted_morton_codes.size() != primitive_count || primitive_aabbs.size() != primitive_count)
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_binary_hploc, "H-PLOC preprocess buffer size mismatch."
        );

    if (primitive_count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() / 2u))
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_topology_binary_hploc,
            "H-PLOC builder primitive count exceeds uint32 range."
        );

    constexpr int k_linear_block_size = k_gwn_default_block_size;
    gwn_device_array<gwn_aabb<Real>> sorted_leaf_aabbs{};
    GWN_RETURN_ON_ERROR(sorted_leaf_aabbs.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_linear_block_size>(
            primitive_count,
            gwn_gather_sorted_aabbs_functor<Real, Index>{
                primitive_aabbs, sorted_primitive_indices, sorted_leaf_aabbs.span()
            },
            stream
        )
    );

    if (primitive_count == 1) {
        GWN_RETURN_ON_ERROR(binary_nodes.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_parent.clear(stream));
        return gwn_status::ok();
    }

    std::size_t const binary_internal_count = primitive_count - 1;
    gwn_binary_parent_temporaries<Index> temps{};
    GWN_RETURN_ON_ERROR(gwn_prepare_binary_topology_buffers(
        primitive_count, binary_nodes, binary_internal_parent, temps, stream
    ));

    constexpr int k_hploc_block_size = 64;

    gwn_device_array<gwn_hploc_node<Real>> full_nodes{};
    gwn_device_array<std::uint32_t> cluster_indices{};
    gwn_device_array<std::uint32_t> boundary_parent{};
    gwn_device_array<unsigned int> cluster_count{};
    gwn_device_array<unsigned int> failure_flag{};
    GWN_RETURN_ON_ERROR(full_nodes.resize(primitive_count * 2 - 1, stream));
    GWN_RETURN_ON_ERROR(cluster_indices.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(boundary_parent.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(cluster_count.resize(1, stream));
    GWN_RETURN_ON_ERROR(failure_flag.resize(1, stream));

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_linear_block_size>(
            primitive_count,
            gwn_hploc_init_full_nodes_functor<Real>{
                sorted_leaf_aabbs.span(), full_nodes.span(), cluster_indices.span()
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        boundary_parent.data(), 0xff, primitive_count * sizeof(std::uint32_t), stream
    )));
    unsigned int const initial_cluster_count = static_cast<unsigned int>(primitive_count);
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        cluster_count.data(), &initial_cluster_count, sizeof(unsigned int), cudaMemcpyHostToDevice,
        stream
    )));
    GWN_RETURN_ON_ERROR(
        gwn_cuda_to_status(cudaMemsetAsync(failure_flag.data(), 0, sizeof(unsigned int), stream))
    );

    int const block_count = gwn_block_count_1d<k_hploc_block_size>(primitive_count);
    gwn_build_binary_hploc_kernel<
        k_hploc_block_size, k_gwn_hploc_search_radius, k_gwn_hploc_merging_threshold, Real,
        MortonCode><<<block_count, k_hploc_block_size, 0, stream>>>(
        static_cast<std::uint32_t>(primitive_count), full_nodes.data(), cluster_indices.data(),
        boundary_parent.data(), cluster_count.data(), sorted_morton_codes.data(),
        failure_flag.data()
    );
    GWN_RETURN_ON_ERROR(gwn_check_last_kernel());

    unsigned int host_cluster_count = 0u;
    unsigned int host_failure_flag = 0u;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_cluster_count, cluster_count.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_failure_flag, failure_flag.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_failure_flag != 0u) {
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_binary_hploc,
            "H-PLOC build kernel failed to converge; possible invalid merge state."
        );
    }
    unsigned int const expected_cluster_count = static_cast<unsigned int>(primitive_count * 2 - 1);
    if (host_cluster_count != expected_cluster_count)
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_binary_hploc,
            "H-PLOC builder did not converge to a single root."
        );

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_linear_block_size>(
            binary_internal_count,
            gwn_hploc_emit_binary_nodes_functor<Real, Index>{
                cuda::std::span<gwn_hploc_node<Real> const>(full_nodes.data(), full_nodes.size()),
                binary_nodes.span(), binary_internal_parent.span(),
                temps.internal_parent_slot.span(), temps.leaf_parent.span(),
                temps.leaf_parent_slot.span(), static_cast<std::uint32_t>(primitive_count)
            },
            stream
        )
    );

    root_internal_index = static_cast<Index>(binary_internal_count - 1);
    return gwn_status::ok();
}

} // namespace detail
} // namespace gwn
