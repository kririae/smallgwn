#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
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

inline constexpr std::uint32_t k_gwn_hploc_search_radius = 4u;
inline constexpr std::uint32_t k_gwn_hploc_merging_threshold = 16u;
inline constexpr std::uint32_t k_gwn_hploc_convergence_inner_iter_limit = 1024u;
inline constexpr std::uint32_t k_gwn_hploc_convergence_outer_iter_per_primitive = 8u;
inline constexpr std::uint32_t k_gwn_hploc_convergence_outer_iter_slack = 1024u;
inline constexpr std::uint32_t k_gwn_hploc_invalid_lane = std::numeric_limits<std::uint32_t>::max();

using gwn_hploc_u64 = unsigned long long;

template <gwn_index_type Index>
using gwn_hploc_node_index_t =
    std::conditional_t<(sizeof(Index) <= sizeof(std::uint32_t)), std::uint32_t, gwn_hploc_u64>;

template <class NodeIndex>
[[nodiscard]] __host__ __device__ inline constexpr NodeIndex gwn_hploc_invalid_index() noexcept {
    static_assert(std::is_unsigned_v<NodeIndex>, "NodeIndex must be unsigned.");
    return std::numeric_limits<NodeIndex>::max();
}

template <gwn_real_type Real, class NodeIndex> struct gwn_hploc_node {
    gwn_aabb<Real> bounds{};
    NodeIndex left_child = gwn_hploc_invalid_index<NodeIndex>();
    NodeIndex right_child = gwn_hploc_invalid_index<NodeIndex>();
};

template <gwn_real_type Real, class NodeIndex> struct gwn_hploc_init_full_nodes_functor {
    cuda::std::span<gwn_aabb<Real> const> sorted_leaf_aabbs{};
    cuda::std::span<gwn_hploc_node<Real, NodeIndex>> full_nodes{};
    cuda::std::span<NodeIndex> cluster_indices{};

    __device__ void operator()(std::size_t const leaf_id) const {
        if (leaf_id >= sorted_leaf_aabbs.size() || leaf_id >= cluster_indices.size() ||
            leaf_id >= full_nodes.size()) {
            return;
        }

        gwn_hploc_node<Real, NodeIndex> node{};
        node.bounds = sorted_leaf_aabbs[leaf_id];
        node.left_child = gwn_hploc_invalid_index<NodeIndex>();
        node.right_child = static_cast<NodeIndex>(leaf_id);
        full_nodes[leaf_id] = node;
        cluster_indices[leaf_id] = static_cast<NodeIndex>(leaf_id);
    }
};

template <class T>
[[nodiscard]] __device__ inline T
gwn_hploc_shfl_value(unsigned int const mask, T const value, int const src_lane) {
    return __shfl_sync(mask, value, src_lane);
}

template <>
[[nodiscard]] __device__ inline gwn_hploc_u64 gwn_hploc_shfl_value<gwn_hploc_u64>(
    unsigned int const mask, gwn_hploc_u64 const value, int const src_lane
) {
    auto const lo = static_cast<unsigned int>(value & 0xffffffffull);
    auto const hi = static_cast<unsigned int>(value >> 32u);
    unsigned int const shuffled_lo = __shfl_sync(mask, lo, src_lane);
    unsigned int const shuffled_hi = __shfl_sync(mask, hi, src_lane);
    return (static_cast<gwn_hploc_u64>(shuffled_hi) << 32u) |
           static_cast<gwn_hploc_u64>(shuffled_lo);
}

template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_aabb<Real>
gwn_hploc_shfl_aabb(unsigned int const mask, gwn_aabb<Real> const &value, int const src_lane) {
    gwn_aabb<Real> result{};
    result.min_x = gwn_hploc_shfl_value(mask, value.min_x, src_lane);
    result.min_y = gwn_hploc_shfl_value(mask, value.min_y, src_lane);
    result.min_z = gwn_hploc_shfl_value(mask, value.min_z, src_lane);
    result.max_x = gwn_hploc_shfl_value(mask, value.max_x, src_lane);
    result.max_y = gwn_hploc_shfl_value(mask, value.max_y, src_lane);
    result.max_z = gwn_hploc_shfl_value(mask, value.max_z, src_lane);
    return result;
}

template <class NodeIndex>
[[nodiscard]] __device__ inline NodeIndex
gwn_hploc_atomic_add(NodeIndex *address, NodeIndex const value) noexcept {
    static_assert(
        std::is_same_v<NodeIndex, std::uint32_t> || std::is_same_v<NodeIndex, gwn_hploc_u64>,
        "NodeIndex must be uint32_t or unsigned long long."
    );
    return atomicAdd(address, value);
}

template <class NodeIndex>
[[nodiscard]] __device__ inline NodeIndex
gwn_hploc_atomic_exch(NodeIndex *address, NodeIndex const value) noexcept {
    static_assert(
        std::is_same_v<NodeIndex, std::uint32_t> || std::is_same_v<NodeIndex, gwn_hploc_u64>,
        "NodeIndex must be uint32_t or unsigned long long."
    );
    return atomicExch(address, value);
}

template <gwn_real_type Real>
[[nodiscard]] __device__ inline bool gwn_hploc_is_better_area_candidate(
    Real const lhs_area, std::uint32_t const lhs_lane, Real const rhs_area,
    std::uint32_t const rhs_lane
) {
    if (lhs_lane == k_gwn_hploc_invalid_lane)
        return false;
    if (rhs_lane == k_gwn_hploc_invalid_lane)
        return true;
    return lhs_area < rhs_area || (lhs_area == rhs_area && lhs_lane < rhs_lane);
}

template <class NodeIndex, gwn_index_type Index>
[[nodiscard]] __device__ inline bool gwn_hploc_decode_child(
    NodeIndex const full_child, NodeIndex const primitive_count, std::uint8_t &kind_out,
    Index &index_out
) {
    if (full_child == gwn_hploc_invalid_index<NodeIndex>())
        return false;
    if (full_child < primitive_count) {
        kind_out = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
        index_out = static_cast<Index>(full_child);
        return true;
    }

    NodeIndex const internal_offset = full_child - primitive_count;
    kind_out = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
    index_out = static_cast<Index>(internal_offset);
    return true;
}

template <gwn_real_type Real, gwn_index_type Index, class NodeIndex>
struct gwn_hploc_emit_binary_nodes_functor {
    cuda::std::span<gwn_hploc_node<Real, NodeIndex> const> full_nodes{};
    cuda::std::span<gwn_binary_node<Index>> binary_nodes{};
    cuda::std::span<gwn_aabb<Real>> binary_internal_bounds{};
    cuda::std::span<Index> internal_parent{};
    cuda::std::span<std::uint8_t> internal_parent_slot{};
    cuda::std::span<Index> leaf_parent{};
    cuda::std::span<std::uint8_t> leaf_parent_slot{};
    NodeIndex primitive_count = 0;

    __device__ void operator()(std::size_t const internal_id_u) const {
        if (internal_id_u >= binary_nodes.size() || internal_id_u >= internal_parent.size() ||
            internal_id_u >= internal_parent_slot.size()) {
            return;
        }

        NodeIndex const full_id = primitive_count + static_cast<NodeIndex>(internal_id_u);
        if (full_id >= full_nodes.size())
            return;

        gwn_hploc_node<Real, NodeIndex> const full_node = full_nodes[full_id];
        gwn_binary_node<Index> node{};

        auto left_kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
        Index left_index = Index(0);
        auto right_kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
        Index right_index = Index(0);

        if (!gwn_hploc_decode_child<NodeIndex, Index>(
                full_node.left_child, primitive_count, left_kind, left_index
            ) ||
            !gwn_hploc_decode_child<NodeIndex, Index>(
                full_node.right_child, primitive_count, right_kind, right_index
            )) {
            return;
        }

        node.left.kind = left_kind;
        node.left.index = left_index;
        node.right.kind = right_kind;
        node.right.index = right_index;
        binary_nodes[internal_id_u] = node;
        if (internal_id_u < binary_internal_bounds.size())
            binary_internal_bounds[internal_id_u] = full_node.bounds;

        auto const parent_index = static_cast<Index>(internal_id_u);
        if (left_kind == static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
            auto const child_u = static_cast<std::size_t>(left_index);
            if (child_u < internal_parent.size() && child_u < internal_parent_slot.size()) {
                internal_parent[child_u] = parent_index;
                internal_parent_slot[child_u] = 0;
            }
        } else {
            auto const child_u = static_cast<std::size_t>(left_index);
            if (child_u < leaf_parent.size() && child_u < leaf_parent_slot.size()) {
                leaf_parent[child_u] = parent_index;
                leaf_parent_slot[child_u] = 0;
            }
        }

        if (right_kind == static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
            auto const child_u = static_cast<std::size_t>(right_index);
            if (child_u < internal_parent.size() && child_u < internal_parent_slot.size()) {
                internal_parent[child_u] = parent_index;
                internal_parent_slot[child_u] = 1;
            }
        } else {
            auto const child_u = static_cast<std::size_t>(right_index);
            if (child_u < leaf_parent.size() && child_u < leaf_parent_slot.size()) {
                leaf_parent[child_u] = parent_index;
                leaf_parent_slot[child_u] = 1;
            }
        }
    }
};

template <
    int BlockSize, std::uint32_t SearchRadius, std::uint32_t MergingThreshold, gwn_real_type Real,
    class MortonCode, class NodeIndex>
__global__ __launch_bounds__(BlockSize) void gwn_build_binary_hploc_kernel(
    NodeIndex const primitive_count, gwn_hploc_node<Real, NodeIndex> *full_nodes,
    NodeIndex *cluster_indices, NodeIndex *boundary_parent, NodeIndex *cluster_count,
    MortonCode const *sorted_morton_codes, unsigned int *failure_flag
) {
    constexpr int k_warp_size = 32;
    constexpr unsigned int k_warp_mask = 0xffffffffu;
    static_assert((BlockSize % k_warp_size) == 0, "BlockSize must be a multiple of 32.");
    static_assert(
        std::is_same_v<NodeIndex, std::uint32_t> || std::is_same_v<NodeIndex, gwn_hploc_u64>,
        "NodeIndex must be uint32_t or unsigned long long."
    );
    auto const lane = static_cast<std::uint32_t>(threadIdx.x) & (k_warp_size - 1u);
    auto const idx = static_cast<NodeIndex>(gwn_global_thread_index_1d());

    auto const delta = [&](NodeIndex const a, NodeIndex const b) {
        MortonCode const key_a = sorted_morton_codes[static_cast<std::size_t>(a)];
        MortonCode const key_b = sorted_morton_codes[static_cast<std::size_t>(b)];
        if constexpr (std::is_same_v<MortonCode, std::uint32_t>) {
            std::uint64_t const combined_a =
                (static_cast<std::uint64_t>(key_a) << 32u) | static_cast<std::uint64_t>(a);
            std::uint64_t const combined_b =
                (static_cast<std::uint64_t>(key_b) << 32u) | static_cast<std::uint64_t>(b);
            return combined_a ^ combined_b;
        }
        auto const key_delta = static_cast<std::uint64_t>(key_a ^ key_b);
        if (key_delta == 0)
            return static_cast<std::uint64_t>(a ^ b);
        return key_delta;
    };

    auto const find_parent_id = [&](NodeIndex const left, NodeIndex const right) {
        if (left == 0 ||
            (right != (primitive_count - 1) && delta(right, right + 1) < delta(left - 1, left))) {
            return right;
        }
        return left - 1;
    };

    auto const load_indices = [&](NodeIndex const start, NodeIndex const end,
                                  NodeIndex &cluster_idx_out,
                                  std::uint32_t const offset) -> std::uint32_t {
        std::uint32_t const index = lane - offset;
        std::uint32_t const load_count =
            std::min(static_cast<std::uint32_t>(end - start), MergingThreshold);
        bool const valid = index < load_count;
        if (valid)
            cluster_idx_out = cluster_indices[start + static_cast<NodeIndex>(index)];

        unsigned int const valid_mask = __ballot_sync(
            k_warp_mask, valid && cluster_idx_out != gwn_hploc_invalid_index<NodeIndex>()
        );
        return static_cast<std::uint32_t>(__popc(valid_mask));
    };

    auto const store_indices = [&](std::uint32_t const previous_num_prim,
                                   NodeIndex const cluster_idx, NodeIndex const left_start) {
        if (lane < previous_num_prim)
            cluster_indices[left_start + static_cast<NodeIndex>(lane)] = cluster_idx;
        __threadfence();
    };

    auto const merge_clusters_create_binary_node =
        [&](std::uint32_t const num_prim, std::uint32_t nearest_neighbor, NodeIndex &cluster_idx,
            gwn_aabb<Real> &cluster_bounds) -> std::uint32_t {
        bool const lane_active = lane < num_prim;
        std::uint32_t safe_neighbor = nearest_neighbor;
        if (safe_neighbor >= num_prim)
            safe_neighbor = lane;
        std::uint32_t const nearest_neighbor_nn =
            gwn_hploc_shfl_value(k_warp_mask, safe_neighbor, static_cast<int>(safe_neighbor));
        bool const mutual_neighbor = lane_active && lane == nearest_neighbor_nn;
        bool const merge = mutual_neighbor && lane < safe_neighbor;

        unsigned int const merge_mask = __ballot_sync(k_warp_mask, merge);
        auto const merge_count = static_cast<std::uint32_t>(__popc(merge_mask));

        NodeIndex base_idx = 0;
        if (lane == 0)
            base_idx = gwn_hploc_atomic_add(cluster_count, static_cast<NodeIndex>(merge_count));
        base_idx = gwn_hploc_shfl_value(k_warp_mask, base_idx, 0);

        std::uint32_t const lower_lane_mask = (lane == 0u) ? 0u : ((1u << lane) - 1u);
        std::uint32_t const relative_idx =
            static_cast<std::uint32_t>(__popc(merge_mask & lower_lane_mask));

        NodeIndex const neighbor_cluster_idx =
            gwn_hploc_shfl_value(k_warp_mask, cluster_idx, static_cast<int>(safe_neighbor));
        gwn_aabb<Real> const neighbor_bounds =
            gwn_hploc_shfl_aabb(k_warp_mask, cluster_bounds, static_cast<int>(safe_neighbor));

        if (merge) {
            gwn_aabb<Real> const merged_bounds = gwn_aabb_union(cluster_bounds, neighbor_bounds);
            NodeIndex const merged_idx = base_idx + static_cast<NodeIndex>(relative_idx);

            gwn_hploc_node<Real, NodeIndex> merged_node{};
            merged_node.bounds = merged_bounds;
            merged_node.left_child = cluster_idx;
            merged_node.right_child = neighbor_cluster_idx;
            full_nodes[merged_idx] = merged_node;

            cluster_idx = merged_idx;
            cluster_bounds = merged_bounds;
        }

        unsigned int const valid_mask = __ballot_sync(k_warp_mask, merge || !mutual_neighbor);
        int const shift = __fns(valid_mask, 0u, lane + 1u);
        int const safe_shift = (shift == -1) ? static_cast<int>(lane) : shift;
        NodeIndex const shifted_cluster_idx =
            gwn_hploc_shfl_value(k_warp_mask, cluster_idx, safe_shift);
        gwn_aabb<Real> const shifted_cluster_bounds =
            gwn_hploc_shfl_aabb(k_warp_mask, cluster_bounds, safe_shift);
        if (shift == -1) {
            cluster_idx = gwn_hploc_invalid_index<NodeIndex>();
        } else {
            cluster_idx = shifted_cluster_idx;
            cluster_bounds = shifted_cluster_bounds;
        }

        return num_prim - merge_count;
    };

    auto const find_nearest_neighbor = [&](std::uint32_t const num_prim,
                                           NodeIndex const cluster_idx,
                                           gwn_aabb<Real> cluster_bounds) {
        (void)cluster_idx;
        Real best_area = std::numeric_limits<Real>::infinity();
        std::uint32_t best_lane = k_gwn_hploc_invalid_lane;

        for (std::uint32_t radius = 1; radius <= SearchRadius; ++radius) {
            std::uint32_t const neighbor_lane = lane + radius;
            bool const neighbor_lane_valid =
                neighbor_lane < static_cast<std::uint32_t>(k_warp_size);
            int const safe_neighbor_lane =
                neighbor_lane_valid ? static_cast<int>(neighbor_lane) : static_cast<int>(lane);

            Real local_area = std::numeric_limits<Real>::infinity();
            bool local_area_valid = false;
            gwn_aabb<Real> const neighbor_bounds =
                gwn_hploc_shfl_aabb(k_warp_mask, cluster_bounds, safe_neighbor_lane);
            if (neighbor_lane_valid && neighbor_lane < num_prim) {
                local_area = gwn_aabb_half_area(gwn_aabb_union(cluster_bounds, neighbor_bounds));
                local_area_valid = true;
                if (gwn_hploc_is_better_area_candidate(
                        local_area, neighbor_lane, best_area, best_lane
                    )) {
                    best_area = local_area;
                    best_lane = neighbor_lane;
                }
            }

            Real neighbor_best_area =
                gwn_hploc_shfl_value(k_warp_mask, best_area, safe_neighbor_lane);
            std::uint32_t neighbor_best_lane =
                gwn_hploc_shfl_value(k_warp_mask, best_lane, safe_neighbor_lane);
            if (!neighbor_lane_valid)
                neighbor_best_lane = k_gwn_hploc_invalid_lane;

            if (local_area_valid && gwn_hploc_is_better_area_candidate(
                                        local_area, lane, neighbor_best_area, neighbor_best_lane
                                    )) {
                neighbor_best_area = local_area;
                neighbor_best_lane = lane;
            }

            auto const left_lane = static_cast<int>(lane) - static_cast<int>(radius);
            int const safe_left_lane = (left_lane >= 0) ? left_lane : static_cast<int>(lane);
            Real const shifted_best_area =
                gwn_hploc_shfl_value(k_warp_mask, neighbor_best_area, safe_left_lane);
            std::uint32_t const shifted_best_lane =
                gwn_hploc_shfl_value(k_warp_mask, neighbor_best_lane, safe_left_lane);
            if (left_lane >= 0) {
                best_area = shifted_best_area;
                best_lane = shifted_best_lane;
            }
        }
        return best_lane;
    };

    auto const ploc_merge = [&](int const selected_lane, NodeIndex const left_boundary,
                                NodeIndex const right_boundary, NodeIndex const split_boundary,
                                bool const is_final_range) {
        NodeIndex const local_left =
            gwn_hploc_shfl_value(k_warp_mask, left_boundary, selected_lane);
        NodeIndex const local_right_end =
            gwn_hploc_shfl_value(k_warp_mask, right_boundary, selected_lane) + 1;
        NodeIndex const local_split =
            gwn_hploc_shfl_value(k_warp_mask, split_boundary, selected_lane);
        NodeIndex const local_right_start = local_split;

        NodeIndex cluster_idx = gwn_hploc_invalid_index<NodeIndex>();
        std::uint32_t const num_left = load_indices(local_left, local_split, cluster_idx, 0u);
        std::uint32_t const num_right =
            load_indices(local_right_start, local_right_end, cluster_idx, num_left);
        std::uint32_t num_prim = num_left + num_right;

        gwn_aabb<Real> cluster_bounds{};
        if (lane < num_prim)
            cluster_bounds = full_nodes[cluster_idx].bounds;

        std::uint32_t const threshold = gwn_hploc_shfl_value(
            k_warp_mask, is_final_range ? 1u : MergingThreshold, selected_lane
        );
        std::uint32_t inner_iter = 0;
        while (num_prim > threshold) {
            if (++inner_iter > k_gwn_hploc_convergence_inner_iter_limit) {
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

    NodeIndex left = idx;
    NodeIndex right = idx;
    NodeIndex split = 0;
    bool lane_active = idx < primitive_count;
    NodeIndex outer_iter = 0;
    NodeIndex const outer_iter_limit =
        primitive_count * static_cast<NodeIndex>(k_gwn_hploc_convergence_outer_iter_per_primitive) +
        static_cast<NodeIndex>(k_gwn_hploc_convergence_outer_iter_slack);

    while (__ballot_sync(k_warp_mask, lane_active) != 0u) {
        if (++outer_iter > outer_iter_limit) {
            if (failure_flag != nullptr)
                atomicExch(failure_flag, 1u);
            break;
        }
        if (lane_active) {
            NodeIndex previous_id = gwn_hploc_invalid_index<NodeIndex>();
            if (find_parent_id(left, right) == right) {
                previous_id = gwn_hploc_atomic_exch(boundary_parent + right, left);
                if (previous_id != gwn_hploc_invalid_index<NodeIndex>()) {
                    split = right + 1;
                    right = previous_id;
                }
            } else {
                previous_id = gwn_hploc_atomic_exch(boundary_parent + (left - 1), right);
                if (previous_id != gwn_hploc_invalid_index<NodeIndex>()) {
                    split = left;
                    left = previous_id;
                }
            }

            if (previous_id == gwn_hploc_invalid_index<NodeIndex>())
                lane_active = false;
        }

        NodeIndex const size = right - left + 1;
        bool const is_final_range = lane_active && (size == primitive_count);
        unsigned int active_mask = __ballot_sync(
            k_warp_mask,
            (lane_active && (size > static_cast<NodeIndex>(MergingThreshold))) || is_final_range
        );
        while (active_mask != 0u) {
            int const selected_lane = __ffs(static_cast<int>(active_mask)) - 1;
            ploc_merge(selected_lane, left, right, split, is_final_range);
            active_mask &= (active_mask - 1u);
        }
    }
}

template <gwn_real_type Real, gwn_index_type Index, class MortonCode = std::uint64_t>
gwn_status gwn_bvh_topology_build_binary_hploc(
    cuda::std::span<Index const> const sorted_primitive_indices,
    cuda::std::span<MortonCode const> const sorted_morton_codes,
    cuda::std::span<gwn_aabb<Real> const> const sorted_primitive_aabbs,
    gwn_device_array<gwn_binary_node<Index>> &binary_nodes,
    gwn_device_array<Index> &binary_internal_parent,
    gwn_device_array<gwn_aabb<Real>> &binary_internal_bounds, Index &root_internal_index,
    cudaStream_t const stream = gwn_default_stream()
) {
    static_assert(
        std::is_same_v<MortonCode, std::uint32_t> || std::is_same_v<MortonCode, std::uint64_t>,
        "MortonCode must be std::uint32_t or std::uint64_t."
    );
    static_assert(std::is_unsigned_v<Index>, "Index must be unsigned.");
    static_assert(sizeof(Index) == 4 || sizeof(Index) == 8, "Index must be uint32_t or uint64_t.");

    using NodeIndex = gwn_hploc_node_index_t<Index>;

    std::size_t const primitive_count = sorted_primitive_indices.size();
    root_internal_index = gwn_invalid_index<Index>();
    if (primitive_count == 0) {
        GWN_RETURN_ON_ERROR(binary_nodes.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_parent.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_bounds.clear(stream));
        return gwn_status::ok();
    }
    if (sorted_morton_codes.size() != primitive_count ||
        sorted_primitive_aabbs.size() != primitive_count)
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_binary_hploc, "H-PLOC preprocess buffer size mismatch."
        );

    NodeIndex const max_by_binary_nodes = (gwn_hploc_invalid_index<NodeIndex>() / 2u) + 1u;
    NodeIndex const max_by_outer_iterations =
        (gwn_hploc_invalid_index<NodeIndex>() -
         static_cast<NodeIndex>(k_gwn_hploc_convergence_outer_iter_slack)) /
        static_cast<NodeIndex>(k_gwn_hploc_convergence_outer_iter_per_primitive);
    NodeIndex const max_supported_primitive_count =
        std::min(max_by_binary_nodes, max_by_outer_iterations);
    if (primitive_count > static_cast<std::size_t>(max_supported_primitive_count))
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_topology_binary_hploc,
            "H-PLOC builder primitive count exceeds supported index range."
        );
    auto const primitive_count_node_index = static_cast<NodeIndex>(primitive_count);

    constexpr int k_linear_block_size = k_gwn_default_block_size;

    if (primitive_count == 1) {
        GWN_RETURN_ON_ERROR(binary_nodes.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_parent.clear(stream));
        GWN_RETURN_ON_ERROR(binary_internal_bounds.clear(stream));
        return gwn_status::ok();
    }

    std::size_t const binary_internal_count = primitive_count - 1;
    gwn_binary_parent_temporaries<Index> temps{};
    GWN_RETURN_ON_ERROR(gwn_prepare_binary_topology_buffers(
        primitive_count, binary_nodes, binary_internal_parent, temps, stream
    ));
    GWN_RETURN_ON_ERROR(binary_internal_bounds.resize(binary_internal_count, stream));

    constexpr int k_hploc_block_size = 64;

    gwn_device_array<gwn_hploc_node<Real, NodeIndex>> full_nodes{};
    gwn_device_array<NodeIndex> cluster_indices{};
    gwn_device_array<NodeIndex> boundary_parent{};
    gwn_device_array<NodeIndex> cluster_count{};
    gwn_device_array<unsigned int> failure_flag{};
    GWN_RETURN_ON_ERROR(full_nodes.resize(primitive_count * 2 - 1, stream));
    GWN_RETURN_ON_ERROR(cluster_indices.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(boundary_parent.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(cluster_count.resize(1, stream));
    GWN_RETURN_ON_ERROR(failure_flag.resize(1, stream));

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_linear_block_size>(
            primitive_count,
            gwn_hploc_init_full_nodes_functor<Real, NodeIndex>{
                sorted_primitive_aabbs, full_nodes.span(), cluster_indices.span()
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(boundary_parent.data(), 0xff, primitive_count * sizeof(NodeIndex), stream)
    ));
    NodeIndex const initial_cluster_count = primitive_count_node_index;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        cluster_count.data(), &initial_cluster_count, sizeof(NodeIndex), cudaMemcpyHostToDevice,
        stream
    )));
    GWN_RETURN_ON_ERROR(
        gwn_cuda_to_status(cudaMemsetAsync(failure_flag.data(), 0, sizeof(unsigned int), stream))
    );

    int const block_count = gwn_block_count_1d<k_hploc_block_size>(primitive_count);
    gwn_build_binary_hploc_kernel<
        k_hploc_block_size, k_gwn_hploc_search_radius, k_gwn_hploc_merging_threshold, Real,
        MortonCode, NodeIndex><<<block_count, k_hploc_block_size, 0, stream>>>(
        primitive_count_node_index, full_nodes.data(), cluster_indices.data(),
        boundary_parent.data(), cluster_count.data(), sorted_morton_codes.data(),
        failure_flag.data()
    );
    GWN_RETURN_ON_ERROR(gwn_check_last_kernel());

    NodeIndex host_cluster_count = 0;
    unsigned int host_failure_flag = 0u;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_cluster_count, cluster_count.data(), sizeof(NodeIndex), cudaMemcpyDeviceToHost, stream
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
    NodeIndex const expected_cluster_count = primitive_count_node_index * 2 - 1;
    if (host_cluster_count != expected_cluster_count)
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_topology_binary_hploc,
            "H-PLOC builder did not converge to a single root."
        );

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_linear_block_size>(
            binary_internal_count,
            gwn_hploc_emit_binary_nodes_functor<Real, Index, NodeIndex>{
                cuda::std::span<gwn_hploc_node<Real, NodeIndex> const>(
                    full_nodes.data(), full_nodes.size()
                ),
                binary_nodes.span(), binary_internal_bounds.span(), binary_internal_parent.span(),
                temps.internal_parent_slot.span(), temps.leaf_parent.span(),
                temps.leaf_parent_slot.span(), primitive_count_node_index
            },
            stream
        )
    );

    root_internal_index = static_cast<Index>(binary_internal_count - 1);
    return gwn_status::ok();
}

} // namespace detail
} // namespace gwn
