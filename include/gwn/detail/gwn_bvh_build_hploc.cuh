#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_bvh_build_binary.cuh"
#include "gwn_bvh_build_common.cuh"
#include "gwn_device_array.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {
namespace detail {

inline constexpr std::uint32_t k_gwn_hploc_max_search_radius = 8u;
inline constexpr std::uint32_t k_gwn_hploc_search_radius_shift = 3u;
inline constexpr std::uint32_t k_gwn_hploc_merging_threshold = 16u;
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

template <class NodeIndex> struct gwn_hploc_init_cluster_indices_functor {
    cuda::std::span<NodeIndex> cluster_indices{};

    __device__ void operator()(std::size_t const leaf_id) const noexcept {
        if (leaf_id < cluster_indices.size())
            cluster_indices[leaf_id] = static_cast<NodeIndex>(leaf_id);
    }
};

template <class T>
[[nodiscard]] __device__ inline T
gwn_hploc_shfl_value(unsigned int const mask, T const value, int const src_lane) noexcept {
    return __shfl_sync(mask, value, src_lane);
}

template <>
[[nodiscard]] __device__ inline gwn_hploc_u64 gwn_hploc_shfl_value<gwn_hploc_u64>(
    unsigned int const mask, gwn_hploc_u64 const value, int const src_lane
) noexcept {
    auto const lo = static_cast<unsigned int>(value & 0xffffffffull);
    auto const hi = static_cast<unsigned int>(value >> 32u);
    unsigned int const shuffled_lo = __shfl_sync(mask, lo, src_lane);
    unsigned int const shuffled_hi = __shfl_sync(mask, hi, src_lane);
    return (static_cast<gwn_hploc_u64>(shuffled_hi) << 32u) |
           static_cast<gwn_hploc_u64>(shuffled_lo);
}

template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_aabb<Real> gwn_hploc_shfl_aabb(
    unsigned int const mask, gwn_aabb<Real> const &value, int const src_lane
) noexcept {
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

/// \brief Build the H-PLOC binary hierarchy directly in the shared binary-builder representation.
///
/// Cluster IDs occupy one continuous space: leaf IDs are in [0, N), and every successful merge
/// allocates an internal ID in [N, 2N-1). The internal part maps directly onto \p binary_nodes and
/// \p binary_internal_bounds, so no second node representation or emit pass is required.
template <
    int BlockSize, std::uint32_t MaxSearchRadius, std::uint32_t MergingThreshold,
    gwn_real_type Real, gwn_index_type Index, class MortonCode, class NodeIndex>
__global__ __launch_bounds__(BlockSize) void gwn_build_binary_hploc_kernel(
    NodeIndex const primitive_count, std::uint32_t const search_radius,
    gwn_aabb<Real> const *sorted_leaf_aabbs, gwn_binary_node<Index> *binary_nodes,
    gwn_aabb<Real> *binary_internal_bounds, Index *binary_internal_parent,
    std::uint8_t *internal_parent_slot, Index *leaf_parent, std::uint8_t *leaf_parent_slot,
    NodeIndex *cluster_indices, NodeIndex *boundary_parent, NodeIndex *cluster_count,
    MortonCode const *sorted_morton_codes, unsigned int *failure_flag
) {
    constexpr int k_warp_size = 32;
    constexpr unsigned int k_warp_mask = 0xffffffffu;
    static_assert((BlockSize % k_warp_size) == 0, "BlockSize must be a multiple of 32.");
    static_assert(MergingThreshold == k_warp_size / 2, "H-PLOC uses half-warp cluster lists.");
    static_assert(
        MaxSearchRadius == (std::uint32_t(1) << k_gwn_hploc_search_radius_shift),
        "H-PLOC packed neighbor keys require a power-of-two maximum search radius."
    );
    static_assert(
        std::is_same_v<NodeIndex, std::uint32_t> || std::is_same_v<NodeIndex, gwn_hploc_u64>,
        "NodeIndex must be uint32_t or unsigned long long."
    );

    auto const lane = static_cast<std::uint32_t>(threadIdx.x) & (k_warp_size - 1u);
    auto const idx = static_cast<NodeIndex>(gwn_global_thread_index_1d());
    NodeIndex const internal_count = primitive_count - 1;

    auto const signal_failure = [&]() noexcept {
        if (failure_flag != nullptr)
            atomicExch(failure_flag, 1u);
    };
    // H-PLOC follows the implicit LBVH hierarchy only to decide when adjacent Morton ranges meet.
    // Equal Morton codes are extended with their sorted positions so every boundary remains
    // ordered.
    auto const delta = [&](NodeIndex const a, NodeIndex const b) noexcept {
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
        return key_delta == 0 ? static_cast<std::uint64_t>(a ^ b) : key_delta;
    };
    auto const find_parent_id = [&](NodeIndex const left, NodeIndex const right) noexcept {
        if (left == 0 ||
            (right != (primitive_count - 1) && delta(right, right + 1) < delta(left - 1, left))) {
            return right;
        }
        return left - 1;
    };
    auto const load_cluster_bounds = [&](NodeIndex const cluster_idx) noexcept {
        if (cluster_idx < primitive_count)
            return sorted_leaf_aabbs[static_cast<std::size_t>(cluster_idx)];
        return binary_internal_bounds[static_cast<std::size_t>(cluster_idx - primitive_count)];
    };
    // A range stores its compact cluster list at the start of that range. Each child contributes
    // at most half a warp, so concatenating two children always fits in one warp.
    auto const load_indices = [&](NodeIndex const start, NodeIndex const end,
                                  NodeIndex &cluster_idx_out,
                                  std::uint32_t const offset) noexcept -> std::uint32_t {
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
                                   NodeIndex const cluster_idx,
                                   NodeIndex const left_start) noexcept {
        if (lane < previous_num_prim)
            cluster_indices[left_start + static_cast<NodeIndex>(lane)] = cluster_idx;
        // The boundary exchange that releases the sibling path must observe the complete list and
        // every internal bound referenced by it.
        __threadfence();
    };

    auto const find_nearest_neighbor = [&](std::uint32_t const num_prim,
                                           gwn_aabb<Real> const cluster_bounds) noexcept {
        // Expanding the fixed eight steps removes loop-control and repeated branch work from the
        // hottest path. The runtime bound still lets callers select a smaller effective radius.
        if constexpr (std::is_same_v<Real, float>) {
            // Positive float area bits preserve numerical order. The low bits hold the relative
            // offset and direction parity. Discarding those low area bits gives equal-area buckets
            // deterministic ordering while keeping all nearest-neighbor state in one register.
            constexpr std::uint32_t k_neighbor_mask =
                (std::uint32_t(1) << (k_gwn_hploc_search_radius_shift + 1u)) - 1u;
            constexpr std::uint32_t k_area_mask = ~k_neighbor_mask;
            std::uint32_t best_key = std::numeric_limits<std::uint32_t>::max();

#pragma unroll
            for (std::uint32_t radius = 1; radius <= MaxSearchRadius; ++radius) {
                if (radius > search_radius)
                    break;
                std::uint32_t const neighbor_lane = lane + radius;
                bool const neighbor_valid = neighbor_lane < num_prim;
                int const safe_neighbor =
                    neighbor_valid ? static_cast<int>(neighbor_lane) : static_cast<int>(lane);
                gwn_aabb<Real> const neighbor_bounds =
                    gwn_hploc_shfl_aabb(k_warp_mask, cluster_bounds, safe_neighbor);

                std::uint32_t right_key = std::numeric_limits<std::uint32_t>::max();
                if (neighbor_valid) {
                    Real const area =
                        gwn_aabb_half_area(gwn_aabb_union(cluster_bounds, neighbor_bounds));
                    std::uint32_t const area_key = (__float_as_uint(area) << 1u) & k_area_mask;
                    std::uint32_t const relative_offset = (radius - 1u) << 1u;
                    std::uint32_t const left_key = area_key | relative_offset | (lane & 1u);
                    right_key = area_key | relative_offset | ((neighbor_lane & 1u) ^ 1u);
                    best_key = std::min(best_key, left_key);
                }

                std::uint32_t neighbor_key =
                    gwn_hploc_shfl_value(k_warp_mask, best_key, safe_neighbor);
                if (neighbor_valid)
                    neighbor_key = std::min(neighbor_key, right_key);

                int const left_lane = static_cast<int>(lane) - static_cast<int>(radius);
                int const safe_left = left_lane >= 0 ? left_lane : static_cast<int>(lane);
                std::uint32_t const shifted_key =
                    gwn_hploc_shfl_value(k_warp_mask, neighbor_key, safe_left);
                if (left_lane >= 0)
                    best_key = shifted_key;
            }

            std::uint32_t const offset = ((best_key & k_neighbor_mask) >> 1u) + 1u;
            bool const points_right = ((best_key ^ lane) & 1u) == 0u;
            return points_right ? lane + offset : lane - offset;
        } else {
            // Double precision keeps the full area value. A unique undirected lane pair breaks
            // equal-area ties identically at both endpoints and retains the mutual-pair guarantee.
            Real best_area = std::numeric_limits<Real>::infinity();
            std::uint32_t best_edge = k_gwn_hploc_invalid_lane;
            std::uint32_t best_lane = k_gwn_hploc_invalid_lane;
            auto const is_better = [](Real const candidate_area, std::uint32_t const candidate_edge,
                                      Real const current_area,
                                      std::uint32_t const current_edge) noexcept {
                return candidate_area < current_area ||
                       (candidate_area == current_area && candidate_edge < current_edge);
            };

#pragma unroll
            for (std::uint32_t radius = 1; radius <= MaxSearchRadius; ++radius) {
                if (radius > search_radius)
                    break;
                std::uint32_t const neighbor_lane = lane + radius;
                bool const neighbor_valid = neighbor_lane < num_prim;
                int const safe_neighbor =
                    neighbor_valid ? static_cast<int>(neighbor_lane) : static_cast<int>(lane);
                gwn_aabb<Real> const neighbor_bounds =
                    gwn_hploc_shfl_aabb(k_warp_mask, cluster_bounds, safe_neighbor);

                Real area = std::numeric_limits<Real>::infinity();
                std::uint32_t edge = k_gwn_hploc_invalid_lane;
                if (neighbor_valid) {
                    area = gwn_aabb_half_area(gwn_aabb_union(cluster_bounds, neighbor_bounds));
                    edge = lane * static_cast<std::uint32_t>(k_warp_size) + neighbor_lane;
                    if (is_better(area, edge, best_area, best_edge)) {
                        best_area = area;
                        best_edge = edge;
                        best_lane = neighbor_lane;
                    }
                }

                Real neighbor_area = gwn_hploc_shfl_value(k_warp_mask, best_area, safe_neighbor);
                std::uint32_t neighbor_edge =
                    gwn_hploc_shfl_value(k_warp_mask, best_edge, safe_neighbor);
                std::uint32_t neighbor_best_lane =
                    gwn_hploc_shfl_value(k_warp_mask, best_lane, safe_neighbor);
                if (neighbor_valid && is_better(area, edge, neighbor_area, neighbor_edge)) {
                    neighbor_area = area;
                    neighbor_edge = edge;
                    neighbor_best_lane = lane;
                }

                int const left_lane = static_cast<int>(lane) - static_cast<int>(radius);
                int const safe_left = left_lane >= 0 ? left_lane : static_cast<int>(lane);
                Real const shifted_area =
                    gwn_hploc_shfl_value(k_warp_mask, neighbor_area, safe_left);
                std::uint32_t const shifted_edge =
                    gwn_hploc_shfl_value(k_warp_mask, neighbor_edge, safe_left);
                std::uint32_t const shifted_lane =
                    gwn_hploc_shfl_value(k_warp_mask, neighbor_best_lane, safe_left);
                if (left_lane >= 0) {
                    best_area = shifted_area;
                    best_edge = shifted_edge;
                    best_lane = shifted_lane;
                }
            }
            return best_lane;
        }
    };

    auto const merge_clusters = [&](std::uint32_t const num_prim,
                                    std::uint32_t const nearest_neighbor, NodeIndex &cluster_idx,
                                    gwn_aabb<Real> &cluster_bounds) noexcept -> std::uint32_t {
        bool const lane_active = lane < num_prim;
        bool const neighbor_valid = nearest_neighbor < num_prim && nearest_neighbor != lane;
        if (lane_active && !neighbor_valid)
            signal_failure();
        std::uint32_t const safe_neighbor = neighbor_valid ? nearest_neighbor : lane;
        std::uint32_t const neighbor_nearest =
            gwn_hploc_shfl_value(k_warp_mask, safe_neighbor, static_cast<int>(safe_neighbor));
        bool const mutual_neighbor = lane_active && neighbor_valid && lane == neighbor_nearest;
        bool const merge = mutual_neighbor && lane < safe_neighbor;

        unsigned int const merge_mask = __ballot_sync(k_warp_mask, merge);
        auto const merge_count = static_cast<std::uint32_t>(__popc(merge_mask));
        // One lane reserves all nodes produced by this warp iteration. Prefix population counts
        // then give each merging lane a unique position without one global atomic per node.
        NodeIndex base_idx = 0;
        if (lane == 0)
            base_idx = gwn_hploc_atomic_add(cluster_count, static_cast<NodeIndex>(merge_count));
        base_idx = gwn_hploc_shfl_value(k_warp_mask, base_idx, 0);

        std::uint32_t const lower_lane_mask = lane == 0u ? 0u : ((1u << lane) - 1u);
        std::uint32_t const relative_idx =
            static_cast<std::uint32_t>(__popc(merge_mask & lower_lane_mask));
        NodeIndex const neighbor_cluster_idx =
            gwn_hploc_shfl_value(k_warp_mask, cluster_idx, static_cast<int>(safe_neighbor));
        gwn_aabb<Real> const neighbor_bounds =
            gwn_hploc_shfl_aabb(k_warp_mask, cluster_bounds, static_cast<int>(safe_neighbor));

        if (merge) {
            NodeIndex const merged_idx = base_idx + static_cast<NodeIndex>(relative_idx);
            if (merged_idx < primitive_count || merged_idx >= primitive_count + internal_count) {
                signal_failure();
            } else {
                gwn_aabb<Real> const merged_bounds =
                    gwn_aabb_union(cluster_bounds, neighbor_bounds);
                auto const internal_id_node = merged_idx - primitive_count;
                auto const internal_id = static_cast<Index>(internal_id_node);
                auto const make_child = [&](NodeIndex const child_idx) noexcept {
                    gwn_binary_child_ref<Index> child{};
                    if (child_idx < primitive_count) {
                        child.kind = static_cast<std::uint8_t>(gwn_binary_child_kind::k_leaf);
                        child.index = static_cast<Index>(child_idx);
                    } else {
                        child.kind = static_cast<std::uint8_t>(gwn_binary_child_kind::k_internal);
                        child.index = static_cast<Index>(child_idx - primitive_count);
                    }
                    return child;
                };
                gwn_binary_node<Index> node{};
                node.left = make_child(cluster_idx);
                node.right = make_child(neighbor_cluster_idx);
                binary_nodes[static_cast<std::size_t>(internal_id_node)] = node;
                binary_internal_bounds[static_cast<std::size_t>(internal_id_node)] = merged_bounds;

                // Each cluster becomes a child exactly once. Parent links can therefore be written
                // beside the merge without atomics or a second hierarchy pass.
                auto const store_parent = [&](gwn_binary_child_ref<Index> const child,
                                              std::uint8_t const slot) noexcept {
                    if (child.kind ==
                        static_cast<std::uint8_t>(gwn_binary_child_kind::k_internal)) {
                        binary_internal_parent[static_cast<std::size_t>(child.index)] = internal_id;
                        internal_parent_slot[static_cast<std::size_t>(child.index)] = slot;
                    } else {
                        leaf_parent[static_cast<std::size_t>(child.index)] = internal_id;
                        leaf_parent_slot[static_cast<std::size_t>(child.index)] = slot;
                    }
                };
                store_parent(node.left, 0u);
                store_parent(node.right, 1u);
                cluster_idx = merged_idx;
                cluster_bounds = merged_bounds;
            }
        }

        // The left endpoint keeps the merged cluster, the right endpoint is removed, and __fns
        // compacts the remaining cluster records back into the low lanes for the next iteration.
        unsigned int const valid_mask = __ballot_sync(k_warp_mask, merge || !mutual_neighbor);
        int const shift = __fns(valid_mask, 0u, lane + 1u);
        int const safe_shift = shift == -1 ? static_cast<int>(lane) : shift;
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

    auto const ploc_merge = [&](int const selected_lane, NodeIndex const left_boundary,
                                NodeIndex const right_boundary, NodeIndex const split_boundary,
                                bool const is_final_range) noexcept {
        NodeIndex const local_left =
            gwn_hploc_shfl_value(k_warp_mask, left_boundary, selected_lane);
        NodeIndex const local_right_end =
            gwn_hploc_shfl_value(k_warp_mask, right_boundary, selected_lane) + 1;
        NodeIndex const local_split =
            gwn_hploc_shfl_value(k_warp_mask, split_boundary, selected_lane);

        NodeIndex cluster_idx = gwn_hploc_invalid_index<NodeIndex>();
        std::uint32_t const num_left = load_indices(local_left, local_split, cluster_idx, 0u);
        std::uint32_t const num_right =
            load_indices(local_split, local_right_end, cluster_idx, num_left);
        std::uint32_t num_prim = num_left + num_right;
        gwn_aabb<Real> cluster_bounds{};
        if (lane < num_prim)
            cluster_bounds = load_cluster_bounds(cluster_idx);

        std::uint32_t const threshold = gwn_hploc_shfl_value(
            k_warp_mask, is_final_range ? 1u : MergingThreshold, selected_lane
        );
        // The whole warp processes one active Morton path at a time. Non-root ranges stop at half
        // a warp so their parent can concatenate two lists; the root reduces to one cluster.
        while (num_prim > threshold) {
            std::uint32_t const previous_num_prim = num_prim;
            std::uint32_t const nearest = find_nearest_neighbor(num_prim, cluster_bounds);
            num_prim = merge_clusters(num_prim, nearest, cluster_idx, cluster_bounds);
            // The packed float tie convention and the double edge tie-break each leave at least
            // one mutual nearest-neighbor pair. Zero progress therefore indicates corrupted
            // cluster state, not an alternate merge case.
            if (num_prim == previous_num_prim) {
                signal_failure();
                break;
            }
        }
        store_indices(num_left + num_right, cluster_idx, local_left);
    };

    NodeIndex left = idx;
    NodeIndex right = idx;
    NodeIndex split = 0;
    bool lane_active = idx < primitive_count;
    // The first path reaching a Morton boundary publishes its range and stops. The atomic exchange
    // lets the second path join both ranges and continue upward without materializing LBVH nodes.
    while (__ballot_sync(k_warp_mask, lane_active) != 0u) {
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
        bool const is_final_range = lane_active && size == primitive_count;
        unsigned int active_mask = __ballot_sync(
            k_warp_mask,
            (lane_active && size > static_cast<NodeIndex>(MergingThreshold)) || is_final_range
        );
        while (active_mask != 0u) {
            int const selected_lane = __ffs(static_cast<int>(active_mask)) - 1;
            ploc_merge(selected_lane, left, right, split, is_final_range);
            active_mask &= active_mask - 1u;
        }
    }
}

/// \brief Build binary hierarchy and internal bounds with the H-PLOC algorithm.
template <gwn_real_type Real, gwn_index_type Index, class MortonCode = std::uint64_t>
void gwn_bvh_build_binary_hploc_impl(
    cuda::std::span<Index const> const sorted_primitive_indices,
    cuda::std::span<MortonCode const> const sorted_morton_codes,
    cuda::std::span<gwn_aabb<Real> const> const sorted_primitive_aabbs,
    gwn_device_array<gwn_binary_node<Index>> &binary_nodes,
    gwn_device_array<Index> &binary_internal_parent,
    gwn_device_array<gwn_aabb<Real>> &binary_internal_bounds, Index &root_internal_index,
    std::uint32_t const search_radius = k_gwn_hploc_max_search_radius,
    cudaStream_t const stream = cudaStreamLegacy
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
        binary_nodes.clear(stream);
        binary_internal_parent.clear(stream);
        binary_internal_bounds.clear(stream);
        return;
    }
    if (sorted_morton_codes.size() != primitive_count ||
        sorted_primitive_aabbs.size() != primitive_count) {
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_build_binary_hploc, "H-PLOC preprocess buffer size mismatch."
        );
    }
    if constexpr (std::is_same_v<MortonCode, std::uint32_t>) {
        if (primitive_count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            return gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_build_binary_hploc,
                "H-PLOC with 32-bit Morton codes supports at most UINT32_MAX primitives."
            );
        }
    }
    if (search_radius == 0 || search_radius > k_gwn_hploc_max_search_radius) {
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_build_binary_hploc, "H-PLOC search radius must be in the range [1, 8]."
        );
    }

    NodeIndex const max_supported_primitive_count =
        (gwn_hploc_invalid_index<NodeIndex>() / 2u) + 1u;
    if (primitive_count > static_cast<std::size_t>(max_supported_primitive_count)) {
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_build_binary_hploc,
            "H-PLOC builder primitive count exceeds supported index range."
        );
    }
    auto const primitive_count_node_index = static_cast<NodeIndex>(primitive_count);
    if (primitive_count == 1) {
        binary_nodes.clear(stream);
        binary_internal_parent.clear(stream);
        binary_internal_bounds.clear(stream);
        return;
    }

    std::size_t const binary_internal_count = primitive_count - 1;
    gwn_binary_parent_temporaries<Index> temps{};
    gwn_prepare_binary_hierarchy_buffers(
        primitive_count, binary_nodes, binary_internal_parent, temps, stream
    );
    binary_internal_bounds.resize(binary_internal_count, stream);

    gwn_device_array<NodeIndex> cluster_indices{};
    gwn_device_array<NodeIndex> boundary_parent{};
    gwn_device_array<NodeIndex> cluster_count{};
    gwn_device_array<unsigned int> failure_flag{};
    cluster_indices.resize(primitive_count, stream);
    boundary_parent.resize(primitive_count, stream);
    cluster_count.resize(1, stream);
    failure_flag.resize(1, stream);

    // Cluster lists begin with one leaf ID per sorted Morton position. Parent arrays were already
    // initialized to invalid by the shared binary-builder buffer preparation above.
    constexpr int k_linear_block_size = k_gwn_default_block_size;
    gwn_throw_status_error(
        gwn_launch_linear_kernel<k_linear_block_size>(
            primitive_count,
            gwn_hploc_init_cluster_indices_functor<NodeIndex>{cluster_indices.span()}, stream
        )
    );
    gwn_throw_status_error(gwn_cuda_to_status(
        cudaMemsetAsync(boundary_parent.data(), 0xff, primitive_count * sizeof(NodeIndex), stream)
    ));
    NodeIndex const initial_cluster_count = primitive_count_node_index;
    gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
        cluster_count.data(), &initial_cluster_count, sizeof(NodeIndex), cudaMemcpyHostToDevice,
        stream
    )));
    gwn_throw_status_error(
        gwn_cuda_to_status(cudaMemsetAsync(failure_flag.data(), 0, sizeof(unsigned int), stream))
    );

    constexpr int k_hploc_block_size = 64;
    if (primitive_count > gwn_max_linear_kernel_elements_1d<k_hploc_block_size>()) {
        return gwn_bvh_invalid_argument(
            k_gwn_bvh_phase_build_binary_hploc,
            "H-PLOC kernel primitive count exceeds linear launch range."
        );
    }
    gwn_build_binary_hploc_kernel<
        k_hploc_block_size, k_gwn_hploc_max_search_radius, k_gwn_hploc_merging_threshold, Real,
        Index, MortonCode, NodeIndex>
        <<<gwn_grid_dim_1d<k_hploc_block_size>(primitive_count),
           gwn_block_dim_1d<k_hploc_block_size>(), 0, stream>>>(
            primitive_count_node_index, search_radius, sorted_primitive_aabbs.data(),
            binary_nodes.data(), binary_internal_bounds.data(), binary_internal_parent.data(),
            temps.internal_parent_slot.data(), temps.leaf_parent.data(),
            temps.leaf_parent_slot.data(), cluster_indices.data(), boundary_parent.data(),
            cluster_count.data(), sorted_morton_codes.data(), failure_flag.data()
        );
    gwn_throw_status_error(gwn_check_last_kernel());

    // The builder is object-facing and reports failures synchronously. A successful H-PLOC pass
    // creates exactly N-1 internal clusters, making 2N-1 the final allocation counter value.
    NodeIndex host_cluster_count = 0;
    unsigned int host_failure_flag = 0u;
    gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_cluster_count, cluster_count.data(), sizeof(NodeIndex), cudaMemcpyDeviceToHost, stream
    )));
    gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_failure_flag, failure_flag.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    gwn_throw_status_error(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    NodeIndex const expected_cluster_count = primitive_count_node_index * 2 - 1;
    if (host_failure_flag != 0u || host_cluster_count != expected_cluster_count) {
        return gwn_bvh_internal_error(
            k_gwn_bvh_phase_build_binary_hploc,
            "H-PLOC builder did not converge to a complete binary hierarchy."
        );
    }

    root_internal_index = static_cast<Index>(binary_internal_count - 1);
}

} // namespace detail
} // namespace gwn
