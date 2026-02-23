#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

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

template <class Index> struct gwn_binary_child_ref {
    std::uint8_t kind{};
    Index index{};
};

template <class Index> struct gwn_binary_node {
    gwn_binary_child_ref<Index> left{};
    gwn_binary_child_ref<Index> right{};
};

template <class Index> struct gwn_build_binary_topology_functor {
    cuda::std::span<std::uint32_t const> morton_codes{};
    cuda::std::span<gwn_binary_node<Index>> binary_nodes{};
    cuda::std::span<Index> internal_parent{};
    cuda::std::span<std::uint8_t> internal_parent_slot{};
    cuda::std::span<Index> leaf_parent{};
    cuda::std::span<std::uint8_t> leaf_parent_slot{};

    [[nodiscard]] __device__ inline int delta(Index const i, Index const j) const noexcept {
        Index const leaf_count = static_cast<Index>(morton_codes.size());
        if (j < Index(0) || j >= leaf_count)
            return -1;

        std::uint32_t const code_i = morton_codes[static_cast<std::size_t>(i)];
        std::uint32_t const code_j = morton_codes[static_cast<std::size_t>(j)];
        if (code_i == code_j) {
            std::uint32_t const diff = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(i) ^ static_cast<std::uint64_t>(j)
            );
            if (diff == 0)
                return 64;
            return 32 + __clz(diff);
        }
        return __clz(code_i ^ code_j);
    }

    __device__ void operator()(std::size_t const internal_id_u) const {
        Index const internal_id = static_cast<Index>(internal_id_u);
        Index const leaf_count = static_cast<Index>(morton_codes.size());
        if (leaf_count <= Index(1))
            return;

        int const direction = (delta(internal_id, internal_id + Index(1)) -
                                   delta(internal_id, internal_id - Index(1)) >=
                               0)
                                  ? 1
                                  : -1;
        int const delta_min = delta(internal_id, internal_id - static_cast<Index>(direction));

        Index range_length_max = Index(2);
        while (delta(internal_id, internal_id + range_length_max * static_cast<Index>(direction)) >
               delta_min) {
            range_length_max <<= 1;
        }

        Index range_length = Index(0);
        for (Index step = range_length_max >> 1; step > 0; step >>= 1) {
            if (delta(
                    internal_id, internal_id + (range_length + step) * static_cast<Index>(direction)
                ) > delta_min) {
                range_length += step;
            }
        }

        Index const j = internal_id + range_length * static_cast<Index>(direction);
        Index const first = std::min(internal_id, j);
        Index const last = std::max(internal_id, j);

        int const delta_node = delta(first, last);
        Index split = first;
        Index step = last - first;
        do {
            step = (step + Index(1)) >> 1;
            Index const candidate = split + step;
            if (candidate < last && delta(first, candidate) > delta_node)
                split = candidate;
        } while (step > Index(1));

        gwn_binary_node<Index> node{};
        if (split == first) {
            node.left.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
            node.left.index = split;
        } else {
            node.left.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
            node.left.index = split;
        }

        if (split + Index(1) == last) {
            node.right.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
            node.right.index = split + Index(1);
        } else {
            node.right.kind = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
            node.right.index = split + Index(1);
        }

        binary_nodes[internal_id_u] = node;
        if (node.left.kind == static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
            internal_parent[static_cast<std::size_t>(node.left.index)] = internal_id;
            internal_parent_slot[static_cast<std::size_t>(node.left.index)] = 0;
        } else {
            leaf_parent[static_cast<std::size_t>(node.left.index)] = internal_id;
            leaf_parent_slot[static_cast<std::size_t>(node.left.index)] = 0;
        }
        if (node.right.kind == static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal)) {
            internal_parent[static_cast<std::size_t>(node.right.index)] = internal_id;
            internal_parent_slot[static_cast<std::size_t>(node.right.index)] = 1;
        } else {
            leaf_parent[static_cast<std::size_t>(node.right.index)] = internal_id;
            leaf_parent_slot[static_cast<std::size_t>(node.right.index)] = 1;
        }
    }
};

template <class Real, class Index> struct gwn_accumulate_binary_bounds_pass_functor {
    cuda::std::span<gwn_aabb<Real> const> sorted_leaf_aabbs{};
    cuda::std::span<Index const> leaf_parent{};
    cuda::std::span<std::uint8_t const> leaf_parent_slot{};
    cuda::std::span<Index const> internal_parent{};
    cuda::std::span<std::uint8_t const> internal_parent_slot{};
    cuda::std::span<gwn_aabb<Real>> internal_bounds{};
    cuda::std::span<gwn_aabb<Real>> pending_left_bounds{};
    cuda::std::span<gwn_aabb<Real>> pending_right_bounds{};
    cuda::std::span<unsigned int> child_arrival_count{};

    __device__ void operator()(std::size_t const leaf_id) const {
        if (leaf_id >= sorted_leaf_aabbs.size() || leaf_id >= leaf_parent.size() ||
            leaf_id >= leaf_parent_slot.size()) {
            return;
        }

        Index parent = leaf_parent[leaf_id];
        if (parent < Index(0))
            return;

        std::uint8_t child_slot = leaf_parent_slot[leaf_id];
        if (child_slot > 1)
            return;

        gwn_aabb<Real> merged_bounds = sorted_leaf_aabbs[leaf_id];
        while (parent >= Index(0)) {
            std::size_t const parent_id = static_cast<std::size_t>(parent);
            if (parent_id >= internal_bounds.size() || parent_id >= pending_left_bounds.size() ||
                parent_id >= pending_right_bounds.size() ||
                parent_id >= child_arrival_count.size() || parent_id >= internal_parent.size() ||
                parent_id >= internal_parent_slot.size()) {
                return;
            }

            if (child_slot == 0)
                pending_left_bounds[parent_id] = merged_bounds;
            else
                pending_right_bounds[parent_id] = merged_bounds;

            __threadfence();
            unsigned int const prev_count = atomicAdd(child_arrival_count.data() + parent_id, 1u);
            if (prev_count == 0)
                return;

            gwn_aabb<Real> const left_bounds = pending_left_bounds[parent_id];
            gwn_aabb<Real> const right_bounds = pending_right_bounds[parent_id];
            merged_bounds = gwn_aabb_union(left_bounds, right_bounds);
            internal_bounds[parent_id] = merged_bounds;

            Index const next_parent = internal_parent[parent_id];
            if (next_parent < Index(0))
                return;

            child_slot = internal_parent_slot[parent_id];
            if (child_slot > 1)
                return;
            parent = next_parent;
        }
    }
};

template <int Value> struct gwn_log2_pow2;

template <> struct gwn_log2_pow2<1> {
    static constexpr int value = 0;
};

template <int Value> struct gwn_log2_pow2 {
    static_assert(Value > 1 && (Value & (Value - 1)) == 0, "Value must be a power of two.");
    static constexpr int value = 1 + gwn_log2_pow2<(Value >> 1)>::value;
};

template <int Width, class Index> struct gwn_collapse_summarize_pass_functor {
    static constexpr int k_collapse_depth = gwn_log2_pow2<Width>::value;

    cuda::std::span<Index const> internal_parent{};
    cuda::std::span<std::uint8_t> internal_is_wide_root{};
    cuda::std::span<Index> internal_wide_node_id{};
    cuda::std::span<Index> wide_node_binary_root{};
    unsigned int *wide_node_count{};
    Index root_internal_index{};

    __device__ void operator()(std::size_t const internal_id_u) const {
        if (internal_id_u >= internal_parent.size() ||
            internal_id_u >= internal_is_wide_root.size() ||
            internal_id_u >= internal_wide_node_id.size() || wide_node_count == nullptr) {
            return;
        }

        Index const internal_id = static_cast<Index>(internal_id_u);
        if (internal_id == root_internal_index)
            return;

        int depth = 0;
        Index cursor = internal_id;
        while (cursor >= Index(0) && cursor != root_internal_index) {
            std::size_t const cursor_u = static_cast<std::size_t>(cursor);
            if (cursor_u >= internal_parent.size())
                return;
            ++depth;
            cursor = internal_parent[cursor_u];
        }
        if (cursor != root_internal_index)
            return;
        if ((depth % k_collapse_depth) != 0)
            return;

        unsigned int const wide_node_id = atomicAdd(wide_node_count, 1u);
        if (wide_node_id >= wide_node_binary_root.size())
            return;

        internal_is_wide_root[internal_id_u] = std::uint8_t(1);
        internal_wide_node_id[internal_id_u] = static_cast<Index>(wide_node_id);
        wide_node_binary_root[wide_node_id] = internal_id;
    }
};

template <int Width, class Real, class Index> struct gwn_collapse_emit_nodes_pass_functor {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static constexpr int k_stack_capacity = Width * 2 + 8;

    cuda::std::span<gwn_binary_node<Index> const> binary_nodes{};
    cuda::std::span<gwn_aabb<Real> const> sorted_leaf_aabbs{};
    cuda::std::span<gwn_aabb<Real> const> internal_bounds{};
    cuda::std::span<std::uint8_t const> internal_is_wide_root{};
    cuda::std::span<Index const> internal_wide_node_id{};
    cuda::std::span<Index const> wide_node_binary_root{};
    cuda::std::span<gwn_bvh_node_soa<Width, Real, Index>> output_nodes{};
    unsigned int *overflow_flag{};

    __device__ static void
    gwn_set_invalid_child(gwn_bvh_node_soa<Width, Real, Index> &node, int const slot) {
        node.child_min_x[slot] = Real(0);
        node.child_min_y[slot] = Real(0);
        node.child_min_z[slot] = Real(0);
        node.child_max_x[slot] = Real(0);
        node.child_max_y[slot] = Real(0);
        node.child_max_z[slot] = Real(0);
        node.child_index[slot] = Index(0);
        node.child_count[slot] = Index(0);
        node.child_kind[slot] = static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
    }

    __device__ void operator()(std::size_t const wide_node_id_u) const {
        if (wide_node_id_u >= output_nodes.size() || wide_node_id_u >= wide_node_binary_root.size())
            return;

        gwn_bvh_node_soa<Width, Real, Index> output_node{};
        Index const binary_root = wide_node_binary_root[wide_node_id_u];
        if (binary_root < Index(0) ||
            static_cast<std::size_t>(binary_root) >= binary_nodes.size()) {
            GWN_PRAGMA_UNROLL
            for (int slot = 0; slot < Width; ++slot)
                gwn_set_invalid_child(output_node, slot);
            output_nodes[wide_node_id_u] = output_node;
            if (overflow_flag != nullptr)
                atomicExch(overflow_flag, 1u);
            return;
        }

        gwn_binary_child_ref<Index> stack[k_stack_capacity]{};
        int stack_size = 0;
        stack[stack_size++] = gwn_binary_child_ref<Index>{
            static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal), binary_root
        };

        int written_children = 0;
        while (stack_size > 0) {
            gwn_binary_child_ref<Index> const ref = stack[--stack_size];
            auto const kind = static_cast<gwn_bvh_child_kind>(ref.kind);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;

            if (written_children >= Width) {
                if (overflow_flag != nullptr)
                    atomicExch(overflow_flag, 1u);
                break;
            }

            if (kind == gwn_bvh_child_kind::k_leaf) {
                if (ref.index < Index(0) ||
                    static_cast<std::size_t>(ref.index) >= sorted_leaf_aabbs.size()) {
                    gwn_set_invalid_child(output_node, written_children++);
                    if (overflow_flag != nullptr)
                        atomicExch(overflow_flag, 1u);
                    continue;
                }
                gwn_aabb<Real> const bounds =
                    sorted_leaf_aabbs[static_cast<std::size_t>(ref.index)];
                int const slot = written_children++;
                output_node.child_min_x[slot] = bounds.min_x;
                output_node.child_min_y[slot] = bounds.min_y;
                output_node.child_min_z[slot] = bounds.min_z;
                output_node.child_max_x[slot] = bounds.max_x;
                output_node.child_max_y[slot] = bounds.max_y;
                output_node.child_max_z[slot] = bounds.max_z;
                output_node.child_index[slot] = ref.index;
                output_node.child_count[slot] = Index(1);
                output_node.child_kind[slot] =
                    static_cast<std::uint8_t>(gwn_bvh_child_kind::k_leaf);
                continue;
            }

            if (kind != gwn_bvh_child_kind::k_internal || ref.index < Index(0) ||
                static_cast<std::size_t>(ref.index) >= binary_nodes.size()) {
                gwn_set_invalid_child(output_node, written_children++);
                if (overflow_flag != nullptr)
                    atomicExch(overflow_flag, 1u);
                continue;
            }

            std::size_t const internal_index_u = static_cast<std::size_t>(ref.index);
            bool const is_child_wide_root = (ref.index != binary_root) &&
                                            (internal_index_u < internal_is_wide_root.size()) &&
                                            (internal_is_wide_root[internal_index_u] != 0);

            if (is_child_wide_root) {
                if (internal_index_u >= internal_bounds.size() ||
                    internal_index_u >= internal_wide_node_id.size()) {
                    gwn_set_invalid_child(output_node, written_children++);
                    if (overflow_flag != nullptr)
                        atomicExch(overflow_flag, 1u);
                    continue;
                }

                Index const child_wide_id = internal_wide_node_id[internal_index_u];
                if (child_wide_id < Index(0) ||
                    static_cast<std::size_t>(child_wide_id) >= output_nodes.size()) {
                    gwn_set_invalid_child(output_node, written_children++);
                    if (overflow_flag != nullptr)
                        atomicExch(overflow_flag, 1u);
                    continue;
                }

                gwn_aabb<Real> const bounds = internal_bounds[internal_index_u];
                int const slot = written_children++;
                output_node.child_min_x[slot] = bounds.min_x;
                output_node.child_min_y[slot] = bounds.min_y;
                output_node.child_min_z[slot] = bounds.min_z;
                output_node.child_max_x[slot] = bounds.max_x;
                output_node.child_max_y[slot] = bounds.max_y;
                output_node.child_max_z[slot] = bounds.max_z;
                output_node.child_index[slot] = child_wide_id;
                output_node.child_count[slot] = Index(0);
                output_node.child_kind[slot] =
                    static_cast<std::uint8_t>(gwn_bvh_child_kind::k_internal);
                continue;
            }

            gwn_binary_node<Index> const binary_node = binary_nodes[internal_index_u];
            if (stack_size + 2 > k_stack_capacity) {
                if (overflow_flag != nullptr)
                    atomicExch(overflow_flag, 1u);
                break;
            }
            stack[stack_size++] = binary_node.right;
            stack[stack_size++] = binary_node.left;
        }

        for (int slot = written_children; slot < Width; ++slot)
            gwn_set_invalid_child(output_node, slot);

        output_nodes[wide_node_id_u] = output_node;
    }
};

} // namespace detail

/// \brief Build a fixed-width LBVH topology from triangle geometry.
///
/// \remark This function only writes topology data (`nodes`, `primitive_indices`, root metadata).
/// \remark Existing topology memory in `bvh` is released before commit.
template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &bvh,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    static_assert(Width >= 2, "BVH width must be at least 2.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor is invalid for BVH construction.");

    if (geometry.triangle_count() == 0) {
        detail::gwn_release_bvh_topology_accessor(bvh, stream);
        return gwn_status::ok();
    }
    if (geometry.vertex_count() == 0)
        return gwn_status::invalid_argument("Cannot build BVH with triangles but zero vertices.");

    std::size_t const primitive_count = geometry.triangle_count();
    constexpr int k_block_size = detail::k_gwn_default_block_size;
    gwn_bvh_topology_accessor<Width, Real, Index> staging_bvh{};
    auto cleanup_staging_bvh = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_topology_accessor(staging_bvh, stream);
    });
    auto commit_staging_bvh = [&]() -> gwn_status {
        detail::gwn_release_bvh_topology_accessor(bvh, stream);
        bvh = staging_bvh;
        cleanup_staging_bvh.release();
        return gwn_status::ok();
    };

    Real scene_min_x = Real(0);
    Real scene_max_x = Real(0);
    Real scene_min_y = Real(0);
    Real scene_max_y = Real(0);
    Real scene_min_z = Real(0);
    Real scene_max_z = Real(0);
    gwn_device_array<Real> scene_axis_min{};
    gwn_device_array<Real> scene_axis_max{};
    gwn_device_array<std::uint8_t> scene_reduce_temp{};
    GWN_RETURN_ON_ERROR(scene_axis_min.resize(1, stream));
    GWN_RETURN_ON_ERROR(scene_axis_max.resize(1, stream));
    GWN_RETURN_ON_ERROR(
        detail::gwn_reduce_minmax_cub(
            geometry.vertex_x, scene_axis_min, scene_axis_max, scene_reduce_temp, scene_min_x,
            scene_max_x, stream
        )
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_reduce_minmax_cub(
            geometry.vertex_y, scene_axis_min, scene_axis_max, scene_reduce_temp, scene_min_y,
            scene_max_y, stream
        )
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_reduce_minmax_cub(
            geometry.vertex_z, scene_axis_min, scene_axis_max, scene_reduce_temp, scene_min_z,
            scene_max_z, stream
        )
    );
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

    Real const scene_inv_x =
        (scene_max_x > scene_min_x) ? Real(1) / (scene_max_x - scene_min_x) : Real(1);
    Real const scene_inv_y =
        (scene_max_y > scene_min_y) ? Real(1) / (scene_max_y - scene_min_y) : Real(1);
    Real const scene_inv_z =
        (scene_max_z > scene_min_z) ? Real(1) / (scene_max_z - scene_min_z) : Real(1);

    gwn_device_array<gwn_aabb<Real>> primitive_aabbs{};
    gwn_device_array<std::uint32_t> morton_codes{};
    gwn_device_array<Index> primitive_indices{};
    GWN_RETURN_ON_ERROR(primitive_aabbs.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(morton_codes.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(primitive_indices.resize(primitive_count, stream));
    auto const primitive_aabbs_span = primitive_aabbs.span();
    auto const morton_codes_span = morton_codes.span();
    auto const primitive_indices_span = primitive_indices.span();
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            detail::gwn_compute_triangle_aabbs_and_morton_functor<Real, Index>{
                geometry, scene_min_x, scene_min_y, scene_min_z, scene_inv_x, scene_inv_y,
                scene_inv_z, primitive_aabbs_span, morton_codes_span, primitive_indices_span
            },
            stream
        )
    );

    if (primitive_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return gwn_status::invalid_argument("CUB radix sort input exceeds int32 item count.");
    int const radix_item_count = static_cast<int>(primitive_count);
    gwn_device_array<std::uint32_t> sorted_morton_codes{};
    gwn_device_array<Index> sorted_primitive_indices{};
    gwn_device_array<std::uint8_t> radix_sort_temp{};
    GWN_RETURN_ON_ERROR(sorted_morton_codes.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(sorted_primitive_indices.resize(primitive_count, stream));

    std::size_t radix_sort_temp_bytes = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            nullptr, radix_sort_temp_bytes, morton_codes_span.data(), sorted_morton_codes.data(),
            primitive_indices_span.data(), sorted_primitive_indices.data(), radix_item_count, 0,
            static_cast<int>(sizeof(std::uint32_t) * 8), stream
        )
    ));
    GWN_RETURN_ON_ERROR(radix_sort_temp.resize(radix_sort_temp_bytes, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            radix_sort_temp.data(), radix_sort_temp_bytes, morton_codes_span.data(),
            sorted_morton_codes.data(), primitive_indices_span.data(),
            sorted_primitive_indices.data(), radix_item_count, 0,
            static_cast<int>(sizeof(std::uint32_t) * 8), stream
        )
    ));
    auto const sorted_morton_codes_span = sorted_morton_codes.span();
    auto const sorted_primitive_indices_span = sorted_primitive_indices.span();

    gwn_device_array<gwn_aabb<Real>> sorted_aabbs{};
    GWN_RETURN_ON_ERROR(sorted_aabbs.resize(primitive_count, stream));
    auto const sorted_aabbs_span = sorted_aabbs.span();
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

    if (primitive_count == 1) {
        staging_bvh.root_kind = gwn_bvh_child_kind::k_leaf;
        staging_bvh.root_index = Index(0);
        staging_bvh.root_count = Index(1);
        return commit_staging_bvh();
    }

    std::size_t const binary_internal_count = primitive_count - 1;
    gwn_device_array<detail::gwn_binary_node<Index>> binary_nodes{};
    gwn_device_array<Index> binary_internal_parent{};
    gwn_device_array<std::uint8_t> binary_internal_parent_slot{};
    gwn_device_array<Index> binary_leaf_parent{};
    gwn_device_array<std::uint8_t> binary_leaf_parent_slot{};
    GWN_RETURN_ON_ERROR(binary_nodes.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_internal_parent.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_internal_parent_slot.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_leaf_parent.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(binary_leaf_parent_slot.resize(primitive_count, stream));
    auto const binary_nodes_span = binary_nodes.span();
    auto const binary_internal_parent_span = binary_internal_parent.span();
    auto const binary_internal_parent_slot_span = binary_internal_parent_slot.span();
    auto const binary_leaf_parent_span = binary_leaf_parent.span();
    auto const binary_leaf_parent_slot_span = binary_leaf_parent_slot.span();
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        binary_internal_parent_span.data(), 0xff, binary_internal_count * sizeof(Index), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        binary_internal_parent_slot_span.data(), 0xff, binary_internal_count * sizeof(std::uint8_t),
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        binary_leaf_parent_span.data(), 0xff, primitive_count * sizeof(Index), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        binary_leaf_parent_slot_span.data(), 0xff, primitive_count * sizeof(std::uint8_t), stream
    )));
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            binary_internal_count,
            detail::gwn_build_binary_topology_functor<Index>{
                sorted_morton_codes_span, binary_nodes_span, binary_internal_parent_span,
                binary_internal_parent_slot_span, binary_leaf_parent_span,
                binary_leaf_parent_slot_span
            },
            stream
        )
    );

    Index const root_internal_index = Index(0);

    gwn_device_array<gwn_aabb<Real>> binary_internal_bounds{};
    gwn_device_array<gwn_aabb<Real>> binary_pending_left{};
    gwn_device_array<gwn_aabb<Real>> binary_pending_right{};
    gwn_device_array<unsigned int> binary_child_arrivals{};
    GWN_RETURN_ON_ERROR(binary_internal_bounds.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_pending_left.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_pending_right.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(binary_child_arrivals.resize(binary_internal_count, stream));
    auto const binary_internal_bounds_span = binary_internal_bounds.span();
    auto const binary_pending_left_span = binary_pending_left.span();
    auto const binary_pending_right_span = binary_pending_right.span();
    auto const binary_child_arrivals_span = binary_child_arrivals.span();
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        binary_child_arrivals_span.data(), 0, binary_internal_count * sizeof(unsigned int), stream
    )));

    // Binary bounds pass: launch one thread per leaf and propagate bounds upward.
    // Each internal node finalizes after both child arrivals.
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            detail::gwn_accumulate_binary_bounds_pass_functor<Real, Index>{
                cuda::std::span<gwn_aabb<Real> const>(
                    sorted_aabbs_span.data(), sorted_aabbs_span.size()
                ),
                cuda::std::span<Index const>(
                    binary_leaf_parent_span.data(), binary_leaf_parent_span.size()
                ),
                cuda::std::span<std::uint8_t const>(
                    binary_leaf_parent_slot_span.data(), binary_leaf_parent_slot_span.size()
                ),
                cuda::std::span<Index const>(
                    binary_internal_parent_span.data(), binary_internal_parent_span.size()
                ),
                cuda::std::span<std::uint8_t const>(
                    binary_internal_parent_slot_span.data(), binary_internal_parent_slot_span.size()
                ),
                binary_internal_bounds_span, binary_pending_left_span, binary_pending_right_span,
                binary_child_arrivals_span
            },
            stream
        )
    );

    static_assert(
        (Width & (Width - 1)) == 0, "BVH collapse currently requires power-of-two Width."
    );
    gwn_device_array<std::uint8_t> collapse_internal_is_wide_root{};
    gwn_device_array<Index> collapse_internal_wide_node_id{};
    gwn_device_array<Index> collapse_wide_node_binary_root{};
    GWN_RETURN_ON_ERROR(collapse_internal_is_wide_root.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(collapse_internal_wide_node_id.resize(binary_internal_count, stream));
    GWN_RETURN_ON_ERROR(collapse_wide_node_binary_root.resize(binary_internal_count, stream));
    auto const collapse_internal_is_wide_root_span = collapse_internal_is_wide_root.span();
    auto const collapse_internal_wide_node_id_span = collapse_internal_wide_node_id.span();
    auto const collapse_wide_node_binary_root_span = collapse_wide_node_binary_root.span();

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        collapse_internal_is_wide_root_span.data(), 0, binary_internal_count * sizeof(std::uint8_t),
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        collapse_internal_wide_node_id_span.data(), 0xff, binary_internal_count * sizeof(Index),
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        collapse_wide_node_binary_root_span.data(), 0xff, binary_internal_count * sizeof(Index),
        stream
    )));

    void *collapse_wide_count_raw = nullptr;
    GWN_RETURN_ON_ERROR(gwn_cuda_malloc(&collapse_wide_count_raw, sizeof(unsigned int), stream));
    auto cleanup_collapse_wide_count = gwn_make_scope_exit([&]() noexcept {
        (void)gwn_cuda_free(collapse_wide_count_raw, stream);
    });
    unsigned int *collapse_wide_count = static_cast<unsigned int *>(collapse_wide_count_raw);
    unsigned int const collapse_wide_count_init = 1u;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_wide_count, &collapse_wide_count_init, sizeof(unsigned int),
        cudaMemcpyHostToDevice, stream
    )));
    std::uint8_t const collapse_root_is_wide = std::uint8_t(1);
    Index const collapse_root_wide_node_id = Index(0);
    Index const collapse_root_binary_root = root_internal_index;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_internal_is_wide_root_span.data(), &collapse_root_is_wide, sizeof(std::uint8_t),
        cudaMemcpyHostToDevice, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_internal_wide_node_id_span.data(), &collapse_root_wide_node_id, sizeof(Index),
        cudaMemcpyHostToDevice, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        collapse_wide_node_binary_root_span.data(), &collapse_root_binary_root, sizeof(Index),
        cudaMemcpyHostToDevice, stream
    )));

    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            binary_internal_count,
            detail::gwn_collapse_summarize_pass_functor<Width, Index>{
                cuda::std::span<Index const>(
                    binary_internal_parent_span.data(), binary_internal_parent_span.size()
                ),
                collapse_internal_is_wide_root_span, collapse_internal_wide_node_id_span,
                collapse_wide_node_binary_root_span, collapse_wide_count, root_internal_index
            },
            stream
        )
    );

    unsigned int host_collapse_wide_count = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_collapse_wide_count, collapse_wide_count, sizeof(unsigned int),
        cudaMemcpyDeviceToHost, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

    if (host_collapse_wide_count == 0 || host_collapse_wide_count > binary_internal_count) {
        return gwn_status::internal_error(
            "BVH collapse summarize produced invalid wide-node count."
        );
    }

    GWN_RETURN_ON_ERROR(
        detail::gwn_allocate_span(
            staging_bvh.nodes, static_cast<std::size_t>(host_collapse_wide_count), stream
        )
    );
    auto const staging_nodes_span = cuda::std::span<gwn_bvh_node_soa<Width, Real, Index>>(
        const_cast<gwn_bvh_node_soa<Width, Real, Index> *>(staging_bvh.nodes.data()),
        static_cast<std::size_t>(host_collapse_wide_count)
    );

    void *collapse_overflow_raw = nullptr;
    GWN_RETURN_ON_ERROR(gwn_cuda_malloc(&collapse_overflow_raw, sizeof(unsigned int), stream));
    auto cleanup_collapse_overflow =
        gwn_make_scope_exit([&]() noexcept { (void)gwn_cuda_free(collapse_overflow_raw, stream); });
    unsigned int *collapse_overflow = static_cast<unsigned int *>(collapse_overflow_raw);
    GWN_RETURN_ON_ERROR(
        gwn_cuda_to_status(cudaMemsetAsync(collapse_overflow, 0, sizeof(unsigned int), stream))
    );

    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            static_cast<std::size_t>(host_collapse_wide_count),
            detail::gwn_collapse_emit_nodes_pass_functor<Width, Real, Index>{
                cuda::std::span<detail::gwn_binary_node<Index> const>(
                    binary_nodes_span.data(), binary_nodes_span.size()
                ),
                cuda::std::span<gwn_aabb<Real> const>(
                    sorted_aabbs_span.data(), sorted_aabbs_span.size()
                ),
                cuda::std::span<gwn_aabb<Real> const>(
                    binary_internal_bounds_span.data(), binary_internal_bounds_span.size()
                ),
                cuda::std::span<std::uint8_t const>(
                    collapse_internal_is_wide_root_span.data(),
                    collapse_internal_is_wide_root_span.size()
                ),
                cuda::std::span<Index const>(
                    collapse_internal_wide_node_id_span.data(),
                    collapse_internal_wide_node_id_span.size()
                ),
                cuda::std::span<Index const>(
                    collapse_wide_node_binary_root_span.data(),
                    static_cast<std::size_t>(host_collapse_wide_count)
                ),
                staging_nodes_span, collapse_overflow
            },
            stream
        )
    );

    unsigned int host_collapse_overflow = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_collapse_overflow, collapse_overflow, sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_collapse_overflow != 0) {
        return gwn_status::internal_error(
            "BVH collapse execute pass overflowed fixed-width node capacity."
        );
    }

    staging_bvh.root_kind = gwn_bvh_child_kind::k_internal;
    staging_bvh.root_index = Index(0);
    staging_bvh.root_count = Index(0);
    return commit_staging_bvh();
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_build_bvh_lbvh.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_build_bvh_lbvh.");
}

/// \brief Convenience wrapper for `Width == 4`.
template <class Real, class Index>
gwn_status gwn_build_bvh4_lbvh(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &bvh,
    cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    return gwn_build_bvh_lbvh<4, Real, Index>(geometry, bvh, stream);
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_build_bvh4_lbvh.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_build_bvh4_lbvh.");
}

namespace detail {

template <int Order, class Real> struct gwn_device_taylor_moment;

template <class Real> struct gwn_device_taylor_moment<0, Real> {
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

template <class Real> struct gwn_device_taylor_moment<1, Real> : gwn_device_taylor_moment<0, Real> {
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
[[nodiscard]] __host__ __device__ inline Real gwn_bounds_max_p_dist2(
    gwn_aabb<Real> const &bounds, Real const average_x, Real const average_y, Real const average_z
) noexcept {
    Real const dx = std::max(average_x - bounds.min_x, bounds.max_x - average_x);
    Real const dy = std::max(average_y - bounds.min_y, bounds.max_y - average_y);
    Real const dz = std::max(average_z - bounds.min_z, bounds.max_z - average_z);
    return dx * dx + dy * dy + dz * dz;
}

template <int Order, class Real>
__device__ inline void
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
__device__ inline void gwn_write_taylor_child(
    gwn_bvh4_taylor_node_soa<Order, Real> &node, int const child_slot,
    gwn_device_taylor_moment<Order, Real> const &moment
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

template <int Order, class Real, class Index>
__device__ inline bool gwn_compute_triangle_taylor_moment(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_device_taylor_moment<Order, Real> &moment, gwn_aabb<Real> &bounds
) noexcept {
    if (primitive_id < Index(0) ||
        static_cast<std::size_t>(primitive_id) >= geometry.triangle_count()) {
        return false;
    }

    std::size_t const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (ia < Index(0) || ib < Index(0) || ic < Index(0))
        return false;

    std::size_t const a_index = static_cast<std::size_t>(ia);
    std::size_t const b_index = static_cast<std::size_t>(ib);
    std::size_t const c_index = static_cast<std::size_t>(ic);
    if (a_index >= geometry.vertex_count() || b_index >= geometry.vertex_count() ||
        c_index >= geometry.vertex_count()) {
        return false;
    }

    Real const ax = geometry.vertex_x[a_index];
    Real const ay = geometry.vertex_y[a_index];
    Real const az = geometry.vertex_z[a_index];
    Real const bx = geometry.vertex_x[b_index];
    Real const by = geometry.vertex_y[b_index];
    Real const bz = geometry.vertex_z[b_index];
    Real const cx = geometry.vertex_x[c_index];
    Real const cy = geometry.vertex_y[c_index];
    Real const cz = geometry.vertex_z[c_index];

    bounds = gwn_aabb<Real>{
        std::min(ax, std::min(bx, cx)), std::min(ay, std::min(by, cy)),
        std::min(az, std::min(bz, cz)), std::max(ax, std::max(bx, cx)),
        std::max(ay, std::max(by, cy)), std::max(az, std::max(bz, cz)),
    };

    Real const abx = bx - ax;
    Real const aby = by - ay;
    Real const abz = bz - az;
    Real const acx = cx - ax;
    Real const acy = cy - ay;
    Real const acz = cz - az;

    moment.n_x = Real(0.5) * (aby * acz - abz * acy);
    moment.n_y = Real(0.5) * (abz * acx - abx * acz);
    moment.n_z = Real(0.5) * (abx * acy - aby * acx);

    Real const area2 = moment.n_x * moment.n_x + moment.n_y * moment.n_y + moment.n_z * moment.n_z;
    moment.area = sqrt(std::max(area2, Real(0)));
    moment.average_x = (ax + bx + cx) / Real(3);
    moment.average_y = (ay + by + cy) / Real(3);
    moment.average_z = (az + bz + cz) / Real(3);
    moment.area_p_x = moment.average_x * moment.area;
    moment.area_p_y = moment.average_y * moment.area;
    moment.area_p_z = moment.average_z * moment.area;
    moment.max_p_dist2 =
        gwn_bounds_max_p_dist2(bounds, moment.average_x, moment.average_y, moment.average_z);
    return true;
}

template <int Order, class Real, class Index>
__device__ inline gwn_device_taylor_moment<Order, Real> gwn_compute_leaf_taylor_moment(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> const &bvh,
    Index const begin, Index const count
) noexcept {
    using moment_type = gwn_device_taylor_moment<Order, Real>;

    moment_type leaf{};
    if (count <= Index(0))
        return leaf;

    bool has_primitive = false;
    gwn_aabb<Real> leaf_bounds{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
    for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
        Index const sorted_slot = begin + primitive_offset;
        if (sorted_slot < Index(0))
            continue;
        std::size_t const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
        if (sorted_slot_u >= bvh.primitive_indices.size())
            continue;

        Index const primitive_id = bvh.primitive_indices[sorted_slot_u];
        moment_type primitive{};
        gwn_aabb<Real> primitive_bounds{};
        if (!gwn_compute_triangle_taylor_moment<Order, Real, Index>(
                geometry, primitive_id, primitive, primitive_bounds
            )) {
            continue;
        }

        if (!has_primitive) {
            leaf_bounds = primitive_bounds;
            has_primitive = true;
        } else {
            leaf_bounds = gwn_aabb_union(leaf_bounds, primitive_bounds);
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
    leaf.max_p_dist2 =
        gwn_bounds_max_p_dist2(leaf_bounds, leaf.average_x, leaf.average_y, leaf.average_z);

    if constexpr (Order >= 1) {
        for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
            Index const sorted_slot = begin + primitive_offset;
            if (sorted_slot < Index(0))
                continue;
            std::size_t const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
            if (sorted_slot_u >= bvh.primitive_indices.size())
                continue;

            Index const primitive_id = bvh.primitive_indices[sorted_slot_u];
            moment_type primitive{};
            gwn_aabb<Real> primitive_bounds{};
            if (!gwn_compute_triangle_taylor_moment<Order, Real, Index>(
                    geometry, primitive_id, primitive, primitive_bounds
                )) {
                continue;
            }

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
}

template <int Order, class Real, class Index>
__device__ inline gwn_device_taylor_moment<Order, Real> gwn_compute_leaf_taylor_moment_cached(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> const &bvh,
    Index const begin, Index const count
) noexcept {
    using moment_type = gwn_device_taylor_moment<Order, Real>;

    constexpr int k_leaf_cache_capacity = 8;
    if (count <= Index(0))
        return moment_type{};
    if (count > Index(k_leaf_cache_capacity))
        return gwn_compute_leaf_taylor_moment<Order>(geometry, bvh, begin, count);

    moment_type primitive_cache[k_leaf_cache_capacity]{};
    gwn_aabb<Real> bounds_cache[k_leaf_cache_capacity]{};
    int cache_count = 0;

    moment_type leaf{};
    bool has_primitive = false;
    gwn_aabb<Real> leaf_bounds{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
    for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
        Index const sorted_slot = begin + primitive_offset;
        if (sorted_slot < Index(0))
            continue;
        std::size_t const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
        if (sorted_slot_u >= bvh.primitive_indices.size())
            continue;

        Index const primitive_id = bvh.primitive_indices[sorted_slot_u];
        moment_type primitive{};
        gwn_aabb<Real> primitive_bounds{};
        if (!gwn_compute_triangle_taylor_moment<Order, Real, Index>(
                geometry, primitive_id, primitive, primitive_bounds
            )) {
            continue;
        }

        primitive_cache[cache_count] = primitive;
        bounds_cache[cache_count] = primitive_bounds;
        ++cache_count;

        if (!has_primitive) {
            leaf_bounds = primitive_bounds;
            has_primitive = true;
        } else {
            leaf_bounds = gwn_aabb_union(leaf_bounds, primitive_bounds);
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
    leaf.max_p_dist2 =
        gwn_bounds_max_p_dist2(leaf_bounds, leaf.average_x, leaf.average_y, leaf.average_z);

    if constexpr (Order >= 1) {
        GWN_PRAGMA_UNROLL
        for (int cache_index = 0; cache_index < k_leaf_cache_capacity; ++cache_index) {
            if (cache_index >= cache_count)
                break;
            moment_type const primitive = primitive_cache[cache_index];
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
}

template <class Real, class Index> struct gwn_prepare_taylor_async_topology_functor {
    gwn_bvh_accessor<Real, Index> bvh{};
    cuda::std::span<Index> internal_parent{};
    cuda::std::span<std::uint8_t> internal_parent_slot{};
    cuda::std::span<std::uint8_t> internal_arity{};
    unsigned int *error_flag = nullptr;

    __device__ inline void gwn_mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ void operator()(std::size_t const node_id) const {
        if (node_id >= bvh.nodes.size() || node_id >= internal_parent.size() ||
            node_id >= internal_parent_slot.size() || node_id >= internal_arity.size()) {
            gwn_mark_error();
            return;
        }

        gwn_bvh4_node_soa<Real, Index> const &node = bvh.nodes[node_id];
        std::uint8_t node_arity = 0;
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;

            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf) {
                gwn_mark_error();
                continue;
            }

            ++node_arity;
            if (kind == gwn_bvh_child_kind::k_internal) {
                Index const child_index = node.child_index[child_slot];
                if (child_index < Index(0) ||
                    static_cast<std::size_t>(child_index) >= bvh.nodes.size()) {
                    gwn_mark_error();
                    continue;
                }
                std::size_t const child_index_u = static_cast<std::size_t>(child_index);
                internal_parent[child_index_u] = static_cast<Index>(node_id);
                internal_parent_slot[child_index_u] = static_cast<std::uint8_t>(child_slot);
                continue;
            }

            Index const leaf_begin = node.child_index[child_slot];
            Index const leaf_count = node.child_count[child_slot];
            if (leaf_begin < Index(0) || leaf_count < Index(0)) {
                gwn_mark_error();
                continue;
            }
            std::size_t const leaf_begin_u = static_cast<std::size_t>(leaf_begin);
            std::size_t const leaf_count_u = static_cast<std::size_t>(leaf_count);
            if (leaf_begin_u > bvh.primitive_indices.size() ||
                leaf_count_u > (bvh.primitive_indices.size() - leaf_begin_u)) {
                gwn_mark_error();
                continue;
            }
        }

        internal_arity[node_id] = node_arity;
        if (node_arity == 0)
            gwn_mark_error();
    }
};

template <int Order, class Real, class Index> struct gwn_build_taylor_async_from_leaves_functor {
    using moment_type = gwn_device_taylor_moment<Order, Real>;

    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_accessor<Real, Index> bvh{};
    cuda::std::span<Index const> internal_parent{};
    cuda::std::span<std::uint8_t const> internal_parent_slot{};
    cuda::std::span<std::uint8_t const> internal_arity{};
    cuda::std::span<unsigned int> internal_arrivals{};
    cuda::std::span<moment_type> pending_child_moments{};
    cuda::std::span<gwn_bvh4_taylor_node_soa<Order, Real>> taylor_nodes{};
    unsigned int *error_flag = nullptr;

    __device__ inline void gwn_mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ inline bool
    gwn_pending_index_is_valid(std::size_t const node_id, int const child_slot) const noexcept {
        if (child_slot < 0 || child_slot >= 4)
            return false;
        if (node_id > (std::numeric_limits<std::size_t>::max() / std::size_t(4)))
            return false;

        std::size_t const pending_index =
            node_id * std::size_t(4) + static_cast<std::size_t>(child_slot);
        return pending_index < pending_child_moments.size();
    }

    __device__ inline std::size_t
    gwn_pending_index(std::size_t const node_id, int const child_slot) const noexcept {
        return node_id * std::size_t(4) + static_cast<std::size_t>(child_slot);
    }

    __device__ bool
    gwn_finalize_node(std::size_t const node_id, moment_type &out_parent_moment) const noexcept {
        if (node_id >= bvh.nodes.size() || node_id >= taylor_nodes.size()) {
            gwn_mark_error();
            return false;
        }

        gwn_bvh4_node_soa<Real, Index> const &node = bvh.nodes[node_id];
        gwn_bvh4_taylor_node_soa<Order, Real> taylor{};
        moment_type parent{};
        bool has_child = false;
        gwn_aabb<Real> merged_bounds{};

        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid) {
                gwn_zero_taylor_child<Order>(taylor, child_slot);
                continue;
            }
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf) {
                gwn_zero_taylor_child<Order>(taylor, child_slot);
                gwn_mark_error();
                continue;
            }
            if (!gwn_pending_index_is_valid(node_id, child_slot)) {
                gwn_zero_taylor_child<Order>(taylor, child_slot);
                gwn_mark_error();
                continue;
            }

            moment_type const child_moment =
                pending_child_moments[gwn_pending_index(node_id, child_slot)];
            gwn_write_taylor_child<Order>(taylor, child_slot, child_moment);

            parent.area += child_moment.area;
            parent.area_p_x += child_moment.area_p_x;
            parent.area_p_y += child_moment.area_p_y;
            parent.area_p_z += child_moment.area_p_z;
            parent.n_x += child_moment.n_x;
            parent.n_y += child_moment.n_y;
            parent.n_z += child_moment.n_z;

            gwn_aabb<Real> const child_bounds{
                node.child_min_x[child_slot], node.child_min_y[child_slot],
                node.child_min_z[child_slot], node.child_max_x[child_slot],
                node.child_max_y[child_slot], node.child_max_z[child_slot],
            };
            if (!has_child) {
                merged_bounds = child_bounds;
                has_child = true;
            } else {
                merged_bounds = gwn_aabb_union(merged_bounds, child_bounds);
            }
        }

        if (has_child) {
            if (parent.area > Real(0)) {
                parent.average_x = parent.area_p_x / parent.area;
                parent.average_y = parent.area_p_y / parent.area;
                parent.average_z = parent.area_p_z / parent.area;
            } else {
                parent.average_x = (merged_bounds.min_x + merged_bounds.max_x) * Real(0.5);
                parent.average_y = (merged_bounds.min_y + merged_bounds.max_y) * Real(0.5);
                parent.average_z = (merged_bounds.min_z + merged_bounds.max_z) * Real(0.5);
            }
            parent.max_p_dist2 = gwn_bounds_max_p_dist2(
                merged_bounds, parent.average_x, parent.average_y, parent.average_z
            );

            if constexpr (Order >= 1) {
                GWN_PRAGMA_UNROLL
                for (int child_slot = 0; child_slot < 4; ++child_slot) {
                    auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
                    if (kind == gwn_bvh_child_kind::k_invalid)
                        continue;
                    if (!gwn_pending_index_is_valid(node_id, child_slot)) {
                        gwn_mark_error();
                        continue;
                    }
                    moment_type const child_moment =
                        pending_child_moments[gwn_pending_index(node_id, child_slot)];
                    Real const dx = child_moment.average_x - parent.average_x;
                    Real const dy = child_moment.average_y - parent.average_y;
                    Real const dz = child_moment.average_z - parent.average_z;

                    parent.nij_xx += child_moment.nij_xx + child_moment.n_x * dx;
                    parent.nij_yy += child_moment.nij_yy + child_moment.n_y * dy;
                    parent.nij_zz += child_moment.nij_zz + child_moment.n_z * dz;
                    parent.nxy += child_moment.nxy + child_moment.n_x * dy;
                    parent.nyx += child_moment.nyx + child_moment.n_y * dx;
                    parent.nyz += child_moment.nyz + child_moment.n_y * dz;
                    parent.nzy += child_moment.nzy + child_moment.n_z * dy;
                    parent.nzx += child_moment.nzx + child_moment.n_z * dx;
                    parent.nxz += child_moment.nxz + child_moment.n_x * dz;
                }
            }
        }

        taylor_nodes[node_id] = taylor;
        out_parent_moment = parent;
        return true;
    }

    __device__ void gwn_propagate_up(
        Index current_parent, std::uint8_t current_slot, moment_type current_moment
    ) const noexcept {
        while (current_parent >= Index(0)) {
            std::size_t const parent_id = static_cast<std::size_t>(current_parent);
            if (parent_id >= bvh.nodes.size() || parent_id >= internal_parent.size() ||
                parent_id >= internal_parent_slot.size() || parent_id >= internal_arity.size() ||
                parent_id >= internal_arrivals.size() || parent_id >= taylor_nodes.size()) {
                gwn_mark_error();
                return;
            }
            if (current_slot >= 4) {
                gwn_mark_error();
                return;
            }
            if (!gwn_pending_index_is_valid(parent_id, static_cast<int>(current_slot))) {
                gwn_mark_error();
                return;
            }

            pending_child_moments[gwn_pending_index(parent_id, static_cast<int>(current_slot))] =
                current_moment;
            __threadfence();

            unsigned int const previous_arrivals =
                atomicAdd(internal_arrivals.data() + parent_id, 1u);
            unsigned int const next_arrivals = previous_arrivals + 1u;
            unsigned int const expected_arrivals =
                static_cast<unsigned int>(internal_arity[parent_id]);
            if (expected_arrivals == 0u || expected_arrivals > 4u) {
                gwn_mark_error();
                return;
            }
            if (next_arrivals < expected_arrivals)
                return;
            if (next_arrivals > expected_arrivals) {
                gwn_mark_error();
                return;
            }

            __threadfence();

            moment_type parent_moment{};
            if (!gwn_finalize_node(parent_id, parent_moment))
                return;

            Index const parent_parent = internal_parent[parent_id];
            if (parent_parent < Index(0))
                return;

            std::uint8_t const parent_parent_slot = internal_parent_slot[parent_id];
            if (parent_parent_slot >= 4) {
                gwn_mark_error();
                return;
            }

            current_parent = parent_parent;
            current_slot = parent_parent_slot;
            current_moment = parent_moment;
        }
    }

    __device__ void operator()(std::size_t const edge_index) const {
        if (edge_index > (std::numeric_limits<std::size_t>::max() / std::size_t(4)))
            return;

        std::size_t const node_id = edge_index >> 2;
        int const child_slot = static_cast<int>(edge_index & std::size_t(3));
        if (node_id >= bvh.nodes.size())
            return;

        gwn_bvh4_node_soa<Real, Index> const &node = bvh.nodes[node_id];
        auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
        if (child_kind != gwn_bvh_child_kind::k_leaf)
            return;

        moment_type const leaf_moment = gwn_compute_leaf_taylor_moment_cached<Order>(
            geometry, bvh, node.child_index[child_slot], node.child_count[child_slot]
        );
        gwn_propagate_up(
            static_cast<Index>(node_id), static_cast<std::uint8_t>(child_slot), leaf_moment
        );
    }
};

template <class Index> struct gwn_validate_taylor_async_convergence_functor {
    cuda::std::span<std::uint8_t const> internal_arity{};
    cuda::std::span<unsigned int const> internal_arrivals{};
    unsigned int *error_flag = nullptr;

    __device__ void operator()(std::size_t const node_id) const {
        if (node_id >= internal_arity.size() || node_id >= internal_arrivals.size()) {
            if (error_flag != nullptr)
                atomicExch(error_flag, 1u);
            return;
        }

        unsigned int const expected_arrivals = static_cast<unsigned int>(internal_arity[node_id]);
        if (expected_arrivals == 0u || expected_arrivals > 4u ||
            internal_arrivals[node_id] != expected_arrivals) {
            if (error_flag != nullptr)
                atomicExch(error_flag, 1u);
        }
    }
};

#ifdef GWN_ENABLE_TAYLOR_LEVELWISE_REFERENCE
template <int Order, class Real, class Index> struct gwn_build_taylor_levelwise_functor {
    using moment_type = gwn_device_taylor_moment<Order, Real>;

    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_accessor<Real, Index> bvh{};
    cuda::std::span<moment_type> node_moments{};
    cuda::std::span<gwn_bvh4_taylor_node_soa<Order, Real>> taylor_nodes{};
    cuda::std::span<Index const> node_ids{};

    __device__ void operator()(std::size_t const local_node_id) const {
        if (local_node_id >= node_ids.size())
            return;
        Index const node_index = node_ids[local_node_id];
        if (node_index < Index(0))
            return;
        std::size_t const node_id = static_cast<std::size_t>(node_index);
        if (node_id >= bvh.nodes.size())
            return;

        gwn_bvh4_node_soa<Real, Index> const &node = bvh.nodes[node_id];
        moment_type child_moments[4]{};
        bool child_valid[4] = {false, false, false, false};
        bool has_child = false;
        gwn_aabb<Real> merged_bounds{};

        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;

            moment_type child{};
            if (kind == gwn_bvh_child_kind::k_internal) {
                Index const child_index = node.child_index[child_slot];
                if (child_index < Index(0) ||
                    static_cast<std::size_t>(child_index) >= bvh.nodes.size()) {
                    continue;
                }
                child = node_moments[static_cast<std::size_t>(child_index)];
            } else if (kind == gwn_bvh_child_kind::k_leaf) {
                child = gwn_compute_leaf_taylor_moment_cached<Order>(
                    geometry, bvh, node.child_index[child_slot], node.child_count[child_slot]
                );
            } else {
                continue;
            }

            child_moments[child_slot] = child;
            child_valid[child_slot] = true;

            gwn_aabb<Real> const bounds{
                node.child_min_x[child_slot], node.child_min_y[child_slot],
                node.child_min_z[child_slot], node.child_max_x[child_slot],
                node.child_max_y[child_slot], node.child_max_z[child_slot],
            };
            if (!has_child) {
                merged_bounds = bounds;
                has_child = true;
            } else {
                merged_bounds = gwn_aabb_union(merged_bounds, bounds);
            }
        }

        moment_type parent{};
        if (has_child) {
            GWN_PRAGMA_UNROLL
            for (int child_slot = 0; child_slot < 4; ++child_slot) {
                if (!child_valid[child_slot])
                    continue;
                moment_type const child = child_moments[child_slot];
                parent.area += child.area;
                parent.area_p_x += child.area_p_x;
                parent.area_p_y += child.area_p_y;
                parent.area_p_z += child.area_p_z;
                parent.n_x += child.n_x;
                parent.n_y += child.n_y;
                parent.n_z += child.n_z;
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
            parent.max_p_dist2 = gwn_bounds_max_p_dist2(
                merged_bounds, parent.average_x, parent.average_y, parent.average_z
            );

            if constexpr (Order >= 1) {
                GWN_PRAGMA_UNROLL
                for (int child_slot = 0; child_slot < 4; ++child_slot) {
                    if (!child_valid[child_slot])
                        continue;
                    moment_type const child = child_moments[child_slot];
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
        }

        gwn_bvh4_taylor_node_soa<Order, Real> taylor{};
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot)
            if (child_valid[child_slot])
                gwn_write_taylor_child<Order>(taylor, child_slot, child_moments[child_slot]);
            else
                gwn_zero_taylor_child<Order>(taylor, child_slot);

        node_moments[node_id] = parent;
        taylor_nodes[node_id] = taylor;
    }
};
#endif

} // namespace detail

/// \brief Build 4-wide LBVH topology and per-node Taylor data (fully GPU async propagation).
///
/// \remark This rebuilds topology first, then computes the requested Taylor order.
/// \remark Previously stored data orders are released before writing the requested one.
template <int Order, class Real, class Index>
gwn_status gwn_build_bvh4_lbvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &topology,
    gwn_bvh_data4_accessor<Real, Index> &data_tree, cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    static_assert(
        Order == 0 || Order == 1,
        "gwn_build_bvh4_lbvh_taylor currently supports Order 0 and Order 1."
    );

    GWN_RETURN_ON_ERROR(gwn_build_bvh4_lbvh(geometry, topology, stream));

    detail::gwn_release_bvh_data_accessor(data_tree, stream);

    if (!topology.has_internal_root())
        return gwn_status::ok();

    using moment_type = detail::gwn_device_taylor_moment<Order, Real>;
    using taylor_node_type = gwn_bvh4_taylor_node_soa<Order, Real>;

    std::size_t const node_count = topology.nodes.size();
    if (node_count == 0)
        return gwn_status::ok();

    if (topology.root_index < Index(0) ||
        static_cast<std::size_t>(topology.root_index) >= node_count)
        return gwn_status::internal_error("BVH root index out of range for Taylor construction.");
    if (node_count > (std::numeric_limits<std::size_t>::max() / std::size_t(4)))
        return gwn_status::internal_error("Taylor async construction node count overflow.");

    std::size_t const pending_count = node_count * std::size_t(4);
    constexpr int k_block_size = detail::k_gwn_default_block_size;

    cuda::std::span<taylor_node_type const> taylor_nodes_device{};
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(taylor_nodes_device, node_count, stream));
    auto cleanup_taylor_nodes = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_span(taylor_nodes_device, stream);
    });

    cuda::std::span<Index const> internal_parent_device{};
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(internal_parent_device, node_count, stream));
    auto cleanup_internal_parent = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_span(internal_parent_device, stream);
    });

    cuda::std::span<std::uint8_t const> internal_parent_slot_device{};
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(internal_parent_slot_device, node_count, stream));
    auto cleanup_internal_parent_slot = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_span(internal_parent_slot_device, stream);
    });

    cuda::std::span<std::uint8_t const> internal_arity_device{};
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(internal_arity_device, node_count, stream));
    auto cleanup_internal_arity = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_span(internal_arity_device, stream);
    });

    cuda::std::span<unsigned int const> internal_arrivals_device{};
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(internal_arrivals_device, node_count, stream));
    auto cleanup_internal_arrivals = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_span(internal_arrivals_device, stream);
    });

    cuda::std::span<moment_type const> pending_child_moments_device{};
    GWN_RETURN_ON_ERROR(
        detail::gwn_allocate_span(pending_child_moments_device, pending_count, stream)
    );
    auto cleanup_pending_child_moments = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_span(pending_child_moments_device, stream);
    });

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        const_cast<taylor_node_type *>(taylor_nodes_device.data()), 0,
        node_count * sizeof(taylor_node_type), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        const_cast<Index *>(internal_parent_device.data()), 0xff, node_count * sizeof(Index), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        const_cast<std::uint8_t *>(internal_parent_slot_device.data()), 0xff,
        node_count * sizeof(std::uint8_t), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        const_cast<std::uint8_t *>(internal_arity_device.data()), 0,
        node_count * sizeof(std::uint8_t), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        const_cast<unsigned int *>(internal_arrivals_device.data()), 0,
        node_count * sizeof(unsigned int), stream
    )));

    void *error_flag_raw = nullptr;
    GWN_RETURN_ON_ERROR(gwn_cuda_malloc(&error_flag_raw, sizeof(unsigned int), stream));
    auto cleanup_error_flag =
        gwn_make_scope_exit([&]() noexcept { (void)gwn_cuda_free(error_flag_raw, stream); });
    unsigned int *error_flag = static_cast<unsigned int *>(error_flag_raw);
    GWN_RETURN_ON_ERROR(
        gwn_cuda_to_status(cudaMemsetAsync(error_flag, 0, sizeof(unsigned int), stream))
    );

    auto const internal_parent = cuda::std::span<Index>(
        const_cast<Index *>(internal_parent_device.data()), internal_parent_device.size()
    );
    auto const internal_parent_slot = cuda::std::span<std::uint8_t>(
        const_cast<std::uint8_t *>(internal_parent_slot_device.data()),
        internal_parent_slot_device.size()
    );
    auto const internal_arity = cuda::std::span<std::uint8_t>(
        const_cast<std::uint8_t *>(internal_arity_device.data()), internal_arity_device.size()
    );
    auto const internal_arrivals = cuda::std::span<unsigned int>(
        const_cast<unsigned int *>(internal_arrivals_device.data()), internal_arrivals_device.size()
    );
    auto const pending_child_moments = cuda::std::span<moment_type>(
        const_cast<moment_type *>(pending_child_moments_device.data()),
        pending_child_moments_device.size()
    );
    auto const taylor_nodes = cuda::std::span<taylor_node_type>(
        const_cast<taylor_node_type *>(taylor_nodes_device.data()), taylor_nodes_device.size()
    );

    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            node_count,
            detail::gwn_prepare_taylor_async_topology_functor<Real, Index>{
                topology, internal_parent, internal_parent_slot, internal_arity, error_flag
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            pending_count,
            detail::gwn_build_taylor_async_from_leaves_functor<Order, Real, Index>{
                geometry, topology,
                cuda::std::span<Index const>(internal_parent.data(), internal_parent.size()),
                cuda::std::span<std::uint8_t const>(
                    internal_parent_slot.data(), internal_parent_slot.size()
                ),
                cuda::std::span<std::uint8_t const>(internal_arity.data(), internal_arity.size()),
                internal_arrivals, pending_child_moments, taylor_nodes, error_flag
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        detail::gwn_launch_linear_kernel<k_block_size>(
            node_count,
            detail::gwn_validate_taylor_async_convergence_functor<Index>{
                cuda::std::span<std::uint8_t const>(internal_arity.data(), internal_arity.size()),
                cuda::std::span<unsigned int const>(
                    internal_arrivals.data(), internal_arrivals.size()
                ),
                error_flag
            },
            stream
        )
    );

    unsigned int host_error_flag = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_error_flag, error_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_error_flag != 0) {
        return gwn_status::internal_error(
            "Taylor async construction failed topology/propagation validation."
        );
    }

    if constexpr (Order == 0)
        data_tree.taylor_order0_nodes = taylor_nodes_device;
    else
        data_tree.taylor_order1_nodes = taylor_nodes_device;
    cleanup_taylor_nodes.release();
    return gwn_status::ok();
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_build_bvh4_lbvh_taylor.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_build_bvh4_lbvh_taylor.");
}

#ifdef GWN_ENABLE_TAYLOR_LEVELWISE_REFERENCE
/// \brief Build 4-wide LBVH topology and Taylor data with host-reconstructed level order.
///
/// \remark This variant computes Taylor nodes from deepest level to root.
template <int Order, class Real, class Index>
gwn_status gwn_build_bvh4_lbvh_taylor_levelwise(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> &topology,
    gwn_bvh_data4_accessor<Real, Index> &data_tree, cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    static_assert(
        Order == 0 || Order == 1,
        "gwn_build_bvh4_lbvh_taylor_levelwise currently supports Order 0 and Order 1."
    );

    GWN_RETURN_ON_ERROR(gwn_build_bvh4_lbvh(geometry, topology, stream));

    detail::gwn_release_bvh_data_accessor(data_tree, stream);

    if (!topology.has_internal_root())
        return gwn_status::ok();

    using moment_type = detail::gwn_device_taylor_moment<Order, Real>;
    using taylor_node_type = gwn_bvh4_taylor_node_soa<Order, Real>;

    std::size_t const node_count = topology.nodes.size();
    if (node_count == 0)
        return gwn_status::ok();
    if (topology.root_index != Index(0))
        return gwn_status::internal_error(
            "Taylor levelwise construction expects root node index to be zero."
        );

    std::vector<std::vector<Index>> level_node_ids{};
    {
        std::vector<gwn_bvh4_node_soa<Real, Index>> host_nodes(node_count);
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
            host_nodes.data(), topology.nodes.data(),
            node_count * sizeof(gwn_bvh4_node_soa<Real, Index>), cudaMemcpyDeviceToHost, stream
        )));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));

        std::vector<int> node_depth(node_count, -1);
        std::vector<Index> queue{};
        queue.reserve(node_count);
        queue.push_back(topology.root_index);
        node_depth[static_cast<std::size_t>(topology.root_index)] = 0;

        std::size_t queue_head = 0;
        while (queue_head < queue.size()) {
            Index const node_index = queue[queue_head++];
            if (node_index < Index(0) || static_cast<std::size_t>(node_index) >= node_count) {
                return gwn_status::internal_error(
                    "Levelwise Taylor traversal encountered out-of-range node index."
                );
            }

            int const depth = node_depth[static_cast<std::size_t>(node_index)];
            if (depth < 0)
                continue;

            if (level_node_ids.size() <= static_cast<std::size_t>(depth))
                level_node_ids.resize(static_cast<std::size_t>(depth) + 1);
            level_node_ids[static_cast<std::size_t>(depth)].push_back(node_index);

            gwn_bvh4_node_soa<Real, Index> const &node =
                host_nodes[static_cast<std::size_t>(node_index)];
            for (int child_slot = 0; child_slot < 4; ++child_slot) {
                if (static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]) !=
                    gwn_bvh_child_kind::k_internal) {
                    continue;
                }

                Index const child_index = node.child_index[child_slot];
                if (child_index < Index(0) || static_cast<std::size_t>(child_index) >= node_count) {
                    return gwn_status::internal_error(
                        "Levelwise Taylor traversal encountered out-of-range child node index."
                    );
                }

                if (node_depth[static_cast<std::size_t>(child_index)] >= 0)
                    continue;

                node_depth[static_cast<std::size_t>(child_index)] = depth + 1;
                queue.push_back(child_index);
            }
        }

        if (level_node_ids.empty())
            return gwn_status::ok();
    }

    std::size_t counted_nodes = 0;
    for (auto const &level_nodes : level_node_ids)
        counted_nodes += level_nodes.size();
    if (counted_nodes != node_count)
        return gwn_status::internal_error("Levelwise Taylor node-count reconstruction mismatch.");

    for (std::size_t level = 0; level < level_node_ids.size(); ++level) {
        std::vector<Index> &level_nodes = level_node_ids[level];
        if (level_nodes.empty())
            continue;
        std::sort(level_nodes.begin(), level_nodes.end());
    }

    cuda::std::span<taylor_node_type const> taylor_nodes_device{};
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(taylor_nodes_device, node_count, stream));
    auto cleanup_taylor_nodes = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_span(taylor_nodes_device, stream);
    });

    cuda::std::span<moment_type const> node_moments_device{};
    GWN_RETURN_ON_ERROR(detail::gwn_allocate_span(node_moments_device, node_count, stream));
    auto cleanup_node_moments = gwn_make_scope_exit([&]() noexcept {
        detail::gwn_release_bvh_span(node_moments_device, stream);
    });

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        const_cast<taylor_node_type *>(taylor_nodes_device.data()), 0,
        node_count * sizeof(taylor_node_type), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        const_cast<moment_type *>(node_moments_device.data()), 0, node_count * sizeof(moment_type),
        stream
    )));

    auto const node_moments = cuda::std::span<moment_type>(
        const_cast<moment_type *>(node_moments_device.data()), node_count
    );
    auto const taylor_nodes = cuda::std::span<taylor_node_type>(
        const_cast<taylor_node_type *>(taylor_nodes_device.data()), node_count
    );

    constexpr int k_block_size = detail::k_gwn_default_block_size;
    for (std::size_t level = level_node_ids.size(); level-- > 0;) {
        std::vector<Index> const &level_nodes = level_node_ids[level];
        if (level_nodes.empty())
            continue;

        gwn_device_array<Index> level_node_ids_device{};
        GWN_RETURN_ON_ERROR(level_node_ids_device.copy_from_host(
            cuda::std::span<Index const>(level_nodes.data(), level_nodes.size()), stream
        ));
        auto const level_node_ids_span = cuda::std::span<Index const>(
            level_node_ids_device.data(), level_node_ids_device.size()
        );

        GWN_RETURN_ON_ERROR(
            detail::gwn_launch_linear_kernel<k_block_size>(
                level_node_ids_span.size(),
                detail::gwn_build_taylor_levelwise_functor<Order, Real, Index>{
                    geometry, topology, node_moments, taylor_nodes, level_node_ids_span
                },
                stream
            )
        );
    }

    if constexpr (Order == 0)
        data_tree.taylor_order0_nodes = taylor_nodes_device;
    else
        data_tree.taylor_order1_nodes = taylor_nodes_device;
    cleanup_taylor_nodes.release();
    return gwn_status::ok();
} catch (std::exception const &) {
    return gwn_status::internal_error(
        "Unhandled std::exception in gwn_build_bvh4_lbvh_taylor_levelwise."
    );
} catch (...) {
    return gwn_status::internal_error(
        "Unhandled unknown exception in gwn_build_bvh4_lbvh_taylor_levelwise."
    );
}
#endif

} // namespace gwn
