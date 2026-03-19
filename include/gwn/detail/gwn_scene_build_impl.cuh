#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <type_traits>
#include <utility>

#include "gwn_bvh_refit_async.cuh"
#include "gwn_bvh_status_helpers.cuh"
#include "gwn_bvh_topology_build_common.cuh"
#include "gwn_bvh_topology_build_impl.cuh"

namespace gwn {
namespace detail {

inline constexpr std::string_view k_gwn_scene_phase_build = "scene.build";
inline constexpr std::string_view k_gwn_scene_phase_preprocess = "scene.build.preprocess";
inline constexpr std::string_view k_gwn_scene_phase_refit_aabb = "scene.build.refit.aabb";
inline constexpr std::string_view k_gwn_scene_phase_refit = "scene.refit";
inline constexpr std::string_view k_gwn_scene_phase_update_blas_table = "scene.update_blas_table";

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ inline gwn_aabb<Real> gwn_scene_zero_aabb() noexcept {
    return gwn_aabb<Real>{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ inline gwn_aabb<Real>
gwn_scene_union_aabb(gwn_aabb<Real> const &lhs, gwn_aabb<Real> const &rhs) noexcept {
    return gwn_aabb_union(lhs, rhs);
}

template <class BlasT>
__device__ inline bool gwn_scene_compute_blas_root_aabb_device(
    BlasT const &blas, gwn_aabb<typename BlasT::real_type> &result
) noexcept {
    using Real = typename BlasT::real_type;
    using Index = typename BlasT::index_type;
    constexpr int Width = std::remove_cvref_t<BlasT>::k_width;

    auto const &topology = blas.topology;
    if (!topology.is_valid())
        return false;

    if (topology.has_internal_root()) {
        if (!blas.aabb.is_valid_for(topology) ||
            !gwn_index_in_bounds(topology.root_index, topology.nodes.size())) {
            return false;
        }
        auto const root_id = static_cast<std::size_t>(topology.root_index);
        gwn_bvh_topology_node_soa<Width, Index> const &root_node = topology.nodes[root_id];
        gwn_bvh_aabb_node_soa<Width, Real> const &root_aabb = blas.aabb.nodes[root_id];

        bool has_bounds = false;
        result = gwn_scene_zero_aabb<Real>();
        GWN_PRAGMA_UNROLL
        for (int slot = 0; slot < Width; ++slot) {
            auto const child_kind = static_cast<gwn_bvh_child_kind>(root_node.child_kind[slot]);
            if (child_kind == gwn_bvh_child_kind::k_invalid)
                continue;
            gwn_aabb<Real> const child_bounds{
                root_aabb.child_min_x[slot], root_aabb.child_min_y[slot],
                root_aabb.child_min_z[slot], root_aabb.child_max_x[slot],
                root_aabb.child_max_y[slot], root_aabb.child_max_z[slot],
            };
            result = has_bounds ? gwn_scene_union_aabb(result, child_bounds) : child_bounds;
            has_bounds = true;
        }
        return has_bounds;
    }

    if (!topology.has_leaf_root())
        return false;

    auto const &geometry = blas.geometry;
    if (!geometry.is_valid())
        return false;

    std::size_t const begin = static_cast<std::size_t>(topology.root_index);
    std::size_t const count = static_cast<std::size_t>(topology.root_count);
    if (begin > topology.primitive_indices.size() ||
        count > (topology.primitive_indices.size() - begin))
        return false;

    bool has_bounds = false;
    result = gwn_scene_zero_aabb<Real>();
    for (std::size_t primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
        Index const primitive_id = topology.primitive_indices[begin + primitive_offset];
        gwn_aabb<Real> primitive_bounds{};
        if (!gwn_compute_triangle_aabb(geometry, primitive_id, primitive_bounds))
            return false;
        result = has_bounds ? gwn_scene_union_aabb(result, primitive_bounds) : primitive_bounds;
        has_bounds = true;
    }
    return has_bounds;
}

template <gwn_real_type Real> struct gwn_scene_preprocess_axis_arrays {
    gwn_device_array<Real> min_x{};
    gwn_device_array<Real> min_y{};
    gwn_device_array<Real> min_z{};
    gwn_device_array<Real> max_x{};
    gwn_device_array<Real> max_y{};
    gwn_device_array<Real> max_z{};
};

template <gwn_real_type Real, gwn_index_type Index, class BlasT>
struct gwn_compute_instance_aabbs_functor {
    cuda::std::span<BlasT const> blas_table{};
    cuda::std::span<gwn_instance_record<Real, Index> const> instances{};
    cuda::std::span<gwn_aabb<Real>> primitive_aabbs{};
    cuda::std::span<Real> min_x{};
    cuda::std::span<Real> min_y{};
    cuda::std::span<Real> min_z{};
    cuda::std::span<Real> max_x{};
    cuda::std::span<Real> max_y{};
    cuda::std::span<Real> max_z{};
    unsigned int *error_flag = nullptr;

    __device__ inline void mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ void operator()(std::size_t const instance_id) const {
        if (instance_id >= instances.size() || instance_id >= primitive_aabbs.size() ||
            instance_id >= min_x.size() || instance_id >= min_y.size() ||
            instance_id >= min_z.size() || instance_id >= max_x.size() ||
            instance_id >= max_y.size() || instance_id >= max_z.size()) {
            mark_error();
            return;
        }

        gwn_instance_record<Real, Index> const instance = instances[instance_id];
        if (!gwn_index_in_bounds(instance.blas_index, blas_table.size())) {
            mark_error();
            return;
        }

        gwn_aabb<Real> local_bounds{};
        if (!gwn_scene_compute_blas_root_aabb_device(
                blas_table[static_cast<std::size_t>(instance.blas_index)], local_bounds
            )) {
            mark_error();
            return;
        }

        gwn_aabb<Real> const world_bounds = instance.transform.transform_aabb(local_bounds);
        primitive_aabbs[instance_id] = world_bounds;
        min_x[instance_id] = world_bounds.min_x;
        min_y[instance_id] = world_bounds.min_y;
        min_z[instance_id] = world_bounds.min_z;
        max_x[instance_id] = world_bounds.max_x;
        max_y[instance_id] = world_bounds.max_y;
        max_z[instance_id] = world_bounds.max_z;
    }
};

template <gwn_real_type Real, gwn_index_type Index, class BlasT>
struct gwn_compute_instance_world_aabbs_functor {
    cuda::std::span<BlasT const> blas_table{};
    cuda::std::span<gwn_instance_record<Real, Index> const> instances{};
    cuda::std::span<gwn_aabb<Real>> instance_aabbs{};
    unsigned int *error_flag = nullptr;

    __device__ inline void mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ void operator()(std::size_t const instance_id) const {
        if (instance_id >= instances.size() || instance_id >= instance_aabbs.size()) {
            mark_error();
            return;
        }

        gwn_instance_record<Real, Index> const instance = instances[instance_id];
        if (!gwn_index_in_bounds(instance.blas_index, blas_table.size())) {
            mark_error();
            return;
        }

        gwn_aabb<Real> local_bounds{};
        if (!gwn_scene_compute_blas_root_aabb_device(
                blas_table[static_cast<std::size_t>(instance.blas_index)], local_bounds
            )) {
            mark_error();
            return;
        }

        instance_aabbs[instance_id] = instance.transform.transform_aabb(local_bounds);
    }
};

template <class MortonCode, gwn_real_type Real, gwn_index_type Index>
struct gwn_compute_instance_morton_functor {
    Real scene_min_x{};
    Real scene_min_y{};
    Real scene_min_z{};
    Real scene_inv_x{};
    Real scene_inv_y{};
    Real scene_inv_z{};
    cuda::std::span<gwn_aabb<Real> const> primitive_aabbs{};
    cuda::std::span<MortonCode> morton_codes{};
    cuda::std::span<Index> primitive_indices{};

    __device__ void operator()(std::size_t const primitive_id) const {
        primitive_indices[primitive_id] = static_cast<Index>(primitive_id);
        gwn_aabb<Real> const bounds = primitive_aabbs[primitive_id];
        Real const center_x = (bounds.min_x + bounds.max_x) * Real(0.5);
        Real const center_y = (bounds.min_y + bounds.max_y) * Real(0.5);
        Real const center_z = (bounds.min_z + bounds.max_z) * Real(0.5);
        morton_codes[primitive_id] = gwn_encode_morton<MortonCode>(
            (center_x - scene_min_x) * scene_inv_x, (center_y - scene_min_y) * scene_inv_y,
            (center_z - scene_min_z) * scene_inv_z
        );
    }
};

template <class MortonCode, gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_build_preprocess(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_topology_build_preprocess<Real, Index, MortonCode> &preprocess, cudaStream_t const stream
) noexcept {
    constexpr int k_block_size = k_gwn_default_block_size;
    std::size_t const primitive_count = instances.size();

    gwn_scene_preprocess_axis_arrays<Real> axis_arrays{};
    GWN_RETURN_ON_ERROR(preprocess.primitive_aabbs.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(axis_arrays.min_x.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(axis_arrays.min_y.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(axis_arrays.min_z.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(axis_arrays.max_x.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(axis_arrays.max_y.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(axis_arrays.max_z.resize(primitive_count, stream));

    gwn_device_array<unsigned int> error_flag_device{};
    GWN_RETURN_ON_ERROR(error_flag_device.resize(1, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(error_flag_device.data(), 0, sizeof(unsigned int), stream)
    ));

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            gwn_compute_instance_aabbs_functor<Real, Index, BlasT>{
                blas_table,
                instances,
                preprocess.primitive_aabbs.span(),
                axis_arrays.min_x.span(),
                axis_arrays.min_y.span(),
                axis_arrays.min_z.span(),
                axis_arrays.max_x.span(),
                axis_arrays.max_y.span(),
                axis_arrays.max_z.span(),
                error_flag_device.data(),
            },
            stream
        )
    );

    unsigned int host_error_flag = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_error_flag, error_flag_device.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_error_flag != 0) {
        return gwn_bvh_internal_error(
            k_gwn_scene_phase_preprocess, "Failed to compute BLAS or instance bounds on device."
        );
    }

    gwn_device_array<Real> axis_min{};
    gwn_device_array<Real> axis_max{};
    gwn_device_array<std::uint8_t> reduce_temp{};
    GWN_RETURN_ON_ERROR(axis_min.resize(1, stream));
    GWN_RETURN_ON_ERROR(axis_max.resize(1, stream));

    Real scene_min_x{};
    Real scene_min_y{};
    Real scene_min_z{};
    Real scene_max_x{};
    Real scene_max_y{};
    Real scene_max_z{};
    Real ignored{};
    GWN_RETURN_ON_ERROR(
        gwn_reduce_minmax<Real>(
            cuda::std::span<Real const>(axis_arrays.min_x.data(), axis_arrays.min_x.size()),
            axis_min, axis_max, reduce_temp, scene_min_x, ignored, stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_reduce_minmax<Real>(
            cuda::std::span<Real const>(axis_arrays.min_y.data(), axis_arrays.min_y.size()),
            axis_min, axis_max, reduce_temp, scene_min_y, ignored, stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_reduce_minmax<Real>(
            cuda::std::span<Real const>(axis_arrays.min_z.data(), axis_arrays.min_z.size()),
            axis_min, axis_max, reduce_temp, scene_min_z, ignored, stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_reduce_minmax<Real>(
            cuda::std::span<Real const>(axis_arrays.max_x.data(), axis_arrays.max_x.size()),
            axis_min, axis_max, reduce_temp, ignored, scene_max_x, stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_reduce_minmax<Real>(
            cuda::std::span<Real const>(axis_arrays.max_y.data(), axis_arrays.max_y.size()),
            axis_min, axis_max, reduce_temp, ignored, scene_max_y, stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_reduce_minmax<Real>(
            cuda::std::span<Real const>(axis_arrays.max_z.data(), axis_arrays.max_z.size()),
            axis_min, axis_max, reduce_temp, ignored, scene_max_z, stream
        )
    );

    auto const safe_inv = [](Real const lo, Real const hi) noexcept {
        return (hi > lo) ? Real(1) / (hi - lo) : Real(1);
    };
    Real const scene_inv_x = safe_inv(scene_min_x, scene_max_x);
    Real const scene_inv_y = safe_inv(scene_min_y, scene_max_y);
    Real const scene_inv_z = safe_inv(scene_min_z, scene_max_z);

    gwn_device_array<MortonCode> morton_codes{};
    gwn_device_array<Index> primitive_indices{};
    GWN_RETURN_ON_ERROR(morton_codes.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(primitive_indices.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            gwn_compute_instance_morton_functor<MortonCode, Real, Index>{
                scene_min_x,
                scene_min_y,
                scene_min_z,
                scene_inv_x,
                scene_inv_y,
                scene_inv_z,
                cuda::std::span<gwn_aabb<Real> const>(
                    preprocess.primitive_aabbs.data(), preprocess.primitive_aabbs.size()
                ),
                morton_codes.span(),
                primitive_indices.span(),
            },
            stream
        )
    );

    gwn_device_array<std::uint8_t> radix_sort_temp{};
    GWN_RETURN_ON_ERROR(preprocess.sorted_morton_codes.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(preprocess.sorted_primitive_indices.resize(primitive_count, stream));
    std::size_t radix_sort_temp_bytes = 0;
    int const radix_sort_end_bit = static_cast<int>(sizeof(MortonCode) * 8);
    auto const radix_item_count = static_cast<std::uint64_t>(primitive_count);
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            nullptr, radix_sort_temp_bytes, morton_codes.data(),
            preprocess.sorted_morton_codes.data(), primitive_indices.data(),
            preprocess.sorted_primitive_indices.data(), radix_item_count, 0, radix_sort_end_bit,
            stream
        )
    ));
    GWN_RETURN_ON_ERROR(radix_sort_temp.resize(radix_sort_temp_bytes, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cub::DeviceRadixSort::SortPairs(
            radix_sort_temp.data(), radix_sort_temp_bytes, morton_codes.data(),
            preprocess.sorted_morton_codes.data(), primitive_indices.data(),
            preprocess.sorted_primitive_indices.data(), radix_item_count, 0, radix_sort_end_bit,
            stream
        )
    ));

    GWN_RETURN_ON_ERROR(preprocess.sorted_primitive_aabbs.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            gwn_gather_sorted_aabbs_functor<Real, Index>{
                preprocess.primitive_aabbs.span(),
                cuda::std::span<Index const>(
                    preprocess.sorted_primitive_indices.data(),
                    preprocess.sorted_primitive_indices.size()
                ),
                preprocess.sorted_primitive_aabbs.span(),
            },
            stream
        )
    );
    return gwn_status::ok();
}

template <int Width, gwn_real_type Real, gwn_index_type Index> struct gwn_scene_aabb_refit_traits {
    using payload_type = gwn_aabb<Real>;

    struct output_context {
        cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real>> nodes{};
        cuda::std::span<gwn_aabb<Real> const> sorted_primitive_aabbs{};
    };

    static constexpr char const *k_error_name = "Scene IAS AABB refit failed.";
    static constexpr std::string_view k_phase = k_gwn_scene_phase_refit_aabb;

    __device__ static bool make_leaf_payload(
        gwn_bvh_topology_accessor<Width, Real, Index> const &topology, std::size_t const node_id,
        int const child_slot, output_context const &context, payload_type &payload
    ) noexcept {
        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        Index const begin = node.child_index[child_slot];
        Index const count = node.child_count[child_slot];

        bool has_bounds = false;
        payload = {};
        for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
            Index const sorted_slot = begin + primitive_offset;
            auto const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
            if (sorted_slot_u >= topology.primitive_indices.size() ||
                sorted_slot_u >= context.sorted_primitive_aabbs.size()) {
                return false;
            }

            gwn_aabb<Real> const primitive_bounds = context.sorted_primitive_aabbs[sorted_slot_u];
            payload =
                has_bounds ? gwn_scene_union_aabb(payload, primitive_bounds) : primitive_bounds;
            has_bounds = true;
        }

        if (!has_bounds)
            payload = gwn_scene_zero_aabb<Real>();
        return true;
    }

    __device__ static void combine(payload_type &dst, payload_type const &src) noexcept {
        dst = gwn_scene_union_aabb(dst, src);
    }

    __device__ static void write_invalid(
        output_context const &context, std::size_t const node_id, int const child_slot
    ) noexcept {
        context.nodes[node_id].child_min_x[child_slot] = Real(0);
        context.nodes[node_id].child_min_y[child_slot] = Real(0);
        context.nodes[node_id].child_min_z[child_slot] = Real(0);
        context.nodes[node_id].child_max_x[child_slot] = Real(0);
        context.nodes[node_id].child_max_y[child_slot] = Real(0);
        context.nodes[node_id].child_max_z[child_slot] = Real(0);
    }

    __device__ static void write_valid(
        output_context const &context, std::size_t const node_id, int const child_slot,
        payload_type const &payload
    ) noexcept {
        context.nodes[node_id].child_min_x[child_slot] = payload.min_x;
        context.nodes[node_id].child_min_y[child_slot] = payload.min_y;
        context.nodes[node_id].child_min_z[child_slot] = payload.min_z;
        context.nodes[node_id].child_max_x[child_slot] = payload.max_x;
        context.nodes[node_id].child_max_y[child_slot] = payload.max_y;
        context.nodes[node_id].child_max_z[child_slot] = payload.max_z;
    }
};

template <class Traits, int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_scene_refit_from_leaves_functor {
    using payload_type = typename Traits::payload_type;
    using output_context = typename Traits::output_context;

    gwn_bvh_topology_accessor<Width, Real, Index> topology{};
    gwn_refit_topology_arrays<Width, Index> arrays{};
    gwn_refit_pending_buffer<Width, Index, payload_type> pending{};
    output_context context{};
    unsigned int *error_flag = nullptr;

    __device__ inline void mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ bool
    finalize_parent(std::size_t const node_id, payload_type &parent_payload) const noexcept {
        if (node_id >= topology.nodes.size())
            return false;

        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        bool has_child = false;
        bool initialized = false;
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf) {
                mark_error();
                continue;
            }
            if (!pending.is_valid(node_id, child_slot)) {
                mark_error();
                continue;
            }

            payload_type const child_payload = pending.pending[pending.index(node_id, child_slot)];
            if (!initialized) {
                parent_payload = child_payload;
                initialized = true;
            } else {
                Traits::combine(parent_payload, child_payload);
            }
            has_child = true;
        }
        return has_child;
    }

    __device__ void propagate_up(
        Index current_parent, std::uint8_t current_slot, payload_type current_payload
    ) const noexcept {
        while (gwn_is_valid_index(current_parent)) {
            auto const parent_id = static_cast<std::size_t>(current_parent);
            if (parent_id >= topology.nodes.size() || parent_id >= arrays.internal_parent.size() ||
                parent_id >= arrays.internal_parent_slot.size() ||
                parent_id >= arrays.internal_arity.size() ||
                parent_id >= arrays.internal_arrivals.size()) {
                mark_error();
                return;
            }
            if (current_slot >= Width ||
                !pending.is_valid(parent_id, static_cast<int>(current_slot))) {
                mark_error();
                return;
            }

            pending.pending[pending.index(parent_id, static_cast<int>(current_slot))] =
                current_payload;
            __threadfence();

            unsigned int const previous_arrivals =
                atomicAdd(arrays.internal_arrivals.data() + parent_id, 1u);
            unsigned int const next_arrivals = previous_arrivals + 1u;
            unsigned int const expected_arrivals =
                static_cast<unsigned int>(arrays.internal_arity[parent_id]);
            if (expected_arrivals == 0u || expected_arrivals > static_cast<unsigned int>(Width) ||
                next_arrivals > expected_arrivals) {
                mark_error();
                return;
            }
            if (next_arrivals < expected_arrivals)
                return;

            __threadfence();

            payload_type parent_payload{};
            if (!finalize_parent(parent_id, parent_payload)) {
                mark_error();
                return;
            }

            Index const parent_parent = arrays.internal_parent[parent_id];
            if (gwn_is_invalid_index(parent_parent))
                return;

            std::uint8_t const parent_parent_slot = arrays.internal_parent_slot[parent_id];
            if (parent_parent_slot >= Width) {
                mark_error();
                return;
            }

            current_parent = parent_parent;
            current_slot = parent_parent_slot;
            current_payload = parent_payload;
        }
    }

    __device__ void operator()(std::size_t const edge_index) const {
        if (edge_index > (std::numeric_limits<std::size_t>::max() / std::size_t(Width)))
            return;

        std::size_t const node_id = edge_index / std::size_t(Width);
        int const child_slot = static_cast<int>(edge_index % std::size_t(Width));
        if (node_id >= topology.nodes.size())
            return;

        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        if (static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]) !=
            gwn_bvh_child_kind::k_leaf) {
            return;
        }

        payload_type leaf_payload{};
        if (!Traits::make_leaf_payload(topology, node_id, child_slot, context, leaf_payload)) {
            mark_error();
            return;
        }

        propagate_up(
            static_cast<Index>(node_id), static_cast<std::uint8_t>(child_slot), leaf_payload
        );
    }
};

template <int Width, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_scene_build_ias_aabb(
    gwn_bvh_topology_accessor<Width, Real, Index> const &topology,
    cuda::std::span<gwn_aabb<Real> const> const sorted_primitive_aabbs,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree, cudaStream_t const stream
) noexcept {
    if (!topology.has_internal_root())
        return gwn_status::ok();

    constexpr int k_block_size = k_gwn_default_block_size;
    std::size_t const node_count = topology.nodes.size();
    if (node_count == 0)
        return gwn_status::ok();
    if (node_count > (std::numeric_limits<std::size_t>::max() / std::size_t(Width)))
        return gwn_bvh_internal_error(k_gwn_scene_phase_refit_aabb, "IAS node count overflow.");

    using traits = gwn_scene_aabb_refit_traits<Width, Real, Index>;
    using payload_type = typename traits::payload_type;
    std::size_t const pending_count = node_count * std::size_t(Width);

    gwn_device_array<Index> internal_parent{};
    gwn_device_array<std::uint8_t> internal_parent_slot{};
    gwn_device_array<std::uint8_t> internal_arity{};
    gwn_device_array<unsigned int> internal_arrivals{};
    gwn_device_array<payload_type> pending_payloads{};
    gwn_device_array<unsigned int> error_flag_device{};

    GWN_RETURN_ON_ERROR(internal_parent.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(internal_parent_slot.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(internal_arity.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(internal_arrivals.resize(node_count, stream));
    GWN_RETURN_ON_ERROR(pending_payloads.resize(pending_count, stream));
    GWN_RETURN_ON_ERROR(error_flag_device.resize(1, stream));

    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(internal_parent.data(), 0xff, node_count * sizeof(Index), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
        internal_parent_slot.data(), 0xff, node_count * sizeof(std::uint8_t), stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(internal_arity.data(), 0, node_count * sizeof(std::uint8_t), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(internal_arrivals.data(), 0, node_count * sizeof(unsigned int), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(pending_payloads.data(), 0, pending_count * sizeof(payload_type), stream)
    ));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(error_flag_device.data(), 0, sizeof(unsigned int), stream)
    ));

    gwn_refit_topology_arrays<Width, Index> const arrays{
        internal_parent.span(), internal_parent_slot.span(), internal_arity.span(),
        internal_arrivals.span()
    };
    gwn_refit_pending_buffer<Width, Index, payload_type> const pending{pending_payloads.span()};
    typename traits::output_context const output_context{
        aabb_tree.nodes,
        sorted_primitive_aabbs,
    };

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            node_count,
            gwn_prepare_refit_topology_functor<Width, Real, Index>{
                topology,
                arrays,
                error_flag_device.data(),
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            pending_count,
            gwn_scene_refit_from_leaves_functor<traits, Width, Real, Index>{
                topology,
                arrays,
                pending,
                output_context,
                error_flag_device.data(),
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            node_count,
            gwn_validate_refit_convergence_functor<Width, Index>{
                cuda::std::span<std::uint8_t const>(
                    arrays.internal_arity.data(), arrays.internal_arity.size()
                ),
                cuda::std::span<unsigned int const>(
                    arrays.internal_arrivals.data(), arrays.internal_arrivals.size()
                ),
                error_flag_device.data(),
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            node_count,
            gwn_finalize_refit_functor<traits, Width, Real, Index>{
                topology,
                pending,
                output_context,
                error_flag_device.data(),
            },
            stream
        )
    );

    unsigned int host_error_flag = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_error_flag, error_flag_device.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_error_flag != 0)
        return gwn_bvh_internal_error(k_gwn_scene_phase_refit_aabb, traits::k_error_name);
    return gwn_status::ok();
}

template <int Width, gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_refit_ias_aabb_from_owned_data(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_bvh_topology_accessor<Width, Real, Index> const &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree, cudaStream_t const stream
) noexcept {
    if (!topology.has_internal_root())
        return gwn_status::ok();
    if (topology.primitive_indices.size() != instances.size()) {
        return gwn_bvh_internal_error(
            k_gwn_scene_phase_refit_aabb, "IAS primitive order does not match scene instance count."
        );
    }

    constexpr int k_block_size = k_gwn_default_block_size;
    std::size_t const primitive_count = instances.size();
    gwn_device_array<gwn_aabb<Real>> instance_aabbs{};
    gwn_device_array<gwn_aabb<Real>> sorted_instance_aabbs{};
    gwn_device_array<unsigned int> error_flag_device{};
    GWN_RETURN_ON_ERROR(instance_aabbs.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(sorted_instance_aabbs.resize(primitive_count, stream));
    GWN_RETURN_ON_ERROR(error_flag_device.resize(1, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(error_flag_device.data(), 0, sizeof(unsigned int), stream)
    ));

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            gwn_compute_instance_world_aabbs_functor<Real, Index, BlasT>{
                blas_table,
                instances,
                instance_aabbs.span(),
                error_flag_device.data(),
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            primitive_count,
            gwn_gather_sorted_aabbs_functor<Real, Index>{
                cuda::std::span<gwn_aabb<Real> const>(instance_aabbs.data(), instance_aabbs.size()),
                cuda::std::span<Index const>(
                    topology.primitive_indices.data(), topology.primitive_indices.size()
                ),
                sorted_instance_aabbs.span(),
            },
            stream
        )
    );

    unsigned int host_error_flag = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_error_flag, error_flag_device.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_error_flag != 0) {
        return gwn_bvh_internal_error(
            k_gwn_scene_phase_refit_aabb,
            "Failed to recompute instance AABBs from scene-owned BLAS data."
        );
    }

    return gwn_scene_build_ias_aabb<Width, Real, Index>(
        topology,
        cuda::std::span<gwn_aabb<Real> const>(
            sorted_instance_aabbs.data(), sorted_instance_aabbs.size()
        ),
        aabb_tree, stream
    );
}

template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_scene_validate_blas_coverage_functor {
    cuda::std::span<gwn_instance_record<Real, Index> const> instances{};
    std::size_t blas_count = 0;
    unsigned int *error_flag = nullptr;

    __device__ void operator()(std::size_t const instance_id) const {
        if (instance_id >= instances.size()) {
            if (error_flag != nullptr)
                atomicExch(error_flag, 1u);
            return;
        }
        if (!gwn_index_in_bounds(instances[instance_id].blas_index, blas_count) &&
            error_flag != nullptr) {
            atomicExch(error_flag, 1u);
        }
    }
};

template <gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_validate_initialized(
    gwn_device_array<BlasT> const &blas_table,
    gwn_device_array<gwn_instance_record<Real, Index>> const &instances,
    gwn_bvh_topology_accessor<BlasT::k_width, Real, Index> const &topology,
    gwn_bvh_aabb_accessor<BlasT::k_width, Real, Index> const &aabb, std::string_view const phase
) noexcept {
    if (blas_table.empty() || instances.empty() || !topology.is_valid() ||
        !aabb.is_valid_for(topology)) {
        return gwn_bvh_invalid_argument(
            phase, "Scene must be built before refit or BLAS table updates."
        );
    }
    return gwn_status::ok();
}

template <gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_validate_refit_inputs(
    cuda::std::span<gwn_instance_record<Real, Index> const> const updated_instances,
    std::size_t const current_instance_count, std::size_t const current_blas_count
) noexcept {
    if (!gwn_span_has_storage(updated_instances)) {
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_refit, "Updated instance span has null storage."
        );
    }
    if (updated_instances.size() != current_instance_count) {
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_refit, "Updated instance count must match the built scene."
        );
    }
    for (std::size_t i = 0; i < updated_instances.size(); ++i) {
        if (!gwn_index_in_bounds(updated_instances[i].blas_index, current_blas_count)) {
            return gwn_bvh_invalid_argument(
                k_gwn_scene_phase_refit,
                std::format("Updated instance {} has an out-of-range BLAS index.", i)
            );
        }
    }
    return gwn_status::ok();
}

template <gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_validate_blas_table_update_inputs(
    cuda::std::span<BlasT const> const updated_blas_table
) noexcept {
    if (!gwn_span_has_storage(updated_blas_table)) {
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_update_blas_table, "Updated BLAS table span has null storage."
        );
    }
    if (updated_blas_table.empty()) {
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_update_blas_table, "Updated BLAS table must not be empty."
        );
    }
    for (std::size_t i = 0; i < updated_blas_table.size(); ++i) {
        if (!updated_blas_table[i].is_valid()) {
            return gwn_bvh_invalid_argument(
                k_gwn_scene_phase_update_blas_table,
                std::format("Updated BLAS table entry {} is invalid.", i)
            );
        }
    }
    return gwn_status::ok();
}

template <int Width, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_scene_validate_instance_blas_coverage_device(
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    std::size_t const blas_count, cudaStream_t const stream
) noexcept {
    constexpr int k_block_size = k_gwn_default_block_size;
    gwn_device_array<unsigned int> error_flag_device{};
    GWN_RETURN_ON_ERROR(error_flag_device.resize(1, stream));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(
        cudaMemsetAsync(error_flag_device.data(), 0, sizeof(unsigned int), stream)
    ));
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            instances.size(),
            gwn_scene_validate_blas_coverage_functor<Width, Real, Index>{
                instances,
                blas_count,
                error_flag_device.data(),
            },
            stream
        )
    );

    unsigned int host_error_flag = 0;
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
        &host_error_flag, error_flag_device.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream
    )));
    GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
    if (host_error_flag != 0) {
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_update_blas_table,
            "Updated BLAS table does not cover all referenced scene BLAS indices."
        );
    }
    return gwn_status::ok();
}

template <int Width, gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_build_staged_ias_aabb(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_bvh_topology_accessor<Width, Real, Index> const &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &staging_aabb, cudaStream_t const stream
) noexcept {
    auto const release_aabb = [](gwn_bvh_aabb_accessor<Width, Real, Index> &tree,
                                 cudaStream_t const stream_to_release) noexcept {
        gwn_release_bvh_aabb_tree_accessor(tree, stream_to_release);
    };

    auto const build_aabb =
        [&](gwn_bvh_aabb_accessor<Width, Real, Index> &staging_accessor) -> gwn_status {
        if (!topology.has_internal_root())
            return gwn_status::ok();

        std::size_t const node_count = topology.nodes.size();
        GWN_RETURN_ON_ERROR(gwn_allocate_span(staging_accessor.nodes, node_count, stream));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemsetAsync(
            staging_accessor.nodes.data(), 0, node_count * sizeof(staging_accessor.nodes[0]), stream
        )));
        return gwn_scene_refit_ias_aabb_from_owned_data<Width, Real, Index, BlasT>(
            blas_table, instances, topology, staging_accessor, stream
        );
    };

    return gwn_replace_accessor_with_staging(
        staging_aabb.accessor(), release_aabb, build_aabb, stream
    );
}

template <
    int Width, gwn_real_type Real, gwn_index_type Index, class MortonCode, class BlasT,
    class BuildBinaryFn>
gwn_status gwn_scene_build_impl(
    char const *entry_name, cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree, BuildBinaryFn &&build_binary_fn,
    cudaStream_t const stream
) noexcept {
    return gwn_try_translate_status(entry_name, [&]() -> gwn_status {
        gwn_topology_build_preprocess<Real, Index, MortonCode> preprocess{};
        GWN_RETURN_ON_ERROR(
            (gwn_scene_build_preprocess<MortonCode>(blas_table, instances, preprocess, stream))
        );
        GWN_RETURN_ON_ERROR(
            (gwn_bvh_topology_build_from_preprocess_impl<Width, Real, Index, MortonCode>(
                entry_name, preprocess, instances.size(), topology, aabb_tree,
                std::forward<BuildBinaryFn>(build_binary_fn), stream
            ))
        );
        return gwn_scene_build_ias_aabb<Width, Real, Index>(
            topology,
            cuda::std::span<gwn_aabb<Real> const>(
                preprocess.sorted_primitive_aabbs.data(), preprocess.sorted_primitive_aabbs.size()
            ),
            aabb_tree, stream
        );
    });
}

template <gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_validate_build_inputs(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances
) noexcept {
    if (!gwn_span_has_storage(blas_table)) {
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_build, "BLAS table span has null storage."
        );
    }
    if (!gwn_span_has_storage(instances))
        return gwn_bvh_invalid_argument(k_gwn_scene_phase_build, "Instance span has null storage.");
    if (blas_table.empty())
        return gwn_bvh_invalid_argument(k_gwn_scene_phase_build, "BLAS table must not be empty.");
    if (instances.empty()) {
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_build, "Instance list must not be empty."
        );
    }
    if (instances.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_build, "Instance count exceeds the representable index range."
        );
    }

    for (std::size_t i = 0; i < blas_table.size(); ++i) {
        if (!blas_table[i].is_valid()) {
            return gwn_bvh_invalid_argument(
                k_gwn_scene_phase_build, std::format("BLAS table entry {} is invalid.", i)
            );
        }
    }
    for (std::size_t i = 0; i < instances.size(); ++i) {
        if (!gwn_index_in_bounds(instances[i].blas_index, blas_table.size())) {
            return gwn_bvh_invalid_argument(
                k_gwn_scene_phase_build,
                std::format("Instance {} has an out-of-range BLAS index.", i)
            );
        }
    }
    return gwn_status::ok();
}

} // namespace detail

template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_build_lbvh(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_scene_object<Width, Real, Index, BlasT> &scene, cudaStream_t const stream
) noexcept {
    GWN_RETURN_ON_ERROR(detail::gwn_scene_validate_build_inputs(blas_table, instances));

    gwn_scene_object<Width, Real, Index, BlasT> staging{};
    staging.set_stream(stream);
    GWN_RETURN_ON_ERROR(staging.blas_table_.copy_from_host(blas_table, stream));
    GWN_RETURN_ON_ERROR(staging.instances_.copy_from_host(instances, stream));
    GWN_RETURN_ON_ERROR((detail::gwn_scene_build_impl<Width, Real, Index, std::uint64_t>(
        "gwn_scene_build_lbvh",
        cuda::std::span<BlasT const>(staging.blas_table_.data(), staging.blas_table_.size()),
        cuda::std::span<gwn_instance_record<Real, Index> const>(
            staging.instances_.data(), staging.instances_.size()
        ),
        staging.ias_topology_.accessor(), staging.ias_aabb_.accessor(),
        [&](detail::gwn_topology_build_preprocess<Real, Index, std::uint64_t> const &preprocess,
            gwn_device_array<detail::gwn_binary_node<Index>> &binary_nodes,
            gwn_device_array<Index> &binary_internal_parent,
            gwn_device_array<gwn_aabb<Real>> &binary_internal_bounds,
            Index &root_internal_index) -> gwn_status {
        GWN_RETURN_ON_ERROR((detail::gwn_bvh_topology_build_binary_lbvh<Real, Index, std::uint64_t>(
            cuda::std::span<std::uint64_t const>(
                preprocess.sorted_morton_codes.data(), preprocess.sorted_morton_codes.size()
            ),
            cuda::std::span<gwn_aabb<Real> const>(
                preprocess.sorted_primitive_aabbs.data(), preprocess.sorted_primitive_aabbs.size()
            ),
            binary_nodes, binary_internal_parent, binary_internal_bounds, stream
        )));
        if (preprocess.sorted_morton_codes.size() > 1)
            root_internal_index = Index(0);
        return gwn_status::ok();
    },
        stream
    )));
    swap(scene, staging);
    return gwn_status::ok();
}

template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_build_hploc(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_scene_object<Width, Real, Index, BlasT> &scene, cudaStream_t const stream
) noexcept {
    GWN_RETURN_ON_ERROR(detail::gwn_scene_validate_build_inputs(blas_table, instances));

    gwn_scene_object<Width, Real, Index, BlasT> staging{};
    staging.set_stream(stream);
    GWN_RETURN_ON_ERROR(staging.blas_table_.copy_from_host(blas_table, stream));
    GWN_RETURN_ON_ERROR(staging.instances_.copy_from_host(instances, stream));
    GWN_RETURN_ON_ERROR((detail::gwn_scene_build_impl<Width, Real, Index, std::uint64_t>(
        "gwn_scene_build_hploc",
        cuda::std::span<BlasT const>(staging.blas_table_.data(), staging.blas_table_.size()),
        cuda::std::span<gwn_instance_record<Real, Index> const>(
            staging.instances_.data(), staging.instances_.size()
        ),
        staging.ias_topology_.accessor(), staging.ias_aabb_.accessor(),
        [&](detail::gwn_topology_build_preprocess<Real, Index, std::uint64_t> const &preprocess,
            gwn_device_array<detail::gwn_binary_node<Index>> &binary_nodes,
            gwn_device_array<Index> &binary_internal_parent,
            gwn_device_array<gwn_aabb<Real>> &binary_internal_bounds,
            Index &root_internal_index) -> gwn_status {
        return detail::gwn_bvh_topology_build_binary_hploc<Real, Index, std::uint64_t>(
            cuda::std::span<Index const>(
                preprocess.sorted_primitive_indices.data(),
                preprocess.sorted_primitive_indices.size()
            ),
            cuda::std::span<std::uint64_t const>(
                preprocess.sorted_morton_codes.data(), preprocess.sorted_morton_codes.size()
            ),
            cuda::std::span<gwn_aabb<Real> const>(
                preprocess.sorted_primitive_aabbs.data(), preprocess.sorted_primitive_aabbs.size()
            ),
            binary_nodes, binary_internal_parent, binary_internal_bounds, root_internal_index,
            stream
        );
    },
        stream
    )));
    swap(scene, staging);
    return gwn_status::ok();
}

template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_refit_transforms(
    cuda::std::span<gwn_instance_record<Real, Index> const> const updated_instances,
    gwn_scene_object<Width, Real, Index, BlasT> &scene, cudaStream_t const stream
) noexcept {
    return detail::gwn_try_translate_status("gwn_scene_refit_transforms", [&]() -> gwn_status {
        GWN_RETURN_ON_ERROR((detail::gwn_scene_validate_initialized<Real, Index, BlasT>(
            scene.blas_table_, scene.instances_, scene.ias_topology_.accessor(),
            scene.ias_aabb_.accessor(), detail::k_gwn_scene_phase_refit
        )));
        GWN_RETURN_ON_ERROR((detail::gwn_scene_validate_refit_inputs<Real, Index, BlasT>(
            updated_instances, scene.instances_.size(), scene.blas_table_.size()
        )));

        gwn_device_array<gwn_instance_record<Real, Index>> staging_instances{};
        staging_instances.set_stream(stream);
        GWN_RETURN_ON_ERROR(staging_instances.copy_from_host(updated_instances, stream));

        gwn_bvh_aabb_tree_object<Width, Real, Index> staging_aabb{};
        staging_aabb.set_stream(stream);
        GWN_RETURN_ON_ERROR((detail::gwn_scene_build_staged_ias_aabb<Width, Real, Index, BlasT>(
            cuda::std::span<BlasT const>(scene.blas_table_.data(), scene.blas_table_.size()),
            cuda::std::span<gwn_instance_record<Real, Index> const>(
                staging_instances.data(), staging_instances.size()
            ),
            scene.ias_topology_.accessor(), staging_aabb, stream
        )));

        swap(scene.instances_, staging_instances);
        swap(scene.ias_aabb_, staging_aabb);
        scene.set_stream(stream);
        return gwn_status::ok();
    });
}

template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_update_blas_table(
    cuda::std::span<BlasT const> const updated_blas_table,
    gwn_scene_object<Width, Real, Index, BlasT> &scene, cudaStream_t const stream
) noexcept {
    return detail::gwn_try_translate_status("gwn_scene_update_blas_table", [&]() -> gwn_status {
        GWN_RETURN_ON_ERROR((detail::gwn_scene_validate_initialized<Real, Index, BlasT>(
            scene.blas_table_, scene.instances_, scene.ias_topology_.accessor(),
            scene.ias_aabb_.accessor(), detail::k_gwn_scene_phase_update_blas_table
        )));
        GWN_RETURN_ON_ERROR(
            (detail::gwn_scene_validate_blas_table_update_inputs<Real, Index, BlasT>(
                updated_blas_table
            ))
        );
        GWN_RETURN_ON_ERROR(
            (detail::gwn_scene_validate_instance_blas_coverage_device<Width, Real, Index>(
                cuda::std::span<gwn_instance_record<Real, Index> const>(
                    scene.instances_.data(), scene.instances_.size()
                ),
                updated_blas_table.size(), stream
            ))
        );

        gwn_device_array<BlasT> staging_blas_table{};
        staging_blas_table.set_stream(stream);
        GWN_RETURN_ON_ERROR(staging_blas_table.copy_from_host(updated_blas_table, stream));

        gwn_bvh_aabb_tree_object<Width, Real, Index> staging_aabb{};
        staging_aabb.set_stream(stream);
        GWN_RETURN_ON_ERROR((detail::gwn_scene_build_staged_ias_aabb<Width, Real, Index, BlasT>(
            cuda::std::span<BlasT const>(staging_blas_table.data(), staging_blas_table.size()),
            cuda::std::span<gwn_instance_record<Real, Index> const>(
                scene.instances_.data(), scene.instances_.size()
            ),
            scene.ias_topology_.accessor(), staging_aabb, stream
        )));

        swap(scene.blas_table_, staging_blas_table);
        swap(scene.ias_aabb_, staging_aabb);
        scene.set_stream(stream);
        return gwn_status::ok();
    });
}

} // namespace gwn
