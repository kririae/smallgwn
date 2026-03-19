#pragma once

#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../gwn_scene.cuh"
#include "gwn_query_common_impl.cuh"
#include "gwn_query_ray_impl.cuh"

namespace gwn {
namespace detail {

template <class Accel>
using gwn_accel_real_t = typename gwn_accel_traits<std::remove_cvref_t<Accel>>::real_type;

template <class Accel>
using gwn_accel_index_t = typename gwn_accel_traits<std::remove_cvref_t<Accel>>::index_type;

template <class Accel>
[[nodiscard]] __host__ __device__ inline bool
gwn_scene_query_accel_has_basic_data_impl(Accel const &accel) noexcept {
    if constexpr (is_blas_accessor_v<std::remove_cvref_t<Accel>>) {
        return accel.geometry.is_valid() && accel.topology.is_valid() &&
               accel.aabb.is_valid_for(accel.topology);
    } else {
        return accel.ias_topology.is_valid() && accel.ias_aabb.is_valid_for(accel.ias_topology) &&
               !accel.blas_table.empty() && !accel.instances.empty() &&
               gwn_span_has_storage(accel.blas_table) && gwn_span_has_storage(accel.instances);
    }
}

template <class BlasAccel, int StackCapacity, typename OverflowCallback>
[[nodiscard]] __device__ inline gwn_ray_hit_result<
    gwn_accel_real_t<BlasAccel>, gwn_accel_index_t<BlasAccel>>
gwn_ray_first_hit_blas_unified_impl(
    BlasAccel const &blas, gwn_accel_real_t<BlasAccel> const ray_ox,
    gwn_accel_real_t<BlasAccel> const ray_oy, gwn_accel_real_t<BlasAccel> const ray_oz,
    gwn_accel_real_t<BlasAccel> const ray_dx, gwn_accel_real_t<BlasAccel> const ray_dy,
    gwn_accel_real_t<BlasAccel> const ray_dz, gwn_accel_real_t<BlasAccel> const t_min,
    gwn_accel_real_t<BlasAccel> const t_max, OverflowCallback const &overflow_callback = {}
) noexcept {
    using accel_type = std::remove_cvref_t<BlasAccel>;
    using Real = gwn_accel_real_t<accel_type>;
    using Index = gwn_accel_index_t<accel_type>;

    auto const hit = gwn_ray_first_hit_bvh_impl<
        accel_type::k_width, Real, Index, StackCapacity, OverflowCallback>(
        blas.geometry, blas.topology, blas.aabb, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
        t_min, t_max, overflow_callback
    );

    gwn_ray_hit_result<Real, Index> result{};
    result.t = hit.t;
    result.instance_id = gwn_invalid_index<Index>();
    result.primitive_id = hit.primitive_id;
    result.u = hit.u;
    result.v = hit.v;
    result.status = hit.status;
    return result;
}

template <class SceneAccel, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline bool gwn_scene_ray_candidate_has_payload_impl(
    gwn_ray_hit_result<Real, Index> const &candidate
) noexcept {
    return !gwn_is_invalid_index(candidate.primitive_id);
}

template <class SceneAccel, gwn_real_type Real, gwn_index_type Index>
__device__ inline void gwn_scene_update_ray_best_hit_impl(
    gwn_ray_hit_result<Real, Index> const &candidate, Index const instance_id,
    gwn_ray_hit_result<Real, Index> &best
) noexcept {
    if (!gwn_scene_ray_candidate_has_payload_impl<SceneAccel, Real, Index>(candidate))
        return;
    if (!gwn_scene_ray_candidate_has_payload_impl<SceneAccel, Real, Index>(best) ||
        candidate.t < best.t) {
        best.t = candidate.t;
        best.instance_id = instance_id;
        best.primitive_id = candidate.primitive_id;
        best.u = candidate.u;
        best.v = candidate.v;
        best.status = gwn_ray_first_hit_status::k_hit;
    }
}

template <class SceneAccel, int StackCapacity, typename OverflowCallback>
[[nodiscard]] __device__ inline bool gwn_scene_visit_instance_range_for_ray_impl(
    SceneAccel const &scene, gwn_accel_real_t<SceneAccel> const ray_ox,
    gwn_accel_real_t<SceneAccel> const ray_oy, gwn_accel_real_t<SceneAccel> const ray_oz,
    gwn_accel_real_t<SceneAccel> const ray_dx, gwn_accel_real_t<SceneAccel> const ray_dy,
    gwn_accel_real_t<SceneAccel> const ray_dz, gwn_accel_real_t<SceneAccel> const t_min,
    gwn_accel_real_t<SceneAccel> const t_max, gwn_accel_index_t<SceneAccel> const begin,
    gwn_accel_index_t<SceneAccel> const count,
    gwn_ray_hit_result<gwn_accel_real_t<SceneAccel>, gwn_accel_index_t<SceneAccel>> &best,
    OverflowCallback const &overflow_callback
) noexcept {
    using scene_type = std::remove_cvref_t<SceneAccel>;
    using Real = gwn_accel_real_t<scene_type>;
    using Index = gwn_accel_index_t<scene_type>;
    using blas_type = typename scene_type::blas_type;

    for (Index offset = 0; offset < count; ++offset) {
        Index const sorted_index = begin + offset;
        if (!gwn_index_in_bounds(sorted_index, scene.ias_topology.primitive_indices.size()))
            continue;

        Index const instance_id =
            scene.ias_topology.primitive_indices[static_cast<std::size_t>(sorted_index)];
        if (!gwn_index_in_bounds(instance_id, scene.instances.size()))
            continue;

        auto const &instance = scene.instances[static_cast<std::size_t>(instance_id)];
        if (!gwn_index_in_bounds(instance.blas_index, scene.blas_table.size()))
            continue;

        blas_type const &blas = scene.blas_table[static_cast<std::size_t>(instance.blas_index)];
        if (!gwn_scene_query_accel_has_basic_data_impl(blas))
            continue;

        Real local_ox = Real(0);
        Real local_oy = Real(0);
        Real local_oz = Real(0);
        Real local_dx = Real(0);
        Real local_dy = Real(0);
        Real local_dz = Real(0);
        instance.transform.inverse_apply_point(
            ray_ox, ray_oy, ray_oz, local_ox, local_oy, local_oz
        );
        instance.transform.inverse_apply_direction(
            ray_dx, ray_dy, ray_dz, local_dx, local_dy, local_dz
        );

        Real nested_t_max = t_max;
        if (gwn_scene_ray_candidate_has_payload_impl<scene_type, Real, Index>(best))
            nested_t_max = best.t < t_max ? best.t : t_max;
        auto const candidate =
            gwn_ray_first_hit_blas_unified_impl<blas_type, StackCapacity, OverflowCallback>(
                blas, local_ox, local_oy, local_oz, local_dx, local_dy, local_dz, t_min,
                nested_t_max, overflow_callback
            );
        gwn_scene_update_ray_best_hit_impl<scene_type, Real, Index>(candidate, instance_id, best);
        if (candidate.status == gwn_ray_first_hit_status::k_overflow) {
            best.status = gwn_ray_first_hit_status::k_overflow;
            return true;
        }
    }

    return false;
}

template <class SceneAccel, int StackCapacity, typename OverflowCallback>
[[nodiscard]] __device__ inline gwn_ray_hit_result<
    gwn_accel_real_t<SceneAccel>, gwn_accel_index_t<SceneAccel>>
gwn_ray_first_hit_scene_impl(
    SceneAccel const &scene, gwn_accel_real_t<SceneAccel> const ray_ox,
    gwn_accel_real_t<SceneAccel> const ray_oy, gwn_accel_real_t<SceneAccel> const ray_oz,
    gwn_accel_real_t<SceneAccel> const ray_dx, gwn_accel_real_t<SceneAccel> const ray_dy,
    gwn_accel_real_t<SceneAccel> const ray_dz, gwn_accel_real_t<SceneAccel> const t_min,
    gwn_accel_real_t<SceneAccel> const t_max, OverflowCallback const &overflow_callback = {}
) noexcept {
    using scene_type = std::remove_cvref_t<SceneAccel>;
    using Real = gwn_accel_real_t<scene_type>;
    using Index = gwn_accel_index_t<scene_type>;

    static_assert(scene_type::k_width >= 2, "Scene width must be at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    gwn_ray_hit_result<Real, Index> best{};
    if (!gwn_scene_query_accel_has_basic_data_impl(scene))
        return best;
    if (!(t_max >= t_min))
        return best;

    Real const dir_len2 = ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz;
    if (!(dir_len2 > Real(0)))
        return best;

    if (scene.ias_topology.root_kind == gwn_bvh_child_kind::k_leaf) {
        if (gwn_scene_visit_instance_range_for_ray_impl<
                scene_type, StackCapacity, OverflowCallback>(
                scene, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, t_min, t_max,
                scene.ias_topology.root_index, scene.ias_topology.root_count, best,
                overflow_callback
            )) {
            return best;
        }
        return best;
    }

    if (scene.ias_topology.root_kind != gwn_bvh_child_kind::k_internal)
        return best;

    auto const ray_dir_precomp = gwn_ray_make_dir_precompute_impl(ray_dx, ray_dy, ray_dz);
    Index stack[StackCapacity];
    int stack_size = 0;
    stack[stack_size++] = scene.ias_topology.root_index;

    while (stack_size > 0) {
        Index const node_index = stack[--stack_size];
        GWN_ASSERT(stack_size >= 0, "scene ray query: IAS stack underflow");
        if (!gwn_index_in_bounds(node_index, scene.ias_topology.nodes.size()) ||
            !gwn_index_in_bounds(node_index, scene.ias_aabb.nodes.size())) {
            continue;
        }

        auto const &topology_node = scene.ias_topology.nodes[static_cast<std::size_t>(node_index)];
        auto const &aabb_node = scene.ias_aabb.nodes[static_cast<std::size_t>(node_index)];

        int child_slot_order[scene_type::k_width];
        Real child_entry_t[scene_type::k_width];
        std::uint8_t child_kind[scene_type::k_width];
        std::uint8_t constexpr k_invalid_kind =
            static_cast<std::uint8_t>(gwn_bvh_child_kind::k_invalid);
        Real constexpr k_infinite_t = std::numeric_limits<Real>::infinity();
        GWN_PRAGMA_UNROLL
        for (int i = 0; i < scene_type::k_width; ++i) {
            child_slot_order[i] = 0;
            child_entry_t[i] = k_infinite_t;
            child_kind[i] = k_invalid_kind;
        }
        int child_count = 0;

        GWN_PRAGMA_UNROLL
        for (int slot = 0; slot < scene_type::k_width; ++slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(topology_node.child_kind[slot]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf)
                continue;

            auto const interval = gwn_ray_aabb_intersect_interval_impl<Real>(
                ray_ox, ray_oy, ray_oz, ray_dir_precomp, aabb_node.child_min_x[slot],
                aabb_node.child_min_y[slot], aabb_node.child_min_z[slot],
                aabb_node.child_max_x[slot], aabb_node.child_max_y[slot],
                aabb_node.child_max_z[slot], t_min, best.hit() ? best.t : t_max
            );
            if (!interval.hit)
                continue;

            child_slot_order[child_count] = slot;
            child_entry_t[child_count] = interval.t_near;
            child_kind[child_count] = topology_node.child_kind[slot];
            ++child_count;
        }

        if (child_count > 1) {
            gwn_sort_children_by_entry_t_impl<scene_type::k_width>(
                child_entry_t, child_slot_order, child_kind
            );
        }

        GWN_PRAGMA_UNROLL
        for (int i = 0; i < scene_type::k_width; ++i) {
            if (static_cast<gwn_bvh_child_kind>(child_kind[i]) != gwn_bvh_child_kind::k_leaf)
                continue;
            if (best.hit() && child_entry_t[i] > best.t)
                continue;

            int const slot = child_slot_order[i];
            if (gwn_scene_visit_instance_range_for_ray_impl<
                    scene_type, StackCapacity, OverflowCallback>(
                    scene, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, t_min, t_max,
                    topology_node.child_index[slot], topology_node.child_count[slot], best,
                    overflow_callback
                )) {
                return best;
            }
        }

        for (int i = scene_type::k_width - 1; i >= 0; --i) {
            if (static_cast<gwn_bvh_child_kind>(child_kind[i]) != gwn_bvh_child_kind::k_internal)
                continue;
            if (best.hit() && child_entry_t[i] > best.t)
                continue;
            if (stack_size >= StackCapacity) {
                overflow_callback();
                best.status = gwn_ray_first_hit_status::k_overflow;
                return best;
            }
            stack[stack_size++] = topology_node.child_index[child_slot_order[i]];
        }
    }

    return best;
}

template <class Accel, int StackCapacity, typename OverflowCallback>
[[nodiscard]] __device__ inline gwn_ray_hit_result<
    gwn_accel_real_t<Accel>, gwn_accel_index_t<Accel>>
gwn_ray_first_hit_accel_impl(
    Accel const &accel, gwn_accel_real_t<Accel> const ray_ox, gwn_accel_real_t<Accel> const ray_oy,
    gwn_accel_real_t<Accel> const ray_oz, gwn_accel_real_t<Accel> const ray_dx,
    gwn_accel_real_t<Accel> const ray_dy, gwn_accel_real_t<Accel> const ray_dz,
    gwn_accel_real_t<Accel> const t_min, gwn_accel_real_t<Accel> const t_max,
    OverflowCallback const &overflow_callback = {}
) noexcept {
    using accel_type = std::remove_cvref_t<Accel>;
    if constexpr (is_blas_accessor_v<accel_type>) {
        return gwn_ray_first_hit_blas_unified_impl<accel_type, StackCapacity, OverflowCallback>(
            accel, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, t_min, t_max, overflow_callback
        );
    } else {
        return gwn_ray_first_hit_scene_impl<accel_type, StackCapacity, OverflowCallback>(
            accel, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, t_min, t_max, overflow_callback
        );
    }
}

template <gwn_real_type Real, gwn_index_type Index>
inline gwn_status gwn_validate_unified_ray_first_hit_batch_spans(
    cuda::std::span<Real const> const ray_origin_x, cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z, cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y, cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<Real> const output_t, cuda::std::span<Index> const output_primitive_id,
    cuda::std::span<Index> const output_instance_id
) noexcept {
    std::size_t const n = ray_origin_x.size();
    if (ray_origin_y.size() != n || ray_origin_z.size() != n || ray_dir_x.size() != n ||
        ray_dir_y.size() != n || ray_dir_z.size() != n || output_t.size() != n ||
        output_primitive_id.size() != n || output_instance_id.size() != n) {
        return gwn_status::invalid_argument("ray first-hit: mismatched span sizes");
    }
    if (!gwn_span_has_storage(ray_origin_x) || !gwn_span_has_storage(ray_origin_y) ||
        !gwn_span_has_storage(ray_origin_z) || !gwn_span_has_storage(ray_dir_x) ||
        !gwn_span_has_storage(ray_dir_y) || !gwn_span_has_storage(ray_dir_z) ||
        !gwn_span_has_storage(output_t) || !gwn_span_has_storage(output_primitive_id) ||
        !gwn_span_has_storage(output_instance_id)) {
        return gwn_status::invalid_argument(
            "ray first-hit: ray/output spans must use non-null storage when non-empty."
        );
    }
    return gwn_status::ok();
}

template <class Accel, int StackCapacity, typename OverflowCallback>
struct gwn_ray_first_hit_batch_accel_functor {
    Accel accel{};
    cuda::std::span<gwn_accel_real_t<Accel> const> ray_origin_x{};
    cuda::std::span<gwn_accel_real_t<Accel> const> ray_origin_y{};
    cuda::std::span<gwn_accel_real_t<Accel> const> ray_origin_z{};
    cuda::std::span<gwn_accel_real_t<Accel> const> ray_dir_x{};
    cuda::std::span<gwn_accel_real_t<Accel> const> ray_dir_y{};
    cuda::std::span<gwn_accel_real_t<Accel> const> ray_dir_z{};
    cuda::std::span<gwn_accel_real_t<Accel>> out_t{};
    cuda::std::span<gwn_accel_index_t<Accel>> out_primitive_id{};
    cuda::std::span<gwn_accel_index_t<Accel>> out_instance_id{};
    gwn_accel_real_t<Accel> t_min{};
    gwn_accel_real_t<Accel> t_max{};
    OverflowCallback overflow_callback{};

    __device__ void operator()(std::size_t const ray_id) const {
        auto const hit = gwn_ray_first_hit_accel_impl<Accel, StackCapacity, OverflowCallback>(
            accel, ray_origin_x[ray_id], ray_origin_y[ray_id], ray_origin_z[ray_id],
            ray_dir_x[ray_id], ray_dir_y[ray_id], ray_dir_z[ray_id], t_min, t_max, overflow_callback
        );
        out_t[ray_id] = hit.t;
        out_primitive_id[ray_id] = hit.primitive_id;
        out_instance_id[ray_id] = hit.instance_id;
    }
};

} // namespace detail
} // namespace gwn
