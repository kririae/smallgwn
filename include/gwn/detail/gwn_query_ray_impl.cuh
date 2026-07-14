#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_common_impl.cuh"
#include "gwn_query_vec3_impl.cuh"

namespace gwn {
namespace detail {

inline constexpr int k_gwn_ray_first_hit_block_size = 256;

/// \brief Validate the shared ray SoA and hit-record output contract of a first-hit batch.
template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] inline gwn_status gwn_validate_ray_first_hit_batch_spans(
    cuda::std::span<Real const> const ray_origin_x, cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z, cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y, cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<gwn_ray_first_hit_result<Real, Index>> const output
) noexcept {
    std::size_t const n = ray_origin_x.size();
    if (ray_origin_y.size() != n || ray_origin_z.size() != n || ray_dir_x.size() != n ||
        ray_dir_y.size() != n || ray_dir_z.size() != n || output.size() != n) {
        return gwn_status::invalid_argument("ray first-hit: mismatched span sizes");
    }

    if (!gwn_span_has_storage(ray_origin_x) || !gwn_span_has_storage(ray_origin_y) ||
        !gwn_span_has_storage(ray_origin_z) || !gwn_span_has_storage(ray_dir_x) ||
        !gwn_span_has_storage(ray_dir_y) || !gwn_span_has_storage(ray_dir_z) ||
        !gwn_span_has_storage(output)) {
        return gwn_status::invalid_argument(
            "ray first-hit: ray/output spans must use non-null storage when non-empty."
        );
    }

    return gwn_status::ok();
}

/// \brief Closed ray-parameter interval surviving an AABB slab test.
template <gwn_real_type Real> struct gwn_ray_aabb_interval {
    bool hit{false};
    Real t_near{Real(0)};
    Real t_far{Real(0)};
};

/// \brief Mutable nearest-hit state shared while one ray traverses the BVH.
template <gwn_real_type Real, gwn_index_type Index> struct gwn_ray_best_hit {
    Real t{Real(-1)};
    Index primitive_id{gwn_invalid_index<Index>()};
    Real u{Real(0)};
    Real v{Real(0)};
    gwn_query_vec3<Real> geometric_normal{};
    bool found{false};
};

/// \brief Per-axis reciprocal and parallel flags reused by ray-AABB tests.
template <gwn_real_type Real> struct gwn_ray_dir_precompute {
    Real inv[3]{Real(0), Real(0), Real(0)};
    bool zero[3]{true, true, true};
};

/// \brief Precompute ray-direction state without dividing parallel components by zero.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_ray_dir_precompute<Real>
gwn_ray_make_dir_precompute_impl(Real const dir_x, Real const dir_y, Real const dir_z) noexcept {
    gwn_ray_dir_precompute<Real> dir{};

    dir.zero[0] = dir_x == Real(0);
    dir.zero[1] = dir_y == Real(0);
    dir.zero[2] = dir_z == Real(0);

    if (!dir.zero[0])
        dir.inv[0] = Real(1) / dir_x;
    if (!dir.zero[1])
        dir.inv[1] = Real(1) / dir_y;
    if (!dir.zero[2])
        dir.inv[2] = Real(1) / dir_z;

    return dir;
}

/// \brief Intersect one closed AABB slab with the current ray interval.
template <gwn_real_type Real>
__device__ inline bool gwn_ray_aabb_update_axis_interval_impl(
    Real const origin, Real const lo, Real const hi, Real const inv_dir, bool const zero_direction,
    Real &t_near, Real &t_far
) noexcept {
    // Keep parallel slabs explicit so a boundary hit never evaluates 0 * infinity. Nonzero
    // directions use min/max ordering without a sign-dependent branch across the warp.
    if (zero_direction)
        return origin >= lo && origin <= hi;

    Real const t0 = (lo - origin) * inv_dir;
    Real const t1 = (hi - origin) * inv_dir;

    t_near = std::max(t_near, std::min(t0, t1));
    t_far = std::min(t_far, std::max(t0, t1));
    return t_near <= t_far;
}

/// \brief Intersect a ray interval with a closed axis-aligned box.
template <gwn_real_type Real>
__device__ inline gwn_ray_aabb_interval<Real> gwn_ray_aabb_intersect_interval_impl(
    Real const ray_ox, Real const ray_oy, Real const ray_oz,
    gwn_ray_dir_precompute<Real> const &ray_dir, Real const min_x, Real const min_y,
    Real const min_z, Real const max_x, Real const max_y, Real const max_z, Real const t_min,
    Real const t_max
) noexcept {
    gwn_ray_aabb_interval<Real> result{};
    if (!(t_max >= t_min))
        return result;

    Real t_near = t_min;
    Real t_far = t_max;

    if (!gwn_ray_aabb_update_axis_interval_impl(
            ray_ox, min_x, max_x, ray_dir.inv[0], ray_dir.zero[0], t_near, t_far
        )) {
        return result;
    }
    if (!gwn_ray_aabb_update_axis_interval_impl(
            ray_oy, min_y, max_y, ray_dir.inv[1], ray_dir.zero[1], t_near, t_far
        )) {
        return result;
    }
    if (!gwn_ray_aabb_update_axis_interval_impl(
            ray_oz, min_z, max_z, ray_dir.inv[2], ray_dir.zero[2], t_near, t_far
        )) {
        return result;
    }

    if (t_far < t_near)
        return result;

    result.hit = true;
    result.t_near = t_near;
    result.t_far = t_far;
    return result;
}

/// \brief Compute an oriented triangle normal with component-wise stable products.
template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ inline gwn_query_vec3<Real> gwn_stable_triangle_normal_impl(
    gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b, gwn_query_vec3<Real> const &c
) noexcept {
    using std::abs;

    Real const ab_mul_x = a.z * b.y;
    Real const ab_mul_y = a.x * b.z;
    Real const ab_mul_z = a.y * b.x;
    Real const bc_mul_x = b.z * c.y;
    Real const bc_mul_y = b.x * c.z;
    Real const bc_mul_z = b.y * c.x;

    gwn_query_vec3<Real> const cross_ab(
        a.y * b.z - ab_mul_x, a.z * b.x - ab_mul_y, a.x * b.y - ab_mul_z
    );
    gwn_query_vec3<Real> const cross_bc(
        b.y * c.z - bc_mul_x, b.z * c.x - bc_mul_y, b.x * c.y - bc_mul_z
    );

    // Select component-wise stable candidates as in Embree's stable normal routine.
    gwn_query_vec3<Real> normal = cross_bc;
    if (abs(ab_mul_x) < abs(bc_mul_x))
        normal.x = cross_ab.x;
    if (abs(ab_mul_y) < abs(bc_mul_y))
        normal.y = cross_ab.y;
    if (abs(ab_mul_z) < abs(bc_mul_z))
        normal.z = cross_ab.z;
    return normal;
}

/// \brief Intersect a ray with a triangle using oriented Pluecker edge functions.
template <gwn_real_type Real>
__device__ inline bool gwn_ray_triangle_intersect_impl(
    gwn_query_vec3<Real> const &origin, gwn_query_vec3<Real> const &direction,
    gwn_query_vec3<Real> const &v0, gwn_query_vec3<Real> const &v1, gwn_query_vec3<Real> const &v2,
    Real const t_min, Real const t_max, Real &t_out, Real &u_out, Real &v_out,
    gwn_query_vec3<Real> &geometric_normal_out
) noexcept {
    // Rebasing at the ray origin keeps the edge functions translation invariant and reduces the
    // magnitude of products for geometry far from the coordinate origin.
    gwn_query_vec3<Real> const p0 = v0 - origin;
    gwn_query_vec3<Real> const p1 = v1 - origin;
    gwn_query_vec3<Real> const p2 = v2 - origin;

    // All three oriented edge functions must agree in sign. The scale-relative tolerance admits
    // their shared boundary so adjacent triangles do not leave a crack.
    gwn_query_vec3<Real> const e0 = p2 - p0;
    gwn_query_vec3<Real> const e1 = p0 - p1;
    gwn_query_vec3<Real> const e2 = p1 - p2;

    Real const u = gwn_query_dot(gwn_query_cross(e0, p2 + p0), direction);
    Real const v = gwn_query_dot(gwn_query_cross(e1, p0 + p1), direction);
    Real const w = gwn_query_dot(gwn_query_cross(e2, p1 + p2), direction);
    Real const uvw = u + v + w;
    using std::abs;
    Real const edge_eps = std::numeric_limits<Real>::epsilon() * abs(uvw);
    Real const min_edge = std::min(u, std::min(v, w));
    Real const max_edge = std::max(u, std::max(v, w));

    if (!(min_edge >= -edge_eps || max_edge <= edge_eps))
        return false;

    // Use the stable geometric normal for both the plane solve and the returned hit record, so t
    // and normal are derived from exactly the same arithmetic.
    gwn_query_vec3<Real> const ng = gwn_stable_triangle_normal_impl(e0, e1, e2);
    Real const den = Real(2) * gwn_query_dot(ng, direction);
    if (den == Real(0))
        return false;

    Real const t_num = Real(2) * gwn_query_dot(p0, ng);
    Real const t = t_num / den;
    if (!(t_min <= t && t <= t_max))
        return false;

    // Preserve ratios for a subnormal sum without slowing the normal reciprocal path.
    if (abs(uvw) >= std::numeric_limits<Real>::min()) {
        Real const inv_uvw = Real(1) / uvw;
        u_out = u * inv_uvw;
        v_out = v * inv_uvw;
    } else {
        if (uvw == Real(0))
            return false;
        u_out = u / uvw;
        v_out = v / uvw;
    }
    t_out = t;
    // Preserve the same stable, unnormalized normal used to solve t. Returning this intersection
    // datum avoids a second triangle load while leaving normalization and facing to the caller.
    geometric_normal_out = ng;
    return true;
}

/// \brief Traverse float BVH4 children in near order with a 32-bit pending-child payload.
template <int StackCapacity, bool HasParallelAxis, typename VisitLeaf, typename OverflowCallback>
[[nodiscard]] __device__ inline bool gwn_ray_first_hit_bvh4_packed_impl(
    gwn_bvh4_accessor<float, std::uint32_t> const &bvh, float const ray_ox, float const ray_oy,
    float const ray_oz, gwn_ray_dir_precompute<float> const &ray_dir,
    gwn_ray_best_hit<float, std::uint32_t> &best, float const t_min, VisitLeaf const &visit_leaf,
    OverflowCallback const &overflow_callback
) noexcept {
    // Leaves and internal children share one near-ordered sequence. Processing the nearest child
    // first tightens best.t early enough to prune farther AABBs and triangle ranges.
    constexpr std::uint32_t k_leaf_bit = std::uint32_t(1) << 31;
    constexpr float k_invalid_order = std::numeric_limits<float>::infinity();

    std::uint32_t stack[StackCapacity];
    int stack_size = 0;
    auto payload = static_cast<std::uint32_t>(bvh.root.offset());
    auto const intersect_bounds = [&](gwn_aabb<float> const &bounds) noexcept {
        if constexpr (!HasParallelAxis) {
            float const tx0 = (bounds.min_x - ray_ox) * ray_dir.inv[0];
            float const tx1 = (bounds.max_x - ray_ox) * ray_dir.inv[0];
            float const ty0 = (bounds.min_y - ray_oy) * ray_dir.inv[1];
            float const ty1 = (bounds.max_y - ray_oy) * ray_dir.inv[1];
            float const tz0 = (bounds.min_z - ray_oz) * ray_dir.inv[2];
            float const tz1 = (bounds.max_z - ray_oz) * ray_dir.inv[2];
            float const t_near = std::max(
                t_min,
                std::max(std::min(tx0, tx1), std::max(std::min(ty0, ty1), std::min(tz0, tz1)))
            );
            float const t_far = std::min(
                best.t,
                std::min(std::max(tx0, tx1), std::min(std::max(ty0, ty1), std::max(tz0, tz1)))
            );
            return gwn_ray_aabb_interval<float>{t_near <= t_far, t_near, t_far};
        }

        float t_near = t_min;
        float t_far = best.t;
        // Parallel axes constrain containment but do not shorten the ray interval. Reusing the
        // scalar slab step keeps packed traversal exact for arbitrarily thin child bounds.
        if (!gwn_ray_aabb_update_axis_interval_impl(
                ray_ox, bounds.min_x, bounds.max_x, ray_dir.inv[0], ray_dir.zero[0], t_near, t_far
            ) ||
            !gwn_ray_aabb_update_axis_interval_impl(
                ray_oy, bounds.min_y, bounds.max_y, ray_dir.inv[1], ray_dir.zero[1], t_near, t_far
            ) ||
            !gwn_ray_aabb_update_axis_interval_impl(
                ray_oz, bounds.min_z, bounds.max_z, ray_dir.inv[2], ray_dir.zero[2], t_near, t_far
            )) {
            return gwn_ray_aabb_interval<float>{};
        }
        return gwn_ray_aabb_interval<float>{true, t_near, t_far};
    };
    while (true) {
        if ((payload & k_leaf_bit) != 0u) {
            int const child_slot = static_cast<int>(payload & 3u);
            std::uint32_t const parent_index = (payload & ~k_leaf_bit) >> 2;
            visit_leaf(bvh.nodes[parent_index].child(child_slot));
        } else if (payload < bvh.nodes.size()) {
            auto const &node = bvh.nodes[payload];
            // Distance and payload move in parallel so the sorting network uses native float
            // comparisons without widening every candidate to a 64-bit composite key.
            float child_entry_t[4];
            std::uint32_t child_payload[4];

            GWN_DETAIL_PRAGMA_UNROLL
            for (int child_slot = 0; child_slot < 4; ++child_slot) {
                child_entry_t[child_slot] = k_invalid_order;
                child_payload[child_slot] = 0u;
                auto const &child = node.child(child_slot);
                if (!child.is_valid())
                    continue;
                if (child.is_internal() && child.offset() >= bvh.nodes.size())
                    continue;

                auto const interval = intersect_bounds(child.bounds);
                if (!interval.hit)
                    continue;

                std::uint32_t const payload_value =
                    child.is_leaf()
                        ? k_leaf_bit | (payload << 2) | static_cast<std::uint32_t>(child_slot)
                        : static_cast<std::uint32_t>(child.offset());
                child_entry_t[child_slot] = interval.t_near;
                child_payload[child_slot] = payload_value;
            }

            auto compare_swap = [&](int const lhs, int const rhs) noexcept {
                if (!(child_entry_t[lhs] > child_entry_t[rhs]))
                    return;
                using std::swap;
                swap(child_entry_t[lhs], child_entry_t[rhs]);
                swap(child_payload[lhs], child_payload[rhs]);
            };
            compare_swap(0, 1);
            compare_swap(2, 3);
            compare_swap(0, 2);
            compare_swap(1, 3);
            compare_swap(1, 2);

            GWN_DETAIL_PRAGMA_UNROLL
            for (int child_slot = 3; child_slot > 0; --child_slot) {
                if (child_entry_t[child_slot] == k_invalid_order)
                    continue;
                if (stack_size >= StackCapacity) {
                    overflow_callback();
                    return true;
                }
                stack[stack_size++] = child_payload[child_slot];
            }
            if (child_entry_t[0] != k_invalid_order) {
                payload = child_payload[0];
                continue;
            }
        }

        if (stack_size == 0)
            return false;
        payload = stack[--stack_size];
    }
}

/// \brief Traverse a canonical BVH and return the nearest ray-triangle hit.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback,
    bool UsePackedTraversal = false>
[[nodiscard]] __device__ inline gwn_ray_first_hit_result<Real, Index> gwn_ray_first_hit_impl(
    gwn_bvh_accessor<Width, Real, Index> const &bvh, Real const ray_ox, Real const ray_oy,
    Real const ray_oz, Real const ray_dx, Real const ray_dy, Real const ray_dz, Real const t_min,
    Real const t_max, OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    gwn_ray_first_hit_result<Real, Index> result{};
    Real const dir_len2 = ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz;
    if (!(dir_len2 > Real(0)))
        return result;

    gwn_query_vec3<Real> const origin(ray_ox, ray_oy, ray_oz);
    gwn_query_vec3<Real> const direction(ray_dx, ray_dy, ray_dz);
    auto const ray_dir = gwn_ray_make_dir_precompute_impl(ray_dx, ray_dy, ray_dz);

    gwn_ray_best_hit<Real, Index> best{};
    best.t = t_max;
    // Publish only complete records. Until a hit survives every test, the result remains the
    // canonical miss record initialized above.
    auto publish_best = [&]() noexcept {
        if (!best.found)
            return;
        result.t = best.t;
        result.primitive_id = best.primitive_id;
        result.u = best.u;
        result.v = best.v;
        result.geometric_normal_x = best.geometric_normal.x;
        result.geometric_normal_y = best.geometric_normal.y;
        result.geometric_normal_z = best.geometric_normal.z;
        result.status = gwn_ray_first_hit_status::k_hit;
    };
    auto visit_leaf = [&](gwn_bvh_child<Real> const &leaf) noexcept {
        for (std::uint32_t primitive_offset = 0; primitive_offset < leaf.primitive_count();
             ++primitive_offset) {
            std::uint64_t const sorted_index = leaf.offset() + primitive_offset;
            if (sorted_index >= bvh.triangles.size())
                continue;

            Real t_hit = Real(0);
            Real u_hit = Real(0);
            Real v_hit = Real(0);
            gwn_query_vec3<Real> geometric_normal{};
            auto const &triangle = bvh.triangles[static_cast<std::size_t>(sorted_index)];
            gwn_query_vec3<Real> const v0(triangle.v0_x, triangle.v0_y, triangle.v0_z);
            gwn_query_vec3<Real> const v1(
                triangle.v0_x + triangle.e1_x, triangle.v0_y + triangle.e1_y,
                triangle.v0_z + triangle.e1_z
            );
            gwn_query_vec3<Real> const v2(
                triangle.v0_x + triangle.e2_x, triangle.v0_y + triangle.e2_y,
                triangle.v0_z + triangle.e2_z
            );
            if (!gwn_ray_triangle_intersect_impl(
                    origin, direction, v0, v1, v2, t_min, best.t, t_hit, u_hit, v_hit,
                    geometric_normal
                )) {
                continue;
            }
            if (best.found && !(t_hit < best.t))
                continue;

            // Triangle records stay on the hot path. The parallel mapping is read only after a
            // hit survives the current distance bound and must report its original mesh ID.
            best.t = t_hit;
            best.primitive_id = bvh.primitive_indices[static_cast<std::size_t>(sorted_index)];
            best.u = u_hit;
            best.v = v_hit;
            best.geometric_normal = geometric_normal;
            best.found = true;
        }
    };

    if constexpr (UsePackedTraversal) {
        static_assert(Width == 4, "Packed ray traversal requires BVH width 4.");
        static_assert(std::is_same_v<Real, float>, "Packed ray traversal requires float.");
        static_assert(
            std::is_same_v<Index, std::uint32_t>, "Packed ray traversal requires uint32_t indices."
        );
        bool overflow = false;
        // Select parallel handling once per ray so the common path keeps a branch-free slab test
        // at every child while zero-direction rays retain exact containment semantics.
        if (ray_dir.zero[0] || ray_dir.zero[1] || ray_dir.zero[2]) {
            overflow = gwn_ray_first_hit_bvh4_packed_impl<StackCapacity, true>(
                bvh, ray_ox, ray_oy, ray_oz, ray_dir, best, t_min, visit_leaf, overflow_callback
            );
        } else {
            overflow = gwn_ray_first_hit_bvh4_packed_impl<StackCapacity, false>(
                bvh, ray_ox, ray_oy, ray_oz, ray_dir, best, t_min, visit_leaf, overflow_callback
            );
        }
        publish_best();
        if (overflow)
            result.status = gwn_ray_first_hit_status::k_overflow;
        return result;
    }

    std::uint64_t stack[StackCapacity];
    int stack_size = 0;
    std::uint64_t reference = bvh.root.reference;
    while (true) {
        gwn_bvh_child<Real> current{};
        current.reference = reference;
        if (current.is_leaf()) {
            visit_leaf(current);
        } else if (current.is_internal() && current.offset() < bvh.nodes.size()) {
            auto const &node = bvh.nodes[static_cast<std::size_t>(current.offset())];
            Real child_entry_t[Width];
            std::uint64_t child_reference[Width];
            int child_count = 0;

            GWN_DETAIL_PRAGMA_UNROLL
            for (int child_slot = 0; child_slot < Width; ++child_slot) {
                auto const &child = node.child(child_slot);
                if (!child.is_valid())
                    continue;

                auto const interval = gwn_ray_aabb_intersect_interval_impl<Real>(
                    ray_ox, ray_oy, ray_oz, ray_dir, child.bounds.min_x, child.bounds.min_y,
                    child.bounds.min_z, child.bounds.max_x, child.bounds.max_y, child.bounds.max_z,
                    t_min, best.t
                );
                if (!interval.hit)
                    continue;
                child_entry_t[child_count] = interval.t_near;
                child_reference[child_count] = child.reference;
                ++child_count;
            }

            auto compare_swap = [&](int const lhs, int const rhs) noexcept {
                if (!(child_entry_t[lhs] > child_entry_t[rhs]))
                    return;
                using std::swap;
                swap(child_entry_t[lhs], child_entry_t[rhs]);
                swap(child_reference[lhs], child_reference[rhs]);
            };
            if constexpr (Width == 2) {
                if (child_count == 2)
                    compare_swap(0, 1);
            } else if constexpr (Width == 4) {
                // Fill inactive lanes with sentinels so the fixed five-comparison network also
                // handles nodes whose arity is below four.
                for (int child_slot = child_count; child_slot < Width; ++child_slot) {
                    child_entry_t[child_slot] = std::numeric_limits<Real>::infinity();
                    child_reference[child_slot] = 0u;
                }
                compare_swap(0, 1);
                compare_swap(2, 3);
                compare_swap(0, 2);
                compare_swap(1, 3);
                compare_swap(1, 2);
            } else {
                for (int pass = 0; pass < child_count; ++pass)
                    for (int child_slot = pass & 1; child_slot + 1 < child_count; child_slot += 2)
                        compare_swap(child_slot, child_slot + 1);
            }

            // Leaf work stays in the current node. Only internal references occupy the traversal
            // stack, which preserves the topology-derived host capacity bound.
            for (int child_slot = 0; child_slot < child_count; ++child_slot) {
                gwn_bvh_child<Real> child{};
                child.reference = child_reference[child_slot];
                if (child.is_leaf() && child_entry_t[child_slot] <= best.t)
                    visit_leaf(child);
            }
            // Reverse insertion makes the closest remaining internal child the next LIFO entry.
            for (int child_slot = child_count - 1; child_slot >= 0; --child_slot) {
                gwn_bvh_child<Real> child{};
                child.reference = child_reference[child_slot];
                if (!child.is_internal() || child_entry_t[child_slot] > best.t)
                    continue;
                if (stack_size >= StackCapacity) {
                    overflow_callback();
                    publish_best();
                    result.status = gwn_ray_first_hit_status::k_overflow;
                    return result;
                }
                stack[stack_size++] = child.reference;
            }
        }

        if (stack_size == 0)
            break;
        reference = stack[--stack_size];
    }

    publish_best();
    return result;
}

/// \brief Invoke canonical ray first-hit traversal for one batch element.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback,
    bool UsePackedTraversal = false>
struct gwn_ray_first_hit_batch_functor {
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    cuda::std::span<Real const> ray_origin_x{};
    cuda::std::span<Real const> ray_origin_y{};
    cuda::std::span<Real const> ray_origin_z{};
    cuda::std::span<Real const> ray_dir_x{};
    cuda::std::span<Real const> ray_dir_y{};
    cuda::std::span<Real const> ray_dir_z{};
    cuda::std::span<gwn_ray_first_hit_result<Real, Index>> out_hit{};
    Real t_min{};
    Real t_max{};
    OverflowCallback overflow_callback{};

    void __device__ operator()(std::size_t const ray_id) const noexcept {
        // The host launcher validates one accessor for the whole batch. Skipping the same span and
        // root checks per ray keeps validation cost independent of query count.
        auto const hit = gwn_ray_first_hit_impl<
            Width, Real, Index, StackCapacity, OverflowCallback, UsePackedTraversal>(
            bvh, ray_origin_x[ray_id], ray_origin_y[ray_id], ray_origin_z[ray_id],
            ray_dir_x[ray_id], ray_dir_y[ray_id], ray_dir_z[ray_id], t_min, t_max, overflow_callback
        );
        out_hit[ray_id] = hit;
    }
};

} // namespace detail
} // namespace gwn
