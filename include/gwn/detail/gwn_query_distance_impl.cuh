#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_common_impl.cuh"
#include "gwn_query_geometry_impl.cuh"
#include "gwn_query_winding_impl.cuh"

namespace gwn {
namespace detail {

/// \brief Compute squared point distance from one leaf-ordered BVH triangle record.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline Real gwn_bvh_triangle_distance_squared_impl(
    gwn_bvh_triangle<Real> const &triangle, gwn_query_vec3<Real> const &query
) noexcept {
    // Reconstruct vertices from the contiguous leaf record before entering the shared numerical
    // kernel. Distance traversal therefore uses the same triangle definition as ray traversal.
    gwn_query_vec3<Real> const v0(triangle.v0_x, triangle.v0_y, triangle.v0_z);
    gwn_query_vec3<Real> const v1(
        triangle.v0_x + triangle.e1_x, triangle.v0_y + triangle.e1_y, triangle.v0_z + triangle.e1_z
    );
    gwn_query_vec3<Real> const v2(
        triangle.v0_x + triangle.e2_x, triangle.v0_y + triangle.e2_y, triangle.v0_z + triangle.e2_z
    );
    return gwn_point_triangle_distance_squared_impl(query, v0, v1, v2);
}

/// \brief Compute unsigned point-to-mesh distance through the canonical BVH.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline Real gwn_unsigned_distance_impl(
    gwn_bvh_accessor<Width, Real, Index> const &bvh, Real const qx, Real const qy, Real const qz,
    Real const culling_band, OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    gwn_query_vec3<Real> const query(qx, qy, qz);
    // The culling band is both the initial upper bound and the value returned when no triangle is
    // closer. Squared bounds keep all node and triangle comparisons free of square roots.
    Real const band = std::max(culling_band, Real(0));
    Real best_dist2 = band * band;

    auto visit_leaf = [&](gwn_bvh_child<Real> const &leaf) noexcept {
        for (std::uint32_t primitive_offset = 0; primitive_offset < leaf.primitive_count();
             ++primitive_offset) {
            std::uint64_t const sorted_index = leaf.offset() + primitive_offset;
            if (sorted_index >= bvh.triangles.size())
                continue;
            Real const distance2 = gwn_bvh_triangle_distance_squared_impl(
                bvh.triangles[static_cast<std::size_t>(sorted_index)], query
            );
            if (distance2 < best_dist2)
                best_dist2 = distance2;
        }
        return best_dist2 > Real(0);
    };

    std::uint64_t stack[StackCapacity];
    int stack_size = 0;
    std::uint64_t reference = bvh.root.reference;
    while (true) {
        std::uint64_t next_reference = 0u;
        gwn_bvh_child<Real> current{};
        current.reference = reference;
        if (current.is_leaf()) {
            // A zero distance is the global lower bound. Returning here also avoids walking nodes
            // that were queued before this leaf tightened the bound.
            if (!visit_leaf(current))
                return Real(0);
        } else if (current.is_internal() && current.offset() < bvh.nodes.size()) {
            auto const &node = bvh.nodes[static_cast<std::size_t>(current.offset())];
            Real child_distance2[Width];
            std::uint64_t child_reference[Width];
            int child_count = 0;

            GWN_DETAIL_PRAGMA_UNROLL
            for (int child_slot = 0; child_slot < Width; ++child_slot) {
                auto const &child = node.child(child_slot);
                if (!child.is_valid())
                    continue;
                Real const distance2 = gwn_aabb_min_distance_squared_impl(
                    qx, qy, qz, child.bounds.min_x, child.bounds.min_y, child.bounds.min_z,
                    child.bounds.max_x, child.bounds.max_y, child.bounds.max_z
                );
                if (distance2 >= best_dist2)
                    continue;
                child_distance2[child_count] = distance2;
                child_reference[child_count] = child.reference;
                ++child_count;
            }

            auto compare_swap = [&](int const lhs, int const rhs) noexcept {
                if (!(child_distance2[lhs] > child_distance2[rhs]))
                    return;
                using std::swap;
                swap(child_distance2[lhs], child_distance2[rhs]);
                swap(child_reference[lhs], child_reference[rhs]);
            };
            if constexpr (Width == 2) {
                if (child_count == 2)
                    compare_swap(0, 1);
            } else if constexpr (Width == 4) {
                // Sentinels let the fixed sorting network cover all valid arities without a
                // data-dependent sort loop on the common width-four path.
                for (int child_slot = child_count; child_slot < Width; ++child_slot) {
                    child_distance2[child_slot] = std::numeric_limits<Real>::infinity();
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

            // Evaluate leaves in the current node before queuing internals. Besides tightening the
            // distance bound early, this keeps leaf references out of the internal stack bound.
            for (int child_slot = 0; child_slot < child_count; ++child_slot) {
                gwn_bvh_child<Real> child{};
                child.reference = child_reference[child_slot];
                if (child.is_leaf() && child_distance2[child_slot] < best_dist2 &&
                    !visit_leaf(child)) {
                    return Real(0);
                }
            }
            // Scan from far to near, queue each previous candidate, and enter the closest internal
            // child directly. The remaining references retain the same near-first LIFO order.
            for (int child_slot = child_count - 1; child_slot >= 0; --child_slot) {
                gwn_bvh_child<Real> child{};
                child.reference = child_reference[child_slot];
                if (!child.is_internal() || child_distance2[child_slot] >= best_dist2)
                    continue;
                if (next_reference != 0u) {
                    if (stack_size >= StackCapacity) {
                        overflow_callback();
                        return std::numeric_limits<Real>::quiet_NaN();
                    }
                    stack[stack_size++] = next_reference;
                }
                next_reference = child.reference;
            }
        }

        if (next_reference != 0u) {
            reference = next_reference;
            continue;
        }
        if (stack_size == 0)
            break;
        reference = stack[--stack_size];
    }

    return sqrt(best_dist2);
}

/// \brief Invoke canonical unsigned-distance traversal for one batch element.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
struct gwn_unsigned_distance_batch_functor {
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_distance{};
    Real culling_band{};
    OverflowCallback overflow_callback{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        // Batch validation runs once on the host. The point implementation can use the canonical
        // spans directly for every query in this launch.
        out_distance[query_id] =
            gwn_unsigned_distance_impl<Width, Real, Index, StackCapacity, OverflowCallback>(
                bvh, query_x[query_id], query_y[query_id], query_z[query_id], culling_band,
                overflow_callback
            );
    }
};

} // namespace detail
} // namespace gwn
