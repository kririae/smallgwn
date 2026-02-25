#pragma once

/// \file gwn_query.cuh
/// \brief Spatial query primitives for triangle meshes on the GPU.
///
/// Public query surface:
///
/// | Query             | Batch (SoA)                                  | Single-point (`__device__`)
/// |
/// |-------------------|----------------------------------------------|-----------------------------------|
/// | Winding number    | `gwn_compute_winding_number_batch_bvh_taylor` | (detail only) | | Unsigned
/// distance | —                                            | `gwn_unsigned_distance_point_bvh` | |
/// Signed distance   | —                                            |
/// `gwn_signed_distance_point_bvh`   |
///
/// Winding-number queries use Taylor-accelerated BVH traversal.  Distance
/// queries use AABB-pruned BVH traversal; sign is determined by the exact
/// winding number.
///
/// \note Signed distance uses the generalised winding number to determine
///       inside/outside: a point whose winding number \f$\ge 0.5\f$ is
///       considered \e inside the surface and receives a negative sign.

#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"

#if !__has_include(<Eigen/Core>) || !__has_include(<Eigen/Geometry>)
#error "gwn_query.cuh requires Eigen/Core and Eigen/Geometry in the include path."
#endif

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gwn {

/// \brief 3-component column vector parameterised on scalar type \p Real.
template <gwn_real_type Real> using gwn_vec3 = Eigen::Matrix<Real, 3, 1>;

/// \brief Default capacity of the per-thread BVH traversal stack.
inline constexpr int k_gwn_default_traversal_stack_capacity = 64;

/// \brief Compute the signed solid angle subtended by triangle \p abc at
///        query point \p q.
///
/// The result is measured in steradians.  A full closed surface encloses
/// \f$4\pi\f$ steradians when the query point is inside the surface.
///
/// \tparam Real Floating-point scalar type (\c float or \c double).
/// \param a First vertex of the triangle.
/// \param b Second vertex of the triangle.
/// \param c Third vertex of the triangle.
/// \param q Query point.
/// \return Signed solid angle in steradians.
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_signed_solid_angle_triangle(
    gwn_vec3<Real> const &a, gwn_vec3<Real> const &b, gwn_vec3<Real> const &c,
    gwn_vec3<Real> const &q
) noexcept {
    gwn_vec3<Real> qa = a - q;
    gwn_vec3<Real> qb = b - q;
    gwn_vec3<Real> qc = c - q;

    Real const a_length = qa.norm();
    Real const b_length = qb.norm();
    Real const c_length = qc.norm();
    if (a_length == Real(0) || b_length == Real(0) || c_length == Real(0))
        return Real(0);

    qa /= a_length;
    qb /= b_length;
    qc /= c_length;

    Real const numerator = qa.dot((qb - qa).cross(qc - qa));
    if (numerator == Real(0))
        return Real(0);

    Real const denominator = Real(1) + qa.dot(qb) + qa.dot(qc) + qb.dot(qc);
    return Real(2) * atan2(numerator, denominator);
}

/// \brief Compute the squared Euclidean distance from point \p p to the
///        closest point on triangle \p abc.
///
/// Uses the Voronoi-region approach (Ericson, "Real-Time Collision Detection",
/// §5.1.5) to compute the exact closest point on—or inside—the triangle.
///
/// \tparam Real Floating-point scalar type.
/// \param p Query point.
/// \param a First vertex of the triangle.
/// \param b Second vertex of the triangle.
/// \param c Third vertex of the triangle.
/// \return Squared distance (\f$\ge 0\f$).
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_point_triangle_distance_squared(
    gwn_vec3<Real> const &p, gwn_vec3<Real> const &a, gwn_vec3<Real> const &b,
    gwn_vec3<Real> const &c
) noexcept {
    gwn_vec3<Real> const ab = b - a;
    gwn_vec3<Real> const ac = c - a;
    gwn_vec3<Real> const ap = p - a;

    Real const d1 = ab.dot(ap);
    Real const d2 = ac.dot(ap);
    if (d1 <= Real(0) && d2 <= Real(0))
        return ap.squaredNorm();

    gwn_vec3<Real> const bp = p - b;
    Real const d3 = ab.dot(bp);
    Real const d4 = ac.dot(bp);
    if (d3 >= Real(0) && d4 <= d3)
        return bp.squaredNorm();

    Real const vc = d1 * d4 - d3 * d2;
    if (vc <= Real(0) && d1 >= Real(0) && d3 <= Real(0)) {
        Real const v = d1 / (d1 - d3);
        return (p - (a + v * ab)).squaredNorm();
    }

    gwn_vec3<Real> const cp = p - c;
    Real const d5 = ab.dot(cp);
    Real const d6 = ac.dot(cp);
    if (d6 >= Real(0) && d5 <= d6)
        return cp.squaredNorm();

    Real const vb = d5 * d2 - d1 * d6;
    if (vb <= Real(0) && d2 >= Real(0) && d6 <= Real(0)) {
        Real const w = d2 / (d2 - d6);
        return (p - (a + w * ac)).squaredNorm();
    }

    Real const va = d3 * d6 - d5 * d4;
    if (va <= Real(0) && (d4 - d3) >= Real(0) && (d5 - d6) >= Real(0)) {
        Real const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + w * (c - b))).squaredNorm();
    }

    Real const denom = Real(1) / (va + vb + vc);
    Real const v = vb * denom;
    Real const w = vc * denom;
    return (p - (a + v * ab + w * ac)).squaredNorm();
}

namespace detail {

/// \brief Evaluate the signed solid angle of a single indexed primitive.
///
/// \tparam Real  Floating-point type.
/// \tparam Index Index type (\c uint32_t or \c uint64_t).
/// \param geometry   Geometry accessor.
/// \param primitive_id  Triangle index.
/// \param query   Query point.
/// \return Signed solid angle in steradians, or \c 0 on invalid input.
template <gwn_real_type Real, gwn_index_type Index>
__device__ inline Real gwn_triangle_solid_angle_from_primitive(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_vec3<Real> const &query
) noexcept {
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return Real(0);

    auto const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ic, geometry.vertex_count())) {
        return Real(0);
    }

    auto const a_index = static_cast<std::size_t>(ia);
    auto const b_index = static_cast<std::size_t>(ib);
    auto const c_index = static_cast<std::size_t>(ic);

    gwn_vec3<Real> const a(
        geometry.vertex_x[a_index], geometry.vertex_y[a_index], geometry.vertex_z[a_index]
    );
    gwn_vec3<Real> const b(
        geometry.vertex_x[b_index], geometry.vertex_y[b_index], geometry.vertex_z[b_index]
    );
    gwn_vec3<Real> const c(
        geometry.vertex_x[c_index], geometry.vertex_y[c_index], geometry.vertex_z[c_index]
    );
    return gwn_signed_solid_angle_triangle(a, b, c, query);
}

/// \brief Compute the squared distance from \p query to the triangle
///        identified by \p primitive_id in \p geometry.
///
/// \tparam Real  Floating-point type.
/// \tparam Index Index type.
/// \param geometry   Geometry accessor.
/// \param primitive_id  Triangle index.
/// \param query   Query point.
/// \return Squared distance, or <tt>numeric_limits<Real>::infinity()</tt> on
///         invalid input.
template <gwn_real_type Real, gwn_index_type Index>
__device__ inline Real gwn_triangle_distance_squared_from_primitive(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_vec3<Real> const &query
) noexcept {
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return std::numeric_limits<Real>::infinity();

    auto const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ic, geometry.vertex_count())) {
        return std::numeric_limits<Real>::infinity();
    }

    auto const a_index = static_cast<std::size_t>(ia);
    auto const b_index = static_cast<std::size_t>(ib);
    auto const c_index = static_cast<std::size_t>(ic);

    gwn_vec3<Real> const a(
        geometry.vertex_x[a_index], geometry.vertex_y[a_index], geometry.vertex_z[a_index]
    );
    gwn_vec3<Real> const b(
        geometry.vertex_x[b_index], geometry.vertex_y[b_index], geometry.vertex_z[b_index]
    );
    gwn_vec3<Real> const c(
        geometry.vertex_x[c_index], geometry.vertex_y[c_index], geometry.vertex_z[c_index]
    );
    return gwn_point_triangle_distance_squared(query, a, b, c);
}

/// \brief Squared minimum distance from a point to an axis-aligned box.
///
/// Returns 0 when the point lies inside the box.
template <gwn_real_type Real>
__host__ __device__ inline Real gwn_aabb_min_distance_squared(
    Real const qx, Real const qy, Real const qz, Real const min_x, Real const min_y,
    Real const min_z, Real const max_x, Real const max_y, Real const max_z
) noexcept {
    auto const clamp_delta = [](Real q, Real lo, Real hi) -> Real {
        if (q < lo)
            return lo - q;
        if (q > hi)
            return q - hi;
        return Real(0);
    };
    Real const dx = clamp_delta(qx, min_x, max_x);
    Real const dy = clamp_delta(qy, min_y, max_y);
    Real const dz = clamp_delta(qz, min_z, max_z);
    return dx * dx + dy * dy + dz * dz;
}

/// \brief Exact winding number via BVH traversal (internal only).
///
/// This function is used as the fallback path for the Taylor traversal
/// and for sign determination in signed-distance queries.
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
__device__ inline Real gwn_winding_number_point_bvh_exact(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh, Real const qx, Real const qy,
    Real const qz
) noexcept {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid() || !bvh.is_valid())
        return Real(0);

    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    Index stack[StackCapacity];
    int stack_size = 0;

    gwn_vec3<Real> const query(qx, qy, qz);
    Real omega_sum = Real(0);
    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        for (Index primitive_offset = 0; primitive_offset < bvh.root_count; ++primitive_offset) {
            Index const sorted_primitive_index = bvh.root_index + primitive_offset;
            if (!gwn_index_in_bounds(sorted_primitive_index, bvh.primitive_indices.size()))
                continue;
            Index const primitive_index =
                bvh.primitive_indices[static_cast<std::size_t>(sorted_primitive_index)];
            omega_sum += gwn_triangle_solid_angle_from_primitive<Real, Index>(
                geometry, primitive_index, query
            );
        }
        return omega_sum / (Real(4) * k_pi);
    }

    if (bvh.root_kind != gwn_bvh_child_kind::k_internal)
        return Real(0);

    stack[stack_size++] = bvh.root_index;
    while (stack_size > 0) {
        Index const node_index = stack[--stack_size];
        if (!gwn_index_in_bounds(node_index, bvh.nodes.size()))
            continue;

        gwn_bvh_topology_node_soa<Width, Index> const &node =
            bvh.nodes[static_cast<std::size_t>(node_index)];
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (child_kind == gwn_bvh_child_kind::k_invalid)
                continue;

            if (child_kind == gwn_bvh_child_kind::k_internal) {
                if (stack_size >= StackCapacity)
                    __trap();
                stack[stack_size++] = node.child_index[child_slot];
                continue;
            }

            if (child_kind != gwn_bvh_child_kind::k_leaf)
                continue;

            Index const begin = node.child_index[child_slot];
            Index const count = node.child_count[child_slot];
            for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
                Index const sorted_primitive_index = begin + primitive_offset;
                if (!gwn_index_in_bounds(sorted_primitive_index, bvh.primitive_indices.size()))
                    continue;
                Index const primitive_index =
                    bvh.primitive_indices[static_cast<std::size_t>(sorted_primitive_index)];
                omega_sum += gwn_triangle_solid_angle_from_primitive<Real, Index>(
                    geometry, primitive_index, query
                );
            }
        }
    }

    return omega_sum / (Real(4) * k_pi);
}

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_taylor_nodes_getter;

template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_taylor_nodes_getter<0, Width, Real, Index> {
    using span_type = cuda::std::span<gwn_bvh_taylor_node_soa<Width, 0, Real> const>;

    [[nodiscard]] __host__ __device__ static span_type
    get(gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree) {
        return data_tree.taylor_order0_nodes;
    }
};

template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_taylor_nodes_getter<1, Width, Real, Index> {
    using span_type = cuda::std::span<gwn_bvh_taylor_node_soa<Width, 1, Real> const>;

    [[nodiscard]] __host__ __device__ static span_type
    get(gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree) {
        return data_tree.taylor_order1_nodes;
    }
};

template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_taylor_nodes_getter<2, Width, Real, Index> {
    using span_type = cuda::std::span<gwn_bvh_taylor_node_soa<Width, 2, Real> const>;

    [[nodiscard]] __host__ __device__ static span_type
    get(gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree) {
        return data_tree.taylor_order2_nodes;
    }
};

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __host__ __device__
    typename gwn_taylor_nodes_getter<Order, Width, Real, Index>::span_type
    gwn_get_taylor_nodes(gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree) {
    return gwn_taylor_nodes_getter<Order, Width, Real, Index>::get(data_tree);
}

/// \brief Taylor-accelerated winding number (single point, detail only).
///
/// Falls back to exact traversal when the moment data is missing or
/// inconsistent.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
__device__ inline Real gwn_winding_number_point_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree, Real const qx, Real const qy,
    Real const qz, Real const accuracy_scale
) noexcept {
    static_assert(
        Order == 0 || Order == 1,
        "gwn_winding_number_point_bvh_taylor currently supports Order 0 and "
        "Order 1."
    );
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid() || !bvh.is_valid())
        return Real(0);
    if (!data_tree.is_valid_for(bvh) || !data_tree.template has_taylor_order<Order>())
        return gwn_winding_number_point_bvh_exact<Width, Real, Index, StackCapacity>(
            geometry, bvh, qx, qy, qz
        );

    auto const taylor_nodes = gwn_get_taylor_nodes<Order, Width>(data_tree);
    if (taylor_nodes.size() != bvh.nodes.size())
        return gwn_winding_number_point_bvh_exact<Width, Real, Index, StackCapacity>(
            geometry, bvh, qx, qy, qz
        );

    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    Index stack[StackCapacity];
    int stack_size = 0;

    Real omega_sum = Real(0);
    Real const accuracy_scale2 = accuracy_scale * accuracy_scale;
    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        for (Index primitive_offset = 0; primitive_offset < bvh.root_count; ++primitive_offset) {
            Index const sorted_primitive_index = bvh.root_index + primitive_offset;
            if (!gwn_index_in_bounds(sorted_primitive_index, bvh.primitive_indices.size()))
                continue;
            Index const primitive_index =
                bvh.primitive_indices[static_cast<std::size_t>(sorted_primitive_index)];
            omega_sum += gwn_triangle_solid_angle_from_primitive<Real, Index>(
                geometry, primitive_index, gwn_vec3<Real>(qx, qy, qz)
            );
        }
        return omega_sum / (Real(4) * k_pi);
    }

    if (bvh.root_kind != gwn_bvh_child_kind::k_internal)
        return Real(0);

    stack[stack_size++] = bvh.root_index;
    while (stack_size > 0) {
        Index const node_index = stack[--stack_size];
        if (!gwn_index_in_bounds(node_index, bvh.nodes.size()))
            continue;

        gwn_bvh_topology_node_soa<Width, Index> const &node =
            bvh.nodes[static_cast<std::size_t>(node_index)];
        auto const &taylor = taylor_nodes[static_cast<std::size_t>(node_index)];
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (child_kind == gwn_bvh_child_kind::k_invalid)
                continue;

            if (child_kind != gwn_bvh_child_kind::k_internal &&
                child_kind != gwn_bvh_child_kind::k_leaf) {
                continue;
            }

            Real const qrx = qx - taylor.child_average_x[child_slot];
            Real const qry = qy - taylor.child_average_y[child_slot];
            Real const qrz = qz - taylor.child_average_z[child_slot];
            Real const qlength2 = qrx * qrx + qry * qry + qrz * qrz;

            bool descend = !(qlength2 > Real(0));
            if (!descend)
                descend = qlength2 <= taylor.child_max_p_dist2[child_slot] * accuracy_scale2;

            if (!descend) {
                Real const qlength_m2 = Real(1) / qlength2;
                Real const qlength_m1 = sqrt(qlength_m2);

                Real const qnx = qrx * qlength_m1;
                Real const qny = qry * qlength_m1;
                Real const qnz = qrz * qlength_m1;
                Real omega_approx = -qlength_m2 * (qnx * taylor.child_n_x[child_slot] +
                                                   qny * taylor.child_n_y[child_slot] +
                                                   qnz * taylor.child_n_z[child_slot]);

                if constexpr (Order >= 1) {
                    Real const qxx = qnx * qnx;
                    Real const qyy = qny * qny;
                    Real const qzz = qnz * qnz;
                    Real const omega_1 =
                        qlength_m2 * qlength_m1 *
                        ((taylor.child_nij_xx[child_slot] + taylor.child_nij_yy[child_slot] +
                          taylor.child_nij_zz[child_slot]) -
                         Real(3) * (qxx * taylor.child_nij_xx[child_slot] +
                                    qyy * taylor.child_nij_yy[child_slot] +
                                    qzz * taylor.child_nij_zz[child_slot] +
                                    qnx * qny * taylor.child_nxy_nyx[child_slot] +
                                    qnx * qnz * taylor.child_nzx_nxz[child_slot] +
                                    qny * qnz * taylor.child_nyz_nzy[child_slot]));
                    omega_approx += omega_1;
                }

                if (isfinite(omega_approx)) {
                    omega_sum += omega_approx;
                    continue;
                }
                descend = true;
            }

            if (child_kind == gwn_bvh_child_kind::k_internal) {
                if (stack_size >= StackCapacity)
                    __trap();
                stack[stack_size++] = node.child_index[child_slot];
                continue;
            }

            Index const begin = node.child_index[child_slot];
            Index const count = node.child_count[child_slot];
            gwn_vec3<Real> const query(qx, qy, qz);
            for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
                Index const sorted_primitive_index = begin + primitive_offset;
                if (!gwn_index_in_bounds(sorted_primitive_index, bvh.primitive_indices.size()))
                    continue;
                Index const primitive_index =
                    bvh.primitive_indices[static_cast<std::size_t>(sorted_primitive_index)];
                omega_sum += gwn_triangle_solid_angle_from_primitive<Real, Index>(
                    geometry, primitive_index, query
                );
            }
        }
    }

    return omega_sum / (Real(4) * k_pi);
}

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
struct gwn_winding_number_batch_bvh_taylor_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> bvh{};
    gwn_bvh_moment_tree_accessor<Width, Real, Index> data_tree{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> output{};
    Real accuracy_scale{};

    __device__ void operator()(std::size_t const query_id) const {
        output[query_id] =
            gwn_winding_number_point_bvh_taylor<Order, Width, Real, Index, StackCapacity>(
                geometry, bvh, data_tree, query_x[query_id], query_y[query_id], query_z[query_id],
                accuracy_scale
            );
    }
};

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
[[nodiscard]] inline gwn_winding_number_batch_bvh_taylor_functor<
    Order, Width, Real, Index, StackCapacity>
gwn_make_winding_number_batch_bvh_taylor_functor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale
) {
    return gwn_winding_number_batch_bvh_taylor_functor<Order, Width, Real, Index, StackCapacity>{
        geometry, bvh, data_tree, query_x, query_y, query_z, output, accuracy_scale
    };
}

} // namespace detail

/// \brief Compute the unsigned distance from a query point to the closest
///        triangle using BVH + AABB-accelerated traversal.
///
/// Traverses the BVH depth-first, pruning child sub-trees whose AABB minimum
/// distance exceeds the current best distance.  Requires both a topology
/// accessor and a matching AABB tree accessor.
///
/// \tparam Width          BVH fan-out.
/// \tparam Real           Floating-point type.
/// \tparam Index          Index type.
/// \tparam StackCapacity  Maximum depth of the per-thread traversal stack.
/// \param geometry   Geometry accessor.
/// \param bvh        BVH topology accessor.
/// \param aabb_tree  AABB payload accessor aligned to \p bvh.
/// \param qx  Query X coordinate.
/// \param qy  Query Y coordinate.
/// \param qz  Query Z coordinate.
/// \return Non-negative distance.  Returns \c 0 when any accessor is invalid.
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
__device__ inline Real gwn_unsigned_distance_point_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, Real const qx, Real const qy,
    Real const qz
) noexcept {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid() || !bvh.is_valid() || !aabb_tree.is_valid_for(bvh))
        return Real(0);

    gwn_vec3<Real> const query(qx, qy, qz);
    Real best_dist2 = std::numeric_limits<Real>::infinity();

    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        for (Index off = 0; off < bvh.root_count; ++off) {
            Index const si = bvh.root_index + off;
            if (!gwn_index_in_bounds(si, bvh.primitive_indices.size()))
                continue;
            Index const pi = bvh.primitive_indices[static_cast<std::size_t>(si)];
            Real const d2 = detail::gwn_triangle_distance_squared_from_primitive<Real, Index>(
                geometry, pi, query
            );
            if (d2 < best_dist2)
                best_dist2 = d2;
        }
        return sqrt(best_dist2);
    }

    if (bvh.root_kind != gwn_bvh_child_kind::k_internal)
        return Real(0);

    Index stack[StackCapacity];
    int stack_size = 0;
    stack[stack_size++] = bvh.root_index;

    while (stack_size > 0) {
        Index const node_index = stack[--stack_size];
        if (!gwn_index_in_bounds(node_index, bvh.nodes.size()))
            continue;

        auto const &topo_node = bvh.nodes[static_cast<std::size_t>(node_index)];
        auto const &aabb_node = aabb_tree.nodes[static_cast<std::size_t>(node_index)];

        GWN_PRAGMA_UNROLL
        for (int s = 0; s < Width; ++s) {
            auto const kind = static_cast<gwn_bvh_child_kind>(topo_node.child_kind[s]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;

            Real const box_dist2 = detail::gwn_aabb_min_distance_squared(
                qx, qy, qz, aabb_node.child_min_x[s], aabb_node.child_min_y[s],
                aabb_node.child_min_z[s], aabb_node.child_max_x[s], aabb_node.child_max_y[s],
                aabb_node.child_max_z[s]
            );
            if (box_dist2 >= best_dist2)
                continue;

            if (kind == gwn_bvh_child_kind::k_internal) {
                if (stack_size >= StackCapacity)
                    __trap();
                stack[stack_size++] = topo_node.child_index[s];
                continue;
            }

            if (kind != gwn_bvh_child_kind::k_leaf)
                continue;

            Index const begin = topo_node.child_index[s];
            Index const count = topo_node.child_count[s];
            for (Index off = 0; off < count; ++off) {
                Index const si = begin + off;
                if (!gwn_index_in_bounds(si, bvh.primitive_indices.size()))
                    continue;
                Index const pi = bvh.primitive_indices[static_cast<std::size_t>(si)];
                Real const d2 = detail::gwn_triangle_distance_squared_from_primitive<Real, Index>(
                    geometry, pi, query
                );
                if (d2 < best_dist2)
                    best_dist2 = d2;
            }
        }
    }

    return sqrt(best_dist2);
}

/// \brief Compute the signed distance from a query point to the mesh using
///        BVH-accelerated traversal.
///
/// Unsigned distance is computed via AABB-pruned BVH traversal, and the sign
/// is determined by an exact winding-number BVH traversal.
///
/// \note Two independent traversals are performed.  This is simpler and more
///       robust than a single fused traversal because the two queries use
///       fundamentally different pruning criteria (AABB distance vs. solid
///       angle).
///
/// \tparam Width          BVH fan-out.
/// \tparam Real           Floating-point type.
/// \tparam Index          Index type.
/// \tparam StackCapacity  Maximum depth of the per-thread traversal stack
///                        (shared by both the distance and winding-number
///                        traversals).
/// \param geometry   Geometry accessor.
/// \param bvh        BVH topology accessor.
/// \param aabb_tree  AABB payload accessor aligned to \p bvh.
/// \param qx  Query X coordinate.
/// \param qy  Query Y coordinate.
/// \param qz  Query Z coordinate.
/// \return Signed distance (negative inside, positive outside).  Returns
///         \c 0 when any accessor is invalid.
template <
    int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
__device__ inline Real gwn_signed_distance_point_bvh(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree, Real const qx, Real const qy,
    Real const qz
) noexcept {
    Real const dist = gwn_unsigned_distance_point_bvh<Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, qx, qy, qz
    );
    Real const wn = detail::gwn_winding_number_point_bvh_exact<Width, Real, Index, StackCapacity>(
        geometry, bvh, qx, qy, qz
    );
    return wn >= Real(0.5) ? -dist : dist;
}

/// \brief Compute winding numbers for a batch using Taylor-accelerated BVH
///        traversal.
///
/// For each query, the BVH is traversed depth-first.  When a child satisfies
/// the far-field criterion the contribution is approximated via the Taylor
/// expansion of the requested \p Order; otherwise the traversal descends.
///
/// \remark Falls back to exact child descent per node when the approximation
///         criterion fails.
///
/// \tparam Order          Taylor expansion order (0 or 1).
/// \tparam Width          BVH fan-out.
/// \tparam Real           Floating-point type.
/// \tparam Index          Index type (default: \c uint32_t).
/// \tparam StackCapacity  Per-thread traversal stack depth.
/// \param geometry        Geometry accessor.
/// \param bvh             BVH topology accessor.
/// \param data_tree       Moment payload accessor.
/// \param query_x         Device-resident X coordinates.
/// \param query_y         Device-resident Y coordinates.
/// \param query_z         Device-resident Z coordinates.
/// \param output          Device-resident output buffer.
/// \param accuracy_scale  Far-field acceptance multiplier (default: 2).
/// \param stream          CUDA stream.
/// \return \c gwn_status::ok() on success.
template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_winding_number_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream()
) noexcept {
    static_assert(
        Order == 0 || Order == 1, "gwn_compute_winding_number_batch_bvh_taylor currently supports "
                                  "Order 0 and Order 1."
    );
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
    if (!data_tree.is_valid_for(bvh))
        return gwn_status::invalid_argument("BVH data tree is invalid for the given topology.");
    if (!data_tree.template has_taylor_order<Order>())
        return gwn_status::invalid_argument("BVH data tree is missing requested Taylor-order data."
        );
    if (query_x.size() != query_y.size() || query_x.size() != query_z.size())
        return gwn_status::invalid_argument("Query SoA spans must have identical lengths.");
    if (query_x.size() != output.size())
        return gwn_status::invalid_argument("Output span size must match query count.");
    if (!gwn_span_has_storage(query_x) || !gwn_span_has_storage(query_y) ||
        !gwn_span_has_storage(query_z) || !gwn_span_has_storage(output)) {
        return gwn_status::invalid_argument(
            "Query/output spans must use non-null storage when non-empty."
        );
    }
    if (output.empty())
        return gwn_status::ok();

    constexpr int k_block_size = detail::k_gwn_default_block_size;
    return detail::gwn_launch_linear_kernel<k_block_size>(
        output.size(),
        detail::gwn_make_winding_number_batch_bvh_taylor_functor<
            Order, Width, Real, Index, StackCapacity>(
            geometry, bvh, data_tree, query_x, query_y, query_z, output, accuracy_scale
        ),
        stream
    );
}

/// \brief Width-4 convenience wrapper for Taylor BVH batch winding-number
///        queries.
///
/// \copydetails gwn_compute_winding_number_batch_bvh_taylor
template <
    int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_winding_number_batch_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> const &bvh,
    gwn_bvh_moment4_accessor<Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale = Real(2), cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, StackCapacity>(
        geometry, bvh, data_tree, query_x, query_y, query_z, output, accuracy_scale, stream
    );
}

} // namespace gwn
