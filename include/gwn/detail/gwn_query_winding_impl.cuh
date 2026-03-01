#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_geometry_impl.cuh"
#include "gwn_query_vec3_impl.cuh"

namespace gwn {
namespace detail {

// Per-triangle gradient of solid angle (Biot-Savart formula).

template <gwn_real_type Real>
__host__ __device__ inline gwn_query_vec3<Real> gwn_gradient_solid_angle_triangle_impl(
    gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b, gwn_query_vec3<Real> const &c,
    gwn_query_vec3<Real> const &q
) noexcept {
    gwn_query_vec3<Real> const ga = a - q;
    gwn_query_vec3<Real> const gb = b - q;
    gwn_query_vec3<Real> const gc = c - q;

    Real const la = gwn_query_norm(ga);
    Real const lb = gwn_query_norm(gb);
    Real const lc = gwn_query_norm(gc);

    gwn_query_vec3<Real> const zero(Real(0), Real(0), Real(0));
    if (la == Real(0) || lb == Real(0) || lc == Real(0))
        return zero;

    gwn_query_vec3<Real> const ga_hat = ga / la;
    gwn_query_vec3<Real> const gb_hat = gb / lb;
    gwn_query_vec3<Real> const gc_hat = gc / lc;

    auto edge_term = [](gwn_query_vec3<Real> const &gi, gwn_query_vec3<Real> const &gj,
                        gwn_query_vec3<Real> const &gi_hat,
                        gwn_query_vec3<Real> const &gj_hat) -> gwn_query_vec3<Real> {
        gwn_query_vec3<Real> const cross = gwn_query_cross(gi, gj);
        Real const cross_sq = gwn_query_squared_norm(cross);
        if (cross_sq == Real(0))
            return gwn_query_vec3<Real>(Real(0), Real(0), Real(0));

        gwn_query_vec3<Real> const diff = gi - gj;
        gwn_query_vec3<Real> const diff_hat = gi_hat - gj_hat;
        Real const scalar = gwn_query_dot(diff, diff_hat) / cross_sq;
        return scalar * cross;
    };

    gwn_query_vec3<Real> result = edge_term(ga, gb, ga_hat, gb_hat);
    result += edge_term(gb, gc, gb_hat, gc_hat);
    result += edge_term(gc, ga, gc_hat, ga_hat);

    return result;
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline gwn_query_vec3<Real> gwn_triangle_gradient_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_query_vec3<Real> const &query
) noexcept {
    gwn_query_vec3<Real> const zero(Real(0), Real(0), Real(0));
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return zero;

    auto const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ic, geometry.vertex_count())) {
        return zero;
    }

    auto const a_index = static_cast<std::size_t>(ia);
    auto const b_index = static_cast<std::size_t>(ib);
    auto const c_index = static_cast<std::size_t>(ic);

    gwn_query_vec3<Real> const a(
        geometry.vertex_x[a_index], geometry.vertex_y[a_index], geometry.vertex_z[a_index]
    );
    gwn_query_vec3<Real> const b(
        geometry.vertex_x[b_index], geometry.vertex_y[b_index], geometry.vertex_z[b_index]
    );
    gwn_query_vec3<Real> const c(
        geometry.vertex_x[c_index], geometry.vertex_y[c_index], geometry.vertex_z[c_index]
    );
    return gwn_gradient_solid_angle_triangle_impl(a, b, c, query);
}

template <gwn_real_type Real> struct gwn_winding_and_gradient_result {
    Real winding{};
    gwn_query_vec3<Real> gradient{};
};

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline gwn_winding_and_gradient_result<Real>
gwn_winding_and_gradient_point_bvh_taylor_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree, Real const qx,
    Real const qy, Real const qz, Real const accuracy_scale
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_winding_and_gradient currently supports Order 0, 1, and 2."
    );
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    gwn_winding_and_gradient_result<Real> result{};
    gwn_query_vec3<Real> const zero(Real(0), Real(0), Real(0));

    if (!geometry.is_valid() || !bvh.is_valid())
        return result;
    if (!data_tree.is_valid_for(bvh))
        return result;

    auto const &taylor_nodes = data_tree.nodes;
    if (taylor_nodes.size() != bvh.nodes.size())
        return result;

    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    constexpr Real k_inv_4pi = Real(1) / (Real(4) * k_pi);
    Index stack[StackCapacity];
    int stack_size = 0;

    Real omega_sum = Real(0);
    gwn_query_vec3<Real> grad_sum(Real(0), Real(0), Real(0));
    Real const accuracy_scale2 = accuracy_scale * accuracy_scale;

    // Handle root-is-leaf case.
    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        gwn_query_vec3<Real> const query(qx, qy, qz);
        for (Index primitive_offset = 0; primitive_offset < bvh.root_count; ++primitive_offset) {
            Index const sorted_primitive_index = bvh.root_index + primitive_offset;
            if (!gwn_index_in_bounds(sorted_primitive_index, bvh.primitive_indices.size()))
                continue;
            Index const primitive_index =
                bvh.primitive_indices[static_cast<std::size_t>(sorted_primitive_index)];
            omega_sum += gwn_triangle_solid_angle_from_primitive_impl<Real, Index>(
                geometry, primitive_index, query
            );
            grad_sum += gwn_triangle_gradient_from_primitive_impl<Real, Index>(
                geometry, primitive_index, query
            );
        }
        result.winding = omega_sum / (Real(4) * k_pi);
        result.gradient = gwn_query_vec3<Real>(
            grad_sum.x * k_inv_4pi, grad_sum.y * k_inv_4pi, grad_sum.z * k_inv_4pi
        );
        return result;
    }

    if (bvh.root_kind != gwn_bvh_child_kind::k_internal)
        return result;

    GWN_ASSERT(!taylor_nodes.empty(), "winding_and_gradient: taylor_nodes empty for internal root");
    stack[stack_size++] = bvh.root_index;
    while (stack_size > 0) {
        Index const node_index = stack[--stack_size];
        GWN_ASSERT(stack_size >= 0, "winding_and_gradient: stack underflow");
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
                child_kind != gwn_bvh_child_kind::k_leaf)
                continue;

            Real const qrx = qx - taylor.child_average_x[child_slot];
            Real const qry = qy - taylor.child_average_y[child_slot];
            Real const qrz = qz - taylor.child_average_z[child_slot];
            Real const qlength2 = qrx * qrx + qry * qry + qrz * qrz;

            bool descend = !(qlength2 > Real(0));
            if (!descend)
                descend = qlength2 <= taylor.child_max_p_dist2[child_slot] * accuracy_scale2;

            if (!descend) {
                Real qlength_m1 = Real(1) / sqrt(qlength2);
                if constexpr (std::is_same_v<Real, float>)
                    qlength_m1 = rsqrtf(qlength2);
                Real const qlength_m2 = qlength_m1 * qlength_m1;
                Real const qlength_m3 = qlength_m2 * qlength_m1;

                Real const qnx = qrx * qlength_m1;
                Real const qny = qry * qlength_m1;
                Real const qnz = qrz * qlength_m1;

                // Shared moment loads
                Real const Nx = taylor.child_n_x[child_slot];
                Real const Ny = taylor.child_n_y[child_slot];
                Real const Nz = taylor.child_n_z[child_slot];
                Real const rdotN = qnx * Nx + qny * Ny + qnz * Nz;

                // Order-0 winding: ω₀ = −(1/r²)(r̂·N)
                Real omega_approx = -qlength_m2 * rdotN;

                // Order-0 gradient: ∇ω₀ = (1/r³)[3r̂(r̂·N) − N]
                Real grad_approx_x = qlength_m3 * (Real(3) * qnx * rdotN - Nx);
                Real grad_approx_y = qlength_m3 * (Real(3) * qny * rdotN - Ny);
                Real grad_approx_z = qlength_m3 * (Real(3) * qnz * rdotN - Nz);

                if constexpr (Order >= 1) {
                    Real const qlength_m4 = qlength_m2 * qlength_m2;
                    Real const qxx = qnx * qnx;
                    Real const qyy = qny * qny;
                    Real const qzz = qnz * qnz;

                    Real const Nij_xx = taylor.child_nij_xx[child_slot];
                    Real const Nij_yy = taylor.child_nij_yy[child_slot];
                    Real const Nij_zz = taylor.child_nij_zz[child_slot];
                    Real const Nxy_Nyx = taylor.child_nxy_nyx[child_slot];
                    Real const Nyz_Nzy = taylor.child_nyz_nzy[child_slot];
                    Real const Nzx_Nxz = taylor.child_nzx_nxz[child_slot];

                    Real const T = Nij_xx + Nij_yy + Nij_zz;
                    Real const S = qxx * Nij_xx + qyy * Nij_yy + qzz * Nij_zz +
                                   qnx * qny * Nxy_Nyx + qnx * qnz * Nzx_Nxz + qny * qnz * Nyz_Nzy;

                    // Winding order-1
                    omega_approx += qlength_m3 * (T - Real(3) * S);

                    // Gradient order-1
                    Real const vu_x = Real(2) * Nij_xx * qnx + Nxy_Nyx * qny + Nzx_Nxz * qnz;
                    Real const vu_y = Nxy_Nyx * qnx + Real(2) * Nij_yy * qny + Nyz_Nzy * qnz;
                    Real const vu_z = Nzx_Nxz * qnx + Nyz_Nzy * qny + Real(2) * Nij_zz * qnz;
                    Real const coeff = Real(15) * S - Real(3) * T;

                    grad_approx_x += qlength_m4 * (coeff * qnx - Real(3) * vu_x);
                    grad_approx_y += qlength_m4 * (coeff * qny - Real(3) * vu_y);
                    grad_approx_z += qlength_m4 * (coeff * qnz - Real(3) * vu_z);
                }

                if constexpr (Order >= 2) {
                    Real const qlength_m4 = qlength_m2 * qlength_m2;
                    Real const qlength_m5 = qlength_m4 * qlength_m1;
                    Real const qnx2 = qnx * qnx;
                    Real const qny2 = qny * qny;
                    Real const qnz2 = qnz * qnz;
                    Real const qnx3 = qnx2 * qnx;
                    Real const qny3 = qny2 * qny;
                    Real const qnz3 = qnz2 * qnz;

                    Real const nijk_xxx = taylor.child_nijk_xxx[child_slot];
                    Real const nijk_yyy = taylor.child_nijk_yyy[child_slot];
                    Real const nijk_zzz = taylor.child_nijk_zzz[child_slot];
                    Real const spn = taylor.child_sum_permute_nxyz[child_slot];

                    Real const Axy = taylor.child_2nxxy_nyxx[child_slot];
                    Real const Axz = taylor.child_2nxxz_nzxx[child_slot];
                    Real const Ayz = taylor.child_2nyyz_nzyy[child_slot];
                    Real const Ayx = taylor.child_2nyyx_nxyy[child_slot];
                    Real const Azx = taylor.child_2nzzx_nxzz[child_slot];
                    Real const Azy = taylor.child_2nzzy_nyzz[child_slot];

                    Real const Px = Real(3) * nijk_xxx + Ayx + Azx;
                    Real const Py = Real(3) * nijk_yyy + Azy + Axy;
                    Real const Pz = Real(3) * nijk_zzz + Axz + Ayz;

                    Real const L2 = qnx * Px + qny * Py + qnz * Pz;

                    Real const C2 = qnx3 * nijk_xxx + qny3 * nijk_yyy + qnz3 * nijk_zzz +
                                    qnx * qny * qnz * spn + qnx2 * (qny * Axy + qnz * Axz) +
                                    qny2 * (qnz * Ayz + qnx * Ayx) + qnz2 * (qnx * Azx + qny * Azy);

                    // Winding order-2 (uses temp variables matching original)
                    Real const temp0_x = Ayx + Azx;
                    Real const temp0_y = Azy + Axy;
                    Real const temp0_z = Axz + Ayz;
                    Real const temp1_x = qny * Axy + qnz * Axz;
                    Real const temp1_y = qnz * Ayz + qnx * Ayx;
                    Real const temp1_z = qnx * Azx + qny * Azy;

                    omega_approx +=
                        qlength_m4 *
                        (Real(1.5) * (qnx * (Real(3) * nijk_xxx + temp0_x) +
                                      qny * (Real(3) * nijk_yyy + temp0_y) +
                                      qnz * (Real(3) * nijk_zzz + temp0_z)) -
                         Real(7.5) * (qnx3 * nijk_xxx + qny3 * nijk_yyy + qnz3 * nijk_zzz +
                                      qnx * qny * qnz * spn + qnx2 * temp1_x + qny2 * temp1_y +
                                      qnz2 * temp1_z));

                    // Gradient order-2
                    Real const Rx = Real(3) * nijk_xxx * qnx2 + Real(2) * Axy * qnx * qny +
                                    Real(2) * Axz * qnx * qnz + Ayx * qny2 + Azx * qnz2 +
                                    spn * qny * qnz;
                    Real const Ry = Real(3) * nijk_yyy * qny2 + Real(2) * Ayx * qnx * qny +
                                    Real(2) * Ayz * qny * qnz + Axy * qnx2 + Azy * qnz2 +
                                    spn * qnx * qnz;
                    Real const Rz = Real(3) * nijk_zzz * qnz2 + Real(2) * Azx * qnx * qnz +
                                    Real(2) * Azy * qny * qnz + Axz * qnx2 + Ayz * qny2 +
                                    spn * qnx * qny;

                    Real const half_105_C = Real(52.5) * C2;

                    grad_approx_x += qlength_m5 * (Real(1.5) * Px - Real(7.5) * (qnx * L2 + Rx) +
                                                   qnx * half_105_C);
                    grad_approx_y += qlength_m5 * (Real(1.5) * Py - Real(7.5) * (qny * L2 + Ry) +
                                                   qny * half_105_C);
                    grad_approx_z += qlength_m5 * (Real(1.5) * Pz - Real(7.5) * (qnz * L2 + Rz) +
                                                   qnz * half_105_C);
                }

                if (isfinite(omega_approx) && isfinite(grad_approx_x) && isfinite(grad_approx_y) &&
                    isfinite(grad_approx_z)) {
                    omega_sum += omega_approx;
                    grad_sum.x += grad_approx_x;
                    grad_sum.y += grad_approx_y;
                    grad_sum.z += grad_approx_z;
                    continue;
                }
                descend = true;
            }

            if (child_kind == gwn_bvh_child_kind::k_internal) {
                if (stack_size >= StackCapacity)
                    gwn_trap();
                stack[stack_size++] = node.child_index[child_slot];
                continue;
            }

            // Leaf: brute-force per-triangle.
            Index const begin = node.child_index[child_slot];
            Index const count = node.child_count[child_slot];
            gwn_query_vec3<Real> const query(qx, qy, qz);
            for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
                Index const sorted_primitive_index = begin + primitive_offset;
                if (!gwn_index_in_bounds(sorted_primitive_index, bvh.primitive_indices.size()))
                    continue;
                Index const primitive_index =
                    bvh.primitive_indices[static_cast<std::size_t>(sorted_primitive_index)];
                omega_sum += gwn_triangle_solid_angle_from_primitive_impl<Real, Index>(
                    geometry, primitive_index, query
                );
                grad_sum += gwn_triangle_gradient_from_primitive_impl<Real, Index>(
                    geometry, primitive_index, query
                );
            }
        }
    }

    result.winding = omega_sum / (Real(4) * k_pi);
    result.gradient = gwn_query_vec3<Real>(
        grad_sum.x * k_inv_4pi, grad_sum.y * k_inv_4pi, grad_sum.z * k_inv_4pi
    );
    return result;
}

template <int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline Real gwn_winding_number_point_bvh_exact_impl(
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

    gwn_query_vec3<Real> const query(qx, qy, qz);
    Real omega_sum = Real(0);
    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        for (Index primitive_offset = 0; primitive_offset < bvh.root_count; ++primitive_offset) {
            Index const sorted_primitive_index = bvh.root_index + primitive_offset;
            if (!gwn_index_in_bounds(sorted_primitive_index, bvh.primitive_indices.size()))
                continue;
            Index const primitive_index =
                bvh.primitive_indices[static_cast<std::size_t>(sorted_primitive_index)];
            omega_sum += gwn_triangle_solid_angle_from_primitive_impl<Real, Index>(
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
        GWN_ASSERT(stack_size >= 0, "winding exact: stack underflow");
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
                    gwn_trap();
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
                omega_sum += gwn_triangle_solid_angle_from_primitive_impl<Real, Index>(
                    geometry, primitive_index, query
                );
            }
        }
    }

    return omega_sum / (Real(4) * k_pi);
}

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline Real gwn_winding_number_point_bvh_taylor_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree, Real const qx,
    Real const qy, Real const qz, Real const accuracy_scale
) noexcept {
    return gwn_winding_and_gradient_point_bvh_taylor_impl<Order, Width, Real, Index, StackCapacity>(
               geometry, bvh, data_tree, qx, qy, qz, accuracy_scale
    )
        .winding;
}

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
struct gwn_winding_number_batch_bvh_taylor_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> bvh{};
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> data_tree{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> output{};
    Real accuracy_scale{};

    __device__ void operator()(std::size_t const query_id) const {
        output[query_id] =
            gwn_winding_number_point_bvh_taylor_impl<Order, Width, Real, Index, StackCapacity>(
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
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &data_tree,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    Real const accuracy_scale
) {
    return gwn_winding_number_batch_bvh_taylor_functor<Order, Width, Real, Index, StackCapacity>{
        geometry, bvh, data_tree, query_x, query_y, query_z, output, accuracy_scale
    };
}

} // namespace detail
} // namespace gwn
