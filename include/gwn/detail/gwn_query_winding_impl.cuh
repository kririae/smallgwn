#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "../gwn_bvh.cuh"
#include "gwn_query_common_impl.cuh"
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

template <gwn_real_type Real> struct gwn_winding_and_gradient_result {
    Real winding{};
    gwn_query_vec3<Real> gradient{};
};

/// \brief Compute exact solid angle from one canonical triangle record.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline Real gwn_bvh_triangle_solid_angle_impl(
    gwn_bvh_triangle<Real> const &triangle, gwn_query_vec3<Real> const &query
) noexcept {
    gwn_query_vec3<Real> const a(triangle.v0_x, triangle.v0_y, triangle.v0_z);
    gwn_query_vec3<Real> const b(
        triangle.v0_x + triangle.e1_x, triangle.v0_y + triangle.e1_y, triangle.v0_z + triangle.e1_z
    );
    gwn_query_vec3<Real> const c(
        triangle.v0_x + triangle.e2_x, triangle.v0_y + triangle.e2_y, triangle.v0_z + triangle.e2_z
    );
    return gwn_signed_solid_angle_triangle_impl(a, b, c, query);
}

/// \brief Compute exact solid angle and gradient from one canonical triangle record.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_winding_and_gradient_result<Real>
gwn_bvh_triangle_winding_gradient_impl(
    gwn_bvh_triangle<Real> const &triangle, gwn_query_vec3<Real> const &query
) noexcept {
    gwn_query_vec3<Real> const a(triangle.v0_x, triangle.v0_y, triangle.v0_z);
    gwn_query_vec3<Real> const b(
        triangle.v0_x + triangle.e1_x, triangle.v0_y + triangle.e1_y, triangle.v0_z + triangle.e1_z
    );
    gwn_query_vec3<Real> const c(
        triangle.v0_x + triangle.e2_x, triangle.v0_y + triangle.e2_y, triangle.v0_z + triangle.e2_z
    );
    return {
        gwn_signed_solid_angle_triangle_impl(a, b, c, query),
        gwn_gradient_solid_angle_triangle_impl(a, b, c, query),
    };
}

/// \brief Evaluate Taylor winding, optionally with its analytic gradient.
template <
    bool ComputeGradient, int Order, int Width, gwn_real_type Real, gwn_index_type Index,
    int StackCapacity, typename OverflowCallback = gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline gwn_winding_and_gradient_result<Real> gwn_winding_taylor_impl(
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_accessor<Width, Order, Real, Index> const &moment, Real const qx, Real const qy,
    Real const qz, Real const accuracy_scale, OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_winding_taylor_impl supports Order 0, 1, and 2."
    );
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

    gwn_winding_and_gradient_result<Real> result{};

    auto const &taylor_nodes = moment.nodes;

    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    constexpr Real k_inv_4pi = Real(1) / (Real(4) * k_pi);
    Index stack[StackCapacity];
    int stack_size = 0;

    Real omega_sum = Real(0);
    gwn_query_vec3<Real> grad_sum(Real(0), Real(0), Real(0));
    Real const accuracy_scale2 = accuracy_scale * accuracy_scale;
    auto const set_overflow_nan_result = [&]() noexcept {
        Real const nan = std::numeric_limits<Real>::quiet_NaN();
        result.winding = nan;
        if constexpr (ComputeGradient)
            result.gradient = gwn_query_vec3<Real>(nan, nan, nan);
    };
    auto const finalize_result = [&]() noexcept {
        result.winding = omega_sum / (Real(4) * k_pi);
        if constexpr (ComputeGradient) {
            result.gradient = gwn_query_vec3<Real>(
                grad_sum.x * k_inv_4pi, grad_sum.y * k_inv_4pi, grad_sum.z * k_inv_4pi
            );
        }
    };

    gwn_query_vec3<Real> const query(qx, qy, qz);
    auto accumulate_leaf = [&](gwn_bvh_child<Real> const &leaf) noexcept {
        for (std::uint32_t primitive_offset = 0; primitive_offset < leaf.primitive_count();
             ++primitive_offset) {
            std::uint64_t const sorted_index = leaf.offset() + primitive_offset;
            if (sorted_index >= bvh.triangles.size())
                continue;
            auto const &triangle = bvh.triangles[static_cast<std::size_t>(sorted_index)];
            if constexpr (ComputeGradient) {
                auto const exact = gwn_bvh_triangle_winding_gradient_impl(triangle, query);
                omega_sum += exact.winding;
                grad_sum += exact.gradient;
            } else {
                omega_sum += gwn_bvh_triangle_solid_angle_impl(triangle, query);
            }
        }
    };

    if (bvh.has_leaf_root()) {
        accumulate_leaf(bvh.root);
        finalize_result();
        return result;
    }

    GWN_ASSERT(!taylor_nodes.empty(), "Taylor moment nodes are empty for an internal BVH root.");
    auto node_index = static_cast<Index>(bvh.root.offset());
    while (true) {
        if (!gwn_index_in_bounds(node_index, bvh.nodes.size()))
            break;

        auto const node_offset = static_cast<std::size_t>(node_index);
        auto const &node = bvh.nodes[node_offset];
        auto const &taylor = taylor_nodes[node_offset];

        GWN_DETAIL_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const &child = node.child(child_slot);
            if (!child.is_valid())
                continue;

            Real const qrx = qx - taylor.child_average_x[child_slot];
            Real const qry = qy - taylor.child_average_y[child_slot];
            Real const qrz = qz - taylor.child_average_z[child_slot];
            Real const qlength2 = qrx * qrx + qry * qry + qrz * qrz;

            bool descend = !(qlength2 > Real(0));
            if (!descend)
                descend = qlength2 <= taylor.child_max_p_dist2[child_slot] * accuracy_scale2;

            if (!descend) {
                // Squared distances implement |q - P| > beta * r without a square root. Equality
                // descends so the acceptance bound remains strict.
                Real qlength_m1 = Real(1) / sqrt(qlength2);
                if constexpr (std::is_same_v<Real, float>)
                    qlength_m1 = rsqrtf(qlength2);
                Real const qlength_m2 = qlength_m1 * qlength_m1;
                Real const qlength_m3 = qlength_m2 * qlength_m1;

                Real const qnx = qrx * qlength_m1;
                Real const qny = qry * qlength_m1;
                Real const qnz = qrz * qlength_m1;

                Real const Nx = taylor.child_n_x[child_slot];
                Real const Ny = taylor.child_n_y[child_slot];
                Real const Nz = taylor.child_n_z[child_slot];
                Real const rdotN = qnx * Nx + qny * Ny + qnz * Nz;

                // Normalizing q keeps the higher inverse powers inside the representable float
                // range while preserving the order-0 expansion.
                Real omega_approx = -qlength_m2 * rdotN;
                Real grad_approx_x = Real(0);
                Real grad_approx_y = Real(0);
                Real grad_approx_z = Real(0);
                if constexpr (ComputeGradient) {
                    grad_approx_x = qlength_m3 * (Real(3) * qnx * rdotN - Nx);
                    grad_approx_y = qlength_m3 * (Real(3) * qny * rdotN - Ny);
                    grad_approx_z = qlength_m3 * (Real(3) * qnz * rdotN - Nz);
                }

                if constexpr (Order >= 1) {
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

                    // Omega_1 uses the six symmetric Nij coefficients stored by the refit.
                    omega_approx += qlength_m3 * (T - Real(3) * S);

                    if constexpr (ComputeGradient) {
                        Real const qlength_m4 = qlength_m2 * qlength_m2;
                        Real const vu_x = Real(2) * Nij_xx * qnx + Nxy_Nyx * qny + Nzx_Nxz * qnz;
                        Real const vu_y = Nxy_Nyx * qnx + Real(2) * Nij_yy * qny + Nyz_Nzy * qnz;
                        Real const vu_z = Nzx_Nxz * qnx + Nyz_Nzy * qny + Real(2) * Nij_zz * qnz;
                        Real const coeff = Real(15) * S - Real(3) * T;

                        grad_approx_x += qlength_m4 * (coeff * qnx - Real(3) * vu_x);
                        grad_approx_y += qlength_m4 * (coeff * qny - Real(3) * vu_y);
                        grad_approx_z += qlength_m4 * (coeff * qnz - Real(3) * vu_z);
                    }
                }

                if constexpr (Order >= 2) {
                    Real const qlength_m4 = qlength_m2 * qlength_m2;
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

                    // Keep the stored symmetric coefficients in these grouped products.
                    // Regrouping changes float rounding and obscures coefficient-to-term mapping.
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

                    if constexpr (ComputeGradient) {
                        Real const qlength_m5 = qlength_m4 * qlength_m1;
                        Real const Px = Real(3) * nijk_xxx + Ayx + Azx;
                        Real const Py = Real(3) * nijk_yyy + Azy + Axy;
                        Real const Pz = Real(3) * nijk_zzz + Axz + Ayz;
                        Real const L2 = qnx * Px + qny * Py + qnz * Pz;
                        Real const C2 = qnx3 * nijk_xxx + qny3 * nijk_yyy + qnz3 * nijk_zzz +
                                        qnx * qny * qnz * spn + qnx2 * (qny * Axy + qnz * Axz) +
                                        qny2 * (qnz * Ayz + qnx * Ayx) +
                                        qnz2 * (qnx * Azx + qny * Azy);
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

                        grad_approx_x +=
                            qlength_m5 *
                            (Real(1.5) * Px - Real(7.5) * (qnx * L2 + Rx) + qnx * half_105_C);
                        grad_approx_y +=
                            qlength_m5 *
                            (Real(1.5) * Py - Real(7.5) * (qny * L2 + Ry) + qny * half_105_C);
                        grad_approx_z +=
                            qlength_m5 *
                            (Real(1.5) * Pz - Real(7.5) * (qnz * L2 + Rz) + qnz * half_105_C);
                    }
                }

                // Accept only a finite far-field expansion. Otherwise traversal descends until
                // leaf triangles contribute their exact solid-angle direct sum.
                if (std::isfinite(omega_approx)) {
                    omega_sum += omega_approx;
                    if constexpr (ComputeGradient) {
                        grad_sum.x += grad_approx_x;
                        grad_sum.y += grad_approx_y;
                        grad_sum.z += grad_approx_z;
                    }
                    continue;
                }
            }

            if (child.is_internal()) {
                if (stack_size >= StackCapacity) {
                    overflow_callback();
                    set_overflow_nan_result();
                    return result;
                }
                // Traversal only needs the internal node offset. The topology's Index-sized
                // offset keeps the default uint32_t stack compact across divergent warps.
                stack[stack_size++] = static_cast<Index>(child.offset());
                continue;
            }

            // A descended leaf contributes the exact solid-angle direct sum of its triangles.
            // Primitive IDs remain cold because winding and its gradient use oriented positions.
            if (child.is_leaf())
                accumulate_leaf(child);
        }

        if (stack_size == 0)
            break;
        node_index = stack[--stack_size];
    }

    finalize_result();
    return result;
}

/// \brief Sum exact solid angles over the canonical triangle sequence.
template <int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline Real gwn_winding_number_exact_impl(
    gwn_bvh_accessor<Width, Real, Index> const &bvh, Real const qx, Real const qy, Real const qz
) noexcept {
    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    gwn_query_vec3<Real> const query(qx, qy, qz);
    Real omega_sum = Real(0);
    // Exact winding has no spatial rejection. A direct linear sequence removes node loads, stack
    // traffic, and hierarchy-dependent omission risk while retaining the BVH's contiguous records.
    for (std::size_t triangle_index = 0; triangle_index < bvh.triangles.size(); ++triangle_index)
        omega_sum += gwn_bvh_triangle_solid_angle_impl(bvh.triangles[triangle_index], query);
    return omega_sum / (Real(4) * k_pi);
}

template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline Real gwn_winding_number_taylor_impl(
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_accessor<Width, Order, Real, Index> const &moment, Real const qx, Real const qy,
    Real const qz, Real const accuracy_scale, OverflowCallback const &overflow_callback = {}
) noexcept {
    return gwn_winding_taylor_impl<
               false, Order, Width, Real, Index, StackCapacity, OverflowCallback>(
               bvh, moment, qx, qy, qz, accuracy_scale, overflow_callback
    )
        .winding;
}

template <
    int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
struct gwn_winding_number_taylor_batch_functor {
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    gwn_bvh_moment_accessor<Width, Order, Real, Index> moment{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_winding{};
    Real accuracy_scale{};
    OverflowCallback overflow_callback{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        // The host launcher validates shared objects once. Threads retain far-field acceptance,
        // exact solid-angle leaf sums, and the stack overflow checks that vary per query.
        out_winding[query_id] = gwn_winding_number_taylor_impl<
            Order, Width, Real, Index, StackCapacity, OverflowCallback>(
            bvh, moment, query_x[query_id], query_y[query_id], query_z[query_id], accuracy_scale,
            overflow_callback
        );
    }
};

template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_winding_number_exact_batch_functor {
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_winding{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        out_winding[query_id] = gwn_winding_number_exact_impl(
            bvh, query_x[query_id], query_y[query_id], query_z[query_id]
        );
    }
};

} // namespace detail
} // namespace gwn
