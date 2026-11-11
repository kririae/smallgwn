#pragma once

#include <cuda/std/cmath>

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

/// \brief Smallest product magnitude that rounded solid-angle terms accept, `16 * min / epsilon`
///        (new term: range floor).
///
/// Products above it are normal with room for the rounding of later sums, so their underflow
/// errors stay negligible relative to the result.
template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ constexpr Real gwn_query_range_floor_impl() noexcept {
    return Real(16) * std::numeric_limits<Real>::min() / std::numeric_limits<Real>::epsilon();
}

/// \brief A vector rescaled by a power of two, together with the exponent removed from it.
template <gwn_real_type Real> struct gwn_query_rescaled_vec3 {
    gwn_query_vec3<Real> value{};
    int exponent{0};
};

/// \brief Rescale a vector by the power of two that brings its largest magnitude into [1, 2).
///
/// Power-of-two scaling keeps every direction and sign, and it is exact unless a component far
/// smaller than the largest one becomes subnormal. A zero vector is returned unchanged.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_query_rescaled_vec3<Real>
gwn_query_rescale_impl(gwn_query_vec3<Real> const &value) noexcept {
    Real const extent = fmax(fmax(fabs(value.x), fabs(value.y)), fabs(value.z));
    if (extent == Real(0))
        return {value, 0};
    int const exponent = ilogb(extent);
    return {
        gwn_query_vec3<Real>(
            scalbn(value.x, -exponent), scalbn(value.y, -exponent), scalbn(value.z, -exponent)
        ),
        exponent
    };
}

/// \brief Kahan's FMA evaluation of the 2x2 determinant `a_x * b_y - a_y * b_x`.
///
/// The second FMA recovers the exact rounding error of the right product, and the first rounds
/// `a_x * b_y - right` once. Without underflow the relative error is at most twice the unit
/// roundoff (Jeannerod, Louvet, and Muller 2013), so the rounded sign is exact, a zero means an
/// exact zero, and the sign is antisymmetric under swapping the two columns. A plain FMA
/// difference is not even antisymmetric.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline Real gwn_query_kahan_determinant_impl(
    Real const a_x, Real const a_y, Real const b_x, Real const b_y
) noexcept {
    Real const right = a_y * b_x;
    return fma(a_x, b_y, -right) + fma(-a_y, b_x, right);
}

/// \brief Cross product whose components are Kahan FMA determinants.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_query_vec3<Real>
gwn_query_kahan_cross_impl(gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b) noexcept {
    return {
        gwn_query_kahan_determinant_impl(a.y, a.z, b.y, b.z),
        gwn_query_kahan_determinant_impl(a.z, a.x, b.z, b.x),
        gwn_query_kahan_determinant_impl(a.x, a.y, b.x, b.y),
    };
}

/// \brief Evaluate one segment's solid-angle gradient from endpoints rescaled to unit magnitude.
///
/// The out-of-line path of \ref gwn_gradient_solid_angle_edge_impl for edges whose unscaled
/// products could underflow or overflow, or that lie within rounding of the query's line. Unit
/// directions and the cosine are scale free, so they come from the rescaled endpoints. The cross
/// product's exponent and the two inverse lengths' exponents are restored only after the bounded
/// angular factors multiply them, so a representable field is never lost to an intermediate that
/// underflows or overflows. The rescaled products stay exact while the nonzero coordinates of each
/// endpoint lie within a factor of `sqrt(1 / floor)` of each other (2^49 for float).
template <bool SingularOnSegment, gwn_real_type Real>
[[nodiscard]] __device__ __noinline__ gwn_query_vec3<Real>
gwn_gradient_solid_angle_edge_rescaled_impl(
    gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b
) noexcept {
    Real const nan = std::numeric_limits<Real>::quiet_NaN();
    gwn_query_vec3<Real> const zero(Real(0), Real(0), Real(0));
    gwn_query_vec3<Real> const singular =
        SingularOnSegment ? gwn_query_vec3<Real>(nan, nan, nan) : zero;
    auto const scaled_a = gwn_query_rescale_impl(a);
    auto const scaled_b = gwn_query_rescale_impl(b);
    Real const a_norm2 = gwn_query_dot(scaled_a.value, scaled_a.value);
    Real const b_norm2 = gwn_query_dot(scaled_b.value, scaled_b.value);
    if (a_norm2 == Real(0) || b_norm2 == Real(0))
        return singular;

    Real const inverse_a = rsqrt(a_norm2);
    Real const inverse_b = rsqrt(b_norm2);
    Real const inverse_norms = inverse_a * inverse_b;
    Real const cosine = gwn_query_dot(scaled_a.value, scaled_b.value) * inverse_norms;
    auto const cross =
        gwn_query_rescale_impl(gwn_query_kahan_cross_impl(scaled_a.value, scaled_b.value));
    if (cross.value.x == Real(0) && cross.value.y == Real(0) && cross.value.z == Real(0))
        return cosine < Real(0) ? singular : zero;

    // u x v is the rescaled cross product times 2^exponent times the inverse norms. For a nearly
    // antiparallel pair, 1 + u.v cancels and equals |u x v|^2 / (1 - u.v), which leaves the unit
    // cross direction times 2^-exponent.
    Real const cross_norm = sqrt(gwn_query_dot(cross.value, cross.value));
    gwn_query_vec3<Real> angular{};
    int exponent = 0;
    if (cosine >= Real(0)) {
        angular = cross.value * (inverse_norms / (Real(1) + cosine));
        exponent = cross.exponent;
    } else {
        angular = (cross.value / cross_norm) * ((Real(1) - cosine) / (cross_norm * inverse_norms));
        exponent = -cross.exponent;
    }
    auto const restore = [](gwn_query_vec3<Real> const &value, int const shift) noexcept {
        return gwn_query_vec3<Real>(
            scalbn(value.x, shift), scalbn(value.y, shift), scalbn(value.z, shift)
        );
    };
    return restore(angular * inverse_a, exponent - scaled_a.exponent) +
           restore(angular * inverse_b, exponent - scaled_b.exponent);
}

/// \brief Solid-angle gradient contribution of one oriented segment (Biot-Savart formula).
///
/// For query-relative endpoints \p a and \p b with unit directions u and v, the segment
/// contributes `(u x v) (1/|a| + 1/|b|) / (1 + u.v)` to the gradient of the signed solid angle.
/// Summed over a closed chain, it is the gradient of the solid angle of any surface the chain
/// bounds. The field vanishes on the segment's line beyond its endpoints and is unbounded on the
/// segment itself, where \p SingularOnSegment selects NaN instead of zero. Per-triangle sums use
/// zero, because the unbounded terms of an interior edge cancel between its two triangles.
///
/// The inline path accepts endpoint norms in the range-floor interval and a cross product whose
/// largest component is above the range floor, with `|u x v|` above its square root. Every
/// product then stays normal and every result finite. Other edges take the rescaled path.
template <bool SingularOnSegment, gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_query_vec3<Real> gwn_gradient_solid_angle_edge_impl(
    gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b
) noexcept {
    Real const lower = gwn_query_range_floor_impl<Real>();
    Real const a_norm2 = gwn_query_dot(a, a);
    Real const b_norm2 = gwn_query_dot(b, b);
    if (fmin(a_norm2, b_norm2) >= lower && fmax(a_norm2, b_norm2) <= Real(1) / lower) {
        Real const inverse_a = rsqrt(a_norm2);
        Real const inverse_b = rsqrt(b_norm2);
        Real const inverse_norms = inverse_a * inverse_b;
        gwn_query_vec3<Real> const cross = gwn_query_kahan_cross_impl(a, b);
        Real const cross_extent = fmax(fmax(fabs(cross.x), fabs(cross.y)), fabs(cross.z));
        if (cross_extent >= lower && cross_extent * inverse_norms >= sqrt(lower)) {
            gwn_query_vec3<Real> const unit_cross = cross * inverse_norms;
            Real const cosine = gwn_query_dot(a, b) * inverse_norms;
            Real const inverse_length_sum = inverse_a + inverse_b;
            if (cosine >= Real(0))
                return unit_cross * (inverse_length_sum / (Real(1) + cosine));
            // 1 + u.v cancels for a nearly antiparallel pair, where it equals
            // |u x v|^2 / (1 - u.v); |u x v| is bounded below, so its square stays normal.
            return (unit_cross * ((Real(1) - cosine) / gwn_query_dot(unit_cross, unit_cross))) *
                   inverse_length_sum;
        }
    }
    return gwn_gradient_solid_angle_edge_rescaled_impl<SingularOnSegment>(a, b);
}

// Per-triangle gradient of solid angle (Biot-Savart formula).

template <gwn_real_type Real>
__device__ inline gwn_query_vec3<Real> gwn_gradient_solid_angle_triangle_impl(
    gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b, gwn_query_vec3<Real> const &c,
    gwn_query_vec3<Real> const &q
) noexcept {
    gwn_query_vec3<Real> const ga = a - q;
    gwn_query_vec3<Real> const gb = b - q;
    gwn_query_vec3<Real> const gc = c - q;

    gwn_query_vec3<Real> const zero(Real(0), Real(0), Real(0));
    auto const is_zero = [](gwn_query_vec3<Real> const &value) noexcept {
        return value.x == Real(0) && value.y == Real(0) && value.z == Real(0);
    };
    if (is_zero(ga) || is_zero(gb) || is_zero(gc))
        return zero;

    gwn_query_vec3<Real> result = gwn_gradient_solid_angle_edge_impl<false>(ga, gb);
    result += gwn_gradient_solid_angle_edge_impl<false>(gb, gc);
    result += gwn_gradient_solid_angle_edge_impl<false>(gc, ga);

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
    gwn_query_vec3<Real> const b(triangle.v1_x, triangle.v1_y, triangle.v1_z);
    gwn_query_vec3<Real> const c(triangle.v2_x, triangle.v2_y, triangle.v2_z);
    return gwn_signed_solid_angle_triangle_impl(a, b, c, query);
}

/// \brief Compute exact solid angle and gradient from one canonical triangle record.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_winding_and_gradient_result<Real>
gwn_bvh_triangle_winding_gradient_impl(
    gwn_bvh_triangle<Real> const &triangle, gwn_query_vec3<Real> const &query
) noexcept {
    gwn_query_vec3<Real> const a(triangle.v0_x, triangle.v0_y, triangle.v0_z);
    gwn_query_vec3<Real> const b(triangle.v1_x, triangle.v1_y, triangle.v1_z);
    gwn_query_vec3<Real> const c(triangle.v2_x, triangle.v2_y, triangle.v2_z);
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
        Index next_node = gwn_invalid_index<Index>();

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
                // TODO: `std::isfinite` will support constexpr starting in C++23.
                if (cuda::std::isfinite(omega_approx)) {
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
                if (!gwn_is_invalid_index(next_node)) {
                    if (stack_size >= StackCapacity) {
                        overflow_callback();
                        set_overflow_nan_result();
                        return result;
                    }
                    // Keep the previously selected child pending so the final internal child can
                    // be entered directly without a matching stack store and load.
                    stack[stack_size++] = next_node;
                }
                next_node = static_cast<Index>(child.offset());
                continue;
            }

            // A descended leaf contributes the exact solid-angle direct sum of its triangles.
            // Primitive IDs remain cold because winding and its gradient use oriented positions.
            if (child.is_leaf())
                accumulate_leaf(child);
        }

        if (!gwn_is_invalid_index(next_node)) {
            node_index = next_node;
            continue;
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
