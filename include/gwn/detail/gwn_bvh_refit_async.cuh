#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../gwn_kernel_utils.cuh"
#include "gwn_bvh_topology_build_common.cuh"

namespace gwn {
namespace detail {

template <int Order, gwn_real_type Real> struct gwn_device_taylor_moment;

template <gwn_real_type Real> struct gwn_device_taylor_moment<0, Real> {
    Real area = Real(0);
    Real area_p_x = Real(0);
    Real area_p_y = Real(0);
    Real area_p_z = Real(0);
    Real n_x = Real(0);
    Real n_y = Real(0);
    Real n_z = Real(0);
};

template <gwn_real_type Real>
struct gwn_device_taylor_moment<1, Real> : gwn_device_taylor_moment<0, Real> {
    Real nx_x = Real(0);
    Real nx_y = Real(0);
    Real nx_z = Real(0);
    Real ny_x = Real(0);
    Real ny_y = Real(0);
    Real ny_z = Real(0);
    Real nz_x = Real(0);
    Real nz_y = Real(0);
    Real nz_z = Real(0);
};

/// Order-2 payload: additive raw second moments about the global origin.
/// For each normal component i in {x,y,z}, stores six jk moments:
/// {xx, yy, zz, xy, yz, zx}.  The final 10 query coefficients are reconstructed
/// at node centroids in write_valid.
template <gwn_real_type Real>
struct gwn_device_taylor_moment<2, Real> : gwn_device_taylor_moment<1, Real> {
    Real raw_x_xx = Real(0);
    Real raw_x_yy = Real(0);
    Real raw_x_zz = Real(0);
    Real raw_x_xy = Real(0);
    Real raw_x_yz = Real(0);
    Real raw_x_zx = Real(0);

    Real raw_y_xx = Real(0);
    Real raw_y_yy = Real(0);
    Real raw_y_zz = Real(0);
    Real raw_y_xy = Real(0);
    Real raw_y_yz = Real(0);
    Real raw_y_zx = Real(0);

    Real raw_z_xx = Real(0);
    Real raw_z_yy = Real(0);
    Real raw_z_zz = Real(0);
    Real raw_z_xy = Real(0);
    Real raw_z_yz = Real(0);
    Real raw_z_zx = Real(0);
};

template <int Order, gwn_real_type Real>
__host__ __device__ inline void gwn_accumulate_taylor_moment(
    gwn_device_taylor_moment<Order, Real> &dst, gwn_device_taylor_moment<Order, Real> const &src
) noexcept {
    dst.area += src.area;
    dst.area_p_x += src.area_p_x;
    dst.area_p_y += src.area_p_y;
    dst.area_p_z += src.area_p_z;
    dst.n_x += src.n_x;
    dst.n_y += src.n_y;
    dst.n_z += src.n_z;
    if constexpr (Order >= 1) {
        dst.nx_x += src.nx_x;
        dst.nx_y += src.nx_y;
        dst.nx_z += src.nx_z;
        dst.ny_x += src.ny_x;
        dst.ny_y += src.ny_y;
        dst.ny_z += src.ny_z;
        dst.nz_x += src.nz_x;
        dst.nz_y += src.nz_y;
        dst.nz_z += src.nz_z;
    }
    if constexpr (Order >= 2) {
        dst.raw_x_xx += src.raw_x_xx;
        dst.raw_x_yy += src.raw_x_yy;
        dst.raw_x_zz += src.raw_x_zz;
        dst.raw_x_xy += src.raw_x_xy;
        dst.raw_x_yz += src.raw_x_yz;
        dst.raw_x_zx += src.raw_x_zx;

        dst.raw_y_xx += src.raw_y_xx;
        dst.raw_y_yy += src.raw_y_yy;
        dst.raw_y_zz += src.raw_y_zz;
        dst.raw_y_xy += src.raw_y_xy;
        dst.raw_y_yz += src.raw_y_yz;
        dst.raw_y_zx += src.raw_y_zx;

        dst.raw_z_xx += src.raw_z_xx;
        dst.raw_z_yy += src.raw_z_yy;
        dst.raw_z_zz += src.raw_z_zz;
        dst.raw_z_xy += src.raw_z_xy;
        dst.raw_z_yz += src.raw_z_yz;
        dst.raw_z_zx += src.raw_z_zx;
    }
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ inline Real gwn_bounds_max_p_dist2(
    gwn_aabb<Real> const &bounds, Real const average_x, Real const average_y, Real const average_z
) noexcept {
    Real const dx = std::max(average_x - bounds.min_x, bounds.max_x - average_x);
    Real const dy = std::max(average_y - bounds.min_y, bounds.max_y - average_y);
    Real const dz = std::max(average_z - bounds.min_z, bounds.max_z - average_z);
    return dx * dx + dy * dy + dz * dz;
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_compute_triangle_aabb(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_aabb<Real> &bounds
) noexcept {
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return false;

    auto const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ic, geometry.vertex_count())) {
        return false;
    }

    auto const a_index = static_cast<std::size_t>(ia);
    auto const b_index = static_cast<std::size_t>(ib);
    auto const c_index = static_cast<std::size_t>(ic);

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
    return true;
}

template <int Order, gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_compute_triangle_taylor_raw_moment(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_device_taylor_moment<Order, Real> &raw_moment
) noexcept {
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return false;

    auto const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ic, geometry.vertex_count())) {
        return false;
    }

    auto const a_index = static_cast<std::size_t>(ia);
    auto const b_index = static_cast<std::size_t>(ib);
    auto const c_index = static_cast<std::size_t>(ic);

    Real const ax = geometry.vertex_x[a_index];
    Real const ay = geometry.vertex_y[a_index];
    Real const az = geometry.vertex_z[a_index];
    Real const bx = geometry.vertex_x[b_index];
    Real const by = geometry.vertex_y[b_index];
    Real const bz = geometry.vertex_z[b_index];
    Real const cx = geometry.vertex_x[c_index];
    Real const cy = geometry.vertex_y[c_index];
    Real const cz = geometry.vertex_z[c_index];

    Real const average_x = (ax + bx + cx) / Real(3);
    Real const average_y = (ay + by + cy) / Real(3);
    Real const average_z = (az + bz + cz) / Real(3);

    Real const abx = bx - ax;
    Real const aby = by - ay;
    Real const abz = bz - az;
    Real const acx = cx - ax;
    Real const acy = cy - ay;
    Real const acz = cz - az;

    Real const n_x = Real(0.5) * (aby * acz - abz * acy);
    Real const n_y = Real(0.5) * (abz * acx - abx * acz);
    Real const n_z = Real(0.5) * (abx * acy - aby * acx);

    Real const area2 = n_x * n_x + n_y * n_y + n_z * n_z;
    Real const area = sqrt(std::max(area2, Real(0)));

    raw_moment.area = area;
    raw_moment.area_p_x = average_x * area;
    raw_moment.area_p_y = average_y * area;
    raw_moment.area_p_z = average_z * area;
    raw_moment.n_x = n_x;
    raw_moment.n_y = n_y;
    raw_moment.n_z = n_z;
    if constexpr (Order >= 1) {
        raw_moment.nx_x = n_x * average_x;
        raw_moment.nx_y = n_x * average_y;
        raw_moment.nx_z = n_x * average_z;
        raw_moment.ny_x = n_y * average_x;
        raw_moment.ny_y = n_y * average_y;
        raw_moment.ny_z = n_y * average_z;
        raw_moment.nz_x = n_z * average_x;
        raw_moment.nz_y = n_z * average_y;
        raw_moment.nz_z = n_z * average_z;
    }
    if constexpr (Order >= 2) {
        // Zero-area triangles contribute nothing to second moments.
        if (area == Real(0)) {
            raw_moment.raw_x_xx = Real(0);
            raw_moment.raw_x_yy = Real(0);
            raw_moment.raw_x_zz = Real(0);
            raw_moment.raw_x_xy = Real(0);
            raw_moment.raw_x_yz = Real(0);
            raw_moment.raw_x_zx = Real(0);

            raw_moment.raw_y_xx = Real(0);
            raw_moment.raw_y_yy = Real(0);
            raw_moment.raw_y_zz = Real(0);
            raw_moment.raw_y_xy = Real(0);
            raw_moment.raw_y_yz = Real(0);
            raw_moment.raw_y_zx = Real(0);

            raw_moment.raw_z_xx = Real(0);
            raw_moment.raw_z_yy = Real(0);
            raw_moment.raw_z_zz = Real(0);
            raw_moment.raw_z_xy = Real(0);
            raw_moment.raw_z_yz = Real(0);
            raw_moment.raw_z_zx = Real(0);
            return true;
        }

        // Unit normal for multiplying integrals.
        Real const inv_area = Real(1) / area;
        Real const nhat_x = n_x * inv_area;
        Real const nhat_y = n_y * inv_area;
        Real const nhat_z = n_z * inv_area;

        // Vertex positions as arrays for axis-indexed access.
        Real const verts[3][3] = {
            {ax, ay, az},
            {bx, by, bz},
            {cx, cy, cz},
        };

        // Compute 6 area-scaled surface integrals: integral_ii and integral_ij
        // using the HDK's sorted-vertex triangle-splitting method.
        // Reference: UT_SolidAngle.cpp lines 306-440.
        // We compute integrals relative to the triangle centroid P (= average),
        // matching the HDK convention.
        Real const P[3] = {average_x, average_y, average_z};

        // Sort vertex indices along each axis independently.
        // order_a[i] gives the 3 vertex indices sorted by axis i.
        int order_a[3][3] = {{0,1,2}, {0,1,2}, {0,1,2}};
        Real span[3];
        for (int axis = 0; axis < 3; ++axis) {
            auto swap2 = [](int &a, int &b) { int t = a; a = b; b = t; };
            if (verts[0][axis] > verts[1][axis])
                swap2(order_a[axis][0], order_a[axis][1]);
            if (verts[order_a[axis][0]][axis] > verts[2][axis])
                swap2(order_a[axis][0], order_a[axis][2]);
            if (verts[order_a[axis][1]][axis] > verts[order_a[axis][2]][axis])
                swap2(order_a[axis][1], order_a[axis][2]);
            span[axis] = verts[order_a[axis][2]][axis] - verts[order_a[axis][0]][axis];
        }

        Real integral_xx = Real(0), integral_yy = Real(0), integral_zz = Real(0);
        Real integral_xy = Real(0), integral_yz = Real(0), integral_zx = Real(0);

        // Lambda-equivalent: compute integrals by splitting triangle at middle
        // vertex along axis i. Computes integral_ii and optionally integral_ij, integral_ik.
        // a, b, c are vertices sorted along axis i (a[i] <= b[i] <= c[i]).
        auto compute_integrals = [&](
            int const va, int const vb, int const vc,
            Real *out_ii, Real *out_ij, Real *out_ik,
            int const i
        ) {
            int const j = (i == 2) ? 0 : (i + 1);
            int const k = (j == 2) ? 0 : (j + 1);

            Real const oab_i = verts[vb][i] - verts[va][i];
            Real const oab_j = verts[vb][j] - verts[va][j];
            Real const oab_k = verts[vb][k] - verts[va][k];
            Real const oac_i = verts[vc][i] - verts[va][i];
            Real const oac_j = verts[vc][j] - verts[va][j];
            Real const oac_k = verts[vc][k] - verts[va][k];
            Real const ocb_i = verts[vb][i] - verts[vc][i];
            Real const ocb_j = verts[vb][j] - verts[vc][j];
            Real const ocb_k = verts[vb][k] - verts[vc][k];

            Real const t = (oac_i > Real(0)) ? (oab_i / oac_i) : Real(0);
            Real const jdiff = t * oac_j - oab_j;
            Real const kdiff = t * oac_k - oab_k;

            // Cross products for area scaling of the two sub-triangles.
            Real const cross_a_0 = jdiff * oab_k - kdiff * oab_j;
            Real const cross_a_1 = kdiff * oab_i;
            Real const cross_a_2 = jdiff * oab_i;
            Real const cross_c_0 = jdiff * ocb_k - kdiff * ocb_j;
            Real const cross_c_1 = kdiff * ocb_i;
            Real const cross_c_2 = jdiff * ocb_i;

            Real const area_scale_a = sqrt(std::max(
                cross_a_0 * cross_a_0 + cross_a_1 * cross_a_1 + cross_a_2 * cross_a_2, Real(0)));
            Real const area_scale_c = sqrt(std::max(
                cross_c_0 * cross_c_0 + cross_c_1 * cross_c_1 + cross_c_2 * cross_c_2, Real(0)));

            Real const Pai = verts[va][i] - P[i];
            Real const Pci = verts[vc][i] - P[i];

            // integral_ii = integral of (p_i - P_i)^2 dA
            *out_ii = area_scale_a * (Real(0.5) * Pai * Pai + Real(2.0/3.0) * Pai * oab_i + Real(0.25) * oab_i * oab_i)
                    + area_scale_c * (Real(0.5) * Pci * Pci + Real(2.0/3.0) * Pci * ocb_i + Real(0.25) * ocb_i * ocb_i);

            // Cross-integrals integral_ij and integral_ik.
            auto compute_cross = [&](int const jk, Real const diff, Real *out) {
                if (!out) return;
                Real const obmidj = verts[vb][jk] + Real(0.5) * diff;
                Real const oabmidj = obmidj - verts[va][jk];
                Real const ocbmidj = obmidj - verts[vc][jk];
                Real const Paj = verts[va][jk] - P[jk];
                Real const Pcj = verts[vc][jk] - P[jk];
                *out = area_scale_a * (Real(0.5) * Pai * Paj + Real(1.0/3.0) * Pai * oabmidj + Real(1.0/3.0) * Paj * oab_i + Real(0.25) * oab_i * oabmidj)
                     + area_scale_c * (Real(0.5) * Pci * Pcj + Real(1.0/3.0) * Pci * ocbmidj + Real(1.0/3.0) * Pcj * ocb_i + Real(0.25) * ocb_i * ocbmidj);
            };

            compute_cross(j, jdiff, out_ij);
            compute_cross(k, kdiff, out_ik);
        };

        // Compute integrals along each axis, choosing which cross-integrals
        // to compute based on axis span (avoids redundant computation).
        // Reference: UT_SolidAngle.cpp lines 423-440.
        Real const dx = span[0], dy = span[1], dz = span[2];
        if (dx > Real(0)) {
            compute_integrals(
                order_a[0][0], order_a[0][1], order_a[0][2],
                &integral_xx,
                (dx >= dy && dy > Real(0)) ? &integral_xy : nullptr,
                (dx >= dz && dz > Real(0)) ? &integral_zx : nullptr,
                0);
        }
        if (dy > Real(0)) {
            compute_integrals(
                order_a[1][0], order_a[1][1], order_a[1][2],
                &integral_yy,
                (dy >= dz && dz > Real(0)) ? &integral_yz : nullptr,
                (dx < dy && dx > Real(0)) ? &integral_xy : nullptr,
                1);
        }
        if (dz > Real(0)) {
            compute_integrals(
                order_a[2][0], order_a[2][1], order_a[2][2],
                &integral_zz,
                (dx < dz && dx > Real(0)) ? &integral_zx : nullptr,
                (dy < dz && dy > Real(0)) ? &integral_yz : nullptr,
                2);
        }

        // Store additive second moments about the global origin:
        //   raw_i_jk = centred_i_jk + N_i * P_j * P_k
        // where centred_i_jk = n_hat_i * integral_jk around triangle centroid P.
        Real const Px = average_x, Py = average_y, Pz = average_z;
        Real const Nx = n_x, Ny = n_y, Nz = n_z;

        Real const cx_xx = nhat_x * integral_xx;
        Real const cx_yy = nhat_x * integral_yy;
        Real const cx_zz = nhat_x * integral_zz;
        Real const cx_xy = nhat_x * integral_xy;
        Real const cx_yz = nhat_x * integral_yz;
        Real const cx_zx = nhat_x * integral_zx;

        Real const cy_xx = nhat_y * integral_xx;
        Real const cy_yy = nhat_y * integral_yy;
        Real const cy_zz = nhat_y * integral_zz;
        Real const cy_xy = nhat_y * integral_xy;
        Real const cy_yz = nhat_y * integral_yz;
        Real const cy_zx = nhat_y * integral_zx;

        Real const cz_xx = nhat_z * integral_xx;
        Real const cz_yy = nhat_z * integral_yy;
        Real const cz_zz = nhat_z * integral_zz;
        Real const cz_xy = nhat_z * integral_xy;
        Real const cz_yz = nhat_z * integral_yz;
        Real const cz_zx = nhat_z * integral_zx;

        raw_moment.raw_x_xx = cx_xx + Nx * Px * Px;
        raw_moment.raw_x_yy = cx_yy + Nx * Py * Py;
        raw_moment.raw_x_zz = cx_zz + Nx * Pz * Pz;
        raw_moment.raw_x_xy = cx_xy + Nx * Px * Py;
        raw_moment.raw_x_yz = cx_yz + Nx * Py * Pz;
        raw_moment.raw_x_zx = cx_zx + Nx * Pz * Px;

        raw_moment.raw_y_xx = cy_xx + Ny * Px * Px;
        raw_moment.raw_y_yy = cy_yy + Ny * Py * Py;
        raw_moment.raw_y_zz = cy_zz + Ny * Pz * Pz;
        raw_moment.raw_y_xy = cy_xy + Ny * Px * Py;
        raw_moment.raw_y_yz = cy_yz + Ny * Py * Pz;
        raw_moment.raw_y_zx = cy_zx + Ny * Pz * Px;

        raw_moment.raw_z_xx = cz_xx + Nz * Px * Px;
        raw_moment.raw_z_yy = cz_yy + Nz * Py * Py;
        raw_moment.raw_z_zz = cz_zz + Nz * Pz * Pz;
        raw_moment.raw_z_xy = cz_xy + Nz * Px * Py;
        raw_moment.raw_z_yz = cz_yz + Nz * Py * Pz;
        raw_moment.raw_z_zx = cz_zx + Nz * Pz * Px;
    }
    return true;
}

template <int Width, gwn_index_type Index> struct gwn_refit_topology_arrays {
    cuda::std::span<Index> internal_parent{};
    cuda::std::span<std::uint8_t> internal_parent_slot{};
    cuda::std::span<std::uint8_t> internal_arity{};
    cuda::std::span<unsigned int> internal_arrivals{};
};

template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_prepare_refit_topology_functor {
    gwn_bvh_topology_accessor<Width, Real, Index> topology{};
    gwn_refit_topology_arrays<Width, Index> arrays{};
    unsigned int *error_flag = nullptr;

    __device__ inline void mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ void operator()(std::size_t const node_id) const {
        if (node_id >= topology.nodes.size() || node_id >= arrays.internal_parent.size() ||
            node_id >= arrays.internal_parent_slot.size() ||
            node_id >= arrays.internal_arity.size()) {
            mark_error();
            return;
        }

        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        std::uint8_t arity = 0;
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid)
                continue;
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf) {
                mark_error();
                continue;
            }

            ++arity;
            if (kind == gwn_bvh_child_kind::k_internal) {
                Index const child_index = node.child_index[child_slot];
                if (!gwn_index_in_bounds(child_index, topology.nodes.size())) {
                    mark_error();
                    continue;
                }
                auto const child_id = static_cast<std::size_t>(child_index);
                arrays.internal_parent[child_id] = static_cast<Index>(node_id);
                arrays.internal_parent_slot[child_id] = static_cast<std::uint8_t>(child_slot);
                continue;
            }

            Index const leaf_begin = node.child_index[child_slot];
            Index const leaf_count = node.child_count[child_slot];
            if (gwn_is_invalid_index(leaf_begin) || gwn_is_invalid_index(leaf_count)) {
                mark_error();
                continue;
            }
            auto const begin = static_cast<std::size_t>(leaf_begin);
            auto const count = static_cast<std::size_t>(leaf_count);
            if (begin > topology.primitive_indices.size() ||
                count > (topology.primitive_indices.size() - begin)) {
                mark_error();
                continue;
            }
        }

        arrays.internal_arity[node_id] = arity;
        if (arity == 0)
            mark_error();
    }
};

template <int Width, gwn_real_type Real, gwn_index_type Index> struct gwn_aabb_refit_traits {
    using payload_type = gwn_aabb<Real>;

    struct output_context {
        cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real>> nodes{};
    };

    static constexpr char const *k_error_name =
        "AABB refit failed topology/propagation validation.";

    __device__ static bool make_leaf_payload(
        gwn_geometry_accessor<Real, Index> const &geometry,
        gwn_bvh_topology_accessor<Width, Real, Index> const &topology, std::size_t const node_id,
        int const child_slot, output_context const &, payload_type &payload
    ) noexcept {
        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        Index const begin = node.child_index[child_slot];
        Index const count = node.child_count[child_slot];

        bool has_bounds = false;
        payload = {};
        for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
            Index const sorted_slot = begin + primitive_offset;
            auto const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
            if (sorted_slot_u >= topology.primitive_indices.size())
                continue;

            Index const primitive_id = topology.primitive_indices[sorted_slot_u];
            gwn_aabb<Real> primitive_bounds{};
            if (!gwn_compute_triangle_aabb(geometry, primitive_id, primitive_bounds))
                continue;

            if (!has_bounds) {
                payload = primitive_bounds;
                has_bounds = true;
            } else {
                payload = gwn_aabb_union(payload, primitive_bounds);
            }
        }

        if (!has_bounds)
            payload = gwn_aabb<Real>{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
        return true;
    }

    __device__ static void combine(payload_type &dst, payload_type const &src) noexcept {
        dst = gwn_aabb_union(dst, src);
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

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_moment_refit_traits {
    using payload_type = gwn_device_taylor_moment<Order, Real>;
    using output_node_type = gwn_bvh_taylor_node_soa<Width, Order, Real>;

    struct output_context {
        cuda::std::span<output_node_type> moment_nodes{};
        cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real> const> aabb_nodes{};
    };

    static constexpr char const *k_error_name =
        "Moment refit failed topology/propagation validation.";

    __device__ static bool make_leaf_payload(
        gwn_geometry_accessor<Real, Index> const &geometry,
        gwn_bvh_topology_accessor<Width, Real, Index> const &topology, std::size_t const node_id,
        int const child_slot, output_context const &, payload_type &payload
    ) noexcept {
        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        Index const begin = node.child_index[child_slot];
        Index const count = node.child_count[child_slot];

        payload = {};
        for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
            Index const sorted_slot = begin + primitive_offset;
            auto const sorted_slot_u = static_cast<std::size_t>(sorted_slot);
            if (sorted_slot_u >= topology.primitive_indices.size())
                continue;

            Index const primitive_id = topology.primitive_indices[sorted_slot_u];
            payload_type primitive_payload{};
            if (!gwn_compute_triangle_taylor_raw_moment<Order>(
                    geometry, primitive_id, primitive_payload
                ))
                continue;
            gwn_accumulate_taylor_moment(payload, primitive_payload);
        }
        return true;
    }

    __device__ static void combine(payload_type &dst, payload_type const &src) noexcept {
        gwn_accumulate_taylor_moment(dst, src);
    }

    __device__ static void zero_child(output_node_type &node, int const child_slot) noexcept {
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
        if constexpr (Order >= 2) {
            node.child_nijk_xxx[child_slot] = Real(0);
            node.child_nijk_yyy[child_slot] = Real(0);
            node.child_nijk_zzz[child_slot] = Real(0);
            node.child_sum_permute_nxyz[child_slot] = Real(0);
            node.child_2nxxy_nyxx[child_slot] = Real(0);
            node.child_2nxxz_nzxx[child_slot] = Real(0);
            node.child_2nyyz_nzyy[child_slot] = Real(0);
            node.child_2nyyx_nxyy[child_slot] = Real(0);
            node.child_2nzzx_nxzz[child_slot] = Real(0);
            node.child_2nzzy_nyzz[child_slot] = Real(0);
        }
    }

    __device__ static void write_invalid(
        output_context const &context, std::size_t const node_id, int const child_slot
    ) noexcept {
        zero_child(context.moment_nodes[node_id], child_slot);
    }

    __device__ static void write_valid(
        output_context const &context, std::size_t const node_id, int const child_slot,
        payload_type const &payload
    ) noexcept {
        output_node_type &node = context.moment_nodes[node_id];
        gwn_aabb<Real> const bounds{
            context.aabb_nodes[node_id].child_min_x[child_slot],
            context.aabb_nodes[node_id].child_min_y[child_slot],
            context.aabb_nodes[node_id].child_min_z[child_slot],
            context.aabb_nodes[node_id].child_max_x[child_slot],
            context.aabb_nodes[node_id].child_max_y[child_slot],
            context.aabb_nodes[node_id].child_max_z[child_slot],
        };

        Real average_x = Real(0);
        Real average_y = Real(0);
        Real average_z = Real(0);
        if (payload.area > Real(0)) {
            average_x = payload.area_p_x / payload.area;
            average_y = payload.area_p_y / payload.area;
            average_z = payload.area_p_z / payload.area;
        } else {
            average_x = (bounds.min_x + bounds.max_x) * Real(0.5);
            average_y = (bounds.min_y + bounds.max_y) * Real(0.5);
            average_z = (bounds.min_z + bounds.max_z) * Real(0.5);
        }

        node.child_max_p_dist2[child_slot] =
            gwn_bounds_max_p_dist2(bounds, average_x, average_y, average_z);
        node.child_average_x[child_slot] = average_x;
        node.child_average_y[child_slot] = average_y;
        node.child_average_z[child_slot] = average_z;
        node.child_n_x[child_slot] = payload.n_x;
        node.child_n_y[child_slot] = payload.n_y;
        node.child_n_z[child_slot] = payload.n_z;

        if constexpr (Order >= 1) {
            Real const nij_xx = payload.nx_x - average_x * payload.n_x;
            Real const nij_yy = payload.ny_y - average_y * payload.n_y;
            Real const nij_zz = payload.nz_z - average_z * payload.n_z;
            Real const nxy = payload.nx_y - average_y * payload.n_x;
            Real const nyx = payload.ny_x - average_x * payload.n_y;
            Real const nyz = payload.ny_z - average_z * payload.n_y;
            Real const nzy = payload.nz_y - average_y * payload.n_z;
            Real const nzx = payload.nz_x - average_x * payload.n_z;
            Real const nxz = payload.nx_z - average_z * payload.n_x;

            node.child_nij_xx[child_slot] = nij_xx;
            node.child_nij_yy[child_slot] = nij_yy;
            node.child_nij_zz[child_slot] = nij_zz;
            node.child_nxy_nyx[child_slot] = nxy + nyx;
            node.child_nyz_nzy[child_slot] = nyz + nzy;
            node.child_nzx_nxz[child_slot] = nzx + nxz;
        }

        if constexpr (Order >= 2) {
            Real const Px = average_x, Py = average_y, Pz = average_z;
            Real const Nx = payload.n_x, Ny = payload.n_y, Nz = payload.n_z;
            // Reconstruct centred second moments from additive raw-origin moments:
            //   centred_i_jk = raw_i_jk - P_j * F_i_k - P_k * F_i_j + P_j*P_k*N_i
            // where F_i_j are first moments (payload.n?_?).
            Real const x_xx =
                payload.raw_x_xx - Real(2) * Px * payload.nx_x + Px * Px * Nx;
            Real const x_yy =
                payload.raw_x_yy - Real(2) * Py * payload.nx_y + Py * Py * Nx;
            Real const x_zz =
                payload.raw_x_zz - Real(2) * Pz * payload.nx_z + Pz * Pz * Nx;
            Real const x_xy =
                payload.raw_x_xy - Px * payload.nx_y - Py * payload.nx_x + Px * Py * Nx;
            Real const x_yz =
                payload.raw_x_yz - Py * payload.nx_z - Pz * payload.nx_y + Py * Pz * Nx;
            Real const x_zx =
                payload.raw_x_zx - Pz * payload.nx_x - Px * payload.nx_z + Pz * Px * Nx;

            Real const y_xx =
                payload.raw_y_xx - Real(2) * Px * payload.ny_x + Px * Px * Ny;
            Real const y_yy =
                payload.raw_y_yy - Real(2) * Py * payload.ny_y + Py * Py * Ny;
            Real const y_zz =
                payload.raw_y_zz - Real(2) * Pz * payload.ny_z + Pz * Pz * Ny;
            Real const y_xy =
                payload.raw_y_xy - Px * payload.ny_y - Py * payload.ny_x + Px * Py * Ny;
            Real const y_yz =
                payload.raw_y_yz - Py * payload.ny_z - Pz * payload.ny_y + Py * Pz * Ny;
            Real const y_zx =
                payload.raw_y_zx - Pz * payload.ny_x - Px * payload.ny_z + Pz * Px * Ny;

            Real const z_xx =
                payload.raw_z_xx - Real(2) * Px * payload.nz_x + Px * Px * Nz;
            Real const z_yy =
                payload.raw_z_yy - Real(2) * Py * payload.nz_y + Py * Py * Nz;
            Real const z_zz =
                payload.raw_z_zz - Real(2) * Pz * payload.nz_z + Pz * Pz * Nz;
            Real const z_xy =
                payload.raw_z_xy - Px * payload.nz_y - Py * payload.nz_x + Px * Py * Nz;
            Real const z_yz =
                payload.raw_z_yz - Py * payload.nz_z - Pz * payload.nz_y + Py * Pz * Nz;
            Real const z_zx =
                payload.raw_z_zx - Pz * payload.nz_x - Px * payload.nz_z + Pz * Px * Nz;

            node.child_nijk_xxx[child_slot] = x_xx;
            node.child_nijk_yyy[child_slot] = y_yy;
            node.child_nijk_zzz[child_slot] = z_zz;
            node.child_sum_permute_nxyz[child_slot] = Real(2) * (x_yz + y_zx + z_xy);
            node.child_2nxxy_nyxx[child_slot] = Real(2) * x_xy + y_xx;
            node.child_2nxxz_nzxx[child_slot] = Real(2) * x_zx + z_xx;
            node.child_2nyyz_nzyy[child_slot] = Real(2) * y_yz + z_yy;
            node.child_2nyyx_nxyy[child_slot] = Real(2) * y_xy + x_yy;
            node.child_2nzzx_nxzz[child_slot] = Real(2) * z_zx + x_zz;
            node.child_2nzzy_nyzz[child_slot] = Real(2) * z_yz + y_zz;
        }
    }
};

template <int Width, gwn_index_type Index, class Payload> struct gwn_refit_pending_buffer {
    cuda::std::span<Payload> pending{};

    __device__ bool is_valid(std::size_t const node_id, int const child_slot) const noexcept {
        if (child_slot < 0 || child_slot >= Width)
            return false;
        if (node_id > (std::numeric_limits<std::size_t>::max() / std::size_t(Width)))
            return false;
        std::size_t const pending_index =
            node_id * std::size_t(Width) + static_cast<std::size_t>(child_slot);
        return pending_index < pending.size();
    }

    __device__ std::size_t index(std::size_t const node_id, int const child_slot) const noexcept {
        return node_id * std::size_t(Width) + static_cast<std::size_t>(child_slot);
    }
};

template <class Traits, int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_refit_from_leaves_functor {
    using payload_type = typename Traits::payload_type;
    using output_context = typename Traits::output_context;

    gwn_geometry_accessor<Real, Index> geometry{};
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
            GWN_ASSERT(
                next_arrivals <= expected_arrivals,
                "refit: arrival count %u exceeds expected %u at node %zu",
                next_arrivals, expected_arrivals, parent_id
            );
            if (expected_arrivals == 0u || expected_arrivals > static_cast<unsigned int>(Width)) {
                mark_error();
                return;
            }
            if (next_arrivals < expected_arrivals)
                return;
            if (next_arrivals > expected_arrivals) {
                mark_error();
                return;
            }

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
        auto const child_slot = static_cast<int>(edge_index % std::size_t(Width));
        if (node_id >= topology.nodes.size())
            return;

        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
        if (child_kind != gwn_bvh_child_kind::k_leaf)
            return;

        payload_type leaf_payload{};
        if (!Traits::make_leaf_payload(
                geometry, topology, node_id, child_slot, context, leaf_payload
            )) {
            mark_error();
            return;
        }

        propagate_up(
            static_cast<Index>(node_id), static_cast<std::uint8_t>(child_slot), leaf_payload
        );
    }
};

template <int Width, gwn_index_type Index> struct gwn_validate_refit_convergence_functor {
    cuda::std::span<std::uint8_t const> internal_arity{};
    cuda::std::span<unsigned int const> internal_arrivals{};
    unsigned int *error_flag = nullptr;

    __device__ void operator()(std::size_t const node_id) const {
        if (node_id >= internal_arity.size() || node_id >= internal_arrivals.size()) {
            if (error_flag != nullptr)
                atomicExch(error_flag, 1u);
            return;
        }

        auto const expected_arrivals = static_cast<unsigned int>(internal_arity[node_id]);
        if (expected_arrivals == 0u || expected_arrivals > static_cast<unsigned int>(Width) ||
            internal_arrivals[node_id] != expected_arrivals) {
            if (error_flag != nullptr)
                atomicExch(error_flag, 1u);
        }
    }
};

template <class Traits, int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_finalize_refit_functor {
    using payload_type = typename Traits::payload_type;
    using output_context = typename Traits::output_context;

    gwn_bvh_topology_accessor<Width, Real, Index> topology{};
    gwn_refit_pending_buffer<Width, Index, payload_type> pending{};
    output_context context{};
    unsigned int *error_flag = nullptr;

    __device__ inline void mark_error() const noexcept {
        if (error_flag != nullptr)
            atomicExch(error_flag, 1u);
    }

    __device__ void operator()(std::size_t const node_id) const {
        if (node_id >= topology.nodes.size()) {
            mark_error();
            return;
        }

        gwn_bvh_topology_node_soa<Width, Index> const &node = topology.nodes[node_id];
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (kind == gwn_bvh_child_kind::k_invalid) {
                Traits::write_invalid(context, node_id, child_slot);
                continue;
            }
            if (kind != gwn_bvh_child_kind::k_internal && kind != gwn_bvh_child_kind::k_leaf) {
                Traits::write_invalid(context, node_id, child_slot);
                mark_error();
                continue;
            }
            if (!pending.is_valid(node_id, child_slot)) {
                Traits::write_invalid(context, node_id, child_slot);
                mark_error();
                continue;
            }
            payload_type const payload = pending.pending[pending.index(node_id, child_slot)];
            Traits::write_valid(context, node_id, child_slot, payload);
        }
    }
};

template <class Traits, int Width, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_run_refit_pass(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &topology,
    typename Traits::output_context const &output_context,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor is invalid for refit.");
    if (!topology.is_valid())
        return gwn_status::invalid_argument("Topology accessor is invalid for refit.");
    if (!topology.has_internal_root())
        return gwn_status::ok();

    constexpr int k_block_size = k_gwn_default_block_size;
    std::size_t const node_count = topology.nodes.size();
    if (node_count == 0)
        return gwn_status::ok();
    if (node_count > (std::numeric_limits<std::size_t>::max() / std::size_t(Width)))
        return gwn_status::internal_error("Refit node count overflow.");

    std::size_t const pending_count = node_count * std::size_t(Width);
    GWN_ASSERT(pending_count / std::size_t(Width) == node_count, "refit: pending_count overflow");
    using payload_type = typename Traits::payload_type;

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

    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            node_count,
            gwn_prepare_refit_topology_functor<Width, Real, Index>{
                topology, arrays, error_flag_device.data()
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            pending_count,
            gwn_refit_from_leaves_functor<Traits, Width, Real, Index>{
                geometry, topology, arrays, pending, output_context, error_flag_device.data()
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
                error_flag_device.data()
            },
            stream
        )
    );
    GWN_RETURN_ON_ERROR(
        gwn_launch_linear_kernel<k_block_size>(
            node_count,
            gwn_finalize_refit_functor<Traits, Width, Real, Index>{
                topology, pending, output_context, error_flag_device.data()
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
        return gwn_status::internal_error(Traits::k_error_name);

    return gwn_status::ok();
}

} // namespace detail
} // namespace gwn
