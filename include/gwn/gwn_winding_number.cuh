#pragma once

#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"

#if !__has_include(<Eigen/Core>) || !__has_include(<Eigen/Geometry>)
#error "gwn_winding_number.cuh requires Eigen/Core and Eigen/Geometry in the include path."
#endif

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gwn {

template <class Real> using gwn_vec3 = Eigen::Matrix<Real, 3, 1>;

template <class Real>
__host__ __device__ inline Real gwn_signed_solid_angle_triangle(
    gwn_vec3<Real> const &a, gwn_vec3<Real> const &b, gwn_vec3<Real> const &c,
    gwn_vec3<Real> const &q
) noexcept;

template <class Real, class Index>
__device__ inline Real gwn_triangle_solid_angle_from_primitive(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_vec3<Real> const &query
) noexcept {
    if (primitive_id < Index(0) ||
        static_cast<std::size_t>(primitive_id) >= geometry.triangle_count()) {
        return Real(0);
    }

    std::size_t const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (ia < Index(0) || ib < Index(0) || ic < Index(0))
        return Real(0);

    std::size_t const a_index = static_cast<std::size_t>(ia);
    std::size_t const b_index = static_cast<std::size_t>(ib);
    std::size_t const c_index = static_cast<std::size_t>(ic);
    if (a_index >= geometry.vertex_count() || b_index >= geometry.vertex_count() ||
        c_index >= geometry.vertex_count()) {
        return Real(0);
    }

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

template <class Real>
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

template <class Real, class Index>
__device__ inline Real gwn_winding_number_point(
    gwn_geometry_accessor<Real, Index> const &geometry, Real const qx, Real const qy, Real const qz
) noexcept {
    if (!geometry.is_valid())
        return Real(0);

    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    std::size_t const triangle_count = geometry.triangle_count();
    gwn_vec3<Real> const query(qx, qy, qz);

    Real omega_sum = Real(0);
    for (std::size_t tri = 0; tri < triangle_count; ++tri) {
        omega_sum += gwn_triangle_solid_angle_from_primitive<Real, Index>(
            geometry, static_cast<Index>(tri), query
        );
    }

    return omega_sum / (Real(4) * k_pi);
}

template <class Real, class Index>
__device__ inline Real gwn_winding_number_point_bvh_exact(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> const &bvh,
    Real const qx, Real const qy, Real const qz
) noexcept {
    if (!geometry.is_valid() || !bvh.is_valid())
        return Real(0);

    constexpr Real k_pi = Real(3.141592653589793238462643383279502884L);
    constexpr int k_stack_capacity = 128;
    Index stack[k_stack_capacity];
    int stack_size = 0;

    gwn_vec3<Real> const query(qx, qy, qz);
    Real omega_sum = Real(0);
    if (bvh.root_kind == gwn_bvh_child_kind::k_leaf) {
        for (Index primitive_offset = 0; primitive_offset < bvh.root_count; ++primitive_offset) {
            Index const primitive_index =
                bvh.primitive_indices[static_cast<std::size_t>(bvh.root_index + primitive_offset)];
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
        if (node_index < Index(0) || static_cast<std::size_t>(node_index) >= bvh.nodes.size())
            continue;

        gwn_bvh4_node_soa<Real, Index> const &node =
            bvh.nodes[static_cast<std::size_t>(node_index)];
        GWN_PRAGMA_UNROLL
        for (int child_slot = 0; child_slot < 4; ++child_slot) {
            auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[child_slot]);
            if (child_kind == gwn_bvh_child_kind::k_invalid)
                continue;

            if (child_kind == gwn_bvh_child_kind::k_internal) {
                if (stack_size < k_stack_capacity)
                    stack[stack_size++] = node.child_index[child_slot];
                continue;
            }

            if (child_kind != gwn_bvh_child_kind::k_leaf)
                continue;

            Index const begin = node.child_index[child_slot];
            Index const count = node.child_count[child_slot];
            for (Index primitive_offset = 0; primitive_offset < count; ++primitive_offset) {
                Index const sorted_primitive_index = begin + primitive_offset;
                if (sorted_primitive_index < Index(0) ||
                    static_cast<std::size_t>(sorted_primitive_index) >=
                        bvh.primitive_indices.size()) {
                    continue;
                }
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

namespace detail {

template <class Real, class Index> struct gwn_winding_number_batch_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> output{};

    __device__ void operator()(std::size_t const query_id) const {
        output[query_id] = gwn_winding_number_point(
            geometry, query_x[query_id], query_y[query_id], query_z[query_id]
        );
    }
};

template <class Real, class Index> struct gwn_winding_number_batch_bvh_exact_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_accessor<Real, Index> bvh{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> output{};

    __device__ void operator()(std::size_t const query_id) const {
        output[query_id] = gwn_winding_number_point_bvh_exact(
            geometry, bvh, query_x[query_id], query_y[query_id], query_z[query_id]
        );
    }
};

template <class Real, class Index>
[[nodiscard]] inline gwn_winding_number_batch_functor<Real, Index>
gwn_make_winding_number_batch_functor(
    gwn_geometry_accessor<Real, Index> const &geometry, cuda::std::span<Real const> const query_x,
    cuda::std::span<Real const> const query_y, cuda::std::span<Real const> const query_z,
    cuda::std::span<Real> const output
) {
    return gwn_winding_number_batch_functor<Real, Index>{
        geometry, query_x, query_y, query_z, output
    };
}

template <class Real, class Index>
[[nodiscard]] inline gwn_winding_number_batch_bvh_exact_functor<Real, Index>
gwn_make_winding_number_batch_bvh_exact_functor(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> const &bvh,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output
) {
    return gwn_winding_number_batch_bvh_exact_functor<Real, Index>{geometry, bvh,     query_x,
                                                                   query_y,  query_z, output};
}

} // namespace detail

template <class Real, class Index = std::int64_t>
gwn_status gwn_compute_winding_number_batch(
    gwn_geometry_accessor<Real, Index> const &geometry, cuda::std::span<Real const> const query_x,
    cuda::std::span<Real const> const query_y, cuda::std::span<Real const> const query_z,
    cuda::std::span<Real> const output, cudaStream_t const stream = gwn_default_stream()
) noexcept {
    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");

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
        detail::gwn_make_winding_number_batch_functor<Real, Index>(
            geometry, query_x, query_y, query_z, output
        ),
        stream
    );
}

template <class Real, class Index = std::int64_t>
gwn_status gwn_compute_winding_number_batch_bvh_exact(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> const &bvh,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    if (!geometry.is_valid())
        return gwn_status::invalid_argument("Geometry accessor contains mismatched span lengths.");
    if (!bvh.is_valid())
        return gwn_status::invalid_argument("BVH accessor is invalid.");
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
        detail::gwn_make_winding_number_batch_bvh_exact_functor<Real, Index>(
            geometry, bvh, query_x, query_y, query_z, output
        ),
        stream
    );
}

} // namespace gwn
