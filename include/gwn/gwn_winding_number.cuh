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
inline constexpr int k_gwn_default_traversal_stack_capacity = 32;

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
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return Real(0);

    std::size_t const triangle_id = static_cast<std::size_t>(primitive_id);
    Index const ia = geometry.tri_i0[triangle_id];
    Index const ib = geometry.tri_i1[triangle_id];
    Index const ic = geometry.tri_i2[triangle_id];
    if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
        !gwn_index_in_bounds(ic, geometry.vertex_count())) {
        return Real(0);
    }

    std::size_t const a_index = static_cast<std::size_t>(ia);
    std::size_t const b_index = static_cast<std::size_t>(ib);
    std::size_t const c_index = static_cast<std::size_t>(ic);

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

template <
    int Width, class Real, class Index, int StackCapacity = k_gwn_default_traversal_stack_capacity>
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

template <int Width, class Real, class Index, int StackCapacity>
struct gwn_winding_number_batch_bvh_exact_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> bvh{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> output{};

    __device__ void operator()(std::size_t const query_id) const {
        output[query_id] = gwn_winding_number_point_bvh_exact<Width, Real, Index, StackCapacity>(
            geometry, bvh, query_x[query_id], query_y[query_id], query_z[query_id]
        );
    }
};

template <int Order, int Width, class Real, class Index> struct gwn_taylor_nodes_getter;

template <int Width, class Real, class Index>
struct gwn_taylor_nodes_getter<0, Width, Real, Index> {
    using span_type = cuda::std::span<gwn_bvh_taylor_node_soa<Width, 0, Real> const>;

    [[nodiscard]] __host__ __device__ static span_type
    get(gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree) {
        return data_tree.taylor_order0_nodes;
    }
};

template <int Width, class Real, class Index>
struct gwn_taylor_nodes_getter<1, Width, Real, Index> {
    using span_type = cuda::std::span<gwn_bvh_taylor_node_soa<Width, 1, Real> const>;

    [[nodiscard]] __host__ __device__ static span_type
    get(gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree) {
        return data_tree.taylor_order1_nodes;
    }
};

template <int Width, class Real, class Index>
struct gwn_taylor_nodes_getter<2, Width, Real, Index> {
    using span_type = cuda::std::span<gwn_bvh_taylor_node_soa<Width, 2, Real> const>;

    [[nodiscard]] __host__ __device__ static span_type
    get(gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree) {
        return data_tree.taylor_order2_nodes;
    }
};

template <int Order, int Width, class Real, class Index>
[[nodiscard]] __host__ __device__
    typename gwn_taylor_nodes_getter<Order, Width, Real, Index>::span_type
    gwn_get_taylor_nodes(gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree) {
    return gwn_taylor_nodes_getter<Order, Width, Real, Index>::get(data_tree);
}

template <
    int Order, int Width, class Real, class Index,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
__device__ inline Real gwn_winding_number_point_bvh_taylor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_tree_accessor<Width, Real, Index> const &data_tree, Real const qx, Real const qy,
    Real const qz, Real const accuracy_scale
) noexcept {
    static_assert(
        Order == 0 || Order == 1,
        "gwn_winding_number_point_bvh_taylor currently supports Order 0 and Order 1."
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

template <int Order, int Width, class Real, class Index, int StackCapacity>
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

template <int Width, class Real, class Index, int StackCapacity>
[[nodiscard]] inline gwn_winding_number_batch_bvh_exact_functor<Width, Real, Index, StackCapacity>
gwn_make_winding_number_batch_bvh_exact_functor(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output
) {
    return gwn_winding_number_batch_bvh_exact_functor<Width, Real, Index, StackCapacity>{
        geometry, bvh, query_x, query_y, query_z, output
    };
}

template <int Order, int Width, class Real, class Index, int StackCapacity>
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

/// \brief Compute winding numbers for a batch of query points by direct triangle summation.
template <class Real, class Index = std::uint32_t>
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

/// \brief Compute winding numbers for a batch using exact BVH traversal.
template <
    int Width, class Real, class Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_winding_number_batch_bvh_exact(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");

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
        detail::gwn_make_winding_number_batch_bvh_exact_functor<Width, Real, Index, StackCapacity>(
            geometry, bvh, query_x, query_y, query_z, output
        ),
        stream
    );
}

/// \brief Width-4 convenience wrapper for exact BVH batch queries.
template <
    class Real, class Index = std::uint32_t,
    int StackCapacity = k_gwn_default_traversal_stack_capacity>
gwn_status gwn_compute_winding_number_batch_bvh_exact(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Real, Index> const &bvh,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_compute_winding_number_batch_bvh_exact<4, Real, Index, StackCapacity>(
        geometry, bvh, query_x, query_y, query_z, output, stream
    );
}

/// \brief Compute winding numbers for a batch using Taylor-accelerated BVH traversal.
///
/// \remark Falls back to exact child descent per node when the approximation criterion fails.
template <
    int Order, int Width, class Real, class Index = std::uint32_t,
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
        Order == 0 || Order == 1,
        "gwn_compute_winding_number_batch_bvh_taylor currently supports Order 0 and Order 1."
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

/// \brief Width-4 convenience wrapper for Taylor BVH batch queries.
template <
    int Order, class Real, class Index = std::uint32_t,
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
