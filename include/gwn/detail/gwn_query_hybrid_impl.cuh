#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_harnack_trace_impl.cuh"
#include "gwn_query_ray_impl.cuh"

namespace gwn {
namespace detail {

enum class gwn_hybrid_hit_kind : std::uint8_t {
    k_miss = 0,
    k_triangle = 1,
    k_harnack = 2,
};

enum class gwn_hybrid_triangle_normal_policy : std::uint8_t {
    k_geometric = 0,
    k_barycentric_vertex = 1,
};

template <gwn_real_type Real, gwn_index_type Index> struct gwn_hybrid_trace_result {
    Real t{Real(-1)};
    Real winding{Real(0)};
    Real normal_x{Real(0)};
    Real normal_y{Real(0)};
    Real normal_z{Real(0)};
    Index primitive_id{gwn_invalid_index<Index>()};
    int iterations{0};
    gwn_hybrid_hit_kind hit_kind{gwn_hybrid_hit_kind::k_miss};

    __host__ __device__ constexpr bool hit() const noexcept { return t >= Real(0); }
};

template <gwn_real_type Real> struct gwn_hybrid_trace_arguments {
    Real target_winding{Real(0.5)};
    Real epsilon{Real(1e-4)};
    int max_iterations{512};
    Real t_min{Real(0)};
    Real t_max{Real(1e6)};
    Real accuracy_scale{Real(2)};
    Real conditioning_epsilon{Real(0)};
    gwn_hybrid_triangle_normal_policy triangle_normal_policy{
        gwn_hybrid_triangle_normal_policy::k_geometric
    };
};

template <gwn_real_type Real>
__device__ inline bool gwn_try_normalize_impl(gwn_query_vec3<Real> &v) noexcept {
    using std::isfinite;

    Real const n2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (!(n2 > Real(0)))
        return false;
    if (!isfinite(n2))
        return false;

    Real const inv_n = rsqrt(n2);
    if (!isfinite(inv_n))
        return false;

    v.x *= inv_n;
    v.y *= inv_n;
    v.z *= inv_n;
    return true;
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_triangle_indices_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id, Index &i0,
    Index &i1, Index &i2
) noexcept {
    if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count()))
        return false;

    std::size_t const tri = static_cast<std::size_t>(primitive_id);
    i0 = geometry.tri_i0[tri];
    i1 = geometry.tri_i1[tri];
    i2 = geometry.tri_i2[tri];

    if (!gwn_index_in_bounds(i0, geometry.vertex_count()) ||
        !gwn_index_in_bounds(i1, geometry.vertex_count()) ||
        !gwn_index_in_bounds(i2, geometry.vertex_count())) {
        return false;
    }

    return true;
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_triangle_vertices_and_indices_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_query_vec3<Real> &a, gwn_query_vec3<Real> &b, gwn_query_vec3<Real> &c, Index &ia, Index &ib,
    Index &ic
) noexcept {
    if (!gwn_triangle_indices_from_primitive_impl<Real, Index>(geometry, primitive_id, ia, ib, ic))
        return false;

    std::size_t const a_index = static_cast<std::size_t>(ia);
    std::size_t const b_index = static_cast<std::size_t>(ib);
    std::size_t const c_index = static_cast<std::size_t>(ic);

    a = gwn_query_vec3<Real>(
        geometry.vertex_x[a_index], geometry.vertex_y[a_index], geometry.vertex_z[a_index]
    );
    b = gwn_query_vec3<Real>(
        geometry.vertex_x[b_index], geometry.vertex_y[b_index], geometry.vertex_z[b_index]
    );
    c = gwn_query_vec3<Real>(
        geometry.vertex_x[c_index], geometry.vertex_y[c_index], geometry.vertex_z[c_index]
    );
    return true;
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_triangle_geometric_normal_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id,
    gwn_query_vec3<Real> &normal_out
) noexcept {
    gwn_query_vec3<Real> a{};
    gwn_query_vec3<Real> b{};
    gwn_query_vec3<Real> c{};
    Index ia{};
    Index ib{};
    Index ic{};
    if (!gwn_triangle_vertices_and_indices_from_primitive_impl<Real, Index>(
            geometry, primitive_id, a, b, c, ia, ib, ic
        )) {
        return false;
    }

    normal_out = gwn_query_cross(b - a, c - a);
    return gwn_try_normalize_impl(normal_out);
}

template <gwn_real_type Real, gwn_index_type Index>
__device__ inline bool gwn_triangle_barycentric_vertex_normal_from_primitive_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, Index const primitive_id, Real const hit_u,
    Real const hit_v, gwn_query_vec3<Real> &normal_out
) noexcept {
    if (geometry.vertex_nx.size() != geometry.vertex_count())
        return false;
    if (geometry.vertex_ny.size() != geometry.vertex_count())
        return false;
    if (geometry.vertex_nz.size() != geometry.vertex_count())
        return false;

    Index ia{};
    Index ib{};
    Index ic{};
    if (!gwn_triangle_indices_from_primitive_impl<Real, Index>(geometry, primitive_id, ia, ib, ic))
        return false;

    std::size_t const a_index = static_cast<std::size_t>(ia);
    std::size_t const b_index = static_cast<std::size_t>(ib);
    std::size_t const c_index = static_cast<std::size_t>(ic);
    gwn_query_vec3<Real> const na(
        geometry.vertex_nx[a_index], geometry.vertex_ny[a_index], geometry.vertex_nz[a_index]
    );
    gwn_query_vec3<Real> const nb(
        geometry.vertex_nx[b_index], geometry.vertex_ny[b_index], geometry.vertex_nz[b_index]
    );
    gwn_query_vec3<Real> const nc(
        geometry.vertex_nx[c_index], geometry.vertex_ny[c_index], geometry.vertex_nz[c_index]
    );

    Real const w1 = hit_u;
    Real const w2 = hit_v;
    Real const w0 = Real(1) - w1 - w2;
    normal_out = na * w0 + nb * w1 + nc * w2;
    return gwn_try_normalize_impl(normal_out);
}

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
__device__ inline gwn_hybrid_trace_result<Real, Index> gwn_hybrid_trace_ray_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_aabb_accessor<Width, Real, Index> const &aabb_tree,
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> const &moment_tree, Real const ray_ox,
    Real const ray_oy, Real const ray_oz, Real const ray_dx, Real const ray_dy, Real const ray_dz,
    gwn_hybrid_trace_arguments<Real> const &arguments
) noexcept {
    static_assert(
        Order == 0 || Order == 1 || Order == 2,
        "gwn_hybrid_trace_ray currently supports Order 0, 1, and 2."
    );

    gwn_hybrid_trace_result<Real, Index> result{};

    if (!(arguments.t_max >= arguments.t_min))
        return result;

    Real const dir_len2 = ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz;
    if (!(dir_len2 > Real(0)))
        return result;

    auto const mesh_hit = gwn_ray_first_hit_bvh_impl<Width, Real, Index, StackCapacity>(
        geometry, bvh, aabb_tree, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, arguments.t_min,
        arguments.t_max
    );

    Real fill_global_t = Real(-1);
    gwn_harnack_trace_result<Real> fill_hit{};
    if (geometry.has_singular_edges()) {
        Real harnack_t_max = arguments.t_max;
        Real const conditioning_epsilon = std::max(arguments.conditioning_epsilon, Real(0));
        if (mesh_hit.hit()) {
            Real const conditioned_t_max = mesh_hit.t - conditioning_epsilon;
            harnack_t_max = std::min(harnack_t_max, conditioned_t_max);
        }

        if (harnack_t_max > arguments.t_min) {
            Real const harnack_t_min = arguments.t_min;
            Real const local_t_max = harnack_t_max - harnack_t_min;
            Real const start_ox = ray_ox + harnack_t_min * ray_dx;
            Real const start_oy = ray_oy + harnack_t_min * ray_dy;
            Real const start_oz = ray_oz + harnack_t_min * ray_dz;

            fill_hit = gwn_harnack_trace_ray_impl<Order, Width, Real, Index, StackCapacity>(
                geometry, bvh, aabb_tree, moment_tree, start_ox, start_oy, start_oz, ray_dx, ray_dy,
                ray_dz, arguments.target_winding, arguments.epsilon, arguments.max_iterations,
                local_t_max, arguments.accuracy_scale
            );
            if (fill_hit.hit())
                fill_global_t = fill_hit.t + harnack_t_min;
        }
    }

    bool choose_fill = false;
    if (fill_global_t >= Real(0)) {
        choose_fill = true;
        if (mesh_hit.hit()) {
            if (fill_global_t >= mesh_hit.t)
                choose_fill = false;
        }
    }

    if (choose_fill) {
        result.t = fill_global_t;
        result.winding = fill_hit.winding;
        result.iterations = fill_hit.iterations;
        result.hit_kind = gwn_hybrid_hit_kind::k_harnack;

        gwn_query_vec3<Real> normal(fill_hit.normal_x, fill_hit.normal_y, fill_hit.normal_z);
        if (!gwn_try_normalize_impl(normal))
            normal = gwn_query_vec3<Real>(Real(0), Real(0), Real(0));
        result.normal_x = normal.x;
        result.normal_y = normal.y;
        result.normal_z = normal.z;
        return result;
    }

    if (!mesh_hit.hit())
        return result;

    result.t = mesh_hit.t;
    result.primitive_id = mesh_hit.primitive_id;
    result.hit_kind = gwn_hybrid_hit_kind::k_triangle;
    result.iterations = fill_hit.iterations;
    result.winding = Real(0);

    gwn_query_vec3<Real> normal(Real(0), Real(0), Real(0));
    bool normal_ok = false;
    if (arguments.triangle_normal_policy == gwn_hybrid_triangle_normal_policy::k_barycentric_vertex)
        normal_ok = gwn_triangle_barycentric_vertex_normal_from_primitive_impl<Real, Index>(
            geometry, mesh_hit.primitive_id, mesh_hit.u, mesh_hit.v, normal
        );
    else if (arguments.triangle_normal_policy == gwn_hybrid_triangle_normal_policy::k_geometric)
        normal_ok = gwn_triangle_geometric_normal_from_primitive_impl<Real, Index>(
            geometry, mesh_hit.primitive_id, normal
        );

    if (!normal_ok)
        normal = gwn_query_vec3<Real>(Real(0), Real(0), Real(0));

    result.normal_x = normal.x;
    result.normal_y = normal.y;
    result.normal_z = normal.z;
    return result;
}

template <int Order, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity>
struct gwn_hybrid_trace_batch_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> bvh{};
    gwn_bvh_aabb_accessor<Width, Real, Index> aabb_tree{};
    gwn_bvh_moment_tree_accessor<Width, Order, Real, Index> moment_tree{};

    cuda::std::span<Real const> ray_origin_x{};
    cuda::std::span<Real const> ray_origin_y{};
    cuda::std::span<Real const> ray_origin_z{};
    cuda::std::span<Real const> ray_dir_x{};
    cuda::std::span<Real const> ray_dir_y{};
    cuda::std::span<Real const> ray_dir_z{};

    cuda::std::span<Real> output_t{};
    cuda::std::span<Real> output_normal_x{};
    cuda::std::span<Real> output_normal_y{};
    cuda::std::span<Real> output_normal_z{};
    cuda::std::span<Index> output_primitive_id{};

    gwn_hybrid_trace_arguments<Real> arguments{};

    __device__ void operator()(std::size_t const ray_id) const {
        auto const hit = gwn_hybrid_trace_ray_impl<Order, Width, Real, Index, StackCapacity>(
            geometry, bvh, aabb_tree, moment_tree, ray_origin_x[ray_id], ray_origin_y[ray_id],
            ray_origin_z[ray_id], ray_dir_x[ray_id], ray_dir_y[ray_id], ray_dir_z[ray_id], arguments
        );
        output_t[ray_id] = hit.t;
        output_normal_x[ray_id] = hit.normal_x;
        output_normal_y[ray_id] = hit.normal_y;
        output_normal_z[ray_id] = hit.normal_z;
        output_primitive_id[ray_id] = hit.primitive_id;
    }
};

} // namespace detail
} // namespace gwn
