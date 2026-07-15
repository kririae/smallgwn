#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/detail/gwn_utils.cuh>
#include <gwn/gwn.cuh>

#include "test_fixtures.cuh"
#include "test_utils.cuh"

using Real = gwn::tests::Real;
using GwnQueryDistanceRayTest = gwn::tests::CudaFixture;

namespace {

constexpr gwn::gwn_query_batch_config k_stack_capacity_one_config{
    .block_size = gwn::k_gwn_default_query_batch_block_size,
    .stack_capacity = 1,
};

template <gwn::gwn_index_type Index> struct HostMesh {
    std::vector<Real> x;
    std::vector<Real> y;
    std::vector<Real> z;
    std::vector<Index> i0;
    std::vector<Index> i1;
    std::vector<Index> i2;
};

template <gwn::gwn_index_type Index>
[[nodiscard]] HostMesh<Index> make_layered_triangles(std::size_t const triangle_count) {
    HostMesh<Index> mesh;
    mesh.x.reserve(3 * triangle_count);
    mesh.y.reserve(3 * triangle_count);
    mesh.z.reserve(3 * triangle_count);
    mesh.i0.reserve(triangle_count);
    mesh.i1.reserve(triangle_count);
    mesh.i2.reserve(triangle_count);
    for (std::size_t triangle_index = 0; triangle_index < triangle_count; ++triangle_index) {
        Index const vertex = static_cast<Index>(mesh.x.size());
        Real const z = static_cast<Real>(triangle_index);
        mesh.x.insert(mesh.x.end(), {Real(0), Real(1), Real(0)});
        mesh.y.insert(mesh.y.end(), {Real(0), Real(0), Real(1)});
        mesh.z.insert(mesh.z.end(), {z, z, z});
        mesh.i0.push_back(vertex);
        mesh.i1.push_back(vertex + Index(1));
        mesh.i2.push_back(vertex + Index(2));
    }
    return mesh;
}

template <gwn::gwn_index_type Index>
void upload_and_build(
    HostMesh<Index> const &mesh, gwn::gwn_geometry_object<Real, Index> &geometry,
    gwn::gwn_bvh4_object<Real, Index> &bvh, gwn::gwn_bvh_build_method const method
) {
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        geometry, gwn::tests::host_span(cuda::std::span<Real const>(mesh.x.data(), mesh.x.size())),
        gwn::tests::host_span(cuda::std::span<Real const>(mesh.y.data(), mesh.y.size())),
        gwn::tests::host_span(cuda::std::span<Real const>(mesh.z.data(), mesh.z.size())),
        gwn::tests::host_span(cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size())),
        gwn::tests::host_span(cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size())),
        gwn::tests::host_span(cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size()))
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    status = gwn::gwn_build_bvh(geometry, bvh, gwn::gwn_bvh_build_options{.method = method});
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
}

struct overflow_probe {
    int *flag = nullptr;

    void __device__ operator()() const noexcept {
        if (flag != nullptr)
            *flag = 1;
    }
};

template <gwn::gwn_index_type Index, int StackCapacity>
__global__ void spatial_point_query_kernel(
    gwn::gwn_bvh4_accessor<Real, Index> const bvh, Real const ray_ox, Real const ray_oy,
    Real const ray_oz, Real const ray_dx, Real const ray_dy, Real const ray_dz, Real const t_min,
    Real const t_max, Real const qx, Real const qy, Real const qz, Real const culling_band,
    gwn::gwn_ray_first_hit_result<Real, Index> *hit, Real *distance, int *overflow_flags
) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    *hit = gwn::gwn_ray_first_hit<4, Real, Index, StackCapacity, overflow_probe>(
        bvh, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, t_min, t_max,
        overflow_probe{overflow_flags}
    );
    *distance = gwn::gwn_unsigned_distance<4, Real, Index, StackCapacity, overflow_probe>(
        bvh, qx, qy, qz, culling_band, overflow_probe{overflow_flags + 1}
    );
}

template <gwn::gwn_index_type Index> struct PointResults {
    gwn::gwn_ray_first_hit_result<Real, Index> hit{};
    Real distance = Real(0);
    std::array<int, 2> overflow_flags{};
};

template <gwn::gwn_index_type Index, int StackCapacity = 64>
[[nodiscard]] PointResults<Index> run_point_queries(
    gwn::gwn_bvh4_object<Real, Index> const &bvh, Real const ray_ox, Real const ray_oy,
    Real const ray_oz, Real const ray_dx, Real const ray_dy, Real const ray_dz, Real const t_min,
    Real const t_max, Real const qx, Real const qy, Real const qz, Real const culling_band
) {
    PointResults<Index> result{};
    gwn::detail::gwn_device_array<gwn::gwn_ray_first_hit_result<Real, Index>> device_hit;
    gwn::detail::gwn_device_array<Real> device_distance;
    gwn::detail::gwn_device_array<int> device_flags;
    device_hit.resize(1);
    device_distance.resize(1);
    device_flags.resize(2);
    device_flags.zero();

    spatial_point_query_kernel<Index, StackCapacity><<<1, 1>>>(
        bvh.accessor(), ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz, t_min, t_max, qx, qy, qz,
        culling_band, device_hit.data(), device_distance.data(), device_flags.data()
    );
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    device_hit.copy_to_host(
        cuda::std::span<gwn::gwn_ray_first_hit_result<Real, Index>>(&result.hit, 1)
    );
    device_distance.copy_to_host(cuda::std::span<Real>(&result.distance, 1));
    device_flags.copy_to_host(cuda::std::span<int>(result.overflow_flags));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return result;
}

template <gwn::gwn_index_type Index> struct BatchResults {
    std::vector<gwn::gwn_ray_first_hit_result<Real, Index>> hits;
    std::vector<Real> distance;
};

template <
    gwn::gwn_index_type Index, int StackCapacity = gwn::k_gwn_default_traversal_stack_capacity>
[[nodiscard]] bool run_batch_queries(
    gwn::gwn_bvh4_object<Real, Index> const &bvh, std::vector<Real> const &ray_ox,
    std::vector<Real> const &ray_oy, std::vector<Real> const &ray_oz,
    std::vector<Real> const &ray_dx, std::vector<Real> const &ray_dy,
    std::vector<Real> const &ray_dz, std::vector<Real> const &query_x,
    std::vector<Real> const &query_y, std::vector<Real> const &query_z, BatchResults<Index> &result,
    Real const culling_band = std::numeric_limits<Real>::infinity(), Real const ray_t_min = Real(0),
    Real const ray_t_max = std::numeric_limits<Real>::infinity()
) {
    constexpr gwn::gwn_query_batch_config config{
        .block_size = gwn::k_gwn_default_query_batch_block_size,
        .stack_capacity = StackCapacity,
    };
    std::size_t const count = ray_ox.size();
    if (ray_oy.size() != count || ray_oz.size() != count || ray_dx.size() != count ||
        ray_dy.size() != count || ray_dz.size() != count || query_x.size() != count ||
        query_y.size() != count || query_z.size() != count) {
        return false;
    }

    gwn::detail::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
    gwn::detail::gwn_device_array<Real> d_qx, d_qy, d_qz, d_distance;
    gwn::detail::gwn_device_array<gwn::gwn_ray_first_hit_result<Real, Index>> d_hits;
    d_ox.resize(count);
    d_oy.resize(count);
    d_oz.resize(count);
    d_dx.resize(count);
    d_dy.resize(count);
    d_dz.resize(count);
    d_qx.resize(count);
    d_qy.resize(count);
    d_qz.resize(count);
    d_hits.resize(count);
    d_distance.resize(count);

    auto upload = [](auto &device, auto const &host) {
        using value_type = typename std::remove_reference_t<decltype(host)>::value_type;
        device.copy_from_host(cuda::std::span<value_type const>(host.data(), host.size()));
    };
    upload(d_ox, ray_ox);
    upload(d_oy, ray_oy);
    upload(d_oz, ray_oz);
    upload(d_dx, ray_dx);
    upload(d_dy, ray_dy);
    upload(d_dz, ray_dz);
    upload(d_qx, query_x);
    upload(d_qy, query_y);
    upload(d_qz, query_z);
    auto as_input = [](gwn::detail::gwn_device_array<Real> const &array) {
        return gwn::gwn_device_span<Real const>(array.data(), array.size());
    };

    gwn::gwn_status status = gwn::gwn_compute_ray_first_hit_batch<config, 4, Real, Index>(
        bvh, as_input(d_ox), as_input(d_oy), as_input(d_oz), as_input(d_dx), as_input(d_dy),
        as_input(d_dz), gwn::tests::device_span(d_hits.span()), ray_t_min, ray_t_max
    );
    if (!status.is_ok()) {
        ADD_FAILURE() << gwn::tests::status_to_debug_string(status);
        return false;
    }
    status = gwn::gwn_compute_unsigned_distance_batch<config, 4, Real, Index>(
        bvh, as_input(d_qx), as_input(d_qy), as_input(d_qz),
        gwn::tests::device_span(d_distance.span()), culling_band
    );
    if (!status.is_ok()) {
        ADD_FAILURE() << gwn::tests::status_to_debug_string(status);
        return false;
    }

    result.hits.resize(count);
    result.distance.resize(count);
    d_hits.copy_to_host(cuda::std::span<gwn::gwn_ray_first_hit_result<Real, Index>>(result.hits));
    d_distance.copy_to_host(cuda::std::span<Real>(result.distance.data(), result.distance.size()));
    return cudaDeviceSynchronize() == cudaSuccess;
}

TEST_F(
    GwnQueryDistanceRayTest, canonical_leaf_queries_preserve_hit_barycentric_and_distance_contracts
) {
    HostMesh<std::uint32_t> mesh = make_layered_triangles<std::uint32_t>(1);
    gwn::gwn_geometry_object<Real, std::uint32_t> geometry;
    gwn::gwn_bvh4_object<Real, std::uint32_t> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_lbvh);
    ASSERT_TRUE(bvh.accessor().has_leaf_root());

    auto result = run_point_queries(
        bvh, Real(0.25), Real(0.25), Real(1), Real(0), Real(0), Real(-1), Real(0),
        std::numeric_limits<Real>::infinity(), Real(0.25), Real(0.25), Real(2),
        std::numeric_limits<Real>::infinity()
    );
    EXPECT_TRUE(result.hit.hit());
    EXPECT_EQ(result.hit.status, gwn::gwn_ray_first_hit_status::k_hit);
    EXPECT_EQ(result.hit.primitive_id, 0u);
    EXPECT_FLOAT_EQ(result.hit.t, Real(1));
    EXPECT_NEAR(result.hit.u, Real(0.25), Real(1e-6));
    EXPECT_NEAR(result.hit.v, Real(0.25), Real(1e-6));
    EXPECT_FLOAT_EQ(result.hit.geometric_normal_x, Real(0));
    EXPECT_FLOAT_EQ(result.hit.geometric_normal_y, Real(0));
    EXPECT_FLOAT_EQ(result.hit.geometric_normal_z, Real(1));
    EXPECT_FLOAT_EQ(result.distance, Real(2));
    EXPECT_EQ(result.overflow_flags, (std::array<int, 2>{0, 0}));

    result = run_point_queries(
        bvh, Real(0.25), Real(0.25), Real(1), Real(0), Real(0), Real(-1), Real(1), Real(1), Real(2),
        Real(0), Real(0), std::numeric_limits<Real>::infinity()
    );
    EXPECT_TRUE(result.hit.hit());
    EXPECT_FLOAT_EQ(result.hit.t, Real(1));
    EXPECT_FLOAT_EQ(result.distance, Real(1));

    result = run_point_queries(
        bvh, Real(0.25), Real(0.25), Real(1), Real(0), Real(0), Real(0), Real(0), Real(10),
        Real(0.25), Real(0.25), Real(2), Real(0.5)
    );
    EXPECT_EQ(result.hit.status, gwn::gwn_ray_first_hit_status::k_miss);
    EXPECT_EQ(result.hit.primitive_id, gwn::gwn_invalid_index<std::uint32_t>());
    EXPECT_FLOAT_EQ(result.hit.t, Real(-1));
    EXPECT_FLOAT_EQ(result.distance, Real(0.5));

    result = run_point_queries(
        bvh, Real(2), Real(2), Real(1), Real(0), Real(0), Real(-1), Real(0), Real(10), Real(0.25),
        Real(0.25), Real(2), Real(-4)
    );
    EXPECT_FLOAT_EQ(result.distance, Real(0));
}

TEST_F(GwnQueryDistanceRayTest, canonical_ray_query_handles_zero_axes_and_coplanar_rays) {
    using Index = std::uint32_t;
    HostMesh<Index> mesh = make_layered_triangles<Index>(1);
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_lbvh);

    auto result = run_point_queries(
        bvh, Real(0.25), Real(0.25), Real(1), Real(1e-30), Real(0), Real(-1), Real(0), Real(2),
        Real(0), Real(0), Real(1), std::numeric_limits<Real>::infinity()
    );
    EXPECT_TRUE(result.hit.hit());
    EXPECT_NEAR(result.hit.t, Real(1), Real(1e-6));

    result = run_point_queries(
        bvh, Real(0.25), Real(0.25), Real(0), Real(1), Real(0), Real(0), Real(0), Real(2), Real(0),
        Real(0), Real(1), std::numeric_limits<Real>::infinity()
    );
    EXPECT_EQ(result.hit.status, gwn::gwn_ray_first_hit_status::k_miss);

    result = run_point_queries(
        bvh, Real(0.25), Real(0.25), Real(1e-6), Real(0), Real(0), Real(-1), Real(0), Real(1e-5),
        Real(0), Real(0), Real(1), std::numeric_limits<Real>::infinity()
    );
    EXPECT_TRUE(result.hit.hit());
    EXPECT_NEAR(result.hit.t, Real(1e-6), Real(1e-10));
}

TEST_F(GwnQueryDistanceRayTest, canonical_ray_query_preserves_tiny_triangle_barycentrics) {
    using Index = std::uint32_t;
    constexpr Real scale = Real(1e-10);
    HostMesh<Index> mesh;
    mesh.x = {Real(0), scale, Real(0)};
    mesh.y = {Real(0), Real(0), scale};
    mesh.z = {Real(0), Real(0), Real(0)};
    mesh.i0 = {0};
    mesh.i1 = {1};
    mesh.i2 = {2};
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_lbvh);

    auto const result = run_point_queries(
        bvh, scale * Real(0.25), scale * Real(0.25), Real(1), Real(0), Real(0), Real(-1), Real(0),
        Real(2), Real(0), Real(0), Real(1), std::numeric_limits<Real>::infinity()
    );

    ASSERT_TRUE(result.hit.hit());
    EXPECT_NEAR(result.hit.u, Real(0.25), Real(1e-6));
    EXPECT_NEAR(result.hit.v, Real(0.25), Real(1e-6));
}

TEST_F(GwnQueryDistanceRayTest, canonical_ray_query_normalizes_subnormal_pluecker_weights) {
    using Index = std::uint32_t;
    constexpr Real scale = Real(1e-20);
    HostMesh<Index> mesh;
    mesh.x = {Real(0), scale, Real(0)};
    mesh.y = {Real(0), Real(0), scale};
    mesh.z = {Real(0), Real(0), Real(0)};
    mesh.i0 = {0};
    mesh.i1 = {1};
    mesh.i2 = {2};
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_lbvh);

    auto const result = run_point_queries(
        bvh, scale * Real(0.25), scale * Real(0.25), Real(1), Real(0), Real(0), Real(-1), Real(0),
        Real(2), Real(0), Real(0), Real(1), std::numeric_limits<Real>::infinity()
    );

    ASSERT_TRUE(result.hit.hit());
    EXPECT_FLOAT_EQ(result.hit.t, Real(1));
    EXPECT_NEAR(result.hit.u, Real(0.25), Real(1e-5));
    EXPECT_NEAR(result.hit.v, Real(0.25), Real(1e-5));
    EXPECT_GT(result.hit.geometric_normal_z, Real(0));
}

TEST_F(
    GwnQueryDistanceRayTest,
    canonical_internal_queries_use_bounds_records_and_original_primitive_ids
) {
    for (gwn::gwn_bvh_build_method const method :
         {gwn::gwn_bvh_build_method::k_lbvh, gwn::gwn_bvh_build_method::k_hploc}) {
        HostMesh<std::uint32_t> mesh = make_layered_triangles<std::uint32_t>(12);
        gwn::gwn_geometry_object<Real, std::uint32_t> geometry;
        gwn::gwn_bvh4_object<Real, std::uint32_t> bvh;
        upload_and_build(mesh, geometry, bvh, method);
        ASSERT_TRUE(bvh.accessor().has_internal_root());

        BatchResults<std::uint32_t> result;
        ASSERT_TRUE(run_batch_queries(
            bvh, {Real(0.25), Real(2), Real(0)}, {Real(0.25), Real(2), Real(0.25)},
            {Real(20), Real(20), Real(20)}, {Real(0), Real(0), Real(0)},
            {Real(0), Real(0), Real(0)}, {Real(-1), Real(-1), Real(-1)},
            {Real(0.25), Real(0.25), Real(0)}, {Real(0.25), Real(0.25), Real(0.25)},
            {Real(10.25), Real(30), Real(10.25)}, result, Real(2)
        ));
        ASSERT_EQ(result.hits.size(), 3u);
        EXPECT_FLOAT_EQ(result.hits[0].t, Real(9));
        EXPECT_EQ(result.hits[0].primitive_id, 11u);
        EXPECT_FLOAT_EQ(result.hits[1].t, Real(-1));
        EXPECT_EQ(result.hits[1].primitive_id, gwn::gwn_invalid_index<std::uint32_t>());
        EXPECT_FLOAT_EQ(result.hits[2].t, Real(9));
        EXPECT_EQ(result.hits[2].primitive_id, 11u);
        EXPECT_FLOAT_EQ(result.distance[0], Real(0.25));
        EXPECT_FLOAT_EQ(result.distance[1], Real(2));
        EXPECT_FLOAT_EQ(result.distance[2], Real(0.25));
    }
}

TEST_F(GwnQueryDistanceRayTest, canonical_ray_query_matches_analytic_oblique_octahedron_hits) {
    using Index = std::uint32_t;
    HostMesh<Index> mesh;
    mesh.x = {Real(1), Real(-1), Real(0), Real(0), Real(0), Real(0)};
    mesh.y = {Real(0), Real(0), Real(1), Real(-1), Real(0), Real(0)};
    mesh.z = {Real(0), Real(0), Real(0), Real(0), Real(1), Real(-1)};
    mesh.i0 = {0, 2, 1, 3, 2, 1, 3, 0};
    mesh.i1 = {2, 1, 3, 0, 0, 2, 1, 3};
    mesh.i2 = {4, 4, 4, 4, 5, 5, 5, 5};
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_hploc);

    BatchResults<Index> result;
    ASSERT_TRUE(run_batch_queries(
        bvh, {Real(2), Real(2), Real(2)}, {Real(2), Real(0.2), Real(2)},
        {Real(2), Real(0.1), Real(2)}, {Real(-1), Real(-1), Real(1)}, {Real(-1), Real(0), Real(1)},
        {Real(-1), Real(0), Real(1)}, {Real(0), Real(0), Real(0)}, {Real(0), Real(0), Real(0)},
        {Real(0), Real(0), Real(0)}, result
    ));
    ASSERT_EQ(result.hits.size(), 3u);
    EXPECT_NEAR(result.hits[0].t, Real(5.0 / 3.0), Real(1e-5));
    EXPECT_EQ(result.hits[0].primitive_id, 0u);
    EXPECT_NEAR(result.hits[1].t, Real(1.3), Real(1e-5));
    EXPECT_EQ(result.hits[1].primitive_id, 0u);
    EXPECT_FLOAT_EQ(result.hits[2].t, Real(-1));
    EXPECT_EQ(result.hits[2].primitive_id, gwn::gwn_invalid_index<Index>());
}

TEST_F(GwnQueryDistanceRayTest, canonical_ray_query_accepts_shared_edge_grazing_hit) {
    HostMesh<std::uint32_t> mesh;
    mesh.x = {Real(0), Real(1), Real(1), Real(0)};
    mesh.y = {Real(0), Real(0), Real(1), Real(1)};
    mesh.z = {Real(0), Real(0), Real(0), Real(0)};
    mesh.i0 = {0, 0};
    mesh.i1 = {1, 2};
    mesh.i2 = {2, 3};
    gwn::gwn_geometry_object<Real, std::uint32_t> geometry;
    gwn::gwn_bvh4_object<Real, std::uint32_t> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_hploc);

    auto const result = run_point_queries(
        bvh, Real(0.5), Real(0.5), Real(1), Real(0), Real(0), Real(-1), Real(0), Real(2), Real(0.5),
        Real(0.5), Real(0), std::numeric_limits<Real>::infinity()
    );
    EXPECT_TRUE(result.hit.hit());
    EXPECT_FLOAT_EQ(result.hit.t, Real(1));
    EXPECT_TRUE(result.hit.primitive_id == 0u || result.hit.primitive_id == 1u);
    EXPECT_TRUE(std::isfinite(result.hit.u));
    EXPECT_TRUE(std::isfinite(result.hit.v));
    EXPECT_FLOAT_EQ(result.distance, Real(0));
}

TEST_F(GwnQueryDistanceRayTest, packed_batch_preserves_parallel_slab_interval_for_thin_bounds) {
    using Index = std::uint32_t;
    HostMesh<Index> mesh = make_layered_triangles<Index>(16);
    for (Real &x : mesh.x)
        x *= Real(1e-25);

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_hploc);
    ASSERT_TRUE(bvh.accessor().has_internal_root());

    Real const ray_x = Real(0.25e-25);
    auto const point = run_point_queries(
        bvh, ray_x, Real(0.25), Real(-1), Real(0), Real(0), Real(1), Real(0.5), Real(2), ray_x,
        Real(0.25), Real(0), std::numeric_limits<Real>::infinity()
    );
    ASSERT_TRUE(point.hit.hit());

    BatchResults<Index> batch;
    ASSERT_TRUE(run_batch_queries(
        bvh, {ray_x}, {Real(0.25)}, {Real(-1)}, {Real(0)}, {Real(0)}, {Real(1)}, {ray_x},
        {Real(0.25)}, {Real(0)}, batch
    ));
    ASSERT_EQ(batch.hits.size(), 1u);
    EXPECT_TRUE(batch.hits[0].hit());
    EXPECT_EQ(batch.hits[0].primitive_id, point.hit.primitive_id);
    EXPECT_FLOAT_EQ(batch.hits[0].t, point.hit.t);
}

TEST_F(GwnQueryDistanceRayTest, packed_batch_orders_negative_child_entry_distances) {
    using Index = std::uint32_t;
    HostMesh<Index> mesh = make_layered_triangles<Index>(16);
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_hploc);
    ASSERT_TRUE(bvh.accessor().has_internal_root());

    BatchResults<Index> batch;
    ASSERT_TRUE(run_batch_queries(
        bvh, {Real(0.25)}, {Real(0.25)}, {Real(20)}, {Real(0)}, {Real(0)}, {Real(1)}, {Real(0.25)},
        {Real(0.25)}, {Real(0)}, batch, std::numeric_limits<Real>::infinity(), Real(-30), Real(0)
    ));
    ASSERT_EQ(batch.hits.size(), 1u);
    EXPECT_TRUE(batch.hits[0].hit());
    EXPECT_EQ(batch.hits[0].primitive_id, 0u);
    EXPECT_FLOAT_EQ(batch.hits[0].t, Real(-20));
}

TEST_F(GwnQueryDistanceRayTest, canonical_batch_queries_reject_invalid_contracts_before_launch) {
    using Index = std::uint32_t;
    gwn::gwn_bvh4_object<Real, Index> empty_bvh;
    gwn::gwn_device_span<Real const> empty_input{};
    gwn::gwn_device_span<Real> empty_real_output{};
    gwn::gwn_device_span<gwn::gwn_ray_first_hit_result<Real, Index>> empty_hit_output{};
    EXPECT_EQ(
        gwn::gwn_compute_ray_first_hit_batch(
            empty_bvh, empty_input, empty_input, empty_input, empty_input, empty_input, empty_input,
            empty_hit_output
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );
    EXPECT_EQ(
        gwn::gwn_compute_unsigned_distance_batch(
            empty_bvh, empty_input, empty_input, empty_input, empty_real_output
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );

    HostMesh<Index> mesh = make_layered_triangles<Index>(16);
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_hploc);
    ASSERT_GT(bvh.accessor().internal_stack_bound, 1u);
    EXPECT_EQ(
        (gwn::gwn_compute_ray_first_hit_batch<k_stack_capacity_one_config, 4, Real, Index>(
             bvh, empty_input, empty_input, empty_input, empty_input, empty_input, empty_input,
             empty_hit_output
        )
             .error()),
        gwn::gwn_error::invalid_argument
    );
    EXPECT_EQ(
        (gwn::gwn_compute_unsigned_distance_batch<k_stack_capacity_one_config, 4, Real, Index>(
             bvh, empty_input, empty_input, empty_input, empty_real_output
        )
             .error()),
        gwn::gwn_error::invalid_argument
    );
    EXPECT_EQ(
        gwn::gwn_compute_ray_first_hit_batch(
            bvh, empty_input, empty_input, empty_input, empty_input, empty_input, empty_input,
            empty_hit_output, Real(2), Real(1)
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );

    gwn::detail::gwn_device_array<Real> one_real;
    one_real.resize(1);
    auto const one_input = gwn::gwn_device_span<Real const>(one_real.data(), one_real.size());
    EXPECT_EQ(
        gwn::gwn_compute_ray_first_hit_batch(
            bvh, one_input, empty_input, empty_input, empty_input, empty_input, empty_input,
            empty_hit_output
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );
    EXPECT_EQ(
        gwn::gwn_compute_unsigned_distance_batch(
            bvh, one_input, empty_input, empty_input, empty_real_output
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );
}

TEST_F(GwnQueryDistanceRayTest, leaf_children_do_not_consume_traversal_stack_capacity) {
    using Index = std::uint32_t;
    HostMesh<Index> mesh = make_layered_triangles<Index>(4);
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_lbvh);
    ASSERT_TRUE(bvh.accessor().has_internal_root());
    ASSERT_EQ(bvh.accessor().internal_stack_bound, 0u);
    ASSERT_EQ(bvh.accessor().packed_stack_bound, 3u);

    gwn::gwn_device_span<Real const> empty_input{};
    gwn::gwn_device_span<Real> empty_real_output{};
    gwn::gwn_device_span<gwn::gwn_ray_first_hit_result<Real, Index>> empty_hit_output{};
    EXPECT_TRUE((gwn::gwn_compute_ray_first_hit_batch<k_stack_capacity_one_config, 4, Real, Index>(
                     bvh, empty_input, empty_input, empty_input, empty_input, empty_input,
                     empty_input, empty_hit_output
                 ))
                    .is_ok());
    EXPECT_TRUE(
        (gwn::gwn_compute_unsigned_distance_batch<k_stack_capacity_one_config, 4, Real, Index>(
             bvh, empty_input, empty_input, empty_input, empty_real_output
         ))
            .is_ok()
    );

    BatchResults<Index> batch;
    ASSERT_TRUE((run_batch_queries<Index, 1>(
        bvh, {Real(0.25)}, {Real(0.25)}, {Real(5)}, {Real(0)}, {Real(0)}, {Real(-1)}, {Real(0.25)},
        {Real(0.25)}, {Real(3.5)}, batch
    )));
    ASSERT_EQ(batch.hits.size(), 1u);
    EXPECT_FLOAT_EQ(batch.hits[0].t, Real(2));
    EXPECT_EQ(batch.hits[0].primitive_id, Index(3));
    EXPECT_FLOAT_EQ(batch.distance[0], Real(0.5));

    auto const result = run_point_queries<Index, 1>(
        bvh, Real(0.25), Real(0.25), Real(5), Real(0), Real(0), Real(-1), Real(0), Real(6),
        Real(0.25), Real(0.25), Real(3.5), std::numeric_limits<Real>::infinity()
    );
    EXPECT_TRUE(result.hit.hit());
    EXPECT_EQ(result.hit.primitive_id, 3u);
    EXPECT_FLOAT_EQ(result.hit.t, Real(2));
    EXPECT_FLOAT_EQ(result.distance, Real(0.5));
    EXPECT_EQ(result.overflow_flags, (std::array<int, 2>{0, 0}));
}

TEST_F(GwnQueryDistanceRayTest, exact_stack_bound_accepts_single_internal_child_chain) {
    using Index = std::uint32_t;
    using Child = gwn::gwn_bvh_child<Real>;
    using Node = gwn::gwn_bvh_node<4, Real>;

    auto const encode_reference = [](std::uint64_t const offset,
                                     std::uint64_t const primitive_count) {
        return Child::k_valid_mask | offset | (primitive_count << Child::k_primitive_count_shift);
    };
    auto const set_bounds = [](Child &child, Real const min_z, Real const max_z) {
        child.bounds = {
            .min_x = Real(0),
            .min_y = Real(0),
            .min_z = min_z,
            .max_x = Real(1),
            .max_y = Real(1),
            .max_z = max_z,
        };
    };

    std::array<Node, 3> nodes{};
    for (std::size_t node_index = 0; node_index < 2; ++node_index) {
        auto &leaf = nodes[node_index].child(0);
        set_bounds(leaf, static_cast<Real>(node_index), static_cast<Real>(node_index));
        leaf.reference = encode_reference(node_index, 1u);

        auto &internal = nodes[node_index].child(1);
        set_bounds(internal, static_cast<Real>(node_index + 1u), Real(3));
        internal.reference = encode_reference(node_index + 1u, 0u);
    }
    for (int child_slot = 0; child_slot < 2; ++child_slot) {
        std::size_t const primitive_index = static_cast<std::size_t>(child_slot + 2);
        auto &leaf = nodes[2].child(child_slot);
        set_bounds(leaf, static_cast<Real>(primitive_index), static_cast<Real>(primitive_index));
        leaf.reference = encode_reference(primitive_index, 1u);
    }

    std::array<Index, 4> const primitive_indices{0u, 1u, 2u, 3u};
    std::array<gwn::gwn_bvh_triangle<Real>, 4> triangles{};
    for (std::size_t triangle_index = 0; triangle_index < triangles.size(); ++triangle_index) {
        auto &triangle = triangles[triangle_index];
        triangle.v0_z = static_cast<Real>(triangle_index);
        triangle.e1_x = Real(1);
        triangle.e2_y = Real(1);
    }

    gwn::gwn_bvh4_object<Real, Index> bvh;
    auto &accessor = bvh.accessor();
    gwn::detail::gwn_allocate_span(accessor.nodes, nodes.size(), cudaStreamLegacy);
    gwn::detail::gwn_allocate_span(
        accessor.primitive_indices, primitive_indices.size(), cudaStreamLegacy
    );
    gwn::detail::gwn_allocate_span(accessor.triangles, triangles.size(), cudaStreamLegacy);
    gwn::detail::gwn_copy_h2d(accessor.nodes, cuda::std::span<Node const>(nodes), cudaStreamLegacy);
    gwn::detail::gwn_copy_h2d(
        accessor.primitive_indices, cuda::std::span<Index const>(primitive_indices),
        cudaStreamLegacy
    );
    gwn::detail::gwn_copy_h2d(
        accessor.triangles, cuda::std::span<gwn::gwn_bvh_triangle<Real> const>(triangles),
        cudaStreamLegacy
    );
    bvh.set_stream(cudaStreamLegacy);
    accessor.root.bounds = {
        .min_x = Real(0),
        .min_y = Real(0),
        .min_z = Real(0),
        .max_x = Real(1),
        .max_y = Real(1),
        .max_z = Real(3),
    };
    accessor.root.reference = encode_reference(0u, 0u);
    accessor.internal_stack_bound = 1u;
    accessor.packed_stack_bound = 3u;
    accessor.revision = 1u;
    ASSERT_TRUE(accessor.is_valid());

    BatchResults<Index> result;
    ASSERT_TRUE((run_batch_queries<Index, 1>(
        bvh, {Real(0.25)}, {Real(0.25)}, {Real(4)}, {Real(0)}, {Real(0)}, {Real(-1)}, {Real(0.25)},
        {Real(0.25)}, {Real(2.5)}, result
    )));
    ASSERT_EQ(result.hits.size(), 1u);
    EXPECT_TRUE(result.hits[0].hit());
    EXPECT_EQ(result.hits[0].primitive_id, Index(3));
    EXPECT_FLOAT_EQ(result.hits[0].t, Real(1));
    EXPECT_FLOAT_EQ(result.distance[0], Real(0.5));
}

TEST_F(GwnQueryDistanceRayTest, canonical_device_queries_report_stack_overflow) {
    using Index = std::uint32_t;
    HostMesh<Index> mesh = make_layered_triangles<Index>(32);
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_hploc);

    auto const result = run_point_queries<Index, 1>(
        bvh, Real(0.25), Real(0.25), Real(40), Real(0), Real(0), Real(-1), Real(0), Real(100),
        Real(0.25), Real(0.25), Real(15.5), std::numeric_limits<Real>::infinity()
    );
    EXPECT_EQ(result.hit.status, gwn::gwn_ray_first_hit_status::k_overflow);
    EXPECT_EQ(result.overflow_flags[0], 1);
    EXPECT_TRUE(std::isnan(result.distance));
    EXPECT_EQ(result.overflow_flags[1], 1);
}

TEST_F(GwnQueryDistanceRayTest, canonical_spatial_queries_support_uint64_indices) {
    using Index = std::uint64_t;
    HostMesh<Index> mesh = make_layered_triangles<Index>(1);
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    upload_and_build(mesh, geometry, bvh, gwn::gwn_bvh_build_method::k_lbvh);

    auto const result = run_point_queries(
        bvh, Real(0.25), Real(0.25), Real(1), Real(0), Real(0), Real(-1), Real(0), Real(2), Real(2),
        Real(0), Real(0), std::numeric_limits<Real>::infinity()
    );
    EXPECT_TRUE(result.hit.hit());
    EXPECT_EQ(result.hit.primitive_id, Index(0));
    EXPECT_FLOAT_EQ(result.distance, Real(1));

    HostMesh<Index> internal_mesh = make_layered_triangles<Index>(12);
    gwn::gwn_geometry_object<Real, Index> internal_geometry;
    gwn::gwn_bvh4_object<Real, Index> internal_bvh;
    upload_and_build(
        internal_mesh, internal_geometry, internal_bvh, gwn::gwn_bvh_build_method::k_hploc
    );
    BatchResults<Index> batch;
    ASSERT_TRUE(run_batch_queries(
        internal_bvh, {Real(0.25)}, {Real(0.25)}, {Real(20)}, {Real(0)}, {Real(0)}, {Real(-1)},
        {Real(0.25)}, {Real(0.25)}, {Real(10.25)}, batch
    ));
    ASSERT_EQ(batch.hits.size(), 1u);
    EXPECT_FLOAT_EQ(batch.hits[0].t, Real(9));
    EXPECT_EQ(batch.hits[0].primitive_id, Index(11));
}

TEST(gwn_vec3, component_access_via_subscript_operator) {
    gwn::gwn_vec3<Real> const value(Real(3), Real(4), Real(5));
    EXPECT_EQ(value[0], Real(3));
    EXPECT_EQ(value[1], Real(4));
    EXPECT_EQ(value[2], Real(5));
}

} // namespace
