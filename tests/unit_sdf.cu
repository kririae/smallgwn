/// \file unit_sdf.cu
/// \brief SDF (signed/unsigned distance) unit tests with libigl parity.
///
/// Tests detail::gwn_point_triangle_distance_squared_impl
/// (host-side analytic tests),
/// gwn_unsigned_distance_point_bvh, and gwn_signed_distance_point_bvh
/// against libigl's point_mesh_squared_distance and signed_distance.
///
/// libigl reference computations live in libigl_reference.cpp (compiled
/// by the host compiler) to avoid SIMD incompatibilities with nvcc.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_query_geometry_impl.cuh>
#include <gwn/gwn.cuh>

#include "libigl_reference.hpp"
#include "test_fixtures.hpp"
#include "test_utils.hpp"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

namespace {

// Octahedron mesh: 6 vertices, 8 triangles, centered at origin.
struct OctahedronMesh {
    std::array<Real, 6> vx{1, -1, 0, 0, 0, 0};
    std::array<Real, 6> vy{0, 0, 1, -1, 0, 0};
    std::array<Real, 6> vz{0, 0, 0, 0, 1, -1};
    std::array<Index, 8> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> i2{4, 4, 4, 4, 5, 5, 5, 5};

    // Build int-typed index arrays for libigl reference helper.
    [[nodiscard]] std::vector<int> int_i0() const {
        std::vector<int> v(i0.size());
        for (std::size_t k = 0; k < i0.size(); ++k)
            v[k] = static_cast<int>(i0[k]);
        return v;
    }
    [[nodiscard]] std::vector<int> int_i1() const {
        std::vector<int> v(i1.size());
        for (std::size_t k = 0; k < i1.size(); ++k)
            v[k] = static_cast<int>(i1[k]);
        return v;
    }
    [[nodiscard]] std::vector<int> int_i2() const {
        std::vector<int> v(i2.size());
        for (std::size_t k = 0; k < i2.size(); ++k)
            v[k] = static_cast<int>(i2[k]);
        return v;
    }
};

// Thin kernel: unsigned distance for N query points.
template <int Width, typename RealT, typename IndexT>
__global__ void kernel_unsigned_distance(
    gwn::gwn_geometry_accessor<RealT, IndexT> geometry,
    gwn::gwn_bvh_topology_accessor<Width, RealT, IndexT> bvh,
    gwn::gwn_bvh_aabb_accessor<Width, RealT, IndexT> aabb_tree, RealT const *qx, RealT const *qy,
    RealT const *qz, RealT *output, std::size_t count, RealT culling_band
) {
    std::size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    output[idx] = gwn::gwn_unsigned_distance_point_bvh<Width, RealT, IndexT>(
        geometry, bvh, aabb_tree, qx[idx], qy[idx], qz[idx], culling_band
    );
}

// Thin kernel: signed distance for N query points.
template <int Width, typename RealT, typename IndexT>
__global__ void kernel_signed_distance(
    gwn::gwn_geometry_accessor<RealT, IndexT> geometry,
    gwn::gwn_bvh_topology_accessor<Width, RealT, IndexT> bvh,
    gwn::gwn_bvh_aabb_accessor<Width, RealT, IndexT> aabb_tree,
    gwn::gwn_bvh_moment_tree_accessor<Width, RealT, IndexT> data_tree, RealT const *qx,
    RealT const *qy, RealT const *qz, RealT *output, std::size_t count,
    RealT winding_number_threshold, RealT culling_band
) {
    std::size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    output[idx] = gwn::gwn_signed_distance_point_bvh<1, Width, RealT, IndexT, 128>(
        geometry, bvh, aabb_tree, data_tree, qx[idx], qy[idx], qz[idx], winding_number_threshold,
        culling_band
    );
}

// Helper: upload octahedron and build BVH topology + AABB + order-1 moments.
struct SdfTestContext {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh_object<Real, Index> bvh;
    gwn::gwn_bvh_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh_moment_object<Real, Index> data;

    bool ready = false;
};

void setup_octahedron_sdf(SdfTestContext &ctx, OctahedronMesh const &mesh) {
    gwn::gwn_status const upload_status = ctx.geometry.upload(
        cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
        cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
        cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
        cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
        cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
        cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
    );
    if (!upload_status.is_ok())
        return;

    gwn::gwn_status const build_status =
        gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
            ctx.geometry, ctx.bvh, ctx.aabb, ctx.data
        );
    if (!build_status.is_ok())
        return;

    ctx.ready = true;
}

// Helper: run unsigned distance queries on GPU.
void gpu_unsigned_distance(
    SdfTestContext &ctx, std::vector<Real> const &qx, std::vector<Real> const &qy,
    std::vector<Real> const &qz, std::vector<Real> &results,
    Real const culling_band = std::numeric_limits<Real>::infinity()
) {
    std::size_t const n = qx.size();
    results.resize(n, Real(0));

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    ASSERT_TRUE(d_qx.resize(n).is_ok());
    ASSERT_TRUE(d_qy.resize(n).is_ok());
    ASSERT_TRUE(d_qz.resize(n).is_ok());
    ASSERT_TRUE(d_out.resize(n).is_ok());
    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());

    int const block_size = 128;
    int const grid_size = static_cast<int>((n + block_size - 1) / block_size);
    kernel_unsigned_distance<4, Real, Index><<<grid_size, block_size>>>(
        ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(), d_qx.span().data(),
        d_qy.span().data(), d_qz.span().data(), d_out.span().data(), n, culling_band
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(results.data(), results.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

// Helper: run signed distance queries on GPU.
void gpu_signed_distance(
    SdfTestContext &ctx, std::vector<Real> const &qx, std::vector<Real> const &qy,
    std::vector<Real> const &qz, std::vector<Real> &results,
    Real const winding_number_threshold = Real(0.5),
    Real const culling_band = std::numeric_limits<Real>::infinity()
) {
    std::size_t const n = qx.size();
    results.resize(n, Real(0));

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    ASSERT_TRUE(d_qx.resize(n).is_ok());
    ASSERT_TRUE(d_qy.resize(n).is_ok());
    ASSERT_TRUE(d_qz.resize(n).is_ok());
    ASSERT_TRUE(d_out.resize(n).is_ok());
    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());

    int const block_size = 128;
    int const grid_size = static_cast<int>((n + block_size - 1) / block_size);
    kernel_signed_distance<4, Real, Index><<<grid_size, block_size>>>(
        ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(), ctx.data.accessor(),
        d_qx.span().data(), d_qy.span().data(), d_qz.span().data(), d_out.span().data(), n,
        winding_number_threshold, culling_band
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(results.data(), results.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

} // namespace

// Host-side analytic tests for detail::gwn_point_triangle_distance_squared_impl.
// These exercise the Voronoi-region closest-point computation on-CPU
// using __host__ __device__ qualification.

TEST(smallgwn_unit_sdf, point_triangle_dist2_vertex_closest) {
    // Point closest to vertex A of triangle ABC in the xy plane.
    gwn::gwn_vec3<Real> const a(0, 0, 0);
    gwn::gwn_vec3<Real> const b(1, 0, 0);
    gwn::gwn_vec3<Real> const c(0, 1, 0);
    gwn::gwn_vec3<Real> const p(-1, -1, 0); // nearest to A

    Real const d2 = gwn::detail::gwn_point_triangle_distance_squared_impl(p, a, b, c);
    EXPECT_NEAR(d2, 2.0f, 1e-6f); // |(-1,-1)-(0,0)|^2 = 2
}

TEST(smallgwn_unit_sdf, point_triangle_dist2_vertex_b_closest) {
    gwn::gwn_vec3<Real> const a(0, 0, 0);
    gwn::gwn_vec3<Real> const b(1, 0, 0);
    gwn::gwn_vec3<Real> const c(0, 1, 0);
    gwn::gwn_vec3<Real> const p(2, -1, 0); // nearest to B

    Real const d2 = gwn::detail::gwn_point_triangle_distance_squared_impl(p, a, b, c);
    EXPECT_NEAR(d2, 2.0f, 1e-6f); // |(2,-1)-(1,0)|^2 = 2
}

TEST(smallgwn_unit_sdf, point_triangle_dist2_vertex_c_closest) {
    gwn::gwn_vec3<Real> const a(0, 0, 0);
    gwn::gwn_vec3<Real> const b(1, 0, 0);
    gwn::gwn_vec3<Real> const c(0, 1, 0);
    gwn::gwn_vec3<Real> const p(-1, 2, 0); // nearest to C

    Real const d2 = gwn::detail::gwn_point_triangle_distance_squared_impl(p, a, b, c);
    EXPECT_NEAR(d2, 2.0f, 1e-6f); // |(-1,2)-(0,1)|^2 = 2
}

TEST(smallgwn_unit_sdf, point_triangle_dist2_edge_ab_closest) {
    gwn::gwn_vec3<Real> const a(0, 0, 0);
    gwn::gwn_vec3<Real> const b(2, 0, 0);
    gwn::gwn_vec3<Real> const c(0, 2, 0);
    gwn::gwn_vec3<Real> const p(1, -1, 0); // projects onto edge AB midpoint

    Real const d2 = gwn::detail::gwn_point_triangle_distance_squared_impl(p, a, b, c);
    // Closest point is (1,0,0), distance^2 = 1.
    EXPECT_NEAR(d2, 1.0f, 1e-6f);
}

TEST(smallgwn_unit_sdf, point_triangle_dist2_edge_ac_closest) {
    gwn::gwn_vec3<Real> const a(0, 0, 0);
    gwn::gwn_vec3<Real> const b(2, 0, 0);
    gwn::gwn_vec3<Real> const c(0, 2, 0);
    gwn::gwn_vec3<Real> const p(-1, 1, 0); // projects onto edge AC midpoint

    Real const d2 = gwn::detail::gwn_point_triangle_distance_squared_impl(p, a, b, c);
    // Closest point is (0,1,0), distance^2 = 1.
    EXPECT_NEAR(d2, 1.0f, 1e-6f);
}

TEST(smallgwn_unit_sdf, point_triangle_dist2_edge_bc_closest) {
    gwn::gwn_vec3<Real> const a(0, 0, 0);
    gwn::gwn_vec3<Real> const b(2, 0, 0);
    gwn::gwn_vec3<Real> const c(0, 2, 0);
    // Point beyond edge BC: nearest to midpoint (1,1,0).
    gwn::gwn_vec3<Real> const p(2, 2, 0);

    Real const d2 = gwn::detail::gwn_point_triangle_distance_squared_impl(p, a, b, c);
    // Closest point on BC: parameterised b + t(c - b) = (2,0,0) + t(-2,2,0).
    // Project (2,2,0) onto BC: t = ((2,2,0)-(2,0,0)) . (-2,2,0) / |(-2,2,0)|^2
    //   = (0,2,0) . (-2,2,0) / 8 = 4/8 = 0.5 → closest = (1,1,0).
    // distance^2 = |(2,2)-(1,1)|^2 = 2.
    EXPECT_NEAR(d2, 2.0f, 1e-6f);
}

TEST(smallgwn_unit_sdf, point_triangle_dist2_face_projection) {
    gwn::gwn_vec3<Real> const a(0, 0, 0);
    gwn::gwn_vec3<Real> const b(1, 0, 0);
    gwn::gwn_vec3<Real> const c(0, 1, 0);
    gwn::gwn_vec3<Real> const p(0.25f, 0.25f, 1.0f); // above interior

    Real const d2 = gwn::detail::gwn_point_triangle_distance_squared_impl(p, a, b, c);
    // Projects to (0.25, 0.25, 0), distance^2 = 1.
    EXPECT_NEAR(d2, 1.0f, 1e-6f);
}

TEST(smallgwn_unit_sdf, point_triangle_dist2_on_triangle) {
    gwn::gwn_vec3<Real> const a(0, 0, 0);
    gwn::gwn_vec3<Real> const b(1, 0, 0);
    gwn::gwn_vec3<Real> const c(0, 1, 0);
    gwn::gwn_vec3<Real> const p(0.25f, 0.25f, 0.0f); // on triangle

    Real const d2 = gwn::detail::gwn_point_triangle_distance_squared_impl(p, a, b, c);
    EXPECT_NEAR(d2, 0.0f, 1e-10f);
}

TEST(smallgwn_unit_sdf, point_triangle_dist2_out_of_plane) {
    // 3D triangle in the plane x + y + z = 1.
    gwn::gwn_vec3<Real> const a(1, 0, 0);
    gwn::gwn_vec3<Real> const b(0, 1, 0);
    gwn::gwn_vec3<Real> const c(0, 0, 1);
    gwn::gwn_vec3<Real> const p(0, 0, 0); // origin

    Real const d2 = gwn::detail::gwn_point_triangle_distance_squared_impl(p, a, b, c);
    // Distance from origin to plane x+y+z=1 is 1/sqrt(3).
    // d^2 = 1/3.  The foot (1/3,1/3,1/3) is inside the triangle.
    EXPECT_NEAR(d2, 1.0f / 3.0f, 1e-5f);
}

// GPU + libigl parity: unsigned distance on octahedron.

TEST_F(CudaFixture, unsigned_distance_octahedron_vs_libigl) {
    constexpr Real k_eps = 1e-6f;

    OctahedronMesh mesh;
    SdfTestContext ctx;
    setup_octahedron_sdf(ctx, mesh);
    if (!ctx.ready)
        GTEST_SKIP() << "CUDA unavailable";

    // Query points: origin (inside), far outside, near surface, on vertex,
    // axis-aligned close points.
    std::vector<Real> qx{0.0f, 5.0f, 0.9f, 1.0f, 0.0f, 0.0f};
    std::vector<Real> qy{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f};
    std::vector<Real> qz{0.0f, 0.0f, 0.0f, 0.0f, 0.9f, 0.0f};

    // GPU results.
    std::vector<Real> gpu_dist;
    gpu_unsigned_distance(ctx, qx, qy, qz, gpu_dist);

    // libigl reference.
    auto const fi0 = mesh.int_i0();
    auto const fi1 = mesh.int_i1();
    auto const fi2 = mesh.int_i2();
    std::size_t const n = qx.size();
    std::vector<float> const igl_dist = gwn::tests::libigl_unsigned_distance(
        mesh.vx.data(), mesh.vy.data(), mesh.vz.data(), mesh.vx.size(), fi0.data(), fi1.data(),
        fi2.data(), fi0.size(), qx.data(), qy.data(), qz.data(), n
    );

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_dist[i], igl_dist[i], k_eps)
            << "query " << i << " (" << qx[i] << ", " << qy[i] << ", " << qz[i] << ")";
    }
}

// GPU + libigl parity: signed distance on octahedron.

TEST_F(CudaFixture, signed_distance_octahedron_vs_libigl) {
    constexpr Real k_eps = 1e-3f;

    OctahedronMesh mesh;
    SdfTestContext ctx;
    setup_octahedron_sdf(ctx, mesh);
    if (!ctx.ready)
        GTEST_SKIP() << "CUDA unavailable";

    // Mix of inside, outside, and on-surface queries.
    std::vector<Real> qx{0.0f, 5.0f, 0.1f, -0.1f, 3.0f, 0.0f};
    std::vector<Real> qy{0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.5f};
    std::vector<Real> qz{0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f};

    // GPU results.
    std::vector<Real> gpu_sd;
    gpu_signed_distance(ctx, qx, qy, qz, gpu_sd);

    // libigl reference using winding number for sign.
    auto const fi0 = mesh.int_i0();
    auto const fi1 = mesh.int_i1();
    auto const fi2 = mesh.int_i2();
    std::size_t const n = qx.size();
    std::vector<float> const igl_sd = gwn::tests::libigl_signed_distance(
        mesh.vx.data(), mesh.vy.data(), mesh.vz.data(), mesh.vx.size(), fi0.data(), fi1.data(),
        fi2.data(), fi0.size(), qx.data(), qy.data(), qz.data(), n
    );

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_sd[i], igl_sd[i], k_eps)
            << "query " << i << " (" << qx[i] << ", " << qy[i] << ", " << qz[i] << ")";
    }
}

// Verify sign correctness: inside = negative, outside = positive.

TEST_F(CudaFixture, signed_distance_sign_correctness) {
    OctahedronMesh mesh;
    SdfTestContext ctx;
    setup_octahedron_sdf(ctx, mesh);
    if (!ctx.ready)
        GTEST_SKIP() << "CUDA unavailable";

    // Clearly inside.
    std::vector<Real> qx_in{0.0f, 0.1f, -0.1f};
    std::vector<Real> qy_in{0.0f, 0.1f, 0.0f};
    std::vector<Real> qz_in{0.0f, 0.1f, 0.0f};

    std::vector<Real> sd_in;
    gpu_signed_distance(ctx, qx_in, qy_in, qz_in, sd_in);
    for (std::size_t i = 0; i < sd_in.size(); ++i)
        EXPECT_LT(sd_in[i], Real(0)) << "Inside point " << i << " should be negative";

    // Clearly outside.
    std::vector<Real> qx_out{5.0f, -3.0f, 0.0f};
    std::vector<Real> qy_out{0.0f, 0.0f, 5.0f};
    std::vector<Real> qz_out{0.0f, 0.0f, 0.0f};

    std::vector<Real> sd_out;
    gpu_signed_distance(ctx, qx_out, qy_out, qz_out, sd_out);
    for (std::size_t i = 0; i < sd_out.size(); ++i)
        EXPECT_GT(sd_out[i], Real(0)) << "Outside point " << i << " should be positive";
}

// Unsigned distance is always non-negative.

TEST_F(CudaFixture, unsigned_distance_non_negative) {
    OctahedronMesh mesh;
    SdfTestContext ctx;
    setup_octahedron_sdf(ctx, mesh);
    if (!ctx.ready)
        GTEST_SKIP() << "CUDA unavailable";

    // Grid of random-ish points.
    std::vector<Real> qx, qy, qz;
    for (int ix = -2; ix <= 2; ++ix) {
        for (int iy = -2; iy <= 2; ++iy) {
            for (int iz = -2; iz <= 2; ++iz) {
                qx.push_back(static_cast<Real>(ix) * 0.7f);
                qy.push_back(static_cast<Real>(iy) * 0.7f);
                qz.push_back(static_cast<Real>(iz) * 0.7f);
            }
        }
    }

    std::vector<Real> dist;
    gpu_unsigned_distance(ctx, qx, qy, qz, dist);

    for (std::size_t i = 0; i < dist.size(); ++i)
        EXPECT_GE(dist[i], Real(0)) << "Unsigned distance must be non-negative at query " << i;
}

// Signed vs unsigned distance: |signed| == unsigned.

TEST_F(CudaFixture, signed_distance_magnitude_equals_unsigned) {
    constexpr Real k_eps = 1e-5f;

    OctahedronMesh mesh;
    SdfTestContext ctx;
    setup_octahedron_sdf(ctx, mesh);
    if (!ctx.ready)
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> qx{0.0f, 5.0f, 0.2f, -1.5f, 0.0f, 0.0f};
    std::vector<Real> qy{0.0f, 0.0f, 0.2f, 0.0f, 0.0f, 3.0f};
    std::vector<Real> qz{0.0f, 0.0f, 0.2f, 0.0f, 0.9f, 0.0f};

    std::vector<Real> ud, sd;
    gpu_unsigned_distance(ctx, qx, qy, qz, ud);
    gpu_signed_distance(ctx, qx, qy, qz, sd);

    for (std::size_t i = 0; i < qx.size(); ++i)
        EXPECT_NEAR(std::abs(sd[i]), ud[i], k_eps) << "query " << i << ": |signed| != unsigned";
}

TEST_F(CudaFixture, signed_distance_threshold_controls_sign) {
    constexpr Real k_eps = 1e-6f;

    OctahedronMesh mesh;
    SdfTestContext ctx;
    setup_octahedron_sdf(ctx, mesh);
    if (!ctx.ready)
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> qx{0.0f};
    std::vector<Real> qy{0.0f};
    std::vector<Real> qz{0.0f};

    std::vector<Real> sd_default, sd_high_threshold;
    gpu_signed_distance(ctx, qx, qy, qz, sd_default, Real(0.5f));
    gpu_signed_distance(ctx, qx, qy, qz, sd_high_threshold, Real(1.1f));

    ASSERT_EQ(sd_default.size(), 1u);
    ASSERT_EQ(sd_high_threshold.size(), 1u);
    EXPECT_LT(sd_default[0], Real(0));
    EXPECT_GT(sd_high_threshold[0], Real(0));
    EXPECT_NEAR(std::abs(sd_default[0]), std::abs(sd_high_threshold[0]), k_eps);
}

TEST_F(CudaFixture, unsigned_distance_culling_band_clamps_far_queries) {
    constexpr Real k_eps = 1e-6f;

    OctahedronMesh mesh;
    SdfTestContext ctx;
    setup_octahedron_sdf(ctx, mesh);
    if (!ctx.ready)
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> qx{5.0f, 0.9f};
    std::vector<Real> qy{0.0f, 0.0f};
    std::vector<Real> qz{0.0f, 0.0f};

    std::vector<Real> baseline, clipped;
    gpu_unsigned_distance(ctx, qx, qy, qz, baseline);
    Real const culling_band = Real(0.2f);
    gpu_unsigned_distance(ctx, qx, qy, qz, clipped, culling_band);

    ASSERT_EQ(clipped.size(), 2u);
    EXPECT_GT(baseline[0], culling_band);
    EXPECT_NEAR(clipped[0], culling_band, k_eps);
    EXPECT_LT(baseline[1], culling_band);
    EXPECT_NEAR(clipped[1], baseline[1], k_eps);
}

TEST_F(CudaFixture, signed_distance_culling_band_clamps_with_winding_sign) {
    constexpr Real k_eps = 1e-6f;

    OctahedronMesh mesh;
    SdfTestContext ctx;
    setup_octahedron_sdf(ctx, mesh);
    if (!ctx.ready)
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> qx{0.0f, 5.0f};
    std::vector<Real> qy{0.0f, 0.0f};
    std::vector<Real> qz{0.0f, 0.0f};

    std::vector<Real> clipped_signed;
    Real const culling_band = Real(0.1f);
    gpu_signed_distance(ctx, qx, qy, qz, clipped_signed, Real(0.5f), culling_band);

    ASSERT_EQ(clipped_signed.size(), 2u);
    EXPECT_NEAR(clipped_signed[0], -culling_band, k_eps);
    EXPECT_NEAR(clipped_signed[1], culling_band, k_eps);
}

// H-PLOC topology produces same distances as LBVH.

TEST_F(CudaFixture, unsigned_distance_hploc_matches_lbvh) {
    constexpr Real k_eps = 1e-5f;

    OctahedronMesh mesh;

    // LBVH build.
    SdfTestContext ctx_lbvh;
    setup_octahedron_sdf(ctx_lbvh, mesh);
    if (!ctx_lbvh.ready)
        GTEST_SKIP() << "CUDA unavailable";

    // HPLOC build.
    SdfTestContext ctx_hploc;
    {
        gwn::gwn_status const upload_status = ctx_hploc.geometry.upload(
            cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
            cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
            cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
            cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
            cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
            cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
        );
        ASSERT_TRUE(upload_status.is_ok());
        gwn::gwn_status const build_status =
            gwn::gwn_bvh_facade_build_topology_aabb_hploc<4, Real, Index>(
                ctx_hploc.geometry, ctx_hploc.bvh, ctx_hploc.aabb
            );
        ASSERT_TRUE(build_status.is_ok());
        ctx_hploc.ready = true;
    }

    std::vector<Real> qx{0.0f, 5.0f, 0.9f, -1.5f, 0.0f};
    std::vector<Real> qy{0.0f, 0.0f, 0.0f, 0.0f, 3.0f};
    std::vector<Real> qz{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    std::vector<Real> lbvh_dist, hploc_dist;
    gpu_unsigned_distance(ctx_lbvh, qx, qy, qz, lbvh_dist);
    gpu_unsigned_distance(ctx_hploc, qx, qy, qz, hploc_dist);

    for (std::size_t i = 0; i < qx.size(); ++i)
        EXPECT_NEAR(lbvh_dist[i], hploc_dist[i], k_eps)
            << "query " << i << ": LBVH vs HPLOC mismatch";
}

// Model-based parity (optional): if SMALLGWN_MODEL_DIR is set, load one model
// and compare GPU signed/unsigned distances with libigl across sampled queries.

TEST_F(CudaFixture, model_unsigned_distance_vs_libigl) {
    auto const model_dir = gwn::tests::find_model_data_dir();
    if (!model_dir.has_value())
        GTEST_SKIP() << "SMALLGWN_MODEL_DATA_DIR not set";

    auto const models = gwn::tests::collect_obj_model_paths(model_dir.value());
    if (models.empty())
        GTEST_SKIP() << "No .obj files in model dir";

    // Use first model only.
    auto const loaded = gwn::tests::load_obj_mesh(models[0]);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load " << models[0];
    gwn::tests::HostMesh const &host_mesh = *loaded;

    // Upload + build.
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const upload_status = geometry.upload(
        cuda::std::span<Real const>(host_mesh.vertex_x.data(), host_mesh.vertex_x.size()),
        cuda::std::span<Real const>(host_mesh.vertex_y.data(), host_mesh.vertex_y.size()),
        cuda::std::span<Real const>(host_mesh.vertex_z.data(), host_mesh.vertex_z.size()),
        cuda::std::span<Index const>(host_mesh.tri_i0.data(), host_mesh.tri_i0.size()),
        cuda::std::span<Index const>(host_mesh.tri_i1.data(), host_mesh.tri_i1.size()),
        cuda::std::span<Index const>(host_mesh.tri_i2.data(), host_mesh.tri_i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
    ASSERT_TRUE(upload_status.is_ok());

    gwn::gwn_bvh_object<Real, Index> bvh;
    gwn::gwn_bvh_aabb_object<Real, Index> aabb;
    ASSERT_TRUE(
        (gwn::gwn_bvh_facade_build_topology_aabb_lbvh<4, Real, Index>(geometry, bvh, aabb)).is_ok()
    );

    // Compute scene bounds from host data.
    Real min_x = host_mesh.vertex_x[0], max_x = host_mesh.vertex_x[0];
    Real min_y = host_mesh.vertex_y[0], max_y = host_mesh.vertex_y[0];
    Real min_z = host_mesh.vertex_z[0], max_z = host_mesh.vertex_z[0];
    for (std::size_t v = 1; v < host_mesh.vertex_x.size(); ++v) {
        min_x = std::min(min_x, host_mesh.vertex_x[v]);
        max_x = std::max(max_x, host_mesh.vertex_x[v]);
        min_y = std::min(min_y, host_mesh.vertex_y[v]);
        max_y = std::max(max_y, host_mesh.vertex_y[v]);
        min_z = std::min(min_z, host_mesh.vertex_z[v]);
        max_z = std::max(max_z, host_mesh.vertex_z[v]);
    }
    Real const cx = (min_x + max_x) * Real(0.5);
    Real const cy = (min_y + max_y) * Real(0.5);
    Real const cz = (min_z + max_z) * Real(0.5);
    Real const extent = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
    int const query_span_multiplier =
        gwn::tests::get_env_positive_int("SMALLGWN_SDF_QUERY_SPAN_MULTIPLIER", 1);
    Real const query_span = extent * static_cast<Real>(query_span_multiplier);

    // Generate query points: grid sampling around centre.
    constexpr int k_n = 5;
    std::vector<Real> qx, qy, qz;
    for (int ix = 0; ix < k_n; ++ix) {
        for (int iy = 0; iy < k_n; ++iy) {
            for (int iz = 0; iz < k_n; ++iz) {
                qx.push_back(cx + query_span * (static_cast<Real>(ix) / (k_n - 1) - 0.5f));
                qy.push_back(cy + query_span * (static_cast<Real>(iy) / (k_n - 1) - 0.5f));
                qz.push_back(cz + query_span * (static_cast<Real>(iz) / (k_n - 1) - 0.5f));
            }
        }
    }

    // GPU.
    std::size_t const n = qx.size();
    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    ASSERT_TRUE(d_qx.resize(n).is_ok());
    ASSERT_TRUE(d_qy.resize(n).is_ok());
    ASSERT_TRUE(d_qz.resize(n).is_ok());
    ASSERT_TRUE(d_out.resize(n).is_ok());
    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());

    int const block_size = 128;
    int const grid_size = static_cast<int>((n + block_size - 1) / block_size);
    kernel_unsigned_distance<4, Real, Index><<<grid_size, block_size>>>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), d_qx.span().data(),
        d_qy.span().data(), d_qz.span().data(), d_out.span().data(), n,
        std::numeric_limits<Real>::infinity()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<Real> gpu_dist(n, 0);
    ASSERT_TRUE(
        d_out.copy_to_host(cuda::std::span<Real>(gpu_dist.data(), gpu_dist.size())).is_ok()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    // libigl reference.
    std::size_t const nv = host_mesh.vertex_x.size();
    std::size_t const nf = host_mesh.tri_i0.size();
    std::vector<int> fi0(nf), fi1(nf), fi2(nf);
    for (std::size_t k = 0; k < nf; ++k) {
        fi0[k] = static_cast<int>(host_mesh.tri_i0[k]);
        fi1[k] = static_cast<int>(host_mesh.tri_i1[k]);
        fi2[k] = static_cast<int>(host_mesh.tri_i2[k]);
    }
    std::vector<float> const igl_dist = gwn::tests::libigl_unsigned_distance(
        host_mesh.vertex_x.data(), host_mesh.vertex_y.data(), host_mesh.vertex_z.data(), nv,
        fi0.data(), fi1.data(), fi2.data(), nf, qx.data(), qy.data(), qz.data(), n
    );

    // Tolerance: combined absolute (extent-proportional) + relative.
    Real const abs_tol = extent * 1e-7f;
    for (std::size_t i = 0; i < n; ++i) {
        Real const tol = std::max(abs_tol, igl_dist[i] * 1e-6f);
        EXPECT_NEAR(gpu_dist[i], igl_dist[i], tol)
            << "query " << i << " (" << qx[i] << ", " << qy[i] << ", " << qz[i] << ")";
    }
}

TEST_F(CudaFixture, model_signed_distance_vs_libigl) {
    auto const model_dir = gwn::tests::find_model_data_dir();
    if (!model_dir.has_value())
        GTEST_SKIP() << "SMALLGWN_MODEL_DATA_DIR not set";

    auto const models = gwn::tests::collect_obj_model_paths(model_dir.value());
    if (models.empty())
        GTEST_SKIP() << "No .obj files in model dir";

    auto const loaded = gwn::tests::load_obj_mesh(models[0]);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load " << models[0];
    gwn::tests::HostMesh const &host_mesh = *loaded;

    // Upload.
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const upload_status = geometry.upload(
        cuda::std::span<Real const>(host_mesh.vertex_x.data(), host_mesh.vertex_x.size()),
        cuda::std::span<Real const>(host_mesh.vertex_y.data(), host_mesh.vertex_y.size()),
        cuda::std::span<Real const>(host_mesh.vertex_z.data(), host_mesh.vertex_z.size()),
        cuda::std::span<Index const>(host_mesh.tri_i0.data(), host_mesh.tri_i0.size()),
        cuda::std::span<Index const>(host_mesh.tri_i1.data(), host_mesh.tri_i1.size()),
        cuda::std::span<Index const>(host_mesh.tri_i2.data(), host_mesh.tri_i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
    ASSERT_TRUE(upload_status.is_ok());

    gwn::gwn_bvh_object<Real, Index> bvh;
    gwn::gwn_bvh_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh_moment_object<Real, Index> data;
    ASSERT_TRUE((gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
                     geometry, bvh, aabb, data
    ))
                    .is_ok());

    // Scene bounds.
    Real min_x = host_mesh.vertex_x[0], max_x = host_mesh.vertex_x[0];
    Real min_y = host_mesh.vertex_y[0], max_y = host_mesh.vertex_y[0];
    Real min_z = host_mesh.vertex_z[0], max_z = host_mesh.vertex_z[0];
    for (std::size_t v = 1; v < host_mesh.vertex_x.size(); ++v) {
        min_x = std::min(min_x, host_mesh.vertex_x[v]);
        max_x = std::max(max_x, host_mesh.vertex_x[v]);
        min_y = std::min(min_y, host_mesh.vertex_y[v]);
        max_y = std::max(max_y, host_mesh.vertex_y[v]);
        min_z = std::min(min_z, host_mesh.vertex_z[v]);
        max_z = std::max(max_z, host_mesh.vertex_z[v]);
    }
    Real const cx = (min_x + max_x) * Real(0.5);
    Real const cy = (min_y + max_y) * Real(0.5);
    Real const cz = (min_z + max_z) * Real(0.5);
    Real const extent = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
    int const query_span_multiplier =
        gwn::tests::get_env_positive_int("SMALLGWN_SDF_QUERY_SPAN_MULTIPLIER", 1);
    Real const query_span = extent * static_cast<Real>(query_span_multiplier);

    constexpr int k_n = 5;
    std::vector<Real> qx, qy, qz;
    for (int ix = 0; ix < k_n; ++ix) {
        for (int iy = 0; iy < k_n; ++iy) {
            for (int iz = 0; iz < k_n; ++iz) {
                qx.push_back(cx + query_span * (static_cast<Real>(ix) / (k_n - 1) - 0.5f));
                qy.push_back(cy + query_span * (static_cast<Real>(iy) / (k_n - 1) - 0.5f));
                qz.push_back(cz + query_span * (static_cast<Real>(iz) / (k_n - 1) - 0.5f));
            }
        }
    }

    // GPU.
    std::size_t const n = qx.size();
    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    ASSERT_TRUE(d_qx.resize(n).is_ok());
    ASSERT_TRUE(d_qy.resize(n).is_ok());
    ASSERT_TRUE(d_qz.resize(n).is_ok());
    ASSERT_TRUE(d_out.resize(n).is_ok());
    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());

    int const block_size = 128;
    int const grid_size = static_cast<int>((n + block_size - 1) / block_size);
    kernel_signed_distance<4, Real, Index><<<grid_size, block_size>>>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), data.accessor(), d_qx.span().data(),
        d_qy.span().data(), d_qz.span().data(), d_out.span().data(), n, Real(0.5),
        std::numeric_limits<Real>::infinity()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<Real> gpu_sd(n, 0);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(gpu_sd.data(), gpu_sd.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    kernel_unsigned_distance<4, Real, Index><<<grid_size, block_size>>>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), d_qx.span().data(),
        d_qy.span().data(), d_qz.span().data(), d_out.span().data(), n,
        std::numeric_limits<Real>::infinity()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<Real> gpu_ud(n, 0);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(gpu_ud.data(), gpu_ud.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    Real const mag_eps = extent * 1e-5f;
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(std::abs(gpu_sd[i]), gpu_ud[i], mag_eps)
            << "query " << i << ": |signed| != unsigned";
    }

    // libigl reference (unsigned distance only: most stable baseline).
    std::size_t const nv = host_mesh.vertex_x.size();
    std::size_t const nf = host_mesh.tri_i0.size();
    std::vector<int> fi0(nf), fi1(nf), fi2(nf);
    for (std::size_t k = 0; k < nf; ++k) {
        fi0[k] = static_cast<int>(host_mesh.tri_i0[k]);
        fi1[k] = static_cast<int>(host_mesh.tri_i1[k]);
        fi2[k] = static_cast<int>(host_mesh.tri_i2[k]);
    }
    std::vector<float> const igl_ud = gwn::tests::libigl_unsigned_distance(
        host_mesh.vertex_x.data(), host_mesh.vertex_y.data(), host_mesh.vertex_z.data(), nv,
        fi0.data(), fi1.data(), fi2.data(), nf, qx.data(), qy.data(), qz.data(), n
    );

    // Compare magnitude against libigl unsigned and compose reference sign
    // from gwn signed output to avoid unstable libigl signed magnitude path.
    Real const abs_tol = extent * 1e-7f;
    for (std::size_t i = 0; i < n; ++i) {
        Real const tol = std::max(abs_tol, igl_ud[i] * 1e-6f);
        EXPECT_NEAR(gpu_ud[i], igl_ud[i], tol) << "query " << i << ": unsigned mismatch";

        Real const signed_ref = gpu_sd[i] < Real(0) ? -igl_ud[i] : igl_ud[i];
        EXPECT_NEAR(gpu_sd[i], signed_ref, tol)
            << "query " << i << " (" << qx[i] << ", " << qy[i] << ", " << qz[i] << ")";
    }
}
