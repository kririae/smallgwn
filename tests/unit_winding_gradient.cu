#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "reference_cpu.cuh"
#include "test_fixtures.hpp"
#include "test_utils.hpp"

// Winding-number gradient unit tests, validate BVH-accelerated Taylor
// gradient against CPU finite differences.
//
// Design notes:
//   Each test asserts BEFORE comparing GPU vs reference that the reference
//   gradient magnitude is non-trivial (> k_min_nontrivial_mag).  This
//   prevents false positives where both GPU and CPU happen to return ~0.

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

namespace {

// The gradient is considered non-trivial if its magnitude exceeds this.
constexpr Real k_min_nontrivial_mag = 1e-3f;

// CPU finite-difference gradient helper.
template <class Real_, class Index_>
void reference_winding_gradient_fd(
    std::span<Real_ const> vx, std::span<Real_ const> vy, std::span<Real_ const> vz,
    std::span<Index_ const> i0, std::span<Index_ const> i1, std::span<Index_ const> i2,
    Real_ const qx, Real_ const qy, Real_ const qz, Real_ const h, Real_ &gx, Real_ &gy,
    Real_ &gz
) {
    auto w = [&](Real_ x, Real_ y, Real_ z) {
        return gwn::tests::reference_winding_number_point<Real_, Index_>(
            vx, vy, vz, i0, i1, i2, x, y, z
        );
    };
    gx = (w(qx + h, qy, qz) - w(qx - h, qy, qz)) / (Real_(2) * h);
    gy = (w(qx, qy + h, qz) - w(qx, qy - h, qz)) / (Real_(2) * h);
    gz = (w(qx, qy, qz + h) - w(qx, qy, qz - h)) / (Real_(2) * h);
}

// Octahedron mesh (8 triangles, closed, circumradius = 1).
// NOTE: perfectly symmetric → monopole moment = 0 → far-field Taylor gradient
//       is trivially 0 for symmetric queries.  Do not use for Taylor accuracy
//       testing.  Use for near-surface (leaf/brute-force) testing only.
struct OctahedronMesh {
    static constexpr int k_nv = 6;
    static constexpr int k_nt = 8;
    std::array<Real, k_nv> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<Real, k_nv> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::array<Real, k_nv> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::array<Index, k_nt> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, k_nt> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, k_nt> i2{4, 4, 4, 4, 5, 5, 5, 5};

    std::span<Real const> sx() const { return {vx.data(), vx.size()}; }
    std::span<Real const> sy() const { return {vy.data(), vy.size()}; }
    std::span<Real const> sz() const { return {vz.data(), vz.size()}; }
    std::span<Index const> si0() const { return {i0.data(), i0.size()}; }
    std::span<Index const> si1() const { return {i1.data(), i1.size()}; }
    std::span<Index const> si2() const { return {i2.data(), i2.size()}; }
};

// Half-octahedron mesh (4 upper triangles, ASYMMETRIC, non-zero monopole).
// The area-weighted normal sum points in +z, giving non-trivial far-field
// gradient.  Use this for testing the Taylor approximation path.
struct HalfOctahedronMesh {
    // Top 4 faces of the octahedron: they share the +z vertex (index 4).
    static constexpr int k_nv = 5;
    static constexpr int k_nt = 4;
    std::array<Real, k_nv> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f};
    std::array<Real, k_nv> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f};
    std::array<Real, k_nv> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    std::array<Index, k_nt> i0{0, 2, 1, 3};
    std::array<Index, k_nt> i1{2, 1, 3, 0};
    std::array<Index, k_nt> i2{4, 4, 4, 4};

    std::span<Real const> sx() const { return {vx.data(), vx.size()}; }
    std::span<Real const> sy() const { return {vy.data(), vy.size()}; }
    std::span<Real const> sz() const { return {vz.data(), vz.size()}; }
    std::span<Index const> si0() const { return {i0.data(), i0.size()}; }
    std::span<Index const> si1() const { return {i1.data(), i1.size()}; }
    std::span<Index const> si2() const { return {i2.data(), i2.size()}; }
};

// Helper: upload geometry, build BVH, query gradient, copy back.
// Returns true on success; skips if CUDA unavailable.
template <int Order, std::size_t Nv, std::size_t Nt>
bool run_gradient_query(
    std::array<Real, Nv> const &vx, std::array<Real, Nv> const &vy,
    std::array<Real, Nv> const &vz, std::array<Index, Nt> const &i0,
    std::array<Index, Nt> const &i1, std::array<Index, Nt> const &i2,
    std::vector<Real> const &qx, std::vector<Real> const &qy, std::vector<Real> const &qz,
    std::vector<Real> &out_gx, std::vector<Real> &out_gy, std::vector<Real> &out_gz,
    Real const accuracy_scale = Real(2)
) {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const upload_status = geometry.upload(
        cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    if (upload_status.error() == gwn::gwn_error::cuda_runtime_error)
        return false;
    if (!upload_status.is_ok()) {
        ADD_FAILURE() << "geometry upload failed: "
                      << gwn::tests::status_to_debug_string(upload_status);
        return false;
    }

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Order, Real, Index> data;
    gwn::gwn_status const build_status =
        gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
            geometry, bvh, aabb, data
        );
    if (!build_status.is_ok()) {
        ADD_FAILURE() << "BVH build failed: "
                      << gwn::tests::status_to_debug_string(build_status);
        return false;
    }

    std::size_t const n = qx.size();
    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_gx, d_gy, d_gz;
    bool ok = d_qx.resize(n).is_ok() && d_qy.resize(n).is_ok() && d_qz.resize(n).is_ok() &&
              d_gx.resize(n).is_ok() && d_gy.resize(n).is_ok() && d_gz.resize(n).is_ok();
    ok = ok &&
         d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), n)).is_ok() &&
         d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), n)).is_ok() &&
         d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device buffer allocation/upload failed";
        return false;
    }

    gwn::gwn_status const query_status =
        gwn::gwn_compute_winding_gradient_batch_bvh_taylor<Order, Real, Index>(
            geometry.accessor(), bvh.accessor(), data.accessor(), d_qx.span(), d_qy.span(),
            d_qz.span(), d_gx.span(), d_gy.span(), d_gz.span(), accuracy_scale
        );
    if (!query_status.is_ok()) {
        ADD_FAILURE() << "gradient query failed: "
                      << gwn::tests::status_to_debug_string(query_status);
        return false;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    out_gx.resize(n);
    out_gy.resize(n);
    out_gz.resize(n);
    ok = d_gx.copy_to_host(cuda::std::span<Real>(out_gx.data(), n)).is_ok() &&
         d_gy.copy_to_host(cuda::std::span<Real>(out_gy.data(), n)).is_ok() &&
         d_gz.copy_to_host(cuda::std::span<Real>(out_gz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device-to-host copy failed";
        return false;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return true;
}

} // namespace

// Test 1: Near-surface half-octahedron, exercises BVH leaf (brute-force) path.
//
// The half-octahedron is an OPEN mesh (4 upper triangles only).  Its winding
// number is not constant anywhere, so ∇w is non-trivially large near the mesh
// surface.  Queries above the mesh faces at small height (z=0.2 above the
// faces at z≈0) have large gradients.  accuracy_scale=1000 forces the BVH
// traversal to use brute-force leaf evaluation on every node.
TEST_F(CudaFixture, gradient_near_surface_half_octahedron_brute_force) {
    constexpr Real k_tol = 5e-3f;
    constexpr Real k_fd_h = 1e-3f;
    constexpr Real k_min_mag = 5e-3f;

    HalfOctahedronMesh mesh;

    // Near-surface queries: above the face interiors of the half-octahedron.
    // Each upper face has a centroid roughly at z≈0.33 (for the top half).
    // Queries at z=0.7 are close to face centers and have non-trivial gradients.
    std::vector<Real> qx{ 0.3f, -0.3f, 0.3f, -0.3f, 0.0f};
    std::vector<Real> qy{ 0.3f,  0.3f,-0.3f, -0.3f, 0.0f};
    std::vector<Real> qz{ 0.7f,  0.7f, 0.7f,  0.7f, 0.5f};
    std::size_t const n = qx.size();

    // CPU finite-difference reference.
    std::vector<Real> ref_gx(n), ref_gy(n), ref_gz(n);
    int nontrivial = 0;
    for (std::size_t i = 0; i < n; ++i) {
        reference_winding_gradient_fd<Real, Index>(
            mesh.sx(), mesh.sy(), mesh.sz(), mesh.si0(), mesh.si1(), mesh.si2(),
            qx[i], qy[i], qz[i], k_fd_h, ref_gx[i], ref_gy[i], ref_gz[i]
        );
        Real const mag = std::sqrt(ref_gx[i] * ref_gx[i] + ref_gy[i] * ref_gy[i] +
                                   ref_gz[i] * ref_gz[i]);
        if (mag > k_min_mag)
            ++nontrivial;
    }
    ASSERT_GE(nontrivial, static_cast<int>(n / 2))
        << "reference gradient is trivially small at most query points, test design issue";

    // GPU gradient with large accuracy_scale to force brute-force leaves.
    std::vector<Real> gpu_gx, gpu_gy, gpu_gz;
    if (!run_gradient_query<1>(
            mesh.vx, mesh.vy, mesh.vz, mesh.i0, mesh.i1, mesh.i2,
            qx, qy, qz, gpu_gx, gpu_gy, gpu_gz, /*accuracy_scale=*/Real(1000)
        ))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_gx[i], ref_gx[i], k_tol)
            << "query " << i << " grad_x: gpu=" << gpu_gx[i] << " ref=" << ref_gx[i];
        EXPECT_NEAR(gpu_gy[i], ref_gy[i], k_tol)
            << "query " << i << " grad_y: gpu=" << gpu_gy[i] << " ref=" << ref_gy[i];
        EXPECT_NEAR(gpu_gz[i], ref_gz[i], k_tol)
            << "query " << i << " grad_z: gpu=" << gpu_gz[i] << " ref=" << ref_gz[i];
    }
}

// Test 2: Single-triangle gradient, isolated Biot-Savart formula test.
//
// Triangle in xy-plane: (0,0,0),(1,0,0),(0,1,0), normal = +z.
// Query is above the triangle interior.  The gradient is large and direction-
// ally predictable (should point mostly toward the centroid in the xy-plane).
// This test is a direct unit test of gwn_gradient_solid_angle_triangle_impl
// via the public batch API with a single-triangle mesh.
TEST_F(CudaFixture, gradient_single_triangle_brute_force) {
    constexpr Real k_tol = 2e-3f;
    constexpr Real k_fd_h = 1e-3f;
    constexpr Real k_min_mag = 1e-3f;

    // Single triangle.
    std::array<Real, 3> tvx{0.0f, 1.0f, 0.0f};
    std::array<Real, 3> tvy{0.0f, 0.0f, 1.0f};
    std::array<Real, 3> tvz{0.0f, 0.0f, 0.0f};
    std::array<Index, 1> ti0{0};
    std::array<Index, 1> ti1{1};
    std::array<Index, 1> ti2{2};

    // Queries above the triangle at various heights.
    std::vector<Real> qx{0.3f, 0.2f, 0.1f};
    std::vector<Real> qy{0.3f, 0.1f, 0.2f};
    std::vector<Real> qz{0.5f, 1.0f, 2.0f};
    std::size_t const n = qx.size();

    std::span<Real const> spvx{tvx.data(), tvx.size()};
    std::span<Real const> spvy{tvy.data(), tvy.size()};
    std::span<Real const> spvz{tvz.data(), tvz.size()};
    std::span<Index const> spi0{ti0.data(), ti0.size()};
    std::span<Index const> spi1{ti1.data(), ti1.size()};
    std::span<Index const> spi2{ti2.data(), ti2.size()};

    // CPU finite-difference reference.
    std::vector<Real> ref_gx(n), ref_gy(n), ref_gz(n);
    int nontrivial = 0;
    for (std::size_t i = 0; i < n; ++i) {
        reference_winding_gradient_fd<Real, Index>(
            spvx, spvy, spvz, spi0, spi1, spi2,
            qx[i], qy[i], qz[i], k_fd_h, ref_gx[i], ref_gy[i], ref_gz[i]
        );
        Real const mag = std::sqrt(ref_gx[i] * ref_gx[i] + ref_gy[i] * ref_gy[i] +
                                   ref_gz[i] * ref_gz[i]);
        if (mag > k_min_mag)
            ++nontrivial;
    }
    ASSERT_GE(nontrivial, 1)
        << "all reference gradients near single triangle are trivially small";

    // GPU gradient (single triangle → always brute-force leaf).
    std::vector<Real> gpu_gx, gpu_gy, gpu_gz;
    if (!run_gradient_query<1>(
            tvx, tvy, tvz, ti0, ti1, ti2,
            qx, qy, qz, gpu_gx, gpu_gy, gpu_gz, /*accuracy_scale=*/Real(2)
        ))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_gx[i], ref_gx[i], k_tol)
            << "query " << i << " grad_x: gpu=" << gpu_gx[i] << " ref=" << ref_gx[i];
        EXPECT_NEAR(gpu_gy[i], ref_gy[i], k_tol)
            << "query " << i << " grad_y: gpu=" << gpu_gy[i] << " ref=" << ref_gy[i];
        EXPECT_NEAR(gpu_gz[i], ref_gz[i], k_tol)
            << "query " << i << " grad_z: gpu=" << gpu_gz[i] << " ref=" << ref_gz[i];
    }
}

// Test 3: Far-field half-octahedron, exercises the Taylor approximation path.
//
// The half-octahedron is asymmetric: its area-weighted normal sum N ≠ 0
// (points in +z direction), so the Order-0 Taylor expansion gives a non-zero
// gradient for far-field queries.  Queries at r≈3 trigger the far-field
// criterion (r > accuracy_scale * max_cluster_radius ≈ 2 × 1), so the Taylor
// expansion is actually used instead of brute-force.  This is the primary
// test that the Taylor gradient coefficients are correct.
TEST_F(CudaFixture, gradient_order0_half_octahedron_far_field_taylor) {
    constexpr Real k_tol = 3e-3f;
    constexpr Real k_fd_h = 1e-3f;

    HalfOctahedronMesh mesh;

    // Far-field queries at r≈3 (use Taylor approximation, not brute force).
    std::vector<Real> qx{3.5f, -3.0f, 0.0f,  0.0f, 2.0f};
    std::vector<Real> qy{0.0f,  0.0f, 3.5f, -3.0f, 2.0f};
    std::vector<Real> qz{0.0f,  0.0f, 0.0f,  0.0f, 2.0f};
    std::size_t const n = qx.size();

    // CPU finite-difference reference.
    std::vector<Real> ref_gx(n), ref_gy(n), ref_gz(n);
    int nontrivial = 0;
    for (std::size_t i = 0; i < n; ++i) {
        reference_winding_gradient_fd<Real, Index>(
            mesh.sx(), mesh.sy(), mesh.sz(), mesh.si0(), mesh.si1(), mesh.si2(),
            qx[i], qy[i], qz[i], k_fd_h, ref_gx[i], ref_gy[i], ref_gz[i]
        );
        Real const mag = std::sqrt(ref_gx[i] * ref_gx[i] + ref_gy[i] * ref_gy[i] +
                                   ref_gz[i] * ref_gz[i]);
        if (mag > k_min_nontrivial_mag)
            ++nontrivial;
    }
    // At least some far-field queries must have non-trivial gradients.
    ASSERT_GE(nontrivial, 2)
        << "half-octahedron far-field reference gradients are trivially small, "
           "mesh symmetry cancelled the monopole? Test design issue.";

    std::vector<Real> gpu_gx, gpu_gy, gpu_gz;
    if (!run_gradient_query<0>(
            mesh.vx, mesh.vy, mesh.vz, mesh.i0, mesh.i1, mesh.i2,
            qx, qy, qz, gpu_gx, gpu_gy, gpu_gz, Real(2)
        ))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_gx[i], ref_gx[i], k_tol)
            << "query " << i << " grad_x: gpu=" << gpu_gx[i] << " ref=" << ref_gx[i];
        EXPECT_NEAR(gpu_gy[i], ref_gy[i], k_tol)
            << "query " << i << " grad_y: gpu=" << gpu_gy[i] << " ref=" << ref_gy[i];
        EXPECT_NEAR(gpu_gz[i], ref_gz[i], k_tol)
            << "query " << i << " grad_z: gpu=" << gpu_gz[i] << " ref=" << ref_gz[i];
    }
}

TEST_F(CudaFixture, gradient_order1_half_octahedron_far_field_taylor) {
    constexpr Real k_tol = 1e-3f;
    constexpr Real k_fd_h = 1e-3f;

    HalfOctahedronMesh mesh;

    std::vector<Real> qx{3.5f, -3.0f, 0.0f,  0.0f, 2.0f};
    std::vector<Real> qy{0.0f,  0.0f, 3.5f, -3.0f, 2.0f};
    std::vector<Real> qz{0.0f,  0.0f, 0.0f,  0.0f, 2.0f};
    std::size_t const n = qx.size();

    std::vector<Real> ref_gx(n), ref_gy(n), ref_gz(n);
    int nontrivial = 0;
    for (std::size_t i = 0; i < n; ++i) {
        reference_winding_gradient_fd<Real, Index>(
            mesh.sx(), mesh.sy(), mesh.sz(), mesh.si0(), mesh.si1(), mesh.si2(),
            qx[i], qy[i], qz[i], k_fd_h, ref_gx[i], ref_gy[i], ref_gz[i]
        );
        Real const mag = std::sqrt(ref_gx[i] * ref_gx[i] + ref_gy[i] * ref_gy[i] +
                                   ref_gz[i] * ref_gz[i]);
        if (mag > k_min_nontrivial_mag)
            ++nontrivial;
    }
    ASSERT_GE(nontrivial, 2)
        << "reference gradients are trivially small, test design issue";

    std::vector<Real> gpu_gx, gpu_gy, gpu_gz;
    if (!run_gradient_query<1>(
            mesh.vx, mesh.vy, mesh.vz, mesh.i0, mesh.i1, mesh.i2,
            qx, qy, qz, gpu_gx, gpu_gy, gpu_gz, Real(2)
        ))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_gx[i], ref_gx[i], k_tol)
            << "query " << i << " grad_x: gpu=" << gpu_gx[i] << " ref=" << ref_gx[i];
        EXPECT_NEAR(gpu_gy[i], ref_gy[i], k_tol)
            << "query " << i << " grad_y: gpu=" << gpu_gy[i] << " ref=" << ref_gy[i];
        EXPECT_NEAR(gpu_gz[i], ref_gz[i], k_tol)
            << "query " << i << " grad_z: gpu=" << gpu_gz[i] << " ref=" << ref_gz[i];
    }
}

// Test 4: Order-1 more accurate than Order-0 for half-octahedron far field.
TEST_F(CudaFixture, gradient_order1_more_accurate_than_order0_half_octahedron) {
    constexpr Real k_fd_h = 1e-4f;

    HalfOctahedronMesh mesh;

    std::vector<Real> qx{3.5f, -3.0f, 0.0f,  0.0f};
    std::vector<Real> qy{0.0f,  0.0f, 3.5f, -3.0f};
    std::vector<Real> qz{0.0f,  0.0f, 0.0f,  0.0f};
    std::size_t const n = qx.size();

    std::vector<Real> ref_gx(n), ref_gy(n), ref_gz(n);
    for (std::size_t i = 0; i < n; ++i) {
        reference_winding_gradient_fd<Real, Index>(
            mesh.sx(), mesh.sy(), mesh.sz(), mesh.si0(), mesh.si1(), mesh.si2(),
            qx[i], qy[i], qz[i], k_fd_h, ref_gx[i], ref_gy[i], ref_gz[i]
        );
    }

    std::vector<Real> gpu0_gx, gpu0_gy, gpu0_gz;
    if (!run_gradient_query<0>(
            mesh.vx, mesh.vy, mesh.vz, mesh.i0, mesh.i1, mesh.i2,
            qx, qy, qz, gpu0_gx, gpu0_gy, gpu0_gz, Real(2)
        ))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    std::vector<Real> gpu1_gx, gpu1_gy, gpu1_gz;
    if (!run_gradient_query<1>(
            mesh.vx, mesh.vy, mesh.vz, mesh.i0, mesh.i1, mesh.i2,
            qx, qy, qz, gpu1_gx, gpu1_gy, gpu1_gz, Real(2)
        ))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    Real max_err0 = 0.0f, max_err1 = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        max_err0 = std::max(max_err0, std::abs(gpu0_gx[i] - ref_gx[i]));
        max_err0 = std::max(max_err0, std::abs(gpu0_gy[i] - ref_gy[i]));
        max_err0 = std::max(max_err0, std::abs(gpu0_gz[i] - ref_gz[i]));
        max_err1 = std::max(max_err1, std::abs(gpu1_gx[i] - ref_gx[i]));
        max_err1 = std::max(max_err1, std::abs(gpu1_gy[i] - ref_gy[i]));
        max_err1 = std::max(max_err1, std::abs(gpu1_gz[i] - ref_gz[i]));
    }
    EXPECT_LE(max_err1, max_err0 + 1e-6f)
        << "Order-1 should be at least as accurate as Order-0"
        << " (err0=" << max_err0 << " err1=" << max_err1 << ")";
}

// Test 5: Order-2 half-octahedron far-field Taylor, tests the Order-2
//         gradient formula (4th derivative of K contracted with N_{ijk}).
TEST_F(CudaFixture, gradient_order2_half_octahedron_far_field_taylor) {
    constexpr Real k_tol = 5e-4f;
    constexpr Real k_fd_h = 1e-3f;

    HalfOctahedronMesh mesh;

    std::vector<Real> qx{3.5f, -3.0f, 0.0f,  0.0f, 2.0f};
    std::vector<Real> qy{0.0f,  0.0f, 3.5f, -3.0f, 2.0f};
    std::vector<Real> qz{0.0f,  0.0f, 0.0f,  0.0f, 2.0f};
    std::size_t const n = qx.size();

    std::vector<Real> ref_gx(n), ref_gy(n), ref_gz(n);
    int nontrivial = 0;
    for (std::size_t i = 0; i < n; ++i) {
        reference_winding_gradient_fd<Real, Index>(
            mesh.sx(), mesh.sy(), mesh.sz(), mesh.si0(), mesh.si1(), mesh.si2(),
            qx[i], qy[i], qz[i], k_fd_h, ref_gx[i], ref_gy[i], ref_gz[i]
        );
        Real const mag = std::sqrt(ref_gx[i] * ref_gx[i] + ref_gy[i] * ref_gy[i] +
                                   ref_gz[i] * ref_gz[i]);
        if (mag > k_min_nontrivial_mag)
            ++nontrivial;
    }
    ASSERT_GE(nontrivial, 2)
        << "reference gradients are trivially small, test design issue";

    std::vector<Real> gpu_gx, gpu_gy, gpu_gz;
    if (!run_gradient_query<2>(
            mesh.vx, mesh.vy, mesh.vz, mesh.i0, mesh.i1, mesh.i2,
            qx, qy, qz, gpu_gx, gpu_gy, gpu_gz, Real(2)
        ))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_gx[i], ref_gx[i], k_tol)
            << "query " << i << " grad_x: gpu=" << gpu_gx[i] << " ref=" << ref_gx[i];
        EXPECT_NEAR(gpu_gy[i], ref_gy[i], k_tol)
            << "query " << i << " grad_y: gpu=" << gpu_gy[i] << " ref=" << ref_gy[i];
        EXPECT_NEAR(gpu_gz[i], ref_gz[i], k_tol)
            << "query " << i << " grad_z: gpu=" << gpu_gz[i] << " ref=" << ref_gz[i];
    }
}

// Test 6: Order-2 more accurate than Order-1 for half-octahedron far field.
TEST_F(CudaFixture, gradient_order2_more_accurate_than_order1_half_octahedron) {
    constexpr Real k_fd_h = 1e-4f;

    HalfOctahedronMesh mesh;

    std::vector<Real> qx{3.5f, -3.0f, 0.0f,  0.0f};
    std::vector<Real> qy{0.0f,  0.0f, 3.5f, -3.0f};
    std::vector<Real> qz{0.0f,  0.0f, 0.0f,  0.0f};
    std::size_t const n = qx.size();

    std::vector<Real> ref_gx(n), ref_gy(n), ref_gz(n);
    for (std::size_t i = 0; i < n; ++i) {
        reference_winding_gradient_fd<Real, Index>(
            mesh.sx(), mesh.sy(), mesh.sz(), mesh.si0(), mesh.si1(), mesh.si2(),
            qx[i], qy[i], qz[i], k_fd_h, ref_gx[i], ref_gy[i], ref_gz[i]
        );
    }

    std::vector<Real> gpu1_gx, gpu1_gy, gpu1_gz;
    if (!run_gradient_query<1>(
            mesh.vx, mesh.vy, mesh.vz, mesh.i0, mesh.i1, mesh.i2,
            qx, qy, qz, gpu1_gx, gpu1_gy, gpu1_gz, Real(2)
        ))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    std::vector<Real> gpu2_gx, gpu2_gy, gpu2_gz;
    if (!run_gradient_query<2>(
            mesh.vx, mesh.vy, mesh.vz, mesh.i0, mesh.i1, mesh.i2,
            qx, qy, qz, gpu2_gx, gpu2_gy, gpu2_gz, Real(2)
        ))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    Real max_err1 = 0.0f, max_err2 = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        max_err1 = std::max(max_err1, std::abs(gpu1_gx[i] - ref_gx[i]));
        max_err1 = std::max(max_err1, std::abs(gpu1_gy[i] - ref_gy[i]));
        max_err1 = std::max(max_err1, std::abs(gpu1_gz[i] - ref_gz[i]));
        max_err2 = std::max(max_err2, std::abs(gpu2_gx[i] - ref_gx[i]));
        max_err2 = std::max(max_err2, std::abs(gpu2_gy[i] - ref_gy[i]));
        max_err2 = std::max(max_err2, std::abs(gpu2_gz[i] - ref_gz[i]));
    }
    EXPECT_LE(max_err2, max_err1 + 1e-6f)
        << "Order-2 should be at least as accurate as Order-1"
        << " (err1=" << max_err1 << " err2=" << max_err2 << ")";
}

// Test 7: Closed-mesh interior points should have near-zero gradient.
//
// For a consistently oriented closed mesh, winding number is constant (=1)
// at interior points. The exact interior gradient is therefore zero.
TEST_F(CudaFixture, gradient_closed_octahedron_interior_near_zero) {
    OctahedronMesh mesh;

    std::vector<Real> qx{0.0f, 0.05f, -0.04f, 0.02f};
    std::vector<Real> qy{0.0f, -0.03f, 0.04f, -0.02f};
    std::vector<Real> qz{0.0f, 0.02f, -0.05f, 0.03f};

    std::vector<Real> gx, gy, gz;
    if (!run_gradient_query<1>(
            mesh.vx, mesh.vy, mesh.vz, mesh.i0, mesh.i1, mesh.i2,
            qx, qy, qz, gx, gy, gz, Real(2)
        ))
        GTEST_SKIP() << "CUDA unavailable or build failed";

    for (std::size_t i = 0; i < qx.size(); ++i) {
        Real const gm = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i]);
        EXPECT_LT(gm, Real(5e-3))
            << "interior gradient should be near zero at query " << i;
    }
}

// Test 8: Error-handling, mismatched output spans.
TEST_F(CudaFixture, gradient_mismatched_output_returns_error) {
    gwn::gwn_geometry_accessor<Real, Index> accessor{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_moment_accessor<0, Real, Index> data{};
    cuda::std::span<Real const> empty{};
    Real dummy[2] = {};
    cuda::std::span<Real> output(dummy, 2);
    cuda::std::span<Real> empty_out{};

    gwn::gwn_status const status =
        gwn::gwn_compute_winding_gradient_batch_bvh_taylor<0, Real, Index>(
            accessor, bvh, data, empty, empty, empty, output, empty_out, empty_out
        );
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}
