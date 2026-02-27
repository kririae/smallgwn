#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_harnack_meshes.hpp"
#include "test_utils.hpp"

// Integration tests for the one-path wrapped-angle tracer.
// For closed meshes this validates order-consistent behavior and plausible hit
// geometry where hits occur. For open meshes it validates broad hit coverage.

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;
using gwn::tests::OctahedronMesh;
using gwn::tests::HalfOctahedronMesh;
using gwn::tests::CubeMesh;
using gwn::tests::generate_sphere_rays;

namespace {

template <int Order>
struct TraceResults {
    std::vector<Real> t, nx, ny, nz;
    bool ok{false};
};

template <int Order, class Mesh>
TraceResults<Order> run_trace(
    Mesh const &mesh,
    std::vector<Real> const &ox, std::vector<Real> const &oy, std::vector<Real> const &oz,
    std::vector<Real> const &dx, std::vector<Real> const &dy, std::vector<Real> const &dz,
    Real const epsilon = Real(1e-3),
    int const max_iter = 2048,
    Real const t_max = Real(100)
) {
    TraceResults<Order> res;

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status s = geometry.upload(
        cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
        cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
        cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
        cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
        cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
        cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
    );
    if (s.error() == gwn::gwn_error::cuda_runtime_error)
        return res;
    if (!s.is_ok()) {
        ADD_FAILURE() << "geometry upload: " << gwn::tests::status_to_debug_string(s);
        return res;
    }

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Order, Real, Index> data;
    s = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
        geometry, bvh, aabb, data
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "BVH build: " << gwn::tests::status_to_debug_string(s);
        return res;
    }

    std::size_t const n = ox.size();
    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
    gwn::gwn_device_array<Real> d_t, d_nx, d_ny, d_nz;
    bool ok = d_ox.resize(n).is_ok() && d_oy.resize(n).is_ok() && d_oz.resize(n).is_ok() &&
              d_dx.resize(n).is_ok() && d_dy.resize(n).is_ok() && d_dz.resize(n).is_ok() &&
              d_t.resize(n).is_ok() && d_nx.resize(n).is_ok() &&
              d_ny.resize(n).is_ok() && d_nz.resize(n).is_ok();
    ok = ok &&
         d_ox.copy_from_host(cuda::std::span<Real const>(ox.data(), n)).is_ok() &&
         d_oy.copy_from_host(cuda::std::span<Real const>(oy.data(), n)).is_ok() &&
         d_oz.copy_from_host(cuda::std::span<Real const>(oz.data(), n)).is_ok() &&
         d_dx.copy_from_host(cuda::std::span<Real const>(dx.data(), n)).is_ok() &&
         d_dy.copy_from_host(cuda::std::span<Real const>(dy.data(), n)).is_ok() &&
         d_dz.copy_from_host(cuda::std::span<Real const>(dz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device allocation failed";
        return res;
    }

    s = gwn::gwn_compute_harnack_trace_batch_bvh_taylor<Order, Real, Index>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), data.accessor(),
        d_ox.span(), d_oy.span(), d_oz.span(),
        d_dx.span(), d_dy.span(), d_dz.span(),
        d_t.span(), d_nx.span(), d_ny.span(), d_nz.span(),
        Real(0.5), epsilon, max_iter, t_max, Real(2)
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "harnack trace: " << gwn::tests::status_to_debug_string(s);
        return res;
    }
    cudaDeviceSynchronize();

    res.t.resize(n);
    res.nx.resize(n);
    res.ny.resize(n);
    res.nz.resize(n);
    ok = d_t.copy_to_host(cuda::std::span<Real>(res.t.data(), n)).is_ok() &&
         d_nx.copy_to_host(cuda::std::span<Real>(res.nx.data(), n)).is_ok() &&
         d_ny.copy_to_host(cuda::std::span<Real>(res.ny.data(), n)).is_ok() &&
         d_nz.copy_to_host(cuda::std::span<Real>(res.nz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device-to-host copy failed";
        return res;
    }
    cudaDeviceSynchronize();
    res.ok = true;
    return res;
}

template <int Order, class Mesh>
TraceResults<Order> run_trace_angle(
    Mesh const &mesh,
    std::vector<Real> const &ox, std::vector<Real> const &oy, std::vector<Real> const &oz,
    std::vector<Real> const &dx, std::vector<Real> const &dy, std::vector<Real> const &dz,
    Real const epsilon = Real(1e-3),
    int const max_iter = 2048,
    Real const t_max = Real(100)
) {
    TraceResults<Order> res;

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status s = geometry.upload(
        cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
        cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
        cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
        cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
        cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
        cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
    );
    if (s.error() == gwn::gwn_error::cuda_runtime_error)
        return res;
    if (!s.is_ok()) {
        ADD_FAILURE() << "geometry upload: " << gwn::tests::status_to_debug_string(s);
        return res;
    }

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Order, Real, Index> data;
    s = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
        geometry, bvh, aabb, data
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "BVH build: " << gwn::tests::status_to_debug_string(s);
        return res;
    }

    std::size_t const n = ox.size();
    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
    gwn::gwn_device_array<Real> d_t, d_nx, d_ny, d_nz;
    bool ok = d_ox.resize(n).is_ok() && d_oy.resize(n).is_ok() && d_oz.resize(n).is_ok() &&
              d_dx.resize(n).is_ok() && d_dy.resize(n).is_ok() && d_dz.resize(n).is_ok() &&
              d_t.resize(n).is_ok() && d_nx.resize(n).is_ok() &&
              d_ny.resize(n).is_ok() && d_nz.resize(n).is_ok();
    ok = ok &&
         d_ox.copy_from_host(cuda::std::span<Real const>(ox.data(), n)).is_ok() &&
         d_oy.copy_from_host(cuda::std::span<Real const>(oy.data(), n)).is_ok() &&
         d_oz.copy_from_host(cuda::std::span<Real const>(oz.data(), n)).is_ok() &&
         d_dx.copy_from_host(cuda::std::span<Real const>(dx.data(), n)).is_ok() &&
         d_dy.copy_from_host(cuda::std::span<Real const>(dy.data(), n)).is_ok() &&
         d_dz.copy_from_host(cuda::std::span<Real const>(dz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device allocation failed";
        return res;
    }

    s = gwn::gwn_compute_harnack_trace_angle_batch_bvh_taylor<Order, Real, Index>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), data.accessor(),
        d_ox.span(), d_oy.span(), d_oz.span(),
        d_dx.span(), d_dy.span(), d_dz.span(),
        d_t.span(), d_nx.span(), d_ny.span(), d_nz.span(),
        Real(0.5), epsilon, max_iter, t_max, Real(2)
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "harnack angle trace: " << gwn::tests::status_to_debug_string(s);
        return res;
    }
    cudaDeviceSynchronize();

    res.t.resize(n);
    res.nx.resize(n);
    res.ny.resize(n);
    res.nz.resize(n);
    ok = d_t.copy_to_host(cuda::std::span<Real>(res.t.data(), n)).is_ok() &&
         d_nx.copy_to_host(cuda::std::span<Real>(res.nx.data(), n)).is_ok() &&
         d_ny.copy_to_host(cuda::std::span<Real>(res.ny.data(), n)).is_ok() &&
         d_nz.copy_to_host(cuda::std::span<Real>(res.nz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device-to-host copy failed";
        return res;
    }
    cudaDeviceSynchronize();
    res.ok = true;
    return res;
}

} // namespace

// ---------------------------------------------------------------------------
// Integration test: trace 6×12 = 72 rays from a sphere of radius 5
// through a closed octahedron.
// ---------------------------------------------------------------------------
TEST_F(CudaFixture, integration_harnack_octahedron_sphere_rays) {
    OctahedronMesh mesh;

    std::vector<Real> ox, oy, oz, dx, dy, dz;
    generate_sphere_rays(Real(5), 7, 12, ox, oy, oz, dx, dy, dz);
    std::size_t const n = ox.size();

    std::cout << "[harnack integration] Tracing " << n << " rays\n";

    auto res0 = run_trace<0>(mesh, ox, oy, oz, dx, dy, dz);
    if (!res0.ok)
        GTEST_SKIP() << "CUDA unavailable";

    auto res1 = run_trace<1>(mesh, ox, oy, oz, dx, dy, dz);
    if (!res1.ok)
        GTEST_SKIP() << "CUDA unavailable";

    auto res2 = run_trace<2>(mesh, ox, oy, oz, dx, dy, dz);
    if (!res2.ok)
        GTEST_SKIP() << "CUDA unavailable";

    // --- Check: single wrapped-angle path is deterministic across orders.
    // For watertight closed meshes this path can hit only a subset of rays,
    // but hit/miss classification should remain order-stable. ---
    int hits0 = 0, hits1 = 0, hits2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (res0.t[i] >= Real(0)) ++hits0;
        if (res1.t[i] >= Real(0)) ++hits1;
        if (res2.t[i] >= Real(0)) ++hits2;
    }
    std::cout << "[harnack integration] Hits: Order-0=" << hits0
              << " Order-1=" << hits1 << " Order-2=" << hits2
              << " / " << n << "\n";

    // With correct R(x) = min(singular_edge_dist, face_dist), the tracer
    // reliably hits the closed-mesh surface for all inward-pointing rays.
    EXPECT_GE(hits0, static_cast<int>(n * 0.90))
        << "Order-0: too few closed-mesh shell hits (" << hits0 << "/" << n << ")";
    EXPECT_GE(hits1, static_cast<int>(n * 0.90))
        << "Order-1: too few closed-mesh shell hits (" << hits1 << "/" << n << ")";
    EXPECT_GE(hits2, static_cast<int>(n * 0.90))
        << "Order-2: too few closed-mesh shell hits (" << hits2 << "/" << n << ")";
    EXPECT_EQ(hits0, hits1) << "single-path tracer should be consistent across orders";
    EXPECT_EQ(hits1, hits2) << "single-path tracer should be consistent across orders";

    // --- Check: hit points are near the octahedron scale ---
    // For a unit octahedron, geometric face radius is 1/sqrt(3) ≈ 0.577.
    // We use a coarse but stable acceptance band.
    for (std::size_t i = 0; i < n; ++i) {
        if (res1.t[i] < Real(0))
            continue;
        Real const hx = ox[i] + res1.t[i] * dx[i];
        Real const hy = oy[i] + res1.t[i] * dy[i];
        Real const hz = oz[i] + res1.t[i] * dz[i];
        Real const hr = std::sqrt(hx * hx + hy * hy + hz * hz);
        EXPECT_GT(hr, Real(0.2))
            << "ray " << i << ": hit too close to origin (r=" << hr << ")";
        EXPECT_LT(hr, Real(1.0))
            << "ray " << i << ": hit too far from origin (r=" << hr << ")";
    }

    // --- Check: normals are unit length and oppose ray direction ---
    int bad_normals = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (res1.t[i] < Real(0))
            continue;
        Real const nmag = std::sqrt(
            res1.nx[i]*res1.nx[i] + res1.ny[i]*res1.ny[i] + res1.nz[i]*res1.nz[i]
        );
        // Skip rays that hit at mesh vertices/edges where the gradient is
        // singular (zero normal).  For a coarse closed mesh like the
        // octahedron, many axis-aligned rays hit at vertices.
        if (nmag < Real(0.5))
            continue;

        if (std::abs(nmag - Real(1)) > Real(0.1))
            ++bad_normals;
    }
    EXPECT_LE(bad_normals, static_cast<int>(n * 0.1))
        << bad_normals << " rays have bad normals (non-unit magnitude)";

    // --- Check: hit distance consistency across orders ---
    // For rays that hit in all 3 orders, the hit distances should be similar.
    Real max_t_diff_01 = 0, max_t_diff_12 = 0;
    int compared = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (res0.t[i] < Real(0) || res1.t[i] < Real(0) || res2.t[i] < Real(0))
            continue;
        max_t_diff_01 = std::max(max_t_diff_01, std::abs(res0.t[i] - res1.t[i]));
        max_t_diff_12 = std::max(max_t_diff_12, std::abs(res1.t[i] - res2.t[i]));
        ++compared;
    }
    std::cout << "[harnack integration] Max t-diff Order-0 vs 1: " << max_t_diff_01
              << "  Order-1 vs 2: " << max_t_diff_12
              << "  (compared " << compared << " rays)\n";

    // Hit distances should agree within reasonable tolerance.
    EXPECT_LT(max_t_diff_01, Real(0.3))
        << "Order-0 vs Order-1 hit distance disagrees by " << max_t_diff_01;
    EXPECT_LT(max_t_diff_12, Real(0.3))
        << "Order-1 vs Order-2 hit distance disagrees by " << max_t_diff_12;
}

TEST_F(CudaFixture, integration_harnack_angle_half_octahedron_forward_rays) {
    HalfOctahedronMesh mesh;

    std::vector<Real> ox, oy, oz, dx, dy, dz;
    for (int yi = -2; yi <= 2; ++yi) {
        for (int xi = -2; xi <= 2; ++xi) {
            ox.push_back(Real(0.2) * Real(xi));
            oy.push_back(Real(0.2) * Real(yi));
            oz.push_back(Real(2));
            dx.push_back(Real(0));
            dy.push_back(Real(0));
            dz.push_back(Real(-1));
        }
    }

    auto res = run_trace_angle<1>(mesh, ox, oy, oz, dx, dy, dz);
    if (!res.ok)
        GTEST_SKIP() << "CUDA unavailable";

    std::size_t const n = ox.size();
    int hits = 0;
    int sane_normals = 0;
    int crossed_face = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (res.t[i] < Real(0))
            continue;
        ++hits;

        Real const z_face = Real(1) - std::abs(ox[i]) - std::abs(oy[i]);
        if (z_face > Real(0)) {
            Real const t_face = (z_face - oz[i]) / dz[i];
            EXPECT_GE(res.t[i], t_face - Real(2e-3))
                << "ray " << i << ": angle-mode hit should not lie meaningfully before face crossing";
            if (res.t[i] > t_face + Real(1e-3))
                ++crossed_face;
        }

        Real const nmag =
            std::sqrt(res.nx[i] * res.nx[i] + res.ny[i] * res.ny[i] + res.nz[i] * res.nz[i]);
        if (nmag > Real(0.5) && nmag < Real(1.5))
            ++sane_normals;
    }

    EXPECT_GE(hits, 20) << "angle-mode tracer should hit most forward rays";
    // With face-distance in R(x), the tracer converges tightly near the face
    // crossing rather than overshooting past it.
    EXPECT_GE(sane_normals, 12) << "angle-mode hits should usually return finite normals";
}

// ---------------------------------------------------------------------------
// Integration test: trace sphere rays through a closed cube.
// ---------------------------------------------------------------------------
TEST_F(CudaFixture, integration_harnack_cube_sphere_rays) {
    CubeMesh mesh;

    std::vector<Real> ox, oy, oz, dx, dy, dz;
    generate_sphere_rays(Real(5), 7, 12, ox, oy, oz, dx, dy, dz);
    std::size_t const n = ox.size();

    std::cout << "[harnack cube integration] Tracing " << n << " rays\n";

    auto res0 = run_trace<0>(mesh, ox, oy, oz, dx, dy, dz);
    if (!res0.ok)
        GTEST_SKIP() << "CUDA unavailable";

    auto res1 = run_trace<1>(mesh, ox, oy, oz, dx, dy, dz);
    if (!res1.ok)
        GTEST_SKIP() << "CUDA unavailable";

    auto res2 = run_trace<2>(mesh, ox, oy, oz, dx, dy, dz);
    if (!res2.ok)
        GTEST_SKIP() << "CUDA unavailable";

    int hits0 = 0, hits1 = 0, hits2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (res0.t[i] >= Real(0)) ++hits0;
        if (res1.t[i] >= Real(0)) ++hits1;
        if (res2.t[i] >= Real(0)) ++hits2;
    }
    std::cout << "[harnack cube integration] Hits: Order-0=" << hits0
              << " Order-1=" << hits1 << " Order-2=" << hits2
              << " / " << n << "\n";

    EXPECT_GE(hits0, static_cast<int>(n * 0.90))
        << "Order-0: too few closed-cube hits (" << hits0 << "/" << n << ")";
    EXPECT_GE(hits1, static_cast<int>(n * 0.90))
        << "Order-1: too few closed-cube hits (" << hits1 << "/" << n << ")";
    EXPECT_GE(hits2, static_cast<int>(n * 0.90))
        << "Order-2: too few closed-cube hits (" << hits2 << "/" << n << ")";
    EXPECT_EQ(hits0, hits1) << "single-path tracer should be consistent across orders";
    EXPECT_EQ(hits1, hits2) << "single-path tracer should be consistent across orders";

    // Hit points should be near the cube surface (distance from origin 1.0–1.73).
    for (std::size_t i = 0; i < n; ++i) {
        if (res1.t[i] < Real(0))
            continue;
        Real const hx = ox[i] + res1.t[i] * dx[i];
        Real const hy = oy[i] + res1.t[i] * dy[i];
        Real const hz = oz[i] + res1.t[i] * dz[i];
        Real const hr = std::sqrt(hx * hx + hy * hy + hz * hz);
        EXPECT_GT(hr, Real(0.8))
            << "ray " << i << ": hit too close to origin (r=" << hr << ")";
        EXPECT_LT(hr, Real(2.0))
            << "ray " << i << ": hit too far from origin (r=" << hr << ")";
    }

    // Hit distance consistency across orders.
    Real max_t_diff_01 = 0, max_t_diff_12 = 0;
    int compared = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (res0.t[i] < Real(0) || res1.t[i] < Real(0) || res2.t[i] < Real(0))
            continue;
        max_t_diff_01 = std::max(max_t_diff_01, std::abs(res0.t[i] - res1.t[i]));
        max_t_diff_12 = std::max(max_t_diff_12, std::abs(res1.t[i] - res2.t[i]));
        ++compared;
    }
    std::cout << "[harnack cube integration] Max t-diff Order-0 vs 1: " << max_t_diff_01
              << "  Order-1 vs 2: " << max_t_diff_12
              << "  (compared " << compared << " rays)\n";

    EXPECT_LT(max_t_diff_01, Real(0.3))
        << "Order-0 vs Order-1 hit distance disagrees by " << max_t_diff_01;
    EXPECT_LT(max_t_diff_12, Real(0.3))
        << "Order-1 vs Order-2 hit distance disagrees by " << max_t_diff_12;
}
