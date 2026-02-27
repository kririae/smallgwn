#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_harnack_meshes.hpp"
#include "test_utils.hpp"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;
using gwn::tests::OctahedronMesh;
using gwn::tests::HalfOctahedronMesh;
using gwn::tests::CubeMesh;
using gwn::tests::OpenCubeMesh;
using gwn::tests::half_octahedron_face_z;
using gwn::tests::generate_sphere_rays;
using gwn::tests::wrapped_angle_residual;

namespace {

// ---------------------------------------------------------------------------
// GPU helper — builds BVH, uploads rays, launches Harnack trace, copies back.
// ---------------------------------------------------------------------------

template <int Order, class Mesh>
bool run_harnack_trace(
    Mesh const &mesh,
    std::vector<Real> const &ox, std::vector<Real> const &oy, std::vector<Real> const &oz,
    std::vector<Real> const &dx, std::vector<Real> const &dy, std::vector<Real> const &dz,
    std::vector<Real> &out_t,
    std::vector<Real> &out_nx, std::vector<Real> &out_ny, std::vector<Real> &out_nz,
    Real const target_winding = Real(0.5),
    Real const epsilon = Real(1e-3),
    int const max_iterations = 2048,
    Real const t_max = Real(100),
    Real const accuracy_scale = Real(2)
) {
    // 1. Upload geometry.
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
        return false;
    if (!s.is_ok()) {
        ADD_FAILURE() << "geometry upload: " << gwn::tests::status_to_debug_string(s);
        return false;
    }

    // 2. Build BVH (topology + AABB + moments).
    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Order, Real, Index> data;
    s = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
        geometry, bvh, aabb, data
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "BVH build: " << gwn::tests::status_to_debug_string(s);
        return false;
    }

    // 3. Allocate device buffers & upload rays.
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
        ADD_FAILURE() << "device allocation/upload failed";
        return false;
    }

    // 4. Launch Harnack trace.
    s = gwn::gwn_compute_harnack_trace_batch_bvh_taylor<Order, Real, Index>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), data.accessor(),
        d_ox.span(), d_oy.span(), d_oz.span(),
        d_dx.span(), d_dy.span(), d_dz.span(),
        d_t.span(), d_nx.span(), d_ny.span(), d_nz.span(),
        target_winding, epsilon, max_iterations, t_max, accuracy_scale
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "harnack trace: " << gwn::tests::status_to_debug_string(s);
        return false;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 5. Copy results back.
    out_t.resize(n);
    out_nx.resize(n);
    out_ny.resize(n);
    out_nz.resize(n);
    ok = d_t.copy_to_host(cuda::std::span<Real>(out_t.data(), n)).is_ok() &&
         d_nx.copy_to_host(cuda::std::span<Real>(out_nx.data(), n)).is_ok() &&
         d_ny.copy_to_host(cuda::std::span<Real>(out_ny.data(), n)).is_ok() &&
         d_nz.copy_to_host(cuda::std::span<Real>(out_nz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device-to-host copy failed";
        return false;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return true;
}

template <int Order, class Mesh>
bool run_harnack_trace_angle(
    Mesh const &mesh,
    std::vector<Real> const &ox, std::vector<Real> const &oy, std::vector<Real> const &oz,
    std::vector<Real> const &dx, std::vector<Real> const &dy, std::vector<Real> const &dz,
    std::vector<Real> &out_t,
    std::vector<Real> &out_nx, std::vector<Real> &out_ny, std::vector<Real> &out_nz,
    Real const target_winding = Real(0.5),
    Real const epsilon = Real(1e-3),
    int const max_iterations = 2048,
    Real const t_max = Real(100),
    Real const accuracy_scale = Real(2)
) {
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
        return false;
    if (!s.is_ok()) {
        ADD_FAILURE() << "geometry upload: " << gwn::tests::status_to_debug_string(s);
        return false;
    }

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Order, Real, Index> data;
    s = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
        geometry, bvh, aabb, data
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "BVH build: " << gwn::tests::status_to_debug_string(s);
        return false;
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
        ADD_FAILURE() << "device allocation/upload failed";
        return false;
    }

    s = gwn::gwn_compute_harnack_trace_angle_batch_bvh_taylor<Order, Real, Index>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), data.accessor(),
        d_ox.span(), d_oy.span(), d_oz.span(),
        d_dx.span(), d_dy.span(), d_dz.span(),
        d_t.span(), d_nx.span(), d_ny.span(), d_nz.span(),
        target_winding, epsilon, max_iterations, t_max, accuracy_scale
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "harnack angle trace: " << gwn::tests::status_to_debug_string(s);
        return false;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    out_t.resize(n);
    out_nx.resize(n);
    out_ny.resize(n);
    out_nz.resize(n);
    ok = d_t.copy_to_host(cuda::std::span<Real>(out_t.data(), n)).is_ok() &&
         d_nx.copy_to_host(cuda::std::span<Real>(out_nx.data(), n)).is_ok() &&
         d_ny.copy_to_host(cuda::std::span<Real>(out_ny.data(), n)).is_ok() &&
         d_nz.copy_to_host(cuda::std::span<Real>(out_nz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device-to-host copy failed";
        return false;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return true;
}

// ---------------------------------------------------------------------------
// GPU helper — evaluate winding number at specific points for cross-checking.
// ---------------------------------------------------------------------------

template <int Order, class Mesh>
bool run_winding_query(
    Mesh const &mesh,
    std::vector<Real> const &qx, std::vector<Real> const &qy, std::vector<Real> const &qz,
    std::vector<Real> &out_w,
    Real const accuracy_scale = Real(2)
) {
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
        return false;
    if (!s.is_ok())
        return false;

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Order, Real, Index> data;
    s = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
        geometry, bvh, aabb, data
    );
    if (!s.is_ok())
        return false;

    std::size_t const n = qx.size();
    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_w;
    bool ok = d_qx.resize(n).is_ok() && d_qy.resize(n).is_ok() &&
              d_qz.resize(n).is_ok() && d_w.resize(n).is_ok();
    ok = ok &&
         d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), n)).is_ok() &&
         d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), n)).is_ok() &&
         d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), n)).is_ok();
    if (!ok)
        return false;

    s = gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, Real, Index>(
        geometry.accessor(), bvh.accessor(), data.accessor(),
        d_qx.span(), d_qy.span(), d_qz.span(), d_w.span(), accuracy_scale
    );
    if (!s.is_ok())
        return false;
    if (cudaDeviceSynchronize() != cudaSuccess)
        return false;

    out_w.resize(n);
    if (!d_w.copy_to_host(cuda::std::span<Real>(out_w.data(), n)).is_ok())
        return false;
    if (cudaDeviceSynchronize() != cudaSuccess)
        return false;
    return true;
}

template <class Mesh>
bool run_edge_distance_query(
    Mesh const &mesh,
    std::vector<Real> const &qx, std::vector<Real> const &qy, std::vector<Real> const &qz,
    std::vector<Real> &out_d
) {
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
        return false;
    if (!s.is_ok())
        return false;

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    s = gwn::gwn_bvh_facade_build_topology_aabb_lbvh<4, Real, Index>(geometry, bvh, aabb);
    if (!s.is_ok())
        return false;

    std::size_t const n = qx.size();
    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    bool ok = d_qx.resize(n).is_ok() && d_qy.resize(n).is_ok() &&
              d_qz.resize(n).is_ok() && d_out.resize(n).is_ok();
    ok = ok &&
         d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), n)).is_ok() &&
         d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), n)).is_ok() &&
         d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), n)).is_ok();
    if (!ok)
        return false;

    s = gwn::gwn_compute_unsigned_edge_distance_batch_bvh<Real, Index>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(),
        d_qx.span(), d_qy.span(), d_qz.span(), d_out.span()
    );
    if (!s.is_ok())
        return false;
    cudaDeviceSynchronize();

    out_d.resize(n);
    if (!d_out.copy_to_host(cuda::std::span<Real>(out_d.data(), n)).is_ok())
        return false;
    cudaDeviceSynchronize();
    return true;
}

inline Real point_segment_distance(
    Real const qx, Real const qy, Real const qz,
    Real const ax, Real const ay, Real const az,
    Real const bx, Real const by, Real const bz
) {
    Real const abx = bx - ax;
    Real const aby = by - ay;
    Real const abz = bz - az;
    Real const apx = qx - ax;
    Real const apy = qy - ay;
    Real const apz = qz - az;
    Real const ab2 = abx * abx + aby * aby + abz * abz;
    if (!(ab2 > Real(0))) {
        Real const dx = apx, dy = apy, dz = apz;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    Real t = (apx * abx + apy * aby + apz * abz) / ab2;
    if (t < Real(0))
        t = Real(0);
    if (t > Real(1))
        t = Real(1);
    Real const cx = ax + t * abx;
    Real const cy = ay + t * aby;
    Real const cz = az + t * abz;
    Real const dx = qx - cx;
    Real const dy = qy - cy;
    Real const dz = qz - cz;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

inline Real cpu_edge_distance(OctahedronMesh const &mesh, Real const qx, Real const qy, Real const qz) {
    Real best = std::numeric_limits<Real>::infinity();
    for (std::size_t ti = 0; ti < mesh.i0.size(); ++ti) {
        auto const ia = static_cast<std::size_t>(mesh.i0[ti]);
        auto const ib = static_cast<std::size_t>(mesh.i1[ti]);
        auto const ic = static_cast<std::size_t>(mesh.i2[ti]);
        best = std::min(
            best, point_segment_distance(
                      qx, qy, qz, mesh.vx[ia], mesh.vy[ia], mesh.vz[ia], mesh.vx[ib], mesh.vy[ib],
                      mesh.vz[ib]
                  )
        );
        best = std::min(
            best, point_segment_distance(
                      qx, qy, qz, mesh.vx[ib], mesh.vy[ib], mesh.vz[ib], mesh.vx[ic], mesh.vy[ic],
                      mesh.vz[ic]
                  )
        );
        best = std::min(
            best, point_segment_distance(
                      qx, qy, qz, mesh.vx[ic], mesh.vy[ic], mesh.vz[ic], mesh.vx[ia], mesh.vy[ia],
                      mesh.vz[ia]
                  )
        );
    }
    return best;
}

} // namespace

// ---------------------------------------------------------------------------
// Test 1: Harnack step-size formula (CPU-only math validation).
// ---------------------------------------------------------------------------
TEST(HarnackStepSize, known_values) {
    using namespace gwn::detail;

    // Exterior approach: w = 0, f* = 0.5, c = −1, R = 1.
    // a = (0+1)/(0.5+1) = 2/3
    // disc = 4/9 + 16/3 = 52/9 → sqrt ≈ 2.404
    // rho = 0.5 × |2/3 + 2 − 2.404| = 0.5 × 0.263 = 0.131
    Real rho = gwn_harnack_step_size(Real(0), Real(0.5), Real(-1), Real(1));
    EXPECT_NEAR(rho, Real(0.131), Real(0.01));

    // At the surface: w = 0.5, f* = 0.5 → a = (0.5+1)/(0.5+1) = 1.
    // disc = 1+8 = 9, sqrt = 3.
    // rho = 0.5 × |1+2−3| = 0.
    rho = gwn_harnack_step_size(Real(0.5), Real(0.5), Real(-1), Real(1));
    EXPECT_NEAR(rho, Real(0), Real(0.001));

    // Interior: w = 1.0, f* = 0.5, c = −1, R = 2.
    // a = (1+1)/(0.5+1) = 4/3
    // disc = 16/9 + 32/3 = 112/9 → sqrt ≈ 3.528
    // rho = (2/2) × |4/3 + 2 − 3.528| ≈ 0.195
    rho = gwn_harnack_step_size(Real(1.0), Real(0.5), Real(-1), Real(2));
    EXPECT_NEAR(rho, Real(0.195), Real(0.02));

    // R = 0 → should return 0 regardless.
    rho = gwn_harnack_step_size(Real(0), Real(0.5), Real(-1), Real(0));
    EXPECT_EQ(rho, Real(0));

    // f* ≤ c → safe full step.
    rho = gwn_harnack_step_size(Real(0), Real(-2), Real(-1), Real(5));
    EXPECT_EQ(rho, Real(5));

    // a ≤ 0 (Taylor error pushes f_t below c) → small positive nudge, not zero.
    rho = gwn_harnack_step_size(Real(-2), Real(0.5), Real(-1), Real(3));
    EXPECT_GT(rho, Real(0)) << "a <= 0 must not stall the ray";
    EXPECT_LE(rho, Real(3)) << "nudge must not exceed R";
    EXPECT_NEAR(rho, Real(3) * Real(1e-4), Real(1e-6));
}

// ---------------------------------------------------------------------------
// Test 1b: Final constrained step must never exceed the safe-ball radius.
// ---------------------------------------------------------------------------
TEST(HarnackStepSize, constrained_step_never_exceeds_radius) {
    using namespace gwn::detail;

    // Tiny radius should remain tiny even when min-step logic is applied.
    Real const R = Real(1e-9);
    Real const rho = gwn_harnack_constrained_step(Real(0), Real(0.5), Real(-1), R);
    EXPECT_LE(rho, R);
}

// ---------------------------------------------------------------------------
// Test 2: Ray parameterization should follow reference semantics.
//
// Reference tracer does NOT normalize ray direction; it scales the step by
// 1/|D|. Therefore scaling ray direction should scale hit t inversely while
// preserving the world-space hit location.
// ---------------------------------------------------------------------------
TEST_F(CudaFixture, harnack_ray_parameterization_nonunit_direction) {
    HalfOctahedronMesh mesh;

    std::vector<Real> ox{Real(0.2), Real(0.2)};
    std::vector<Real> oy{Real(0.1), Real(0.1)};
    std::vector<Real> oz{Real(2.0), Real(2.0)};
    std::vector<Real> dx{Real(0), Real(0)};
    std::vector<Real> dy{Real(0), Real(0)};
    std::vector<Real> dz{Real(-1), Real(-2)};

    std::vector<Real> t, nx, ny, nz;
    if (!run_harnack_trace<1>(mesh, ox, oy, oz, dx, dy, dz, t, nx, ny, nz))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_GE(t[0], Real(0));
    ASSERT_GE(t[1], Real(0));

    // Same world-space hit point, parameter t scales inversely with |D|.
    Real const hx0 = ox[0] + t[0] * dx[0];
    Real const hx1 = ox[1] + t[1] * dx[1];
    EXPECT_NEAR(hx0, hx1, Real(1e-3));
    EXPECT_NEAR(t[0] / Real(2), t[1], Real(2e-2));
}

// ---------------------------------------------------------------------------
// Test 3: Open-mesh convergence against wrapped target level set.
// ---------------------------------------------------------------------------
TEST_F(CudaFixture, harnack_convergence_at_mesh_surface) {
    HalfOctahedronMesh mesh;

    std::vector<Real> ox{0.0f, 0.25f, -0.25f};
    std::vector<Real> oy{0.0f, 0.10f, 0.10f};
    std::vector<Real> oz{2.0f, 2.0f, 2.0f};
    std::vector<Real> dx{0.0f, 0.0f, 0.0f};
    std::vector<Real> dy{0.0f, 0.0f, 0.0f};
    std::vector<Real> dz{-1.0f, -1.0f, -1.0f};

    constexpr Real k_eps = 1e-3f;
    std::vector<Real> ht, hnx, hny, hnz;
    if (!run_harnack_trace<1>(mesh, ox, oy, oz, dx, dy, dz,
                              ht, hnx, hny, hnz, Real(0.5), k_eps))
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> qx_hit, qy_hit, qz_hit;
    std::vector<Real> t_hit;
    std::vector<Real> t_face_hit;
    for (std::size_t i = 0; i < ox.size(); ++i) {
        if (ht[i] < Real(0))
            continue;
        Real const z_face = half_octahedron_face_z(ox[i], oy[i]);
        if (!(z_face > Real(0)))
            continue;
        Real const t_face = (z_face - oz[i]) / dz[i];
        qx_hit.push_back(ox[i] + ht[i] * dx[i]);
        qy_hit.push_back(oy[i] + ht[i] * dy[i]);
        qz_hit.push_back(oz[i] + ht[i] * dz[i]);
        t_hit.push_back(ht[i]);
        t_face_hit.push_back(t_face);
    }
    ASSERT_GE(qx_hit.size(), 2u) << "expected at least two open-mesh hits";

    std::vector<Real> w_hit;
    if (!run_winding_query<1>(mesh, qx_hit, qy_hit, qz_hit, w_hit))
        GTEST_SKIP() << "CUDA unavailable";
    ASSERT_EQ(w_hit.size(), qx_hit.size());
    for (std::size_t i = 0; i < w_hit.size(); ++i) {
        if (!(t_hit[i] > t_face_hit[i] + Real(1e-3)))
            continue;
        Real const residual = wrapped_angle_residual(w_hit[i], Real(0.5));
        EXPECT_LT(residual, Real(0.2))
            << "open-mesh wrapped residual too large at hit " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 4: Surface normal sanity — at hit points where the gradient is
//         well-defined, the returned normal should be approximately unit
//         length and should oppose the incoming ray direction.
//
// Uses axis-aligned rays on the half-octahedron.  Rays through mesh
// vertices (where the gradient may be singular) are skipped.
// ---------------------------------------------------------------------------
TEST_F(CudaFixture, harnack_normal_sanity) {
    HalfOctahedronMesh mesh;

    // Rays from above the half-octahedron pointing down.
    // Use off-center origins to avoid hitting the apex vertex directly.
    std::vector<Real> ox{0.4f, -0.2f, 0.1f};
    std::vector<Real> oy{0.1f,  0.3f, -0.4f};
    std::vector<Real> oz{5,     5,     5};
    std::vector<Real> dx{0,     0,     0};
    std::vector<Real> dy{0,     0,     0};
    std::vector<Real> dz{-1,   -1,    -1};

    std::vector<Real> ht, hnx, hny, hnz;
    if (!run_harnack_trace<1>(mesh, ox, oy, oz, dx, dy, dz, ht, hnx, hny, hnz))
        GTEST_SKIP() << "CUDA unavailable";

    int valid_normals = 0;
    for (std::size_t i = 0; i < ox.size(); ++i) {
        if (ht[i] < Real(0))
            continue;

        Real const nmag = std::sqrt(hnx[i]*hnx[i] + hny[i]*hny[i] + hnz[i]*hnz[i]);
        if (nmag < Real(0.5))
            continue; // Skip degenerate normals (vertex/edge singularity).

        ++valid_normals;

        EXPECT_NEAR(nmag, Real(1), Real(0.1))
            << "ray " << i << ": normal magnitude = " << nmag;

        Real const ndotd = hnx[i] * dx[i] + hny[i] * dy[i] + hnz[i] * dz[i];
        EXPECT_LT(ndotd, Real(0))
            << "ray " << i << ": normal should oppose incoming direction";
    }
    // We expect at least one ray to produce a valid normal.
    EXPECT_GE(valid_normals, 1) << "No rays produced valid normals";
}

// ---------------------------------------------------------------------------
// Test 5: No-hit — rays pointing away from the mesh should not find a hit.
// ---------------------------------------------------------------------------
TEST_F(CudaFixture, harnack_no_hit_rays) {
    OctahedronMesh mesh;

    // Rays starting outside, pointing away from the mesh.
    std::vector<Real> ox{5,  0, 0};
    std::vector<Real> oy{0,  5, 0};
    std::vector<Real> oz{0,  0, 5};
    std::vector<Real> dx{1, 0, 0};  // pointing away
    std::vector<Real> dy{0, 1, 0};
    std::vector<Real> dz{0, 0, 1};

    std::vector<Real> ht, hnx, hny, hnz;
    if (!run_harnack_trace<1>(mesh, ox, oy, oz, dx, dy, dz, ht, hnx, hny, hnz,
                              Real(0.5), Real(1e-3), 256, Real(10)))
        GTEST_SKIP() << "CUDA unavailable";

    for (std::size_t i = 0; i < ox.size(); ++i) {
        EXPECT_LT(ht[i], Real(0))
            << "ray " << i << " should not hit (t=" << ht[i] << ")";
    }
}

// ---------------------------------------------------------------------------
// Test 7: Mismatched output spans return error.
// ---------------------------------------------------------------------------
TEST_F(CudaFixture, harnack_mismatched_spans_returns_error) {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};
    gwn::gwn_bvh4_moment_accessor<0, Real, Index> moment{};
    cuda::std::span<Real const> empty{};
    Real dummy[2] = {};
    cuda::std::span<Real> out2(dummy, 2);
    cuda::std::span<Real> empty_out{};

    gwn::gwn_status const s =
        gwn::gwn_compute_harnack_trace_batch_bvh_taylor<0, Real, Index>(
            geometry, bvh, aabb, moment,
            empty, empty, empty,  // origins (0)
            empty, empty, empty,  // dirs (0)
            out2,                 // output_t (2) — MISMATCH
            empty_out, empty_out, empty_out
        );
    EXPECT_EQ(s.error(), gwn::gwn_error::invalid_argument);
}

// ---------------------------------------------------------------------------
// Test 9: Invalid non-empty accessors/spans must be rejected pre-launch.
// ---------------------------------------------------------------------------
TEST_F(CudaFixture, harnack_batch_rejects_invalid_accessors) {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};
    gwn::gwn_bvh4_moment_accessor<0, Real, Index> moment{};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
    gwn::gwn_device_array<Real> d_t, d_nx, d_ny, d_nz;
    bool ok = d_ox.resize(1).is_ok() && d_oy.resize(1).is_ok() && d_oz.resize(1).is_ok() &&
              d_dx.resize(1).is_ok() && d_dy.resize(1).is_ok() && d_dz.resize(1).is_ok() &&
              d_t.resize(1).is_ok() && d_nx.resize(1).is_ok() &&
              d_ny.resize(1).is_ok() && d_nz.resize(1).is_ok();
    if (!ok)
        GTEST_SKIP() << "CUDA unavailable";

    gwn::gwn_status const s = gwn::gwn_compute_harnack_trace_batch_bvh_taylor<0, Real, Index>(
        geometry, bvh, aabb, moment,
        d_ox.span(), d_oy.span(), d_oz.span(),
        d_dx.span(), d_dy.span(), d_dz.span(),
        d_t.span(), d_nx.span(), d_ny.span(), d_nz.span()
    );
    EXPECT_EQ(s.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, harnack_angle_batch_rejects_invalid_accessors) {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};
    gwn::gwn_bvh4_moment_accessor<0, Real, Index> moment{};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
    gwn::gwn_device_array<Real> d_t, d_nx, d_ny, d_nz;
    bool ok = d_ox.resize(1).is_ok() && d_oy.resize(1).is_ok() && d_oz.resize(1).is_ok() &&
              d_dx.resize(1).is_ok() && d_dy.resize(1).is_ok() && d_dz.resize(1).is_ok() &&
              d_t.resize(1).is_ok() && d_nx.resize(1).is_ok() &&
              d_ny.resize(1).is_ok() && d_nz.resize(1).is_ok();
    if (!ok)
        GTEST_SKIP() << "CUDA unavailable";

    gwn::gwn_status const s =
        gwn::gwn_compute_harnack_trace_angle_batch_bvh_taylor<0, Real, Index>(
            geometry, bvh, aabb, moment,
            d_ox.span(), d_oy.span(), d_oz.span(),
            d_dx.span(), d_dy.span(), d_dz.span(),
            d_t.span(), d_nx.span(), d_ny.span(), d_nz.span()
        );
    EXPECT_EQ(s.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, edge_distance_batch_rejects_invalid_accessors) {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    bool ok = d_qx.resize(1).is_ok() && d_qy.resize(1).is_ok() &&
              d_qz.resize(1).is_ok() && d_out.resize(1).is_ok();
    if (!ok)
        GTEST_SKIP() << "CUDA unavailable";

    gwn::gwn_status const s = gwn::gwn_compute_unsigned_edge_distance_batch_bvh<Real, Index>(
        geometry, bvh, aabb, d_qx.span(), d_qy.span(), d_qz.span(), d_out.span()
    );
    EXPECT_EQ(s.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, edge_distance_matches_cpu_reference) {
    OctahedronMesh mesh;

    std::vector<Real> qx{Real(0), Real(0.15), Real(2.0), Real(-0.4)};
    std::vector<Real> qy{Real(0), Real(-0.25), Real(0.5), Real(0.8)};
    std::vector<Real> qz{Real(0), Real(0.6), Real(-1.2), Real(0.1)};
    std::vector<Real> d_gpu;

    if (!run_edge_distance_query(mesh, qx, qy, qz, d_gpu))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(d_gpu.size(), qx.size());
    for (std::size_t i = 0; i < qx.size(); ++i) {
        Real const d_cpu = cpu_edge_distance(mesh, qx[i], qy[i], qz[i]);
        EXPECT_NEAR(d_gpu[i], d_cpu, Real(2e-4))
            << "query " << i << " edge distance mismatch";
    }
}

TEST_F(CudaFixture, harnack_angle_half_octahedron_hits) {
    HalfOctahedronMesh mesh;

    std::vector<Real> ox{-0.4f, 0.0f, 0.4f, -0.3f, 0.1f, 0.3f};
    std::vector<Real> oy{-0.3f, 0.2f, 0.1f,  0.4f, -0.2f, 0.0f};
    std::vector<Real> oz{2.0f, 2.0f, 2.0f, 1.8f, 1.8f, 1.8f};
    std::vector<Real> dx{0, 0, 0, 0, 0, 0};
    std::vector<Real> dy{0, 0, 0, 0, 0, 0};
    std::vector<Real> dz{-1, -1, -1, -1, -1, -1};

    std::vector<Real> ht, hnx, hny, hnz;
    if (!run_harnack_trace_angle<1>(mesh, ox, oy, oz, dx, dy, dz, ht, hnx, hny, hnz))
        GTEST_SKIP() << "CUDA unavailable";

    int hits = 0;
    int behind_face_hits = 0;
    std::vector<Real> qx, qy, qz;
    std::vector<Real> t_face_hits;
    std::vector<Real> t_hit_values;
    for (std::size_t i = 0; i < ht.size(); ++i) {
        if (ht[i] < Real(0))
            continue;
        ++hits;

        Real const z_face = half_octahedron_face_z(ox[i], oy[i]);
        ASSERT_GT(z_face, Real(0));
        Real const t_face = (z_face - oz[i]) / dz[i];
        EXPECT_GE(ht[i], t_face - Real(2e-3))
            << "ray " << i << ": angle-mode hit should not lie meaningfully before face crossing";
        if (ht[i] > t_face + Real(1e-3))
            ++behind_face_hits;

        qx.push_back(ox[i] + ht[i] * dx[i]);
        qy.push_back(oy[i] + ht[i] * dy[i]);
        qz.push_back(oz[i] + ht[i] * dz[i]);
        t_face_hits.push_back(t_face);
        t_hit_values.push_back(ht[i]);
    }
    EXPECT_GE(hits, 5) << "angle-mode tracer should hit most open-mesh test rays";
    // With face-distance in R(x), the tracer converges tightly near the face
    // crossing rather than overshooting past it.  The important check is the
    // winding-number residual below, not how far past the face the hit lands.

    std::vector<Real> hw;
    if (!run_winding_query<1>(mesh, qx, qy, qz, hw))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(hw.size(), qx.size());
    ASSERT_EQ(t_face_hits.size(), qx.size());
    ASSERT_EQ(t_hit_values.size(), qx.size());
    for (std::size_t i = 0; i < hw.size(); ++i) {
        // Very near the face crossing, the wrapped value can sit on the branch
        // cut; check residual only for hits that are clearly behind the face.
        if (!(t_hit_values[i] > t_face_hits[i] + Real(1e-3)))
            continue;
        Real const residual = wrapped_angle_residual(hw[i], Real(0.5));
        EXPECT_LT(residual, Real(0.15))
            << "angle residual too large at hit " << i << " (residual=" << residual << ")";
    }
}

// ---------------------------------------------------------------------------
// Cube geometry tests
// ---------------------------------------------------------------------------

TEST_F(CudaFixture, harnack_cube_closed_no_singular_edges) {
    CubeMesh mesh;

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
        GTEST_SKIP() << "CUDA unavailable";
    ASSERT_TRUE(s.is_ok());

    EXPECT_EQ(geometry.singular_edge_count(), 0u)
        << "closed cube should have zero singular edges";
}

TEST_F(CudaFixture, harnack_open_cube_has_singular_edges) {
    OpenCubeMesh mesh;

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
        GTEST_SKIP() << "CUDA unavailable";
    ASSERT_TRUE(s.is_ok());

    EXPECT_EQ(geometry.singular_edge_count(), 4u)
        << "open cube (missing z=+1 face) should have 4 singular edges";
}

TEST_F(CudaFixture, harnack_cube_closed_hits) {
    CubeMesh mesh;

    // Axis-aligned rays from outside, pointing inward.
    std::vector<Real> ox{Real(0), Real(5), Real(0)};
    std::vector<Real> oy{Real(0), Real(0), Real(5)};
    std::vector<Real> oz{Real(5), Real(0), Real(0)};
    std::vector<Real> dx{Real(0), Real(-1), Real(0)};
    std::vector<Real> dy{Real(0), Real(0), Real(-1)};
    std::vector<Real> dz{Real(-1), Real(0), Real(0)};

    std::vector<Real> ht, hnx, hny, hnz;
    if (!run_harnack_trace<1>(mesh, ox, oy, oz, dx, dy, dz, ht, hnx, hny, hnz))
        GTEST_SKIP() << "CUDA unavailable";

    int hits = 0;
    for (std::size_t i = 0; i < ox.size(); ++i) {
        if (ht[i] < Real(0))
            continue;
        ++hits;
        Real const hx = ox[i] + ht[i] * dx[i];
        Real const hy = oy[i] + ht[i] * dy[i];
        Real const hz = oz[i] + ht[i] * dz[i];
        // Hit point should be near the cube surface (max coord ≈ 1).
        Real const max_coord = std::max({std::abs(hx), std::abs(hy), std::abs(hz)});
        EXPECT_NEAR(max_coord, Real(1), Real(0.15))
            << "ray " << i << ": hit not near cube surface";
    }
    EXPECT_GE(hits, 1) << "closed cube should produce at least one hit";
}

TEST_F(CudaFixture, harnack_open_cube_hits) {
    OpenCubeMesh mesh;

    // Ray from +z through the open face, should hit the opposite z=-1 face.
    // Ray from -z toward the closed face.
    std::vector<Real> ox{Real(0), Real(0)};
    std::vector<Real> oy{Real(0), Real(0)};
    std::vector<Real> oz{Real(5), Real(-5)};
    std::vector<Real> dx{Real(0), Real(0)};
    std::vector<Real> dy{Real(0), Real(0)};
    std::vector<Real> dz{Real(-1), Real(1)};

    std::vector<Real> ht, hnx, hny, hnz;
    if (!run_harnack_trace<1>(mesh, ox, oy, oz, dx, dy, dz, ht, hnx, hny, hnz))
        GTEST_SKIP() << "CUDA unavailable";

    int hits = 0;
    for (std::size_t i = 0; i < ox.size(); ++i) {
        if (ht[i] >= Real(0))
            ++hits;
    }
    EXPECT_GE(hits, 1) << "open cube should produce at least one hit";
}

TEST_F(CudaFixture, harnack_open_cube_winding_convergence) {
    OpenCubeMesh mesh;

    // Rays from +z through the open face toward the interior.
    // The winding number transitions smoothly through 0.5 inside the open cube.
    std::vector<Real> ox{Real(0), Real(0.3)};
    std::vector<Real> oy{Real(0), Real(0.2)};
    std::vector<Real> oz{Real(5), Real(5)};
    std::vector<Real> dx{Real(0), Real(0)};
    std::vector<Real> dy{Real(0), Real(0)};
    std::vector<Real> dz{Real(-1), Real(-1)};

    std::vector<Real> ht, hnx, hny, hnz;
    if (!run_harnack_trace_angle<1>(mesh, ox, oy, oz, dx, dy, dz, ht, hnx, hny, hnz))
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> qx, qy, qz;
    for (std::size_t i = 0; i < ox.size(); ++i) {
        if (ht[i] < Real(0))
            continue;
        qx.push_back(ox[i] + ht[i] * dx[i]);
        qy.push_back(oy[i] + ht[i] * dy[i]);
        qz.push_back(oz[i] + ht[i] * dz[i]);
    }
    if (qx.empty())
        GTEST_SKIP() << "no hits to verify winding convergence";

    std::vector<Real> w;
    if (!run_winding_query<1>(mesh, qx, qy, qz, w))
        GTEST_SKIP() << "CUDA unavailable";

    for (std::size_t i = 0; i < w.size(); ++i) {
        Real const residual = wrapped_angle_residual(w[i], Real(0.5));
        EXPECT_LT(residual, Real(0.5))
            << "open cube winding residual too large at hit " << i
            << " (w=" << w[i] << ", residual=" << residual << ")";
    }
}
