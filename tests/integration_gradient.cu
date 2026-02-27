// Integration test: BVH-accelerated gradient vs CPU finite-difference reference.
//
// Uses the upper half-octahedron as the built-in mesh (open, asymmetric mesh:
// the area-weighted normal sum ≠ 0 so both Taylor orders give non-trivial
// gradients).  A 5×5×5 lattice grid surrounds the mesh.
// The CPU finite-difference gradient is the reference; the GPU Taylor gradient
// (Order 0 and 1) is the system under test.
//
// Key correctness properties verified:
//   1. Order-0 max component error < threshold.
//   2. Order-1 max component error < threshold.
//   3. Order-1 is at least as accurate as Order-0 (max error monotone).
//   4. At least 20% of the query points have |FD gradient| > 0.01, confirming
//      the test is not a false positive due to geometric symmetry cancellation.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "reference_cpu.cuh"
#include "test_fixtures.hpp"
#include "test_utils.hpp"

namespace {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

// ---------------------------------------------------------------------------
// Half-octahedron mesh — 5 vertices, 4 triangles, upper faces only.
// ASYMMETRIC: area-weighted normal sum N ≠ 0 (points in +z), so both
// Order-0 and Order-1 Taylor produce non-trivial gradients everywhere.
// ---------------------------------------------------------------------------
struct HalfOctaMesh {
    static constexpr std::size_t k_nv = 5;
    static constexpr std::size_t k_nt = 4;
    std::array<Real, k_nv> vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f};
    std::array<Real, k_nv> vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f};
    std::array<Real, k_nv> vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    std::array<Index, k_nt> i0{0, 2, 1, 3};
    std::array<Index, k_nt> i1{2, 1, 3, 0};
    std::array<Index, k_nt> i2{4, 4, 4, 4};
};

// ---------------------------------------------------------------------------
// CPU finite-difference gradient of the winding number (6-point stencil).
// ---------------------------------------------------------------------------
void fd_gradient(
    std::span<Real const> vx, std::span<Real const> vy, std::span<Real const> vz,
    std::span<Index const> i0, std::span<Index const> i1, std::span<Index const> i2,
    Real const qx, Real const qy, Real const qz, Real const h, Real &gx, Real &gy, Real &gz
) {
    auto w = [&](Real x, Real y, Real z) {
        return gwn::tests::reference_winding_number_point<Real, Index>(
            vx, vy, vz, i0, i1, i2, x, y, z
        );
    };
    gx = (w(qx + h, qy, qz) - w(qx - h, qy, qz)) / (Real(2) * h);
    gy = (w(qx, qy + h, qz) - w(qx, qy - h, qz)) / (Real(2) * h);
    gz = (w(qx, qy, qz + h) - w(qx, qy, qz - h)) / (Real(2) * h);
}

// ---------------------------------------------------------------------------
// Build BVH and run GPU gradient query for a given Order.
// Returns false if CUDA is unavailable.
// ---------------------------------------------------------------------------
template <int Order, class Mesh>
bool run_gpu_gradient(
    Mesh const &mesh, std::vector<Real> const &qx, std::vector<Real> const &qy,
    std::vector<Real> const &qz, std::vector<Real> &gx_out, std::vector<Real> &gy_out,
    std::vector<Real> &gz_out, Real const accuracy_scale
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

    std::size_t const n = qx.size();
    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_gx, d_gy, d_gz;
    bool ok = d_qx.resize(n).is_ok() && d_qy.resize(n).is_ok() && d_qz.resize(n).is_ok() &&
              d_gx.resize(n).is_ok() && d_gy.resize(n).is_ok() && d_gz.resize(n).is_ok();
    ok = ok &&
         d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), n)).is_ok() &&
         d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), n)).is_ok() &&
         d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device allocation/upload failed";
        return false;
    }

    s = gwn::gwn_compute_winding_gradient_batch_bvh_taylor<Order, Real, Index>(
        geometry.accessor(), bvh.accessor(), data.accessor(), d_qx.span(), d_qy.span(),
        d_qz.span(), d_gx.span(), d_gy.span(), d_gz.span(), accuracy_scale
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "gradient query: " << gwn::tests::status_to_debug_string(s);
        return false;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gx_out.resize(n);
    gy_out.resize(n);
    gz_out.resize(n);
    ok = d_gx.copy_to_host(cuda::std::span<Real>(gx_out.data(), n)).is_ok() &&
         d_gy.copy_to_host(cuda::std::span<Real>(gy_out.data(), n)).is_ok() &&
         d_gz.copy_to_host(cuda::std::span<Real>(gz_out.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device-to-host copy failed";
        return false;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return true;
}

// ---------------------------------------------------------------------------
// Compute per-component max error.
// ---------------------------------------------------------------------------
Real max_error_3d(
    std::vector<Real> const &ax, std::vector<Real> const &ay, std::vector<Real> const &az,
    std::vector<Real> const &bx, std::vector<Real> const &by, std::vector<Real> const &bz
) {
    Real err = 0.0f;
    for (std::size_t i = 0; i < ax.size(); ++i) {
        err = std::max(err, std::abs(ax[i] - bx[i]));
        err = std::max(err, std::abs(ay[i] - by[i]));
        err = std::max(err, std::abs(az[i] - bz[i]));
    }
    return err;
}

} // namespace

// ---------------------------------------------------------------------------
// Main integration test: 5×5×5 lattice grid around the half-octahedron.
// ---------------------------------------------------------------------------
TEST_F(CudaFixture, integration_gradient_half_octahedron_order0_and_order1) {
    constexpr Real k_fd_h = 1e-3f;           // FD step (float: avoid cancellation)
    constexpr Real k_accuracy_scale = Real(2);
    constexpr Real k_order0_max_err = 5e-2f;
    constexpr Real k_order1_max_err = 1e-2f;
    constexpr Real k_nontrivial_mag = 1e-2f;  // gradient magnitude threshold
    constexpr Real k_nontrivial_frac = 0.20f; // at least 20% non-trivial

    HalfOctaMesh mesh;
    std::span<Real const> spvx{mesh.vx.data(), mesh.vx.size()};
    std::span<Real const> spvy{mesh.vy.data(), mesh.vy.size()};
    std::span<Real const> spvz{mesh.vz.data(), mesh.vz.size()};
    std::span<Index const> spi0{mesh.i0.data(), mesh.i0.size()};
    std::span<Index const> spi1{mesh.i1.data(), mesh.i1.size()};
    std::span<Index const> spi2{mesh.i2.data(), mesh.i2.size()};

    // Build 5×5×5 lattice from -1.3 to +1.3.
    // The half-octahedron occupies x,y ∈ [-1,1], z ∈ [0,1], so the grid
    // covers points both inside the "dome" and outside/below it.
    constexpr int k_grid = 5;
    constexpr Real k_lo = -1.3f;
    constexpr Real k_hi = 1.3f;
    constexpr Real k_step = (k_hi - k_lo) / static_cast<Real>(k_grid - 1);

    std::vector<Real> qx, qy, qz;
    for (int ix = 0; ix < k_grid; ++ix) {
        for (int iy = 0; iy < k_grid; ++iy) {
            for (int iz = 0; iz < k_grid; ++iz) {
                Real const x = k_lo + static_cast<Real>(ix) * k_step;
                Real const y = k_lo + static_cast<Real>(iy) * k_step;
                Real const z = k_lo + static_cast<Real>(iz) * k_step;
                qx.push_back(x);
                qy.push_back(y);
                qz.push_back(z);
            }
        }
    }
    std::size_t const n = qx.size();
    ASSERT_GT(n, std::size_t(10)) << "Too few query points generated";

    // CPU finite-difference reference.
    std::vector<Real> ref_gx(n), ref_gy(n), ref_gz(n);
    int nontrivial_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        fd_gradient(spvx, spvy, spvz, spi0, spi1, spi2,
                    qx[i], qy[i], qz[i], k_fd_h,
                    ref_gx[i], ref_gy[i], ref_gz[i]);
        Real const mag = std::sqrt(ref_gx[i] * ref_gx[i] + ref_gy[i] * ref_gy[i] +
                                   ref_gz[i] * ref_gz[i]);
        if (mag > k_nontrivial_mag)
            ++nontrivial_count;
    }

    // Guard: the open half-octahedron has non-trivial gradient everywhere
    // (unlike the symmetric closed octahedron which has ∇w=0 inside).
    double const nontrivial_frac = static_cast<double>(nontrivial_count) / static_cast<double>(n);
    ASSERT_GE(nontrivial_frac, static_cast<double>(k_nontrivial_frac))
        << "Only " << nontrivial_count << "/" << n
        << " query points have |FD gradient| > " << k_nontrivial_mag
        << " — test may be a false positive due to geometric symmetry";

    // --- GPU Order-0 ---
    std::vector<Real> gpu0_gx, gpu0_gy, gpu0_gz;
    if (!run_gpu_gradient<0>(mesh, qx, qy, qz, gpu0_gx, gpu0_gy, gpu0_gz, k_accuracy_scale))
        GTEST_SKIP() << "CUDA unavailable";

    Real const err0 = max_error_3d(gpu0_gx, gpu0_gy, gpu0_gz, ref_gx, ref_gy, ref_gz);
    std::cout << "[gradient integration] Order-0 max component error: " << err0 << "\n";
    EXPECT_LE(err0, k_order0_max_err) << "Order-0 gradient max error exceeds threshold";

    // --- GPU Order-1 ---
    std::vector<Real> gpu1_gx, gpu1_gy, gpu1_gz;
    if (!run_gpu_gradient<1>(mesh, qx, qy, qz, gpu1_gx, gpu1_gy, gpu1_gz, k_accuracy_scale))
        GTEST_SKIP() << "CUDA unavailable";

    Real const err1 = max_error_3d(gpu1_gx, gpu1_gy, gpu1_gz, ref_gx, ref_gy, ref_gz);
    std::cout << "[gradient integration] Order-1 max component error: " << err1 << "\n";
    EXPECT_LE(err1, k_order1_max_err) << "Order-1 gradient max error exceeds threshold";

    // Order-1 must be at least as accurate as Order-0 (within float noise).
    EXPECT_LE(err1, err0 + 1e-5f)
        << "Order-1 (" << err1 << ") is less accurate than Order-0 (" << err0 << ")";

    // --- GPU Order-2 ---
    std::vector<Real> gpu2_gx, gpu2_gy, gpu2_gz;
    if (!run_gpu_gradient<2>(mesh, qx, qy, qz, gpu2_gx, gpu2_gy, gpu2_gz, k_accuracy_scale))
        GTEST_SKIP() << "CUDA unavailable";

    constexpr Real k_order2_max_err = 5e-3f;
    Real const err2 = max_error_3d(gpu2_gx, gpu2_gy, gpu2_gz, ref_gx, ref_gy, ref_gz);
    std::cout << "[gradient integration] Order-2 max component error: " << err2 << "\n";
    EXPECT_LE(err2, k_order2_max_err) << "Order-2 gradient max error exceeds threshold";

    // Order-2 must be at least as accurate as Order-1.
    EXPECT_LE(err2, err1 + 1e-5f)
        << "Order-2 (" << err2 << ") is less accurate than Order-1 (" << err1 << ")";
}
