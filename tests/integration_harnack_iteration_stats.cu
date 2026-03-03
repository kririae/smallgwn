/// @file integration_harnack_iteration_stats.cu
/// Measures per-ray iteration counts for closed vs open octahedron to quantify
/// R-bound conservativeness.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
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
using gwn::tests::generate_sphere_rays;

namespace {

constexpr int k_width = 4;
constexpr int k_stack = gwn::k_gwn_default_traversal_stack_capacity;
constexpr int k_block_size = 128;

struct IterStats {
    int hit{0};
    int iterations{0};
};

template <int Order>
__global__ void trace_collect_iters_kernel(
    gwn::gwn_geometry_accessor<Real, Index> geometry,
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh,
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb_tree,
    gwn::gwn_bvh4_moment_accessor<Order, Real, Index> moment_tree,
    Real const *ray_ox, Real const *ray_oy, Real const *ray_oz,
    Real const *ray_dx, Real const *ray_dy, Real const *ray_dz,
    std::size_t const ray_count,
    Real const target_winding, Real const epsilon, int const max_iterations,
    Real const t_max, Real const accuracy_scale,
    IterStats *out_stats
) {
    std::size_t const ray_id = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray_id >= ray_count)
        return;

    auto const res = gwn::detail::gwn_harnack_trace_ray_impl<
        Order, k_width, Real, Index, k_stack>(
        geometry, bvh, aabb_tree, moment_tree,
        ray_ox[ray_id], ray_oy[ray_id], ray_oz[ray_id],
        ray_dx[ray_id], ray_dy[ray_id], ray_dz[ray_id],
        target_winding, epsilon, max_iterations, t_max, accuracy_scale
    );

    IterStats stats{};
    stats.hit = res.hit() ? 1 : 0;
    stats.iterations = res.iterations;
    out_stats[ray_id] = stats;
}

template <int Order, class Mesh>
void measure_iteration_stats(
    Mesh const &mesh,
    std::vector<Real> const &ox, std::vector<Real> const &oy, std::vector<Real> const &oz,
    std::vector<Real> const &dx, std::vector<Real> const &dy, std::vector<Real> const &dz,
    Real const epsilon, int const max_iterations, Real const t_max, Real const accuracy_scale,
    char const *label
) {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status s = gwn::gwn_upload_geometry(
        geometry,
        cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
        cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
        cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
        cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
        cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
        cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
    );
    ASSERT_TRUE(s.is_ok()) << "geometry upload: " << gwn::tests::status_to_debug_string(s);

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Order, Real, Index> moment;
    s = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, k_width, Real, Index>(
        geometry, bvh, aabb, moment
    );
    ASSERT_TRUE(s.is_ok()) << "BVH build: " << gwn::tests::status_to_debug_string(s);

    std::size_t const n = ox.size();

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
    gwn::gwn_device_array<IterStats> d_out;
    bool ok = d_ox.resize(n).is_ok() && d_oy.resize(n).is_ok() && d_oz.resize(n).is_ok() &&
              d_dx.resize(n).is_ok() && d_dy.resize(n).is_ok() && d_dz.resize(n).is_ok() &&
              d_out.resize(n).is_ok();
    ok = ok &&
         d_ox.copy_from_host(cuda::std::span<Real const>(ox.data(), n)).is_ok() &&
         d_oy.copy_from_host(cuda::std::span<Real const>(oy.data(), n)).is_ok() &&
         d_oz.copy_from_host(cuda::std::span<Real const>(oz.data(), n)).is_ok() &&
         d_dx.copy_from_host(cuda::std::span<Real const>(dx.data(), n)).is_ok() &&
         d_dy.copy_from_host(cuda::std::span<Real const>(dy.data(), n)).is_ok() &&
         d_dz.copy_from_host(cuda::std::span<Real const>(dz.data(), n)).is_ok();
    ASSERT_TRUE(ok) << "device alloc/upload failed";

    dim3 block(k_block_size);
    dim3 grid(static_cast<unsigned int>((n + k_block_size - 1) / k_block_size));
    trace_collect_iters_kernel<Order><<<grid, block>>>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), moment.accessor(),
        d_ox.data(), d_oy.data(), d_oz.data(), d_dx.data(), d_dy.data(), d_dz.data(),
        n, Real(0.5), epsilon, max_iterations, Real(100), accuracy_scale,
        d_out.data()
    );
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<IterStats> host_out(n);
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<IterStats>(host_out.data(), n)).is_ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Collect stats for hits only
    std::vector<int> hit_iters;
    std::vector<int> miss_iters;
    for (auto const &st : host_out) {
        if (st.hit)
            hit_iters.push_back(st.iterations);
        else
            miss_iters.push_back(st.iterations);
    }

    auto print_stats = [](char const *tag, std::vector<int> &iters) {
        if (iters.empty()) {
            std::cout << "  " << tag << ": (none)\n";
            return;
        }
        std::sort(iters.begin(), iters.end());
        long long sum = 0;
        for (int v : iters) sum += v;
        double avg = static_cast<double>(sum) / static_cast<double>(iters.size());
        int min_v = iters.front();
        int max_v = iters.back();
        int median = iters[iters.size() / 2];
        int p90 = iters[static_cast<std::size_t>(static_cast<double>(iters.size()) * 0.9)];
        int p99 = iters[std::min(iters.size() - 1, static_cast<std::size_t>(static_cast<double>(iters.size()) * 0.99))];
        std::cout << "  " << tag << ": n=" << iters.size()
                  << " avg=" << avg
                  << " min=" << min_v
                  << " median=" << median
                  << " p90=" << p90
                  << " p99=" << p99
                  << " max=" << max_v << "\n";
    };

    std::cout << "\n=== " << label << " ===\n";
    std::cout << "  total rays=" << n
              << " hits=" << hit_iters.size()
              << " misses=" << miss_iters.size() << "\n";
    print_stats("hit_iters", hit_iters);
    print_stats("miss_iters", miss_iters);
}

} // namespace

TEST_F(CudaFixture, iteration_stats_closed_octahedron) {
    OctahedronMesh mesh;
    std::vector<Real> ox, oy, oz, dx, dy, dz;
    generate_sphere_rays(Real(5), 20, 40, ox, oy, oz, dx, dy, dz);

    measure_iteration_stats<1>(
        mesh, ox, oy, oz, dx, dy, dz,
        Real(1e-3), 2048, Real(100), Real(2),
        "Closed Octahedron (sphere rays R=5, 20x40)"
    );
}

TEST_F(CudaFixture, iteration_stats_half_octahedron) {
    HalfOctahedronMesh mesh;
    std::vector<Real> ox, oy, oz, dx, dy, dz;
    generate_sphere_rays(Real(5), 20, 40, ox, oy, oz, dx, dy, dz);

    measure_iteration_stats<1>(
        mesh, ox, oy, oz, dx, dy, dz,
        Real(1e-3), 2048, Real(100), Real(2),
        "Half Octahedron (sphere rays R=5, 20x40)"
    );
}

TEST_F(CudaFixture, iteration_stats_closed_cube) {
    CubeMesh mesh;
    std::vector<Real> ox, oy, oz, dx, dy, dz;
    generate_sphere_rays(Real(5), 20, 40, ox, oy, oz, dx, dy, dz);

    measure_iteration_stats<1>(
        mesh, ox, oy, oz, dx, dy, dz,
        Real(1e-3), 2048, Real(100), Real(2),
        "Closed Cube (sphere rays R=5, 20x40)"
    );
}

TEST_F(CudaFixture, iteration_stats_open_cube) {
    OpenCubeMesh mesh;
    std::vector<Real> ox, oy, oz, dx, dy, dz;
    generate_sphere_rays(Real(5), 20, 40, ox, oy, oz, dx, dy, dz);

    measure_iteration_stats<1>(
        mesh, ox, oy, oz, dx, dy, dz,
        Real(1e-3), 2048, Real(100), Real(2),
        "Open Cube (sphere rays R=5, 20x40)"
    );
}
