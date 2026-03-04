#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_harnack_meshes.hpp"
#include "test_utils.hpp"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CubeMesh;
using gwn::tests::CudaFixture;
using gwn::tests::OpenCubeMesh;

namespace {

struct TetraMesh {
    static constexpr std::size_t Nv = 4;
    static constexpr std::size_t Nt = 4;
    std::array<Real, 4> vx{Real(0), Real(1), Real(0), Real(0)};
    std::array<Real, 4> vy{Real(0), Real(0), Real(1), Real(0)};
    std::array<Real, 4> vz{Real(0), Real(0), Real(0), Real(1)};

    // Outward orientation.
    std::array<Index, 4> i0{0, 0, 0, 1};
    std::array<Index, 4> i1{2, 1, 3, 2};
    std::array<Index, 4> i2{1, 3, 2, 3};
};

struct DoubleSidedSingleTriangleMesh {
    std::array<Real, 3> vx{Real(0), Real(1), Real(0)};
    std::array<Real, 3> vy{Real(0), Real(0), Real(1)};
    std::array<Real, 3> vz{Real(0), Real(0), Real(0)};

    // Two opposite-winding copies of the same triangle.
    std::array<Index, 2> i0{0, 0};
    std::array<Index, 2> i1{1, 2};
    std::array<Index, 2> i2{2, 1};
};

template <int Order, class Mesh>
bool build_scene(
    Mesh const &mesh, gwn::gwn_geometry_object<Real, Index> &geometry,
    gwn::gwn_bvh4_topology_object<Real, Index> &bvh, gwn::gwn_bvh4_aabb_object<Real, Index> &aabb,
    gwn::gwn_bvh4_moment_object<Order, Real, Index> &moment
) {
    gwn::gwn_status s = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
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

    s = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<Order, 4, Real, Index>(
        geometry, bvh, aabb, moment
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "BVH build: " << gwn::tests::status_to_debug_string(s);
        return false;
    }

    return true;
}

template <class Mesh>
bool run_ray_first_hit_query(
    Mesh const &mesh, std::vector<Real> const &ox, std::vector<Real> const &oy,
    std::vector<Real> const &oz, std::vector<Real> const &dx, std::vector<Real> const &dy,
    std::vector<Real> const &dz, std::vector<Real> &out_t, std::vector<Index> &out_pi,
    Real const t_min = Real(0), Real const t_max = std::numeric_limits<Real>::infinity()
) {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status s = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
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
    s = gwn::gwn_bvh_facade_build_topology_aabb_lbvh<4, Real, Index>(geometry, bvh, aabb);
    if (!s.is_ok()) {
        ADD_FAILURE() << "BVH build: " << gwn::tests::status_to_debug_string(s);
        return false;
    }

    std::size_t const n = ox.size();
    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t;
    gwn::gwn_device_array<Index> d_pi;
    bool ok = d_ox.resize(n).is_ok() && d_oy.resize(n).is_ok() && d_oz.resize(n).is_ok() &&
              d_dx.resize(n).is_ok() && d_dy.resize(n).is_ok() && d_dz.resize(n).is_ok() &&
              d_t.resize(n).is_ok() && d_pi.resize(n).is_ok();
    ok = ok && d_ox.copy_from_host(cuda::std::span<Real const>(ox.data(), n)).is_ok() &&
         d_oy.copy_from_host(cuda::std::span<Real const>(oy.data(), n)).is_ok() &&
         d_oz.copy_from_host(cuda::std::span<Real const>(oz.data(), n)).is_ok() &&
         d_dx.copy_from_host(cuda::std::span<Real const>(dx.data(), n)).is_ok() &&
         d_dy.copy_from_host(cuda::std::span<Real const>(dy.data(), n)).is_ok() &&
         d_dz.copy_from_host(cuda::std::span<Real const>(dz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device allocation/upload failed";
        return false;
    }

    s = gwn::gwn_compute_ray_first_hit_batch_bvh<Real, Index>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), d_ox.span(), d_oy.span(), d_oz.span(),
        d_dx.span(), d_dy.span(), d_dz.span(), d_t.span(), d_pi.span(), t_min, t_max
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "ray first-hit query: " << gwn::tests::status_to_debug_string(s);
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "CUDA synchronize failed after ray query";
        return false;
    }

    out_t.resize(n);
    out_pi.resize(n);
    ok = d_t.copy_to_host(cuda::std::span<Real>(out_t.data(), n)).is_ok() &&
         d_pi.copy_to_host(cuda::std::span<Index>(out_pi.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device-to-host copy failed";
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "CUDA synchronize failed after ray copy";
        return false;
    }

    return true;
}

template <int Order> struct hybrid_single_api_functor {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};
    gwn::gwn_bvh4_moment_accessor<Order, Real, Index> moment{};

    cuda::std::span<Real const> ox{};
    cuda::std::span<Real const> oy{};
    cuda::std::span<Real const> oz{};
    cuda::std::span<Real const> dx{};
    cuda::std::span<Real const> dy{};
    cuda::std::span<Real const> dz{};

    cuda::std::span<Real> out_t{};
    cuda::std::span<Real> out_nx{};
    cuda::std::span<Real> out_ny{};
    cuda::std::span<Real> out_nz{};
    cuda::std::span<Real> out_winding{};
    cuda::std::span<Index> out_pi{};
    cuda::std::span<std::uint8_t> out_hit_kind{};
    cuda::std::span<int> out_iterations{};

    gwn::gwn_hybrid_trace_arguments<Real> args{};

    __device__ void operator()(std::size_t const i) const {
        auto const hit = gwn::gwn_hybrid_trace_ray_bvh_taylor<Order, 4, Real, Index>(
            geometry, bvh, aabb, moment, ox[i], oy[i], oz[i], dx[i], dy[i], dz[i], args
        );
        out_t[i] = hit.t;
        out_nx[i] = hit.normal_x;
        out_ny[i] = hit.normal_y;
        out_nz[i] = hit.normal_z;
        if (!out_winding.empty())
            out_winding[i] = hit.winding;
        out_pi[i] = hit.primitive_id;
        out_hit_kind[i] = static_cast<std::uint8_t>(hit.hit_kind);
        out_iterations[i] = hit.iterations;
    }
};

template <int Order, class Mesh>
bool run_hybrid_single_api(
    Mesh const &mesh, std::vector<Real> const &ox, std::vector<Real> const &oy,
    std::vector<Real> const &oz, std::vector<Real> const &dx, std::vector<Real> const &dy,
    std::vector<Real> const &dz, gwn::gwn_hybrid_trace_arguments<Real> const &args,
    std::vector<Real> &out_t, std::vector<Real> &out_nx, std::vector<Real> &out_ny,
    std::vector<Real> &out_nz, std::vector<Index> &out_pi, std::vector<std::uint8_t> &out_hit_kind,
    std::vector<int> &out_iterations, std::vector<Real> *out_winding = nullptr
) {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<Order, Real, Index> moment;
    if (!build_scene<Order>(mesh, geometry, bvh, aabb, moment))
        return false;

    std::size_t const n = ox.size();
    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
    gwn::gwn_device_array<Real> d_t, d_nx, d_ny, d_nz;
    gwn::gwn_device_array<Real> d_winding;
    gwn::gwn_device_array<Index> d_pi;
    gwn::gwn_device_array<std::uint8_t> d_hit_kind;
    gwn::gwn_device_array<int> d_iterations;

    bool ok = d_ox.resize(n).is_ok() && d_oy.resize(n).is_ok() && d_oz.resize(n).is_ok() &&
              d_dx.resize(n).is_ok() && d_dy.resize(n).is_ok() && d_dz.resize(n).is_ok() &&
              d_t.resize(n).is_ok() && d_nx.resize(n).is_ok() && d_ny.resize(n).is_ok() &&
              d_nz.resize(n).is_ok() && d_pi.resize(n).is_ok() && d_hit_kind.resize(n).is_ok() &&
              d_iterations.resize(n).is_ok();
    if (out_winding != nullptr)
        ok = ok && d_winding.resize(n).is_ok();
    ok = ok && d_ox.copy_from_host(cuda::std::span<Real const>(ox.data(), n)).is_ok() &&
         d_oy.copy_from_host(cuda::std::span<Real const>(oy.data(), n)).is_ok() &&
         d_oz.copy_from_host(cuda::std::span<Real const>(oz.data(), n)).is_ok() &&
         d_dx.copy_from_host(cuda::std::span<Real const>(dx.data(), n)).is_ok() &&
         d_dy.copy_from_host(cuda::std::span<Real const>(dy.data(), n)).is_ok() &&
         d_dz.copy_from_host(cuda::std::span<Real const>(dz.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device allocation/upload failed";
        return false;
    }

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    gwn::gwn_status s = gwn::detail::gwn_launch_linear_kernel<k_block_size>(
        n, hybrid_single_api_functor<Order>{
               geometry.accessor(),
               bvh.accessor(),
               aabb.accessor(),
               moment.accessor(),
               d_ox.span(),
               d_oy.span(),
               d_oz.span(),
               d_dx.span(),
               d_dy.span(),
               d_dz.span(),
               d_t.span(),
               d_nx.span(),
               d_ny.span(),
               d_nz.span(),
               (out_winding != nullptr) ? d_winding.span() : cuda::std::span<Real>{},
               d_pi.span(),
               d_hit_kind.span(),
               d_iterations.span(),
               args,
           }
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "hybrid single API: " << gwn::tests::status_to_debug_string(s);
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "CUDA synchronize failed after hybrid single API";
        return false;
    }

    out_t.resize(n);
    out_nx.resize(n);
    out_ny.resize(n);
    out_nz.resize(n);
    out_pi.resize(n);
    out_hit_kind.resize(n);
    out_iterations.resize(n);
    ok = d_t.copy_to_host(cuda::std::span<Real>(out_t.data(), n)).is_ok() &&
         d_nx.copy_to_host(cuda::std::span<Real>(out_nx.data(), n)).is_ok() &&
         d_ny.copy_to_host(cuda::std::span<Real>(out_ny.data(), n)).is_ok() &&
         d_nz.copy_to_host(cuda::std::span<Real>(out_nz.data(), n)).is_ok() &&
         d_pi.copy_to_host(cuda::std::span<Index>(out_pi.data(), n)).is_ok() &&
         d_hit_kind.copy_to_host(cuda::std::span<std::uint8_t>(out_hit_kind.data(), n)).is_ok() &&
         d_iterations.copy_to_host(cuda::std::span<int>(out_iterations.data(), n)).is_ok();
    if (out_winding != nullptr) {
        out_winding->resize(n);
        ok = ok && d_winding.copy_to_host(cuda::std::span<Real>(out_winding->data(), n)).is_ok();
    }
    if (!ok) {
        ADD_FAILURE() << "device-to-host copy failed";
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "CUDA synchronize failed after hybrid copy";
        return false;
    }

    return true;
}

inline Real norm3(Real const x, Real const y, Real const z) {
    return std::sqrt(x * x + y * y + z * z);
}

} // namespace

TEST_F(CudaFixture, hybrid_closed_cube_matches_ray_first_hit_and_skips_harnack_iterations) {
    CubeMesh mesh;

    std::vector<Real> ox{Real(0), Real(5), Real(0)};
    std::vector<Real> oy{Real(0), Real(0), Real(5)};
    std::vector<Real> oz{Real(5), Real(0), Real(0)};
    std::vector<Real> dx{Real(0), Real(-1), Real(0)};
    std::vector<Real> dy{Real(0), Real(0), Real(-1)};
    std::vector<Real> dz{Real(-1), Real(0), Real(0)};

    gwn::gwn_hybrid_trace_arguments<Real> args{};
    gwn::gwn_init_hybrid_trace_arguments(args);

    std::vector<Real> ht, hnx, hny, hnz;
    std::vector<Index> hpi;
    std::vector<std::uint8_t> hkind;
    std::vector<int> hiters;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, args, ht, hnx, hny, hnz, hpi, hkind, hiters
        ))
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> rt;
    std::vector<Index> rpi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, rt, rpi))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(ht.size(), rt.size());
    ASSERT_EQ(hpi.size(), rpi.size());
    for (std::size_t i = 0; i < ht.size(); ++i) {
        EXPECT_NEAR(ht[i], rt[i], Real(1e-5)) << "t mismatch at ray " << i;
        EXPECT_EQ(hpi[i], rpi[i]) << "primitive id mismatch at ray " << i;
        EXPECT_EQ(hkind[i], static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_triangle))
            << "closed mesh should resolve via triangle branch";
        EXPECT_EQ(hiters[i], 0) << "closed mesh should skip Harnack stage";

        Real const nrm = norm3(hnx[i], hny[i], hnz[i]);
        EXPECT_NEAR(nrm, Real(1), Real(1e-4));
    }
}

TEST_F(CudaFixture, hybrid_open_cube_prefers_harnack_when_fill_is_before_triangle) {
    OpenCubeMesh mesh;

    std::vector<Real> ox{Real(0)};
    std::vector<Real> oy{Real(0)};
    std::vector<Real> oz{Real(5)};
    std::vector<Real> dx{Real(0)};
    std::vector<Real> dy{Real(0)};
    std::vector<Real> dz{Real(-1)};

    gwn::gwn_hybrid_trace_arguments<Real> args{};
    gwn::gwn_init_hybrid_trace_arguments(args);
    args.epsilon = Real(1e-3);
    args.max_iterations = 2048;
    args.t_max = Real(100);

    std::vector<Real> ht, hnx, hny, hnz;
    std::vector<Index> hpi;
    std::vector<std::uint8_t> hkind;
    std::vector<int> hiters;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, args, ht, hnx, hny, hnz, hpi, hkind, hiters
        ))
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> rt;
    std::vector<Index> rpi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, rt, rpi))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(ht.size(), 1u);
    ASSERT_EQ(rt.size(), 1u);
    ASSERT_GE(rt[0], Real(0));
    ASSERT_GE(ht[0], Real(0));
    EXPECT_EQ(hkind[0], static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_harnack))
        << "expected fill branch for open-cube center ray";
    EXPECT_LT(ht[0], rt[0] - Real(1e-3)) << "fill hit should be before mesh triangle hit";
    EXPECT_NE(hiters[0], 0) << "open mesh fill branch should consume Harnack iterations";
}

TEST_F(CudaFixture, hybrid_open_cube_prefers_triangle_when_triangle_is_before_fill) {
    OpenCubeMesh mesh;

    std::vector<Real> ox{Real(5)};
    std::vector<Real> oy{Real(0)};
    std::vector<Real> oz{Real(0)};
    std::vector<Real> dx{Real(-1)};
    std::vector<Real> dy{Real(0)};
    std::vector<Real> dz{Real(0)};

    gwn::gwn_hybrid_trace_arguments<Real> args{};
    gwn::gwn_init_hybrid_trace_arguments(args);
    args.epsilon = Real(1e-3);
    args.max_iterations = 2048;
    args.t_max = Real(100);

    std::vector<Real> ht, hnx, hny, hnz;
    std::vector<Index> hpi;
    std::vector<std::uint8_t> hkind;
    std::vector<int> hiters;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, args, ht, hnx, hny, hnz, hpi, hkind, hiters
        ))
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> rt;
    std::vector<Index> rpi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, rt, rpi))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(ht.size(), 1u);
    ASSERT_EQ(rt.size(), 1u);
    ASSERT_GE(rt[0], Real(0));
    ASSERT_GE(ht[0], Real(0));
    EXPECT_EQ(hkind[0], static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_triangle))
        << "triangle should win when it is the first event";
    EXPECT_NEAR(ht[0], rt[0], Real(1e-4));
    EXPECT_EQ(hpi[0], rpi[0]);
}

TEST_F(CudaFixture, hybrid_fill_hit_normal_is_policy_independent_when_triangle_policy_changes) {
    OpenCubeMesh mesh;

    std::vector<Real> ox{
        Real(0),   Real(0.7),  Real(-0.7), Real(0.8),  Real(-0.8),
        Real(0.4), Real(-0.4), Real(0.0),  Real(0.25),
    };
    std::vector<Real> oy{
        Real(0),    Real(0.7), Real(0.7),  Real(-0.8),  Real(-0.8),
        Real(-0.4), Real(0.4), Real(0.85), Real(-0.25),
    };
    std::vector<Real> oz(ox.size(), Real(5));
    std::vector<Real> dx(ox.size(), Real(0));
    std::vector<Real> dy(ox.size(), Real(0));
    std::vector<Real> dz(ox.size(), Real(-1));

    gwn::gwn_hybrid_trace_arguments<Real> geometric_args{};
    gwn::gwn_init_hybrid_trace_arguments(geometric_args);
    geometric_args.epsilon = Real(1e-3);
    geometric_args.max_iterations = 2048;
    geometric_args.t_max = Real(100);
    geometric_args.triangle_normal_policy = gwn::gwn_hybrid_triangle_normal_policy::k_geometric;

    gwn::gwn_hybrid_trace_arguments<Real> barycentric_args{};
    gwn::gwn_init_hybrid_trace_arguments(barycentric_args);
    barycentric_args.epsilon = Real(1e-3);
    barycentric_args.max_iterations = 2048;
    barycentric_args.t_max = Real(100);
    barycentric_args.triangle_normal_policy =
        gwn::gwn_hybrid_triangle_normal_policy::k_barycentric_vertex;

    std::vector<Real> t_geo, nx_geo, ny_geo, nz_geo;
    std::vector<Index> pi_geo;
    std::vector<std::uint8_t> kind_geo;
    std::vector<int> iter_geo;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, geometric_args, t_geo, nx_geo, ny_geo, nz_geo, pi_geo,
            kind_geo, iter_geo
        )) {
        GTEST_SKIP() << "CUDA unavailable";
    }

    std::vector<Real> t_bary, nx_bary, ny_bary, nz_bary;
    std::vector<Index> pi_bary;
    std::vector<std::uint8_t> kind_bary;
    std::vector<int> iter_bary;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, barycentric_args, t_bary, nx_bary, ny_bary, nz_bary,
            pi_bary, kind_bary, iter_bary
        )) {
        GTEST_SKIP() << "CUDA unavailable";
    }

    ASSERT_EQ(t_geo.size(), ox.size());
    ASSERT_EQ(t_bary.size(), ox.size());

    std::size_t harnack_count = 0;
    for (std::size_t i = 0; i < ox.size(); ++i) {
        if (kind_geo[i] != static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_harnack))
            continue;
        ASSERT_EQ(kind_bary[i], static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_harnack));
        ++harnack_count;

        EXPECT_NEAR(t_geo[i], t_bary[i], Real(1e-4));
        EXPECT_EQ(iter_geo[i], iter_bary[i]);
        EXPECT_EQ(pi_geo[i], pi_bary[i]);
        EXPECT_NEAR(nx_geo[i], nx_bary[i], Real(1e-4));
        EXPECT_NEAR(ny_geo[i], ny_bary[i], Real(1e-4));
        EXPECT_NEAR(nz_geo[i], nz_bary[i], Real(1e-4));
    }

    EXPECT_GT(harnack_count, 0u) << "test setup should include at least one fill hit";
}

TEST_F(CudaFixture, hybrid_respects_t_min_and_hits_second_surface_on_closed_cube) {
    CubeMesh mesh;

    std::vector<Real> ox{Real(0)};
    std::vector<Real> oy{Real(0)};
    std::vector<Real> oz{Real(5)};
    std::vector<Real> dx{Real(0)};
    std::vector<Real> dy{Real(0)};
    std::vector<Real> dz{Real(-1)};

    gwn::gwn_hybrid_trace_arguments<Real> args{};
    gwn::gwn_init_hybrid_trace_arguments(args);
    args.t_min = Real(4.5);
    args.t_max = Real(100);

    std::vector<Real> ht, hnx, hny, hnz;
    std::vector<Index> hpi;
    std::vector<std::uint8_t> hkind;
    std::vector<int> hiters;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, args, ht, hnx, hny, hnz, hpi, hkind, hiters
        ))
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> rt;
    std::vector<Index> rpi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, rt, rpi, args.t_min, args.t_max))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(ht.size(), 1u);
    ASSERT_EQ(rt.size(), 1u);
    EXPECT_NEAR(ht[0], Real(6), Real(1e-4));
    EXPECT_NEAR(ht[0], rt[0], Real(1e-4));
    EXPECT_EQ(hpi[0], rpi[0]);
    EXPECT_EQ(hkind[0], static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_triangle));
    EXPECT_EQ(hiters[0], 0);
}

TEST_F(CudaFixture, hybrid_barycentric_vertex_normal_policy_differs_from_geometric_on_closed_mesh) {
    TetraMesh mesh;

    std::vector<Real> ox{Real(0.2)};
    std::vector<Real> oy{Real(0.2)};
    std::vector<Real> oz{Real(-1)};
    std::vector<Real> dx{Real(0)};
    std::vector<Real> dy{Real(0)};
    std::vector<Real> dz{Real(1)};

    gwn::gwn_hybrid_trace_arguments<Real> geometric_args{};
    gwn::gwn_init_hybrid_trace_arguments(geometric_args);
    geometric_args.triangle_normal_policy = gwn::gwn_hybrid_triangle_normal_policy::k_geometric;

    gwn::gwn_hybrid_trace_arguments<Real> bary_args{};
    gwn::gwn_init_hybrid_trace_arguments(bary_args);
    bary_args.triangle_normal_policy = gwn::gwn_hybrid_triangle_normal_policy::k_barycentric_vertex;

    std::vector<Real> t_geo, nx_geo, ny_geo, nz_geo;
    std::vector<Index> pi_geo;
    std::vector<std::uint8_t> kind_geo;
    std::vector<int> iter_geo;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, geometric_args, t_geo, nx_geo, ny_geo, nz_geo, pi_geo,
            kind_geo, iter_geo
        )) {
        GTEST_SKIP() << "CUDA unavailable";
    }

    std::vector<Real> t_bary, nx_bary, ny_bary, nz_bary;
    std::vector<Index> pi_bary;
    std::vector<std::uint8_t> kind_bary;
    std::vector<int> iter_bary;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, bary_args, t_bary, nx_bary, ny_bary, nz_bary, pi_bary,
            kind_bary, iter_bary
        )) {
        GTEST_SKIP() << "CUDA unavailable";
    }

    ASSERT_EQ(t_geo.size(), 1u);
    ASSERT_EQ(t_bary.size(), 1u);
    ASSERT_GE(t_geo[0], Real(0));
    ASSERT_GE(t_bary[0], Real(0));
    EXPECT_EQ(kind_geo[0], static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_triangle));
    EXPECT_EQ(kind_bary[0], static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_triangle));

    Real const n_geo = norm3(nx_geo[0], ny_geo[0], nz_geo[0]);
    Real const n_bary = norm3(nx_bary[0], ny_bary[0], nz_bary[0]);
    EXPECT_NEAR(n_geo, Real(1), Real(1e-4));
    EXPECT_NEAR(n_bary, Real(1), Real(1e-4));

    Real const dot = nx_geo[0] * nx_bary[0] + ny_geo[0] * ny_bary[0] + nz_geo[0] * nz_bary[0];
    EXPECT_LT(
        dot, Real(0.98)
    ) << "barycentric vertex normal should differ on non-planar closed mesh";
}

TEST_F(CudaFixture, hybrid_triangle_branch_does_not_populate_winding) {
    CubeMesh mesh;

    std::vector<Real> ox{Real(0)};
    std::vector<Real> oy{Real(0)};
    std::vector<Real> oz{Real(5)};
    std::vector<Real> dx{Real(0)};
    std::vector<Real> dy{Real(0)};
    std::vector<Real> dz{Real(-1)};

    gwn::gwn_hybrid_trace_arguments<Real> args{};
    gwn::gwn_init_hybrid_trace_arguments(args);
    args.triangle_normal_policy = gwn::gwn_hybrid_triangle_normal_policy::k_geometric;

    std::vector<Real> t, nx, ny, nz, winding;
    std::vector<Index> pi;
    std::vector<std::uint8_t> kind;
    std::vector<int> iterations;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, args, t, nx, ny, nz, pi, kind, iterations, &winding
        )) {
        GTEST_SKIP() << "CUDA unavailable";
    }

    ASSERT_EQ(t.size(), 1u);
    ASSERT_EQ(kind.size(), 1u);
    ASSERT_EQ(winding.size(), 1u);
    EXPECT_EQ(kind[0], static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_triangle));
    EXPECT_NEAR(winding[0], Real(0), Real(1e-8));
}

TEST_F(
    CudaFixture,
    hybrid_barycentric_policy_returns_zero_normal_when_vertex_normal_interpolation_is_degenerate
) {
    DoubleSidedSingleTriangleMesh mesh;

    std::vector<Real> ox{Real(0.2)};
    std::vector<Real> oy{Real(0.2)};
    std::vector<Real> oz{Real(1)};
    std::vector<Real> dx{Real(0)};
    std::vector<Real> dy{Real(0)};
    std::vector<Real> dz{Real(-1)};

    gwn::gwn_hybrid_trace_arguments<Real> geometric_args{};
    gwn::gwn_init_hybrid_trace_arguments(geometric_args);
    geometric_args.triangle_normal_policy = gwn::gwn_hybrid_triangle_normal_policy::k_geometric;

    gwn::gwn_hybrid_trace_arguments<Real> barycentric_args{};
    gwn::gwn_init_hybrid_trace_arguments(barycentric_args);
    barycentric_args.triangle_normal_policy =
        gwn::gwn_hybrid_triangle_normal_policy::k_barycentric_vertex;

    std::vector<Real> t_geo, nx_geo, ny_geo, nz_geo;
    std::vector<Index> pi_geo;
    std::vector<std::uint8_t> kind_geo;
    std::vector<int> iter_geo;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, geometric_args, t_geo, nx_geo, ny_geo, nz_geo, pi_geo,
            kind_geo, iter_geo
        )) {
        GTEST_SKIP() << "CUDA unavailable";
    }

    std::vector<Real> t_bary, nx_bary, ny_bary, nz_bary;
    std::vector<Index> pi_bary;
    std::vector<std::uint8_t> kind_bary;
    std::vector<int> iter_bary;
    if (!run_hybrid_single_api<1>(
            mesh, ox, oy, oz, dx, dy, dz, barycentric_args, t_bary, nx_bary, ny_bary, nz_bary,
            pi_bary, kind_bary, iter_bary
        )) {
        GTEST_SKIP() << "CUDA unavailable";
    }

    ASSERT_EQ(t_geo.size(), 1u);
    ASSERT_EQ(t_bary.size(), 1u);
    EXPECT_EQ(kind_geo[0], static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_triangle));
    EXPECT_EQ(kind_bary[0], static_cast<std::uint8_t>(gwn::gwn_hybrid_hit_kind::k_triangle));
    EXPECT_GE(t_geo[0], Real(0));
    EXPECT_GE(t_bary[0], Real(0));

    Real const geo_norm = norm3(nx_geo[0], ny_geo[0], nz_geo[0]);
    Real const bary_norm = norm3(nx_bary[0], ny_bary[0], nz_bary[0]);
    EXPECT_NEAR(geo_norm, Real(1), Real(1e-4));
    EXPECT_NEAR(bary_norm, Real(0), Real(1e-7));
}
