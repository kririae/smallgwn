#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "../benchmarks/benchmark_utils.hpp"
#include "test_fixtures.hpp"
#include "test_harnack_meshes.hpp"
#include "test_utils.hpp"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;
using gwn::tests::OctahedronMesh;

namespace {

struct SingleTriangleMesh {
    static constexpr std::size_t Nv = 3;
    static constexpr std::size_t Nt = 1;
    std::array<Real, 3> vx{Real(0), Real(1), Real(0)};
    std::array<Real, 3> vy{Real(0), Real(0), Real(1)};
    std::array<Real, 3> vz{Real(0), Real(0), Real(0)};
    std::array<Index, 1> i0{0};
    std::array<Index, 1> i1{1};
    std::array<Index, 1> i2{2};
};

struct TwoLayerTriangleMesh {
    static constexpr std::size_t Nv = 6;
    static constexpr std::size_t Nt = 2;
    std::array<Real, 6> vx{
        Real(0), Real(1), Real(0), // far (z=-2), primitive 0
        Real(0), Real(1), Real(0)  // near (z=0), primitive 1
    };
    std::array<Real, 6> vy{Real(0), Real(0), Real(1), Real(0), Real(0), Real(1)};
    std::array<Real, 6> vz{Real(-2), Real(-2), Real(-2), Real(0), Real(0), Real(0)};
    std::array<Index, 2> i0{0, 3};
    std::array<Index, 2> i1{1, 4};
    std::array<Index, 2> i2{2, 5};
};

struct SharedEdgeSquareMesh {
    static constexpr std::size_t Nv = 4;
    static constexpr std::size_t Nt = 2;
    std::array<Real, 4> vx{Real(0), Real(1), Real(1), Real(0)};
    std::array<Real, 4> vy{Real(0), Real(0), Real(1), Real(1)};
    std::array<Real, 4> vz{Real(0), Real(0), Real(0), Real(0)};
    std::array<Index, 2> i0{0, 0};
    std::array<Index, 2> i1{1, 2};
    std::array<Index, 2> i2{2, 3};
};

struct single_ray_api_functor {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};
    cuda::std::span<Real const> ox{};
    cuda::std::span<Real const> oy{};
    cuda::std::span<Real const> oz{};
    cuda::std::span<Real const> dx{};
    cuda::std::span<Real const> dy{};
    cuda::std::span<Real const> dz{};
    cuda::std::span<Real> out_t{};
    cuda::std::span<Index> out_pi{};

    __device__ void operator()(std::size_t const i) const {
        auto const hit = gwn::gwn_ray_first_hit_bvh<4, Real, Index>(
            geometry, bvh, aabb, ox[i], oy[i], oz[i], dx[i], dy[i], dz[i]
        );
        out_t[i] = hit.t;
        out_pi[i] = hit.primitive_id;
    }
};

struct single_ray_api_uv_functor {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};
    cuda::std::span<Real const> ox{};
    cuda::std::span<Real const> oy{};
    cuda::std::span<Real const> oz{};
    cuda::std::span<Real const> dx{};
    cuda::std::span<Real const> dy{};
    cuda::std::span<Real const> dz{};
    cuda::std::span<Real> out_t{};
    cuda::std::span<Index> out_pi{};
    cuda::std::span<Real> out_u{};
    cuda::std::span<Real> out_v{};

    __device__ void operator()(std::size_t const i) const {
        auto const hit = gwn::gwn_ray_first_hit_bvh<4, Real, Index>(
            geometry, bvh, aabb, ox[i], oy[i], oz[i], dx[i], dy[i], dz[i]
        );
        out_t[i] = hit.t;
        out_pi[i] = hit.primitive_id;
        out_u[i] = hit.u;
        out_v[i] = hit.v;
    }
};

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
        ADD_FAILURE() << "CUDA synchronize failed after ray first-hit query";
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
        ADD_FAILURE() << "CUDA synchronize failed after result copy";
        return false;
    }

    return true;
}

template <class Mesh>
bool run_ray_first_hit_query_single_api(
    Mesh const &mesh, std::vector<Real> const &ox, std::vector<Real> const &oy,
    std::vector<Real> const &oz, std::vector<Real> const &dx, std::vector<Real> const &dy,
    std::vector<Real> const &dz, std::vector<Real> &out_t, std::vector<Index> &out_pi
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

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    s = gwn::detail::gwn_launch_linear_kernel<k_block_size>(
        n, single_ray_api_functor{
               geometry.accessor(), bvh.accessor(), aabb.accessor(), d_ox.span(), d_oy.span(),
               d_oz.span(), d_dx.span(), d_dy.span(), d_dz.span(), d_t.span(), d_pi.span()
           }
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "ray first-hit single API query: "
                      << gwn::tests::status_to_debug_string(s);
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "CUDA synchronize failed after single API query";
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
        ADD_FAILURE() << "CUDA synchronize failed after result copy";
        return false;
    }

    return true;
}

template <class Mesh>
bool run_ray_first_hit_query_single_api_with_uv(
    Mesh const &mesh, std::vector<Real> const &ox, std::vector<Real> const &oy,
    std::vector<Real> const &oz, std::vector<Real> const &dx, std::vector<Real> const &dy,
    std::vector<Real> const &dz, std::vector<Real> &out_t, std::vector<Index> &out_pi,
    std::vector<Real> &out_u, std::vector<Real> &out_v
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
    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t, d_u, d_v;
    gwn::gwn_device_array<Index> d_pi;
    bool ok = d_ox.resize(n).is_ok() && d_oy.resize(n).is_ok() && d_oz.resize(n).is_ok() &&
              d_dx.resize(n).is_ok() && d_dy.resize(n).is_ok() && d_dz.resize(n).is_ok() &&
              d_t.resize(n).is_ok() && d_pi.resize(n).is_ok() && d_u.resize(n).is_ok() &&
              d_v.resize(n).is_ok();
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
    s = gwn::detail::gwn_launch_linear_kernel<k_block_size>(
        n, single_ray_api_uv_functor{
               geometry.accessor(), bvh.accessor(), aabb.accessor(), d_ox.span(), d_oy.span(),
               d_oz.span(), d_dx.span(), d_dy.span(), d_dz.span(), d_t.span(), d_pi.span(),
               d_u.span(), d_v.span()
           }
    );
    if (!s.is_ok()) {
        ADD_FAILURE() << "ray first-hit single API uv query: "
                      << gwn::tests::status_to_debug_string(s);
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "CUDA synchronize failed after single API uv query";
        return false;
    }

    out_t.resize(n);
    out_pi.resize(n);
    out_u.resize(n);
    out_v.resize(n);
    ok = d_t.copy_to_host(cuda::std::span<Real>(out_t.data(), n)).is_ok() &&
         d_pi.copy_to_host(cuda::std::span<Index>(out_pi.data(), n)).is_ok() &&
         d_u.copy_to_host(cuda::std::span<Real>(out_u.data(), n)).is_ok() &&
         d_v.copy_to_host(cuda::std::span<Real>(out_v.data(), n)).is_ok();
    if (!ok) {
        ADD_FAILURE() << "device-to-host copy failed";
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "CUDA synchronize failed after uv result copy";
        return false;
    }

    return true;
}

inline bool cpu_ray_triangle_intersect_moller(
    Real const ox, Real const oy, Real const oz, Real const dx, Real const dy, Real const dz,
    Real const ax, Real const ay, Real const az, Real const bx, Real const by, Real const bz,
    Real const cx, Real const cy, Real const cz, Real const t_min, Real const t_max, Real &t_out
) {
    double const e1x = static_cast<double>(bx - ax);
    double const e1y = static_cast<double>(by - ay);
    double const e1z = static_cast<double>(bz - az);
    double const e2x = static_cast<double>(cx - ax);
    double const e2y = static_cast<double>(cy - ay);
    double const e2z = static_cast<double>(cz - az);

    double const pvec_x = static_cast<double>(dy) * e2z - static_cast<double>(dz) * e2y;
    double const pvec_y = static_cast<double>(dz) * e2x - static_cast<double>(dx) * e2z;
    double const pvec_z = static_cast<double>(dx) * e2y - static_cast<double>(dy) * e2x;
    double const det = e1x * pvec_x + e1y * pvec_y + e1z * pvec_z;

    double constexpr k_eps = 1e-12;
    if (std::abs(det) < k_eps)
        return false;

    double const inv_det = 1.0 / det;
    double const tvec_x = static_cast<double>(ox - ax);
    double const tvec_y = static_cast<double>(oy - ay);
    double const tvec_z = static_cast<double>(oz - az);

    double const u = (tvec_x * pvec_x + tvec_y * pvec_y + tvec_z * pvec_z) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    double const qvec_x = tvec_y * e1z - tvec_z * e1y;
    double const qvec_y = tvec_z * e1x - tvec_x * e1z;
    double const qvec_z = tvec_x * e1y - tvec_y * e1x;

    double const v = (static_cast<double>(dx) * qvec_x + static_cast<double>(dy) * qvec_y +
                      static_cast<double>(dz) * qvec_z) *
                     inv_det;
    if (v < 0.0 || u + v > 1.0)
        return false;

    double const t = (e2x * qvec_x + e2y * qvec_y + e2z * qvec_z) * inv_det;
    if (t < static_cast<double>(t_min) || t > static_cast<double>(t_max))
        return false;

    t_out = static_cast<Real>(t);
    return true;
}

template <class Mesh>
bool cpu_ray_first_hit_mesh(
    Mesh const &mesh, Real const ox, Real const oy, Real const oz, Real const dx, Real const dy,
    Real const dz, Real const t_min, Real const t_max, Real &best_t, Index &best_pi
) {
    bool found = false;
    best_t = t_max;
    best_pi = gwn::gwn_invalid_index<Index>();
    for (std::size_t ti = 0; ti < mesh.i0.size(); ++ti) {
        std::size_t const ia = static_cast<std::size_t>(mesh.i0[ti]);
        std::size_t const ib = static_cast<std::size_t>(mesh.i1[ti]);
        std::size_t const ic = static_cast<std::size_t>(mesh.i2[ti]);

        Real t_hit = Real(0);
        if (!cpu_ray_triangle_intersect_moller(
                ox, oy, oz, dx, dy, dz, mesh.vx[ia], mesh.vy[ia], mesh.vz[ia], mesh.vx[ib],
                mesh.vy[ib], mesh.vz[ib], mesh.vx[ic], mesh.vy[ic], mesh.vz[ic], t_min, best_t,
                t_hit
            )) {
            continue;
        }

        if (t_hit < best_t) {
            best_t = t_hit;
            best_pi = static_cast<Index>(ti);
            found = true;
        }
    }
    return found;
}

inline Real next_unit_random(std::uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<Real>((state >> 8) * (1.0 / 16777216.0));
}

inline void normalize3(Real &x, Real &y, Real &z) {
    Real const n2 = x * x + y * y + z * z;
    if (!(n2 > Real(0))) {
        x = Real(0);
        y = Real(0);
        z = Real(1);
        return;
    }
    Real const inv_n = Real(1) / std::sqrt(n2);
    x *= inv_n;
    y *= inv_n;
    z *= inv_n;
}

template <class Mesh> gwn::tests::HostMesh make_host_mesh_impl(Mesh const &mesh) {
    gwn::tests::HostMesh host{};
    host.vertex_x.assign(mesh.vx.begin(), mesh.vx.end());
    host.vertex_y.assign(mesh.vy.begin(), mesh.vy.end());
    host.vertex_z.assign(mesh.vz.begin(), mesh.vz.end());
    host.tri_i0.assign(mesh.i0.begin(), mesh.i0.end());
    host.tri_i1.assign(mesh.i1.begin(), mesh.i1.end());
    host.tri_i2.assign(mesh.i2.begin(), mesh.i2.end());
    return host;
}

} // namespace

TEST_F(CudaFixture, ray_first_hit_single_triangle_basic_cases) {
    SingleTriangleMesh mesh;
    std::vector<Real> ox{Real(0.25), Real(0.25), Real(0.25), Real(0.25), Real(0.5), Real(0.25)};
    std::vector<Real> oy{Real(0.25), Real(0.25), Real(0.25), Real(0.25), Real(0.0), Real(0.25)};
    std::vector<Real> oz{Real(1.0), Real(-1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0)};
    std::vector<Real> dx{Real(0), Real(0), Real(1), Real(0), Real(0), Real(0)};
    std::vector<Real> dy{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
    std::vector<Real> dz{Real(-1), Real(1), Real(0), Real(1), Real(-1), Real(0)};

    std::vector<Real> t;
    std::vector<Index> pi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, t, pi))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(t.size(), ox.size());
    ASSERT_EQ(pi.size(), ox.size());

    Index const invalid = gwn::gwn_invalid_index<Index>();
    EXPECT_NEAR(t[0], Real(1), Real(1e-5));
    EXPECT_EQ(pi[0], Index(0));

    EXPECT_NEAR(t[1], Real(1), Real(1e-5)) << "back-face hit should be accepted";
    EXPECT_EQ(pi[1], Index(0));

    EXPECT_LT(t[2], Real(0)) << "parallel ray should miss";
    EXPECT_EQ(pi[2], invalid);

    EXPECT_LT(t[3], Real(0)) << "ray pointing away should miss";
    EXPECT_EQ(pi[3], invalid);

    EXPECT_NEAR(t[4], Real(1), Real(1e-5)) << "edge hit should remain robust";
    EXPECT_EQ(pi[4], Index(0));

    EXPECT_LT(t[5], Real(0)) << "zero-length ray direction should miss";
    EXPECT_EQ(pi[5], invalid);
}

TEST_F(CudaFixture, ray_first_hit_handles_near_zero_direction_components) {
    SingleTriangleMesh mesh;
    std::vector<Real> ox{Real(0.25)};
    std::vector<Real> oy{Real(0.25)};
    std::vector<Real> oz{Real(1.0)};
    std::vector<Real> dx{Real(1e-8)};
    std::vector<Real> dy{Real(-1e-8)};
    std::vector<Real> dz{Real(-1.0)};

    std::vector<Real> t;
    std::vector<Index> pi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, t, pi))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(t.size(), 1u);
    ASSERT_EQ(pi.size(), 1u);
    EXPECT_NEAR(t[0], Real(1), Real(1e-5));
    EXPECT_EQ(pi[0], Index(0));
}

TEST_F(CudaFixture, ray_first_hit_coplanar_and_near_coplanar_behavior) {
    SingleTriangleMesh mesh;
    std::vector<Real> ox{Real(0.25), Real(0.25)};
    std::vector<Real> oy{Real(0.25), Real(0.25)};
    std::vector<Real> oz{Real(0.0), Real(1e-6)};
    std::vector<Real> dx{Real(1.0), Real(0.0)};
    std::vector<Real> dy{Real(0.0), Real(0.0)};
    std::vector<Real> dz{Real(0.0), Real(-1.0)};

    std::vector<Real> t;
    std::vector<Index> pi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, t, pi))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(t.size(), 2u);
    ASSERT_EQ(pi.size(), 2u);
    EXPECT_LT(t[0], Real(0)) << "coplanar ray should miss for first-hit query";
    EXPECT_EQ(pi[0], gwn::gwn_invalid_index<Index>());
    EXPECT_NEAR(t[1], Real(1e-6), Real(1e-7));
    EXPECT_EQ(pi[1], Index(0));
}

TEST_F(CudaFixture, ray_first_hit_respects_t_interval) {
    SingleTriangleMesh mesh;
    std::vector<Real> ox{Real(0.25)};
    std::vector<Real> oy{Real(0.25)};
    std::vector<Real> oz{Real(1.0)};
    std::vector<Real> dx{Real(0)};
    std::vector<Real> dy{Real(0)};
    std::vector<Real> dz{Real(-1)};

    std::vector<Real> t;
    std::vector<Index> pi;

    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, t, pi, Real(0), Real(1)))
        GTEST_SKIP() << "CUDA unavailable";
    ASSERT_EQ(t.size(), 1u);
    EXPECT_NEAR(t[0], Real(1), Real(1e-5));

    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, t, pi, Real(1.1), Real(10)))
        GTEST_SKIP() << "CUDA unavailable";
    EXPECT_LT(t[0], Real(0));

    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, t, pi, Real(0), Real(0.9)))
        GTEST_SKIP() << "CUDA unavailable";
    EXPECT_LT(t[0], Real(0));
}

TEST_F(CudaFixture, ray_first_hit_accepts_exact_t_interval_boundary) {
    SingleTriangleMesh mesh;
    std::vector<Real> ox{Real(0.25)};
    std::vector<Real> oy{Real(0.25)};
    std::vector<Real> oz{Real(1.0)};
    std::vector<Real> dx{Real(0)};
    std::vector<Real> dy{Real(0)};
    std::vector<Real> dz{Real(-1)};

    std::vector<Real> t;
    std::vector<Index> pi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, t, pi, Real(1), Real(1)))
        GTEST_SKIP() << "CUDA unavailable";
    ASSERT_EQ(t.size(), 1u);
    ASSERT_EQ(pi.size(), 1u);
    EXPECT_NEAR(t[0], Real(1), Real(1e-6));
    EXPECT_EQ(pi[0], Index(0));
}

TEST_F(CudaFixture, ray_first_hit_returns_nearest_triangle) {
    TwoLayerTriangleMesh mesh;
    std::vector<Real> ox{Real(0.25), Real(0.25), Real(0.25)};
    std::vector<Real> oy{Real(0.25), Real(0.25), Real(0.25)};
    std::vector<Real> oz{Real(1.0), Real(-1.0), Real(1.0)};
    std::vector<Real> dx{Real(0), Real(0), Real(0)};
    std::vector<Real> dy{Real(0), Real(0), Real(0)};
    std::vector<Real> dz{Real(-1), Real(-1), Real(1)};

    std::vector<Real> t;
    std::vector<Index> pi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, t, pi))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(t.size(), 3u);
    ASSERT_EQ(pi.size(), 3u);

    EXPECT_NEAR(t[0], Real(1), Real(1e-5));
    EXPECT_EQ(pi[0], Index(1)) << "near triangle should win";

    EXPECT_NEAR(t[1], Real(1), Real(1e-5));
    EXPECT_EQ(pi[1], Index(0)) << "from between layers, far plane is next hit";

    EXPECT_LT(t[2], Real(0));
    EXPECT_EQ(pi[2], gwn::gwn_invalid_index<Index>());
}

TEST_F(CudaFixture, ray_first_hit_shared_edge_has_no_cracks) {
    SharedEdgeSquareMesh mesh;
    int constexpr n = 128;
    std::vector<Real> ox;
    std::vector<Real> oy;
    std::vector<Real> oz;
    std::vector<Real> dx;
    std::vector<Real> dy;
    std::vector<Real> dz;
    ox.reserve(n);
    oy.reserve(n);
    oz.reserve(n);
    dx.reserve(n);
    dy.reserve(n);
    dz.reserve(n);

    for (int i = 0; i < n; ++i) {
        Real const alpha = (static_cast<Real>(i) + Real(0.5)) / static_cast<Real>(n);
        ox.push_back(alpha);
        oy.push_back(alpha);
        oz.push_back(Real(1));
        dx.push_back(Real(0));
        dy.push_back(Real(0));
        dz.push_back(Real(-1));
    }

    std::vector<Real> t;
    std::vector<Index> pi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, t, pi))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(t.size(), static_cast<std::size_t>(n));
    ASSERT_EQ(pi.size(), static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(t[static_cast<std::size_t>(i)], Real(1), Real(1e-5))
            << "shared-edge miss/regression at sample " << i;
        EXPECT_TRUE(
            pi[static_cast<std::size_t>(i)] == Index(0) ||
            pi[static_cast<std::size_t>(i)] == Index(1)
        ) << "unexpected primitive id at sample "
          << i;
    }
}

TEST_F(CudaFixture, ray_first_hit_batch_rejects_invalid_accessors) {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t;
    gwn::gwn_device_array<Index> d_pi;
    bool ok = d_ox.resize(1).is_ok() && d_oy.resize(1).is_ok() && d_oz.resize(1).is_ok() &&
              d_dx.resize(1).is_ok() && d_dy.resize(1).is_ok() && d_dz.resize(1).is_ok() &&
              d_t.resize(1).is_ok() && d_pi.resize(1).is_ok();
    if (!ok)
        GTEST_SKIP() << "CUDA unavailable";

    gwn::gwn_status const s = gwn::gwn_compute_ray_first_hit_batch_bvh<Real, Index>(
        geometry, bvh, aabb, d_ox.span(), d_oy.span(), d_oz.span(), d_dx.span(), d_dy.span(),
        d_dz.span(), d_t.span(), d_pi.span()
    );
    EXPECT_EQ(s.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, ray_first_hit_batch_mismatched_spans_and_invalid_interval) {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh{};
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb{};
    cuda::std::span<Real const> empty{};
    Real t_buf[2] = {};
    Index pi_buf[2] = {};
    cuda::std::span<Real> out_t(t_buf, 2);
    cuda::std::span<Index> out_pi(pi_buf, 2);
    cuda::std::span<Real> empty_t{};
    cuda::std::span<Index> empty_pi{};

    gwn::gwn_status s = gwn::gwn_compute_ray_first_hit_batch_bvh<Real, Index>(
        geometry, bvh, aabb, empty, empty, empty, empty, empty, empty, out_t, out_pi
    );
    EXPECT_EQ(s.error(), gwn::gwn_error::invalid_argument);

    s = gwn::gwn_compute_ray_first_hit_batch_bvh<Real, Index>(
        geometry, bvh, aabb, empty, empty, empty, empty, empty, empty, empty_t, empty_pi, Real(2),
        Real(1)
    );
    EXPECT_EQ(s.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, ray_first_hit_matches_cpu_reference_on_octahedron) {
    OctahedronMesh mesh;

    int constexpr n = 512;
    std::vector<Real> ox;
    std::vector<Real> oy;
    std::vector<Real> oz;
    std::vector<Real> dx;
    std::vector<Real> dy;
    std::vector<Real> dz;
    ox.reserve(n);
    oy.reserve(n);
    oz.reserve(n);
    dx.reserve(n);
    dy.reserve(n);
    dz.reserve(n);

    std::uint32_t rng = 0x1234abcdu;
    constexpr Real pi = Real(3.14159265358979323846);
    for (int i = 0; i < n; ++i) {
        Real const u = next_unit_random(rng);
        Real const v = next_unit_random(rng);
        Real const z = Real(2) * u - Real(1);
        Real const phi = Real(2) * pi * v;
        Real const r = std::sqrt(std::max(Real(0), Real(1) - z * z));
        Real ux = r * std::cos(phi);
        Real uy = r * std::sin(phi);
        Real uz = z;
        normalize3(ux, uy, uz);

        ox.push_back(Real(4) * ux);
        oy.push_back(Real(4) * uy);
        oz.push_back(Real(4) * uz);

        if ((i & 1) == 0) {
            Real jx = Real(2) * next_unit_random(rng) - Real(1);
            Real jy = Real(2) * next_unit_random(rng) - Real(1);
            Real jz = Real(2) * next_unit_random(rng) - Real(1);
            normalize3(jx, jy, jz);
            Real dir_x = -ux + Real(0.03) * jx;
            Real dir_y = -uy + Real(0.03) * jy;
            Real dir_z = -uz + Real(0.03) * jz;
            normalize3(dir_x, dir_y, dir_z);
            dx.push_back(dir_x);
            dy.push_back(dir_y);
            dz.push_back(dir_z);
        } else {
            // Point away from the object to generate robust miss cases.
            dx.push_back(ux);
            dy.push_back(uy);
            dz.push_back(uz);
        }
    }

    std::vector<Real> gpu_t;
    std::vector<Index> gpu_pi;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, gpu_t, gpu_pi))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(gpu_t.size(), static_cast<std::size_t>(n));
    ASSERT_EQ(gpu_pi.size(), static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        Real cpu_t = Real(-1);
        Index cpu_pi = gwn::gwn_invalid_index<Index>();
        bool const cpu_hit = cpu_ray_first_hit_mesh(
            mesh, ox[static_cast<std::size_t>(i)], oy[static_cast<std::size_t>(i)],
            oz[static_cast<std::size_t>(i)], dx[static_cast<std::size_t>(i)],
            dy[static_cast<std::size_t>(i)], dz[static_cast<std::size_t>(i)], Real(0),
            std::numeric_limits<Real>::infinity(), cpu_t, cpu_pi
        );
        bool const gpu_hit = gpu_t[static_cast<std::size_t>(i)] >= Real(0);

        EXPECT_EQ(gpu_hit, cpu_hit) << "hit/miss mismatch at ray " << i;
        if (!gpu_hit) {
            EXPECT_EQ(gpu_pi[static_cast<std::size_t>(i)], gwn::gwn_invalid_index<Index>())
                << "miss should output invalid primitive id at ray " << i;
            continue;
        }

        EXPECT_NEAR(gpu_t[static_cast<std::size_t>(i)], cpu_t, Real(2e-3))
            << "t mismatch at ray " << i;
        EXPECT_NE(gpu_pi[static_cast<std::size_t>(i)], gwn::gwn_invalid_index<Index>())
            << "hit should output valid primitive id at ray " << i;
    }
}

TEST_F(CudaFixture, ray_first_hit_single_api_matches_batch_api) {
    TwoLayerTriangleMesh mesh;
    std::vector<Real> ox{Real(0.25), Real(0.25), Real(0.25), Real(0.4)};
    std::vector<Real> oy{Real(0.25), Real(0.25), Real(0.25), Real(0.1)};
    std::vector<Real> oz{Real(1.0), Real(-1.0), Real(1.0), Real(1.5)};
    std::vector<Real> dx{Real(0), Real(0), Real(0), Real(0)};
    std::vector<Real> dy{Real(0), Real(0), Real(0), Real(0)};
    std::vector<Real> dz{Real(-1), Real(-1), Real(1), Real(-1)};

    std::vector<Real> t_batch;
    std::vector<Index> pi_batch;
    if (!run_ray_first_hit_query(mesh, ox, oy, oz, dx, dy, dz, t_batch, pi_batch))
        GTEST_SKIP() << "CUDA unavailable";

    std::vector<Real> t_single;
    std::vector<Index> pi_single;
    if (!run_ray_first_hit_query_single_api(mesh, ox, oy, oz, dx, dy, dz, t_single, pi_single))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(t_batch.size(), t_single.size());
    ASSERT_EQ(pi_batch.size(), pi_single.size());
    for (std::size_t i = 0; i < t_batch.size(); ++i) {
        EXPECT_NEAR(t_single[i], t_batch[i], Real(1e-6)) << "t mismatch at ray " << i;
        EXPECT_EQ(pi_single[i], pi_batch[i]) << "primitive id mismatch at ray " << i;
    }
}

TEST_F(CudaFixture, ray_first_hit_single_api_reports_embree_style_uv_barycentrics) {
    SingleTriangleMesh mesh;
    std::vector<Real> ox{Real(0.2)};
    std::vector<Real> oy{Real(0.3)};
    std::vector<Real> oz{Real(1.0)};
    std::vector<Real> dx{Real(0)};
    std::vector<Real> dy{Real(0)};
    std::vector<Real> dz{Real(-1)};

    std::vector<Real> t;
    std::vector<Index> pi;
    std::vector<Real> u;
    std::vector<Real> v;
    if (!run_ray_first_hit_query_single_api_with_uv(mesh, ox, oy, oz, dx, dy, dz, t, pi, u, v))
        GTEST_SKIP() << "CUDA unavailable";

    ASSERT_EQ(t.size(), 1u);
    ASSERT_EQ(pi.size(), 1u);
    ASSERT_EQ(u.size(), 1u);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(pi[0], Index(0));
    EXPECT_NEAR(t[0], Real(1), Real(1e-6));
    EXPECT_NEAR(u[0], Real(0.2), Real(1e-5));
    EXPECT_NEAR(v[0], Real(0.3), Real(1e-5));
    EXPECT_NEAR(u[0] + v[0], Real(0.5), Real(1e-5));
}

TEST_F(CudaFixture, ray_first_hit_benchmark_mix_contains_hits_and_misses) {
    OctahedronMesh mesh;
    gwn::tests::HostMesh const host_mesh = make_host_mesh_impl(mesh);
    auto const ray_mix = gwn::bench::gwn_make_mixed_ray_soa(host_mesh, 4096, 0xA55A5AA5u);

    std::vector<Real> t;
    std::vector<Index> pi;
    if (!run_ray_first_hit_query(
            mesh, ray_mix.origin[0], ray_mix.origin[1], ray_mix.origin[2], ray_mix.direction[0],
            ray_mix.direction[1], ray_mix.direction[2], t, pi
        )) {
        GTEST_SKIP() << "CUDA unavailable";
    }

    std::size_t const hit_count = static_cast<std::size_t>(
        std::count_if(t.begin(), t.end(), [](Real const value) { return value >= Real(0); })
    );
    ASSERT_GT(hit_count, 0u) << "benchmark ray mix should contain at least one hit";
    ASSERT_LT(hit_count, t.size()) << "benchmark ray mix should contain at least one miss";

    double const hit_ratio = static_cast<double>(hit_count) / static_cast<double>(t.size());
    EXPECT_GT(hit_ratio, 0.05) << "hit ratio too small -> benchmark mix degenerates to misses";
    EXPECT_LT(hit_ratio, 0.95) << "hit ratio too large -> benchmark mix degenerates to hits";
}

TEST(ray_query_vec3, component_access_via_subscript_operator) {
    gwn::detail::gwn_query_vec3<Real> const v(Real(3), Real(4), Real(5));
    EXPECT_EQ(v[0], Real(3));
    EXPECT_EQ(v[1], Real(4));
    EXPECT_EQ(v[2], Real(5));
}
