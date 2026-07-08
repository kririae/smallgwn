#include <array>
#include <cmath>
#include <limits>
#include <string>
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

template <class Mesh> struct AntipodalContext {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_boundary_chain_object<Index> boundary;
    bool ready{false};
};

template <class Mesh> void setup_context(Mesh const &mesh, AntipodalContext<Mesh> &ctx) {
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        ctx.geometry, cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
        cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
        cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
        cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
        cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
        cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    status = gwn::gwn_bvh_facade_build_topology_aabb_lbvh<4, Real, Index>(
        ctx.geometry, ctx.bvh, ctx.aabb
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    status = gwn::gwn_build_boundary_chain(ctx.geometry.accessor(), ctx.boundary);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ctx.ready = true;
}

std::vector<Real> run_exact(
    gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_accessor<Real, Index> const &bvh, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::gwn_device_array<Real> d_qx;
    gwn::gwn_device_array<Real> d_qy;
    gwn::gwn_device_array<Real> d_qz;
    gwn::gwn_device_array<Real> d_out;
    EXPECT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    EXPECT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    EXPECT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());
    EXPECT_TRUE(d_out.resize(qx.size()).is_ok());

    gwn::gwn_status const status = gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
        geometry, bvh, d_qx.span(), d_qy.span(), d_qz.span(), d_out.span()
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> out(qx.size(), Real(0));
    EXPECT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(out.data(), out.size())).is_ok());
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return out;
}

std::vector<Real> run_antipodal(
    gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_accessor<Real, Index> const &bvh,
    gwn::gwn_bvh4_aabb_accessor<Real, Index> const &aabb,
    gwn::gwn_boundary_chain_accessor<Index> const &boundary, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::gwn_device_array<Real> d_qx;
    gwn::gwn_device_array<Real> d_qy;
    gwn::gwn_device_array<Real> d_qz;
    gwn::gwn_device_array<Real> d_out;
    EXPECT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    EXPECT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    EXPECT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());
    EXPECT_TRUE(d_out.resize(qx.size()).is_ok());

    gwn::gwn_status const status = gwn::gwn_compute_winding_number_batch_bvh_antipodal<Real, Index>(
        geometry, bvh, aabb, boundary, d_qx.span(), d_qy.span(), d_qz.span(), d_out.span()
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> out(qx.size(), Real(0));
    EXPECT_TRUE(d_out.copy_to_host(cuda::std::span<Real>(out.data(), out.size())).is_ok());
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return out;
}

std::array<std::vector<Real>, 3> run_antipodal_gradient(
    gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_boundary_chain_accessor<Index> const &boundary, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::gwn_device_array<Real> d_qx;
    gwn::gwn_device_array<Real> d_qy;
    gwn::gwn_device_array<Real> d_qz;
    gwn::gwn_device_array<Real> d_gx;
    gwn::gwn_device_array<Real> d_gy;
    gwn::gwn_device_array<Real> d_gz;
    EXPECT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    EXPECT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    EXPECT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());
    EXPECT_TRUE(d_gx.resize(qx.size()).is_ok());
    EXPECT_TRUE(d_gy.resize(qx.size()).is_ok());
    EXPECT_TRUE(d_gz.resize(qx.size()).is_ok());

    gwn::gwn_status const status = gwn::gwn_compute_winding_gradient_batch_antipodal<Real, Index>(
        geometry, boundary, d_qx.span(), d_qy.span(), d_qz.span(), d_gx.span(), d_gy.span(),
        d_gz.span()
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<std::vector<Real>, 3> out{
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
    };
    EXPECT_TRUE(d_gx.copy_to_host(cuda::std::span<Real>(out[0].data(), out[0].size())).is_ok());
    EXPECT_TRUE(d_gy.copy_to_host(cuda::std::span<Real>(out[1].data(), out[1].size())).is_ok());
    EXPECT_TRUE(d_gz.copy_to_host(cuda::std::span<Real>(out[2].data(), out[2].size())).is_ok());
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return out;
}

struct antipodal_gradient_point_functor {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_boundary_chain_accessor<Index> boundary{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_x{};
    cuda::std::span<Real> out_y{};
    cuda::std::span<Real> out_z{};

    __device__ void operator()(std::size_t const query_id) const {
        gwn::gwn_vec3<Real> const gradient = gwn::gwn_winding_gradient_point_antipodal(
            geometry, boundary, query_x[query_id], query_y[query_id], query_z[query_id]
        );
        out_x[query_id] = gradient.x;
        out_y[query_id] = gradient.y;
        out_z[query_id] = gradient.z;
    }
};

struct antipodal_projected_edge_sign_functor {
    cuda::std::span<int> out{};

    __device__ void operator()(std::size_t const) const {
        using gwn::detail::gwn_antipodal_projected_edge_classify_impl;
        using gwn::detail::gwn_antipodal_projected_edge_sign;
        using gwn::detail::gwn_antipodal_ray_axis;
        using gwn::detail::gwn_query_vec3;

        auto encode = [](gwn_antipodal_projected_edge_sign const sign) {
            return static_cast<int>(sign);
        };

        out[0] =
            encode(gwn_antipodal_projected_edge_classify_impl(
                       gwn_query_vec3<Real>(Real(1), Real(0), Real(0)),
                       gwn_query_vec3<Real>(Real(0), Real(1), Real(0)), gwn_antipodal_ray_axis::k_z
            )
                       .sign);
        out[1] =
            encode(gwn_antipodal_projected_edge_classify_impl(
                       gwn_query_vec3<Real>(Real(0), Real(1), Real(0)),
                       gwn_query_vec3<Real>(Real(1), Real(0), Real(0)), gwn_antipodal_ray_axis::k_z
            )
                       .sign);
        out[2] =
            encode(gwn_antipodal_projected_edge_classify_impl(
                       gwn_query_vec3<Real>(Real(0), Real(1), Real(0)),
                       gwn_query_vec3<Real>(Real(0), Real(0), Real(1)), gwn_antipodal_ray_axis::k_x
            )
                       .sign);
        out[3] =
            encode(gwn_antipodal_projected_edge_classify_impl(
                       gwn_query_vec3<Real>(Real(1), Real(0), Real(0)),
                       gwn_query_vec3<Real>(Real(0), Real(0), Real(1)), gwn_antipodal_ray_axis::k_y
            )
                       .sign);
        out[4] =
            encode(gwn_antipodal_projected_edge_classify_impl(
                       gwn_query_vec3<Real>(Real(1), Real(0), Real(0)),
                       gwn_query_vec3<Real>(Real(2), Real(0), Real(0)), gwn_antipodal_ray_axis::k_z
            )
                       .sign);
        out[5] =
            encode(gwn_antipodal_projected_edge_classify_impl(
                       gwn_query_vec3<Real>(Real(0), Real(0), Real(0)),
                       gwn_query_vec3<Real>(Real(0), Real(0), Real(0)), gwn_antipodal_ray_axis::k_z
            )
                       .sign);
    }
};

struct antipodal_projected_edge_classifier_functor {
    cuda::std::span<int> out_sign{};
    cuda::std::span<int> out_status{};

    __device__ void operator()(std::size_t const) const {
        using gwn::detail::gwn_antipodal_projected_edge_classify_impl;
        using gwn::detail::gwn_antipodal_ray_axis;
        using gwn::detail::gwn_query_vec3;

        auto const identical_projection = gwn_antipodal_projected_edge_classify_impl(
            gwn_query_vec3<Real>(Real(1), Real(0), Real(0)),
            gwn_query_vec3<Real>(Real(1), Real(0), Real(1)), gwn_antipodal_ray_axis::k_z
        );
        out_sign[0] = static_cast<int>(identical_projection.sign);
        out_status[0] = static_cast<int>(identical_projection.status);

        auto const contains_origin = gwn_antipodal_projected_edge_classify_impl(
            gwn_query_vec3<Real>(Real(1), Real(0), Real(0)),
            gwn_query_vec3<Real>(Real(-1), Real(0), Real(1)), gwn_antipodal_ray_axis::k_z
        );
        out_sign[1] = static_cast<int>(contains_origin.sign);
        out_status[1] = static_cast<int>(contains_origin.status);

        auto const regular_edge = gwn_antipodal_projected_edge_classify_impl(
            gwn_query_vec3<Real>(Real(1), Real(0), Real(0)),
            gwn_query_vec3<Real>(Real(0), Real(1), Real(0)), gwn_antipodal_ray_axis::k_z
        );
        out_sign[2] = static_cast<int>(regular_edge.sign);
        out_status[2] = static_cast<int>(regular_edge.status);
    }
};

struct antipodal_triangle_crossing_functor {
    cuda::std::span<int> out_status{};
    cuda::std::span<Real> out_crossing_sign{};

    __device__ void operator()(std::size_t const) const {
        using gwn::detail::gwn_antipodal_axis_result;
        using gwn::detail::gwn_antipodal_ray_axis;
        using gwn::detail::gwn_query_vec3;
        using gwn::detail::gwn_ray_triangle_crossing_antipodal_impl;

        Real crossing_sign = Real(0);
        gwn_antipodal_axis_result const status = gwn_ray_triangle_crossing_antipodal_impl(
            gwn_query_vec3<Real>(Real(0), Real(0), Real(0)), gwn_antipodal_ray_axis::k_z,
            gwn_query_vec3<Real>(Real(1), Real(0), Real(0)),
            gwn_query_vec3<Real>(Real(-1), Real(0), Real(1)),
            gwn_query_vec3<Real>(Real(0), Real(1), Real(1)), crossing_sign
        );
        out_status[0] = static_cast<int>(status);
        out_crossing_sign[0] = crossing_sign;
    }
};

std::array<std::vector<Real>, 3> run_antipodal_gradient_point_kernel(
    gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_boundary_chain_accessor<Index> const &boundary, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::gwn_device_array<Real> d_qx;
    gwn::gwn_device_array<Real> d_qy;
    gwn::gwn_device_array<Real> d_qz;
    gwn::gwn_device_array<Real> d_gx;
    gwn::gwn_device_array<Real> d_gy;
    gwn::gwn_device_array<Real> d_gz;
    EXPECT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    EXPECT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    EXPECT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());
    EXPECT_TRUE(d_gx.resize(qx.size()).is_ok());
    EXPECT_TRUE(d_gy.resize(qx.size()).is_ok());
    EXPECT_TRUE(d_gz.resize(qx.size()).is_ok());

    gwn::gwn_status const status =
        gwn::detail::gwn_launch_linear_kernel<gwn::detail::k_gwn_default_block_size>(
            qx.size(), antipodal_gradient_point_functor{
                           geometry, boundary, d_qx.span(), d_qy.span(), d_qz.span(), d_gx.span(),
                           d_gy.span(), d_gz.span()
                       }
        );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<std::vector<Real>, 3> out{
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
    };
    EXPECT_TRUE(d_gx.copy_to_host(cuda::std::span<Real>(out[0].data(), out[0].size())).is_ok());
    EXPECT_TRUE(d_gy.copy_to_host(cuda::std::span<Real>(out[1].data(), out[1].size())).is_ok());
    EXPECT_TRUE(d_gz.copy_to_host(cuda::std::span<Real>(out[2].data(), out[2].size())).is_ok());
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return out;
}

void expect_near_vector(std::vector<Real> const &actual, std::vector<Real> const &expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], Real(3e-4)) << "query " << i;
}

void expect_near_gradient(
    std::array<std::vector<Real>, 3> const &actual,
    std::array<std::vector<Real>, 3> const &expected, Real const tolerance
) {
    for (int axis = 0; axis < 3; ++axis) {
        ASSERT_EQ(actual[axis].size(), expected[axis].size());
        for (std::size_t i = 0; i < actual[axis].size(); ++i)
            EXPECT_NEAR(actual[axis][i], expected[axis][i], tolerance)
                << "axis " << axis << " query " << i;
    }
}

void expect_nan_gradient(std::array<std::vector<Real>, 3> const &actual) {
    for (int axis = 0; axis < 3; ++axis)
        for (std::size_t i = 0; i < actual[axis].size(); ++i)
            EXPECT_TRUE(std::isnan(actual[axis][i])) << "axis " << axis << " query " << i;
}

void expect_finite_gradient(std::array<std::vector<Real>, 3> const &actual) {
    for (int axis = 0; axis < 3; ++axis)
        for (std::size_t i = 0; i < actual[axis].size(); ++i)
            EXPECT_TRUE(std::isfinite(actual[axis][i])) << "axis " << axis << " query " << i;
}

void expect_nan_vector(std::vector<Real> const &actual) {
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_TRUE(std::isnan(actual[i])) << "query " << i;
}

} // namespace

TEST_F(CudaFixture, antipodal_closed_mesh_matches_existing_winding) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.23), Real(2.2), Real(-0.41), Real(-2.3)};
    std::array<Real, 4> const qy{Real(0.17), Real(0.3), Real(0.26), Real(-0.4)};
    std::array<Real, 4> const qz{Real(0.11), Real(0.2), Real(1.7), Real(0.5)};

    expect_near_vector(
        run_antipodal(
            ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(),
            ctx.boundary.accessor(), qx, qy, qz
        ),
        run_exact(ctx.geometry.accessor(), ctx.bvh.accessor(), qx, qy, qz)
    );
}

TEST_F(CudaFixture, antipodal_projected_edge_sign_uses_coordinate_planes) {
    gwn::gwn_device_array<int> d_out;
    ASSERT_TRUE(d_out.resize(6).is_ok());
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<1>(
                     1, antipodal_projected_edge_sign_functor{d_out.span()}
                 ))
                    .is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<int, 6> out{};
    ASSERT_TRUE(d_out.copy_to_host(cuda::std::span<int>(out.data(), out.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], -1);
    EXPECT_EQ(out[2], 1);
    EXPECT_EQ(out[3], -1);
    EXPECT_EQ(out[4], 1);
    EXPECT_EQ(out[5], 0);
}

TEST_F(CudaFixture, antipodal_projected_edge_classifier_marks_singular_axes) {
    gwn::gwn_device_array<int> d_sign;
    gwn::gwn_device_array<int> d_status;
    ASSERT_TRUE(d_sign.resize(3).is_ok());
    ASSERT_TRUE(d_status.resize(3).is_ok());
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<1>(
                     1, antipodal_projected_edge_classifier_functor{d_sign.span(), d_status.span()}
                 ))
                    .is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<int, 3> sign{};
    std::array<int, 3> status{};
    ASSERT_TRUE(d_sign.copy_to_host(cuda::std::span<int>(sign.data(), sign.size())).is_ok());
    ASSERT_TRUE(d_status.copy_to_host(cuda::std::span<int>(status.data(), status.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_EQ(sign[0], 0);
    EXPECT_EQ(status[0], static_cast<int>(gwn::detail::gwn_antipodal_axis_result::k_done));

    EXPECT_EQ(sign[1], -1);
    EXPECT_EQ(status[1], static_cast<int>(gwn::detail::gwn_antipodal_axis_result::k_singular));

    EXPECT_EQ(sign[2], 1);
    EXPECT_EQ(status[2], static_cast<int>(gwn::detail::gwn_antipodal_axis_result::k_done));
}

TEST_F(CudaFixture, antipodal_triangle_crossing_marks_projected_origin_edge_as_singular) {
    gwn::gwn_device_array<int> d_status;
    gwn::gwn_device_array<Real> d_crossing_sign;
    ASSERT_TRUE(d_status.resize(1).is_ok());
    ASSERT_TRUE(d_crossing_sign.resize(1).is_ok());
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<1>(
                     1, antipodal_triangle_crossing_functor{d_status.span(), d_crossing_sign.span()}
                 ))
                    .is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<int, 1> status{};
    std::array<Real, 1> crossing_sign{};
    ASSERT_TRUE(d_status.copy_to_host(cuda::std::span<int>(status.data(), status.size())).is_ok());
    ASSERT_TRUE(d_crossing_sign
                    .copy_to_host(cuda::std::span<Real>(crossing_sign.data(), crossing_sign.size()))
                    .is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_EQ(status[0], static_cast<int>(gwn::detail::gwn_antipodal_axis_result::k_singular));
    EXPECT_EQ(crossing_sign[0], Real(0));
}

TEST_F(CudaFixture, antipodal_center_axis_ray_returns_nan_after_all_retries) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0), Real(0), Real(0), Real(0)};
    std::array<Real, 4> const qy{Real(0), Real(0), Real(0), Real(0)};
    std::array<Real, 4> const qz{Real(0), Real(-0.25), Real(0.25), Real(-0.5)};

    std::vector<Real> const antipodal = run_antipodal(
        ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(), ctx.boundary.accessor(),
        qx, qy, qz
    );
    std::vector<Real> const exact =
        run_exact(ctx.geometry.accessor(), ctx.bvh.accessor(), qx, qy, qz);

    EXPECT_TRUE(std::isnan(antipodal[0]));
    for (std::size_t i = 1; i < antipodal.size(); ++i)
        EXPECT_NEAR(antipodal[i], exact[i], Real(3e-4)) << "query " << i;
}

TEST_F(CudaFixture, antipodal_open_mesh_matches_existing_winding) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.23), Real(2.2), Real(-0.41), Real(-2.3)};
    std::array<Real, 4> const qy{Real(0.17), Real(0.3), Real(0.26), Real(-0.4)};
    std::array<Real, 4> const qz{Real(0.11), Real(0.2), Real(1.7), Real(0.5)};

    expect_near_vector(
        run_antipodal(
            ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(),
            ctx.boundary.accessor(), qx, qy, qz
        ),
        run_exact(ctx.geometry.accessor(), ctx.bvh.accessor(), qx, qy, qz)
    );
}

TEST_F(CudaFixture, antipodal_open_cube_boundary_line_returns_nan_after_all_retries) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.25), Real(-0.25), Real(0.5), Real(-0.5)};
    std::array<Real, 4> const qy{Real(1), Real(1), Real(1), Real(1)};
    std::array<Real, 4> const qz{Real(0.2), Real(0.2), Real(0.4), Real(0.4)};

    expect_nan_vector(run_antipodal(
        ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(), ctx.boundary.accessor(),
        qx, qy, qz
    ));
}

TEST_F(CudaFixture, antipodal_gradient_closed_mesh_is_zero) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.23), Real(2.2), Real(-0.41), Real(-2.3)};
    std::array<Real, 4> const qy{Real(0.17), Real(0.3), Real(0.26), Real(-0.4)};
    std::array<Real, 4> const qz{Real(0.11), Real(0.2), Real(1.7), Real(0.5)};

    std::array<std::vector<Real>, 3> const gradient =
        run_antipodal_gradient(ctx.geometry.accessor(), ctx.boundary.accessor(), qx, qy, qz);
    std::array<std::vector<Real>, 3> const zero{
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
    };
    expect_near_gradient(gradient, zero, Real(1e-6));
}

TEST_F(CudaFixture, antipodal_gradient_matches_finite_difference) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.21), Real(1.3), Real(-1.4), Real(0.35)};
    std::array<Real, 4> const qy{Real(0.18), Real(0.25), Real(-0.33), Real(1.4)};
    std::array<Real, 4> const qz{Real(0.15), Real(0.2), Real(0.42), Real(1.55)};
    Real const eps = Real(1e-2);

    std::array<std::vector<Real>, 3> expected{
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
    };
    for (int axis = 0; axis < 3; ++axis) {
        std::array<Real, 4> qx_plus = qx;
        std::array<Real, 4> qy_plus = qy;
        std::array<Real, 4> qz_plus = qz;
        std::array<Real, 4> qx_minus = qx;
        std::array<Real, 4> qy_minus = qy;
        std::array<Real, 4> qz_minus = qz;
        auto &plus = axis == 0 ? qx_plus : (axis == 1 ? qy_plus : qz_plus);
        auto &minus = axis == 0 ? qx_minus : (axis == 1 ? qy_minus : qz_minus);
        for (std::size_t i = 0; i < plus.size(); ++i) {
            plus[i] += eps;
            minus[i] -= eps;
        }

        std::vector<Real> const value_plus = run_antipodal(
            ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(),
            ctx.boundary.accessor(), qx_plus, qy_plus, qz_plus
        );
        std::vector<Real> const value_minus = run_antipodal(
            ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(),
            ctx.boundary.accessor(), qx_minus, qy_minus, qz_minus
        );
        for (std::size_t i = 0; i < value_plus.size(); ++i)
            expected[axis][i] = (value_plus[i] - value_minus[i]) / (Real(2) * eps);
    }

    std::array<std::vector<Real>, 3> const gradient =
        run_antipodal_gradient(ctx.geometry.accessor(), ctx.boundary.accessor(), qx, qy, qz);
    expect_near_gradient(gradient, expected, Real(6e-3));
}

TEST_F(CudaFixture, antipodal_gradient_retries_axis_when_boundary_projects_singular) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(1), Real(1), Real(1), Real(1)};
    std::array<Real, 4> const qy{Real(0.25), Real(-0.25), Real(0.5), Real(-0.5)};
    std::array<Real, 4> const qz{Real(0.2), Real(0.2), Real(0.4), Real(0.4)};

    std::array<std::vector<Real>, 3> const gradient =
        run_antipodal_gradient(ctx.geometry.accessor(), ctx.boundary.accessor(), qx, qy, qz);
    expect_finite_gradient(gradient);

    EXPECT_NEAR(gradient[0][0], gradient[0][1], Real(1e-6));
    EXPECT_NEAR(gradient[1][0], -gradient[1][1], Real(1e-6));
    EXPECT_NEAR(gradient[2][0], gradient[2][1], Real(1e-6));

    EXPECT_NEAR(gradient[0][2], gradient[0][3], Real(1e-6));
    EXPECT_NEAR(gradient[1][2], -gradient[1][3], Real(1e-6));
    EXPECT_NEAR(gradient[2][2], gradient[2][3], Real(1e-6));
}

TEST_F(CudaFixture, antipodal_gradient_point_matches_batch) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.21), Real(1.3), Real(-1.4), Real(0.35)};
    std::array<Real, 4> const qy{Real(0.18), Real(0.25), Real(-0.33), Real(1.4)};
    std::array<Real, 4> const qz{Real(0.15), Real(0.2), Real(0.42), Real(1.55)};

    std::array<std::vector<Real>, 3> const point = run_antipodal_gradient_point_kernel(
        ctx.geometry.accessor(), ctx.boundary.accessor(), qx, qy, qz
    );
    std::array<std::vector<Real>, 3> const batch =
        run_antipodal_gradient(ctx.geometry.accessor(), ctx.boundary.accessor(), qx, qy, qz);
    expect_near_gradient(point, batch, Real(1e-6));
}

TEST_F(CudaFixture, antipodal_gradient_returns_nan_when_all_retry_axes_are_singular) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(1), Real(-1), Real(1), Real(-1)};
    std::array<Real, 4> const qy{Real(1), Real(1), Real(-1), Real(-1)};
    std::array<Real, 4> const qz{Real(1), Real(1), Real(1), Real(1)};

    std::array<std::vector<Real>, 3> const gradient =
        run_antipodal_gradient(ctx.geometry.accessor(), ctx.boundary.accessor(), qx, qy, qz);
    expect_nan_gradient(gradient);
}

TEST_F(CudaFixture, antipodal_winding_returns_nan_when_all_retry_axes_are_singular) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(1), Real(-1), Real(1), Real(-1)};
    std::array<Real, 4> const qy{Real(1), Real(1), Real(-1), Real(-1)};
    std::array<Real, 4> const qz{Real(1), Real(1), Real(1), Real(1)};

    std::vector<Real> const winding = run_antipodal(
        ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(), ctx.boundary.accessor(),
        qx, qy, qz
    );
    expect_nan_vector(winding);
}

TEST_F(CudaFixture, antipodal_rejects_missing_or_mismatched_boundary) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 1> const qx{Real(0.23)};
    std::array<Real, 1> const qy{Real(0.17)};
    std::array<Real, 1> const qz{Real(0.11)};
    gwn::gwn_device_array<Real> d_qx;
    gwn::gwn_device_array<Real> d_qy;
    gwn::gwn_device_array<Real> d_qz;
    gwn::gwn_device_array<Real> d_out;
    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());
    ASSERT_TRUE(d_out.resize(qx.size()).is_ok());

    gwn::gwn_boundary_chain_object<Index> missing;
    EXPECT_FALSE((gwn::gwn_compute_winding_number_batch_bvh_antipodal<Real, Index>(
                      ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(),
                      missing.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_out.span()
                  ))
                     .is_ok());

    auto mismatched = ctx.boundary.accessor();
    mismatched.mesh_triangle_count += 1;
    EXPECT_FALSE((gwn::gwn_compute_winding_number_batch_bvh_antipodal<Real, Index>(
                      ctx.geometry.accessor(), ctx.bvh.accessor(), ctx.aabb.accessor(), mismatched,
                      d_qx.span(), d_qy.span(), d_qz.span(), d_out.span()
                  ))
                     .is_ok());
}

TEST_F(CudaFixture, antipodal_gradient_rejects_bad_inputs) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 1> const qx{Real(0.23)};
    std::array<Real, 1> const qy{Real(0.17)};
    std::array<Real, 1> const qz{Real(0.11)};
    gwn::gwn_device_array<Real> d_qx;
    gwn::gwn_device_array<Real> d_qy;
    gwn::gwn_device_array<Real> d_qz;
    gwn::gwn_device_array<Real> d_gx;
    gwn::gwn_device_array<Real> d_gy;
    gwn::gwn_device_array<Real> d_gz;
    ASSERT_TRUE(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())).is_ok());
    ASSERT_TRUE(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())).is_ok());
    ASSERT_TRUE(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())).is_ok());
    ASSERT_TRUE(d_gx.resize(qx.size()).is_ok());
    ASSERT_TRUE(d_gy.resize(qx.size()).is_ok());
    ASSERT_TRUE(d_gz.resize(qx.size()).is_ok());

    gwn::gwn_boundary_chain_object<Index> missing;
    EXPECT_FALSE((gwn::gwn_compute_winding_gradient_batch_antipodal<Real, Index>(
                      ctx.geometry.accessor(), missing.accessor(), d_qx.span(), d_qy.span(),
                      d_qz.span(), d_gx.span(), d_gy.span(), d_gz.span()
                  ))
                     .is_ok());

    auto mismatched = ctx.boundary.accessor();
    mismatched.mesh_triangle_count += 1;
    EXPECT_FALSE((gwn::gwn_compute_winding_gradient_batch_antipodal<Real, Index>(
                      ctx.geometry.accessor(), mismatched, d_qx.span(), d_qy.span(), d_qz.span(),
                      d_gx.span(), d_gy.span(), d_gz.span()
                  ))
                     .is_ok());

    cuda::std::span<Real> const bad_output{};
    EXPECT_FALSE((gwn::gwn_compute_winding_gradient_batch_antipodal<Real, Index>(
                      ctx.geometry.accessor(), ctx.boundary.accessor(), d_qx.span(), d_qy.span(),
                      d_qz.span(), bad_output, d_gy.span(), d_gz.span()
                  ))
                     .is_ok());
}
