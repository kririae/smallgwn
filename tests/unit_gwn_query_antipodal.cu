#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "test_fixtures.cuh"
#include "test_meshes.hpp"
#include "test_utils.cuh"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CubeMesh;
using gwn::tests::OpenCubeMesh;

namespace {

class GwnQueryAntipodalTest : public gwn::tests::CudaFixture {};

template <class Mesh> struct AntipodalContext {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    gwn::gwn_boundary_chain_object<Index> boundary;
    bool ready{false};
};

template <class Mesh>
void setup_context(
    Mesh const &mesh, AntipodalContext<Mesh> &ctx,
    gwn::gwn_bvh_build_method const method = gwn::gwn_bvh_build_method::k_lbvh
) {
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        ctx.geometry, gwn::gwn_host_span<Real const>(mesh.vx.data(), mesh.vx.size()),
        gwn::gwn_host_span<Real const>(mesh.vy.data(), mesh.vy.size()),
        gwn::gwn_host_span<Real const>(mesh.vz.data(), mesh.vz.size()),
        gwn::gwn_host_span<Index const>(mesh.i0.data(), mesh.i0.size()),
        gwn::gwn_host_span<Index const>(mesh.i1.data(), mesh.i1.size()),
        gwn::gwn_host_span<Index const>(mesh.i2.data(), mesh.i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    status =
        gwn::gwn_build_bvh(ctx.geometry, ctx.bvh, gwn::gwn_bvh_build_options{.method = method});
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    status = gwn::gwn_build_boundary_chain(ctx.geometry, ctx.boundary);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ctx.ready = true;
}

std::vector<Real> run_exact(
    gwn::gwn_bvh4_object<Real, Index> const &bvh, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_out;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_out.resize(qx.size());

    gwn::gwn_status const status = gwn::gwn_compute_winding_number_exact_batch(
        bvh, gwn::tests::device_input_span(d_qx.span()), gwn::tests::device_input_span(d_qy.span()),
        gwn::tests::device_input_span(d_qz.span()), gwn::tests::device_span(d_out.span())
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> out(qx.size(), Real(0));
    d_out.copy_to_host(cuda::std::span<Real>(out.data(), out.size()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return out;
}

std::vector<Real> run_antipodal(
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_object<Real, Index> const &bvh,
    gwn::gwn_boundary_chain_object<Index> const &boundary, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_out;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_out.resize(qx.size());

    gwn::gwn_status const status = gwn::gwn_compute_winding_number_antipodal_batch(
        geometry, bvh, boundary, gwn::tests::device_input_span(d_qx.span()),
        gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
        gwn::tests::device_span(d_out.span())
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> out(qx.size(), Real(0));
    d_out.copy_to_host(cuda::std::span<Real>(out.data(), out.size()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return out;
}

std::array<std::vector<Real>, 3> run_antipodal_gradient(
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_boundary_chain_object<Index> const &boundary, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_gx;
    gwn::detail::gwn_device_array<Real> d_gy;
    gwn::detail::gwn_device_array<Real> d_gz;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_gx.resize(qx.size());
    d_gy.resize(qx.size());
    d_gz.resize(qx.size());

    gwn::gwn_status const status = gwn::gwn_compute_winding_gradient_antipodal_batch(
        geometry, boundary, gwn::tests::device_input_span(d_qx.span()),
        gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
        gwn::tests::device_span(d_gx.span()), gwn::tests::device_span(d_gy.span()),
        gwn::tests::device_span(d_gz.span())
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<std::vector<Real>, 3> out{
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
    };
    d_gx.copy_to_host(cuda::std::span<Real>(out[0].data(), out[0].size()));
    d_gy.copy_to_host(cuda::std::span<Real>(out[1].data(), out[1].size()));
    d_gz.copy_to_host(cuda::std::span<Real>(out[2].data(), out[2].size()));
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
        gwn::gwn_vec3<Real> const gradient = gwn::gwn_winding_gradient_antipodal(
            geometry, boundary, query_x[query_id], query_y[query_id], query_z[query_id]
        );
        out_x[query_id] = gradient.x;
        out_y[query_id] = gradient.y;
        out_z[query_id] = gradient.z;
    }
};

struct antipodal_overflow_probe {
    int *flag{};

    void __device__ operator()() const noexcept {
        if (flag != nullptr)
            *flag = 1;
    }
};

template <int StackCapacity> struct antipodal_point_functor {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_accessor<Real, Index> bvh{};
    gwn::gwn_boundary_chain_accessor<Index> boundary{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out{};
    cuda::std::span<int> overflow_flag{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        out[query_id] = gwn::gwn_winding_number_antipodal<
            4, Real, Index, StackCapacity, antipodal_overflow_probe>(
            geometry, bvh, boundary, query_x[query_id], query_y[query_id], query_z[query_id],
            antipodal_overflow_probe{overflow_flag.data() + query_id}
        );
    }
};

template <int StackCapacity = 64>
[[nodiscard]] std::pair<std::vector<Real>, std::vector<int>> run_antipodal_point_kernel(
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_object<Real, Index> const &bvh,
    gwn::gwn_boundary_chain_object<Index> const &boundary, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::detail::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    gwn::detail::gwn_device_array<int> d_overflow;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_out.resize(qx.size());
    d_overflow.resize(qx.size());
    d_overflow.zero();
    EXPECT_TRUE((gwn::detail::gwn_launch_linear_kernel<gwn::detail::k_gwn_default_block_size>(
                     qx.size(),
                     antipodal_point_functor<StackCapacity>{
                         geometry.accessor(), bvh.accessor(), boundary.accessor(), d_qx.span(),
                         d_qy.span(), d_qz.span(), d_out.span(), d_overflow.span()
                     }
                 ))
                    .is_ok());
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> out(qx.size());
    std::vector<int> overflow(qx.size());
    d_out.copy_to_host(cuda::std::span<Real>(out.data(), out.size()));
    d_overflow.copy_to_host(cuda::std::span<int>(overflow.data(), overflow.size()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return {out, overflow};
}

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
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_gx;
    gwn::detail::gwn_device_array<Real> d_gy;
    gwn::detail::gwn_device_array<Real> d_gz;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_gx.resize(qx.size());
    d_gy.resize(qx.size());
    d_gz.resize(qx.size());

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
    d_gx.copy_to_host(cuda::std::span<Real>(out[0].data(), out[0].size()));
    d_gy.copy_to_host(cuda::std::span<Real>(out[1].data(), out[1].size()));
    d_gz.copy_to_host(cuda::std::span<Real>(out[2].data(), out[2].size()));
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

TEST_F(GwnQueryAntipodalTest, antipodal_closed_mesh_matches_existing_winding) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.23), Real(2.2), Real(-0.41), Real(-2.3)};
    std::array<Real, 4> const qy{Real(0.17), Real(0.3), Real(0.26), Real(-0.4)};
    std::array<Real, 4> const qz{Real(0.11), Real(0.2), Real(1.7), Real(0.5)};

    expect_near_vector(
        run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz),
        run_exact(ctx.bvh, qx, qy, qz)
    );
}

TEST_F(GwnQueryAntipodalTest, antipodal_triangle_crossing_marks_projected_origin_edge_as_singular) {
    gwn::detail::gwn_device_array<int> d_status;
    gwn::detail::gwn_device_array<Real> d_crossing_sign;
    d_status.resize(1);
    d_crossing_sign.resize(1);
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<1>(
                     1, antipodal_triangle_crossing_functor{d_status.span(), d_crossing_sign.span()}
                 ))
                    .is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<int, 1> status{};
    std::array<Real, 1> crossing_sign{};
    d_status.copy_to_host(cuda::std::span<int>(status.data(), status.size()));
    d_crossing_sign.copy_to_host(cuda::std::span<Real>(crossing_sign.data(), crossing_sign.size()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_EQ(status[0], static_cast<int>(gwn::detail::gwn_antipodal_axis_result::k_singular));
    EXPECT_EQ(crossing_sign[0], Real(0));
}

TEST_F(GwnQueryAntipodalTest, antipodal_open_mesh_matches_existing_winding) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.23), Real(2.2), Real(-0.41), Real(-2.3)};
    std::array<Real, 4> const qy{Real(0.17), Real(0.3), Real(0.26), Real(-0.4)};
    std::array<Real, 4> const qz{Real(0.11), Real(0.2), Real(1.7), Real(0.5)};

    expect_near_vector(
        run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz),
        run_exact(ctx.bvh, qx, qy, qz)
    );
}

TEST_F(GwnQueryAntipodalTest, antipodal_open_mesh_matches_on_collinear_boundary_projections) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(2), Real(2), Real(-2), Real(-2)};
    std::array<Real, 4> const qy{Real(-1), Real(1), Real(-1), Real(1)};
    std::array<Real, 4> const qz{Real(2), Real(2), Real(2), Real(2)};

    expect_near_vector(
        run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz),
        run_exact(ctx.bvh, qx, qy, qz)
    );
}

TEST_F(GwnQueryAntipodalTest, antipodal_gradient_matches_finite_difference) {
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

        std::vector<Real> const value_plus =
            run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx_plus, qy_plus, qz_plus);
        std::vector<Real> const value_minus =
            run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx_minus, qy_minus, qz_minus);
        for (std::size_t i = 0; i < value_plus.size(); ++i)
            expected[axis][i] = (value_plus[i] - value_minus[i]) / (Real(2) * eps);
    }

    std::array<std::vector<Real>, 3> const gradient =
        run_antipodal_gradient(ctx.geometry, ctx.boundary, qx, qy, qz);
    expect_near_gradient(gradient, expected, Real(6e-3));
}

TEST_F(GwnQueryAntipodalTest, antipodal_gradient_retries_axis_when_boundary_projects_singular) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(1), Real(1), Real(1), Real(1)};
    std::array<Real, 4> const qy{Real(0.25), Real(-0.25), Real(0.5), Real(-0.5)};
    std::array<Real, 4> const qz{Real(0.2), Real(0.2), Real(0.4), Real(0.4)};

    std::array<std::vector<Real>, 3> const gradient =
        run_antipodal_gradient(ctx.geometry, ctx.boundary, qx, qy, qz);
    expect_finite_gradient(gradient);

    EXPECT_NEAR(gradient[0][0], gradient[0][1], Real(1e-6));
    EXPECT_NEAR(gradient[1][0], -gradient[1][1], Real(1e-6));
    EXPECT_NEAR(gradient[2][0], gradient[2][1], Real(1e-6));

    EXPECT_NEAR(gradient[0][2], gradient[0][3], Real(1e-6));
    EXPECT_NEAR(gradient[1][2], -gradient[1][3], Real(1e-6));
    EXPECT_NEAR(gradient[2][2], gradient[2][3], Real(1e-6));
}

TEST_F(GwnQueryAntipodalTest, antipodal_gradient_point_matches_batch) {
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
        run_antipodal_gradient(ctx.geometry, ctx.boundary, qx, qy, qz);
    expect_near_gradient(point, batch, Real(1e-6));
}

TEST_F(GwnQueryAntipodalTest, antipodal_winding_point_matches_batch) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.21), Real(1.3), Real(-1.4), Real(0.35)};
    std::array<Real, 4> const qy{Real(0.18), Real(0.25), Real(-0.33), Real(1.4)};
    std::array<Real, 4> const qz{Real(0.15), Real(0.2), Real(0.42), Real(1.55)};
    auto const [point, overflow] =
        run_antipodal_point_kernel(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz);
    std::vector<Real> const batch = run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz);
    expect_near_vector(point, batch);
    EXPECT_EQ(overflow, (std::vector<int>{0, 0, 0, 0}));
}

TEST_F(GwnQueryAntipodalTest, antipodal_device_query_reports_stack_overflow) {
    constexpr std::size_t triangle_count = 32;
    std::vector<Real> vx, vy, vz;
    std::vector<Index> i0, i1, i2;
    vx.reserve(3 * triangle_count);
    vy.reserve(3 * triangle_count);
    vz.reserve(3 * triangle_count);
    i0.reserve(triangle_count);
    i1.reserve(triangle_count);
    i2.reserve(triangle_count);
    for (std::size_t triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
        Index const first_vertex = static_cast<Index>(vx.size());
        Real const z = static_cast<Real>(triangle_id);
        vx.insert(vx.end(), {Real(0), Real(1), Real(0)});
        vy.insert(vy.end(), {Real(0), Real(0), Real(1)});
        vz.insert(vz.end(), {z, z, z});
        i0.push_back(first_vertex);
        i1.push_back(first_vertex + Index(1));
        i2.push_back(first_vertex + Index(2));
    }

    gwn::gwn_geometry_object<Real, Index> geometry;
    ASSERT_TRUE(
        gwn::gwn_upload_geometry(
            geometry, gwn::gwn_host_span<Real const>(vx.data(), vx.size()),
            gwn::gwn_host_span<Real const>(vy.data(), vy.size()),
            gwn::gwn_host_span<Real const>(vz.data(), vz.size()),
            gwn::gwn_host_span<Index const>(i0.data(), i0.size()),
            gwn::gwn_host_span<Index const>(i1.data(), i1.size()),
            gwn::gwn_host_span<Index const>(i2.data(), i2.size())
        )
            .is_ok()
    );
    gwn::gwn_bvh4_object<Real, Index> bvh;
    ASSERT_TRUE(
        gwn::gwn_build_bvh(
            geometry, bvh, gwn::gwn_bvh_build_options{.method = gwn::gwn_bvh_build_method::k_hploc}
        )
            .is_ok()
    );
    gwn::gwn_boundary_chain_object<Index> boundary;
    ASSERT_TRUE(gwn::gwn_build_boundary_chain(geometry, boundary).is_ok());
    ASSERT_GT(bvh.accessor().internal_stack_bound, 1u);

    std::array<Real, 4> const qx{Real(0.25), Real(0.25), Real(0.25), Real(0.25)};
    std::array<Real, 4> const qy{Real(0.25), Real(0.25), Real(0.25), Real(0.25)};
    std::array<Real, 4> const qz{Real(-1), Real(-2), Real(-3), Real(-4)};
    auto const [winding, overflow] =
        run_antipodal_point_kernel<1>(geometry, bvh, boundary, qx, qy, qz);
    for (std::size_t query_id = 0; query_id < winding.size(); ++query_id) {
        EXPECT_TRUE(std::isnan(winding[query_id]));
        EXPECT_EQ(overflow[query_id], 1);
    }
}

TEST_F(GwnQueryAntipodalTest, antipodal_gradient_returns_nan_when_all_retry_axes_are_singular) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(1), Real(-1), Real(1), Real(-1)};
    std::array<Real, 4> const qy{Real(1), Real(1), Real(-1), Real(-1)};
    std::array<Real, 4> const qz{Real(1), Real(1), Real(1), Real(1)};

    std::array<std::vector<Real>, 3> const gradient =
        run_antipodal_gradient(ctx.geometry, ctx.boundary, qx, qy, qz);
    expect_nan_gradient(gradient);
}

TEST_F(GwnQueryAntipodalTest, antipodal_winding_returns_nan_when_all_retry_axes_are_singular) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(1), Real(-1), Real(1), Real(-1)};
    std::array<Real, 4> const qy{Real(1), Real(1), Real(-1), Real(-1)};
    std::array<Real, 4> const qz{Real(1), Real(1), Real(1), Real(1)};

    std::vector<Real> const winding =
        run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz);
    expect_nan_vector(winding);
}

TEST_F(GwnQueryAntipodalTest, antipodal_rejects_missing_or_mismatched_boundary) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 1> const qx{Real(0.23)};
    std::array<Real, 1> const qy{Real(0.17)};
    std::array<Real, 1> const qz{Real(0.11)};
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_out;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_out.resize(qx.size());

    gwn::gwn_boundary_chain_object<Index> missing;
    EXPECT_FALSE(
        gwn::gwn_compute_winding_number_antipodal_batch(
            ctx.geometry, ctx.bvh, missing, gwn::tests::device_input_span(d_qx.span()),
            gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
            gwn::tests::device_span(d_out.span())
        )
            .is_ok()
    );

    gwn::gwn_boundary_chain_object<Index> mismatched;
    ASSERT_TRUE(gwn::gwn_build_boundary_chain(ctx.geometry, mismatched).is_ok());
    ++mismatched.accessor().mesh_vertex_count;
    EXPECT_FALSE(
        gwn::gwn_compute_winding_number_antipodal_batch(
            ctx.geometry, ctx.bvh, mismatched, gwn::tests::device_input_span(d_qx.span()),
            gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
            gwn::tests::device_span(d_out.span())
        )
            .is_ok()
    );
}

TEST_F(GwnQueryAntipodalTest, antipodal_gradient_rejects_bad_inputs) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 1> const qx{Real(0.23)};
    std::array<Real, 1> const qy{Real(0.17)};
    std::array<Real, 1> const qz{Real(0.11)};
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_gx;
    gwn::detail::gwn_device_array<Real> d_gy;
    gwn::detail::gwn_device_array<Real> d_gz;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_gx.resize(qx.size());
    d_gy.resize(qx.size());
    d_gz.resize(qx.size());

    gwn::gwn_boundary_chain_object<Index> missing;
    EXPECT_FALSE(
        gwn::gwn_compute_winding_gradient_antipodal_batch(
            ctx.geometry, missing, gwn::tests::device_input_span(d_qx.span()),
            gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
            gwn::tests::device_span(d_gx.span()), gwn::tests::device_span(d_gy.span()),
            gwn::tests::device_span(d_gz.span())
        )
            .is_ok()
    );

    gwn::gwn_boundary_chain_object<Index> mismatched;
    ASSERT_TRUE(gwn::gwn_build_boundary_chain(ctx.geometry, mismatched).is_ok());
    ++mismatched.accessor().mesh_vertex_count;
    EXPECT_FALSE(
        gwn::gwn_compute_winding_gradient_antipodal_batch(
            ctx.geometry, mismatched, gwn::tests::device_input_span(d_qx.span()),
            gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
            gwn::tests::device_span(d_gx.span()), gwn::tests::device_span(d_gy.span()),
            gwn::tests::device_span(d_gz.span())
        )
            .is_ok()
    );

    gwn::gwn_device_span<Real> const bad_output{};
    EXPECT_FALSE(
        gwn::gwn_compute_winding_gradient_antipodal_batch(
            ctx.geometry, ctx.boundary, gwn::tests::device_input_span(d_qx.span()),
            gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
            bad_output, gwn::tests::device_span(d_gy.span()), gwn::tests::device_span(d_gz.span())
        )
            .is_ok()
    );
}
