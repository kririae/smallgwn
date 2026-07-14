#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "reference_cpu.cuh"
#include "test_fixtures.cuh"
#include "test_utils.cuh"

namespace {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;

class GwnQueryGradientTest : public gwn::tests::CudaFixture {};

struct asymmetric_octahedron {
    std::array<Real, 6> x{Real(1.7), Real(-0.8), Real(0.15), Real(-0.1), Real(0.2), Real(-0.2)};
    std::array<Real, 6> y{Real(0.1), Real(-0.2), Real(1.2), Real(-0.7), Real(0.1), Real(-0.1)};
    std::array<Real, 6> z{Real(0), Real(0.1), Real(-0.1), Real(0.15), Real(1.5), Real(-0.9)};
    std::array<Index, 8> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> i2{4, 4, 4, 4, 5, 5, 5, 5};
};

template <class Mesh>
[[nodiscard]] gwn::gwn_status
upload_mesh(Mesh const &mesh, gwn::gwn_geometry_object<Real, Index> &geometry) {
    return gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(mesh.x), cuda::std::span<Real const>(mesh.y),
        cuda::std::span<Real const>(mesh.z), cuda::std::span<Index const>(mesh.i0),
        cuda::std::span<Index const>(mesh.i1), cuda::std::span<Index const>(mesh.i2)
    );
}

template <class Mesh>
[[nodiscard]] gwn::gwn_vec3<Real> reference_gradient(
    Mesh const &mesh, Real const qx, Real const qy, Real const qz, Real const step = Real(1e-3)
) {
    auto winding = [&](Real const x, Real const y, Real const z) {
        return gwn::tests::reference_winding_number_point<Real, Index>(
            cuda::std::span<Real const>(mesh.x), cuda::std::span<Real const>(mesh.y),
            cuda::std::span<Real const>(mesh.z), cuda::std::span<Index const>(mesh.i0),
            cuda::std::span<Index const>(mesh.i1), cuda::std::span<Index const>(mesh.i2), x, y, z
        );
    };
    Real const scale = Real(1) / (Real(2) * step);
    return {
        (winding(qx + step, qy, qz) - winding(qx - step, qy, qz)) * scale,
        (winding(qx, qy + step, qz) - winding(qx, qy - step, qz)) * scale,
        (winding(qx, qy, qz + step) - winding(qx, qy, qz - step)) * scale,
    };
}

struct gradient_buffers {
    gwn::detail::gwn_device_array<Real> x{};
    gwn::detail::gwn_device_array<Real> y{};
    gwn::detail::gwn_device_array<Real> z{};
    gwn::detail::gwn_device_array<Real> gradient_x{};
    gwn::detail::gwn_device_array<Real> gradient_y{};
    gwn::detail::gwn_device_array<Real> gradient_z{};

    void upload(
        cuda::std::span<Real const> const host_x, cuda::std::span<Real const> const host_y,
        cuda::std::span<Real const> const host_z
    ) {
        ASSERT_EQ(host_x.size(), host_y.size());
        ASSERT_EQ(host_x.size(), host_z.size());
        x.resize(host_x.size());
        y.resize(host_y.size());
        z.resize(host_z.size());
        gradient_x.resize(host_x.size());
        gradient_y.resize(host_x.size());
        gradient_z.resize(host_x.size());
        x.copy_from_host(host_x);
        y.copy_from_host(host_y);
        z.copy_from_host(host_z);
    }
};

struct host_gradients {
    std::vector<Real> x{};
    std::vector<Real> y{};
    std::vector<Real> z{};
};

template <int Order>
[[nodiscard]] host_gradients compute_gradients(
    gwn::gwn_bvh4_object<Real, Index> const &bvh,
    gwn::gwn_bvh4_moment_object<Order, Real, Index> const &moment, gradient_buffers &queries,
    Real const accuracy_scale
) {
    gwn::gwn_status const status = gwn::gwn_compute_winding_gradient_taylor_batch<Order>(
        bvh, moment, queries.x.span(), queries.y.span(), queries.z.span(),
        queries.gradient_x.span(), queries.gradient_y.span(), queries.gradient_z.span(),
        accuracy_scale
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    host_gradients output{
        std::vector<Real>(queries.x.size()), std::vector<Real>(queries.x.size()),
        std::vector<Real>(queries.x.size())
    };
    queries.gradient_x.copy_to_host(cuda::std::span<Real>(output.x.data(), output.x.size()));
    queries.gradient_y.copy_to_host(cuda::std::span<Real>(output.y.data(), output.y.size()));
    queries.gradient_z.copy_to_host(cuda::std::span<Real>(output.z.data(), output.z.size()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return output;
}

[[nodiscard]] Real
gradient_error(host_gradients const &actual, std::vector<gwn::gwn_vec3<Real>> const &expected) {
    EXPECT_EQ(actual.x.size(), expected.size());
    Real error = Real(0);
    for (std::size_t query_id = 0; query_id < expected.size(); ++query_id) {
        Real const dx = actual.x[query_id] - expected[query_id].x;
        Real const dy = actual.y[query_id] - expected[query_id].y;
        Real const dz = actual.z[query_id] - expected[query_id].z;
        error += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    return error / static_cast<Real>(expected.size());
}

void expect_far_field_order_progression(gwn::gwn_bvh_build_method const method) {
    asymmetric_octahedron const mesh{};
    gwn::gwn_geometry_object<Real, Index> geometry{};
    ASSERT_TRUE(upload_mesh(mesh, geometry).is_ok());
    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(
        gwn::gwn_build_bvh(geometry, bvh, gwn::gwn_bvh_build_options{.method = method}).is_ok()
    );
    ASSERT_TRUE(bvh.accessor().has_internal_root());

    gwn::gwn_bvh4_moment_object<0, Real, Index> moment0{};
    gwn::gwn_bvh4_moment_object<1, Real, Index> moment1{};
    gwn::gwn_bvh4_moment_object<2, Real, Index> moment2{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<0>(bvh, moment0).is_ok());
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<1>(bvh, moment1).is_ok());
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<2>(bvh, moment2).is_ok());

    std::array<Real, 8> const query_x{
        Real(2.9), Real(-2.7), Real(0.4), Real(-0.6), Real(3.2), Real(-2.8), Real(1.8), Real(-2.1),
    };
    std::array<Real, 8> const query_y{
        Real(0.4), Real(-0.8), Real(3.1), Real(-2.9), Real(2.2), Real(1.7), Real(-2.6), Real(2.4),
    };
    std::array<Real, 8> const query_z{
        Real(1.1), Real(-0.5), Real(0.8), Real(-1.2), Real(-2.3), Real(2.5), Real(2.8), Real(-2.5),
    };
    std::vector<gwn::gwn_vec3<Real>> reference;
    reference.reserve(query_x.size());
    for (std::size_t query_id = 0; query_id < query_x.size(); ++query_id) {
        reference.push_back(reference_gradient(
            mesh, query_x[query_id], query_y[query_id], query_z[query_id], Real(5e-4)
        ));
    }

    gradient_buffers queries{};
    queries.upload(query_x, query_y, query_z);
    host_gradients const order0 = compute_gradients(bvh, moment0, queries, Real(0));
    host_gradients const order1 = compute_gradients(bvh, moment1, queries, Real(0));
    host_gradients const order2 = compute_gradients(bvh, moment2, queries, Real(0));

    Real const error0 = gradient_error(order0, reference);
    Real const error1 = gradient_error(order1, reference);
    Real const error2 = gradient_error(order2, reference);
    EXPECT_LT(error1, error0);
    EXPECT_LT(error2, error1);
    EXPECT_LT(error2, Real(3e-3));
}

__global__ void evaluate_gradient_point(
    gwn::gwn_bvh4_accessor<Real, Index> const bvh,
    gwn::gwn_bvh4_moment_accessor<1, Real, Index> const moment, Real const qx, Real const qy,
    Real const qz, gwn::gwn_vec3<Real> *output
) {
    *output = gwn::gwn_winding_gradient_taylor<1>(bvh, moment, qx, qy, qz, Real(0));
}

} // namespace

TEST_F(GwnQueryGradientTest, descended_leaf_gradient_matches_finite_difference) {
    asymmetric_octahedron const mesh{};
    gwn::gwn_geometry_object<Real, Index> geometry{};
    gwn::gwn_status const upload_status = upload_mesh(mesh, geometry);
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
    ASSERT_TRUE(upload_status.is_ok());
    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh).is_ok());
    gwn::gwn_bvh4_moment_object<2, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<2>(bvh, moment).is_ok());

    std::array<Real, 4> const query_x{Real(0.3), Real(-0.2), Real(1.8), Real(-1.4)};
    std::array<Real, 4> const query_y{Real(0.2), Real(-0.3), Real(0.4), Real(0.8)};
    std::array<Real, 4> const query_z{Real(1.1), Real(-0.8), Real(0.7), Real(-0.5)};
    gradient_buffers queries{};
    queries.upload(query_x, query_y, query_z);
    host_gradients const actual = compute_gradients(bvh, moment, queries, Real(1e6));

    for (std::size_t query_id = 0; query_id < query_x.size(); ++query_id) {
        auto const expected =
            reference_gradient(mesh, query_x[query_id], query_y[query_id], query_z[query_id]);
        EXPECT_NEAR(actual.x[query_id], expected.x, Real(3e-3)) << "query " << query_id;
        EXPECT_NEAR(actual.y[query_id], expected.y, Real(3e-3)) << "query " << query_id;
        EXPECT_NEAR(actual.z[query_id], expected.z, Real(3e-3)) << "query " << query_id;
    }
}

TEST_F(GwnQueryGradientTest, far_field_orders_reduce_error_for_both_builders) {
    expect_far_field_order_progression(gwn::gwn_bvh_build_method::k_lbvh);
    expect_far_field_order_progression(gwn::gwn_bvh_build_method::k_hploc);
}

TEST_F(GwnQueryGradientTest, device_point_gradient_matches_independent_finite_difference) {
    asymmetric_octahedron const mesh{};
    gwn::gwn_geometry_object<Real, Index> geometry{};
    ASSERT_TRUE(upload_mesh(mesh, geometry).is_ok());
    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh).is_ok());
    gwn::gwn_bvh4_moment_object<1, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<1>(bvh, moment).is_ok());

    Real constexpr qx = Real(3.2);
    Real constexpr qy = Real(-1.1);
    Real constexpr qz = Real(2.4);
    gwn::detail::gwn_device_array<gwn::gwn_vec3<Real>> output{};
    output.resize(1);
    evaluate_gradient_point<<<1, 1>>>(bvh.accessor(), moment.accessor(), qx, qy, qz, output.data());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<gwn::gwn_vec3<Real>, 1> actual{};
    output.copy_to_host(cuda::std::span<gwn::gwn_vec3<Real>>(actual));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    auto const expected = reference_gradient(mesh, qx, qy, qz, Real(5e-4));
    EXPECT_NEAR(actual[0].x, expected.x, Real(3e-3));
    EXPECT_NEAR(actual[0].y, expected.y, Real(3e-3));
    EXPECT_NEAR(actual[0].z, expected.z, Real(3e-3));
}

TEST_F(GwnQueryGradientTest, nan_query_coordinate_propagates_to_every_gradient_component) {
    asymmetric_octahedron const mesh{};
    gwn::gwn_geometry_object<Real, Index> geometry{};
    ASSERT_TRUE(upload_mesh(mesh, geometry).is_ok());
    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh).is_ok());
    gwn::gwn_bvh4_moment_object<2, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<2>(bvh, moment).is_ok());

    std::array<Real, 1> const query_x{std::numeric_limits<Real>::quiet_NaN()};
    std::array<Real, 1> const query_y{Real(0.25)};
    std::array<Real, 1> const query_z{Real(3)};
    gradient_buffers queries{};
    queries.upload(query_x, query_y, query_z);
    host_gradients const actual = compute_gradients(bvh, moment, queries, Real(0));
    ASSERT_EQ(actual.x.size(), 1u);
    EXPECT_TRUE(std::isnan(actual.x[0]));
    EXPECT_TRUE(std::isnan(actual.y[0]));
    EXPECT_TRUE(std::isnan(actual.z[0]));
}

TEST_F(GwnQueryGradientTest, batch_contract_rejects_invalid_state_and_output_mismatch) {
    gwn::gwn_bvh4_object<Real, Index> empty_bvh{};
    gwn::gwn_bvh4_moment_object<0, Real, Index> empty_moment{};
    cuda::std::span<Real const> const empty_input{};
    cuda::std::span<Real> const empty_output{};
    EXPECT_EQ(
        gwn::gwn_compute_winding_gradient_taylor_batch<0>(
            empty_bvh, empty_moment, empty_input, empty_input, empty_input, empty_output,
            empty_output, empty_output
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );

    asymmetric_octahedron const mesh{};
    gwn::gwn_geometry_object<Real, Index> geometry{};
    ASSERT_TRUE(upload_mesh(mesh, geometry).is_ok());
    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh).is_ok());
    gwn::gwn_bvh4_moment_object<0, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<0>(bvh, moment).is_ok());

    EXPECT_TRUE(
        gwn::gwn_compute_winding_gradient_taylor_batch<0>(
            bvh, moment, empty_input, empty_input, empty_input, empty_output, empty_output,
            empty_output
        )
            .is_ok()
    );
    for (Real const invalid_scale : {Real(-1), std::numeric_limits<Real>::quiet_NaN()}) {
        EXPECT_EQ(
            gwn::gwn_compute_winding_gradient_taylor_batch<0>(
                bvh, moment, empty_input, empty_input, empty_input, empty_output, empty_output,
                empty_output, invalid_scale
            )
                .error(),
            gwn::gwn_error::invalid_argument
        );
    }
    std::array<Real, 1> output_storage{};
    cuda::std::span<Real> const mismatched_output(output_storage);
    EXPECT_EQ(
        gwn::gwn_compute_winding_gradient_taylor_batch<0>(
            bvh, moment, empty_input, empty_input, empty_input, mismatched_output, empty_output,
            empty_output
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );
}
