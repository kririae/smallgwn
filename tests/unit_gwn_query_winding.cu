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

class GwnQueryWindingTest : public gwn::tests::CudaFixture {};

struct asymmetric_octahedron {
    std::array<Real, 6> x{Real(1.7), Real(-0.8), Real(0.15), Real(-0.1), Real(0.2), Real(-0.2)};
    std::array<Real, 6> y{Real(0.1), Real(-0.2), Real(1.2), Real(-0.7), Real(0.1), Real(-0.1)};
    std::array<Real, 6> z{Real(0), Real(0.1), Real(-0.1), Real(0.15), Real(1.5), Real(-0.9)};
    std::array<Index, 8> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> i2{4, 4, 4, 4, 5, 5, 5, 5};
};

[[nodiscard]] gwn::gwn_status
upload_mesh(asymmetric_octahedron const &mesh, gwn::gwn_geometry_object<Real, Index> &geometry) {
    return gwn::gwn_upload_geometry(
        geometry, gwn::tests::host_span(cuda::std::span<Real const>(mesh.x)),
        gwn::tests::host_span(cuda::std::span<Real const>(mesh.y)),
        gwn::tests::host_span(cuda::std::span<Real const>(mesh.z)),
        gwn::tests::host_span(cuda::std::span<Index const>(mesh.i0)),
        gwn::tests::host_span(cuda::std::span<Index const>(mesh.i1)),
        gwn::tests::host_span(cuda::std::span<Index const>(mesh.i2))
    );
}

struct query_buffers {
    gwn::detail::gwn_device_array<Real> x{};
    gwn::detail::gwn_device_array<Real> y{};
    gwn::detail::gwn_device_array<Real> z{};
    gwn::detail::gwn_device_array<Real> output{};

    void upload(
        cuda::std::span<Real const> const host_x, cuda::std::span<Real const> const host_y,
        cuda::std::span<Real const> const host_z
    ) {
        ASSERT_EQ(host_x.size(), host_y.size());
        ASSERT_EQ(host_x.size(), host_z.size());
        x.resize(host_x.size());
        y.resize(host_y.size());
        z.resize(host_z.size());
        output.resize(host_x.size());
        x.copy_from_host(host_x);
        y.copy_from_host(host_y);
        z.copy_from_host(host_z);
    }

    [[nodiscard]] std::vector<Real> download() {
        std::vector<Real> host(output.size());
        output.copy_to_host(cuda::std::span<Real>(host.data(), host.size()));
        EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
        return host;
    }

    [[nodiscard]] gwn::gwn_device_span<Real const> x_device() const {
        return gwn::tests::device_span(x.span());
    }
    [[nodiscard]] gwn::gwn_device_span<Real const> y_device() const {
        return gwn::tests::device_span(y.span());
    }
    [[nodiscard]] gwn::gwn_device_span<Real const> z_device() const {
        return gwn::tests::device_span(z.span());
    }
    [[nodiscard]] gwn::gwn_device_span<Real> output_device() {
        return gwn::tests::device_span(output.span());
    }
};

template <int Order>
[[nodiscard]] std::vector<Real> compute_taylor(
    gwn::gwn_bvh4_object<Real, Index> const &bvh,
    gwn::gwn_bvh4_moment_object<Order, Real, Index> const &moment, query_buffers &queries,
    Real const accuracy_scale
) {
    gwn::gwn_status const status = gwn::gwn_compute_winding_number_taylor_batch<Order>(
        bvh, moment, queries.x_device(), queries.y_device(), queries.z_device(),
        queries.output_device(), accuracy_scale
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return queries.download();
}

[[nodiscard]] std::vector<Real>
compute_exact(gwn::gwn_bvh4_object<Real, Index> const &bvh, query_buffers &queries) {
    gwn::gwn_status const status = gwn::gwn_compute_winding_number_exact_batch(
        bvh, queries.x_device(), queries.y_device(), queries.z_device(), queries.output_device()
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return queries.download();
}

[[nodiscard]] std::vector<Real> reference_winding(
    asymmetric_octahedron const &mesh, cuda::std::span<Real const> const query_x,
    cuda::std::span<Real const> const query_y, cuda::std::span<Real const> const query_z
) {
    std::vector<Real> output(query_x.size());
    gwn::gwn_status const status = gwn::tests::reference_winding_number_batch<Real, Index>(
        cuda::std::span<Real const>(mesh.x), cuda::std::span<Real const>(mesh.y),
        cuda::std::span<Real const>(mesh.z), cuda::std::span<Index const>(mesh.i0),
        cuda::std::span<Index const>(mesh.i1), cuda::std::span<Index const>(mesh.i2), query_x,
        query_y, query_z, cuda::std::span<Real>(output)
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    return output;
}

template <std::size_t N>
[[nodiscard]] Real
mean_absolute_error(std::vector<Real> const &actual, std::array<Real, N> const &expected) {
    EXPECT_EQ(actual.size(), expected.size());
    Real error = Real(0);
    for (std::size_t i = 0; i < expected.size(); ++i)
        error += std::abs(actual[i] - expected[i]);
    return error / static_cast<Real>(expected.size());
}

template <std::size_t N>
[[nodiscard]] Real
mean_absolute_error(std::vector<Real> const &actual, std::vector<Real> const &expected) {
    EXPECT_EQ(actual.size(), expected.size());
    Real error = Real(0);
    for (std::size_t i = 0; i < expected.size(); ++i)
        error += std::abs(actual[i] - expected[i]);
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
    std::vector<Real> const reference = reference_winding(mesh, query_x, query_y, query_z);

    query_buffers queries{};
    queries.upload(query_x, query_y, query_z);
    // Zero accepts every non-singular child, so changes in the three Taylor orders are observable
    // independently of the hierarchy's near-field descent choices.
    std::vector<Real> const order0 = compute_taylor(bvh, moment0, queries, Real(0));
    std::vector<Real> const order1 = compute_taylor(bvh, moment1, queries, Real(0));
    std::vector<Real> const order2 = compute_taylor(bvh, moment2, queries, Real(0));

    Real const error0 = mean_absolute_error<8>(order0, reference);
    Real const error1 = mean_absolute_error<8>(order1, reference);
    Real const error2 = mean_absolute_error<8>(order2, reference);
    EXPECT_LT(error1, error0);
    EXPECT_LT(error2, error1);
    EXPECT_LT(error2, Real(2e-3));
}

} // namespace

TEST_F(GwnQueryWindingTest, exact_batch_matches_independent_cpu_reference) {
    asymmetric_octahedron const mesh{};
    gwn::gwn_geometry_object<Real, Index> geometry{};
    gwn::gwn_status const upload_status = upload_mesh(mesh, geometry);
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
    ASSERT_TRUE(upload_status.is_ok());

    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh).is_ok());

    std::array<Real, 5> const query_x{Real(0), Real(4), Real(-2), Real(0.2), Real(1.5)};
    std::array<Real, 5> const query_y{Real(0), Real(3), Real(1), Real(-1.4), Real(0.2)};
    std::array<Real, 5> const query_z{Real(0), Real(2), Real(-3), Real(2.2), Real(-1.8)};
    std::vector<Real> const reference = reference_winding(mesh, query_x, query_y, query_z);

    query_buffers queries{};
    queries.upload(query_x, query_y, query_z);
    std::vector<Real> const actual = compute_exact(bvh, queries);

    ASSERT_EQ(actual.size(), reference.size());
    for (std::size_t query_id = 0; query_id < actual.size(); ++query_id)
        EXPECT_NEAR(actual[query_id], reference[query_id], Real(2e-6)) << "query " << query_id;
}

TEST_F(GwnQueryWindingTest, far_field_orders_reduce_error_for_both_builders) {
    expect_far_field_order_progression(gwn::gwn_bvh_build_method::k_lbvh);
    expect_far_field_order_progression(gwn::gwn_bvh_build_method::k_hploc);
}

TEST_F(GwnQueryWindingTest, nan_query_coordinate_propagates_through_exact_and_taylor) {
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
    query_buffers queries{};
    queries.upload(query_x, query_y, query_z);

    std::vector<Real> const exact = compute_exact(bvh, queries);
    std::vector<Real> const taylor = compute_taylor(bvh, moment, queries, Real(0));
    ASSERT_EQ(exact.size(), 1u);
    ASSERT_EQ(taylor.size(), 1u);
    EXPECT_TRUE(std::isnan(exact[0]));
    EXPECT_TRUE(std::isnan(taylor[0]));
}

TEST_F(GwnQueryWindingTest, batch_contract_rejects_invalid_state_and_span_mismatch) {
    gwn::gwn_bvh4_object<Real, Index> empty_bvh{};
    gwn::gwn_bvh4_moment_object<0, Real, Index> empty_moment{};
    gwn::gwn_device_span<Real const> const empty_input{};
    gwn::gwn_device_span<Real> const empty_output{};
    EXPECT_EQ(
        gwn::gwn_compute_winding_number_taylor_batch<0>(
            empty_bvh, empty_moment, empty_input, empty_input, empty_input, empty_output
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );
    EXPECT_EQ(
        gwn::gwn_compute_winding_number_exact_batch(
            empty_bvh, empty_input, empty_input, empty_input, empty_output
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
        gwn::gwn_compute_winding_number_taylor_batch<0>(
            bvh, moment, empty_input, empty_input, empty_input, empty_output
        )
            .is_ok()
    );
    for (Real const invalid_scale : {Real(-1), std::numeric_limits<Real>::quiet_NaN()}) {
        EXPECT_EQ(
            gwn::gwn_compute_winding_number_taylor_batch<0>(
                bvh, moment, empty_input, empty_input, empty_input, empty_output, invalid_scale
            )
                .error(),
            gwn::gwn_error::invalid_argument
        );
    }
    EXPECT_TRUE(
        gwn::gwn_compute_winding_number_exact_batch(
            bvh, empty_input, empty_input, empty_input, empty_output
        )
            .is_ok()
    );

    std::array<Real, 1> output_storage{};
    gwn::gwn_device_span<Real> const mismatched_output(
        output_storage.data(), output_storage.size()
    );
    EXPECT_EQ(
        gwn::gwn_compute_winding_number_taylor_batch<0>(
            bvh, moment, empty_input, empty_input, empty_input, mismatched_output
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );
    EXPECT_EQ(
        gwn::gwn_compute_winding_number_exact_batch(
            bvh, empty_input, empty_input, empty_input, mismatched_output
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );
}
