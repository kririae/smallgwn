#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "test_fixtures.cuh"
#include "test_utils.cuh"

namespace {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using GwnBvhMomentTest = gwn::tests::CudaFixture;
using GwnBvhMomentStreamTest = gwn::tests::CudaStreamFixture;

struct octahedron_mesh {
    std::array<Real, 6> x{Real(1), Real(-1), Real(0), Real(0), Real(0), Real(0)};
    std::array<Real, 6> y{Real(0), Real(0), Real(1), Real(-1), Real(0), Real(0)};
    std::array<Real, 6> z{Real(0), Real(0), Real(0), Real(0), Real(1), Real(-1)};
    std::array<Index, 8> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> i2{4, 4, 4, 4, 5, 5, 5, 5};
};

[[nodiscard]] gwn::gwn_status upload_octahedron(
    gwn::gwn_geometry_object<Real, Index> &geometry, cudaStream_t const stream = cudaStreamLegacy
) {
    octahedron_mesh const mesh{};
    return gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(mesh.x.data(), mesh.x.size()),
        cuda::std::span<Real const>(mesh.y.data(), mesh.y.size()),
        cuda::std::span<Real const>(mesh.z.data(), mesh.z.size()),
        cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
        cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
        cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size()), stream
    );
}

template <int Order>
void expect_moment_query_matches_exact(gwn::gwn_bvh_build_method const method) {
    gwn::gwn_geometry_object<Real, Index> geometry{};
    ASSERT_TRUE(upload_octahedron(geometry).is_ok());

    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(
        gwn::gwn_build_bvh(geometry, bvh, gwn::gwn_bvh_build_options{.method = method}).is_ok()
    );

    gwn::gwn_bvh4_moment_object<Order, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<Order>(bvh, moment).is_ok());
    ASSERT_TRUE(moment.has_data());

    std::array<Real, 2> const query_x{Real(0.13), Real(3.25)};
    std::array<Real, 2> const query_y{Real(-0.21), Real(-2.5)};
    std::array<Real, 2> const query_z{Real(0.17), Real(4.75)};
    gwn::detail::gwn_device_array<Real> d_x{};
    gwn::detail::gwn_device_array<Real> d_y{};
    gwn::detail::gwn_device_array<Real> d_z{};
    gwn::detail::gwn_device_array<Real> d_taylor{};
    gwn::detail::gwn_device_array<Real> d_exact{};
    d_x.resize(query_x.size());
    d_y.resize(query_y.size());
    d_z.resize(query_z.size());
    d_taylor.resize(query_x.size());
    d_exact.resize(query_x.size());
    d_x.copy_from_host(cuda::std::span<Real const>(query_x));
    d_y.copy_from_host(cuda::std::span<Real const>(query_y));
    d_z.copy_from_host(cuda::std::span<Real const>(query_z));

    // A deliberately large beta makes every finite cluster descend, so both public APIs evaluate
    // the same triangle set without using Taylor coefficients.
    ASSERT_TRUE((gwn::gwn_compute_winding_number_taylor_batch<Order>(
                     bvh, moment, d_x.span(), d_y.span(), d_z.span(), d_taylor.span(), Real(1e6)
    )
                     .is_ok()));
    ASSERT_TRUE((gwn::gwn_compute_winding_number_exact_batch(
                     bvh, d_x.span(), d_y.span(), d_z.span(), d_exact.span()
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 2> taylor{};
    std::array<Real, 2> exact{};
    d_taylor.copy_to_host(cuda::std::span<Real>(taylor));
    d_exact.copy_to_host(cuda::std::span<Real>(exact));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_NEAR(taylor[0], exact[0], Real(1e-6));
    EXPECT_NEAR(taylor[1], exact[1], Real(1e-6));
}

} // namespace

TEST(smallgwn_unit_bvh_taylor, default_moment_is_empty) {
    gwn::gwn_bvh4_moment_object<0, Real, Index> moment{};
    EXPECT_FALSE(moment.has_data());
}

TEST_F(GwnBvhMomentTest, all_orders_and_builders_reach_exact_leaf_fallback) {
    for (gwn::gwn_bvh_build_method const method :
         {gwn::gwn_bvh_build_method::k_lbvh, gwn::gwn_bvh_build_method::k_hploc}) {
        expect_moment_query_matches_exact<0>(method);
        expect_moment_query_matches_exact<1>(method);
        expect_moment_query_matches_exact<2>(method);
    }
}

TEST_F(GwnBvhMomentTest, moment_from_replaced_bvh_is_rejected) {
    gwn::gwn_geometry_object<Real, Index> geometry{};
    ASSERT_TRUE(upload_octahedron(geometry).is_ok());

    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh).is_ok());
    gwn::gwn_bvh4_moment_object<1, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<1>(bvh, moment).is_ok());

    // Strong replacement allocates a new triangle sequence even when counts are unchanged.
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh).is_ok());
    EXPECT_FALSE(moment.accessor().is_valid_for(bvh.accessor()));

    gwn::gwn_status const stale_status = gwn::gwn_compute_winding_number_taylor_batch<1>(
        bvh, moment, cuda::std::span<Real const>{}, cuda::std::span<Real const>{},
        cuda::std::span<Real const>{}, cuda::std::span<Real>{}
    );
    EXPECT_EQ(stale_status.error(), gwn::gwn_error::invalid_argument);

    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<1>(bvh, moment).is_ok());
    EXPECT_TRUE(moment.accessor().is_valid_for(bvh.accessor()));
}

TEST_F(GwnBvhMomentTest, moment_from_refit_bvh_is_rejected) {
    gwn::gwn_geometry_object<Real, Index> geometry{};
    ASSERT_TRUE(upload_octahedron(geometry).is_ok());

    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh).is_ok());
    gwn::gwn_bvh4_moment_object<1, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<1>(bvh, moment).is_ok());

    ASSERT_TRUE(gwn::gwn_refit_bvh(geometry, bvh).is_ok());
    EXPECT_FALSE(moment.accessor().is_valid_for(bvh.accessor()));

    gwn::gwn_status const stale_status = gwn::gwn_compute_winding_number_taylor_batch<1>(
        bvh, moment, cuda::std::span<Real const>{}, cuda::std::span<Real const>{},
        cuda::std::span<Real const>{}, cuda::std::span<Real>{}
    );
    EXPECT_EQ(stale_status.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(GwnBvhMomentTest, leaf_root_moment_uses_exact_triangle_record) {
    gwn::tests::SingleTriangleMesh const mesh{};
    gwn::gwn_geometry_object<Real, Index> geometry{};
    ASSERT_TRUE((gwn::gwn_upload_geometry(
                     geometry, cuda::std::span<Real const>(mesh.vx),
                     cuda::std::span<Real const>(mesh.vy), cuda::std::span<Real const>(mesh.vz),
                     cuda::std::span<Index const>(mesh.i0), cuda::std::span<Index const>(mesh.i1),
                     cuda::std::span<Index const>(mesh.i2)
    )
                     .is_ok()));

    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh).is_ok());
    ASSERT_TRUE(bvh.accessor().has_leaf_root());
    gwn::gwn_bvh4_moment_object<2, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<2>(bvh, moment).is_ok());
    EXPECT_TRUE(moment.has_data());
    EXPECT_TRUE(moment.accessor().is_valid_for(bvh.accessor()));

    gwn::detail::gwn_device_array<Real> query{};
    gwn::detail::gwn_device_array<Real> taylor{};
    gwn::detail::gwn_device_array<Real> exact{};
    query.resize(1);
    taylor.resize(1);
    exact.resize(1);
    std::array<Real, 1> const host_query{Real(2)};
    query.copy_from_host(cuda::std::span<Real const>(host_query));
    ASSERT_TRUE((gwn::gwn_compute_winding_number_taylor_batch<2>(
                     bvh, moment, query.span(), query.span(), query.span(), taylor.span(), Real(0)
    )
                     .is_ok()));
    ASSERT_TRUE((gwn::gwn_compute_winding_number_exact_batch(
                     bvh, query.span(), query.span(), query.span(), exact.span()
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> taylor_value{};
    std::array<Real, 1> exact_value{};
    taylor.copy_to_host(cuda::std::span<Real>(taylor_value));
    exact.copy_to_host(cuda::std::span<Real>(exact_value));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_FLOAT_EQ(taylor_value[0], exact_value[0]);
}

TEST_F(GwnBvhMomentStreamTest, clear_resets_state_before_explicit_rebind) {
    gwn::gwn_geometry_object<Real, Index> geometry{};
    ASSERT_TRUE(upload_octahedron(geometry, stream_a_).is_ok());
    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh, {}, stream_a_).is_ok());
    gwn::gwn_bvh4_moment_object<0, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<0>(bvh, moment, stream_a_).is_ok());
    ASSERT_EQ(moment.stream(), stream_a_);

    moment.clear();
    moment.set_stream(stream_b_);
    EXPECT_FALSE(moment.has_data());
    EXPECT_EQ(moment.stream(), stream_b_);
}

TEST_F(GwnBvhMomentStreamTest, moment_replacement_preserves_old_stream_ordering) {
    gwn::gwn_geometry_object<Real, Index> geometry{};
    ASSERT_TRUE(upload_octahedron(geometry, stream_a_).is_ok());
    gwn::gwn_bvh4_object<Real, Index> bvh{};
    ASSERT_TRUE(gwn::gwn_build_bvh(geometry, bvh, {}, stream_a_).is_ok());
    gwn::gwn_bvh4_moment_object<0, Real, Index> moment{};
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<0>(bvh, moment, stream_a_).is_ok());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));

    constexpr std::size_t k_query_count = 1u << 20;
    gwn::detail::gwn_device_array<Real> query_x{};
    gwn::detail::gwn_device_array<Real> query_y{};
    gwn::detail::gwn_device_array<Real> query_z{};
    gwn::detail::gwn_device_array<Real> output{};
    query_x.resize(k_query_count, stream_a_);
    query_y.resize(k_query_count, stream_a_);
    query_z.resize(k_query_count, stream_a_);
    output.resize(k_query_count, stream_a_);
    ASSERT_EQ(
        cudaSuccess, cudaMemsetAsync(query_x.data(), 0, query_x.size() * sizeof(Real), stream_a_)
    );
    ASSERT_EQ(
        cudaSuccess, cudaMemsetAsync(query_y.data(), 0, query_y.size() * sizeof(Real), stream_a_)
    );
    ASSERT_EQ(
        cudaSuccess, cudaMemsetAsync(query_z.data(), 0, query_z.size() * sizeof(Real), stream_a_)
    );

    // Keep readers of the old coefficients queued on A while replacement is built on B. The old
    // allocation must be released behind those readers on A, not on the replacement stream.
    ASSERT_TRUE((gwn::gwn_compute_winding_number_taylor_batch<0>(
                     bvh, moment, query_x.span(), query_y.span(), query_z.span(), output.span(),
                     Real(2), stream_a_
    )
                     .is_ok()));
    ASSERT_TRUE(gwn::gwn_refit_bvh_moment<0>(bvh, moment, stream_b_).is_ok());
    EXPECT_EQ(moment.stream(), stream_b_);
    EXPECT_TRUE(moment.accessor().is_valid_for(bvh.accessor()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream_b_));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));
}
