#include <array>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "test_fixtures.cuh"
#include "test_utils.cuh"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using GwnQueryStackTest = gwn::tests::CudaFixture;

namespace {

struct query_stack_context {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    gwn::gwn_bvh4_moment_object<1, Real, Index> moment;
    gwn::gwn_boundary_chain_object<Index> boundary_chain;
    gwn::detail::gwn_device_array<Real> qx;
    gwn::detail::gwn_device_array<Real> qy;
    gwn::detail::gwn_device_array<Real> qz;
    gwn::detail::gwn_device_array<Real> out;
    gwn::detail::gwn_device_array<Real> out_y;
    gwn::detail::gwn_device_array<Real> out_z;
    gwn::detail::gwn_device_array<gwn::gwn_ray_first_hit_result<Real, Index>> out_hit;
};

void expect_stack_capacity_error(gwn::gwn_status const &status) {
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument)
        << gwn::tests::status_to_debug_string(status);
    EXPECT_NE(status.message().find("Traversal stack capacity"), std::string::npos)
        << gwn::tests::status_to_debug_string(status);
}

gwn::gwn_status setup_query_stack_context(query_stack_context &ctx) {
    std::array<Real, 6> const vx{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vy{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::array<Real, 6> const vz{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    std::array<Index, 8> const i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> const i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> const i2{4, 4, 4, 4, 5, 5, 5, 5};

    gwn::gwn_status status = gwn::gwn_upload_geometry(
        ctx.geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    if (!status.is_ok())
        return status;

    status = gwn::gwn_build_bvh(ctx.geometry, ctx.bvh);
    if (!status.is_ok())
        return status;

    status = gwn::gwn_refit_bvh_moment<1>(ctx.bvh, ctx.moment);
    if (!status.is_ok())
        return status;

    status = gwn::gwn_build_boundary_chain(ctx.geometry, ctx.boundary_chain);
    if (!status.is_ok())
        return status;

    std::array<Real, 1> const qx{0.25f};
    std::array<Real, 1> const qy{0.25f};
    std::array<Real, 1> const qz{2.0f};
    ctx.qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    ctx.qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    ctx.qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    ctx.out.resize(qx.size());
    ctx.out_y.resize(qx.size());
    ctx.out_z.resize(qx.size());
    ctx.out_hit.resize(qx.size());
    return gwn::gwn_status::ok();
}

} // namespace

TEST_F(GwnQueryStackTest, traversing_batch_queries_reject_capacity_below_internal_stack_bound) {
    query_stack_context ctx;
    gwn::gwn_status const setup_status = setup_query_stack_context(ctx);
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(setup_status);
    ASSERT_TRUE(setup_status.is_ok()) << gwn::tests::status_to_debug_string(setup_status);

    ASSERT_TRUE(ctx.bvh.accessor().has_internal_root());
    ASSERT_GT(ctx.bvh.accessor().internal_stack_bound, 1u);

    expect_stack_capacity_error((gwn::gwn_compute_unsigned_distance_batch<4, Real, Index, 1>(
        ctx.bvh, ctx.qx.span(), ctx.qy.span(), ctx.qz.span(), ctx.out.span()
    )));
    expect_stack_capacity_error((gwn::gwn_compute_ray_first_hit_batch<4, Real, Index, 1>(
        ctx.bvh, ctx.qx.span(), ctx.qy.span(), ctx.qz.span(), ctx.qx.span(), ctx.qy.span(),
        ctx.qz.span(), ctx.out_hit.span()
    )));
    expect_stack_capacity_error((gwn::gwn_compute_winding_number_taylor_batch<1, 4, Real, Index, 1>(
        ctx.bvh, ctx.moment, ctx.qx.span(), ctx.qy.span(), ctx.qz.span(), ctx.out.span()
    )));
    expect_stack_capacity_error((gwn::gwn_compute_winding_number_antipodal_batch<4, Real, Index, 1>(
        ctx.geometry, ctx.bvh, ctx.boundary_chain, ctx.qx.span(), ctx.qy.span(), ctx.qz.span(),
        ctx.out.span()
    )));
    expect_stack_capacity_error(
        (gwn::gwn_compute_winding_gradient_taylor_batch<1, 4, Real, Index, 1>(
            ctx.bvh, ctx.moment, ctx.qx.span(), ctx.qy.span(), ctx.qz.span(), ctx.out.span(),
            ctx.out_y.span(), ctx.out_z.span()
        ))
    );
    // Exact winding scans the triangle sequence and therefore has no stack-capacity contract.
    EXPECT_TRUE(
        gwn::gwn_compute_winding_number_exact_batch(
            ctx.bvh, ctx.qx.span(), ctx.qy.span(), ctx.qz.span(), ctx.out.span()
        )
            .is_ok()
    );
}
