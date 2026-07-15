#include <array>
#include <cstdint>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "test_fixtures.cuh"
#include "test_utils.cuh"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using GwnQueryStackTest = gwn::tests::CudaFixture;

namespace {

constexpr gwn::gwn_query_batch_config k_stack_capacity_one_config{
    .block_size = gwn::k_gwn_default_query_batch_block_size,
    .stack_capacity = 1,
};
constexpr gwn::gwn_query_batch_config k_invalid_block_size_config{
    .block_size = 0,
    .stack_capacity = gwn::k_gwn_default_traversal_stack_capacity,
};
constexpr gwn::gwn_query_batch_config k_non_traversal_config{
    .block_size = gwn::k_gwn_default_query_batch_block_size,
    .stack_capacity = 0,
};

static_assert(gwn::gwn_query_batch_config_value<gwn::gwn_query_batch_config{}>);
static_assert(gwn::gwn_traversal_batch_config_value<gwn::gwn_query_batch_config{}>);
static_assert(!gwn::gwn_query_batch_config_value<k_invalid_block_size_config>);
static_assert(gwn::gwn_query_batch_config_value<k_non_traversal_config>);
static_assert(!gwn::gwn_traversal_batch_config_value<k_non_traversal_config>);
static_assert(!std::is_convertible_v<cuda::std::span<Real>, gwn::gwn_device_span<Real>>);
static_assert(!std::is_convertible_v<cuda::std::span<Real>, gwn::gwn_host_span<Real>>);
static_assert(!std::is_convertible_v<gwn::gwn_host_span<Real>, gwn::gwn_device_span<Real>>);
static_assert(std::is_convertible_v<gwn::gwn_device_span<Real>, gwn::gwn_device_span<Real const>>);

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
        ctx.geometry, gwn::gwn_host_span<Real const>(vx.data(), vx.size()),
        gwn::gwn_host_span<Real const>(vy.data(), vy.size()),
        gwn::gwn_host_span<Real const>(vz.data(), vz.size()),
        gwn::gwn_host_span<Index const>(i0.data(), i0.size()),
        gwn::gwn_host_span<Index const>(i1.data(), i1.size()),
        gwn::gwn_host_span<Index const>(i2.data(), i2.size())
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

    expect_stack_capacity_error(
        (gwn::gwn_compute_unsigned_distance_batch<k_stack_capacity_one_config, 4, Real, Index>(
            ctx.bvh, gwn::tests::device_input_span(ctx.qx.span()),
            gwn::tests::device_input_span(ctx.qy.span()),
            gwn::tests::device_input_span(ctx.qz.span()), gwn::tests::device_span(ctx.out.span())
        ))
    );
    expect_stack_capacity_error(
        (gwn::gwn_compute_ray_first_hit_batch<k_stack_capacity_one_config, 4, Real, Index>(
            ctx.bvh, gwn::tests::device_input_span(ctx.qx.span()),
            gwn::tests::device_input_span(ctx.qy.span()),
            gwn::tests::device_input_span(ctx.qz.span()),
            gwn::tests::device_input_span(ctx.qx.span()),
            gwn::tests::device_input_span(ctx.qy.span()),
            gwn::tests::device_input_span(ctx.qz.span()),
            gwn::tests::device_span(ctx.out_hit.span())
        ))
    );
    expect_stack_capacity_error((gwn::gwn_compute_winding_number_taylor_batch<
                                 1, k_stack_capacity_one_config, 4, Real, Index>(
        ctx.bvh, ctx.moment, gwn::tests::device_input_span(ctx.qx.span()),
        gwn::tests::device_input_span(ctx.qy.span()), gwn::tests::device_input_span(ctx.qz.span()),
        gwn::tests::device_span(ctx.out.span())
    )));
    expect_stack_capacity_error((gwn::gwn_compute_winding_number_antipodal_batch<
                                 k_stack_capacity_one_config, 4, Real, Index>(
        ctx.geometry, ctx.bvh, ctx.boundary_chain, gwn::tests::device_input_span(ctx.qx.span()),
        gwn::tests::device_input_span(ctx.qy.span()), gwn::tests::device_input_span(ctx.qz.span()),
        gwn::tests::device_span(ctx.out.span())
    )));
    expect_stack_capacity_error((gwn::gwn_compute_winding_gradient_taylor_batch<
                                 1, k_stack_capacity_one_config, 4, Real, Index>(
        ctx.bvh, ctx.moment, gwn::tests::device_input_span(ctx.qx.span()),
        gwn::tests::device_input_span(ctx.qy.span()), gwn::tests::device_input_span(ctx.qz.span()),
        gwn::tests::device_span(ctx.out.span()), gwn::tests::device_span(ctx.out_y.span()),
        gwn::tests::device_span(ctx.out_z.span())
    )));
    // Exact winding scans the triangle sequence and therefore has no stack-capacity contract.
    EXPECT_TRUE(
        (gwn::gwn_compute_winding_number_exact_batch<k_non_traversal_config, 4, Real, Index>(
             ctx.bvh, gwn::tests::device_input_span(ctx.qx.span()),
             gwn::tests::device_input_span(ctx.qy.span()),
             gwn::tests::device_input_span(ctx.qz.span()), gwn::tests::device_span(ctx.out.span())
         ))
            .is_ok()
    );
}
