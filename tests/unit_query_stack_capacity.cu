#include <array>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_utils.hpp"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

namespace {

struct query_stack_context {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_topology_object<Real, Index> topology;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<1, Real, Index> moment;
    gwn::gwn_boundary_chain_object<Index> boundary_chain;
    gwn::gwn_device_array<Real> qx;
    gwn::gwn_device_array<Real> qy;
    gwn::gwn_device_array<Real> qz;
    gwn::gwn_device_array<Real> out;
    gwn::gwn_device_array<Real> out_y;
    gwn::gwn_device_array<Real> out_z;
    gwn::gwn_device_array<Real> out_w;
    gwn::gwn_device_array<Index> out_index;
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

    status = gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
        ctx.geometry, ctx.topology, ctx.aabb, ctx.moment
    );
    if (!status.is_ok())
        return status;

    status = gwn::gwn_build_boundary_chain(ctx.geometry.accessor(), ctx.boundary_chain);
    if (!status.is_ok())
        return status;

    std::array<Real, 1> const qx{0.25f};
    std::array<Real, 1> const qy{0.25f};
    std::array<Real, 1> const qz{2.0f};
    status = ctx.qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    if (!status.is_ok())
        return status;
    status = ctx.qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    if (!status.is_ok())
        return status;
    status = ctx.qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    if (!status.is_ok())
        return status;
    status = ctx.out.resize(qx.size());
    if (!status.is_ok())
        return status;
    status = ctx.out_y.resize(qx.size());
    if (!status.is_ok())
        return status;
    status = ctx.out_z.resize(qx.size());
    if (!status.is_ok())
        return status;
    status = ctx.out_w.resize(qx.size());
    if (!status.is_ok())
        return status;
    return ctx.out_index.resize(qx.size());
}

} // namespace

TEST_F(CudaFixture, batch_queries_reject_stack_capacity_below_topology_depth_bound) {
    query_stack_context ctx;
    gwn::gwn_status const setup_status = setup_query_stack_context(ctx);
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(setup_status);
    ASSERT_TRUE(setup_status.is_ok()) << gwn::tests::status_to_debug_string(setup_status);

    ASSERT_TRUE(ctx.topology.accessor().has_internal_root());
    ASSERT_GT(ctx.topology.accessor().max_depth, 0u);

    expect_stack_capacity_error(
        (gwn::gwn_compute_unsigned_boundary_edge_distance_batch_bvh<Real, Index, 1>(
            ctx.geometry.accessor(), ctx.topology.accessor(), ctx.aabb.accessor(), ctx.qx.span(),
            ctx.qy.span(), ctx.qz.span(), ctx.out.span()
        ))
    );
    expect_stack_capacity_error((gwn::gwn_compute_ray_first_hit_batch_bvh<Real, Index, 1>(
        ctx.geometry.accessor(), ctx.topology.accessor(), ctx.aabb.accessor(), ctx.qx.span(),
        ctx.qy.span(), ctx.qz.span(), ctx.qx.span(), ctx.qy.span(), ctx.qz.span(), ctx.out.span(),
        ctx.out_index.span()
    )));
    expect_stack_capacity_error(
        (gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index, 1>(
            ctx.geometry.accessor(), ctx.topology.accessor(), ctx.moment.accessor(), ctx.qx.span(),
            ctx.qy.span(), ctx.qz.span(), ctx.out.span()
        ))
    );
    expect_stack_capacity_error((gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index, 1>(
        ctx.geometry.accessor(), ctx.topology.accessor(), ctx.qx.span(), ctx.qy.span(),
        ctx.qz.span(), ctx.out.span()
    )));
    expect_stack_capacity_error(
        (gwn::gwn_compute_winding_number_batch_bvh_antipodal<Real, Index, 1>(
            ctx.geometry.accessor(), ctx.topology.accessor(), ctx.aabb.accessor(),
            ctx.boundary_chain.accessor(), ctx.qx.span(), ctx.qy.span(), ctx.qz.span(),
            ctx.out.span()
        ))
    );
    expect_stack_capacity_error((gwn::gwn_compute_unsigned_distance_batch_bvh<Real, Index, 1>(
        ctx.geometry.accessor(), ctx.topology.accessor(), ctx.aabb.accessor(), ctx.qx.span(),
        ctx.qy.span(), ctx.qz.span(), ctx.out.span()
    )));
    expect_stack_capacity_error((gwn::gwn_compute_signed_distance_batch_bvh<1, Real, Index, 1>(
        ctx.geometry.accessor(), ctx.topology.accessor(), ctx.aabb.accessor(),
        ctx.moment.accessor(), ctx.qx.span(), ctx.qy.span(), ctx.qz.span(), ctx.out.span()
    )));
    expect_stack_capacity_error(
        (gwn::gwn_compute_winding_gradient_batch_bvh_taylor<1, Real, Index, 1>(
            ctx.geometry.accessor(), ctx.topology.accessor(), ctx.moment.accessor(), ctx.qx.span(),
            ctx.qy.span(), ctx.qz.span(), ctx.out.span(), ctx.out_y.span(), ctx.out_z.span()
        ))
    );
    expect_stack_capacity_error((gwn::gwn_compute_harnack_trace_batch_bvh_taylor<1, Real, Index, 1>(
        ctx.geometry.accessor(), ctx.topology.accessor(), ctx.aabb.accessor(),
        ctx.moment.accessor(), ctx.qx.span(), ctx.qy.span(), ctx.qz.span(), ctx.qx.span(),
        ctx.qy.span(), ctx.qz.span(), ctx.out.span(), ctx.out_y.span(), ctx.out_z.span(),
        ctx.out_w.span()
    )));
    expect_stack_capacity_error((gwn::gwn_compute_hybrid_trace_batch_bvh_taylor<1, Real, Index, 1>(
        ctx.geometry.accessor(), ctx.topology.accessor(), ctx.aabb.accessor(),
        ctx.moment.accessor(), ctx.qx.span(), ctx.qy.span(), ctx.qz.span(), ctx.qx.span(),
        ctx.qy.span(), ctx.qz.span(), ctx.out.span(), ctx.out_y.span(), ctx.out_z.span(),
        ctx.out_w.span(), ctx.out_index.span()
    )));
}
