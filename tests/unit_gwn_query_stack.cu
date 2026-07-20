#include <array>
#include <cstdint>
#include <limits>
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
    gwn::detail::gwn_device_array<Real> ray_dx;
    gwn::detail::gwn_device_array<Real> ray_dy;
    gwn::detail::gwn_device_array<Real> ray_dz;
    gwn::detail::gwn_device_array<int> overflow_flag;
};

struct overflow_probe {
    int *flag{};

    void __device__ operator()() const noexcept {
        if (flag != nullptr)
            *flag = 1;
    }
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

[[nodiscard]] gwn::gwn_status setup_exact_stack_bound_context(query_stack_context &ctx) {
    using Child = gwn::gwn_bvh_child<Real>;
    using Node = gwn::gwn_bvh4_node<Real>;

    std::array<Real, 12> const vx{
        Real(0), Real(1), Real(0), Real(0), Real(1), Real(0),
        Real(0), Real(1), Real(0), Real(0), Real(1), Real(0),
    };
    std::array<Real, 12> const vy{
        Real(0), Real(0), Real(1), Real(0), Real(0), Real(1),
        Real(0), Real(0), Real(1), Real(0), Real(0), Real(1),
    };
    std::array<Real, 12> const vz{
        Real(0), Real(0), Real(0), Real(1), Real(1), Real(1),
        Real(2), Real(2), Real(2), Real(3), Real(3), Real(3),
    };
    std::array<Index, 4> const i0{0, 3, 6, 9};
    std::array<Index, 4> const i1{1, 4, 7, 10};
    std::array<Index, 4> const i2{2, 5, 8, 11};

    GWN_RETURN_ON_ERROR(
        gwn::gwn_upload_geometry(
            ctx.geometry, gwn::gwn_host_span<Real const>(vx.data(), vx.size()),
            gwn::gwn_host_span<Real const>(vy.data(), vy.size()),
            gwn::gwn_host_span<Real const>(vz.data(), vz.size()),
            gwn::gwn_host_span<Index const>(i0.data(), i0.size()),
            gwn::gwn_host_span<Index const>(i1.data(), i1.size()),
            gwn::gwn_host_span<Index const>(i2.data(), i2.size())
        )
    );

    auto const encode_reference = [](std::uint64_t const offset,
                                     std::uint64_t const primitive_count) {
        return Child::k_valid_mask | offset | (primitive_count << Child::k_primitive_count_shift);
    };
    auto const set_bounds = [](Child &child, Real const min_z, Real const max_z) {
        child.bounds = {
            .min_x = Real(0),
            .min_y = Real(0),
            .min_z = min_z,
            .max_x = Real(1),
            .max_y = Real(1),
            .max_z = max_z,
        };
    };

    // The root has two internal children, each owning one two-triangle leaf. Direct descent leaves
    // one pending internal reference, so the topology-exact internal stack bound is one.
    std::array<Node, 3> nodes{};
    auto &left_internal = nodes[0].child(0);
    set_bounds(left_internal, Real(0), Real(1));
    left_internal.reference = encode_reference(1, 0);
    auto &right_internal = nodes[0].child(1);
    set_bounds(right_internal, Real(2), Real(3));
    right_internal.reference = encode_reference(2, 0);

    auto &left_leaf = nodes[1].child(0);
    set_bounds(left_leaf, Real(0), Real(1));
    left_leaf.reference = encode_reference(0, 2);
    auto &right_leaf = nodes[2].child(0);
    set_bounds(right_leaf, Real(2), Real(3));
    right_leaf.reference = encode_reference(2, 2);

    std::array<Index, 4> const primitive_indices{0, 1, 2, 3};
    std::array<gwn::gwn_bvh_triangle<Real>, 4> triangles{};
    for (std::size_t triangle_index = 0; triangle_index < triangles.size(); ++triangle_index) {
        auto &triangle = triangles[triangle_index];
        triangle.v0_z = static_cast<Real>(triangle_index);
        triangle.e1_x = Real(1);
        triangle.e2_y = Real(1);
    }

    auto &bvh = ctx.bvh.accessor();
    gwn::detail::gwn_allocate_span(bvh.nodes, nodes.size(), cudaStreamLegacy);
    gwn::detail::gwn_allocate_span(
        bvh.primitive_indices, primitive_indices.size(), cudaStreamLegacy
    );
    gwn::detail::gwn_allocate_span(bvh.triangles, triangles.size(), cudaStreamLegacy);
    gwn::detail::gwn_copy_h2d(bvh.nodes, cuda::std::span<Node const>(nodes), cudaStreamLegacy);
    gwn::detail::gwn_copy_h2d(
        bvh.primitive_indices, cuda::std::span<Index const>(primitive_indices), cudaStreamLegacy
    );
    gwn::detail::gwn_copy_h2d(
        bvh.triangles, cuda::std::span<gwn::gwn_bvh_triangle<Real> const>(triangles),
        cudaStreamLegacy
    );
    bvh.root.bounds = {
        .min_x = Real(0),
        .min_y = Real(0),
        .min_z = Real(0),
        .max_x = Real(1),
        .max_y = Real(1),
        .max_z = Real(3),
    };
    bvh.root.reference = encode_reference(0, 0);
    bvh.internal_stack_bound = 1;
    // Keep the ray batch on canonical traversal so this test covers the shared internal bound.
    bvh.packed_stack_bound = 2;
    bvh.revision = 1;

    GWN_RETURN_ON_ERROR(gwn::gwn_refit_bvh_moment<1>(ctx.bvh, ctx.moment));
    GWN_RETURN_ON_ERROR(gwn::gwn_build_boundary_chain(ctx.geometry, ctx.boundary_chain));

    std::array<Real, 1> const qx{Real(0.25)};
    std::array<Real, 1> const qy{Real(0.25)};
    std::array<Real, 1> const qz{Real(-1)};
    std::array<Real, 1> const ray_dx{Real(0)};
    std::array<Real, 1> const ray_dy{Real(0)};
    std::array<Real, 1> const ray_dz{Real(1)};
    ctx.qx.copy_from_host(cuda::std::span<Real const>(qx));
    ctx.qy.copy_from_host(cuda::std::span<Real const>(qy));
    ctx.qz.copy_from_host(cuda::std::span<Real const>(qz));
    ctx.ray_dx.copy_from_host(cuda::std::span<Real const>(ray_dx));
    ctx.ray_dy.copy_from_host(cuda::std::span<Real const>(ray_dy));
    ctx.ray_dz.copy_from_host(cuda::std::span<Real const>(ray_dz));
    ctx.out.resize(1);
    ctx.out_hit.resize(1);
    ctx.overflow_flag.resize(1);
    ctx.overflow_flag.zero();
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

TEST_F(GwnQueryStackTest, exact_internal_stack_bound_accepts_direct_child_traversal) {
    query_stack_context ctx;
    gwn::gwn_status const setup_status = setup_exact_stack_bound_context(ctx);
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(setup_status);
    ASSERT_TRUE(setup_status.is_ok()) << gwn::tests::status_to_debug_string(setup_status);
    ASSERT_EQ(ctx.bvh.accessor().internal_stack_bound, 1u);

    overflow_probe const probe{ctx.overflow_flag.data()};
    auto const qx = gwn::tests::device_input_span(ctx.qx.span());
    auto const qy = gwn::tests::device_input_span(ctx.qy.span());
    auto const qz = gwn::tests::device_input_span(ctx.qz.span());
    auto const output = gwn::tests::device_span(ctx.out.span());

    EXPECT_TRUE((gwn::gwn_compute_unsigned_distance_batch<k_stack_capacity_one_config>(
                     ctx.bvh, qx, qy, qz, output, std::numeric_limits<Real>::infinity(),
                     cudaStreamLegacy, probe
                 ))
                    .is_ok());
    EXPECT_TRUE((gwn::gwn_compute_ray_first_hit_batch<k_stack_capacity_one_config>(
                     ctx.bvh, qx, qy, qz, gwn::tests::device_input_span(ctx.ray_dx.span()),
                     gwn::tests::device_input_span(ctx.ray_dy.span()),
                     gwn::tests::device_input_span(ctx.ray_dz.span()),
                     gwn::tests::device_span(ctx.out_hit.span()), Real(0),
                     std::numeric_limits<Real>::infinity(), cudaStreamLegacy, probe
                 ))
                    .is_ok());
    EXPECT_TRUE((gwn::gwn_compute_winding_number_taylor_batch<1, k_stack_capacity_one_config>(
                     ctx.bvh, ctx.moment, qx, qy, qz, output, std::numeric_limits<Real>::infinity(),
                     cudaStreamLegacy, probe
                 ))
                    .is_ok());
    EXPECT_TRUE((gwn::gwn_compute_winding_number_antipodal_batch<k_stack_capacity_one_config>(
                     ctx.geometry, ctx.bvh, ctx.boundary_chain, qx, qy, qz, output,
                     cudaStreamLegacy, probe
                 ))
                    .is_ok());
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<int, 1> overflow{};
    ctx.overflow_flag.copy_to_host(cuda::std::span<int>(overflow));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(overflow[0], 0);
}
