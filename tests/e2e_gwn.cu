#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "reference_hdk.cuh"
#include "test_utils.cuh"

namespace {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::HostMesh;
constexpr int k_stack_capacity = gwn::tests::k_test_stack_capacity;
constexpr gwn::gwn_query_batch_config k_query_batch_config{
    .block_size = gwn::k_gwn_default_query_batch_block_size,
    .stack_capacity = k_stack_capacity,
};

// Query generation helpers.

std::array<std::vector<Real>, 3> make_integration_query_soa(HostMesh const &mesh) {
    Real min_x = std::numeric_limits<Real>::max();
    Real min_y = std::numeric_limits<Real>::max();
    Real min_z = std::numeric_limits<Real>::max();
    Real max_x = std::numeric_limits<Real>::lowest();
    Real max_y = std::numeric_limits<Real>::lowest();
    Real max_z = std::numeric_limits<Real>::lowest();

    for (std::size_t i = 0; i < mesh.vertex_x.size(); ++i) {
        min_x = std::min(min_x, mesh.vertex_x[i]);
        min_y = std::min(min_y, mesh.vertex_y[i]);
        min_z = std::min(min_z, mesh.vertex_z[i]);
        max_x = std::max(max_x, mesh.vertex_x[i]);
        max_y = std::max(max_y, mesh.vertex_y[i]);
        max_z = std::max(max_z, mesh.vertex_z[i]);
    }

    Real const center_x = (min_x + max_x) * Real(0.5);
    Real const center_y = (min_y + max_y) * Real(0.5);
    Real const center_z = (min_z + max_z) * Real(0.5);
    Real const extent_x = std::max(max_x - min_x, Real(1e-2));
    Real const extent_y = std::max(max_y - min_y, Real(1e-2));
    Real const extent_z = std::max(max_z - min_z, Real(1e-2));

    std::array<std::vector<Real>, 3> soa{};
    auto append_query = [&](Real const x, Real const y, Real const z) {
        soa[0].push_back(x);
        soa[1].push_back(y);
        soa[2].push_back(z);
    };

    constexpr std::array<Real, 3> k_lattice_scale = {Real(-0.61), Real(0.13), Real(0.77)};
    for (Real const sx : k_lattice_scale) {
        for (Real const sy : k_lattice_scale) {
            for (Real const sz : k_lattice_scale) {
                append_query(
                    center_x + sx * extent_x, center_y + sy * extent_y, center_z + sz * extent_z
                );
            }
        }
    }

    constexpr std::array<Real, 2> k_far_scale = {Real(-2.5), Real(2.5)};
    for (Real const scale : k_far_scale) {
        append_query(center_x + scale * extent_x, center_y, center_z);
        append_query(center_x, center_y + scale * extent_y, center_z);
        append_query(center_x, center_y, center_z + scale * extent_z);
    }

    constexpr std::array<Real, 2> k_corner_scale = {Real(-1.8), Real(1.8)};
    for (Real const sx : k_corner_scale) {
        for (Real const sy : k_corner_scale) {
            for (Real const sz : k_corner_scale) {
                append_query(
                    center_x + sx * extent_x, center_y + sy * extent_y, center_z + sz * extent_z
                );
            }
        }
    }

    return soa;
}

// Integration Tests.

TEST(gwn_e2e, taylor_orders_match_hdk_on_mesh_directory) {
    std::vector<std::filesystem::path> const model_paths = gwn::tests::collect_mesh_paths();
    ASSERT_FALSE(model_paths.empty());

    constexpr Real k_accuracy_scale = Real(2);
    constexpr Real k_order0_epsilon = Real(5e-2);
    constexpr Real k_order1_epsilon = Real(3e-2);
    constexpr Real k_order2_epsilon = Real(2e-2);

    std::size_t tested_model_count = 0;
    std::size_t input_skip_count = 0;
    for (std::filesystem::path const &model_path : model_paths) {
        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value()) {
            std::cerr << "smallgwn input skip: failed to parse " << model_path << '\n';
            ++input_skip_count;
            continue;
        }
        HostMesh const &mesh = *maybe_mesh;
        auto const query_soa = make_integration_query_soa(mesh);
        std::size_t const query_count = query_soa[0].size();
        ASSERT_GT(query_count, 0u);

        std::vector<Real> cpu_order0(query_count, Real(0));
        std::vector<Real> cpu_order1(query_count, Real(0));
        std::vector<Real> cpu_order2(query_count, Real(0));
        ASSERT_TRUE((gwn::tests::reference_winding_number_batch_hdk_taylor<Real, Index>(
                         cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
                         cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
                         cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
                         cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
                         cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
                         cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()),
                         cuda::std::span<Real const>(query_soa[0].data(), query_soa[0].size()),
                         cuda::std::span<Real const>(query_soa[1].data(), query_soa[1].size()),
                         cuda::std::span<Real const>(query_soa[2].data(), query_soa[2].size()),
                         cuda::std::span<Real>(cpu_order0.data(), cpu_order0.size()), 0,
                         k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::tests::reference_winding_number_batch_hdk_taylor<Real, Index>(
                         cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
                         cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
                         cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
                         cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
                         cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
                         cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()),
                         cuda::std::span<Real const>(query_soa[0].data(), query_soa[0].size()),
                         cuda::std::span<Real const>(query_soa[1].data(), query_soa[1].size()),
                         cuda::std::span<Real const>(query_soa[2].data(), query_soa[2].size()),
                         cuda::std::span<Real>(cpu_order1.data(), cpu_order1.size()), 1,
                         k_accuracy_scale
        )
                         .is_ok()));
        ASSERT_TRUE((gwn::tests::reference_winding_number_batch_hdk_taylor<Real, Index>(
                         cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
                         cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
                         cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
                         cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
                         cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
                         cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()),
                         cuda::std::span<Real const>(query_soa[0].data(), query_soa[0].size()),
                         cuda::std::span<Real const>(query_soa[1].data(), query_soa[1].size()),
                         cuda::std::span<Real const>(query_soa[2].data(), query_soa[2].size()),
                         cuda::std::span<Real>(cpu_order2.data(), cpu_order2.size()), 2,
                         k_accuracy_scale
        )
                         .is_ok()));

        gwn::gwn_geometry_object<Real, Index> geometry;
        gwn::gwn_status const upload_status = gwn::gwn_upload_geometry(
            geometry, gwn::gwn_host_span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            gwn::gwn_host_span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            gwn::gwn_host_span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            gwn::gwn_host_span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            gwn::gwn_host_span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            gwn::gwn_host_span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size())
        );
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
        ASSERT_TRUE(upload_status.is_ok()) << gwn::tests::status_to_debug_string(upload_status);

        gwn::detail::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
        d_qx.resize(query_count);
        d_qy.resize(query_count);
        d_qz.resize(query_count);
        d_out.resize(query_count);
        d_qx.copy_from_host(cuda::std::span<Real const>(query_soa[0].data(), query_count));
        d_qy.copy_from_host(cuda::std::span<Real const>(query_soa[1].data(), query_count));
        d_qz.copy_from_host(cuda::std::span<Real const>(query_soa[2].data(), query_count));

        gwn::gwn_bvh4_object<Real, Index> bvh;
        ASSERT_TRUE((gwn::gwn_build_bvh(
                         geometry, bvh,
                         gwn::gwn_bvh_build_options{.method = gwn::gwn_bvh_build_method::k_lbvh}
        )
                         .is_ok()));

        gwn::gwn_bvh4_moment_object<0, Real, Index> data_o0;
        ASSERT_TRUE(gwn::gwn_refit_bvh_moment<0>(bvh, data_o0).is_ok());
        ASSERT_TRUE(
            (gwn::gwn_compute_winding_number_taylor_batch<0, k_query_batch_config, 4, Real, Index>(
                 bvh, data_o0, gwn::tests::device_input_span(d_qx.span()),
                 gwn::tests::device_input_span(d_qy.span()),
                 gwn::tests::device_input_span(d_qz.span()), gwn::tests::device_span(d_out.span()),
                 k_accuracy_scale
            )
                 .is_ok())
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> gpu_order0(query_count, Real(0));
        d_out.copy_to_host(cuda::std::span<Real>(gpu_order0.data(), gpu_order0.size()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        gwn::gwn_bvh4_moment_object<1, Real, Index> data_o1;
        ASSERT_TRUE(gwn::gwn_refit_bvh_moment<1>(bvh, data_o1).is_ok());
        ASSERT_TRUE(
            (gwn::gwn_compute_winding_number_taylor_batch<1, k_query_batch_config, 4, Real, Index>(
                 bvh, data_o1, gwn::tests::device_input_span(d_qx.span()),
                 gwn::tests::device_input_span(d_qy.span()),
                 gwn::tests::device_input_span(d_qz.span()), gwn::tests::device_span(d_out.span()),
                 k_accuracy_scale
            )
                 .is_ok())
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> gpu_order1(query_count, Real(0));
        d_out.copy_to_host(cuda::std::span<Real>(gpu_order1.data(), gpu_order1.size()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        gwn::gwn_bvh4_moment_object<2, Real, Index> data_o2;
        ASSERT_TRUE(gwn::gwn_refit_bvh_moment<2>(bvh, data_o2).is_ok());
        ASSERT_TRUE(
            (gwn::gwn_compute_winding_number_taylor_batch<2, k_query_batch_config, 4, Real, Index>(
                 bvh, data_o2, gwn::tests::device_input_span(d_qx.span()),
                 gwn::tests::device_input_span(d_qy.span()),
                 gwn::tests::device_input_span(d_qz.span()), gwn::tests::device_span(d_out.span()),
                 k_accuracy_scale
            )
                 .is_ok())
        );
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<Real> gpu_order2(query_count, Real(0));
        d_out.copy_to_host(cuda::std::span<Real>(gpu_order2.data(), gpu_order2.size()));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ++tested_model_count;
        for (std::size_t qi = 0; qi < query_count; ++qi) {
            EXPECT_NEAR(gpu_order0[qi], cpu_order0[qi], k_order0_epsilon) << "query index: " << qi;
            EXPECT_NEAR(gpu_order1[qi], cpu_order1[qi], k_order1_epsilon) << "query index: " << qi;
            EXPECT_NEAR(gpu_order2[qi], cpu_order2[qi], k_order2_epsilon) << "query index: " << qi;
        }
    }

    ASSERT_EQ(tested_model_count + input_skip_count, model_paths.size());
    ASSERT_GT(tested_model_count, 0u);
}

} // namespace
