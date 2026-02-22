#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <gwn/gwn.cuh>
#include <span>
#include <string_view>
#include <vector>

#include "reference_cpu.hpp"
#include "reference_hdk/UT_SolidAngle.h"

namespace {

bool is_cuda_runtime_unavailable(cudaError_t const result) {
    return result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
           result == cudaErrorOperatingSystem || result == cudaErrorSystemNotReady;
}

bool is_cuda_runtime_unavailable_message(std::string_view const message) {
    return message.find("cudaErrorNoDevice") != std::string_view::npos ||
           message.find("cudaErrorInsufficientDriver") != std::string_view::npos ||
           message.find("cudaErrorOperatingSystem") != std::string_view::npos ||
           message.find("cudaErrorSystemNotReady") != std::string_view::npos;
}

} // namespace

TEST(smallgwn_parity_scaffold, cpu_reference_matches_repository_triangle_oracle) {
    constexpr float k_pi = 3.14159265358979323846f;
    constexpr float k_epsilon = 1e-6f;

    std::array<float, 3> const vertex_x{1.0f, 0.0f, 0.0f};
    std::array<float, 3> const vertex_y{0.0f, 1.0f, 0.0f};
    std::array<float, 3> const vertex_z{0.0f, 0.0f, 1.0f};
    std::array<std::int64_t, 1> const tri_i0{0};
    std::array<std::int64_t, 1> const tri_i1{1};
    std::array<std::int64_t, 1> const tri_i2{2};

    std::array<float, 2> const query_x{0.0f, 0.25f};
    std::array<float, 2> const query_y{0.0f, 0.25f};
    std::array<float, 2> const query_z{0.0f, 0.25f};

    std::vector<float> reference_output(query_x.size(), 0.0f);
    gwn::gwn_status const status = gwn::tests::reference_winding_number_batch<float, std::int64_t>(
        std::span<float const>(vertex_x.data(), vertex_x.size()),
        std::span<float const>(vertex_y.data(), vertex_y.size()),
        std::span<float const>(vertex_z.data(), vertex_z.size()),
        std::span<std::int64_t const>(tri_i0.data(), tri_i0.size()),
        std::span<std::int64_t const>(tri_i1.data(), tri_i1.size()),
        std::span<std::int64_t const>(tri_i2.data(), tri_i2.size()),
        std::span<float const>(query_x.data(), query_x.size()),
        std::span<float const>(query_y.data(), query_y.size()),
        std::span<float const>(query_z.data(), query_z.size()),
        std::span<float>(reference_output.data(), reference_output.size())
    );
    ASSERT_TRUE(status.is_ok()) << status.message();

    for (std::size_t query_id = 0; query_id < query_x.size(); ++query_id) {
        HDK_Sample::UT_Vector3T<float> a;
        HDK_Sample::UT_Vector3T<float> b;
        HDK_Sample::UT_Vector3T<float> c;
        HDK_Sample::UT_Vector3T<float> q;
        a[0] = vertex_x[0];
        a[1] = vertex_y[0];
        a[2] = vertex_z[0];
        b[0] = vertex_x[1];
        b[1] = vertex_y[1];
        b[2] = vertex_z[1];
        c[0] = vertex_x[2];
        c[1] = vertex_y[2];
        c[2] = vertex_z[2];
        q[0] = query_x[query_id];
        q[1] = query_y[query_id];
        q[2] = query_z[query_id];

        float const oracle_solid_angle = HDK_Sample::UTsignedSolidAngleTri(a, b, c, q);
        float const oracle_winding = oracle_solid_angle / (4.0f * k_pi);
        EXPECT_NEAR(reference_output[query_id], oracle_winding, k_epsilon);
    }
}

TEST(smallgwn_parity_scaffold, bvh_exact_batch_matches_cpu_reference) {
    constexpr float k_epsilon = 1e-4f;

    std::array<float, 6> const vertex_x{1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 6> const vertex_y{0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    std::array<float, 6> const vertex_z{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f};

    std::array<std::int64_t, 8> const tri_i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<std::int64_t, 8> const tri_i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<std::int64_t, 8> const tri_i2{4, 4, 4, 4, 5, 5, 5, 5};

    std::array<float, 4> const query_x{0.0f, 2.0f, 0.2f, -1.5f};
    std::array<float, 4> const query_y{0.0f, 0.0f, 0.2f, 0.1f};
    std::array<float, 4> const query_z{0.0f, 0.0f, 0.2f, 0.0f};

    std::vector<float> reference_output(query_x.size(), 0.0f);
    gwn::gwn_status const reference_status =
        gwn::tests::reference_winding_number_batch<float, std::int64_t>(
            std::span<float const>(vertex_x.data(), vertex_x.size()),
            std::span<float const>(vertex_y.data(), vertex_y.size()),
            std::span<float const>(vertex_z.data(), vertex_z.size()),
            std::span<std::int64_t const>(tri_i0.data(), tri_i0.size()),
            std::span<std::int64_t const>(tri_i1.data(), tri_i1.size()),
            std::span<std::int64_t const>(tri_i2.data(), tri_i2.size()),
            std::span<float const>(query_x.data(), query_x.size()),
            std::span<float const>(query_y.data(), query_y.size()),
            std::span<float const>(query_z.data(), query_z.size()),
            std::span<float>(reference_output.data(), reference_output.size())
        );
    ASSERT_TRUE(reference_status.is_ok()) << reference_status.message();

    gwn::gwn_geometry_object<float, std::int64_t> geometry;
    gwn::gwn_status const upload_status = geometry.upload(
        cuda::std::span<float const>(vertex_x.data(), vertex_x.size()),
        cuda::std::span<float const>(vertex_y.data(), vertex_y.size()),
        cuda::std::span<float const>(vertex_z.data(), vertex_z.size()),
        cuda::std::span<std::int64_t const>(tri_i0.data(), tri_i0.size()),
        cuda::std::span<std::int64_t const>(tri_i1.data(), tri_i1.size()),
        cuda::std::span<std::int64_t const>(tri_i2.data(), tri_i2.size())
    );
    if (!upload_status.is_ok() && upload_status.error() == gwn::gwn_error::cuda_runtime_error &&
        is_cuda_runtime_unavailable_message(upload_status.message())) {
        GTEST_SKIP() << "CUDA runtime unavailable: " << upload_status.message();
    }
    ASSERT_TRUE(upload_status.is_ok()) << upload_status.message();

    gwn::gwn_bvh_object<float, std::int64_t> bvh;
    gwn::gwn_status const build_status =
        gwn::gwn_build_bvh4_lbvh<float, std::int64_t>(geometry.accessor(), bvh.accessor());
    ASSERT_TRUE(build_status.is_ok()) << build_status.message();
    ASSERT_TRUE(bvh.has_bvh());
    auto const &bvh_accessor = bvh.accessor();
    EXPECT_TRUE(bvh_accessor.is_valid());
    EXPECT_EQ(bvh_accessor.root_kind, gwn::gwn_bvh_child_kind::k_internal);
    EXPECT_EQ(bvh_accessor.primitive_indices.size(), tri_i0.size());

    float *d_query_x = nullptr;
    float *d_query_y = nullptr;
    float *d_query_z = nullptr;
    float *d_output = nullptr;
    std::size_t const query_bytes = query_x.size() * sizeof(float);
    cudaError_t result = cudaMalloc(&d_query_x, query_bytes);
    if (is_cuda_runtime_unavailable(result))
        GTEST_SKIP() << "CUDA runtime unavailable: " << cudaGetErrorString(result);
    ASSERT_EQ(cudaSuccess, result);
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_query_y, query_bytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_query_z, query_bytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_output, query_bytes));
    auto const cleanup = [&]() {
        if (d_output != nullptr) {
            (void)cudaFree(d_output);
            d_output = nullptr;
        }
        if (d_query_z != nullptr) {
            (void)cudaFree(d_query_z);
            d_query_z = nullptr;
        }
        if (d_query_y != nullptr) {
            (void)cudaFree(d_query_y);
            d_query_y = nullptr;
        }
        if (d_query_x != nullptr) {
            (void)cudaFree(d_query_x);
            d_query_x = nullptr;
        }
    };

    ASSERT_EQ(
        cudaSuccess, cudaMemcpy(d_query_x, query_x.data(), query_bytes, cudaMemcpyHostToDevice)
    );
    ASSERT_EQ(
        cudaSuccess, cudaMemcpy(d_query_y, query_y.data(), query_bytes, cudaMemcpyHostToDevice)
    );
    ASSERT_EQ(
        cudaSuccess, cudaMemcpy(d_query_z, query_z.data(), query_bytes, cudaMemcpyHostToDevice)
    );

    gwn::gwn_status const bvh_query_status =
        gwn::gwn_compute_winding_number_batch_bvh_exact<float, std::int64_t>(
            geometry.accessor(),
            bvh.accessor(),
            cuda::std::span<float const>(d_query_x, query_x.size()),
            cuda::std::span<float const>(d_query_y, query_y.size()),
            cuda::std::span<float const>(d_query_z, query_z.size()),
            cuda::std::span<float>(d_output, query_x.size())
        );
    ASSERT_TRUE(bvh_query_status.is_ok()) << bvh_query_status.message();
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float> gpu_output(query_x.size(), 0.0f);
    ASSERT_EQ(
        cudaSuccess, cudaMemcpy(gpu_output.data(), d_output, query_bytes, cudaMemcpyDeviceToHost)
    );
    cleanup();

    for (std::size_t query_id = 0; query_id < query_x.size(); ++query_id)
        EXPECT_NEAR(gpu_output[query_id], reference_output[query_id], k_epsilon);
}
