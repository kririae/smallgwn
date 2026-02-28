#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn_utils.cuh>

#include "test_fixtures.hpp"

// gwn_device_array unit tests, RAII, stream binding, resize semantics.

using gwn::tests::CudaStreamFixture;

TEST_F(CudaStreamFixture, default_constructed_is_empty) {
    gwn::gwn_device_array<float> buffer;
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.stream(), gwn::gwn_default_stream());
}

TEST_F(CudaStreamFixture, explicit_stream_constructor) {
    gwn::gwn_device_array<float> buffer(stream_a_);
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.stream(), stream_a_);
}

TEST_F(CudaStreamFixture, resize_allocates_and_binds_stream) {
    gwn::gwn_device_array<float> buffer;
    gwn::gwn_status const status = buffer.resize(64, stream_a_);
    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_EQ(buffer.size(), 64u);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.stream(), stream_a_);
    EXPECT_FALSE(buffer.empty());
}

TEST_F(CudaStreamFixture, resize_same_size_preserves_stream) {
    gwn::gwn_device_array<float> buffer;
    ASSERT_TRUE(buffer.resize(32, stream_a_).is_ok());
    EXPECT_EQ(buffer.stream(), stream_a_);

    ASSERT_TRUE(buffer.resize(32, stream_b_).is_ok());
    EXPECT_EQ(buffer.stream(), stream_a_);
    EXPECT_EQ(buffer.size(), 32u);
}

TEST_F(CudaStreamFixture, resize_to_zero_clears) {
    gwn::gwn_device_array<float> buffer;
    ASSERT_TRUE(buffer.resize(32, stream_a_).is_ok());

    gwn::gwn_status const status = buffer.resize(0, stream_b_);
    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.stream(), stream_b_);
}

TEST_F(CudaStreamFixture, resize_to_different_size) {
    gwn::gwn_device_array<float> buffer;
    ASSERT_TRUE(buffer.resize(32, stream_a_).is_ok());
    float *const old_ptr = buffer.data();

    ASSERT_TRUE(buffer.resize(128, stream_a_).is_ok());
    EXPECT_EQ(buffer.size(), 128u);
    EXPECT_NE(buffer.data(), nullptr);
    // Pointer may or may not change, but size must update.
}

TEST_F(CudaStreamFixture, clear_with_stream_rebinds) {
    gwn::gwn_device_array<float> buffer;
    ASSERT_TRUE(buffer.resize(32, stream_a_).is_ok());
    EXPECT_EQ(buffer.stream(), stream_a_);

    gwn::gwn_status const status = buffer.clear(stream_b_);
    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.stream(), stream_b_);
}

TEST_F(CudaStreamFixture, clear_no_arg_uses_bound_stream) {
    gwn::gwn_device_array<float> buffer;
    ASSERT_TRUE(buffer.resize(32, stream_a_).is_ok());

    gwn::gwn_status const status = buffer.clear();
    ASSERT_TRUE(status.is_ok()) << status.message();
    EXPECT_TRUE(buffer.empty());
}

TEST_F(CudaStreamFixture, double_clear_is_idempotent) {
    gwn::gwn_device_array<float> buffer;
    ASSERT_TRUE(buffer.resize(32, stream_a_).is_ok());

    ASSERT_TRUE(buffer.clear().is_ok());
    ASSERT_TRUE(buffer.clear().is_ok());
    EXPECT_TRUE(buffer.empty());
}

TEST_F(CudaStreamFixture, zero_fills_with_zeros) {
    gwn::gwn_device_array<float> buffer;
    ASSERT_TRUE(buffer.resize(16, stream_a_).is_ok());

    gwn::gwn_status const zero_status = buffer.zero(stream_a_);
    ASSERT_TRUE(zero_status.is_ok()) << zero_status.message();

    std::vector<float> host(16, 1.0f);
    gwn::gwn_status const copy_status =
        buffer.copy_to_host(cuda::std::span<float>(host.data(), host.size()), stream_a_);
    ASSERT_TRUE(copy_status.is_ok()) << copy_status.message();
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));

    for (float const val : host)
        EXPECT_EQ(val, 0.0f);
}

TEST_F(CudaStreamFixture, copy_from_host_and_back) {
    gwn::gwn_device_array<float> buffer;
    ASSERT_TRUE(buffer.resize(4, stream_a_).is_ok());

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    gwn::gwn_status const upload_status =
        buffer.copy_from_host(cuda::std::span<float const>(input.data(), input.size()), stream_a_);
    ASSERT_TRUE(upload_status.is_ok()) << upload_status.message();

    std::vector<float> output(4, 0.0f);
    gwn::gwn_status const download_status =
        buffer.copy_to_host(cuda::std::span<float>(output.data(), output.size()), stream_a_);
    ASSERT_TRUE(download_status.is_ok()) << download_status.message();
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));

    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_EQ(output[i], input[i]);
}

TEST_F(CudaStreamFixture, move_constructor_transfers_ownership) {
    gwn::gwn_device_array<float> src;
    ASSERT_TRUE(src.resize(32, stream_a_).is_ok());
    float *const ptr = src.data();
    std::size_t const sz = src.size();

    gwn::gwn_device_array<float> dst(std::move(src));
    EXPECT_EQ(dst.data(), ptr);
    EXPECT_EQ(dst.size(), sz);
    EXPECT_TRUE(src.empty());
    EXPECT_EQ(src.data(), nullptr);
}

TEST_F(CudaStreamFixture, move_assignment_transfers_ownership) {
    gwn::gwn_device_array<float> src;
    ASSERT_TRUE(src.resize(32, stream_a_).is_ok());
    float *const ptr = src.data();

    gwn::gwn_device_array<float> dst;
    ASSERT_TRUE(dst.resize(16, stream_b_).is_ok());

    dst = std::move(src);
    EXPECT_EQ(dst.data(), ptr);
    EXPECT_EQ(dst.size(), 32u);
    EXPECT_TRUE(src.empty());
}

TEST_F(CudaStreamFixture, span_accessors) {
    gwn::gwn_device_array<float> buffer;
    ASSERT_TRUE(buffer.resize(8, stream_a_).is_ok());

    cuda::std::span<float> mutable_span = buffer.span();
    EXPECT_EQ(mutable_span.data(), buffer.data());
    EXPECT_EQ(mutable_span.size(), 8u);

    cuda::std::span<float const> const_span =
        static_cast<gwn::gwn_device_array<float> const &>(buffer).span();
    EXPECT_EQ(const_span.data(), buffer.data());
    EXPECT_EQ(const_span.size(), 8u);
}

TEST_F(CudaStreamFixture, swap_exchanges_state) {
    gwn::gwn_device_array<float> a;
    ASSERT_TRUE(a.resize(16, stream_a_).is_ok());
    float *const a_ptr = a.data();

    gwn::gwn_device_array<float> b;
    ASSERT_TRUE(b.resize(32, stream_b_).is_ok());
    float *const b_ptr = b.data();

    swap(a, b);

    EXPECT_EQ(a.data(), b_ptr);
    EXPECT_EQ(a.size(), 32u);
    EXPECT_EQ(b.data(), a_ptr);
    EXPECT_EQ(b.size(), 16u);
}

// noexcept guarantees.

TEST(smallgwn_unit_device_array_noexcept, methods_are_noexcept) {
    gwn::gwn_device_array<float> buffer;
    static_assert(noexcept(buffer.resize(0)));
    static_assert(noexcept(buffer.clear()));
    static_assert(noexcept(buffer.zero()));
    static_assert(noexcept(buffer.copy_from_host(cuda::std::span<float const>{})));
    static_assert(noexcept(std::as_const(buffer).copy_to_host(cuda::std::span<float>{})));
}

// Null-storage argument validation.

TEST(smallgwn_unit_device_array_null_storage, cuda_malloc_null_output_pointer) {
    gwn::gwn_status const status = gwn::gwn_cuda_malloc(nullptr, sizeof(float));
    ASSERT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST(smallgwn_unit_device_array_null_storage, copy_h2d_null_destination) {
    float const source[1] = {1.0f};
    gwn::gwn_status const status = gwn::detail::gwn_copy_h2d<float>(
        cuda::std::span<float const>(static_cast<float const *>(nullptr), 1),
        cuda::std::span<float const>(source, 1), gwn::gwn_default_stream()
    );
    ASSERT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST(smallgwn_unit_device_array_null_storage, copy_h2d_null_source) {
    float const destination_storage[1] = {0.0f};
    gwn::gwn_status const status = gwn::detail::gwn_copy_h2d<float>(
        cuda::std::span<float const>(destination_storage, 1),
        cuda::std::span<float const>(static_cast<float const *>(nullptr), 1),
        gwn::gwn_default_stream()
    );
    ASSERT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST(smallgwn_unit_device_array_null_storage, copy_d2h_null_source) {
    float destination[1] = {0.0f};
    gwn::gwn_status const status = gwn::detail::gwn_copy_d2h<float>(
        cuda::std::span<float>(destination, 1),
        cuda::std::span<float const>(static_cast<float const *>(nullptr), 1),
        gwn::gwn_default_stream()
    );
    ASSERT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST(smallgwn_unit_device_array_null_storage, copy_d2h_null_destination) {
    float const source_storage[1] = {0.0f};
    gwn::gwn_status const status = gwn::detail::gwn_copy_d2h<float>(
        cuda::std::span<float>(static_cast<float *>(nullptr), 1),
        cuda::std::span<float const>(source_storage, 1), gwn::gwn_default_stream()
    );
    ASSERT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST(smallgwn_unit_device_array_null_storage, copy_d2d_null_source) {
    float destination_storage[1] = {0.0f};
    gwn::gwn_status const status = gwn::detail::gwn_copy_d2d<float>(
        cuda::std::span<float const>(destination_storage, 1),
        cuda::std::span<float const>(static_cast<float const *>(nullptr), 1),
        gwn::gwn_default_stream()
    );
    ASSERT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST(smallgwn_unit_device_array_null_storage, copy_d2d_null_destination) {
    float const source_storage[1] = {0.0f};
    gwn::gwn_status const status = gwn::detail::gwn_copy_d2d<float>(
        cuda::std::span<float const>(static_cast<float const *>(nullptr), 1),
        cuda::std::span<float const>(source_storage, 1), gwn::gwn_default_stream()
    );
    ASSERT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}
