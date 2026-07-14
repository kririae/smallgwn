#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/detail/gwn_utils.cuh>

#include "test_fixtures.cuh"

namespace {

class GwnDeviceArrayTest : public gwn::tests::CudaStreamFixture {};

TEST(gwn_device_array, default_state_has_nothrow_lifetime_operations) {
    gwn::detail::gwn_device_array<float> array;
    EXPECT_TRUE(array.empty());
    EXPECT_EQ(array.data(), nullptr);
    EXPECT_EQ(array.stream(), gwn::gwn_default_stream());

    static_assert(std::is_nothrow_destructible_v<decltype(array)>);
    static_assert(std::is_nothrow_move_constructible_v<decltype(array)>);
    static_assert(std::is_nothrow_move_assignable_v<decltype(array)>);
    static_assert(noexcept(array.clear()));
    static_assert(noexcept(array.set_stream(gwn::gwn_default_stream())));
    static_assert(!noexcept(array.resize(0)));
    static_assert(!noexcept(array.zero()));
    static_assert(!noexcept(array.copy_from_host(cuda::std::span<float const>{})));
    static_assert(!noexcept(array.copy_to_host(cuda::std::span<float>{})));
}

TEST_F(GwnDeviceArrayTest, resize_updates_binding_only_when_storage_changes) {
    gwn::detail::gwn_device_array<float> array;
    array.resize(32, stream_a_);
    EXPECT_EQ(array.size(), 32u);
    EXPECT_EQ(array.stream(), stream_a_);

    // Same-size resize is the one mutation that intentionally preserves the old binding.
    array.resize(32, stream_b_);
    EXPECT_EQ(array.stream(), stream_a_);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));
    array.resize(64, stream_b_);
    EXPECT_EQ(array.size(), 64u);
    EXPECT_EQ(array.stream(), stream_b_);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_b_));
    array.resize(0, stream_a_);
    EXPECT_TRUE(array.empty());
    EXPECT_EQ(array.stream(), stream_a_);
}

TEST_F(GwnDeviceArrayTest, copy_and_zero_round_trip_on_the_bound_stream) {
    std::array<float, 4> const input{1, 2, 3, 4};
    std::array<float, 4> output{};
    gwn::detail::gwn_device_array<float> array(stream_a_);

    array.copy_from_host(cuda::std::span<float const>(input), stream_a_);
    array.copy_to_host(cuda::std::span<float>(output), stream_a_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));
    EXPECT_EQ(output, input);

    array.zero(stream_a_);
    array.copy_to_host(cuda::std::span<float>(output), stream_a_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));
    EXPECT_EQ(output, (std::array<float, 4>{0, 0, 0, 0}));
}

TEST_F(GwnDeviceArrayTest, move_and_swap_transfer_storage_with_stream_binding) {
    gwn::detail::gwn_device_array<float> first;
    gwn::detail::gwn_device_array<float> second;
    first.resize(16, stream_a_);
    second.resize(32, stream_b_);
    float *const first_data = first.data();
    float *const second_data = second.data();

    swap(first, second);
    EXPECT_EQ(first.data(), second_data);
    EXPECT_EQ(first.stream(), stream_b_);
    EXPECT_EQ(second.data(), first_data);
    EXPECT_EQ(second.stream(), stream_a_);

    gwn::detail::gwn_device_array<float> moved(std::move(first));
    EXPECT_TRUE(first.empty());
    EXPECT_EQ(moved.data(), second_data);
    EXPECT_EQ(moved.stream(), stream_b_);
    EXPECT_EQ(moved.span().size(), 32u);

    second = std::move(moved);
    EXPECT_TRUE(moved.empty());
    EXPECT_EQ(second.data(), second_data);
    EXPECT_EQ(second.size(), 32u);
    EXPECT_EQ(second.stream(), stream_b_);
}

TEST_F(GwnDeviceArrayTest, clear_is_idempotent_and_rebinds) {
    gwn::detail::gwn_device_array<float> array;
    array.resize(8, stream_a_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));
    array.clear(stream_b_);
    array.clear();
    EXPECT_TRUE(array.empty());
    EXPECT_EQ(array.stream(), stream_b_);
}

TEST_F(GwnDeviceArrayTest, implicit_operations_use_the_bound_stream) {
    std::array<float, 2> const input{3, 7};
    std::array<float, 2> output{};
    gwn::detail::gwn_device_array<float> array(stream_a_);

    array.copy_from_host(cuda::std::span<float const>(input));
    EXPECT_EQ(array.stream(), stream_a_);
    array.copy_to_host(cuda::std::span<float>(output));
    EXPECT_EQ(array.stream(), stream_a_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));
    EXPECT_EQ(output, input);
}

TEST_F(GwnDeviceArrayTest, empty_operations_without_work_preserve_binding) {
    gwn::detail::gwn_device_array<float> array(stream_a_);

    array.zero(stream_b_);
    array.copy_from_host(cuda::std::span<float const>{}, stream_b_);
    array.copy_to_host(cuda::std::span<float>{}, stream_b_);
    EXPECT_EQ(array.stream(), stream_a_);

    array.clear(stream_b_);
    EXPECT_EQ(array.stream(), stream_b_);
}

TEST_F(GwnDeviceArrayTest, nonempty_copy_to_host_publishes_its_stream) {
    std::array<float, 1> const input{5};
    std::array<float, 1> output{};
    gwn::detail::gwn_device_array<float> array(stream_a_);
    array.copy_from_host(cuda::std::span<float const>(input), stream_a_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));

    array.copy_to_host(cuda::std::span<float>(output), stream_b_);
    EXPECT_EQ(array.stream(), stream_b_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_b_));
    EXPECT_EQ(output, input);
}

TEST_F(GwnDeviceArrayTest, invalid_copy_arguments_preserve_the_owning_state) {
    gwn::detail::gwn_device_array<float> array(stream_a_);
    array.resize(2, stream_a_);
    float *const storage = array.data();
    cuda::std::span<float const> const null_source(static_cast<float const *>(nullptr), 3);

    EXPECT_THROW(array.copy_from_host(null_source, stream_b_), std::invalid_argument);
    EXPECT_EQ(array.data(), storage);
    EXPECT_EQ(array.size(), 2u);
    EXPECT_EQ(array.stream(), stream_a_);

    std::array<float, 1> output{};
    EXPECT_THROW(
        array.copy_to_host(cuda::std::span<float>(output), stream_b_), std::invalid_argument
    );
    EXPECT_EQ(array.data(), storage);
    EXPECT_EQ(array.stream(), stream_a_);
}

TEST(gwn_device_array, oversized_resize_throws_before_changing_empty_state) {
    gwn::detail::gwn_device_array<std::uint64_t> array;
    std::size_t constexpr oversized = std::numeric_limits<std::size_t>::max();

    EXPECT_THROW(array.resize(oversized), std::invalid_argument);
    EXPECT_TRUE(array.empty());
    EXPECT_EQ(array.stream(), gwn::gwn_default_stream());
}

TEST(gwn_memory_copy, null_nonempty_storage_is_rejected) {
    float mutable_value = 0;
    float const const_value = 1;
    cuda::std::span<float> const null_output(static_cast<float *>(nullptr), 1);
    cuda::std::span<float const> const null_input(static_cast<float const *>(nullptr), 1);
    cuda::std::span<float> const output(&mutable_value, 1);
    cuda::std::span<float const> const input(&const_value, 1);

    EXPECT_THROW(
        gwn::detail::gwn_copy_h2d(null_output, input, cudaStreamLegacy), std::invalid_argument
    );
    EXPECT_THROW(
        gwn::detail::gwn_copy_h2d(output, null_input, cudaStreamLegacy), std::invalid_argument
    );
    EXPECT_THROW(
        gwn::detail::gwn_copy_d2h(null_output, input, cudaStreamLegacy), std::invalid_argument
    );
    EXPECT_THROW(
        gwn::detail::gwn_copy_d2h(output, null_input, cudaStreamLegacy), std::invalid_argument
    );
    EXPECT_THROW(
        gwn::detail::gwn_copy_d2d(null_output, input, cudaStreamLegacy), std::invalid_argument
    );
    EXPECT_THROW(
        gwn::detail::gwn_copy_d2d(output, null_input, cudaStreamLegacy), std::invalid_argument
    );
}

} // namespace
