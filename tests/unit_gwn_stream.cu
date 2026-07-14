#include <cstdint>

#include <gtest/gtest.h>

#include <gwn/gwn_utils.cuh>

TEST(gwn_stream_mixin, set_and_swap_preserve_explicit_bindings) {
    gwn::gwn_stream_mixin first;
    gwn::gwn_stream_mixin second;
    EXPECT_EQ(first.stream(), gwn::gwn_default_stream());

    cudaStream_t const first_stream = reinterpret_cast<cudaStream_t>(std::uintptr_t(0xAAAA));
    cudaStream_t const second_stream = reinterpret_cast<cudaStream_t>(std::uintptr_t(0xBBBB));
    first.set_stream(first_stream);
    second.set_stream(second_stream);
    swap(first, second);

    EXPECT_EQ(first.stream(), second_stream);
    EXPECT_EQ(second.stream(), first_stream);
}
