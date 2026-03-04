#include <cstddef>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <gwn/gwn_kernel_utils.cuh>

#include "test_fixtures.hpp"
#include "test_utils.hpp"

using gwn::tests::CudaFixture;

namespace {

struct noop_linear_functor {
    __device__ void operator()(std::size_t const) const {}
};

} // namespace

TEST_F(CudaFixture, launch_linear_kernel_rejects_count_beyond_supported_range) {
    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    constexpr std::size_t k_too_many_elements =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) *
            static_cast<std::size_t>(k_block_size) +
        1u;

    gwn::gwn_status const status = gwn::detail::gwn_launch_linear_kernel<k_block_size>(
        k_too_many_elements, noop_linear_functor{}, gwn::gwn_default_stream()
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}
