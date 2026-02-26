#include <cstdlib>
#if !defined(_WIN32)
#include <csignal>
#endif

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

namespace {

__global__ void gwn_assert_device_true_path_kernel(int const value) {
    GWN_FORCE_ASSERT(value == 1, "device-value=%d", value);
}

} // namespace

static_assert(gwn::detail::gwn_assert_filename("/tmp/a/b/file.cuh") == "file.cuh");
static_assert(gwn::detail::gwn_assert_filename("C:\\tmp\\a\\b\\file.cuh") == "file.cuh");

#if defined(_WIN32)
#define GWN_EXPECT_ABORT_WITH_STDERR(statement, pattern) ASSERT_DEATH(statement, pattern)
#else
#define GWN_EXPECT_ABORT_WITH_STDERR(statement, pattern)                                    \
    ASSERT_EXIT(statement, ::testing::KilledBySignal(SIGABRT), pattern)
#endif

TEST(smallgwn_unit_assert, force_assert_true_does_not_terminate) {
    ASSERT_EXIT(
        {
            GWN_FORCE_ASSERT(true);
            std::exit(77);
        },
        ::testing::ExitedWithCode(77),
        "");
}

TEST(smallgwn_unit_assert, force_assert_false_terminates_with_default_message) {
    GWN_EXPECT_ABORT_WITH_STDERR({ GWN_FORCE_ASSERT(false); }, "Assertion \\(false\\) failed");
}

TEST(smallgwn_unit_assert, force_assert_false_terminates_with_formatted_message) {
    GWN_EXPECT_ABORT_WITH_STDERR(
        { GWN_FORCE_ASSERT(false, "value=%d", 7); },
        "Assertion \\(false\\) failed[[:space:][:print:]]*value=7");
}

#if !defined(NDEBUG)
TEST(smallgwn_unit_assert, assert_false_terminates_in_debug_builds) {
    GWN_EXPECT_ABORT_WITH_STDERR(
        { GWN_ASSERT(false, "debug-value=%d", 9); },
        "Assertion \\(false\\) failed[[:space:][:print:]]*debug-value=9");
}
#endif

TEST(smallgwn_unit_assert, force_assert_device_true_path_compiles_and_runs) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
        GTEST_SKIP() << "CUDA unavailable";

    gwn_assert_device_true_path_kernel<<<1, 1>>>(1);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

#undef GWN_EXPECT_ABORT_WITH_STDERR
