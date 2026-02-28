#include <source_location>
#include <string>

#include <gtest/gtest.h>

#include <gwn/gwn_utils.cuh>

// gwn_status unit tests, error code paths, factory methods, noexcept guarantees.

TEST(smallgwn_unit_status, default_constructed_is_ok) {
    gwn::gwn_status const status;
    EXPECT_TRUE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::success);
}

TEST(smallgwn_unit_status, ok_factory_produces_success) {
    gwn::gwn_status const status = gwn::gwn_status::ok();
    EXPECT_TRUE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::success);
}

TEST(smallgwn_unit_status, invalid_argument_factory) {
    gwn::gwn_status const status = gwn::gwn_status::invalid_argument("bad input");
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
    EXPECT_NE(status.message().find("bad input"), std::string::npos);
}

TEST(smallgwn_unit_status, internal_error_factory_default_message) {
    gwn::gwn_status const status = gwn::gwn_status::internal_error();
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::internal_error);
    EXPECT_FALSE(status.message().empty());
}

TEST(smallgwn_unit_status, internal_error_factory_custom_message) {
    gwn::gwn_status const status = gwn::gwn_status::internal_error("custom failure");
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::internal_error);
    EXPECT_NE(status.message().find("custom failure"), std::string::npos);
}

TEST(smallgwn_unit_status, cuda_runtime_error_factory) {
    gwn::gwn_status const status = gwn::gwn_status::cuda_runtime_error(cudaErrorInvalidValue);
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::cuda_runtime_error);
    EXPECT_TRUE(status.has_location());
}

TEST(smallgwn_unit_status, cuda_runtime_error_records_source_location) {
    auto const loc = std::source_location::current();
    gwn::gwn_status const status = gwn::gwn_status::cuda_runtime_error(cudaErrorInvalidValue, loc);
    EXPECT_TRUE(status.has_location());
    std::source_location const stored_loc = status.location();
    EXPECT_EQ(stored_loc.line(), loc.line());
}

TEST(smallgwn_unit_status, ok_status_has_no_location) {
    gwn::gwn_status const status = gwn::gwn_status::ok();
    EXPECT_FALSE(status.has_location());
}

TEST(smallgwn_unit_status, cuda_to_status_success_returns_ok) {
    gwn::gwn_status const status = gwn::gwn_cuda_to_status(cudaSuccess);
    EXPECT_TRUE(status.is_ok());
}

TEST(smallgwn_unit_status, cuda_to_status_error_returns_cuda_error) {
    gwn::gwn_status const status = gwn::gwn_cuda_to_status(cudaErrorInvalidValue);
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.error(), gwn::gwn_error::cuda_runtime_error);
}

TEST(smallgwn_unit_status, throw_if_error_does_not_throw_on_ok) {
    gwn::gwn_status const status = gwn::gwn_status::ok();
    EXPECT_NO_THROW(gwn::gwn_throw_if_error(status));
}

TEST(smallgwn_unit_status, throw_if_error_throws_on_failure) {
    gwn::gwn_status const status = gwn::gwn_status::invalid_argument("test");
    EXPECT_THROW(gwn::gwn_throw_if_error(status), std::runtime_error);
}

// gwn_status noexcept guarantees.

TEST(smallgwn_unit_status, factories_are_noexcept) {
    static_assert(noexcept(gwn::gwn_status::ok()));

    // Factory functions accept by-value std::string, whose construction is not
    // noexcept.  We verify the factory itself does not add a throw‐spec by
    // passing a pre-constructed (moved) string.
    std::string msg("x");
    EXPECT_TRUE(noexcept(gwn::gwn_status::invalid_argument(std::move(msg))));
    msg = "y";
    EXPECT_TRUE(noexcept(gwn::gwn_status::internal_error(std::move(msg))));

    // cuda_runtime_error and gwn_cuda_to_status use std::format internally,
    // which may or may not be noexcept depending on the implementation.
    // Just verify they compile and don't throw for a normal error code.
    EXPECT_NO_THROW({
        auto s1 = gwn::gwn_status::cuda_runtime_error(cudaSuccess);
        (void)s1;
        auto s2 = gwn::gwn_cuda_to_status(cudaSuccess);
        (void)s2;
    });
}

// gwn_scope_exit.

TEST(smallgwn_unit_status, scope_exit_fires_on_destruction) {
    int counter = 0;
    {
        auto guard = gwn::gwn_make_scope_exit([&]() noexcept { ++counter; });
        EXPECT_EQ(counter, 0);
    }
    EXPECT_EQ(counter, 1);
}

TEST(smallgwn_unit_status, scope_exit_release_prevents_callback) {
    int counter = 0;
    {
        auto guard = gwn::gwn_make_scope_exit([&]() noexcept { ++counter; });
        guard.release();
    }
    EXPECT_EQ(counter, 0);
}

TEST(smallgwn_unit_status, scope_exit_fires_once_on_move) {
    int counter = 0;
    {
        auto guard1 = gwn::gwn_make_scope_exit([&]() noexcept { ++counter; });
        auto guard2 = std::move(guard1);
        EXPECT_EQ(counter, 0);
    }
    EXPECT_EQ(counter, 1);
}

// GWN_RETURN_ON_ERROR macro.

namespace {

gwn::gwn_status helper_return_on_error_ok() {
    GWN_RETURN_ON_ERROR(gwn::gwn_status::ok());
    return gwn::gwn_status::ok();
}

gwn::gwn_status helper_return_on_error_fail() {
    GWN_RETURN_ON_ERROR(gwn::gwn_status::invalid_argument("early exit"));
    return gwn::gwn_status::ok(); // should not reach
}

} // namespace

TEST(smallgwn_unit_status, return_on_error_passes_through_ok) {
    gwn::gwn_status const result = helper_return_on_error_ok();
    EXPECT_TRUE(result.is_ok());
}

TEST(smallgwn_unit_status, return_on_error_returns_early_on_failure) {
    gwn::gwn_status const result = helper_return_on_error_fail();
    EXPECT_FALSE(result.is_ok());
    EXPECT_EQ(result.error(), gwn::gwn_error::invalid_argument);
}
