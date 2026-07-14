#include <source_location>
#include <string>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_bvh_status_helpers.cuh>
#include <gwn/detail/gwn_utils.cuh>
#include <gwn/gwn_utils.cuh>

namespace {

[[nodiscard]] gwn::gwn_status return_first_error(bool const fail) {
    GWN_RETURN_ON_ERROR(
        fail ? gwn::gwn_status::invalid_argument("first error") : gwn::gwn_status::ok()
    );
    return gwn::gwn_status::ok();
}

TEST(gwn_status, factories_preserve_error_and_message_contracts) {
    gwn::gwn_status const success;
    EXPECT_TRUE(success.is_ok());
    EXPECT_EQ(success.error(), gwn::gwn_error::success);
    EXPECT_FALSE(success.has_location());

    gwn::gwn_status const invalid = gwn::gwn_status::invalid_argument("bad input");
    EXPECT_FALSE(invalid.is_ok());
    EXPECT_EQ(invalid.error(), gwn::gwn_error::invalid_argument);
    EXPECT_NE(invalid.message().find("bad input"), std::string::npos);

    gwn::gwn_status const internal = gwn::gwn_status::internal_error();
    EXPECT_EQ(internal.error(), gwn::gwn_error::internal_error);
    EXPECT_FALSE(internal.message().empty());

    static_assert(noexcept(gwn::gwn_status::ok()));
}

TEST(gwn_status, cuda_conversion_preserves_error_and_source_location) {
    EXPECT_TRUE(gwn::gwn_cuda_to_status(cudaSuccess).is_ok());

    std::source_location const location = std::source_location::current();
    gwn::gwn_status const status =
        gwn::gwn_status::cuda_runtime_error(cudaErrorInvalidValue, location);
    EXPECT_EQ(status.error(), gwn::gwn_error::cuda_runtime_error);
    ASSERT_TRUE(status.has_location());
    EXPECT_EQ(status.location().line(), location.line());
}

TEST(gwn_status, throw_if_error_uses_status_success_state) {
    EXPECT_NO_THROW(gwn::gwn_throw_if_error(gwn::gwn_status::ok()));
    EXPECT_THROW(
        gwn::gwn_throw_if_error(gwn::gwn_status::invalid_argument("bad input")), std::runtime_error
    );
}

TEST(gwn_scope_exit, move_and_release_preserve_single_callback_ownership) {
    int calls = 0;
    {
        auto first = gwn::gwn_make_scope_exit([&]() noexcept { ++calls; });
        auto second = std::move(first);
        EXPECT_EQ(calls, 0);
    }
    EXPECT_EQ(calls, 1);

    {
        auto released = gwn::gwn_make_scope_exit([&]() noexcept { ++calls; });
        released.release();
    }
    EXPECT_EQ(calls, 1);
}

TEST(gwn_status, return_on_error_propagates_the_first_failure) {
    EXPECT_TRUE(return_first_error(false).is_ok());
    gwn::gwn_status const failure = return_first_error(true);
    EXPECT_EQ(failure.error(), gwn::gwn_error::invalid_argument);
    EXPECT_NE(failure.message().find("first error"), std::string::npos);
}

TEST(gwn_status, detail_exception_boundary_preserves_error_categories) {
    auto const stored = gwn::detail::gwn_try_translate_status("stored", []() {
        throw gwn::detail::gwn_status_exception(gwn::gwn_status::invalid_argument("stored input"));
    });
    EXPECT_EQ(stored.error(), gwn::gwn_error::invalid_argument);
    EXPECT_NE(stored.message().find("stored input"), std::string::npos);

    auto const invalid = gwn::detail::gwn_try_translate_status("invalid", []() {
        throw std::invalid_argument("invalid input");
    });
    EXPECT_EQ(invalid.error(), gwn::gwn_error::invalid_argument);

    std::source_location const location = std::source_location::current();
    auto const cuda = gwn::detail::gwn_try_translate_status("cuda", [location]() {
        throw gwn::detail::gwn_cuda_exception(cudaErrorInvalidValue, "cuda test", location);
    });
    EXPECT_EQ(cuda.error(), gwn::gwn_error::cuda_runtime_error);
    EXPECT_EQ(cuda.location().line(), location.line());

    auto const unknown = gwn::detail::gwn_try_translate_status("unknown", []() { throw 1; });
    EXPECT_EQ(unknown.error(), gwn::gwn_error::internal_error);

    static_assert(noexcept(gwn::detail::gwn_try_translate_status("noexcept", []() {})));
}

} // namespace
