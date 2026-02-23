#pragma once

#include <cuda_runtime_api.h>

#include <exception>
#include <format>
#include <utility>

#include "gwn/gwn_bvh.cuh"

namespace gwn {
namespace detail {

template <class Function>
gwn_status gwn_try_translate_status(char const *const function_name, Function &&function) noexcept {
    try {
        return std::forward<Function>(function)();
    } catch (std::exception const &exception) {
        return gwn_status::internal_error(
            std::format("Unhandled std::exception in {}: {}", function_name, exception.what())
        );
    } catch (...) {
        return gwn_status::internal_error(
            std::format("Unhandled unknown exception in {}.", function_name)
        );
    }
}

template <class Accessor, class ReleaseFunction, class BuildFunction>
gwn_status gwn_replace_accessor_with_staging(
    Accessor &target_accessor, ReleaseFunction &&release_function, BuildFunction &&build_function,
    cudaStream_t const stream
) {
    Accessor staging_accessor{};
    auto cleanup_staging_accessor =
        gwn_make_scope_exit([&]() noexcept { release_function(staging_accessor, stream); });

    GWN_RETURN_ON_ERROR(build_function(staging_accessor));
    release_function(target_accessor, stream);
    target_accessor = staging_accessor;
    cleanup_staging_accessor.release();
    return gwn_status::ok();
}

} // namespace detail
} // namespace gwn
