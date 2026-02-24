#pragma once

#include <cuda_runtime_api.h>

#include <exception>
#include <format>
#include <string>
#include <string_view>
#include <utility>

#include "../gwn_bvh.cuh"

namespace gwn {
namespace detail {

inline constexpr std::string_view k_gwn_bvh_phase_topology_build = "bvh.topology.build";
inline constexpr std::string_view k_gwn_bvh_phase_topology_preprocess = "bvh.topology.preprocess";
inline constexpr std::string_view k_gwn_bvh_phase_topology_binary_lbvh = "bvh.topology.binary.lbvh";
inline constexpr std::string_view k_gwn_bvh_phase_topology_binary_hploc =
    "bvh.topology.binary.hploc";
inline constexpr std::string_view k_gwn_bvh_phase_topology_collapse = "bvh.topology.collapse";
inline constexpr std::string_view k_gwn_bvh_phase_refit_aabb = "bvh.refit.aabb";
inline constexpr std::string_view k_gwn_bvh_phase_refit_moment = "bvh.refit.moment";

[[nodiscard]] inline std::string
gwn_bvh_format_message(std::string_view const phase, std::string_view const message) {
    return std::format("{}: {}", phase, message);
}

[[nodiscard]] inline gwn_status
gwn_bvh_invalid_argument(std::string_view const phase, std::string_view const message) noexcept {
    return gwn_status::invalid_argument(gwn_bvh_format_message(phase, message));
}

[[nodiscard]] inline gwn_status
gwn_bvh_internal_error(std::string_view const phase, std::string_view const message) noexcept {
    return gwn_status::internal_error(gwn_bvh_format_message(phase, message));
}

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
