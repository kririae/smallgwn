#pragma once

#include <cuda_runtime_api.h>

#include <exception>
#include <format>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "../gwn_bvh.cuh"
#include "gwn_utils.cuh"

namespace gwn {
namespace detail {

inline constexpr std::string_view k_gwn_bvh_phase_build = "bvh.build";
inline constexpr std::string_view k_gwn_bvh_phase_build_preprocess = "bvh.build.preprocess";
inline constexpr std::string_view k_gwn_bvh_phase_build_binary_lbvh = "bvh.build.binary.lbvh";
inline constexpr std::string_view k_gwn_bvh_phase_build_binary_hploc = "bvh.build.binary.hploc";
inline constexpr std::string_view k_gwn_bvh_phase_build_collapse = "bvh.build.collapse";
inline constexpr std::string_view k_gwn_bvh_phase_build_reorder = "bvh.build.reorder";
inline constexpr std::string_view k_gwn_bvh_phase_refit = "bvh.refit";
inline constexpr std::string_view k_gwn_bvh_phase_refit_moment = "bvh.refit.moment";

[[nodiscard]] inline std::string
gwn_bvh_format_message(std::string_view const phase, std::string_view const message) {
    return std::format("{}: {}", phase, message);
}

[[noreturn]] inline void
gwn_bvh_invalid_argument(std::string_view const phase, std::string_view const message) {
    throw std::invalid_argument(gwn_bvh_format_message(phase, message));
}

[[noreturn]] inline void
gwn_bvh_internal_error(std::string_view const phase, std::string_view const message) {
    throw std::runtime_error(gwn_bvh_format_message(phase, message));
}

/// \brief Translate exceptions from a BVH host phase into its public status contract.
template <class Function>
[[nodiscard]] gwn_status
gwn_try_translate_status(char const *const function_name, Function &&function) noexcept {
    try {
        if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
            std::forward<Function>(function)();
            return gwn_status::ok();
        } else {
            return std::forward<Function>(function)();
        }
    } catch (gwn_status_exception const &exception) {
        return exception.status();
    } catch (gwn_cuda_exception const &exception) {
        return gwn_status::cuda_runtime_error(
            exception.result(), exception.location(),
            std::format("{}: {}", function_name, exception.what())
        );
    } catch (std::invalid_argument const &exception) {
        return gwn_status::invalid_argument(exception.what());
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

/// \brief Build an accessor off to the side and publish it only after complete success.
template <class Accessor, class ReleaseFunction, class BuildFunction>
void gwn_replace_accessor_with_staging(
    Accessor &target_accessor, ReleaseFunction &&release_function, BuildFunction &&build_function,
    cudaStream_t const stream
) {
    Accessor staging_accessor{};
    auto cleanup_staging_accessor =
        gwn_make_scope_exit([&]() noexcept { release_function(staging_accessor, stream); });

    build_function(staging_accessor);
    release_function(target_accessor, stream);
    target_accessor = staging_accessor;
    cleanup_staging_accessor.release();
}

} // namespace detail
} // namespace gwn
