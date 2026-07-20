#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

#include <gwn/gwn_utils.cuh>

#include "test_mesh_io.hpp"

namespace gwn::tests {

template <class T>
[[nodiscard]] inline gwn_host_span<T> host_span(cuda::std::span<T> const span) noexcept {
    return gwn_host_span<T>(span);
}

template <class T>
[[nodiscard]] inline gwn_device_span<T> device_span(cuda::std::span<T> const span) noexcept {
    return gwn_device_span<T>(span);
}

template <class T>
[[nodiscard]] inline gwn_device_span<T const>
device_input_span(cuda::std::span<T> const span) noexcept {
    return gwn_device_span<T const>(span.data(), span.size());
}

inline constexpr int k_test_stack_capacity = 64;

// CUDA runtime availability checks.

[[nodiscard]] inline bool is_cuda_runtime_unavailable(cudaError_t const result) noexcept {
    return result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
           result == cudaErrorOperatingSystem || result == cudaErrorSystemNotReady;
}

[[nodiscard]] inline bool
is_cuda_runtime_unavailable_message(std::string_view const message) noexcept {
    return message.find("cudaErrorNoDevice") != std::string_view::npos ||
           message.find("cudaErrorInsufficientDriver") != std::string_view::npos ||
           message.find("cudaErrorInitializationError") != std::string_view::npos ||
           message.find("cudaErrorSystemDriverMismatch") != std::string_view::npos ||
           message.find("cudaErrorOperatingSystem") != std::string_view::npos ||
           message.find("cudaErrorSystemNotReady") != std::string_view::npos ||
           message.find("cudaErrorNotSupported") != std::string_view::npos;
}

// gwn_status debug formatting.

[[nodiscard]] inline std::string status_to_debug_string(gwn::gwn_status const &status) {
    std::ostringstream out;
    out << status.message();
    if (status.has_location()) {
        std::source_location const loc = status.location();
        out << " at " << loc.file_name() << ":" << loc.line();
    }
    return out.str();
}

// CUDA skip macros for GTest.

#define SMALLGWN_SKIP_IF_NO_CUDA()                                                                 \
    do {                                                                                           \
        cudaError_t const __cuda_check_result = cudaFree(nullptr);                                 \
        if (gwn::tests::is_cuda_runtime_unavailable(__cuda_check_result))                          \
            GTEST_SKIP() << "CUDA runtime unavailable: "                                           \
                         << cudaGetErrorString(__cuda_check_result);                               \
    } while (false)

#define SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status)                                           \
    do {                                                                                           \
        if (!(status).is_ok() && (status).error() == gwn::gwn_error::cuda_runtime_error &&         \
            gwn::tests::is_cuda_runtime_unavailable_message((status).message())) {                 \
            GTEST_SKIP() << "CUDA runtime unavailable: " << (status).message();                    \
        }                                                                                          \
    } while (false)

} // namespace gwn::tests
