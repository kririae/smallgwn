#pragma once

/// \file detail/gwn_utils.cuh
/// \brief Shared exception-based CUDA memory utilities for detail code.

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <format>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../gwn_utils.cuh"

namespace gwn::detail {

/// \brief Cross-platform loop unroll pragma.
#ifndef GWN_DETAIL_PRAGMA_UNROLL
#define GWN_DETAIL_PRAGMA_UNROLL _Pragma("unroll")
#endif

/// \brief CUDA Runtime API failure raised by exception-based detail code.
class gwn_cuda_exception final : public std::runtime_error {
public:
    explicit gwn_cuda_exception(
        cudaError_t const result, char const *const operation,
        std::source_location const location = std::source_location::current()
    )
        : std::runtime_error(std::format("{} failed.", operation)), result_{result},
          location_{location} {}

    [[nodiscard]] cudaError_t result() const noexcept { return result_; }
    [[nodiscard]] std::source_location location() const noexcept { return location_; }

private:
    cudaError_t result_;
    std::source_location location_;
};

/// \brief Exception adapter for a status returned by a non-throwing primitive.
class gwn_status_exception final : public std::runtime_error {
public:
    explicit gwn_status_exception(gwn_status status)
        : std::runtime_error(status.message()), status_{std::move(status)} {}

    [[nodiscard]] gwn_status const &status() const noexcept { return status_; }

private:
    gwn_status status_;
};

/// \brief Throw when a CUDA Runtime API call fails in exception-based detail code.
inline void gwn_throw_if_cuda_error(
    cudaError_t const result, char const *const operation,
    std::source_location const location = std::source_location::current()
) {
    if (result != cudaSuccess)
        throw gwn_cuda_exception(result, operation, location);
}

/// \brief Adapt a non-throwing primitive to the detail exception model.
inline void gwn_throw_status_error(gwn_status status) {
    if (!status.is_ok())
        throw gwn_status_exception(std::move(status));
}

template <class T>
    requires(!std::is_const_v<T>)
void gwn_allocate_span(
    cuda::std::span<T> &dst, std::size_t const count, cudaStream_t const stream
) {
    if (dst.data() != nullptr || dst.size() != 0)
        throw std::invalid_argument("gwn_allocate_span expects an empty destination span.");
    if (count == 0) {
        dst = {};
        return;
    }
    if (count > (std::numeric_limits<std::size_t>::max() / sizeof(T)))
        throw std::invalid_argument(
            "gwn_allocate_span element count exceeds addressable byte range."
        );

    void *ptr = nullptr;
    gwn_throw_if_cuda_error(cudaMallocAsync(&ptr, count * sizeof(T), stream), "cudaMallocAsync");
    GWN_ASSERT(
        ptr != nullptr, "gwn_allocate_span allocated %zu elements but returned null storage.", count
    );
    dst = cuda::std::span<T>(static_cast<T *>(ptr), count);
    GWN_ASSERT(gwn_span_has_storage(dst), "gwn_allocate_span produced invalid span storage.");
}

template <class T>
    requires(!std::is_const_v<T>)
void gwn_free_span(cuda::std::span<T> &span_view, cudaStream_t const stream) noexcept {
    GWN_ASSERT(gwn_span_has_storage(span_view), "gwn_free_span requires valid span storage.");

    if (span_view.data() != nullptr) {
        cudaError_t const result = cudaFreeAsync(span_view.data(), stream);
        GWN_ASSERT(result == cudaSuccess, "cudaFreeAsync failed with error %d.", int(result));
        if (result != cudaSuccess)
            std::terminate();
        span_view = {};
    }
}

template <class T>
    requires(!std::is_const_v<T>)
void gwn_copy_h2d(
    cuda::std::span<T> const dst_device, cuda::std::span<T const> const src_host,
    cudaStream_t const stream
) {
    if (dst_device.size() != src_host.size())
        throw std::invalid_argument("gwn_copy_h2d span size mismatch.");
    if (!gwn_span_has_storage(dst_device))
        throw std::invalid_argument("gwn_copy_h2d destination span has null storage.");
    if (!gwn_span_has_storage(src_host))
        throw std::invalid_argument("gwn_copy_h2d source span has null storage.");

    if (src_host.empty())
        return;

    gwn_throw_if_cuda_error(
        cudaMemcpyAsync(
            dst_device.data(), src_host.data(), src_host.size_bytes(), cudaMemcpyHostToDevice,
            stream
        ),
        "cudaMemcpyAsync"
    );
}

template <class T>
void gwn_copy_d2h(
    cuda::std::span<T> const dst_host, cuda::std::span<T const> const src_device,
    cudaStream_t const stream
) {
    if (dst_host.size() != src_device.size())
        throw std::invalid_argument("gwn_copy_d2h span size mismatch.");
    if (!gwn_span_has_storage(dst_host))
        throw std::invalid_argument("gwn_copy_d2h destination span has null storage.");
    if (!gwn_span_has_storage(src_device))
        throw std::invalid_argument("gwn_copy_d2h source span has null storage.");

    if (src_device.empty())
        return;

    gwn_throw_if_cuda_error(
        cudaMemcpyAsync(
            dst_host.data(), src_device.data(), src_device.size_bytes(), cudaMemcpyDeviceToHost,
            stream
        ),
        "cudaMemcpyAsync"
    );
}

template <class T>
    requires(!std::is_const_v<T>)
void gwn_copy_d2d(
    cuda::std::span<T> const dst_device, cuda::std::span<T const> const src_device,
    cudaStream_t const stream
) {
    if (dst_device.size() != src_device.size())
        throw std::invalid_argument("gwn_copy_d2d span size mismatch.");
    if (!gwn_span_has_storage(dst_device))
        throw std::invalid_argument("gwn_copy_d2d destination span has null storage.");
    if (!gwn_span_has_storage(src_device))
        throw std::invalid_argument("gwn_copy_d2d source span has null storage.");

    if (src_device.empty())
        return;

    gwn_throw_if_cuda_error(
        cudaMemcpyAsync(
            dst_device.data(), src_device.data(), src_device.size_bytes(), cudaMemcpyDeviceToDevice,
            stream
        ),
        "cudaMemcpyAsync"
    );
}

} // namespace gwn::detail
