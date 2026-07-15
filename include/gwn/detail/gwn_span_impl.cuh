#pragma once

/// \file detail/gwn_span_impl.cuh
/// \brief Adapt public memory-space views to internal CUDA spans.

#include "../gwn_utils.cuh"

namespace gwn::detail {

template <class T>
[[nodiscard]] constexpr cuda::std::span<T>
gwn_cuda_span_view_impl(gwn_host_span<T> const span) noexcept {
    return span.span_;
}

template <class T>
[[nodiscard]] constexpr cuda::std::span<T>
gwn_cuda_span_view_impl(gwn_device_span<T> const span) noexcept {
    return span.span_;
}

} // namespace gwn::detail
