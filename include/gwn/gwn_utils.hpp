#pragma once

#include <cuda_runtime_api.h>
#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace gwn {

/// \brief Utility base that disables copy construction/assignment.
class gwn_noncopyable {
 protected:
  constexpr gwn_noncopyable() noexcept = default;
  ~gwn_noncopyable() = default;
  gwn_noncopyable(gwn_noncopyable&&) noexcept = default;
  gwn_noncopyable& operator=(gwn_noncopyable&&) noexcept = default;

 public:
  gwn_noncopyable(const gwn_noncopyable&) = delete;
  gwn_noncopyable& operator=(const gwn_noncopyable&) = delete;
};

/// \brief Returns true when span data pointer is valid for its size.
template <class T>
[[nodiscard]] __host__ __device__ constexpr bool gwn_span_has_storage(
    const cuda::std::span<const T> span) noexcept {
  return span.size() == 0 || span.data() != nullptr;
}

template <class T>
[[nodiscard]] __host__ __device__ constexpr bool gwn_span_has_storage(
    const cuda::std::span<T> span) noexcept {
  return span.size() == 0 || span.data() != nullptr;
}

/// \brief High-level status categories exposed by the public API.
enum class gwn_error {
  success = 0,
  invalid_argument,
  cuda_runtime_error,
  internal_error,
};

/// \brief Error-code object used by public APIs.
/// \remark Message storage is non-owning; pass stable-lifetime text.
/// \remark `detail_code` is backend-agnostic; CUDA errors encode `cudaError_t`.
class gwn_status {
 public:
  constexpr gwn_status() noexcept = default;

  [[nodiscard]] static constexpr gwn_status ok() noexcept {
    return gwn_status();
  }

  [[nodiscard]] static constexpr gwn_status invalid_argument(
      const std::string_view message) noexcept {
    return gwn_status(gwn_error::invalid_argument, message);
  }

  [[nodiscard]] static constexpr gwn_status internal_error(
      const std::string_view message = "Internal error.") noexcept {
    return gwn_status(gwn_error::internal_error, message);
  }

  [[nodiscard]] static constexpr gwn_status cuda_runtime_error(
      const cudaError_t cuda_result,
      const std::source_location loc = std::source_location::current(),
      const std::string_view message = "CUDA API call failed.") noexcept {
    return gwn_status(gwn_error::cuda_runtime_error, message,
                      static_cast<std::int64_t>(static_cast<int>(cuda_result)),
                      loc, true);
  }

  [[nodiscard]] constexpr bool is_ok() const noexcept {
    return error_ == gwn_error::success;
  }
  [[nodiscard]] constexpr gwn_error error() const noexcept { return error_; }
  [[nodiscard]] constexpr bool has_detail_code() const noexcept {
    return has_detail_code_;
  }
  [[nodiscard]] constexpr std::int64_t detail_code() const noexcept {
    return detail_code_;
  }
  [[nodiscard]] constexpr std::source_location location() const noexcept {
    return location_;
  }
  [[nodiscard]] constexpr bool has_location() const noexcept {
    return location_.line() != 0;
  }
  [[nodiscard]] constexpr bool is_cuda_runtime_error() const noexcept {
    return error_ == gwn_error::cuda_runtime_error && has_detail_code_;
  }
  [[nodiscard]] constexpr cudaError_t cuda_error() const noexcept {
    if (!is_cuda_runtime_error()) {
      return cudaSuccess;
    }
    return static_cast<cudaError_t>(static_cast<int>(detail_code_));
  }
  [[nodiscard]] constexpr std::string_view message() const noexcept {
    return message_;
  }

 private:
  constexpr gwn_status(const gwn_error error_code,
                       const std::string_view message) noexcept
      : error_(error_code), message_(message) {}

  constexpr gwn_status(const gwn_error error_code,
                       const std::string_view message,
                       const std::int64_t detail_code,
                       const std::source_location loc,
                       const bool has_detail_code) noexcept
      : error_(error_code),
        detail_code_(detail_code),
        has_detail_code_(has_detail_code),
        location_(loc),
        message_(message) {}

  gwn_error error_ = gwn_error::success;
  std::int64_t detail_code_ = 0;
  bool has_detail_code_ = false;
  std::source_location location_{};
  std::string_view message_{};
};

/// \brief Scope guard used for deterministic cleanup in status-based code.
///
/// \remark Callback must be `noexcept` to preserve cleanup safety.
template <class Callback>
class gwn_scope_exit {
 public:
  static_assert(std::is_nothrow_invocable_v<Callback&>,
                "gwn_scope_exit callback must be noexcept.");

  explicit gwn_scope_exit(Callback callback) noexcept(
      std::is_nothrow_move_constructible_v<Callback>)
      : callback_(std::move(callback)) {}

  gwn_scope_exit(const gwn_scope_exit&) = delete;
  gwn_scope_exit& operator=(const gwn_scope_exit&) = delete;

  gwn_scope_exit(gwn_scope_exit&& other) noexcept(
      std::is_nothrow_move_constructible_v<Callback>)
      : callback_(std::move(other.callback_)),
        is_active_(std::exchange(other.is_active_, false)) {}

  gwn_scope_exit& operator=(gwn_scope_exit&&) = delete;

  ~gwn_scope_exit() noexcept {
    if (is_active_) {
      callback_();
    }
  }

  void release() noexcept { is_active_ = false; }

 private:
  Callback callback_;
  bool is_active_ = true;
};

/// \brief Helper for creating `gwn_scope_exit` with template argument
/// deduction.
template <class Callback>
[[nodiscard]] auto gwn_make_scope_exit(Callback&& callback)
    -> gwn_scope_exit<std::decay_t<Callback>> {
  return gwn_scope_exit<std::decay_t<Callback>>(
      std::forward<Callback>(callback));
}

#ifndef GWN_HANDLE_STATUS_FAIL
#define GWN_HANDLE_STATUS_FAIL(status) ((void)(status))
#endif

/// \brief Evaluate status expression and return on failure.
#ifndef GWN_RETURN_ON_ERROR
#define GWN_RETURN_ON_ERROR(expr)                \
  do {                                           \
    const ::gwn::gwn_status gwn_status = (expr); \
    if (!gwn_status.is_ok()) {                   \
      GWN_HANDLE_STATUS_FAIL(gwn_status);        \
      return gwn_status;                         \
    }                                            \
  } while (false)
#endif

/// \brief Cross-platform loop unroll pragma.
#ifndef GWN_PRAGMA_UNROLL
#define GWN_PRAGMA_UNROLL _Pragma("unroll")
#endif

/// \brief Throw `std::runtime_error` if `status` is not success.
inline void gwn_throw_if_error(const gwn_status& status) {
  if (status.is_ok()) {
    return;
  }

  std::ostringstream out;
  out << status.message();
  if (status.is_cuda_runtime_error()) {
    const cudaError_t result = status.cuda_error();
    out << " [cuda_error=" << static_cast<int>(result) << " ("
        << cudaGetErrorName(result) << "): \"" << cudaGetErrorString(result)
        << "\"]";
  } else if (status.has_detail_code()) {
    out << " [detail_code=" << status.detail_code() << "]";
  }
  if (status.has_location()) {
    const std::source_location loc = status.location();
    out << " at " << loc.file_name() << ":" << loc.line();
  }
  throw std::runtime_error(out.str());
}

/// \brief Convert CUDA runtime result to `gwn_status`.
inline gwn_status gwn_cuda_to_status(
    const cudaError_t cuda_result,
    const std::source_location loc = std::source_location::current()) noexcept {
  if (cuda_result == cudaSuccess) {
    return gwn_status::ok();
  }
  return gwn_status::cuda_runtime_error(cuda_result, loc);
}

/// \brief Default stream used by API helpers.
[[nodiscard]] inline cudaStream_t gwn_default_stream() noexcept {
  return cudaStreamLegacy;
}

/// \brief Allocate device memory from CUDA stream-ordered allocator only.
/// \remark Uses `cudaMallocAsync` only; no synchronous fallback path.
/// \remark `bytes == 0` returns success with `*ptr = nullptr`.
inline gwn_status gwn_cuda_malloc(
    void** ptr,
    const std::size_t bytes,
    const cudaStream_t stream = gwn_default_stream()) noexcept {
  if (bytes == 0) {
    *ptr = nullptr;
    return gwn_status::ok();
  }

  return gwn_cuda_to_status(cudaMallocAsync(ptr, bytes, stream));
}

/// \brief Free device memory through CUDA stream-ordered allocator only.
/// \remark Uses `cudaFreeAsync` only; no synchronous fallback path.
/// \remark `ptr == nullptr` is treated as success.
inline gwn_status gwn_cuda_free(
    void* ptr,
    const cudaStream_t stream = gwn_default_stream()) noexcept {
  if (ptr == nullptr) {
    return gwn_status::ok();
  }

  return gwn_cuda_to_status(cudaFreeAsync(ptr, stream));
}

}  // namespace gwn
