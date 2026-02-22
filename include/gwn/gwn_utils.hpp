#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace gwn {

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

enum class gwn_error {
  success = 0,
  invalid_argument,
  cuda_runtime_error,
};

class gwn_status {
 public:
  gwn_status() = default;

  [[nodiscard]] static gwn_status ok() { return gwn_status(); }

  [[nodiscard]] static gwn_status invalid_argument(std::string message) {
    return gwn_status(gwn_error::invalid_argument, cudaSuccess,
                      std::move(message));
  }

  [[nodiscard]] static gwn_status cuda_runtime_error(
      const cudaError_t cuda_result,
      const std::source_location loc = std::source_location::current()) {
    std::ostringstream out;
    out << "CUDA API call failed with error " << static_cast<int>(cuda_result)
        << " (" << cudaGetErrorName(cuda_result) << "): \""
        << cudaGetErrorString(cuda_result) << "\" at " << loc.file_name() << ":"
        << loc.line();
    return gwn_status(gwn_error::cuda_runtime_error, cuda_result, out.str());
  }

  [[nodiscard]] bool is_ok() const noexcept {
    return error_ == gwn_error::success;
  }
  [[nodiscard]] gwn_error error() const noexcept { return error_; }
  [[nodiscard]] cudaError_t cuda_result() const noexcept {
    return cuda_result_;
  }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }

 private:
  gwn_status(const gwn_error error_code,
             const cudaError_t cuda_result,
             std::string message)
      : error_(error_code),
        cuda_result_(cuda_result),
        message_(std::move(message)) {}

  gwn_error error_ = gwn_error::success;
  cudaError_t cuda_result_ = cudaSuccess;
  std::string message_{};
};

inline void gwn_throw_if_error(const gwn_status& status) {
  if (!status.is_ok()) {
    throw std::runtime_error(status.message());
  }
}

inline gwn_status gwn_cuda_to_status(
    const cudaError_t cuda_result,
    const std::source_location loc = std::source_location::current()) {
  if (cuda_result == cudaSuccess) {
    return gwn_status::ok();
  }
  return gwn_status::cuda_runtime_error(cuda_result, loc);
}

inline bool gwn_supports_stream_ordered_allocator() noexcept {
#if CUDART_VERSION >= 11020
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return false;
  }

  int supported = 0;
  if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported,
                             device) != cudaSuccess) {
    return false;
  }
  return supported != 0;
#else
  return false;
#endif
}

inline bool gwn_is_async_allocator_unavailable(const cudaError_t result) {
  return result == cudaErrorNotSupported || result == cudaErrorOperatingSystem;
}

inline gwn_status gwn_cuda_malloc(void** ptr,
                                  const std::size_t bytes,
                                  const cudaStream_t stream = 0) noexcept {
  if (bytes == 0) {
    *ptr = nullptr;
    return gwn_status::ok();
  }

  if (gwn_supports_stream_ordered_allocator()) {
    const cudaError_t async_result = cudaMallocAsync(ptr, bytes, stream);
    if (async_result == cudaSuccess) {
      return gwn_status::ok();
    }
    if (!gwn_is_async_allocator_unavailable(async_result)) {
      return gwn_cuda_to_status(async_result);
    }
    (void)cudaGetLastError();
  }

  return gwn_cuda_to_status(cudaMalloc(ptr, bytes));
}

inline gwn_status gwn_cuda_free(void* ptr,
                                const cudaStream_t stream = 0) noexcept {
  if (ptr == nullptr) {
    return gwn_status::ok();
  }

  if (gwn_supports_stream_ordered_allocator()) {
    const cudaError_t async_result = cudaFreeAsync(ptr, stream);
    if (async_result == cudaSuccess) {
      return gwn_status::ok();
    }
    if (!gwn_is_async_allocator_unavailable(async_result)) {
      return gwn_cuda_to_status(async_result);
    }
    (void)cudaGetLastError();
  }

  return gwn_cuda_to_status(cudaFree(ptr));
}

}  // namespace gwn
