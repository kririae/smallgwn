#pragma once

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace gwn {

/// \brief Utility base that disables copy construction/assignment.
class gwn_noncopyable {
protected:
    constexpr gwn_noncopyable() noexcept = default;
    ~gwn_noncopyable() = default;
    gwn_noncopyable(gwn_noncopyable &&) noexcept = default;
    gwn_noncopyable &operator=(gwn_noncopyable &&) noexcept = default;

public:
    gwn_noncopyable(gwn_noncopyable const &) = delete;
    gwn_noncopyable &operator=(gwn_noncopyable const &) = delete;
};

/// \brief Default stream used by API helpers.
[[nodiscard]] inline cudaStream_t gwn_default_stream() noexcept;

/// \brief Mixin that binds an object to a CUDA stream.
///
/// \remark This mixin only stores stream state and accessor methods.
/// \remark It does not own memory and does not perform synchronization.
class gwn_stream_mixin {
public:
    gwn_stream_mixin() noexcept = default;
    explicit gwn_stream_mixin(cudaStream_t const stream) noexcept : stream_(stream) {}

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    void set_stream(cudaStream_t const stream) noexcept { stream_ = stream; }

    friend void swap(gwn_stream_mixin &lhs, gwn_stream_mixin &rhs) noexcept {
        using std::swap;
        swap(lhs.stream_, rhs.stream_);
    }

private:
    cudaStream_t stream_ = gwn_default_stream();
};

/// \brief Returns true when span data pointer is valid for its size.
template <class T>
[[nodiscard]] __host__ __device__ constexpr bool
gwn_span_has_storage(cuda::std::span<T const> const span) noexcept {
    return span.size() == 0 || span.data() != nullptr;
}

template <class T>
[[nodiscard]] __host__ __device__ constexpr bool
gwn_span_has_storage(cuda::std::span<T> const span) noexcept {
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
/// \remark Message storage is owned by `std::string`.
class gwn_status {
public:
    gwn_status() noexcept = default;

    [[nodiscard]] static gwn_status ok() noexcept { return gwn_status(); }

    [[nodiscard]] static gwn_status invalid_argument(std::string message) noexcept {
        return gwn_status(gwn_error::invalid_argument, std::move(message));
    }

    [[nodiscard]] static gwn_status
    internal_error(std::string message = "Internal error.") noexcept {
        return gwn_status(gwn_error::internal_error, std::move(message));
    }

    [[nodiscard]] static gwn_status cuda_runtime_error(
        cudaError_t const cuda_result,
        std::source_location const loc = std::source_location::current(),
        std::string message = "CUDA API call failed."
    ) noexcept {
        char const *const error_name = cudaGetErrorName(cuda_result);
        char const *const error_message = cudaGetErrorString(cuda_result);
        message += std::format(
            " [cuda_error={} ({}): \"{}\"]", static_cast<int>(cuda_result),
            error_name != nullptr ? error_name : "unknown",
            error_message != nullptr ? error_message : "unknown"
        );
        return gwn_status(gwn_error::cuda_runtime_error, std::move(message), loc);
    }

    [[nodiscard]] bool is_ok() const noexcept { return error_ == gwn_error::success; }
    [[nodiscard]] gwn_error error() const noexcept { return error_; }
    [[nodiscard]] std::source_location location() const noexcept { return location_; }
    [[nodiscard]] bool has_location() const noexcept { return location_.line() != 0; }
    [[nodiscard]] std::string const &message() const noexcept { return message_; }

private:
    gwn_status(gwn_error const error_code, std::string message) noexcept
        : error_(error_code), message_(std::move(message)) {}

    gwn_status(
        gwn_error const error_code, std::string message, std::source_location const loc
    ) noexcept
        : error_(error_code), location_(loc), message_(std::move(message)) {}

    gwn_error error_ = gwn_error::success;
    std::source_location location_{};
    std::string message_{};
};

/// \brief Scope guard used for deterministic cleanup in status-based code.
///
/// \remark Callback must be `noexcept` to preserve cleanup safety.
template <class Callback> class gwn_scope_exit {
public:
    static_assert(
        std::is_nothrow_invocable_v<Callback &>, "gwn_scope_exit callback must be noexcept."
    );

    explicit gwn_scope_exit(
        Callback callback
    ) noexcept(std::is_nothrow_move_constructible_v<Callback>)
        : callback_(std::move(callback)) {}

    gwn_scope_exit(gwn_scope_exit const &) = delete;
    gwn_scope_exit &operator=(gwn_scope_exit const &) = delete;

    gwn_scope_exit(gwn_scope_exit &&other) noexcept(std::is_nothrow_move_constructible_v<Callback>)
        : callback_(std::move(other.callback_)),
          is_active_(std::exchange(other.is_active_, false)) {}

    gwn_scope_exit &operator=(gwn_scope_exit &&) = delete;

    ~gwn_scope_exit() noexcept {
        if (is_active_)
            callback_();
    }

    void release() noexcept { is_active_ = false; }

private:
    Callback callback_;
    bool is_active_ = true;
};

/// \brief Helper for creating `gwn_scope_exit` with template argument
/// deduction.
template <class Callback>
[[nodiscard]] auto gwn_make_scope_exit(Callback &&callback)
    -> gwn_scope_exit<std::decay_t<Callback>> {
    return gwn_scope_exit<std::decay_t<Callback>>(std::forward<Callback>(callback));
}

#ifndef GWN_HANDLE_STATUS_FAIL
#define GWN_HANDLE_STATUS_FAIL(status) ((void)(status))
#endif

/// \brief Evaluate status expression and return on failure.
#ifndef GWN_RETURN_ON_ERROR
#define GWN_RETURN_ON_ERROR(expr)                                                                  \
    do {                                                                                           \
        const ::gwn::gwn_status gwn_status = (expr);                                               \
        if (!gwn_status.is_ok()) {                                                                 \
            GWN_HANDLE_STATUS_FAIL(gwn_status);                                                    \
            return gwn_status;                                                                     \
        }                                                                                          \
    } while (false)
#endif

/// \brief Cross-platform loop unroll pragma.
#ifndef GWN_PRAGMA_UNROLL
#define GWN_PRAGMA_UNROLL _Pragma("unroll")
#endif

/// \brief Throw `std::runtime_error` if `status` is not success.
inline void gwn_throw_if_error(gwn_status const &status) {
    if (status.is_ok())
        return;

    std::string out = status.message();
    if (status.has_location()) {
        std::source_location const loc = status.location();
        out += std::format(" at {}:{}", loc.file_name(), loc.line());
    }
    throw std::runtime_error(out);
}

/// \brief Convert CUDA runtime result to `gwn_status`.
inline gwn_status gwn_cuda_to_status(
    cudaError_t const cuda_result, std::source_location const loc = std::source_location::current()
) noexcept {
    if (cuda_result == cudaSuccess)
        return gwn_status::ok();
    return gwn_status::cuda_runtime_error(cuda_result, loc);
}

[[nodiscard]] inline cudaStream_t gwn_default_stream() noexcept { return cudaStreamLegacy; }

/// \brief Allocate device memory from CUDA stream-ordered allocator only.
/// \remark Uses `cudaMallocAsync` only; no synchronous fallback path.
/// \remark `bytes == 0` returns success with `*ptr = nullptr`.
inline gwn_status gwn_cuda_malloc(
    void **ptr, std::size_t const bytes, cudaStream_t const stream = gwn_default_stream()
) noexcept {
    if (bytes == 0) {
        *ptr = nullptr;
        return gwn_status::ok();
    }

    return gwn_cuda_to_status(cudaMallocAsync(ptr, bytes, stream));
}

/// \brief Free device memory through CUDA stream-ordered allocator only.
/// \remark Uses `cudaFreeAsync` only; no synchronous fallback path.
/// \remark `ptr == nullptr` is treated as success.
inline gwn_status
gwn_cuda_free(void *ptr, cudaStream_t const stream = gwn_default_stream()) noexcept {
    if (ptr == nullptr)
        return gwn_status::ok();

    return gwn_cuda_to_status(cudaFreeAsync(ptr, stream));
}

/// \brief RAII device buffer using CUDA stream-ordered allocation.
///
/// \remark Allocation and release use `cudaMallocAsync` / `cudaFreeAsync` only.
/// \remark `resize()` does not preserve old contents.
/// \remark Assignment uses copy-and-swap; copy construction remains disabled.
/// \remark `clear()` and destructor release on the currently bound stream.
template <class T> class gwn_device_array final : public gwn_noncopyable {
public:
    using value_type = T;

    gwn_device_array() = default;
    explicit gwn_device_array(cudaStream_t const stream) noexcept : stream_(stream) {}

    gwn_device_array(gwn_device_array &&other) noexcept { swap(*this, other); }

    gwn_device_array &operator=(gwn_device_array other) noexcept {
        swap(*this, other);
        return *this;
    }

    ~gwn_device_array() {
        gwn_status const status = clear();
        if (!status.is_ok())
            GWN_HANDLE_STATUS_FAIL(status);
    }

    [[nodiscard]] gwn_status
    resize(std::size_t const count, cudaStream_t const stream = gwn_default_stream()) noexcept {
        if (count == size_) {
            stream_ = stream;
            return gwn_status::ok();
        }
        if (count == 0)
            return clear(stream);

        void *new_ptr = nullptr;
        gwn_status const alloc_status = gwn_cuda_malloc(&new_ptr, count * sizeof(T), stream);
        if (!alloc_status.is_ok())
            return alloc_status;

        auto cleanup_new_ptr = gwn_make_scope_exit([&]() noexcept {
            if (new_ptr != nullptr)
                (void)gwn_cuda_free(new_ptr, stream);
        });

        if (data_ != nullptr) {
            gwn_status const release_status = gwn_cuda_free(data_, stream);
            if (!release_status.is_ok())
                return release_status;
        }

        data_ = static_cast<T *>(new_ptr);
        size_ = count;
        stream_ = stream;
        new_ptr = nullptr;
        cleanup_new_ptr.release();
        return gwn_status::ok();
    }

    [[nodiscard]] gwn_status clear() noexcept { return clear(stream_); }

    [[nodiscard]] gwn_status clear(cudaStream_t const stream) noexcept {
        stream_ = stream;
        if (data_ == nullptr) {
            size_ = 0;
            return gwn_status::ok();
        }

        gwn_status const release_status = gwn_cuda_free(data_, stream);
        if (!release_status.is_ok())
            return release_status;
        data_ = nullptr;
        size_ = 0;
        return gwn_status::ok();
    }

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    void set_stream(cudaStream_t const stream) noexcept { stream_ = stream; }

    [[nodiscard]] gwn_status zero(cudaStream_t const stream = gwn_default_stream()) noexcept {
        if (empty())
            return gwn_status::ok();
        return gwn_cuda_to_status(cudaMemsetAsync(data_, 0, size_ * sizeof(T), stream));
    }

    [[nodiscard]] gwn_status copy_from_host(
        cuda::std::span<T const> const src, cudaStream_t const stream = gwn_default_stream()
    ) noexcept {
        GWN_RETURN_ON_ERROR(resize(src.size(), stream));
        if (src.empty())
            return gwn_status::ok();

        return gwn_cuda_to_status(
            cudaMemcpyAsync(data_, src.data(), src.size_bytes(), cudaMemcpyHostToDevice, stream)
        );
    }

    [[nodiscard]] gwn_status copy_to_host(
        cuda::std::span<T> const dst, cudaStream_t const stream = gwn_default_stream()
    ) const noexcept {
        if (dst.size() != size_)
            return gwn_status::invalid_argument(
                "gwn_device_array::copy_to_host destination size mismatch."
            );
        if (!dst.empty() && dst.data() == nullptr)
            return gwn_status::invalid_argument(
                "gwn_device_array::copy_to_host destination span has null storage."
            );
        if (dst.empty())
            return gwn_status::ok();

        return gwn_cuda_to_status(
            cudaMemcpyAsync(dst.data(), data_, dst.size_bytes(), cudaMemcpyDeviceToHost, stream)
        );
    }

    [[nodiscard]] T *data() noexcept { return data_; }
    [[nodiscard]] T const *data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] cuda::std::span<T> span() noexcept { return cuda::std::span<T>(data_, size_); }
    [[nodiscard]] cuda::std::span<T const> span() const noexcept {
        return cuda::std::span<T const>(data_, size_);
    }

    friend void swap(gwn_device_array &lhs, gwn_device_array &rhs) noexcept {
        using std::swap;
        swap(lhs.data_, rhs.data_);
        swap(lhs.size_, rhs.size_);
        swap(lhs.stream_, rhs.stream_);
    }

private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
    cudaStream_t stream_ = gwn_default_stream();
};

} // namespace gwn
