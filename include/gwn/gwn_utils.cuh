#pragma once

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <format>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "gwn_assert.cuh"

namespace gwn {
template <class Real>
concept gwn_real_type = std::floating_point<Real>;

template <class Index>
concept gwn_index_type = std::same_as<Index, std::uint32_t> || std::same_as<Index, std::uint64_t>;

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

/// \brief Unconditional trap helper for host/device code paths.
[[noreturn]] __host__ __device__ inline void gwn_trap() noexcept {
#if defined(__CUDA_ARCH__)
    asm volatile("trap;");
    __builtin_unreachable();
#else
    __builtin_trap();
#endif
}

/// \brief Mixin that binds an object to a CUDA stream.
///
/// \remark This mixin only stores stream state and accessor methods.
/// \remark It does not own memory and does not perform synchronization.
class gwn_stream_mixin {
public:
    gwn_stream_mixin() noexcept = default;
    explicit gwn_stream_mixin(cudaStream_t const stream) noexcept : stream_{stream} {}

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    void set_stream(cudaStream_t const stream) noexcept { stream_ = stream; }

    friend void swap(gwn_stream_mixin &lhs, gwn_stream_mixin &rhs) noexcept {
        using std::swap;
        swap(lhs.stream_, rhs.stream_);
    }

private:
    cudaStream_t stream_{gwn_default_stream()};
};

/// \brief Check whether a span has valid storage (non-null data for non-empty spans).
template <class T>
[[nodiscard]] __host__ __device__ constexpr bool
gwn_span_has_storage(cuda::std::span<T const> const span) noexcept {
    return span.size() == 0 || span.data() != nullptr;
}

/// \overload
template <class T>
[[nodiscard]] __host__ __device__ constexpr bool
gwn_span_has_storage(cuda::std::span<T> const span) noexcept {
    return span.size() == 0 || span.data() != nullptr;
}

/// \brief Return the sentinel value used for invalid integral indices.
template <class Index>
[[nodiscard]] __host__ __device__ constexpr Index gwn_invalid_index() noexcept {
    static_assert(std::is_integral_v<Index>, "Index must be an integral type.");
    return std::numeric_limits<Index>::max();
}

/// \brief Return whether an index value is invalid.
template <class Index>
[[nodiscard]] __host__ __device__ constexpr bool gwn_is_invalid_index(Index const index) noexcept {
    static_assert(std::is_integral_v<Index>, "Index must be an integral type.");
    if constexpr (std::is_signed_v<Index>)
        return index < Index(0);
    else
        return index == gwn_invalid_index<Index>();
}

/// \brief Return whether an index value is valid.
template <class Index>
[[nodiscard]] __host__ __device__ constexpr bool gwn_is_valid_index(Index const index) noexcept {
    return !gwn_is_invalid_index(index);
}

/// \brief Return whether an index is valid and within `[0, bound)`.
template <class Index>
[[nodiscard]] __host__ __device__ constexpr bool
gwn_index_in_bounds(Index const index, std::size_t const bound) noexcept {
    return gwn_is_valid_index(index) && static_cast<std::size_t>(index) < bound;
}

/// \brief High-level status categories exposed by the public API.
enum class gwn_error {
    success = 0,
    invalid_argument,
    cuda_runtime_error,
    internal_error,
};

/// \brief Lightweight error-code object returned by public APIs.
///
/// \details `std::string` construction and `std::format` are assumed never to throw
///          (OOM is treated as a fatal precondition violation).  All factory methods
///          are therefore `noexcept`.
class gwn_status {
public:
    gwn_status() noexcept = default;

    /// \brief Construct a success status (no message, no location).
    [[nodiscard]] static gwn_status ok() noexcept { return gwn_status(); }

    /// \brief Construct an invalid-argument error with a diagnostic message.
    [[nodiscard]] static gwn_status invalid_argument(std::string message) noexcept {
        return gwn_status(gwn_error::invalid_argument, std::move(message));
    }

    /// \brief Construct an internal-error status with an optional diagnostic message.
    [[nodiscard]] static gwn_status
    internal_error(std::string message = "Internal error.") noexcept {
        return gwn_status(gwn_error::internal_error, std::move(message));
    }

    /// \brief Construct a CUDA runtime error status from a `cudaError_t`.
    ///
    /// \details The error name and description are appended to \p message
    ///          using `cudaGetErrorName` / `cudaGetErrorString`.
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

    /// \brief Return `true` when the status represents success.
    [[nodiscard]] bool is_ok() const noexcept { return error_ == gwn_error::success; }

    /// \brief Return the error category.
    [[nodiscard]] gwn_error error() const noexcept { return error_; }

    /// \brief Return the captured source location (meaningful only for CUDA errors).
    [[nodiscard]] std::source_location location() const noexcept { return location_; }

    /// \brief Return `true` when a non-default source location is present.
    [[nodiscard]] bool has_location() const noexcept { return location_.line() != 0; }

    /// \brief Return the diagnostic message string.
    [[nodiscard]] std::string const &message() const noexcept { return message_; }

private:
    gwn_status(gwn_error const error_code, std::string message) noexcept
        : error_{error_code}, message_{std::move(message)} {}

    gwn_status(
        gwn_error const error_code, std::string message, std::source_location const loc
    ) noexcept
        : error_{error_code}, location_{loc}, message_{std::move(message)} {}

    gwn_error error_{gwn_error::success};
    std::source_location location_{};
    std::string message_{};
};

/// \brief RAII scope guard for deterministic cleanup in status-based code.
///
/// \tparam Callback  Callable type; must be nothrow-invocable.
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
    bool is_active_{true};
};

/// \brief Construct a `gwn_scope_exit` with class template argument deduction.
template <class Callback>
[[nodiscard]] auto gwn_make_scope_exit(Callback &&callback)
    -> gwn_scope_exit<std::decay_t<Callback>> {
    return gwn_scope_exit<std::decay_t<Callback>>(std::forward<Callback>(callback));
}

#ifndef GWN_HANDLE_STATUS_FAIL
#define GWN_HANDLE_STATUS_FAIL(status) ((void)(status))
#endif

/// \brief Evaluate a `gwn_status`-returning expression and return on failure.
///
/// \details The result is stored in a uniquely named local to avoid shadowing
///          the `gwn_status` type name in the enclosing scope.
#ifndef GWN_RETURN_ON_ERROR
#define GWN_RETURN_ON_ERROR(expr)                                                                  \
    do {                                                                                           \
        ::gwn::gwn_status const gwn_status_result_ = (expr);                                       \
        if (!gwn_status_result_.is_ok()) {                                                         \
            GWN_HANDLE_STATUS_FAIL(gwn_status_result_);                                            \
            return gwn_status_result_;                                                             \
        }                                                                                          \
    } while (false)
#endif

/// \brief Cross-platform loop unroll pragma.
#ifndef GWN_PRAGMA_UNROLL
#define GWN_PRAGMA_UNROLL _Pragma("unroll")
#endif

/// \brief Throw `std::runtime_error` with diagnostic details if \p status is not success.
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

/// \brief Translate a `cudaError_t` into a `gwn_status`.
inline gwn_status gwn_cuda_to_status(
    cudaError_t const cuda_result, std::source_location const loc = std::source_location::current()
) noexcept {
    if (cuda_result == cudaSuccess)
        return gwn_status::ok();
    return gwn_status::cuda_runtime_error(cuda_result, loc);
}

[[nodiscard]] inline cudaStream_t gwn_default_stream() noexcept { return cudaStreamLegacy; }

/// \brief Allocate device memory via the CUDA stream-ordered allocator.
///
/// \remark No synchronous fallback path (`cudaMallocAsync` only).
/// \remark Zero-byte requests succeed with `*ptr = nullptr`.
inline gwn_status gwn_cuda_malloc(
    void **ptr, std::size_t const bytes, cudaStream_t const stream = gwn_default_stream()
) noexcept {
    if (ptr == nullptr)
        return gwn_status::invalid_argument("gwn_cuda_malloc requires non-null output pointer.");

    if (bytes == 0) {
        *ptr = nullptr;
        return gwn_status::ok();
    }

    gwn_status const alloc_status = gwn_cuda_to_status(cudaMallocAsync(ptr, bytes, stream));
    if (!alloc_status.is_ok())
        return alloc_status;

    GWN_ASSERT(
        *ptr != nullptr, "gwn_cuda_malloc succeeded for %zu bytes but returned a null pointer.",
        bytes
    );
    return gwn_status::ok();
}

/// \brief Free device memory via the CUDA stream-ordered allocator.
///
/// \remark No synchronous fallback path (`cudaFreeAsync` only).
/// \remark Null pointer is treated as a no-op success.
inline gwn_status
gwn_cuda_free(void *ptr, cudaStream_t const stream = gwn_default_stream()) noexcept {
    if (ptr == nullptr)
        return gwn_status::ok();

    return gwn_cuda_to_status(cudaFreeAsync(ptr, stream));
}

/// \brief RAII device buffer backed by the CUDA stream-ordered allocator.
///
/// \tparam T  Element type (trivially copyable assumed by memset/memcpy).
///
/// \details Allocation and deallocation use `cudaMallocAsync` / `cudaFreeAsync`
///          exclusively.  The buffer remembers its bound stream and releases on
///          that stream by default.
///
/// \remark `resize()` does not preserve old contents.
/// \remark Move-assignment uses copy-and-swap; copy construction is deleted.
template <class T> class gwn_device_array final : public gwn_noncopyable {
public:
    using value_type = T;

    gwn_device_array() = default;
    explicit gwn_device_array(cudaStream_t const stream) noexcept : stream_{stream} {}

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

    /// \brief Resize the buffer (contents are not preserved).
    ///
    /// \remark When \p count equals the current size, the operation is a no-op;
    ///         the bound stream is preserved unchanged.
    /// \remark When reallocation happens, new storage is allocated on \p stream while
    ///         old storage is released on the previously bound stream.
    [[nodiscard]] gwn_status
    resize(std::size_t const count, cudaStream_t const stream = gwn_default_stream()) noexcept {
        GWN_ASSERT(
            invariant_storage_ok_(), "gwn_device_array storage invariant violated before resize."
        );

        if (count == size_) {
            // No-op: size unchanged, stream binding preserved
            return gwn_status::ok();
        }
        if (count == 0)
            return clear(stream);

        cudaStream_t const release_stream = stream_;
        T *const old_ptr = data_;

        void *new_ptr = nullptr;
        gwn_status const alloc_status = gwn_cuda_malloc(&new_ptr, count * sizeof(T), stream);
        if (!alloc_status.is_ok())
            return alloc_status;

        GWN_ASSERT(
            new_ptr != nullptr,
            "gwn_device_array::resize allocated %zu elements but returned null storage.", count
        );

        auto cleanup_new_ptr = gwn_make_scope_exit([&]() noexcept {
            if (new_ptr != nullptr)
                (void)gwn_cuda_free(new_ptr, stream);
        });

        if (old_ptr != nullptr) {
            gwn_status const release_status = gwn_cuda_free(old_ptr, release_stream);
            if (!release_status.is_ok())
                return release_status;
        }

        data_ = static_cast<T *>(new_ptr);
        size_ = count;
        stream_ = stream;
        new_ptr = nullptr;
        cleanup_new_ptr.release();
        GWN_ASSERT(
            invariant_storage_ok_(), "gwn_device_array storage invariant violated after resize."
        );
        return gwn_status::ok();
    }

    /// \brief Release device memory on the currently bound stream.
    [[nodiscard]] gwn_status clear() noexcept { return clear(stream_); }

    /// \brief Release device memory on the currently bound stream and rebind to \p stream.
    ///
    /// \remark The free operation is enqueued on the stream that was bound before this call.
    /// \remark On success, the object is rebound to \p stream; on failure, the old binding stays.
    [[nodiscard]] gwn_status clear(cudaStream_t const stream) noexcept {
        GWN_ASSERT(
            invariant_storage_ok_(), "gwn_device_array storage invariant violated before clear."
        );

        cudaStream_t const release_stream = stream_;
        if (data_ == nullptr) {
            size_ = 0;
            stream_ = stream;
            GWN_ASSERT(
                invariant_storage_ok_(), "gwn_device_array storage invariant violated after clear."
            );
            return gwn_status::ok();
        }

        gwn_status const release_status = gwn_cuda_free(data_, release_stream);
        if (!release_status.is_ok())
            return release_status;
        data_ = nullptr;
        size_ = 0;
        stream_ = stream;
        GWN_ASSERT(
            invariant_storage_ok_(), "gwn_device_array storage invariant violated after clear."
        );
        return gwn_status::ok();
    }

    /// \brief Return the currently bound stream.
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

    /// \brief Rebind to a different stream without releasing memory.
    void set_stream(cudaStream_t const stream) noexcept { stream_ = stream; }

    /// \brief Zero-fill the buffer asynchronously on \p stream.
    [[nodiscard]] gwn_status zero(cudaStream_t const stream = gwn_default_stream()) noexcept {
        if (empty())
            return gwn_status::ok();
        return gwn_cuda_to_status(cudaMemsetAsync(data_, 0, size_ * sizeof(T), stream));
    }

    /// \brief Upload host data into the buffer (resizes to match \p src).
    [[nodiscard]] gwn_status copy_from_host(
        cuda::std::span<T const> const src, cudaStream_t const stream = gwn_default_stream()
    ) noexcept {
        GWN_RETURN_ON_ERROR(resize(src.size(), stream));
        if (src.empty())
            return gwn_status::ok();

        GWN_ASSERT(
            data_ != nullptr,
            "gwn_device_array::copy_from_host requires non-null device storage for non-empty copy."
        );

        return gwn_cuda_to_status(
            cudaMemcpyAsync(data_, src.data(), src.size_bytes(), cudaMemcpyHostToDevice, stream)
        );
    }

    /// \brief Download buffer contents into host span \p dst (sizes must match).
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

        GWN_ASSERT(
            data_ != nullptr,
            "gwn_device_array::copy_to_host requires non-null device storage for non-empty copy."
        );

        return gwn_cuda_to_status(
            cudaMemcpyAsync(dst.data(), data_, dst.size_bytes(), cudaMemcpyDeviceToHost, stream)
        );
    }

    /// \brief Pointer to device storage (may be null when empty).
    [[nodiscard]] T *data() noexcept { return data_; }
    /// \overload
    [[nodiscard]] T const *data() const noexcept { return data_; }

    /// \brief Number of elements.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// \brief Return `true` when the buffer holds no elements.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// \brief Return a device span over the buffer contents.
    [[nodiscard]] cuda::std::span<T> span() noexcept { return {data_, size_}; }
    /// \overload
    [[nodiscard]] cuda::std::span<T const> span() const noexcept { return {data_, size_}; }

    /// \brief Release ownership of the device pointer without freeing it.
    ///
    /// \remark Caller becomes responsible for eventually freeing the returned pointer.
    [[nodiscard]] T *release() noexcept {
        T *const raw = data_;
        data_ = nullptr;
        size_ = 0;
        return raw;
    }

    friend void swap(gwn_device_array &lhs, gwn_device_array &rhs) noexcept {
        using std::swap;
        swap(lhs.data_, rhs.data_);
        swap(lhs.size_, rhs.size_);
        swap(lhs.stream_, rhs.stream_);
    }

private:
    [[nodiscard]] bool invariant_storage_ok_() const noexcept {
        return size_ == 0 || data_ != nullptr;
    }

    T *data_{nullptr};
    std::size_t size_{0};
    cudaStream_t stream_{gwn_default_stream()};
};

namespace detail {

template <class T>
[[nodiscard]] constexpr T *gwn_mutable_data(cuda::std::span<T const> const span) noexcept {
    return const_cast<T *>(span.data());
}

template <class T>
gwn_status gwn_allocate_span(
    cuda::std::span<T const> &dst, std::size_t const count, cudaStream_t const stream
) noexcept {
    if (dst.data() != nullptr || dst.size() != 0)
        return gwn_status::invalid_argument("gwn_allocate_span expects an empty destination span.");
    if (count == 0) {
        dst = {};
        return gwn_status::ok();
    }

    void *ptr = nullptr;
    GWN_RETURN_ON_ERROR(gwn_cuda_malloc(&ptr, count * sizeof(T), stream));
    GWN_ASSERT(
        ptr != nullptr, "gwn_allocate_span allocated %zu elements but returned null storage.", count
    );
    dst = cuda::std::span<T const>(static_cast<T const *>(ptr), count);
    GWN_ASSERT(gwn_span_has_storage(dst), "gwn_allocate_span produced invalid span storage.");
    return gwn_status::ok();
}

template <class T>
void gwn_free_span(cuda::std::span<T const> &span_view, cudaStream_t const stream) noexcept {
    GWN_ASSERT(gwn_span_has_storage(span_view), "gwn_free_span requires valid span storage.");

    if (span_view.data() != nullptr) {
        gwn_status const status = gwn_cuda_free(gwn_mutable_data(span_view), stream);
        if (!status.is_ok())
            GWN_HANDLE_STATUS_FAIL(status);
        span_view = {};
    }
}

template <class T>
gwn_status gwn_copy_h2d(
    cuda::std::span<T const> const dst_device, cuda::std::span<T const> const src_host,
    cudaStream_t const stream
) noexcept {
    if (dst_device.size() != src_host.size())
        return gwn_status::invalid_argument("gwn_copy_h2d span size mismatch.");
    if (!gwn_span_has_storage(dst_device))
        return gwn_status::invalid_argument("gwn_copy_h2d destination span has null storage.");
    if (!gwn_span_has_storage(src_host))
        return gwn_status::invalid_argument("gwn_copy_h2d source span has null storage.");

    if (src_host.empty())
        return gwn_status::ok();

    return gwn_cuda_to_status(cudaMemcpyAsync(
        gwn_mutable_data(dst_device), src_host.data(), src_host.size_bytes(),
        cudaMemcpyHostToDevice, stream
    ));
}

template <class T>
gwn_status gwn_copy_d2h(
    cuda::std::span<T> const dst_host, cuda::std::span<T const> const src_device,
    cudaStream_t const stream
) noexcept {
    if (dst_host.size() != src_device.size())
        return gwn_status::invalid_argument("gwn_copy_d2h span size mismatch.");
    if (!gwn_span_has_storage(dst_host))
        return gwn_status::invalid_argument("gwn_copy_d2h destination span has null storage.");
    if (!gwn_span_has_storage(src_device))
        return gwn_status::invalid_argument("gwn_copy_d2h source span has null storage.");

    if (src_device.empty())
        return gwn_status::ok();

    return gwn_cuda_to_status(cudaMemcpyAsync(
        dst_host.data(), src_device.data(), src_device.size_bytes(), cudaMemcpyDeviceToHost, stream
    ));
}

template <class T>
gwn_status gwn_copy_d2d(
    cuda::std::span<T const> const dst_device, cuda::std::span<T const> const src_device,
    cudaStream_t const stream
) noexcept {
    if (dst_device.size() != src_device.size())
        return gwn_status::invalid_argument("gwn_copy_d2d span size mismatch.");
    if (!gwn_span_has_storage(dst_device))
        return gwn_status::invalid_argument("gwn_copy_d2d destination span has null storage.");
    if (!gwn_span_has_storage(src_device))
        return gwn_status::invalid_argument("gwn_copy_d2d source span has null storage.");

    if (src_device.empty())
        return gwn_status::ok();

    return gwn_cuda_to_status(cudaMemcpyAsync(
        gwn_mutable_data(dst_device), src_device.data(), src_device.size_bytes(),
        cudaMemcpyDeviceToDevice, stream
    ));
}

} // namespace detail

} // namespace gwn
