#pragma once

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "gwn_utils.cuh"

namespace gwn::detail {

/// \brief Unique-owning typed device storage with stream-ordered lifetime.
///
/// The bound stream is the stream ordered after the allocation's current uses. Operations without
/// an explicit stream use that binding. Operations that enqueue work on an explicit stream publish
/// it as the new binding. Callers establish any required ordering before changing streams.
template <class T> class gwn_device_array final : public gwn_noncopyable {
    static_assert(std::is_trivially_copyable_v<T>);

public:
    using value_type = T;

    gwn_device_array() noexcept = default;
    explicit gwn_device_array(cudaStream_t const stream) noexcept : stream_{stream} {}

    gwn_device_array(gwn_device_array &&other) noexcept { swap(*this, other); }

    gwn_device_array &operator=(gwn_device_array other) noexcept {
        swap(*this, other);
        return *this;
    }

    ~gwn_device_array() noexcept { clear(); }

    /// Resize without preserving contents, using the currently bound stream.
    void resize(std::size_t const count) { resize(count, stream_); }

    /// Resize without preserving contents, using \p stream for allocation and replacement.
    void resize(std::size_t const count, cudaStream_t const stream) {
        GWN_ASSERT(invariant_holds_(), "gwn_device_array invariant violated before resize.");

        // Same-size resize is a true no-op. In particular, it does not silently change the
        // lifetime stream of an allocation that was not used by this operation.
        if (count == size_)
            return;
        if (count == 0) {
            clear(stream);
            return;
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::invalid_argument(
                "gwn_device_array element count exceeds addressable bytes."
            );

        T *const replacement = allocate_(count, stream);
        // Deallocation failure cannot preserve a trustworthy owning state, so it is fatal rather
        // than represented as a recoverable third state beside empty and owning.
        deallocate_(data_, stream);
        data_ = replacement;
        size_ = count;
        stream_ = stream;
        GWN_ASSERT(invariant_holds_(), "gwn_device_array invariant violated after resize.");
    }

    /// Release storage on the currently bound stream.
    void clear() noexcept { clear(stream_); }

    /// Release storage on \p stream and bind the reusable empty object to that stream.
    void clear(cudaStream_t const stream) noexcept {
        GWN_ASSERT(invariant_holds_(), "gwn_device_array invariant violated before clear.");
        deallocate_(data_, stream);
        data_ = nullptr;
        size_ = 0;
        stream_ = stream;
    }

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

    /// Publish a lifetime stream without adding synchronization.
    void set_stream(cudaStream_t const stream) noexcept { stream_ = stream; }

    /// Zero-fill the allocation on the currently bound stream.
    void zero() { zero(stream_); }

    /// Zero-fill the allocation on \p stream.
    void zero(cudaStream_t const stream) {
        if (empty())
            return;

        // Publish before the CUDA call because a reported error can include prior asynchronous
        // work; cleanup must still remain ordered after this attempted use.
        stream_ = stream;
        gwn_throw_if_cuda_error(
            cudaMemsetAsync(data_, 0, size_ * sizeof(T), stream), "cudaMemsetAsync"
        );
    }

    /// Replace contents from host storage on the currently bound stream.
    ///
    /// The host span follows `cudaMemcpyAsync` lifetime rules.
    void copy_from_host(cuda::std::span<T const> const src) { copy_from_host(src, stream_); }

    /// Replace contents from host storage on \p stream.
    void copy_from_host(cuda::std::span<T const> const src, cudaStream_t const stream) {
        if (!gwn_span_has_storage(src))
            throw std::invalid_argument("gwn_device_array source span has null storage.");

        if (src.size() == size_) {
            if (src.empty())
                return;
            stream_ = stream;
            gwn_throw_if_cuda_error(
                cudaMemcpyAsync(
                    data_, src.data(), src.size_bytes(), cudaMemcpyHostToDevice, stream
                ),
                "cudaMemcpyAsync"
            );
            return;
        }
        if (src.empty()) {
            clear(stream);
            return;
        }
        if (src.size() > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::invalid_argument("gwn_device_array source span exceeds addressable bytes.");

        T *replacement = allocate_(src.size(), stream);
        auto release_replacement =
            gwn_make_scope_exit([&]() noexcept { deallocate_(replacement, stream); });
        gwn_throw_if_cuda_error(
            cudaMemcpyAsync(
                replacement, src.data(), src.size_bytes(), cudaMemcpyHostToDevice, stream
            ),
            "cudaMemcpyAsync"
        );

        // Size-changing copy is a replacement operation: the old state remains intact until both
        // allocation and copy enqueue have succeeded.
        deallocate_(data_, stream);
        data_ = replacement;
        size_ = src.size();
        stream_ = stream;
        release_replacement.release();
    }

    /// Copy contents to host storage on the currently bound stream.
    ///
    /// The host span follows `cudaMemcpyAsync` lifetime rules.
    void copy_to_host(cuda::std::span<T> const dst) { copy_to_host(dst, stream_); }

    /// Copy contents to host storage on \p stream and publish that read as a lifetime use.
    void copy_to_host(cuda::std::span<T> const dst, cudaStream_t const stream) {
        if (dst.size() != size_)
            throw std::invalid_argument("gwn_device_array destination size mismatch.");
        if (!gwn_span_has_storage(dst))
            throw std::invalid_argument("gwn_device_array destination span has null storage.");
        if (dst.empty())
            return;

        stream_ = stream;
        gwn_throw_if_cuda_error(
            cudaMemcpyAsync(dst.data(), data_, dst.size_bytes(), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync"
        );
    }

    [[nodiscard]] T *data() noexcept { return data_; }
    [[nodiscard]] T const *data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] cuda::std::span<T> span() noexcept { return {data_, size_}; }
    [[nodiscard]] cuda::std::span<T const> span() const noexcept { return {data_, size_}; }

    friend void swap(gwn_device_array &lhs, gwn_device_array &rhs) noexcept {
        using std::swap;
        swap(lhs.data_, rhs.data_);
        swap(lhs.size_, rhs.size_);
        swap(lhs.stream_, rhs.stream_);
    }

private:
    [[nodiscard]] static T *allocate_(std::size_t const count, cudaStream_t const stream) {
        void *storage = nullptr;
        gwn_throw_if_cuda_error(
            cudaMallocAsync(&storage, count * sizeof(T), stream), "cudaMallocAsync"
        );
        GWN_ASSERT(storage != nullptr, "cudaMallocAsync returned null storage for nonzero size.");
        return static_cast<T *>(storage);
    }

    static void deallocate_(T *const storage, cudaStream_t const stream) noexcept {
        if (storage == nullptr)
            return;
        cudaError_t const result = cudaFreeAsync(storage, stream);
        GWN_ASSERT(result == cudaSuccess, "cudaFreeAsync failed with error %d.", int(result));
        if (result != cudaSuccess)
            std::terminate();
    }

    [[nodiscard]] bool invariant_holds_() const noexcept {
        return (data_ == nullptr) == (size_ == 0);
    }

    T *data_{nullptr};
    std::size_t size_{0};
    cudaStream_t stream_{cudaStreamLegacy};
};

} // namespace gwn::detail
