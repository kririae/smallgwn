#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>

#include "../gwn_kernel_utils.cuh"
#include "../gwn_utils.cuh"

namespace gwn {
namespace detail {

template <gwn_index_type Index>
struct gwn_edge_key_traits;

template <>
struct gwn_edge_key_traits<std::uint32_t> {
    using key_type = std::uint64_t;

    static __host__ __device__ inline key_type
    pack(std::uint32_t const a, std::uint32_t const b) noexcept {
        std::uint32_t const lo = (a < b) ? a : b;
        std::uint32_t const hi = (a < b) ? b : a;
        return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
    }

    static __host__ __device__ inline void
    unpack(key_type const key, std::uint32_t &i0, std::uint32_t &i1) noexcept {
        i0 = static_cast<std::uint32_t>(key >> 32);
        i1 = static_cast<std::uint32_t>(key & 0xFFFFFFFFu);
    }
};

template <gwn_index_type Index>
struct gwn_expand_oriented_edges_u32_functor {
    static_assert(std::is_same_v<Index, std::uint32_t>);
    using Key = typename gwn_edge_key_traits<Index>::key_type;

    cuda::std::span<Index const> tri_i0{};
    cuda::std::span<Index const> tri_i1{};
    cuda::std::span<Index const> tri_i2{};
    cuda::std::span<Key> edge_keys{};
    cuda::std::span<std::int32_t> edge_signs{};

    __device__ void operator()(std::size_t const tri_id) const {
        Index const a = tri_i0[tri_id];
        Index const b = tri_i1[tri_id];
        Index const c = tri_i2[tri_id];

        auto const emit = [&](std::size_t const slot, Index const u, Index const v) {
            if (u == v) {
                edge_keys[slot] = Key(0);
                edge_signs[slot] = 0;
                return;
            }
            edge_keys[slot] = gwn_edge_key_traits<Index>::pack(u, v);
            edge_signs[slot] = (u < v) ? std::int32_t(1) : std::int32_t(-1);
        };

        std::size_t const base = tri_id * 3;
        emit(base + 0, a, b);
        emit(base + 1, b, c);
        emit(base + 2, c, a);
    }
};

template <gwn_index_type Index>
struct gwn_decode_singular_edge_keys_u32_functor {
    static_assert(std::is_same_v<Index, std::uint32_t>);
    using Key = typename gwn_edge_key_traits<Index>::key_type;

    cuda::std::span<Key const> keys{};
    cuda::std::span<Index> out_i0{};
    cuda::std::span<Index> out_i1{};

    __device__ void operator()(std::size_t const idx) const {
        Index a{}, b{};
        gwn_edge_key_traits<Index>::unpack(keys[idx], a, b);
        out_i0[idx] = a;
        out_i1[idx] = b;
    }
};

struct gwn_edge_key_u64x2 {
    std::uint64_t lo{};
    std::uint64_t hi{};

    __host__ __device__ inline bool operator==(gwn_edge_key_u64x2 const &other) const noexcept {
        return lo == other.lo && hi == other.hi;
    }
};

struct gwn_u64_edge_lo_sign {
    std::uint64_t lo{};
    std::int32_t sign{};
};

struct gwn_u64_edge_hi_sign {
    std::uint64_t hi{};
    std::int32_t sign{};
};

struct gwn_expand_oriented_edges_u64_functor {
    cuda::std::span<std::uint64_t const> tri_i0{};
    cuda::std::span<std::uint64_t const> tri_i1{};
    cuda::std::span<std::uint64_t const> tri_i2{};

    cuda::std::span<std::uint64_t> key_hi{};
    cuda::std::span<gwn_u64_edge_lo_sign> value_lo_sign{};

    __device__ void operator()(std::size_t const tri_id) const {
        std::uint64_t const a = tri_i0[tri_id];
        std::uint64_t const b = tri_i1[tri_id];
        std::uint64_t const c = tri_i2[tri_id];

        auto const emit = [&](std::size_t const slot, std::uint64_t const u, std::uint64_t const v) {
            if (u == v) {
                key_hi[slot] = 0;
                value_lo_sign[slot] = gwn_u64_edge_lo_sign{0, 0};
                return;
            }
            std::uint64_t const lo = (u < v) ? u : v;
            std::uint64_t const hi = (u < v) ? v : u;
            std::int32_t const sign = (u < v) ? std::int32_t(1) : std::int32_t(-1);
            key_hi[slot] = hi;
            value_lo_sign[slot] = gwn_u64_edge_lo_sign{lo, sign};
        };

        std::size_t const base = tri_id * 3;
        emit(base + 0, a, b);
        emit(base + 1, b, c);
        emit(base + 2, c, a);
    }
};

struct gwn_prepare_second_sort_u64_functor {
    cuda::std::span<std::uint64_t const> sorted_hi{};
    cuda::std::span<gwn_u64_edge_lo_sign const> sorted_lo_sign{};

    cuda::std::span<std::uint64_t> key_lo{};
    cuda::std::span<gwn_u64_edge_hi_sign> value_hi_sign{};

    __device__ void operator()(std::size_t const idx) const {
        key_lo[idx] = sorted_lo_sign[idx].lo;
        value_hi_sign[idx] = gwn_u64_edge_hi_sign{sorted_hi[idx], sorted_lo_sign[idx].sign};
    }
};

struct gwn_pack_reduce_u64_functor {
    cuda::std::span<std::uint64_t const> sorted_lo{};
    cuda::std::span<gwn_u64_edge_hi_sign const> sorted_hi_sign{};

    cuda::std::span<gwn_edge_key_u64x2> out_keys{};
    cuda::std::span<std::int32_t> out_signs{};

    __device__ void operator()(std::size_t const idx) const {
        out_keys[idx] = gwn_edge_key_u64x2{sorted_lo[idx], sorted_hi_sign[idx].hi};
        out_signs[idx] = sorted_hi_sign[idx].sign;
    }
};

struct gwn_decode_singular_edge_keys_u64_functor {
    cuda::std::span<gwn_edge_key_u64x2 const> keys{};
    cuda::std::span<std::uint64_t> out_i0{};
    cuda::std::span<std::uint64_t> out_i1{};

    __device__ void operator()(std::size_t const idx) const {
        out_i0[idx] = keys[idx].lo;
        out_i1[idx] = keys[idx].hi;
    }
};

template <gwn_index_type Index>
[[nodiscard]] gwn_status gwn_build_singular_edges(
    cuda::std::span<Index const> const d_tri_i0,
    cuda::std::span<Index const> const d_tri_i1,
    cuda::std::span<Index const> const d_tri_i2,
    gwn_device_array<Index> &out_i0,
    gwn_device_array<Index> &out_i1,
    std::size_t &out_count,
    cudaStream_t const stream
) noexcept {
    static_assert(
        std::is_same_v<Index, std::uint32_t> || std::is_same_v<Index, std::uint64_t>,
        "Singular-edge builder requires uint32_t or uint64_t Index."
    );

    constexpr int k_block = k_gwn_default_block_size;
    std::size_t const T = d_tri_i0.size();
    out_count = 0;

    if (T == 0) {
        GWN_RETURN_ON_ERROR(out_i0.clear(stream));
        GWN_RETURN_ON_ERROR(out_i1.clear(stream));
        return gwn_status::ok();
    }

    std::size_t const E = T * 3;
    auto const item_count = static_cast<std::uint64_t>(E);

    if constexpr (std::is_same_v<Index, std::uint32_t>) {
        using Key = typename gwn_edge_key_traits<Index>::key_type;

        gwn_device_array<Key> edge_keys{};
        gwn_device_array<std::int32_t> edge_signs{};
        GWN_RETURN_ON_ERROR(edge_keys.resize(E, stream));
        GWN_RETURN_ON_ERROR(edge_signs.resize(E, stream));

        GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
            T,
            gwn_expand_oriented_edges_u32_functor<Index>{
                d_tri_i0, d_tri_i1, d_tri_i2, edge_keys.span(), edge_signs.span()
            },
            stream
        ));

        gwn_device_array<Key> sorted_keys{};
        gwn_device_array<std::int32_t> sorted_signs{};
        GWN_RETURN_ON_ERROR(sorted_keys.resize(E, stream));
        GWN_RETURN_ON_ERROR(sorted_signs.resize(E, stream));

        gwn_device_array<std::uint8_t> cub_temp{};
        constexpr int k_sort_end_bit = static_cast<int>(sizeof(Key) * 8);

        std::size_t sort_temp_bytes = 0;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_bytes,
            edge_keys.data(), sorted_keys.data(),
            edge_signs.data(), sorted_signs.data(),
            item_count, 0, k_sort_end_bit, stream
        )));
        GWN_RETURN_ON_ERROR(cub_temp.resize(sort_temp_bytes, stream));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceRadixSort::SortPairs(
            cub_temp.data(), sort_temp_bytes,
            edge_keys.data(), sorted_keys.data(),
            edge_signs.data(), sorted_signs.data(),
            item_count, 0, k_sort_end_bit, stream
        )));

        GWN_RETURN_ON_ERROR(edge_keys.clear(stream));
        GWN_RETURN_ON_ERROR(edge_signs.clear(stream));

        gwn_device_array<Key> unique_keys{};
        gwn_device_array<std::int32_t> reduced_sums{};
        gwn_device_array<std::size_t> num_runs_dev{};
        GWN_RETURN_ON_ERROR(unique_keys.resize(E, stream));
        GWN_RETURN_ON_ERROR(reduced_sums.resize(E, stream));
        GWN_RETURN_ON_ERROR(num_runs_dev.resize(1, stream));
        GWN_RETURN_ON_ERROR(reduced_sums.zero(stream));

        ::cuda::std::plus<std::int32_t> sum_op{};
        std::size_t reduce_temp_bytes = 0;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceReduce::ReduceByKey(
            nullptr, reduce_temp_bytes,
            sorted_keys.data(), unique_keys.data(),
            sorted_signs.data(), reduced_sums.data(),
            num_runs_dev.data(), sum_op, item_count, stream
        )));
        if (reduce_temp_bytes > cub_temp.size())
            GWN_RETURN_ON_ERROR(cub_temp.resize(reduce_temp_bytes, stream));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceReduce::ReduceByKey(
            cub_temp.data(), reduce_temp_bytes,
            sorted_keys.data(), unique_keys.data(),
            sorted_signs.data(), reduced_sums.data(),
            num_runs_dev.data(), sum_op, item_count, stream
        )));

        GWN_RETURN_ON_ERROR(sorted_keys.clear(stream));
        GWN_RETURN_ON_ERROR(sorted_signs.clear(stream));
        GWN_RETURN_ON_ERROR(num_runs_dev.clear(stream));

        gwn_device_array<Key> singular_keys{};
        gwn_device_array<std::size_t> num_singular_dev{};
        GWN_RETURN_ON_ERROR(singular_keys.resize(E, stream));
        GWN_RETURN_ON_ERROR(num_singular_dev.resize(1, stream));

        std::size_t select_temp_bytes = 0;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceSelect::Flagged(
            nullptr, select_temp_bytes,
            unique_keys.data(), reduced_sums.data(),
            singular_keys.data(), num_singular_dev.data(),
            item_count, stream
        )));
        if (select_temp_bytes > cub_temp.size())
            GWN_RETURN_ON_ERROR(cub_temp.resize(select_temp_bytes, stream));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceSelect::Flagged(
            cub_temp.data(), select_temp_bytes,
            unique_keys.data(), reduced_sums.data(),
            singular_keys.data(), num_singular_dev.data(),
            item_count, stream
        )));

        GWN_RETURN_ON_ERROR(unique_keys.clear(stream));
        GWN_RETURN_ON_ERROR(reduced_sums.clear(stream));
        GWN_RETURN_ON_ERROR(cub_temp.clear(stream));

        std::size_t num_singular = 0;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
            &num_singular, num_singular_dev.data(), sizeof(std::size_t),
            cudaMemcpyDeviceToHost, stream
        )));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
        GWN_RETURN_ON_ERROR(num_singular_dev.clear(stream));

        out_count = num_singular;
        if (num_singular == 0) {
            GWN_RETURN_ON_ERROR(singular_keys.clear(stream));
            GWN_RETURN_ON_ERROR(out_i0.clear(stream));
            GWN_RETURN_ON_ERROR(out_i1.clear(stream));
            return gwn_status::ok();
        }

        GWN_RETURN_ON_ERROR(out_i0.resize(num_singular, stream));
        GWN_RETURN_ON_ERROR(out_i1.resize(num_singular, stream));
        GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
            num_singular,
            gwn_decode_singular_edge_keys_u32_functor<Index>{
                cuda::std::span<Key const>(singular_keys.data(), num_singular),
                out_i0.span(), out_i1.span()
            },
            stream
        ));

        GWN_RETURN_ON_ERROR(singular_keys.clear(stream));
        return gwn_status::ok();
    } else {
        gwn_device_array<std::uint64_t> key_hi{};
        gwn_device_array<gwn_u64_edge_lo_sign> value_lo_sign{};
        GWN_RETURN_ON_ERROR(key_hi.resize(E, stream));
        GWN_RETURN_ON_ERROR(value_lo_sign.resize(E, stream));

        GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
            T,
            gwn_expand_oriented_edges_u64_functor{
                d_tri_i0, d_tri_i1, d_tri_i2, key_hi.span(), value_lo_sign.span()
            },
            stream
        ));

        gwn_device_array<std::uint64_t> key_hi_sorted{};
        gwn_device_array<gwn_u64_edge_lo_sign> value_lo_sign_sorted{};
        GWN_RETURN_ON_ERROR(key_hi_sorted.resize(E, stream));
        GWN_RETURN_ON_ERROR(value_lo_sign_sorted.resize(E, stream));

        gwn_device_array<std::uint8_t> cub_temp{};
        constexpr int k_u64_sort_end_bit = static_cast<int>(sizeof(std::uint64_t) * 8);

        std::size_t sort_temp_bytes = 0;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_bytes,
            key_hi.data(), key_hi_sorted.data(),
            value_lo_sign.data(), value_lo_sign_sorted.data(),
            item_count, 0, k_u64_sort_end_bit, stream
        )));
        GWN_RETURN_ON_ERROR(cub_temp.resize(sort_temp_bytes, stream));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceRadixSort::SortPairs(
            cub_temp.data(), sort_temp_bytes,
            key_hi.data(), key_hi_sorted.data(),
            value_lo_sign.data(), value_lo_sign_sorted.data(),
            item_count, 0, k_u64_sort_end_bit, stream
        )));

        GWN_RETURN_ON_ERROR(key_hi.clear(stream));
        GWN_RETURN_ON_ERROR(value_lo_sign.clear(stream));

        gwn_device_array<std::uint64_t> key_lo{};
        gwn_device_array<gwn_u64_edge_hi_sign> value_hi_sign{};
        GWN_RETURN_ON_ERROR(key_lo.resize(E, stream));
        GWN_RETURN_ON_ERROR(value_hi_sign.resize(E, stream));

        GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
            E,
            gwn_prepare_second_sort_u64_functor{
                key_hi_sorted.span(), value_lo_sign_sorted.span(),
                key_lo.span(), value_hi_sign.span()
            },
            stream
        ));

        GWN_RETURN_ON_ERROR(key_hi_sorted.clear(stream));
        GWN_RETURN_ON_ERROR(value_lo_sign_sorted.clear(stream));

        gwn_device_array<std::uint64_t> key_lo_sorted{};
        gwn_device_array<gwn_u64_edge_hi_sign> value_hi_sign_sorted{};
        GWN_RETURN_ON_ERROR(key_lo_sorted.resize(E, stream));
        GWN_RETURN_ON_ERROR(value_hi_sign_sorted.resize(E, stream));

        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_bytes,
            key_lo.data(), key_lo_sorted.data(),
            value_hi_sign.data(), value_hi_sign_sorted.data(),
            item_count, 0, k_u64_sort_end_bit, stream
        )));
        if (sort_temp_bytes > cub_temp.size())
            GWN_RETURN_ON_ERROR(cub_temp.resize(sort_temp_bytes, stream));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceRadixSort::SortPairs(
            cub_temp.data(), sort_temp_bytes,
            key_lo.data(), key_lo_sorted.data(),
            value_hi_sign.data(), value_hi_sign_sorted.data(),
            item_count, 0, k_u64_sort_end_bit, stream
        )));

        GWN_RETURN_ON_ERROR(key_lo.clear(stream));
        GWN_RETURN_ON_ERROR(value_hi_sign.clear(stream));

        gwn_device_array<gwn_edge_key_u64x2> edge_keys{};
        gwn_device_array<std::int32_t> edge_signs{};
        GWN_RETURN_ON_ERROR(edge_keys.resize(E, stream));
        GWN_RETURN_ON_ERROR(edge_signs.resize(E, stream));

        GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
            E,
            gwn_pack_reduce_u64_functor{
                key_lo_sorted.span(), value_hi_sign_sorted.span(),
                edge_keys.span(), edge_signs.span()
            },
            stream
        ));

        GWN_RETURN_ON_ERROR(key_lo_sorted.clear(stream));
        GWN_RETURN_ON_ERROR(value_hi_sign_sorted.clear(stream));

        gwn_device_array<gwn_edge_key_u64x2> unique_keys{};
        gwn_device_array<std::int32_t> reduced_sums{};
        gwn_device_array<std::size_t> num_runs_dev{};
        GWN_RETURN_ON_ERROR(unique_keys.resize(E, stream));
        GWN_RETURN_ON_ERROR(reduced_sums.resize(E, stream));
        GWN_RETURN_ON_ERROR(num_runs_dev.resize(1, stream));
        GWN_RETURN_ON_ERROR(reduced_sums.zero(stream));

        ::cuda::std::plus<std::int32_t> sum_op{};
        std::size_t reduce_temp_bytes = 0;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceReduce::ReduceByKey(
            nullptr, reduce_temp_bytes,
            edge_keys.data(), unique_keys.data(),
            edge_signs.data(), reduced_sums.data(),
            num_runs_dev.data(), sum_op, item_count, stream
        )));
        if (reduce_temp_bytes > cub_temp.size())
            GWN_RETURN_ON_ERROR(cub_temp.resize(reduce_temp_bytes, stream));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceReduce::ReduceByKey(
            cub_temp.data(), reduce_temp_bytes,
            edge_keys.data(), unique_keys.data(),
            edge_signs.data(), reduced_sums.data(),
            num_runs_dev.data(), sum_op, item_count, stream
        )));

        GWN_RETURN_ON_ERROR(edge_keys.clear(stream));
        GWN_RETURN_ON_ERROR(edge_signs.clear(stream));
        GWN_RETURN_ON_ERROR(num_runs_dev.clear(stream));

        gwn_device_array<gwn_edge_key_u64x2> singular_keys{};
        gwn_device_array<std::size_t> num_singular_dev{};
        GWN_RETURN_ON_ERROR(singular_keys.resize(E, stream));
        GWN_RETURN_ON_ERROR(num_singular_dev.resize(1, stream));

        std::size_t select_temp_bytes = 0;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceSelect::Flagged(
            nullptr, select_temp_bytes,
            unique_keys.data(), reduced_sums.data(),
            singular_keys.data(), num_singular_dev.data(),
            item_count, stream
        )));
        if (select_temp_bytes > cub_temp.size())
            GWN_RETURN_ON_ERROR(cub_temp.resize(select_temp_bytes, stream));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cub::DeviceSelect::Flagged(
            cub_temp.data(), select_temp_bytes,
            unique_keys.data(), reduced_sums.data(),
            singular_keys.data(), num_singular_dev.data(),
            item_count, stream
        )));

        GWN_RETURN_ON_ERROR(unique_keys.clear(stream));
        GWN_RETURN_ON_ERROR(reduced_sums.clear(stream));
        GWN_RETURN_ON_ERROR(cub_temp.clear(stream));

        std::size_t num_singular = 0;
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaMemcpyAsync(
            &num_singular, num_singular_dev.data(), sizeof(std::size_t),
            cudaMemcpyDeviceToHost, stream
        )));
        GWN_RETURN_ON_ERROR(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
        GWN_RETURN_ON_ERROR(num_singular_dev.clear(stream));

        out_count = num_singular;
        if (num_singular == 0) {
            GWN_RETURN_ON_ERROR(singular_keys.clear(stream));
            GWN_RETURN_ON_ERROR(out_i0.clear(stream));
            GWN_RETURN_ON_ERROR(out_i1.clear(stream));
            return gwn_status::ok();
        }

        GWN_RETURN_ON_ERROR(out_i0.resize(num_singular, stream));
        GWN_RETURN_ON_ERROR(out_i1.resize(num_singular, stream));
        GWN_RETURN_ON_ERROR(gwn_launch_linear_kernel<k_block>(
            num_singular,
            gwn_decode_singular_edge_keys_u64_functor{
                cuda::std::span<gwn_edge_key_u64x2 const>(singular_keys.data(), num_singular),
                out_i0.span(), out_i1.span()
            },
            stream
        ));

        GWN_RETURN_ON_ERROR(singular_keys.clear(stream));
        return gwn_status::ok();
    }
}

} // namespace detail
} // namespace gwn
