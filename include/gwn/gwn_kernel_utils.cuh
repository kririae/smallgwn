#pragma once

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "gwn_utils.hpp"

namespace gwn::detail {

inline constexpr int k_gwn_default_block_size = 128;

template <int BlockSize>
[[nodiscard]] constexpr int gwn_block_count_1d(std::size_t const element_count) noexcept {
    static_assert(BlockSize > 0, "BlockSize must be positive.");
    if (element_count == 0)
        return 0;

    constexpr std::size_t k_block_size = static_cast<std::size_t>(BlockSize);
    std::size_t const block_count = (element_count + k_block_size - 1) / k_block_size;
    return static_cast<int>(block_count);
}

template <int BlockSize>
[[nodiscard]] constexpr dim3 gwn_grid_dim_1d(std::size_t const element_count) noexcept {
    return dim3{static_cast<unsigned int>(gwn_block_count_1d<BlockSize>(element_count)), 1u, 1u};
}

template <int BlockSize> [[nodiscard]] constexpr dim3 gwn_block_dim_1d() noexcept {
    static_assert(BlockSize > 0, "BlockSize must be positive.");
    return dim3{static_cast<unsigned int>(BlockSize), 1u, 1u};
}

[[nodiscard]] __device__ inline std::size_t gwn_global_thread_index_1d() noexcept {
    return static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
}

[[nodiscard]] inline gwn_status gwn_check_last_kernel() noexcept {
    return gwn_cuda_to_status(cudaGetLastError());
}

template <typename Functor>
concept gwn_linear_index_functor = std::is_trivially_copyable_v<Functor> &&
                                   requires(Functor const functor, std::size_t const index) {
                                       { functor(index) };
                                   };

template <int BlockSize, gwn_linear_index_functor Functor>
__global__ __launch_bounds__(BlockSize) void gwn_linear_kernel(
    std::size_t const element_count, Functor const __grid_constant__ functor
) {
    std::size_t const global_index = gwn_global_thread_index_1d();
    if (global_index >= element_count)
        return;

    functor(global_index);
}

template <int BlockSize, gwn_linear_index_functor Functor>
gwn_status gwn_launch_linear_kernel(
    std::size_t const element_count, Functor const &functor,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    if (element_count == 0)
        return gwn_status::ok();

    int const block_count = gwn_block_count_1d<BlockSize>(element_count);
    gwn_linear_kernel<BlockSize, Functor>
        <<<block_count, BlockSize, 0, stream>>>(element_count, functor);
    return gwn_check_last_kernel();
}

} // namespace gwn::detail
