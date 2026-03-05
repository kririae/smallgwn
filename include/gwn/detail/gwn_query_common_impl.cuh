#pragma once

#include <cstdint>

#include "../gwn_utils.cuh"

namespace gwn {

enum class gwn_ray_first_hit_status : std::uint8_t {
    k_miss = 0,
    k_hit = 1,
    k_overflow = 2,
};

enum class gwn_closest_triangle_normal_status : std::uint8_t {
    k_miss = 0,
    k_hit = 1,
    k_overflow = 2,
};

} // namespace gwn

namespace gwn::detail {

struct gwn_traversal_overflow_trap_callback {
    __device__ inline void operator()() const noexcept { gwn_trap(); }
};

} // namespace gwn::detail
