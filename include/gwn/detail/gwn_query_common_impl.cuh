#pragma once

#include <cstdint>

#include "../gwn_bvh.cuh"

namespace gwn::detail {

/// \brief Trap the calling thread when traversal cannot represent its pending work.
struct gwn_traversal_overflow_trap_callback {
    __device__ inline void operator()() const noexcept { gwn_trap(); }
};

/// \brief Validate the internal-reference stack bound shared by every batch traversal.
template <int StackCapacity, int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] inline gwn_status gwn_validate_traversal_stack_capacity_impl(
    gwn_bvh_accessor<Width, Real, Index> const &bvh
) noexcept {
    static_assert(StackCapacity > 0, "Traversal stack capacity must be positive.");
    if (!bvh.has_internal_root())
        return gwn_status::ok();

    if (bvh.internal_stack_bound > std::uint64_t(StackCapacity)) {
        return gwn_status::invalid_argument(
            "Traversal stack capacity is below the BVH internal stack bound."
        );
    }
    return gwn_status::ok();
}

} // namespace gwn::detail
