#pragma once

#include <atomic>
#include <cstdint>
#include <limits>

namespace gwn::detail {

/// \brief Reserve a process-unique nonzero revision for a BVH mutation.
[[nodiscard]] inline std::uint64_t gwn_reserve_bvh_revision() noexcept {
    static std::atomic<std::uint64_t> next_revision{1};
    std::uint64_t revision = next_revision.load(std::memory_order_relaxed);
    for (;;) {
        // Leave the counter saturated. Wrapping would eventually let an old moment pass an
        // ABA-style identity comparison after cudaMallocAsync reused the BVH allocations.
        if (revision == 0 || revision == std::numeric_limits<std::uint64_t>::max())
            return 0;
        if (next_revision.compare_exchange_weak(
                revision, revision + 1, std::memory_order_relaxed, std::memory_order_relaxed
            )) {
            return revision;
        }
    }
}

} // namespace gwn::detail
