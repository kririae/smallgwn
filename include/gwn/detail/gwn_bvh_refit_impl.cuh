#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "../gwn_bvh_refit.cuh"
#include "gwn_bvh_refit_common.cuh"
#include "gwn_bvh_revision.cuh"
#include "gwn_bvh_status_helpers.cuh"
#include "gwn_bvh_triangle_impl.cuh"
#include "gwn_device_array.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {
namespace detail {

/// \brief Compute leaf bounds and propagate completed internal bounds to the root.
///
/// One thread owns each leaf range. The final child arriving at a node combines every pending
/// child bound and continues upward. Release fences before the arrival counter make sibling writes
/// visible to that final thread without serializing unrelated subtrees.
template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_refit_bvh_from_leaves_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    gwn_bvh_refit_state<Index> state{};
    cuda::std::span<gwn_aabb<Real>> pending{};
    cuda::std::span<gwn_aabb<Real>> node_bounds{};
    unsigned int *error_flag = nullptr;

    void __device__ operator()(std::size_t const edge_index) const noexcept {
        auto signal_error = [&]() noexcept { atomicExch(error_flag, 1u); };
        auto pending_index = [](std::size_t const node_index, int const child_slot) noexcept {
            return node_index * static_cast<std::size_t>(Width) +
                   static_cast<std::size_t>(child_slot);
        };
        auto combine_node = [&](std::size_t const node_index, gwn_aabb<Real> &bounds) noexcept {
            bool initialized = false;
            auto const &node = bvh.nodes[node_index];
            for (int child_slot = 0; child_slot < Width; ++child_slot) {
                if (!node.child(child_slot).is_valid())
                    continue;
                auto const &child_bounds = pending[pending_index(node_index, child_slot)];
                bounds = initialized ? gwn_aabb_union(bounds, child_bounds) : child_bounds;
                initialized = true;
            }
            return initialized;
        };

        std::size_t const node_index = edge_index / static_cast<std::size_t>(Width);
        int const child_slot = static_cast<int>(edge_index % static_cast<std::size_t>(Width));
        if (node_index >= bvh.nodes.size())
            return;
        auto const &leaf = bvh.nodes[node_index].child(child_slot);
        if (!leaf.is_leaf())
            return;

        gwn_aabb<Real> leaf_bounds{};
        bool initialized = false;
        // Leaf ranges partition primitive order. One thread can replace each triangle record and
        // reduce its bounds without atomics or duplicate geometry loads.
        for (std::uint32_t primitive_offset = 0; primitive_offset < leaf.primitive_count();
             ++primitive_offset) {
            std::uint64_t const sorted_index = leaf.offset() + primitive_offset;
            if (sorted_index >= bvh.primitive_indices.size()) {
                signal_error();
                return;
            }

            auto const payload = gwn_compute_bvh_triangle(
                geometry, bvh.primitive_indices[static_cast<std::size_t>(sorted_index)]
            );
            if (!payload.is_valid) {
                signal_error();
                return;
            }
            bvh.triangles[static_cast<std::size_t>(sorted_index)] = payload.triangle;
            leaf_bounds =
                initialized ? gwn_aabb_union(leaf_bounds, payload.bounds) : payload.bounds;
            initialized = true;
        }
        if (!initialized) {
            signal_error();
            return;
        }

        Index current_parent = static_cast<Index>(node_index);
        std::uint32_t current_slot = static_cast<std::uint32_t>(child_slot);
        gwn_aabb<Real> current_bounds = leaf_bounds;
        while (gwn_is_valid_index(current_parent)) {
            auto const parent_index = static_cast<std::size_t>(current_parent);
            if (parent_index >= bvh.nodes.size() || current_slot >= Width) {
                signal_error();
                return;
            }

            pending[pending_index(parent_index, current_slot)] = current_bounds;
            __threadfence();
            // The last child to arrive owns the parent reduction. It can continue up several
            // levels, which removes one kernel launch per depth from dynamic refit.
            unsigned int const arrival = atomicAdd(state.arrivals.data() + parent_index, 1u) + 1u;
            unsigned int const expected = state.arity[parent_index];
            if (expected == 0u || arrival > expected) {
                signal_error();
                return;
            }
            if (arrival < expected)
                return;

            __threadfence();
            if (!combine_node(parent_index, current_bounds)) {
                signal_error();
                return;
            }
            node_bounds[parent_index] = current_bounds;

            Index const next_parent = state.parent[parent_index];
            if (gwn_is_invalid_index(next_parent))
                return;
            current_parent = next_parent;
            current_slot = state.parent_slot[parent_index];
        }
    }
};

/// \brief Commit propagated child bounds after the bottom-up pass converges.
template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_finalize_bvh_refit_functor {
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    gwn_bvh_refit_state<Index> state{};
    cuda::std::span<gwn_aabb<Real> const> pending{};
    unsigned int *error_flag = nullptr;

    void __device__ operator()(std::size_t const node_index) const noexcept {
        if (state.arrivals[node_index] != state.arity[node_index]) {
            atomicExch(error_flag, 1u);
            return;
        }

        auto &node = bvh.nodes[node_index];
        // Administration is untouched. Only the bounds beside each reference are committed.
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto &child = node.child(child_slot);
            if (!child.is_valid())
                continue;
            child.bounds = pending
                [node_index * static_cast<std::size_t>(Width) +
                 static_cast<std::size_t>(child_slot)];
        }
    }
};

/// \brief Refit a root leaf range without allocating internal propagation state.
template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_refit_bvh_leaf_root_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    gwn_aabb<Real> *root_bounds = nullptr;
    unsigned int *error_flag = nullptr;

    void __device__ operator()(std::size_t const) const noexcept {
        gwn_aabb<Real> bounds{};
        bool initialized = false;
        // Packed root leaves are capped by the 16-bit count field. One thread owns the complete
        // range, so triangle replacement and the final root bound require no arrival counter.
        for (std::uint32_t primitive_offset = 0; primitive_offset < bvh.root.primitive_count();
             ++primitive_offset) {
            std::uint64_t const sorted_index = bvh.root.offset() + primitive_offset;
            if (sorted_index >= bvh.primitive_indices.size()) {
                atomicExch(error_flag, 1u);
                return;
            }
            auto const payload = gwn_compute_bvh_triangle(
                geometry, bvh.primitive_indices[static_cast<std::size_t>(sorted_index)]
            );
            if (!payload.is_valid) {
                atomicExch(error_flag, 1u);
                return;
            }
            bvh.triangles[static_cast<std::size_t>(sorted_index)] = payload.triangle;
            bounds = initialized ? gwn_aabb_union(bounds, payload.bounds) : payload.bounds;
            initialized = true;
        }
        if (!initialized) {
            atomicExch(error_flag, 1u);
            return;
        }
        *root_bounds = bounds;
    }
};

/// \brief Refit canonical BVH bounds and triangle records without changing hierarchy state.
template <int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_refit_bvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Width, Real, Index> &bvh,
    cudaStream_t const stream
) noexcept {
    return gwn_try_translate_status("gwn_refit_bvh_impl", [&]() -> gwn_status {
        if (!geometry.is_valid())
            gwn_bvh_invalid_argument(k_gwn_bvh_phase_refit, "Geometry accessor is invalid.");
        if (!bvh.is_valid())
            gwn_bvh_invalid_argument(k_gwn_bvh_phase_refit, "BVH accessor is invalid.");
        if (geometry.triangle_count() != bvh.primitive_indices.size()) {
            gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_refit,
                "Geometry triangle count differs from the BVH primitive order."
            );
        }
        std::uint64_t const next_revision = gwn_reserve_bvh_revision();
        if (next_revision == 0) {
            gwn_bvh_internal_error(
                k_gwn_bvh_phase_refit, "BVH geometry-derived state revision is exhausted."
            );
        }
        constexpr int k_block_size = k_gwn_default_block_size;
        gwn_device_array<unsigned int> error_flag{};
        gwn_device_array<gwn_aabb<Real>> root_bounds{};
        error_flag.resize(1, stream);
        error_flag.zero(stream);
        root_bounds.resize(1, stream);

        if (bvh.has_leaf_root()) {
            // A leaf root has no node array or parent state. Its bounded primitive count makes a
            // single range worker cheaper than setting up the internal propagation buffers.
            // Clear the published identity before the first in-place write can be enqueued. A
            // later failure then makes this BVH and every moment formed from it unqueryable.
            bvh.revision = 0;
            gwn_throw_status_error(
                gwn_launch_linear_kernel<k_block_size>(
                    1,
                    gwn_refit_bvh_leaf_root_functor<Width, Real, Index>{
                        geometry, bvh, root_bounds.data(), error_flag.data()
                    },
                    stream
                )
            );
        } else {
            std::size_t const node_count = bvh.nodes.size();
            if (node_count >
                std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(Width)) {
                gwn_bvh_invalid_argument(
                    k_gwn_bvh_phase_refit, "BVH child count exceeds the addressable range."
                );
            }
            std::size_t const child_count = node_count * static_cast<std::size_t>(Width);
            gwn_device_array<Index> parent{};
            gwn_device_array<std::uint32_t> parent_slot{};
            gwn_device_array<std::uint32_t> arity{};
            gwn_device_array<unsigned int> arrivals{};
            gwn_device_array<gwn_aabb<Real>> pending{};
            gwn_device_array<gwn_aabb<Real>> node_bounds{};
            parent.resize(node_count, stream);
            parent_slot.resize(node_count, stream);
            arity.resize(node_count, stream);
            arrivals.resize(node_count, stream);
            pending.resize(child_count, stream);
            node_bounds.resize(node_count, stream);
            gwn_throw_status_error(gwn_cuda_to_status(
                cudaMemsetAsync(parent.data(), 0xff, parent.size() * sizeof(Index), stream)
            ));
            gwn_throw_status_error(gwn_cuda_to_status(cudaMemsetAsync(
                parent_slot.data(), 0xff, parent_slot.size() * sizeof(std::uint32_t), stream
            )));
            arity.zero(stream);
            arrivals.zero(stream);

            gwn_bvh_refit_state<Index> const state{
                parent.span(), parent_slot.span(), arity.span(), arrivals.span()
            };
            // Phase 1 recovers parent links and checks every packed range. Phase 2 starts from all
            // leaves and propagates completed bounds. Phase 3 writes child-local bounds only after
            // every node has received its declared arity. Clear the published identity before
            // these launches because a later failure cannot roll back their in-place writes.
            bvh.revision = 0;
            gwn_throw_status_error(
                gwn_launch_linear_kernel<k_block_size>(
                    node_count,
                    gwn_prepare_bvh_refit_functor<Width, Real, Index>{
                        bvh, state, error_flag.data()
                    },
                    stream
                )
            );
            gwn_throw_status_error(
                gwn_launch_linear_kernel<k_block_size>(
                    child_count,
                    gwn_refit_bvh_from_leaves_functor<Width, Real, Index>{
                        geometry, bvh, state, pending.span(), node_bounds.span(), error_flag.data()
                    },
                    stream
                )
            );
            gwn_throw_status_error(
                gwn_launch_linear_kernel<k_block_size>(
                    node_count,
                    gwn_finalize_bvh_refit_functor<Width, Real, Index>{
                        bvh,
                        state,
                        cuda::std::span<gwn_aabb<Real> const>(pending.data(), pending.size()),
                        error_flag.data(),
                    },
                    stream
                )
            );
            gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
                root_bounds.data(),
                node_bounds.data() + static_cast<std::size_t>(bvh.root.offset()),
                sizeof(gwn_aabb<Real>), cudaMemcpyDeviceToDevice, stream
            )));
        }

        // Root bounds are stored in the host-side accessor value, not in a separate device node.
        // The same synchronization also turns device validation failures into gwn_status.
        unsigned int host_error = 0;
        gwn_aabb<Real> host_root_bounds{};
        gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
            &host_error, error_flag.data(), sizeof(host_error), cudaMemcpyDeviceToHost, stream
        )));
        gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
            &host_root_bounds, root_bounds.data(), sizeof(host_root_bounds), cudaMemcpyDeviceToHost,
            stream
        )));
        gwn_throw_status_error(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
        if (host_error != 0u)
            gwn_bvh_internal_error(k_gwn_bvh_phase_refit, "Bottom-up BVH refit did not converge.");

        bvh.root.bounds = host_root_bounds;
        // Publish the new revision only after every device write and validation has succeeded.
        // Existing moments then fail alignment checks until explicitly refit for this BVH state.
        bvh.revision = next_revision;
        return gwn_status::ok();
    });
}

} // namespace detail

template <int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_refit_bvh(
    gwn_geometry_object<Real, Index> const &geometry, gwn_bvh_object<Width, Real, Index> &bvh,
    cudaStream_t const stream
) noexcept {
    std::uint64_t const revision_before = bvh.accessor().revision;
    gwn_status const status =
        detail::gwn_refit_bvh_impl<Width, Real, Index>(geometry.accessor(), bvh.accessor(), stream);

    // A revision cleared by this call means device writes may have been enqueued. Its allocations
    // must thereafter be released or replaced on the stream that received those writes.
    if (status.is_ok() || (revision_before != 0 && bvh.accessor().revision == 0))
        bvh.set_stream(stream);
    return status;
}

} // namespace gwn
