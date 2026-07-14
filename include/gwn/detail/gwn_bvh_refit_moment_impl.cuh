#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "gwn_bvh_refit_common.cuh"
#include "gwn_bvh_status_helpers.cuh"
#include "gwn_device_array.cuh"

namespace gwn {
namespace detail {

/// \brief Propagate canonical leaf moments to every parent child slot.
template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_refit_bvh_moment_from_leaves_functor {
    using traits = gwn_bvh_moment_refit_traits<Order, Width, Real, Index>;
    using payload_type = typename traits::payload_type;

    gwn_bvh_accessor<Width, Real, Index> bvh{};
    gwn_bvh_refit_state<Index> state{};
    cuda::std::span<payload_type> pending{};
    unsigned int *error_flag = nullptr;

    void __device__ operator()(std::size_t const edge_index) const noexcept {
        auto signal_error = [&]() noexcept { atomicExch(error_flag, 1u); };
        auto pending_index = [](std::size_t const node_index, int const child_slot) noexcept {
            return node_index * static_cast<std::size_t>(Width) +
                   static_cast<std::size_t>(child_slot);
        };
        auto combine_node = [&](std::size_t const node_index, payload_type &payload) noexcept {
            bool initialized = false;
            auto const &node = bvh.nodes[node_index];
            for (int child_slot = 0; child_slot < Width; ++child_slot) {
                if (!node.child(child_slot).is_valid())
                    continue;
                auto const &child_payload = pending[pending_index(node_index, child_slot)];
                if (initialized)
                    traits::combine(payload, child_payload);
                else
                    payload = child_payload;
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

        payload_type current_payload{};
        if (!traits::make_leaf_payload(bvh, leaf, current_payload)) {
            signal_error();
            return;
        }

        Index current_parent = static_cast<Index>(node_index);
        std::uint32_t current_slot = static_cast<std::uint32_t>(child_slot);
        while (gwn_is_valid_index(current_parent)) {
            auto const parent_index = static_cast<std::size_t>(current_parent);
            if (parent_index >= bvh.nodes.size() || current_slot >= Width) {
                signal_error();
                return;
            }

            pending[pending_index(parent_index, current_slot)] = current_payload;
            __threadfence();
            // The last child to arrive owns this node's reduction and continues upward. This is
            // the same bottom-up dependency as canonical bounds refit, but the payload contains
            // additive raw moments instead of an AABB.
            unsigned int const arrival = atomicAdd(state.arrivals.data() + parent_index, 1u) + 1u;
            unsigned int const expected = state.arity[parent_index];
            if (expected == 0u || arrival > expected) {
                signal_error();
                return;
            }
            if (arrival < expected)
                return;

            __threadfence();
            if (!combine_node(parent_index, current_payload)) {
                signal_error();
                return;
            }

            Index const next_parent = state.parent[parent_index];
            if (gwn_is_invalid_index(next_parent))
                return;
            current_parent = next_parent;
            current_slot = state.parent_slot[parent_index];
        }
    }
};

/// \brief Convert propagated raw moments into field-SoA query coefficients.
template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_finalize_bvh_moment_refit_functor {
    using traits = gwn_bvh_moment_refit_traits<Order, Width, Real, Index>;
    using payload_type = typename traits::payload_type;

    gwn_bvh_accessor<Width, Real, Index> bvh{};
    gwn_bvh_refit_state<Index> state{};
    cuda::std::span<payload_type const> pending{};
    typename traits::output_context context{};
    unsigned int *error_flag = nullptr;

    void __device__ operator()(std::size_t const node_index) const noexcept {
        if (state.arity[node_index] == 0u ||
            state.arrivals[node_index] != state.arity[node_index]) {
            atomicExch(error_flag, 1u);
            return;
        }

        auto const &node = bvh.nodes[node_index];
        for (int child_slot = 0; child_slot < Width; ++child_slot) {
            auto const &child = node.child(child_slot);
            if (!child.is_valid()) {
                traits::write_invalid(context, node_index, child_slot);
                continue;
            }
            std::size_t const payload_index =
                node_index * static_cast<std::size_t>(Width) + static_cast<std::size_t>(child_slot);
            traits::write_valid(
                context, child.bounds, node_index, child_slot, pending[payload_index]
            );
        }
    }
};

/// \brief Construct one Taylor order in an empty staging accessor aligned to a canonical BVH.
template <int Order, int Width, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_refit_bvh_moment_impl(
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_bvh_moment_accessor<Width, Order, Real, Index> &moment,
    cudaStream_t const stream = cudaStreamLegacy
) noexcept {
    return gwn_try_translate_status("gwn_refit_bvh_moment_impl", [&]() -> gwn_status {
        static_assert(
            Order == 0 || Order == 1 || Order == 2,
            "gwn_refit_bvh_moment supports Order 0, 1, and 2."
        );

        if (!bvh.is_valid())
            gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_refit_moment, "BVH accessor is invalid for moment refit."
            );
        if (!moment.empty()) {
            gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_refit_moment, "Moment refit output accessor must be empty."
            );
        }

        auto const build_moment = [&](gwn_bvh_moment_accessor<Width, Order, Real, Index> &staging) {
            // The revision identifies this exact built or refit BVH state. It remains unique when
            // cudaMallocAsync reuses storage, so stale coefficients cannot become valid again.
            staging.bvh_revision = bvh.revision;
            if (bvh.has_leaf_root())
                return;

            std::size_t const node_count = bvh.nodes.size();
            if (node_count >
                std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(Width)) {
                gwn_bvh_invalid_argument(
                    k_gwn_bvh_phase_refit_moment, "BVH child count exceeds the addressable range."
                );
            }
            if constexpr (sizeof(Index) < sizeof(std::size_t)) {
                if (node_count > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
                    gwn_bvh_invalid_argument(
                        k_gwn_bvh_phase_refit_moment,
                        "BVH node count exceeds the moment refit's index range."
                    );
                }
            }

            using traits = gwn_bvh_moment_refit_traits<Order, Width, Real, Index>;
            using node_type = typename traits::output_node_type;
            using payload_type = typename traits::payload_type;
            std::size_t const child_count = node_count * static_cast<std::size_t>(Width);

            cuda::std::span<node_type> staging_nodes{};
            gwn_allocate_span(staging_nodes, node_count, stream);
            auto release_staging =
                gwn_make_scope_exit([&]() noexcept { gwn_free_span(staging_nodes, stream); });
            gwn_throw_status_error(gwn_cuda_to_status(cudaMemsetAsync(
                staging_nodes.data(), 0, staging_nodes.size() * sizeof(node_type), stream
            )));

            gwn_device_array<Index> parent{};
            gwn_device_array<std::uint32_t> parent_slot{};
            gwn_device_array<std::uint32_t> arity{};
            gwn_device_array<unsigned int> arrivals{};
            gwn_device_array<payload_type> pending{};
            gwn_device_array<unsigned int> error_flag{};
            parent.resize(node_count, stream);
            parent_slot.resize(node_count, stream);
            arity.resize(node_count, stream);
            arrivals.resize(node_count, stream);
            pending.resize(child_count, stream);
            error_flag.resize(1, stream);
            gwn_throw_status_error(gwn_cuda_to_status(
                cudaMemsetAsync(parent.data(), 0xff, parent.size() * sizeof(Index), stream)
            ));
            gwn_throw_status_error(gwn_cuda_to_status(cudaMemsetAsync(
                parent_slot.data(), 0xff, parent_slot.size() * sizeof(std::uint32_t), stream
            )));
            arity.zero(stream);
            arrivals.zero(stream);
            pending.zero(stream);
            error_flag.zero(stream);

            gwn_bvh_refit_state<Index> const state{
                parent.span(), parent_slot.span(), arity.span(), arrivals.span()
            };
            constexpr int k_block_size = k_gwn_default_block_size;
            // Preparation reconstructs parent links, propagation computes one additive payload per
            // child slot, and finalization recentres those payloads into the stored coefficients.
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
                    gwn_refit_bvh_moment_from_leaves_functor<Order, Width, Real, Index>{
                        bvh, state, pending.span(), error_flag.data()
                    },
                    stream
                )
            );
            gwn_throw_status_error(
                gwn_launch_linear_kernel<k_block_size>(
                    node_count,
                    gwn_finalize_bvh_moment_refit_functor<Order, Width, Real, Index>{
                        bvh,
                        state,
                        cuda::std::span<payload_type const>(pending.data(), pending.size()),
                        typename traits::output_context{staging_nodes},
                        error_flag.data(),
                    },
                    stream
                )
            );

            unsigned int host_error = 0;
            gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
                &host_error, error_flag.data(), sizeof(host_error), cudaMemcpyDeviceToHost, stream
            )));
            gwn_throw_status_error(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
            if (host_error != 0u)
                gwn_bvh_internal_error(
                    k_gwn_bvh_phase_refit_moment, "Bottom-up moment refit did not converge."
                );

            staging.nodes = staging_nodes;
            release_staging.release();
        };

        gwn_bvh_moment_accessor<Width, Order, Real, Index> staging{};
        auto release_staging = gwn_make_scope_exit([&]() noexcept {
            gwn_release_bvh_moment_accessor(staging, stream);
        });
        build_moment(staging);
        moment = staging;
        release_staging.release();
        return gwn_status::ok();
    });
}

} // namespace detail
} // namespace gwn
