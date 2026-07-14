#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../gwn_bvh_build.cuh"
#include "gwn_bvh_build_collapse.cuh"
#include "gwn_bvh_build_hploc.cuh"
#include "gwn_bvh_build_lbvh.cuh"
#include "gwn_bvh_reorder.cuh"
#include "gwn_bvh_revision.cuh"
#include "gwn_bvh_status_helpers.cuh"
#include "gwn_bvh_triangle_impl.cuh"
#include "gwn_device_array.cuh"
#include "gwn_kernel_utils.cuh"

namespace gwn {
namespace detail {

/// \brief Materialize leaf-ordered \c v0/e1/e2 records from canonical primitive order.
template <gwn_real_type Real, gwn_index_type Index> struct gwn_build_bvh_triangles_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    cuda::std::span<Index const> primitive_indices{};
    cuda::std::span<gwn_bvh_triangle<Real>> triangles{};
    unsigned int *error_flag = nullptr;

    void __device__ operator()(std::size_t const sorted_index) const noexcept {
        // The source index names the original mesh triangle while the destination remains in BVH
        // primitive order. Traversal can therefore read one contiguous triangle record and still
        // report the original primitive ID from the parallel index span.
        Index const primitive_index = primitive_indices[sorted_index];
        auto const payload = gwn_compute_bvh_triangle(geometry, primitive_index);
        if (!payload.is_valid) {
            atomicExch(error_flag, 1u);
            return;
        }
        triangles[sorted_index] = payload.triangle;
    }
};

/// \brief Build a canonical BVH with native child-AoS collapse and strong replacement semantics.
template <int Width, gwn_real_type Real, gwn_index_type Index, class MortonCode>
void gwn_build_bvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry, gwn_bvh_accessor<Width, Real, Index> &bvh,
    gwn_bvh_build_options const options, cudaStream_t const stream
) {
    static_assert(Width >= 2, "BVH width must be at least 2.");
    static_assert(
        std::is_same_v<MortonCode, std::uint32_t> || std::is_same_v<MortonCode, std::uint64_t>,
        "MortonCode must be std::uint32_t or std::uint64_t."
    );

    [&]() {
        if (!geometry.is_valid()) {
            return gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_build, "Geometry accessor is invalid for BVH build."
            );
        }

        if (options.method != gwn_bvh_build_method::k_hploc &&
            options.method != gwn_bvh_build_method::k_lbvh) {
            return gwn_bvh_invalid_argument(k_gwn_bvh_phase_build, "BVH build method is invalid.");
        }
        if (options.method == gwn_bvh_build_method::k_hploc &&
            (options.hploc_search_radius == 0 ||
             options.hploc_search_radius > k_gwn_hploc_max_search_radius)) {
            return gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_build, "H-PLOC search radius must be in the range [1, 8]."
            );
        }

        std::size_t const primitive_count = geometry.triangle_count();
        if (primitive_count > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
            return gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_build, "Triangle count exceeds the selected BVH index range."
            );
        }
        if (primitive_count != 0 && geometry.vertex_count() == 0) {
            return gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_build, "Cannot build a BVH with triangles but no vertices."
            );
        }
        if (primitive_count > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
            return gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_build,
                "Triangle count exceeds the native wide-collapse counter range."
            );
        }
        if (primitive_count != 0 && !gwn_bvh_child<Real>::can_encode_offset(primitive_count - 1)) {
            return gwn_bvh_invalid_argument(
                k_gwn_bvh_phase_build, "Triangle count exceeds the packed BVH child offset range."
            );
        }

        auto const release = [](gwn_bvh_accessor<Width, Real, Index> &target,
                                cudaStream_t const release_stream) noexcept {
            gwn_release_bvh_accessor(target, release_stream);
        };
        auto const build = [&](gwn_bvh_accessor<Width, Real, Index> &staging) {
            if (primitive_count == 0)
                return;

            // Morton preprocessing and the binary algorithms are representation-independent. The
            // first persistent wide representation they feed is the canonical child-AoS tree.
            gwn_bvh_build_preprocess_data<Real, Index, MortonCode> preprocess{};
            gwn_bvh_build_preprocess<MortonCode>(geometry, preprocess, stream);

            gwn_device_array<gwn_binary_node<Index>> binary_nodes{};
            gwn_device_array<Index> binary_parent{};
            gwn_device_array<gwn_aabb<Real>> binary_bounds{};
            Index binary_root = gwn_invalid_index<Index>();
            switch (options.method) {
            case gwn_bvh_build_method::k_hploc:
                gwn_bvh_build_binary_hploc_impl<Real, Index, MortonCode>(
                    cuda::std::span<Index const>(
                        preprocess.sorted_primitive_indices.data(), primitive_count
                    ),
                    cuda::std::span<MortonCode const>(
                        preprocess.sorted_morton_codes.data(), primitive_count
                    ),
                    cuda::std::span<gwn_aabb<Real> const>(
                        preprocess.sorted_primitive_aabbs.data(), primitive_count
                    ),
                    binary_nodes, binary_parent, binary_bounds, binary_root,
                    options.hploc_search_radius, stream
                );
                break;
            case gwn_bvh_build_method::k_lbvh:
                gwn_bvh_build_binary_lbvh<Real, Index, MortonCode>(
                    cuda::std::span<MortonCode const>(
                        preprocess.sorted_morton_codes.data(), primitive_count
                    ),
                    cuda::std::span<gwn_aabb<Real> const>(
                        preprocess.sorted_primitive_aabbs.data(), primitive_count
                    ),
                    binary_nodes, binary_parent, binary_bounds, stream
                );
                if (primitive_count > 1)
                    binary_root = Index(0);
                break;
            default:
                GWN_ASSERT(false, "Validated BVH build method reached default case.");
                return gwn_bvh_internal_error(
                    k_gwn_bvh_phase_build,
                    "Validated BVH build method reached an unreachable branch."
                );
            }

            gwn_allocate_span(staging.primitive_indices, primitive_count, stream);
            gwn_allocate_span(staging.triangles, primitive_count, stream);
            gwn_copy_d2d(
                staging.primitive_indices,
                cuda::std::span<Index const>(
                    preprocess.sorted_primitive_indices.data(), primitive_count
                ),
                stream
            );

            // Root bounds live beside the packed root reference because leaf-root queries have no
            // node allocation from which to recover them.
            gwn_aabb<Real> root_bounds{};
            if (primitive_count == 1) {
                gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
                    &root_bounds, preprocess.sorted_primitive_aabbs.data(), sizeof(root_bounds),
                    cudaMemcpyDeviceToHost, stream
                )));
                staging.root.reference =
                    gwn_bvh_child<Real>::k_valid_mask |
                    (std::uint64_t(1) << gwn_bvh_child<Real>::k_primitive_count_shift);
            } else {
                if (!gwn_index_in_bounds(binary_root, binary_bounds.size())) {
                    return gwn_bvh_internal_error(
                        k_gwn_bvh_phase_build, "Binary BVH builder produced an invalid root."
                    );
                }
                gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
                    &root_bounds, binary_bounds.data() + static_cast<std::size_t>(binary_root),
                    sizeof(root_bounds), cudaMemcpyDeviceToHost, stream
                )));
                gwn_allocate_span(staging.nodes, binary_nodes.size(), stream);
                gwn_device_array<std::uint64_t> reorder_key{};
                reorder_key.resize(binary_nodes.size(), stream);
                std::size_t wide_node_count = 0;
                gwn_bvh_collapse_binary_wide_impl<Width, Real, Index>(
                    cuda::std::span<gwn_binary_node<Index> const>(
                        binary_nodes.data(), binary_nodes.size()
                    ),
                    cuda::std::span<gwn_aabb<Real> const>(
                        binary_bounds.data(), binary_bounds.size()
                    ),
                    cuda::std::span<gwn_aabb<Real> const>(
                        preprocess.sorted_primitive_aabbs.data(), primitive_count
                    ),
                    staging.nodes, reorder_key.span(), wide_node_count,
                    staging.internal_stack_bound, staging.packed_stack_bound, binary_root, stream
                );
                staging.nodes = staging.nodes.first(wide_node_count);
                auto const initialized_reorder_key =
                    cuda::std::span<std::uint64_t const>(reorder_key.data(), wide_node_count);

                gwn_bvh_reorder_impl<Width, Real, Index>(
                    staging.nodes, initialized_reorder_key, stream
                );
                staging.root.reference = gwn_bvh_child<Real>::k_valid_mask;
            }

            gwn_device_array<unsigned int> error_flag{};
            error_flag.resize(1, stream);
            error_flag.zero(stream);
            constexpr int k_block_size = k_gwn_default_block_size;
            gwn_throw_status_error(
                gwn_launch_linear_kernel<k_block_size>(
                    primitive_count,
                    gwn_build_bvh_triangles_functor<Real, Index>{
                        geometry,
                        cuda::std::span<Index const>(
                            staging.primitive_indices.data(), staging.primitive_indices.size()
                        ),
                        staging.triangles,
                        error_flag.data(),
                    },
                    stream
                )
            );

            // The triangle kernel is the last writer of the staging accessor. Synchronizing here
            // prevents a failed geometry lookup from publishing a partially initialized BVH.
            unsigned int host_error = 0;
            gwn_throw_status_error(gwn_cuda_to_status(cudaMemcpyAsync(
                &host_error, error_flag.data(), sizeof(host_error), cudaMemcpyDeviceToHost, stream
            )));
            gwn_throw_status_error(gwn_cuda_to_status(cudaStreamSynchronize(stream)));
            if (host_error != 0u) {
                return gwn_bvh_internal_error(
                    k_gwn_bvh_phase_build,
                    "Leaf-ordered triangle construction encountered invalid geometry."
                );
            }

            staging.root.bounds = root_bounds;
            // Every successful build or refit receives a process-unique identity. A stale moment
            // therefore remains stale even if cudaMallocAsync later reuses every BVH allocation.
            staging.revision = gwn_reserve_bvh_revision();
            if (staging.revision == 0) {
                return gwn_bvh_internal_error(
                    k_gwn_bvh_phase_build, "BVH geometry-derived state revision is exhausted."
                );
            }
        };

        gwn_replace_accessor_with_staging(bvh, release, build, stream);
    }();
}

} // namespace detail

template <int Width, gwn_real_type Real, gwn_index_type Index, gwn_index_type MortonCode>
[[nodiscard]] gwn_status gwn_build_bvh(
    gwn_geometry_object<Real, Index> const &geometry, gwn_bvh_object<Width, Real, Index> &bvh,
    gwn_bvh_build_options const options, cudaStream_t const stream
) noexcept {
    return detail::gwn_try_translate_status("gwn_build_bvh", [&]() {
        gwn_bvh_object<Width, Real, Index> staging;
        detail::gwn_build_bvh_impl<Width, Real, Index, MortonCode>(
            geometry.accessor(), staging.accessor(), options, stream
        );
        staging.set_stream(stream);
        swap(bvh, staging);
        // staging now owns the replaced BVH and its original stream binding. Its destructor
        // releases that storage after all work previously enqueued on the old stream.
    });
}

} // namespace gwn
