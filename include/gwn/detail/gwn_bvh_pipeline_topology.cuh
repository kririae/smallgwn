#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "gwn/detail/gwn_bvh_build_lbvh.cuh"
#include "gwn/detail/gwn_bvh_pipeline_common.cuh"

namespace gwn {
namespace detail {

template <int Width, class Real, class Index>
gwn_status gwn_build_bvh_topology_lbvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_try_translate_status("gwn_build_bvh_topology_lbvh_impl", [&]() -> gwn_status {
        static_assert(Width >= 2, "BVH width must be at least 2.");

        if (!geometry.is_valid()) {
            return gwn_status::invalid_argument(
                "Geometry accessor is invalid for BVH construction."
            );
        }

        if (geometry.triangle_count() == 0) {
            gwn_release_bvh_topology_tree_accessor(topology, stream);
            return gwn_status::ok();
        }
        if (geometry.vertex_count() == 0) {
            return gwn_status::invalid_argument(
                "Cannot build BVH with triangles but zero vertices."
            );
        }

        std::size_t const primitive_count = geometry.triangle_count();
        auto const release_topology = [](gwn_bvh_topology_accessor<Width, Real, Index> &tree,
                                         cudaStream_t const stream_to_release) noexcept {
            gwn_release_bvh_topology_tree_accessor(tree, stream_to_release);
        };

        auto const build_topology =
            [&](gwn_bvh_topology_accessor<Width, Real, Index> &staging_topology) -> gwn_status {
            gwn_device_array<Index> sorted_primitive_indices{};
            gwn_device_array<gwn_binary_node<Index>> binary_nodes{};
            gwn_device_array<Index> binary_internal_parent{};
            GWN_RETURN_ON_ERROR(gwn_build_binary_lbvh_topology(
                geometry, sorted_primitive_indices, binary_nodes, binary_internal_parent, stream
            ));

            GWN_RETURN_ON_ERROR(
                gwn_allocate_span(staging_topology.primitive_indices, primitive_count, stream)
            );
            GWN_RETURN_ON_ERROR(gwn_copy_d2d(
                staging_topology.primitive_indices,
                cuda::std::span<Index const>(
                    sorted_primitive_indices.data(), sorted_primitive_indices.size()
                ),
                stream
            ));

            if (primitive_count == 1) {
                staging_topology.root_kind = gwn_bvh_child_kind::k_leaf;
                staging_topology.root_index = Index(0);
                staging_topology.root_count = Index(1);
                return gwn_status::ok();
            }

            GWN_RETURN_ON_ERROR((gwn_collapse_binary_lbvh_topology<Width, Real, Index>(
                cuda::std::span<gwn_binary_node<Index> const>(
                    binary_nodes.data(), binary_nodes.size()
                ),
                cuda::std::span<Index const>(
                    binary_internal_parent.data(), binary_internal_parent.size()
                ),
                staging_topology, stream
            )));

            staging_topology.root_kind = gwn_bvh_child_kind::k_internal;
            staging_topology.root_index = Index(0);
            staging_topology.root_count = Index(0);
            return gwn_status::ok();
        };

        return gwn_replace_accessor_with_staging(
            topology, release_topology, build_topology, stream
        );
    });
}

} // namespace detail
} // namespace gwn
