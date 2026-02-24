#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "gwn_bvh_status_helpers.cuh"
#include "gwn_bvh_topology_build_hploc.cuh"
#include "gwn_bvh_topology_build_lbvh.cuh"

namespace gwn {
namespace detail {

template <int Width, class Real, class Index, class BuildBinaryFn>
gwn_status gwn_bvh_topology_build_from_binary_impl(
    char const *entry_name, gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology, BuildBinaryFn &&build_binary_fn,
    cudaStream_t const stream
) noexcept {
    return gwn_try_translate_status(entry_name, [&]() -> gwn_status {
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
            Index root_internal_index = gwn_invalid_index<Index>();
            GWN_RETURN_ON_ERROR(build_binary_fn(
                sorted_primitive_indices, binary_nodes, binary_internal_parent, root_internal_index
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

            GWN_RETURN_ON_ERROR((gwn_bvh_topology_build_collapse_binary_lbvh<Width, Real, Index>(
                cuda::std::span<gwn_binary_node<Index> const>(
                    binary_nodes.data(), binary_nodes.size()
                ),
                cuda::std::span<Index const>(
                    binary_internal_parent.data(), binary_internal_parent.size()
                ),
                staging_topology, root_internal_index, stream
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

template <int Width, class Real, class Index, class MortonCode = std::uint64_t>
gwn_status gwn_bvh_topology_build_lbvh_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_bvh_topology_build_from_binary_impl<Width, Real, Index>(
        "gwn_bvh_topology_build_lbvh_impl", geometry, topology,
        [&](gwn_device_array<Index> &sorted_primitive_indices,
            gwn_device_array<gwn_binary_node<Index>> &binary_nodes,
            gwn_device_array<Index> &binary_internal_parent,
            Index &root_internal_index) -> gwn_status {
        GWN_RETURN_ON_ERROR((gwn_bvh_topology_build_binary_lbvh<Real, Index, MortonCode>(
            geometry, sorted_primitive_indices, binary_nodes, binary_internal_parent, stream
        )));
        if (geometry.triangle_count() > 1)
            root_internal_index = Index(0);
        return gwn_status::ok();
    },
        stream
    );
}

template <int Width, class Real, class Index, class MortonCode = std::uint64_t>
gwn_status gwn_bvh_topology_build_hploc_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology,
    cudaStream_t const stream = gwn_default_stream()
) noexcept {
    return gwn_bvh_topology_build_from_binary_impl<Width, Real, Index>(
        "gwn_bvh_topology_build_hploc_impl", geometry, topology,
        [&](gwn_device_array<Index> &sorted_primitive_indices,
            gwn_device_array<gwn_binary_node<Index>> &binary_nodes,
            gwn_device_array<Index> &binary_internal_parent,
            Index &root_internal_index) -> gwn_status {
        return gwn_bvh_topology_build_binary_hploc<Real, Index, MortonCode>(
            geometry, sorted_primitive_indices, binary_nodes, binary_internal_parent,
            root_internal_index, stream
        );
    },
        stream
    );
}

} // namespace detail
} // namespace gwn
