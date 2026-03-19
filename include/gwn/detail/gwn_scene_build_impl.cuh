#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <utility>
#include <vector>

#include "gwn_bvh_status_helpers.cuh"
#include "gwn_bvh_topology_build_common.cuh"
#include "gwn_bvh_topology_build_impl.cuh"

namespace gwn {
namespace detail {

inline constexpr std::string_view k_gwn_scene_phase_build = "scene.build";
inline constexpr std::string_view k_gwn_scene_phase_preprocess = "scene.build.preprocess";
inline constexpr std::string_view k_gwn_scene_phase_refit_aabb = "scene.build.refit.aabb";

template <class T>
gwn_status gwn_scene_copy_device_span_to_host(
    cuda::std::span<T const> const src, std::vector<T> &dst, cudaStream_t const stream
) noexcept {
    dst.resize(src.size());
    GWN_RETURN_ON_ERROR(gwn_copy_d2h(cuda::std::span<T>(dst.data(), dst.size()), src, stream));
    return gwn_cuda_to_status(cudaStreamSynchronize(stream));
}

template <class T>
gwn_status gwn_scene_copy_device_value_to_host(
    T const *const src, T &dst, cudaStream_t const stream
) noexcept {
    GWN_RETURN_ON_ERROR(gwn_copy_d2h(
        cuda::std::span<T>(&dst, std::size_t(1)), cuda::std::span<T const>(src, std::size_t(1)),
        stream
    ));
    return gwn_cuda_to_status(cudaStreamSynchronize(stream));
}

template <gwn_real_type Real> [[nodiscard]] inline gwn_aabb<Real> gwn_scene_zero_aabb() noexcept {
    return gwn_aabb<Real>{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)};
}

template <gwn_real_type Real>
[[nodiscard]] inline gwn_aabb<Real> gwn_scene_union_assign(
    gwn_aabb<Real> const &lhs, gwn_aabb<Real> const &rhs, bool &has_value
) noexcept {
    if (!has_value) {
        has_value = true;
        return rhs;
    }
    return gwn_aabb_union(lhs, rhs);
}

template <int Width, gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_compute_blas_leaf_root_aabb(
    BlasT const &blas, gwn_aabb<Real> &result, cudaStream_t const stream
) noexcept {
    auto const &topology = blas.topology;
    auto const &geometry = blas.geometry;
    if (!topology.has_leaf_root())
        return gwn_bvh_internal_error(
            k_gwn_scene_phase_preprocess, "BLAS leaf-root AABB requested for non-leaf topology."
        );

    std::size_t const begin = static_cast<std::size_t>(topology.root_index);
    std::size_t const count = static_cast<std::size_t>(topology.root_count);
    if (begin > topology.primitive_indices.size() ||
        count > (topology.primitive_indices.size() - begin)) {
        return gwn_bvh_internal_error(
            k_gwn_scene_phase_preprocess, "BLAS leaf-root primitive range is out of bounds."
        );
    }

    std::vector<Index> primitive_ids{};
    GWN_RETURN_ON_ERROR(gwn_scene_copy_device_span_to_host(
        cuda::std::span<Index const>(topology.primitive_indices.data() + begin, count),
        primitive_ids, stream
    ));

    bool has_bounds = false;
    gwn_aabb<Real> bounds = gwn_scene_zero_aabb<Real>();
    for (Index const primitive_id : primitive_ids) {
        if (!gwn_index_in_bounds(primitive_id, geometry.triangle_count())) {
            return gwn_bvh_internal_error(
                k_gwn_scene_phase_preprocess, "BLAS leaf-root primitive index is out of bounds."
            );
        }

        Index ia{};
        Index ib{};
        Index ic{};
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.tri_i0.data() + primitive_id, ia, stream)
        );
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.tri_i1.data() + primitive_id, ib, stream)
        );
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.tri_i2.data() + primitive_id, ic, stream)
        );
        if (!gwn_index_in_bounds(ia, geometry.vertex_count()) ||
            !gwn_index_in_bounds(ib, geometry.vertex_count()) ||
            !gwn_index_in_bounds(ic, geometry.vertex_count())) {
            return gwn_bvh_internal_error(
                k_gwn_scene_phase_preprocess, "BLAS triangle vertex index is out of bounds."
            );
        }

        Real ax{};
        Real ay{};
        Real az{};
        Real bx{};
        Real by{};
        Real bz{};
        Real cx{};
        Real cy{};
        Real cz{};
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.vertex_x.data() + ia, ax, stream)
        );
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.vertex_y.data() + ia, ay, stream)
        );
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.vertex_z.data() + ia, az, stream)
        );
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.vertex_x.data() + ib, bx, stream)
        );
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.vertex_y.data() + ib, by, stream)
        );
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.vertex_z.data() + ib, bz, stream)
        );
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.vertex_x.data() + ic, cx, stream)
        );
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.vertex_y.data() + ic, cy, stream)
        );
        GWN_RETURN_ON_ERROR(
            gwn_scene_copy_device_value_to_host(geometry.vertex_z.data() + ic, cz, stream)
        );

        gwn_aabb<Real> const triangle_bounds{
            std::min(ax, std::min(bx, cx)), std::min(ay, std::min(by, cy)),
            std::min(az, std::min(bz, cz)), std::max(ax, std::max(bx, cx)),
            std::max(ay, std::max(by, cy)), std::max(az, std::max(bz, cz)),
        };
        bounds = gwn_scene_union_assign(bounds, triangle_bounds, has_bounds);
    }

    if (!has_bounds) {
        return gwn_bvh_internal_error(
            k_gwn_scene_phase_preprocess, "BLAS leaf-root primitive range is empty."
        );
    }
    result = bounds;
    return gwn_status::ok();
}

template <int Width, gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_compute_blas_root_aabb(
    BlasT const &blas, gwn_aabb<Real> &result, cudaStream_t const stream
) noexcept {
    auto const &topology = blas.topology;
    if (topology.has_leaf_root())
        return gwn_scene_compute_blas_leaf_root_aabb<Width, Real, Index>(blas, result, stream);

    if (!gwn_index_in_bounds(topology.root_index, topology.nodes.size())) {
        return gwn_bvh_internal_error(
            k_gwn_scene_phase_preprocess, "BLAS internal root index is out of bounds."
        );
    }

    gwn_bvh_topology_node_soa<Width, Index> root_node{};
    gwn_bvh_aabb_node_soa<Width, Real> root_aabb{};
    GWN_RETURN_ON_ERROR(gwn_scene_copy_device_value_to_host(
        blas.topology.nodes.data() + topology.root_index, root_node, stream
    ));
    GWN_RETURN_ON_ERROR(gwn_scene_copy_device_value_to_host(
        blas.aabb.nodes.data() + topology.root_index, root_aabb, stream
    ));

    bool has_bounds = false;
    gwn_aabb<Real> bounds = gwn_scene_zero_aabb<Real>();
    for (int slot = 0; slot < Width; ++slot) {
        auto const child_kind = static_cast<gwn_bvh_child_kind>(root_node.child_kind[slot]);
        if (child_kind == gwn_bvh_child_kind::k_invalid)
            continue;
        gwn_aabb<Real> const child_bounds{
            root_aabb.child_min_x[slot], root_aabb.child_min_y[slot], root_aabb.child_min_z[slot],
            root_aabb.child_max_x[slot], root_aabb.child_max_y[slot], root_aabb.child_max_z[slot],
        };
        bounds = gwn_scene_union_assign(bounds, child_bounds, has_bounds);
    }

    if (!has_bounds) {
        return gwn_bvh_internal_error(
            k_gwn_scene_phase_preprocess, "BLAS root node has no valid children."
        );
    }
    result = bounds;
    return gwn_status::ok();
}

template <class MortonCode, gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_build_preprocess(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_topology_build_preprocess<Real, Index, MortonCode> &preprocess,
    std::vector<gwn_aabb<Real>> &instance_aabbs, cudaStream_t const stream
) noexcept {
    struct morton_item {
        MortonCode code{};
        Index primitive_index{};
        gwn_aabb<Real> bounds{};
    };

    instance_aabbs.resize(instances.size());
    std::vector<morton_item> sorted_items{};
    sorted_items.reserve(instances.size());

    bool has_scene_bounds = false;
    gwn_aabb<Real> scene_bounds = gwn_scene_zero_aabb<Real>();
    for (std::size_t i = 0; i < instances.size(); ++i) {
        auto const &instance = instances[i];
        gwn_aabb<Real> local_bounds{};
        GWN_RETURN_ON_ERROR((gwn_scene_compute_blas_root_aabb<BlasT::k_width, Real, Index>(
            blas_table[static_cast<std::size_t>(instance.blas_index)], local_bounds, stream
        )));
        gwn_aabb<Real> const world_bounds = instance.transform.transform_aabb(local_bounds);
        instance_aabbs[i] = world_bounds;
        scene_bounds = gwn_scene_union_assign(scene_bounds, world_bounds, has_scene_bounds);
    }

    if (!has_scene_bounds) {
        return gwn_bvh_internal_error(
            k_gwn_scene_phase_preprocess, "Scene preprocess produced no instance bounds."
        );
    }

    auto const safe_inv = [](Real const lo, Real const hi) noexcept {
        return (hi > lo) ? Real(1) / (hi - lo) : Real(1);
    };
    Real const inv_x = safe_inv(scene_bounds.min_x, scene_bounds.max_x);
    Real const inv_y = safe_inv(scene_bounds.min_y, scene_bounds.max_y);
    Real const inv_z = safe_inv(scene_bounds.min_z, scene_bounds.max_z);

    for (std::size_t i = 0; i < instance_aabbs.size(); ++i) {
        gwn_aabb<Real> const &bounds = instance_aabbs[i];
        Real const center_x = (bounds.min_x + bounds.max_x) * Real(0.5);
        Real const center_y = (bounds.min_y + bounds.max_y) * Real(0.5);
        Real const center_z = (bounds.min_z + bounds.max_z) * Real(0.5);
        sorted_items.push_back(
            morton_item{
                gwn_encode_morton<MortonCode>(
                    (center_x - scene_bounds.min_x) * inv_x,
                    (center_y - scene_bounds.min_y) * inv_y, (center_z - scene_bounds.min_z) * inv_z
                ),
                static_cast<Index>(i),
                bounds,
            }
        );
    }

    std::sort(
        sorted_items.begin(), sorted_items.end(),
        [](morton_item const &lhs, morton_item const &rhs) {
        if (lhs.code != rhs.code)
            return lhs.code < rhs.code;
        return lhs.primitive_index < rhs.primitive_index;
    }
    );

    std::vector<Index> sorted_primitive_indices{};
    std::vector<MortonCode> sorted_morton_codes{};
    std::vector<gwn_aabb<Real>> sorted_primitive_aabbs{};
    sorted_primitive_indices.reserve(sorted_items.size());
    sorted_morton_codes.reserve(sorted_items.size());
    sorted_primitive_aabbs.reserve(sorted_items.size());
    for (morton_item const &item : sorted_items) {
        sorted_primitive_indices.push_back(item.primitive_index);
        sorted_morton_codes.push_back(item.code);
        sorted_primitive_aabbs.push_back(item.bounds);
    }

    GWN_RETURN_ON_ERROR(preprocess.sorted_primitive_indices.copy_from_host(
        cuda::std::span<Index const>(
            sorted_primitive_indices.data(), sorted_primitive_indices.size()
        ),
        stream
    ));
    GWN_RETURN_ON_ERROR(preprocess.sorted_morton_codes.copy_from_host(
        cuda::std::span<MortonCode const>(sorted_morton_codes.data(), sorted_morton_codes.size()),
        stream
    ));
    GWN_RETURN_ON_ERROR(preprocess.primitive_aabbs.copy_from_host(
        cuda::std::span<gwn_aabb<Real> const>(instance_aabbs.data(), instance_aabbs.size()), stream
    ));
    return preprocess.sorted_primitive_aabbs.copy_from_host(
        cuda::std::span<gwn_aabb<Real> const>(
            sorted_primitive_aabbs.data(), sorted_primitive_aabbs.size()
        ),
        stream
    );
}

template <int Width, gwn_real_type Real, gwn_index_type Index>
gwn_status gwn_scene_build_ias_aabb(
    gwn_bvh_topology_accessor<Width, Real, Index> const &topology,
    cuda::std::span<gwn_aabb<Real> const> const instance_aabbs,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree, cudaStream_t const stream
) noexcept {
    if (!topology.has_internal_root())
        return gwn_status::ok();

    std::vector<gwn_bvh_topology_node_soa<Width, Index>> host_nodes{};
    std::vector<Index> host_primitive_indices{};
    GWN_RETURN_ON_ERROR(gwn_scene_copy_device_span_to_host(
        cuda::std::span<gwn_bvh_topology_node_soa<Width, Index> const>(
            topology.nodes.data(), topology.nodes.size()
        ),
        host_nodes, stream
    ));
    GWN_RETURN_ON_ERROR(gwn_scene_copy_device_span_to_host(
        cuda::std::span<Index const>(
            topology.primitive_indices.data(), topology.primitive_indices.size()
        ),
        host_primitive_indices, stream
    ));

    std::vector<gwn_bvh_aabb_node_soa<Width, Real>> host_aabb_nodes(host_nodes.size());
    std::vector<gwn_aabb<Real>> subtree_bounds(host_nodes.size(), gwn_scene_zero_aabb<Real>());

    for (std::size_t node_index = host_nodes.size(); node_index-- > 0;) {
        auto const &node = host_nodes[node_index];
        bool has_child = false;
        gwn_aabb<Real> node_bounds = gwn_scene_zero_aabb<Real>();

        for (int slot = 0; slot < Width; ++slot) {
            auto const child_kind = static_cast<gwn_bvh_child_kind>(node.child_kind[slot]);
            if (child_kind == gwn_bvh_child_kind::k_invalid)
                continue;

            gwn_aabb<Real> child_bounds{};
            if (child_kind == gwn_bvh_child_kind::k_internal) {
                if (!gwn_index_in_bounds(node.child_index[slot], subtree_bounds.size())) {
                    return gwn_bvh_internal_error(
                        k_gwn_scene_phase_refit_aabb, "IAS internal child index is out of bounds."
                    );
                }
                child_bounds = subtree_bounds[static_cast<std::size_t>(node.child_index[slot])];
            } else if (child_kind == gwn_bvh_child_kind::k_leaf) {
                std::size_t const begin = static_cast<std::size_t>(node.child_index[slot]);
                std::size_t const count = static_cast<std::size_t>(node.child_count[slot]);
                if (begin > host_primitive_indices.size() ||
                    count > (host_primitive_indices.size() - begin)) {
                    return gwn_bvh_internal_error(
                        k_gwn_scene_phase_refit_aabb, "IAS leaf primitive range is out of bounds."
                    );
                }
                bool has_leaf = false;
                child_bounds = gwn_scene_zero_aabb<Real>();
                for (std::size_t i = 0; i < count; ++i) {
                    Index const primitive_index = host_primitive_indices[begin + i];
                    if (!gwn_index_in_bounds(primitive_index, instance_aabbs.size())) {
                        return gwn_bvh_internal_error(
                            k_gwn_scene_phase_refit_aabb,
                            "IAS leaf primitive index is out of bounds."
                        );
                    }
                    child_bounds = gwn_scene_union_assign(
                        child_bounds, instance_aabbs[static_cast<std::size_t>(primitive_index)],
                        has_leaf
                    );
                }
                if (!has_leaf) {
                    return gwn_bvh_internal_error(
                        k_gwn_scene_phase_refit_aabb, "IAS leaf child has no primitives."
                    );
                }
            } else {
                return gwn_bvh_internal_error(
                    k_gwn_scene_phase_refit_aabb, "IAS child slot has an unknown kind tag."
                );
            }

            host_aabb_nodes[node_index].child_min_x[slot] = child_bounds.min_x;
            host_aabb_nodes[node_index].child_min_y[slot] = child_bounds.min_y;
            host_aabb_nodes[node_index].child_min_z[slot] = child_bounds.min_z;
            host_aabb_nodes[node_index].child_max_x[slot] = child_bounds.max_x;
            host_aabb_nodes[node_index].child_max_y[slot] = child_bounds.max_y;
            host_aabb_nodes[node_index].child_max_z[slot] = child_bounds.max_z;
            node_bounds = gwn_scene_union_assign(node_bounds, child_bounds, has_child);
        }

        if (!has_child) {
            return gwn_bvh_internal_error(
                k_gwn_scene_phase_refit_aabb, "IAS node has no valid children."
            );
        }
        subtree_bounds[node_index] = node_bounds;
    }

    GWN_RETURN_ON_ERROR(gwn_copy_h2d(
        aabb_tree.nodes,
        cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real> const>(
            host_aabb_nodes.data(), host_aabb_nodes.size()
        ),
        stream
    ));
    return gwn_status::ok();
}

template <
    int Width, gwn_real_type Real, gwn_index_type Index, class MortonCode, class BlasT,
    class BuildBinaryFn>
gwn_status gwn_scene_build_impl(
    char const *entry_name, cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_bvh_topology_accessor<Width, Real, Index> &topology,
    gwn_bvh_aabb_accessor<Width, Real, Index> &aabb_tree, BuildBinaryFn &&build_binary_fn,
    cudaStream_t const stream
) noexcept {
    return gwn_try_translate_status(entry_name, [&]() -> gwn_status {
        gwn_topology_build_preprocess<Real, Index, MortonCode> preprocess{};
        std::vector<gwn_aabb<Real>> instance_aabbs{};
        GWN_RETURN_ON_ERROR(
            gwn_scene_build_preprocess<MortonCode>(
                blas_table, instances, preprocess, instance_aabbs, stream
            )
        );
        GWN_RETURN_ON_ERROR(
            (gwn_bvh_topology_build_from_preprocess_impl<Width, Real, Index, MortonCode>(
                entry_name, preprocess, instances.size(), topology, aabb_tree,
                std::forward<BuildBinaryFn>(build_binary_fn), stream
            ))
        );
        return gwn_scene_build_ias_aabb<Width, Real, Index>(
            topology,
            cuda::std::span<gwn_aabb<Real> const>(instance_aabbs.data(), instance_aabbs.size()),
            aabb_tree, stream
        );
    });
}

template <gwn_real_type Real, gwn_index_type Index, class BlasT>
gwn_status gwn_scene_validate_build_inputs(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances
) noexcept {
    if (!gwn_span_has_storage(blas_table))
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_build, "BLAS table span has null storage."
        );
    if (!gwn_span_has_storage(instances))
        return gwn_bvh_invalid_argument(k_gwn_scene_phase_build, "Instance span has null storage.");
    if (blas_table.empty())
        return gwn_bvh_invalid_argument(k_gwn_scene_phase_build, "BLAS table must not be empty.");
    if (instances.empty())
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_build, "Instance list must not be empty."
        );
    if (instances.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        return gwn_bvh_invalid_argument(
            k_gwn_scene_phase_build, "Instance count exceeds the representable index range."
        );
    }

    for (std::size_t i = 0; i < blas_table.size(); ++i) {
        if (!blas_table[i].is_valid()) {
            return gwn_bvh_invalid_argument(
                k_gwn_scene_phase_build, std::format("BLAS table entry {} is invalid.", i)
            );
        }
    }
    for (std::size_t i = 0; i < instances.size(); ++i) {
        if (!gwn_index_in_bounds(instances[i].blas_index, blas_table.size())) {
            return gwn_bvh_invalid_argument(
                k_gwn_scene_phase_build,
                std::format("Instance {} has an out-of-range BLAS index.", i)
            );
        }
    }
    return gwn_status::ok();
}

} // namespace detail

template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_build_lbvh(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_scene_object<Width, Real, Index, BlasT> &scene, cudaStream_t const stream
) noexcept {
    GWN_RETURN_ON_ERROR(detail::gwn_scene_validate_build_inputs(blas_table, instances));

    gwn_scene_object<Width, Real, Index, BlasT> staging{};
    staging.set_stream(stream);
    GWN_RETURN_ON_ERROR(staging.blas_table_.copy_from_host(blas_table, stream));
    GWN_RETURN_ON_ERROR(staging.instances_.copy_from_host(instances, stream));
    GWN_RETURN_ON_ERROR((detail::gwn_scene_build_impl<Width, Real, Index, std::uint64_t>(
        "gwn_scene_build_lbvh", blas_table, instances, staging.ias_topology_.accessor(),
        staging.ias_aabb_.accessor(),
        [&](detail::gwn_topology_build_preprocess<Real, Index, std::uint64_t> const &preprocess,
            gwn_device_array<detail::gwn_binary_node<Index>> &binary_nodes,
            gwn_device_array<Index> &binary_internal_parent,
            gwn_device_array<gwn_aabb<Real>> &binary_internal_bounds,
            Index &root_internal_index) -> gwn_status {
        GWN_RETURN_ON_ERROR((detail::gwn_bvh_topology_build_binary_lbvh<Real, Index, std::uint64_t>(
            cuda::std::span<std::uint64_t const>(
                preprocess.sorted_morton_codes.data(), preprocess.sorted_morton_codes.size()
            ),
            cuda::std::span<gwn_aabb<Real> const>(
                preprocess.sorted_primitive_aabbs.data(), preprocess.sorted_primitive_aabbs.size()
            ),
            binary_nodes, binary_internal_parent, binary_internal_bounds, stream
        )));
        if (preprocess.sorted_morton_codes.size() > 1)
            root_internal_index = Index(0);
        return gwn_status::ok();
    },
        stream
    )));
    swap(scene, staging);
    return gwn_status::ok();
}

template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_build_hploc(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_scene_object<Width, Real, Index, BlasT> &scene, cudaStream_t const stream
) noexcept {
    GWN_RETURN_ON_ERROR(detail::gwn_scene_validate_build_inputs(blas_table, instances));

    gwn_scene_object<Width, Real, Index, BlasT> staging{};
    staging.set_stream(stream);
    GWN_RETURN_ON_ERROR(staging.blas_table_.copy_from_host(blas_table, stream));
    GWN_RETURN_ON_ERROR(staging.instances_.copy_from_host(instances, stream));
    GWN_RETURN_ON_ERROR((detail::gwn_scene_build_impl<Width, Real, Index, std::uint64_t>(
        "gwn_scene_build_hploc", blas_table, instances, staging.ias_topology_.accessor(),
        staging.ias_aabb_.accessor(),
        [&](detail::gwn_topology_build_preprocess<Real, Index, std::uint64_t> const &preprocess,
            gwn_device_array<detail::gwn_binary_node<Index>> &binary_nodes,
            gwn_device_array<Index> &binary_internal_parent,
            gwn_device_array<gwn_aabb<Real>> &binary_internal_bounds,
            Index &root_internal_index) -> gwn_status {
        return detail::gwn_bvh_topology_build_binary_hploc<Real, Index, std::uint64_t>(
            cuda::std::span<Index const>(
                preprocess.sorted_primitive_indices.data(),
                preprocess.sorted_primitive_indices.size()
            ),
            cuda::std::span<std::uint64_t const>(
                preprocess.sorted_morton_codes.data(), preprocess.sorted_morton_codes.size()
            ),
            cuda::std::span<gwn_aabb<Real> const>(
                preprocess.sorted_primitive_aabbs.data(), preprocess.sorted_primitive_aabbs.size()
            ),
            binary_nodes, binary_internal_parent, binary_internal_bounds, root_internal_index,
            stream
        );
    },
        stream
    )));
    swap(scene, staging);
    return gwn_status::ok();
}

} // namespace gwn
