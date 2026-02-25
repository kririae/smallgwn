#pragma once

/// \file gwn_bvh.cuh
/// \brief Core BVH data structures, accessors, and owning host-side objects.
///
/// This header defines the full BVH type hierarchy used by \c smallgwn:
/// - SoA node structs for topology, AABB bounds, and Taylor moments.
/// - Non-owning accessor types (\c gwn_bvh_topology_tree_accessor etc.) that
///   hold \c cuda::std::span views into device memory — safe to copy to device.
/// - RAII owning-object wrappers (\c gwn_bvh_topology_tree_object etc.) that
///   manage device allocations, implement the \c gwn_stream_mixin binding
///   protocol, and are non-copyable.
/// - Width-4 convenience aliases (\c gwn_bvh4_topology_object,
///   \c gwn_bvh4_aabb_object, \c gwn_bvh4_moment_object) for the most common
///   use-case.

#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "gwn_utils.cuh"

namespace gwn {

/// \brief Discriminant tag stored per child slot in every BVH node.
///
/// \c k_invalid is the zero-initialised sentinel used before a tree is built.
/// \c k_internal means the child slot points to another internal node (its
/// index is an offset into the \c nodes span). \c k_leaf means the child slot
/// covers a contiguous run of primitives in the \c primitive_indices array.
enum class gwn_bvh_child_kind : std::uint8_t {
    k_invalid = 0,  ///< Unset / uninitialised slot.
    k_internal = 1, ///< Child is an internal node.
    k_leaf = 2,     ///< Child is a primitive leaf.
};

/// \brief Axis-aligned bounding box with SoA-friendly scalar components.
///
/// All six scalars are stored as plain \c Real values with no padding.  Used
/// both for per-primitive bounds during BVH construction and for per-node AABB
/// payload nodes in \c gwn_bvh_aabb_node_soa.
template <gwn_real_type Real> struct gwn_aabb {
    Real min_x; ///< Minimum extent along the X axis.
    Real min_y; ///< Minimum extent along the Y axis.
    Real min_z; ///< Minimum extent along the Z axis.
    Real max_x; ///< Maximum extent along the X axis.
    Real max_y; ///< Maximum extent along the Y axis.
    Real max_z; ///< Maximum extent along the Z axis.
};

/// \brief Topology-only BVH node (no per-child bounds payload).
template <int Width, gwn_index_type Index = std::uint32_t> struct gwn_bvh_topology_node_soa {
    static_assert(Width >= 2, "BVH node width must be at least 2.");

    Index child_index[Width];
    Index child_count[Width];
    std::uint8_t child_kind[Width];
};

/// \brief Width-4 topology node alias.
template <gwn_index_type Index = std::uint32_t>
using gwn_bvh4_topology_node_soa = gwn_bvh_topology_node_soa<4, Index>;

/// \brief AABB payload tree node aligned with a topology node.
template <int Width, gwn_real_type Real> struct gwn_bvh_aabb_node_soa {
    static_assert(Width >= 2, "BVH node width must be at least 2.");

    Real child_min_x[Width];
    Real child_min_y[Width];
    Real child_min_z[Width];
    Real child_max_x[Width];
    Real child_max_y[Width];
    Real child_max_z[Width];
};

/// \brief Width-4 AABB node alias.
template <gwn_real_type Real> using gwn_bvh4_aabb_node_soa = gwn_bvh_aabb_node_soa<4, Real>;

/// \brief SoA node storing Taylor multipole moment coefficients per child slot.
///
/// \tparam Width  BVH node fan-out.
/// \tparam Order  Taylor expansion order (0, 1, or 2).
/// \tparam Real   Floating-point scalar type.
///
/// The primary template is declared but not defined; only the Order = 0, 1,
/// and 2 specialisations are provided.
template <int Width, int Order, gwn_real_type Real> struct gwn_bvh_taylor_node_soa;

/// \brief Order-0 (monopole) Taylor node: area-weighted normal and far-field radius.
///
/// \c child_max_p_dist2 is the squared maximum distance from the child's
/// centroid to any of its primitive centroids, used as the far-field criterion
/// during approximate winding-number traversal.
template <int Width, gwn_real_type Real> struct gwn_bvh_taylor_node_soa<Width, 0, Real> {
    Real child_max_p_dist2[Width]; ///< Squared max primitive-centroid spread.
    Real child_average_x[Width];   ///< Area-weighted average centroid X.
    Real child_average_y[Width];   ///< Area-weighted average centroid Y.
    Real child_average_z[Width];   ///< Area-weighted average centroid Z.
    Real child_n_x[Width];         ///< Summed area-weighted normal X (Nx).
    Real child_n_y[Width];         ///< Summed area-weighted normal Y (Ny).
    Real child_n_z[Width];         ///< Summed area-weighted normal Z (Nz).
};

/// \brief Order-1 Taylor node: monopole + first-order moment (symmetric Jacobian).
///
/// Extends the order-0 layout with the 6 independent components of the
/// symmetrised area-weighted normal Jacobian \f$N_{ij} + N_{ji}\f$.
template <int Width, gwn_real_type Real> struct gwn_bvh_taylor_node_soa<Width, 1, Real> {
    Real child_max_p_dist2[Width]; ///< Squared max primitive-centroid spread.
    Real child_average_x[Width];   ///< Area-weighted average centroid X.
    Real child_average_y[Width];   ///< Area-weighted average centroid Y.
    Real child_average_z[Width];   ///< Area-weighted average centroid Z.
    Real child_n_x[Width];         ///< Summed area-weighted normal X.
    Real child_n_y[Width];         ///< Summed area-weighted normal Y.
    Real child_n_z[Width];         ///< Summed area-weighted normal Z.
    Real child_nij_xx[Width];      ///< Nxx diagonal moment.
    Real child_nij_yy[Width];      ///< Nyy diagonal moment.
    Real child_nij_zz[Width];      ///< Nzz diagonal moment.
    Real child_nxy_nyx[Width];     ///< Nxy + Nyx symmetrised moment.
    Real child_nyz_nzy[Width];     ///< Nyz + Nzy symmetrised moment.
    Real child_nzx_nxz[Width];     ///< Nzx + Nxz symmetrised moment.
};

/// \brief Order-2 Taylor node: monopole + order-1 + second-order moment tensor.
///
/// Extends the order-1 layout with 10 independent components of the
/// symmetrised second-order moment tensor \f$N_{ijk}\f$.
template <int Width, gwn_real_type Real> struct gwn_bvh_taylor_node_soa<Width, 2, Real> {
    Real child_max_p_dist2[Width];
    Real child_average_x[Width];
    Real child_average_y[Width];
    Real child_average_z[Width];
    Real child_n_x[Width];
    Real child_n_y[Width];
    Real child_n_z[Width];
    Real child_nij_xx[Width];
    Real child_nij_yy[Width];
    Real child_nij_zz[Width];
    Real child_nxy_nyx[Width];
    Real child_nyz_nzy[Width];
    Real child_nzx_nxz[Width];
    Real child_nijk_xxx[Width];
    Real child_nijk_yyy[Width];
    Real child_nijk_zzz[Width];
    Real child_sum_permute_nxyz[Width];
    Real child_2nxxy_nyxx[Width];
    Real child_2nxxz_nzxx[Width];
    Real child_2nyyz_nzyy[Width];
    Real child_2nyyx_nxyy[Width];
    Real child_2nzzx_nxzz[Width];
    Real child_2nzzy_nyzz[Width];
};

/// \brief Width-4 Taylor node alias.
template <int Order, gwn_real_type Real>
using gwn_bvh4_taylor_node_soa = gwn_bvh_taylor_node_soa<4, Order, Real>;

/// \brief Topology tree accessor for a fixed-width BVH.
///
/// \remark This tree stores only hierarchy and primitive indirection.
template <int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t>
struct gwn_bvh_topology_tree_accessor {
    static_assert(Width >= 2, "BVH accessor width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    static constexpr int k_width = Width;

    cuda::std::span<gwn_bvh_topology_node_soa<Width, Index> const> nodes{}; ///< Internal nodes.
    cuda::std::span<Index const> primitive_indices{}; ///< Sorted primitive index buffer.
    gwn_bvh_child_kind root_kind = gwn_bvh_child_kind::k_invalid; ///< Root child kind.
    Index root_index = 0; ///< Root node index (internal) or begin offset (leaf).
    Index root_count = 0; ///< Primitive count at the root when it is a leaf.

    /// \brief Return \c true when the root is an internal node.
    [[nodiscard]] __host__ __device__ constexpr bool has_internal_root() const noexcept {
        return root_kind == gwn_bvh_child_kind::k_internal;
    }

    /// \brief Return \c true when the root covers a leaf primitive run.
    [[nodiscard]] __host__ __device__ constexpr bool has_leaf_root() const noexcept {
        return root_kind == gwn_bvh_child_kind::k_leaf;
    }

    /// \brief Return \c true when the accessor describes a structurally consistent tree.
    [[nodiscard]] __host__ __device__ constexpr bool is_valid() const noexcept {
        if (root_kind == gwn_bvh_child_kind::k_invalid)
            return false;

        if (root_kind == gwn_bvh_child_kind::k_internal) {
            return !nodes.empty() && gwn_span_has_storage(nodes) &&
                   gwn_index_in_bounds(root_index, nodes.size());
        }

        if (root_kind == gwn_bvh_child_kind::k_leaf) {
            if (!gwn_span_has_storage(primitive_indices) || gwn_is_invalid_index(root_index) ||
                gwn_is_invalid_index(root_count)) {
                return false;
            }
            auto const begin = static_cast<std::size_t>(root_index);
            auto const count = static_cast<std::size_t>(root_count);
            return begin <= primitive_indices.size() && count <= (primitive_indices.size() - begin);
        }

        return false;
    }
};

/// \brief AABB payload tree accessor aligned to a topology tree.
template <int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t>
struct gwn_bvh_aabb_tree_accessor {
    static_assert(Width >= 2, "BVH accessor width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    static constexpr int k_width = Width;

    cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real> const> nodes{};

    /// \brief Return \c true when no AABB node data is held.
    [[nodiscard]] __host__ __device__ constexpr bool empty() const noexcept {
        return nodes.empty();
    }

    /// \brief Return \c true when the AABB tree is consistent with a given topology.
    ///
    /// Requires \p topology to be valid.  For a leaf-root topology the AABB
    /// tree must be empty; otherwise the node counts must match.
    [[nodiscard]] __host__ __device__ constexpr bool is_valid_for(
        gwn_bvh_topology_tree_accessor<Width, Real, Index> const &topology
    ) const noexcept {
        if (!topology.is_valid())
            return false;
        if (topology.has_leaf_root())
            return empty();
        return gwn_span_has_storage(nodes) && nodes.size() == topology.nodes.size();
    }
};

/// \brief Moment payload tree accessor storing Taylor data aligned to topology.
template <int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t>
struct gwn_bvh_moment_tree_accessor {
    static_assert(Width >= 2, "BVH accessor width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    static constexpr int k_width = Width;

    cuda::std::span<gwn_bvh_taylor_node_soa<Width, 0, Real> const> taylor_order0_nodes{};
    cuda::std::span<gwn_bvh_taylor_node_soa<Width, 1, Real> const> taylor_order1_nodes{};
    cuda::std::span<gwn_bvh_taylor_node_soa<Width, 2, Real> const> taylor_order2_nodes{};

    /// \brief Return \c true when the specified Taylor order slot is populated.
    /// \tparam Order  Expansion order to test (0, 1, or 2).
    template <int Order>
    [[nodiscard]] __host__ __device__ constexpr bool has_taylor_order() const noexcept {
        if constexpr (Order == 0)
            return !taylor_order0_nodes.empty();
        if constexpr (Order == 1)
            return !taylor_order1_nodes.empty();
        if constexpr (Order == 2)
            return !taylor_order2_nodes.empty();
        return false;
    }

    /// \brief Return \c true when no Taylor data is held (all order slots empty).
    [[nodiscard]] __host__ __device__ constexpr bool empty() const noexcept {
        return taylor_order0_nodes.empty() && taylor_order1_nodes.empty() &&
               taylor_order2_nodes.empty();
    }

    /// \brief Return \c true when every non-empty Taylor order is consistent with
    ///        the given topology.
    [[nodiscard]] __host__ __device__ constexpr bool is_valid_for(
        gwn_bvh_topology_tree_accessor<Width, Real, Index> const &topology
    ) const noexcept {
        auto const validate_taylor = [&](auto const taylor_nodes) constexpr {
            return taylor_nodes.empty() || (gwn_span_has_storage(taylor_nodes) &&
                                            taylor_nodes.size() == topology.nodes.size());
        };

        if (!topology.is_valid())
            return false;
        if (topology.has_leaf_root())
            return empty();

        return validate_taylor(taylor_order0_nodes) && validate_taylor(taylor_order1_nodes) &&
               validate_taylor(taylor_order2_nodes);
    }
};

/// \brief Topology accessor alias — use when \c Width is a template parameter.
template <int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_bvh_topology_accessor = gwn_bvh_topology_tree_accessor<Width, Real, Index>;

/// \brief AABB accessor alias — use when \c Width is a template parameter.
template <int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_bvh_aabb_accessor = gwn_bvh_aabb_tree_accessor<Width, Real, Index>;

/// \brief Moment accessor alias — use when \c Width is a template parameter.
template <int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_bvh_moment_accessor = gwn_bvh_moment_tree_accessor<Width, Real, Index>;

/// \brief Width-4 topology accessor (most common use-case).
template <gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_bvh4_topology_accessor = gwn_bvh_topology_tree_accessor<4, Real, Index>;

/// \brief Width-4 AABB accessor.
template <gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_bvh4_aabb_accessor = gwn_bvh_aabb_tree_accessor<4, Real, Index>;

/// \brief Width-4 moment accessor.
template <gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_bvh4_moment_accessor = gwn_bvh_moment_tree_accessor<4, Real, Index>;

namespace detail {

template <int Width, gwn_real_type Real, gwn_index_type Index>
void gwn_release_bvh_topology_tree_accessor(
    gwn_bvh_topology_tree_accessor<Width, Real, Index> &tree, cudaStream_t const stream
) noexcept {
    gwn_free_span(tree.primitive_indices, stream);
    gwn_free_span(tree.nodes, stream);
    tree.root_kind = gwn_bvh_child_kind::k_invalid;
    tree.root_index = 0;
    tree.root_count = 0;
}

template <int Width, gwn_real_type Real, gwn_index_type Index>
void gwn_release_bvh_aabb_tree_accessor(
    gwn_bvh_aabb_tree_accessor<Width, Real, Index> &tree, cudaStream_t const stream
) noexcept {
    gwn_free_span(tree.nodes, stream);
}

template <int Width, gwn_real_type Real, gwn_index_type Index>
void gwn_release_bvh_moment_tree_accessor(
    gwn_bvh_moment_tree_accessor<Width, Real, Index> &tree, cudaStream_t const stream
) noexcept {
    gwn_free_span(tree.taylor_order2_nodes, stream);
    gwn_free_span(tree.taylor_order1_nodes, stream);
    gwn_free_span(tree.taylor_order0_nodes, stream);
}

} // namespace detail

/// \brief Owning host-side RAII wrapper for a topology tree accessor.
template <int Width, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
class gwn_bvh_topology_tree_object final : public gwn_noncopyable, public gwn_stream_mixin {
public:
    static_assert(Width >= 2, "BVH object width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    using accessor_type = gwn_bvh_topology_tree_accessor<Width, Real, Index>;

    gwn_bvh_topology_tree_object() = default;

    gwn_bvh_topology_tree_object(gwn_bvh_topology_tree_object &&other) noexcept {
        swap(*this, other);
    }

    gwn_bvh_topology_tree_object &operator=(gwn_bvh_topology_tree_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    ~gwn_bvh_topology_tree_object() { clear(); }

    void clear() noexcept { detail::gwn_release_bvh_topology_tree_accessor(accessor_, stream()); }

    void clear(cudaStream_t const clear_stream) noexcept {
        set_stream(clear_stream);
        detail::gwn_release_bvh_topology_tree_accessor(accessor_, stream());
    }

    [[nodiscard]] accessor_type &accessor() noexcept { return accessor_; }
    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }

    /// \brief Return `true` when the object holds a valid built topology tree.
    [[nodiscard]] bool has_data() const noexcept { return accessor_.is_valid(); }

    friend void
    swap(gwn_bvh_topology_tree_object &lhs, gwn_bvh_topology_tree_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    accessor_type accessor_{};
};

/// \brief Owning host-side RAII wrapper for an AABB payload tree accessor.
template <int Width, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
class gwn_bvh_aabb_tree_object final : public gwn_noncopyable, public gwn_stream_mixin {
public:
    static_assert(Width >= 2, "BVH object width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    using accessor_type = gwn_bvh_aabb_tree_accessor<Width, Real, Index>;

    gwn_bvh_aabb_tree_object() = default;

    gwn_bvh_aabb_tree_object(gwn_bvh_aabb_tree_object &&other) noexcept { swap(*this, other); }

    gwn_bvh_aabb_tree_object &operator=(gwn_bvh_aabb_tree_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    ~gwn_bvh_aabb_tree_object() { clear(); }

    void clear() noexcept { detail::gwn_release_bvh_aabb_tree_accessor(accessor_, stream()); }

    void clear(cudaStream_t const clear_stream) noexcept {
        set_stream(clear_stream);
        detail::gwn_release_bvh_aabb_tree_accessor(accessor_, stream());
    }

    [[nodiscard]] accessor_type &accessor() noexcept { return accessor_; }
    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }

    /// \brief Return `true` when the object holds AABB payload data.
    [[nodiscard]] bool has_data() const noexcept { return !accessor_.empty(); }

    friend void swap(gwn_bvh_aabb_tree_object &lhs, gwn_bvh_aabb_tree_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    accessor_type accessor_{};
};

/// \brief Owning host-side RAII wrapper for a moment payload tree accessor.
template <int Width, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
class gwn_bvh_moment_tree_object final : public gwn_noncopyable, public gwn_stream_mixin {
public:
    static_assert(Width >= 2, "BVH object width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    using accessor_type = gwn_bvh_moment_tree_accessor<Width, Real, Index>;

    gwn_bvh_moment_tree_object() = default;

    gwn_bvh_moment_tree_object(gwn_bvh_moment_tree_object &&other) noexcept { swap(*this, other); }

    gwn_bvh_moment_tree_object &operator=(gwn_bvh_moment_tree_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    ~gwn_bvh_moment_tree_object() { clear(); }

    void clear() noexcept { detail::gwn_release_bvh_moment_tree_accessor(accessor_, stream()); }

    void clear(cudaStream_t const clear_stream) noexcept {
        set_stream(clear_stream);
        detail::gwn_release_bvh_moment_tree_accessor(accessor_, stream());
    }

    [[nodiscard]] accessor_type &accessor() noexcept { return accessor_; }
    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }

    /// \brief Return `true` when the object holds any Taylor moment data.
    [[nodiscard]] bool has_data() const noexcept { return !accessor_.empty(); }

    friend void swap(gwn_bvh_moment_tree_object &lhs, gwn_bvh_moment_tree_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    accessor_type accessor_{};
};

/// \brief Width-4 topology-only BVH owning object.
///
/// \remark This alias holds only the BVH topology (hierarchy + primitive
///         indirection).  AABB bounds and Taylor moment payloads are stored
///         in separate \c gwn_bvh4_aabb_object / \c gwn_bvh4_moment_object
///         instances.
template <gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
using gwn_bvh4_topology_object = gwn_bvh_topology_tree_object<4, Real, Index>;

/// \brief Width-4 AABB payload owning object.
template <gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
using gwn_bvh4_aabb_object = gwn_bvh_aabb_tree_object<4, Real, Index>;

/// \brief Width-4 Taylor moment payload owning object.
template <gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
using gwn_bvh4_moment_object = gwn_bvh_moment_tree_object<4, Real, Index>;

/// \brief Width-parameterised topology-only BVH owning object.
template <int Width, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
using gwn_bvh_topology_object = gwn_bvh_topology_tree_object<Width, Real, Index>;

} // namespace gwn
