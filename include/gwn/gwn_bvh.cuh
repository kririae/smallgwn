#pragma once

#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "gwn_utils.cuh"

namespace gwn {

enum class gwn_bvh_child_kind : std::uint8_t {
    k_invalid = 0,
    k_internal = 1,
    k_leaf = 2,
};

template <class Real> struct gwn_aabb {
    Real min_x;
    Real min_y;
    Real min_z;
    Real max_x;
    Real max_y;
    Real max_z;
};

template <int Width, std::size_t NaturalAlignment>
inline constexpr std::size_t k_gwn_bvh_node_alignment =
    (Width == 4) ? std::size_t(128) : NaturalAlignment;

/// \brief Topology-only BVH node (no per-child bounds payload).
template <int Width, class Index = std::int64_t>
struct alignas(k_gwn_bvh_node_alignment<Width, alignof(Index)>) gwn_bvh_topology_node_soa {
    static_assert(Width >= 2, "BVH node width must be at least 2.");

    Index child_index[Width];
    Index child_count[Width];
    std::uint8_t child_kind[Width];
};

template <class Index = std::int64_t>
using gwn_bvh4_topology_node_soa = gwn_bvh_topology_node_soa<4, Index>;

/// \brief AABB payload tree node aligned with a topology node.
template <int Width, class Real>
struct alignas(k_gwn_bvh_node_alignment<Width, alignof(Real)>) gwn_bvh_aabb_node_soa {
    static_assert(Width >= 2, "BVH node width must be at least 2.");

    Real child_min_x[Width];
    Real child_min_y[Width];
    Real child_min_z[Width];
    Real child_max_x[Width];
    Real child_max_y[Width];
    Real child_max_z[Width];
};

template <class Real> using gwn_bvh4_aabb_node_soa = gwn_bvh_aabb_node_soa<4, Real>;

template <int Width, int Order, class Real> struct gwn_bvh_taylor_node_soa;

template <int Width, class Real>
struct alignas(k_gwn_bvh_node_alignment<Width, alignof(Real)>)
    gwn_bvh_taylor_node_soa<Width, 0, Real> {
    Real child_max_p_dist2[Width];
    Real child_average_x[Width];
    Real child_average_y[Width];
    Real child_average_z[Width];
    Real child_n_x[Width];
    Real child_n_y[Width];
    Real child_n_z[Width];
};

template <int Width, class Real>
struct alignas(k_gwn_bvh_node_alignment<Width, alignof(Real)>)
    gwn_bvh_taylor_node_soa<Width, 1, Real> {
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
};

template <int Width, class Real>
struct alignas(k_gwn_bvh_node_alignment<Width, alignof(Real)>)
    gwn_bvh_taylor_node_soa<Width, 2, Real> {
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

template <int Order, class Real>
using gwn_bvh4_taylor_node_soa = gwn_bvh_taylor_node_soa<4, Order, Real>;

/// \brief Topology tree accessor for a fixed-width BVH.
///
/// \remark This tree stores only hierarchy and primitive indirection.
template <int Width, class Real, class Index = std::int64_t> struct gwn_bvh_topology_tree_accessor {
    static_assert(Width >= 2, "BVH accessor width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    static constexpr int k_width = Width;

    cuda::std::span<gwn_bvh_topology_node_soa<Width, Index> const> nodes{};
    cuda::std::span<Index const> primitive_indices{};
    gwn_bvh_child_kind root_kind = gwn_bvh_child_kind::k_invalid;
    Index root_index = 0;
    Index root_count = 0;

    [[nodiscard]] __host__ __device__ constexpr bool has_internal_root() const noexcept {
        return root_kind == gwn_bvh_child_kind::k_internal;
    }

    [[nodiscard]] __host__ __device__ constexpr bool has_leaf_root() const noexcept {
        return root_kind == gwn_bvh_child_kind::k_leaf;
    }

    [[nodiscard]] __host__ __device__ constexpr bool is_valid() const noexcept {
        if (root_kind == gwn_bvh_child_kind::k_invalid)
            return false;

        if (root_kind == gwn_bvh_child_kind::k_internal) {
            return !nodes.empty() && gwn_span_has_storage(nodes) && root_index >= 0 &&
                   static_cast<std::size_t>(root_index) < nodes.size();
        }

        if (root_kind == gwn_bvh_child_kind::k_leaf) {
            if (!gwn_span_has_storage(primitive_indices) || root_index < 0 || root_count < 0)
                return false;
            std::size_t const begin = static_cast<std::size_t>(root_index);
            std::size_t const count = static_cast<std::size_t>(root_count);
            return begin <= primitive_indices.size() && count <= (primitive_indices.size() - begin);
        }

        return false;
    }
};

/// \brief AABB payload tree accessor aligned to a topology tree.
template <int Width, class Real, class Index = std::int64_t> struct gwn_bvh_aabb_tree_accessor {
    static_assert(Width >= 2, "BVH accessor width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    static constexpr int k_width = Width;

    cuda::std::span<gwn_bvh_aabb_node_soa<Width, Real> const> nodes{};

    [[nodiscard]] __host__ __device__ constexpr bool empty() const noexcept {
        return nodes.empty();
    }

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
template <int Width, class Real, class Index = std::int64_t> struct gwn_bvh_moment_tree_accessor {
    static_assert(Width >= 2, "BVH accessor width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    static constexpr int k_width = Width;

    cuda::std::span<gwn_bvh_taylor_node_soa<Width, 0, Real> const> taylor_order0_nodes{};
    cuda::std::span<gwn_bvh_taylor_node_soa<Width, 1, Real> const> taylor_order1_nodes{};
    cuda::std::span<gwn_bvh_taylor_node_soa<Width, 2, Real> const> taylor_order2_nodes{};

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

    [[nodiscard]] __host__ __device__ constexpr bool empty() const noexcept {
        return taylor_order0_nodes.empty() && taylor_order1_nodes.empty() &&
               taylor_order2_nodes.empty();
    }

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

template <int Width, class Real, class Index = std::int64_t>
using gwn_bvh_topology_accessor = gwn_bvh_topology_tree_accessor<Width, Real, Index>;

template <int Width, class Real, class Index = std::int64_t>
using gwn_bvh_aabb_accessor = gwn_bvh_aabb_tree_accessor<Width, Real, Index>;

template <int Width, class Real, class Index = std::int64_t>
using gwn_bvh_moment_accessor = gwn_bvh_moment_tree_accessor<Width, Real, Index>;

template <class Real, class Index = std::int64_t>
using gwn_bvh_accessor = gwn_bvh_topology_tree_accessor<4, Real, Index>;

template <class Real, class Index = std::int64_t>
using gwn_bvh_aabb4_accessor = gwn_bvh_aabb_tree_accessor<4, Real, Index>;

template <class Real, class Index = std::int64_t>
using gwn_bvh_moment4_accessor = gwn_bvh_moment_tree_accessor<4, Real, Index>;

namespace detail {

template <int Width, class Real, class Index>
void gwn_release_bvh_topology_tree_accessor(
    gwn_bvh_topology_tree_accessor<Width, Real, Index> &tree, cudaStream_t const stream
) noexcept {
    gwn_free_span(tree.primitive_indices, stream);
    gwn_free_span(tree.nodes, stream);
    tree.root_kind = gwn_bvh_child_kind::k_invalid;
    tree.root_index = 0;
    tree.root_count = 0;
}

template <int Width, class Real, class Index>
void gwn_release_bvh_aabb_tree_accessor(
    gwn_bvh_aabb_tree_accessor<Width, Real, Index> &tree, cudaStream_t const stream
) noexcept {
    gwn_free_span(tree.nodes, stream);
}

template <int Width, class Real, class Index>
void gwn_release_bvh_moment_tree_accessor(
    gwn_bvh_moment_tree_accessor<Width, Real, Index> &tree, cudaStream_t const stream
) noexcept {
    gwn_free_span(tree.taylor_order2_nodes, stream);
    gwn_free_span(tree.taylor_order1_nodes, stream);
    gwn_free_span(tree.taylor_order0_nodes, stream);
}

} // namespace detail

/// \brief Owning host-side RAII wrapper for a topology tree accessor.
template <int Width, class Real = float, class Index = std::int64_t>
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
template <int Width, class Real = float, class Index = std::int64_t>
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
template <int Width, class Real = float, class Index = std::int64_t>
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
///         in separate \c gwn_bvh_aabb_object / \c gwn_bvh_moment_object
///         instances.
template <class Real = float, class Index = std::int64_t>
using gwn_bvh_object = gwn_bvh_topology_tree_object<4, Real, Index>;

template <class Real = float, class Index = std::int64_t>
using gwn_bvh_aabb_object = gwn_bvh_aabb_tree_object<4, Real, Index>;

template <class Real = float, class Index = std::int64_t>
using gwn_bvh_moment_object = gwn_bvh_moment_tree_object<4, Real, Index>;

template <int Width, class Real = float, class Index = std::int64_t>
using gwn_bvh_topology_object = gwn_bvh_topology_tree_object<Width, Real, Index>;

} // namespace gwn
