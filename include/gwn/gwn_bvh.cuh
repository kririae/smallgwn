#pragma once

/// \file gwn_bvh.cuh
/// \brief Core BVH data structures, accessors, and owning host-side objects.
///
/// A canonical BVH owns child-local bounds, hierarchy references, primitive order, and triangle
/// records. Taylor moments remain independent payloads aligned to that BVH. Accessors are
/// trivially copyable device views; owning objects provide stream-bound RAII storage.

#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "detail/gwn_utils.cuh"
#include "gwn_utils.cuh"

namespace gwn {

/// \brief Axis-aligned bounding box with scalar components.
///
/// All six scalars are stored as plain \c Real values. BVH construction uses this record for
/// primitive and child bounds.
template <gwn_real_type Real> struct gwn_aabb {
    Real min_x; ///< Minimum extent along the X axis.
    Real min_y; ///< Minimum extent along the Y axis.
    Real min_z; ///< Minimum extent along the Z axis.
    Real max_x; ///< Maximum extent along the X axis.
    Real max_y; ///< Maximum extent along the Y axis.
    Real max_z; ///< Maximum extent along the Z axis.
};

/// \brief Geometry record used by triangle-consuming BVH queries.
///
/// Records follow BVH primitive order. Edges are stored relative to \c v0 so a query can load the
/// complete triangle with nine contiguous scalars and without gathering vertex indices.
template <gwn_real_type Real> struct gwn_bvh_triangle {
    Real v0_x{}; ///< First vertex X coordinate.
    Real v0_y{}; ///< First vertex Y coordinate.
    Real v0_z{}; ///< First vertex Z coordinate.
    Real e1_x{}; ///< X component of the edge from the first to the second vertex.
    Real e1_y{}; ///< Y component of the edge from the first to the second vertex.
    Real e1_z{}; ///< Z component of the edge from the first to the second vertex.
    Real e2_x{}; ///< X component of the edge from the first to the third vertex.
    Real e2_y{}; ///< Y component of the edge from the first to the third vertex.
    Real e2_z{}; ///< Z component of the edge from the first to the third vertex.
};

/// \brief Bounds and packed hierarchy reference for one BVH child.
///
/// The 64-bit reference stores a 47-bit offset, one valid bit, and a 16-bit primitive count. A
/// valid zero-count child is internal; a valid positive-count child is a leaf. Explicit masks make
/// the device representation independent of implementation-defined C++ bit-field layout.
template <gwn_real_type Real> struct gwn_bvh_child {
    static constexpr int k_offset_bits = 47;
    static constexpr int k_primitive_count_bits = 16;
    static constexpr std::uint64_t k_offset_mask = (std::uint64_t(1) << k_offset_bits) - 1;
    static constexpr std::uint64_t k_valid_mask = std::uint64_t(1) << k_offset_bits;
    static constexpr int k_primitive_count_shift = k_offset_bits + 1;
    static constexpr std::uint64_t k_primitive_count_mask =
        (std::uint64_t(1) << k_primitive_count_bits) - 1;

    gwn_aabb<Real> bounds{};      ///< Spatial bounds of the referenced child.
    std::uint64_t reference = 0u; ///< Packed valid flag, offset, and primitive count.

    /// \brief Return whether \p value fits the packed child offset.
    [[nodiscard]] __host__ __device__ static constexpr bool
    can_encode_offset(std::uint64_t const value) noexcept {
        return value <= k_offset_mask;
    }

    /// \brief Return whether \p value fits the packed leaf primitive count.
    [[nodiscard]] __host__ __device__ static constexpr bool
    can_encode_primitive_count(std::uint64_t const value) noexcept {
        return value <= k_primitive_count_mask;
    }

    /// \brief Return whether this child references an internal node or leaf range.
    [[nodiscard]] __host__ __device__ constexpr bool is_valid() const noexcept {
        return (reference & k_valid_mask) != 0u;
    }

    /// \brief Return whether this child references an internal node.
    [[nodiscard]] __host__ __device__ constexpr bool is_internal() const noexcept {
        return is_valid() && primitive_count() == 0u;
    }

    /// \brief Return whether this child references a non-empty primitive range.
    [[nodiscard]] __host__ __device__ constexpr bool is_leaf() const noexcept {
        return is_valid() && primitive_count() != 0u;
    }

    /// \brief Return the internal-node or leaf-range offset.
    [[nodiscard]] __host__ __device__ constexpr std::uint64_t offset() const noexcept {
        return reference & k_offset_mask;
    }

    /// \brief Return the number of primitives in a leaf, or zero for an internal child.
    [[nodiscard]] __host__ __device__ constexpr std::uint32_t primitive_count() const noexcept {
        return static_cast<std::uint32_t>(reference >> k_primitive_count_shift);
    }
};

/// \brief Child-AoS node used by the canonical BVH.
template <int Width, gwn_real_type Real> struct gwn_bvh_node {
    static_assert(Width >= 2, "BVH node width must be at least 2.");

    gwn_bvh_child<Real> children[Width]{}; ///< Child-local bounds and hierarchy references.

    /// \brief Return the child at \p slot.
    [[nodiscard]] __host__ __device__ constexpr gwn_bvh_child<Real> &
    child(int const slot) noexcept {
        return children[slot];
    }

    /// \brief Return the child at \p slot.
    [[nodiscard]] __host__ __device__ constexpr gwn_bvh_child<Real> const &
    child(int const slot) const noexcept {
        return children[slot];
    }
};

/// \brief Width-4 canonical BVH node alias.
template <gwn_real_type Real> using gwn_bvh4_node = gwn_bvh_node<4, Real>;

// These sizes are part of the float query-memory contract: one child is one 32-byte transaction,
// and a leaf triangle has no padding beyond its nine scalar values.
static_assert(sizeof(gwn_bvh_child<float>) == 32);
static_assert(sizeof(gwn_bvh_triangle<float>) == 36);

/// \brief SoA node storing Taylor multipole moment coefficients per child slot.
///
/// \tparam Order  Taylor expansion order (0, 1, or 2).
///
/// The primary template is declared but not defined; only the Order = 0, 1,
/// and 2 specialisations are provided.
template <int Width, int Order, gwn_real_type Real> struct gwn_bvh_moment_node;

/// \brief Order-0 (monopole) Taylor node: area-weighted normal and far-field radius.
///
/// \c child_max_p_dist2 is a conservative squared radius from the expansion center to the
/// farthest point of the child's bounds. Taylor traversal uses it for far-field acceptance.
template <int Width, gwn_real_type Real> struct gwn_bvh_moment_node<Width, 0, Real> {
    Real child_max_p_dist2[Width]; ///< Squared upper bound on expansion radius.
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
template <int Width, gwn_real_type Real> struct gwn_bvh_moment_node<Width, 1, Real> {
    Real child_max_p_dist2[Width]; ///< Squared upper bound on expansion radius.
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
template <int Width, gwn_real_type Real> struct gwn_bvh_moment_node<Width, 2, Real> {
    Real child_max_p_dist2[Width];      ///< Squared upper bound on expansion radius.
    Real child_average_x[Width];        ///< Area-weighted average centroid X.
    Real child_average_y[Width];        ///< Area-weighted average centroid Y.
    Real child_average_z[Width];        ///< Area-weighted average centroid Z.
    Real child_n_x[Width];              ///< Summed area-weighted normal X.
    Real child_n_y[Width];              ///< Summed area-weighted normal Y.
    Real child_n_z[Width];              ///< Summed area-weighted normal Z.
    Real child_nij_xx[Width];           ///< Nxx diagonal first-order moment.
    Real child_nij_yy[Width];           ///< Nyy diagonal first-order moment.
    Real child_nij_zz[Width];           ///< Nzz diagonal first-order moment.
    Real child_nxy_nyx[Width];          ///< Nxy + Nyx symmetrised first-order moment.
    Real child_nyz_nzy[Width];          ///< Nyz + Nzy symmetrised first-order moment.
    Real child_nzx_nxz[Width];          ///< Nzx + Nxz symmetrised first-order moment.
    Real child_nijk_xxx[Width];         ///< Nxxx second-order moment.
    Real child_nijk_yyy[Width];         ///< Nyyy second-order moment.
    Real child_nijk_zzz[Width];         ///< Nzzz second-order moment.
    Real child_sum_permute_nxyz[Width]; ///< Sum of all Nxyz index permutations.
    Real child_2nxxy_nyxx[Width];       ///< 2Nxxy + Nyxx symmetrised moment.
    Real child_2nxxz_nzxx[Width];       ///< 2Nxxz + Nzxx symmetrised moment.
    Real child_2nyyz_nzyy[Width];       ///< 2Nyyz + Nzyy symmetrised moment.
    Real child_2nyyx_nxyy[Width];       ///< 2Nyyx + Nxyy symmetrised moment.
    Real child_2nzzx_nxzz[Width];       ///< 2Nzzx + Nxzz symmetrised moment.
    Real child_2nzzy_nyzz[Width];       ///< 2Nzzy + Nyzz symmetrised moment.
};

/// \brief Width-4 moment node alias.
template <int Order, gwn_real_type Real>
using gwn_bvh4_moment_node = gwn_bvh_moment_node<4, Order, Real>;

/// \brief Non-owning device view of a complete canonical BVH.
template <int Width, gwn_real_type Real, gwn_index_type Index = std::uint32_t>
struct gwn_bvh_accessor {
    static_assert(Width >= 2, "BVH accessor width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    static constexpr int k_width = Width;

    cuda::std::span<gwn_bvh_node<Width, Real>> nodes{};  ///< Internal child-AoS nodes.
    cuda::std::span<Index> primitive_indices{};          ///< Original IDs in BVH primitive order.
    cuda::std::span<gwn_bvh_triangle<Real>> triangles{}; ///< Geometry in BVH primitive order.
    gwn_bvh_child<Real> root{};                          ///< Root bounds and hierarchy reference.
    std::uint32_t internal_stack_bound = 0; ///< Worst-case pending internal references.
    std::uint32_t packed_stack_bound = 0;   ///< Worst-case pending packed ray children.
    std::uint64_t revision = 0; ///< Geometry-derived state revision; zero denotes no built BVH.

    /// \brief Return whether the root references an internal node.
    [[nodiscard]] __host__ __device__ constexpr bool has_internal_root() const noexcept {
        return root.is_internal();
    }

    /// \brief Return whether the root directly references a primitive range.
    [[nodiscard]] __host__ __device__ constexpr bool has_leaf_root() const noexcept {
        return root.is_leaf();
    }

    /// \brief Return whether this accessor describes a complete queryable BVH.
    [[nodiscard]] __host__ __device__ constexpr bool is_valid() const noexcept {
        // Primitive order and triangle records are indexed together by every leaf query. Accepting
        // different lengths would turn an otherwise valid packed leaf range into an out-of-bounds
        // triangle load.
        if (revision == 0 || !root.is_valid() || primitive_indices.empty() ||
            primitive_indices.size() != triangles.size()) {
            return false;
        }
        if (!gwn_span_has_storage(nodes) || !gwn_span_has_storage(primitive_indices) ||
            !gwn_span_has_storage(triangles)) {
            return false;
        }

        std::uint64_t const root_offset = root.offset();
        if (root.is_internal())
            return root_offset < nodes.size();

        std::uint64_t const count = root.primitive_count();
        // Subtract only after checking the begin offset so begin + count cannot overflow.
        return root_offset <= primitive_indices.size() &&
               count <= primitive_indices.size() - root_offset;
    }
};

/// \brief Width-4 canonical BVH accessor.
template <gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_bvh4_accessor = gwn_bvh_accessor<4, Real, Index>;

/// \brief Moment payload accessor storing field-SoA Taylor data aligned to a canonical BVH.
///
/// \tparam Order  Taylor expansion order (0, 1, or 2).
template <int Width, int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t>
struct gwn_bvh_moment_accessor {
    static_assert(Width >= 2, "BVH accessor width must be at least 2.");
    static_assert(Order == 0 || Order == 1 || Order == 2, "Taylor order must be 0, 1, or 2.");

    using real_type = Real;
    using index_type = Index;
    static constexpr int k_width = Width;
    static constexpr int k_order = Order;

    cuda::std::span<gwn_bvh_moment_node<Width, Order, Real>> nodes{}; ///< Field-SoA moments.
    std::uint64_t bvh_revision = 0; ///< Unique BVH state used to form the coefficients.

    /// \brief Return whether the accessor has not been refit for a BVH.
    [[nodiscard]] __host__ __device__ constexpr bool empty() const noexcept {
        return bvh_revision == 0;
    }

    /// \brief Return whether the Taylor data is structurally aligned with \p bvh.
    [[nodiscard]] __host__ __device__ constexpr bool
    is_valid_for(gwn_bvh_accessor<Width, Real, Index> const &bvh) const noexcept {
        if (!bvh.is_valid() || bvh_revision != bvh.revision)
            return false;
        if (bvh.has_leaf_root())
            return nodes.empty();
        return gwn_span_has_storage(nodes) && nodes.size() == bvh.nodes.size();
    }
};

/// \brief Width-4 moment accessor.
template <int Order, gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_bvh4_moment_accessor = gwn_bvh_moment_accessor<4, Order, Real, Index>;

namespace detail {

/// \brief Release every allocation owned by \p bvh on \p stream and reset its device view.
template <int Width, gwn_real_type Real, gwn_index_type Index>
void gwn_release_bvh_accessor(
    gwn_bvh_accessor<Width, Real, Index> &bvh, cudaStream_t const stream
) noexcept {
    gwn_free_span(bvh.triangles, stream);
    gwn_free_span(bvh.primitive_indices, stream);
    gwn_free_span(bvh.nodes, stream);
    bvh.root = {};
    bvh.internal_stack_bound = 0;
    bvh.packed_stack_bound = 0;
    bvh.revision = 0;
}

/// \brief Release moment coefficients on \p stream and clear their BVH alignment identity.
template <int Width, int Order, gwn_real_type Real, gwn_index_type Index>
void gwn_release_bvh_moment_accessor(
    gwn_bvh_moment_accessor<Width, Order, Real, Index> &tree, cudaStream_t const stream
) noexcept {
    gwn_free_span(tree.nodes, stream);
    tree.bvh_revision = 0;
}

} // namespace detail

/// \brief Owning host-side RAII container for a complete canonical BVH.
///
/// The object owns child-AoS nodes, primitive order, and leaf-ordered triangle records. \c clear()
/// and destruction release storage on the currently bound stream.
template <int Width, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
class gwn_bvh_object final : public gwn_noncopyable, public gwn_stream_mixin {
public:
    static_assert(Width >= 2, "BVH object width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    using accessor_type = gwn_bvh_accessor<Width, Real, Index>;

    /// \brief Construct an empty BVH bound to the default stream.
    gwn_bvh_object() = default;

    /// \brief Move storage and stream binding from \p other.
    gwn_bvh_object(gwn_bvh_object &&other) noexcept { swap(*this, other); }

    /// \brief Replace this object using move-and-swap.
    gwn_bvh_object &operator=(gwn_bvh_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    /// \brief Release owned storage on the bound stream.
    ~gwn_bvh_object() { clear(); }

    /// \brief Release owned storage on the currently bound stream.
    void clear() noexcept { detail::gwn_release_bvh_accessor(accessor_, stream()); }

    /// \brief Release owned storage on the current stream, then bind \p next_stream.
    void clear(cudaStream_t const next_stream) noexcept {
        detail::gwn_release_bvh_accessor(accessor_, stream());
        set_stream(next_stream);
    }

    /// \brief Return the mutable non-owning device accessor.
    [[nodiscard]] accessor_type &accessor() noexcept { return accessor_; }

    /// \brief Return the immutable non-owning device accessor.
    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }

    /// \brief Return whether this object owns a complete queryable BVH.
    [[nodiscard]] bool has_data() const noexcept { return accessor_.is_valid(); }

    /// \brief Exchange storage and stream binding with \p rhs.
    friend void swap(gwn_bvh_object &lhs, gwn_bvh_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    accessor_type accessor_{};
};

/// \brief Width-4 canonical BVH owning object.
template <gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
using gwn_bvh4_object = gwn_bvh_object<4, Real, Index>;

/// \brief Owning host-side RAII wrapper for a canonical BVH moment payload.
template <int Width, int Order, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
class gwn_bvh_moment_object final : public gwn_noncopyable, public gwn_stream_mixin {
public:
    static_assert(Width >= 2, "BVH object width must be at least 2.");
    static_assert(Order == 0 || Order == 1 || Order == 2, "Taylor order must be 0, 1, or 2.");

    using real_type = Real;
    using index_type = Index;
    using accessor_type = gwn_bvh_moment_accessor<Width, Order, Real, Index>;

    /// \brief Construct an empty moment object bound to the default stream.
    gwn_bvh_moment_object() = default;

    /// \brief Move storage and stream binding from \p other.
    gwn_bvh_moment_object(gwn_bvh_moment_object &&other) noexcept { swap(*this, other); }

    /// \brief Replace this object using move-and-swap.
    gwn_bvh_moment_object &operator=(gwn_bvh_moment_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    /// \brief Release owned storage on the bound stream.
    ~gwn_bvh_moment_object() { clear(); }

    /// \brief Release owned storage on the currently bound stream.
    void clear() noexcept { detail::gwn_release_bvh_moment_accessor(accessor_, stream()); }

    /// \brief Release owned storage on the current stream, then bind \p next_stream.
    void clear(cudaStream_t const next_stream) noexcept {
        detail::gwn_release_bvh_moment_accessor(accessor_, stream());
        set_stream(next_stream);
    }

    /// \brief Return the mutable non-owning device accessor.
    [[nodiscard]] accessor_type &accessor() noexcept { return accessor_; }

    /// \brief Return the immutable non-owning device accessor.
    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }

    /// \brief Return whether the object has been refit for a BVH.
    [[nodiscard]] bool has_data() const noexcept { return !accessor_.empty(); }

    /// \brief Exchange storage and stream binding with \p rhs.
    friend void swap(gwn_bvh_moment_object &lhs, gwn_bvh_moment_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    accessor_type accessor_{};
};

/// \brief Width-4 Taylor moment payload owning object.
template <int Order, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
using gwn_bvh4_moment_object = gwn_bvh_moment_object<4, Order, Real, Index>;

} // namespace gwn
