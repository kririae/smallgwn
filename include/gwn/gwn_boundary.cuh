#pragma once

/// \file gwn_boundary.cuh
/// \brief Public mesh boundary-chain data structures and builders.
///
/// The boundary chain is the algebraic boundary of an oriented triangle-index
/// chain. It stores directed mesh edges with positive multiplicity independently
/// from BVH topology and moment data.

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "gwn_geometry.cuh"
#include "gwn_utils.cuh"

namespace gwn {

/// \brief Non-owning device view of a mesh boundary chain.
///
/// Each row represents one directed boundary edge of the oriented triangle
/// chain.  \c start_vertex and \c end_vertex store indices into the source
/// mesh vertex arrays.  \c multiplicity stores the positive net multiplicity
/// for that directed edge after opposite orientations cancel.
template <gwn_index_type Index = std::uint32_t> struct gwn_boundary_chain_accessor {
    using index_type = Index;

    cuda::std::span<Index> start_vertex{};         ///< Directed boundary edge start vertex.
    cuda::std::span<Index> end_vertex{};           ///< Directed boundary edge end vertex.
    cuda::std::span<std::uint64_t> multiplicity{}; ///< Positive net edge multiplicity.
    std::size_t mesh_vertex_count{0};   ///< Vertex count of the mesh used to build this chain.
    std::size_t mesh_triangle_count{0}; ///< Triangle count of the mesh used to build this chain.
    bool is_built{false};               ///< True after a successful build, including closed meshes.

    /// \brief Return the number of directed boundary edges in the chain.
    [[nodiscard]] __host__ __device__ constexpr std::size_t edge_count() const noexcept {
        return start_vertex.size();
    }

    /// \brief Return \c true when the built chain has no boundary edges.
    [[nodiscard]] __host__ __device__ constexpr bool empty() const noexcept {
        return edge_count() == 0;
    }

    /// \brief Return \c true when the accessor stores a built chain with matching spans.
    [[nodiscard]] __host__ __device__ constexpr bool is_valid() const noexcept {
        return is_built && start_vertex.size() == end_vertex.size() &&
               start_vertex.size() == multiplicity.size() && gwn_span_has_storage(start_vertex) &&
               gwn_span_has_storage(end_vertex) && gwn_span_has_storage(multiplicity);
    }
};

template <gwn_index_type Index = std::uint32_t> class gwn_boundary_chain_object;

namespace detail {
template <gwn_index_type Index>
void gwn_release_boundary_chain_accessor(
    gwn_boundary_chain_accessor<Index> &accessor, cudaStream_t stream
) noexcept;
} // namespace detail

/// \brief Owning host-side wrapper for a boundary-chain accessor.
///
/// \remark `clear()` and the destructor release device storage on the currently
///         bound stream.
/// \remark `has_data()` means the object has been built.  A closed mesh can
///         have `has_data() == true` and `edge_count() == 0`.
template <gwn_index_type Index>
class gwn_boundary_chain_object final : public gwn_noncopyable, public gwn_stream_mixin {
public:
    using index_type = Index;
    using accessor_type = gwn_boundary_chain_accessor<Index>;

    /// \brief Construct an empty, unbuilt boundary-chain object.
    gwn_boundary_chain_object() = default;

    /// \brief Move-construct by swapping with \p other.
    gwn_boundary_chain_object(gwn_boundary_chain_object &&other) noexcept { swap(*this, other); }

    /// \brief Move-assign with copy-and-swap semantics.
    gwn_boundary_chain_object &operator=(gwn_boundary_chain_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    /// \brief Release owned device storage on the currently bound stream.
    ~gwn_boundary_chain_object() { clear(); }

    /// \brief Release owned device storage on the currently bound stream.
    void clear() noexcept { detail::gwn_release_boundary_chain_accessor(accessor_, stream()); }

    /// \brief Return a mutable accessor view for device launches.
    [[nodiscard]] accessor_type &accessor() noexcept { return accessor_; }

    /// \brief Return a const accessor view for device launches.
    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }

    /// \brief Return \c true when a boundary-chain build has succeeded.
    [[nodiscard]] bool has_data() const noexcept { return accessor_.is_built; }

    /// \brief Swap two boundary-chain objects and their stream bindings.
    friend void swap(gwn_boundary_chain_object &lhs, gwn_boundary_chain_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    accessor_type accessor_{};
};

/// \brief Build the algebraic boundary chain from host triangle-index arrays.
///
/// \param[in] mesh_vertex_count Number of vertices in the source mesh.
/// \param[in] i0 First triangle-index array.
/// \param[in] i1 Second triangle-index array.
/// \param[in] i2 Third triangle-index array.
/// \param[in,out] out Destination boundary-chain object.
/// \param[in] stream CUDA stream used for allocation and preprocessing.
///
/// \return \c ok on success.  Returns \c invalid_argument for mismatched spans,
///         missing storage, or out-of-range triangle indices.
///
/// A closed mesh produces a built object with zero boundary edges.
template <gwn_index_type Index>
[[nodiscard]] gwn_status gwn_build_boundary_chain(
    std::size_t mesh_vertex_count, cuda::std::span<Index const> const i0,
    cuda::std::span<Index const> const i1, cuda::std::span<Index const> const i2,
    gwn_boundary_chain_object<Index> &out, cudaStream_t stream = gwn_default_stream()
) noexcept;

/// \brief Build the algebraic boundary chain from an uploaded geometry object.
///
/// The output records the geometry vertex and triangle counts.  Use the chain
/// with the geometry that produced it.
///
/// \return \c ok on success, or \c invalid_argument when \p geometry is invalid.
template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] gwn_status gwn_build_boundary_chain(
    gwn_geometry_object<Real, Index> const &geometry, gwn_boundary_chain_object<Index> &out,
    cudaStream_t stream = gwn_default_stream()
) noexcept;

} // namespace gwn

#include "detail/gwn_boundary_impl.cuh"
