#pragma once

#include <cuda/std/tuple>

#include <type_traits>
#include <utility>

#include "detail/gwn_similarity_transform.cuh"
#include "gwn_bvh.cuh"
#include "gwn_geometry.cuh"
#include "gwn_utils.cuh"

namespace gwn {

template <
    int Width, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t, class... DataTrees>
struct gwn_blas_accessor {
    static_assert(Width >= 2, "BLAS accessor width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    static constexpr int k_width = Width;

    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_topology_accessor<Width, Real, Index> topology{};
    gwn_bvh_aabb_accessor<Width, Real, Index> aabb{};
    cuda::std::tuple<DataTrees...> data{};

    template <typename T>
    [[nodiscard]] __host__ __device__ constexpr T const &get() const noexcept {
        static_assert(
            (std::is_same_v<T, DataTrees> || ...),
            "gwn_blas_accessor::get<T>() requires T to appear in DataTrees..."
        );
        return cuda::std::get<T>(data);
    }

    [[nodiscard]] __host__ __device__ constexpr bool is_valid() const noexcept {
        return geometry.is_valid() && topology.is_valid() && aabb.is_valid_for(topology);
    }
};

template <int Width, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t>
class gwn_blas_object final : public gwn_noncopyable, public gwn_stream_mixin {
public:
    static_assert(Width >= 2, "BLAS object width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    using accessor_type = gwn_blas_accessor<Width, Real, Index>;
    static constexpr int k_width = Width;

    gwn_blas_object() = default;

    gwn_blas_object(gwn_blas_object &&other) noexcept { swap(*this, other); }

    gwn_blas_object &operator=(gwn_blas_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    [[nodiscard]] cudaStream_t stream() const noexcept { return gwn_stream_mixin::stream(); }

    void set_stream(cudaStream_t const stream) noexcept {
        gwn_stream_mixin::set_stream(stream);
        geometry_.set_stream(stream);
        topology_.set_stream(stream);
        aabb_.set_stream(stream);
    }

    [[nodiscard]] gwn_geometry_object<Real, Index> &geometry() noexcept { return geometry_; }
    [[nodiscard]] gwn_geometry_object<Real, Index> const &geometry() const noexcept {
        return geometry_;
    }

    [[nodiscard]] gwn_bvh_topology_tree_object<Width, Real, Index> &topology() noexcept {
        return topology_;
    }
    [[nodiscard]] gwn_bvh_topology_tree_object<Width, Real, Index> const &
    topology() const noexcept {
        return topology_;
    }

    [[nodiscard]] gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb() noexcept { return aabb_; }
    [[nodiscard]] gwn_bvh_aabb_tree_object<Width, Real, Index> const &aabb() const noexcept {
        return aabb_;
    }

    [[nodiscard]] bool has_data() const noexcept {
        return geometry_.has_data() && accessor().is_valid();
    }

    [[nodiscard]] accessor_type accessor() const noexcept {
        return accessor_type{
            geometry_.accessor(),
            topology_.accessor(),
            aabb_.accessor(),
            cuda::std::tuple<>{},
        };
    }

    friend void swap(gwn_blas_object &lhs, gwn_blas_object &rhs) noexcept {
        using std::swap;
        swap(lhs.geometry_, rhs.geometry_);
        swap(lhs.topology_, rhs.topology_);
        swap(lhs.aabb_, rhs.aabb_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    gwn_geometry_object<Real, Index> geometry_{};
    gwn_bvh_topology_tree_object<Width, Real, Index> topology_{};
    gwn_bvh_aabb_tree_object<Width, Real, Index> aabb_{};
};

template <class Accel> struct gwn_accel_traits;

template <int W, gwn_real_type R, gwn_index_type I, class... DataTrees>
struct gwn_accel_traits<gwn_blas_accessor<W, R, I, DataTrees...>> {
    using Real = R;
    using Index = I;
    static constexpr int Width = W;

    using real_type = R;
    using index_type = I;
    static constexpr int k_width = W;
};

template <class T> struct is_blas_accessor : std::false_type {};

template <int Width, gwn_real_type Real, gwn_index_type Index, class... DataTrees>
struct is_blas_accessor<gwn_blas_accessor<Width, Real, Index, DataTrees...>> : std::true_type {};

template <class T> inline constexpr bool is_blas_accessor_v = is_blas_accessor<T>::value;

template <gwn_real_type Real, gwn_index_type Index = std::uint32_t, class... DataTrees>
using gwn_blas4_accessor = gwn_blas_accessor<4, Real, Index, DataTrees...>;

template <gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_blas4_object = gwn_blas_object<4, Real, Index>;

} // namespace gwn
