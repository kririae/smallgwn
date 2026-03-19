#pragma once

#include <cuda/std/tuple>

#include <type_traits>
#include <utility>

#include "detail/gwn_query_common_impl.cuh"
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

template <gwn_real_type Real, gwn_index_type Index = std::uint32_t> struct gwn_instance_record {
    Index blas_index{gwn_invalid_index<Index>()};
    gwn_similarity_transform<Real> transform{gwn_similarity_transform<Real>::identity()};
};

template <int Width, gwn_real_type Real, gwn_index_type Index, class BlasT> class gwn_scene_object;

template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_build_lbvh(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_scene_object<Width, Real, Index, BlasT> &scene,
    cudaStream_t const stream = gwn_default_stream()
) noexcept;

template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_build_hploc(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_scene_object<Width, Real, Index, BlasT> &scene,
    cudaStream_t const stream = gwn_default_stream()
) noexcept;

template <
    int Width, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t,
    class BlasT = gwn_blas_accessor<Width, Real, Index>>
struct gwn_scene_accessor {
    static_assert(Width >= 2, "Scene accessor width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    using blas_type = BlasT;
    static constexpr int k_width = Width;

    gwn_bvh_topology_accessor<Width, Real, Index> ias_topology{};
    gwn_bvh_aabb_accessor<Width, Real, Index> ias_aabb{};
    cuda::std::span<BlasT const> blas_table{};
    cuda::std::span<gwn_instance_record<Real, Index> const> instances{};

    [[nodiscard]] __host__ __device__ constexpr bool is_valid() const noexcept {
        if (!ias_topology.is_valid() || !ias_aabb.is_valid_for(ias_topology))
            return false;
        if (blas_table.empty() || instances.empty())
            return false;
        if (!gwn_span_has_storage(blas_table) || !gwn_span_has_storage(instances))
            return false;

#if defined(__CUDA_ARCH__)
        // Host code must not dereference device spans; deep referential checks stay device-only.
        for (std::size_t i = 0; i < blas_table.size(); ++i)
            if (!blas_table[i].is_valid())
                return false;
        for (std::size_t i = 0; i < instances.size(); ++i)
            if (!gwn_index_in_bounds(instances[i].blas_index, blas_table.size()))
                return false;
#endif
        return true;
    }
};

template <
    int Width, gwn_real_type Real = float, gwn_index_type Index = std::uint32_t,
    class BlasT = gwn_blas_accessor<Width, Real, Index>>
class gwn_scene_object final : public gwn_noncopyable, public gwn_stream_mixin {
public:
    static_assert(Width >= 2, "Scene object width must be at least 2.");

    using real_type = Real;
    using index_type = Index;
    using blas_type = BlasT;
    using accessor_type = gwn_scene_accessor<Width, Real, Index, BlasT>;
    static constexpr int k_width = Width;

    gwn_scene_object() = default;

    gwn_scene_object(gwn_scene_object &&other) noexcept { swap(*this, other); }

    gwn_scene_object &operator=(gwn_scene_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    [[nodiscard]] cudaStream_t stream() const noexcept { return gwn_stream_mixin::stream(); }

    void set_stream(cudaStream_t const stream) noexcept {
        gwn_stream_mixin::set_stream(stream);
        ias_topology_.set_stream(stream);
        ias_aabb_.set_stream(stream);
        blas_table_.set_stream(stream);
        instances_.set_stream(stream);
    }

    [[nodiscard]] accessor_type accessor() const noexcept {
        return accessor_type{
            ias_topology_.accessor(),
            ias_aabb_.accessor(),
            blas_table_.span(),
            instances_.span(),
        };
    }

    [[nodiscard]] bool has_data() const noexcept {
        gwn_bvh_topology_accessor<Width, Real, Index> const &ias_topology =
            ias_topology_.accessor();
        return ias_topology.is_valid() && ias_aabb_.accessor().is_valid_for(ias_topology) &&
               !blas_table_.empty() && !instances_.empty();
    }

    friend void swap(gwn_scene_object &lhs, gwn_scene_object &rhs) noexcept {
        using std::swap;
        swap(lhs.ias_topology_, rhs.ias_topology_);
        swap(lhs.ias_aabb_, rhs.ias_aabb_);
        swap(lhs.blas_table_, rhs.blas_table_);
        swap(lhs.instances_, rhs.instances_);
        swap(static_cast<gwn_stream_mixin &>(lhs), static_cast<gwn_stream_mixin &>(rhs));
    }

private:
    template <int W, gwn_real_type R, gwn_index_type I, typename B>
    friend gwn_status gwn_scene_build_lbvh(
        cuda::std::span<B const> const blas_table,
        cuda::std::span<gwn_instance_record<R, I> const> const instances,
        gwn_scene_object<W, R, I, B> &scene, cudaStream_t const stream
    ) noexcept;

    template <int W, gwn_real_type R, gwn_index_type I, typename B>
    friend gwn_status gwn_scene_build_hploc(
        cuda::std::span<B const> const blas_table,
        cuda::std::span<gwn_instance_record<R, I> const> const instances,
        gwn_scene_object<W, R, I, B> &scene, cudaStream_t const stream
    ) noexcept;

    gwn_bvh_topology_tree_object<Width, Real, Index> ias_topology_{};
    gwn_bvh_aabb_tree_object<Width, Real, Index> ias_aabb_{};
    gwn_device_array<BlasT> blas_table_{};
    gwn_device_array<gwn_instance_record<Real, Index>> instances_{};
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

template <int W, gwn_real_type R, gwn_index_type I, class B>
struct gwn_accel_traits<gwn_scene_accessor<W, R, I, B>> {
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

template <class T> struct is_scene_accessor : std::false_type {};

template <int Width, gwn_real_type Real, gwn_index_type Index, class BlasT>
struct is_scene_accessor<gwn_scene_accessor<Width, Real, Index, BlasT>> : std::true_type {};

template <class T> inline constexpr bool is_scene_accessor_v = is_scene_accessor<T>::value;

template <class T>
inline constexpr bool is_traversable_v = is_blas_accessor_v<T> || is_scene_accessor_v<T>;

template <gwn_real_type Real, gwn_index_type Index = std::uint32_t> struct gwn_ray_hit_result {
    Real t{Real(-1)};
    Index instance_id{gwn_invalid_index<Index>()};
    Index primitive_id{gwn_invalid_index<Index>()};
    Real u{Real(0)};
    Real v{Real(0)};
    gwn_ray_first_hit_status status{gwn_ray_first_hit_status::k_miss};

    __host__ __device__ constexpr bool hit() const noexcept {
        return status == gwn_ray_first_hit_status::k_hit;
    }
};

template <gwn_real_type Real, gwn_index_type Index = std::uint32_t, class... DataTrees>
using gwn_blas4_accessor = gwn_blas_accessor<4, Real, Index, DataTrees...>;

template <gwn_real_type Real, gwn_index_type Index = std::uint32_t>
using gwn_blas4_object = gwn_blas_object<4, Real, Index>;

template <
    gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    class BlasT = gwn_blas_accessor<4, Real, Index>>
using gwn_scene4_accessor = gwn_scene_accessor<4, Real, Index, BlasT>;

template <
    gwn_real_type Real, gwn_index_type Index = std::uint32_t,
    class BlasT = gwn_blas_accessor<4, Real, Index>>
using gwn_scene4_object = gwn_scene_object<4, Real, Index, BlasT>;

} // namespace gwn

#include "detail/gwn_scene_build_impl.cuh"
