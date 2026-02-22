#pragma once

#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "gwn_utils.hpp"

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

template <class Real, class Index = std::int64_t> struct gwn_bvh4_node_soa {
    Real child_min_x[4];
    Real child_min_y[4];
    Real child_min_z[4];
    Real child_max_x[4];
    Real child_max_y[4];
    Real child_max_z[4];
    Index child_index[4];
    Index child_count[4];
    std::uint8_t child_kind[4];
};

template <int Order, class Real> struct gwn_bvh4_taylor_node_soa;

template <class Real> struct gwn_bvh4_taylor_node_soa<0, Real> {
    Real child_max_p_dist2[4];
    Real child_average_x[4];
    Real child_average_y[4];
    Real child_average_z[4];
    Real child_n_x[4];
    Real child_n_y[4];
    Real child_n_z[4];
};

template <class Real> struct gwn_bvh4_taylor_node_soa<1, Real> {
    Real child_max_p_dist2[4];
    Real child_average_x[4];
    Real child_average_y[4];
    Real child_average_z[4];
    Real child_n_x[4];
    Real child_n_y[4];
    Real child_n_z[4];
    Real child_nij_xx[4];
    Real child_nij_yy[4];
    Real child_nij_zz[4];
    Real child_nxy_nyx[4];
    Real child_nyz_nzy[4];
    Real child_nzx_nxz[4];
};

template <class Real> struct gwn_bvh4_taylor_node_soa<2, Real> {
    Real child_max_p_dist2[4];
    Real child_average_x[4];
    Real child_average_y[4];
    Real child_average_z[4];
    Real child_n_x[4];
    Real child_n_y[4];
    Real child_n_z[4];
    Real child_nij_xx[4];
    Real child_nij_yy[4];
    Real child_nij_zz[4];
    Real child_nxy_nyx[4];
    Real child_nyz_nzy[4];
    Real child_nzx_nxz[4];
    Real child_nijk_xxx[4];
    Real child_nijk_yyy[4];
    Real child_nijk_zzz[4];
    Real child_sum_permute_nxyz[4];
    Real child_2nxxy_nyxx[4];
    Real child_2nxxz_nzxx[4];
    Real child_2nyyz_nzyy[4];
    Real child_2nyyx_nxyy[4];
    Real child_2nzzx_nxzz[4];
    Real child_2nzzy_nyzz[4];
};

template <class Real, class Index = std::int64_t> struct gwn_bvh_accessor {
    using real_type = Real;
    using index_type = Index;

    cuda::std::span<gwn_bvh4_node_soa<Real, Index> const> nodes{};
    cuda::std::span<Index const> primitive_indices{};
    cuda::std::span<gwn_bvh4_taylor_node_soa<0, Real> const> taylor_order0_nodes{};
    cuda::std::span<gwn_bvh4_taylor_node_soa<1, Real> const> taylor_order1_nodes{};
    cuda::std::span<gwn_bvh4_taylor_node_soa<2, Real> const> taylor_order2_nodes{};
    gwn_bvh_child_kind root_kind = gwn_bvh_child_kind::k_invalid;
    Index root_index = 0;
    Index root_count = 0;

    __host__ __device__ constexpr bool has_internal_root() const noexcept {
        return root_kind == gwn_bvh_child_kind::k_internal;
    }

    __host__ __device__ constexpr bool has_leaf_root() const noexcept {
        return root_kind == gwn_bvh_child_kind::k_leaf;
    }

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

    __host__ __device__ constexpr bool is_valid() const noexcept {
        auto const validate_taylor = [&](auto const taylor_nodes) constexpr {
            return taylor_nodes.empty() ||
                   (gwn_span_has_storage(taylor_nodes) && taylor_nodes.size() == nodes.size());
        };

        if (root_kind == gwn_bvh_child_kind::k_invalid)
            return false;

        if (root_kind == gwn_bvh_child_kind::k_internal) {
            return !nodes.empty() && gwn_span_has_storage(nodes) && root_index >= 0 &&
                   static_cast<std::size_t>(root_index) < nodes.size() &&
                   validate_taylor(taylor_order0_nodes) && validate_taylor(taylor_order1_nodes) &&
                   validate_taylor(taylor_order2_nodes);
        }

        if (root_kind == gwn_bvh_child_kind::k_leaf) {
            if (!gwn_span_has_storage(primitive_indices) || root_index < 0 || root_count < 0 ||
                !taylor_order0_nodes.empty() || !taylor_order1_nodes.empty() ||
                !taylor_order2_nodes.empty()) {
                return false;
            }
            std::size_t const begin = static_cast<std::size_t>(root_index);
            std::size_t const count = static_cast<std::size_t>(root_count);
            return begin <= primitive_indices.size() && count <= (primitive_indices.size() - begin);
        }

        return false;
    }
};

namespace detail {

template <class T>
void gwn_release_bvh_span(cuda::std::span<T const> &span_view, cudaStream_t const stream) noexcept {
    if (span_view.data() != nullptr) {
        gwn_status const status = gwn_cuda_free(const_cast<T *>(span_view.data()), stream);
        if (!status.is_ok())
            GWN_HANDLE_STATUS_FAIL(status);
        span_view = {};
    }
}

template <class Real, class Index>
void gwn_release_bvh_accessor(
    gwn_bvh_accessor<Real, Index> &bvh, cudaStream_t const stream
) noexcept {
    gwn_release_bvh_span(bvh.taylor_order2_nodes, stream);
    gwn_release_bvh_span(bvh.taylor_order1_nodes, stream);
    gwn_release_bvh_span(bvh.taylor_order0_nodes, stream);
    gwn_release_bvh_span(bvh.primitive_indices, stream);
    gwn_release_bvh_span(bvh.nodes, stream);
    bvh.root_kind = gwn_bvh_child_kind::k_invalid;
    bvh.root_index = 0;
    bvh.root_count = 0;
}

} // namespace detail

template <class Real = float, class Index = std::int64_t>
class gwn_bvh_object final : public gwn_noncopyable {
public:
    using real_type = Real;
    using index_type = Index;
    using accessor_type = gwn_bvh_accessor<Real, Index>;

    gwn_bvh_object() = default;

    gwn_bvh_object(gwn_bvh_object &&other) noexcept { swap(*this, other); }

    gwn_bvh_object &operator=(gwn_bvh_object other) noexcept {
        swap(*this, other);
        return *this;
    }

    ~gwn_bvh_object() { clear(); }

    void clear(cudaStream_t const stream = gwn_default_stream()) noexcept {
        detail::gwn_release_bvh_accessor(accessor_, stream);
    }

    [[nodiscard]] accessor_type &accessor() noexcept { return accessor_; }
    [[nodiscard]] accessor_type const &accessor() const noexcept { return accessor_; }
    [[nodiscard]] bool has_bvh() const noexcept { return accessor_.is_valid(); }

    friend void swap(gwn_bvh_object &lhs, gwn_bvh_object &rhs) noexcept {
        using std::swap;
        swap(lhs.accessor_, rhs.accessor_);
    }

private:
    accessor_type accessor_{};
};

} // namespace gwn
