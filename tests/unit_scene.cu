#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_harnack_meshes.hpp"
#include "test_utils.hpp"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

namespace {

std::array<Real, 3> apply_point_host(
    gwn::gwn_similarity_transform<Real> const &transform, std::array<Real, 3> const &point
) {
    std::array<Real, 3> result{};
    transform.apply_point(point[0], point[1], point[2], result[0], result[1], result[2]);
    return result;
}

std::array<Real, 3> apply_direction_host(
    gwn::gwn_similarity_transform<Real> const &transform, std::array<Real, 3> const &direction
) {
    std::array<Real, 3> result{};
    transform.apply_direction(
        direction[0], direction[1], direction[2], result[0], result[1], result[2]
    );
    return result;
}

gwn::gwn_aabb<Real> compute_expected_aabb(
    gwn::gwn_similarity_transform<Real> const &transform, gwn::gwn_aabb<Real> const &local
) {
    std::array<std::array<Real, 3>, 8> const corners{{
        {{local.min_x, local.min_y, local.min_z}},
        {{local.min_x, local.min_y, local.max_z}},
        {{local.min_x, local.max_y, local.min_z}},
        {{local.min_x, local.max_y, local.max_z}},
        {{local.max_x, local.min_y, local.min_z}},
        {{local.max_x, local.min_y, local.max_z}},
        {{local.max_x, local.max_y, local.min_z}},
        {{local.max_x, local.max_y, local.max_z}},
    }};

    gwn::gwn_aabb<Real> expected{
        std::numeric_limits<Real>::max(),    std::numeric_limits<Real>::max(),
        std::numeric_limits<Real>::max(),    std::numeric_limits<Real>::lowest(),
        std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest(),
    };

    for (auto const &corner : corners) {
        std::array<Real, 3> const world = apply_point_host(transform, corner);
        expected.min_x = std::min(expected.min_x, world[0]);
        expected.min_y = std::min(expected.min_y, world[1]);
        expected.min_z = std::min(expected.min_z, world[2]);
        expected.max_x = std::max(expected.max_x, world[0]);
        expected.max_y = std::max(expected.max_y, world[1]);
        expected.max_z = std::max(expected.max_z, world[2]);
    }

    return expected;
}

gwn::gwn_aabb<Real>
union_aabb_host(gwn::gwn_aabb<Real> const &lhs, gwn::gwn_aabb<Real> const &rhs) {
    return gwn::gwn_aabb<Real>{
        std::min(lhs.min_x, rhs.min_x), std::min(lhs.min_y, rhs.min_y),
        std::min(lhs.min_z, rhs.min_z), std::max(lhs.max_x, rhs.max_x),
        std::max(lhs.max_y, rhs.max_y), std::max(lhs.max_z, rhs.max_z),
    };
}

void expect_aabb_near(
    gwn::gwn_aabb<Real> const &actual, gwn::gwn_aabb<Real> const &expected,
    Real const tolerance = Real(1e-6)
) {
    EXPECT_NEAR(actual.min_x, expected.min_x, tolerance);
    EXPECT_NEAR(actual.min_y, expected.min_y, tolerance);
    EXPECT_NEAR(actual.min_z, expected.min_z, tolerance);
    EXPECT_NEAR(actual.max_x, expected.max_x, tolerance);
    EXPECT_NEAR(actual.max_y, expected.max_y, tolerance);
    EXPECT_NEAR(actual.max_z, expected.max_z, tolerance);
}

template <class T> T copy_device_value(T const *const src) {
    T host{};
    gwn::gwn_status const status = gwn::detail::gwn_copy_d2h(
        cuda::std::span<T>(&host, std::size_t(1)), cuda::std::span<T const>(src, std::size_t(1)),
        gwn::gwn_default_stream()
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(gwn::gwn_default_stream()));
    return host;
}

gwn::gwn_aabb<Real> copy_scene_root_bounds(
    gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> const &accessor
) {
    EXPECT_EQ(accessor.ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_internal);
    auto const root_node =
        copy_device_value(accessor.ias_topology.nodes.data() + accessor.ias_topology.root_index);
    auto const root_aabb =
        copy_device_value(accessor.ias_aabb.nodes.data() + accessor.ias_topology.root_index);

    bool has_root_child = false;
    gwn::gwn_aabb<Real> root_bounds{};
    for (int slot = 0; slot < 4; ++slot) {
        if (static_cast<gwn::gwn_bvh_child_kind>(root_node.child_kind[slot]) ==
            gwn::gwn_bvh_child_kind::k_invalid) {
            continue;
        }

        gwn::gwn_aabb<Real> const child_bounds{
            root_aabb.child_min_x[slot], root_aabb.child_min_y[slot], root_aabb.child_min_z[slot],
            root_aabb.child_max_x[slot], root_aabb.child_max_y[slot], root_aabb.child_max_z[slot],
        };
        root_bounds = has_root_child ? union_aabb_host(root_bounds, child_bounds) : child_bounds;
        has_root_child = true;
    }

    EXPECT_TRUE(has_root_child);
    return root_bounds;
}

template <class T> std::vector<T> copy_device_span(cuda::std::span<T const> const src) {
    std::vector<T> host(src.size());
    gwn::gwn_status const status = gwn::detail::gwn_copy_d2h(
        cuda::std::span<T>(host.data(), host.size()), src, gwn::gwn_default_stream()
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(gwn::gwn_default_stream()));
    return host;
}

struct TestBlasStorage {
    gwn::gwn_geometry_object<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_object<Real, Index> topology{};
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb{};

    [[nodiscard]] gwn::gwn_blas_accessor<4, Real, Index> accessor() const noexcept {
        return gwn::gwn_blas_accessor<4, Real, Index>{
            geometry.accessor(),
            topology.accessor(),
            aabb.accessor(),
            cuda::std::tuple<>{},
        };
    }
};

struct unified_scene_and_blas_ray_functor {
    gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_blas_accessor<4, Real, Index> blas{};
    cuda::std::span<Real const> ox{};
    cuda::std::span<Real const> oy{};
    cuda::std::span<Real const> oz{};
    cuda::std::span<Real const> dx{};
    cuda::std::span<Real const> dy{};
    cuda::std::span<Real const> dz{};
    cuda::std::span<Real> scene_t{};
    cuda::std::span<Index> scene_primitive_id{};
    cuda::std::span<Index> scene_instance_id{};
    cuda::std::span<Real> scene_u{};
    cuda::std::span<Real> scene_v{};
    cuda::std::span<Real> blas_t{};
    cuda::std::span<Index> blas_primitive_id{};
    cuda::std::span<Index> blas_instance_id{};
    cuda::std::span<Real> blas_u{};
    cuda::std::span<Real> blas_v{};

    __device__ void operator()(std::size_t const i) const {
        auto const scene_hit =
            gwn::gwn_ray_first_hit(scene, ox[i], oy[i], oz[i], dx[i], dy[i], dz[i]);
        auto const blas_hit =
            gwn::gwn_ray_first_hit(blas, ox[i], oy[i], oz[i], dx[i], dy[i], dz[i]);

        scene_t[i] = scene_hit.t;
        scene_primitive_id[i] = scene_hit.primitive_id;
        scene_instance_id[i] = scene_hit.instance_id;
        scene_u[i] = scene_hit.u;
        scene_v[i] = scene_hit.v;

        blas_t[i] = blas_hit.t;
        blas_primitive_id[i] = blas_hit.primitive_id;
        blas_instance_id[i] = blas_hit.instance_id;
        blas_u[i] = blas_hit.u;
        blas_v[i] = blas_hit.v;
    }
};

struct unified_scene_ray_functor {
    gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    cuda::std::span<Real const> ox{};
    cuda::std::span<Real const> oy{};
    cuda::std::span<Real const> oz{};
    cuda::std::span<Real const> dx{};
    cuda::std::span<Real const> dy{};
    cuda::std::span<Real const> dz{};
    cuda::std::span<Real> out_t{};
    cuda::std::span<Index> out_primitive_id{};
    cuda::std::span<Index> out_instance_id{};
    cuda::std::span<Real> out_u{};
    cuda::std::span<Real> out_v{};

    __device__ void operator()(std::size_t const i) const {
        auto const hit = gwn::gwn_ray_first_hit(scene, ox[i], oy[i], oz[i], dx[i], dy[i], dz[i]);
        out_t[i] = hit.t;
        out_primitive_id[i] = hit.primitive_id;
        out_instance_id[i] = hit.instance_id;
        out_u[i] = hit.u;
        out_v[i] = hit.v;
    }
};

struct unified_scene_winding_functor {
    gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    cuda::std::span<Real const> qx{};
    cuda::std::span<Real const> qy{};
    cuda::std::span<Real const> qz{};
    cuda::std::span<Real> out_winding{};

    __device__ void operator()(std::size_t const i) const {
        out_winding[i] = gwn::gwn_winding_number_point(scene, qx[i], qy[i], qz[i]);
    }
};

struct unified_scene_and_blas_winding_functor {
    gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_blas_accessor<4, Real, Index> blas{};
    cuda::std::span<Real const> scene_qx{};
    cuda::std::span<Real const> scene_qy{};
    cuda::std::span<Real const> scene_qz{};
    cuda::std::span<Real const> blas_qx{};
    cuda::std::span<Real const> blas_qy{};
    cuda::std::span<Real const> blas_qz{};
    cuda::std::span<Real> scene_winding{};
    cuda::std::span<Real> blas_winding{};

    __device__ void operator()(std::size_t const i) const {
        scene_winding[i] =
            gwn::gwn_winding_number_point(scene, scene_qx[i], scene_qy[i], scene_qz[i]);
        blas_winding[i] = gwn::gwn_winding_number_point(blas, blas_qx[i], blas_qy[i], blas_qz[i]);
    }
};

struct ray_overflow_callback_probe {
    int *flag{};

    __device__ void operator()() const noexcept {
        if (flag != nullptr)
            *flag = 1;
    }
};

template <int StackCapacity> struct unified_scene_ray_interval_functor {
    gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    cuda::std::span<Real const> ox{};
    cuda::std::span<Real const> oy{};
    cuda::std::span<Real const> oz{};
    cuda::std::span<Real const> dx{};
    cuda::std::span<Real const> dy{};
    cuda::std::span<Real const> dz{};
    cuda::std::span<Real> out_t{};
    cuda::std::span<Index> out_primitive_id{};
    cuda::std::span<Index> out_instance_id{};
    cuda::std::span<std::uint8_t> out_status{};
    Real t_min{};
    Real t_max{};

    __device__ void operator()(std::size_t const i) const {
        auto const hit = gwn::gwn_ray_first_hit<
            gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>>,
            StackCapacity>(scene, ox[i], oy[i], oz[i], dx[i], dy[i], dz[i], t_min, t_max);
        out_t[i] = hit.t;
        out_primitive_id[i] = hit.primitive_id;
        out_instance_id[i] = hit.instance_id;
        out_status[i] = static_cast<std::uint8_t>(hit.status);
    }
};

template <int StackCapacity> struct unified_scene_ray_overflow_functor {
    gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    cuda::std::span<Real const> ox{};
    cuda::std::span<Real const> oy{};
    cuda::std::span<Real const> oz{};
    cuda::std::span<Real const> dx{};
    cuda::std::span<Real const> dy{};
    cuda::std::span<Real const> dz{};
    cuda::std::span<int> callback_flag{};
    cuda::std::span<Real> out_t{};
    cuda::std::span<Index> out_primitive_id{};
    cuda::std::span<Index> out_instance_id{};
    cuda::std::span<std::uint8_t> out_status{};
    Real t_min{};
    Real t_max{};

    __device__ void operator()(std::size_t const i) const {
        auto const callback = ray_overflow_callback_probe{callback_flag.data()};
        auto const hit = gwn::gwn_ray_first_hit<
            gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>>,
            StackCapacity>(scene, ox[i], oy[i], oz[i], dx[i], dy[i], dz[i], t_min, t_max, callback);
        out_t[i] = hit.t;
        out_primitive_id[i] = hit.primitive_id;
        out_instance_id[i] = hit.instance_id;
        out_status[i] = static_cast<std::uint8_t>(hit.status);
    }
};

template <class T>
void expect_copy_from_host(gwn::gwn_device_array<T> &dst, cuda::std::span<T const> const src) {
    ASSERT_TRUE(dst.copy_from_host(src).is_ok());
}

bool run_scene_ray_first_hit_overflow_probe(
    bool &callback_called, Real &out_t, Index &out_primitive_id, Index &out_instance_id,
    std::uint8_t &out_status
) {
    using TopologyNode = gwn::gwn_bvh4_topology_node_soa<Index>;
    using AabbNode = gwn::gwn_bvh4_aabb_node_soa<Real>;

    std::array<Real, 3> const h_vx{Real(0), Real(1), Real(0)};
    std::array<Real, 3> const h_vy{Real(0), Real(0), Real(1)};
    std::array<Real, 3> const h_vz{Real(0), Real(0), Real(0)};
    std::array<Index, 1> const h_i0{Index(0)};
    std::array<Index, 1> const h_i1{Index(1)};
    std::array<Index, 1> const h_i2{Index(2)};
    std::array<Index, 1> const h_primitive_indices{Index(0)};

    gwn::gwn_geometry_object<Real, Index> geometry_object;
    if (!gwn::gwn_upload_geometry(
             geometry_object, cuda::std::span<Real const>(h_vx.data(), h_vx.size()),
             cuda::std::span<Real const>(h_vy.data(), h_vy.size()),
             cuda::std::span<Real const>(h_vz.data(), h_vz.size()),
             cuda::std::span<Index const>(h_i0.data(), h_i0.size()),
             cuda::std::span<Index const>(h_i1.data(), h_i1.size()),
             cuda::std::span<Index const>(h_i2.data(), h_i2.size())
        )
             .is_ok()) {
        return false;
    }

    std::array<TopologyNode, 3> h_topology{};
    std::array<AabbNode, 3> h_aabb{};
    std::uint8_t const invalid_kind = static_cast<std::uint8_t>(gwn::gwn_bvh_child_kind::k_invalid);
    std::uint8_t const leaf_kind = static_cast<std::uint8_t>(gwn::gwn_bvh_child_kind::k_leaf);
    std::uint8_t const internal_kind =
        static_cast<std::uint8_t>(gwn::gwn_bvh_child_kind::k_internal);

    for (auto &node : h_topology) {
        for (int i = 0; i < 4; ++i) {
            node.child_index[i] = Index(0);
            node.child_count[i] = Index(0);
            node.child_kind[i] = invalid_kind;
        }
    }
    h_topology[0].child_index[0] = Index(0);
    h_topology[0].child_count[0] = Index(1);
    h_topology[0].child_kind[0] = leaf_kind;
    h_topology[0].child_index[1] = Index(1);
    h_topology[0].child_kind[1] = internal_kind;
    h_topology[0].child_index[2] = Index(2);
    h_topology[0].child_kind[2] = internal_kind;

    for (auto &node : h_aabb) {
        for (int i = 0; i < 4; ++i) {
            node.child_min_x[i] = Real(2);
            node.child_min_y[i] = Real(2);
            node.child_min_z[i] = Real(2);
            node.child_max_x[i] = Real(3);
            node.child_max_y[i] = Real(3);
            node.child_max_z[i] = Real(3);
        }
    }
    h_aabb[0].child_min_x[0] = Real(0);
    h_aabb[0].child_min_y[0] = Real(0);
    h_aabb[0].child_min_z[0] = Real(-1);
    h_aabb[0].child_max_x[0] = Real(1);
    h_aabb[0].child_max_y[0] = Real(1);
    h_aabb[0].child_max_z[0] = Real(1);
    for (int child_slot = 1; child_slot < 3; ++child_slot) {
        h_aabb[0].child_min_x[child_slot] = Real(-1);
        h_aabb[0].child_min_y[child_slot] = Real(-1);
        h_aabb[0].child_min_z[child_slot] = Real(-2);
        h_aabb[0].child_max_x[child_slot] = Real(1);
        h_aabb[0].child_max_y[child_slot] = Real(1);
        h_aabb[0].child_max_z[child_slot] = Real(2);
    }

    gwn::gwn_device_array<TopologyNode> d_topology;
    gwn::gwn_device_array<AabbNode> d_aabb;
    gwn::gwn_device_array<Index> d_primitive_indices;
    if (!gwn::tests::resize_device_arrays(h_topology.size(), d_topology, d_aabb) ||
        !d_primitive_indices.resize(h_primitive_indices.size()).is_ok()) {
        return false;
    }
    if (!d_topology
             .copy_from_host(
                 cuda::std::span<TopologyNode const>(h_topology.data(), h_topology.size())
             )
             .is_ok() ||
        !d_aabb.copy_from_host(cuda::std::span<AabbNode const>(h_aabb.data(), h_aabb.size()))
             .is_ok() ||
        !d_primitive_indices
             .copy_from_host(
                 cuda::std::span<Index const>(
                     h_primitive_indices.data(), h_primitive_indices.size()
                 )
             )
             .is_ok()) {
        return false;
    }

    gwn::gwn_blas_accessor<4, Real, Index> const overflow_blas{
        geometry_object.accessor(),
        gwn::gwn_bvh4_topology_accessor<Real, Index>{
            d_topology.span(),
            d_primitive_indices.span(),
            gwn::gwn_bvh_child_kind::k_internal,
            Index(0),
            Index(0),
        },
        gwn::gwn_bvh4_aabb_accessor<Real, Index>{d_aabb.span()},
        cuda::std::tuple<>{},
    };

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{overflow_blas};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    if (!gwn::gwn_scene_build_lbvh<4, Real, Index>(
             cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                 blas_table.data(), blas_table.size()
             ),
             cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
                 instances.data(), instances.size()
             ),
             scene
        )
             .is_ok()) {
        return false;
    }

    std::array<Real, 1> const h_ox{Real(0)};
    std::array<Real, 1> const h_oy{Real(0)};
    std::array<Real, 1> const h_oz{Real(-10)};
    std::array<Real, 1> const h_dx{Real(0)};
    std::array<Real, 1> const h_dy{Real(0)};
    std::array<Real, 1> const h_dz{Real(1)};
    std::array<Real, 1> h_t{Real(123)};
    std::array<Index, 1> h_primitive_id{Index(7)};
    std::array<Index, 1> h_instance_id{Index(9)};
    std::array<std::uint8_t, 1> h_status{std::uint8_t(0)};
    std::array<int, 1> h_callback_flag{0};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t;
    gwn::gwn_device_array<Index> d_primitive_id, d_instance_id;
    gwn::gwn_device_array<std::uint8_t> d_status;
    gwn::gwn_device_array<int> d_callback_flag;
    if (!gwn::tests::resize_device_arrays(
            1, d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t, d_primitive_id, d_instance_id, d_status,
            d_callback_flag
        )) {
        return false;
    }
    if (!d_ox.copy_from_host(cuda::std::span<Real const>(h_ox.data(), h_ox.size())).is_ok() ||
        !d_oy.copy_from_host(cuda::std::span<Real const>(h_oy.data(), h_oy.size())).is_ok() ||
        !d_oz.copy_from_host(cuda::std::span<Real const>(h_oz.data(), h_oz.size())).is_ok() ||
        !d_dx.copy_from_host(cuda::std::span<Real const>(h_dx.data(), h_dx.size())).is_ok() ||
        !d_dy.copy_from_host(cuda::std::span<Real const>(h_dy.data(), h_dy.size())).is_ok() ||
        !d_dz.copy_from_host(cuda::std::span<Real const>(h_dz.data(), h_dz.size())).is_ok()) {
        return false;
    }

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    gwn::gwn_status const launch_status = gwn::detail::gwn_launch_linear_kernel<k_block_size>(
        1, unified_scene_ray_overflow_functor<1>{
               scene.accessor(),
               d_ox.span(),
               d_oy.span(),
               d_oz.span(),
               d_dx.span(),
               d_dy.span(),
               d_dz.span(),
               d_callback_flag.span(),
               d_t.span(),
               d_primitive_id.span(),
               d_instance_id.span(),
               d_status.span(),
               Real(0),
               std::numeric_limits<Real>::infinity(),
           }
    );
    if (!launch_status.is_ok() || cudaDeviceSynchronize() != cudaSuccess)
        return false;

    if (!d_t.copy_to_host(cuda::std::span<Real>(h_t.data(), h_t.size())).is_ok() ||
        !d_primitive_id
             .copy_to_host(cuda::std::span<Index>(h_primitive_id.data(), h_primitive_id.size()))
             .is_ok() ||
        !d_instance_id
             .copy_to_host(cuda::std::span<Index>(h_instance_id.data(), h_instance_id.size()))
             .is_ok() ||
        !d_status.copy_to_host(cuda::std::span<std::uint8_t>(h_status.data(), h_status.size()))
             .is_ok() ||
        !d_callback_flag
             .copy_to_host(cuda::std::span<int>(h_callback_flag.data(), h_callback_flag.size()))
             .is_ok()) {
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
        return false;

    callback_called = h_callback_flag[0] != 0;
    out_t = h_t[0];
    out_primitive_id = h_primitive_id[0];
    out_instance_id = h_instance_id[0];
    out_status = h_status[0];
    return true;
}

TestBlasStorage build_test_blas(
    std::vector<Real> const &vx, std::vector<Real> const &vy, std::vector<Real> const &vz,
    std::vector<Index> const &i0, std::vector<Index> const &i1, std::vector<Index> const &i2
) {
    TestBlasStorage blas{};
    gwn::gwn_status const upload_status = gwn::gwn_upload_geometry(
        blas.geometry, cuda::std::span<Real const>(vx.data(), vx.size()),
        cuda::std::span<Real const>(vy.data(), vy.size()),
        cuda::std::span<Real const>(vz.data(), vz.size()),
        cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size())
    );
    EXPECT_TRUE(upload_status.is_ok()) << gwn::tests::status_to_debug_string(upload_status);
    if (!upload_status.is_ok())
        return blas;

    gwn::gwn_status const build_status =
        gwn::gwn_bvh_facade_build_topology_aabb_lbvh<4, Real, Index>(
            blas.geometry, blas.topology, blas.aabb
        );
    EXPECT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);
    return blas;
}

template <class BuildSceneFn> void expect_scene_build_success(BuildSceneFn &&build_scene) {
    gwn::tests::SingleTriangleMesh mesh_a{};
    gwn::gwn_aabb<Real> const local_a{Real(0), Real(0), Real(0), Real(1), Real(1), Real(0)};
    gwn::gwn_aabb<Real> const local_b{Real(3), Real(2), Real(1), Real(4), Real(3), Real(1)};
    std::vector<Real> const vx_b{Real(3), Real(4), Real(3)};
    std::vector<Real> const vy_b{Real(2), Real(2), Real(3)};
    std::vector<Real> const vz_b{Real(1), Real(1), Real(1)};
    std::vector<Index> const i0_b{0};
    std::vector<Index> const i1_b{1};
    std::vector<Index> const i2_b{2};

    TestBlasStorage const blas_a = build_test_blas(
        std::vector<Real>(mesh_a.vx.begin(), mesh_a.vx.end()),
        std::vector<Real>(mesh_a.vy.begin(), mesh_a.vy.end()),
        std::vector<Real>(mesh_a.vz.begin(), mesh_a.vz.end()),
        std::vector<Index>(mesh_a.i0.begin(), mesh_a.i0.end()),
        std::vector<Index>(mesh_a.i1.begin(), mesh_a.i1.end()),
        std::vector<Index>(mesh_a.i2.begin(), mesh_a.i2.end())
    );
    TestBlasStorage const blas_b = build_test_blas(vx_b, vy_b, vz_b, i0_b, i1_b, i2_b);

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 2> blas_table{
        blas_a.accessor(),
        blas_b.accessor(),
    };
    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 2> const expected_blas_table = blas_table;

    std::array<gwn::gwn_instance_record<Real, Index>, 3> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].blas_index = Index(1);
    instances[1].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].transform.translation[0] = Real(10);
    instances[1].transform.translation[1] = Real(-2);
    instances[1].transform.translation[2] = Real(1);
    instances[2].blas_index = Index(0);
    instances[2].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[2].transform.translation[0] = Real(-5);
    instances[2].transform.translation[1] = Real(3);
    instances[2].transform.translation[2] = Real(2);
    std::array<gwn::gwn_instance_record<Real, Index>, 3> const expected_instances = instances;

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const status = build_scene(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );

    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_TRUE(scene.has_data());

    auto const accessor = scene.accessor();
    EXPECT_EQ(accessor.ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_internal);
    EXPECT_GT(accessor.ias_topology.nodes.size(), 0u);
    EXPECT_TRUE(accessor.ias_aabb.is_valid_for(accessor.ias_topology));
    EXPECT_EQ(accessor.ias_topology.primitive_indices.size(), 3u);

    auto const root_node =
        copy_device_value(accessor.ias_topology.nodes.data() + accessor.ias_topology.root_index);
    auto const root_aabb =
        copy_device_value(accessor.ias_aabb.nodes.data() + accessor.ias_topology.root_index);

    bool has_root_child = false;
    gwn::gwn_aabb<Real> actual_root_bounds{};
    for (int slot = 0; slot < 4; ++slot) {
        if (static_cast<gwn::gwn_bvh_child_kind>(root_node.child_kind[slot]) ==
            gwn::gwn_bvh_child_kind::k_invalid) {
            continue;
        }
        gwn::gwn_aabb<Real> const child_bounds{
            root_aabb.child_min_x[slot], root_aabb.child_min_y[slot], root_aabb.child_min_z[slot],
            root_aabb.child_max_x[slot], root_aabb.child_max_y[slot], root_aabb.child_max_z[slot],
        };
        actual_root_bounds =
            has_root_child ? union_aabb_host(actual_root_bounds, child_bounds) : child_bounds;
        has_root_child = true;
    }
    ASSERT_TRUE(has_root_child);

    gwn::gwn_aabb<Real> const expected_root_bounds = union_aabb_host(
        union_aabb_host(
            compute_expected_aabb(expected_instances[0].transform, local_a),
            compute_expected_aabb(expected_instances[1].transform, local_b)
        ),
        compute_expected_aabb(expected_instances[2].transform, local_a)
    );
    expect_aabb_near(actual_root_bounds, expected_root_bounds);

    instances[0].blas_index = Index(1);
    instances[0].transform.translation[0] = Real(123);
    instances[1].transform.translation[1] = Real(456);
    instances[2].transform.translation[2] = Real(789);
    blas_table[0] = {};
    blas_table[1] = {};

    std::vector<gwn::gwn_instance_record<Real, Index>> const owned_instances =
        copy_device_span(accessor.instances);
    ASSERT_EQ(owned_instances.size(), expected_instances.size());
    for (std::size_t i = 0; i < expected_instances.size(); ++i) {
        EXPECT_EQ(owned_instances[i].blas_index, expected_instances[i].blas_index);
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_NEAR(
                owned_instances[i].transform.translation[axis],
                expected_instances[i].transform.translation[axis], Real(1e-6)
            );
        EXPECT_NEAR(
            owned_instances[i].transform.scale, expected_instances[i].transform.scale, Real(1e-6)
        );
    }

    std::vector<gwn::gwn_blas_accessor<4, Real, Index>> const owned_blas_table =
        copy_device_span(accessor.blas_table);
    ASSERT_EQ(owned_blas_table.size(), expected_blas_table.size());
    for (std::size_t i = 0; i < expected_blas_table.size(); ++i) {
        EXPECT_TRUE(owned_blas_table[i].is_valid());
        EXPECT_EQ(
            owned_blas_table[i].geometry.vertex_x.data(),
            expected_blas_table[i].geometry.vertex_x.data()
        );
        EXPECT_EQ(
            owned_blas_table[i].topology.root_kind, expected_blas_table[i].topology.root_kind
        );
        EXPECT_EQ(
            owned_blas_table[i].topology.root_index, expected_blas_table[i].topology.root_index
        );
    }
}

} // namespace

TEST_F(CudaFixture, SimilarityTransformIdentity) {
    auto const transform = gwn::gwn_similarity_transform<Real>::identity();

    std::array<Real, 3> const point{{Real(1.5), Real(-2.0), Real(0.25)}};
    std::array<Real, 3> const direction{{Real(-3.0), Real(4.0), Real(2.5)}};
    std::array<Real, 3> const out_point = apply_point_host(transform, point);
    std::array<Real, 3> const out_direction = apply_direction_host(transform, direction);

    EXPECT_NEAR(out_point[0], point[0], Real(1e-6));
    EXPECT_NEAR(out_point[1], point[1], Real(1e-6));
    EXPECT_NEAR(out_point[2], point[2], Real(1e-6));
    EXPECT_NEAR(out_direction[0], direction[0], Real(1e-6));
    EXPECT_NEAR(out_direction[1], direction[1], Real(1e-6));
    EXPECT_NEAR(out_direction[2], direction[2], Real(1e-6));

    gwn::gwn_aabb<Real> const local{Real(-1), Real(-2), Real(-3), Real(4), Real(5), Real(6)};
    gwn::gwn_aabb<Real> const world = transform.transform_aabb(local);
    EXPECT_NEAR(world.min_x, local.min_x, Real(1e-6));
    EXPECT_NEAR(world.min_y, local.min_y, Real(1e-6));
    EXPECT_NEAR(world.min_z, local.min_z, Real(1e-6));
    EXPECT_NEAR(world.max_x, local.max_x, Real(1e-6));
    EXPECT_NEAR(world.max_y, local.max_y, Real(1e-6));
    EXPECT_NEAR(world.max_z, local.max_z, Real(1e-6));
}

TEST_F(CudaFixture, SimilarityTransformInverse) {
    gwn::gwn_similarity_transform<Real> transform{};
    transform.rotation[0][0] = Real(0);
    transform.rotation[0][1] = Real(-1);
    transform.rotation[0][2] = Real(0);
    transform.rotation[1][0] = Real(1);
    transform.rotation[1][1] = Real(0);
    transform.rotation[1][2] = Real(0);
    transform.rotation[2][0] = Real(0);
    transform.rotation[2][1] = Real(0);
    transform.rotation[2][2] = Real(1);
    transform.translation[0] = Real(10);
    transform.translation[1] = Real(-4);
    transform.translation[2] = Real(1);
    transform.scale = Real(2);

    std::array<Real, 3> const local_point{{Real(1), Real(2), Real(-3)}};
    std::array<Real, 3> const local_direction{{Real(4), Real(-5), Real(6)}};
    std::array<Real, 3> world_point{};
    std::array<Real, 3> world_direction{};
    std::array<Real, 3> recovered_point{};
    std::array<Real, 3> recovered_direction{};

    transform.apply_point(
        local_point[0], local_point[1], local_point[2], world_point[0], world_point[1],
        world_point[2]
    );
    transform.apply_direction(
        local_direction[0], local_direction[1], local_direction[2], world_direction[0],
        world_direction[1], world_direction[2]
    );
    transform.inverse_apply_point(
        world_point[0], world_point[1], world_point[2], recovered_point[0], recovered_point[1],
        recovered_point[2]
    );
    transform.inverse_apply_direction(
        world_direction[0], world_direction[1], world_direction[2], recovered_direction[0],
        recovered_direction[1], recovered_direction[2]
    );

    EXPECT_NEAR(recovered_point[0], local_point[0], Real(1e-6));
    EXPECT_NEAR(recovered_point[1], local_point[1], Real(1e-6));
    EXPECT_NEAR(recovered_point[2], local_point[2], Real(1e-6));
    EXPECT_NEAR(recovered_direction[0], local_direction[0], Real(1e-6));
    EXPECT_NEAR(recovered_direction[1], local_direction[1], Real(1e-6));
    EXPECT_NEAR(recovered_direction[2], local_direction[2], Real(1e-6));
}

TEST_F(CudaFixture, SimilarityTransformAABB) {
    gwn::gwn_similarity_transform<Real> transform{};
    transform.rotation[0][0] = Real(0);
    transform.rotation[0][1] = Real(-1);
    transform.rotation[0][2] = Real(0);
    transform.rotation[1][0] = Real(1);
    transform.rotation[1][1] = Real(0);
    transform.rotation[1][2] = Real(0);
    transform.rotation[2][0] = Real(0);
    transform.rotation[2][1] = Real(0);
    transform.rotation[2][2] = Real(1);
    transform.translation[0] = Real(10);
    transform.translation[1] = Real(-4);
    transform.translation[2] = Real(1);
    transform.scale = Real(2);

    gwn::gwn_aabb<Real> const local{Real(-1), Real(0), Real(2), Real(3), Real(2), Real(4)};
    gwn::gwn_aabb<Real> const expected = compute_expected_aabb(transform, local);
    gwn::gwn_aabb<Real> const world = transform.transform_aabb(local);

    EXPECT_NEAR(world.min_x, expected.min_x, Real(1e-6));
    EXPECT_NEAR(world.min_y, expected.min_y, Real(1e-6));
    EXPECT_NEAR(world.min_z, expected.min_z, Real(1e-6));
    EXPECT_NEAR(world.max_x, expected.max_x, Real(1e-6));
    EXPECT_NEAR(world.max_y, expected.max_y, Real(1e-6));
    EXPECT_NEAR(world.max_z, expected.max_z, Real(1e-6));
}

TEST_F(CudaFixture, BlasAccessorValid) {
    gwn::gwn_blas_accessor<4, Real, Index> empty{};
    EXPECT_FALSE(empty.is_valid());
}

TEST(smallgwn_unit_scene, BlasAccessorDataGet) {
    gwn::gwn_blas_accessor<4, Real, Index, int> blas{};
    blas.data = cuda::std::make_tuple(42);
    EXPECT_EQ(blas.get<int>(), 42);
}

TEST(smallgwn_unit_scene, SceneAccessorDefaultConstructedIsInvalid) {
    gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    EXPECT_FALSE(scene.is_valid());
}

TEST(smallgwn_unit_scene, SceneObjectDefaultConstructedHasNoData) {
    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    EXPECT_FALSE(scene.has_data());
}

TEST(smallgwn_unit_scene, SceneTask6ApisCompileForMixedSceneAndBlasWidths) {
    using mixed_blas_type = gwn::gwn_blas_accessor<4, Real, Index>;
    gwn::gwn_scene_object<8, Real, Index, mixed_blas_type> scene{};

    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    std::array<mixed_blas_type, 1> blas_table{};

    gwn::gwn_status const refit_status = gwn::gwn_scene_refit_transforms<8, Real, Index>(
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    EXPECT_EQ(refit_status.error(), gwn::gwn_error::invalid_argument);

    gwn::gwn_status const update_status = gwn::gwn_scene_update_blas_table<8, Real, Index>(
        cuda::std::span<mixed_blas_type const>(blas_table.data(), blas_table.size()), scene
    );
    EXPECT_EQ(update_status.error(), gwn::gwn_error::invalid_argument);
}

TEST(smallgwn_unit_scene, SceneBuildRejectsInvalidBlasAccessor) {
    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{
        gwn::gwn_blas_accessor<4, Real, Index>{}
    };
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const status = gwn::gwn_scene_build_lbvh<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );

    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, SceneBuildRejectsOutOfRangeBlasIndex) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(1);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const status = gwn::gwn_scene_build_hploc<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );

    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
}

TEST_F(CudaFixture, SceneBuildSingleInstanceLeafRoot) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const status = gwn::gwn_scene_build_lbvh<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );

    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_TRUE(scene.has_data());
    EXPECT_TRUE(scene.accessor().is_valid());
    EXPECT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_leaf);
    EXPECT_TRUE(scene.accessor().ias_aabb.empty());
}

TEST_F(CudaFixture, SceneBuildLBVH) {
    expect_scene_build_success([](auto const blas_table, auto const instances, auto &scene) {
        return gwn::gwn_scene_build_lbvh<4, Real, Index>(blas_table, instances, scene);
    });
}

TEST_F(CudaFixture, SceneBuildHPLOC) {
    expect_scene_build_success([](auto const blas_table, auto const instances, auto &scene) {
        return gwn::gwn_scene_build_hploc<4, Real, Index>(blas_table, instances, scene);
    });
}

TEST_F(CudaFixture, SceneRefitTransforms) {
    gwn::tests::SingleTriangleMesh mesh_a{};
    TestBlasStorage const blas_a = build_test_blas(
        std::vector<Real>(mesh_a.vx.begin(), mesh_a.vx.end()),
        std::vector<Real>(mesh_a.vy.begin(), mesh_a.vy.end()),
        std::vector<Real>(mesh_a.vz.begin(), mesh_a.vz.end()),
        std::vector<Index>(mesh_a.i0.begin(), mesh_a.i0.end()),
        std::vector<Index>(mesh_a.i1.begin(), mesh_a.i1.end()),
        std::vector<Index>(mesh_a.i2.begin(), mesh_a.i2.end())
    );
    gwn::gwn_aabb<Real> const local_a{Real(0), Real(0), Real(0), Real(1), Real(1), Real(0)};
    gwn::gwn_aabb<Real> const local_b{Real(3), Real(2), Real(1), Real(4), Real(3), Real(1)};
    TestBlasStorage const blas_b = build_test_blas(
        std::vector<Real>{Real(3), Real(4), Real(3)}, std::vector<Real>{Real(2), Real(2), Real(3)},
        std::vector<Real>{Real(1), Real(1), Real(1)}, std::vector<Index>{0}, std::vector<Index>{1},
        std::vector<Index>{2}
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 2> const blas_table{
        blas_a.accessor(),
        blas_b.accessor(),
    };
    std::array<gwn::gwn_instance_record<Real, Index>, 3> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].blas_index = Index(1);
    instances[1].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].transform.translation[0] = Real(10);
    instances[1].transform.translation[1] = Real(-2);
    instances[1].transform.translation[2] = Real(1);
    instances[2].blas_index = Index(0);
    instances[2].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[2].transform.translation[0] = Real(-5);
    instances[2].transform.translation[1] = Real(3);
    instances[2].transform.translation[2] = Real(2);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_lbvh<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);

    gwn::gwn_aabb<Real> const initial_root_bounds = copy_scene_root_bounds(scene.accessor());

    std::array<gwn::gwn_instance_record<Real, Index>, 3> updated_instances = instances;
    updated_instances[1].transform.translation[0] = Real(20);
    updated_instances[1].transform.translation[1] = Real(5);
    updated_instances[1].transform.translation[2] = Real(-3);

    gwn::gwn_status const refit_status = gwn::gwn_scene_refit_transforms<4, Real, Index>(
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            updated_instances.data(), updated_instances.size()
        ),
        scene
    );
    ASSERT_TRUE(refit_status.is_ok()) << gwn::tests::status_to_debug_string(refit_status);

    gwn::gwn_aabb<Real> const updated_root_bounds = copy_scene_root_bounds(scene.accessor());
    EXPECT_NEAR(initial_root_bounds.max_x, Real(14), Real(1e-6));
    EXPECT_NEAR(updated_root_bounds.max_x, Real(24), Real(1e-6));
    EXPECT_NEAR(updated_root_bounds.min_y, Real(0), Real(1e-6));
    EXPECT_NEAR(updated_root_bounds.min_z, Real(-2), Real(1e-6));

    gwn::gwn_aabb<Real> const expected_root_bounds = union_aabb_host(
        union_aabb_host(
            compute_expected_aabb(updated_instances[0].transform, local_a),
            compute_expected_aabb(updated_instances[1].transform, local_b)
        ),
        compute_expected_aabb(updated_instances[2].transform, local_a)
    );
    expect_aabb_near(updated_root_bounds, expected_root_bounds);

    std::vector<gwn::gwn_instance_record<Real, Index>> const owned_instances =
        copy_device_span(scene.accessor().instances);
    ASSERT_EQ(owned_instances.size(), updated_instances.size());
    for (std::size_t i = 0; i < updated_instances.size(); ++i) {
        EXPECT_EQ(owned_instances[i].blas_index, updated_instances[i].blas_index);
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(
                owned_instances[i].transform.translation[axis],
                updated_instances[i].transform.translation[axis], Real(1e-6)
            );
        }
    }
}

TEST_F(CudaFixture, SceneRefitTransformsFailureDoesNotMutateScene) {
    gwn::tests::SingleTriangleMesh mesh_a{};
    gwn::gwn_aabb<Real> const local_a{Real(0), Real(0), Real(0), Real(1), Real(1), Real(0)};
    gwn::gwn_aabb<Real> const local_b{Real(3), Real(2), Real(1), Real(4), Real(3), Real(1)};
    TestBlasStorage const blas_a = build_test_blas(
        std::vector<Real>(mesh_a.vx.begin(), mesh_a.vx.end()),
        std::vector<Real>(mesh_a.vy.begin(), mesh_a.vy.end()),
        std::vector<Real>(mesh_a.vz.begin(), mesh_a.vz.end()),
        std::vector<Index>(mesh_a.i0.begin(), mesh_a.i0.end()),
        std::vector<Index>(mesh_a.i1.begin(), mesh_a.i1.end()),
        std::vector<Index>(mesh_a.i2.begin(), mesh_a.i2.end())
    );
    TestBlasStorage const blas_b = build_test_blas(
        std::vector<Real>{Real(3), Real(4), Real(3)}, std::vector<Real>{Real(2), Real(2), Real(3)},
        std::vector<Real>{Real(1), Real(1), Real(1)}, std::vector<Index>{0}, std::vector<Index>{1},
        std::vector<Index>{2}
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 2> const blas_table{
        blas_a.accessor(),
        blas_b.accessor(),
    };
    std::array<gwn::gwn_instance_record<Real, Index>, 3> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].blas_index = Index(1);
    instances[1].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].transform.translation[0] = Real(10);
    instances[1].transform.translation[1] = Real(-2);
    instances[1].transform.translation[2] = Real(1);
    instances[2].blas_index = Index(0);
    instances[2].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[2].transform.translation[0] = Real(-5);
    instances[2].transform.translation[1] = Real(3);
    instances[2].transform.translation[2] = Real(2);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_lbvh<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);

    gwn::gwn_aabb<Real> const initial_root_bounds = copy_scene_root_bounds(scene.accessor());
    std::vector<gwn::gwn_instance_record<Real, Index>> const initial_owned_instances =
        copy_device_span(scene.accessor().instances);

    std::array<gwn::gwn_instance_record<Real, Index>, 3> invalid_instances = instances;
    invalid_instances[1].blas_index = Index(99);
    gwn::gwn_status const refit_status = gwn::gwn_scene_refit_transforms<4, Real, Index>(
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            invalid_instances.data(), invalid_instances.size()
        ),
        scene
    );
    EXPECT_EQ(refit_status.error(), gwn::gwn_error::invalid_argument);

    expect_aabb_near(copy_scene_root_bounds(scene.accessor()), initial_root_bounds);
    std::vector<gwn::gwn_instance_record<Real, Index>> const final_owned_instances =
        copy_device_span(scene.accessor().instances);
    ASSERT_EQ(final_owned_instances.size(), initial_owned_instances.size());
    for (std::size_t i = 0; i < initial_owned_instances.size(); ++i) {
        EXPECT_EQ(final_owned_instances[i].blas_index, initial_owned_instances[i].blas_index);
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(
                final_owned_instances[i].transform.translation[axis],
                initial_owned_instances[i].transform.translation[axis], Real(1e-6)
            );
        }
    }
    gwn::gwn_aabb<Real> const expected_root_bounds = union_aabb_host(
        union_aabb_host(
            compute_expected_aabb(instances[0].transform, local_a),
            compute_expected_aabb(instances[1].transform, local_b)
        ),
        compute_expected_aabb(instances[2].transform, local_a)
    );
    expect_aabb_near(initial_root_bounds, expected_root_bounds);
}

TEST_F(CudaFixture, SceneRefitTransformsLeafRoot) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_lbvh<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);
    ASSERT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_leaf);
    ASSERT_TRUE(scene.accessor().ias_aabb.empty());

    std::array<gwn::gwn_instance_record<Real, Index>, 1> updated_instances = instances;
    updated_instances[0].transform.translation[0] = Real(7);
    updated_instances[0].transform.translation[1] = Real(-2);
    updated_instances[0].transform.translation[2] = Real(3);

    gwn::gwn_status const refit_status = gwn::gwn_scene_refit_transforms<4, Real, Index>(
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            updated_instances.data(), updated_instances.size()
        ),
        scene
    );
    ASSERT_TRUE(refit_status.is_ok()) << gwn::tests::status_to_debug_string(refit_status);

    EXPECT_TRUE(scene.accessor().is_valid());
    EXPECT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_leaf);
    EXPECT_TRUE(scene.accessor().ias_aabb.empty());
    std::vector<gwn::gwn_instance_record<Real, Index>> const owned_instances =
        copy_device_span(scene.accessor().instances);
    ASSERT_EQ(owned_instances.size(), updated_instances.size());
    EXPECT_EQ(owned_instances[0].blas_index, updated_instances[0].blas_index);
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(
            owned_instances[0].transform.translation[axis],
            updated_instances[0].transform.translation[axis], Real(1e-6)
        );
    }
}

TEST_F(CudaFixture, SceneUpdateBlasTable) {
    gwn::tests::SingleTriangleMesh mesh_a{};
    TestBlasStorage const blas_a = build_test_blas(
        std::vector<Real>(mesh_a.vx.begin(), mesh_a.vx.end()),
        std::vector<Real>(mesh_a.vy.begin(), mesh_a.vy.end()),
        std::vector<Real>(mesh_a.vz.begin(), mesh_a.vz.end()),
        std::vector<Index>(mesh_a.i0.begin(), mesh_a.i0.end()),
        std::vector<Index>(mesh_a.i1.begin(), mesh_a.i1.end()),
        std::vector<Index>(mesh_a.i2.begin(), mesh_a.i2.end())
    );
    TestBlasStorage const blas_b = build_test_blas(
        std::vector<Real>{Real(3), Real(4), Real(3)}, std::vector<Real>{Real(2), Real(2), Real(3)},
        std::vector<Real>{Real(1), Real(1), Real(1)}, std::vector<Index>{0}, std::vector<Index>{1},
        std::vector<Index>{2}
    );
    TestBlasStorage const updated_blas_b = build_test_blas(
        std::vector<Real>{Real(20), Real(26), Real(20)},
        std::vector<Real>{Real(-1), Real(-1), Real(5)},
        std::vector<Real>{Real(2), Real(2), Real(2)}, std::vector<Index>{0}, std::vector<Index>{1},
        std::vector<Index>{2}
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 2> blas_table{
        blas_a.accessor(),
        blas_b.accessor(),
    };
    std::array<gwn::gwn_instance_record<Real, Index>, 3> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].blas_index = Index(1);
    instances[1].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].transform.translation[0] = Real(10);
    instances[1].transform.translation[1] = Real(-2);
    instances[1].transform.translation[2] = Real(1);
    instances[2].blas_index = Index(0);
    instances[2].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[2].transform.translation[0] = Real(-5);
    instances[2].transform.translation[1] = Real(3);
    instances[2].transform.translation[2] = Real(2);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_hploc<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);

    gwn::gwn_aabb<Real> const initial_root_bounds = copy_scene_root_bounds(scene.accessor());

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 2> updated_blas_table{
        blas_a.accessor(),
        updated_blas_b.accessor(),
    };
    gwn::gwn_status const update_status = gwn::gwn_scene_update_blas_table<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            updated_blas_table.data(), updated_blas_table.size()
        ),
        scene
    );
    ASSERT_TRUE(update_status.is_ok()) << gwn::tests::status_to_debug_string(update_status);

    gwn::gwn_aabb<Real> const updated_root_bounds = copy_scene_root_bounds(scene.accessor());
    EXPECT_NEAR(initial_root_bounds.max_x, Real(14), Real(1e-6));
    EXPECT_NEAR(updated_root_bounds.max_x, Real(36), Real(1e-6));
    EXPECT_NEAR(updated_root_bounds.max_y, Real(4), Real(1e-6));
    EXPECT_NEAR(updated_root_bounds.max_z, Real(3), Real(1e-6));

    gwn::gwn_aabb<Real> const local_a{Real(0), Real(0), Real(0), Real(1), Real(1), Real(0)};
    gwn::gwn_aabb<Real> const updated_local_b{Real(20), Real(-1), Real(2),
                                              Real(26), Real(5),  Real(2)};
    gwn::gwn_aabb<Real> const expected_root_bounds = union_aabb_host(
        union_aabb_host(
            compute_expected_aabb(instances[0].transform, local_a),
            compute_expected_aabb(instances[1].transform, updated_local_b)
        ),
        compute_expected_aabb(instances[2].transform, local_a)
    );
    expect_aabb_near(updated_root_bounds, expected_root_bounds);

    std::vector<gwn::gwn_blas_accessor<4, Real, Index>> const owned_blas_table =
        copy_device_span(scene.accessor().blas_table);
    ASSERT_EQ(owned_blas_table.size(), updated_blas_table.size());
    EXPECT_EQ(
        owned_blas_table[1].geometry.vertex_x.data(), updated_blas_table[1].geometry.vertex_x.data()
    );
    EXPECT_EQ(owned_blas_table[1].topology.root_index, updated_blas_table[1].topology.root_index);
    EXPECT_EQ(owned_blas_table[1].aabb.nodes.data(), updated_blas_table[1].aabb.nodes.data());
}

TEST_F(CudaFixture, SceneUpdateBlasTableLeafRoot) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );
    TestBlasStorage const updated_blas = build_test_blas(
        std::vector<Real>{Real(10), Real(12), Real(10)},
        std::vector<Real>{Real(-1), Real(-1), Real(4)},
        std::vector<Real>{Real(2), Real(2), Real(2)}, std::vector<Index>{0}, std::vector<Index>{1},
        std::vector<Index>{2}
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_hploc<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);
    ASSERT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_leaf);
    ASSERT_TRUE(scene.accessor().ias_aabb.empty());

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const updated_blas_table{
        updated_blas.accessor()
    };
    gwn::gwn_status const update_status = gwn::gwn_scene_update_blas_table<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            updated_blas_table.data(), updated_blas_table.size()
        ),
        scene
    );
    ASSERT_TRUE(update_status.is_ok()) << gwn::tests::status_to_debug_string(update_status);

    EXPECT_TRUE(scene.accessor().is_valid());
    EXPECT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_leaf);
    EXPECT_TRUE(scene.accessor().ias_aabb.empty());
    std::vector<gwn::gwn_blas_accessor<4, Real, Index>> const owned_blas_table =
        copy_device_span(scene.accessor().blas_table);
    ASSERT_EQ(owned_blas_table.size(), updated_blas_table.size());
    EXPECT_EQ(
        owned_blas_table[0].geometry.vertex_x.data(), updated_blas_table[0].geometry.vertex_x.data()
    );
    EXPECT_EQ(owned_blas_table[0].topology.root_index, updated_blas_table[0].topology.root_index);
    EXPECT_EQ(owned_blas_table[0].aabb.nodes.data(), updated_blas_table[0].aabb.nodes.data());
}

TEST_F(CudaFixture, SceneBuildAndRefitFromDeviceSpans) {
    gwn::tests::SingleTriangleMesh mesh_a{};
    gwn::gwn_aabb<Real> const local_a{Real(0), Real(0), Real(0), Real(1), Real(1), Real(0)};
    gwn::gwn_aabb<Real> const local_b{Real(3), Real(2), Real(1), Real(4), Real(3), Real(1)};
    TestBlasStorage const blas_a = build_test_blas(
        std::vector<Real>(mesh_a.vx.begin(), mesh_a.vx.end()),
        std::vector<Real>(mesh_a.vy.begin(), mesh_a.vy.end()),
        std::vector<Real>(mesh_a.vz.begin(), mesh_a.vz.end()),
        std::vector<Index>(mesh_a.i0.begin(), mesh_a.i0.end()),
        std::vector<Index>(mesh_a.i1.begin(), mesh_a.i1.end()),
        std::vector<Index>(mesh_a.i2.begin(), mesh_a.i2.end())
    );
    TestBlasStorage const blas_b = build_test_blas(
        std::vector<Real>{Real(3), Real(4), Real(3)}, std::vector<Real>{Real(2), Real(2), Real(3)},
        std::vector<Real>{Real(1), Real(1), Real(1)}, std::vector<Index>{0}, std::vector<Index>{1},
        std::vector<Index>{2}
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 2> blas_table{
        blas_a.accessor(),
        blas_b.accessor(),
    };
    std::array<gwn::gwn_instance_record<Real, Index>, 3> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].blas_index = Index(1);
    instances[1].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].transform.translation[0] = Real(10);
    instances[1].transform.translation[1] = Real(-2);
    instances[1].transform.translation[2] = Real(1);
    instances[2].blas_index = Index(0);
    instances[2].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[2].transform.translation[0] = Real(-5);
    instances[2].transform.translation[1] = Real(3);
    instances[2].transform.translation[2] = Real(2);

    gwn::gwn_device_array<gwn::gwn_blas_accessor<4, Real, Index>> d_blas_table{};
    ASSERT_TRUE(d_blas_table
                    .copy_from_host(
                        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                            blas_table.data(), blas_table.size()
                        )
                    )
                    .is_ok());
    gwn::gwn_device_array<gwn::gwn_instance_record<Real, Index>> d_instances{};
    ASSERT_TRUE(d_instances
                    .copy_from_host(
                        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
                            instances.data(), instances.size()
                        )
                    )
                    .is_ok());

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_hploc<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            d_blas_table.data(), d_blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            d_instances.data(), d_instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);

    gwn::gwn_aabb<Real> const expected_initial_root_bounds = union_aabb_host(
        union_aabb_host(
            compute_expected_aabb(instances[0].transform, local_a),
            compute_expected_aabb(instances[1].transform, local_b)
        ),
        compute_expected_aabb(instances[2].transform, local_a)
    );
    expect_aabb_near(copy_scene_root_bounds(scene.accessor()), expected_initial_root_bounds);

    std::array<gwn::gwn_instance_record<Real, Index>, 3> updated_instances = instances;
    updated_instances[1].transform.translation[0] = Real(20);
    updated_instances[1].transform.translation[1] = Real(5);
    updated_instances[1].transform.translation[2] = Real(-3);
    gwn::gwn_device_array<gwn::gwn_instance_record<Real, Index>> d_updated_instances{};
    ASSERT_TRUE(d_updated_instances
                    .copy_from_host(
                        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
                            updated_instances.data(), updated_instances.size()
                        )
                    )
                    .is_ok());

    gwn::gwn_status const refit_status = gwn::gwn_scene_refit_transforms<4, Real, Index>(
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            d_updated_instances.data(), d_updated_instances.size()
        ),
        scene
    );
    ASSERT_TRUE(refit_status.is_ok()) << gwn::tests::status_to_debug_string(refit_status);

    gwn::gwn_aabb<Real> const expected_updated_root_bounds = union_aabb_host(
        union_aabb_host(
            compute_expected_aabb(updated_instances[0].transform, local_a),
            compute_expected_aabb(updated_instances[1].transform, local_b)
        ),
        compute_expected_aabb(updated_instances[2].transform, local_a)
    );
    expect_aabb_near(copy_scene_root_bounds(scene.accessor()), expected_updated_root_bounds);
}

TEST_F(CudaFixture, SceneUpdateBlasTableFromDeviceSpans) {
    gwn::tests::SingleTriangleMesh mesh_a{};
    TestBlasStorage const blas_a = build_test_blas(
        std::vector<Real>(mesh_a.vx.begin(), mesh_a.vx.end()),
        std::vector<Real>(mesh_a.vy.begin(), mesh_a.vy.end()),
        std::vector<Real>(mesh_a.vz.begin(), mesh_a.vz.end()),
        std::vector<Index>(mesh_a.i0.begin(), mesh_a.i0.end()),
        std::vector<Index>(mesh_a.i1.begin(), mesh_a.i1.end()),
        std::vector<Index>(mesh_a.i2.begin(), mesh_a.i2.end())
    );
    TestBlasStorage const blas_b = build_test_blas(
        std::vector<Real>{Real(3), Real(4), Real(3)}, std::vector<Real>{Real(2), Real(2), Real(3)},
        std::vector<Real>{Real(1), Real(1), Real(1)}, std::vector<Index>{0}, std::vector<Index>{1},
        std::vector<Index>{2}
    );
    TestBlasStorage const updated_blas_b = build_test_blas(
        std::vector<Real>{Real(20), Real(26), Real(20)},
        std::vector<Real>{Real(-1), Real(-1), Real(5)},
        std::vector<Real>{Real(2), Real(2), Real(2)}, std::vector<Index>{0}, std::vector<Index>{1},
        std::vector<Index>{2}
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 2> blas_table{
        blas_a.accessor(),
        blas_b.accessor(),
    };
    std::array<gwn::gwn_instance_record<Real, Index>, 3> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].blas_index = Index(1);
    instances[1].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].transform.translation[0] = Real(10);
    instances[1].transform.translation[1] = Real(-2);
    instances[1].transform.translation[2] = Real(1);
    instances[2].blas_index = Index(0);
    instances[2].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[2].transform.translation[0] = Real(-5);
    instances[2].transform.translation[1] = Real(3);
    instances[2].transform.translation[2] = Real(2);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_lbvh<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 2> updated_blas_table{
        blas_a.accessor(),
        updated_blas_b.accessor(),
    };
    gwn::gwn_device_array<gwn::gwn_blas_accessor<4, Real, Index>> d_updated_blas_table{};
    ASSERT_TRUE(d_updated_blas_table
                    .copy_from_host(
                        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                            updated_blas_table.data(), updated_blas_table.size()
                        )
                    )
                    .is_ok());

    gwn::gwn_status const update_status = gwn::gwn_scene_update_blas_table<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            d_updated_blas_table.data(), d_updated_blas_table.size()
        ),
        scene
    );
    ASSERT_TRUE(update_status.is_ok()) << gwn::tests::status_to_debug_string(update_status);

    gwn::gwn_aabb<Real> const local_a{Real(0), Real(0), Real(0), Real(1), Real(1), Real(0)};
    gwn::gwn_aabb<Real> const updated_local_b{Real(20), Real(-1), Real(2),
                                              Real(26), Real(5),  Real(2)};
    gwn::gwn_aabb<Real> const expected_root_bounds = union_aabb_host(
        union_aabb_host(
            compute_expected_aabb(instances[0].transform, local_a),
            compute_expected_aabb(instances[1].transform, updated_local_b)
        ),
        compute_expected_aabb(instances[2].transform, local_a)
    );
    expect_aabb_near(copy_scene_root_bounds(scene.accessor()), expected_root_bounds);
}

TEST_F(CudaFixture, SceneUpdateBlasTableFailureDoesNotMutateScene) {
    gwn::tests::SingleTriangleMesh mesh_a{};
    TestBlasStorage const blas_a = build_test_blas(
        std::vector<Real>(mesh_a.vx.begin(), mesh_a.vx.end()),
        std::vector<Real>(mesh_a.vy.begin(), mesh_a.vy.end()),
        std::vector<Real>(mesh_a.vz.begin(), mesh_a.vz.end()),
        std::vector<Index>(mesh_a.i0.begin(), mesh_a.i0.end()),
        std::vector<Index>(mesh_a.i1.begin(), mesh_a.i1.end()),
        std::vector<Index>(mesh_a.i2.begin(), mesh_a.i2.end())
    );
    TestBlasStorage const blas_b = build_test_blas(
        std::vector<Real>{Real(3), Real(4), Real(3)}, std::vector<Real>{Real(2), Real(2), Real(3)},
        std::vector<Real>{Real(1), Real(1), Real(1)}, std::vector<Index>{0}, std::vector<Index>{1},
        std::vector<Index>{2}
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 2> const blas_table{
        blas_a.accessor(),
        blas_b.accessor(),
    };
    std::array<gwn::gwn_instance_record<Real, Index>, 2> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].blas_index = Index(1);
    instances[1].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].transform.translation[0] = Real(10);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_lbvh<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);

    gwn::gwn_aabb<Real> const initial_root_bounds = copy_scene_root_bounds(scene.accessor());
    std::vector<gwn::gwn_blas_accessor<4, Real, Index>> const initial_owned_blas_table =
        copy_device_span(scene.accessor().blas_table);

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const missing_blas_table{
        blas_a.accessor()
    };
    gwn::gwn_status const update_status = gwn::gwn_scene_update_blas_table<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            missing_blas_table.data(), missing_blas_table.size()
        ),
        scene
    );
    EXPECT_EQ(update_status.error(), gwn::gwn_error::invalid_argument);

    gwn::gwn_aabb<Real> const final_root_bounds = copy_scene_root_bounds(scene.accessor());
    expect_aabb_near(final_root_bounds, initial_root_bounds);

    std::vector<gwn::gwn_blas_accessor<4, Real, Index>> const final_owned_blas_table =
        copy_device_span(scene.accessor().blas_table);
    ASSERT_EQ(final_owned_blas_table.size(), initial_owned_blas_table.size());
    for (std::size_t i = 0; i < initial_owned_blas_table.size(); ++i) {
        EXPECT_EQ(
            final_owned_blas_table[i].geometry.vertex_x.data(),
            initial_owned_blas_table[i].geometry.vertex_x.data()
        );
        EXPECT_EQ(
            final_owned_blas_table[i].topology.root_index,
            initial_owned_blas_table[i].topology.root_index
        );
        EXPECT_EQ(
            final_owned_blas_table[i].aabb.nodes.data(),
            initial_owned_blas_table[i].aabb.nodes.data()
        );
    }
}

TEST_F(CudaFixture, SceneRayFirstHitSingleInstance) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_lbvh<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);
    ASSERT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_leaf);

    std::array<Real, 1> const h_ox{Real(0.25)};
    std::array<Real, 1> const h_oy{Real(0.25)};
    std::array<Real, 1> const h_oz{Real(-1)};
    std::array<Real, 1> const h_dx{Real(0)};
    std::array<Real, 1> const h_dy{Real(0)};
    std::array<Real, 1> const h_dz{Real(1)};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
    gwn::gwn_device_array<Real> d_scene_t, d_scene_u, d_scene_v, d_blas_t, d_blas_u, d_blas_v;
    gwn::gwn_device_array<Index> d_scene_primitive_id, d_scene_instance_id, d_blas_primitive_id,
        d_blas_instance_id;
    expect_copy_from_host(d_ox, cuda::std::span<Real const>(h_ox.data(), h_ox.size()));
    expect_copy_from_host(d_oy, cuda::std::span<Real const>(h_oy.data(), h_oy.size()));
    expect_copy_from_host(d_oz, cuda::std::span<Real const>(h_oz.data(), h_oz.size()));
    expect_copy_from_host(d_dx, cuda::std::span<Real const>(h_dx.data(), h_dx.size()));
    expect_copy_from_host(d_dy, cuda::std::span<Real const>(h_dy.data(), h_dy.size()));
    expect_copy_from_host(d_dz, cuda::std::span<Real const>(h_dz.data(), h_dz.size()));
    ASSERT_TRUE(
        gwn::tests::resize_device_arrays(
            1, d_scene_t, d_scene_u, d_scene_v, d_blas_t, d_blas_u, d_blas_v, d_scene_primitive_id,
            d_scene_instance_id, d_blas_primitive_id, d_blas_instance_id
        )
    );

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    gwn::gwn_status const launch_status = gwn::detail::gwn_launch_linear_kernel<k_block_size>(
        1, unified_scene_and_blas_ray_functor{
               scene.accessor(),
               blas.accessor(),
               d_ox.span(),
               d_oy.span(),
               d_oz.span(),
               d_dx.span(),
               d_dy.span(),
               d_dz.span(),
               d_scene_t.span(),
               d_scene_primitive_id.span(),
               d_scene_instance_id.span(),
               d_scene_u.span(),
               d_scene_v.span(),
               d_blas_t.span(),
               d_blas_primitive_id.span(),
               d_blas_instance_id.span(),
               d_blas_u.span(),
               d_blas_v.span(),
           }
    );
    ASSERT_TRUE(launch_status.is_ok()) << gwn::tests::status_to_debug_string(launch_status);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> scene_t{};
    std::array<Real, 1> scene_u{};
    std::array<Real, 1> scene_v{};
    std::array<Real, 1> blas_t{};
    std::array<Real, 1> blas_u{};
    std::array<Real, 1> blas_v{};
    std::array<Index, 1> scene_primitive_id{};
    std::array<Index, 1> scene_instance_id{};
    std::array<Index, 1> blas_primitive_id{};
    std::array<Index, 1> blas_instance_id{};
    ASSERT_TRUE(
        d_scene_t.copy_to_host(cuda::std::span<Real>(scene_t.data(), scene_t.size())).is_ok()
    );
    ASSERT_TRUE(
        d_scene_u.copy_to_host(cuda::std::span<Real>(scene_u.data(), scene_u.size())).is_ok()
    );
    ASSERT_TRUE(
        d_scene_v.copy_to_host(cuda::std::span<Real>(scene_v.data(), scene_v.size())).is_ok()
    );
    ASSERT_TRUE(d_blas_t.copy_to_host(cuda::std::span<Real>(blas_t.data(), blas_t.size())).is_ok());
    ASSERT_TRUE(d_blas_u.copy_to_host(cuda::std::span<Real>(blas_u.data(), blas_u.size())).is_ok());
    ASSERT_TRUE(d_blas_v.copy_to_host(cuda::std::span<Real>(blas_v.data(), blas_v.size())).is_ok());
    ASSERT_TRUE(d_scene_primitive_id
                    .copy_to_host(
                        cuda::std::span<Index>(scene_primitive_id.data(), scene_primitive_id.size())
                    )
                    .is_ok());
    ASSERT_TRUE(d_scene_instance_id
                    .copy_to_host(
                        cuda::std::span<Index>(scene_instance_id.data(), scene_instance_id.size())
                    )
                    .is_ok());
    ASSERT_TRUE(d_blas_primitive_id
                    .copy_to_host(
                        cuda::std::span<Index>(blas_primitive_id.data(), blas_primitive_id.size())
                    )
                    .is_ok());
    ASSERT_TRUE(
        d_blas_instance_id
            .copy_to_host(cuda::std::span<Index>(blas_instance_id.data(), blas_instance_id.size()))
            .is_ok()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_NEAR(scene_t[0], blas_t[0], Real(1e-6));
    EXPECT_NEAR(scene_u[0], blas_u[0], Real(1e-6));
    EXPECT_NEAR(scene_v[0], blas_v[0], Real(1e-6));
    EXPECT_EQ(scene_primitive_id[0], blas_primitive_id[0]);
    EXPECT_EQ(scene_instance_id[0], Index(0));
    EXPECT_EQ(blas_instance_id[0], gwn::gwn_invalid_index<Index>());
    EXPECT_NEAR(scene_t[0], Real(1), Real(1e-6));
    EXPECT_EQ(scene_primitive_id[0], Index(0));
    EXPECT_NEAR(scene_u[0], Real(0.25), Real(1e-6));
    EXPECT_NEAR(scene_v[0], Real(0.25), Real(1e-6));
}

TEST_F(CudaFixture, SceneRayFirstHitMultiInstance) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 3> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[0].transform.translation[2] = Real(5);
    instances[1].blas_index = Index(0);
    instances[1].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].transform.translation[2] = Real(1);
    instances[2].blas_index = Index(0);
    instances[2].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[2].transform.translation[2] = Real(3);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_lbvh<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);
    ASSERT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_internal);

    std::array<Real, 1> const h_ox{Real(0.25)};
    std::array<Real, 1> const h_oy{Real(0.25)};
    std::array<Real, 1> const h_oz{Real(-2)};
    std::array<Real, 1> const h_dx{Real(0)};
    std::array<Real, 1> const h_dy{Real(0)};
    std::array<Real, 1> const h_dz{Real(1)};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t, d_u, d_v;
    gwn::gwn_device_array<Index> d_primitive_id, d_instance_id;
    expect_copy_from_host(d_ox, cuda::std::span<Real const>(h_ox.data(), h_ox.size()));
    expect_copy_from_host(d_oy, cuda::std::span<Real const>(h_oy.data(), h_oy.size()));
    expect_copy_from_host(d_oz, cuda::std::span<Real const>(h_oz.data(), h_oz.size()));
    expect_copy_from_host(d_dx, cuda::std::span<Real const>(h_dx.data(), h_dx.size()));
    expect_copy_from_host(d_dy, cuda::std::span<Real const>(h_dy.data(), h_dy.size()));
    expect_copy_from_host(d_dz, cuda::std::span<Real const>(h_dz.data(), h_dz.size()));
    ASSERT_TRUE(gwn::tests::resize_device_arrays(1, d_t, d_u, d_v, d_primitive_id, d_instance_id));

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    gwn::gwn_status const launch_status = gwn::detail::gwn_launch_linear_kernel<k_block_size>(
        1, unified_scene_ray_functor{
               scene.accessor(),
               d_ox.span(),
               d_oy.span(),
               d_oz.span(),
               d_dx.span(),
               d_dy.span(),
               d_dz.span(),
               d_t.span(),
               d_primitive_id.span(),
               d_instance_id.span(),
               d_u.span(),
               d_v.span(),
           }
    );
    ASSERT_TRUE(launch_status.is_ok()) << gwn::tests::status_to_debug_string(launch_status);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> point_t{};
    std::array<Real, 1> point_u{};
    std::array<Real, 1> point_v{};
    std::array<Index, 1> point_primitive_id{};
    std::array<Index, 1> point_instance_id{};
    ASSERT_TRUE(d_t.copy_to_host(cuda::std::span<Real>(point_t.data(), point_t.size())).is_ok());
    ASSERT_TRUE(d_u.copy_to_host(cuda::std::span<Real>(point_u.data(), point_u.size())).is_ok());
    ASSERT_TRUE(d_v.copy_to_host(cuda::std::span<Real>(point_v.data(), point_v.size())).is_ok());
    ASSERT_TRUE(d_primitive_id
                    .copy_to_host(
                        cuda::std::span<Index>(point_primitive_id.data(), point_primitive_id.size())
                    )
                    .is_ok());
    ASSERT_TRUE(d_instance_id
                    .copy_to_host(
                        cuda::std::span<Index>(point_instance_id.data(), point_instance_id.size())
                    )
                    .is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_NEAR(point_t[0], Real(3), Real(1e-6));
    EXPECT_EQ(point_instance_id[0], Index(1));
    EXPECT_EQ(point_primitive_id[0], Index(0));
    EXPECT_NEAR(point_u[0], Real(0.25), Real(1e-6));
    EXPECT_NEAR(point_v[0], Real(0.25), Real(1e-6));

    gwn::gwn_device_array<Real> batch_t{};
    gwn::gwn_device_array<Index> batch_primitive_id{};
    gwn::gwn_device_array<Index> batch_instance_id{};
    ASSERT_TRUE(
        gwn::tests::resize_device_arrays(1, batch_t, batch_primitive_id, batch_instance_id)
    );
    gwn::gwn_status const batch_status = gwn::gwn_compute_ray_first_hit_batch(
        scene.accessor(), d_ox.span(), d_oy.span(), d_oz.span(), d_dx.span(), d_dy.span(),
        d_dz.span(), batch_t.span(), batch_primitive_id.span(), batch_instance_id.span()
    );
    ASSERT_TRUE(batch_status.is_ok()) << gwn::tests::status_to_debug_string(batch_status);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> batch_t_host{};
    std::array<Index, 1> batch_primitive_id_host{};
    std::array<Index, 1> batch_instance_id_host{};
    ASSERT_TRUE(batch_t
                    .copy_to_host(cuda::std::span<Real>(batch_t_host.data(), batch_t_host.size()))
                    .is_ok());
    ASSERT_TRUE(batch_primitive_id
                    .copy_to_host(
                        cuda::std::span<Index>(
                            batch_primitive_id_host.data(), batch_primitive_id_host.size()
                        )
                    )
                    .is_ok());
    ASSERT_TRUE(
        batch_instance_id
            .copy_to_host(
                cuda::std::span<Index>(batch_instance_id_host.data(), batch_instance_id_host.size())
            )
            .is_ok()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_NEAR(batch_t_host[0], point_t[0], Real(1e-6));
    EXPECT_EQ(batch_primitive_id_host[0], point_primitive_id[0]);
    EXPECT_EQ(batch_instance_id_host[0], point_instance_id[0]);
}

TEST_F(CudaFixture, SceneRayFirstHitScaledInstance) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[0].transform.translation[2] = Real(4);
    instances[0].transform.scale = Real(2);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    gwn::gwn_status const build_status = gwn::gwn_scene_build_lbvh<4, Real, Index>(
        cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
            blas_table.data(), blas_table.size()
        ),
        cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
            instances.data(), instances.size()
        ),
        scene
    );
    ASSERT_TRUE(build_status.is_ok()) << gwn::tests::status_to_debug_string(build_status);

    std::array<Real, 1> const h_ox{Real(0.5)};
    std::array<Real, 1> const h_oy{Real(0.5)};
    std::array<Real, 1> const h_oz{Real(0)};
    std::array<Real, 1> const h_dx{Real(0)};
    std::array<Real, 1> const h_dy{Real(0)};
    std::array<Real, 1> const h_dz{Real(1)};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t, d_u, d_v;
    gwn::gwn_device_array<Index> d_primitive_id, d_instance_id;
    expect_copy_from_host(d_ox, cuda::std::span<Real const>(h_ox.data(), h_ox.size()));
    expect_copy_from_host(d_oy, cuda::std::span<Real const>(h_oy.data(), h_oy.size()));
    expect_copy_from_host(d_oz, cuda::std::span<Real const>(h_oz.data(), h_oz.size()));
    expect_copy_from_host(d_dx, cuda::std::span<Real const>(h_dx.data(), h_dx.size()));
    expect_copy_from_host(d_dy, cuda::std::span<Real const>(h_dy.data(), h_dy.size()));
    expect_copy_from_host(d_dz, cuda::std::span<Real const>(h_dz.data(), h_dz.size()));
    ASSERT_TRUE(gwn::tests::resize_device_arrays(1, d_t, d_u, d_v, d_primitive_id, d_instance_id));

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    gwn::gwn_status const launch_status = gwn::detail::gwn_launch_linear_kernel<k_block_size>(
        1, unified_scene_ray_functor{
               scene.accessor(),
               d_ox.span(),
               d_oy.span(),
               d_oz.span(),
               d_dx.span(),
               d_dy.span(),
               d_dz.span(),
               d_t.span(),
               d_primitive_id.span(),
               d_instance_id.span(),
               d_u.span(),
               d_v.span(),
           }
    );
    ASSERT_TRUE(launch_status.is_ok()) << gwn::tests::status_to_debug_string(launch_status);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> hit_t{};
    std::array<Real, 1> hit_u{};
    std::array<Real, 1> hit_v{};
    std::array<Index, 1> hit_primitive_id{};
    std::array<Index, 1> hit_instance_id{};
    ASSERT_TRUE(d_t.copy_to_host(cuda::std::span<Real>(hit_t.data(), hit_t.size())).is_ok());
    ASSERT_TRUE(d_u.copy_to_host(cuda::std::span<Real>(hit_u.data(), hit_u.size())).is_ok());
    ASSERT_TRUE(d_v.copy_to_host(cuda::std::span<Real>(hit_v.data(), hit_v.size())).is_ok());
    ASSERT_TRUE(
        d_primitive_id
            .copy_to_host(cuda::std::span<Index>(hit_primitive_id.data(), hit_primitive_id.size()))
            .is_ok()
    );
    ASSERT_TRUE(
        d_instance_id
            .copy_to_host(cuda::std::span<Index>(hit_instance_id.data(), hit_instance_id.size()))
            .is_ok()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_NEAR(hit_t[0], Real(4), Real(1e-6));
    EXPECT_EQ(hit_primitive_id[0], Index(0));
    EXPECT_EQ(hit_instance_id[0], Index(0));
    EXPECT_NEAR(hit_u[0], Real(0.25), Real(1e-6));
    EXPECT_NEAR(hit_v[0], Real(0.25), Real(1e-6));
}

TEST_F(CudaFixture, SceneRayFirstHitFiniteTMaxMiss) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    ASSERT_TRUE((gwn::gwn_scene_build_lbvh<4, Real, Index>(
                     cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                         blas_table.data(), blas_table.size()
                     ),
                     cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
                         instances.data(), instances.size()
                     ),
                     scene
    )
                     .is_ok()));
    ASSERT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_leaf);

    std::array<Real, 1> const h_ox{Real(0.25)};
    std::array<Real, 1> const h_oy{Real(0.25)};
    std::array<Real, 1> const h_oz{Real(-1)};
    std::array<Real, 1> const h_dx{Real(0)};
    std::array<Real, 1> const h_dy{Real(0)};
    std::array<Real, 1> const h_dz{Real(1)};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t;
    gwn::gwn_device_array<Index> d_primitive_id, d_instance_id;
    gwn::gwn_device_array<std::uint8_t> d_status;
    expect_copy_from_host(d_ox, cuda::std::span<Real const>(h_ox.data(), h_ox.size()));
    expect_copy_from_host(d_oy, cuda::std::span<Real const>(h_oy.data(), h_oy.size()));
    expect_copy_from_host(d_oz, cuda::std::span<Real const>(h_oz.data(), h_oz.size()));
    expect_copy_from_host(d_dx, cuda::std::span<Real const>(h_dx.data(), h_dx.size()));
    expect_copy_from_host(d_dy, cuda::std::span<Real const>(h_dy.data(), h_dy.size()));
    expect_copy_from_host(d_dz, cuda::std::span<Real const>(h_dz.data(), h_dz.size()));
    ASSERT_TRUE(gwn::tests::resize_device_arrays(1, d_t, d_primitive_id, d_instance_id, d_status));

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<k_block_size>(
                     1,
                     unified_scene_ray_interval_functor<64>{
                         scene.accessor(),
                         d_ox.span(),
                         d_oy.span(),
                         d_oz.span(),
                         d_dx.span(),
                         d_dy.span(),
                         d_dz.span(),
                         d_t.span(),
                         d_primitive_id.span(),
                         d_instance_id.span(),
                         d_status.span(),
                         Real(0),
                         Real(0.5),
                     }
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> hit_t{};
    std::array<Index, 1> hit_primitive_id{};
    std::array<Index, 1> hit_instance_id{};
    std::array<std::uint8_t, 1> hit_status{};
    ASSERT_TRUE(d_t.copy_to_host(cuda::std::span<Real>(hit_t.data(), hit_t.size())).is_ok());
    ASSERT_TRUE(
        d_primitive_id
            .copy_to_host(cuda::std::span<Index>(hit_primitive_id.data(), hit_primitive_id.size()))
            .is_ok()
    );
    ASSERT_TRUE(
        d_instance_id
            .copy_to_host(cuda::std::span<Index>(hit_instance_id.data(), hit_instance_id.size()))
            .is_ok()
    );
    ASSERT_TRUE(
        d_status.copy_to_host(cuda::std::span<std::uint8_t>(hit_status.data(), hit_status.size()))
            .is_ok()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_EQ(hit_status[0], static_cast<std::uint8_t>(gwn::gwn_ray_first_hit_status::k_miss));
    EXPECT_EQ(hit_t[0], Real(-1));
    EXPECT_EQ(hit_primitive_id[0], gwn::gwn_invalid_index<Index>());
    EXPECT_EQ(hit_instance_id[0], gwn::gwn_invalid_index<Index>());
}

TEST_F(CudaFixture, SceneRayFirstHitNestedOverflowPreservesPayload) {
    bool callback_called = false;
    Real hit_t = Real(-1);
    Index hit_primitive_id = gwn::gwn_invalid_index<Index>();
    Index hit_instance_id = gwn::gwn_invalid_index<Index>();
    std::uint8_t hit_status = 0;

    ASSERT_TRUE(run_scene_ray_first_hit_overflow_probe(
        callback_called, hit_t, hit_primitive_id, hit_instance_id, hit_status
    ));

    EXPECT_TRUE(callback_called);
    EXPECT_EQ(hit_status, static_cast<std::uint8_t>(gwn::gwn_ray_first_hit_status::k_overflow));
    EXPECT_NEAR(hit_t, Real(10), Real(1e-6));
    EXPECT_EQ(hit_primitive_id, Index(0));
    EXPECT_EQ(hit_instance_id, Index(0));
}

TEST_F(CudaFixture, UnifiedBlasBatchWritesInvalidInstanceId) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<Real, 1> const h_ox{Real(0.25)};
    std::array<Real, 1> const h_oy{Real(0.25)};
    std::array<Real, 1> const h_oz{Real(-1)};
    std::array<Real, 1> const h_dx{Real(0)};
    std::array<Real, 1> const h_dy{Real(0)};
    std::array<Real, 1> const h_dz{Real(1)};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t;
    gwn::gwn_device_array<Index> d_primitive_id, d_instance_id;
    expect_copy_from_host(d_ox, cuda::std::span<Real const>(h_ox.data(), h_ox.size()));
    expect_copy_from_host(d_oy, cuda::std::span<Real const>(h_oy.data(), h_oy.size()));
    expect_copy_from_host(d_oz, cuda::std::span<Real const>(h_oz.data(), h_oz.size()));
    expect_copy_from_host(d_dx, cuda::std::span<Real const>(h_dx.data(), h_dx.size()));
    expect_copy_from_host(d_dy, cuda::std::span<Real const>(h_dy.data(), h_dy.size()));
    expect_copy_from_host(d_dz, cuda::std::span<Real const>(h_dz.data(), h_dz.size()));
    ASSERT_TRUE(gwn::tests::resize_device_arrays(1, d_t, d_primitive_id, d_instance_id));

    ASSERT_TRUE((gwn::gwn_compute_ray_first_hit_batch(
                     blas.accessor(), d_ox.span(), d_oy.span(), d_oz.span(), d_dx.span(),
                     d_dy.span(), d_dz.span(), d_t.span(), d_primitive_id.span(),
                     d_instance_id.span()
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Index, 1> hit_instance_id{};
    ASSERT_TRUE(
        d_instance_id
            .copy_to_host(cuda::std::span<Index>(hit_instance_id.data(), hit_instance_id.size()))
            .is_ok()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(hit_instance_id[0], gwn::gwn_invalid_index<Index>());
}

TEST_F(CudaFixture, SceneRayFirstHitNegativeIntervalLeafRoot) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    ASSERT_TRUE((gwn::gwn_scene_build_lbvh<4, Real, Index>(
                     cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                         blas_table.data(), blas_table.size()
                     ),
                     cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
                         instances.data(), instances.size()
                     ),
                     scene
    )
                     .is_ok()));
    ASSERT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_leaf);

    std::array<Real, 1> const h_ox{Real(0.25)};
    std::array<Real, 1> const h_oy{Real(0.25)};
    std::array<Real, 1> const h_oz{Real(1)};
    std::array<Real, 1> const h_dx{Real(0)};
    std::array<Real, 1> const h_dy{Real(0)};
    std::array<Real, 1> const h_dz{Real(1)};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t;
    gwn::gwn_device_array<Index> d_primitive_id, d_instance_id;
    gwn::gwn_device_array<std::uint8_t> d_status;
    expect_copy_from_host(d_ox, cuda::std::span<Real const>(h_ox.data(), h_ox.size()));
    expect_copy_from_host(d_oy, cuda::std::span<Real const>(h_oy.data(), h_oy.size()));
    expect_copy_from_host(d_oz, cuda::std::span<Real const>(h_oz.data(), h_oz.size()));
    expect_copy_from_host(d_dx, cuda::std::span<Real const>(h_dx.data(), h_dx.size()));
    expect_copy_from_host(d_dy, cuda::std::span<Real const>(h_dy.data(), h_dy.size()));
    expect_copy_from_host(d_dz, cuda::std::span<Real const>(h_dz.data(), h_dz.size()));
    ASSERT_TRUE(gwn::tests::resize_device_arrays(1, d_t, d_primitive_id, d_instance_id, d_status));

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<k_block_size>(
                     1,
                     unified_scene_ray_interval_functor<64>{
                         scene.accessor(),
                         d_ox.span(),
                         d_oy.span(),
                         d_oz.span(),
                         d_dx.span(),
                         d_dy.span(),
                         d_dz.span(),
                         d_t.span(),
                         d_primitive_id.span(),
                         d_instance_id.span(),
                         d_status.span(),
                         Real(-2),
                         Real(-0.5),
                     }
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> hit_t{};
    std::array<Index, 1> hit_primitive_id{};
    std::array<Index, 1> hit_instance_id{};
    std::array<std::uint8_t, 1> hit_status{};
    ASSERT_TRUE(d_t.copy_to_host(cuda::std::span<Real>(hit_t.data(), hit_t.size())).is_ok());
    ASSERT_TRUE(
        d_primitive_id
            .copy_to_host(cuda::std::span<Index>(hit_primitive_id.data(), hit_primitive_id.size()))
            .is_ok()
    );
    ASSERT_TRUE(
        d_instance_id
            .copy_to_host(cuda::std::span<Index>(hit_instance_id.data(), hit_instance_id.size()))
            .is_ok()
    );
    ASSERT_TRUE(
        d_status.copy_to_host(cuda::std::span<std::uint8_t>(hit_status.data(), hit_status.size()))
            .is_ok()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_EQ(hit_status[0], static_cast<std::uint8_t>(gwn::gwn_ray_first_hit_status::k_hit));
    EXPECT_NEAR(hit_t[0], Real(-1), Real(1e-6));
    EXPECT_EQ(hit_primitive_id[0], Index(0));
    EXPECT_EQ(hit_instance_id[0], Index(0));

    gwn::gwn_device_array<Real> batch_t{};
    gwn::gwn_device_array<Index> batch_primitive_id{};
    gwn::gwn_device_array<Index> batch_instance_id{};
    ASSERT_TRUE(
        gwn::tests::resize_device_arrays(1, batch_t, batch_primitive_id, batch_instance_id)
    );
    ASSERT_TRUE((gwn::gwn_compute_ray_first_hit_batch(
                     scene.accessor(), d_ox.span(), d_oy.span(), d_oz.span(), d_dx.span(),
                     d_dy.span(), d_dz.span(), batch_t.span(), batch_primitive_id.span(),
                     batch_instance_id.span(), Real(-2), Real(-0.5)
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> batch_t_host{};
    std::array<Index, 1> batch_primitive_id_host{};
    std::array<Index, 1> batch_instance_id_host{};
    ASSERT_TRUE(batch_t
                    .copy_to_host(cuda::std::span<Real>(batch_t_host.data(), batch_t_host.size()))
                    .is_ok());
    ASSERT_TRUE(batch_primitive_id
                    .copy_to_host(
                        cuda::std::span<Index>(
                            batch_primitive_id_host.data(), batch_primitive_id_host.size()
                        )
                    )
                    .is_ok());
    ASSERT_TRUE(
        batch_instance_id
            .copy_to_host(
                cuda::std::span<Index>(batch_instance_id_host.data(), batch_instance_id_host.size())
            )
            .is_ok()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_NEAR(batch_t_host[0], Real(-1), Real(1e-6));
    EXPECT_EQ(batch_primitive_id_host[0], Index(0));
    EXPECT_EQ(batch_instance_id_host[0], Index(0));
}

TEST_F(CudaFixture, SceneRayFirstHitNegativeIntervalInternalRoot) {
    gwn::tests::SingleTriangleMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 2> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].blas_index = Index(0);
    instances[1].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].transform.translation[2] = Real(3);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    ASSERT_TRUE((gwn::gwn_scene_build_lbvh<4, Real, Index>(
                     cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                         blas_table.data(), blas_table.size()
                     ),
                     cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
                         instances.data(), instances.size()
                     ),
                     scene
    )
                     .is_ok()));
    ASSERT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_internal);

    std::array<Real, 1> const h_ox{Real(0.25)};
    std::array<Real, 1> const h_oy{Real(0.25)};
    std::array<Real, 1> const h_oz{Real(1)};
    std::array<Real, 1> const h_dx{Real(0)};
    std::array<Real, 1> const h_dy{Real(0)};
    std::array<Real, 1> const h_dz{Real(1)};

    gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz, d_t;
    gwn::gwn_device_array<Index> d_primitive_id, d_instance_id;
    gwn::gwn_device_array<std::uint8_t> d_status;
    expect_copy_from_host(d_ox, cuda::std::span<Real const>(h_ox.data(), h_ox.size()));
    expect_copy_from_host(d_oy, cuda::std::span<Real const>(h_oy.data(), h_oy.size()));
    expect_copy_from_host(d_oz, cuda::std::span<Real const>(h_oz.data(), h_oz.size()));
    expect_copy_from_host(d_dx, cuda::std::span<Real const>(h_dx.data(), h_dx.size()));
    expect_copy_from_host(d_dy, cuda::std::span<Real const>(h_dy.data(), h_dy.size()));
    expect_copy_from_host(d_dz, cuda::std::span<Real const>(h_dz.data(), h_dz.size()));
    ASSERT_TRUE(gwn::tests::resize_device_arrays(1, d_t, d_primitive_id, d_instance_id, d_status));

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<k_block_size>(
                     1,
                     unified_scene_ray_interval_functor<64>{
                         scene.accessor(),
                         d_ox.span(),
                         d_oy.span(),
                         d_oz.span(),
                         d_dx.span(),
                         d_dy.span(),
                         d_dz.span(),
                         d_t.span(),
                         d_primitive_id.span(),
                         d_instance_id.span(),
                         d_status.span(),
                         Real(-2),
                         Real(-0.5),
                     }
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> hit_t{};
    std::array<Index, 1> hit_primitive_id{};
    std::array<Index, 1> hit_instance_id{};
    std::array<std::uint8_t, 1> hit_status{};
    ASSERT_TRUE(d_t.copy_to_host(cuda::std::span<Real>(hit_t.data(), hit_t.size())).is_ok());
    ASSERT_TRUE(
        d_primitive_id
            .copy_to_host(cuda::std::span<Index>(hit_primitive_id.data(), hit_primitive_id.size()))
            .is_ok()
    );
    ASSERT_TRUE(
        d_instance_id
            .copy_to_host(cuda::std::span<Index>(hit_instance_id.data(), hit_instance_id.size()))
            .is_ok()
    );
    ASSERT_TRUE(
        d_status.copy_to_host(cuda::std::span<std::uint8_t>(hit_status.data(), hit_status.size()))
            .is_ok()
    );
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_EQ(hit_status[0], static_cast<std::uint8_t>(gwn::gwn_ray_first_hit_status::k_hit));
    EXPECT_NEAR(hit_t[0], Real(-1), Real(1e-6));
    EXPECT_EQ(hit_primitive_id[0], Index(0));
    EXPECT_EQ(hit_instance_id[0], Index(0));
}

TEST_F(CudaFixture, SceneWindingExactSingleInstance) {
    gwn::tests::CubeMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[0].transform.translation[0] = Real(3);
    instances[0].transform.translation[1] = Real(-2);
    instances[0].transform.translation[2] = Real(1);
    instances[0].transform.scale = Real(1.5);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    ASSERT_TRUE((gwn::gwn_scene_build_lbvh<4, Real, Index>(
                     cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                         blas_table.data(), blas_table.size()
                     ),
                     cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
                         instances.data(), instances.size()
                     ),
                     scene
    )
                     .is_ok()));
    ASSERT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_leaf);

    std::array<std::array<Real, 3>, 2> const local_queries{{
        {{Real(0), Real(0), Real(0)}},
        {{Real(3), Real(0), Real(0)}},
    }};
    std::array<Real, 2> scene_qx{};
    std::array<Real, 2> scene_qy{};
    std::array<Real, 2> scene_qz{};
    std::array<Real, 2> blas_qx{};
    std::array<Real, 2> blas_qy{};
    std::array<Real, 2> blas_qz{};
    for (std::size_t i = 0; i < local_queries.size(); ++i) {
        auto const world = apply_point_host(instances[0].transform, local_queries[i]);
        scene_qx[i] = world[0];
        scene_qy[i] = world[1];
        scene_qz[i] = world[2];
        blas_qx[i] = local_queries[i][0];
        blas_qy[i] = local_queries[i][1];
        blas_qz[i] = local_queries[i][2];
    }

    gwn::gwn_device_array<Real> d_scene_qx, d_scene_qy, d_scene_qz;
    gwn::gwn_device_array<Real> d_blas_qx, d_blas_qy, d_blas_qz;
    gwn::gwn_device_array<Real> d_scene_point, d_blas_point, d_scene_batch, d_blas_batch;
    expect_copy_from_host(
        d_scene_qx, cuda::std::span<Real const>(scene_qx.data(), scene_qx.size())
    );
    expect_copy_from_host(
        d_scene_qy, cuda::std::span<Real const>(scene_qy.data(), scene_qy.size())
    );
    expect_copy_from_host(
        d_scene_qz, cuda::std::span<Real const>(scene_qz.data(), scene_qz.size())
    );
    expect_copy_from_host(d_blas_qx, cuda::std::span<Real const>(blas_qx.data(), blas_qx.size()));
    expect_copy_from_host(d_blas_qy, cuda::std::span<Real const>(blas_qy.data(), blas_qy.size()));
    expect_copy_from_host(d_blas_qz, cuda::std::span<Real const>(blas_qz.data(), blas_qz.size()));
    ASSERT_TRUE(
        gwn::tests::resize_device_arrays(
            scene_qx.size(), d_scene_point, d_blas_point, d_scene_batch, d_blas_batch
        )
    );

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<k_block_size>(
                     scene_qx.size(),
                     unified_scene_and_blas_winding_functor{
                         scene.accessor(),
                         blas.accessor(),
                         d_scene_qx.span(),
                         d_scene_qy.span(),
                         d_scene_qz.span(),
                         d_blas_qx.span(),
                         d_blas_qy.span(),
                         d_blas_qz.span(),
                         d_scene_point.span(),
                         d_blas_point.span(),
                     }
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch(
                     scene.accessor(), d_scene_qx.span(), d_scene_qy.span(), d_scene_qz.span(),
                     d_scene_batch.span()
    )
                     .is_ok()));
    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch(
                     blas.accessor(), d_blas_qx.span(), d_blas_qy.span(), d_blas_qz.span(),
                     d_blas_batch.span()
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 2> scene_point{};
    std::array<Real, 2> blas_point{};
    std::array<Real, 2> scene_batch{};
    std::array<Real, 2> blas_batch{};
    ASSERT_TRUE(d_scene_point
                    .copy_to_host(cuda::std::span<Real>(scene_point.data(), scene_point.size()))
                    .is_ok());
    ASSERT_TRUE(d_blas_point
                    .copy_to_host(cuda::std::span<Real>(blas_point.data(), blas_point.size()))
                    .is_ok());
    ASSERT_TRUE(d_scene_batch
                    .copy_to_host(cuda::std::span<Real>(scene_batch.data(), scene_batch.size()))
                    .is_ok());
    ASSERT_TRUE(d_blas_batch
                    .copy_to_host(cuda::std::span<Real>(blas_batch.data(), blas_batch.size()))
                    .is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_NEAR(scene_point[0], blas_point[0], Real(1e-4));
    EXPECT_NEAR(scene_batch[0], blas_batch[0], Real(1e-4));
    EXPECT_NEAR(scene_point[0], Real(1), Real(1e-4));
    EXPECT_NEAR(scene_batch[0], Real(1), Real(1e-4));
    EXPECT_NEAR(scene_point[1], blas_point[1], Real(1e-4));
    EXPECT_NEAR(scene_batch[1], blas_batch[1], Real(1e-4));
    EXPECT_NEAR(scene_point[1], Real(0), Real(1e-4));
    EXPECT_NEAR(scene_batch[1], Real(0), Real(1e-4));
}

TEST_F(CudaFixture, SceneWindingExactMultiInstance) {
    gwn::tests::CubeMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 2> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[1].blas_index = Index(0);
    instances[1].transform = gwn::gwn_similarity_transform<Real>::identity();

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    ASSERT_TRUE((gwn::gwn_scene_build_lbvh<4, Real, Index>(
                     cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                         blas_table.data(), blas_table.size()
                     ),
                     cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
                         instances.data(), instances.size()
                     ),
                     scene
    )
                     .is_ok()));
    ASSERT_EQ(scene.accessor().ias_topology.root_kind, gwn::gwn_bvh_child_kind::k_internal);

    std::array<Real, 2> const h_qx{Real(0), Real(3)};
    std::array<Real, 2> const h_qy{Real(0), Real(0)};
    std::array<Real, 2> const h_qz{Real(0), Real(0)};

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_point, d_batch;
    expect_copy_from_host(d_qx, cuda::std::span<Real const>(h_qx.data(), h_qx.size()));
    expect_copy_from_host(d_qy, cuda::std::span<Real const>(h_qy.data(), h_qy.size()));
    expect_copy_from_host(d_qz, cuda::std::span<Real const>(h_qz.data(), h_qz.size()));
    ASSERT_TRUE(gwn::tests::resize_device_arrays(h_qx.size(), d_point, d_batch));

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<k_block_size>(
                     h_qx.size(),
                     unified_scene_winding_functor{
                         scene.accessor(),
                         d_qx.span(),
                         d_qy.span(),
                         d_qz.span(),
                         d_point.span(),
                     }
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch(
                     scene.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_batch.span()
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 2> point{};
    std::array<Real, 2> batch{};
    ASSERT_TRUE(d_point.copy_to_host(cuda::std::span<Real>(point.data(), point.size())).is_ok());
    ASSERT_TRUE(d_batch.copy_to_host(cuda::std::span<Real>(batch.data(), batch.size())).is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_NEAR(point[0], Real(2), Real(1e-4));
    EXPECT_NEAR(batch[0], Real(2), Real(1e-4));
    EXPECT_NEAR(point[1], Real(0), Real(1e-4));
    EXPECT_NEAR(batch[1], Real(0), Real(1e-4));
}

TEST_F(CudaFixture, SceneWindingExactDoesNotRequireBlasAabb) {
    gwn::tests::CubeMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    ASSERT_TRUE((gwn::gwn_scene_build_lbvh<4, Real, Index>(
                     cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                         blas_table.data(), blas_table.size()
                     ),
                     cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
                         instances.data(), instances.size()
                     ),
                     scene
    )
                     .is_ok()));

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> overridden_blas_table{blas.accessor()};
    overridden_blas_table[0].aabb = {};
    gwn::gwn_device_array<gwn::gwn_blas_accessor<4, Real, Index>> d_overridden_blas_table{};
    expect_copy_from_host(
        d_overridden_blas_table, cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                                     overridden_blas_table.data(), overridden_blas_table.size()
                                 )
    );

    auto const built_scene = scene.accessor();
    auto const overridden_scene =
        gwn::gwn_scene_accessor<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>>{
            built_scene.ias_topology,
            built_scene.ias_aabb,
            d_overridden_blas_table.span(),
            built_scene.instances,
        };

    std::array<Real, 1> const h_qx{Real(0)};
    std::array<Real, 1> const h_qy{Real(0)};
    std::array<Real, 1> const h_qz{Real(0)};

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_scene_point, d_blas_point, d_scene_batch,
        d_blas_batch;
    expect_copy_from_host(d_qx, cuda::std::span<Real const>(h_qx.data(), h_qx.size()));
    expect_copy_from_host(d_qy, cuda::std::span<Real const>(h_qy.data(), h_qy.size()));
    expect_copy_from_host(d_qz, cuda::std::span<Real const>(h_qz.data(), h_qz.size()));
    ASSERT_TRUE(
        gwn::tests::resize_device_arrays(
            h_qx.size(), d_scene_point, d_blas_point, d_scene_batch, d_blas_batch
        )
    );

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<k_block_size>(
                     h_qx.size(),
                     unified_scene_and_blas_winding_functor{
                         overridden_scene,
                         blas.accessor(),
                         d_qx.span(),
                         d_qy.span(),
                         d_qz.span(),
                         d_qx.span(),
                         d_qy.span(),
                         d_qz.span(),
                         d_scene_point.span(),
                         d_blas_point.span(),
                     }
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch(
                     overridden_scene, d_qx.span(), d_qy.span(), d_qz.span(), d_scene_batch.span()
    )
                     .is_ok()));
    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch(
                     blas.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_blas_batch.span()
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> scene_point{};
    std::array<Real, 1> blas_point{};
    std::array<Real, 1> scene_batch{};
    std::array<Real, 1> blas_batch{};
    ASSERT_TRUE(d_scene_point
                    .copy_to_host(cuda::std::span<Real>(scene_point.data(), scene_point.size()))
                    .is_ok());
    ASSERT_TRUE(d_blas_point
                    .copy_to_host(cuda::std::span<Real>(blas_point.data(), blas_point.size()))
                    .is_ok());
    ASSERT_TRUE(d_scene_batch
                    .copy_to_host(cuda::std::span<Real>(scene_batch.data(), scene_batch.size()))
                    .is_ok());
    ASSERT_TRUE(d_blas_batch
                    .copy_to_host(cuda::std::span<Real>(blas_batch.data(), blas_batch.size()))
                    .is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_NEAR(scene_point[0], blas_point[0], Real(1e-4));
    EXPECT_NEAR(scene_batch[0], blas_batch[0], Real(1e-4));
    EXPECT_NEAR(scene_point[0], Real(1), Real(1e-4));
    EXPECT_NEAR(scene_batch[0], Real(1), Real(1e-4));
}

TEST_F(CudaFixture, SceneWindingExactNegativeScaleFlipsSign) {
    gwn::tests::CubeMesh mesh{};
    TestBlasStorage const blas = build_test_blas(
        std::vector<Real>(mesh.vx.begin(), mesh.vx.end()),
        std::vector<Real>(mesh.vy.begin(), mesh.vy.end()),
        std::vector<Real>(mesh.vz.begin(), mesh.vz.end()),
        std::vector<Index>(mesh.i0.begin(), mesh.i0.end()),
        std::vector<Index>(mesh.i1.begin(), mesh.i1.end()),
        std::vector<Index>(mesh.i2.begin(), mesh.i2.end())
    );

    std::array<gwn::gwn_blas_accessor<4, Real, Index>, 1> const blas_table{blas.accessor()};
    std::array<gwn::gwn_instance_record<Real, Index>, 1> instances{};
    instances[0].blas_index = Index(0);
    instances[0].transform = gwn::gwn_similarity_transform<Real>::identity();
    instances[0].transform.scale = Real(-1);

    gwn::gwn_scene_object<4, Real, Index, gwn::gwn_blas_accessor<4, Real, Index>> scene{};
    ASSERT_TRUE((gwn::gwn_scene_build_lbvh<4, Real, Index>(
                     cuda::std::span<gwn::gwn_blas_accessor<4, Real, Index> const>(
                         blas_table.data(), blas_table.size()
                     ),
                     cuda::std::span<gwn::gwn_instance_record<Real, Index> const>(
                         instances.data(), instances.size()
                     ),
                     scene
    )
                     .is_ok()));

    std::array<Real, 1> const h_qx{Real(0)};
    std::array<Real, 1> const h_qy{Real(0)};
    std::array<Real, 1> const h_qz{Real(0)};

    gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_scene_point, d_blas_point, d_scene_batch,
        d_blas_batch;
    expect_copy_from_host(d_qx, cuda::std::span<Real const>(h_qx.data(), h_qx.size()));
    expect_copy_from_host(d_qy, cuda::std::span<Real const>(h_qy.data(), h_qy.size()));
    expect_copy_from_host(d_qz, cuda::std::span<Real const>(h_qz.data(), h_qz.size()));
    ASSERT_TRUE(
        gwn::tests::resize_device_arrays(
            h_qx.size(), d_scene_point, d_blas_point, d_scene_batch, d_blas_batch
        )
    );

    constexpr int k_block_size = gwn::detail::k_gwn_default_block_size;
    ASSERT_TRUE((gwn::detail::gwn_launch_linear_kernel<k_block_size>(
                     h_qx.size(),
                     unified_scene_and_blas_winding_functor{
                         scene.accessor(),
                         blas.accessor(),
                         d_qx.span(),
                         d_qy.span(),
                         d_qz.span(),
                         d_qx.span(),
                         d_qy.span(),
                         d_qz.span(),
                         d_scene_point.span(),
                         d_blas_point.span(),
                     }
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch(
                     scene.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_scene_batch.span()
    )
                     .is_ok()));
    ASSERT_TRUE((gwn::gwn_compute_winding_number_batch(
                     blas.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_blas_batch.span()
    )
                     .is_ok()));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<Real, 1> scene_point{};
    std::array<Real, 1> blas_point{};
    std::array<Real, 1> scene_batch{};
    std::array<Real, 1> blas_batch{};
    ASSERT_TRUE(d_scene_point
                    .copy_to_host(cuda::std::span<Real>(scene_point.data(), scene_point.size()))
                    .is_ok());
    ASSERT_TRUE(d_blas_point
                    .copy_to_host(cuda::std::span<Real>(blas_point.data(), blas_point.size()))
                    .is_ok());
    ASSERT_TRUE(d_scene_batch
                    .copy_to_host(cuda::std::span<Real>(scene_batch.data(), scene_batch.size()))
                    .is_ok());
    ASSERT_TRUE(d_blas_batch
                    .copy_to_host(cuda::std::span<Real>(blas_batch.data(), blas_batch.size()))
                    .is_ok());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_NEAR(scene_point[0], -blas_point[0], Real(1e-4));
    EXPECT_NEAR(scene_batch[0], -blas_batch[0], Real(1e-4));
    EXPECT_NEAR(scene_point[0], Real(-1), Real(1e-4));
    EXPECT_NEAR(scene_batch[0], Real(-1), Real(1e-4));
}
