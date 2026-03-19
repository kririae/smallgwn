#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
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
