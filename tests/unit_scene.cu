#include <algorithm>
#include <array>
#include <limits>

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
    transform.apply_point(point.data(), result.data());
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

} // namespace

TEST_F(CudaFixture, SimilarityTransformIdentity) {
    auto const transform = gwn::gwn_similarity_transform<Real>::identity();

    std::array<Real, 3> const point{{Real(1.5), Real(-2.0), Real(0.25)}};
    std::array<Real, 3> const direction{{Real(-3.0), Real(4.0), Real(2.5)}};
    std::array<Real, 3> out_point{};
    std::array<Real, 3> out_direction{};

    transform.apply_point(point.data(), out_point.data());
    transform.apply_direction(direction.data(), out_direction.data());

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

    transform.apply_point(local_point.data(), world_point.data());
    transform.apply_direction(local_direction.data(), world_direction.data());
    transform.inverse_apply_point(world_point.data(), recovered_point.data());
    transform.inverse_apply_direction(world_direction.data(), recovered_direction.data());

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
