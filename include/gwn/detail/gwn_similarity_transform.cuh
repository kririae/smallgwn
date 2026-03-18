#pragma once

#include "../gwn_bvh.cuh"
#include "../gwn_utils.cuh"

namespace gwn {

template <gwn_real_type Real> struct gwn_similarity_transform {
    Real rotation[3][3]{};
    Real translation[3]{};
    Real scale{Real(1)};

    __host__ __device__ constexpr void
    apply_point(Real const local[3], Real world[3]) const noexcept {
        Real rotated[3]{};
        apply_rotation(local, rotated);
        world[0] = scale * rotated[0] + translation[0];
        world[1] = scale * rotated[1] + translation[1];
        world[2] = scale * rotated[2] + translation[2];
    }

    __host__ __device__ constexpr void
    apply_direction(Real const local[3], Real world[3]) const noexcept {
        Real rotated[3]{};
        apply_rotation(local, rotated);
        world[0] = scale * rotated[0];
        world[1] = scale * rotated[1];
        world[2] = scale * rotated[2];
    }

    __host__ __device__ constexpr void
    inverse_apply_point(Real const world[3], Real local[3]) const noexcept {
        Real translated[3]{
            world[0] - translation[0],
            world[1] - translation[1],
            world[2] - translation[2],
        };
        Real rotated[3]{};
        inverse_apply_rotation(translated, rotated);
        local[0] = rotated[0] / scale;
        local[1] = rotated[1] / scale;
        local[2] = rotated[2] / scale;
    }

    __host__ __device__ constexpr void
    inverse_apply_direction(Real const world[3], Real local[3]) const noexcept {
        Real rotated[3]{};
        inverse_apply_rotation(world, rotated);
        local[0] = rotated[0] / scale;
        local[1] = rotated[1] / scale;
        local[2] = rotated[2] / scale;
    }

    [[nodiscard]] __host__ __device__ constexpr gwn_aabb<Real>
    transform_aabb(gwn_aabb<Real> const &local) const noexcept {
        Real const local_center[3]{
            (local.min_x + local.max_x) * Real(0.5),
            (local.min_y + local.max_y) * Real(0.5),
            (local.min_z + local.max_z) * Real(0.5),
        };
        Real const local_extent[3]{
            (local.max_x - local.min_x) * Real(0.5),
            (local.max_y - local.min_y) * Real(0.5),
            (local.max_z - local.min_z) * Real(0.5),
        };

        Real world_center[3]{};
        apply_point(local_center, world_center);

        Real const abs_scale = abs_value(scale);
        Real world_extent[3]{};
        for (int axis = 0; axis < 3; ++axis) {
            Real extent = Real(0);
            for (int component = 0; component < 3; ++component)
                extent += abs_value(rotation[axis][component]) * local_extent[component];
            world_extent[axis] = abs_scale * extent;
        }

        return gwn_aabb<Real>{
            world_center[0] - world_extent[0], world_center[1] - world_extent[1],
            world_center[2] - world_extent[2], world_center[0] + world_extent[0],
            world_center[1] + world_extent[1], world_center[2] + world_extent[2],
        };
    }

    [[nodiscard]] __host__ __device__ static constexpr gwn_similarity_transform
    identity() noexcept {
        gwn_similarity_transform transform{};
        transform.rotation[0][0] = Real(1);
        transform.rotation[1][1] = Real(1);
        transform.rotation[2][2] = Real(1);
        return transform;
    }

private:
    [[nodiscard]] __host__ __device__ static constexpr Real abs_value(Real const value) noexcept {
        return value < Real(0) ? -value : value;
    }

    __host__ __device__ constexpr void
    apply_rotation(Real const local[3], Real rotated[3]) const noexcept {
        rotated[0] =
            rotation[0][0] * local[0] + rotation[0][1] * local[1] + rotation[0][2] * local[2];
        rotated[1] =
            rotation[1][0] * local[0] + rotation[1][1] * local[1] + rotation[1][2] * local[2];
        rotated[2] =
            rotation[2][0] * local[0] + rotation[2][1] * local[1] + rotation[2][2] * local[2];
    }

    __host__ __device__ constexpr void
    inverse_apply_rotation(Real const world[3], Real rotated[3]) const noexcept {
        rotated[0] =
            rotation[0][0] * world[0] + rotation[1][0] * world[1] + rotation[2][0] * world[2];
        rotated[1] =
            rotation[0][1] * world[0] + rotation[1][1] * world[1] + rotation[2][1] * world[2];
        rotated[2] =
            rotation[0][2] * world[0] + rotation[1][2] * world[1] + rotation[2][2] * world[2];
    }
};

} // namespace gwn
