#pragma once

#include "../gwn_bvh.cuh"
#include "../gwn_utils.cuh"

namespace gwn {

template <gwn_real_type Real> struct gwn_similarity_transform {
    Real rotation[3][3]{
        {Real(1), Real(0), Real(0)}, {Real(0), Real(1), Real(0)}, {Real(0), Real(0), Real(1)}
    };
    Real translation[3]{};
    Real scale{Real(1)};

    __host__ __device__ constexpr void
    apply_point(Real px, Real py, Real pz, Real &ox, Real &oy, Real &oz) const noexcept {
        assert_scale_positive();
        Real rx = Real(0);
        Real ry = Real(0);
        Real rz = Real(0);
        apply_rotation(px, py, pz, rx, ry, rz);
        ox = scale * rx + translation[0];
        oy = scale * ry + translation[1];
        oz = scale * rz + translation[2];
    }

    __host__ __device__ constexpr void
    apply_direction(Real dx, Real dy, Real dz, Real &ox, Real &oy, Real &oz) const noexcept {
        assert_scale_positive();
        Real rx = Real(0);
        Real ry = Real(0);
        Real rz = Real(0);
        apply_rotation(dx, dy, dz, rx, ry, rz);
        ox = scale * rx;
        oy = scale * ry;
        oz = scale * rz;
    }

    __host__ __device__ constexpr void
    inverse_apply_point(Real px, Real py, Real pz, Real &ox, Real &oy, Real &oz) const noexcept {
        assert_scale_positive();
        Real rx = Real(0);
        Real ry = Real(0);
        Real rz = Real(0);
        inverse_apply_rotation(
            px - translation[0], py - translation[1], pz - translation[2], rx, ry, rz
        );
        ox = rx / scale;
        oy = ry / scale;
        oz = rz / scale;
    }

    __host__ __device__ constexpr void inverse_apply_direction(
        Real dx, Real dy, Real dz, Real &ox, Real &oy, Real &oz
    ) const noexcept {
        assert_scale_positive();
        Real rx = Real(0);
        Real ry = Real(0);
        Real rz = Real(0);
        inverse_apply_rotation(dx, dy, dz, rx, ry, rz);
        ox = rx / scale;
        oy = ry / scale;
        oz = rz / scale;
    }

    [[nodiscard]] __host__ __device__ constexpr gwn_aabb<Real>
    transform_aabb(gwn_aabb<Real> const &local) const noexcept {
        assert_scale_positive();
        Real const local_center_x = (local.min_x + local.max_x) * Real(0.5);
        Real const local_center_y = (local.min_y + local.max_y) * Real(0.5);
        Real const local_center_z = (local.min_z + local.max_z) * Real(0.5);
        Real const local_extent_x = (local.max_x - local.min_x) * Real(0.5);
        Real const local_extent_y = (local.max_y - local.min_y) * Real(0.5);
        Real const local_extent_z = (local.max_z - local.min_z) * Real(0.5);

        Real world_center_x = Real(0);
        Real world_center_y = Real(0);
        Real world_center_z = Real(0);
        apply_point(
            local_center_x, local_center_y, local_center_z, world_center_x, world_center_y,
            world_center_z
        );

        Real world_extent_x = scale * (abs_value(rotation[0][0]) * local_extent_x +
                                       abs_value(rotation[0][1]) * local_extent_y +
                                       abs_value(rotation[0][2]) * local_extent_z);
        Real world_extent_y = scale * (abs_value(rotation[1][0]) * local_extent_x +
                                       abs_value(rotation[1][1]) * local_extent_y +
                                       abs_value(rotation[1][2]) * local_extent_z);
        Real world_extent_z = scale * (abs_value(rotation[2][0]) * local_extent_x +
                                       abs_value(rotation[2][1]) * local_extent_y +
                                       abs_value(rotation[2][2]) * local_extent_z);

        return gwn_aabb<Real>{
            world_center_x - world_extent_x, world_center_y - world_extent_y,
            world_center_z - world_extent_z, world_center_x + world_extent_x,
            world_center_y + world_extent_y, world_center_z + world_extent_z,
        };
    }

    [[nodiscard]] __host__ __device__ static constexpr gwn_similarity_transform
    identity() noexcept {
        return gwn_similarity_transform{};
    }

private:
    [[nodiscard]] __host__ __device__ static constexpr Real abs_value(Real const value) noexcept {
        return value < Real(0) ? -value : value;
    }

    __host__ __device__ constexpr void assert_scale_positive() const noexcept {
        GWN_ASSERT(scale > Real(0), "gwn_similarity_transform scale must be positive.");
    }

    __host__ __device__ constexpr void
    apply_rotation(Real px, Real py, Real pz, Real &ox, Real &oy, Real &oz) const noexcept {
        ox = rotation[0][0] * px + rotation[0][1] * py + rotation[0][2] * pz;
        oy = rotation[1][0] * px + rotation[1][1] * py + rotation[1][2] * pz;
        oz = rotation[2][0] * px + rotation[2][1] * py + rotation[2][2] * pz;
    }

    __host__ __device__ constexpr void
    inverse_apply_rotation(Real px, Real py, Real pz, Real &ox, Real &oy, Real &oz) const noexcept {
        ox = rotation[0][0] * px + rotation[1][0] * py + rotation[2][0] * pz;
        oy = rotation[0][1] * px + rotation[1][1] * py + rotation[2][1] * pz;
        oz = rotation[0][2] * px + rotation[1][2] * py + rotation[2][2] * pz;
    }
};

} // namespace gwn
