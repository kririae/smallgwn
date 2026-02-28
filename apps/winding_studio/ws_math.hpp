#pragma once

#include "ws_types.hpp"

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace winding_studio::app {

[[nodiscard]] inline Mat4 mat4_identity() {
    return Mat4(1.0f);
}

[[nodiscard]] inline Mat4 mat4_mul(Mat4 const &a, Mat4 const &b) {
    return a * b;
}

[[nodiscard]] inline Vec3 vec3_sub(Vec3 const &a, Vec3 const &b) {
    return a - b;
}

[[nodiscard]] inline float vec3_dot(Vec3 const &a, Vec3 const &b) {
    return glm::dot(a, b);
}

[[nodiscard]] inline Vec3 vec3_cross(Vec3 const &a, Vec3 const &b) {
    return glm::cross(a, b);
}

[[nodiscard]] inline Vec3 vec3_normalize(Vec3 const &a) {
    float const len2 = glm::dot(a, a);
    if (!(len2 > 0.0f))
        return Vec3{0.0f, 0.0f, 1.0f};
    return glm::normalize(a);
}

[[nodiscard]] inline Mat4 mat4_rotate_x(float const radians) {
    return glm::rotate(mat4_identity(), radians, Vec3{1.0f, 0.0f, 0.0f});
}

[[nodiscard]] inline Mat4 mat4_rotate_y(float const radians) {
    return glm::rotate(mat4_identity(), radians, Vec3{0.0f, 1.0f, 0.0f});
}

[[nodiscard]] inline Mat4 mat4_perspective(
    float const fovy_radians, float const aspect, float const z_near, float const z_far
) {
    return glm::perspective(fovy_radians, aspect, z_near, z_far);
}

[[nodiscard]] inline Mat4 mat4_look_at(Vec3 const &eye, Vec3 const &target, Vec3 const &up_hint) {
    return glm::lookAt(eye, target, up_hint);
}

[[nodiscard]] inline CameraBasis build_camera_basis(AppState const &state) {
    CameraBasis basis{};
    basis.eye = Vec3{
        state.camera_target.x + std::sin(state.yaw) * state.camera_radius,
        state.camera_target.y + std::sin(state.pitch) * state.camera_radius + 0.3f,
        state.camera_target.z + std::cos(state.yaw) * state.camera_radius,
    };
    basis.target = state.camera_target;
    basis.up = Vec3{0.0f, 1.0f, 0.0f};
    basis.forward = vec3_normalize(vec3_sub(basis.target, basis.eye));
    basis.right = vec3_normalize(vec3_cross(basis.forward, basis.up));
    basis.ortho_up = vec3_cross(basis.right, basis.forward);
    return basis;
}

} // namespace winding_studio::app
