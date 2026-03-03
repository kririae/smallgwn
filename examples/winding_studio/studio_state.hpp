#pragma once

#include "studio_domain.hpp"
#include "voxel_grid_spec.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace winding_studio::app {

/**
 * @brief Shared PI constant used by camera and projection helpers.
 */
inline constexpr float k_pi = 3.14159265358979323846f;
inline constexpr float k_default_camera_yaw = 0.7853981634f;        // 45 deg
inline constexpr float k_default_camera_pitch = 0.6154797087f;      // 35.264 deg
inline constexpr float k_default_camera_radius = 3.1f;
inline constexpr float k_default_camera_fovy_radians = 0.6911503838f; // 39.6 deg (Blender-like)

using Vec3 = glm::vec3;
using Mat4 = glm::mat4;

/**
 * @brief Per-frame UI output consumed by app orchestration.
 */
struct UiLayoutResult {
    bool harnack_params_changed{false};
    bool voxel_params_changed{false};
    int activate_mesh_index{-1};
    int remove_mesh_index{-1};
    std::string mesh_file_to_add{};
    bool request_harnack_refresh{false};
    bool request_voxel_refresh{false};
    float viewport_pos_x{0.0f};
    float viewport_pos_y{0.0f};
    float viewport_w{1.0f};
    float viewport_h{1.0f};
};

/**
 * @brief Integer framebuffer region for GL rendering.
 */
struct FramebufferRect {
    int x{0};
    int y{0};
    int w{0};
    int h{0};
};

/**
 * @brief Derived camera basis vectors used by UI and rendering.
 */
struct CameraBasis {
    Vec3 eye{};
    Vec3 target{};
    Vec3 up{};
    Vec3 forward{};
    Vec3 right{};
    Vec3 ortho_up{};
};

/**
 * @brief Application runtime state.
 */
struct AppState {
    bool harnack_live_update = true;
    ViewMode view_mode = ViewMode::k_split;
    bool auto_rotate = false;
    bool wireframe = false;
    float yaw = k_default_camera_yaw;
    float pitch = k_default_camera_pitch;
    float camera_radius = k_default_camera_radius;
    float camera_fovy_radians = k_default_camera_fovy_radians;
    Vec3 camera_target{0.0f, 0.0f, 0.0f};

    float epsilon = 1e-3f;
    float t_max = 100.0f;
    int max_iterations = 2048;
    float accuracy_scale = 2.0f;
    float target_winding = 0.5f;
    float harnack_resolution_scale = 0.75f;

    float voxel_dx = 0.05f;
    float voxel_target_w = 0.5f;
    std::size_t voxel_max_voxels = 10'000'000u;
    std::size_t voxel_max_voxels_pending = 10'000'000u;
    bool voxel_max_voxels_confirm_popup_open = false;
    float voxel_actual_dx = 0.05f;
    std::size_t voxel_grid_nx = 1u;
    std::size_t voxel_grid_ny = 1u;
    std::size_t voxel_grid_nz = 1u;
    std::size_t voxel_grid_total = 0u;
    std::size_t voxel_occupied_count = 0u;
    std::size_t voxel_requested_count = 0u;
    bool voxel_truncated = false;
    float last_voxel_ms = 0.0f;

    std::size_t triangle_count = 0;
    std::size_t harnack_hit_count = 0;
    std::size_t harnack_pixel_count = 0;
    float last_harnack_ms = 0.0f;

    std::vector<MeshLibraryEntry> mesh_library{};
    int active_mesh_index = -1;
    int selected_mesh_index = -1;
    std::string active_mesh_name{"None"};
    std::string status_line{"Ready"};

    bool force_harnack_refresh = true;
    bool force_voxel_refresh = true;
    winding_studio::voxel::MeshBounds mesh_bounds{};
};

} // namespace winding_studio::app
