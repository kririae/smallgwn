#pragma once

#include "imgui.h"
#include "imfilebrowser.h"
#include "voxel_grid_spec.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace winding_studio::app {

constexpr float k_pi = 3.14159265358979323846f;

using Vec3 = glm::vec3;
using Mat4 = glm::mat4;

enum class ViewMode : int {
    k_split = 0,
    k_raster = 1,
    k_harnack = 2,
    k_voxel = 3,
};

struct CliOptions {
    bool show_help{false};
    int width{1600};
    int height{960};
    ViewMode view_mode{ViewMode::k_split};
    std::string mesh_file{};
    float voxel_dx{-1.0f};
    float camera_distance{-1.0f};
    float harnack_resolution_scale{-1.0f};
    float harnack_target_w{-1.0f};
};

struct MeshData {
    std::vector<float> positions;
    std::vector<std::uint32_t> indices;
};

struct MeshLibraryEntry {
    std::string name{};
    MeshData mesh{};
    std::size_t triangle_count{0};
    bool is_builtin{false};
};

struct AppState {
    bool harnack_live_update = true;
    ViewMode view_mode = ViewMode::k_split;
    bool auto_rotate = false;
    bool wireframe = false;
    float yaw = 0.0f;
    float pitch = 0.35f;
    float camera_radius = 2.7f;
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
    float voxel_actual_dx = 0.05f;
    std::size_t voxel_grid_nx = 1u;
    std::size_t voxel_grid_ny = 1u;
    std::size_t voxel_grid_nz = 1u;
    std::size_t voxel_grid_total = 0u;
    std::size_t voxel_occupied_count = 0u;
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
    ImGui::FileBrowser file_browser{
        ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_ConfirmOnEnter
    };
    bool file_browser_initialized = false;
};

struct UiLayoutResult {
    bool harnack_params_changed{false};
    bool voxel_params_changed{false};
    int activate_mesh_index{-1};
    int remove_mesh_index{-1};
    std::string mesh_file_to_add{};
    bool request_harnack_refresh{false};
    bool request_voxel_refresh{false};
    ImVec2 viewport_pos{0.0f, 0.0f};
    ImVec2 viewport_size{1.0f, 1.0f};
};

struct FramebufferRect {
    int x{0};
    int y{0};
    int w{0};
    int h{0};
};

struct CameraBasis {
    Vec3 eye{};
    Vec3 target{};
    Vec3 up{};
    Vec3 forward{};
    Vec3 right{};
    Vec3 ortho_up{};
};

} // namespace winding_studio::app
