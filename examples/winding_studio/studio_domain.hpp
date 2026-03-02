#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace winding_studio::app {

/**
 * @brief View mode for the main viewport.
 */
enum class ViewMode : int {
    k_split = 0,
    k_raster = 1,
    k_harnack = 2,
    k_voxel = 3,
};

/**
 * @brief Command-line options for launching Winding Studio.
 */
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

/**
 * @brief Triangle mesh in CPU AoS form: xyz-interleaved positions + triangle indices.
 */
struct MeshData {
    std::vector<float> positions;
    std::vector<std::uint32_t> indices;
};

/**
 * @brief Entry in the session mesh library.
 */
struct MeshLibraryEntry {
    std::string name{};
    MeshData mesh{};
    std::size_t triangle_count{0};
    bool is_builtin{false};
};

} // namespace winding_studio::app
