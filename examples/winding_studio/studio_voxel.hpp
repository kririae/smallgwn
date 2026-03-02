#pragma once

#include "studio_render.hpp"
#include "studio_state.hpp"
#include "voxelizer.hpp"

namespace winding_studio::app {

/**
 * @brief Refresh voxelization output for current frame when needed.
 */
void update_voxelization_if_needed(
    AppState &state, winding_studio::Voxelizer &voxelizer, VoxelRenderer &voxel_renderer,
    winding_studio::VoxelizeStats &voxel_stats
);

} // namespace winding_studio::app
