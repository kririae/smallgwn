#pragma once

#include "harnack_tracer.hpp"
#include "studio_mesh_library.hpp"
#include "studio_render.hpp"
#include "voxelizer.hpp"

#include <string>

namespace winding_studio::app {

/**
 * @brief Activate a mesh entry and upload it to all backends atomically.
 *
 * On backend upload failure, this function attempts to restore previously active backend resources.
 */
[[nodiscard]] bool activate_mesh_by_index(
    AppState &state, int index, std::string const &status_prefix, MeshRenderer &renderer,
    winding_studio::HarnackTracer &harnack_tracer, winding_studio::Voxelizer &voxelizer,
    std::string &error
);

} // namespace winding_studio::app
