#include "studio_voxel.hpp"

#include <chrono>
#include <string>

#include "studio_mesh_library.hpp"

namespace winding_studio::app {

void update_voxelization_if_needed(
    AppState &state, winding_studio::Voxelizer &voxelizer, VoxelRenderer &voxel_renderer,
    winding_studio::VoxelizeStats &voxel_stats
) {
    bool const needs_voxel = state.view_mode == ViewMode::k_voxel;
    if (!(needs_voxel && has_active_mesh(state) && voxelizer.has_mesh() &&
          state.force_voxel_refresh))
        return;

    winding_studio::voxel::VoxelGridSpec const grid =
        winding_studio::voxel::make_voxel_grid_from_dx(
            state.mesh_bounds, state.voxel_dx, state.voxel_max_voxels
        );
    state.voxel_grid_nx = grid.nx;
    state.voxel_grid_ny = grid.ny;
    state.voxel_grid_nz = grid.nz;
    state.voxel_grid_total = static_cast<std::size_t>(grid.total_voxels);
    state.voxel_actual_dx = grid.actual_dx;

    if (!voxel_renderer.ensure_instance_capacity(state.voxel_grid_total)) {
        state.status_line = "Voxelize failed: cannot allocate instance buffer";
        state.voxel_occupied_count = 0;
        state.voxel_requested_count = 0;
        state.voxel_truncated = false;
        state.force_voxel_refresh = false;
        return;
    }

    winding_studio::VoxelizeConfig const config{
        state.voxel_target_w,
        state.accuracy_scale,
    };

    auto const voxel_begin = std::chrono::steady_clock::now();
    std::string voxel_error;
    bool const ok = voxelizer.voxelize(
        grid, config, voxel_renderer.instance_buffer(), voxel_renderer.instance_capacity(),
        voxel_stats, voxel_error
    );
    auto const voxel_end = std::chrono::steady_clock::now();
    state.last_voxel_ms = std::chrono::duration<float, std::milli>(voxel_end - voxel_begin).count();

    if (!ok) {
        state.status_line = "Voxelize failed: " + voxel_error;
        state.voxel_occupied_count = 0;
        state.voxel_requested_count = 0;
        state.voxel_truncated = false;
        state.force_voxel_refresh = false;
        return;
    }

    state.voxel_occupied_count = voxel_stats.occupied_count;
    state.voxel_grid_total = voxel_stats.total_voxels;
    state.voxel_requested_count = voxel_stats.requested_selected_count;
    state.voxel_truncated = voxel_stats.truncated;
    if (voxel_stats.truncated) {
        state.status_line = "Voxel truncated: " + std::to_string(state.voxel_occupied_count) +
                            " / " + std::to_string(state.voxel_requested_count);
    }

    state.force_voxel_refresh = false;
}

} // namespace winding_studio::app
